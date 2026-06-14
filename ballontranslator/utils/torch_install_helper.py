import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


TORCH_FAMILY_PACKAGES = {'torch', 'torchvision', 'torchaudio'}
TORCH_CUDA_CUTOFF = 7.5
NVIDIA_SMI_TIMEOUT = 5
ALIYUN_PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple'
ALIYUN_PYTORCH_WHEEL_ROOT = 'https://mirrors.aliyun.com/pytorch-wheels'


@dataclass(frozen=True)
class NvidiaGpuInfo:
    """Detected NVIDIA GPU metadata used for torch wheel selection.

    >>> gpu = NvidiaGpuInfo('NVIDIA GeForce RTX 4090', 8.9)
    >>> (gpu.name, gpu.compute_capability)
    ('NVIDIA GeForce RTX 4090', 8.9)
    """

    name: str
    compute_capability: Optional[float] = None


@dataclass(frozen=True)
class TorchInstallProfile:
    """Pinned torch-family wheel selection for a CUDA runtime index.

    >>> profile = TorchInstallProfile('cu118', ('torch==2.7.1',), 'https://example.invalid')
    >>> profile.name
    'cu118'
    """

    name: str
    requirements: Sequence[str]
    index_url: str


@dataclass(frozen=True)
class TorchInstallRequest:
    """Requirements and environment after optional torch-profile rewriting.

    >>> request = TorchInstallRequest(['torch'], {'PATH': '/bin'})
    >>> request.profile is None
    True
    """

    requirements: List[str]
    env: dict
    profile: Optional[TorchInstallProfile] = None
    backend: Optional[str] = None


OLDER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu118',
    requirements=(
        'torch==2.7.1',
        'torchvision==0.22.1',
        'torchaudio==2.7.1',
    ),
    index_url='https://download.pytorch.org/whl/cu118',
)

NEWER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu128',
    requirements=(
        'torch==2.10.0',
        'torchvision==0.25.0',
        'torchaudio==2.10.0',
    ),
    index_url='https://download.pytorch.org/whl/cu128',
)


def prepare_torch_install_request(
    requirements: Iterable[str],
    env: Optional[dict] = None,
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
) -> TorchInstallRequest:
    """Rewrite a plain ``torch`` install request for detected NVIDIA GPUs.

    >>> request = prepare_torch_install_request(
    ...     ['torch', 'torchvision', 'einops'],
    ...     env={'PATH': '/bin', 'INDEX_URL': 'https://mirrors.aliyun.com/pypi/simple'},
    ...     gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
    ... )
    >>> request.requirements[:4]
    ['torch==2.10.0', 'torchvision==0.25.0', 'torchaudio==2.10.0', 'einops']
    >>> request.env['FIND_LINKS']
    'https://mirrors.aliyun.com/pytorch-wheels/cu128'
    """

    reqs = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
    request_env = dict(env or os.environ.copy())
    if not _has_plain_unpinned_torch(reqs):
        return TorchInstallRequest(reqs, request_env)

    detector = gpu_detector or detect_nvidia_gpus
    profile = select_torch_install_profile(detector())
    if profile is None:
        return TorchInstallRequest(reqs, request_env)

    request_env = _env_for_torch_profile(request_env, profile)
    return TorchInstallRequest(
        requirements=_rewrite_torch_family_requirements(reqs, profile),
        env=request_env,
        profile=profile,
        # uv can reject PyTorch CUDA wheel-link installs during resolution; the
        # PyTorch-provided commands use pip directly, so mirror that path here.
        backend='pip',
    )


def _env_for_torch_profile(env: dict, profile: TorchInstallProfile) -> dict:
    """Return installer env for a selected torch CUDA wheel profile.

    >>> env = _env_for_torch_profile({'INDEX_URL': ALIYUN_PYPI_MIRROR}, NEWER_NVIDIA_PROFILE)
    >>> env['FIND_LINKS']
    'https://mirrors.aliyun.com/pytorch-wheels/cu128'
    >>> env['INDEX_URL']
    'https://mirrors.aliyun.com/pypi/simple'
    """

    result = dict(env)
    if _is_aliyun_pypi_mirror(result.get('INDEX_URL')):
        result['INDEX_URL'] = ALIYUN_PYPI_MIRROR
        result['FIND_LINKS'] = f'{ALIYUN_PYTORCH_WHEEL_ROOT}/{profile.name}'
    else:
        result['INDEX_URL'] = profile.index_url
        result.pop('FIND_LINKS', None)
    return result


def select_torch_install_profile(gpus: Sequence[NvidiaGpuInfo]) -> Optional[TorchInstallProfile]:
    """Choose a torch CUDA wheel profile from detected NVIDIA GPUs.

    >>> select_torch_install_profile([]) is None
    True
    >>> select_torch_install_profile([NvidiaGpuInfo('GTX 1080', 6.1)]).name
    'cu118'
    >>> select_torch_install_profile([NvidiaGpuInfo('RTX 2080', 7.5)]).name
    'cu128'
    """

    if not gpus:
        return None
    for gpu in gpus:
        if gpu.compute_capability is None or gpu.compute_capability < TORCH_CUDA_CUTOFF:
            return OLDER_NVIDIA_PROFILE
    return NEWER_NVIDIA_PROFILE


def detect_nvidia_gpus() -> List[NvidiaGpuInfo]:
    """Detect NVIDIA GPUs through driver tools without importing torch.

    >>> isinstance(detect_nvidia_gpus(), list)
    True
    """

    if sys.platform not in {'win32', 'linux'}:
        return []
    command_path = _find_nvidia_smi()
    if not command_path:
        return []

    output = _run_nvidia_smi([
        command_path,
        '--query-gpu=name,compute_cap',
        '--format=csv,noheader,nounits',
    ])
    gpus = _parse_nvidia_smi_compute_output(output)
    if gpus:
        return gpus

    # Older driver builds may not expose compute_cap through nvidia-smi.
    name_output = _run_nvidia_smi([
        command_path,
        '--query-gpu=name',
        '--format=csv,noheader,nounits',
    ])
    return [NvidiaGpuInfo(name.strip(), None) for name in name_output.splitlines() if name.strip()]


def _find_nvidia_smi() -> Optional[str]:
    found = shutil.which('nvidia-smi')
    if found:
        return found

    candidates = []
    if sys.platform == 'win32':
        system_root = os.environ.get('SystemRoot')
        program_files = os.environ.get('ProgramFiles')
        if system_root:
            candidates.append(os.path.join(system_root, 'System32', 'nvidia-smi.exe'))
        if program_files:
            candidates.append(os.path.join(program_files, 'NVIDIA Corporation', 'NVSMI', 'nvidia-smi.exe'))
    elif sys.platform == 'linux':
        candidates.extend(['/usr/bin/nvidia-smi', '/usr/local/bin/nvidia-smi'])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _run_nvidia_smi(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=NVIDIA_SMI_TIMEOUT,
        )
    except Exception:
        return ''
    if completed.returncode != 0:
        return ''
    return completed.stdout or ''


def _parse_nvidia_smi_compute_output(output: str) -> List[NvidiaGpuInfo]:
    """Parse ``name, compute_cap`` rows from ``nvidia-smi``.

    >>> _parse_nvidia_smi_compute_output('NVIDIA GeForce RTX 4090, 8.9\\n')[0].compute_capability
    8.9
    >>> _parse_nvidia_smi_compute_output('NVIDIA GPU, N/A\\n')[0].compute_capability is None
    True
    """

    gpus = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        name, _, capability_text = line.partition(',')
        name = name.strip()
        if not name:
            continue
        gpus.append(NvidiaGpuInfo(name, _parse_compute_capability(capability_text)))
    return gpus


def _parse_compute_capability(value: str) -> Optional[float]:
    match = re.search(r'(\d+)(?:\.(\d+))?', value or '')
    if match is None:
        return None
    major = match.group(1)
    minor = match.group(2) or '0'
    try:
        return float(f'{major}.{minor}')
    except ValueError:
        return None


def _is_aliyun_pypi_mirror(index_url: Optional[str]) -> bool:
    if not isinstance(index_url, str):
        return False
    normalized = index_url.strip().lower().rstrip('/')
    return normalized in {
        'https://mirrors.aliyun.com/pypi/simple',
        'http://mirrors.aliyun.com/pypi/simple',
    }


def _has_plain_unpinned_torch(requirements: Sequence[str]) -> bool:
    for req_text in requirements:
        req = Requirement(req_text)
        if canonicalize_name(req.name) != 'torch':
            continue
        if req.specifier or req.marker or req.url or req.extras:
            continue
        return True
    return False


def _rewrite_torch_family_requirements(
    requirements: Sequence[str],
    profile: TorchInstallProfile,
) -> List[str]:
    rewritten = []
    inserted_profile = False
    for req_text in requirements:
        req = Requirement(req_text)
        package_name = canonicalize_name(req.name)
        if package_name in TORCH_FAMILY_PACKAGES:
            if package_name == 'torch' and not inserted_profile:
                rewritten.extend(profile.requirements)
                inserted_profile = True
            continue
        rewritten.append(str(req))
    if not inserted_profile:
        rewritten = [*profile.requirements, *rewritten]
    return list(dict.fromkeys(rewritten))
