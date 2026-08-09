import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ballontranslator.utils import shared


TORCH_FAMILY_PACKAGES = {'torch', 'torchvision'}
TORCH_INSTALL_DEVICE_OPTIONS = ('cpu', 'cuda', 'xpu')
TORCH_CUDA_VERSION_OPTIONS = ('cu128', 'cu118')
TORCH_CUDA_CUTOFF = 7.5
NVIDIA_SMI_TIMEOUT = 5
XPU_SMI_TIMEOUT = 5
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
class IntelXpuInfo:
    """Detected Intel XPU metadata used for torch wheel selection.

    >>> xpu = IntelXpuInfo('Intel(R) Arc(TM) A770 Graphics', '0')
    >>> (xpu.name, xpu.device_id)
    ('Intel(R) Arc(TM) A770 Graphics', '0')
    """

    name: str
    device_id: Optional[str] = None


@dataclass(frozen=True)
class TorchInstallProfile:
    """Torch-family wheel selection for a PyTorch runtime index.

    >>> profile = TorchInstallProfile('cu118', ('torch==2.7.1',), 'https://example.invalid')
    >>> profile.name
    'cu118'
    """

    name: str
    requirements: Sequence[str]
    index_url: str
    use_aliyun_find_links: bool = True


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
    device: str = 'cpu'
    cuda_version: Optional[str] = None


OLDER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu118',
    requirements=(
        'torch==2.7.1',
        'torchvision==0.22.1',
    ),
    index_url='https://download.pytorch.org/whl/cu118',
)

NEWER_NVIDIA_PROFILE = TorchInstallProfile(
    name='cu128',
    requirements=(
        'torch==2.10.0',
        'torchvision==0.25.0',
    ),
    index_url='https://download.pytorch.org/whl/cu128',
)

CPU_PROFILE = TorchInstallProfile(
    name='cpu',
    requirements=NEWER_NVIDIA_PROFILE.requirements,
    index_url='https://download.pytorch.org/whl/cpu',
)

INTEL_XPU_PROFILE = TorchInstallProfile(
    name='xpu',
    requirements=(
        'torch',
        'torchvision',
    ),
    index_url='https://download.pytorch.org/whl/xpu',
    use_aliyun_find_links=False,
)


def prepare_torch_install_request(
    requirements: Iterable[str],
    env: Optional[dict] = None,
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
    torch_device: Optional[str] = None,
    torch_cuda_version: Optional[str] = None,
) -> TorchInstallRequest:
    """Rewrite a plain ``torch`` install request for detected GPU backends.

    >>> request = prepare_torch_install_request(
    ...     ['torch', 'torchvision', 'einops'],
    ...     env={'PATH': '/bin', 'INDEX_URL': 'https://mirrors.aliyun.com/pypi/simple'},
    ...     gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
    ... )
    >>> request.requirements[:4]
    ['torch==2.10.0', 'torchvision==0.25.0', 'einops']
    >>> request.env['FIND_LINKS']
    'https://mirrors.aliyun.com/pytorch-wheels/cu128'
    >>> prepare_torch_install_request(['torch'], torch_device='xpu').device
    'xpu'
    >>> prepare_torch_install_request(['torch'], torch_device='cuda', torch_cuda_version='cu118').profile.name
    'cu118'
    """

    reqs = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
    request_env = dict(env or os.environ.copy())
    if not _has_plain_unpinned_torch(reqs):
        return TorchInstallRequest(reqs, request_env)

    profile, device = select_torch_install_profile_for_device(
        torch_device,
        torch_cuda_version=torch_cuda_version,
        gpu_detector=gpu_detector,
        xpu_detector=xpu_detector,
    )
    if profile is None:
        return TorchInstallRequest(reqs, request_env, device=device)

    request_env = _env_for_torch_profile(request_env, profile)
    return TorchInstallRequest(
        requirements=_rewrite_torch_family_requirements(reqs, profile),
        env=request_env,
        profile=profile,
        # uv can reject PyTorch wheel-index installs during resolution; the
        # PyTorch-provided commands use pip directly, so mirror that path here.
        backend='pip',
        device=device,
        cuda_version=profile.name if device == 'cuda' else None,
    )


def _env_for_torch_profile(env: dict, profile: TorchInstallProfile) -> dict:
    """Return installer env for a selected torch wheel profile.

    >>> env = _env_for_torch_profile({'INDEX_URL': ALIYUN_PYPI_MIRROR}, NEWER_NVIDIA_PROFILE)
    >>> env['FIND_LINKS']
    'https://mirrors.aliyun.com/pytorch-wheels/cu128'
    >>> env['INDEX_URL']
    'https://mirrors.aliyun.com/pypi/simple'
    >>> _env_for_torch_profile({}, CPU_PROFILE)['INDEX_URL']
    'https://download.pytorch.org/whl/cpu'
    >>> _env_for_torch_profile({'INDEX_URL': ALIYUN_PYPI_MIRROR}, INTEL_XPU_PROFILE)['INDEX_URL']
    'https://download.pytorch.org/whl/xpu'
    """

    result = dict(env)
    if profile.use_aliyun_find_links and _is_aliyun_pypi_mirror(result.get('INDEX_URL')):
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


def select_torch_install_profile_for_device(
    torch_device: Optional[str] = None,
    torch_cuda_version: Optional[str] = None,
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    """Choose a torch wheel profile for an automatic or user-selected device.

    >>> select_torch_install_profile_for_device('cpu')[1]
    'cpu'
    >>> select_torch_install_profile_for_device('xpu')[0].name
    'xpu'
    >>> select_torch_install_profile_for_device('cuda', torch_cuda_version='cu118')[0].name
    'cu118'
    """

    if torch_device is not None:
        torch_device = torch_device.lower()
    if torch_device not in (None, *TORCH_INSTALL_DEVICE_OPTIONS):
        torch_device = None
    profile_by_cuda_version = {
        OLDER_NVIDIA_PROFILE.name: OLDER_NVIDIA_PROFILE,
        NEWER_NVIDIA_PROFILE.name: NEWER_NVIDIA_PROFILE,
    }

    if torch_device == 'cpu':
        profile = CPU_PROFILE if sys.platform in {'win32', 'linux'} else None
        return profile, 'cpu'
    if torch_device == 'cuda':
        if torch_cuda_version in profile_by_cuda_version:
            return profile_by_cuda_version[torch_cuda_version], 'cuda'
        detector = gpu_detector or detect_nvidia_gpus
        return select_torch_install_profile(detector()) or NEWER_NVIDIA_PROFILE, 'cuda'
    if torch_device == 'xpu':
        return INTEL_XPU_PROFILE, 'xpu'

    return _cached_preferred_torch_install_profile(gpu_detector, xpu_detector)


def _cached_preferred_torch_install_profile(
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    """Return the cached automatic torch install target.

    >>> profile, device = _cached_preferred_torch_install_profile(lambda: [], lambda: [])
    >>> (profile.name, device)
    ('cpu', 'cpu')
    """

    if gpu_detector is not None or xpu_detector is not None:
        return _detect_preferred_torch_install_profile(gpu_detector, xpu_detector)

    cached_device = shared.TORCH_INSTALL_PREFERRED_DEVICE
    if cached_device in TORCH_INSTALL_DEVICE_OPTIONS:
        return shared.TORCH_INSTALL_PREFERRED_PROFILE, cached_device

    profile, device = _detect_preferred_torch_install_profile()
    shared.TORCH_INSTALL_PREFERRED_DEVICE = device
    shared.TORCH_INSTALL_PREFERRED_PROFILE = profile
    return profile, device


def _detect_preferred_torch_install_profile(
    gpu_detector: Optional[Callable[[], List[NvidiaGpuInfo]]] = None,
    xpu_detector: Optional[Callable[[], List[IntelXpuInfo]]] = None,
) -> tuple:
    detector = gpu_detector or detect_nvidia_gpus
    profile = select_torch_install_profile(detector())
    if profile is not None:
        return profile, 'cuda'

    detector = xpu_detector or detect_intel_xpus
    profile = select_torch_xpu_install_profile(detector())
    if profile is not None:
        return profile, 'xpu'
    profile = CPU_PROFILE if sys.platform in {'win32', 'linux'} else None
    return profile, 'cpu'


def select_torch_xpu_install_profile(xpus: Sequence[IntelXpuInfo]) -> Optional[TorchInstallProfile]:
    """Choose the torch XPU wheel profile from detected Intel XPU devices.

    >>> select_torch_xpu_install_profile([]) is None
    True
    >>> select_torch_xpu_install_profile([IntelXpuInfo('Intel Arc')]).name
    'xpu'
    """

    if not xpus:
        return None
    return INTEL_XPU_PROFILE


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


def detect_intel_xpus() -> List[IntelXpuInfo]:
    """Detect Intel XPU devices through driver tools without importing torch.

    >>> isinstance(detect_intel_xpus(), list)
    True
    """

    if sys.platform not in {'win32', 'linux'}:
        return []
    command_path = _find_xpu_smi()
    if not command_path:
        return []

    output = _run_xpu_smi([command_path, 'discovery', '-j'])
    xpus = _parse_xpu_smi_discovery_json(output)
    if xpus:
        return xpus

    output = _run_xpu_smi([command_path, 'discovery'])
    return _parse_xpu_smi_discovery_text(output)


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


def _find_xpu_smi() -> Optional[str]:
    found = shutil.which('xpu-smi')
    if found:
        return found

    candidates = []
    if sys.platform == 'win32':
        program_dirs = [
            os.environ.get('ProgramFiles'),
            os.environ.get('ProgramFiles(x86)'),
        ]
        for program_dir in program_dirs:
            if not program_dir:
                continue
            candidates.extend([
                os.path.join(program_dir, 'Intel', 'oneAPI', 'tools', 'latest', 'xpu-smi', 'xpu-smi.exe'),
                os.path.join(program_dir, 'Intel', 'oneAPI', 'tools', 'latest', 'bin', 'xpu-smi.exe'),
            ])
    elif sys.platform == 'linux':
        candidates.extend([
            '/usr/bin/xpu-smi',
            '/usr/local/bin/xpu-smi',
            '/opt/intel/xpumanager/bin/xpu-smi',
        ])

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


def _run_xpu_smi(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False,
            timeout=XPU_SMI_TIMEOUT,
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


def _parse_xpu_smi_discovery_json(output: str) -> List[IntelXpuInfo]:
    """Parse ``xpu-smi discovery -j`` output.

    >>> _parse_xpu_smi_discovery_json('{"device_list": [{"device_id": 0, "device_name": "Intel Arc"}]}')
    [IntelXpuInfo(name='Intel Arc', device_id='0')]
    >>> _parse_xpu_smi_discovery_json('not json')
    []
    """

    if not output:
        return []
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return []
    if isinstance(data, dict):
        devices = data.get('device_list') or data.get('devices') or []
    elif isinstance(data, list):
        devices = data
    else:
        return []
    return _parse_xpu_smi_device_entries(devices)


def _parse_xpu_smi_device_entries(devices: Iterable[object]) -> List[IntelXpuInfo]:
    xpus = []
    for entry in devices:
        if not isinstance(entry, dict):
            continue
        device_type = _first_entry_value(entry, ('device_type', 'type', 'Device Type'))
        if device_type and not any(token in device_type.lower() for token in ('gpu', 'xpu')):
            continue
        name = _first_entry_value(entry, ('device_name', 'name', 'Device Name'))
        device_id = _first_entry_value(entry, ('device_id', 'id', 'Device ID'))
        if not name and device_id is None:
            continue
        xpus.append(IntelXpuInfo(name or 'Intel XPU', device_id))
    return xpus


def _first_entry_value(entry: dict, keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parse_xpu_smi_discovery_text(output: str) -> List[IntelXpuInfo]:
    """Parse plain ``xpu-smi discovery`` output as a best-effort fallback.

    >>> _parse_xpu_smi_discovery_text('| 0 | Intel(R) Arc(TM) A770 Graphics |')[0].name
    'Intel(R) Arc(TM) A770 Graphics'
    """

    xpus = []
    seen = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or 'intel' not in line.lower():
            continue
        parts = [part.strip() for part in line.split('|') if part.strip()]
        candidates = parts or [line]
        for candidate in candidates:
            if 'intel' not in candidate.lower():
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            xpus.append(IntelXpuInfo(candidate))
            break
    return xpus


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


def has_plain_unpinned_torch(requirements: Iterable[str]) -> bool:
    """Return whether requirements include plain ``torch`` needing a device choice.

    >>> has_plain_unpinned_torch(['torch', 'einops'])
    True
    >>> has_plain_unpinned_torch(['torch==2.7.1'])
    False
    """

    reqs = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
    return _has_plain_unpinned_torch(reqs)


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
