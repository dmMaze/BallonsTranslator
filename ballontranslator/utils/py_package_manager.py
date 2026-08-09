import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion
from ballontranslator.utils import package_installer
from ballontranslator.utils.package_installer import InstallResult
from ballontranslator.utils.torch_install_helper import (
    TORCH_FAMILY_PACKAGES,
    TorchInstallRequest,
    detect_nvidia_gpus,
    has_plain_unpinned_torch,
    prepare_torch_install_request,
)

try:
    import importlib.metadata as importlib_metadata
except (ModuleNotFoundError, ImportError):
    import importlib_metadata


DEFAULT_PACKAGE_IMPORT_NAMES = {
    'hf-transfer': ['hf_transfer'],
    'opencv-contrib-python': ['cv2'],
    'opencv-python': ['cv2'],
    'opencv-python-headless': ['cv2'],
    'opencc-python-reimplemented': ['opencc'],
    'onnxruntime': ['onnxruntime'],
    'onnxruntime-gpu': ['onnxruntime'],
    'pillow': ['PIL'],
    'pillow-jxl-plugin': ['pillow_jxl'],
    'protobuf': ['google.protobuf'],
    'pyqt5-qt5': [],
    'pyqt6-qt6': [],
    'pywin32': ['win32api'],
    'python-docx': ['docx'],
    'pyobjc-core': ['objc'],
    'pyobjc-framework-cocoa': ['Cocoa'],
    'pyobjc-framework-coreml': ['CoreML'],
    'pyobjc-framework-quartz': ['Quartz'],
    'pyyaml': ['yaml'],
    'pyspellchecker': ['spellchecker'],
    'spacy-pkuseg': ['spacy_pkuseg'],
}

RUNTIME_CONSTRAINED_PACKAGES = (
    'numpy',
    'opencv-python',
    'opencv-contrib-python',
    'opencv-python-headless',
)
ALLOW_RUNTIME_PACKAGE_UPGRADE_ENV = 'BALLONTRANSLATOR_ALLOW_RUNTIME_PACKAGE_UPGRADE'


def _requirement_with_package_name(
    req: Requirement,
    package_name: str,
    extra_specifier: str = '',
) -> str:
    """Return ``req`` with a different distribution name.

    >>> _requirement_with_package_name(Requirement('onnxruntime>=1'), 'onnxruntime-gpu')
    'onnxruntime-gpu>=1'
    >>> _requirement_with_package_name(
    ...     Requirement('onnxruntime>=1'), 'onnxruntime-gpu', '<1.27.0'
    ... )
    'onnxruntime-gpu<1.27.0,>=1'
    """
    requirement = package_name
    if req.extras:
        requirement += '[' + ','.join(sorted(req.extras)) + ']'
    specifier = str(req.specifier)
    if extra_specifier:
        specifier = str(
            SpecifierSet(','.join(filter(None, (specifier, extra_specifier))))
        )
    requirement += specifier
    if req.marker:
        requirement += f'; {req.marker}'
    return requirement


def _torch_cuda_package_suffix(
    version_lookup: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Return the CUDA suffix from installed torch metadata.

    >>> _torch_cuda_package_suffix(lambda _: '2.7.1+cu118')
    'cu118'
    >>> _torch_cuda_package_suffix(lambda _: '2.7.1') is None
    True
    >>> _torch_cuda_package_suffix(lambda _: '2.7.1+cpu') is None
    True
    """
    version_lookup = version_lookup or importlib_metadata.version
    try:
        version = version_lookup('torch')
    except importlib_metadata.PackageNotFoundError:
        return None
    if '+' not in version:
        return None
    suffix = version.split('+', 1)[1].lower()
    if suffix.startswith('cu'):
        return suffix
    return None


def _onnxruntime_requirement_for_cuda(req: Requirement, cuda_suffix: str) -> str:
    """Return the ONNX Runtime package matching detected torch CUDA.

    >>> _onnxruntime_requirement_for_cuda(Requirement('onnxruntime'), 'cu118')
    'onnxruntime'
    >>> _onnxruntime_requirement_for_cuda(Requirement('onnxruntime'), 'cu128')
    'onnxruntime-gpu<1.27.0'
    >>> _onnxruntime_requirement_for_cuda(Requirement('onnxruntime'), 'cu130')
    'onnxruntime-gpu>=1.27.0'
    """
    if cuda_suffix == 'cu118':
        return _requirement_with_package_name(req, 'onnxruntime')
    if cuda_suffix.startswith('cu12'):
        return _requirement_with_package_name(req, 'onnxruntime-gpu', '<1.27.0')
    return _requirement_with_package_name(req, 'onnxruntime-gpu', '>=1.27.0')


def _resolve_onnxruntime_requirement(
    req: Requirement,
    gpu_detector=detect_nvidia_gpus,
    cuda_suffix_lookup=_torch_cuda_package_suffix,
    torch_cuda_version: Optional[str] = None,
) -> str:
    """Return the install requirement for ONNX Runtime in this environment.

    >>> _resolve_onnxruntime_requirement(
    ...     Requirement('onnxruntime'),
    ...     gpu_detector=lambda: [],
    ...     cuda_suffix_lookup=lambda: None,
    ... )
    'onnxruntime'
    """
    if sys.platform not in {'win32', 'linux'}:
        return str(req)

    cuda_suffix = torch_cuda_version or cuda_suffix_lookup()
    if cuda_suffix is not None:
        return _onnxruntime_requirement_for_cuda(req, cuda_suffix)

    try:
        has_gpu = bool(gpu_detector())
    except Exception:
        has_gpu = False
    if has_gpu:
        return _requirement_with_package_name(req, 'onnxruntime-gpu', '>=1.27.0')
    return str(req)


def _distribution_installed(package_name: str) -> bool:
    """Return whether package metadata exists without importing the package.

    >>> _distribution_installed('__ballontranslator_missing_package__')
    False
    """

    try:
        importlib_metadata.distribution(package_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    return True


def _opencv_python_installed() -> bool:
    """Return whether the GUI OpenCV wheel is present.

    >>> isinstance(_opencv_python_installed(), bool)
    True
    """

    return _distribution_installed('opencv-python')


def _runtime_package_constraints(
    version_lookup: Optional[Callable[[str], Optional[str]]] = None,
) -> List[str]:
    """Return constraints that keep loaded core native packages in place.

    >>> _runtime_package_constraints(lambda name: {'numpy': '1.26.4'}.get(name))
    ['numpy==1.26.4']
    """

    version_lookup = version_lookup or importlib_metadata.version
    constraints = []
    for package_name in RUNTIME_CONSTRAINED_PACKAGES:
        try:
            version = version_lookup(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        if version:
            constraints.append(f'{package_name}=={version}')
    return constraints


def _write_runtime_constraints_file(
    constraints: Iterable[str],
    directory: Optional[str] = None,
) -> str:
    """Write runtime package constraints for pip-compatible installers.

    >>> path = _write_runtime_constraints_file(['numpy==1.26.4'], directory=tempfile.gettempdir())
    >>> os.path.basename(path).startswith('ballontranslator-runtime-constraints-')
    True
    """

    lines = list(dict.fromkeys(constraints))
    if not lines:
        return ''
    directory = directory or tempfile.gettempdir()
    path = os.path.join(directory, f'ballontranslator-runtime-constraints-{os.getpid()}.txt')
    with open(path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))
        f.write('\n')
    return path


@dataclass
class MissingRequirement:
    """A package requirement that is unavailable in the current environment.

    >>> missing = MissingRequirement('pyyaml', 'pyyaml', ['yaml'])
    >>> missing.import_names
    ['yaml']
    """

    requirement: str
    package_name: str
    import_names: List[str] = field(default_factory=list)


@dataclass
class MissingModuleRequirements:
    """Missing requirements grouped by the module that needs them.

    >>> item = MissingModuleRequirements('ocr', 'mit48px', ['torch'])
    >>> (item.module_key, item.module_name, item.requirements)
    ('ocr', 'mit48px', ['torch'])
    """

    module_key: str
    module_name: str
    requirements: List[str] = field(default_factory=list)
    missing: List[MissingRequirement] = field(default_factory=list)


class PyPackageManager:
    """Check and install Python packages for selectable modules.

    >>> manager = PyPackageManager(backend='pip', package_import_names={'pyyaml': ['yaml']})
    >>> manager.import_names_for_requirement('pyyaml')
    ['yaml']
    >>> manager.requirement_for_import_name('yaml', ['pyyaml', 'openai>=2.8.1'])
    'pyyaml'
    """

    BACKENDS = package_installer.BACKENDS

    def __init__(
        self,
        backend: str = 'auto',
        extra_args: str = '',
        package_import_names: Optional[Dict[str, List[str]]] = None,
        env: Optional[dict] = None,
    ) -> None:
        self.backend = backend if backend in self.BACKENDS else 'auto'
        self.extra_args = extra_args or ''
        self.package_import_names = dict(DEFAULT_PACKAGE_IMPORT_NAMES)
        if package_import_names:
            for key, value in package_import_names.items():
                self.package_import_names[canonicalize_name(key)] = value
        self.env = env or os.environ.copy()

    def missing_requirements(self, requirements: Iterable[str]) -> List[MissingRequirement]:
        """Return requirements that fail package metadata or import checks.

        >>> manager = PyPackageManager()
        >>> manager.missing_requirements([])
        []
        """

        missing = []
        for original_req_text in dict.fromkeys(requirements):
            if not original_req_text:
                continue
            resolved_requirements = self.resolve_runtime_requirements([original_req_text])
            if not resolved_requirements:
                continue
            resolved_req_text = resolved_requirements[0]
            req = Requirement(resolved_req_text)
            if req.marker and not req.marker.evaluate():
                continue
            package_name = canonicalize_name(req.name)
            import_names = self.import_names_for_requirement(resolved_req_text)
            if self._requirement_satisfied(req) and all(self._import_available(name) for name in import_names):
                continue
            missing.append(MissingRequirement(str(Requirement(original_req_text)), package_name, import_names))
        return missing

    def resolve_runtime_requirements(
        self,
        requirements: Iterable[str],
        torch_cuda_version: Optional[str] = None,
    ) -> List[str]:
        """Resolve backend-specific runtime packages without importing them.

        >>> manager = PyPackageManager()
        >>> manager.resolve_runtime_requirements([])
        []
        """
        resolved = []
        for req_text in dict.fromkeys(requirements):
            if not req_text:
                continue
            req = Requirement(req_text)
            package_name = canonicalize_name(req.name)
            if package_name == 'opencv-python-headless' and _opencv_python_installed():
                continue
            if package_name == 'onnxruntime-gpu' and req.specifier:
                raise ValueError(
                    'Do not specify onnxruntime-gpu versions directly; '
                    'depend on onnxruntime and let the package manager choose '
                    'the CUDA-compatible package.'
                )
            if package_name in {'onnxruntime', 'onnxruntime-gpu'}:
                resolved.append(_resolve_onnxruntime_requirement(
                    req,
                    torch_cuda_version=torch_cuda_version,
                ))
            else:
                resolved.append(str(req))
        return resolved

    def import_names_for_requirement(self, requirement: str) -> List[str]:
        req = Requirement(requirement)
        package_name = canonicalize_name(req.name)
        if package_name in self.package_import_names:
            return self.package_import_names[package_name]
        return [req.name.replace('-', '_')]

    def requirement_for_import_name(self, import_name: str, requirements: Iterable[str]) -> Optional[str]:
        for original_requirement in dict.fromkeys(requirements):
            resolved = self.resolve_runtime_requirements([original_requirement])
            if resolved and import_name in self.import_names_for_requirement(resolved[0]):
                return str(Requirement(original_requirement))
        return None

    def build_install_command(
        self,
        requirements: Iterable[str],
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> List[str]:
        """Build an install command for this manager's backend.

        >>> manager = PyPackageManager(backend='pip')
        >>> manager.build_install_command(['einops']).count('einops')
        1
        """

        return self.build_install_commands(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )[0]

    def build_install_commands(
        self,
        requirements: Iterable[str],
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> List[List[str]]:
        """Build install command(s), splitting torch CUDA wheels when needed.

        >>> manager = PyPackageManager(backend='pip')
        >>> len(manager.build_install_commands(['einops']))
        1
        """

        requests = self._prepare_install_requests(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        return self._build_install_commands(requests)

    def _build_install_commands(
        self,
        requests: Iterable[TorchInstallRequest],
    ) -> List[List[str]]:
        constraint_files = self._runtime_constraint_files()
        return [
            package_installer.build_install_command(
                requirements=request.requirements,
                constraint_files=constraint_files,
                backend=request.backend or self.backend,
                extra_args=self.extra_args,
                env=request.env,
            )
            for request in requests
        ]

    def install(
        self,
        requirements: Iterable[str],
        progress_callback: Optional[Callable[[dict], None]] = None,
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> InstallResult:
        requirements = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
        requests = self._prepare_install_requests(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        uninstall_command = self._torch_uninstall_command(requests)
        if progress_callback is not None:
            progress_callback({
                'event': 'installing_packages',
                'message': self._installing_packages_summary(requirements),
            })
        if uninstall_command:
            try:
                completed = subprocess.run(
                    uninstall_command,
                    env=self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    shell=False,
                )
                result = InstallResult(
                    completed.returncode == 0,
                    uninstall_command,
                    returncode=completed.returncode,
                    stdout=completed.stdout or '',
                )
            except Exception as e:
                result = InstallResult(False, uninstall_command, returncode=-1, error=str(e))
            if not result.ok:
                return result
        constraint_files = self._runtime_constraint_files()
        final_result = None
        for request in requests:
            result = package_installer.install(
                requirements=request.requirements,
                constraint_files=constraint_files,
                backend=request.backend or self.backend,
                extra_args=self.extra_args,
                env=request.env,
                progress_callback=progress_callback,
            )
            final_result = result
            if not result.ok:
                return result
        return final_result or InstallResult(True, [])

    def resolve_backend(self) -> str:
        return package_installer.resolve_backend(self.backend, env=self.env)

    def preview_command(
        self,
        requirements: Iterable[str],
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> str:
        requirements = list(requirements)
        requests = self._prepare_install_requests(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        commands = self._build_install_commands(requests)
        uninstall_command = self._torch_uninstall_command(requests)
        if uninstall_command:
            commands.insert(0, uninstall_command)
        return '\n'.join(shlex.join(command) for command in commands)

    def torch_install_device(self, requirements: Iterable[str]) -> str:
        """Return the device selected by torch install probing.

        >>> manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})
        >>> manager.torch_install_device(['einops'])
        'cpu'
        """

        request = prepare_torch_install_request(requirements=requirements, env=self.env)
        return request.device

    def torch_install_cuda_version(self, requirements: Iterable[str]) -> Optional[str]:
        """Return the CUDA profile selected by torch install probing.

        >>> manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})
        >>> manager.torch_install_cuda_version(['einops']) is None
        True
        """

        request = prepare_torch_install_request(requirements=requirements, env=self.env)
        return request.cuda_version

    @staticmethod
    def needs_torch_install_choice(requirements: Iterable[str]) -> bool:
        """Return whether a torch install should ask for a device target.

        >>> PyPackageManager.needs_torch_install_choice(['torch'])
        True
        >>> PyPackageManager.needs_torch_install_choice(['torch==2.7.1'])
        False
        """

        return has_plain_unpinned_torch(requirements)

    @staticmethod
    def _installing_packages_summary(requirements: Iterable[str]) -> str:
        """Return a compact package summary for the progress panel.

        >>> PyPackageManager._installing_packages_summary(['torch', 'torchvision'])
        'torch...'
        >>> PyPackageManager._installing_packages_summary(['einops'])
        'einops'
        """

        reqs = list(dict.fromkeys(requirements))
        if not reqs:
            return 'packages'
        first = Requirement(reqs[0]).name
        return first + ('...' if len(reqs) > 1 else '')

    def _prepare_install_requests(
        self,
        requirements: Iterable[str],
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> List[TorchInstallRequest]:
        request = prepare_torch_install_request(
            requirements=requirements,
            env=self.env,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        resolved_requirements = self.resolve_runtime_requirements(
            request.requirements,
            torch_cuda_version=request.cuda_version,
        )
        request = type(request)(
            requirements=resolved_requirements,
            env=request.env,
            profile=request.profile,
            backend=request.backend,
            device=request.device,
            cuda_version=request.cuda_version,
        )
        if request.profile is None:
            return [request]
        torch_requirements, other_requirements = self._split_torch_family_requirements(request.requirements)
        requests = []
        if torch_requirements:
            requests.append(type(request)(
                requirements=torch_requirements,
                env=request.env,
                profile=request.profile,
                backend=request.backend,
            ))
        if other_requirements:
            # Non-torch packages must resolve against the user's normal package
            # source; the PyTorch CUDA wheel index does not host packages like einops.
            requests.append(type(request)(
                requirements=other_requirements,
                env=dict(self.env),
            ))
        return requests

    @staticmethod
    def _torch_uninstall_command(requests: Iterable[TorchInstallRequest]) -> List[str]:
        requested_packages = {
            canonicalize_name(Requirement(requirement).name)
            for request in requests
            for requirement in request.requirements
        }
        installed_packages = [
            package_name
            for package_name in ('torch', 'torchvision')
            if package_name in requested_packages and _distribution_installed(package_name)
        ]
        if not installed_packages:
            return []
        return [sys.executable, '-m', 'pip', 'uninstall', '-y', *installed_packages]

    def _runtime_constraint_files(self) -> List[str]:
        """Return constraint files that protect loaded core native packages.

        >>> manager = PyPackageManager(env={ALLOW_RUNTIME_PACKAGE_UPGRADE_ENV: '1'})
        >>> manager._runtime_constraint_files()
        []
        """

        if self.env.get(ALLOW_RUNTIME_PACKAGE_UPGRADE_ENV, '').lower() in {'1', 'true', 'yes'}:
            return []
        constraint_file = _write_runtime_constraints_file(_runtime_package_constraints())
        return [constraint_file] if constraint_file else []

    @staticmethod
    def _split_torch_family_requirements(requirements: Iterable[str]):
        torch_requirements = []
        other_requirements = []
        for requirement in requirements:
            package_name = canonicalize_name(Requirement(requirement).name)
            if package_name in TORCH_FAMILY_PACKAGES:
                torch_requirements.append(requirement)
            else:
                other_requirements.append(requirement)
        return torch_requirements, other_requirements

    def _requirement_satisfied(self, req: Requirement) -> bool:
        try:
            dist = importlib_metadata.distribution(req.name)
        except importlib_metadata.PackageNotFoundError:
            return False
        if req.specifier:
            version = dist.version
            if not version:
                return False
            try:
                if not req.specifier.contains(version, prereleases=True):
                    return False
            except (InvalidVersion, TypeError):
                return False
        return True

    @staticmethod
    def _import_available(import_name: str) -> bool:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                return False
            # Check if it is a corrupted package (e.g. folder exists but missing __init__.py,
            # which Python imports as a namespace package).
            if spec.loader is None and spec.submodule_search_locations is not None:
                has_init = False
                for loc in spec.submodule_search_locations:
                    for ext in ('.py', '.pyc', '.pyd', '.so'):
                        if os.path.exists(os.path.join(loc, f'__init__{ext}')):
                            has_init = True
                            break
                    if has_init:
                        break
                if not has_init:
                    return False
            return True
        except (ImportError, ModuleNotFoundError, ValueError):
            return False


def collect_missing_module_requirements(
    module_specs: Iterable[tuple],
    package_manager: PyPackageManager,
) -> List[MissingModuleRequirements]:
    """Collect missing packages for several selected module specs.

    >>> class FakeManager:
    ...     def missing_requirements(self, requirements):
    ...         return [MissingRequirement(req, req, [req]) for req in requirements if req == 'torch']
    >>> class Spec:
    ...     dependencies = ['torch', 'einops', 'torch']
    >>> missing = collect_missing_module_requirements([('ocr', 'mit48px', Spec())], FakeManager())
    >>> [(item.module_key, item.module_name, item.requirements) for item in missing]
    [('ocr', 'mit48px', ['torch'])]
    """

    missing_modules = []
    for module_key, module_name, spec in module_specs:
        requirements = list(dict.fromkeys(getattr(spec, 'dependencies', []) or []))
        if not requirements:
            continue
        missing = package_manager.missing_requirements(requirements)
        if missing:
            missing_modules.append(MissingModuleRequirements(
                module_key=module_key,
                module_name=module_name,
                requirements=[item.requirement for item in missing],
                missing=missing,
            ))
    return missing_modules
