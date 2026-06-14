import importlib.util
import os
import shlex
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from ballontranslator.utils import package_installer
from ballontranslator.utils.package_installer import InstallResult
from ballontranslator.utils.torch_install_helper import TORCH_FAMILY_PACKAGES, prepare_torch_install_request

try:
    import importlib.metadata as importlib_metadata
except (ModuleNotFoundError, ImportError):
    import importlib_metadata


DEFAULT_PACKAGE_IMPORT_NAMES = {
    'hf-transfer': ['hf_transfer'],
    'opencv-python': ['cv2'],
    'opencc-python-reimplemented': ['opencc'],
    'pillow': ['PIL'],
    'pillow-jxl-plugin': ['pillow_jxl'],
    'pyyaml': ['yaml'],
    'spacy-pkuseg': ['spacy_pkuseg'],
}


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
        for req_text in dict.fromkeys(requirements):
            if not req_text:
                continue
            req = Requirement(req_text)
            if req.marker and not req.marker.evaluate():
                continue
            package_name = canonicalize_name(req.name)
            import_names = self.import_names_for_requirement(req_text)
            if self._requirement_satisfied(req) and all(self._import_available(name) for name in import_names):
                continue
            missing.append(MissingRequirement(str(req), package_name, import_names))
        return missing

    def import_names_for_requirement(self, requirement: str) -> List[str]:
        req = Requirement(requirement)
        package_name = canonicalize_name(req.name)
        if package_name in self.package_import_names:
            return self.package_import_names[package_name]
        return [req.name.replace('-', '_')]

    def requirement_for_import_name(self, import_name: str, requirements: Iterable[str]) -> Optional[str]:
        for requirement in requirements:
            if import_name in self.import_names_for_requirement(requirement):
                return str(Requirement(requirement))
        return None

    def build_install_command(self, requirements: Iterable[str]) -> List[str]:
        """Build an install command for this manager's backend.

        >>> manager = PyPackageManager(backend='pip')
        >>> manager.build_install_command(['einops']).count('einops')
        1
        """

        return self.build_install_commands(requirements)[0]

    def build_install_commands(self, requirements: Iterable[str]) -> List[List[str]]:
        """Build install command(s), splitting torch CUDA wheels when needed.

        >>> manager = PyPackageManager(backend='pip')
        >>> len(manager.build_install_commands(['einops']))
        1
        """

        requests = self._prepare_install_requests(requirements)
        return [
            package_installer.build_install_command(
                requirements=request.requirements,
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
    ) -> InstallResult:
        requirements = [str(Requirement(req)) for req in dict.fromkeys(requirements) if req]
        requests = self._prepare_install_requests(requirements)
        if progress_callback is not None:
            progress_callback({
                'event': 'installing_packages',
                'message': self._installing_packages_summary(requirements),
            })
        final_result = None
        for request in requests:
            result = package_installer.install(
                requirements=request.requirements,
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

    def preview_command(self, requirements: Iterable[str]) -> str:
        return '\n'.join(shlex.join(command) for command in self.build_install_commands(requirements))

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

    def _prepare_install_requests(self, requirements: Iterable[str]):
        request = prepare_torch_install_request(requirements=requirements, env=self.env)
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
        if req.specifier and not req.specifier.contains(dist.version, prereleases=True):
            return False
        return True

    @staticmethod
    def _import_available(import_name: str) -> bool:
        try:
            return importlib.util.find_spec(import_name) is not None
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
