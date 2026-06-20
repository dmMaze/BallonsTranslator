from typing import List
import os.path as osp
import os

from .base import LOGGER
import ballontranslator.utils.shared as shared
from ballontranslator.utils.download_util import DownloadCancelled, download_and_check_files
from ballontranslator.utils.py_package_manager import MissingRequirement, PyPackageManager
from ballontranslator.utils.registry import ModuleSpec
from .package_import_names import PACKAGE_IMPORT_NAMES


class MissingDependency(ModuleNotFoundError):
    """Raised when a module's declared Python dependencies are missing.

    >>> spec = ModuleSpec(key='mit48px', import_path='pkg.mod', class_name='OCR')
    >>> missing = [MissingRequirement('torch', 'torch', ['torch'])]
    >>> err = MissingDependency(spec, missing)
    >>> err.requirements
    ['torch']
    """

    def __init__(self, spec: ModuleSpec, missing: List[MissingRequirement]):
        self.spec = spec
        self.missing = missing
        self.requirements = [item.requirement for item in missing]
        deps = ', '.join(self.requirements)
        super().__init__(
            f'Module "{spec.key}" requires Python package(s): {deps}. '
            f'Install them or select a module that does not require them.'
        )


def _spec_from_module(module_spec_or_class) -> ModuleSpec:
    """Normalize either a ``ModuleSpec`` or loaded class into a ``ModuleSpec``.

    >>> class DemoModule:
    ...     name = 'demo'
    ...     dependencies = ['torch']
    >>> _spec_from_module(DemoModule).dependencies
    ['torch']
    """

    if isinstance(module_spec_or_class, ModuleSpec):
        return module_spec_or_class
    module_class = module_spec_or_class
    spec = getattr(module_class, '_module_spec', None)
    if isinstance(spec, ModuleSpec):
        return spec
    return ModuleSpec(
        key=getattr(module_class, 'name', getattr(module_class, '__name__', '')),
        import_path=getattr(module_class, '__module__', ''),
        class_name=getattr(module_class, '__name__', ''),
        params=getattr(module_class, 'params', None),
        download_file_list=getattr(module_class, 'download_file_list', None),
        download_file_on_load=getattr(module_class, 'download_file_on_load', False),
        dependencies=getattr(module_class, 'dependencies', []),
        resolved_class=module_class,
    )


def _missing_module_requirements(module_spec_or_class) -> List[MissingRequirement]:
    """Check a module spec's declared dependencies with ``PyPackageManager``.

    >>> spec = ModuleSpec(key='demo', import_path='pkg.mod', class_name='Demo', dependencies=[])
    >>> _missing_module_requirements(spec)
    []
    """

    spec = _spec_from_module(module_spec_or_class)
    dependencies = spec.dependencies if spec.dependencies is not None else []
    dependencies = list(dict.fromkeys(dependencies))
    package_manager = PyPackageManager(package_import_names=PACKAGE_IMPORT_NAMES)
    return package_manager.missing_requirements(dependencies)


def ensure_module_files(module_spec_or_class, progress_callback=None, cancel_event=None):
    """Prepare optional dependencies and declared files before module import.

    Example:
        >>> spec = ModuleSpec(key='demo', import_path='demo.module', class_name='Demo')
        >>> ensure_module_files(spec)
        True
    """

    # Called from the selected module thread before importing or instantiating it.
    spec = _spec_from_module(module_spec_or_class)
    missing = _missing_module_requirements(spec)
    if missing:
        raise MissingDependency(spec, missing)
    if spec.download_file_list is None:
        return True

    all_successful = True
    for download_kwargs in spec.download_file_list:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled('Module preparation cancelled by user.')
        if progress_callback is not None:
            progress_callback({'event': 'module_file_group', 'module': spec.key})
        # Download helpers write temp files first, so cancelled downloads stay retryable.
        all_successful = download_and_check_files(
            **download_kwargs,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        if not all_successful:
            LOGGER.error(f'Please save these files manually to sepcified path and retry, otherwise {spec.key} will be unavailable.')
            return False

    if shared.CACHE_UPDATED:
        shared.dump_cache()

    return all_successful


def prepare_pkuseg(progress_callback=None, cancel_event=None):
    try:
        import pkuseg
    except:
        import spacy_pkuseg as pkuseg

    flist = [
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/postag.zip',
            'files': ['features.pkl', 'weights.npz'],
            'sha256_pre_calculated': ['17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d', '2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'],
            'archived_files': 'postag.zip',
            'save_dir': 'data/models/pkuseg/postag'
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/spacy_ontonotes.zip',
            'files': ['features.msgpack', 'weights.npz'],
            'sha256_pre_calculated': ['fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18', '5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'],
            'archived_files': 'spacy_ontonotes.zip',
            'save_dir': 'data/models/pkuseg/spacy_ontonotes'
        },
    ]
    for files_download_kwargs in flist:
        download_and_check_files(**files_download_kwargs, progress_callback=progress_callback, cancel_event=cancel_event)

    PKUSEG_HOME = osp.join(shared.PROGRAM_PATH, 'data/models/pkuseg')
    pkuseg.config.pkuseg_home = PKUSEG_HOME

    # there must be data/models/pkuseg/postag.zip and data/models/pkuseg/spacy_ontonotes.zip
    # otherwise the dumb package download these models again becuz its dumb checking
    p = osp.join(PKUSEG_HOME, 'postag.zip')
    if not osp.exists(p):
        os.makedirs(p)

    p = osp.join(PKUSEG_HOME, 'spacy_ontonotes.zip')
    if not osp.exists(p):
        os.makedirs(p)
