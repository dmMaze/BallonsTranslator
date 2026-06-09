import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

from ballontranslator.utils import package_installer


CORE_IMPORT_PROBES = (
    ('packaging', ()),
    ('qtpy', ()),
    ('qtpy.QtCore', ('Qt',)),
    ('numpy', ()),
    ('PIL', ()),
    ('pillow_jxl', ()),
    ('requests', ()),
    ('tqdm', ()),
    ('natsort', ()),
    ('cv2', ('IMREAD_COLOR', 'IMREAD_GRAYSCALE', 'cvtColor')),
)


def _platform_import_probes() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    if sys.platform == 'win32':
        return (('win32api', ()),)
    if sys.platform == 'darwin':
        return (('objc', ()), ('Cocoa', ()), ('Quartz', ()),)
    return ()


def check_core_imports(probes: Iterable[Tuple[str, Iterable[str]]] = None) -> List[str]:
    """Return failed core import probes.

    >>> check_core_imports([('math', ('sqrt',))])
    []
    >>> check_core_imports([('math', ('not_a_real_attr',))])
    ['math: missing not_a_real_attr']
    """

    failures = []
    probes = tuple(probes or CORE_IMPORT_PROBES) + _platform_import_probes()
    for module_name, attrs in probes:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            failures.append(f'{module_name}: {e}')
            continue
        missing_attrs = [attr for attr in attrs if not hasattr(module, attr)]
        if missing_attrs:
            failures.append(f'{module_name}: missing {", ".join(missing_attrs)}')
    return failures


def _drop_probe_modules(probes: Iterable[Tuple[str, Iterable[str]]]):
    for module_name, _ in probes:
        root = module_name.split('.', 1)[0]
        for loaded_name in list(sys.modules):
            if loaded_name == root or loaded_name.startswith(root + '.'):
                sys.modules.pop(loaded_name, None)
    importlib.invalidate_caches()


def install_core_requirements(
    requirements_file: str,
    backend: str = 'auto',
    env: dict = None,
    extra_args: str = '',
):
    """Install the core requirements file through the shared installer.

    >>> from unittest import mock
    >>> with mock.patch('ballontranslator.utils.package_installer.install') as install:
    ...     install.return_value = 'ok'
    ...     install_core_requirements('/tmp/requirements.txt', backend='pip')
    'ok'
    >>> install.call_args.kwargs['requirements_file']
    '/tmp/requirements.txt'
    """

    return package_installer.install(
        requirements_file=requirements_file,
        backend=backend,
        extra_args=extra_args,
        env=env,
    )


def ensure_core_requirements(
    repo_root: str = '',
    requirements_file: str = '',
    backend: str = 'auto',
    env: dict = None,
    force: bool = False,
) -> bool:
    """Repair core launch requirements when probes fail.

    The launch path restarts after this returns ``True`` because native modules
    can be left in a bad import state after a failed probe.

    >>> from unittest import mock
    >>> with mock.patch(f'{__name__}.check_core_imports', return_value=[]):
    ...     ensure_core_requirements(repo_root='/tmp/repo')
    False
    """

    if getattr(sys, 'frozen', False):
        return False

    repo_path = Path(repo_root or Path(__file__).resolve().parents[2])
    requirements_path = Path(requirements_file) if requirements_file else repo_path / 'requirements.txt'
    probes = tuple(CORE_IMPORT_PROBES) + _platform_import_probes()
    failures = check_core_imports(probes)
    if not force and not failures:
        return False

    print('Core Python requirements are missing or invalid.')
    if failures:
        print('Missing/invalid core imports:')
        for failure in failures:
            print(f'  - {failure}')
    print(f'Installing core requirements from {requirements_path}...')

    result = install_core_requirements(
        str(requirements_path),
        backend=backend,
        env=env or os.environ.copy(),
    )
    if not result.ok:
        raise RuntimeError(
            'Failed to install core Python requirements.\n'
            f'Command: {result.command_text}\n'
            f'Exit code: {result.returncode}\n'
            f'{result.stderr or result.stdout or result.error}'
        )

    _drop_probe_modules(probes)
    print('Core Python requirements were installed.')
    return True
