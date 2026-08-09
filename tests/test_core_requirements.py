import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from ballontranslator.utils import core_requirements
from ballontranslator.utils.py_package_manager import MissingRequirement
from ballontranslator.utils.package_installer import InstallResult


class FakeStream:
    def __init__(self, text='installed\n'):
        self.text = text
        self.index = 0

    def read(self, size=1):
        if self.index >= len(self.text):
            return ''
        chunk = self.text[self.index:self.index + size]
        self.index += size
        return chunk


class FakeProcess:
    def __init__(self, text='installed\n', returncode=0):
        self.stdout = FakeStream(text)
        self.returncode = returncode

    def poll(self):
        return self.returncode if self.stdout.index >= len(self.stdout.text) else None

    def wait(self):
        return self.returncode


class CoreRequirementsTests(unittest.TestCase):

    def test_missing_win32gui_forces_pywin32_reinstall(self):
        with mock.patch.object(core_requirements.sys, 'platform', 'win32'), mock.patch(
            'ballontranslator.utils.core_requirements.package_installer.install',
            return_value=InstallResult(True, []),
        ) as install:
            probes = core_requirements._platform_import_probes()
            core_requirements._install_core_requirements_for_failures(
                core_requirements.Path('/tmp/requirements.txt'), [],
                ["win32gui: No module named 'win32gui'"], 'auto', {},
            )

        self.assertIn(('win32gui', ()), probes)
        self.assertEqual(install.call_args.kwargs['requirements'], ['pywin32'])
        self.assertEqual(install.call_args.kwargs['extra_args'], '--force-reinstall')

    def test_healthy_core_imports_do_not_install(self):
        with mock.patch('ballontranslator.utils.core_requirements.check_core_imports', return_value=[]), \
                mock.patch('ballontranslator.utils.core_requirements.install_core_requirements') as install:
            did_install = core_requirements.ensure_core_requirements(repo_root='/tmp/repo')

        self.assertFalse(did_install)
        install.assert_not_called()

    def test_missing_core_import_installs_once(self):
        with mock.patch(
            'ballontranslator.utils.core_requirements.check_core_imports',
            return_value=['numpy: missing'],
        ), mock.patch(
            'ballontranslator.utils.core_requirements.install_core_requirements',
            return_value=InstallResult(True, ['python', '-m', 'pip']),
        ) as install, mock.patch('ballontranslator.utils.core_requirements._drop_probe_modules') as drop:
            did_install = core_requirements.ensure_core_requirements(repo_root='/tmp/repo')

        self.assertTrue(did_install)
        install.assert_called_once()
        drop.assert_called_once()

    def test_missing_requirement_file_entry_installs_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            requirements_path = f'{tmpdir}/requirements.txt'
            with open(requirements_path, 'w', encoding='utf8') as f:
                f.write('spacy-pkuseg\n')

            with mock.patch(
                'ballontranslator.utils.core_requirements.check_core_imports',
                return_value=[],
            ), mock.patch(
                'ballontranslator.utils.py_package_manager.PyPackageManager.missing_requirements',
                return_value=[
                    MissingRequirement('spacy-pkuseg', 'spacy-pkuseg', ['spacy_pkuseg']),
                ],
            ), mock.patch(
                'ballontranslator.utils.core_requirements.install_core_requirements',
                return_value=InstallResult(True, ['python', '-m', 'pip']),
            ) as install, mock.patch('ballontranslator.utils.core_requirements._drop_probe_modules') as drop:
                did_install = core_requirements.ensure_core_requirements(
                    repo_root=tmpdir,
                    requirements_file=requirements_path,
                )

        self.assertTrue(did_install)
        install.assert_called_once()
        drop.assert_called_once()

    def test_requirement_file_failure_installs_before_import_probes(self):
        with mock.patch(
            'ballontranslator.utils.core_requirements.check_core_requirements_file',
            return_value=['numpy>=2: missing package or import (numpy)'],
        ), mock.patch(
            'ballontranslator.utils.core_requirements.check_core_imports',
        ) as check_imports, mock.patch(
            'ballontranslator.utils.core_requirements.install_core_requirements',
            return_value=InstallResult(True, ['python', '-m', 'pip']),
        ) as install, mock.patch('ballontranslator.utils.core_requirements._drop_probe_modules'):
            did_install = core_requirements.ensure_core_requirements(repo_root='/tmp/repo')

        self.assertTrue(did_install)
        install.assert_called_once()
        check_imports.assert_not_called()

    def test_broken_cv2_attr_is_reported(self):
        def import_module(name):
            if name == 'cv2':
                return SimpleNamespace()
            return SimpleNamespace(IMREAD_COLOR=1, IMREAD_GRAYSCALE=0, cvtColor=lambda: None)

        with mock.patch('ballontranslator.utils.core_requirements.importlib.import_module', side_effect=import_module):
            failures = core_requirements.check_core_imports([('cv2', ('IMREAD_COLOR',))])

        self.assertEqual(failures, ['cv2: missing IMREAD_COLOR'])

    def test_failed_core_install_raises_clear_error(self):
        with mock.patch(
            'ballontranslator.utils.core_requirements.check_core_imports',
            return_value=['cv2: missing IMREAD_COLOR'],
        ), mock.patch(
            'ballontranslator.utils.core_requirements.install_core_requirements',
            return_value=InstallResult(False, ['python', '-m', 'pip'], returncode=1, stderr='boom'),
        ):
            with self.assertRaisesRegex(RuntimeError, 'Failed to install core Python requirements'):
                core_requirements.ensure_core_requirements(repo_root='/tmp/repo')

    def test_index_url_reaches_core_installer(self):
        env = {'INDEX_URL': 'https://example.invalid/simple'}
        with mock.patch(
            'ballontranslator.utils.core_requirements.check_core_imports',
            return_value=['qtpy: missing'],
        ), mock.patch(
            'ballontranslator.utils.package_installer.subprocess.Popen',
        ) as popen, mock.patch(
            'ballontranslator.utils.package_installer.LOGGER.info',
        ) as log_info, mock.patch('ballontranslator.utils.core_requirements._drop_probe_modules'):
            popen.return_value = FakeProcess()
            core_requirements.ensure_core_requirements(repo_root='/tmp/repo', env=env)

        command = popen.call_args.args[0]
        self.assertIn('-i', command)
        self.assertIn('https://example.invalid/simple', command)
        log_info.assert_any_call(
            'Using PyPI package mirror for package install: https://example.invalid/simple'
        )


if __name__ == '__main__':
    unittest.main()
