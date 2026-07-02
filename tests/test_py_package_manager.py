import unittest
from unittest import mock

from ballontranslator.utils.package_installer import InstallResult
from ballontranslator.utils.py_package_manager import (
    ALLOW_RUNTIME_PACKAGE_UPGRADE_ENV,
    PyPackageManager,
    _runtime_package_constraints,
)


class PyPackageManagerTests(unittest.TestCase):

    def test_runtime_package_constraints_pin_installed_core_packages(self):
        constraints = _runtime_package_constraints(
            lambda name: {
                'numpy': '1.26.4',
                'opencv-python': '4.8.1.78',
            }.get(name)
        )

        self.assertEqual(
            constraints,
            ['numpy==1.26.4', 'opencv-python==4.8.1.78'],
        )

    def test_resolve_runtime_requirements_skips_headless_opencv_when_gui_opencv_exists(self):
        manager = PyPackageManager()

        with mock.patch(
            'ballontranslator.utils.py_package_manager._opencv_python_installed',
            return_value=True,
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._onnxruntime_cuda_available',
            return_value=False,
        ):
            resolved = manager.resolve_runtime_requirements([
                'opencv-python-headless>=4.8.1.78',
                'onnxruntime',
            ])

        self.assertEqual(resolved, ['onnxruntime'])

    def test_resolve_runtime_requirements_keeps_headless_opencv_without_gui_opencv(self):
        manager = PyPackageManager()

        with mock.patch(
            'ballontranslator.utils.py_package_manager._opencv_python_installed',
            return_value=False,
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._onnxruntime_cuda_available',
            return_value=False,
        ):
            resolved = manager.resolve_runtime_requirements(['opencv-python-headless>=4.8.1.78'])

        self.assertEqual(resolved, ['opencv-python-headless>=4.8.1.78'])

    def test_build_install_commands_include_runtime_constraint_file(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.py_package_manager._onnxruntime_cuda_available',
            return_value=False,
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._runtime_package_constraints',
            return_value=['numpy==1.26.4', 'opencv-python==4.8.1.78'],
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._write_runtime_constraints_file',
            return_value='/tmp/ballontranslator-runtime-constraints-test.txt',
        ), mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(['onnxruntime'])

        self.assertEqual(len(commands), 1)
        self.assertIn('onnxruntime', commands[0])
        self.assertIn('-c', commands[0])
        self.assertIn('/tmp/ballontranslator-runtime-constraints-test.txt', commands[0])

    def test_runtime_constraint_file_is_passed_to_installer(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.py_package_manager._onnxruntime_cuda_available',
            return_value=False,
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._runtime_package_constraints',
            return_value=['numpy==1.26.4'],
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._write_runtime_constraints_file',
            return_value='/tmp/ballontranslator-runtime-constraints-test.txt',
        ), mock.patch(
            'ballontranslator.utils.package_installer.install',
            return_value=InstallResult(True, ['python', '-m', 'pip', 'install', 'onnxruntime']),
        ) as install:
            result = manager.install(['onnxruntime'])

        self.assertTrue(result.ok)
        self.assertEqual(
            install.call_args.kwargs['constraint_files'],
            ['/tmp/ballontranslator-runtime-constraints-test.txt'],
        )

    def test_runtime_constraints_can_be_disabled_for_manual_upgrade(self):
        manager = PyPackageManager(env={ALLOW_RUNTIME_PACKAGE_UPGRADE_ENV: '1'})

        with mock.patch(
            'ballontranslator.utils.py_package_manager._write_runtime_constraints_file',
        ) as write_constraints:
            self.assertEqual(manager._runtime_constraint_files(), [])

        write_constraints.assert_not_called()


if __name__ == '__main__':
    unittest.main()
