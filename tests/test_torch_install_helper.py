import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ballontranslator.utils import package_installer
from ballontranslator.utils import shared
from ballontranslator.utils.py_package_manager import PyPackageManager
from ballontranslator.utils.torch_install_helper import (
    ALIYUN_PYPI_MIRROR,
    IntelXpuInfo,
    NvidiaGpuInfo,
    _parse_xpu_smi_discovery_json,
    _parse_xpu_smi_discovery_text,
    has_plain_unpinned_torch,
    prepare_torch_install_request,
)


class TorchInstallHelperTests(unittest.TestCase):

    def setUp(self):
        shared.TORCH_INSTALL_PREFERRED_DEVICE = None
        shared.TORCH_INSTALL_PREFERRED_PROFILE = None

    def tearDown(self):
        shared.TORCH_INSTALL_PREFERRED_DEVICE = None
        shared.TORCH_INSTALL_PREFERRED_PROFILE = None

    def test_xpu_discovery_routes_plain_torch_to_xpu_index(self):
        request = prepare_torch_install_request(
            ['torch', 'torchvision', 'einops'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
        )

        self.assertEqual(request.profile.name, 'xpu')
        self.assertEqual(request.backend, 'pip')
        self.assertEqual(
            request.requirements,
            ['torch', 'torchvision', 'einops'],
        )
        self.assertEqual(request.env['INDEX_URL'], 'https://download.pytorch.org/whl/xpu')
        self.assertNotIn('FIND_LINKS', request.env)

    def test_nvidia_profile_takes_priority_over_xpu_profile(self):
        request = prepare_torch_install_request(
            ['torch'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
        )

        self.assertEqual(request.profile.name, 'cu128')

    def test_pinned_torch_requirement_does_not_route_to_xpu_index(self):
        request = prepare_torch_install_request(
            ['torch==2.7.1', 'torchvision'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
        )

        self.assertIsNone(request.profile)
        self.assertEqual(request.requirements, ['torch==2.7.1', 'torchvision'])

    def test_forced_cpu_uses_cpu_profile(self):
        request = prepare_torch_install_request(
            ['torch', 'einops'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
            torch_device='cpu',
        )

        self.assertEqual(request.profile.name, 'cpu')
        self.assertEqual(request.backend, 'pip')
        self.assertEqual(request.device, 'cpu')
        self.assertEqual(
            request.requirements,
            ['torch==2.10.0', 'torchvision==0.25.0', 'einops'],
        )
        self.assertEqual(request.env['INDEX_URL'], 'https://download.pytorch.org/whl/cpu')
        self.assertNotIn('FIND_LINKS', request.env)

    def test_cpu_profile_uses_aliyun_cpu_wheels_with_pypi_mirror(self):
        manager = PyPackageManager(
            backend='auto',
            env={'PATH': '/bin', 'INDEX_URL': ALIYUN_PYPI_MIRROR},
        )

        command = manager.build_install_commands(['torch'], torch_device='cpu')[0]

        self.assertIn('torch==2.10.0', command)
        self.assertIn(ALIYUN_PYPI_MIRROR, command)
        self.assertIn('https://mirrors.aliyun.com/pytorch-wheels/cpu', command)
        self.assertNotIn('--extra-index-url', command)

    def test_auto_cpu_keeps_default_install_path_on_macos(self):
        with mock.patch(
            'ballontranslator.utils.torch_install_helper.sys.platform',
            'darwin',
        ):
            request = prepare_torch_install_request(
                ['torch'],
                env={'PATH': '/bin'},
                gpu_detector=lambda: [],
                xpu_detector=lambda: [],
            )

        self.assertIsNone(request.profile)
        self.assertEqual(request.requirements, ['torch'])

    def test_forced_cuda_uses_default_cuda_profile_without_probe_result(self):
        request = prepare_torch_install_request(
            ['torch'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [],
            xpu_detector=lambda: [],
            torch_device='cuda',
        )

        self.assertEqual(request.profile.name, 'cu128')
        self.assertEqual(request.device, 'cuda')
        self.assertEqual(request.cuda_version, 'cu128')

    def test_forced_cuda_version_selects_matching_profile(self):
        request = prepare_torch_install_request(
            ['torch'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
            torch_device='cuda',
            torch_cuda_version='cu118',
        )

        self.assertEqual(request.profile.name, 'cu118')
        self.assertEqual(request.cuda_version, 'cu118')

    def test_forced_xpu_uses_xpu_profile_without_probe_result(self):
        request = prepare_torch_install_request(
            ['torch'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [],
            xpu_detector=lambda: [],
            torch_device='xpu',
        )

        self.assertEqual(request.profile.name, 'xpu')
        self.assertEqual(request.device, 'xpu')

    def test_xpu_profile_uses_official_index_with_aliyun_pypi_mirror(self):
        request = prepare_torch_install_request(
            ['torch', 'einops'],
            env={'PATH': '/bin', 'INDEX_URL': ALIYUN_PYPI_MIRROR},
            gpu_detector=lambda: [],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
        )

        self.assertEqual(request.env['INDEX_URL'], 'https://download.pytorch.org/whl/xpu')
        self.assertNotIn('FIND_LINKS', request.env)

    def test_auto_detection_caches_preferred_device(self):
        calls = {'nvidia': 0, 'xpu': 0}

        def detect_nvidia():
            calls['nvidia'] += 1
            return []

        def detect_xpu():
            calls['xpu'] += 1
            return [IntelXpuInfo('Intel Arc')]

        with mock.patch(
            'ballontranslator.utils.torch_install_helper.detect_nvidia_gpus',
            side_effect=detect_nvidia,
        ), mock.patch(
            'ballontranslator.utils.torch_install_helper.detect_intel_xpus',
            side_effect=detect_xpu,
        ):
            first = prepare_torch_install_request(['torch'], env={'PATH': '/bin'})
            second = prepare_torch_install_request(['torch'], env={'PATH': '/bin'})

        self.assertEqual(first.device, 'xpu')
        self.assertEqual(second.device, 'xpu')
        self.assertEqual(calls, {'nvidia': 1, 'xpu': 1})

    def test_plain_unpinned_torch_requirement_detection(self):
        self.assertTrue(has_plain_unpinned_torch(['torch', 'einops']))
        self.assertFalse(has_plain_unpinned_torch(['torch==2.7.1']))
        self.assertFalse(has_plain_unpinned_torch(['torch; python_version < "3.12"']))

    def test_parse_xpu_smi_discovery_json(self):
        output = '''
        {
            "device_list": [
                {"device_id": 0, "device_type": "GPU", "device_name": "Intel Arc"},
                {"device_id": 1, "device_type": "CPU", "device_name": "Intel CPU"}
            ]
        }
        '''

        self.assertEqual(
            _parse_xpu_smi_discovery_json(output),
            [IntelXpuInfo('Intel Arc', '0')],
        )

    def test_parse_xpu_smi_discovery_text(self):
        output = '''
        +----+-----------------------------------+
        | 0  | Intel(R) Arc(TM) A770 Graphics    |
        +----+-----------------------------------+
        '''

        self.assertEqual(
            _parse_xpu_smi_discovery_text(output),
            [IntelXpuInfo('Intel(R) Arc(TM) A770 Graphics')],
        )

    def test_package_manager_builds_split_xpu_install_commands(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.torch_install_helper.detect_nvidia_gpus',
            return_value=[],
        ), mock.patch(
            'ballontranslator.utils.torch_install_helper.detect_intel_xpus',
            return_value=[IntelXpuInfo('Intel Arc')],
        ), mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(['torch', 'einops'])

        self.assertEqual(len(commands), 2)
        torch_command, other_command = commands
        self.assertIn('torch', torch_command)
        self.assertIn('torchvision', torch_command)
        self.assertNotIn('einops', torch_command)
        self.assertIn('-i', torch_command)
        self.assertIn('https://download.pytorch.org/whl/xpu', torch_command)
        self.assertIn('einops', other_command)
        self.assertNotIn('https://download.pytorch.org/whl/xpu', other_command)

    def test_package_manager_forced_cpu_builds_cpu_install_command(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(['torch', 'einops'], torch_device='cpu')

        self.assertEqual(len(commands), 2)
        torch_command, other_command = commands
        self.assertIn('torch==2.10.0', torch_command)
        self.assertIn('https://download.pytorch.org/whl/cpu', torch_command)
        self.assertIn('-i', torch_command)
        self.assertNotIn('-f', torch_command)
        self.assertNotIn('einops', torch_command)
        self.assertIn('einops', other_command)
        self.assertNotIn('--force-reinstall', torch_command)
        self.assertNotIn('https://download.pytorch.org/whl/xpu', torch_command)

    def test_package_manager_uninstalls_existing_torch_packages_before_install(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch.object(
            manager,
            '_runtime_constraint_files',
            return_value=[],
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._distribution_installed',
            side_effect=lambda package: package in {'torch', 'torchvision'},
        ), mock.patch(
            'ballontranslator.utils.py_package_manager.subprocess.run',
            return_value=mock.Mock(returncode=0, stdout='removed'),
        ) as uninstall, mock.patch(
            'ballontranslator.utils.py_package_manager.package_installer.install',
            return_value=package_installer.InstallResult(True, []),
        ) as install:
            result = manager.install(['torch'], torch_device='cpu')

        self.assertTrue(result.ok)
        self.assertEqual(
            uninstall.call_args.args[0][3:],
            ['uninstall', '-y', 'torch', 'torchvision'],
        )
        self.assertNotIn('--force-reinstall', install.call_args.kwargs['extra_args'])

    def test_package_manager_stops_when_torch_uninstall_fails(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})
        with mock.patch(
            'ballontranslator.utils.py_package_manager._distribution_installed',
            side_effect=lambda package: package == 'torch',
        ), mock.patch(
            'ballontranslator.utils.py_package_manager.subprocess.run',
            return_value=mock.Mock(returncode=1, stdout='locked'),
        ), mock.patch(
            'ballontranslator.utils.py_package_manager.package_installer.install',
        ) as install:
            result = manager.install(['torch'], torch_device='cpu')

        self.assertFalse(result.ok)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, 'locked')
        install.assert_not_called()

    def test_pinned_torch_uninstalls_only_the_requested_family_package(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.py_package_manager._distribution_installed',
            return_value=True,
        ):
            requests = manager._prepare_install_requests(['torch==2.7.1'])
            command = manager._torch_uninstall_command(requests)

        self.assertEqual(command[3:], ['uninstall', '-y', 'torch'])

    def test_cpu_profile_uses_pip_for_pytorch_index(self):
        manager = PyPackageManager(backend='uv', env={'PATH': '/bin'})

        commands = manager.build_install_commands(['torch'], torch_device='cpu')

        self.assertIn('-m', commands[0])
        self.assertIn('pip', commands[0])
        self.assertIn('https://download.pytorch.org/whl/cpu', commands[0])

    def test_package_manager_forced_cuda_version_builds_matching_command(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(
                ['torch', 'einops'],
                torch_device='cuda',
                torch_cuda_version='cu118',
            )

        self.assertEqual(len(commands), 2)
        self.assertIn('torch==2.7.1', commands[0])
        self.assertIn('https://download.pytorch.org/whl/cu118', commands[0])
        self.assertIn('einops', commands[1])

    def test_package_manager_forced_cuda_12_caps_onnxruntime_gpu(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(
                ['torch', 'onnxruntime'],
                torch_device='cuda',
                torch_cuda_version='cu128',
            )

        self.assertEqual(len(commands), 2)
        self.assertIn('torch==2.10.0', commands[0])
        self.assertIn('onnxruntime-gpu<1.27.0', commands[1])
        self.assertNotIn('onnxruntime-gpu>=1.27.0', commands[1])

    def test_package_manager_rejects_versioned_onnxruntime_gpu_dependency(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with self.assertRaisesRegex(ValueError, 'Do not specify onnxruntime-gpu versions'):
            manager.build_install_commands(
                ['torch', 'onnxruntime-gpu>=1.27.0'],
                torch_device='cuda',
                torch_cuda_version='cu128',
            )

    def test_package_manager_reports_original_onnxruntime_requirement_as_missing(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch.object(
            PyPackageManager,
            '_requirement_satisfied',
            return_value=False,
        ), mock.patch.object(
            PyPackageManager,
            '_import_available',
            return_value=False,
        ), mock.patch(
            'ballontranslator.utils.py_package_manager._resolve_onnxruntime_requirement',
            return_value='onnxruntime-gpu>=1.27.0',
        ):
            missing = manager.missing_requirements(['onnxruntime', 'torch'])

        self.assertEqual([item.requirement for item in missing], ['onnxruntime', 'torch'])

    def test_package_manager_auto_uses_bundled_uv_for_non_torch_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_dir = Path(tmpdir) / 'ballontrans_pylibs_win'
            runtime_dir.mkdir()
            python_path = runtime_dir / 'python.exe'
            uv_path = runtime_dir / 'uv.exe'
            python_path.write_text('', encoding='utf8')
            uv_path.write_text('', encoding='utf8')
            manager = PyPackageManager(backend='auto', env={'PATH': ''})

            with mock.patch.object(
                package_installer.sys,
                'executable',
                str(python_path),
            ), mock.patch(
                'ballontranslator.utils.package_installer._pip_supports_raw_progress',
                return_value=False,
            ):
                commands = manager.build_install_commands(
                    ['torch', 'einops'],
                    torch_device='cuda',
                    torch_cuda_version='cu128',
                )

        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][0], str(python_path))
        self.assertEqual(commands[1][0], str(uv_path))
        self.assertIn('einops', commands[1])


if __name__ == '__main__':
    unittest.main()
