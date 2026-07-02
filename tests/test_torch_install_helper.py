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
            ['torch', 'torchvision', 'torchaudio', 'einops'],
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

    def test_forced_cpu_keeps_plain_torch_install(self):
        request = prepare_torch_install_request(
            ['torch', 'einops'],
            env={'PATH': '/bin'},
            gpu_detector=lambda: [NvidiaGpuInfo('RTX 4090', 8.9)],
            xpu_detector=lambda: [IntelXpuInfo('Intel Arc')],
            torch_device='cpu',
        )

        self.assertIsNone(request.profile)
        self.assertEqual(request.device, 'cpu')
        self.assertEqual(request.requirements, ['torch', 'einops'])

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
        self.assertIn('torchaudio', torch_command)
        self.assertNotIn('einops', torch_command)
        self.assertIn('-i', torch_command)
        self.assertIn('https://download.pytorch.org/whl/xpu', torch_command)
        self.assertIn('einops', other_command)
        self.assertNotIn('https://download.pytorch.org/whl/xpu', other_command)

    def test_package_manager_forced_cpu_builds_plain_install_command(self):
        manager = PyPackageManager(backend='pip', env={'PATH': '/bin'})

        with mock.patch(
            'ballontranslator.utils.package_installer._pip_supports_raw_progress',
            return_value=False,
        ):
            commands = manager.build_install_commands(['torch', 'einops'], torch_device='cpu')

        self.assertEqual(len(commands), 1)
        self.assertIn('torch', commands[0])
        self.assertIn('einops', commands[0])
        self.assertNotIn('https://download.pytorch.org/whl/xpu', commands[0])

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
