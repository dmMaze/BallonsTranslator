import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from ballontranslator.utils import network_mirrors


class NetworkMirrorsTests(unittest.TestCase):

    def test_heuristic_accepts_mainland_locale(self):
        self.assertTrue(network_mirrors.should_use_china_mirrors(locale_names=['zh_CN']))
        self.assertTrue(network_mirrors.should_use_china_mirrors(locale_names=['CN']))

    def test_heuristic_rejects_generic_chinese_and_utc8(self):
        self.assertFalse(network_mirrors.should_use_china_mirrors(locale_names=['zh']))
        self.assertFalse(network_mirrors.should_use_china_mirrors(timezone_names=['UTC+08:00']))

    def test_heuristic_accepts_mainland_timezone_names(self):
        self.assertTrue(network_mirrors.should_use_china_mirrors(timezone_names=['Asia/Shanghai']))
        self.assertTrue(network_mirrors.should_use_china_mirrors(timezone_names=['China Standard Time']))

    def test_huggingface_url_rewrite_is_origin_scoped(self):
        mirror = 'https://hf-mirror.com'
        self.assertEqual(
            network_mirrors.rewrite_huggingface_url(
                'https://huggingface.co/dreMaz/model/resolve/main/file.bin',
                mirror,
            ),
            'https://hf-mirror.com/dreMaz/model/resolve/main/file.bin',
        )
        self.assertEqual(
            network_mirrors.rewrite_huggingface_url('https://example.com/file.bin', mirror),
            'https://example.com/file.bin',
        )

    def test_pypi_mirror_env_injection(self):
        env = network_mirrors.installer_env_with_pypi_mirror(
            {'PATH': os.pathsep.join(['/bin'])},
            'https://pypi.tuna.tsinghua.edu.cn/simple',
        )

        self.assertEqual(env['INDEX_URL'], 'https://pypi.tuna.tsinghua.edu.cn/simple')
        self.assertEqual(env['PATH'], '/bin')

    def test_explicit_null_mirror_fields_are_not_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            with open(path, 'w', encoding='utf8') as f:
                json.dump({'mirrors': {'huggingface': None, 'pypi': None}}, f)

            self.assertEqual(network_mirrors.missing_mirror_fields(path), set())

    def test_missing_mirror_fields_backfill_for_china(self):
        mirrors = SimpleNamespace(huggingface=None, pypi=None)

        updated = network_mirrors.backfill_missing_mirror_defaults(
            mirrors,
            {'huggingface', 'pypi'},
            locale_names=['zh_CN'],
        )

        self.assertEqual(updated, ['huggingface', 'pypi'])
        self.assertEqual(mirrors.huggingface, 'https://hf-mirror.com')
        self.assertEqual(mirrors.pypi, 'https://pypi.tuna.tsinghua.edu.cn/simple')

    def test_explicit_null_is_preserved_when_field_is_not_missing(self):
        mirrors = SimpleNamespace(huggingface=None, pypi=None)

        updated = network_mirrors.backfill_missing_mirror_defaults(
            mirrors,
            set(),
            locale_names=['zh_CN'],
        )

        self.assertEqual(updated, [])
        self.assertIsNone(mirrors.huggingface)
        self.assertIsNone(mirrors.pypi)

    def test_read_saved_pypi_mirror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            with open(path, 'w', encoding='utf8') as f:
                json.dump({'mirrors': {'pypi': 'https://example.invalid/simple'}}, f)

            self.assertEqual(
                network_mirrors.read_saved_pypi_mirror(path),
                'https://example.invalid/simple',
            )

    def test_plain_dict_module_param_can_be_saved(self):
        try:
            from ballontranslator.utils.config import ModuleConfig
        except ModuleNotFoundError as e:
            self.skipTest(f'config dependencies unavailable: {e}')

        cfg = ModuleConfig(textdetector_params={
            'ysgyolo': {
                'label': {'balloon': True, 'other': False},
                'device': 'cuda',
            },
        })

        self.assertEqual(
            cfg.get_params('textdetector', for_saving=True)['ysgyolo']['label'],
            {'balloon': True, 'other': False},
        )

    def test_huggingface_download_rewrite_logs_mirror_use(self):
        try:
            from ballontranslator.utils import download_util
        except ModuleNotFoundError as e:
            self.skipTest(f'download dependencies unavailable: {e}')

        with mock.patch.object(
            download_util,
            '_configured_huggingface_mirror',
            return_value='https://hf-mirror.com',
        ), mock.patch.object(download_util.LOGGER, 'info') as log_info:
            rewritten_url = download_util._rewrite_configured_url(
                'https://huggingface.co/demo/model/resolve/main/file.bin',
                log_mirror=True,
            )

        self.assertEqual(
            rewritten_url,
            'https://hf-mirror.com/demo/model/resolve/main/file.bin',
        )
        log_info.assert_called_once_with(
            'Using Hugging Face mirror for model download: '
            'https://huggingface.co/demo/model/resolve/main/file.bin -> '
            'https://hf-mirror.com/demo/model/resolve/main/file.bin'
        )


if __name__ == '__main__':
    unittest.main()
