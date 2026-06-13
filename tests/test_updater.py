import tempfile
import unittest
from pathlib import Path

from ballontranslator.utils import updater
from ballontranslator.utils.version import _read_pyproject_version


class UpdaterTests(unittest.TestCase):

    def test_pyproject_version_reader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'pyproject.toml'
            path.write_text('[project]\nname = "ballontranslator"\nversion = "1.4.2"\n', encoding='utf8')

            self.assertEqual(_read_pyproject_version(path), '1.4.2')

    def test_release_version_comparison_strips_tag_prefix(self):
        self.assertTrue(updater.is_remote_newer('1.4.0', 'v1.4.1'))
        self.assertFalse(updater.is_remote_newer('1.4.1', 'v1.4.1'))

    def test_release_payload_requires_source_zip(self):
        info = updater.release_info_from_api_payload({
            'tag_name': 'v1.4.1',
            'html_url': 'https://example.invalid/release',
            'zipball_url': 'https://example.invalid/source.zip',
        })

        self.assertEqual(info.version, '1.4.1')
        self.assertEqual(info.zip_url, 'https://example.invalid/source.zip')


if __name__ == '__main__':
    unittest.main()
