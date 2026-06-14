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

    def test_git_update_message_is_user_facing(self):
        message = updater.format_git_update_message('userspace_update', local_changes_saved=True, local_branch='dev')

        self.assertIn('Local git changes on branch "dev" were saved before updating.', message)
        self.assertIn('The update was applied on branch: userspace_update', message)
        self.assertIn('git switch dev', message)
        self.assertIn('git stash pop', message)
        self.assertNotIn('Saved working directory', message)

    def test_release_payload_requires_source_zip(self):
        info = updater.release_info_from_api_payload({
            'tag_name': 'v1.4.1',
            'name': 'BallonsTranslator 1.4.1',
            'html_url': 'https://example.invalid/release',
            'zipball_url': 'https://example.invalid/source.zip',
            'body': 'Release notes',
            'published_at': '2026-06-13T00:00:00Z',
        })

        self.assertEqual(info.version, '1.4.1')
        self.assertEqual(info.zip_url, 'https://example.invalid/source.zip')
        self.assertEqual(info.name, 'BallonsTranslator 1.4.1')
        self.assertEqual(info.body, 'Release notes')

    def test_check_latest_release_does_not_apply_update(self):
        class FakeUpdater(updater.BallonsTranslatorUpdater):
            def query_latest_release(self):
                return updater.ReleaseInfo(
                    tag_name='v1.4.3',
                    version='1.4.3',
                    html_url='https://example.invalid/release',
                    zip_url='https://example.invalid/source.zip',
                )

            def download_source_zip(self, release_info):
                raise AssertionError('check-only flow should not download')

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'pyproject.toml').write_text('[project]\nversion = "1.4.2"\n', encoding='utf8')

            result = FakeUpdater(program_path=str(root), cache_dir=str(root / '.btrans_cache')).check_latest_release()

        self.assertEqual(result.status, 'available')
        self.assertEqual(result.current_version, '1.4.2')
        self.assertEqual(result.latest_version, '1.4.3')


if __name__ == '__main__':
    unittest.main()
