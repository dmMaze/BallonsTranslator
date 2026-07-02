import tempfile
import unittest
import zipfile
from pathlib import Path

from ballontranslator.utils import updater
from ballontranslator.utils.version import _read_pyproject_version


def _write_update_root_files(root: Path, marker: str) -> None:
    for filename in updater.SOURCE_UPDATE_FILES:
        (root / filename).write_text(f'{marker} {filename}', encoding='utf8')


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

    def test_backup_source_includes_root_update_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            (root / 'ballontranslator').mkdir()
            (root / 'ballontranslator' / '__init__.py').write_text('app', encoding='utf8')
            (root / 'resources').mkdir()
            (root / 'resources' / 'themes.json').write_text('{}', encoding='utf8')
            _write_update_root_files(root, 'current')

            backup_path = updater.BallonsTranslatorUpdater(
                program_path=str(root),
                cache_dir=str(cache_dir),
            ).backup_source('1.5.0')

            self.assertEqual((backup_path / 'ballontranslator' / '__init__.py').read_text(encoding='utf8'), 'app')
            self.assertEqual((backup_path / 'resources' / 'themes.json').read_text(encoding='utf8'), '{}')
            for filename in updater.SOURCE_UPDATE_FILES:
                self.assertEqual((backup_path / filename).read_text(encoding='utf8'), f'current {filename}')

    def test_install_source_zip_removes_downloaded_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            (root / 'ballontranslator').mkdir()
            (root / 'ballontranslator' / '__init__.py').write_text('old app', encoding='utf8')
            (root / 'resources').mkdir()
            (root / 'resources' / 'themes.json').write_text('{"old": true}', encoding='utf8')
            _write_update_root_files(root, 'old')

            release_root = root / 'release' / 'BallonsTranslator-1.5.1'
            (release_root / 'ballontranslator').mkdir(parents=True)
            (release_root / 'ballontranslator' / '__init__.py').write_text('new app', encoding='utf8')
            (release_root / 'resources').mkdir()
            (release_root / 'resources' / 'themes.json').write_text('{"new": true}', encoding='utf8')
            _write_update_root_files(release_root, 'new')
            zip_path = cache_dir / 'BallonsTranslator_1.5.1_source.zip'
            cache_dir.mkdir()
            with zipfile.ZipFile(zip_path, 'w') as archive:
                for path in release_root.rglob('*'):
                    archive.write(path, path.relative_to(root / 'release'))

            updater.BallonsTranslatorUpdater(
                program_path=str(root),
                cache_dir=str(cache_dir),
            ).install_source_zip(zip_path)

            self.assertEqual((root / 'ballontranslator' / '__init__.py').read_text(encoding='utf8'), 'new app')
            self.assertEqual((root / 'resources' / 'themes.json').read_text(encoding='utf8'), '{"new": true}')
            for filename in updater.SOURCE_UPDATE_FILES:
                self.assertEqual((root / filename).read_text(encoding='utf8'), f'new {filename}')
            self.assertFalse(zip_path.exists())
            self.assertFalse((cache_dir / 'BallonsTranslator_1.5.1_source_extracted').exists())

    def test_install_source_zip_ignores_cleanup_failure(self):
        class CleanupFailingUpdater(updater.BallonsTranslatorUpdater):
            def _cleanup_downloaded_source(self, zip_path, extract_root):
                raise OSError('cleanup failed')

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            (root / 'ballontranslator').mkdir()
            (root / 'ballontranslator' / '__init__.py').write_text('old app', encoding='utf8')
            (root / 'resources').mkdir()
            (root / 'resources' / 'themes.json').write_text('{"old": true}', encoding='utf8')
            _write_update_root_files(root, 'old')

            release_root = root / 'release' / 'BallonsTranslator-1.5.1'
            (release_root / 'ballontranslator').mkdir(parents=True)
            (release_root / 'ballontranslator' / '__init__.py').write_text('new app', encoding='utf8')
            (release_root / 'resources').mkdir()
            (release_root / 'resources' / 'themes.json').write_text('{"new": true}', encoding='utf8')
            _write_update_root_files(release_root, 'new')
            zip_path = cache_dir / 'BallonsTranslator_1.5.1_source.zip'
            cache_dir.mkdir()
            with zipfile.ZipFile(zip_path, 'w') as archive:
                for path in release_root.rglob('*'):
                    archive.write(path, path.relative_to(root / 'release'))

            CleanupFailingUpdater(
                program_path=str(root),
                cache_dir=str(cache_dir),
            ).install_source_zip(zip_path)

            self.assertEqual((root / 'ballontranslator' / '__init__.py').read_text(encoding='utf8'), 'new app')
            self.assertEqual((root / 'resources' / 'themes.json').read_text(encoding='utf8'), '{"new": true}')


if __name__ == '__main__':
    unittest.main()
