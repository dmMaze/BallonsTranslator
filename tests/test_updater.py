import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from ballontranslator.utils import updater
from ballontranslator.utils.version import _read_pyproject_version


def _write_update_root_files(root: Path, marker: str) -> None:
    for filename in updater.SOURCE_UPDATE_FILES:
        (root / filename).write_text(f'{marker} {filename}', encoding='utf8')


def _write_update_dirs(root: Path, marker: str) -> None:
    (root / 'ballontranslator').mkdir(parents=True)
    (root / 'ballontranslator' / '__init__.py').write_text(f'{marker} app', encoding='utf8')
    (root / 'resources').mkdir()
    (root / 'resources' / 'themes.json').write_text(f'{{"{marker}": true}}', encoding='utf8')
    builtin_dir = root / 'config' / 'llm_profile_builtin'
    builtin_dir.mkdir(parents=True)
    (builtin_dir / 'deepseek.yaml').write_text(f'{marker} profile', encoding='utf8')
    (builtin_dir / 'nested').mkdir()
    (builtin_dir / 'nested' / 'extra.yaml').write_text(f'{marker} nested profile', encoding='utf8')


class UpdaterTests(unittest.TestCase):

    def test_managed_payload_includes_builtin_profiles_and_launchers(self):
        self.assertIn('config/llm_profile_builtin', updater.SOURCE_UPDATE_DIRS)
        self.assertNotIn('custom_modules', updater.SOURCE_UPDATE_DIRS)
        self.assertIn('launch_win.bat', updater.SOURCE_UPDATE_FILES)
        self.assertIn('launch.sh', updater.SOURCE_UPDATE_FILES)

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

        with self.assertRaises(RuntimeError):
            updater.release_info_from_api_payload([])

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

    def test_cached_release_preview_does_not_query_github(self):
        payload = {
            'tag_name': 'v1.4.2',
            'html_url': 'https://example.invalid/release',
            'zipball_url': 'https://example.invalid/source.zip',
            'body': 'Cached release notes',
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            cache_dir.mkdir()
            (root / 'pyproject.toml').write_text('[project]\nversion = "1.4.2"\n', encoding='utf8')
            (cache_dir / updater.RELEASE_RESPONSE_CACHE_FILENAME).write_text(
                json.dumps(payload),
                encoding='utf8',
            )

            with mock.patch('ballontranslator.utils.updater.urlopen') as urlopen:
                result = updater.BallonsTranslatorUpdater(
                    program_path=str(root),
                    cache_dir=str(cache_dir),
                ).preview_cached_release()

        urlopen.assert_not_called()
        self.assertEqual(result.status, 'preview')
        self.assertEqual(result.release_info.body, 'Cached release notes')

    def test_successful_github_query_caches_raw_response(self):
        payload = {
            'tag_name': 'v1.4.3',
            'html_url': 'https://example.invalid/release',
            'zipball_url': 'https://example.invalid/source.zip',
            'body': 'Fresh release notes',
        }
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode('utf8')

        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch('ballontranslator.utils.updater.urlopen', return_value=response):
            cache_dir = Path(tmpdir) / '.btrans_cache'
            info = updater.BallonsTranslatorUpdater(
                program_path=tmpdir,
                cache_dir=str(cache_dir),
            ).query_latest_release()
            cached_payload = json.loads(
                (cache_dir / updater.RELEASE_RESPONSE_CACHE_FILENAME).read_text(encoding='utf8')
            )

        self.assertEqual(info.version, '1.4.3')
        self.assertEqual(cached_payload, payload)

    def test_backup_source_replaces_last_version_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            _write_update_dirs(root, 'current')
            _write_update_root_files(root, 'current')
            stale_backup = cache_dir / 'last_version'
            stale_backup.mkdir(parents=True)
            (stale_backup / 'stale.txt').write_text('stale', encoding='utf8')

            backup_path = updater.BallonsTranslatorUpdater(
                program_path=str(root),
                cache_dir=str(cache_dir),
            ).backup_source()

            self.assertEqual(backup_path, stale_backup)
            self.assertFalse((backup_path / 'stale.txt').exists())
            self.assertFalse((cache_dir / '.last_version_tmp').exists())
            for dirname in updater.SOURCE_UPDATE_DIRS:
                source_dir = root / dirname
                if not source_dir.exists():
                    continue
                self.assertTrue((backup_path / dirname).is_dir())
                for source_file in source_dir.rglob('*'):
                    if source_file.is_file():
                        relative_path = source_file.relative_to(root)
                        self.assertEqual(
                            (backup_path / relative_path).read_bytes(),
                            source_file.read_bytes(),
                        )
            for filename in updater.SOURCE_UPDATE_FILES:
                self.assertEqual(
                    (backup_path / filename).read_bytes(),
                    (root / filename).read_bytes(),
                )

    def test_apply_update_creates_backup_before_other_actions(self):
        calls = []

        class RecordingUpdater(updater.BallonsTranslatorUpdater):
            def backup_source(self):
                calls.append('backup')
                return self.cache_dir / 'last_version'

            def download_source_zip(self, release_info):
                calls.append('download')
                return self.cache_dir / 'source.zip'

            def prepare_git_worktree(self, latest_version):
                calls.append('git')
                return ''

            def install_source_zip(self, zip_path):
                calls.append('install')

        release_info = updater.ReleaseInfo(
            tag_name='v1.4.3',
            version='1.4.3',
            html_url='https://example.invalid/release',
            zip_url='https://example.invalid/source.zip',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = RecordingUpdater(
                program_path=tmpdir,
                cache_dir=str(Path(tmpdir) / '.btrans_cache'),
            ).apply_update(release_info, current_version='1.4.2')

        self.assertEqual(calls, ['backup', 'download', 'git', 'install'])
        self.assertEqual(result.status, 'updated')

    def test_backup_failure_stops_update_before_download(self):
        class BackupFailingUpdater(updater.BallonsTranslatorUpdater):
            def backup_source(self):
                raise OSError('backup failed')

            def download_source_zip(self, release_info):
                raise AssertionError('download must not start after a backup failure')

        release_info = updater.ReleaseInfo(
            tag_name='v1.4.3',
            version='1.4.3',
            html_url='https://example.invalid/release',
            zip_url='https://example.invalid/source.zip',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(OSError, 'backup failed'):
                BackupFailingUpdater(
                    program_path=tmpdir,
                    cache_dir=str(Path(tmpdir) / '.btrans_cache'),
                ).apply_update(release_info, current_version='1.4.2')

    def test_install_source_zip_removes_downloaded_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            _write_update_dirs(root, 'old')
            (root / 'custom_modules').mkdir()
            (root / 'custom_modules' / 'trans_user.py').write_text('user module', encoding='utf8')
            (root / 'config' / 'textstyles').mkdir()
            (root / 'config' / 'textstyles' / 'user.json').write_text('local textstyle', encoding='utf8')
            _write_update_root_files(root, 'old')

            release_root = root / 'release' / 'BallonsTranslator-1.5.1'
            _write_update_dirs(release_root, 'new')
            (release_root / 'config' / 'textstyles').mkdir()
            (release_root / 'config' / 'textstyles' / 'user.json').write_text('release textstyle', encoding='utf8')
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
            self.assertEqual(
                (root / 'custom_modules' / 'trans_user.py').read_text(encoding='utf8'),
                'user module',
            )
            self.assertEqual((root / 'resources' / 'themes.json').read_text(encoding='utf8'), '{"new": true}')
            self.assertEqual((root / 'config' / 'llm_profile_builtin' / 'deepseek.yaml').read_text(encoding='utf8'), 'new profile')
            self.assertEqual(
                (root / 'config' / 'llm_profile_builtin' / 'nested' / 'extra.yaml').read_text(encoding='utf8'),
                'new nested profile',
            )
            self.assertEqual((root / 'config' / 'textstyles' / 'user.json').read_text(encoding='utf8'), 'local textstyle')
            for filename in updater.SOURCE_UPDATE_FILES:
                self.assertEqual((root / filename).read_text(encoding='utf8'), f'new {filename}')
            self.assertFalse(zip_path.exists())
            self.assertFalse((cache_dir / 'BallonsTranslator_1.5.1_source_extracted').exists())

    def test_install_source_zip_deletes_removed_managed_directory_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            _write_update_dirs(root, 'old')
            (root / 'config' / 'textstyles').mkdir()
            (root / 'config' / 'textstyles' / 'user.json').write_text('local textstyle', encoding='utf8')
            _write_update_root_files(root, 'old')

            release_root = root / 'release' / 'BallonsTranslator-1.5.1'
            _write_update_dirs(release_root, 'new')
            shutil.rmtree(release_root / 'config' / 'llm_profile_builtin')
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

            self.assertFalse((root / 'config' / 'llm_profile_builtin').exists())
            self.assertTrue((root / 'config').is_dir())
            self.assertEqual((root / 'config' / 'textstyles' / 'user.json').read_text(encoding='utf8'), 'local textstyle')
            self.assertEqual((root / 'ballontranslator' / '__init__.py').read_text(encoding='utf8'), 'new app')
            self.assertEqual((root / 'resources' / 'themes.json').read_text(encoding='utf8'), '{"new": true}')

    def test_install_source_zip_ignores_cleanup_failure(self):
        class CleanupFailingUpdater(updater.BallonsTranslatorUpdater):
            def _cleanup_downloaded_source(self, zip_path, extract_root):
                raise OSError('cleanup failed')

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / '.btrans_cache'
            _write_update_dirs(root, 'old')
            _write_update_root_files(root, 'old')

            release_root = root / 'release' / 'BallonsTranslator-1.5.1'
            _write_update_dirs(release_root, 'new')
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
            self.assertEqual((root / 'config' / 'llm_profile_builtin' / 'deepseek.yaml').read_text(encoding='utf8'), 'new profile')


if __name__ == '__main__':
    unittest.main()
