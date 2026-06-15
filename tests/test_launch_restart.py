import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ballontranslator import launch


class LaunchRestartTests(unittest.TestCase):

    def test_restart_preserves_module_launch(self):
        main_path = str(Path(launch.__file__).resolve().parent / '__main__.py')
        with mock.patch.object(sys, 'argv', [main_path, '--debug']), \
                mock.patch('ballontranslator.launch.os.execv') as execv:
            launch.restart()

        self.assertEqual(
            execv.call_args.args[1],
            [sys.executable, '-m', 'ballontranslator', '--debug'],
        )

    def test_core_requirements_env_uses_saved_pypi_mirror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w', encoding='utf8') as f:
                json.dump({'mirrors': {'pypi': 'https://example.invalid/simple'}}, f)

            env = launch.core_requirements_env(config_path)

        self.assertEqual(env['INDEX_URL'], 'https://example.invalid/simple')

    def test_resource_theme_fallback_copies_old_config_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'config').mkdir()
            (root / 'config' / 'stylesheet.css').write_text('old css', encoding='utf8')
            (root / 'config' / 'themes.json').write_text('{"old": true}', encoding='utf8')

            copied = launch.ensure_resource_theme_files(str(root))

            self.assertEqual(copied, ['stylesheet.css', 'themes.json'])
            self.assertEqual((root / 'resources' / 'stylesheet.css').read_text(encoding='utf8'), 'old css')
            self.assertEqual((root / 'resources' / 'themes.json').read_text(encoding='utf8'), '{"old": true}')

    def test_resource_theme_fallback_does_not_overwrite_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / 'config').mkdir()
            (root / 'resources').mkdir()
            (root / 'config' / 'stylesheet.css').write_text('old css', encoding='utf8')
            (root / 'resources' / 'stylesheet.css').write_text('new css', encoding='utf8')

            copied = launch.ensure_resource_theme_files(str(root))

            self.assertEqual(copied, [])
            self.assertEqual((root / 'resources' / 'stylesheet.css').read_text(encoding='utf8'), 'new css')


if __name__ == '__main__':
    unittest.main()
