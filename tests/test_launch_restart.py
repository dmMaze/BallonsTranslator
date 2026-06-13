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


if __name__ == '__main__':
    unittest.main()
