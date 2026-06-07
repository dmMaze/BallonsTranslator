import sys
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


if __name__ == '__main__':
    unittest.main()
