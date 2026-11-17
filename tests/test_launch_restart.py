import json
import os
import subprocess
import sys
import threading
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ballontranslator import launch


class LaunchRestartTests(unittest.TestCase):

    def test_exception_hooks_log_and_route_only_main_thread_errors_to_dialog(self) -> None:
        from ballontranslator.utils import message
        from ballontranslator.utils.logger import logger

        original_hook = mock.Mock()
        with mock.patch.object(sys, 'excepthook', original_hook), \
                mock.patch.object(threading, 'excepthook'), \
                mock.patch.object(logger, 'error') as log, \
                mock.patch.object(message, 'create_error_dialog') as dialog:
            launch.install_exception_hooks()
            try:
                raise ValueError('test failure')
            except ValueError as error:
                traceback = error.__traceback__
                sys.excepthook(type(error), error, traceback)
                dialog.assert_called_once_with(error, exception_type='UnhandledPythonException')
                log.assert_called_once_with('Unhandled Python exception',
                                            exc_info=(ValueError, error, traceback))
                thread_args = threading.ExceptHookArgs((ValueError, error, traceback, None))
                threading.excepthook(thread_args)
                self.assertEqual(log.call_count, 2)
                self.assertEqual(log.call_args.kwargs['exc_info'], (ValueError, error, traceback))
                dialog.assert_called_once()
            interrupt = KeyboardInterrupt()
            sys.excepthook(KeyboardInterrupt, interrupt, None)
            original_hook.assert_called_once_with(KeyboardInterrupt, interrupt, None)
            self.assertEqual(log.call_count, 2)

    def test_exception_hook_is_safe_before_window_exists_and_when_headless(self) -> None:
        from ballontranslator.utils import shared
        from ballontranslator.utils.logger import logger

        for headless in (False, True):
            with self.subTest(headless=headless), \
                    mock.patch.object(sys, 'excepthook'), \
                    mock.patch.object(threading, 'excepthook'), \
                    mock.patch.object(shared, 'HEADLESS', headless), \
                    mock.patch.object(shared, 'create_errdialog_in_mainthread') as dispatch, \
                    mock.patch.object(shared, 'showed_exception', set()), \
                    mock.patch.object(logger, 'error') as log:
                launch.install_exception_hooks()
                sys.excepthook(ValueError, ValueError('test failure'), None)
                self.assertTrue(log.called)
                self.assertEqual(dispatch.call_count, 0 if headless else 1)

    def test_numba_cache_uses_persistent_app_directory(self):
        self.assertEqual(
            os.environ['NUMBA_CACHE_DIR'],
            os.path.join(launch.shared.cache_dir, 'numba'),
        )

    def test_restart_preserves_module_launch(self):
        main_path = str(Path(launch.__file__).resolve().parent / '__main__.py')
        with mock.patch.object(sys, 'argv', [main_path, '--debug']), \
                mock.patch('ballontranslator.launch.os.execv') as execv:
            launch.restart()

        self.assertEqual(
            execv.call_args.args[1],
            [sys.executable, '-m', 'ballontranslator', '--debug'],
        )

    def test_restart_closes_window_before_replacing_process(self):
        events = []
        window = mock.Mock()
        window.close.side_effect = lambda: events.append('close')

        with mock.patch.object(launch, 'BT', window), \
                mock.patch.object(sys, 'argv', ['ballontranslator']), \
                mock.patch('ballontranslator.launch.os.execv', side_effect=lambda *_: events.append('exec')):
            launch.restart()

        self.assertEqual(events, ['close', 'exec'])

    def test_core_requirements_env_uses_saved_pypi_mirror(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            with open(config_path, 'w', encoding='utf8') as f:
                json.dump({'mirrors': {'pypi': 'https://example.invalid/simple'}}, f)

            env = launch.core_requirements_env(config_path)

        self.assertEqual(env['INDEX_URL'], 'https://example.invalid/simple')

    def test_config_alias_sets_config_path_argument(self):
        config_path = os.path.join('profiles', 'custom.json')

        args = launch.parser.parse_args(['--config', config_path])
        legacy_args = launch.parser.parse_args(['--config_path', config_path])

        self.assertEqual(args.config_path, config_path)
        self.assertEqual(legacy_args.config_path, config_path)

    def test_show_release_info_is_opt_in(self):
        self.assertFalse(launch.parser.parse_args([]).show_release_info)
        self.assertTrue(launch.parser.parse_args(['--show_release_info']).show_release_info)
        self.assertTrue(launch.parser.parse_args(['--show-release-info']).show_release_info)

    def test_load_config_then_save_uses_custom_config_path(self):
        try:
            from ballontranslator.utils import config as program_config
            from ballontranslator.utils import shared
        except ImportError as e:
            self.skipTest(f'config dependencies unavailable: {e}')

        original_path = shared.CONFIG_PATH
        original_config = program_config.pcfg.copy()
        original_created_on_load = program_config.config_created_on_load
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = os.path.join(tmpdir, 'profile', 'custom.json')

                program_config.load_config(config_path)
                program_config.pcfg.display_lang = 'English'
                saved = program_config.save_config()

                self.assertTrue(saved)
                self.assertTrue(os.path.exists(config_path))
                with open(config_path, 'r', encoding='utf8') as f:
                    saved_config = json.load(f)
                self.assertEqual(saved_config['display_lang'], 'English')
        finally:
            shared.CONFIG_PATH = original_path
            program_config.pcfg.merge(original_config)
            program_config.config_created_on_load = original_created_on_load

    def test_bundled_windows_runtime_disables_user_site(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            user_site = str(root / 'AppData' / 'Roaming' / 'Python' / 'Python312' / 'site-packages')
            bundled_executable = str(root / 'ballontrans_pylibs_win' / 'python.exe')
            original_path = sys.path[:]
            original_env = os.environ.get('PYTHONNOUSERSITE')
            import site
            original_enable_user_site = site.ENABLE_USER_SITE

            try:
                sys.path[:] = ['project', user_site, 'bundled-site']
                with mock.patch.object(launch.sys, 'platform', 'win32'), \
                        mock.patch.object(launch.sys, 'executable', bundled_executable), \
                        mock.patch('site.getusersitepackages', return_value=user_site):
                    removed = launch.disable_bundled_windows_user_site()
                    current_path = sys.path[:]
                    current_env = os.environ.get('PYTHONNOUSERSITE')
                    current_enable_user_site = site.ENABLE_USER_SITE
            finally:
                sys.path[:] = original_path
                site.ENABLE_USER_SITE = original_enable_user_site
                if original_env is None:
                    os.environ.pop('PYTHONNOUSERSITE', None)
                else:
                    os.environ['PYTHONNOUSERSITE'] = original_env

            self.assertEqual(removed, [user_site])
            self.assertNotIn(user_site, current_path)
            self.assertEqual(current_env, '1')
            self.assertFalse(current_enable_user_site)

    def test_non_bundled_windows_runtime_keeps_user_site(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            user_site = str(root / 'AppData' / 'Roaming' / 'Python' / 'Python312' / 'site-packages')
            system_executable = str(root / 'Python312' / 'python.exe')
            original_path = sys.path[:]

            try:
                sys.path[:] = ['project', user_site, 'system-site']
                with mock.patch.object(launch.sys, 'platform', 'win32'), \
                        mock.patch.object(launch.sys, 'executable', system_executable), \
                        mock.patch('site.getusersitepackages', return_value=user_site):
                    removed = launch.disable_bundled_windows_user_site()
                    current_path = sys.path[:]
            finally:
                sys.path[:] = original_path

            self.assertEqual(removed, [])
            self.assertIn(user_site, current_path)

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


    def test_font_backend_maps_to_platform_arguments(self) -> None:
        for platform, version, backend, expected in (
            ('win32', '5.15.14', 'gdi', 'windows:nodirectwrite'),
            ('win32', '6.7.3', 'gdi', 'windows:nodirectwrite'),
            ('win32', '6.8.0', 'gdi', 'windows:fontengine=gdi'),
            ('win32', '6.8.0', 'freetype', 'windows:fontengine=freetype'),
            ('darwin', '6.8.0', 'freetype', 'cocoa:fontengine=freetype'),
            ('darwin', '6.8.0', 'gdi', None),
            ('linux', '6.8.0', 'freetype', None),
            ('win32', '6.8.0', 'default', None),
        ):
            with self.subTest(platform=platform, version=version, backend=backend):
                argv = ['ballontranslator']
                result = launch.application_arguments(
                    argv, headless=False, platform_name=platform,
                    qt_version=version, font_backend=backend,
                )
                self.assertEqual(result, argv + (['-platform', expected] if expected else []))
                self.assertEqual(argv, ['ballontranslator'])

    def test_platform_selection_reaches_qt(self) -> None:
        try:
            from qtpy import API_NAME, QT_VERSION
        except ImportError as error:
            self.skipTest(f'Qt dependencies unavailable: {error}')

        qt_probe = (
            'import json, sys\n'
            'from qtpy.QtWidgets import QApplication\n'
            'app = QApplication(json.loads(sys.argv[1]))\n'
            'print(app.platformName())\n'
        )
        for options, headless in (
            (['-platform', 'minimal'], False),
            (['--platform', 'minimal'], False),
            ([], True),
            (['-platform', 'minimal'], True),
            (['--platform', 'minimal'], True),
        ):
            with self.subTest(options=options, headless=headless):
                argv = ['ballontranslator', *options]
                result = launch.application_arguments(
                    argv, headless=headless, platform_name='win32',
                    qt_version=QT_VERSION, font_backend='gdi',
                )
                self.assertEqual(result, argv + (['-platform', 'offscreen'] if headless else []))
                # Each QApplication needs a fresh process to select its platform.
                process = subprocess.run(
                    [sys.executable, '-c', qt_probe, json.dumps(result)],
                    env={**os.environ, 'QT_API': API_NAME.lower(), 'QT_QPA_PLATFORM': 'offscreen'},
                    capture_output=True, text=True, timeout=15,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stdout.strip(), 'offscreen' if headless else 'minimal')

    def test_unsupported_platform_syntax_does_not_suppress_saved_backend(self) -> None:
        for option in ('-platform=minimal', '--platform=minimal'):
            with self.subTest(option=option):
                argv = ['ballontranslator', option]
                result = launch.application_arguments(
                    argv, headless=False, platform_name='win32',
                    qt_version='6.8.0', font_backend='gdi',
                )
                self.assertEqual(result, argv + ['-platform', 'windows:fontengine=gdi'])


if __name__ == '__main__':
    unittest.main()
