import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.modules import codex
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.utils import config as program_config, shared
from ballontranslator.utils.config import ProgramConfig, json_dump_program_config, pcfg


class FontBackendConfigTests(unittest.TestCase):
    def test_original_pr_config_keeps_both_platform_choices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            backends = {'windows': 'gdi', 'macos': 'freetype'}
            with open(path, 'w', encoding='utf8') as stream:
                json.dump({'font_backend': backends}, stream)
            config = ProgramConfig.load(path)
            self.assertEqual(config.font_backend, backends)
            self.assertEqual(json.loads(json_dump_program_config(config))['font_backend'], backends)

    def test_new_configs_do_not_share_backend_preferences(self) -> None:
        first, second = ProgramConfig(), ProgramConfig()
        first.font_backend['windows'] = 'gdi'
        self.assertEqual(second.font_backend, {'windows': 'default', 'macos': 'default'})

    def test_loading_defaults_and_invalid_values_does_not_rewrite_file(self) -> None:
        defaults = {'windows': 'default', 'macos': 'default'}
        cases = [({}, defaults, False)]
        cases.extend(({'font_backend': value}, defaults, True)
                     for value in (None, False, 12, [], 'gdi'))
        cases.extend(({'font_backend': value}, expected, warns) for value, expected, warns in (
            ({}, defaults, False),
            ({'windows': 'gdi'}, {'windows': 'gdi', 'macos': 'default'}, False),
            ({'macos': 'freetype'}, {'windows': 'default', 'macos': 'freetype'}, False),
            ({'windows': 'gdi', 'macos': False}, {'windows': 'gdi', 'macos': 'default'}, True),
            ({'windows': [], 'macos': 'freetype'}, {'windows': 'default', 'macos': 'freetype'}, True),
            ({'windows': 'gdi', 'macos': 'gdi'}, {'windows': 'gdi', 'macos': 'default'}, True),
            ({'windows': 'freetype', 'removed': 'value'}, {'windows': 'freetype', 'macos': 'default'}, True),
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            for payload, expected, warns in cases:
                with self.subTest(payload=payload):
                    payload['display_lang'] = 'en_US'
                    original = json.dumps(payload)
                    with open(path, 'w', encoding='utf8') as stream:
                        stream.write(original)
                    with mock.patch('ballontranslator.utils.config.LOGGER.warning') as warning:
                        config = ProgramConfig.load(path)
                    self.assertEqual(config.font_backend, expected)
                    self.assertEqual(config.display_lang, 'en_US')
                    self.assertEqual(warning.called, warns)
                    self.assertEqual(json.loads(json_dump_program_config(config))['font_backend'], expected)
                    with open(path, encoding='utf8') as stream:
                        self.assertEqual(stream.read(), original)


class FontBackendConfigPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        for patcher in (
            mock.patch.object(pcfg, 'font_backend', {'windows': 'default', 'macos': 'default'}),
            mock.patch('ballontranslator.ui.configpanel.sys', SimpleNamespace(platform='win32')),
            mock.patch('ballontranslator.ui.configpanel.probe_torch_package', return_value=(None, None)),
            mock.patch.object(codex.account, '_load', side_effect=AssertionError('Unexpected credential IO')),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.panel = ConfigPanel()
        self.panel.setupConfig()
        self.addCleanup(self.panel.deleteLater)

    def tearDown(self) -> None:
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        super().tearDown()

    def test_user_choice_saves_once_and_prompts_for_restart(self) -> None:
        saved, restarted = mock.Mock(), mock.Mock()
        self.panel.save_config.connect(saved)
        self.panel.restart_requested.connect(restarted)
        combo = self.panel.font_backend_combobox
        self.assertEqual([combo.itemData(i) for i in range(combo.count())],
                         ['default', 'gdi', 'freetype'])

        QTest.keyClick(combo, Qt.Key.Key_Down)

        self.assertEqual(pcfg.font_backend['windows'], 'gdi')
        self.assertFalse(self.panel.font_backend_restart_button.isHidden())
        saved.assert_called_once()
        combo.activated.emit(combo.currentIndex())
        saved.assert_called_once()
        QTest.keyClick(combo, Qt.Key.Key_Return)
        restarted.assert_not_called()
        self.panel.font_backend_restart_button.click()
        restarted.assert_called_once()

        QTest.keyClick(combo, Qt.Key.Key_Up)

        self.assertEqual(pcfg.font_backend['windows'], 'default')
        self.assertTrue(self.panel.font_backend_restart_button.isHidden())
        self.assertEqual(saved.call_count, 2)

    def test_setup_and_programmatic_selection_do_not_save(self) -> None:
        saved = mock.Mock()
        self.panel.save_config.connect(saved)
        pcfg.font_backend['windows'] = 'freetype'
        self.panel.setupConfig()
        self.assertEqual(self.panel.font_backend_combobox.currentData(), 'freetype')
        self.assertFalse(self.panel.font_backend_restart_button.isHidden())
        self.panel.font_backend_combobox.setCurrentIndex(1)
        self.assertEqual(pcfg.font_backend['windows'], 'freetype')
        saved.assert_not_called()

    def test_platform_choices_are_independent_and_linux_keeps_them(self) -> None:
        pcfg.font_backend['windows'] = 'gdi'
        with mock.patch('ballontranslator.ui.configpanel.sys', SimpleNamespace(platform='darwin')):
            panel = ConfigPanel()
            try:
                panel.setupConfig()
                combo = panel.font_backend_combobox
                self.assertEqual([combo.itemData(i) for i in range(combo.count())], ['default', 'freetype'])
                self.assertEqual(combo.currentData(), 'default')
                QTest.keyClick(combo, Qt.Key.Key_Down)
                self.assertEqual(pcfg.font_backend, {'windows': 'gdi', 'macos': 'freetype'})
                self.assertFalse(panel.font_backend_restart_button.isHidden())
            finally:
                panel.deleteLater()
        with mock.patch('ballontranslator.ui.configpanel.sys', SimpleNamespace(platform='linux')):
            panel = ConfigPanel()
            try:
                panel.setupConfig()
                self.assertIsNone(panel.font_backend_combobox)
                self.assertEqual(pcfg.font_backend, {'windows': 'gdi', 'macos': 'freetype'})
            finally:
                panel.deleteLater()

    def test_user_choice_round_trips_through_config_save(self) -> None:
        pcfg.font_backend['macos'] = 'freetype'
        self.panel.save_config.connect(program_config.save_config)
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            shared, 'CONFIG_PATH', os.path.join(tmpdir, 'config.json'),
        ):
            QTest.keyClick(self.panel.font_backend_combobox, Qt.Key.Key_Down)
            loaded = ProgramConfig.load(shared.CONFIG_PATH)
            self.assertEqual(loaded.font_backend, {'windows': 'gdi', 'macos': 'freetype'})


if __name__ == '__main__':
    unittest.main()
