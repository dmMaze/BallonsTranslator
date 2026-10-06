import json
import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QTranslator
from qtpy.QtWidgets import QApplication

from ballontranslator.modules import codex
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.utils.config import (
    FONT_BACKEND_DEFAULT,
    FontBackendConfig,
    ProgramConfig,
    font_backend_options,
    pcfg,
)


class FontBackendConfigTests(unittest.TestCase):
    def test_platform_options_are_explicit(self) -> None:
        self.assertEqual(
            font_backend_options('win32'),
            ('default', 'gdi', 'freetype'),
        )
        self.assertEqual(
            font_backend_options('darwin'),
            ('default', 'freetype'),
        )
        self.assertEqual(font_backend_options('linux'), ('default',))

    def test_invalid_saved_values_fall_back_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            with open(path, 'w', encoding='utf8') as stream:
                json.dump(
                    {
                        'font_backend': {
                            'windows': 'unknown',
                            'macos': 12,
                            'removed': 'value',
                        }
                    },
                    stream,
                )
            with mock.patch(
                'ballontranslator.utils.config.LOGGER.warning'
            ) as warning:
                config = ProgramConfig.load(path)

        self.assertEqual(
            config.font_backend,
            FontBackendConfig(
                windows=FONT_BACKEND_DEFAULT,
                macos=FONT_BACKEND_DEFAULT,
            ),
        )
        self.assertGreaterEqual(warning.call_count, 3)


@unittest.skipUnless(sys.platform == 'win32', 'Windows backend UI')
class FontBackendConfigPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.original_config = pcfg.font_backend.copy()
        pcfg.font_backend.windows = FONT_BACKEND_DEFAULT
        patches = (
            mock.patch(
                'ballontranslator.ui.configpanel.probe_torch_package',
                return_value=(None, None),
            ),
            mock.patch.object(
                codex.account,
                '_load',
                side_effect=AssertionError('Unexpected credential IO'),
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.panel = ConfigPanel()
        self.addCleanup(self.panel.deleteLater)
        self.addCleanup(
            lambda: setattr(pcfg, 'font_backend', self.original_config)
        )

    def tearDown(self) -> None:
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        super().tearDown()

    def test_typesetting_advanced_backend_change_prompts_for_restart(self) -> None:
        self.assertEqual(
            [
                self.panel.font_backend_combobox.itemText(index)
                for index in range(self.panel.font_backend_combobox.count())
            ],
            ['Default', 'GDI', 'FreeType'],
        )
        self.assertEqual(
            self.panel.font_backend_advanced_group.objectName(),
            'FontBackendAdvancedGroup',
        )
        self.assertEqual(self.panel.font_backend_help_button.text(), '?')
        self.assertIn(
            'NexusFont', self.panel.font_backend_help_button.toolTip()
        )
        self.assertEqual(self.panel.font_backend_restart_button.text(), '')
        self.assertFalse(self.panel.font_backend_restart_button.icon().isNull())
        self.assertEqual(
            self.panel.font_backend_restart_notice.text(),
            'Restart required to apply changes',
        )
        self.assertEqual(
            self.panel.font_backend_restart_row.parentWidget().objectName(),
            'FontBackendSelectorRow',
        )
        saved = mock.Mock()
        restarted = mock.Mock()
        self.panel.save_config.connect(saved)
        self.panel.restart_requested.connect(restarted)

        self.panel.font_backend_combobox.setCurrentIndex(1)

        self.assertEqual(pcfg.font_backend.windows, 'gdi')
        self.assertFalse(self.panel.font_backend_restart_row.isHidden())
        self.assertEqual(saved.call_count, 1)
        self.panel.font_backend_restart_button.click()
        self.assertEqual(restarted.call_count, 1)

        self.panel.font_backend_combobox.setCurrentIndex(0)

        self.assertEqual(pcfg.font_backend.windows, FONT_BACKEND_DEFAULT)
        self.assertTrue(self.panel.font_backend_restart_row.isHidden())
        self.assertEqual(saved.call_count, 2)

    def test_font_backend_controls_use_config_panel_translations(self) -> None:
        translations = {
            'Default': 'Translated Default',
            'Default: system recommended\n'
            'GDI: compatible with NexusFont\n'
            'FreeType: Qt font engine': 'Translated help',
            'Font backend help': 'Translated accessible help',
            'Restart required to apply changes': 'Translated restart notice',
            'Restart application now': 'Translated restart action',
            'Restart': 'Translated restart',
        }

        class BackendTranslator(QTranslator):
            def translate(
                self, context, source_text, disambiguation=None, n=-1,
            ) -> str:
                if context == 'ConfigPanel':
                    return translations.get(source_text, '')
                return ''

        translator = BackendTranslator()
        self.app.installTranslator(translator)
        panel = ConfigPanel()
        try:
            self.assertEqual(
                panel.font_backend_combobox.itemText(0),
                'Translated Default',
            )
            self.assertEqual(
                panel.font_backend_help_button.toolTip(),
                'Translated help',
            )
            self.assertEqual(
                panel.font_backend_help_button.accessibleName(),
                'Translated accessible help',
            )
            self.assertEqual(
                panel.font_backend_restart_notice.text(),
                'Translated restart notice',
            )
            self.assertEqual(
                panel.font_backend_restart_button.toolTip(),
                'Translated restart action',
            )
            self.assertEqual(
                panel.font_backend_restart_button.accessibleName(),
                'Translated restart',
            )
        finally:
            panel.deleteLater()
            self.app.removeTranslator(translator)


if __name__ == '__main__':
    unittest.main()
