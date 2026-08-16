import gc
import json
import os
import tempfile
import unittest
import weakref
from typing import List
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import QT6
from qtpy.QtCore import QEvent, QPointF, QTranslator, Qt
from qtpy.QtGui import QFontDatabase, QMouseEvent
from qtpy.QtWidgets import QApplication, QDialog, QWidget

from ballontranslator.ui.configpanel import ConfigPanel, FontExcludeDialog
from ballontranslator.ui.text_engine.formatting.panel import (
    FontFamilyComboBox,
    FontFormatPanel,
)
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.config import ProgramConfig, pcfg
from ballontranslator.utils.fontformat import FontFormat


def get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def item_names(list_widget) -> List[str]:
    return [
        list_widget.item(i).data(Qt.ItemDataRole.UserRole)
        for i in range(list_widget.count())
    ]


def mouse_press_event() -> QMouseEvent:
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(1, 1),
        QPointF(1, 1),
        QPointF(1, 1),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


class FontExclusionConfigTests(unittest.TestCase):

    @staticmethod
    def _load_config(payload) -> ProgramConfig:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'config.json')
            with open(path, 'w', encoding='utf8') as f:
                json.dump(payload, f)
            return ProgramConfig.load(path)

    def test_font_filter_is_sorted_and_excludes_names(self):
        self.assertEqual(
            shared.get_filtered_font_list(
                {'Times', 'Arial', 'Courier'},
                ['Times'],
            ),
            ['Arial', 'Courier'],
        )

    def test_config_discards_invalid_exclusion_value_only(self):
        config = self._load_config({
            'darkmode': True,
            'excluded_fonts': 1,
        })

        self.assertTrue(config.darkmode)
        self.assertEqual(config.excluded_fonts, [])

    def test_config_normalizes_mixed_and_duplicate_font_names(self):
        config = self._load_config({
            'excluded_fonts': ['Zulu', 1, '', 'Alpha', 'Zulu'],
        })

        self.assertEqual(config.excluded_fonts, ['Alpha', 'Zulu'])


class FontExclusionUiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def setUp(self):
        self.old_font_families = shared.FONT_FAMILIES
        self.old_excluded_fonts = pcfg.excluded_fonts
        self.old_only_custom = pcfg.let_show_only_custom_fonts_flag
        self.old_active_format = C.active_format

    def tearDown(self):
        shared.FONT_FAMILIES = self.old_font_families
        pcfg.excluded_fonts = self.old_excluded_fonts
        pcfg.let_show_only_custom_fonts_flag = self.old_only_custom
        C.active_format = self.old_active_format

    def test_search_clears_hidden_selection_before_moving_fonts(self):
        shared.FONT_FAMILIES = {'Zulu', 'Alpha', 'Beta'}
        pcfg.excluded_fonts = ['Beta']
        dialog = FontExcludeDialog()
        self.addCleanup(dialog.close)

        self.assertEqual(item_names(dialog.available_list), ['Alpha', 'Zulu'])
        for i in range(dialog.available_list.count()):
            dialog.available_list.item(i).setSelected(True)

        dialog.search_edit.setText('Alpha')
        dialog._hide_fonts()

        self.assertEqual(dialog.get_excluded_fonts(), ['Alpha', 'Beta'])
        self.assertEqual(item_names(dialog.available_list), ['Zulu'])

    def test_legacy_addition_respects_active_search(self):
        shared.FONT_FAMILIES = {'Alpha', 'MS Sans Serif'}
        pcfg.excluded_fonts = []
        dialog = FontExcludeDialog()
        self.addCleanup(dialog.close)
        dialog.search_edit.setText('Alpha')

        with patch('ballontranslator.ui.configpanel.QMessageBox.information'):
            dialog._on_add_legacy_fonts()

        legacy_item = dialog.excluded_list.item(0)
        self.assertEqual(dialog._real_name(legacy_item), 'MS Sans Serif')
        self.assertTrue(legacy_item.isHidden())

    def test_outside_click_rejects_frameless_dialog(self):
        shared.FONT_FAMILIES = {'Alpha', 'Beta'}
        pcfg.excluded_fonts = ['Beta']
        panel = ConfigPanel()
        outside = QWidget()
        self.addCleanup(panel.deleteLater)
        self.addCleanup(outside.close)
        panel.show_font_exclusion_dialog()
        dialog = panel.font_exclude_dialog
        self.app.processEvents()
        dialog.available_list.item(0).setSelected(True)
        dialog._hide_fonts()
        window_type = getattr(Qt, 'WindowType', Qt)

        dialog.eventFilter(outside, mouse_press_event())

        self.assertTrue(dialog.windowFlags() & window_type.FramelessWindowHint)
        self.assertFalse(dialog.isVisible())
        self.assertEqual(dialog.result(), QDialog.DialogCode.Rejected)
        self.assertEqual(pcfg.excluded_fonts, ['Beta'])

    def test_closed_font_exclude_dialog_is_deleted_and_recreated(self):
        shared.FONT_FAMILIES = {'Alpha'}
        panel = ConfigPanel()
        self.addCleanup(panel.deleteLater)
        panel.show_font_exclusion_dialog()
        first_dialog = panel.font_exclude_dialog
        first_dialog_ref = weakref.ref(first_dialog)
        destroyed = []
        first_dialog.destroyed.connect(lambda: destroyed.append(True))

        first_dialog.reject()
        self.assertIsNone(panel.font_exclude_dialog)
        del first_dialog
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        self.assertEqual(destroyed, [True])
        self.assertIsNone(first_dialog_ref())

        panel.show_font_exclusion_dialog()
        self.assertIsNotNone(panel.font_exclude_dialog)

    def test_config_panel_outside_click_hides_and_saves(self):
        panel = ConfigPanel()
        outside = QWidget()
        self.addCleanup(panel.deleteLater)
        self.addCleanup(outside.close)
        saves = []
        panel.save_config.connect(lambda: saves.append(True))
        panel.showConfigDialog()
        self.app.processEvents()

        panel.eventFilter(outside, mouse_press_event())

        self.assertFalse(panel.isVisible())
        self.assertEqual(saves, [True])

    def test_font_combo_preserves_an_applied_hidden_font(self):
        shared.FONT_FAMILIES = {'Alpha', 'Beta'}
        combo = FontFamilyComboBox()
        self.addCleanup(combo.close)
        combo.update_font_list(['Alpha', 'Beta'])
        combo.setCurrentText('Beta')
        changes = []
        combo.param_changed.connect(lambda *args: changes.append(args))

        combo.update_font_list(['Alpha'])

        self.assertEqual(combo.currentText(), 'Beta')
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ['Alpha'])
        self.assertEqual(changes, [])

    def test_selecting_item_with_hidden_font_keeps_filtered_popup(self):
        font_database = QFontDatabase if QT6 else QFontDatabase()
        font_families = sorted(font_database.families(), key=str.casefold)
        self.assertGreaterEqual(len(font_families), 2)
        allowed_font, hidden_font = font_families[0], font_families[-1]
        shared.FONT_FAMILIES = set(font_families)
        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        active_format = FontFormat(font_family=hidden_font)
        panel.global_format = active_format
        panel.familybox.update_font_list([allowed_font])
        panel.familybox.param_changed.disconnect(
            panel.on_font_family_changed
        )
        changes = []
        panel.familybox.param_changed.connect(lambda *args: changes.append(args))

        panel.set_active_format(active_format)

        self.assertEqual(panel.familybox.currentText(), hidden_font)
        self.assertEqual(
            [panel.familybox.itemText(i) for i in range(panel.familybox.count())],
            [allowed_font],
        )

        panel.familybox.setCurrentIndex(0)

        self.assertEqual(changes, [('font_family', allowed_font)])

    def test_config_panel_emits_only_for_changed_exclusions(self):
        pcfg.excluded_fonts = ['Beta']
        pcfg.let_show_only_custom_fonts_flag = True
        panel = ConfigPanel()
        self.addCleanup(panel.deleteLater)
        font_changes = []
        saves = []
        panel.font_list_changed.connect(font_changes.append)
        panel.save_config.connect(lambda: saves.append(True))

        panel._apply_font_exclusions(['Beta'])

        self.assertEqual(font_changes, [])
        self.assertEqual(saves, [])

        panel._apply_font_exclusions(['Alpha'])

        self.assertEqual(font_changes, [True])
        self.assertEqual(saves, [True])
        self.assertEqual(pcfg.excluded_fonts, ['Alpha'])

    def test_compiled_chinese_font_exclusion_translation_loads(self):
        translator = QTranslator()

        self.assertTrue(translator.load('zh_CN', 'resources/translate'))
        self.assertEqual(
            translator.translate('FontExcludeDialog', 'Font Exclusion'),
            '字体排除',
        )


if __name__ == '__main__':
    unittest.main()
