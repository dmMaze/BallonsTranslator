import json
import os
import tempfile
import unittest
from typing import List
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QTranslator, Qt
from qtpy.QtWidgets import QApplication, QDialog

from ballontranslator.ui.configpanel import ConfigPanel, FontExcludeDialog
from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox
from ballontranslator.utils import shared
from ballontranslator.utils.config import ProgramConfig, pcfg


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

    def tearDown(self):
        shared.FONT_FAMILIES = self.old_font_families
        pcfg.excluded_fonts = self.old_excluded_fonts
        pcfg.let_show_only_custom_fonts_flag = self.old_only_custom

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

    def test_config_panel_emits_only_for_changed_exclusions(self):
        pcfg.excluded_fonts = ['Beta']
        pcfg.let_show_only_custom_fonts_flag = True
        panel = ConfigPanel()
        self.addCleanup(panel.deleteLater)
        font_changes = []
        saves = []
        panel.font_list_changed.connect(font_changes.append)
        panel.save_config.connect(lambda: saves.append(True))

        unchanged = Mock()
        unchanged.exec.return_value = QDialog.DialogCode.Accepted
        unchanged.get_excluded_fonts.return_value = ['Beta']
        with patch('ballontranslator.ui.configpanel.FontExcludeDialog', return_value=unchanged):
            panel.on_exclude_fonts_clicked()

        self.assertEqual(font_changes, [])
        self.assertEqual(saves, [])

        changed = Mock()
        changed.exec.return_value = QDialog.DialogCode.Accepted
        changed.get_excluded_fonts.return_value = ['Alpha']
        with patch('ballontranslator.ui.configpanel.FontExcludeDialog', return_value=changed):
            panel.on_exclude_fonts_clicked()

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
