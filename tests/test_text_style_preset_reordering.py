import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.formatting.presets import TextStylePresetPanel
from ballontranslator.utils import config
from ballontranslator.utils.fontformat import FontFormat


class TextStylePresetReorderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_styles = list(config.text_styles)
        config.text_styles.clear()

    def tearDown(self):
        config.text_styles[:] = self.original_styles

    def test_reordering_moves_widget_and_saved_style_together(self):
        first = FontFormat(_style_name='First')
        second = FontFormat(_style_name='Second')
        third = FontFormat(_style_name='Third')
        config.text_styles.extend([first, second, third])
        panel = TextStylePresetPanel('Text Style', 'text_style', 'expand_tstyle_panel')
        panel.initStyles(config.text_styles)

        first_label = panel.flayout.itemAt(0).widget()
        third_label = panel.flayout.itemAt(2).widget()
        with patch(
            'ballontranslator.ui.text_engine.formatting.presets.save_text_styles'
        ) as save_styles:
            panel.reorderStyleLabel(third_label, first_label, False)

        self.assertEqual(
            [style._style_name for style in config.text_styles],
            ['Third', 'First', 'Second'],
        )
        self.assertIs(panel.flayout.itemAt(0).widget(), third_label)
        save_styles.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
