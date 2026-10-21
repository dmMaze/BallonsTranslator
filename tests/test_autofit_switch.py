"""The switch gates the vertical auto-fit, and the settings panel exposes it."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.vertical_fit import (
    fit_vertical_font_size,
)
from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


def make_item(text, bounds, font_size) -> TextBlkItem:
    block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(bounds)
    block.vertical = True
    block.alignment = TextAlignment.Center
    block.translation = text
    block.fontformat.font_size = font_size
    block.fontformat.letter_spacing = 1.0
    block.fontformat.vertical = True
    return TextBlkItem(block, 0)


class AutofitSwitchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_switch_defaults_to_off(self) -> None:
        """A layout-changing feature must not surprise existing projects."""
        self.assertFalse(C.ProgramConfig().vertical_font_autofit_enabled)

    def test_disabled_switch_leaves_the_font_size_alone(self) -> None:
        item = make_item('短い', (0, 0, 200, 400), 20)
        before = item.fontformat.font_size
        with patch.object(C.pcfg, 'vertical_font_autofit_enabled', False):
            if C.pcfg.vertical_font_autofit_enabled:
                fit_vertical_font_size(item)
        self.assertEqual(item.fontformat.font_size, before)

    def test_enabled_switch_grows_the_block(self) -> None:
        item = make_item('短い', (0, 0, 200, 400), 20)
        before = item.fontformat.font_size
        with patch.object(C.pcfg, 'vertical_font_autofit_enabled', True):
            if C.pcfg.vertical_font_autofit_enabled:
                fit_vertical_font_size(item)
        self.assertGreater(item.fontformat.font_size, before)


if __name__ == '__main__':
    unittest.main()
