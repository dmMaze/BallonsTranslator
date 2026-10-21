"""Vertical font auto-fit: grow a short block to fill its box."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.vertical_fit import (
    content_fits,
    fit_vertical_font_size,
)
from ballontranslator.utils.fontformat import TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


def make_item(text, bounds, font_size, vertical=True) -> TextBlkItem:
    block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(bounds)
    block.vertical = vertical
    block.alignment = TextAlignment.Center
    block.translation = text
    block.fontformat.font_size = font_size
    block.fontformat.letter_spacing = 1.0
    block.fontformat.vertical = vertical
    return TextBlkItem(block, 0)


class VerticalFitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_short_text_grows_towards_the_box(self) -> None:
        item = make_item('短い', (0, 0, 200, 400), 20)
        before = item.fontformat.font_size
        ratio = fit_vertical_font_size(item)
        self.assertGreater(ratio, 1.0)
        self.assertGreater(item.fontformat.font_size, before)
        self.assertTrue(content_fits(item))

    def test_already_full_block_is_left_alone(self) -> None:
        item = make_item('これはとても長い文章です。', (0, 0, 200, 400), 60)
        before = item.fontformat.font_size
        self.assertEqual(fit_vertical_font_size(item), 1.0)
        self.assertEqual(item.fontformat.font_size, before)

    def test_horizontal_block_is_not_touched(self) -> None:
        item = make_item('短い', (0, 0, 300, 200), 20, vertical=False)
        before = item.fontformat.font_size
        self.assertEqual(fit_vertical_font_size(item), 1.0)
        self.assertEqual(item.fontformat.font_size, before)

    def test_growth_is_bounded(self) -> None:
        item = make_item('あ', (0, 0, 200, 900), 20)
        ratio = fit_vertical_font_size(item, max_ratio=1.6)
        self.assertLessEqual(ratio, 1.6)
        self.assertTrue(content_fits(item))


if __name__ == '__main__':
    unittest.main()
