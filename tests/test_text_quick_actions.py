"""Quick actions: drag scaling and the no-forced-wrap switch."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_quick_actions import (
    DRAG_SPAN,
    MAX_FONT_SIZE,
    MIN_FONT_SIZE,
    TextQuickActions,
    scaled_font_size,
)
from ballontranslator.utils.fontformat import TextAlignment
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


def make_item(text, bounds, font_size=30, vertical=True) -> TextBlkItem:
    block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
    block._bounding_rect = list(bounds)
    block.vertical = vertical
    block.alignment = TextAlignment.Left
    block.translation = text
    block.fontformat.font_size = font_size
    block.fontformat.letter_spacing = 1.0
    block.fontformat.vertical = vertical
    return TextBlkItem(block, 0)


class ScaledSizeTest(unittest.TestCase):

    def test_drag_down_grows_and_up_shrinks(self) -> None:
        self.assertGreater(scaled_font_size(40.0, 60.0), 40.0)
        self.assertLess(scaled_font_size(40.0, -60.0), 40.0)

    def test_zero_drag_keeps_the_size(self) -> None:
        self.assertAlmostEqual(scaled_font_size(40.0, 0.0), 40.0)

    def test_bounds_are_respected(self) -> None:
        self.assertEqual(scaled_font_size(40.0, -1e6), MIN_FONT_SIZE)
        self.assertEqual(scaled_font_size(40.0, 1e6), MAX_FONT_SIZE)

    def test_full_span_doubles_the_size(self) -> None:
        self.assertAlmostEqual(
            scaled_font_size(40.0, DRAG_SPAN), 80.0, places=6
        )


class ActionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.scene = QGraphicsScene()
        self.items = [make_item('一二三四', (0, 0, 200, 300))]
        self.scene.addItem(self.items[0])
        self.view = QGraphicsView(self.scene)
        self.view.resize(400, 400)
        self.actions = TextQuickActions(self.view, lambda: self.items)
        self.actions.set_target(self.items[0])

    def test_drag_applies_a_scaled_size(self) -> None:
        self.actions.start_scale(100.0)
        self.actions.drag_scale(100.0 + DRAG_SPAN / 2)
        self.assertAlmostEqual(
            self.items[0].fontformat.font_size, 45.0, places=3
        )
        self.actions.end_scale()

    def test_wrap_toggle_flips_the_flag_and_back(self) -> None:
        self.actions._on_wrap_clicked()
        self.assertTrue(self.items[0].fontformat.no_forced_wrap)
        self.actions._on_wrap_clicked()
        self.assertFalse(self.items[0].fontformat.no_forced_wrap)

    def test_unconstrained_vertical_block_keeps_one_column(self) -> None:
        item = make_item('一二三四五六七八九十', (0, 0, 120, 80))
        item.fontformat.no_forced_wrap = True
        item.layout.reLayoutEverything()
        block = item.document().firstBlock().layout()
        columns = {round(block.lineAt(i).x(), 1) for i in range(block.lineCount())}
        self.assertEqual(len(columns), 1, columns)

        # Turning it off must restore the wrapped layout instead of reusing the
        # cached single-column geometry.
        item.fontformat.no_forced_wrap = False
        item.layout.reLayoutEverything()
        block = item.document().firstBlock().layout()
        columns = {round(block.lineAt(i).x(), 1) for i in range(block.lineCount())}
        self.assertGreater(len(columns), 1, columns)


if __name__ == '__main__':
    unittest.main()
