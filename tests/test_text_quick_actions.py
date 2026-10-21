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

    def test_growing_leaves_the_frame_to_the_layout(self) -> None:
        """Enlarging must not pin the frame; the layout grows it instead."""
        calls = []
        self.actions._restore_frames = lambda: calls.append(1)
        self.actions.start_scale(100.0)
        self.actions.drag_scale(100.0 + DRAG_SPAN * 0.2)
        self.assertEqual(calls, [])

    def test_shrinking_restores_the_captured_frame(self) -> None:
        """One gesture: grow, then shrink past the start."""
        calls = []
        item = self.items[0]
        captured = item.boundingRect().height()
        self.actions.start_scale(100.0)
        self.actions.drag_scale(100.0 + DRAG_SPAN * 0.4)
        self.actions._restore_frames = lambda: calls.append(1)
        self.actions.drag_scale(100.0 - DRAG_SPAN * 0.4)
        self.actions.end_scale()
        self.assertTrue(calls)
        self.assertAlmostEqual(
            item.boundingRect().height(), captured, places=1
        )

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

        wrapped = make_item('一二三四五六七八九十', (0, 0, 120, 80))
        wrapped.layout.reLayoutEverything()
        block = wrapped.document().firstBlock().layout()
        columns = {round(block.lineAt(i).x(), 1) for i in range(block.lineCount())}
        self.assertGreater(len(columns), 1, columns)

    def test_flag_is_part_of_the_layout_cache_key(self) -> None:
        """Turning the flag off must not reuse the free-wrapped geometry."""
        item = make_item('一二三四五六七八九十', (0, 0, 120, 80))
        item.layout.reLayoutEverything()
        before = item.layout._plain_column_context
        item.fontformat.no_forced_wrap = True
        item.layout.reLayoutEverything()
        after = item.layout._plain_column_context
        self.assertNotEqual(before, after)

    def test_switching_off_restores_the_original_height(self) -> None:
        item = make_item('一二三四五六七八九十', (0, 0, 120, 80))
        original = item.boundingRect().height()
        actions = TextQuickActions(self.view, lambda: [item])
        actions.set_target(item)
        actions._on_wrap_clicked()
        grown = item.boundingRect().height()
        self.assertGreater(grown, original)
        actions._on_wrap_clicked()
        self.assertAlmostEqual(
            item.boundingRect().height(), original, places=1
        )
        self.assertFalse(item.fontformat.no_forced_wrap)

    def test_frame_grows_to_hold_free_wrapped_text(self) -> None:
        item = make_item('一二三四五六七八九十', (0, 0, 120, 80))
        before = item.boundingRect().height()
        item.fontformat.no_forced_wrap = True
        item.layout.reLayoutEverything()
        self.assertGreater(item.boundingRect().height(), before)


class NotifyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_binding_notifies_the_quick_actions(self) -> None:
        """The handles follow the control's binding, including clearing."""
        from types import SimpleNamespace

        from qtpy.QtWidgets import QGraphicsScene

        from ballontranslator.ui.text_engine.shape_control import (
            TextBlkShapeControl,
        )

        scene = QGraphicsScene()
        seen = []
        scene.text_quick_actions = SimpleNamespace(
            set_target=lambda item: seen.append(item)
        )
        from qtpy.QtWidgets import QGraphicsView

        view = QGraphicsView(scene)
        view.resize(400, 400)
        control = TextBlkShapeControl(view)
        scene.addItem(control)

        item = make_item('一二', (0, 0, 100, 100))
        control.setBlkItem(item)
        self.assertTrue(seen)
        self.assertIs(seen[-1], item)

        control.setBlkItem(None)
        self.assertIsNone(seen[-1])


if __name__ == '__main__':
    unittest.main()
