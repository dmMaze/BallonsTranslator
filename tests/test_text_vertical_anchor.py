"""Vertical anchor placement inside the logical rectangle.

The anchor only translates settled lines; it must never change wrapping,
bookkeeping coordinates, or the persisted rectangle.
"""

import os
import unittest
import warnings


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, Qt
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.vertical_layout import (
    VerticalTextDocumentLayout,
)
from ballontranslator.utils.fontformat import (
    FontFormat,
    TextAlignment,
    TextVerticalAlignment,
)
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


BOUNDS = (0, 0, 320, 300)
TEXT = '天地玄黃宇'
FONT_SIZE = 30
TOLERANCE = 0.05


class VerticalAnchorLayoutTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def _make_item(
        self,
        *,
        vertical: bool,
        anchor: int,
        alignment: TextAlignment = TextAlignment.Center,
        text: str = TEXT,
        bounds=BOUNDS,
    ) -> TextBlkItem:
        block = TextBlock(list(bounds), text_layout_version=TEXT_LAYOUT_VERSION)
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.alignment = alignment
        block.translation = text
        block.fontformat.font_size = FONT_SIZE
        block.fontformat.letter_spacing = 1.0
        block.fontformat.vertical_alignment = anchor
        return TextBlkItem(block, 0)

    @staticmethod
    def _lines(item: TextBlkItem):
        geometry = []
        block = item.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            for index in range(layout.lineCount()):
                line = layout.lineAt(index)
                geometry.append(
                    (line.y(), line.textStart(), line.textLength())
                )
            block = block.next()
        return geometry

    @staticmethod
    def _content_height(item: TextBlkItem) -> float:
        layout = item.layout
        if isinstance(layout, VerticalTextDocumentLayout):
            return layout.shrink_height
        return layout.shrink_height - layout.effectPadding()

    def _slack(self, item: TextBlkItem) -> float:
        return max(0.0, item.layout.available_height - self._content_height(item))

    def _expected_shift(self, item: TextBlkItem, anchor) -> float:
        slack = self._slack(item)
        if anchor is TextVerticalAlignment.Center:
            return slack / 2
        if anchor is TextVerticalAlignment.Bottom:
            return slack
        return 0.0

    def _assert_shift(self, item, anchor, label: str) -> None:
        lines = self._lines(item)
        self.assertTrue(lines, f'{label}: no settled lines')
        expected = self._expected_shift(item, anchor)
        self.assertGreater(
            expected, 1.0, f'{label}: the case needs free space to be meaningful'
        )
        self.assertAlmostEqual(
            lines[0][0], expected, delta=TOLERANCE,
            msg=f'{label}: first line y {lines[0][0]} != {expected}',
        )

    def test_auto_keeps_vertical_blocks_top_anchored(self) -> None:
        item = self._make_item(
            vertical=True, anchor=TextVerticalAlignment.Auto
        )
        self.assertIs(
            item.layout.effective_vertical_alignment(),
            TextVerticalAlignment.Top,
        )
        self.assertAlmostEqual(self._lines(item)[0][0], 0.0, delta=TOLERANCE)

    def test_auto_matches_the_historical_horizontal_rule(self) -> None:
        centred = self._make_item(
            vertical=False, anchor=TextVerticalAlignment.Auto
        )
        self.assertIs(
            centred.layout.effective_vertical_alignment(),
            TextVerticalAlignment.Center,
        )
        self._assert_shift(centred, TextVerticalAlignment.Center, 'auto/centre')

        left = self._make_item(
            vertical=False,
            anchor=TextVerticalAlignment.Auto,
            alignment=TextAlignment.Left,
        )
        self.assertIs(
            left.layout.effective_vertical_alignment(),
            TextVerticalAlignment.Top,
        )
        self.assertAlmostEqual(self._lines(left)[0][0], 0.0, delta=TOLERANCE)

    def test_explicit_anchor_settles_both_writing_modes(self) -> None:
        for vertical in (True, False):
            for anchor in (
                TextVerticalAlignment.Top,
                TextVerticalAlignment.Center,
                TextVerticalAlignment.Bottom,
            ):
                with self.subTest(vertical=vertical, anchor=anchor.name):
                    item = self._make_item(vertical=vertical, anchor=anchor)
                    if anchor is TextVerticalAlignment.Top:
                        self.assertAlmostEqual(
                            self._lines(item)[0][0], 0.0, delta=TOLERANCE
                        )
                    else:
                        self._assert_shift(
                            item, anchor, f'vertical={vertical}'
                        )

    def test_bottom_anchor_ends_at_the_rect_bottom(self) -> None:
        for vertical in (True, False):
            with self.subTest(vertical=vertical):
                item = self._make_item(
                    vertical=vertical, anchor=TextVerticalAlignment.Bottom
                )
                layout = item.layout
                lines = self._lines(item)
                last = max(lines, key=lambda entry: entry[0])
                line = item.document().firstBlock().layout().lineAt(0)
                self.assertAlmostEqual(
                    last[0] + line.height(),
                    layout.available_height,
                    delta=1.0,
                )

    def test_anchor_never_changes_wrapping(self) -> None:
        for vertical in (True, False):
            with self.subTest(vertical=vertical):
                baseline = self._make_item(
                    vertical=vertical, anchor=TextVerticalAlignment.Top
                )
                expected = [
                    (start, length) for _y, start, length in self._lines(baseline)
                ]
                for anchor in (
                    TextVerticalAlignment.Center,
                    TextVerticalAlignment.Bottom,
                ):
                    item = self._make_item(vertical=vertical, anchor=anchor)
                    self.assertEqual(
                        [
                            (start, length)
                            for _y, start, length in self._lines(item)
                        ],
                        expected,
                    )

    def test_anchor_never_changes_the_rectangle(self) -> None:
        for vertical in (True, False):
            with self.subTest(vertical=vertical):
                rects = []
                for anchor in (
                    TextVerticalAlignment.Top,
                    TextVerticalAlignment.Bottom,
                ):
                    item = self._make_item(vertical=vertical, anchor=anchor)
                    rects.append(item.logical_unpadded_rect())
                self.assertAlmostEqual(
                    rects[0].width(), rects[1].width(), delta=TOLERANCE
                )
                self.assertAlmostEqual(
                    rects[0].height(), rects[1].height(), delta=TOLERANCE
                )

    def test_vertical_absolute_cells_follow_the_lines(self) -> None:
        """Selection cells and slanted-glyph painting read absolute extents."""
        item = self._make_item(
            vertical=True, anchor=TextVerticalAlignment.Bottom
        )
        layout = item.layout
        lines = self._lines(item)
        for index, (line_y, _start, _length) in enumerate(lines):
            top, bottom = layout.y_offset_lst[0][index]
            self.assertAlmostEqual(float(top), line_y, delta=TOLERANCE)
            self.assertAlmostEqual(
                float(layout.line_spaces_lst[0][index][2][0]),
                line_y,
                delta=TOLERANCE,
            )
            self.assertGreater(float(bottom), float(top))

    def test_hit_testing_follows_the_moved_content(self) -> None:
        """One screen point resolves to different text under each anchor.

        Vertical hit testing returns the nearest character boundary, so the
        probe sits in the upper half of a cell: inside it the point resolves to
        that character, while the same point in a shorter top-anchored block
        falls past the text and clamps to its end.
        """
        bottom_item = self._make_item(
            vertical=True, anchor=TextVerticalAlignment.Bottom
        )
        top_item = self._make_item(
            vertical=True, anchor=TextVerticalAlignment.Top
        )
        layout = bottom_item.layout
        last_line = bottom_item.document().firstBlock().layout().lineAt(
            len(self._lines(bottom_item)) - 1
        )
        cell_top, _cell_bottom = layout.y_offset_lst[0][-1]
        probe = QPointF(last_line.x() + 2.0, float(cell_top) + 5.0)
        self.assertEqual(
            layout.hitTest(probe, Qt.HitTestAccuracy.FuzzyHit), len(TEXT) - 1
        )
        self.assertEqual(
            top_item.layout.hitTest(probe, Qt.HitTestAccuracy.FuzzyHit),
            len(TEXT),
        )

    def test_setting_the_same_anchor_is_a_noop(self) -> None:
        item = self._make_item(
            vertical=True, anchor=TextVerticalAlignment.Center
        )
        before = self._lines(item)
        item.setVerticalAlignment(TextVerticalAlignment.Center)
        self.assertEqual(self._lines(item), before)

    def test_live_change_matches_fresh_construction(self) -> None:
        for vertical in (True, False):
            with self.subTest(vertical=vertical):
                live = self._make_item(
                    vertical=vertical, anchor=TextVerticalAlignment.Top
                )
                live.setVerticalAlignment(TextVerticalAlignment.Bottom)
                fresh = self._make_item(
                    vertical=vertical, anchor=TextVerticalAlignment.Bottom
                )
                live_lines = [(round(y, 2), s, n) for y, s, n in self._lines(live)]
                fresh_lines = [
                    (round(y, 2), s, n) for y, s, n in self._lines(fresh)
                ]
                self.assertEqual(live_lines, fresh_lines)


class VerticalAlignmentPersistenceTest(unittest.TestCase):

    def test_default_is_auto_and_round_trips(self) -> None:
        self.assertIs(
            FontFormat().vertical_alignment, TextVerticalAlignment.Auto
        )
        format_ = FontFormat()
        format_.vertical_alignment = TextVerticalAlignment.Bottom
        self.assertEqual(
            format_.to_serializable_dict()['vertical_alignment'], 2
        )
        restored = FontFormat(**format_.to_serializable_dict())
        self.assertIs(
            restored.vertical_alignment, TextVerticalAlignment.Bottom
        )

    def test_invalid_value_falls_back_to_auto(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertLogs(
                'BallonTranslator', level='WARNING'
            ):
                format_ = FontFormat(vertical_alignment='sideways')
        self.assertIs(
            format_.vertical_alignment, TextVerticalAlignment.Auto
        )

    def test_out_of_range_integer_falls_back_to_auto(self) -> None:
        with self.assertLogs(
            'BallonTranslator', level='WARNING'
        ):
            format_ = FontFormat(vertical_alignment=7)
        self.assertIs(
            format_.vertical_alignment, TextVerticalAlignment.Auto
        )


class VerticalAlignmentCommandTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_command_is_registered_for_the_format_field(self) -> None:
        from ballontranslator.ui.text_engine.formatting.commands import (
            handle_ffmt_change,
        )

        self.assertIn('vertical_alignment', handle_ffmt_change)
        self.assertTrue(callable(handle_ffmt_change['vertical_alignment']))

    def test_command_applies_the_anchor_as_one_undo_step(self) -> None:
        from unittest.mock import patch

        from ballontranslator.ui import shared_widget as SW
        from ballontranslator.ui.text_engine.formatting import commands

        item = VerticalAnchorLayoutTest()._make_item(
            vertical=True, anchor=TextVerticalAlignment.Top
        )
        pushed = []

        class _Canvas:
            def push_undo_command(self, command):
                pushed.append(command)
                command.redo()

        with patch.object(commands.SW, 'canvas', _Canvas(), create=True):
            commands.ffmt_change_vertical_alignment(
                'vertical_alignment',
                int(TextVerticalAlignment.Bottom),
                FontFormat(),
                False,
                [item],
                set_focus=False,
            )

        self.assertEqual(len(pushed), 1)
        self.assertIs(
            item.fontformat.vertical_alignment,
            TextVerticalAlignment.Bottom,
        )
        pushed[0].undo()
        self.assertIs(
            item.fontformat.vertical_alignment, TextVerticalAlignment.Top
        )


if __name__ == '__main__':
    unittest.main()
