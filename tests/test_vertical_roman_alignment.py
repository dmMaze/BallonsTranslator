import os
import unittest
from types import SimpleNamespace


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QRectF
from qtpy.QtGui import QTextCursor
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QCheckBox
try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.formatting.commands import (
    ffmt_change_standard_vertical_roman_alignment,
    ffmt_change_vertical,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.layout import (
    PUNSET_ALIGNCENTER,
    PUNSET_BRACKET,
    PUNSET_HALF,
    PUNSET_NONBRACKET,
    PUNSET_PAUSEORSTOP,
    PUNSET_STANDARD_VERTICAL_ROMAN,
)
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


class VerticalRomanAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        text: str,
        standard_vertical_roman_alignment: bool,
    ) -> TextBlkItem:
        block = TextBlock([0, 0, 220, 900])
        block._bounding_rect = [0, 0, 220, 900]
        block.translation = text
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.letter_spacing = 1.0
        block.fontformat.standard_vertical_roman_alignment = (
            standard_vertical_roman_alignment
        )
        return TextBlkItem(block, 0)

    @staticmethod
    def _orientation(item: TextBlkItem, position: int):
        block = item.document().firstBlock()
        line = block.layout().lineForTextPosition(position)
        placement = item.layout.vertical_line_placement(
            block, line.lineNumber()
        )
        return placement[2]

    @staticmethod
    def _ink_and_cell(
        item: TextBlkItem,
        position: int,
    ) -> tuple[QRectF, QRectF]:
        block = item.document().firstBlock()
        line = block.layout().lineForTextPosition(position)
        line_number = line.lineNumber()
        char_format = item.layout.get_char_fontfmt(0, position)
        actual = char_format.punc_actual_rect(
            line, block.text()[position], cache=True
        )
        x_offset, y_offset = item.layout._draw_offset[0][line_number]
        ink = QRectF(
            line.x() + x_offset + actual[0],
            line.y() + y_offset + actual[1],
            actual[2],
            actual[3],
        )
        line_width = item.layout.per_char_records[0][position][
            'line_width'
        ]
        top, bottom = item.layout.y_offset_lst[0][position]
        return ink, QRectF(line.x(), top, line_width, bottom - top)

    def test_missing_project_field_uses_default_enabled_alignment(self):
        legacy_block = TextBlock(
            xyxy=[0, 0, 100, 100],
            fontformat={'vertical': True},
        )

        self.assertTrue(
            legacy_block.fontformat.standard_vertical_roman_alignment
        )
        disabled = FontFormat(standard_vertical_roman_alignment=False)
        self.assertFalse(
            disabled.to_serializable_dict()[
                'standard_vertical_roman_alignment'
            ]
        )
        malformed = FontFormat(
            standard_vertical_roman_alignment='sideways'
        )
        self.assertTrue(malformed.standard_vertical_roman_alignment)

    def test_standard_mode_keeps_roman_upright(self):
        text = 'A9(éＡ。（—'
        standard = self._make_item(text, True)
        chinese = self._make_item(text, False)

        for position in (0, 1, 2, 3):
            self.assertTrue(
                self._orientation(standard, position).isIdentity()
            )
            self.assertFalse(
                self._orientation(chinese, position).isIdentity()
            )
        for position in (4, 5):
            self.assertTrue(
                self._orientation(standard, position).isIdentity()
            )
            self.assertTrue(
                self._orientation(chinese, position).isIdentity()
            )
        for position in (6, 7):
            self.assertFalse(
                self._orientation(standard, position).isIdentity()
            )
            self.assertFalse(
                self._orientation(chinese, position).isIdentity()
            )

    def test_standard_roman_ink_is_centered_in_its_cell(self):
        text = 'Aa09!?()é'
        item = self._make_item(text, True)

        for position, char in enumerate(text):
            with self.subTest(char=char):
                ink, cell = self._ink_and_cell(item, position)
                self.assertAlmostEqual(
                    ink.center().x(), cell.center().x(), delta=1.0
                )
                self.assertAlmostEqual(
                    ink.center().y(), cell.center().y(), delta=1.0
                )

    def test_item_switch_relayouts_existing_vertical_text(self):
        item = self._make_item('ABC。', True)
        previous_generation = item.layout.layout_generation

        item.setStandardVerticalRomanAlignment(False)

        self.assertFalse(
            item.fontformat.standard_vertical_roman_alignment
        )
        self.assertGreater(item.layout.layout_generation, previous_generation)
        self.assertFalse(self._orientation(item, 0).isIdentity())

    def test_item_switch_uses_canvas_undo_history(self):
        item = self._make_item('ABC', True)
        stack = QUndoStack()
        previous_canvas = SW.canvas
        SW.canvas = SimpleNamespace(push_undo_command=stack.push)
        try:
            ffmt_change_standard_vertical_roman_alignment(
                'standard_vertical_roman_alignment',
                False,
                FontFormat(),
                False,
                [item],
            )
            self.assertFalse(
                item.blk.fontformat.standard_vertical_roman_alignment
            )
            stack.undo()
            self.assertTrue(
                item.blk.fontformat.standard_vertical_roman_alignment
            )
            stack.redo()
            self.assertFalse(
                item.blk.fontformat.standard_vertical_roman_alignment
            )
        finally:
            SW.canvas = previous_canvas

    def test_item_switches_restore_canvas_keyboard_focus(self):
        canvas = Canvas()
        canvas.gv.resize(800, 500)
        canvas.gv.show()
        focus_thief = QCheckBox(canvas.gv)
        focus_thief.show()
        previous_canvas = SW.canvas
        SW.canvas = canvas
        try:
            for switch, param_name, value in (
                (ffmt_change_vertical, 'vertical', False),
                (
                    ffmt_change_standard_vertical_roman_alignment,
                    'standard_vertical_roman_alignment',
                    False,
                ),
            ):
                with self.subTest(param_name=param_name):
                    item = self._make_item('ABC', True)
                    item.setParentItem(canvas.textLayer)
                    item.startEdit()
                    cursor = item.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    item.setTextCursor(cursor)

                    focus_thief.setFocus()
                    self.app.processEvents()
                    self.assertTrue(focus_thief.hasFocus())

                    switch(
                        param_name,
                        value,
                        item.fontformat,
                        False,
                        [item],
                        set_focus=True,
                    )
                    self.app.processEvents()

                    self.assertTrue(item.isEditing())
                    self.assertTrue(canvas.gv.hasFocus())
                    QTest.keyClicks(QApplication.focusWidget(), 'x')
                    self.assertEqual(item.toPlainText(), 'ABCx')
                    item.setParentItem(None)
        finally:
            SW.canvas = previous_canvas
            focus_thief.close()
            canvas.gv.close()

    def test_standard_punctuation_is_centered_and_chinese_is_upper_right(self):
        for char in PUNSET_PAUSEORSTOP:
            with self.subTest(char=char, mode='standard'):
                standard = self._make_item(char, True)
                ink, cell = self._ink_and_cell(standard, 0)
                self.assertAlmostEqual(
                    ink.center().x(), cell.center().x(), delta=1.0
                )
                self.assertAlmostEqual(
                    ink.center().y(), cell.center().y(), delta=1.0
                )
            with self.subTest(char=char, mode='chinese'):
                chinese = self._make_item(char, False)
                ink, cell = self._ink_and_cell(chinese, 0)
                self.assertAlmostEqual(
                    ink.right(), cell.right(), delta=1.0
                )
                self.assertAlmostEqual(
                    ink.top(), cell.top(), delta=1.0
                )

    def test_interpuncts_and_bullets_stay_centered(self):
        for char in PUNSET_ALIGNCENTER:
            for enabled in (True, False):
                with self.subTest(char=char, enabled=enabled):
                    item = self._make_item(char, enabled)
                    ink, cell = self._ink_and_cell(item, 0)
                    self.assertAlmostEqual(
                        ink.center().x(), cell.center().x(), delta=1.0
                    )
                    self.assertAlmostEqual(
                        ink.center().y(), cell.center().y(), delta=1.0
                    )

    def test_clreq_punctuation_orientation_groups(self):
        standard = self._make_item('字', True)
        chinese = self._make_item('字', False)

        for char in PUNSET_PAUSEORSTOP:
            self.assertFalse(
                chinese.layout.needs_vertical_rotation(char), char
            )
        for char in PUNSET_NONBRACKET:
            self.assertTrue(
                chinese.layout.needs_vertical_rotation(char), char
            )
        for char in PUNSET_BRACKET | PUNSET_HALF:
            self.assertTrue(
                chinese.layout.needs_vertical_rotation(char), char
            )
        for char in PUNSET_STANDARD_VERTICAL_ROMAN:
            self.assertTrue(
                standard.layout.needs_vertical_rotation(char), char
            )
        for char in PUNSET_HALF - PUNSET_STANDARD_VERTICAL_ROMAN:
            self.assertFalse(
                standard.layout.needs_vertical_rotation(char), char
            )


if __name__ == '__main__':
    unittest.main()
