import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


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
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.layout import (
    PUNSET_ALIGNCENTER,
    PUNSET_BRACKET,
    PUNSET_HALF,
    PUNSET_NONBRACKET,
    PUNSET_PAUSEORSTOP,
    PUNSET_STANDARD_VERTICAL_ROMAN,
)
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import FontFormat, TextAlignment
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

    def test_rotated_glyph_slant_uses_visible_baseline_in_both_modes(self):
        for text, roman in (('（A', True), ('AＡ', False)):
            with self.subTest(text=text, roman=roman):
                item = self._make_item(text, roman)
                block = item.document().firstBlock()
                line, offset, orientation = (
                    item.layout.vertical_line_placement(block, 0)
                )
                self.assertFalse(orientation.isIdentity())
                neutral = glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    0.0,
                )
                slanted = glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    45.0,
                )
                points = [
                    (
                        source.elementAt(index),
                        target.elementAt(index),
                    )
                    for source, target in zip(
                        neutral.paths, slanted.paths
                    )
                    for index in range(source.elementCount())
                ]
                self.assertTrue(points)
                top_source, top_target = min(
                    points, key=lambda pair: pair[0].y
                )
                bottom_source, bottom_target = max(
                    points, key=lambda pair: pair[0].y
                )
                self.assertGreater(
                    top_target.x - top_source.x,
                    bottom_target.x - bottom_source.x,
                )

    def test_mirrored_rotated_punctuation_keeps_the_same_column(self):
        item = self._make_item('（）', True)
        block = item.document().firstBlock()
        for angle in (-45.0, 45.0):
            shifts = []
            for line_number in range(2):
                line, offset, orientation = (
                    item.layout.vertical_line_placement(
                        block, line_number
                    )
                )
                neutral = glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    0.0,
                )
                slanted = glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    angle,
                )
                shifts.append(
                    slanted.bounds.center().x()
                    - neutral.bounds.center().x()
                )
            self.assertAlmostEqual(shifts[0], shifts[1], places=6)
            self.assertGreater(shifts[0] * angle, 0.0)

    def test_glyph_slant_enlargement_uses_stable_bounds_and_anchor(self):
        block = TextBlock([0, 0, 90, 70])
        block._bounding_rect = [0, 0, 90, 70]
        block.translation = 'A'
        block.fontformat.vertical = True
        block.fontformat.alignment = TextAlignment.Right
        block.fontformat.standard_vertical_roman_alignment = False
        block.fontformat.glyph_slant_angle = 24.0
        item = TextBlkItem(block, 0)
        renderer = item.geometry_controller.layout_renderer
        probes = []
        anchor = item.mapToScene(item.logical_unpadded_rect().topRight())

        def probe_geometry() -> None:
            probes.append(renderer.ink_bounds())

        item.layout.size_enlarged.connect(probe_geometry)
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText('(—BCDEFGH')
        item.setTextCursor(cursor)
        self.app.processEvents()

        self.assertTrue(probes)
        self.assertEqual(
            item.mapToScene(item.logical_unpadded_rect().topRight()),
            anchor,
        )

        wide_block = TextBlock([0, 0, 140, 220])
        wide_block._bounding_rect = [0, 0, 140, 220]
        wide_block.translation = '縦'
        wide_block.fontformat.vertical = True
        wide_block.fontformat.alignment = TextAlignment.Right
        wide_block.fontformat.standard_vertical_roman_alignment = True
        wide_block.fontformat.glyph_slant_angle = 18.0
        wide_item = TextBlkItem(wide_block, 1)
        wide_anchor = wide_item.mapToScene(
            wide_item.logical_unpadded_rect().topRight()
        )
        cursor = wide_item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText('書き、句読点。（ABC）—テスト縦書き。')
        wide_item.setTextCursor(cursor)
        self.app.processEvents()
        self.assertEqual(
            wide_item.mapToScene(
                wide_item.logical_unpadded_rect().topRight()
            ),
            wide_anchor,
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

    def test_deselection_keeps_roman_layout_and_effect_format_synchronized(self):
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        canvas = Canvas()
        SW.canvas = canvas
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(
            setattr, C, 'active_format', previous_active_format
        )
        self.addCleanup(canvas.gv.deleteLater)

        item = self._make_item('（A）', True)
        item.fontformat.stroke_width = 0.08
        item.setParentItem(canvas.textLayer)
        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        panel.global_format = FontFormat()
        self.addCleanup(panel.deleteLater)

        panel.set_textblk_item(item)
        panel.on_param_changed(
            'standard_vertical_roman_alignment', False
        )
        panel.set_textblk_item()
        panel.on_param_changed(
            'standard_vertical_roman_alignment', False
        )
        panel.set_textblk_item(item)
        panel.on_param_changed(
            'standard_vertical_roman_alignment', True
        )

        self.assertIs(item.fontformat, item.blk.fontformat)
        self.assertIs(item.fontformat, item.layout.fontformat)
        self.assertTrue(self._orientation(item, 1).isIdentity())
        self.assertIsNotNone(item.effect_renderer.background_pixmap)

        panel.set_textblk_item(item)
        panel.text_transform_session.detach_scene_owner()
        self.assertIs(item.fontformat, item.blk.fontformat)
        self.assertIs(item.fontformat, item.layout.fontformat)

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
