import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QTextCharFormat,
    QTextCursor,
    QTextLayout,
)
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsScene,
    QLineEdit,
)
try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.custom_widget import SizeComboBox
from ballontranslator.ui.text_engine.formatting.commands import (
    ffmt_change_letter_spacing,
    ffmt_change_standard_vertical_roman_alignment,
    ffmt_change_vertical,
)
from ballontranslator.ui.text_engine.formatting.panel import FontFormatPanel
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.layout import CharFontFormat
from ballontranslator.ui.text_engine.vertical_layout import (
    PUNSET_ALIGNCENTER,
    PUNSET_BRACKET,
    PUNSET_HALF,
    PUNSET_NONBRACKET,
    PUNSET_PAUSEORSTOP,
    PUNSET_STANDARD_VERTICAL_ROMAN,
    format_punc_actual_rect,
    punc_actual_rect_cached,
)
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import (
    BendTextTransform,
    FontFormat,
    TextAlignment,
    TextTransformStack,
)
from ballontranslator.utils.textblock import TEXT_LAYOUT_VERSION, TextBlock


class VerticalRomanAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        text: str,
        standard_vertical_roman_alignment: bool,
        letter_spacing: float = 1.0,
    ) -> TextBlkItem:
        block = TextBlock([0, 0, 220, 900])
        block._bounding_rect = [0, 0, 220, 900]
        block.translation = text
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.letter_spacing = letter_spacing
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
        actual = format_punc_actual_rect(
            char_format,
            line,
            block.text()[position],
            cache=True,
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

    @staticmethod
    def _make_overflow_item(
        *,
        alignment: TextAlignment = TextAlignment.Left,
        standard: bool = False,
        vertical: bool = True,
        stroke_width: float = 0.0,
        text: str = 'g一般ajpqy',
        bounds: tuple[float, float] = (260, 210),
        font_size: float = 64,
    ) -> TextBlkItem:
        width, height = bounds
        block = TextBlock(
            [0, 0, width, height],
            text_layout_version=TEXT_LAYOUT_VERSION,
        )
        block._bounding_rect = [0, 0, width, height]
        block.translation = text
        block.fontformat.vertical = vertical
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = font_size
        block.fontformat.alignment = alignment
        block.fontformat.standard_vertical_roman_alignment = standard
        block.fontformat.stroke_width = stroke_width
        return TextBlkItem(block, 0)

    @staticmethod
    def _transformed_line_ink(item: TextBlkItem) -> tuple[QRectF, ...]:
        ink = []
        block = item.document().firstBlock()
        while block.isValid():
            text_layout = block.layout()
            placement_for_line = getattr(
                item.layout, 'vertical_line_placement', None
            )
            if placement_for_line is None:
                block = block.next()
                continue
            for line_number in range(text_layout.lineCount()):
                placement = placement_for_line(block, line_number)
                if placement is None or placement[2].isIdentity():
                    continue
                line, offset, orientation = placement
                ink.append(glyph_geometry(
                    line,
                    line.textStart(),
                    line.textLength(),
                    offset,
                    orientation,
                    0.0,
                ).bounds)
            block = block.next()
        return tuple(ink)

    def test_rotated_roman_ink_extends_real_item_geometry_after_resize(self):
        for alignment in (
            TextAlignment.Left,
            TextAlignment.Center,
            TextAlignment.Right,
        ):
            with self.subTest(alignment=alignment):
                item = self._make_overflow_item(alignment=alignment)
                logical = QRectF(item.logical_unpadded_rect())

                for width in (260.0, 315.0, 205.0):
                    item.set_size(
                        width,
                        logical.height(),
                        set_layout_maxsize=True,
                    )
                    self.app.processEvents()
                    logical = QRectF(item.logical_unpadded_rect())
                    ink = self._transformed_line_ink(item)

                    self.assertGreater(len({
                        item.document().firstBlock().layout()
                        .lineAt(index).x()
                        for index in range(
                            item.document().firstBlock().layout().lineCount()
                        )
                    }), 1)
                    self.assertTrue(ink)
                    self.assertEqual(logical.width(), width)
                    for bounds in ink:
                        self.assertTrue(
                            item.geometry_controller.source_paint_rect()
                            .contains(bounds)
                        )
                        self.assertTrue(item.boundingRect().contains(bounds))
                        self.assertTrue(
                            item.shape().boundingRect().contains(bounds)
                        )
                        self.assertTrue(item.contains(bounds.center()))

    def test_upright_roman_and_horizontal_paths_keep_native_bounds(self):
        for alignment in (
            TextAlignment.Left,
            TextAlignment.Center,
            TextAlignment.Right,
        ):
            with self.subTest(alignment=alignment):
                item = self._make_overflow_item(
                    alignment=alignment,
                    standard=True,
                )
                for width in (260.0, 205.0):
                    item.set_size(
                        width,
                        item.logical_unpadded_rect().height(),
                        set_layout_maxsize=True,
                    )
                    self.app.processEvents()
                    self.assertEqual(
                        item.logical_unpadded_rect().width(), width
                    )
                    self.assertTrue(item.layout.base_ink_bounds().isEmpty())
                    self.assertEqual(
                        item.geometry_controller.source_paint_rect(),
                        item.geometry_controller.source_rect(),
                    )

        item = self._make_overflow_item(vertical=False)

        self.assertTrue(item.layout.base_ink_bounds().isEmpty())
        self.assertEqual(
            item.geometry_controller.source_paint_rect(),
            item.geometry_controller.source_rect(),
        )

    def test_rotated_ink_measurement_is_cached_by_settled_layout(self):
        with patch(
            'ballontranslator.ui.text_engine.vertical_layout.glyph_geometry',
            wraps=glyph_geometry,
        ) as measure:
            item = self._make_overflow_item()
            settled_calls = measure.call_count
            line_count = item.document().firstBlock().layout().lineCount()
            self.assertGreater(settled_calls, 0)
            self.assertLessEqual(settled_calls, line_count)

            for _ in range(20):
                item.layout.base_ink_bounds()
                item.geometry_controller.source_paint_rect()
                item.boundingRect()
                item.shape()

            self.assertEqual(measure.call_count, settled_calls)

    def test_rotated_ink_effect_and_transform_bounds_share_one_owner(self):
        item = self._make_overflow_item(stroke_width=0.14)
        neutral_ink = item.layout.base_ink_bounds()
        padding = item.padding()
        self.assertGreater(padding, 0.0)
        self.assertTrue(
            item.geometry_controller.source_paint_rect().contains(
                neutral_ink.adjusted(-padding, -padding, padding, padding)
            )
        )
        scene = QGraphicsScene()
        scene.addItem(item)
        scene_rect = QRectF(item.boundingRect())
        scene.setSceneRect(scene_rect)
        scale = 2.0
        image = QImage(
            math.ceil(scene_rect.width() * scale),
            math.ceil(scene_rect.height() * scale),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            scene.render(painter)
        finally:
            painter.end()
        logical_left = math.floor(
            (item.logical_unpadded_rect().left() - scene_rect.left())
            * image.width() / scene_rect.width()
        )
        self.assertGreater(logical_left, 0)
        self.assertTrue(any(
            image.pixelColor(x, y).alpha() > 0
            for x in range(logical_left)
            for y in range(image.height())
        ))

        item.set_text_transform(TextTransformStack((
            BendTextTransform(0.2),
        )))
        controller = item.geometry_controller
        self.assertIsNone(controller.layout_renderer)
        self.assertEqual(
            controller.layout_ink_bounds(), item.layout.base_ink_bounds()
        )
        mapped_center = controller.visual_mapper.forward_point(
            neutral_ink.center()
        )
        self.assertTrue(item.boundingRect().contains(mapped_center))
        self.assertTrue(item.shape().contains(mapped_center))

        item.set_text_transform(TextTransformStack(
            (BendTextTransform(0.2),),
            18.0,
        ))
        renderer_ink = controller.layout_renderer.ink_bounds()
        padding = item.padding()
        self.assertEqual(controller.layout_ink_bounds(), renderer_ink)
        self.assertTrue(
            controller.source_paint_rect().contains(
                renderer_ink.adjusted(
                    -padding, -padding, padding, padding
                )
            )
        )
        mapped_center = controller.visual_mapper.forward_point(
            renderer_ink.center()
        )
        self.assertTrue(item.boundingRect().contains(mapped_center))
        self.assertTrue(item.shape().contains(mapped_center))

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

    def test_orientation_toggle_updates_scene_overflow_in_both_directions(self):
        item = self._make_overflow_item(
            standard=True,
            text='一般abcg',
            bounds=(180, 520),
            font_size=96,
        )
        scene = QGraphicsScene()
        scene.setItemIndexMethod(
            QGraphicsScene.ItemIndexMethod.BspTreeIndex
        )
        scene.setSceneRect(QRectF(-100, -100, 400, 800))
        scene.addItem(item)
        logical = QRectF(item.logical_unpadded_rect())
        prepare_states = []
        prepare_geometry_change = item.prepareGeometryChange

        def observe_prepare() -> None:
            prepare_states.append((
                item.fontformat.standard_vertical_roman_alignment,
                item.layout.base_ink_bounds().isEmpty(),
            ))
            prepare_geometry_change()

        with patch.object(
            item,
            'prepareGeometryChange',
            side_effect=observe_prepare,
        ) as prepare:
            item.setStandardVerticalRomanAlignment(False)
            self.app.processEvents()
            g_ink = self._transformed_line_ink(item)[-1]
            outside = QPointF(
                (g_ink.left() + logical.left()) / 2,
                g_ink.center().y(),
            )
            scene_point = item.mapToScene(outside)
            self.assertLess(outside.x(), logical.left())
            self.assertIn(item, scene.items(scene_point))
            self.assertTrue(item.sceneBoundingRect().contains(scene_point))
            self.assertEqual(item.logical_unpadded_rect(), logical)

            item.setStandardVerticalRomanAlignment(True)
            self.app.processEvents()
            self.assertNotIn(item, scene.items(scene_point))
            self.assertFalse(item.sceneBoundingRect().contains(scene_point))
            self.assertEqual(item.logical_unpadded_rect(), logical)

            self.assertEqual(prepare.call_count, 2)

        self.assertEqual(prepare_states, [(True, True), (False, False)])

        item.setStandardVerticalRomanAlignment(False)
        self.app.processEvents()
        self.assertIn(item, scene.items(scene_point))

    def test_document_edit_updates_scene_overflow_without_item_hook(self):
        item = self._make_overflow_item(
            text='一般abc',
            bounds=(180, 520),
            font_size=96,
        )
        scene = QGraphicsScene()
        scene.setItemIndexMethod(
            QGraphicsScene.ItemIndexMethod.BspTreeIndex
        )
        scene.setSceneRect(QRectF(-100, -100, 400, 800))
        scene.addItem(item)
        logical = QRectF(item.logical_unpadded_rect())
        before_bounds = QRectF(item.sceneBoundingRect())

        cursor = QTextCursor(item.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText('g')
        self.app.processEvents()
        g_ink = self._transformed_line_ink(item)[-1]
        outside = QPointF(
            (g_ink.left() + logical.left()) / 2,
            g_ink.center().y(),
        )
        scene_point = item.mapToScene(outside)
        self.assertIn(item, scene.items(scene_point))
        self.assertTrue(item.sceneBoundingRect().contains(scene_point))
        self.assertLess(item.sceneBoundingRect().left(), before_bounds.left())
        self.assertEqual(item.logical_unpadded_rect(), logical)

        cursor.deletePreviousChar()
        self.app.processEvents()
        self.assertNotIn(item, scene.items(scene_point))
        self.assertEqual(item.sceneBoundingRect(), before_bounds)
        self.assertEqual(item.logical_unpadded_rect(), logical)

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

    def test_live_formatting_keeps_keyboard_editor_focus(self):
        canvas = Canvas()
        canvas.gv.resize(800, 500)
        canvas.gv.show()
        previous_canvas = SW.canvas
        SW.canvas = canvas
        try:
            item = self._make_item('ABC', True)
            item.setParentItem(canvas.textLayer)
            item.startEdit()
            editors = (
                QLineEdit(canvas.gv),
                SizeComboBox([0, 10], 'letter_spacing', canvas.gv),
            )
            for editor in editors:
                with self.subTest(editor=type(editor).__name__):
                    editor.show()
                    editor.setFocus()
                    self.app.processEvents()
                    self.assertTrue(editor.hasFocus())

                    ffmt_change_letter_spacing(
                        'letter_spacing',
                        1.5,
                        item.fontformat,
                        False,
                        [item],
                        set_focus=True,
                    )
                    self.app.processEvents()

                    self.assertTrue(editor.hasFocus())
                    QTest.keyClicks(editor, '7')
                    self.assertIn('7', editor.text() if isinstance(
                        editor, QLineEdit
                    ) else editor.currentText())
                    editor.close()
            item.setParentItem(None)
        finally:
            SW.canvas = previous_canvas
            canvas.gv.close()

    def test_joined_rotated_punctuation_spacing_keeps_its_column(self):
        for roman, height in ((False, 85), (True, 100)):
            with self.subTest(roman=roman):
                block = TextBlock(
                    [0, 0, 240, height],
                    text_layout_version=TEXT_LAYOUT_VERSION,
                )
                block._bounding_rect = [0, 0, 240, height]
                block.translation = 'A——B'
                block.fontformat.vertical = True
                block.fontformat.font_family = 'Noto Sans CJK SC'
                block.fontformat.font_size = 40
                block.fontformat.letter_spacing = 1.0
                block.fontformat.alignment = TextAlignment.Right
                block.fontformat.standard_vertical_roman_alignment = roman
                item = TextBlkItem(block, 0)

                text_block = item.document().firstBlock()
                before_x = text_block.layout().lineForTextPosition(1).x()
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(1)
                cursor.setPosition(
                    3, QTextCursor.MoveMode.KeepAnchor
                )
                item.setTextCursor(cursor)
                item.setLetterSpacing(1.5)
                self.app.processEvents()

                text_block = item.document().firstBlock()
                after_x = text_block.layout().lineForTextPosition(1).x()
                self.assertEqual(after_x, before_x)

    def test_preceding_spacing_keeps_joined_punctuation_in_its_column(self):
        for roman in (False, True):
            for alignment in (
                TextAlignment.Left,
                TextAlignment.Center,
                TextAlignment.Right,
            ):
                with self.subTest(roman=roman, alignment=alignment):
                    self._assert_preceding_spacing_keeps_joined_column(
                        roman, alignment
                    )

    def _assert_preceding_spacing_keeps_joined_column(
        self,
        roman: bool,
        alignment: TextAlignment,
    ) -> None:
        block = TextBlock(
            [0, 0, 240, 500],
            text_layout_version=TEXT_LAYOUT_VERSION,
        )
        block._bounding_rect = [0, 0, 240, 500]
        block.translation = '木——'
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.letter_spacing = 1.0
        block.fontformat.alignment = alignment
        block.fontformat.standard_vertical_roman_alignment = roman
        item = TextBlkItem(block, 0)
        item.squeezeBoundingRect()

        text_block = item.document().firstBlock()
        dash_line = text_block.layout().lineForTextPosition(1)
        before = item.mapToScene(dash_line.position())
        before_height = item.logical_unpadded_rect().height()
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.3)
        self.app.processEvents()

        text_block = item.document().firstBlock()
        dash_line = text_block.layout().lineForTextPosition(1)
        after = item.mapToScene(dash_line.position())
        self.assertAlmostEqual(after.x(), before.x())
        self.assertGreater(after.y(), before.y())
        self.assertGreater(
            item.logical_unpadded_rect().height(), before_height
        )

    def test_nonediting_spacing_keeps_tight_single_column(self):
        for roman in (False, True):
            with self.subTest(roman=roman):
                source = self._make_item('木——', roman, 1.1)
                source.squeezeBoundingRect()
                rect = source.absBoundingRect(qrect=True)
                xyxy = [
                    rect.left(), rect.top(), rect.right(), rect.bottom()
                ]
                loaded_block = TextBlock(
                    xyxy,
                    fontformat=source.fontformat.to_serializable_dict(),
                    text_layout_version=TEXT_LAYOUT_VERSION,
                )
                loaded_block._bounding_rect = xyxy
                loaded_block.translation = '木——'
                loaded_block.rich_text = source.toHtml()
                item = TextBlkItem(loaded_block, 0)
                item.startEdit()
                item.endEdit()

                text_block = item.document().firstBlock()
                before = item.mapToScene(
                    text_block.layout().lineForTextPosition(1).position()
                )
                item.setLetterSpacing(1.5)
                self.app.processEvents()

                text_block = item.document().firstBlock()
                after = item.mapToScene(
                    text_block.layout().lineForTextPosition(1).position()
                )
                self.assertAlmostEqual(after.x(), before.x())
                self.assertGreater(after.y(), before.y())

    def test_punctuation_rect_cache_tracks_line_geometry(self):
        font = QFont('Noto Sans CJK SC')
        font.setPointSizeF(40)
        char_format = QTextCharFormat()
        char_format.setFont(font)
        cached_format = CharFontFormat(char_format)

        def make_line(first_spacing: float, second_spacing: float):
            line_font = QFont(font)
            layout = QTextLayout('((', line_font)
            formats = []
            for position, spacing in enumerate((
                first_spacing,
                second_spacing,
            )):
                format_range = QTextLayout.FormatRange()
                format_range.start = position
                format_range.length = 1
                format_range.format = QTextCharFormat(char_format)
                format_range.format.setFontLetterSpacingType(
                    QFont.SpacingType.PercentageSpacing
                )
                format_range.format.setFontLetterSpacing(spacing)
                formats.append(format_range)
            layout.setFormats(formats)
            layout.beginLayout()
            line = layout.createLine()
            line.setLineWidth(1000)
            layout.endLayout()
            return layout, line

        first_layout, first_line = make_line(200, 100)
        second_layout, second_line = make_line(100, 200)
        self.assertEqual(
            first_line.naturalTextWidth(), second_line.naturalTextWidth()
        )
        punc_actual_rect_cached.cache_clear()
        first_rect = format_punc_actual_rect(
            cached_format, first_line, '((', cache=True
        )
        cached_second = format_punc_actual_rect(
            cached_format, second_line, '((', cache=True
        )
        uncached_second = format_punc_actual_rect(
            cached_format, second_line, '((', cache=False
        )
        self.assertNotEqual(first_rect, uncached_second)
        self.assertEqual(cached_second, uncached_second)

    def test_invalid_spacing_does_not_mutate_format_or_geometry(self):
        item = self._make_item('木——', True)
        item.squeezeBoundingRect()
        before_rect = QRectF(item.logical_unpadded_rect())
        before_spacing = item.fontformat.letter_spacing

        with self.assertRaises(ValueError):
            item.setLetterSpacing(11.0)

        self.assertEqual(item.logical_unpadded_rect(), before_rect)
        self.assertEqual(item.fontformat.letter_spacing, before_spacing)

    def test_zero_spacing_keeps_vertical_cursor_cells_monotonic(self):
        for roman in (False, True):
            for text in ('木——木', '木ii木'):
                with self.subTest(roman=roman, text=text):
                    item = self._make_item(text, roman)
                    item.startEdit()
                    cursor = item.textCursor()
                    cursor.setPosition(1)
                    cursor.setPosition(
                        3, QTextCursor.MoveMode.KeepAnchor
                    )
                    item.setTextCursor(cursor)
                    item.setLetterSpacing(0)
                    self.app.processEvents()

                    offsets = item.layout.y_offset_lst[0]
                    self.assertTrue(
                        all(top <= bottom for top, bottom in offsets)
                    )
                    self.assertEqual(offsets, sorted(offsets))

    def test_joined_punctuation_negative_spacing_reduces_run_advance(self):
        for roman in (False, True):
            with self.subTest(roman=roman):
                item = self._make_item('木——木', roman)
                block = item.document().firstBlock()
                before = block.layout().lineForTextPosition(3).position()

                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(1)
                cursor.setPosition(
                    3, QTextCursor.MoveMode.KeepAnchor
                )
                item.setTextCursor(cursor)
                item.setLetterSpacing(0.5)
                self.app.processEvents()

                block = item.document().firstBlock()
                after = block.layout().lineForTextPosition(3).position()
                self.assertEqual(after.x(), before.x())
                self.assertLess(after.y(), before.y())

    def test_lone_final_glyph_uses_configured_column_spacing(self):
        block = TextBlock(
            [0, 0, 300, 140],
            text_layout_version=TEXT_LAYOUT_VERSION,
        )
        block._bounding_rect = [0, 0, 300, 140]
        block.translation = '木木木—'
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = 40
        block.fontformat.line_spacing = 1.5
        item = TextBlkItem(block, 0)

        text_block = item.document().firstBlock()
        first_line = text_block.layout().lineForTextPosition(0)
        final_line = text_block.layout().lineForTextPosition(3)
        final_width = item.layout._line_record(
            text_block, final_line.lineNumber()
        )['line_width']

        self.assertNotEqual(first_line.x(), final_line.x())
        self.assertAlmostEqual(
            first_line.x() - final_line.x(),
            item.layout.calculate_line_spacing(final_width, 1.5),
        )

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
