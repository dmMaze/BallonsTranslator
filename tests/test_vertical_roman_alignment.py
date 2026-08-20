import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
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
from ballontranslator.ui.text_engine.vertical_layout import (
    _LINE_INK_BOUNDS_CACHE,
    _line_ink_bounds,
    _line_ink_cache_key,
    _uncached_line_ink_bounds,
    PUNSET_ALIGNCENTER,
    PUNSET_BRACKET,
    PUNSET_COMPACT,
    PUNSET_HALF,
    PUNSET_NONBRACKET,
    PUNSET_PAUSEORSTOP,
    PUNSET_STANDARD_VERTICAL_ROMAN,
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
        font_size: float = 40.0,
    ) -> TextBlkItem:
        block = TextBlock([0, 0, 220, 900])
        block._bounding_rect = [0, 0, 220, 900]
        block.translation = text
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Noto Sans CJK SC'
        block.fontformat.font_size = font_size
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
        placement = item.layout.vertical_line_placement(block, line_number)
        placed_line, offset, orientation = placement
        ink = glyph_geometry(
            placed_line,
            placed_line.textStart(),
            placed_line.textLength(),
            offset,
            orientation,
            0.0,
        ).bounds
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

    def test_fresh_project_item_does_not_add_glyph_bearing_padding(self):
        def make_item(text: str) -> TextBlkItem:
            payload = {
                'xyxy': [81, 111, 220, 362],
                '_bounding_rect': [234, 93, 30, 269],
                'translation': text,
                'rich_text': (
                    '<html><body style="font-family:\'Noto Sans\'; '
                    'font-size:22.5pt; font-weight:700">'
                    '<p><span data-btrans-letter-spacing="1.23">'
                    f'{text}</span></p></body></html>'
                ),
                'fontformat': {
                    'font_family': 'Noto Sans',
                    'font_size': 30.0,
                    'font_weight': 700,
                    'bold': True,
                    'alignment': TextAlignment.Left,
                    'vertical': True,
                    'standard_vertical_roman_alignment': False,
                    'letter_spacing': 1.23,
                },
                'text_layout_version': TEXT_LAYOUT_VERSION,
            }
            return TextBlkItem(TextBlock(**payload), 12, show_rect=True)

        reference = make_item('一般abc')
        item = make_item('一般abcg')
        scene = QGraphicsScene()
        scene.addItem(reference)
        scene.addItem(item)
        self.app.processEvents()

        # Only the settled native column may enlarge the saved box; glyph
        # bearings are paint overflow, not layout width.
        native_width = max(30.0, item.layout._column_content_width())
        actual_rect = item.absBoundingRect(qrect=True)
        actual_width = actual_rect.width()
        self.assertEqual(actual_rect.topLeft(), QPointF(234, 93))
        self.assertEqual(actual_rect.height(), 269.0)
        self.assertGreaterEqual(actual_width, native_width)
        self.assertLessEqual(actual_width - native_width, 1.0)
        self.assertEqual(
            item.blk._bounding_rect,
            [234, 93, math.ceil(actual_width), 269],
        )
        self.assertEqual(item.logical_unpadded_rect().width(), actual_width)
        self.assertEqual(
            item.geometry_controller.visual_outline_in_item().boundingRect(),
            item.logical_unpadded_rect(),
        )

        for position in (0, 1):
            reference_line = (
                reference.document().firstBlock().layout()
                .lineForTextPosition(position)
            )
            item_line = (
                item.document().firstBlock().layout()
                .lineForTextPosition(position)
            )
            self.assertEqual(item_line.position(), reference_line.position())
            self.assertEqual(
                item.mapToScene(item_line.position()),
                reference.mapToScene(reference_line.position()),
            )

        first_ink, first_cell = self._ink_and_cell(item, 0)
        reference_first_ink = self._ink_and_cell(reference, 0)[0]
        self.assertEqual(
            item.mapRectToScene(first_ink),
            reference.mapRectToScene(reference_first_ink),
        )
        self.assertAlmostEqual(
            first_ink.center().x(),
            first_cell.center().x(),
            delta=1 / 64 + 0.001,
        )
        g_ink = self._transformed_line_ink(item)[-1]
        outside = QPointF(
            (g_ink.left() + item.logical_unpadded_rect().left()) / 2,
            g_ink.center().y(),
        )
        self.assertLess(g_ink.left(), item.logical_unpadded_rect().left())
        self.assertTrue(
            item.geometry_controller.source_paint_rect().contains(g_ink)
        )
        self.assertTrue(item.boundingRect().contains(g_ink))
        self.assertTrue(item.shape().contains(outside))
        self.assertIn(item, scene.items(item.mapToScene(outside)))

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

    def test_small_vertical_ink_shares_exact_column_center(self):
        tolerance = 1 / 64 + 0.001
        for font_size in (5.0, 16.0):
            for standard in (True, False):
                item = self._make_item(
                    '啊大木・—…!', standard, font_size=font_size
                )
                for position, char in enumerate(item.toPlainText()):
                    with self.subTest(
                        font_size=font_size,
                        standard=standard,
                        char=char,
                    ):
                        ink, cell = self._ink_and_cell(item, position)
                        self.assertAlmostEqual(
                            ink.center().x(),
                            cell.center().x(),
                            delta=tolerance,
                        )

    def test_line_ink_bounds_cache_reuses_exact_shaping(self):
        font = QFont('Noto Sans CJK SC')
        font.setPointSizeF(5)
        self.addCleanup(_LINE_INK_BOUNDS_CACHE.clear)

        def make_line(text='木', spacings=()):
            layout = QTextLayout(text, font)
            formats = []
            for position, spacing in enumerate(spacings):
                char_format = QTextCharFormat()
                char_format.setFont(font)
                char_format.setFontLetterSpacingType(
                    QFont.SpacingType.PercentageSpacing
                )
                char_format.setFontLetterSpacing(spacing)
                format_range = QTextLayout.FormatRange()
                format_range.start = position
                format_range.length = 1
                format_range.format = char_format
                formats.append(format_range)
            layout.setFormats(formats)
            layout.beginLayout()
            line = layout.createLine()
            line.setLineWidth(1000)
            layout.endLayout()
            return layout, line

        repeated = [make_line() for _ in range(8)]
        _LINE_INK_BOUNDS_CACHE.clear()
        with patch(
            'ballontranslator.ui.text_engine.vertical_layout.'
            '_uncached_line_ink_bounds',
            wraps=_uncached_line_ink_bounds,
        ) as measure:
            for layout, line in repeated:
                _line_ink_bounds(line)
            for layout, line in repeated:
                _line_ink_bounds(line)
        self.assertEqual(measure.call_count, 1)
        self.assertEqual(len(_LINE_INK_BOUNDS_CACHE), 1)

        first_layout, first_line = make_line('((', (200.0, 100.0))
        second_layout, second_line = make_line('((', (100.0, 200.0))
        self.assertEqual(
            first_line.naturalTextWidth(), second_line.naturalTextWidth()
        )
        first_bounds = _line_ink_bounds(first_line)
        second_bounds = _line_ink_bounds(second_line)
        self.assertNotEqual(first_bounds, second_bounds)
        self.assertEqual(
            second_bounds, _uncached_line_ink_bounds(second_line)
        )

        _LINE_INK_BOUNDS_CACHE.clear()
        cache_keys = []
        with patch.object(_LINE_INK_BOUNDS_CACHE, 'max_entries', 2):
            layouts = []
            for text in ('木', '大', '啊'):
                layout, line = make_line(text)
                layouts.append(layout)
                cache_keys.append(_line_ink_cache_key(line, 0.0))
                _line_ink_bounds(line)
        self.assertEqual(len(_LINE_INK_BOUNDS_CACHE), 2)
        self.assertNotIn(cache_keys[0], _LINE_INK_BOUNDS_CACHE)

        _LINE_INK_BOUNDS_CACHE.clear()
        with patch(
            'ballontranslator.ui.text_engine.vertical_layout.'
            '_uncached_line_ink_bounds',
            wraps=_uncached_line_ink_bounds,
        ) as measure:
            item = self._make_item('（木', False, font_size=5.0)
        line_count = item.document().firstBlock().layout().lineCount()
        self.assertEqual(measure.call_count, line_count)
        self.assertEqual(len(_LINE_INK_BOUNDS_CACHE), line_count)

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

    def test_rotated_item_uses_logical_pivot_when_overflow_appears(self):
        item = self._make_overflow_item(
            standard=True,
            text='一般abcg',
            bounds=(180, 700),
            font_size=96,
        )
        scene = QGraphicsScene()
        scene.setItemIndexMethod(
            QGraphicsScene.ItemIndexMethod.BspTreeIndex
        )
        scene.setSceneRect(QRectF(-200, -200, 600, 1100))
        scene.addItem(item)
        item.setRotation(12.0)
        logical = QRectF(item.logical_unpadded_rect())

        def cjk_geometry() -> tuple[QPointF, QPointF, QPointF]:
            block = item.document().firstBlock()
            line = block.layout().lineForTextPosition(1)
            placement = item.layout.vertical_line_placement(
                block, line.lineNumber()
            )
            glyph = glyph_geometry(
                placement[0],
                placement[0].textStart(),
                placement[0].textLength(),
                placement[1],
                placement[2],
                0.0,
            ).bounds
            return (
                QPointF(line.position()),
                item.mapToScene(line.position()),
                item.mapToScene(glyph.center()),
            )

        logical_anchor = item.mapToScene(logical.center())
        cjk_before = cjk_geometry()
        self.assertEqual(item.transformOriginPoint(), logical.center())

        item.setStandardVerticalRomanAlignment(False)
        self.app.processEvents()

        self.assertEqual(item.logical_unpadded_rect(), logical)
        self.assertEqual(item.transformOriginPoint(), logical.center())
        self.assertEqual(item.mapToScene(logical.center()), logical_anchor)
        self.assertEqual(cjk_geometry(), cjk_before)
        g_ink = self._transformed_line_ink(item)[-1]
        outside = QPointF(
            (g_ink.left() + logical.left()) / 2,
            g_ink.center().y(),
        )
        scene_point = item.mapToScene(outside)
        self.assertTrue(
            item.geometry_controller.source_paint_rect().contains(g_ink)
        )
        self.assertTrue(item.boundingRect().contains(g_ink))
        self.assertTrue(item.shape().boundingRect().contains(g_ink))
        self.assertIn(item, scene.items(scene_point))

        item.setStandardVerticalRomanAlignment(True)
        self.app.processEvents()
        self.assertEqual(item.transformOriginPoint(), logical.center())
        self.assertEqual(item.mapToScene(logical.center()), logical_anchor)
        self.assertEqual(cjk_geometry(), cjk_before)

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
        self.assertTrue(panel.romanAlignmentChecker.isChecked())
        panel.romanAlignmentChecker.click()
        self.assertFalse(panel.romanAlignmentChecker.isChecked())
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

    def test_global_compact_punctuation_uses_half_cells_without_document_edits(self):
        original = C.pcfg.compact_vertical_punctuation_spacing
        try:
            for standard in (True, False):
                with self.subTest(standard=standard):
                    C.pcfg.compact_vertical_punctuation_spacing = False
                    item = self._make_item('木，。（』木', standard)
                    document = item.document()
                    legacy_final_top = item.layout.y_offset_lst[0][5][0]
                    revision = document.revision()
                    undo_steps = document.availableUndoSteps()
                    html = item.toHtml()

                    C.pcfg.compact_vertical_punctuation_spacing = True
                    item.refreshVerticalLayout()

                    offsets = item.layout.y_offset_lst[0]
                    regular_height = offsets[0][1] - offsets[0][0]
                    for position in range(1, 5):
                        top, bottom = offsets[position]
                        self.assertAlmostEqual(
                            bottom - top,
                            regular_height / 2,
                        )
                        ink, cell = self._ink_and_cell(item, position)
                        self.assertGreaterEqual(ink.top(), cell.top() - 1.0)
                        self.assertLessEqual(ink.bottom(), cell.bottom() + 1.0)
                        self.assertIn(
                            item.layout.hitTest(
                                cell.center(),
                                Qt.HitTestAccuracy.FuzzyHit,
                            ),
                            (position, position + 1),
                        )

                    self.assertLess(offsets[5][0], legacy_final_top)
                    self.assertEqual(document.revision(), revision)
                    self.assertEqual(document.availableUndoSteps(), undo_steps)
                    self.assertEqual(item.toHtml(), html)

                    for char in PUNSET_COMPACT:
                        punctuation = self._make_item(char, standard)
                        ink, cell = self._ink_and_cell(punctuation, 0)
                        self.assertGreaterEqual(ink.top(), cell.top() - 2.0)
                        self.assertLessEqual(ink.bottom(), cell.bottom() + 2.0)
        finally:
            C.pcfg.compact_vertical_punctuation_spacing = original

    def test_compact_punctuation_keeps_normal_character_spacing(self):
        original = C.pcfg.compact_vertical_punctuation_spacing
        C.pcfg.compact_vertical_punctuation_spacing = True
        try:
            for spacing in (1.1, 1.5, 2.0):
                with self.subTest(spacing=spacing):
                    item = self._make_item('木。木', True, spacing)
                    centers = [
                        self._ink_and_cell(item, position)[0].center().y()
                        for position in range(3)
                    ]
                    self.assertAlmostEqual(
                        centers[1],
                        (centers[0] + centers[2]) / 2,
                        delta=1.0,
                    )
        finally:
            C.pcfg.compact_vertical_punctuation_spacing = original

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
            self.assertTrue(
                standard.layout.needs_vertical_rotation(char), char
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
