import json
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor, QAbstractTextDocumentLayout
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.utils.bubble_shape import BubbleShape, load_bubble_shape, partition_bubble, trace_bubble
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import FontFormat, VerticalAlignment, LOGGER as FONT_LOGGER
from ballontranslator.utils.proj_imgtrans import TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.ui.text_engine.bubble_layout import BubbleProfile, FlowUnit, balanced_flow_breaks, fit_text_in_shape
from ballontranslator.ui.text_engine.editing.commands import AutoLayoutCommand
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.editing.widgets import TransTextEdit
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.glyph import glyph_geometry
from ballontranslator.ui.text_engine.annotations import apply_ruby, apply_text_combine_upright, ruby_containers


class BubbleLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def tearDown(self) -> None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    @staticmethod
    def shape(angle: float = 0) -> BubbleShape:
        mask = np.zeros((160, 200), np.uint8)
        cv2.ellipse(mask, (100, 80), (90, 70), 0, 0, 360, 1, -1)
        return trace_bubble(mask, (600, 480), angle)

    def item(self, text: str, *, vertical: bool = False, shape: bool = True) -> TextBlkItem:
        block = TextBlock(
            [10, 20, 310, 220], _bounding_rect=[10, 20, 300, 200],
            translation=text, det_model='koharu',
            bubble_shape=self.shape() if shape else None,
            bubble_layout=shape,
            fontformat=FontFormat(font_family='Arial', font_size=24, vertical=vertical, alignment=1),
            text_layout_version=1,
        )
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        return item

    @staticmethod
    def column_ink(item: TextBlkItem) -> dict[float, QRectF]:
        columns = {}
        block = item.document().firstBlock()
        while block.isValid():
            for number in range(block.layout().lineCount()):
                source = block.layout().lineAt(number)
                line, offset, orientation = item.layout.vertical_line_placement(block, number)
                ink = glyph_geometry(line, line.textStart(), line.textLength(), offset, orientation, 0.).bounds
                if not ink.isEmpty():
                    columns[source.x()] = columns.get(source.x(), QRectF()).united(ink)
            block = block.next()
        return columns

    def test_primitives_use_four_editable_nodes_and_round_trip(self) -> None:
        for ellipse in (False, True):
            mask = np.zeros((160, 200), np.uint8)
            if ellipse:
                cv2.ellipse(mask, (100, 80), (90, 70), 0, 0, 360, 1, -1)
            else:
                mask[10:150, 10:190] = 1
            shape = trace_bubble(mask, (2000, 1600))
            self.assertEqual(len(shape.nodes), 4)
            self.assertEqual(any(node[:2] != node[2:4] for node in shape.nodes), ellipse)
            block = TextBlock(bubble_shape=shape, bubble_layout=True)
            restored = TextBlock(**json.loads(json.dumps(block, cls=TextBlkEncoder)))
            self.assertEqual(restored.bubble_shape, shape)
            self.assertTrue(restored.bubble_layout)
            block.adjust_pos(15, -10)
            self.assertEqual(block.bubble_shape.nodes, shape.nodes)
            self.assertEqual(block.bubble_shape.frame[0], shape.frame[0] + 15)
            self.assertLess(len(json.dumps(shape.to_serializable_dict())), 600)

    def test_anchor_insertion_preserves_cubics_including_the_closing_segment(self) -> None:
        original = self.shape()
        def evaluate(shape, index, t):
            a = shape.nodes[index]
            b = shape.nodes[(index + 1) % len(shape.nodes)]
            points = np.array((a[:2], a[4:], b[2:4], b[:2]))
            return np.array(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3)) @ points
        for index in (0, len(original.nodes) - 1):
            divided = original.split_segment(index, .37)
            self.assertEqual(divided.frame, original.frame)
            self.assertEqual(len(divided.nodes), len(original.nodes) + 1)
            for t in np.linspace(0, 1, 41):
                actual = (evaluate(divided, index, t / .37) if t <= .37
                          else evaluate(divided, index + 1, (t - .37) / .63))
                np.testing.assert_allclose(actual, evaluate(original, index, t), atol=1e-12)

    def test_invalid_optional_shape_preserves_translation(self) -> None:
        for payload in (None, {'nodes': []}, {'frame': [0, 0, -1, 3], 'nodes': self.shape().nodes},
                        {'frame': [10**400, 0, 30, 30], 'nodes': self.shape().nodes},
                        {'frame': [0, 0, 30, 30], 'nodes': [[float('nan')] * 6] * 4}, 'bad'):
            with self.subTest(payload=payload), patch('ballontranslator.utils.bubble_shape.LOGGER.warning') as warning:
                block = TextBlock(translation='keep me', bubble_shape=payload, bubble_layout=True)
                self.assertEqual(block.translation, 'keep me')
                self.assertIsNone(block.bubble_shape)
                self.assertFalse(block.bubble_layout)
                warning.assert_called()
        self.assertIsNone(TextBlock().bubble_shape)
        value = self.shape().to_serializable_dict() | {'future_field': True}
        with patch('ballontranslator.utils.bubble_shape.LOGGER.warning') as warning:
            self.assertEqual(load_bubble_shape(value), self.shape())
            warning.assert_called_once()

    def test_clipped_bubbles_do_not_extrapolate_beyond_the_page(self) -> None:
        mask = np.zeros((160, 200), np.uint8)
        cv2.ellipse(mask, (30, 15), (90, 70), 0, 0, 360, 1, -1)
        shape = trace_bubble(mask, (200, 160))
        self.assertGreater(len(shape.nodes), 4)
        x, y, width, height = shape.frame
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 200)
        self.assertLessEqual(y + height, 160)

    def test_profiles_preserve_taper_and_concavity_and_are_reused(self) -> None:
        profile = BubbleProfile(self.shape())
        low, high = profile.band(.45, .55)
        top_low, top_high = profile.band(.1, .2)
        self.assertGreater(high - low, top_high - top_low)
        mask = np.zeros((120, 160), np.uint8)
        cv2.fillPoly(mask, [np.array([[10, 10], [150, 10], [150, 40], [80, 40],
                                     [80, 80], [150, 80], [150, 110], [10, 110]])], 1)
        shape = trace_bubble(mask, (160, 120))
        self.assertGreater(len(shape.nodes), 4)
        self.assertLessEqual(len(shape.nodes), 64)
        item = self.item('Editable bubble text')
        fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
        saved = item.layout.flow_profile
        item.setRect([0, 0, 500, 400])
        item.setFontSize(18)
        self.assertIs(item.layout.flow_profile, saved)

    def test_joined_flows_split_at_the_neck_before_source_bisectors(self) -> None:
        mask = np.zeros((100, 110), np.uint8)
        cv2.fillPoly(mask, [np.array([(0, 20), (40, 20), (50, 30), (60, 20),
                                     (100, 20), (100, 80), (60, 80), (50, 70),
                                     (40, 80), (0, 80)])], 1)
        anchors = np.array([[15., 50.], [65., 50.]])
        for scale in ((1, 1), (2, 3)):
            cells = partition_bubble(mask, (110 * scale[0], 100 * scale[1]), anchors * scale)
            self.assertTrue(cells[0][50, 45])  # Source bisectors would cut at x=40.
            self.assertTrue(cells[1][50, 55])
            self.assertFalse(np.any(cells[0] & cells[1]))
            self.assertFalse(np.any((cells[0] | cells[1]) & (mask == 0)))

    def test_coincident_bubble_anchors_keep_separate_nonempty_flows(self) -> None:
        mask = np.zeros((80, 100), np.uint8)
        cv2.ellipse(mask, (50, 40), (45, 35), 0, 0, 360, 1, -1)
        cells = partition_bubble(mask, (100, 80), np.array([[50., 40.], [50., 40.]]))
        self.assertGreater(np.count_nonzero(cells[0]), 1000)
        self.assertGreater(np.count_nonzero(cells[1]), 1000)
        self.assertFalse(np.any(cells[0] & cells[1]))
        np.testing.assert_array_equal(cells[0] | cells[1], mask != 0)

    def test_both_modes_fit_preserve_paragraphs_and_repeat(self) -> None:
        for vertical, text in ((False, 'Hello 😀 cafe\u0301! This is a longer paragraph.\n\nKeep this paragraph too.'),
                               (True, 'これはテストです。新しい形の吹き出しに文章を配置します。')):
            with self.subTest(vertical=vertical):
                item = self.item(text, vertical=vertical)
                result = fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
                self.assertFalse(result.overflow)
                self.assertGreater(result.font_size, 9)
                self.assertEqual(item.toPlainText(), text)
                self.assertEqual(item.layout.flow_line, len(item.layout.flow_slots))
                rect = item.absBoundingRect(qrect=True)
                again = fit_text_in_shape(item, auto_size=True)
                self.assertAlmostEqual(result.font_size, again.font_size, delta=.2)
                self.assertEqual(item.absBoundingRect(qrect=True), rect)
                self.assertEqual(item.toPlainText(), text)

    def test_off_center_flow_uses_the_available_body_in_both_modes(self) -> None:
        for vertical in (False, True):
            points = [(0.6, 0), (1, 0), (1, 1), (0.6, 1)]
            if vertical:
                points = [(y, x) for x, y in points]
            shape = BubbleShape((0, 0, 300, 300), tuple(tuple(point) * 3 for point in points))
            item = self.item('这里有文字', vertical=vertical)
            item.blk.bubble_shape = shape
            result = fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
            self.assertFalse(result.overflow)
            self.assertGreater(result.font_size, 30)
            slots = item.layout.flow_slots
            centers = [slot.center().y() if vertical else slot.center().x() for slot in slots]
            self.assertTrue(all(abs(center - 240) < 2 for center in centers))

    def test_vertical_flow_centers_short_column_ink_and_keeps_hit_testing(self) -> None:
        item = self.item('長い縦の文章。\n短い。', vertical=True)
        item.setVerticalAlignment(1)
        result = fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
        self.assertFalse(result.overflow)
        columns = self.column_ink(item)
        self.assertGreaterEqual(len(columns), 2)
        centers = [bounds.center().y() for bounds in columns.values()]
        self.assertLess(max(centers) - min(centers), .1)
        caret = item.layout.source_cursor_rect(0)
        self.assertEqual(item.layout.hitTest(QPointF(caret.center().x(), caret.top()), Qt.HitTestAccuracy.FuzzyHit), 0)

    def test_vertical_alignment_defaults_and_round_trip(self) -> None:
        for horizontal in (0, 1, 2):
            old = TextBlock(fontformat={'vertical': True, 'alignment': horizontal})
            self.assertEqual(old.fontformat.vertical_alignment, VerticalAlignment.Top)
        for value in (None, 'bottom', [], {}, True, 1.0, -1, 3):
            with self.subTest(value=value), self.assertLogs(FONT_LOGGER, 'WARNING'):
                fmt = FontFormat(vertical_alignment=value, font_size=37)
                self.assertEqual(fmt.vertical_alignment, VerticalAlignment.Top)
                self.assertEqual(fmt.font_size, 37)
            with self.assertRaises(ValueError):
                fmt.vertical_alignment = value
        for alignment in VerticalAlignment:
            fmt = FontFormat(vertical_alignment=alignment)
            loaded = FontFormat(**json.loads(json.dumps(fmt.to_serializable_dict())))
            self.assertEqual(loaded.vertical_alignment, alignment)

    def test_vertical_top_center_bottom_keeps_ink_and_carets_together(self) -> None:
        for shape in (False, True):
            item = self.item('天地玄黄\n宇宙', vertical=True, shape=shape)
            self.assertFalse(fit_text_in_shape(item, auto_size=False, use_bubble_frame=shape).overflow)
            original = None
            for alignment in (0, 1, 2, 0):
                with self.subTest(shape=shape, alignment=alignment):
                    item.setVerticalAlignment(alignment)
                    columns = self.column_ink(item)
                    self.assertEqual(len(columns), 2)
                    padding = item.layout.effectPadding()
                    top = padding + (max(slot.top() for slot in item.layout.flow_slots) if shape else 0)
                    bottom = padding + (min(slot.bottom() for slot in item.layout.flow_slots) if shape else item.layout.available_height)
                    if alignment == 1:
                        self.assertTrue(all(abs(ink.center().y() - (top + bottom) / 2) < .05 for ink in columns.values()))
                    elif alignment == 2:
                        self.assertTrue(all(abs(ink.bottom() - bottom) < .05 for ink in columns.values()))
                    elif shape:
                        self.assertTrue(all(abs(ink.top() - top) < .05 for ink in columns.values()))
                    if alignment == 0:
                        if original is not None:
                            self.assertEqual(columns, original)
                        original = columns
                    for position in (0, 5):
                        caret = item.layout.source_cursor_rect(position)
                        self.assertEqual(item.layout.hitTest(QPointF(caret.center().x(), caret.top()), Qt.HitTestAccuracy.FuzzyHit), position)
                    caret = item.layout.source_cursor_rect(0)
                    for _ in range(3):
                        item.layout.reLayout()
                        self.assertEqual(item.layout.source_cursor_rect(0), caret)

    def test_overflow_columns_keep_vertical_alignment_after_resize(self) -> None:
        block = TextBlock(
            _bounding_rect=[0, 0, 315, 477], translation='刚才那位是……雨宫雫小姐吧？',
            bubble_shape=self.shape(), bubble_layout=True, text_layout_version=1,
            fontformat=FontFormat(font_family='Arial', font_size=96, vertical=True, alignment=1),
        )
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        item.setRect([0, 0, 483, 345])
        original_text = item.toPlainText()
        original_size = item.layout.max_font_size(to_px=True)
        for alignment in (0, 1, 2):
            with self.subTest(alignment=alignment):
                item.setVerticalAlignment(alignment)
                self.assertTrue(item.layout.flow_overflow)
                columns = self.column_ink(item)
                self.assertGreaterEqual(len(columns), 3)
                positions = [(ink.top(), ink.center().y(), ink.bottom())[alignment]
                             for ink in columns.values()]
                self.assertLess(max(positions) - min(positions), .05)
                # The final overflow column must remain editable after moving.
                position = len(original_text) - 1
                caret = item.layout.source_cursor_rect(position)
                self.assertEqual(item.layout.hitTest(QPointF(caret.center().x(), caret.top()),
                                                    Qt.HitTestAccuracy.FuzzyHit), position)
                self.assertEqual(item.toPlainText(), original_text)
                self.assertEqual(item.layout.max_font_size(to_px=True), original_size)
                item.layout.reLayout()
                self.assertEqual(self.column_ink(item), columns)

    def test_overflow_group_alignment_and_paint_bounds_follow_all_columns(self) -> None:
        block = TextBlock(
            _bounding_rect=[0, 0, 315, 477], translation='刚才那位是……雨宫雫小姐吧？',
            bubble_shape=self.shape(), bubble_layout=True, text_layout_version=1,
            fontformat=FontFormat(font_family='Arial', font_size=96, vertical=True,
                                  alignment=1, vertical_alignment=1),
        )
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        item.setRect([40, 80, 184, 246])
        frame = list(item.blk._bounding_rect)
        scene = QGraphicsScene()
        scene.addItem(item)
        try:
            for alignment in (0, 1, 2):
                with self.subTest(alignment=alignment):
                    item.setAlignment(alignment)
                    self.assertTrue(item.layout.flow_overflow)
                    ink = QRectF()
                    for column in self.column_ink(item).values():
                        ink = ink.united(column)
                    logical = item.logical_unpadded_rect()
                    position = (ink.left(), ink.center().x(), ink.right())[alignment]
                    target = (logical.left(), logical.center().x(), logical.right())[alignment]
                    self.assertAlmostEqual(position, target, delta=.05)
                    self.assertTrue(item.boundingRect().contains(ink))
                    self.assertTrue(item.shape().boundingRect().contains(ink))
                    self.assertEqual(item.blk._bounding_rect, frame)
                    item.layout.reLayout()
                    settled = QRectF()
                    for column in self.column_ink(item).values():
                        settled = settled.united(column)
                    self.assertEqual(settled, ink)
        finally:
            scene.removeItem(item)

    def test_overflow_scene_render_keeps_every_glyph(self) -> None:
        block = TextBlock(
            _bounding_rect=[40, 80, 184, 246], translation='刚才那位是……雨宫雫小姐吧？',
            bubble_shape=self.shape(), bubble_layout=True, text_layout_version=1,
            fontformat=FontFormat(font_family='Arial', font_size=96, vertical=True,
                                  alignment=1, vertical_alignment=1),
        )
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        scene = QGraphicsScene()
        scene.addItem(item)
        try:
            for width in (394, 184, 394):
                item.setRect([40, 80, width, 246])
                self.assertTrue(item.layout.flow_overflow)
                ink = QRectF()
                for column in self.column_ink(item).values():
                    ink = ink.united(column)
                area = ink.united(item.logical_unpadded_rect()).adjusted(-10, -10, 10, 10).toAlignedRect()
                images = []
                for direct in (False, True):
                    image = QImage(area.size(), QImage.Format.Format_ARGB32)
                    image.fill(Qt.GlobalColor.white)
                    painter = QPainter(image)
                    if direct:
                        painter.translate(-area.left(), -area.top())
                        item.layout.draw(painter, QAbstractTextDocumentLayout.PaintContext())
                    else:
                        scene.render(painter, QRectF(0, 0, area.width(), area.height()),
                                     QRectF(area).translated(item.pos()))
                    painter.end()
                    images.append(image)
                self.assertEqual(images[0], images[1])
        finally:
            scene.removeItem(item)

    def test_vertical_fit_accepts_native_font_ink_wider_than_ordinary_bounds(self) -> None:
        # CoreText can report wider tight ink than ordinary bounds at these sizes.
        with patch.object(shared, 'LDPI', self.app.primaryScreen().logicalDotsPerInch()):
            for family in ('Microsoft YaHei UI', 'Arial'):
                for size in (9., 24., 30., 40.):
                    with self.subTest(family=family, size=size):
                        item = self.item('我今天一大早就遇到一个怪人……', vertical=True)
                        item.setFontFamily(family)
                        item.setFontSize(size * 72 / shared.LDPI)
                        result = fit_text_in_shape(item, auto_size=False, use_bubble_frame=True)
                        self.assertFalse(result.overflow)
                        self.assertAlmostEqual(result.font_size, size, delta=.1)

    def test_vertical_inline_alignment_tracks_height_resize(self) -> None:
        for alignment in (0, 1, 2):
            with self.subTest(alignment=alignment):
                item = self.item('天地玄黄\n宇宙', vertical=True, shape=False)
                item.setVerticalAlignment(alignment)
                before = item.layout.source_cursor_rect(0)
                rect = QRectF(*item.blk._bounding_rect)
                rect.setHeight(rect.height() + 120)
                item.setRect(rect)
                after = item.layout.source_cursor_rect(0)
                self.assertAlmostEqual(after.top() - before.top(), alignment * 60, delta=.05)
                rect.setWidth(rect.width() + 80)
                item.setRect(rect)
                self.assertAlmostEqual(item.layout.source_cursor_rect(0).top(), after.top(), delta=.05)
                self.assertEqual(item.toPlainText(), '天地玄黄\n宇宙')

    def test_free_text_fit_tolerates_subpixel_font_bearings(self) -> None:
        with patch.object(shared, 'LDPI', self.app.primaryScreen().logicalDotsPerInch()):
            item = self.item('如果母亲非常勤劳，女儿最终可能什么都做不了，这确实是事实。',
                             vertical=True, shape=False)
            item.setFontFamily('Microsoft YaHei UI')
            item.setFontSize(14 * 72 / shared.LDPI)
            result = fit_text_in_shape(item, auto_size=False, bounding_rect=QRectF(0, 0, 114, 138))
            self.assertFalse(result.overflow)
            self.assertAlmostEqual(result.font_size, 14, delta=.1)

    def test_balanced_breaks_fit_without_shrinking_or_changing_translation(self) -> None:
        mask = np.zeros((200, 200), np.uint8)
        cv2.ellipse(mask, (100, 100), (90, 90), 0, 0, 360, 1, -1)
        block = TextBlock(
            _bounding_rect=[0, 0, 220, 300], translation='我今天一大早就遇到一个怪人……',
            bubble_shape=trace_bubble(mask, (200, 200)), bubble_layout=True,
            fontformat=FontFormat(font_family='Arial', font_size=30, alignment=1, letter_spacing=1.15),
        )
        item = TextBlkItem(block, 0)
        self.addCleanup(item.deleteLater)
        result = fit_text_in_shape(item, auto_size=False)
        self.assertFalse(result.overflow)
        self.assertAlmostEqual(result.font_size, 30, delta=.1)
        self.assertEqual(item.toPlainText(), block.translation)
        layout = item.document().firstBlock().layout()
        lengths = [layout.lineAt(i).textLength() for i in range(layout.lineCount())]
        self.assertLessEqual(max(lengths) - min(lengths), 1)
        result = fit_text_in_shape(item, auto_size=True)
        self.assertFalse(result.overflow)
        layout = item.document().firstBlock().layout()
        ends = {layout.lineAt(i).textStart() + layout.lineAt(i).textLength()
                for i in range(layout.lineCount())}
        self.assertTrue(ends.isdisjoint({13, 14}))  # Keep the final ellipsis run attached.

    def test_long_documents_keep_native_wrapping_when_balancing_is_bounded(self) -> None:
        text = '文' * 1100
        item = self.item(text)
        item.setRect([0, 0, 1700, 1700])
        item.setFontSize(9)
        result = fit_text_in_shape(item, auto_size=False)
        self.assertFalse(result.overflow)
        self.assertEqual(item.toPlainText(), text)

    def test_balanced_capacity_keeps_discardable_line_end_spacing(self) -> None:
        units = [FlowUnit(1, 15., 5., True, False, 0.), FlowUnit(2, 15., 5., True, True, 0.)]
        self.assertEqual(balanced_flow_breaks(units, [10., 10.]), [1, 2])
        self.assertIsNone(balanced_flow_breaks(units, [9., 9.]))
        self.assertIsNone(balanced_flow_breaks(units, [19.]))

    def test_inline_hard_breaks_and_emoji_keep_their_authored_rows(self) -> None:
        item = self.item('placeholder')
        item.load_rich_text_html('<p>A😀<br/><br/>B</p>')
        before = item.toPlainText()
        result = fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
        self.assertFalse(result.overflow)
        self.assertEqual(item.toPlainText(), before)
        self.assertEqual(item.document().blockCount(), 1)
        self.assertEqual(item.document().firstBlock().layout().lineCount(), 3)

    def test_manual_size_and_frame_survive_fit_and_overflow_is_reported(self) -> None:
        item = self.item('Keep the chosen size. ' * 12)
        item.setRect([30, 40, 100, 50])
        rect = item.absBoundingRect(qrect=True)
        size = item.layout.max_font_size(to_px=True)
        with patch('ballontranslator.ui.text_engine.bubble_layout.LOGGER.warning') as warning:
            result = fit_text_in_shape(item, auto_size=False)
        self.assertTrue(result.overflow)
        warning.assert_called()
        self.assertEqual(item.layout.max_font_size(to_px=True), size)
        self.assertEqual(item.absBoundingRect(qrect=True), rect)
        layout = item.document().firstBlock().layout()
        self.assertGreater(len({layout.lineAt(i).y() for i in range(layout.lineCount())}), 1)

    def test_ruby_and_combined_runs_survive_fitting(self) -> None:
        for vertical in (False, True):
            item = self.item('東京で12人に会いました。', vertical=vertical)
            cursor = QTextCursor(item.document())
            cursor.setPosition(0)
            cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
            apply_ruby(cursor, 'group', 'とうきょう')
            if vertical:
                cursor.setPosition(3)
                cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
                apply_text_combine_upright(cursor, True)
            for alignment in ((0, 1, 2) if vertical else (0,)):
                item.setVerticalAlignment(alignment)
                result = fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
                self.assertFalse(result.overflow)
                self.assertEqual(item.toPlainText(), '東京で12人に会いました。')
                self.assertEqual(len(ruby_containers(item.document())), 1)
                if vertical:
                    self.assertTrue(any(item.layout.text_combine_ranges))

    def test_free_text_has_resolution_dependent_size_and_keeps_text(self) -> None:
        item = self.item('Large free text', shape=False)
        item.setRect([0, 0, 800, 300])
        result = fit_text_in_shape(item, auto_size=True)
        self.assertFalse(result.overflow)
        self.assertGreater(result.font_size, 24)
        self.assertEqual(item.toPlainText(), 'Large free text')

    def test_vertical_fitting_excludes_trailing_tracking_from_ink(self) -> None:
        item = self.item('似乎有些东西起到了弯曲的缠绕机制的作用，而且还有倾斜、松弛和末端的情况。', vertical=True, shape=False)
        item.setLetterSpacing(1.15)
        result = fit_text_in_shape(item, auto_size=True, bounding_rect=QRectF(0, 0, 562, 173))
        self.assertFalse(result.overflow)
        self.assertGreater(result.font_size, 9)

    def test_vertical_shape_alignment_is_configurable(self) -> None:
        item = self.item('文', vertical=True)
        item.setFontSize(9)
        fit_text_in_shape(item, auto_size=False, use_bubble_frame=True)
        positions = []
        for alignment in (0, 1, 2):
            item.setAlignment(alignment)
            positions.append(item.document().firstBlock().layout().lineAt(0).x())
        self.assertLess(positions[0], positions[1])
        self.assertLess(positions[1], positions[2])

    def test_formatting_after_emoji_and_undo_round_trip(self) -> None:
        item = self.item('A😀B\n\nThis stays a paragraph.')
        cursor = QTextCursor(item.document())
        cursor.setPosition(3)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor('red'))
        cursor.setCharFormat(fmt)
        old_html, old_rect = item.toHtml(), item.absBoundingRect(qrect=True)
        old_flow = (item.blk.bubble_layout, item.fontformat.font_size)
        editor = TransTextEdit(0, None)
        self.addCleanup(editor.deleteLater)
        editor.setPlainText(item.toPlainText())
        def layout(target: TextBlkItem) -> bool:
            fit_text_in_shape(target, auto_size=True, use_bubble_frame=True)
            return True
        command = AutoLayoutCommand([item], [editor], layout)
        new_html = item.toHtml()
        cursor.setPosition(3)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        self.assertEqual(cursor.charFormat().foreground().color(), QColor('red'))
        command.redo()
        command.undo()
        self.assertEqual(item.blk.bubble_layout, old_flow[0])
        self.assertEqual(item.toHtml(), old_html)
        self.assertEqual(item.absBoundingRect(qrect=True), old_rect)
        command.redo()
        self.assertTrue(item.blk.bubble_layout)
        self.assertEqual(item.toHtml(), new_html)
        self.assertEqual(editor.toPlainText(), item.toPlainText())

    def test_legacy_format_restoration_uses_utf16_positions(self) -> None:
        item = self.item('A😀B')
        cursor = QTextCursor(item.document())
        cursor.setPosition(3)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        fmt = cursor.charFormat()
        fmt.setForeground(QColor('red'))
        cursor.setCharFormat(fmt)
        formats = item.get_char_fmts()
        item.setPlainText('A😀\nB')
        SceneTextManager.restore_charfmts(None, item, 'A😀B', 'A😀\nB', formats)
        cursor.setPosition(4)
        cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        self.assertEqual(cursor.charFormat().foreground().color(), QColor('red'))

    def test_project_reload_and_export_share_flow(self) -> None:
        item = self.item('A persisted bubble with enough text to wrap.')
        fit_text_in_shape(item, auto_size=True, use_bubble_frame=True)
        item.blk.rich_text = item.toHtml()
        restored = TextBlock(**json.loads(json.dumps(item.blk, cls=TextBlkEncoder)))
        loaded = TextBlkItem(restored, 0)
        self.addCleanup(loaded.deleteLater)
        self.assertEqual(loaded.toPlainText(), item.toPlainText())
        self.assertEqual(loaded.layout.flow_line, item.layout.flow_line)
        scene = QGraphicsScene()
        scene.addItem(loaded)
        image = QImage(800, 600, QImage.Format.Format_ARGB32)
        image.fill(QColor('white'))
        painter = QPainter(image)
        scene.render(painter, QRectF(0, 0, 800, 600), QRectF(0, 0, 800, 600))
        painter.end()
        self.assertTrue(any(image.pixelColor(x, y) != QColor('white')
                            for y in range(0, 500, 5) for x in range(0, 600, 5)))
        scene.removeItem(loaded)

    def test_rendered_ink_stays_inside_ellipse_in_both_modes(self) -> None:
        for vertical in (False, True):
            item = self.item('形の中に翻訳した文章を配置します。複数の行も確認します。', vertical=vertical)
            self.assertFalse(fit_text_in_shape(item, auto_size=True, use_bubble_frame=True).overflow)
            rect = item.absBoundingRect(qrect=True)
            width, height = int(rect.width()), int(rect.height())
            image = QImage(width, height, QImage.Format.Format_ARGB32)
            image.fill(0)
            scene = QGraphicsScene()
            scene.addItem(item)
            painter = QPainter(image)
            scene.render(painter, QRectF(0, 0, width, height), rect)
            painter.end()
            scene.removeItem(item)
            bits = image.constBits()
            if hasattr(bits, 'setsize'):
                bits.setsize(image.sizeInBytes())
            ink = np.frombuffer(bits, np.uint8).reshape(height, width, 4)[..., 3] > 40
            allowed = np.zeros((height, width), np.uint8)
            cv2.ellipse(allowed, (width // 2, height // 2), (width // 2, height // 2), 0, 0, 360, 1, -1)
            allowed = cv2.dilate(allowed, np.ones((3, 3), np.uint8))
            self.assertTrue(ink.any())
            self.assertFalse(np.any(ink & (allowed == 0)))


if __name__ == '__main__':
    unittest.main()
