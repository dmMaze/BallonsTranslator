import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from shapely.geometry import LinearRing
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QRectF, Qt, QTextBoundaryFinder
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QTextCursor, QAbstractTextDocumentLayout, QTransform
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.utils.bubble_shape import (
    BubbleShape, _cover_text_region, _text_mask_path, bubble_path, load_bubble_shape, partition_bubble, trace_bubble,
)
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import FontFormat, VerticalAlignment, LOGGER as FONT_LOGGER
from ballontranslator.utils.proj_imgtrans import TextBlkEncoder
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.ui.text_engine.bubble_layout import (
    BubbleProfile, FlowUnit, _flow_break_penalty, balanced_flow_breaks, fit_text_in_shape, measure_flow_units,
)
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
        self.assertLessEqual(len(shape.nodes), 6)
        x, y, width, height = shape.frame
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 200)
        self.assertLessEqual(y + height, 160)
        path = bubble_path(shape.nodes, QRectF(*shape.frame))
        self.assertTrue(path.contains(QPointF(30, 30)))
        self.assertFalse(path.contains(QPointF(115, 80)))

    def test_tracing_covers_text_cells_after_scaling_rotation_and_save(self) -> None:
        mask = np.zeros((160, 200), np.uint8)
        cv2.ellipse(mask, (100, 80), (80, 60), 0, 0, 360, 1, -1)
        # Text extends below the bubble, including isolated pixels on page edges.
        for box in ((35, 20, 165, 155), (0, 0, 165, 160)):
            text_mask = np.zeros_like(mask)
            x1, y1, x2, y2 = box
            text_mask[y1:y2, 95:105] = 1
            text_mask[y1, x1] = text_mask[y2 - 1, x2 - 1] = 1
            required = QTransform().scale(10, 7.5).translate(.5, .5).map(_text_mask_path(text_mask))
            for angle in (0, 27, -73.5):
                with self.subTest(box=box, angle=angle):
                    shape = trace_bubble(mask, (2000, 1200), angle, text_mask)
                    restored = TextBlock(**json.loads(json.dumps(TextBlock(
                        angle=angle, bubble_shape=shape, bubble_layout=True,
                    ), cls=TextBlkEncoder)))
                    shape = restored.bubble_shape
                    self.assertLessEqual(len(shape.nodes), 8)
                    path = bubble_path(shape.nodes, QRectF(*shape.frame))
                    center = QRectF(*shape.frame).center()
                    transform = QTransform().translate(center.x(), center.y()).rotate(restored.angle)
                    transform.translate(-center.x(), -center.y())
                    self.assertTrue(required.subtracted(transform.map(path)).isEmpty())

    def test_text_mask_coverage_includes_thin_strokes_and_disconnected_components(self) -> None:
        mask = np.zeros((120, 160), np.uint8)
        cv2.fillPoly(mask, [np.array([[10, 10], [150, 10], [150, 40], [80, 40],
                                     [80, 80], [150, 80], [150, 110], [10, 110]])], 1)
        text_mask = np.zeros_like(mask)
        text_mask[25, 25:136] = text_mask[95, 25:136] = 1
        text_mask[60, 25:136] = 1
        required = _text_mask_path(text_mask).translated(.5, .5)
        original = trace_bubble(mask, (160, 120))
        original_path = bubble_path(original.nodes, QRectF(*original.frame))
        self.assertTrue(all(original_path.contains(QPointF(x, y))
                            for x in (25, 135) for y in (25, 95)))
        self.assertFalse(required.subtracted(original_path).isEmpty())
        shape = trace_bubble(mask, (160, 120), text_mask=text_mask)
        self.assertTrue(required.subtracted(bubble_path(shape.nodes, QRectF(*shape.frame))).isEmpty())
        self.assertLessEqual(len(shape.nodes), len(original.nodes) + 4)

    def test_tracing_rejects_mismatched_text_masks(self) -> None:
        mask = np.ones((30, 30), np.uint8)
        for shape in ((30,), (30, 29), (30, 30, 1)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                trace_bubble(mask, (30, 30), text_mask=np.zeros(shape, np.uint8))

    def test_text_mask_path_covers_full_pixels_without_filling_empty_centers(self) -> None:
        for bits in range(16):
            mask = np.array([(bits >> i) & 1 for i in range(4)], np.uint8).reshape(2, 2)
            path = _text_mask_path(mask)
            for y in range(2):
                for x in range(2):
                    self.assertEqual(path.contains(QPointF(x, y)), bool(mask[y, x]))
                    if mask[y, x]:
                        cell = QPainterPath()
                        cell.addRect(QRectF(x - .5, y - .5, 1, 1))
                        self.assertTrue(cell.subtracted(path).isEmpty())
        ring = np.ones((5, 5), np.uint8)
        ring[1:4, 1:4] = 0
        self.assertFalse(_text_mask_path(ring).contains(QPointF(2, 2)))

    def test_mask_coverage_does_not_require_empty_bounding_box_corners(self) -> None:
        mask = np.zeros((160, 200), np.uint8)
        cv2.ellipse(mask, (100, 80), (80, 60), 0, 0, 360, 1, -1)
        text_mask = np.zeros_like(mask)
        text_mask[25:136, 97:104] = 1
        text_mask[77:84, 30:171] = 1
        original = trace_bubble(mask, (200, 160))
        self.assertEqual(trace_bubble(mask, (200, 160), text_mask=text_mask), original)
        self.assertEqual(trace_bubble(mask, (200, 160), text_mask=np.zeros_like(mask)), original)
        required = _text_mask_path(text_mask).translated(.5, .5)
        path = bubble_path(original.nodes, QRectF(*original.frame))
        self.assertTrue(required.subtracted(path).isEmpty())
        self.assertFalse(path.contains(required.boundingRect()))

    def test_coverage_preserves_smooth_joins(self) -> None:
        mask = np.zeros((160, 200), np.uint8)
        cv2.ellipse(mask, (100, 80), (80, 60), 0, 0, 360, 1, -1)
        original = trace_bubble(mask, (200, 160))
        text_mask = np.zeros_like(mask)
        text_mask[20:155, 35:165] = 1
        adjusted = trace_bubble(mask, (200, 160), text_mask=text_mask)
        self.assertEqual(len(adjusted.nodes), len(original.nodes))
        self.assertEqual(len(adjusted.nodes), 4)
        nodes = np.array(adjusted.nodes).reshape(-1, 3, 2)
        incoming, outgoing = nodes[:, 1] - nodes[:, 0], nodes[:, 2] - nodes[:, 0]
        self.assertTrue(np.all(np.linalg.norm(incoming, axis=1) > .01))
        self.assertTrue(np.all(np.linalg.norm(outgoing, axis=1) > .01))
        np.testing.assert_allclose(incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0],
                                   0, atol=1e-5)
        self.assertTrue(np.all(np.sum(incoming * outgoing, axis=1) < 0))

    def test_coverage_keeps_polygon_order_and_adds_at_most_four_anchors(self) -> None:
        notch = np.array([[10, 10], [150, 10], [150, 40], [80, 40],
                          [80, 80], [150, 80], [150, 110], [10, 110]])
        triangle = np.array([[20, 10], [160, 90], [10, 110]])
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        radii = np.where(np.arange(64) % 2, 25, 70)
        star = np.column_stack((80 + radii * np.cos(angles), 60 + radii * np.sin(angles)))
        required = QPainterPath()
        required.addRect(QRectF(25, 25, 110, 70))
        for anchors in (notch, triangle, star):
            for reverse in (False, True):
                with self.subTest(count=len(anchors), reverse=reverse):
                    original = np.tile(anchors[::-1] if reverse else anchors, (1, 3))
                    saved = original.copy()
                    nodes = _cover_text_region(original, required)
                    np.testing.assert_array_equal(original, saved)
                    self.assertGreaterEqual(len(nodes), len(original))
                    self.assertLessEqual(len(nodes), min(64, len(original) + 4))
                    self.assertTrue(required.subtracted(bubble_path(nodes)).isEmpty())

    def test_tracing_and_coverage_preserve_a_simple_outline_at_deep_notches(self) -> None:
        notch = np.zeros((103, 159), np.uint8)
        notch[2:-2, 2:-2] = 1
        cv2.fillPoly(notch, [np.array([[53, 34], [159, 51], [53, 68]])], 0)
        cutouts = np.zeros((40, 53), np.uint8)
        cutouts[2:-2, 2:-2] = 1
        for center in ((15, 13), (19, 17), (32, 23), (12, 26)):
            cv2.circle(cutouts, center, 10, 0, -1)
        for mask in (notch, cutouts):
            page_size = mask.shape[::-1]
            original = trace_bubble(mask, page_size)
            for sparse in (False, True):
                with self.subTest(grid=mask.shape, sparse=sparse):
                    text_mask = mask.copy()
                    if sparse:
                        text_mask[np.arange(len(mask)) % 8 > 1] = 0
                    adjusted = trace_bubble(mask, page_size, text_mask=text_mask)
                    for shape in (original, adjusted):
                        path = bubble_path(shape.nodes, QRectF(*shape.frame))
                        outline = [path.pointAtPercent(t) for t in np.linspace(0, 1, 513)]
                        self.assertTrue(LinearRing([(p.x(), p.y()) for p in outline]).is_simple)
                    required = _text_mask_path(text_mask).translated(.5, .5)
                    self.assertTrue(required.subtracted(path).isEmpty())
                    self.assertLessEqual(len(adjusted.nodes), len(original.nodes) + 4)

    def test_concave_mask_coverage_on_a_detection_worker(self) -> None:
        # Isolate native crashes: the main thread's larger stack hides this regression.
        program = textwrap.dedent('''
            import cv2
            import numpy as np
            from qtpy.QtCore import QCoreApplication, QThread, QRectF
            from ballontranslator.utils.bubble_shape import trace_bubble, bubble_path, _text_mask_path

            class Worker(QThread):
                completed = False

                def run(self) -> None:
                    mask = np.zeros((120, 180), np.uint8)
                    cv2.fillPoly(mask, [np.array([[10, 10], [150, 10], [150, 40], [80, 40],
                                                 [80, 80], [150, 80], [150, 110], [10, 110]])], 1)
                    text = np.zeros_like(mask)
                    text[25:96, 25:136] = 1
                    original = trace_bubble(mask, (180, 120))
                    shape = trace_bubble(mask, (180, 120), text_mask=text)
                    assert shape is not None and len(shape.nodes) <= len(original.nodes) + 4
                    required = _text_mask_path(text).translated(.5, .5)
                    assert required.subtracted(bubble_path(shape.nodes, QRectF(*shape.frame))).isEmpty()
                    self.completed = True

            app = QCoreApplication([])
            worker = Worker()
            worker.start()
            assert worker.wait(10000) and worker.completed
        ''')
        result = subprocess.run(
            [sys.executable, '-X', 'faulthandler', '-c', program],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tails_and_spikes_keep_compact_body_primitives(self) -> None:
        for kind in ('rectangle', 'ellipse', 'spikes'):
            with self.subTest(kind=kind):
                body = np.zeros((200, 240), np.uint8)
                if kind == 'rectangle':
                    body[25:130, 30:200] = 1
                else:
                    cv2.ellipse(body, (120, 100), (85, 65), 0, 0, 360, 1, -1)
                mask = body.copy()
                if kind == 'spikes':
                    angles = np.linspace(0, 2 * np.pi, 48, endpoint=False)
                    radii = np.where(np.arange(48) % 2, 1., 1.12)
                    points = np.column_stack((120 + 85 * radii * np.cos(angles),
                                              100 + 65 * radii * np.sin(angles)))
                    cv2.fillPoly(mask, [points.astype(np.int32)], 1)
                else:
                    cv2.fillPoly(mask, [np.array([[170, 110], [225, 190], [183, 110]])], 1)
                shape = trace_bubble(mask, (240, 200))
                self.assertEqual(len(shape.nodes), 4)
                path = bubble_path(shape.nodes, QRectF(*shape.frame))
                self.assertTrue(path.contains(QPointF(120, 100)))
                self.assertFalse(path.contains(QPointF(222, 185)))
                polygon = np.rint([(p.x(), p.y()) for p in path.toFillPolygon()]).astype(np.int32)
                raster = np.zeros_like(mask)
                cv2.fillPoly(raster, [polygon], 1)
                self.assertGreater(np.count_nonzero(raster & body) / np.count_nonzero(body), .9)
                self.assertLess(np.count_nonzero(raster & (mask == 0)) / np.count_nonzero(raster), .06)

    def test_compound_curves_keep_lobes_and_merge_unneeded_anchors(self) -> None:
        for scale in (1, 2):
            with self.subTest(scale=scale):
                mask = np.zeros((200 * scale, 260 * scale), np.uint8)
                cv2.ellipse(mask, (85 * scale, 95 * scale), (60 * scale, 70 * scale), 0, 0, 360, 1, -1)
                cv2.ellipse(mask, (180 * scale, 95 * scale), (60 * scale, 80 * scale), 0, 0, 360, 1, -1)
                shape = trace_bubble(mask, (260, 200))
                self.assertLessEqual(len(shape.nodes), 8)
                self.assertTrue(any(node[:2] != node[2:4] for node in shape.nodes))
                path = bubble_path(shape.nodes, QRectF(*shape.frame))
                self.assertTrue(path.contains(QPointF(85, 95)))
                self.assertTrue(path.contains(QPointF(180, 95)))
                self.assertFalse(path.contains(QPointF(130, 22)))

    def test_narrow_neck_does_not_discard_a_second_usable_lobe(self) -> None:
        for axes in ((40, 45), (20, 20)):
            with self.subTest(axes=axes):
                mask = np.zeros((120, 240), np.uint8)
                cv2.ellipse(mask, (55, 60), (40, 45), 0, 0, 360, 1, -1)
                cv2.ellipse(mask, (180, 60), axes, 0, 0, 360, 1, -1)
                mask[59:62, 90:165] = 1
                shape = trace_bubble(mask, (240, 120))
                self.assertLessEqual(len(shape.nodes), 12)
                self.assertTrue(any(node[:2] != node[2:4] for node in shape.nodes))
                path = bubble_path(shape.nodes, QRectF(*shape.frame))
                self.assertTrue(path.contains(QPointF(55, 60)))
                self.assertTrue(path.contains(QPointF(180, 60)))
                self.assertFalse(path.contains(QPointF(115, 30)))

    def test_profiles_preserve_taper_and_concavity_and_are_reused(self) -> None:
        profile = BubbleProfile(self.shape())
        low, high = profile.band(.45, .55)
        top_low, top_high = profile.band(.1, .2)
        self.assertGreater(high - low, top_high - top_low)
        mask = np.zeros((120, 160), np.uint8)
        cv2.fillPoly(mask, [np.array([[10, 10], [150, 10], [150, 40], [80, 40],
                                     [80, 80], [150, 80], [150, 110], [10, 110]])], 1)
        shape = trace_bubble(mask, (160, 120))
        self.assertLessEqual(len(shape.nodes), 12)
        path = bubble_path(shape.nodes, QRectF(*shape.frame))
        self.assertTrue(path.contains(QPointF(40, 60)))
        self.assertFalse(path.contains(QPointF(140, 60)))
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

    def test_english_breaks_score_phrases_across_spaces_and_emoji(self) -> None:
        item = self.item('Stop! Now, go and find the notebook. 😀 Go home.')
        item.setFontSize(14)
        text = item.toPlainText().encode('utf-16-le')
        scores = {text[:unit.end * 2].decode('utf-16-le').rstrip(): unit.penalty
                  for unit in measure_flow_units(item.layout, False)}
        self.assertLess(scores['Stop!'], scores['Stop! Now,'])
        self.assertLess(scores['Stop! Now,'], scores['Stop! Now, go'])
        self.assertLess(scores['Stop! Now, go'], scores['Stop! Now, go and find'])
        self.assertLess(scores['Stop! Now, go and find'], scores['Stop! Now, go and find the'])
        self.assertEqual(scores['Stop! Now, go and find the notebook.'], scores['Stop!'])
        self.assertEqual(scores['Stop! Now, go and find the notebook. 😀 Go home.'], scores['Stop!'])

    def test_english_word_measurements_keep_hanging_spaces_when_reflowing(self) -> None:
        item = self.item('one two three four seven eight nine ten')
        item.setFontSize(14)
        item.setLetterSpacing(1.15)
        measurements = []
        for width in (180, 700):
            item.setRect([0, 0, width, 500])
            units = measure_flow_units(item.layout, False)
            measurements.append([(unit.end, unit.advance, unit.trailing) for unit in units])
        np.testing.assert_allclose(measurements[0], measurements[1], atol=.05)
        self.assertTrue(all(trailing > 0 for _, _, trailing in measurements[0][:-1]))

    def test_phrase_scoring_keeps_whole_words_after_long_mixed_script_tokens(self) -> None:
        prefix = 'x' * 10000 + '文'
        for word, expected in (('', 100.), ('within', 100.), ('thereof', 100.),
                               (' FROM ', 300.), ('with', 300.), ('the', 300.)):
            with self.subTest(word=word):
                self.assertEqual(_flow_break_penalty(prefix + word, 'book'), expected)

    def test_horizontal_free_text_fit_does_not_gain_capacity_by_splitting_words(self) -> None:
        for family in ('Arial', 'Times New Roman', 'Comic Sans MS'):
            with self.subTest(family=family):
                item = self.item('Put the notebook beside the refrigerator.', shape=False)
                item.setFontFamily(family)
                item.setRect([0, 0, 140, 400])
                original = item.toPlainText()
                result = fit_text_in_shape(item, auto_size=True)
                self.assertFalse(result.overflow)
                self.assertEqual(item.toPlainText(), original)
                item.layout.reLayout()
                block = item.document().firstBlock()
                finder = QTextBoundaryFinder(QTextBoundaryFinder.BoundaryType.Line, block.text())
                for index in range(block.layout().lineCount()):
                    line = block.layout().lineAt(index)
                    finder.setPosition(line.textStart() + line.textLength())
                    self.assertTrue(finder.isAtBoundary())

    def test_horizontal_word_that_cannot_fit_reports_overflow_at_the_floor(self) -> None:
        text = 'supercalifragilisticexpialidocious'
        item = self.item(text, shape=False)
        item.setRect([0, 0, 20, 400])
        with patch('ballontranslator.ui.text_engine.bubble_layout.LOGGER.warning') as warning:
            result = fit_text_in_shape(item, auto_size=True)
        self.assertTrue(result.overflow)
        self.assertAlmostEqual(result.font_size, 9., delta=.1)
        self.assertEqual(item.toPlainText(), text)
        warning.assert_called_once()

    def test_word_insertion_keeps_occupied_flow_bands_and_editing_history(self) -> None:
        text = 'Hey, what do you think of Sakamoto, Mei?'

        def positions(item: TextBlkItem) -> tuple[QPointF, ...]:
            layout = item.document().firstBlock().layout()
            return tuple(layout.lineAt(i).position() for i in range(layout.lineCount()))

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self.item(text, vertical=vertical)
                if not vertical:
                    shape = BubbleShape.rectangle((10, 20, 248, 342))
                    item.set_bubble_shape(shape, True, QRectF(*shape.frame))
                self.assertFalse(fit_text_in_shape(item, auto_size=True).overflow)
                frame = item.absBoundingRect(qrect=True)
                size = item.layout.max_font_size(to_px=True)
                before = positions(item)
                cursor = QTextCursor(item.document())
                cursor.setPosition(text.index('Sakamoto') + 3)
                cursor.insertText('s')
                after = positions(item)
                self.assertTrue(item.layout.flow_overflow)
                self.assertEqual(item.layout.flow_line, len(item.layout.flow_slots))
                self.assertEqual(item.absBoundingRect(qrect=True), frame)
                self.assertEqual(item.layout.max_font_size(to_px=True), size)
                self.assertLess((after[0] - before[0]).manhattanLength(),
                                item.layout.flow_step)
                if vertical:
                    point = item.layout.source_cursor_rect(0).center()
                else:
                    line = item.document().firstBlock().layout().lineAt(0)
                    point = QPointF(item.layout._cursor_x(line, 0), line.y() + line.height() / 2)
                self.assertEqual(item.layout.hitTest(point, Qt.HitTestAccuracy.FuzzyHit), 0)
                item.undo()
                self.assertEqual(item.toPlainText(), text)
                self.assertEqual(positions(item), before)
                item.redo()
                self.assertEqual(item.toPlainText(), text.replace('Sakamoto', 'Saksamoto'))
                self.assertEqual(positions(item), after)
                item.blk.rich_text = item.toHtml()
                restored = TextBlock(**json.loads(json.dumps(item.blk, cls=TextBlkEncoder)))
                loaded = TextBlkItem(restored, 0)
                self.addCleanup(loaded.deleteLater)
                self.assertEqual(positions(loaded), after)
                self.assertTrue(loaded.layout.flow_overflow)
                self.assertFalse(fit_text_in_shape(loaded, auto_size=True).overflow)
                self.assertLess(loaded.layout.max_font_size(to_px=True), size)

    def test_vertical_word_overflow_keeps_columns_in_their_bubble_slots(self) -> None:
        text = 'Hey, what do you think of Sakamoto, Mei?'
        for alignment in (0, 2):
            with self.subTest(alignment=alignment):
                item = self.item(text, vertical=True)
                shape = BubbleShape.rectangle((10, 20, 248, 342))
                item.set_bubble_shape(shape, True, QRectF(*shape.frame))
                item.setAlignment(alignment)
                self.assertFalse(fit_text_in_shape(item, auto_size=True).overflow)
                columns = item.layout.flow_line
                first_x = item.layout.source_cursor_rect(0).x()
                cursor = QTextCursor(item.document())
                cursor.setPosition(text.index('Sakamoto') + 3)
                cursor.insertText('s')
                self.assertTrue(item.layout.flow_overflow)
                self.assertEqual(item.layout.flow_line, columns)
                self.assertEqual(item.layout.flow_line, len(item.layout.flow_slots))
                self.assertAlmostEqual(item.layout.source_cursor_rect(0).x(), first_x, delta=.05)

    def test_horizontal_bubble_fits_and_centers_actual_ink_across_fonts(self) -> None:
        for family in ('Arial', 'Times New Roman', 'Comic Sans MS'):
            with self.subTest(family=family):
                shape = BubbleShape.rectangle((0, 0, 260, 130))
                block = TextBlock(
                    _bounding_rect=list(shape.frame), translation='...',
                    bubble_shape=shape, bubble_layout=True,
                    fontformat=FontFormat(font_family=family, font_size=24, alignment=1),
                )
                item = TextBlkItem(block, 0)
                self.addCleanup(item.deleteLater)
                result = fit_text_in_shape(item, auto_size=True)
                self.assertFalse(result.overflow)
                self.assertGreater(result.font_size, shape.frame[3])
                line = item.document().firstBlock().layout().lineAt(0)
                ink = glyph_geometry(line, 0, line.textLength(), QPointF(), QTransform(), 0.).bounds
                frame = item.logical_unpadded_rect()
                self.assertTrue(frame.contains(ink))
                self.assertAlmostEqual(ink.center().y(), frame.center().y(), delta=2.)

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
