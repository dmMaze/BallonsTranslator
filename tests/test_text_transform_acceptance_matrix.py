import math
import os
import unittest
from types import SimpleNamespace

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QFontDatabase, QImage, QPainter, QPen, QPixmap, QPolygonF
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.utils import shared as C

C.FLAG_QT6 = QT_VERSION.startswith('6')
C.USE_PYSIDE6 = API_NAME == 'PySide6'

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_transform import rect_polygon
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


def _render_font_family():
    windows_arial = r'C:\Windows\Fonts\arial.ttf'
    if os.path.exists(windows_arial):
        font_id = QFontDatabase.addApplicationFont(windows_arial)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    families = QFontDatabase.families()
    return families[0] if families else None


_FONT_FAMILY = _render_font_family()


def _make_item(
    *,
    text='Transform',
    html=None,
    vertical=False,
    transform=(1.55, 0.72, 18.0),
    angle=13.0,
    stroke_width=0.0,
    shadow_radius=0.0,
    shadow_strength=0.0,
    shadow_offset=(0.0, 0.0),
    gradient=False,
):
    font_format = FontFormat(
        font_family=_FONT_FAMILY or 'Sans Serif',
        font_size=34,
        vertical=vertical,
        frgb=[25, 185, 70],
        srgb=[225, 30, 35],
        stroke_width=stroke_width,
        shadow_radius=shadow_radius,
        shadow_strength=shadow_strength,
        shadow_offset=list(shadow_offset),
        shadow_color=[25, 45, 205],
        gradient_enabled=gradient,
        gradient_start_color=[245, 170, 20],
        gradient_end_color=[20, 80, 245],
        gradient_angle=35,
        gradient_size=0.55,
        horizontal_scale=transform[0],
        vertical_scale=transform[1],
        slant_angle=transform[2],
    )
    block = TextBlock(
        xyxy=[90, 70, 390, 270],
        _bounding_rect=[90, 70, 300, 200],
        translation=text,
        angle=angle,
        fontformat=font_format,
    )
    item = TextBlkItem(block)
    if html is not None:
        item.setHtml(html)
        item.repaint_background()
    return item


def _image_array(image):
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    size = converted.width() * converted.height() * 4
    bits = converted.bits()
    if hasattr(bits, 'asstring'):
        raw = bits.asstring(size)
    else:
        raw = bits.tobytes()
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        converted.height(), converted.width(), 4
    ).copy()


def _render_item(item, *, scale=1.0, source_rect=None):
    scene = QGraphicsScene()
    scene.addItem(item)
    if source_rect is None:
        bounds = item.mapToScene(item.boundingRect()).boundingRect()
        source_rect = bounds.adjusted(-12, -12, 12, 12)
    scene.setSceneRect(source_rect)
    width = max(1, math.ceil(source_rect.width() * scale))
    height = max(1, math.ceil(source_rect.height() * scale))
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        scene.render(painter, QRectF(0, 0, width, height), source_rect)
    finally:
        painter.end()
        scene.removeItem(item)
    return _image_array(image)


def _alpha_bbox(alpha):
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _alpha_centroid(alpha):
    ys, xs = np.nonzero(alpha > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _bbox_aspect_ratio(bbox):
    left, top, right, bottom = bbox
    return (right - left + 1) / (bottom - top + 1)


def _polygon_envelope_mask(shape, polygon, source_rect):
    """Rasterize a scene polygon with a one-pixel AA coverage guard."""
    height, width = shape[:2]
    scale_x = width / source_rect.width()
    scale_y = height / source_rect.height()
    pixel_polygon = QPolygonF(
        [
            QPointF(
                (point.x() - source_rect.left()) * scale_x,
                (point.y() - source_rect.top()) * scale_y,
            )
            for point in polygon
        ]
    )
    mask_image = QImage(width, height, QImage.Format.Format_ARGB32)
    mask_image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(mask_image)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        guard_pen = QPen(Qt.GlobalColor.white)
        guard_pen.setWidthF(2.0)
        painter.setPen(guard_pen)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawPolygon(pixel_polygon)
    finally:
        painter.end()
    return _image_array(mask_image)[..., 3] > 0


def _assert_alpha_within_polygon(testcase, alpha, polygon, source_rect):
    envelope_mask = _polygon_envelope_mask(alpha.shape, polygon, source_rect)
    outside_alpha = np.logical_and(alpha > 0, ~envelope_mask)
    testcase.assertEqual(int(outside_alpha.sum()), 0)


def _assert_unclipped(testcase, pixels):
    alpha = pixels[..., 3]
    testcase.assertFalse(alpha[0].any())
    testcase.assertFalse(alpha[-1].any())
    testcase.assertFalse(alpha[:, 0].any())
    testcase.assertFalse(alpha[:, -1].any())


@unittest.skipUnless(_FONT_FAMILY, 'No usable font is available for raster tests')
class TextTransformAcceptanceMatrixTests(unittest.TestCase):

    def test_content_matrix_in_horizontal_and_vertical_layouts(self):
        rich = (
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:24pt;'
            'font-weight:700;color:#c62828">Rich</span>'
            f'<span style="font-family:{_FONT_FAMILY};font-size:38pt;'
            'font-style:italic;text-decoration:underline;color:#1565c0"> Text</span></p>'
        )
        formatted_empty = (
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:42pt;'
            'font-style:italic;color:#1565c0"><br></span></p>'
        )
        cases = (
            ('plain-latin', 'Plain Latin 123', None, True),
            ('cjk', '한국어 日本語 中文', None, True),
            ('mixed-script', 'ABC 한국語 123', None, True),
            ('emoji-variation-selectors', '☺️ ✈︎ ❤️', None, True),
            ('combining-marks', 'e\u0301 A\u030a n\u0303', None, True),
            ('multiline', 'first line\nsecond 행\nthird', None, True),
            ('empty', '', None, False),
            ('formatted-empty', '', formatted_empty, False),
            ('partial-rich-formatting', '', rich, True),
            ('vertical-punctuation', '「日本語」、。！？…', None, True),
        )

        for vertical in (False, True):
            for name, text, html, visible in cases:
                with self.subTest(vertical=vertical, content=name):
                    item = _make_item(text=text, html=html, vertical=vertical)
                    html_before = item.document().toHtml()
                    plain_before = item.document().toPlainText()
                    pixels = _render_item(item)
                    alpha_pixels = int((pixels[..., 3] > 0).sum())
                    if visible:
                        self.assertGreater(alpha_pixels, 20)
                    else:
                        self.assertEqual(alpha_pixels, 0)
                    _assert_unclipped(self, pixels)
                    self.assertEqual(item.document().toHtml(), html_before)
                    self.assertEqual(item.document().toPlainText(), plain_before)
                    self.assertEqual(item.fontformat.vertical, vertical)
                    self.assertEqual(
                        item.fontformat.text_transform, (1.55, 0.72, 18.0)
                    )

                    if name == 'partial-rich-formatting':
                        point_sizes = []
                        block = item.document().firstBlock()
                        while block.isValid():
                            iterator = block.begin()
                            while not iterator.atEnd():
                                point_sizes.append(
                                    iterator.fragment().charFormat().fontPointSize()
                                )
                                iterator += 1
                            block = block.next()
                        self.assertGreaterEqual(len(point_sizes), 2)
                        self.assertGreater(max(point_sizes), min(point_sizes))

    def test_effect_combinations_removal_and_extreme_transforms(self):
        effect_cases = (
            ('fill', 0.0, 0.0, 0.0, False),
            ('stroke', 0.18, 0.0, 0.0, False),
            ('shadow', 0.0, 0.2, 0.8, False),
            ('gradient', 0.0, 0.0, 0.0, True),
            ('stroke-gradient', 0.18, 0.0, 0.0, True),
            ('stroke-shadow', 0.18, 0.2, 0.8, False),
            ('all', 0.18, 0.2, 0.8, True),
        )
        transforms = ((0.1, 4.0, 45.0), (4.0, 0.1, -45.0))
        plain_area = {}

        for vertical in (False, True):
            for transform in transforms:
                for name, stroke, shadow_radius, shadow_strength, gradient in effect_cases:
                    with self.subTest(
                        vertical=vertical, transform=transform, effects=name
                    ):
                        item = _make_item(
                            text='Fx 한국語\nEDGE',
                            vertical=vertical,
                            transform=transform,
                            angle=-27,
                            stroke_width=stroke,
                            shadow_radius=shadow_radius,
                            shadow_strength=shadow_strength,
                            shadow_offset=(-0.25, 0.2),
                            gradient=gradient,
                        )
                        pixels = _render_item(item, scale=1.25)
                        _assert_unclipped(self, pixels)
                        area = int((pixels[..., 3] > 0).sum())
                        self.assertGreater(area, 30)
                        key = vertical, transform
                        if name == 'fill':
                            plain_area[key] = area
                            self.assertEqual(item.padding(), 0)
                            self.assertIsNone(item.background_pixmap)
                        elif stroke or shadow_strength:
                            self.assertGreater(item.padding(), 0)
                            self.assertIsNotNone(item.background_pixmap)
                            self.assertGreater(area, plain_area[key] * 0.9)

                        if gradient:
                            ink = pixels[pixels[..., 3] > 128, :3]
                            self.assertGreater(len(ink), 20)
                            self.assertGreater(int(np.ptp(ink[:, 0])), 10)
                            self.assertGreater(int(np.ptp(ink[:, 2])), 10)

        item = _make_item(
            text='REMOVE',
            transform=(2.1, 0.55, 24),
            angle=19,
            stroke_width=0.2,
            shadow_radius=0.22,
            shadow_strength=0.85,
            shadow_offset=(-0.3, 0.25),
            gradient=True,
        )
        fixed_source = QRectF(0, 0, 520, 360)
        with_effects = _render_item(item, source_rect=fixed_source)
        self.assertGreater(item.padding(), 0)
        item.setStrokeWidth(0)
        item.setBGAttribute('shadow_strength', 0)
        item.setGradientEnabled(False)
        removed = _render_item(item, source_rect=fixed_source)
        plain = _render_item(
            _make_item(
                text='REMOVE', transform=(2.1, 0.55, 24), angle=19
            ),
            source_rect=fixed_source,
        )
        self.assertEqual(item.padding(), 0)
        self.assertIsNone(item.background_pixmap)
        self.assertGreater(
            int((with_effects[..., 3] > 0).sum()),
            int((removed[..., 3] > 0).sum()),
        )
        self.assertEqual(
            _alpha_bbox(removed[..., 3]), _alpha_bbox(plain[..., 3])
        )
        removed_area = int((removed[..., 3] > 0).sum())
        plain_area_value = int((plain[..., 3] > 0).sum())
        self.assertLessEqual(abs(removed_area - plain_area_value), 4)

    def test_editing_and_nonediting_scene_renders_have_matching_ink(self):
        html = (
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:25pt;'
            'font-weight:700;color:#1b9e3e">Edit</span>'
            f'<span style="font-family:{_FONT_FAMILY};font-size:37pt;'
            'font-style:italic;color:#1565c0"> 한국語</span></p>'
        )
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = _make_item(
                    text='',
                    html=html,
                    vertical=vertical,
                    transform=(1.8, 0.65, -22),
                    angle=17,
                    stroke_width=0.16,
                    shadow_radius=0.18,
                    shadow_strength=0.75,
                    shadow_offset=(0.2, -0.18),
                    gradient=True,
                )
                source = QRectF(0, 0, 560, 400)
                nonediting_corners = QPolygonF(
                    [
                        item.mapToScene(point)
                        for point in rect_polygon(item.boundingRect())
                    ]
                )
                nonediting = _render_item(item, source_rect=source)
                item.startEdit()
                item.clearFocus()
                editing_corners = QPolygonF(
                    [
                        item.mapToScene(point)
                        for point in rect_polygon(item.boundingRect())
                    ]
                )
                editing = _render_item(item, source_rect=source)
                self.assertTrue(item.is_editting())
                self.assertEqual(len(nonediting_corners), 4)
                self.assertEqual(len(editing_corners), 4)
                for nonediting_corner, editing_corner in zip(
                    nonediting_corners, editing_corners
                ):
                    self.assertAlmostEqual(
                        nonediting_corner.x(), editing_corner.x(), places=6
                    )
                    self.assertAlmostEqual(
                        nonediting_corner.y(), editing_corner.y(), places=6
                    )
                self.assertEqual(
                    _alpha_bbox(nonediting[..., 3]), _alpha_bbox(editing[..., 3])
                )
                nonediting_ink = nonediting[..., 3] > 0
                editing_ink = editing[..., 3] > 0
                nonediting_bbox = _alpha_bbox(nonediting[..., 3])
                editing_bbox = _alpha_bbox(editing[..., 3])
                self.assertIsNotNone(nonediting_bbox)
                self.assertIsNotNone(editing_bbox)

                nonediting_centroid = _alpha_centroid(nonediting[..., 3])
                editing_centroid = _alpha_centroid(editing[..., 3])
                self.assertIsNotNone(nonediting_centroid)
                self.assertIsNotNone(editing_centroid)
                self.assertLessEqual(
                    abs(nonediting_centroid[0] - editing_centroid[0]),
                    0.5,
                )
                self.assertLessEqual(
                    abs(nonediting_centroid[1] - editing_centroid[1]),
                    0.5,
                )

                self.assertAlmostEqual(
                    _bbox_aspect_ratio(nonediting_bbox),
                    _bbox_aspect_ratio(editing_bbox),
                    places=6,
                )

                _assert_alpha_within_polygon(
                    self, nonediting[..., 3], nonediting_corners, source
                )
                _assert_alpha_within_polygon(
                    self, editing[..., 3], editing_corners, source
                )
                disagreement = np.logical_xor(nonediting_ink, editing_ink).sum()
                union = np.logical_or(nonediting_ink, editing_ink).sum()
                self.assertLessEqual(disagreement / max(1, union), 0.01)
                item.endEdit(keep_focus=False)

    def test_actual_canvas_export_matches_nonediting_and_ends_editing(self):
        width, height = 520, 360
        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(
            img_valid=True,
            inpainted_valid=True,
            inpainted_array=np.zeros((height, width, 4), dtype=np.uint8),
        )
        transparent = QPixmap(width, height)
        transparent.fill(Qt.GlobalColor.transparent)
        canvas.inpaintLayer.setPixmap(transparent)
        canvas.textLayer.setPixmap(transparent)
        canvas.baseLayer.setRect(QRectF(0, 0, width, height))
        canvas.setSceneRect(QRectF(0, 0, width, height))

        item = _make_item(
            text='Canvas 한국語',
            transform=(1.75, 0.7, 21),
            angle=-14,
            stroke_width=0.17,
            shadow_radius=0.18,
            shadow_strength=0.8,
            shadow_offset=(-0.2, 0.2),
            gradient=True,
        )
        item.setParentItem(canvas.textLayer)
        canvas._set_scene_scale(0.75)
        canvas.textLayer.setOpacity(0.35)
        canvas.textLayer.hide()
        nonediting_corners = QPolygonF(
            [
                item.mapToItem(canvas.baseLayer, point)
                for point in rect_polygon(item.boundingRect())
            ]
        )
        nonediting = _image_array(canvas.render_result_img())
        self.assertAlmostEqual(canvas.scale_factor, 0.75)
        self.assertAlmostEqual(canvas.textLayer.opacity(), 0.35)
        self.assertFalse(canvas.textLayer.isVisible())

        canvas.editor_index = 1
        canvas.txtblkShapeControl.blk_item = item
        item.startEdit()
        item.setSelected(True)
        editing_corners = QPolygonF(
            [
                item.mapToItem(canvas.baseLayer, point)
                for point in rect_polygon(item.boundingRect())
            ]
        )
        editing_export = _image_array(canvas.render_result_img())

        self.assertFalse(item.is_editting())
        self.assertFalse(item.isSelected())
        self.assertAlmostEqual(canvas.scale_factor, 0.75)
        self.assertAlmostEqual(canvas.textLayer.opacity(), 0.35)
        self.assertFalse(canvas.textLayer.isVisible())
        self.assertEqual(len(nonediting_corners), 4)
        self.assertEqual(len(editing_corners), 4)
        for nonediting_corner, editing_corner in zip(
            nonediting_corners, editing_corners
        ):
            self.assertAlmostEqual(
                nonediting_corner.x(), editing_corner.x(), places=6
            )
            self.assertAlmostEqual(
                nonediting_corner.y(), editing_corner.y(), places=6
            )
        for label, pixels in (
            ('nonediting', nonediting),
            ('editing-export', editing_export),
        ):
            red_stroke = (
                (pixels[..., 0] > 150)
                & (pixels[..., 1] < 100)
                & (pixels[..., 2] < 100)
                & (pixels[..., 3] > 0)
            )
            blue_shadow = (
                (pixels[..., 2] > 130)
                & (pixels[..., 0] < 100)
                & (pixels[..., 1] < 120)
                & (pixels[..., 3] > 0)
            )
            with self.subTest(export=label):
                self.assertGreater(int(red_stroke.sum()), 50)
                self.assertGreater(int(blue_shadow.sum()), 50)
        # Qt 6 bindings can choose a slightly different grayscale antialiasing
        # phase after focus/interaction flags change.  Export parity is about
        # the same final ink/effects geometry, not byte-identical edge pixels.
        nonediting_ink = nonediting[..., 3] > 0
        editing_ink = editing_export[..., 3] > 0
        self.assertGreater(int(nonediting_ink.sum()), 50)
        nonediting_bbox = _alpha_bbox(nonediting[..., 3])
        editing_bbox = _alpha_bbox(editing_export[..., 3])
        self.assertIsNotNone(nonediting_bbox)
        self.assertIsNotNone(editing_bbox)
        for actual, expected in zip(editing_bbox, nonediting_bbox):
            self.assertLessEqual(abs(actual - expected), 1)
        nonediting_centroid = _alpha_centroid(nonediting[..., 3])
        editing_centroid = _alpha_centroid(editing_export[..., 3])
        self.assertIsNotNone(nonediting_centroid)
        self.assertIsNotNone(editing_centroid)
        self.assertLessEqual(
            abs(editing_centroid[0] - nonediting_centroid[0]), 0.75
        )
        self.assertLessEqual(
            abs(editing_centroid[1] - nonediting_centroid[1]), 0.75
        )
        self.assertLessEqual(
            abs(
                _bbox_aspect_ratio(editing_bbox)
                - _bbox_aspect_ratio(nonediting_bbox)
            ),
            0.02,
        )
        export_source = QRectF(0, 0, width, height)
        _assert_alpha_within_polygon(
            self, nonediting[..., 3], nonediting_corners, export_source
        )
        _assert_alpha_within_polygon(
            self, editing_export[..., 3], editing_corners, export_source
        )
        disagreement = np.logical_xor(nonediting_ink, editing_ink).sum()
        union = np.logical_or(nonediting_ink, editing_ink).sum()
        self.assertLess(disagreement / max(1, union), 0.03)
        pixel_delta = np.abs(
            editing_export.astype(np.int16) - nonediting.astype(np.int16)
        )
        union_rgb_delta = pixel_delta[..., :3][
            np.logical_or(nonediting_ink, editing_ink)
        ]
        # PySide6 rasterizes a newly unfocused QTextItem at a different
        # grayscale-AA phase.  Compare the affected union only, normalized to
        # the channel range: geometry is constrained above and the remaining
        # mean color delta must stay below eight percent.
        self.assertLess(float(union_rgb_delta.mean() / 255.0), 0.08)


if __name__ == '__main__':
    unittest.main()
