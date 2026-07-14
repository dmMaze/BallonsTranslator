import os
import math
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QFontDatabase, QImage, QPainter
from qtpy.QtWidgets import QApplication, QGraphicsScene

from ballontranslator.utils import shared as C

C.FLAG_QT6 = QT_VERSION.startswith('6')
C.USE_PYSIDE6 = API_NAME == 'PySide6'

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui import textitem as textitem_module
from ballontranslator.ui.scene_textlayout import _grapheme_count
from ballontranslator.ui.textitem import TextBlkItem
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


_APP = QApplication.instance() or QApplication([])


def _font_family():
    arial = r'C:\Windows\Fonts\arial.ttf'
    if os.path.exists(arial):
        font_id = QFontDatabase.addApplicationFont(arial)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[0]
    try:
        families = QFontDatabase.families()
    except TypeError:  # PyQt5 exposes this as an instance method.
        families = QFontDatabase().families()
    return families[0] if families else None


_FONT_FAMILY = _font_family()


def _make_vertical_item(stroke_width=0.2):
    font_format = FontFormat(
        font_family=_FONT_FAMILY or 'Sans Serif',
        font_size=36,
        vertical=True,
        frgb=[0, 200, 0],
        srgb=[230, 0, 0],
        stroke_width=stroke_width,
    )
    block = TextBlock(
        xyxy=[30, 30, 310, 330],
        _bounding_rect=[30, 30, 280, 300],
        translation='',
        fontformat=font_format,
    )
    return TextBlkItem(block)


def _alpha(pixmap):
    return pixmap2ndarray(pixmap, keep_alpha=True)[..., 3]


def _stroke_pixmap(item, render_scale=2.0):
    pixmap = item._new_effect_pixmap(render_scale)
    painter = QPainter(pixmap)
    try:
        painter.translate(-item.boundingRect().topLeft())
        item.paint_stroke(painter, render_scale)
    finally:
        painter.end()
    return pixmap


def _layout_pixmap(item, render_scale=2.0):
    pixmap = item._new_effect_pixmap(render_scale)
    painter = QPainter(pixmap)
    try:
        painter.translate(-item.boundingRect().topLeft())
        item._paint_live_layout(painter, item._effect_paint_context())
    finally:
        painter.end()
    return pixmap


def _render_item(item, render_scale=2.0):
    source = QRectF(0, 0, 360, 380)
    image = QImage(
        math.ceil(source.width() * render_scale),
        math.ceil(source.height() * render_scale),
        QImage.Format.Format_ARGB32,
    )
    image.fill(Qt.GlobalColor.transparent)
    scene = QGraphicsScene()
    scene.setSceneRect(source)
    scene.addItem(item)
    painter = QPainter(image)
    try:
        scene.render(painter, QRectF(image.rect()), source)
    finally:
        painter.end()
        scene.removeItem(item)
    return pixmap2ndarray(image, keep_alpha=True)


@unittest.skipUnless(_FONT_FAMILY, 'No usable font is available for raster tests')
class VerticalTextStrokeTests(unittest.TestCase):

    def test_mixed_rich_fragments_use_their_own_selection_and_stroke_width(self):
        item = _make_vertical_item(0.2)
        item.setHtml(
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:20pt">A</span>'
            f'<span style="font-family:{_FONT_FAMILY};font-size:50pt">B</span></p>'
        )
        item.repaint_background()
        document = item.document()
        snapshot = (document.toHtml(), document.revision(), id(item.layout))
        expected = []
        context = item._stroke_paint_context()
        for selection in context.selections:
            expected.append(
                (
                    selection.cursor.selectionStart(),
                    selection.cursor.selectionEnd(),
                    selection.format.textOutline().widthF(),
                )
            )
        self.assertGreaterEqual(len(expected), 2)
        widths = [record[2] for record in expected]
        self.assertGreater(max(widths), min(widths) * 2)

        calls = []
        kernels = []
        original = item.layout.draw_glyph_selection_mask
        original_kernel = textitem_module.cv2.getStructuringElement

        def tracking_mask(painter, fragment_context):
            self.assertTrue(fragment_context.selections)
            for selection in fragment_context.selections:
                calls.append(
                    (
                        selection.cursor.selectionStart(),
                        selection.cursor.selectionEnd(),
                        selection.format.textOutline().widthF(),
                    )
                )
            return original(painter, fragment_context)

        def tracking_kernel(shape, size):
            kernels.append(size)
            return original_kernel(shape, size)

        item.layout.draw_glyph_selection_mask = tracking_mask
        textitem_module.cv2.getStructuringElement = tracking_kernel
        try:
            stroke = _stroke_pixmap(item)
        finally:
            item.layout.draw_glyph_selection_mask = original
            textitem_module.cv2.getStructuringElement = original_kernel

        self.assertEqual(sorted(calls), sorted(expected))
        expected_kernels = sorted(
            {
                (2 * math.ceil(width / 2 * 2.0) + 1,) * 2
                for width in widths
            }
        )
        self.assertEqual(sorted(kernels), expected_kernels)
        self.assertGreater(int((_alpha(stroke) > 0).sum()), 100)
        self.assertEqual(
            (document.toHtml(), document.revision(), id(item.layout)),
            snapshot,
        )

    def test_vertical_stroke_mask_excludes_underline_decoration(self):
        plain = _make_vertical_item(0.22)
        underlined = _make_vertical_item(0.22)
        plain.setHtml(
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:38pt">AB</span></p>'
        )
        underlined.setHtml(
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:38pt;'
            'text-decoration:underline">AB</span></p>'
        )
        plain_layout = _alpha(_layout_pixmap(plain))
        underlined_layout_before = _alpha(_layout_pixmap(underlined))
        self.assertFalse(np.array_equal(plain_layout, underlined_layout_before))

        plain_alpha = _alpha(_stroke_pixmap(plain))
        underline_alpha = _alpha(_stroke_pixmap(underlined))
        underlined_layout_after = _alpha(_layout_pixmap(underlined))
        self.assertEqual(plain_alpha.shape, underline_alpha.shape)
        np.testing.assert_array_equal(underline_alpha, plain_alpha)
        np.testing.assert_array_equal(
            underlined_layout_after, underlined_layout_before
        )

    def test_normal_vertical_fill_restores_rich_fragment_interiors(self):
        item = _make_vertical_item(0.22)
        item.setHtml(
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:30pt;'
            'color:#00c800">A</span>'
            f'<span style="font-family:{_FONT_FAMILY};font-size:48pt;'
            'color:#0044dd">B</span></p>'
        )
        pixels = _render_item(item)
        red_stroke = (
            (pixels[..., 0] > 150)
            & (pixels[..., 1] < 90)
            & (pixels[..., 2] < 90)
            & (pixels[..., 3] > 128)
        )
        green_fill = (
            (pixels[..., 1] > 130)
            & (pixels[..., 0] < 90)
            & (pixels[..., 2] < 90)
            & (pixels[..., 3] > 128)
        )
        blue_fill = (
            (pixels[..., 2] > 130)
            & (pixels[..., 0] < 90)
            & (pixels[..., 1] < 130)
            & (pixels[..., 3] > 128)
        )
        self.assertGreater(int(red_stroke.sum()), 100)
        self.assertGreater(int(green_fill.sum()), 50)
        self.assertGreater(int(blue_fill.sum()), 50)

    def test_utf16_lines_for_emoji_variation_selector_and_combining_mark(self):
        item = _make_vertical_item(0.18)
        item.setPlainText('A\U0001f600\ufe0fB\u0301')
        item.repaint_background()
        block = item.document().firstBlock()
        layout = block.layout()
        starts = []
        lengths = []
        positions = []
        for line_number in range(layout.lineCount()):
            line = layout.lineAt(line_number)
            starts.append(line.textStart())
            lengths.append(line.textLength())
            positions.append(line.position())

        self.assertEqual(starts, [0, 1, 4])
        self.assertEqual(lengths, [1, 3, 2])
        self.assertTrue(all(position != QPointF() for position in positions))
        self.assertEqual(len(item.layout.line_spaces_lst[0]), 3)
        self.assertEqual(len(item.layout._draw_offset[0]), 3)

        document = item.document()
        snapshot = (document.toHtml(), document.revision(), id(item.layout))
        stroke = _stroke_pixmap(item)
        self.assertGreater(int((_alpha(stroke) > 0).sum()), 100)
        self.assertEqual(
            (document.toHtml(), document.revision(), id(item.layout)),
            snapshot,
        )

    def test_vertical_hit_test_never_returns_inside_a_grapheme(self):
        item = _make_vertical_item(0.18)
        item.setPlainText('A\U0001f600\ufe0fB\u0301')
        block = item.document().firstBlock()
        layout = block.layout()

        for line_number in range(layout.lineCount()):
            line = layout.lineAt(line_number)
            start = line.textStart()
            end = start + line.textLength()
            top, bottom = item.layout.y_offset_lst[0][line_number]
            column_left = item.layout.x_offset_lst[1]
            column_right = item.layout.x_offset_lst[0]
            x_points = (
                column_left + 0.1,
                (column_left + column_right) / 2,
                column_right - 0.1,
            )
            y_points = (top + 0.1, (top + bottom) / 2, bottom - 0.1)
            for x in x_points:
                for y in y_points:
                    with self.subTest(
                        line=line_number,
                        point=(x, y),
                        utf16_range=(start, end),
                    ):
                        hit = item.layout.hitTest(
                            QPointF(x, y), Qt.HitTestAccuracy.FuzzyHit
                        )
                        self.assertIn(hit, (start, end))

    def test_qt_grapheme_boundaries_cover_keycap_enclosing_and_zwj(self):
        for text in (
            '1\ufe0f\u20e3',
            'A\u20dd',
            '\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466',
        ):
            with self.subTest(text=text):
                self.assertEqual(_grapheme_count(text), 1)

        item = _make_vertical_item(0.18)
        item.setPlainText('1\ufe0f\u20e32')
        block = item.document().firstBlock()
        layout = block.layout()
        self.assertEqual(
            [layout.lineAt(index).textStart() for index in range(layout.lineCount())],
            [0, 3],
        )
        self.assertEqual(
            [layout.lineAt(index).textLength() for index in range(layout.lineCount())],
            [3, 1],
        )
        self.assertEqual(len(item.layout.y_offset_lst[0]), 2)


if __name__ == '__main__':
    unittest.main()
