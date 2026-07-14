import math
import os
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import API_NAME, QT_VERSION
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QFontDatabase,
    QImage,
    QInputMethodEvent,
    QLinearGradient,
    QPainter,
    QTextCursor,
)
from qtpy.QtWidgets import QApplication, QGraphicsItem, QGraphicsScene

from ballontranslator.utils import shared as C

C.FLAG_QT6 = QT_VERSION.startswith('6')
C.USE_PYSIDE6 = API_NAME == 'PySide6'

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.textitem import (
    GRADIENT_LAYOUT_FORMAT_PROPERTY,
    TextBlkItem,
)
from ballontranslator.utils.fontformat import FontFormat, pt2px
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
    vertical=False,
    transform=(1.0, 1.0, 0.0),
    angle=0.0,
    stroke_width=0.0,
    shadow_radius=0.0,
    shadow_strength=0.0,
    shadow_offset=(0.0, 0.0),
    gradient=False,
    text='TEST',
):
    font_format = FontFormat(
        font_family=_FONT_FAMILY or 'Sans Serif',
        font_size=40,
        vertical=vertical,
        frgb=[0, 220, 0],
        srgb=[230, 0, 0],
        stroke_width=stroke_width,
        shadow_radius=shadow_radius,
        shadow_strength=shadow_strength,
        shadow_offset=list(shadow_offset),
        shadow_color=[0, 0, 160],
        gradient_enabled=gradient,
        gradient_start_color=[255, 180, 0],
        gradient_end_color=[0, 80, 255],
        horizontal_scale=transform[0],
        vertical_scale=transform[1],
        slant_angle=transform[2],
    )
    block = TextBlock(
        xyxy=[80, 70, 300, 220],
        _bounding_rect=[80, 70, 220, 150],
        translation=text,
        angle=angle,
        fontformat=font_format,
    )
    return TextBlkItem(block)


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


def _render_item(item, source_rect, scale=1, background=None):
    scene = QGraphicsScene()
    scene.setSceneRect(source_rect)
    scene.addItem(item)
    width = max(1, math.ceil(source_rect.width() * scale))
    height = max(1, math.ceil(source_rect.height() * scale))
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent if background is None else background)
    painter = QPainter(image)
    try:
        scene.render(
            painter,
            QRectF(0, 0, width, height),
            source_rect,
        )
    finally:
        painter.end()
        scene.removeItem(item)
    return image


@unittest.skipUnless(_FONT_FAMILY, 'No usable font is available for raster tests')
class TextTransformRenderingTests(unittest.TestCase):

    def test_effect_passes_use_the_attached_document_and_layout_without_mutation(self):
        item = _make_item(
            stroke_width=0.18,
            shadow_radius=0.2,
            shadow_strength=0.8,
            shadow_offset=(0.2, -0.15),
            text='A😀B',
        )
        document = item.document()
        layout = item.layout
        cursor = item.textCursor()
        snapshot = (
            document.toHtml(),
            document.toPlainText(),
            document.revision(),
            document.availableUndoSteps(),
            document.isModified(),
            cursor.position(),
            cursor.anchor(),
            item.documentSize(),
        )
        line_positions = []
        block = document.firstBlock()
        while block.isValid():
            text_layout = block.layout()
            line_positions.append(
                [text_layout.lineAt(i).position() for i in range(text_layout.lineCount())]
            )
            block = block.next()

        contexts = []
        original_draw = layout.draw

        def tracking_draw(painter, context):
            contexts.append(context)
            return original_draw(painter, context)

        layout.draw = tracking_draw
        try:
            item.repaint_background(2.0)
        finally:
            layout.draw = original_draw

        self.assertIs(document.documentLayout(), layout)
        self.assertIs(item.layout, layout)
        self.assertGreaterEqual(len(contexts), 3)
        self.assertTrue(any(len(context.selections) == 0 for context in contexts))
        stroke_contexts = [context for context in contexts if context.selections]
        self.assertTrue(stroke_contexts)
        for context in contexts:
            self.assertEqual(context.cursorPosition, -1)
        for context in stroke_contexts:
            for selection in context.selections:
                self.assertIs(selection.cursor.document(), document)

        cursor = item.textCursor()
        self.assertEqual(
            (
                document.toHtml(),
                document.toPlainText(),
                document.revision(),
                document.availableUndoSteps(),
                document.isModified(),
                cursor.position(),
                cursor.anchor(),
                item.documentSize(),
            ),
            snapshot,
        )
        current_positions = []
        block = document.firstBlock()
        while block.isValid():
            text_layout = block.layout()
            current_positions.append(
                [text_layout.lineAt(i).position() for i in range(text_layout.lineCount())]
            )
            block = block.next()
        self.assertEqual(current_positions, line_positions)
        self.assertEqual(item._background_pixmap_scale, 2.0)

        def failing_draw(*_args):
            raise RuntimeError('paint failure')

        layout.draw = failing_draw
        try:
            with self.assertRaisesRegex(RuntimeError, 'paint failure'):
                item.repaint_background(4.0)
        finally:
            layout.draw = original_draw
        self.assertFalse(item.repainting)

    def test_direction_switch_preserves_rich_html_revision_and_selection_direction(self):
        item = _make_item(text='ABCD')
        item.setHtml(
            '<p><span style="font-weight:700;color:#2277cc">AB</span>'
            '<span style="font-style:italic;color:#cc4422">CD</span></p>'
        )
        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(3)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        document = item.document()
        html = document.toHtml()
        revision = document.revision()

        item.setVertical(True)
        cursor = item.textCursor()
        self.assertEqual((cursor.position(), cursor.anchor()), (1, 3))
        self.assertEqual(document.toHtml(), html)
        self.assertEqual(document.revision(), revision)

        item.setVertical(False)
        cursor = item.textCursor()
        self.assertEqual((cursor.position(), cursor.anchor()), (1, 3))
        self.assertEqual(document.toHtml(), html)
        self.assertEqual(document.revision(), revision)
        item.endEdit()
        scene.removeItem(item)

    def test_normal_fill_restores_stroke_interiors(self):
        plain = _make_item(text='FILL')
        outlined = _make_item(stroke_width=0.22, text='FILL')
        source = QRectF(0, 0, 380, 280)
        plain_pixels = _image_array(_render_item(plain, source, 2))
        outlined_pixels = _image_array(_render_item(outlined, source, 2))

        plain_green = (
            (plain_pixels[..., 1] > 150)
            & (plain_pixels[..., 0] < 80)
            & (plain_pixels[..., 3] > 128)
        ).sum()
        outlined_green = (
            (outlined_pixels[..., 1] > 150)
            & (outlined_pixels[..., 0] < 80)
            & (outlined_pixels[..., 3] > 128)
        ).sum()
        outlined_red = (
            (outlined_pixels[..., 0] > 150)
            & (outlined_pixels[..., 1] < 80)
            & (outlined_pixels[..., 3] > 128)
        ).sum()
        self.assertGreater(plain_green, 100)
        self.assertGreaterEqual(outlined_green, plain_green * 0.9)
        self.assertGreater(outlined_red, 100)

        effect = pixmap2ndarray(outlined.background_pixmap, keep_alpha=True)[..., 3]
        self.assertEqual(int(effect[0].max()), 0)
        self.assertEqual(int(effect[-1].max()), 0)
        self.assertEqual(int(effect[:, 0].max()), 0)
        self.assertEqual(int(effect[:, -1].max()), 0)

    def test_vertical_rich_text_stroke_covers_every_fragment(self):
        item = _make_item(vertical=True, stroke_width=0.16, text='')
        item.setHtml(
            f'<p><span style="font-family:{_FONT_FAMILY};font-size:24pt;color:#00bb00">A</span>'
            f'<span style="font-family:{_FONT_FAMILY};font-size:40pt;color:#0044dd">B</span></p>'
        )
        item.repaint_background()
        document = item.document()
        fragment_sizes = []
        iterator = document.firstBlock().begin()
        while not iterator.atEnd():
            fragment_sizes.append(iterator.fragment().charFormat().fontPointSize())
            iterator += 1
        self.assertGreaterEqual(len(fragment_sizes), 2)
        self.assertGreater(max(fragment_sizes), min(fragment_sizes))

        plain = item._new_effect_pixmap()
        painter = QPainter(plain)
        try:
            painter.translate(-item.boundingRect().topLeft())
            item._paint_live_layout(painter, item._effect_paint_context())
        finally:
            painter.end()
        plain_alpha = pixmap2ndarray(plain, keep_alpha=True)[..., 3]
        effect = pixmap2ndarray(item.background_pixmap, keep_alpha=True)
        glyph_pixels = plain_alpha > 16
        red_effect = (
            (effect[..., 0] > 150)
            & (effect[..., 1] < 80)
            & (effect[..., 3] > 0)
        )
        self.assertGreater(int(glyph_pixels.sum()), 100)
        self.assertTrue(np.all(red_effect[glyph_pixels]))

    def test_empty_formatted_document_has_no_effect_envelope(self):
        item = _make_item(
            text='',
            stroke_width=0.2,
            shadow_radius=0.25,
            shadow_strength=0.9,
            shadow_offset=(0.3, -0.2),
        )
        before_html = item.toHtml()
        before_revision = item.document().revision()

        item.repaint_background(2.0)

        self.assertTrue(item.document().isEmpty())
        self.assertEqual(item.padding(), 0.0)
        self.assertIsNone(item.background_pixmap)
        self.assertEqual(item.toHtml(), before_html)
        self.assertEqual(item.document().revision(), before_revision)

    def test_effect_removal_shrinks_from_zero_and_keeps_outer_transform_separate(self):
        item = _make_item(
            transform=(4.0, 0.1, 45.0),
            angle=27,
            stroke_width=0.2,
            shadow_radius=0.25,
            shadow_strength=0.9,
            shadow_offset=(-0.3, 0.2),
            gradient=True,
        )
        neutral = _make_item(
            stroke_width=0.2,
            shadow_radius=0.25,
            shadow_strength=0.9,
            shadow_offset=(-0.3, 0.2),
        )
        self.assertAlmostEqual(item.padding(), neutral.padding(), places=5)
        self.assertEqual(item.cacheMode(), QGraphicsItem.CacheMode.NoCache)
        logical_rect = item.absBoundingRect(qrect=True)
        scene_pivot = item.mapToScene(item.transformOriginPoint())
        canonical_transform = item.fontformat.text_transform
        visual_polygon = item.visual_polygon_in_scene()
        self.assertGreater(item.padding(), 0)

        item.setStrokeWidth(0)
        after_stroke = item.padding()
        item.setBGAttribute('shadow_strength', 0)
        self.assertLess(after_stroke, neutral.padding())
        self.assertEqual(item.padding(), 0)
        self.assertEqual(item.absBoundingRect(qrect=True), logical_rect)
        self.assertEqual(item.mapToScene(item.transformOriginPoint()), scene_pivot)
        self.assertEqual(item.fontformat.text_transform, canonical_transform)
        for actual, expected in zip(item.visual_polygon_in_scene(), visual_polygon):
            self.assertAlmostEqual(actual.x(), expected.x())
            self.assertAlmostEqual(actual.y(), expected.y())
        self.assertIsNone(item.background_pixmap)

        gradient = item.get_text_gradient()
        midpoint = (gradient.start() + gradient.finalStop()) / 2
        self.assertAlmostEqual(midpoint.x(), item.logical_unpadded_rect().center().x())
        self.assertAlmostEqual(midpoint.y(), item.logical_unpadded_rect().center().y())

    def test_extreme_transform_rotation_has_alpha_and_resolution_parity(self):
        cases = (
            (0.1, 4.0, 45.0, 23),
            (4.0, 0.1, -45.0, -31),
        )
        for horizontal, vertical, slant, angle in cases:
            with self.subTest(case=(horizontal, vertical, slant, angle)):
                item = _make_item(
                    transform=(horizontal, vertical, slant),
                    angle=angle,
                    stroke_width=0.16,
                    shadow_radius=0.18,
                    shadow_strength=0.75,
                    shadow_offset=(0.2, -0.2),
                    text='EDGE',
                )
                bounds = item.mapToScene(item.boundingRect()).boundingRect()
                source = bounds.adjusted(-8, -8, 8, 8)
                low = _image_array(_render_item(item, source, 1))
                high = _image_array(_render_item(item, source, 2))
                low_alpha = low[..., 3] > 0
                high_alpha = high[..., 3] > 0
                self.assertGreater(int(low_alpha.sum()), 50)
                self.assertGreater(int(high_alpha.sum()), 200)
                self.assertFalse(low_alpha[0].any())
                self.assertFalse(low_alpha[-1].any())
                self.assertFalse(low_alpha[:, 0].any())
                self.assertFalse(low_alpha[:, -1].any())
                normalized_high_area = high_alpha.sum() / 4
                self.assertLess(
                    abs(normalized_high_area - low_alpha.sum()) / low_alpha.sum(),
                    0.3,
                )
                self.assertGreaterEqual(item._background_pixmap_scale, 2.0)

    def test_selection_and_ime_preedit_do_not_enter_effect_cache(self):
        item = _make_item(
            stroke_width=0.16,
            shadow_radius=0.2,
            shadow_strength=0.8,
            text='AB',
        )
        scene = QGraphicsScene()
        scene.addItem(item)
        before = pixmap2ndarray(item.background_pixmap, keep_alpha=True)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.repaint_background()
        np.testing.assert_array_equal(
            pixmap2ndarray(item.background_pixmap, keep_alpha=True), before
        )

        cached = item.background_pixmap
        cached_scale = item._background_pixmap_scale
        item.inputMethodEvent(QInputMethodEvent('X', []))
        self.assertTrue(item.pre_editing)
        item.repaint_background(4.0)
        self.assertIs(item.background_pixmap, cached)
        self.assertEqual(item._background_pixmap_scale, cached_scale)
        np.testing.assert_array_equal(
            pixmap2ndarray(item.background_pixmap, keep_alpha=True), before
        )

        item.inputMethodEvent(QInputMethodEvent('', []))
        item.endEdit()
        scene.removeItem(item)

    def test_transformed_effects_survive_opaque_nonediting_paint(self):
        item = _make_item(
            transform=(1.8, 0.65, -22.0),
            angle=17,
            stroke_width=0.2,
            shadow_radius=0.2,
            shadow_strength=0.9,
            shadow_offset=(0.25, -0.2),
            text='OPAQUE',
        )
        source = QRectF(0, 0, 520, 360)
        nonediting = _image_array(
            _render_item(item, source, 2, QColor(Qt.GlobalColor.white))
        )
        item.startEdit()
        item.clearFocus()
        editing = _image_array(
            _render_item(item, source, 2, QColor(Qt.GlobalColor.white))
        )

        counts = []
        for pixels in (nonediting, editing):
            red = (
                (pixels[..., 0] > 150)
                & (pixels[..., 1] < 100)
                & (pixels[..., 2] < 100)
            ).sum()
            blue = (
                (pixels[..., 2] > 120)
                & (pixels[..., 0] < 100)
                & (pixels[..., 1] < 120)
            ).sum()
            green = (
                (pixels[..., 1] > 140)
                & (pixels[..., 0] < 100)
                & (pixels[..., 2] < 100)
            ).sum()
            self.assertGreater(red, 100)
            self.assertGreater(blue, 100)
            self.assertGreater(green, 100)
            counts.append((red, blue, green))
        for nonediting_count, editing_count in zip(*counts):
            ratio = nonediting_count / editing_count
            self.assertGreater(ratio, 0.5)
            self.assertLess(ratio, 2.0)
        item.endEdit(keep_focus=False)

    def test_vertical_effect_envelope_converges_with_clear_cache_borders(self):
        cases = (
            dict(stroke_width=0.2),
            dict(
                shadow_radius=0.25,
                shadow_strength=0.8,
                shadow_offset=(0.2, -0.2),
            ),
        )
        for effect in cases:
            with self.subTest(effect=effect):
                item = _make_item(vertical=True, text='fjÅW')
                item.fontformat.font_size = pt2px(40)
                item.set_fontformat(item.fontformat.deepcopy())
                if 'stroke_width' in effect:
                    item.setStrokeWidth(effect['stroke_width'])
                else:
                    item.setBGAttribute('shadow_radius', effect['shadow_radius'])
                    item.setBGAttribute('shadow_strength', effect['shadow_strength'])
                    item.setBGAttribute('shadow_offset', list(effect['shadow_offset']))
                item.setRect(QRectF(80, 70, 40, 40))
                item.repaint_background(2.0)
                padding = item.padding()
                revision = item.document().revision()
                item.repaint_background(2.0)
                self.assertEqual(item.padding(), padding)
                self.assertEqual(item.document().revision(), revision)

                alpha = pixmap2ndarray(
                    item.background_pixmap, keep_alpha=True
                )[..., 3]
                self.assertEqual(int(alpha[0].max()), 0)
                self.assertEqual(int(alpha[-1].max()), 0)
                self.assertEqual(int(alpha[:, 0].max()), 0)
                self.assertEqual(int(alpha[:, -1].max()), 0)

    def test_gradient_geometry_refreshes_after_logical_resize(self):
        item = _make_item(gradient=True, text='Gradient')
        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(6)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        plain_text = item.toPlainText()
        cursor_state = (cursor.position(), cursor.anchor())
        document = item.document()
        revision = document.revision()
        undo_steps = document.availableUndoSteps()

        item.setRect(QRectF(80, 70, 400, 100))

        gradient_ranges = [
            format_range
            for format_range in document.firstBlock().layout().formats()
            if bool(
                format_range.format.property(
                    GRADIENT_LAYOUT_FORMAT_PROPERTY
                )
            )
        ]
        self.assertEqual(len(gradient_ranges), 1)
        live_gradient = QLinearGradient(
            gradient_ranges[0].format.foreground().gradient()
        )
        expected = item.get_text_gradient()
        self.assertEqual(live_gradient.start(), expected.start())
        self.assertEqual(live_gradient.finalStop(), expected.finalStop())
        self.assertEqual(item.toPlainText(), plain_text)
        self.assertEqual(document.revision(), revision)
        self.assertEqual(document.availableUndoSteps(), undo_steps)
        self.assertEqual(
            (item.textCursor().position(), item.textCursor().anchor()),
            cursor_state,
        )
        item.endEdit(keep_focus=False)
        scene.removeItem(item)


if __name__ == '__main__':
    unittest.main()
