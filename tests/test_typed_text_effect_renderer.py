import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np

from qtpy.QtCore import QRectF
from qtpy.QtGui import (
    QColor,
    QImage,
    QPainter,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import (
    EffectRasterPlan,
    EffectRasterAllocationError,
)
from ballontranslator.ui.text_engine.rendering.effect_paint import (
    colorize_effect_paint_rgba,
    effect_paint_preview_image,
    rasterize_effect_paint,
)
from ballontranslator.ui.text_engine.rendering.shadow import (
    render_shadow_rgba,
)
from ballontranslator.utils.fontformat import (
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_effects import (
    GradientStop,
    HollowEffect,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class TypedShadowRasterTest(unittest.TestCase):
    def test_drop_blur_spread_and_inner_clipping(self):
        alpha = np.zeros((21, 21), dtype=np.uint8)
        alpha[8:13, 8:13] = 255

        hard = render_shadow_rgba(
            alpha, 'drop', (1, 2, 3), 1.0, (4, -3), 0, 0
        )
        self.assertEqual(hard[5, 12].tolist(), [1, 2, 3, 255])
        self.assertEqual(hard[8, 8, 3], 0)

        soft = render_shadow_rgba(
            alpha, 'drop', (0, 0, 0), 1.0, (0, 0), 2, 2
        )[..., 3]
        self.assertGreater(np.count_nonzero(soft), np.count_nonzero(alpha))
        self.assertTrue(np.any((soft > 0) & (soft < 255)))

        inner = render_shadow_rgba(
            alpha, 'inner', (9, 8, 7), 1.0, (2, 0), 1, 1
        )[..., 3]
        self.assertEqual(np.count_nonzero(inner[alpha == 0]), 0)
        self.assertGreater(np.count_nonzero(inner), 0)

        partial = np.array([[128]], dtype=np.uint8)
        partial_inner = render_shadow_rgba(
            partial, 'inner', (0, 0, 0), 1.0, (0, 0), 0, 0
        )[0, 0, 3]
        self.assertEqual(partial_inner, 64)

    def test_long_shadow_is_connected_for_large_diagonals(self):
        alpha = np.zeros((40, 40), dtype=np.uint8)
        alpha[5, 5] = 255

        long_alpha = render_shadow_rgba(
            alpha, 'long', (0, 0, 0), 1.0, (23, 11), 9, 9
        )[..., 3]

        self.assertEqual(long_alpha[5, 5], 255)
        self.assertEqual(long_alpha[16, 28], 255)
        components, _labels = cv2.connectedComponents(
            (long_alpha > 0).astype(np.uint8), connectivity=8
        )
        self.assertEqual(components, 2)

    def test_blur_is_translation_invariant_at_array_edges(self):
        source = np.zeros((11, 13), dtype=np.uint8)
        source[:4, :5] = 255
        padding = 8
        padded = np.pad(source, padding)

        for shadow_type in ('drop', 'inner'):
            with self.subTest(shadow_type=shadow_type):
                direct = render_shadow_rgba(
                    source, shadow_type, (1, 2, 3), 1.0,
                    (0, 0), 2, 0,
                )
                translated = render_shadow_rgba(
                    padded, shadow_type, (1, 2, 3), 1.0,
                    (0, 0), 2, 0,
                )[padding:padding + 11, padding:padding + 13]
                np.testing.assert_array_equal(direct, translated)


class TypedTextEffectRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def _item(cls, stack: TextEffectStack, vertical: bool = False):
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = 'Typed effects'
        block.vertical = vertical
        block.fontformat.frgb = [240, 20, 20]
        block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _render(item: TextBlkItem) -> np.ndarray:
        scene = item.scene()
        owns_scene = scene is None
        if owns_scene:
            scene = QGraphicsScene()
            scene.addItem(item)
        image = QImage(
            420, 260, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 420, 260),
            QRectF(-30, -30, 420, 260),
        )
        painter.end()
        if owns_scene:
            scene.removeItem(item)
        return pixmap2ndarray(image, keep_alpha=True)

    def test_inner_survives_public_paint_and_vertical_path(self):
        plain = self._item(TextEffectStack())
        inner_stack = TextEffectStack(effects=(ShadowEffect(
            shadow_type='inner',
            color=(0, 0, 0),
            offset=(0.2, 0.0),
            blur=0.12,
            spread=0.03,
        ),))
        inner = self._item(inner_stack)

        plain_pixels = self._render(plain)
        inner_pixels = self._render(inner)

        self.assertFalse(np.array_equal(plain_pixels, inner_pixels))
        foreground = inner_pixels[..., 3] > 0
        self.assertTrue(np.any(inner_pixels[..., 0][foreground] < 180))
        vertical = self._item(inner_stack, vertical=True)
        self.assertGreater(np.count_nonzero(self._render(vertical)[..., 3]), 0)

    def test_gradient_preview_matches_renderer_straight_rgba(self):
        paint = LinearGradientPaint(
            stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(1.0, (0, 0, 255), 0.0),
            ),
        )
        rect = QRectF(0, 0, 1, 1)
        raster = rasterize_effect_paint(
            paint, rect, rect, 1.0, 1, 1
        )
        preview = pixmap2ndarray(
            effect_paint_preview_image(paint, rect, 1.0),
            keep_alpha=True,
        )
        np.testing.assert_array_equal(preview, raster)
        self.assertEqual(raster[0, 0].tolist(), [128, 0, 128, 128])

        coverage = np.full((1, 1, 4), 255, dtype=np.uint8)
        coverage[..., 3] = 128
        result = colorize_effect_paint_rgba(
            paint, coverage, rect, rect, 1.0
        )
        self.assertIs(result, coverage)
        self.assertEqual(coverage[0, 0].tolist(), [128, 0, 128, 64])

        hard = LinearGradientPaint(stops=(
            GradientStop(0.5, (255, 0, 0), 1.0),
            GradientStop(0.5, (0, 0, 255), 1.0),
        ))
        raster = rasterize_effect_paint(
            hard, QRectF(0, 0, 2, 1), QRectF(0, 0, 2, 1), 1.0, 2, 1
        )
        self.assertEqual(raster[0, 0, :3].tolist(), [255, 0, 0])
        self.assertEqual(raster[0, 1, :3].tolist(), [0, 0, 255])

    def test_solid_center_stays_native_while_gradient_center_is_generated(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        solid = TextEffectStack(effects=(StrokeEffect(
            width=0.2, paint=SolidPaint((0, 0, 255))
        ),))
        gradient = TextEffectStack(effects=(StrokeEffect(
            width=0.2,
            paint=LinearGradientPaint(),
        ),))

        with patch.object(
            renderer,
            '_positioned_stroke_band',
            wraps=renderer._positioned_stroke_band,
        ) as band:
            item.set_text_effects(solid)
            self.assertEqual(band.call_count, 0)
            self.assertEqual(renderer._effect_flags(), (True, False))
            item.set_text_effects(gradient)
            self.assertGreater(band.call_count, 0)
            self.assertEqual(renderer._effect_flags(), (True, True))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

    def test_gradient_stroke_positions_flip_hollow_and_multiple_paints(self):
        def gradient(angle=0.0):
            return LinearGradientPaint(stops=(
                GradientStop(0.0, (255, 0, 0), 1.0),
                GradientStop(0.1, (0, 255, 0), 0.5),
                GradientStop(0.2, (0, 0, 255), 1.0),
            ), angle=angle)

        rendered = {}
        for position in ('inside', 'center', 'outside'):
            item = self._item(TextEffectStack(effects=(StrokeEffect(
                width=0.28,
                position=position,
                paint=gradient(),
            ),)))
            item.setPlainText('\N{FULL BLOCK}' * 5)
            item.layout.reLayoutEverything()
            item.repaint_background()
            rendered[position] = self._render(item)
            pixels = rendered[position]
            self.assertTrue(np.any(
                (pixels[..., 0] > 140) & (pixels[..., 2] < 100)
            ))
            self.assertTrue(np.any(
                (pixels[..., 2] > 130)
                & (pixels[..., 2] > pixels[..., 0])
            ))

        normal = rendered['center']
        flipped_item = self._item(TextEffectStack(effects=(StrokeEffect(
            width=0.28, paint=gradient(180)
        ),)))
        flipped_item.setPlainText('\N{FULL BLOCK}')
        flipped_item.layout.reLayoutEverything()
        flipped_item.repaint_background()
        flipped = self._render(flipped_item)

        def channel_center(pixels, channel):
            mask = (pixels[..., channel] > 140) & (pixels[..., 3] > 20)
            return np.argwhere(mask)[:, 1].mean()

        self.assertLess(channel_center(normal, 0), channel_center(normal, 2))
        self.assertGreater(
            channel_center(flipped, 0), channel_center(flipped, 2)
        )

        hollow = self._item(TextEffectStack(effects=(
            StrokeEffect(width=0.28, paint=gradient()),
            HollowEffect(),
        )))
        hollow.setPlainText('\N{FULL BLOCK}')
        hollow.layout.reLayoutEverything()
        hollow.repaint_background()
        hollow_pixels = self._render(hollow)
        bounds = np.argwhere(hollow_pixels[..., 3] > 20)
        center = (bounds.min(0) + bounds.max(0)) // 2
        self.assertEqual(hollow_pixels[center[0], center[1], 3], 0)

        layered = self._item(TextEffectStack(effects=(
            StrokeEffect(width=0.12, paint=gradient()),
            StrokeEffect(
                width=0.28, paint=SolidPaint((255, 255, 0))
            ),
        )))
        layered_pixels = self._render(layered)
        self.assertTrue(np.any(layered_pixels[..., 2] > 140))
        self.assertTrue(np.any(
            (layered_pixels[..., 0] > 140)
            & (layered_pixels[..., 1] > 140)
        ))

    def test_gradient_stop_alpha_reduces_shadow_stroke_silhouette(self):
        def silhouette_sum(stop_opacity):
            item = self._item(TextEffectStack(effects=(
                StrokeEffect(
                    width=0.3,
                    position='outside',
                    paint=LinearGradientPaint(stops=(
                        GradientStop(0.0, (255, 0, 0), stop_opacity),
                        GradientStop(1.0, (0, 0, 255), stop_opacity),
                    )),
                ),
                ShadowEffect(offset=(0.2, 0.1)),
            )))
            item.setPlainText('\N{FULL BLOCK}' * 3)
            item.layout.reLayoutEverything()
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            canonical = renderer._capture_effect_source(bounds, 1.0)
            canonical_alpha = renderer._pixmap_alpha(canonical)
            silhouette = renderer._stroke_silhouette(
                canonical, canonical_alpha, bounds, 1.0
            )
            return int(pixmap2ndarray(
                silhouette, keep_alpha=True
            )[..., 3].sum())

        self.assertLess(silhouette_sum(0.2), silhouette_sum(1.0))

    def test_stroke_positions_control_coverage_and_external_padding(self):
        paint = SolidPaint((20, 60, 220))

        def rendered(position=None):
            options = {'width': 0.24, 'paint': paint}
            if position is not None:
                options['position'] = position
            item = self._item(TextEffectStack(effects=(
                StrokeEffect(**options),
            )))
            item.setPlainText('\N{FULL BLOCK}')
            item.layout.reLayoutEverything()
            item.repaint_background()
            return item, self._render(item)

        default, default_pixels = rendered()
        center, center_pixels = rendered('center')
        inside, inside_pixels = rendered('inside')
        outside, outside_pixels = rendered('outside')
        plain = self._item(TextEffectStack())
        plain.setPlainText('\N{FULL BLOCK}')
        plain.layout.reLayoutEverything()
        plain_pixels = self._render(plain)

        np.testing.assert_array_equal(default_pixels, center_pixels)
        self.assertEqual(inside.padding(), 0.0)
        self.assertAlmostEqual(
            outside.effect_renderer._conservative_effect_padding(),
            center.effect_renderer._conservative_effect_padding() * 2.0,
        )
        # Committed padding is rounded outward to 1/64 layout units.
        self.assertAlmostEqual(
            outside.padding(),
            center.padding() * 2.0,
            delta=1.0 / 32.0,
        )

        glyph = plain_pixels[..., 3] > 16
        deep_glyph = cv2.erode(
            glyph.astype(np.uint8), np.ones((7, 7), dtype=np.uint8)
        ) > 0

        def blue(pixels):
            return (
                (pixels[..., 2] > 140)
                & (pixels[..., 0] < 100)
                & (pixels[..., 3] > 16)
            )

        inside_blue = blue(inside_pixels)
        outside_blue = blue(outside_pixels)
        center_blue = blue(center_pixels)
        self.assertGreater(np.count_nonzero(inside_blue), 0)
        self.assertEqual(np.count_nonzero(inside_blue & ~glyph), 0)
        self.assertGreater(np.count_nonzero(outside_blue & ~glyph), 0)
        self.assertEqual(np.count_nonzero(outside_blue & deep_glyph), 0)
        self.assertGreater(
            np.count_nonzero(outside_blue & ~glyph),
            np.count_nonzero(center_blue & ~glyph),
        )

    def test_hollow_and_shadow_silhouette_use_positioned_stroke_band(self):
        plain = self._item(TextEffectStack())
        plain.setPlainText('\N{FULL BLOCK}')
        plain.layout.reLayoutEverything()
        glyph = self._render(plain)[..., 3] > 16
        distance = cv2.distanceTransform(
            glyph.astype(np.uint8), cv2.DIST_L2, 5
        )
        deepest = np.unravel_index(np.argmax(distance), distance.shape)

        for position in ('inside', 'center', 'outside'):
            with self.subTest(position=position):
                stroke = StrokeEffect(
                    width=0.24,
                    position=position,
                    paint=SolidPaint((20, 60, 220)),
                )
                item = self._item(TextEffectStack(effects=(
                    stroke,
                    HollowEffect(),
                )))
                item.setPlainText('\N{FULL BLOCK}')
                item.layout.reLayoutEverything()
                item.repaint_background()
                pixels = self._render(item)
                blue = (
                    (pixels[..., 2] > 140)
                    & (pixels[..., 0] < 100)
                    & (pixels[..., 3] > 16)
                )

                self.assertGreater(np.count_nonzero(blue), 0)
                self.assertEqual(pixels[deepest][3], 0)
                if position == 'inside':
                    self.assertEqual(np.count_nonzero(blue & ~glyph), 0)
                elif position == 'outside':
                    self.assertGreater(np.count_nonzero(blue & ~glyph), 0)

                renderer = item.effect_renderer
                bounds = renderer.boundingRect()
                canonical = renderer._capture_effect_source(bounds, 1.0)
                canonical_alpha = renderer._pixmap_alpha(canonical)
                silhouette = renderer._stroke_silhouette(
                    canonical, canonical_alpha, bounds, 1.0
                )
                silhouette_alpha = pixmap2ndarray(
                    silhouette, keep_alpha=True
                )[..., 3]
                outside_source = (
                    (silhouette_alpha > 16) & (canonical_alpha <= 16)
                )
                if position == 'inside':
                    self.assertEqual(np.count_nonzero(outside_source), 0)
                else:
                    self.assertGreater(np.count_nonzero(outside_source), 0)

    def test_inside_does_not_add_nonlinear_external_padding(self):
        transform = TextTransformStack((SineTextTransform(),))
        plain = self._item(TextEffectStack())
        inside = self._item(TextEffectStack(effects=(
            StrokeEffect(width=0.24, position='inside'),
        )))
        outside = self._item(TextEffectStack(effects=(
            StrokeEffect(width=0.24, position='outside'),
        )))

        for item in (plain, inside, outside):
            item.set_text_transform(transform)

        self.assertEqual(inside.padding(), plain.padding())
        self.assertGreater(outside.padding(), inside.padding())
        self.assertGreater(np.count_nonzero(self._render(inside)[..., 3]), 0)
        self.assertGreater(np.count_nonzero(self._render(outside)[..., 3]), 0)

    def test_first_shadow_card_paints_on_top_within_exterior_phase(self):
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(color=(255, 0, 0), offset=(0.0, 0.0)),
            ShadowEffect(color=(0, 0, 255), offset=(0.0, 0.0)),
        )))
        pixels = pixmap2ndarray(
            item.effect_renderer.background_pixmap, keep_alpha=True
        )
        opaque = pixels[..., 3] > 200

        self.assertGreater(np.count_nonzero(opaque), 0)
        self.assertTrue(np.all(pixels[..., 0][opaque] > pixels[..., 2][opaque]))

    def test_drop_preserves_fractional_relative_offset(self):
        item = self._item(TextEffectStack())
        source_alpha = np.zeros((7, 7), dtype=np.uint8)
        source_alpha[3, 3] = 255

        pixmap = item.effect_renderer._shadow_pixmap(
            source_alpha,
            ShadowEffect(offset=(0.01, 0.0)),
            1.0,
        )
        alpha = pixmap2ndarray(pixmap, keep_alpha=True)[..., 3]

        shifted = alpha[3, 3:5]
        self.assertTrue(np.all(shifted > 0))
        self.assertTrue(np.all(shifted < 255))

    def test_fractional_exterior_offsets_keep_horizontal_raster_guard(self):
        for shadow_type in ('drop', 'long'):
            for direction in (-1, 1):
                with self.subTest(
                    shadow_type=shadow_type, direction=direction
                ):
                    item = self._item(TextEffectStack(effects=(
                        ShadowEffect(
                            shadow_type=shadow_type,
                            color=(0, 0, 255),
                            offset=(direction * 0.01, 0.0),
                        ),
                    )))
                    item.setPlainText('\N{FULL BLOCK}')
                    item.layout.reLayoutEverything()
                    item.repaint_background()

                    alpha = pixmap2ndarray(
                        item.effect_renderer.background_pixmap,
                        keep_alpha=True,
                    )[..., 3]
                    self.assertGreater(np.count_nonzero(alpha), 0)
                    self.assertEqual(np.max(alpha[:, 0]), 0)
                    self.assertEqual(np.max(alpha[:, -1]), 0)

        inner = self._item(TextEffectStack(effects=(
            ShadowEffect(shadow_type='inner'),
        )))
        self.assertEqual(inner.padding(), 0.0)

    def test_hollow_has_no_blank_cache_and_preserves_stroke_coverage(self):
        hollow = self._item(TextEffectStack(effects=(HollowEffect(),)))
        self.assertIsNone(hollow.effect_renderer._effect_raster_state)
        self.assertEqual(np.count_nonzero(self._render(hollow)[..., 3]), 0)
        hollow_inner = self._item(TextEffectStack(effects=(
            HollowEffect(),
            ShadowEffect(shadow_type='inner', blur=0.2),
        )))
        self.assertIsNone(hollow_inner.effect_renderer._effect_raster_state)
        self.assertEqual(hollow_inner.padding(), 0.0)

        stroke = StrokeEffect(
            width=0.28, paint=SolidPaint((20, 60, 220))
        )
        normal = self._item(TextEffectStack(effects=(stroke,)))
        outlined = self._item(TextEffectStack(effects=(stroke, HollowEffect())))
        normal.setPlainText('\N{FULL BLOCK}')
        outlined.setPlainText('\N{FULL BLOCK}')
        normal_pixels = self._render(normal)
        hollow_pixels = self._render(outlined)
        normal_alpha = normal_pixels[..., 3]
        hollow_alpha = hollow_pixels[..., 3]

        self.assertGreater(np.count_nonzero(hollow_alpha), 0)
        normal_bounds = np.argwhere(normal_alpha > 0)
        hollow_bounds = np.argwhere(hollow_alpha > 0)
        self.assertEqual(
            (normal_bounds.min(0).tolist(), normal_bounds.max(0).tolist()),
            (hollow_bounds.min(0).tolist(), hollow_bounds.max(0).tolist()),
        )
        deep_interior = cv2.erode(
            (normal_alpha > 200).astype(np.uint8),
            np.ones((13, 13), dtype=np.uint8),
        ) > 0
        self.assertGreater(np.count_nonzero(deep_interior), 0)
        center = ((normal_bounds.min(0) + normal_bounds.max(0)) // 2)
        self.assertEqual(hollow_alpha[center[0], center[1]], 0)
        self.assertLess(
            np.count_nonzero(hollow_alpha[deep_interior] > 20),
            np.count_nonzero(deep_interior) * 0.25,
        )
        visible = hollow_alpha > 100
        self.assertTrue(np.any(
            (hollow_pixels[..., 2] > 150)
            & (hollow_pixels[..., 0] < 80)
            & visible
        ))
        self.assertFalse(np.any(
            (hollow_pixels[..., 0] > 150)
            & (hollow_pixels[..., 2] < 100)
            & visible
        ))

    def test_hollow_strokes_follow_each_rich_text_fragment_size(self):
        item = self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.10,
                opacity=0.8,
                paint=SolidPaint((0, 0, 255)),
            ),
            StrokeEffect(
                width=0.22,
                paint=SolidPaint((0, 255, 0)),
            ),
            HollowEffect(),
        )))
        item.setPlainText(
            '\N{FULL BLOCK}  \N{FULL BLOCK}'
        )
        for position, point_size in ((0, 64), (3, 10)):
            cursor = QTextCursor(item.document())
            cursor.setPosition(position)
            cursor.setPosition(
                position + 1, QTextCursor.MoveMode.KeepAnchor
            )
            char_format = QTextCharFormat()
            char_format.setFontPointSize(point_size)
            cursor.mergeCharFormat(char_format)
        item.layout.reLayoutEverything()
        item.repaint_background()

        pixels = self._render(item)
        alpha = pixels[..., 3]
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (alpha > 20).astype(np.uint8), connectivity=8
        )

        self.assertEqual(count, 3)
        for x, y, width, height, _area in stats[1:]:
            self.assertEqual(alpha[y + height // 2, x + width // 2], 0)
        visible = alpha > 20
        self.assertTrue(np.any((pixels[..., 2] > 150) & visible))
        self.assertTrue(np.any((pixels[..., 1] > 150) & visible))
        self.assertFalse(np.any(
            (pixels[..., 0] > 150)
            & (pixels[..., 1] < 100)
            & visible
        ))

    def test_hollow_knocks_foreground_from_exterior_but_keeps_shadow(self):
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(
                shadow_type='drop',
                color=(0, 0, 0),
                offset=(0.45, 0.0),
            ),
            HollowEffect(),
        )))
        pixels = self._render(item)
        visible = pixels[..., 3] > 0

        self.assertGreater(np.count_nonzero(visible), 0)
        self.assertFalse(np.any(
            (pixels[..., 0] > 180)
            & (pixels[..., 1] < 80)
            & visible
        ))

    def test_hollow_selection_feedback_is_transient_native_and_nonlinear(self):
        item = self._item(TextEffectStack(effects=(HollowEffect(),)))
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.show()
        scene.addItem(item)
        item.startEdit()
        view.setFocus()
        item.setFocus()
        self.app.processEvents()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)
        clipped = QImage(
            160, 100, QImage.Format.Format_ARGB32_Premultiplied
        )
        clipped.fill(QColor(0, 0, 0, 0))
        clipped_painter = QPainter(clipped)
        renderer = item.effect_renderer
        with patch.object(
            renderer,
            '_new_effect_pixmap',
            wraps=renderer._new_effect_pixmap,
        ) as allocate:
            scene.render(
                clipped_painter,
                QRectF(0, 0, 160, 100),
                QRectF(0, 0, 80, 50),
            )
        clipped_painter.end()
        render_scale, interaction_rect = allocate.call_args.args
        self.assertEqual(render_scale, 2.0)
        self.assertLessEqual(interaction_rect.width(), 80.0)
        self.assertLessEqual(interaction_rect.height(), 50.0)
        self.assertLess(
            interaction_rect.width(), renderer.boundingRect().width()
        )

        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

        item.set_export_effect_render(True)
        try:
            self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)
        finally:
            item.set_export_effect_render(False)

    def test_inner_failure_falls_back_to_base_and_export_remains_strict(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        inner = TextEffectStack(effects=(ShadowEffect(
            shadow_type='inner', blur=0.2
        ),))
        failure = EffectRasterAllocationError('mock typed effect failure')

        with patch.object(
            renderer, '_render_effect_surface', side_effect=failure
        ):
            item.set_text_effects(inner)
            pixels = self._render(item)
        self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)

        item.set_export_effect_render(True)
        try:
            with patch.object(
                renderer, '_render_effect_surface', side_effect=failure
            ):
                self._render(item)
            self.assertIsInstance(item.export_effect_error, Exception)
        finally:
            item.set_export_effect_render(False)

    def test_generated_stroke_failure_never_uses_center_vector_fallback(self):
        failure = EffectRasterAllocationError('mock positioned Stroke failure')
        cases = (
            ('inside', SolidPaint((0, 0, 255))),
            ('outside', SolidPaint((0, 0, 255))),
            ('center', LinearGradientPaint()),
        )
        for position, paint in cases:
            with self.subTest(position=position, paint=paint.paint_type):
                item = self._item(TextEffectStack())
                renderer = item.effect_renderer
                stack = TextEffectStack(effects=(StrokeEffect(
                    width=0.24,
                    position=position,
                    paint=paint,
                ),))

                with patch.object(
                    renderer, '_render_effect_surface', side_effect=failure
                ):
                    item.set_text_effects(stack)
                    pixels = self._render(item)
                self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)
                self.assertFalse(renderer.direct_stroke)

                item.set_export_effect_render(True)
                try:
                    with patch.object(
                        renderer,
                        '_render_effect_surface',
                        side_effect=failure,
                    ):
                        self._render(item)
                    self.assertIsInstance(
                        item.export_effect_error,
                        EffectRasterAllocationError,
                    )
                    self.assertFalse(renderer.direct_stroke)
                finally:
                    item.set_export_effect_render(False)

    def test_shadow_preview_promotes_cache_and_reshape_rebuilds_once(self):
        before = TextEffectStack(effects=(ShadowEffect(
            offset=(0.1, 0.1), blur=0.08
        ),))
        after = TextEffectStack(effects=(ShadowEffect(
            offset=(0.3, -0.1), blur=0.12
        ),))
        item = self._item(before)
        renderer = item.effect_renderer

        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render:
            item.set_text_effects(after, preview=True)
            scratch = renderer._preview_effect_raster_state
            self.assertEqual(render.call_count, 1)
            item.set_text_effects(after)
            self.assertEqual(render.call_count, 1)
            self.assertIs(renderer._effect_raster_state, scratch)

            item.startReshape()
            item.setRect(QRectF(0, 0, 300, 170))
            item.repaint_background()
            item.setRect(QRectF(0, 0, 290, 160))
            item.repaint_background()
            self.assertEqual(render.call_count, 1)
            item.endReshape()
            self.assertEqual(render.call_count, 2)
        self.assertEqual(item.blk.fontformat.text_effects, after)

    def test_gradient_stroke_preview_promotes_completed_cache(self):
        before = TextEffectStack(effects=(StrokeEffect(
            width=0.12, paint=LinearGradientPaint()
        ),))
        after = TextEffectStack(effects=(StrokeEffect(
            width=0.18,
            paint=LinearGradientPaint(angle=90.0, scale=1.5),
        ),))
        item = self._item(before)
        renderer = item.effect_renderer
        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render:
            item.set_text_effects(after, preview=True)
            scratch = renderer._preview_effect_raster_state
            self.assertEqual(render.call_count, 1)
            item.set_text_effects(after)
            self.assertEqual(render.call_count, 1)
            self.assertIs(renderer._effect_raster_state, scratch)

    def test_forced_tiles_match_full_typed_effect_surface(self):
        stacks = (
            TextEffectStack(effects=(
                ShadowEffect(
                    offset=(0.18, 0.12), blur=0.08, spread=0.04
                ),
                StrokeEffect(
                    width=0.12,
                    position='outside',
                    paint=LinearGradientPaint(
                        stops=(
                            GradientStop(0.0, (255, 0, 0), 0.25),
                            GradientStop(0.5, (0, 255, 0), 0.75),
                            GradientStop(1.0, (0, 0, 255), 1.0),
                        ),
                        angle=37.0,
                        scale=1.4,
                    ),
                ),
                ShadowEffect(
                    shadow_type='inner',
                    offset=(0.08, 0.04),
                    blur=0.06,
                    spread=0.02,
                ),
            )),
            TextEffectStack(effects=(
                ShadowEffect(
                    shadow_type='long', offset=(0.30, 0.22)
                ),
                StrokeEffect(width=0.12, position='inside'),
                HollowEffect(),
            )),
        )
        for stack in stacks:
            for scale in (1.0, 2.0):
                with self.subTest(stack=stack, scale=scale):
                    item = self._item(stack)
                    renderer = item.effect_renderer
                    bounds = renderer.boundingRect()
                    full = renderer._render_effect_surface(bounds, scale)
                    tiled = renderer._new_effect_pixmap(scale, bounds)
                    painter = QPainter(tiled)
                    painter.translate(-bounds.topLeft())
                    renderer.tile_cache.clear()
                    try:
                        with patch.object(
                            renderer,
                            '_render_effect_surface',
                            wraps=renderer._render_effect_surface,
                        ) as render_tile:
                            renderer._draw_tiled_effects(
                                painter,
                                EffectRasterPlan(
                                    'tiles', scale, 0, 0, 64
                                ),
                                bounds,
                            )
                        self.assertGreater(render_tile.call_count, 1)
                    finally:
                        painter.end()

                    np.testing.assert_array_equal(
                        pixmap2ndarray(full, keep_alpha=True),
                        pixmap2ndarray(tiled, keep_alpha=True),
                    )

    def test_hollow_bridge_failures_use_safe_fallback_and_strict_export(self):
        stack = TextEffectStack(effects=(
            StrokeEffect(width=0.2), HollowEffect()
        ))
        for error in (
            RuntimeError('runtime bridge failure'),
            ValueError('value bridge failure'),
            BufferError('buffer bridge failure'),
        ):
            with self.subTest(error=type(error).__name__):
                item = self._item(TextEffectStack())
                with patch(
                    'ballontranslator.ui.text_engine.effect_renderer.'
                    'pixmap2ndarray',
                    side_effect=error,
                ):
                    item.set_text_effects(stack)
                    pixels = self._render(item)
                self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)

        item = self._item(stack)
        item.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'pixmap2ndarray',
                side_effect=BufferError('strict bridge failure'),
            ):
                self._render(item)
            self.assertIsInstance(
                item.export_effect_error, EffectRasterAllocationError
            )
        finally:
            item.set_export_effect_render(False)

    def test_hollow_failure_keeps_direct_stroke_without_foreground(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        stack = TextEffectStack(effects=(
            StrokeEffect(width=0.2, paint=SolidPaint((0, 0, 255))),
            ShadowEffect(blur=0.2),
            HollowEffect(),
        ))
        failure = EffectRasterAllocationError('mock typed effect failure')

        with patch.object(
            renderer, '_render_effect_surface', side_effect=failure
        ):
            item.set_text_effects(stack)
            pixels = self._render(item)

        self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)
        visible = pixels[..., 3] > 100
        self.assertFalse(np.any(
            (pixels[..., 0] > 180)
            & (pixels[..., 1] < 80)
            & visible
        ))


if __name__ == '__main__':
    unittest.main()
