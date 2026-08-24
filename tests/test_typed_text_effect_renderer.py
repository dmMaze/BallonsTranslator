import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPalette,
    QTextCharFormat,
    QTextCursor,
)
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QGraphicsView,
    QStyle,
    QStyleOptionGraphicsItem,
)

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import (
    EFFECT_RASTER_GUARD,
    EffectRasterPlan,
    EffectRasterAllocationError,
)
from ballontranslator.ui.text_engine.rendering.effect_paint import (
    colorize_effect_paint_rgba,
    effect_paint_preview_image,
    rasterize_effect_paint,
)
from ballontranslator.ui.text_engine.rendering.shadow import (
    render_glow_alpha,
    render_shadow_alpha,
)
from ballontranslator.utils.fontformat import (
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_effects import (
    GlowEffect,
    GradientOverlayEffect,
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
    def test_outer_and_inner_glow_alpha_clip_to_their_semantic_side(self):
        alpha = np.zeros((21, 21), dtype=np.uint8)
        alpha[7:14, 7:14] = 255

        outer = render_glow_alpha(alpha, 'outer', 2, 1)
        inner = render_glow_alpha(alpha, 'inner', 2, 1)

        self.assertGreater(np.count_nonzero(outer), 0)
        self.assertEqual(np.count_nonzero(outer[alpha > 0]), 0)
        self.assertGreater(np.count_nonzero(inner), 0)
        self.assertEqual(np.count_nonzero(inner[alpha == 0]), 0)
        with self.assertRaises(ValueError):
            render_glow_alpha(alpha, 'future', 1, 0)

        padding = 8
        padded = np.pad(alpha, padding)
        for glow_type in ('outer', 'inner'):
            with self.subTest(glow_type=glow_type):
                translated = render_glow_alpha(
                    padded, glow_type, 2, 1
                )[padding:padding + 21, padding:padding + 21]
                np.testing.assert_array_equal(
                    render_glow_alpha(alpha, glow_type, 2, 1),
                    translated,
                )

    def test_drop_blur_spread_and_inner_clipping(self):
        alpha = np.zeros((21, 21), dtype=np.uint8)
        alpha[8:13, 8:13] = 255

        hard = render_shadow_alpha(
            alpha, 'drop', 1.0, (4, -3), 0, 0
        )
        self.assertEqual(hard[5, 12], 255)
        self.assertEqual(hard[8, 8], 0)

        soft = render_shadow_alpha(
            alpha, 'drop', 1.0, (0, 0), 2, 2
        )
        self.assertGreater(np.count_nonzero(soft), np.count_nonzero(alpha))
        self.assertTrue(np.any((soft > 0) & (soft < 255)))

        inner = render_shadow_alpha(
            alpha, 'inner', 1.0, (2, 0), 1, 1
        )
        self.assertEqual(np.count_nonzero(inner[alpha == 0]), 0)
        self.assertGreater(np.count_nonzero(inner), 0)

        partial = np.array([[128]], dtype=np.uint8)
        partial_inner = render_shadow_alpha(
            partial, 'inner', 1.0, (0, 0), 0, 0
        )[0, 0]
        self.assertEqual(partial_inner, 64)

    def test_long_shadow_is_connected_for_large_diagonals(self):
        alpha = np.zeros((40, 40), dtype=np.uint8)
        alpha[5, 5] = 255

        long_alpha = render_shadow_alpha(
            alpha, 'long', 1.0, (23, 11), 9, 9
        )

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
                direct = render_shadow_alpha(
                    source, shadow_type, 1.0, (0, 0), 2, 0,
                )
                translated = render_shadow_alpha(
                    padded, shadow_type, 1.0, (0, 0), 2, 0,
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

    def test_glow_phases_source_hollow_padding_and_opacity(self):
        outer = GlowEffect(
            paint=SolidPaint((0, 0, 255)), size=0.12, spread=0.04
        )
        inner = GlowEffect(
            glow_type='inner',
            paint=SolidPaint((0, 255, 0)),
            size=0.12,
            spread=0.03,
        )
        item = self._item(TextEffectStack(effects=(
            outer,
            StrokeEffect(
                width=0.18,
                position='outside',
                paint=SolidPaint((255, 0, 0)),
            ),
            GradientOverlayEffect(
                paint=self._constant_gradient((220, 80, 20))
            ),
            inner,
        )))
        item.setPlainText('\N{FULL BLOCK}')
        item.layout.reLayoutEverything()
        item.repaint_background()
        pixels = self._render(item)
        self.assertTrue(np.any(
            (pixels[..., 2] > pixels[..., 0]) & (pixels[..., 3] > 20)
        ))
        self.assertTrue(np.any(
            (pixels[..., 1] > pixels[..., 0]) & (pixels[..., 3] > 20)
        ))

        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        canonical = renderer._capture_effect_source(bounds, 1.0)
        canonical_alpha = renderer._pixmap_alpha(canonical)
        silhouette = renderer._stroke_silhouette(
            canonical, canonical_alpha, bounds, 1.0
        )
        silhouette_alpha = renderer._pixmap_alpha(silhouette)
        self.assertGreater(
            int(silhouette_alpha.sum()), int(canonical_alpha.sum())
        )
        with patch.object(
            renderer, '_glow_pixmap', wraps=renderer._glow_pixmap
        ) as glow_pixmap:
            renderer._render_pre_mask_effect_surface(bounds, 1.0)
        outer_call = next(
            call for call in glow_pixmap.call_args_list
            if call.args[1].glow_type == 'outer'
        )
        np.testing.assert_array_equal(
            outer_call.args[0], silhouette_alpha
        )

        inner_only = self._item(TextEffectStack(effects=(inner,)))
        self.assertEqual(inner_only.padding(), 0.0)
        outer_only = self._item(TextEffectStack(effects=(outer,)))
        font_size = outer_only.layout.max_font_size(to_px=True)
        self.assertAlmostEqual(
            outer_only.effect_renderer._conservative_effect_padding(),
            (outer.size + outer.spread) * font_size + EFFECT_RASTER_GUARD,
        )

        hollow_inner = self._item(TextEffectStack(effects=(
            HollowEffect(), inner,
        )))
        self.assertIsNone(hollow_inner.effect_renderer._effect_raster_state)
        self.assertEqual(np.count_nonzero(self._render(hollow_inner)[..., 3]), 0)
        hollow_outer = self._item(TextEffectStack(effects=(
            outer, HollowEffect(),
        )))
        self.assertGreater(
            np.count_nonzero(self._render(hollow_outer)[..., 3]), 0
        )

        half = self._item(TextEffectStack(
            overall_opacity=0.5, effects=(outer,)
        ))
        opaque_alpha = self._render(outer_only)[..., 3].sum()
        half_alpha = self._render(half)[..., 3].sum()
        self.assertLess(half_alpha, opaque_alpha * 0.65)
        self.assertGreater(half_alpha, opaque_alpha * 0.35)

    def test_glow_pixmap_applies_coverage_and_each_opacity_once(self):
        source_alpha = np.array([[0, 255]], dtype=np.uint8)
        generated_alpha = np.array([[255, 0]], dtype=np.uint8)
        half_paint = LinearGradientPaint(stops=(
            GradientStop(0.0, (20, 40, 60), 0.5),
            GradientStop(1.0, (20, 40, 60), 0.5),
        ))
        cases = (
            (GlowEffect(opacity=0.5), 128),
            (GlowEffect(paint=half_paint), 128),
            (GlowEffect(opacity=0.5, paint=half_paint), 64),
        )
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer

        for glow, expected_alpha in cases:
            with self.subTest(glow=glow), patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'render_glow_alpha',
                return_value=generated_alpha.copy(),
            ) as render_alpha:
                pixmap = renderer._glow_pixmap(
                    source_alpha, glow, QRectF(0, 0, 2, 1), 1.0
                )
                pixels = pixmap2ndarray(pixmap, keep_alpha=True)

            self.assertEqual(int(pixels[0, 0, 3]), expected_alpha)
            self.assertEqual(int(pixels[0, 1, 3]), 0)
            np.testing.assert_array_equal(
                render_alpha.call_args.args[0], source_alpha
            )
            self.assertEqual(render_alpha.call_args.args[1], 'outer')

    def test_shadow_and_glow_share_retained_phase_order(self):
        effects = (
            ShadowEffect(spread=0.1),
            GlowEffect(size=0.1, spread=0.05),
            GlowEffect(glow_type='inner', size=0.1),
            ShadowEffect(shadow_type='inner', blur=0.1),
        )
        item = self._item(TextEffectStack(effects=effects))
        renderer = item.effect_renderer
        calls = []
        original = renderer._generated_effect_pixmap

        def record(source_alpha, effect, surface_rect, render_scale):
            calls.append(effect)
            return original(
                source_alpha, effect, surface_rect, render_scale
            )

        with patch.object(
            renderer, '_generated_effect_pixmap', side_effect=record
        ):
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )

        self.assertEqual(tuple(calls), (
            effects[1], effects[0], effects[3], effects[2]
        ))

    def test_outer_glow_uses_each_positioned_stroke_silhouette(self):
        for position in ('inside', 'center', 'outside'):
            with self.subTest(position=position):
                item = self._item(TextEffectStack(effects=(
                    GlowEffect(size=0.08),
                    StrokeEffect(width=0.16, position=position),
                )))
                renderer = item.effect_renderer
                bounds = renderer.boundingRect()
                canonical = renderer._capture_effect_source(bounds, 1.0)
                canonical_alpha = renderer._pixmap_alpha(canonical)
                expected = renderer._pixmap_alpha(
                    renderer._stroke_silhouette(
                        canonical, canonical_alpha, bounds, 1.0
                    )
                )
                with patch.object(
                    renderer, '_glow_pixmap', wraps=renderer._glow_pixmap
                ) as glow_pixmap:
                    renderer._render_pre_mask_effect_surface(bounds, 1.0)
                np.testing.assert_array_equal(
                    glow_pixmap.call_args.args[0], expected
                )

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

        exact = LinearGradientPaint(stops=(
            GradientStop(0.0, (10, 20, 30), 0.25),
            GradientStop(1.0, (210, 120, 50), 0.75),
        ))
        coverage = np.full((1, 4, 4), 255, dtype=np.uint8)
        coverage[..., 3] = (255, 128, 64, 1)
        colorize_effect_paint_rgba(
            exact,
            coverage,
            QRectF(0, 0, 4, 1),
            QRectF(0, 0, 4, 1),
            1.0,
        )
        np.testing.assert_array_equal(coverage[0], np.array((
            (35, 32, 32, 80),
            (85, 58, 38, 56),
            (135, 82, 42, 36),
            (185, 108, 48, 1),
        ), dtype=np.uint8))

        hard = LinearGradientPaint(stops=(
            GradientStop(0.5, (255, 0, 0), 1.0),
            GradientStop(0.5, (0, 0, 255), 1.0),
        ))
        raster = rasterize_effect_paint(
            hard, QRectF(0, 0, 3, 1), QRectF(0, 0, 3, 1), 1.0, 3, 1
        )
        self.assertEqual(raster[0, 0, :3].tolist(), [255, 0, 0])
        self.assertEqual(raster[0, 1, :3].tolist(), [0, 0, 255])
        self.assertEqual(raster[0, 2, :3].tolist(), [0, 0, 255])

        opaque = LinearGradientPaint(stops=(
            GradientStop(0.0, (0, 100, 240), 1.0),
            GradientStop(1.0, (0, 100, 240), 1.0),
        ))
        canonical = np.array([[[200, 20, 40, 127]]], dtype=np.uint8)
        recolored = colorize_effect_paint_rgba(
            opaque,
            canonical.copy(),
            rect,
            rect,
            1.0,
            source_atop_opacity=1.0,
        )
        self.assertEqual(recolored[0, 0].tolist(), [0, 100, 240, 127])
        tinted = colorize_effect_paint_rgba(
            opaque,
            canonical.copy(),
            rect,
            rect,
            1.0,
            source_atop_opacity=0.5,
        )
        self.assertEqual(tinted[0, 0].tolist(), [100, 60, 140, 127])
        half_alpha = LinearGradientPaint(stops=(
            GradientStop(0.0, (0, 100, 240), 0.5),
            GradientStop(1.0, (0, 100, 240), 0.5),
        ))
        stop_tinted = colorize_effect_paint_rgba(
            half_alpha,
            canonical.copy(),
            rect,
            rect,
            1.0,
            source_atop_opacity=1.0,
        )
        np.testing.assert_array_equal(stop_tinted, tinted)

    def test_canonical_alpha_is_only_extracted_for_alpha_consumers(self):
        overlay = GradientOverlayEffect(
            paint=self._constant_gradient((20, 60, 220))
        )
        item = self._item(TextEffectStack(effects=(overlay,)))
        renderer = item.effect_renderer
        with patch.object(
            renderer, '_pixmap_alpha', wraps=renderer._pixmap_alpha
        ) as pixmap_alpha:
            rendered = renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )
        self.assertEqual(pixmap_alpha.call_count, 0)
        self.assertGreater(np.count_nonzero(
            pixmap2ndarray(rendered, keep_alpha=True)[..., 3]
        ), 0)

        center_gradient = self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.2,
                position='center',
                paint=LinearGradientPaint(),
            ),
        )))
        renderer = center_gradient.effect_renderer
        with patch.object(
            renderer, '_pixmap_alpha', wraps=renderer._pixmap_alpha
        ) as pixmap_alpha:
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )
        self.assertEqual(pixmap_alpha.call_count, 0)

        consumers = (
            ShadowEffect(),
            ShadowEffect(shadow_type='inner'),
            StrokeEffect(width=0.2, position='outside'),
        )
        for effect in consumers:
            with self.subTest(effect=effect):
                item = self._item(TextEffectStack(effects=(effect,)))
                renderer = item.effect_renderer
                with patch.object(
                    renderer, '_pixmap_alpha', wraps=renderer._pixmap_alpha
                ) as pixmap_alpha:
                    renderer._render_pre_mask_effect_surface(
                        renderer.boundingRect(), 1.0
                    )
                self.assertGreater(pixmap_alpha.call_count, 0)

    @staticmethod
    def _constant_gradient(color, stop_opacity=1.0):
        return LinearGradientPaint(stops=(
            GradientStop(0.0, color, stop_opacity),
            GradientStop(1.0, color, stop_opacity),
        ))

    def test_gradient_overlay_replaces_and_tints_canonical_face(self):
        plain = self._item(TextEffectStack())
        opaque = self._item(TextEffectStack(effects=(
            GradientOverlayEffect(
                paint=self._constant_gradient((0, 0, 255))
            ),
        )))
        partial = self._item(TextEffectStack(effects=(
            GradientOverlayEffect(
                opacity=0.5,
                paint=self._constant_gradient((0, 0, 255)),
            ),
        )))

        plain_pixels = self._render(plain)
        opaque_pixels = self._render(opaque)
        partial_pixels = self._render(partial)
        self.assertLessEqual(
            np.max(np.abs(
                plain_pixels[..., 3].astype(np.int16)
                - opaque_pixels[..., 3].astype(np.int16)
            )),
            4,
        )
        visible = opaque_pixels[..., 3] > 160
        self.assertTrue(np.any(visible))
        self.assertGreater(
            np.mean(opaque_pixels[..., 2][visible]),
            np.mean(opaque_pixels[..., 0][visible]),
        )
        partial_visible = partial_pixels[..., 3] > 160
        self.assertGreater(np.mean(partial_pixels[..., 0][partial_visible]), 40)
        self.assertGreater(np.mean(partial_pixels[..., 2][partial_visible]), 40)

    def test_gradient_overlay_phase_hollow_and_shadow_silhouette(self):
        overlay = GradientOverlayEffect(
            paint=self._constant_gradient((0, 0, 255))
        )
        shadow = ShadowEffect(
            paint=SolidPaint((0, 255, 0)),
            offset=(0.2, 0.1),
            blur=0.05,
        )
        without_overlay = self._render(self._item(TextEffectStack(effects=(
            shadow,
        ))))
        with_overlay = self._render(self._item(TextEffectStack(effects=(
            shadow, overlay,
        ))))
        np.testing.assert_array_equal(
            without_overlay[..., 3], with_overlay[..., 3]
        )

        hollow = HollowEffect()
        hollow_plain = self._render(self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.2,
                position='outside',
                paint=SolidPaint((0, 255, 0)),
            ),
            hollow,
        ))))
        hollow_overlay = self._render(self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.2,
                position='outside',
                paint=SolidPaint((0, 255, 0)),
            ),
            hollow,
            overlay,
        ))))
        np.testing.assert_array_equal(hollow_plain, hollow_overlay)

        inside = self._render(self._item(TextEffectStack(effects=(
            overlay,
            StrokeEffect(
                width=0.35,
                position='inside',
                paint=SolidPaint((0, 255, 0)),
            ),
            ShadowEffect(
                shadow_type='inner',
                offset=(0.12, 0.0),
            ),
        ))))
        visible_inside = inside[..., 3] > 160
        self.assertTrue(np.any(
            (inside[..., 1] > inside[..., 2]) & visible_inside
        ))

    def test_solid_center_stays_native_while_gradient_center_is_generated(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        solid = TextEffectStack(effects=(StrokeEffect(
            width=0.2,
            paint=SolidPaint((0, 0, 255)),
            position='center',
        ),))
        gradient = TextEffectStack(effects=(StrokeEffect(
            width=0.2,
            paint=LinearGradientPaint(),
            position='center',
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
            width=0.28,
            paint=gradient(180),
            position='center',
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

        np.testing.assert_array_equal(default_pixels, outside_pixels)
        self.assertEqual(default.fontformat.text_effects[0].position, 'outside')
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
            ShadowEffect(
                paint=SolidPaint((255, 0, 0)), offset=(0.0, 0.0)
            ),
            ShadowEffect(
                paint=SolidPaint((0, 0, 255)), offset=(0.0, 0.0)
            ),
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
            item.effect_renderer.boundingRect(),
            1.0,
        )
        alpha = pixmap2ndarray(pixmap, keep_alpha=True)[..., 3]

        shifted = alpha[3, 3:5]
        self.assertTrue(np.all(shifted > 0))
        self.assertTrue(np.all(shifted < 255))

    def test_shadow_gradient_color_and_stop_opacity_share_block_coordinates(
        self,
    ):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        source_alpha = np.full((1, 4), 255, dtype=np.uint8)
        rect = QRectF(0, 0, 4, 1)
        shadow = ShadowEffect(
            opacity=0.5,
            offset=(0.0, 0.0),
            paint=LinearGradientPaint(stops=(
                GradientStop(0.0, (255, 0, 0), 0.5),
                GradientStop(1.0, (0, 0, 255), 0.5),
            )),
        )

        with patch.object(
            renderer, 'logical_unpadded_rect', return_value=rect
        ):
            pixmap = renderer._shadow_pixmap(
                source_alpha, shadow, rect, 1.0
            )
        pixels = pixmap2ndarray(pixmap, keep_alpha=True)

        self.assertGreater(pixels[0, 0, 0], pixels[0, 0, 2])
        self.assertGreater(pixels[0, -1, 2], pixels[0, -1, 0])
        np.testing.assert_array_equal(
            pixels[..., 3], np.full((1, 4), 64, dtype=np.uint8)
        )

    def test_fractional_exterior_offsets_keep_horizontal_raster_guard(self):
        for shadow_type in ('drop', 'long'):
            for direction in (-1, 1):
                with self.subTest(
                    shadow_type=shadow_type, direction=direction
                ):
                    item = self._item(TextEffectStack(effects=(
                        ShadowEffect(
                            shadow_type=shadow_type,
                            paint=SolidPaint((0, 0, 255)),
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

        feedback_pixels = self._render(item)
        self.assertGreater(np.count_nonzero(feedback_pixels[..., 3]), 0)
        highlighted_text = view.palette().color(
            QPalette.ColorRole.HighlightedText
        )
        highlighted_rgb = np.array([
            highlighted_text.red(),
            highlighted_text.green(),
            highlighted_text.blue(),
        ])
        hollow_highlighted_text = np.count_nonzero(
            np.all(
                feedback_pixels[..., :3] == highlighted_rgb,
                axis=2,
            )
            & (feedback_pixels[..., 3] > 0)
        )
        plain = self._item(TextEffectStack())
        plain_scene = QGraphicsScene()
        plain_view = QGraphicsView(plain_scene)
        plain_scene.addItem(plain)
        plain_view.show()
        plain.startEdit()
        plain_view.setFocus()
        plain.setFocus()
        self.app.processEvents()
        plain_cursor = plain.textCursor()
        plain_cursor.setPosition(0)
        plain_cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        plain.setTextCursor(plain_cursor)
        plain_pixels = self._render(plain)
        plain_highlighted_text = np.count_nonzero(
            np.all(plain_pixels[..., :3] == highlighted_rgb, axis=2)
            & (plain_pixels[..., 3] > 0)
        )
        self.assertGreater(plain_highlighted_text, 0)
        self.assertEqual(hollow_highlighted_text, plain_highlighted_text)
        plain_view.close()
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
        self.assertEqual(allocate.call_count, 0)
        self.assertGreater(
            np.count_nonzero(
                pixmap2ndarray(clipped, keep_alpha=True)[..., 3]
            ),
            0,
        )

        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        with patch.object(
            item.geometry_controller,
            'paint_deferred_cursor',
            wraps=item.geometry_controller.paint_deferred_cursor,
        ) as paint_cursor:
            self.assertGreater(
                np.count_nonzero(self._render(item)[..., 3]), 0
            )
        self.assertGreater(paint_cursor.call_count, 0)
        self.assertTrue(all(
            call.args[1] is not None
            for call in paint_cursor.call_args_list
        ))

        item.set_export_effect_render(True)
        try:
            self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)
        finally:
            item.set_export_effect_render(False)

    def test_hollow_interaction_defers_caret_and_hides_native_item_frame(self):
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(
                shadow_type='drop',
                offset=(0.5, 0.3),
                blur=0.2,
            ),
            HollowEffect(),
        )))
        self.assertGreater(item.padding(), 0.0)
        item.startEdit()
        renderer = item.effect_renderer
        option = QStyleOptionGraphicsItem()
        option.exposedRect = item.boundingRect()
        option.state = QStyle.StateFlag.State_Selected
        image = QImage(
            420, 260, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        observed = []

        def base_paint(source_painter, source_option, _widget):
            observed.append((
                source_option.state,
                item.layout.defer_cursor_paint,
                source_painter.opacity(),
            ))
            item.layout.deferred_cursor_position = 3

        try:
            with patch.object(
                item.geometry_controller, 'paint_deferred_cursor'
            ) as paint_cursor:
                renderer.paint_item(painter, option, None, base_paint)
        finally:
            painter.end()
            item.endEdit()

        self.assertEqual(observed, [(
            QStyle.StateFlag.State_Selected,
            True,
            0.0,
        )])
        self.assertEqual(item.layout.deferred_cursor_position, 3)
        paint_cursor.assert_called_once()
        self.assertIsNone(paint_cursor.call_args.args[1])

    def test_vertical_hollow_selection_keeps_native_feedback(self):
        item = self._item(
            TextEffectStack(effects=(HollowEffect(),)),
            vertical=True,
        )
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        scene.addItem(item)
        view.show()
        item.startEdit()
        view.setFocus()
        item.setFocus()
        self.app.processEvents()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        pixels = self._render(item)
        selection_background = view.palette().color(
            QPalette.ColorRole.Highlight
        )
        selection_rgb = np.array([
            selection_background.red(),
            selection_background.green(),
            selection_background.blue(),
        ])
        self.assertGreater(
            np.count_nonzero(
                np.all(pixels[..., :3] == selection_rgb, axis=2)
                & (pixels[..., 3] > 0)
            ),
            0,
        )
        self.assertGreater(
            np.count_nonzero(
                (pixels[..., 0] > 180)
                & (pixels[..., 1] < 80)
                & (pixels[..., 2] < 80)
                & (pixels[..., 3] > 0)
            ),
            0,
        )
        view.close()

    def test_deferred_cursor_is_visible_over_neutral_effect_surface(self):
        item = self._item(TextEffectStack(effects=(HollowEffect(),)))
        cursor_rect = QRectF(20, 20, 3, 18)
        image = QImage(
            80, 60, QImage.Format.Format_ARGB32_Premultiplied
        )
        background = QColor(10, 20, 30, 255)
        image.fill(background)
        item.layout.deferred_cursor_position = 0
        painter = QPainter(image)
        try:
            with patch.object(
                item.layout,
                'source_cursor_rect',
                return_value=cursor_rect,
            ):
                item.geometry_controller.paint_deferred_cursor(
                    painter, None, export_render=False
                )
        finally:
            painter.end()

        self.assertNotEqual(image.pixelColor(21, 25), background)
        self.assertEqual(image.pixelColor(5, 5), background)

    def test_effect_padding_does_not_expand_neutral_interaction_shape(self):
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(
                shadow_type='drop',
                offset=(0.5, 0.3),
                blur=0.2,
            ),
            HollowEffect(),
        )))
        source_rect = item.geometry_controller.source_rect()
        logical_rect = item.logical_unpadded_rect()
        padded_corner = QPointF(
            source_rect.left() + 1.0,
            source_rect.bottom() - 1.0,
        )

        self.assertGreater(item.padding(), 0.0)
        self.assertTrue(item.boundingRect().contains(padded_corner))
        self.assertFalse(logical_rect.contains(padded_corner))
        self.assertFalse(item.shape().contains(padded_corner))
        self.assertFalse(item.contains(padded_corner))
        scene = QGraphicsScene()
        scene.addItem(item)
        self.assertNotIn(
            item,
            scene.items(item.mapToScene(padded_corner)),
        )
        self.assertIn(
            item,
            scene.items(item.mapToScene(logical_rect.center())),
        )

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
        overlay = GradientOverlayEffect(opacity=0.7)
        before = TextEffectStack(effects=(
            ShadowEffect(
                offset=(0.1, 0.1),
                blur=0.08,
                paint=LinearGradientPaint(angle=15.0),
            ),
            overlay,
        ))
        after = TextEffectStack(effects=(
            ShadowEffect(
                offset=(0.3, -0.1),
                blur=0.12,
                paint=LinearGradientPaint(angle=75.0),
            ),
            overlay,
        ))
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

    def test_gradient_overlay_preview_promotes_completed_cache(self):
        before = TextEffectStack(effects=(GradientOverlayEffect(),))
        after = TextEffectStack(effects=(GradientOverlayEffect(
            opacity=0.65,
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

    def test_glow_preview_promotes_cache_and_reshape_rebuilds_once(self):
        before = TextEffectStack(effects=(GlowEffect(size=0.08),))
        after = TextEffectStack(effects=(GlowEffect(
            paint=LinearGradientPaint(angle=60.0),
            size=0.16,
            spread=0.04,
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

    def test_glow_allocation_fallback_and_strict_export(self):
        stack = TextEffectStack(effects=(GlowEffect(size=0.2),))
        interactive = self._item(TextEffectStack())
        with patch(
            'ballontranslator.ui.text_engine.effect_renderer.'
            'render_glow_alpha',
            side_effect=BufferError('Glow bridge failure'),
        ):
            interactive.set_text_effects(stack)
            pixels = self._render(interactive)
        self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)

        exported = self._item(stack)
        exported.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'render_glow_alpha',
                side_effect=BufferError('strict Glow bridge failure'),
            ):
                self._render(exported)
            self.assertIsInstance(
                exported.export_effect_error, EffectRasterAllocationError
            )
        finally:
            exported.set_export_effect_render(False)

    def test_gradient_overlay_allocation_fallback_and_strict_export(self):
        stack = TextEffectStack(effects=(GradientOverlayEffect(),))
        item = self._item(stack)
        with patch(
            'ballontranslator.ui.text_engine.effect_renderer.'
            'colorize_effect_paint_rgba',
            side_effect=BufferError('overlay bridge failure'),
        ):
            interactive = self._render(item)
        self.assertGreater(np.count_nonzero(interactive[..., 3]), 0)

        exported = self._item(stack)
        exported.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'colorize_effect_paint_rgba',
                side_effect=BufferError('strict overlay bridge failure'),
            ):
                self._render(exported)
            self.assertIsInstance(
                exported.export_effect_error, EffectRasterAllocationError
            )
        finally:
            exported.set_export_effect_render(False)

    def test_forced_tiles_match_full_typed_effect_surface(self):
        stacks = (
            TextEffectStack(effects=(
                ShadowEffect(
                    offset=(0.18, 0.12),
                    blur=0.08,
                    spread=0.04,
                    paint=LinearGradientPaint(angle=41.0),
                ),
                GlowEffect(
                    paint=LinearGradientPaint(angle=17.0),
                    size=0.08,
                    spread=0.03,
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
                    paint=LinearGradientPaint(angle=203.0),
                ),
                GlowEffect(
                    glow_type='inner',
                    paint=LinearGradientPaint(angle=113.0),
                    size=0.06,
                    spread=0.02,
                ),
                GradientOverlayEffect(
                    opacity=0.7,
                    paint=LinearGradientPaint(
                        stops=(
                            GradientStop(0.0, (40, 80, 220), 0.4),
                            GradientStop(1.0, (220, 40, 80), 1.0),
                        ),
                        angle=71.0,
                        scale=0.9,
                    ),
                ),
            )),
            TextEffectStack(effects=(
                ShadowEffect(
                    shadow_type='long',
                    offset=(0.30, 0.22),
                    paint=LinearGradientPaint(angle=127.0),
                ),
                GlowEffect(size=0.08, spread=0.03),
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
            StrokeEffect(width=0.2, position='center'), HollowEffect()
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
            StrokeEffect(
                width=0.2,
                paint=SolidPaint((0, 0, 255)),
                position='center',
            ),
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
