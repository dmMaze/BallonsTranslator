import hashlib
import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PIL import Image

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QImage,
    QInputMethodEvent,
    QPainter,
    QPalette,
    QPixmap,
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
from ballontranslator.ui.text_engine.effects.paint import (
    colorize_effect_paint_rgba,
    effect_paint_preview_image,
    rasterize_effect_paint,
)
from ballontranslator.ui.text_engine.effects.blend import (
    CUSTOM_BLEND_MODES,
    composite_custom_blend_rgba,
)
from ballontranslator.ui.text_engine.effects.shadow import (
    render_glow_alpha,
    render_shadow_alpha,
)
from ballontranslator.utils.fontformat import (
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    TextFillEffect,
    GradientStop,
    HollowEffect,
    ImageEffect,
    LinearGradientPaint,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TEXT_EFFECT_BLEND_MODES,
    TextEffectStack,
    TexturePaint,
)
from ballontranslator.utils.textblock import TextBlock


class TypedShadowRasterTest(unittest.TestCase):
    def test_custom_blends_use_straight_rgba_source_over(self):
        destination = np.array(
            [[[100, 150, 200, 128]]], dtype=np.uint8
        )
        source = np.array([[[200, 50, 100, 128]]], dtype=np.uint8)
        expected = {
            'linear_burn': [115, 66, 115, 192],
            'darker_color': [167, 83, 133, 192],
            'linear_dodge': [185, 134, 185, 192],
            'lighter_color': [133, 117, 167, 192],
        }
        for blend_mode, pixel in expected.items():
            with self.subTest(blend_mode=blend_mode):
                result = composite_custom_blend_rgba(
                    destination, source, blend_mode
                )
                self.assertEqual(result[0, 0].tolist(), pixel)
                self.assertEqual(result[0, 0, 3], 192)

        transparent = np.zeros((1, 1, 4), dtype=np.uint8)
        for blend_mode in expected:
            with self.subTest(blend_mode=blend_mode, identity='source'):
                np.testing.assert_array_equal(
                    composite_custom_blend_rgba(
                        transparent, source, blend_mode
                    ),
                    source,
                )
            with self.subTest(blend_mode=blend_mode, identity='destination'):
                np.testing.assert_array_equal(
                    composite_custom_blend_rgba(
                        destination, transparent, blend_mode
                    ),
                    destination,
                )

        tie_destination = np.array(
            [[[10, 20, 30, 255]]], dtype=np.uint8
        )
        tie_source = np.array([[[30, 20, 10, 255]]], dtype=np.uint8)
        for blend_mode in ('darker_color', 'lighter_color'):
            with self.subTest(blend_mode=blend_mode, tie=True):
                np.testing.assert_array_equal(
                    composite_custom_blend_rgba(
                        tie_destination, tie_source, blend_mode
                    ),
                    tie_destination,
                )

        adversarial_destination = np.array(
            [[[166, 51, 173, 239]]], dtype=np.uint8
        )
        adversarial_source = np.array(
            [[[201, 229, 136, 244]]], dtype=np.uint8
        )
        self.assertEqual(
            composite_custom_blend_rgba(
                adversarial_destination,
                adversarial_source,
                'lighter_color',
            )[0, 0].tolist(),
            [200, 222, 137, 254],
        )
        half_destination = np.array(
            [[[204, 0, 0, 44]]], dtype=np.uint8
        )
        half_source = np.array([[[29, 0, 0, 44]]], dtype=np.uint8)
        self.assertEqual(
            composite_custom_blend_rgba(
                half_destination, half_source, 'linear_dodge'
            )[0, 0].tolist(),
            [128, 0, 0, 80],
        )

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
            alpha, 'long', 1.0, (23, 11), 9, 9, alpha
        )

        self.assertEqual(long_alpha[5, 5], 0)
        self.assertEqual(long_alpha[16, 28], 255)
        components, _labels = cv2.connectedComponents(
            (long_alpha > 0).astype(np.uint8), connectivity=8
        )
        self.assertEqual(components, 2)

    def test_drop_and_long_are_clipped_outside_canonical_alpha(self):
        alpha = np.zeros((21, 21), dtype=np.uint8)
        alpha[8:13, 8:13] = 255

        for shadow_type in ('drop', 'long'):
            with self.subTest(shadow_type=shadow_type):
                zero = render_shadow_alpha(
                    alpha, shadow_type, 1.0, (0, 0), 0, 0, alpha
                )
                self.assertEqual(np.count_nonzero(zero), 0)

                shifted = render_shadow_alpha(
                    alpha, shadow_type, 1.0, (4, 3), 0, 0, alpha
                )
                self.assertEqual(np.count_nonzero(shifted[alpha > 0]), 0)
                self.assertGreater(np.count_nonzero(shifted), 0)

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

    @staticmethod
    def _all_stroke_silhouette(
        renderer,
        canonical: QPixmap,
        canonical_alpha: np.ndarray,
        bounds: QRectF,
    ) -> QPixmap:
        silhouette = QPixmap(canonical)
        renderer._paint_stroke_silhouette(
            silhouette,
            canonical_alpha,
            tuple(reversed(renderer._active_strokes())),
            bounds,
            1.0,
            {},
        )
        return silhouette

    def test_inner_survives_public_paint_and_vertical_path(self):
        plain = self._item(TextEffectStack())
        inner_stack = TextEffectStack(effects=(ShadowEffect(
            shadow_type='inner',
            angle=0.0,
            distance=0.2,
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

    def test_drop_and_long_preserve_face_and_tile_exact_in_both_layouts(self):
        black = SolidPaint((0, 0, 0))
        for vertical in (False, True):
            for shadow_type in ('drop', 'long'):
                with self.subTest(
                    vertical=vertical, shadow_type=shadow_type
                ):
                    zero_item = self._item(TextEffectStack(effects=(
                        ShadowEffect(
                            shadow_type=shadow_type,
                            angle=0.0,
                            distance=0.0,
                            blur=0.0,
                            spread=0.0,
                            paint=black,
                        ),
                    )), vertical=vertical)
                    zero_renderer = zero_item.effect_renderer
                    zero_bounds = zero_renderer.boundingRect()
                    base = pixmap2ndarray(
                        zero_renderer._render_effect_base(
                            zero_bounds, 1.0
                        ),
                        keep_alpha=True,
                    )
                    zero = pixmap2ndarray(
                        zero_renderer._render_effect_surface(
                            zero_bounds, 1.0
                        ),
                        keep_alpha=True,
                    )
                    opaque_face = base[..., 3] == 255
                    self.assertGreater(np.count_nonzero(opaque_face), 0)
                    np.testing.assert_array_equal(
                        zero[..., :3][opaque_face],
                        base[..., :3][opaque_face],
                    )

                    shifted_item = self._item(TextEffectStack(effects=(
                        ShadowEffect(
                            shadow_type=shadow_type,
                            angle=33.69,
                            distance=0.288,
                            blur=0.0,
                            spread=0.0,
                            paint=black,
                        ),
                    )), vertical=vertical)
                    renderer = shifted_item.effect_renderer
                    bounds = renderer.boundingRect()
                    shifted_base = pixmap2ndarray(
                        renderer._render_effect_base(bounds, 1.0),
                        keep_alpha=True,
                    )
                    full = renderer._render_effect_surface(bounds, 1.0)
                    full_pixels = pixmap2ndarray(full, keep_alpha=True)
                    exterior = (
                        (shifted_base[..., 3] == 0)
                        & (full_pixels[..., 3] > 0)
                    )
                    self.assertGreater(np.count_nonzero(exterior), 0)

                    tiled = renderer._new_effect_pixmap(1.0, bounds)
                    painter = QPainter(tiled)
                    painter.translate(-bounds.topLeft())
                    renderer.tile_cache.clear()
                    try:
                        renderer._draw_tiled_effects(
                            painter,
                            EffectRasterPlan(
                                'tiles', 1.0, 0, 0, 96
                            ),
                            bounds,
                        )
                    finally:
                        painter.end()
                    np.testing.assert_array_equal(
                        full_pixels,
                        pixmap2ndarray(tiled, keep_alpha=True),
                    )

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
            TextFillEffect(
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
        silhouette = self._all_stroke_silhouette(
            renderer, canonical, canonical_alpha, bounds
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
                'ballontranslator.ui.text_engine.effects.renderer.'
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

    def test_shadow_and_glow_follow_global_card_order(self):
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

        def record(
            source_alpha,
            effect,
            surface_rect,
            render_scale,
            canonical_alpha,
        ):
            calls.append(effect)
            return original(
                source_alpha,
                effect,
                surface_rect,
                render_scale,
                canonical_alpha,
            )

        with patch.object(
            renderer, '_generated_effect_pixmap', side_effect=record
        ):
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )

        self.assertEqual(tuple(calls), tuple(reversed(effects)))

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
                    self._all_stroke_silhouette(
                        renderer, canonical, canonical_alpha, bounds
                    )
                )
                with patch.object(
                    renderer, '_glow_pixmap', wraps=renderer._glow_pixmap
                ) as glow_pixmap:
                    renderer._render_pre_mask_effect_surface(bounds, 1.0)
                np.testing.assert_array_equal(
                    glow_pixmap.call_args.args[0], expected
                )

    def test_exterior_effects_use_only_preceding_strokes(self):
        stroke = StrokeEffect(
            width=0.16,
            position='outside',
            paint=SolidPaint((20, 60, 220)),
        )
        for generated in (ShadowEffect(), GlowEffect(size=0.08)):
            for vertical in (False, True):
                for effects, includes_stroke in (
                    ((generated, stroke), True),
                    ((stroke, generated), False),
                ):
                    with self.subTest(
                        effect=generated.effect_type,
                        includes_stroke=includes_stroke,
                        vertical=vertical,
                    ):
                        item = self._item(
                            TextEffectStack(effects=effects),
                            vertical=vertical,
                        )
                        renderer = item.effect_renderer
                        bounds = renderer.boundingRect()
                        canonical = renderer._capture_effect_source(
                            bounds, 1.0
                        )
                        canonical_alpha = renderer._pixmap_alpha(canonical)
                        expected = canonical_alpha
                        if includes_stroke:
                            expected = renderer._pixmap_alpha(
                                self._all_stroke_silhouette(
                                    renderer,
                                    canonical,
                                    canonical_alpha,
                                    bounds,
                                )
                            )
                        with patch.object(
                            renderer,
                            '_generated_effect_pixmap',
                            wraps=renderer._generated_effect_pixmap,
                        ) as render_generated:
                            renderer._render_pre_mask_effect_surface(
                                bounds, 1.0
                            )
                        np.testing.assert_array_equal(
                            render_generated.call_args.args[0], expected
                        )

    def test_exterior_source_grows_silhouette_once_per_preceding_stroke(self):
        first = StrokeEffect(width=0.12, position='outside')
        second = StrokeEffect(width=0.20, position='center')
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(),
            second,
            GlowEffect(size=0.08),
            first,
        )))
        renderer = item.effect_renderer

        with patch.object(
            renderer,
            '_paint_stroke_silhouette',
            wraps=renderer._paint_stroke_silhouette,
        ) as paint_silhouette:
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )

        self.assertEqual(
            [call.args[2] for call in paint_silhouette.call_args_list],
            [(first,), (second,)],
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
        )
        self.assertEqual(recolored[0, 0].tolist(), [0, 100, 240, 127])
        half_alpha = LinearGradientPaint(stops=(
            GradientStop(0.0, (0, 100, 240), 0.5),
            GradientStop(1.0, (0, 100, 240), 0.5),
        ))
        stop_colored = colorize_effect_paint_rgba(
            half_alpha,
            canonical.copy(),
            rect,
            rect,
            1.0,
        )
        self.assertEqual(stop_colored[0, 0].tolist(), [0, 100, 240, 64])

    def test_canonical_alpha_is_extracted_for_consumers_and_center_clip(self):
        text_fill = TextFillEffect(
            paint=self._constant_gradient((20, 60, 220))
        )
        item = self._item(TextEffectStack(effects=(text_fill,)))
        renderer = item.effect_renderer
        renderer._effect_raster_state.effect_source_cache.clear()
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
        renderer._effect_raster_state.effect_source_cache.clear()
        with patch.object(
            renderer, '_pixmap_alpha', wraps=renderer._pixmap_alpha
        ) as pixmap_alpha:
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )
        self.assertEqual(pixmap_alpha.call_count, 1)

        consumers = (
            ShadowEffect(),
            ShadowEffect(shadow_type='inner'),
            StrokeEffect(width=0.2, position='outside'),
        )
        for effect in consumers:
            with self.subTest(effect=effect):
                item = self._item(TextEffectStack(effects=(effect,)))
                renderer = item.effect_renderer
                renderer._effect_raster_state.effect_source_cache.clear()
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

    @staticmethod
    def _solid_pixmap(color) -> QPixmap:
        pixmap = QPixmap(1, 1)
        pixmap.fill(QColor(*color))
        return pixmap

    def test_repeatable_text_fill_order_blend_and_opacity(self):
        bottom = TextFillEffect(
            paint=self._constant_gradient((100, 150, 200))
        )
        top_colors = {
            'normal': (200, 50, 100),
            'darken': (100, 50, 100),
            'multiply': (78, 29, 78),
            'color_burn': (57, 0, 115),
            'linear_burn': (45, 0, 45),
            'darker_color': (200, 50, 100),
            'lighten': (200, 150, 200),
            'screen': (222, 171, 222),
            'color_dodge': (255, 187, 255),
            'linear_dodge': (255, 200, 255),
            'lighter_color': (100, 150, 200),
        }
        self.assertEqual(tuple(top_colors), TEXT_EFFECT_BLEND_MODES)
        item = self._item(TextEffectStack(effects=(
            TextFillEffect(
                paint=self._constant_gradient((200, 50, 100))
            ),
            StrokeEffect(enabled=False),
            bottom,
        )))
        renderer = item.effect_renderer
        rect = QRectF(0, 0, 1, 1)
        canonical = self._solid_pixmap((12, 34, 56, 255))

        self.assertEqual(renderer._active_text_fills(), (
            bottom, item.blk.fontformat.text_effects.effects[0]
        ))
        for blend_mode, expected in top_colors.items():
            with self.subTest(blend_mode=blend_mode):
                top = TextFillEffect(
                    blend_mode=blend_mode,
                    paint=self._constant_gradient((200, 50, 100)),
                )
                group = renderer._text_fill_group_pixmap(
                    canonical, rect, 1.0, (bottom, top)
                )
                pixel = pixmap2ndarray(group, keep_alpha=True)[0, 0]
                np.testing.assert_allclose(pixel[:3], expected, atol=1)
                self.assertEqual(pixel[3], 255)

        custom_fill = TextFillEffect(
            blend_mode='linear_dodge',
            paint=self._constant_gradient((200, 50, 100)),
        )
        with patch.object(
            renderer,
            '_custom_blend_surface_pixmaps',
            wraps=renderer._custom_blend_surface_pixmaps,
        ) as custom_blend:
            renderer._text_fill_group_pixmap(
                canonical, rect, 1.0, (custom_fill,)
            )
            self.assertEqual(custom_blend.call_count, 0)
            renderer._text_fill_group_pixmap(
                canonical, rect, 1.0, (custom_fill, custom_fill)
            )
            self.assertEqual(custom_blend.call_count, 1)

        partial_canonical = self._solid_pixmap((12, 34, 56, 128))
        repeated = renderer._text_fill_group_pixmap(
            partial_canonical, rect, 1.0, (bottom, bottom)
        )
        self.assertEqual(
            pixmap2ndarray(repeated, keep_alpha=True)[0, 0, 3], 128
        )

        half = renderer._text_fill_group_pixmap(
            canonical,
            rect,
            1.0,
            (TextFillEffect(
                opacity=0.5,
                paint=self._constant_gradient((1, 2, 3)),
            ),),
        )
        quarter = renderer._text_fill_group_pixmap(
            canonical,
            rect,
            1.0,
            (TextFillEffect(
                opacity=0.5,
                paint=self._constant_gradient((1, 2, 3), 0.5),
            ),),
        )
        erased = renderer._text_fill_group_pixmap(
            canonical,
            rect,
            1.0,
            (TextFillEffect(opacity=0.0),),
        )
        self.assertEqual(pixmap2ndarray(half, keep_alpha=True)[0, 0, 3], 128)
        self.assertEqual(
            pixmap2ndarray(quarter, keep_alpha=True)[0, 0, 3], 64
        )
        self.assertEqual(
            pixmap2ndarray(erased, keep_alpha=True)[0, 0, 3], 0
        )

    def test_missing_texture_fill_is_skipped_without_hiding_valid_sibling(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        rect = QRectF(0, 0, 1, 1)
        canonical = self._solid_pixmap((12, 34, 56, 255))
        missing = TextFillEffect(paint=TexturePaint(RasterAssetRef(
            'assets/' + 'a' * 64 + '.png', 'missing.png'
        )))
        valid = TextFillEffect(
            paint=self._constant_gradient((20, 60, 220))
        )

        with patch.object(renderer, '_project_raster', return_value=None):
            self.assertIsNone(renderer._text_fill_group_pixmap(
                canonical, rect, 1.0, (missing,)
            ))
            group = renderer._text_fill_group_pixmap(
                canonical, rect, 1.0, (valid, missing)
            )

        pixel = pixmap2ndarray(group, keep_alpha=True)[0, 0]
        np.testing.assert_allclose(pixel, (20, 60, 220, 255), atol=1)

    def test_empty_texture_and_image_do_not_enter_the_raster_path(self):
        stack = TextEffectStack(effects=(TextFillEffect(
            paint=TexturePaint()
        ), ImageEffect()))
        item = self._item(stack)
        renderer = item.effect_renderer

        self.assertFalse(stack.has_active_effects)
        self.assertFalse(renderer.has_raster_effects())
        with patch.object(renderer, '_project_raster') as project_raster:
            pixels = self._render(item)
        project_raster.assert_not_called()
        self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)

        item.set_export_effect_render(True)
        try:
            self._render(item)
            self.assertIsNone(item.export_effect_error)
        finally:
            item.set_export_effect_render(False)

    def test_generated_layers_apply_blend_modes_in_isolated_surface(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        rect = QRectF(0, 0, 1, 1)
        source = self._solid_pixmap((100, 150, 200, 255))
        layer = self._solid_pixmap((200, 50, 100, 255))
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        expected_colors = {
            'normal': (200, 50, 100),
            'darken': (100, 50, 100),
            'multiply': (78, 29, 78),
            'color_burn': (57, 0, 115),
            'linear_burn': (45, 0, 45),
            'darker_color': (200, 50, 100),
            'lighten': (200, 150, 200),
            'screen': (222, 171, 222),
            'color_dodge': (255, 187, 255),
            'linear_dodge': (255, 200, 255),
            'lighter_color': (100, 150, 200),
        }

        for blend_mode, expected in expected_colors.items():
            with self.subTest(blend_mode=blend_mode), patch.object(
                renderer,
                '_cached_effect_source',
                return_value=(source, alpha),
            ), patch.object(
                renderer, '_generated_effect_pixmap', return_value=layer
            ), patch.object(
                renderer,
                '_custom_blend_surface_pixmaps',
                wraps=renderer._custom_blend_surface_pixmaps,
            ) as custom_blend:
                effect = ShadowEffect(blend_mode=blend_mode)
                result = renderer._composite_generated_layer_batch(
                    source, ((0, effect),), rect, 1.0
                )
                self.assertEqual(
                    custom_blend.call_count,
                    int(blend_mode in CUSTOM_BLEND_MODES),
                )
            pixel = pixmap2ndarray(result, keep_alpha=True)[0, 0]
            np.testing.assert_allclose(pixel[:3], expected, atol=1)
            self.assertEqual(pixel[3], 255)

        stroke_item = self._item(TextEffectStack(effects=(StrokeEffect(
            blend_mode='linear_burn', position='center'
        ),)))
        self.assertEqual(stroke_item.effect_renderer._effect_flags(), (
            True, True
        ))
        self.assertFalse(
            stroke_item.effect_renderer._all_strokes_vector_compatible()
        )

    def test_generated_layer_painter_setup_failure_restores_state(self):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        rect = QRectF(0, 0, 1, 1)
        source = self._solid_pixmap((100, 150, 200, 255))
        alpha = np.full((1, 1), 255, dtype=np.uint8)
        previous_error = EffectRasterAllocationError('previous')

        with patch.object(
            renderer,
            '_prepare_effect_surface_painter',
            side_effect=RuntimeError('setup failed'),
        ):
            with self.assertRaises(EffectRasterAllocationError) as direct:
                renderer._begin_effect_layer_painter(source, rect, 1.0)
            self.assertIsInstance(direct.exception.__cause__, RuntimeError)
            probe = QPainter(source)
            self.assertTrue(probe.isActive())
            probe.end()

        def fail_during_guarded_setup(
            _painter: QPainter, _render_scale: float
        ) -> None:
            self.assertTrue(renderer.capturing_surface)
            self.assertIsNone(renderer.surface_raster_error)
            raise RuntimeError('guarded setup failed')

        renderer.capturing_surface = False
        renderer.surface_raster_error = previous_error
        with patch.object(
            renderer,
            '_prepare_effect_surface_painter',
            side_effect=fail_during_guarded_setup,
        ), patch.object(
            renderer,
            '_cached_effect_source',
            return_value=(source, alpha),
        ), self.assertRaises(EffectRasterAllocationError) as composite:
            renderer._composite_generated_layer_batch(
                source, ((0, ShadowEffect()),), rect, 1.0
            )

        self.assertIsInstance(composite.exception.__cause__, RuntimeError)
        self.assertFalse(renderer.capturing_surface)
        self.assertIs(renderer.surface_raster_error, previous_error)

    def test_gradient_replaces_foreground_and_stop_opacity_overwrites_alpha(self):
        plain = self._item(TextEffectStack())
        opaque = self._item(TextEffectStack(effects=(
            TextFillEffect(
                paint=self._constant_gradient((0, 0, 255))
            ),
        )))
        partial = self._item(TextEffectStack(effects=(
            TextFillEffect(
                paint=self._constant_gradient((0, 0, 255), 0.5),
            ),
        )))
        transparent = self._item(TextEffectStack(effects=(
            TextFillEffect(
                paint=self._constant_gradient((0, 0, 255), 0.0),
            ),
        )))

        plain_pixels = self._render(plain)
        opaque_pixels = self._render(opaque)
        partial_pixels = self._render(partial)
        transparent_pixels = self._render(transparent)
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
        expected_partial_alpha = np.rint(
            opaque_pixels[..., 3].astype(np.float32) * 0.5
        ).astype(np.uint8)
        self.assertLessEqual(
            np.max(np.abs(
                partial_pixels[..., 3].astype(np.int16)
                - expected_partial_alpha.astype(np.int16)
            )),
            2,
        )
        self.assertEqual(np.max(transparent_pixels[..., 3]), 0)

    def test_text_fill_phase_hollow_and_shadow_silhouette(self):
        text_fill = TextFillEffect(
            paint=self._constant_gradient((0, 0, 255))
        )
        shadow = ShadowEffect(
            paint=SolidPaint((0, 255, 0)),
            angle=26.565,
            distance=0.224,
            blur=0.05,
        )
        without_fill_item = self._item(TextEffectStack(effects=(shadow,)))
        with_fill_item = self._item(TextEffectStack(effects=(
            shadow, text_fill,
        )))
        without_fill = pixmap2ndarray(
            without_fill_item.effect_renderer._render_pre_mask_effect_surface(
                without_fill_item.effect_renderer.boundingRect(), 1.0
            ),
            keep_alpha=True,
        )
        with_fill = pixmap2ndarray(
            with_fill_item.effect_renderer._render_pre_mask_effect_surface(
                with_fill_item.effect_renderer.boundingRect(), 1.0
            ),
            keep_alpha=True,
        )
        np.testing.assert_array_equal(
            without_fill[..., 3], with_fill[..., 3]
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
        hollow_fill = self._render(self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.2,
                position='outside',
                paint=SolidPaint((0, 255, 0)),
            ),
            hollow,
            text_fill,
        ))))
        np.testing.assert_array_equal(hollow_plain, hollow_fill)

        inside = self._render(self._item(TextEffectStack(effects=(
            text_fill,
            StrokeEffect(
                width=0.35,
                position='inside',
                paint=SolidPaint((0, 255, 0)),
            ),
            ShadowEffect(
                shadow_type='inner',
                angle=0.0,
                distance=0.12,
            ),
        ))))
        visible_inside = inside[..., 3] > 160
        self.assertTrue(np.any(
            (inside[..., 1] > inside[..., 2]) & visible_inside
        ))

    def test_completed_surface_bands_solid_and_gradient_center_strokes(self):
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
            solid_calls = band.call_count
            self.assertGreater(solid_calls, 0)
            self.assertEqual(renderer._effect_flags(), (True, False))
            item.set_text_effects(gradient)
            self.assertGreater(band.call_count, solid_calls)
            self.assertEqual(renderer._effect_flags(), (True, True))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

    def test_noop_filter_preserves_public_center_stroke_appearance(self):
        noop = FilterEffect('builtin:noise', params={
            'amount': 0.0, 'mode': 'monochrome', 'seed': 1,
        })
        paints = (
            SolidPaint((20, 60, 220)),
            LinearGradientPaint(stops=(
                GradientStop(0.0, (20, 60, 220)),
                GradientStop(1.0, (220, 60, 20)),
            )),
        )
        for vertical in (False, True):
            for paint in paints:
                stroke = StrokeEffect(
                    width=0.20, position='center', paint=paint
                )
                baseline = self._render(self._item(
                    TextEffectStack(effects=(stroke,)), vertical=vertical
                ))
                for effects in ((noop, stroke), (stroke, noop)):
                    with self.subTest(
                        vertical=vertical,
                        paint=type(paint).__name__,
                        effects=effects,
                    ):
                        filtered = self._render(self._item(
                            TextEffectStack(effects=effects),
                            vertical=vertical,
                        ))
                        delta = np.abs(
                            filtered.astype(np.int16)
                            - baseline.astype(np.int16)
                        )
                        changed = np.count_nonzero(
                            np.any(delta, axis=2)
                        )
                        # The direct Qt face-over-outline and isolated
                        # surface bridge round antialias coverage differently.
                        # Keep this below the former 200/1.1k regression.
                        self.assertLessEqual(
                            int(delta.max()), 100,
                            (int(delta.max()), int(changed)),
                        )
                        self.assertLessEqual(changed, 1000)

    def test_center_coverage_cache_is_reused_when_hollow_changes(self):
        stroke = StrokeEffect(
            width=0.20,
            position='center',
            paint=SolidPaint((20, 60, 220)),
        )
        item = self._item(TextEffectStack(effects=(stroke,)))
        renderer = item.effect_renderer
        renderer._render_effect_surface(renderer.boundingRect(), 0.5)

        with patch.object(
            renderer, 'paint_stroke', wraps=renderer.paint_stroke
        ) as paint_stroke:
            item.set_text_effects(TextEffectStack(effects=(
                stroke, HollowEffect()
            )), preview=True)
        self.assertEqual(paint_stroke.call_count, 0)

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
                ShadowEffect(angle=26.565, distance=0.224),
            )))
            item.setPlainText('\N{FULL BLOCK}' * 3)
            item.layout.reLayoutEverything()
            renderer = item.effect_renderer
            bounds = renderer.boundingRect()
            canonical = renderer._capture_effect_source(bounds, 1.0)
            canonical_alpha = renderer._pixmap_alpha(canonical)
            silhouette = self._all_stroke_silhouette(
                renderer, canonical, canonical_alpha, bounds
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
        for item in (center, inside, outside):
            self.assertEqual(item.effect_renderer._stroke_width(), 0.24)
        self.assertAlmostEqual(
            outside.effect_renderer._conservative_effect_padding(),
            center.effect_renderer._conservative_effect_padding(),
        )
        # Committed padding is rounded outward to 1/64 layout units.
        self.assertAlmostEqual(
            outside.padding(),
            center.padding(),
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
        np.testing.assert_array_equal(outside_blue, center_blue)

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
                silhouette = self._all_stroke_silhouette(
                    renderer, canonical, canonical_alpha, bounds
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

    def test_first_shadow_card_paints_on_top_in_shared_exterior_pixels(self):
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(
                paint=SolidPaint((255, 0, 0)),
                angle=34.0,
                distance=0.216,
            ),
            ShadowEffect(
                paint=SolidPaint((0, 0, 255)),
                angle=34.0,
                distance=0.216,
            ),
        )))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        base = pixmap2ndarray(
            renderer._render_effect_base(bounds, 1.0), keep_alpha=True
        )
        pixels = pixmap2ndarray(
            renderer._render_effect_surface(bounds, 1.0), keep_alpha=True
        )
        exterior = (base[..., 3] == 0) & (pixels[..., 3] > 200)

        self.assertGreater(np.count_nonzero(exterior), 0)
        self.assertTrue(np.all(
            pixels[..., 0][exterior] > pixels[..., 2][exterior]
        ))

    def test_drop_preserves_fractional_relative_offset(self):
        item = self._item(TextEffectStack())
        source_alpha = np.zeros((7, 7), dtype=np.uint8)
        source_alpha[3, 3] = 255

        pixmap = item.effect_renderer._shadow_pixmap(
            source_alpha,
            ShadowEffect(angle=0.0, distance=0.01),
            item.effect_renderer.boundingRect(),
            1.0,
        )
        alpha = pixmap2ndarray(pixmap, keep_alpha=True)[..., 3]

        self.assertEqual(alpha[3, 3], 0)
        self.assertGreater(alpha[3, 4], 0)
        self.assertLess(alpha[3, 4], 255)

    def test_shadow_gradient_color_and_stop_opacity_share_block_coordinates(
        self,
    ):
        item = self._item(TextEffectStack())
        renderer = item.effect_renderer
        source_alpha = np.zeros((1, 6), dtype=np.uint8)
        source_alpha[0, :2] = 255
        rect = QRectF(0, 0, 6, 1)
        shadow = ShadowEffect(
            opacity=0.5,
            angle=0.0,
            distance=0.0,
            paint=LinearGradientPaint(stops=(
                GradientStop(0.0, (255, 0, 0), 0.5),
                GradientStop(1.0, (0, 0, 255), 0.5),
            )),
        )

        with patch.object(
            renderer, 'logical_unpadded_rect', return_value=rect
        ), patch.object(
            renderer, '_shadow_metrics', return_value=(0.0, 0.0, 2.0, 0.0)
        ):
            pixmap = renderer._shadow_pixmap(
                source_alpha, shadow, rect, 1.0
            )
        pixels = pixmap2ndarray(pixmap, keep_alpha=True)

        self.assertGreater(pixels[0, 2, 0], pixels[0, 2, 2])
        self.assertGreater(pixels[0, 3, 2], pixels[0, 3, 0])
        np.testing.assert_array_equal(
            pixels[..., 3],
            np.array([[0, 0, 64, 64, 0, 0]], dtype=np.uint8),
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
                            angle=0.0 if direction > 0 else 180.0,
                            distance=0.01,
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
                angle=0.0,
                distance=0.45,
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
                angle=31.0,
                distance=0.583,
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
        self.assertEqual(
            np.count_nonzero(
                (pixels[..., 0] > 180)
                & (pixels[..., 1] < 80)
                & (pixels[..., 2] < 80)
                & (pixels[..., 3] > 0)
            ),
            0,
        )
        view.close()

    def test_custom_glyph_editing_keeps_completed_effect_surface(self):
        gradient = TextFillEffect(
            paint=self._constant_gradient((0, 0, 255))
        )
        cases = (
            ('vertical-gradient', True, TextTransformStack(), (gradient,)),
            (
                'vertical-hollow',
                True,
                TextTransformStack(),
                (
                    StrokeEffect(width=0.2, position='outside'),
                    HollowEffect(),
                ),
            ),
            (
                'horizontal-glyph-slant',
                False,
                TextTransformStack(glyph_slant_angle=18.0),
                (gradient,),
            ),
        )
        for name, vertical, transform, effects in cases:
            with self.subTest(name=name):
                item = self._item(
                    TextEffectStack(effects=effects), vertical=vertical
                )
                item.set_text_transform(transform)
                scene = QGraphicsScene()
                view = QGraphicsView(scene)
                scene.addItem(item)
                view.show()
                view.setFocus()
                self.app.processEvents()
                settled = self._render(item)

                item.startEdit()
                view.setFocus()
                item.setFocus()
                self.app.processEvents()
                renderer = item.effect_renderer
                with patch.object(
                    item.geometry_controller, 'paint_deferred_cursor'
                ), patch.object(
                    renderer,
                    '_paint_live_layout',
                    wraps=renderer._paint_live_layout,
                ) as replay_layout:
                    editing = self._render(item)

                np.testing.assert_array_equal(editing, settled)
                replay_layout.assert_not_called()
                view.close()

    def test_feedback_context_does_not_restore_unspecified_foreground(self):
        item = self._item(TextEffectStack(effects=(HollowEffect(),)))
        cursor = QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        selection = QAbstractTextDocumentLayout.Selection()
        selection.cursor = cursor
        selection.format = QTextCharFormat()
        selection.format.setBackground(QColor('#3f51b5'))
        context = QAbstractTextDocumentLayout.PaintContext()
        context.selections = [selection]

        feedback = item.effect_renderer._editing_feedback_context(context)

        self.assertEqual(len(feedback.selections), 2)
        replay_format = feedback.selections[1].format
        self.assertEqual(replay_format.foreground().color().alpha(), 0)
        self.assertEqual(
            replay_format.background().color(), QColor('#3f51b5')
        )
        self.assertEqual(
            replay_format.textOutline().style(), Qt.PenStyle.NoPen
        )
        self.assertEqual(replay_format.underlineColor().alpha(), 0)

    def test_vertical_glyph_slant_selection_stays_above_gradient(self):
        item = self._item(
            TextEffectStack(effects=(TextFillEffect(
                paint=self._constant_gradient((0, 0, 255))
            ),)),
            vertical=True,
        )
        item.set_text_transform(TextTransformStack(glyph_slant_angle=15.0))
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
        selection = view.palette().color(QPalette.ColorRole.Highlight)
        selection_rgb = np.array([
            selection.red(), selection.green(), selection.blue()
        ])
        self.assertGreater(np.count_nonzero(
            np.all(pixels[..., :3] == selection_rgb, axis=2)
            & (pixels[..., 3] > 0)
        ), 0)
        self.assertEqual(np.count_nonzero(
            (pixels[..., 0] > 180)
            & (pixels[..., 1] < 80)
            & (pixels[..., 2] < 80)
            & (pixels[..., 3] > 0)
        ), 0)
        view.close()

    def test_vertical_gradient_keeps_transient_ime_preedit_visible(self):
        block = TextBlock([0, 0, 220, 260])
        block._bounding_rect = [0, 0, 220, 260]
        block.translation = '字'
        block.vertical = True
        block.fontformat.frgb = [240, 20, 20]
        block.fontformat.text_effects = TextEffectStack(effects=(
            TextFillEffect(
                paint=self._constant_gradient((0, 0, 255))
            ),
        ))
        item = TextBlkItem(block, 1)
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        scene.addItem(item)
        view.show()
        item.startEdit()
        view.setFocus()
        item.setFocus()
        self.app.processEvents()

        with patch.object(
            item.geometry_controller, 'paint_deferred_cursor'
        ):
            before = self._render(item)
            item.inputMethodEvent(QInputMethodEvent('かな', []))
            self.app.processEvents()
            after = self._render(item)

        before_red = np.count_nonzero(
            (before[..., 0] > 180)
            & (before[..., 1] < 80)
            & (before[..., 2] < 80)
            & (before[..., 3] > 0)
        )
        after_red = np.count_nonzero(
            (after[..., 0] > 180)
            & (after[..., 1] < 80)
            & (after[..., 2] < 80)
            & (after[..., 3] > 0)
        )
        self.assertEqual(before_red, 0)
        self.assertGreater(after_red, 0)
        self.assertGreater(np.count_nonzero(
            (after[..., 2] > 180) & (after[..., 3] > 0)
        ), 0)
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
                angle=31.0,
                distance=0.583,
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

    def test_effect_variants_rerender_preview_at_persistent_quality(self):
        cases = (
            (
                TextEffectStack(effects=(ShadowEffect(blur=0.08),)),
                TextEffectStack(effects=(ShadowEffect(
                    angle=341.565, distance=0.316, blur=0.12
                ),)),
            ),
            (
                TextEffectStack(effects=(StrokeEffect(
                    width=0.12, paint=LinearGradientPaint()
                ),)),
                TextEffectStack(effects=(StrokeEffect(
                    width=0.18,
                    paint=LinearGradientPaint(angle=90.0, scale=1.5),
                ),)),
            ),
            (
                TextEffectStack(effects=(TextFillEffect(),)),
                TextEffectStack(effects=(TextFillEffect(
                    paint=LinearGradientPaint(angle=90.0, scale=1.5),
                ),)),
            ),
            (
                TextEffectStack(effects=(GlowEffect(size=0.08),)),
                TextEffectStack(effects=(GlowEffect(
                    paint=LinearGradientPaint(angle=60.0),
                    size=0.16,
                    spread=0.04,
                ),)),
            ),
        )
        for before, after in cases:
            with self.subTest(effect=after.effects[0].effect_type):
                item = self._item(before)
                renderer = item.effect_renderer
                renderer.set_faster_preview(True)
                with patch.object(
                    renderer,
                    '_render_effect_surface',
                    wraps=renderer._render_effect_surface,
                ) as render:
                    item.set_text_effects(after, preview=True)
                    scratch = renderer._preview_effect_raster_state
                    self.assertEqual(render.call_count, 1)
                    self.assertEqual(
                        scratch.background_pixmap_scale, 0.5
                    )
                    item.set_text_effects(after)
                    renderer.repaint_background()
                    self.assertEqual(render.call_count, 2)
                    self.assertIsNot(
                        renderer._effect_raster_state, scratch
                    )

    def test_paint_previews_reuse_canonical_source_and_match_cold_output(self):
        stroke = StrokeEffect(
            width=0.18,
            position='inside',
            paint=LinearGradientPaint(),
        )
        text_fill = TextFillEffect()
        before = TextEffectStack(effects=(stroke, text_fill))
        previews = tuple(
            TextEffectStack(effects=(
                replace(
                    stroke,
                    opacity=opacity,
                    paint=(
                        SolidPaint(first)
                        if angle == 75.0
                        else LinearGradientPaint(
                            stops=(
                                GradientStop(0.0, first, 0.4),
                                GradientStop(1.0, second, 0.9),
                            ),
                            angle=angle,
                        )
                    ),
                ),
                replace(
                    text_fill,
                    paint=LinearGradientPaint(
                        stops=(
                            GradientStop(0.0, second, 0.7 * opacity),
                            GradientStop(1.0, first, opacity),
                        ),
                        angle=angle + 45.0,
                    ),
                ),
            ))
            for angle, opacity, first, second in (
                (15.0, 0.4, (220, 30, 20), (20, 60, 220)),
                (75.0, 0.7, (30, 200, 80), (180, 40, 160)),
                (135.0, 0.501, (240, 180, 20), (20, 160, 220)),
            )
        )
        item = self._item(before)
        renderer = item.effect_renderer
        renderer._effect_raster_state.effect_source_cache.clear()
        committed_coverage_keys = tuple(
            renderer._effect_raster_state.positioned_stroke_coverage_cache
        )
        self.assertTrue(committed_coverage_keys)

        with patch.object(
            renderer,
            '_capture_effect_source',
            wraps=renderer._capture_effect_source,
        ) as capture:
            with patch.object(
                renderer,
                '_pixmap_alpha',
                wraps=renderer._pixmap_alpha,
            ) as alpha:
                with patch.object(
                    renderer,
                    '_paint_cloned_document_stroke',
                    wraps=renderer._paint_cloned_document_stroke,
                ) as native_outline:
                    for preview in previews:
                        item.set_text_effects(preview, preview=True)
                        renderer.repaint_background()

        self.assertEqual(capture.call_count, 1)
        self.assertEqual(alpha.call_count, 1)
        self.assertEqual(native_outline.call_count, 0)
        scratch = renderer._preview_effect_raster_state
        self.assertEqual(len(scratch.effect_source_cache), 1)
        scratch_coverage_keys = tuple(
            scratch.positioned_stroke_coverage_cache
        )
        self.assertLessEqual(len(scratch_coverage_keys), 2)
        self.assertTrue(
            set(scratch_coverage_keys).intersection(
                committed_coverage_keys
            )
        )
        hot = pixmap2ndarray(scratch.background_pixmap, keep_alpha=True)

        cold = self._item(before)
        cold_state = cold.effect_renderer._effect_raster_state
        cold_state.effect_source_cache.clear()
        cold_state.positioned_stroke_coverage_cache.clear()
        cold.set_text_effects(previews[-1], preview=True)
        cold.effect_renderer.repaint_background()
        cold_pixels = pixmap2ndarray(
            cold.effect_renderer.background_pixmap, keep_alpha=True
        )
        np.testing.assert_array_equal(hot, cold_pixels)

    def test_effect_source_cache_key_is_bounded_and_lifecycle_owned(self):
        stack = TextEffectStack(effects=(StrokeEffect(
            width=0.18,
            position='inside',
            paint=LinearGradientPaint(),
        ),))
        item = self._item(stack)
        renderer = item.effect_renderer
        state = renderer._effect_raster_state
        state.effect_source_cache.clear()
        bounds = renderer.boundingRect()

        with patch.object(
            renderer,
            '_capture_effect_source',
            wraps=renderer._capture_effect_source,
        ) as capture:
            renderer._cached_effect_source(
                bounds, 0.5, needs_alpha=True
            )
            renderer._cached_effect_source(
                bounds, 0.5, needs_alpha=True
            )
            self.assertEqual(capture.call_count, 1)

            renderer._cached_effect_source(
                bounds, 1.0, needs_alpha=True
            )
            self.assertEqual(capture.call_count, 2)
            renderer._cached_effect_source(
                bounds.adjusted(0.0, 0.0, 1.0, 0.0),
                0.5,
                needs_alpha=True,
            )
            self.assertEqual(capture.call_count, 3)
            self.assertEqual(len(state.effect_source_cache), 2)

            item.repaint_on_changed = False
            item.setPlainText('Changed canonical source')
            changed_bounds = renderer.boundingRect()
            renderer._cached_effect_source(
                changed_bounds, 0.5, needs_alpha=True
            )
            self.assertEqual(capture.call_count, 4)
            self.assertLessEqual(len(state.effect_source_cache), 2)

            geometry_rect = QRectF(*item.blk.bounding_rect())
            geometry_rect.setWidth(geometry_rect.width() + 24.0)
            item.setRect(geometry_rect, repaint=False)
            changed_bounds = renderer.boundingRect()
            renderer._cached_effect_source(
                changed_bounds, 0.5, needs_alpha=True
            )
            self.assertEqual(capture.call_count, 5)
            self.assertLessEqual(len(state.effect_source_cache), 2)

        committed_keys = tuple(state.effect_source_cache)
        renderer.set_export_effect_render(True)
        try:
            exported = renderer._export_effect_raster_state
            self.assertEqual(exported.effect_source_cache, {})
            renderer._cached_effect_source(
                changed_bounds, 0.5, needs_alpha=True
            )
            self.assertEqual(len(exported.effect_source_cache), 1)
            self.assertEqual(tuple(state.effect_source_cache), committed_keys)
        finally:
            renderer.set_export_effect_render(False)

        item.startReshape()
        self.assertEqual(state.effect_source_cache, {})
        item.endReshape()
        renderer.release_caches()
        self.assertIsNone(renderer._effect_raster_state)
        self.assertIsNone(renderer._preview_effect_raster_state)
        self.assertIsNone(renderer._export_effect_raster_state)

    def test_positioned_stroke_coverage_cache_misses_and_lifecycle(self):
        stroke = StrokeEffect(
            width=0.18,
            position='inside',
            paint=LinearGradientPaint(),
        )
        item = self._item(TextEffectStack(effects=(stroke,)))
        renderer = item.effect_renderer
        state = renderer._effect_raster_state
        state.positioned_stroke_coverage_cache.clear()

        def coverage(
            current: StrokeEffect,
            rect: QRectF,
            scale: float,
        ) -> np.ndarray:
            _source, alpha = renderer._cached_effect_source(
                rect,
                scale,
                needs_alpha=current.position != 'center',
            )
            return renderer._positioned_stroke_coverage(
                rect, scale, current, alpha
            )

        bounds = renderer.boundingRect()
        with patch.object(
            renderer,
            '_paint_cloned_document_stroke',
            wraps=renderer._paint_cloned_document_stroke,
        ) as native_outline:
            first = coverage(stroke, bounds, 0.5)
            self.assertIs(first, coverage(
                replace(
                    stroke,
                    opacity=0.37,
                    paint=SolidPaint((20, 80, 220)),
                ),
                bounds,
                0.5,
            ))
            self.assertEqual(native_outline.call_count, 1)
            self.assertFalse(first.flags.writeable)
            with self.assertRaises(ValueError):
                first[0, 0] = 255

            coverage(replace(stroke, width=0.24), bounds, 0.5)
            self.assertEqual(native_outline.call_count, 2)
            coverage(replace(stroke, position='outside'), bounds, 0.5)
            self.assertEqual(native_outline.call_count, 3)
            coverage(stroke, bounds, 1.0)
            self.assertEqual(native_outline.call_count, 4)
            adjusted = bounds.adjusted(0.0, 0.0, 1.0, 0.0)
            coverage(stroke, adjusted, 0.5)
            self.assertEqual(native_outline.call_count, 5)
            self.assertEqual(
                len(state.positioned_stroke_coverage_cache), 2
            )

            item.repaint_on_changed = False
            item.setPlainText('Changed positioned Stroke source')
            changed_bounds = renderer.boundingRect()
            coverage(stroke, changed_bounds, 0.5)
            self.assertEqual(native_outline.call_count, 6)
            self.assertLessEqual(
                len(state.positioned_stroke_coverage_cache), 2
            )

            committed_keys = tuple(
                state.positioned_stroke_coverage_cache
            )
            renderer.set_export_effect_render(True)
            try:
                exported = renderer._export_effect_raster_state
                self.assertEqual(
                    exported.positioned_stroke_coverage_cache, {}
                )
                coverage(stroke, changed_bounds, 0.5)
                self.assertEqual(native_outline.call_count, 7)
                self.assertEqual(
                    len(exported.positioned_stroke_coverage_cache), 1
                )
                self.assertEqual(
                    tuple(state.positioned_stroke_coverage_cache),
                    committed_keys,
                )
            finally:
                renderer.set_export_effect_render(False)

        item.startReshape()
        self.assertEqual(state.positioned_stroke_coverage_cache, {})
        item.endReshape()
        renderer.release_caches()
        self.assertIsNone(renderer._effect_raster_state)
        self.assertIsNone(renderer._preview_effect_raster_state)
        self.assertIsNone(renderer._export_effect_raster_state)

    def test_cached_coverage_handles_compound_hollow_output(self):
        stroke = StrokeEffect(
            width=0.18,
            position='outside',
            opacity=0.501,
            paint=LinearGradientPaint(angle=37.0),
        )
        item = self._item(TextEffectStack(effects=(
            ShadowEffect(blur=0.08),
            GlowEffect(size=0.07),
            stroke,
            HollowEffect(),
        )))
        renderer = item.effect_renderer
        renderer._effect_raster_state.positioned_stroke_coverage_cache.clear()
        bounds = renderer.boundingRect()
        with patch.object(
            renderer,
            '_paint_cloned_document_stroke',
            wraps=renderer._paint_cloned_document_stroke,
        ) as native_outline:
            cold = renderer._render_pre_mask_effect_surface(bounds, 1.0)
            hot = renderer._render_pre_mask_effect_surface(bounds, 1.0)
        self.assertEqual(native_outline.call_count, 1)
        np.testing.assert_array_equal(
            pixmap2ndarray(cold, keep_alpha=True),
            pixmap2ndarray(hot, keep_alpha=True),
        )

    def test_positioned_stroke_band_is_reused_within_one_composite(self):
        generated_strokes = (
            StrokeEffect(width=0.18, position='outside'),
            StrokeEffect(width=0.18, position='center'),
            StrokeEffect(
                width=0.18,
                position='center',
                paint=LinearGradientPaint(angle=35.0),
            ),
            StrokeEffect(width=0.18, position='inside'),
        )
        exterior_effects = (
            ShadowEffect(blur=0.08),
            GlowEffect(size=0.08),
        )
        for exterior in exterior_effects:
            for stroke in generated_strokes:
                for hollow in (False, True):
                    with self.subTest(
                        exterior=exterior.effect_type,
                        position=stroke.position,
                        hollow=hollow,
                    ):
                        effects = [exterior, stroke]
                        if hollow:
                            effects.append(HollowEffect(enabled=True))
                        item = self._item(TextEffectStack(
                            effects=tuple(effects)
                        ))
                        renderer = item.effect_renderer
                        with patch.object(
                            renderer,
                            '_positioned_stroke_band',
                            wraps=renderer._positioned_stroke_band,
                        ) as band:
                            renderer._render_pre_mask_effect_surface(
                                renderer.boundingRect(), 1.0
                            )
                        self.assertEqual(band.call_count, 1)

        item = self._item(TextEffectStack(effects=(
            ShadowEffect(blur=0.08),
            StrokeEffect(width=0.18, position='center'),
        )))
        renderer = item.effect_renderer
        with patch.object(
            renderer,
            '_positioned_stroke_band',
            wraps=renderer._positioned_stroke_band,
        ) as band:
            with patch.object(
                renderer,
                '_positioned_stroke_coverage',
                wraps=renderer._positioned_stroke_coverage,
            ) as coverage:
                renderer._render_pre_mask_effect_surface(
                    renderer.boundingRect(), 1.0
                )
        self.assertEqual(band.call_count, 1)
        self.assertEqual(coverage.call_count, 1)

    def test_glow_allocation_fallback_and_strict_export(self):
        stack = TextEffectStack(effects=(GlowEffect(size=0.2),))
        interactive = self._item(TextEffectStack())
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
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
                'ballontranslator.ui.text_engine.effects.renderer.'
                'render_glow_alpha',
                side_effect=BufferError('strict Glow bridge failure'),
            ):
                self._render(exported)
            self.assertIsInstance(
                exported.export_effect_error, EffectRasterAllocationError
            )
        finally:
            exported.set_export_effect_render(False)

    def test_text_fill_allocation_fallback_and_strict_export(self):
        stack = TextEffectStack(effects=(TextFillEffect(),))
        item = self._item(stack)
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'colorize_effect_paint_rgba',
            side_effect=BufferError('Text Fill bridge failure'),
        ):
            interactive = self._render(item)
        self.assertGreater(np.count_nonzero(interactive[..., 3]), 0)

        exported = self._item(stack)
        exported.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effects.renderer.'
                'colorize_effect_paint_rgba',
                side_effect=BufferError('strict Text Fill bridge failure'),
            ):
                self._render(exported)
            self.assertIsInstance(
                exported.export_effect_error, EffectRasterAllocationError
            )
        finally:
            exported.set_export_effect_render(False)

    def test_texture_fill_uses_project_asset_and_missing_export_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'blue.png')
            Image.fromarray(np.full(
                (2, 2, 4), (20, 60, 230, 255), dtype=np.uint8
            ), 'RGBA').save(source_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)
            stack = TextEffectStack(effects=(TextFillEffect(
                paint=TexturePaint(asset, mapping='tile', scale=1.5)
            ),))

            for vertical, transformed in (
                (False, False), (True, False), (False, True)
            ):
                with self.subTest(vertical=vertical, transformed=transformed):
                    item = self._item(stack, vertical=vertical)
                    if transformed:
                        item.set_text_transform(
                            TextTransformStack((SineTextTransform(),))
                        )
                    scene = QGraphicsScene()
                    scene.imgtrans_proj = project
                    scene.addItem(item)
                    item.effect_renderer.project_assets_changed()
                    pixels = self._render(item)
                    visible = pixels[..., 3] > 160
                    self.assertTrue(np.any(visible))
                    self.assertGreater(
                        np.mean(pixels[..., 2][visible]),
                        np.mean(pixels[..., 0][visible]),
                    )

            missing = RasterAssetRef(
                'assets/' + 'b' * 64 + '.png', 'missing.png'
            )
            item = self._item(TextEffectStack(effects=(TextFillEffect(
                paint=TexturePaint(missing)
            ),)))
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            scene.addItem(item)
            item.effect_renderer.project_assets_changed()
            interactive = self._render(item)
            visible = interactive[..., 3] > 160
            self.assertTrue(np.any(visible))
            self.assertGreater(
                np.mean(interactive[..., 0][visible]),
                np.mean(interactive[..., 2][visible]),
            )

            item.set_export_effect_render(True)
            try:
                self._render(item)
                self.assertIsInstance(
                    item.export_effect_error, EffectRasterAllocationError
                )
            finally:
                item.set_export_effect_render(False)

    def test_cached_texture_strict_export_rechecks_deleted_and_corrupt_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'blue.png')
            Image.fromarray(np.full(
                (3, 4, 4), (20, 60, 230, 255), dtype=np.uint8
            ), 'RGBA').save(source_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)
            item = self._item(TextEffectStack(effects=(TextFillEffect(
                paint=TexturePaint(asset)
            ),)))
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            scene.addItem(item)
            item.effect_renderer.project_assets_changed()

            self._render(item)
            installed_path = project.resolve_raster_asset(asset)
            os.unlink(installed_path)
            item.effect_renderer.project_assets_changed()
            bypassed = self._render(item)
            visible = bypassed[..., 3] > 160
            self.assertGreater(
                np.mean(bypassed[..., 0][visible]),
                np.mean(bypassed[..., 2][visible]),
            )
            item.set_export_effect_render(True)
            try:
                self._render(item)
                self.assertIsInstance(
                    item.export_effect_error, EffectRasterAllocationError
                )
            finally:
                item.set_export_effect_render(False)

            restored = project.import_raster_asset(source_path)
            self.assertEqual(restored, asset)
            item.effect_renderer.project_assets_changed()
            self._render(item)
            with open(project.resolve_raster_asset(asset), 'wb') as installed:
                installed.write(b'corrupt')
            item.set_export_effect_render(True)
            try:
                self._render(item)
                self.assertIsInstance(
                    item.export_effect_error, EffectRasterAllocationError
                )
            finally:
                item.set_export_effect_render(False)

    def test_missing_texture_recovers_after_restore_and_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'recover.png')
            Image.fromarray(np.full(
                (3, 4, 4), (20, 190, 230, 255), dtype=np.uint8
            ), 'RGBA').save(source_path)
            with open(source_path, 'rb') as source:
                digest = hashlib.sha256(source.read()).hexdigest()
            asset = RasterAssetRef(
                f'assets/{digest}.png', 'recover.png'
            )
            project = ProjImgTrans()
            project.directory = directory
            item = self._item(TextEffectStack(effects=(TextFillEffect(
                paint=TexturePaint(asset)
            ),)))
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            scene.addItem(item)
            item.effect_renderer.project_assets_changed()

            missing = self._render(item)
            visible = missing[..., 3] > 160
            self.assertGreater(
                np.mean(missing[..., 0][visible]),
                np.mean(missing[..., 2][visible]),
            )

            os.makedirs(project.assets_dir())
            shutil.copyfile(
                source_path,
                os.path.join(project.assets_dir(), f'{digest}.png'),
            )
            item.effect_renderer.project_assets_changed()
            recovered = self._render(item)
            visible = recovered[..., 3] > 160
            self.assertGreater(
                np.mean(recovered[..., 2][visible]),
                np.mean(recovered[..., 0][visible]),
            )

    def test_forced_tiles_match_full_texture_surface_and_reuse_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, 'pattern.png')
            pattern = np.array(
                [
                    [[230, 20, 40, 255], [20, 200, 60, 255]],
                    [[40, 60, 230, 255], [210, 190, 20, 255]],
                ],
                dtype=np.uint8,
            )
            Image.fromarray(pattern, 'RGBA').save(source_path)
            project = ProjImgTrans()
            project.directory = directory
            asset = project.import_raster_asset(source_path)
            stack = TextEffectStack(effects=(TextFillEffect(
                paint=TexturePaint(asset, mapping='tile', scale=1.5)
            ),))
            item = self._item(stack)
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            scene.addItem(item)
            renderer = item.effect_renderer
            renderer.project_assets_changed()
            bounds = renderer.boundingRect()

            project._raster_asset_cache.clear()
            renderer.release_caches()
            with patch.object(
                project,
                '_decode_raster_asset_snapshot',
                wraps=project._decode_raster_asset_snapshot,
            ) as decode, patch.object(
                renderer,
                '_capture_effect_source',
                wraps=renderer._capture_effect_source,
            ) as capture:
                full = renderer._render_effect_surface(bounds, 1.0)
                changed = replace(
                    stack.effects[0].paint, mapping='crop', scale=0.75
                )
                item.set_text_effects(TextEffectStack(effects=(
                    replace(stack.effects[0], paint=changed),
                )))
                renderer._render_effect_surface(bounds, 1.0)
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(capture.call_count, 1)

            item.set_text_effects(stack)
            renderer.tile_cache.clear()
            full = renderer._render_effect_surface(bounds, 1.0)
            tiled = renderer._new_effect_pixmap(1.0, bounds)
            painter = QPainter(tiled)
            painter.translate(-bounds.topLeft())
            renderer.tile_cache.clear()
            try:
                renderer._draw_tiled_effects(
                    painter,
                    EffectRasterPlan('tiles', 1.0, 0, 0, 64),
                    bounds,
                )
            finally:
                painter.end()
            np.testing.assert_array_equal(
                pixmap2ndarray(full, keep_alpha=True),
                pixmap2ndarray(tiled, keep_alpha=True),
            )

    def test_forced_tiles_match_full_typed_effect_surface(self):
        stacks = (
            TextEffectStack(effects=(
                TextFillEffect(
                    opacity=0.7,
                    blend_mode='linear_burn',
                    paint=LinearGradientPaint(
                        stops=(
                            GradientStop(0.0, (220, 180, 40), 0.6),
                            GradientStop(1.0, (40, 200, 180), 0.9),
                        ),
                        angle=149.0,
                        scale=1.2,
                    ),
                ),
                ShadowEffect(
                    blend_mode='darker_color',
                    angle=34.0,
                    distance=0.216,
                    blur=0.08,
                    spread=0.04,
                    paint=LinearGradientPaint(angle=41.0),
                ),
                GlowEffect(
                    blend_mode='linear_dodge',
                    paint=LinearGradientPaint(angle=17.0),
                    size=0.08,
                    spread=0.03,
                ),
                StrokeEffect(
                    width=0.12,
                    position='outside',
                    blend_mode='lighter_color',
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
                    blend_mode='multiply',
                    shadow_type='inner',
                    angle=26.565,
                    distance=0.089,
                    blur=0.06,
                    spread=0.02,
                    paint=LinearGradientPaint(angle=203.0),
                ),
                GlowEffect(
                    blend_mode='screen',
                    glow_type='inner',
                    paint=LinearGradientPaint(angle=113.0),
                    size=0.06,
                    spread=0.02,
                ),
                TextFillEffect(
                    blend_mode='color_dodge',
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
                    angle=36.0,
                    distance=0.372,
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

    def test_two_filter_cumulative_halos_match_full_at_absolute_origins(self):
        stack = TextEffectStack(effects=(
            FilterEffect('builtin:grain', params={
                'amount': 0.55, 'size': 3.4, 'seed': 17,
            }),
            GlowEffect(size=0.08, spread=0.03),
            FilterEffect('builtin:gaussian_blur', params={'radius': 2.7}),
        ))
        item = self._item(stack)
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        logical = renderer.logical_unpadded_rect()
        self.assertLess(bounds.left() - logical.left(), 0.0)
        self.assertLess(bounds.top() - logical.top(), 0.0)

        for scale in (1.0, 2.0):
            with self.subTest(scale=scale):
                renderer.release_caches()
                full = renderer._render_effect_surface(bounds, scale)
                tiled = renderer._new_effect_pixmap(scale, bounds)
                painter = QPainter(tiled)
                painter.translate(-bounds.topLeft())
                try:
                    renderer._draw_tiled_effects(
                        painter,
                        EffectRasterPlan('tiles', scale, 0, 0, 96),
                        bounds,
                    )
                finally:
                    painter.end()
                np.testing.assert_array_equal(
                    pixmap2ndarray(full, keep_alpha=True),
                    pixmap2ndarray(tiled, keep_alpha=True),
                )

    def test_filter_only_receives_horizontal_and_vertical_canonical_text(self):
        stack = TextEffectStack(effects=(FilterEffect(
            'builtin:noise', params={
                'amount': 0.8, 'mode': 'monochrome', 'seed': 5,
            }
        ),))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._item(stack, vertical=vertical)
                pixels = pixmap2ndarray(
                    item.effect_renderer._render_effect_surface(
                        item.effect_renderer.boundingRect(), 1.0
                    ),
                    keep_alpha=True,
                )
                self.assertGreater(np.count_nonzero(pixels[:, :, 3]), 0)
                visible = pixels[:, :, 3] > 128
                self.assertGreater(
                    np.unique(pixels[:, :, 0][visible]).size, 1
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
                    'ballontranslator.ui.text_engine.effects.renderer.'
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
                'ballontranslator.ui.text_engine.effects.renderer.'
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
