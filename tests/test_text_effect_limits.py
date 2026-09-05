import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.effects.limits import limit_effect_radii
from ballontranslator.ui.text_engine.effects.shadow import _blur
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.morphology import dilate_alpha_disc
from ballontranslator.ui.text_engine.rendering.raster import (
    EFFECT_CACHE_MAX_BYTES, plan_effect_raster,
)
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.text_effects import (
    GlowEffect, ShadowEffect, StrokeEffect, TextEffectStack,
)


class EffectRadiusTest(unittest.TestCase):
    def test_disc_matches_opencv_for_soft_edges_and_cropped_sources(self) -> None:
        random = np.random.default_rng(42)
        for radius in (16, 17, 31, 64, 128):
            for source in (
                random.integers(0, 256, (49, 63), dtype=np.uint8),
                np.pad(np.full((3, 5), 100, np.uint8), ((7, 19), (5, 41))),
                np.zeros((40, 50), np.uint8),
            ):
                with self.subTest(radius=radius, shape=source.shape):
                    expected = cv2.dilate(source, cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                    ))
                    np.testing.assert_array_equal(dilate_alpha_disc(source, radius), expected)

    def test_large_blur_keeps_kernel_extent_and_alpha_precision(self) -> None:
        mask = np.zeros((181, 203), dtype=np.uint8)
        mask[40:110, 30:150] = 100
        for radius in (25, 64):
            expected = cv2.GaussianBlur(
                mask, (2 * radius + 1, 2 * radius + 1), (2 * radius + 1) / 6,
                borderType=cv2.BORDER_CONSTANT,
            )
            actual = _blur(mask, radius)
            self.assertLessEqual(int(np.abs(actual.astype(int) - expected.astype(int)).max()), 2)
            self.assertLessEqual(int(actual.max()), 100)

    def test_spread_saturates_without_changing_saved_values_or_offset(self) -> None:
        stack = TextEffectStack(effects=(ShadowEffect(
            distance=0.2, blur=0.1, spread=10.0,
        ),))
        limited = limit_effect_radii(stack, 200.0, 500.0)
        effect = limited.effects[0]
        self.assertEqual(effect.distance, 0.2)
        self.assertEqual(effect.blur, 0.1)
        self.assertAlmostEqual(effect.spread, 2.2)
        self.assertEqual(stack.effects[0].spread, 10.0)
        larger_font = limit_effect_radii(stack, 400.0, 500.0).effects[0]
        self.assertLess(larger_font.spread, effect.spread)

    def test_generated_reaches_share_stroke_budget(self) -> None:
        stack = TextEffectStack(effects=(
            ShadowEffect(spread=10.0, blur=0.2, distance=0.1),
            GlowEffect(size=0.2, spread=10.0),
            StrokeEffect(width=2.0),
        ))
        shadow, glow, stroke = limit_effect_radii(stack, 200.0, 500.0).effects
        self.assertLessEqual(stroke.width / 2 + shadow.distance + shadow.blur + shadow.spread, 2.5)
        self.assertLessEqual(stroke.width / 2 + glow.size + glow.spread, 2.5)


class BoundedEffectRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(effect) -> TextBlkItem:
        block = TextBlock([0, 0, 1800, 1600])
        block._bounding_rect = [0, 0, 1800, 1600]
        block.translation = 'Sh'
        block.fontformat.font_size = 200.0
        block.fontformat.text_effects = TextEffectStack(effects=(effect,))
        return TextBlkItem(block, 0)

    def test_large_view_keeps_alpha_and_reuses_visible_cores(self) -> None:
        item = self._item(ShadowEffect(spread=10.0, blur=0.05, distance=0.1))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        plan = plan_effect_raster(bounds.width(), bounds.height(), 1.0)
        self.assertEqual(plan.mode, 'tiles')
        image = QImage(math.ceil(bounds.width()), math.ceil(bounds.height()),
                       QImage.Format.Format_ARGB32_Premultiplied)

        def render_tile(rect, scale, **kwargs):
            tile = renderer._new_effect_pixmap(scale, rect)
            tile.fill(QColor(40, 80, 120, 128))
            return tile

        with patch.object(renderer, '_render_effect_surface', side_effect=render_tile) as render:
            counts = []
            for _ in range(2):
                image.fill(0)
                painter = QPainter(image)
                painter.translate(-bounds.topLeft())
                try:
                    renderer._draw_tiled_effects(painter, plan, bounds)
                finally:
                    painter.end()
                counts.append(render.call_count)
            self.assertGreater(counts[0], 2)
            self.assertEqual(counts[0], counts[1])
        alpha = pixmap2ndarray(image, keep_alpha=True)[..., 3]
        self.assertTrue(np.all(alpha[2:-2, 2:-2] == 128))
        self.assertEqual(renderer.allocation_warning_generation, -1)
        self.assertLessEqual(sum(
            pixmap.width() * pixmap.height() * 4
            for _rect, pixmap in renderer.tile_cache.values()
        ), EFFECT_CACHE_MAX_BYTES)

    def test_inputs_above_limit_do_not_invalidate_same_pixels(self) -> None:
        item = self._item(ShadowEffect(spread=8.0))
        renderer = item.effect_renderer
        before = renderer._effect_cache_input_key()
        generation = renderer.cache_generation
        item.set_text_effects(TextEffectStack(effects=(ShadowEffect(spread=10.0),)))
        self.assertEqual(before, renderer._effect_cache_input_key())
        self.assertEqual(generation, renderer.cache_generation)
        self.assertEqual(item.fontformat.text_effects.effects[0].spread, 10.0)
        item.set_text_effects(TextEffectStack(effects=(ShadowEffect(spread=9.0),)), preview=True)
        self.assertFalse(renderer._effect_preview_changes_pixels())


if __name__ == '__main__':
    unittest.main()
