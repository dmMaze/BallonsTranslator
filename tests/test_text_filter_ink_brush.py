"""Contracts for the built-in Ink Brush text filter."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtGui import QPainter
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.raster import EffectRasterPlan
from ballontranslator.ui.text_engine.effects.filters import (
    FilterContext,
    FilterRuntime,
    get_filter_registry,
)
from ballontranslator.utils.text_effects import FilterEffect, TextEffectStack
from ballontranslator.utils.textblock import TextBlock

FILTER_ID = 'builtin:ink_brush'
DEFAULTS = {
    'start_density': 0.05,
    'end_density': 0.25,
    'angle': 90.0,
    'size': 2.0,
    'streak': 0.6,
    'streak_angle': 0.0,
    'seed': 0,
}


def _solid_rect(height=240, width=240, inset=20, value=255):
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[inset:height - inset, inset:width - inset] = (20, 30, 40, value)
    return rgba


def _pattern(height=200, width=220):
    alpha = np.zeros((height, width), dtype=np.uint8)
    alpha[20:height - 20, width // 3:width // 2] = 255
    alpha[height // 2:height // 2 + 18, 16:width - 16] = 255
    alpha[height // 4:3 * height // 4, width // 5:width // 5 + 22] = 255
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha
    rgba[:, :, :3][alpha > 0] = (20, 30, 40)
    return rgba


def _eroded_fraction(source, result, rows):
    ink = source[rows, :, 3] > 0
    gone = ink & (result[rows, :, 3] == 0)
    return float(gone.sum()) / max(int(ink.sum()), 1)


class InkBrushAlgorithmTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = get_filter_registry()
        cls.spec = cls.registry.get_spec(FILTER_ID)
        cls.runtime = cls.registry.resolve(FilterEffect(
            FILTER_ID, schema_version=cls.spec.schema_version
        ))

    def test_metadata_defaults_and_neutral_halo(self):
        self.assertIsNotNone(self.spec)
        self.assertEqual(self.spec.name, 'Ink Brush')
        self.assertEqual(self.spec.order, 70)
        self.assertEqual(self.spec.schema_version, 2)
        self.assertFalse(self.spec.expands_alpha)
        self.assertEqual(self.spec.default_params(), DEFAULTS)
        self.assertEqual(
            self.runtime.params, self.spec.normalize_params(DEFAULTS)
        )
        neutral = dict(
            self.runtime.params, start_density=0.0, end_density=0.0
        )
        self.assertEqual(self.runtime.tile_halo(neutral, 1.0), 0)
        self.assertGreater(self.runtime.tile_halo(self.runtime.params, 1.0), 0)
        self.assertGreater(
            self.runtime.tile_halo(self.runtime.params, 2.0),
            self.runtime.tile_halo(self.runtime.params, 1.0),
        )

    def test_zero_density_is_identity(self):
        image = _pattern()
        params = dict(
            self.runtime.params, start_density=0.0, end_density=0.0
        )
        result = self.runtime.apply(
            image, params, FilterContext(1.0, 0, 0, False, 220, 200)
        )
        self.assertIs(result, image)

    def test_full_density_erases_the_ink(self):
        image = _solid_rect(height=200, width=200, inset=10)
        params = dict(
            self.runtime.params, start_density=1.0, end_density=1.0
        )
        result = self.runtime.apply(
            image.copy(), params, FilterContext(1.0, 0, 0, False, 200, 200)
        )
        self.assertEqual(int(np.count_nonzero(result[:, :, 3])), 0)

    def test_density_interpolates_linearly_along_direction(self):
        image = _solid_rect(height=300, width=120, inset=20)
        params = dict(
            self.runtime.params, start_density=0.10, end_density=0.50,
            angle=90.0, size=2.0,
        )
        result = self.runtime.apply(
            image.copy(), params, FilterContext(1.0, 0, 0, False, 120, 300)
        )
        center = slice(55, 65)
        bands = (slice(35, 65), slice(135, 165), slice(235, 265))
        fractions = [
            _eroded_fraction(image, result, band)
            for band in bands
        ]
        self.assertLess(fractions[0], fractions[1])
        self.assertLess(fractions[1], fractions[2])
        for fraction, expected in zip(fractions, (0.15, 0.30, 0.45)):
            self.assertAlmostEqual(fraction, expected, delta=0.10)
        self.assertGreater(
            int(np.count_nonzero(result[40:260, center, 3] == 0)), 0
        )

    def test_uniform_density_matches_hole_fraction(self):
        image = _solid_rect(height=240, width=240, inset=20)
        params = dict(
            self.runtime.params, start_density=0.4, end_density=0.4,
            angle=90.0, size=2.0, streak=0.6,
        )
        result = self.runtime.apply(
            image.copy(), params, FilterContext(1.0, 0, 0, False, 240, 240)
        )
        holes = result[60:180, 60:180, 3] == 0
        self.assertAlmostEqual(float(holes.mean()), 0.4, delta=0.06)

    def test_streak_direction_rotates_anisotropy(self):
        image = _solid_rect(height=240, width=240, inset=20)
        params = dict(
            self.runtime.params, start_density=0.4, end_density=0.4,
            angle=90.0, size=2.0, streak=0.9,
        )

        def correlations(streak_angle):
            result = self.runtime.apply(
                image.copy(),
                dict(params, streak_angle=streak_angle),
                FilterContext(1.0, 0, 0, False, 240, 240),
            )
            holes = (result[60:180, 60:180, 3] == 0).astype(np.float32)
            return (
                float((holes[:, 1:] * holes[:, :-1]).mean()),
                float((holes[1:, :] * holes[:-1, :]).mean()),
            )

        horizontal, vertical = correlations(0.0)
        self.assertGreater(horizontal, vertical)
        rotated_horizontal, rotated_vertical = correlations(90.0)
        self.assertGreater(rotated_vertical, rotated_horizontal)

    def test_schema_one_density_and_variation_migrate_to_endpoints(self):
        effect = FilterEffect(
            FILTER_ID, schema_version=1,
            params={'density': 1.0, 'variation': 0.5, 'opaque': 'kept'},
        )
        runtime = self.registry.resolve(effect)
        self.assertAlmostEqual(runtime.params['start_density'], 0.3)
        self.assertAlmostEqual(runtime.params['end_density'], 0.6)
        self.assertEqual(effect.params_dict()['opaque'], 'kept')

    def test_alpha_contract_and_determinism(self):
        image = _pattern()
        context = FilterContext(1.0, -7, 5, False, 220, 200)
        first = self.runtime.apply(image.copy(), self.runtime.params, context)
        second = self.runtime.apply(image.copy(), self.runtime.params, context)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, image.dtype)
        self.assertTrue(first.flags.c_contiguous)
        self.assertFalse(np.any(first[:, :, 3] > image[:, :, 3]))
        np.testing.assert_array_equal(first[:, :, :3], image[:, :, :3])
        empty = image[:, :, 3] == 0
        self.assertTrue(np.all(first[:, :, 3][empty] == 0))

    def test_density_gradient_follows_direction(self):
        image = _solid_rect()
        params = dict(
            self.runtime.params, start_density=0.15, end_density=0.75
        )
        top = slice(20, 80)
        bottom = slice(160, 220)

        down = self.runtime.apply(
            image.copy(), dict(params, angle=90.0),
            FilterContext(1.0, 0, 0, False, 240, 240),
        )
        up = self.runtime.apply(
            image.copy(), dict(params, angle=270.0),
            FilterContext(1.0, 0, 0, False, 240, 240),
        )

        self.assertGreater(
            _eroded_fraction(image, down, bottom),
            _eroded_fraction(image, down, top),
        )
        self.assertGreater(
            _eroded_fraction(image, up, top),
            _eroded_fraction(image, up, bottom),
        )

    def test_tiles_match_full_surface_at_one_and_two_x(self):
        params = dict(
            self.runtime.params, start_density=0.1, end_density=0.45
        )
        for scale in (1.0, 2.0):
            with self.subTest(scale=scale):
                height = int(200 * scale)
                width = int(220 * scale)
                source = _pattern(height, width)
                halo = self.runtime.tile_halo(params, scale)
                full = self.runtime.apply(
                    source.copy(), params,
                    FilterContext(scale, 0, 0, False, 220, 200),
                )
                for top in range(0, height, 56):
                    for left in range(0, width, 56):
                        bottom = min(top + 56, height)
                        right = min(left + 56, width)
                        crop_top = max(0, top - halo)
                        crop_left = max(0, left - halo)
                        crop_bottom = min(height, bottom + halo)
                        crop_right = min(width, right + halo)
                        crop = source[
                            crop_top:crop_bottom, crop_left:crop_right
                        ].copy()
                        filtered = self.runtime.apply(
                            crop, params,
                            FilterContext(
                                scale, crop_left, crop_top, False, 220, 200
                            ),
                        )
                        core = filtered[
                            top - crop_top:bottom - crop_top,
                            left - crop_left:right - crop_left,
                        ]
                        np.testing.assert_array_equal(
                            core, full[top:bottom, left:right]
                        )

    def test_invalid_active_params_use_declared_defaults(self):
        effect = FilterEffect(
            FILTER_ID, schema_version=2,
            params={'start_density': 9.0, 'opaque_future': 'preserved'},
        )
        runtime = self.registry.resolve(effect)
        self.assertEqual(
            runtime.params['start_density'], DEFAULTS['start_density']
        )
        self.assertEqual(
            effect.params_dict()['opaque_future'], 'preserved'
        )

    def test_edge_outside_tile_halo_cannot_change_core(self) -> None:
        params = dict(
            self.runtime.params, start_density=0.4, end_density=0.4
        )
        source = np.full((256, 256, 4), 255, dtype=np.uint8)
        source[:, 0, 3] = 0
        for scale in (1.0, 1.25, 2.0, 4.0):
            with self.subTest(scale=scale):
                halo = self.runtime.tile_halo(params, scale)
                full = self.runtime.apply(
                    source.copy(), params,
                    FilterContext(scale, -7, -13, False, 256, 256),
                )
                # The tile omits the only transparent edge, beyond its halo.
                tile = self.runtime.apply(
                    source[:, 1:].copy(), params,
                    FilterContext(scale, -6, -13, False, 256, 256),
                )
                np.testing.assert_array_equal(
                    full[:, 1 + halo:], tile[:, halo:]
                )


class InkBrushRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(stack, *, text='Ink brush text', vertical=False):
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = text
        block.vertical = vertical
        block.fontformat.frgb = [0, 0, 0]
        block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _full_and_tiled_pixels(item, scale, tile_edge=128):
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        renderer.release_caches()
        full = renderer._render_effect_surface(bounds, scale)
        tiled = renderer._new_effect_pixmap(scale, bounds)
        painter = QPainter(tiled)
        painter.translate(-bounds.topLeft())
        renderer.tile_cache.clear()
        try:
            renderer._draw_tiled_effects(
                painter,
                EffectRasterPlan('tiles', scale, 0, 0, tile_edge),
                bounds,
            )
        finally:
            painter.end()
        return (
            pixmap2ndarray(full, keep_alpha=True),
            pixmap2ndarray(tiled, keep_alpha=True),
        )

    def test_renderer_passes_unpadded_logical_size_to_filters(self):
        contexts = []

        def apply(rgba, _params, context):
            contexts.append(context)
            return rgba

        effect = FilterEffect(FILTER_ID, schema_version=2)
        spec = SimpleNamespace(
            filter_id=FILTER_ID,
            schema_version=2,
            expands_alpha=False,
        )
        runtime = FilterRuntime(
            spec, effect.params_dict(), apply, lambda _params, _scale: 0
        )
        registry = SimpleNamespace(resolve=lambda _effect: runtime)
        item = self._item(TextEffectStack(effects=(effect,)))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        logical = renderer.logical_unpadded_rect()
        renderer.release_caches()
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            renderer._render_effect_surface(bounds, 1.0)

        self.assertTrue(contexts)
        for context in contexts:
            self.assertAlmostEqual(context.logical_width, logical.width())
            self.assertAlmostEqual(context.logical_height, logical.height())
            self.assertGreater(context.logical_width, 0.0)
            self.assertGreater(context.logical_height, 0.0)

    def test_percent_parameter_editors_accept_values(self):
        from ballontranslator.ui.text_engine.effects.cards import (
            FilterEffectCard,
        )

        spec = get_filter_registry().get_spec(FILTER_ID)
        card = FilterEffectCard(0, FILTER_ID, spec)
        commits = []
        card.value_commit_requested.connect(
            lambda _index, name, value: commits.append((name, value))
        )
        try:
            for key, text, expected in (
                ('start_density', '35%', 0.35),
                ('end_density', '65%', 0.65),
                ('streak', '30%', 0.3),
                ('angle', '45\u00b0', 45.0),
                ('streak_angle', '90\u00b0', 90.0),
                ('size', '3.5', 3.5),
                ('seed', '7', 7),
            ):
                with self.subTest(parameter=key):
                    control = card.numeric_controls[key]
                    control.editor.setText(text)
                    control.editor.textEdited.emit(text)
                    self.assertTrue(control.commit_pending())
                    self.assertAlmostEqual(commits[-1][1], expected, places=4)
        finally:
            card.deleteLater()

    def test_full_and_tiled_rendering_match_at_one_and_two_x(self):
        stack = TextEffectStack(effects=(
            FilterEffect(
                FILTER_ID, schema_version=2,
                params=dict(DEFAULTS, end_density=0.6),
            ),
        ))
        for vertical in (False, True):
            for scale in (1.0, 2.0):
                with self.subTest(vertical=vertical, scale=scale):
                    item = self._item(stack, vertical=vertical)
                    full, tiled = self._full_and_tiled_pixels(item, scale)
                    np.testing.assert_array_equal(full, tiled)
                    self.assertGreater(np.count_nonzero(full[:, :, 3]), 0)


if __name__ == '__main__':
    unittest.main()
