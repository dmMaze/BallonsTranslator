import builtins
import os
import os.path as osp
import threading
import unittest
from unittest.mock import patch

import numpy as np
from qtpy.QtCore import QRectF

from ballontranslator.ui.text_engine.rendering import effect_paint
from ballontranslator.ui.text_engine.rendering import effect_paint_numba
from ballontranslator.ui.text_engine.rendering.effect_paint import (
    colorize_effect_paint_rgba,
    start_effect_paint_numba_warmup,
)
from ballontranslator.ui.text_engine.rendering.effect_paint_numba import (
    NUMBA_CACHE_DIR,
    colorize_linear_gradient_rgba,
    warm_effect_paint_numba_cache,
)
from ballontranslator.utils import shared
from ballontranslator.utils.text_effects import (
    GradientStop,
    LinearGradientPaint,
    SolidPaint,
)


class EffectPaintNumbaTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        warm_effect_paint_numba_cache()
        cls._backend = effect_paint._numba_colorize_linear_gradient_rgba
        effect_paint._numba_colorize_linear_gradient_rgba = (
            colorize_linear_gradient_rgba
        )

    @classmethod
    def tearDownClass(cls) -> None:
        effect_paint._numba_colorize_linear_gradient_rgba = cls._backend

    @staticmethod
    def _numpy_result(
        paint,
        rgba,
        surface_rect,
        logical_rect,
        render_scale,
    ):
        with patch.object(
            effect_paint,
            '_compiled_colorize_linear_gradient_rgba',
            return_value=False,
        ):
            return colorize_effect_paint_rgba(
                paint,
                rgba,
                surface_rect,
                logical_rect,
                render_scale,
            )

    def test_cache_and_background_warmup_use_existing_ownership(self):
        expected = osp.join(shared.cache_dir, 'numba')
        self.assertEqual(NUMBA_CACHE_DIR, expected)
        self.assertEqual(os.environ['NUMBA_CACHE_DIR'], expected)

        main_thread = threading.current_thread()
        warmed = threading.Event()

        def record_thread() -> None:
            self.assertIsNot(threading.current_thread(), main_thread)
            warmed.set()

        with patch.object(
            effect_paint_numba,
            'warm_effect_paint_numba_cache',
            side_effect=record_thread,
        ), patch.object(
            effect_paint, '_numba_colorize_linear_gradient_rgba', None
        ):
            thread = start_effect_paint_numba_warmup()
            thread.join(5.0)
            self.assertIs(
                effect_paint._numba_colorize_linear_gradient_rgba,
                colorize_linear_gradient_rgba,
            )
        self.assertFalse(thread.is_alive())
        self.assertTrue(thread.daemon)
        self.assertTrue(warmed.is_set())

        with patch.object(
            effect_paint_numba,
            'warm_effect_paint_numba_cache',
            side_effect=RuntimeError('warmup failed'),
        ), patch.object(
            effect_paint, '_numba_colorize_linear_gradient_rgba', None
        ):
            failed = start_effect_paint_numba_warmup()
            failed.join(5.0)
            self.assertIsNone(
                effect_paint._numba_colorize_linear_gradient_rgba
            )

    def test_first_use_and_unavailable_backend_fall_back_without_mutation(self):
        paint = LinearGradientPaint()
        surface = QRectF(0.25, -1.75, 4.0, 2.0)
        logical = QRectF(-2.5, 0.75, 9.5, 3.25)
        original = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)
        expected = self._numpy_result(
            paint, original.copy(), surface, logical, 1.25
        )

        original_import = builtins.__import__

        def reject_numba_import(name, *args, **kwargs):
            if name.endswith('effect_paint_numba'):
                raise AssertionError('pre-warm caller tried to import Numba')
            return original_import(name, *args, **kwargs)

        before_warmup = original.copy()
        with patch.object(
            effect_paint, '_numba_colorize_linear_gradient_rgba', None
        ), patch('builtins.__import__', side_effect=reject_numba_import):
            result = colorize_effect_paint_rgba(
                paint,
                before_warmup,
                surface,
                logical,
                1.25,
            )
        self.assertIs(result, before_warmup)
        np.testing.assert_array_equal(result, expected)

        unavailable = original.copy()
        with patch.object(
            effect_paint, '_numba_colorize_linear_gradient_rgba', None
        ):
            result = colorize_effect_paint_rgba(
                paint,
                unavailable,
                surface,
                logical,
                1.25,
            )
        self.assertIs(result, unavailable)
        np.testing.assert_array_equal(result, expected)

    def test_solid_and_strided_targets_keep_the_numpy_path(self):
        solid = np.full((2, 3, 4), 255, dtype=np.uint8)
        with patch.object(
            effect_paint,
            '_compiled_colorize_linear_gradient_rgba',
            side_effect=AssertionError('SolidPaint reached gradient backend'),
        ):
            colorize_effect_paint_rgba(
                SolidPaint((1, 2, 3)),
                solid,
                QRectF(),
                QRectF(),
                1.0,
            )
        self.assertTrue(np.all(solid[..., :3] == (1, 2, 3)))

        storage = np.arange(8 * 22 * 4, dtype=np.uint8).reshape(8, 22, 4)
        gaps = storage[:, 1::2].copy()
        target = storage[:, ::2]
        surface = QRectF(-3.25, 2.75, 8.8, 6.4)
        logical = QRectF(-7.5, -4.25, 19.75, 12.5)
        paint = LinearGradientPaint(angle=117.0, scale=1.35)
        expected = self._numpy_result(
            paint, target.copy(), surface, logical, 1.25
        )
        with patch.object(
            effect_paint_numba,
            '_colorize_linear_gradient_rgba',
            side_effect=AssertionError('strided target reached Numba'),
        ):
            result = colorize_effect_paint_rgba(
                paint, target, surface, logical, 1.25
            )
        self.assertIs(result, target)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(storage[:, 1::2], gaps)

    def test_randomized_compiled_results_match_numpy_exactly(self):
        rng = np.random.default_rng(20260825)
        stop_counts = (2, 3, 5, 8, 32)
        heights = (1, 3, 17, 263)
        widths = (1, 9, 31)
        scales = (0.5, 1.0, 1.25, 2.0, 3.0)
        opacity_values = (0.0, 0.125, 0.5, 0.875, 1.0)

        for case in range(256):
            stop_count = stop_counts[case % len(stop_counts)]
            positions = np.sort(rng.choice(
                np.linspace(0.0, 1.0, 21),
                stop_count,
                replace=True,
            ))
            colors = rng.integers(0, 256, (stop_count, 3))
            opacities = rng.choice(opacity_values, stop_count)
            paint = LinearGradientPaint(
                stops=tuple(
                    GradientStop(
                        float(positions[index]),
                        tuple(int(value) for value in colors[index]),
                        float(opacities[index]),
                    )
                    for index in range(stop_count)
                ),
                angle=float(rng.uniform(0.0, 360.0)),
                scale=float(rng.uniform(0.1, 4.0)),
            )
            height = heights[case % len(heights)]
            width = widths[case % len(widths)]
            render_scale = scales[case % len(scales)]
            surface = QRectF(
                float(rng.uniform(-50.0, 50.0)),
                float(rng.uniform(-50.0, 50.0)),
                width / render_scale,
                height / render_scale,
            )
            logical = QRectF(
                float(rng.uniform(-20.0, 20.0)),
                float(rng.uniform(-20.0, 20.0)),
                float(rng.uniform(1.0, 120.0)),
                float(rng.uniform(1.0, 120.0)),
            )
            original = rng.integers(
                0, 256, (height, width, 4), dtype=np.uint8
            )
            expected = self._numpy_result(
                paint,
                original.copy(),
                surface,
                logical,
                render_scale,
            )
            actual = colorize_effect_paint_rgba(
                paint,
                original.copy(),
                surface,
                logical,
                render_scale,
            )
            np.testing.assert_array_equal(
                actual, expected, err_msg=f'random gradient case {case}'
            )

    def test_compiled_full_and_tiles_are_identical(self):
        paint = LinearGradientPaint(
            stops=(
                GradientStop(0.0, (10, 20, 30), 0.25),
                GradientStop(0.35, (220, 40, 80), 0.8),
                GradientStop(0.35, (30, 210, 90), 0.4),
                GradientStop(1.0, (80, 60, 240), 1.0),
            ),
            angle=73.0,
            scale=1.4,
        )
        render_scale = 2.0
        surface = QRectF(-13.25, 7.75, 38.5, 131.5)
        logical = QRectF(-3.5, 2.25, 71.75, 83.5)
        rng = np.random.default_rng(91)
        original = rng.integers(0, 256, (263, 77, 4), dtype=np.uint8)
        full = colorize_effect_paint_rgba(
            paint,
            original.copy(),
            surface,
            logical,
            render_scale,
        )
        tiled = np.empty_like(full)
        for top, bottom in ((0, 127), (127, 263)):
            for left, right in ((0, 29), (29, 77)):
                tile_rect = QRectF(
                    surface.left() + left / render_scale,
                    surface.top() + top / render_scale,
                    (right - left) / render_scale,
                    (bottom - top) / render_scale,
                )
                tiled[top:bottom, left:right] = colorize_effect_paint_rgba(
                    paint,
                    original[top:bottom, left:right].copy(),
                    tile_rect,
                    logical,
                    render_scale,
                )
        np.testing.assert_array_equal(tiled, full)


if __name__ == '__main__':
    unittest.main()
