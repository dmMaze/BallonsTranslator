import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
from PIL import Image
from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.misc import ndarray2pixmap, pixmap2ndarray
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.effects.filters import FilterRuntime
from ballontranslator.ui.text_engine.rendering.raster import (
    EffectRasterPlan,
    EffectRasterAllocationError,
)
from ballontranslator.utils.fontformat import SineTextTransform, TextTransformStack
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.text_alpha_mask import AlphaBrushStroke, TextAlphaMask
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    ImageEffect,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
)
from ballontranslator.utils.textblock import TextBlock


class _RuntimeRegistry:
    def __init__(self, runtimes):
        self.runtimes = runtimes
        self.resolve_calls = {filter_id: 0 for filter_id in runtimes}

    def resolve(self, effect):
        self.resolve_calls[effect.filter_id] += 1
        return self.runtimes[effect.filter_id]


def _runtime(effect, apply, halo=0, *, expands_alpha=False):
    spec = SimpleNamespace(
        filter_id=effect.filter_id,
        schema_version=effect.schema_version,
        expands_alpha=expands_alpha,
    )
    return FilterRuntime(spec, effect.params_dict(), apply, lambda _p, _s: halo)


class TextFilterRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _item(stack, *, text='Filtered text', vertical=False):
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = text
        block.vertical = vertical
        block.fontformat.frgb = [180, 90, 40]
        block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _public_pixels(item):
        scene = QGraphicsScene()
        scene.addItem(item)
        image = QImage(
            420, 260, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            scene.render(
                painter,
                QRectF(0, 0, 420, 260),
                QRectF(-30, -30, 420, 260),
            )
        finally:
            painter.end()
            scene.removeItem(item)
        return pixmap2ndarray(image, keep_alpha=True)

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

    @staticmethod
    def _import_asset(project, directory, pixels):
        source = Path(directory) / 'rendered.png'
        Image.fromarray(pixels, 'RGBA').save(source)
        return project.import_raster_asset(str(source))

    def test_chain_runs_bottom_to_top_through_one_rgba_bridge(self):
        first = FilterEffect('custom:first')
        second = FilterEffect('custom:second')
        calls = []

        def apply_first(rgba, _params, _context):
            calls.append('first')
            rgba[:, :, 0] = np.minimum(rgba[:, :, 0], 210)
            return rgba

        def apply_second(rgba, _params, _context):
            calls.append('second')
            rgba[:, :, 1] = np.minimum(rgba[:, :, 1], 200)
            return rgba

        registry = _RuntimeRegistry({
            first.filter_id: _runtime(first, apply_first),
            second.filter_id: _runtime(second, apply_second),
        })
        item = self._item(TextEffectStack(effects=(first, second)))
        renderer = item.effect_renderer
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ), patch(
            'ballontranslator.ui.text_engine.effects.renderer.pixmap2ndarray',
            wraps=pixmap2ndarray,
        ) as to_array, patch(
            'ballontranslator.ui.text_engine.effects.renderer.ndarray2pixmap',
            wraps=ndarray2pixmap,
        ) as to_pixmap:
            renderer._render_pre_mask_effect_surface(
                renderer.boundingRect(), 1.0
            )

        self.assertEqual(calls, ['second', 'first'])
        self.assertEqual(to_array.call_count, 1)
        self.assertEqual(to_pixmap.call_count, 1)

    def test_image_then_custom_filter_then_eraser(self):
        effect = FilterEffect('custom:image-filter')
        order = []

        def filter_image(rgba, _params, _context):
            visible = rgba[:, :, 3] > 200
            self.assertGreater(np.count_nonzero(visible), 0)
            self.assertGreater(
                np.mean(rgba[:, :, 2][visible]),
                np.mean(rgba[:, :, 0][visible]),
            )
            order.append('filter')
            rgba[:, :, 1] = 220
            return rgba

        registry = _RuntimeRegistry({
            effect.filter_id: _runtime(effect, filter_image)
        })
        with tempfile.TemporaryDirectory() as directory:
            project = ProjImgTrans()
            project.directory = directory
            asset = self._import_asset(
                project,
                directory,
                np.full((3, 5, 4), (20, 60, 230, 255), np.uint8),
            )
            item = self._item(TextEffectStack(effects=(
                effect,
                ImageEffect(asset, mode='foreground'),
            )), text='')
            item.blk.text_alpha_mask = TextAlphaMask(strokes=(
                AlphaBrushStroke('erase', 1000, ((160, 90),)),
            ))
            scene = QGraphicsScene()
            scene.imgtrans_proj = project
            scene.addItem(item)
            renderer = item.effect_renderer
            renderer.project_assets_changed()
            original_mask = renderer._apply_text_alpha_mask

            def apply_mask(*args, **kwargs):
                order.append('eraser')
                return original_mask(*args, **kwargs)

            with patch(
                'ballontranslator.ui.text_engine.effects.renderer.'
                'get_filter_registry',
                return_value=registry,
            ), patch.object(
                renderer,
                '_apply_text_alpha_mask',
                side_effect=apply_mask,
            ):
                renderer.release_caches()
                rendered = pixmap2ndarray(
                    renderer._render_effect_surface(
                        renderer.boundingRect(), 1.0
                    ),
                    keep_alpha=True,
                )

            self.assertEqual(order, ['filter', 'eraser'])
            self.assertEqual(np.count_nonzero(rendered[:, :, 3]), 0)
            scene.removeItem(item)

    def test_global_order_selects_stroke_and_glow_filter_input(self):
        effect = FilterEffect('custom:blackout')

        def blackout(rgba, _params, _context):
            rgba[:, :, :3] = 0
            return rgba

        registry = _RuntimeRegistry({
            effect.filter_id: _runtime(effect, blackout)
        })
        layers = (
            GlowEffect(
                size=0.10,
                spread=0.04,
                paint=SolidPaint((240, 40, 20)),
            ),
            StrokeEffect(
                width=0.20,
                position='outside',
                paint=SolidPaint((20, 240, 40)),
            ),
        )

        def render(stack):
            item = self._item(stack)
            renderer = item.effect_renderer
            return pixmap2ndarray(
                renderer._render_effect_surface(
                    renderer.boundingRect(), 1.0
                ),
                keep_alpha=True,
            )

        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            filter_above = render(TextEffectStack(effects=(effect,) + layers))
            layers_above = render(TextEffectStack(effects=layers + (effect,)))

        self.assertEqual(np.count_nonzero(filter_above[:, :, :3]), 0)
        self.assertGreater(np.count_nonzero(layers_above[:, :, :3]), 0)

    def test_expanding_builtins_match_full_tiles_at_one_and_two_x_h_and_v(self):
        effects = (
            FilterEffect('builtin:gaussian_blur', params={'radius': 2.0}),
            FilterEffect('builtin:bloom', params={
                'threshold': 0.45, 'radius': 2.0, 'intensity': 0.9,
            }),
            FilterEffect('builtin:glitch', params={
                'shift': 4.0, 'block_size': 5.0, 'activity': 0.75,
                'rgb_split': 2.0, 'seed': 23,
            }),
        )
        for effect in effects:
            for vertical in (False, True):
                for scale in (1.0, 2.0):
                    with self.subTest(
                        filter_id=effect.filter_id,
                        vertical=vertical,
                        scale=scale,
                    ):
                        item = self._item(
                            TextEffectStack(effects=(effect,)),
                            vertical=vertical,
                        )
                        full, tiled = self._full_and_tiled_pixels(
                            item, scale
                        )
                        np.testing.assert_array_equal(full, tiled)
                        self.assertGreater(
                            np.count_nonzero(full[:, :, 3]), 0
                        )

    def test_cumulative_interleaved_expanding_chain_matches_full_tiles(self):
        stack = TextEffectStack(effects=(
            FilterEffect('builtin:glitch', params={
                'shift': 3.0, 'block_size': 4.0, 'activity': 1.0,
                'rgb_split': 2.0, 'seed': 31,
            }),
            FilterEffect('builtin:gaussian_blur', params={'radius': 1.5}),
            GlowEffect(size=0.06, spread=0.02),
            FilterEffect('builtin:bloom', params={
                'threshold': 0.4, 'radius': 2.0, 'intensity': 0.7,
            }),
        ))
        for vertical in (False, True):
            for scale in (1.0, 2.0):
                with self.subTest(vertical=vertical, scale=scale):
                    item = self._item(stack, vertical=vertical)
                    full, tiled = self._full_and_tiled_pixels(
                        item, scale, tile_edge=160
                    )
                    np.testing.assert_array_equal(full, tiled)

    def test_expanding_builtins_respect_generated_layer_order(self):
        stroke = StrokeEffect(
            width=0.18,
            position='outside',
            paint=SolidPaint((255, 255, 255)),
        )
        filters = (
            FilterEffect('builtin:gaussian_blur', params={'radius': 2.0}),
            FilterEffect('builtin:bloom', params={
                'threshold': 0.6, 'radius': 2.0, 'intensity': 1.0,
            }),
            FilterEffect('builtin:glitch', params={
                'shift': 5.0, 'block_size': 3.0, 'activity': 1.0,
                'rgb_split': 3.0, 'seed': 9,
            }),
        )
        for effect in filters:
            with self.subTest(filter_id=effect.filter_id):
                above = self._item(TextEffectStack(effects=(effect, stroke)))
                below = self._item(TextEffectStack(effects=(stroke, effect)))
                above_pixels = pixmap2ndarray(
                    above.effect_renderer._render_effect_surface(
                        above.effect_renderer.boundingRect(), 1.0
                    ),
                    keep_alpha=True,
                )
                below_pixels = pixmap2ndarray(
                    below.effect_renderer._render_effect_surface(
                        below.effect_renderer.boundingRect(), 1.0
                    ),
                    keep_alpha=True,
                )
                self.assertFalse(np.array_equal(above_pixels, below_pixels))

    def test_expanding_builtin_chain_is_strict_export_eligible_h_and_v(self):
        stack = TextEffectStack(effects=(
            FilterEffect('builtin:gaussian_blur', params={'radius': 1.0}),
            FilterEffect('builtin:bloom', params={
                'threshold': 0.5, 'radius': 1.0, 'intensity': 0.8,
            }),
            FilterEffect('builtin:glitch', params={
                'shift': 2.0, 'block_size': 4.0, 'activity': 1.0,
                'rgb_split': 1.0, 'seed': 7,
            }),
        ))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._item(stack, vertical=vertical)
                renderer = item.effect_renderer
                renderer.set_export_effect_render(True)
                try:
                    pixels = pixmap2ndarray(
                        renderer._render_effect_surface(
                            renderer.boundingRect(), 1.0
                        ),
                        keep_alpha=True,
                    )
                finally:
                    renderer.set_export_effect_render(False)
                self.assertGreater(np.count_nonzero(pixels[:, :, 3]), 0)

    def test_neutral_expanding_builtins_do_not_add_effect_padding(self):
        effects = (
            FilterEffect('builtin:gaussian_blur', params={'radius': 0.0}),
            FilterEffect('builtin:bloom', params={
                'threshold': 0.0, 'radius': 32.0, 'intensity': 0.0,
            }),
            FilterEffect('builtin:glitch', params={
                'shift': 64.0, 'block_size': 1.0, 'activity': 0.0,
                'rgb_split': 32.0, 'seed': 0,
            }),
        )
        for effect in effects:
            with self.subTest(filter_id=effect.filter_id):
                item = self._item(TextEffectStack(effects=(effect,)))
                item.effect_renderer._update_effect_padding()
                self.assertEqual(item.padding(), 0.0)

    def test_center_stroke_keeps_face_and_band_above_or_below_filter(self):
        effect = FilterEffect('custom:identity')
        registry = _RuntimeRegistry({
            effect.filter_id: _runtime(
                effect, lambda rgba, _params, _context: rgba
            )
        })
        stroke = StrokeEffect(
            width=0.20,
            position='center',
            paint=SolidPaint((20, 40, 240)),
        )

        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            for stack in (
                TextEffectStack(effects=(effect, stroke)),
                TextEffectStack(effects=(stroke, effect)),
            ):
                with self.subTest(stack=stack):
                    item = self._item(stack)
                    renderer = item.effect_renderer
                    bounds = renderer.boundingRect()
                    base = pixmap2ndarray(
                        renderer._render_effect_base(bounds, 1.0),
                        keep_alpha=True,
                    )
                    pixels = pixmap2ndarray(
                        renderer._render_effect_surface(bounds, 1.0),
                        keep_alpha=True,
                    )
                    rgb = pixels[..., :3].astype(np.int16)
                    face = (
                        (pixels[..., 3] > 200)
                        & (rgb[..., 0] > rgb[..., 2] + 80)
                    )
                    band = (
                        (base[..., 3] == 0)
                        & (pixels[..., 3] > 20)
                        & (rgb[..., 2] > rgb[..., 0] + 80)
                    )
                    self.assertGreater(np.count_nonzero(face), 0)
                    self.assertGreater(np.count_nonzero(band), 0)

    def test_shadow_protects_face_but_keeps_global_stroke_z_order(self):
        effect = FilterEffect('custom:identity')
        registry = _RuntimeRegistry({
            effect.filter_id: _runtime(
                effect, lambda rgba, _params, _context: rgba
            )
        })
        zero = ShadowEffect(
            offset=(0.0, 0.0),
            blur=0.0,
            spread=0.0,
            paint=SolidPaint((0, 0, 0)),
        )

        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            plain = self._public_pixels(self._item(
                TextEffectStack(effects=(effect,))
            ))
            opaque_face = plain[..., 3] == 255
            self.assertGreater(np.count_nonzero(opaque_face), 0)
            for shadow_type in ('drop', 'long'):
                shadow = replace(zero, shadow_type=shadow_type)
                for stack in (
                    TextEffectStack(effects=(effect, shadow)),
                    TextEffectStack(effects=(shadow, effect)),
                ):
                    with self.subTest(
                        shadow_type=shadow_type, stack=stack
                    ):
                        pixels = self._public_pixels(self._item(stack))
                        np.testing.assert_array_equal(
                            pixels[..., :3][opaque_face],
                            plain[..., :3][opaque_face],
                        )

                        shifted = replace(shadow, offset=(0.20, 0.12))
                        shifted_stack = replace(
                            stack,
                            effects=tuple(
                                shifted if value is shadow else value
                                for value in stack.effects
                            ),
                        )
                        shifted_pixels = self._public_pixels(
                            self._item(shifted_stack)
                        )
                        exterior = (
                            (plain[..., 3] == 0)
                            & (shifted_pixels[..., 3] > 0)
                        )
                        self.assertGreater(
                            np.count_nonzero(exterior), 0
                        )

            stroke = StrokeEffect(
                width=0.20,
                position='outside',
                paint=SolidPaint((20, 40, 240)),
            )
            shadow = replace(zero, paint=SolidPaint((20, 220, 40)))
            stroke_above = self._public_pixels(self._item(
                TextEffectStack(effects=(effect, stroke, shadow))
            ))
            shadow_above = self._public_pixels(self._item(
                TextEffectStack(effects=(effect, shadow, stroke))
            ))
            stroke_rgb = stroke_above[..., :3].astype(np.int16)
            band = (
                (plain[..., 3] == 0)
                & (stroke_above[..., 3] > 200)
                & (stroke_rgb[..., 2] > stroke_rgb[..., 1] + 80)
            )
            self.assertGreater(np.count_nonzero(band), 0)
            shadow_rgb = shadow_above[..., :3].astype(np.int16)
            self.assertTrue(np.all(
                shadow_rgb[..., 1][band] > shadow_rgb[..., 2][band]
            ))

    def test_generated_layer_splits_filter_groups_into_two_bridges(self):
        top = FilterEffect('custom:top')
        bottom = FilterEffect('custom:bottom')
        calls = []

        def runtime_for(effect):
            def apply(rgba, _params, _context):
                calls.append(effect.filter_id)
                return rgba

            return _runtime(effect, apply)

        registry = _RuntimeRegistry({
            top.filter_id: runtime_for(top),
            bottom.filter_id: runtime_for(bottom),
        })
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            item = self._item(TextEffectStack(effects=(
                top, GlowEffect(size=0.08), bottom
            )))
            renderer = item.effect_renderer
            renderer.release_caches()
            calls.clear()
            with patch.object(
                renderer,
                '_apply_filter_chain',
                wraps=renderer._apply_filter_chain,
            ) as bridge:
                renderer._render_effect_surface(renderer.boundingRect(), 1.0)

        self.assertEqual(calls, ['custom:bottom', 'custom:top'])
        self.assertEqual(bridge.call_count, 2)

    def test_reorder_toggle_and_exterior_stroke_dependency_miss_prefix_cache(self):
        effect = FilterEffect('custom:identity')
        runtime = _runtime(effect, lambda rgba, _params, _context: rgba)
        registry = _RuntimeRegistry({effect.filter_id: runtime})
        stroke = StrokeEffect(
            width=0.20,
            opacity=0.25,
            position='outside',
            paint=SolidPaint((20, 240, 40)),
        )
        shadow = GlowEffect(size=0.10, spread=0.03)
        initial = TextEffectStack(effects=(stroke, effect, shadow))
        item = self._item(initial)
        renderer = item.effect_renderer
        prefix_key = renderer._effect_cache_key_before_bottom_filter
        input_key = renderer._effect_cache_input_key

        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            renderer._render_effect_surface(renderer.boundingRect(), 1.0)
            preview_filter = replace(effect, params={'value': 0.25})
            baseline = TextEffectStack(effects=(
                stroke, preview_filter, shadow
            ))
            item.set_text_effects(baseline, preview=True)
            renderer.repaint_background()

            # The lower exterior prefix depends on every canonical Stroke,
            # including this one above the Filter.
            changed_stroke = replace(stroke, opacity=1.0)
            changed_stack = TextEffectStack(effects=(
                changed_stroke, preview_filter, shadow
            ))
            self.assertNotEqual(
                prefix_key(input_key(baseline)),
                prefix_key(input_key(changed_stack)),
            )
            with patch.object(
                renderer,
                '_render_pre_filter_effect_surface',
                wraps=renderer._render_pre_filter_effect_surface,
            ) as prefix:
                item.set_text_effects(changed_stack, preview=True)
                renderer.repaint_background()
            self.assertGreater(prefix.call_count, 0)
            item.set_text_effects(baseline, preview=True)
            renderer.repaint_background()

            # Moving the Filter changes the below-filter prefix even though
            # the non-Filter values themselves are identical.
            moved_stack = TextEffectStack(effects=(
                stroke, shadow, preview_filter
            ))
            self.assertNotEqual(
                prefix_key(input_key(baseline)),
                prefix_key(input_key(moved_stack)),
            )
            with patch.object(
                renderer,
                '_render_pre_filter_effect_surface',
                wraps=renderer._render_pre_filter_effect_surface,
            ) as prefix:
                item.set_text_effects(moved_stack, preview=True)
                renderer.repaint_background()
            self.assertGreater(prefix.call_count, 0)

            disabled = replace(preview_filter, enabled=False)
            disabled_stack = TextEffectStack(effects=(
                stroke, shadow, disabled
            ))
            self.assertNotEqual(
                prefix_key(input_key(moved_stack)),
                prefix_key(input_key(disabled_stack)),
            )
            with patch.object(
                renderer,
                '_render_pre_filter_effect_surface',
                wraps=renderer._render_pre_filter_effect_surface,
            ) as prefix:
                item.set_text_effects(disabled_stack, preview=True)
                renderer.repaint_background()
            self.assertGreater(prefix.call_count, 0)
            disabled_pixels = pixmap2ndarray(
                renderer._render_effect_surface(
                    renderer.boundingRect(), 0.5
                ),
                keep_alpha=True,
            )
        self.assertGreater(np.count_nonzero(disabled_pixels[:, :, :3]), 0)

        disabled_stroke = replace(stroke, enabled=False)
        alignment_item = self._item(TextEffectStack(effects=(
            disabled_stroke, effect
        )))
        alignment_renderer = alignment_item.effect_renderer
        disabled_alignment = TextEffectStack(effects=(
            disabled_stroke, effect
        ))
        enabled_alignment = TextEffectStack(effects=(stroke, effect))
        self.assertNotEqual(
            alignment_renderer._effect_cache_key_before_bottom_filter(
                alignment_renderer._effect_cache_input_key(disabled_alignment)
            ),
            alignment_renderer._effect_cache_key_before_bottom_filter(
                alignment_renderer._effect_cache_input_key(enabled_alignment)
            ),
        )
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.'
            'get_filter_registry',
            return_value=registry,
        ):
            alignment_renderer._render_effect_surface(
                alignment_renderer.boundingRect(), 1.0
            )
            preview_filter = replace(effect, params={'value': 0.25})
            alignment_item.set_text_effects(TextEffectStack(effects=(
                disabled_stroke, preview_filter
            )), preview=True)
            alignment_renderer.repaint_background()
            with patch.object(
                alignment_renderer,
                '_render_pre_filter_effect_surface',
                wraps=alignment_renderer._render_pre_filter_effect_surface,
            ) as prefix:
                alignment_item.set_text_effects(TextEffectStack(effects=(
                    stroke, preview_filter
                )), preview=True)
                alignment_renderer.repaint_background()
        self.assertGreater(prefix.call_count, 0)

    def test_filter_only_preview_reuses_upper_stroke_geometry(self):
        before = FilterEffect('builtin:noise', params={
            'amount': 0.2, 'mode': 'monochrome', 'seed': 1,
        })
        stroke = StrokeEffect(width=0.2, position='outside')
        item = self._item(TextEffectStack(effects=(stroke, before)))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        renderer._render_effect_surface(bounds, 1.0)

        first_preview = replace(before, params={
            'amount': 0.5, 'mode': 'monochrome', 'seed': 1,
        })
        item.set_text_effects(
            TextEffectStack(effects=(stroke, first_preview)), preview=True
        )
        renderer.repaint_background()
        second_preview = replace(before, params={
            'amount': 0.8, 'mode': 'monochrome', 'seed': 1,
        })
        with patch.object(
            renderer, 'paint_stroke', wraps=renderer.paint_stroke
        ) as paint_stroke, patch.object(
            renderer,
            '_capture_effect_source',
            wraps=renderer._capture_effect_source,
        ) as capture, patch.object(
            renderer,
            '_composite_generated_layer_batch',
            wraps=renderer._composite_generated_layer_batch,
        ) as upper_batch:
            item.set_text_effects(
                TextEffectStack(effects=(stroke, second_preview)),
                preview=True,
            )
            renderer.repaint_background()

        self.assertEqual(paint_stroke.call_count, 0)
        self.assertEqual(capture.call_count, 0)
        # The upper layer is intentionally outside the reusable filtered
        # prefix, so only its cached geometry is recolored/recomposited.
        self.assertGreater(upper_batch.call_count, 0)

    def test_tiled_render_reuses_one_immutable_validated_plan(self):
        effects = (
            FilterEffect('custom:first'),
            FilterEffect('custom:skipped'),
            FilterEffect('custom:last'),
        )
        halo_calls = {effect.filter_id: 0 for effect in effects}
        apply_calls = {effect.filter_id: 0 for effect in effects}
        immutable_params = []

        def runtime_for(effect, halo):
            def tile_halo(_params, _scale):
                halo_calls[effect.filter_id] += 1
                return halo

            def apply(rgba, params, _context):
                apply_calls[effect.filter_id] += 1
                try:
                    params['changed'] = True
                except TypeError:
                    immutable_params.append(True)
                else:
                    immutable_params.append(False)
                return rgba

            return FilterRuntime(
                SimpleNamespace(
                    filter_id=effect.filter_id,
                    schema_version=effect.schema_version,
                ),
                {'amount': 0.5},
                apply,
                tile_halo,
            )

        registry = _RuntimeRegistry({
            effects[0].filter_id: runtime_for(effects[0], 1),
            effects[1].filter_id: runtime_for(effects[1], 60),
            effects[2].filter_id: runtime_for(effects[2], 1),
        })
        item = self._item(TextEffectStack(effects=effects))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        target = renderer._new_effect_pixmap(1.0, bounds)
        painter = QPainter(target)
        painter.translate(-bounds.topLeft())
        try:
            with patch(
                'ballontranslator.ui.text_engine.effects.renderer.'
                'get_filter_registry',
                return_value=registry,
            ):
                renderer._draw_tiled_effects(
                    painter,
                    EffectRasterPlan('tiles', 1.0, 0, 0, 96),
                    bounds,
                )
        finally:
            painter.end()

        self.assertEqual(
            registry.resolve_calls,
            {effect.filter_id: 1 for effect in effects},
        )
        self.assertEqual(
            halo_calls,
            {effect.filter_id: 1 for effect in effects},
        )
        self.assertGreater(apply_calls['custom:first'], 1)
        self.assertEqual(apply_calls['custom:skipped'], 0)
        self.assertGreater(apply_calls['custom:last'], 1)
        self.assertTrue(immutable_params)
        self.assertTrue(all(immutable_params))

    def test_filter_only_preview_reuses_prefilter_and_canonical_capture(self):
        before = FilterEffect('builtin:noise', params={
            'amount': 0.2, 'mode': 'monochrome', 'seed': 1,
        })
        after = FilterEffect('builtin:noise', params={
            'amount': 0.8, 'mode': 'monochrome', 'seed': 1,
        })
        item = self._item(TextEffectStack(effects=(before,)))
        renderer = item.effect_renderer
        bounds = renderer.boundingRect()
        renderer._render_effect_surface(bounds, 1.0)
        item.set_text_effects(
            TextEffectStack(effects=(after,)), preview=True
        )
        renderer._render_effect_surface(bounds, 1.0)
        next_preview = FilterEffect('builtin:noise', params={
            'amount': 0.6, 'mode': 'monochrome', 'seed': 1,
        })

        with patch.object(
            renderer,
            '_render_pre_filter_effect_surface',
            wraps=renderer._render_pre_filter_effect_surface,
        ) as upstream, patch.object(
            renderer,
            '_capture_effect_source',
            wraps=renderer._capture_effect_source,
        ) as capture:
            item.set_text_effects(
                TextEffectStack(effects=(next_preview,)), preview=True
            )
            renderer._render_effect_surface(bounds, 1.0)
        self.assertEqual(upstream.call_count, 0)
        self.assertEqual(capture.call_count, 0)

    def test_plugin_failures_bypass_interactively_and_fail_strict(self):
        def expand_alpha(rgba, _params, _context):
            rgba[:, :, 3] = 255
            return rgba

        failures = (
            lambda _rgba, _params, _context: (_ for _ in ()).throw(
                RuntimeError('plugin failed')
            ),
            lambda rgba, _params, _context: rgba[:, :, :3],
            lambda rgba, _params, _context: rgba.astype(np.float32),
            lambda rgba, _params, _context: np.asfortranarray(rgba),
            expand_alpha,
        )
        for apply in failures:
            with self.subTest(apply=apply):
                effect = FilterEffect('custom:failure')
                item = self._item(TextEffectStack(effects=(effect,)))
                renderer = item.effect_renderer
                registry = _RuntimeRegistry({
                    effect.filter_id: _runtime(effect, apply)
                })
                with patch(
                    'ballontranslator.ui.text_engine.effects.renderer.'
                    'get_filter_registry',
                    return_value=registry,
                ):
                    interactive = renderer._render_effect_surface(
                        renderer.boundingRect(), 1.0
                    )
                    self.assertGreater(np.count_nonzero(
                        pixmap2ndarray(interactive, keep_alpha=True)[:, :, 3]
                    ), 0)
                    renderer.set_export_effect_render(True)
                    try:
                        with self.assertRaises(EffectRasterAllocationError):
                            renderer._render_effect_surface(
                                renderer.boundingRect(), 1.0
                            )
                    finally:
                        renderer.set_export_effect_render(False)

    def test_declared_alpha_expansion_is_limited_to_its_halo(self):
        effect = FilterEffect('custom:expand')
        item = self._item(TextEffectStack(effects=(effect,)))
        renderer = item.effect_renderer
        source_pixels = np.zeros((11, 13, 4), dtype=np.uint8)
        source_pixels[4:7, 5:8] = (30, 80, 160, 255)
        source = ndarray2pixmap(source_pixels)

        def expand_one(rgba, _params, _context):
            rgba[:, :, 3] = cv2.dilate(
                rgba[:, :, 3], np.ones((3, 3), dtype=np.uint8)
            )
            return rgba

        runtime = _runtime(
            effect, expand_one, halo=1, expands_alpha=True
        )
        result = renderer._apply_filter_chain(
            source,
            QRectF(0, 0, source.width(), source.height()),
            1.0,
            frozenset(),
            ((0, effect, runtime, 1),),
        )
        result_alpha = pixmap2ndarray(result, keep_alpha=True)[:, :, 3]
        self.assertGreater(
            np.count_nonzero(result_alpha),
            np.count_nonzero(source_pixels[:, :, 3]),
        )

        def expand_everywhere(rgba, _params, _context):
            rgba[:, :, 3] = 255
            return rgba

        invalid_runtime = _runtime(
            effect, expand_everywhere, halo=1, expands_alpha=True
        )
        with patch.object(renderer, '_filter_failure') as failure:
            bypassed = renderer._apply_filter_chain(
                source,
                QRectF(0, 0, source.width(), source.height()),
                1.0,
                frozenset(),
                ((0, effect, invalid_runtime, 1),),
            )
        self.assertEqual(failure.call_count, 1)
        np.testing.assert_array_equal(
            pixmap2ndarray(bypassed, keep_alpha=True), source_pixels
        )

    def test_rough_edge_adds_padding_and_opaque_jagged_growth(self):
        effect = FilterEffect('builtin:rough_edge', params={
            'amount': 1.0, 'size': 2.7,
            'hardness': 0.8, 'seed': 17,
        })
        item = self._item(TextEffectStack(effects=(effect,)))
        renderer = item.effect_renderer
        renderer._update_effect_padding()
        # The 0.5x preview rounds 2.7 logical pixels to a two-pixel halo.
        self.assertGreaterEqual(item.padding(), 4.0)
        bounds = renderer.boundingRect()
        upstream = pixmap2ndarray(
            renderer._render_pre_filter_effect_surface(bounds, 1.0),
            keep_alpha=True,
        )
        result = pixmap2ndarray(
            renderer._render_pre_mask_effect_surface(bounds, 1.0),
            keep_alpha=True,
        )
        expanded = (upstream[:, :, 3] == 0) & (result[:, :, 3] > 0)
        self.assertGreater(np.count_nonzero(expanded), 0)
        self.assertGreater(int(result[:, :, 3][expanded].max()), 200)

    def test_invalid_halos_bypass_interactively_and_fail_strict(self):
        effect = FilterEffect('custom:bad_halo')
        for halo in (None, -1, float('nan'), 513):
            with self.subTest(halo=halo):
                item = self._item(TextEffectStack(effects=(effect,)))
                renderer = item.effect_renderer
                registry = _RuntimeRegistry({
                    effect.filter_id: _runtime(
                        effect,
                        lambda rgba, _params, _context: rgba,
                        halo,
                    )
                })
                with patch(
                    'ballontranslator.ui.text_engine.effects.renderer.'
                    'get_filter_registry',
                    return_value=registry,
                ):
                    interactive = renderer._render_effect_surface(
                        renderer.boundingRect(), 1.0
                    )
                    self.assertGreater(np.count_nonzero(
                        pixmap2ndarray(interactive, keep_alpha=True)[:, :, 3]
                    ), 0)
                    renderer.set_export_effect_render(True)
                    try:
                        with self.assertRaises(EffectRasterAllocationError):
                            renderer._render_effect_surface(
                                renderer.boundingRect(), 1.0
                            )
                    finally:
                        renderer.set_export_effect_render(False)

    def test_interactive_filter_warnings_are_stable_once_and_capped(self):
        renderer = self._item(TextEffectStack()).effect_renderer
        repeated = FilterEffect('custom:repeated')
        with patch(
            'ballontranslator.ui.text_engine.effects.renderer.LOGGER.warning'
        ) as warning:
            renderer._filter_failure(
                repeated, 'apply', ValueError('message one')
            )
            renderer._filter_failure(
                repeated, 'apply', ValueError('message two')
            )
            for index in range(100):
                renderer._filter_failure(
                    FilterEffect(f'custom:{index}'),
                    'resolution',
                    ValueError(f'changing message {index}'),
                )

        self.assertEqual(warning.call_count, 64)
        self.assertEqual(len(renderer._filter_warnings), 64)
        renderer.set_export_effect_render(True)
        try:
            with self.assertRaises(EffectRasterAllocationError):
                renderer._filter_failure(
                    repeated, 'apply', ValueError('strict failure')
                )
        finally:
            renderer.set_export_effect_render(False)

    def test_filters_stay_before_interaction_feedback_while_editing_h_and_v(self):
        effect = FilterEffect('custom:editing')
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                order = []

                def apply(rgba, _params, _context):
                    order.append('filter')
                    rgba[:, :, 0] //= 2
                    return rgba

                item = self._item(
                    TextEffectStack(effects=(effect,)), vertical=vertical
                )
                scene = QGraphicsScene()
                view = QGraphicsView(scene)
                scene.addItem(item)
                view.show()
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(0)
                cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
                item.setTextCursor(cursor)
                registry = _RuntimeRegistry({
                    effect.filter_id: _runtime(effect, apply)
                })

                def cursor_paint(*_args, **_kwargs):
                    order.append('interaction')

                with patch(
                    'ballontranslator.ui.text_engine.effects.renderer.'
                    'get_filter_registry',
                    return_value=registry,
                ), patch.object(
                    item.geometry_controller,
                    'paint_deferred_cursor',
                    side_effect=cursor_paint,
                ):
                    image = QImage(
                        420, 260, QImage.Format.Format_ARGB32_Premultiplied
                    )
                    image.fill(QColor(0, 0, 0, 0))
                    painter = QPainter(image)
                    scene.render(painter)
                    painter.end()
                self.assertIn('filter', order)
                self.assertIn('interaction', order)
                self.assertLess(order.index('filter'), order.index('interaction'))
                item.endEdit()
                view.deleteLater()

    def test_empty_missing_filter_is_strictly_export_eligible_in_canvas(self):
        project = ProjImgTrans()
        project.inpainted_array = np.zeros((220, 360, 3), dtype=np.uint8)
        canvas = Canvas()
        canvas.imgtrans_proj = project
        canvas.baseLayer.setRect(QRectF(0, 0, 360, 220))
        item = self._item(
            TextEffectStack(effects=(FilterEffect('missing:strict'),)),
            text='',
        )
        item.setParentItem(canvas.textLayer)
        try:
            self.assertTrue(item.effect_renderer.has_raster_effects())
            with self.assertRaises(EffectRasterAllocationError):
                canvas.render_result_img()
        finally:
            canvas.deleteLater()
            self.app.processEvents()

    def test_filter_output_survives_nonlinear_transform(self):
        item = self._item(TextEffectStack(effects=(FilterEffect(
            'builtin:rough_edge', params={
                'amount': 0.8, 'size': 3.0, 'hardness': 0.5, 'seed': 11,
            }
        ),)))
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        scene = QGraphicsScene()
        scene.addItem(item)
        image = QImage(420, 260, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(painter)
        painter.end()
        self.assertGreater(
            np.count_nonzero(pixmap2ndarray(image, keep_alpha=True)[:, :, 3]),
            0,
        )


if __name__ == '__main__':
    unittest.main()
