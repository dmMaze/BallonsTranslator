import os
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QPixmap, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene, QGraphicsView

from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.annotations import (
    apply_emphasis,
    apply_ruby,
    apply_text_combine_upright,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering.alpha_mask import (
    render_text_alpha_mask,
)
from ballontranslator.ui.text_engine.rendering.raster import (
    EFFECT_RASTER_GUARD,
    EffectRasterAllocationError,
    EffectRasterPlan,
)
from ballontranslator.utils.fontformat import (
    FontFormat,
    ProjectiveTextTransform,
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils.text_alpha_mask import (
    AlphaBrushStroke,
    TextAlphaMask,
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


def _mask_pixels(mask, rect, scale=1.0):
    image = render_text_alpha_mask(mask, rect, QPointF(), scale)
    return pixmap2ndarray(QPixmap.fromImage(image), keep_alpha=True)[..., 3]


class TextAlphaMaskRasterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_erase_restore_order_and_outside_points(self):
        rect = QRectF(-20, -20, 80, 80)
        erase_then_restore = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 24, ((-10, 10), (30, 10))),
            AlphaBrushStroke('restore', 8, ((10, 10),)),
        ))
        restore_then_erase = TextAlphaMask(strokes=tuple(
            reversed(erase_then_restore.strokes)
        ))

        first = _mask_pixels(erase_then_restore, rect)
        second = _mask_pixels(restore_then_erase, rect)
        # Item point (10, 10) maps to array point (30, 30).
        self.assertEqual(first[30, 30], 255)
        self.assertEqual(second[30, 30], 0)
        # The negative item-local sample remains valid inside effect overflow.
        self.assertEqual(first[30, 10], 0)

    def test_raster_is_translation_invariant_at_two_scales(self):
        mask = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 9, ((-4.5, 3.25), (44.5, 28.75))),
            AlphaBrushStroke('restore', 3, ((20.25, 16.5),)),
        ))
        for scale in (1.0, 2.0):
            with self.subTest(scale=scale):
                full = _mask_pixels(mask, QRectF(-16, -12, 80, 64), scale)
                crop = _mask_pixels(mask, QRectF(0, 0, 32, 32), scale)
                offset_x = int(16 * scale)
                offset_y = int(12 * scale)
                np.testing.assert_array_equal(
                    crop,
                    full[
                        offset_y:offset_y + int(32 * scale),
                        offset_x:offset_x + int(32 * scale),
                    ],
                )

        with self.assertRaises(EffectRasterAllocationError):
            render_text_alpha_mask(
                mask, QRectF(0, 0, 9000, 1), QPointF(), 1.0
            )


class TextAlphaMaskRendererTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @classmethod
    def _item(
        cls,
        stack=TextEffectStack(),
        mask=None,
        vertical=False,
        text='Masked text',
    ):
        block = TextBlock([0, 0, 320, 180], text_alpha_mask=mask)
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = text
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
            440, 280, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, 440, 280),
            QRectF(-40, -40, 440, 280),
        )
        painter.end()
        if owns_scene:
            scene.removeItem(item)
        return pixmap2ndarray(image, keep_alpha=True)

    @staticmethod
    def _erase_all() -> TextAlphaMask:
        return TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 1000, ((160, 90),)),
        ))

    @staticmethod
    def _partial_mask() -> TextAlphaMask:
        return TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 26, ((-18, 20), (250, 130))),
            AlphaBrushStroke('restore', 7, ((120, 82), (160, 82))),
        ))

    def test_fill_only_mask_owns_surface_without_growing_bounds(self):
        item = self._item()
        before = QRectF(item.boundingRect())
        unmasked = self._render(item)

        mask = self._erase_all()
        self.assertTrue(item.set_text_alpha_mask(mask))
        self.assertEqual(item.boundingRect(), before)
        self.assertTrue(item.effect_renderer.has_raster_effects())
        self.assertIsNotNone(item.effect_renderer.background_pixmap)
        pixmap_key = item.effect_renderer.background_pixmap.cacheKey()
        self.assertFalse(item.set_text_alpha_mask(self._erase_all()))
        self.assertEqual(
            item.effect_renderer.background_pixmap.cacheKey(), pixmap_key
        )
        self.assertGreater(np.count_nonzero(unmasked[..., 3]), 0)
        self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)

        self.assertTrue(item.set_text_alpha_mask(None))
        self.assertFalse(item.effect_renderer.has_raster_effects())
        self.assertIsNone(item.effect_renderer._effect_raster_state)
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

        neutral = TextAlphaMask()
        self.assertTrue(item.set_text_alpha_mask(neutral))
        self.assertFalse(item.set_text_alpha_mask(neutral))
        with self.assertRaises(TypeError):
            item.set_text_alpha_mask({})

    def test_applying_font_style_does_not_replace_block_mask(self):
        mask = self._partial_mask()
        item = self._item(mask=mask)

        item.set_fontformat(FontFormat(font_size=32))

        self.assertIs(item.blk.text_alpha_mask, mask)

    def test_mask_clips_stroke_drop_long_and_hollow_output(self):
        stacks = (
            TextEffectStack(effects=(
                ShadowEffect(
                    paint=SolidPaint((0, 255, 0)),
                    offset=(-0.35, 0),
                    blur=0.04,
                ),
                GlowEffect(
                    paint=SolidPaint((255, 255, 0)),
                    size=0.08,
                    spread=0.03,
                ),
                StrokeEffect(
                    width=0.18,
                    position='outside',
                    paint=SolidPaint((0, 0, 255)),
                ),
                GradientOverlayEffect(
                    paint=LinearGradientPaint(angle=90.0)
                ),
                GlowEffect(
                    glow_type='inner',
                    paint=SolidPaint((255, 255, 0)),
                    size=0.06,
                ),
            )),
            TextEffectStack(effects=(
                ShadowEffect(
                    shadow_type='long',
                    paint=SolidPaint((0, 255, 0)),
                    offset=(-0.5, 0.2),
                ),
                GlowEffect(size=0.08, spread=0.03),
                StrokeEffect(
                    width=0.18,
                    position='inside',
                    paint=SolidPaint((0, 0, 255)),
                ),
                HollowEffect(),
            )),
        )
        for stack in stacks:
            with self.subTest(stack=stack):
                item = self._item(stack)
                unmasked = self._render(item)
                bounds = QRectF(item.boundingRect())
                item.set_text_alpha_mask(self._partial_mask())
                masked = self._render(item)
                self.assertEqual(item.boundingRect(), bounds)
                self.assertGreater(np.count_nonzero(unmasked[..., 3]), 0)
                self.assertFalse(np.array_equal(unmasked, masked))
                self.assertLess(
                    np.count_nonzero(masked[..., 3]),
                    np.count_nonzero(unmasked[..., 3]),
                )

    def test_full_and_forced_tiles_match_at_multiple_scales(self):
        stacks = (
            TextEffectStack(effects=(
                ShadowEffect(offset=(0.18, 0.12), blur=0.08, spread=0.04),
                StrokeEffect(
                    width=0.12,
                    position='outside',
                    paint=LinearGradientPaint(
                        stops=(
                            GradientStop(0.0, (255, 0, 0), 0.3),
                            GradientStop(1.0, (0, 0, 255), 1.0),
                        ),
                        angle=28.0,
                        scale=1.7,
                    ),
                ),
                ShadowEffect(
                    shadow_type='inner', offset=(0.08, 0.04), blur=0.06
                ),
            )),
            TextEffectStack(effects=(
                ShadowEffect(shadow_type='long', offset=(0.30, 0.22)),
                StrokeEffect(width=0.12, position='inside'),
                HollowEffect(),
            )),
        )
        for stack in stacks:
            for scale in (1.0, 2.0):
                with self.subTest(stack=stack, scale=scale):
                    item = self._item(stack, self._partial_mask())
                    renderer = item.effect_renderer
                    bounds = renderer.boundingRect()
                    full = renderer._render_effect_surface(bounds, scale)
                    tiled = renderer._new_effect_pixmap(scale, bounds)
                    painter = QPainter(tiled)
                    painter.translate(-bounds.topLeft())
                    renderer.tile_cache.clear()
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

    def test_strict_tiled_hollow_stroke_never_bypasses_active_mask(self):
        hollow_only = self._item(
            TextEffectStack(effects=(HollowEffect(),)),
            self._erase_all(),
        )
        self.assertEqual(
            hollow_only.effect_renderer._effect_flags(), (False, False)
        )
        self.assertIsNone(hollow_only.effect_renderer._effect_raster_state)

        item = self._item(
            TextEffectStack(effects=(
                StrokeEffect(width=1.5),
                HollowEffect(),
            )),
            self._erase_all(),
        )
        renderer = item.effect_renderer
        renderer.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'EFFECT_TILE_MAX_EDGE',
                32,
            ):
                stroke_overlap = (
                    renderer._stroke_outset() + EFFECT_RASTER_GUARD
                )
                self.assertGreaterEqual(
                    2 * int(np.ceil(stroke_overlap)), 32
                )
                renderer.force_tiles = True
                pixels = self._render(item)
            self.assertEqual(np.count_nonzero(pixels[..., 3]), 0)
            self.assertIsInstance(
                item.export_effect_error, EffectRasterAllocationError
            )
            self.assertFalse(renderer.direct_stroke)
        finally:
            renderer.set_export_effect_render(False)

    def test_mask_covers_vertical_annotations_and_distorted_glyphs(self):
        item = self._item(
            TextEffectStack(effects=(
                StrokeEffect(
                    width=0.12,
                    position='outside',
                    paint=LinearGradientPaint(angle=90.0),
                ),
            )),
            vertical=True,
            text='東京12',
        )
        cursor = QTextCursor(item.document())
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_ruby(cursor, 'group', 'とうきょう')
        apply_emphasis(cursor, 'filled sesame', 'under left')
        cursor.setPosition(2)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)
        item.layout.reLayoutEverything()
        item.repaint_background()
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

        item.set_text_transform(TextTransformStack(glyph_slant_angle=12))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)
        item.set_text_transform(TextTransformStack(
            (SineTextTransform(amplitude_x=0.2),),
            glyph_slant_angle=12,
        ))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)
        item.set_text_alpha_mask(self._erase_all())
        self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)

        projective = self._item(
            TextEffectStack(effects=(
                StrokeEffect(
                    width=0.12,
                    position='inside',
                    paint=LinearGradientPaint(angle=215.0, scale=1.3),
                ),
            )),
            text='Projective mask',
        )
        projective.set_text_transform(TextTransformStack((
            ProjectiveTextTransform(rotation_y=25, perspective=0.3),
        )))
        self.assertGreater(
            np.count_nonzero(self._render(projective)[..., 3]), 0
        )
        projective.set_text_alpha_mask(self._erase_all())
        self.assertEqual(
            np.count_nonzero(self._render(projective)[..., 3]), 0
        )

    def test_editing_feedback_is_unmasked_but_export_has_no_controls(self):
        item = self._item(mask=self._erase_all())
        scene = QGraphicsScene()
        view = QGraphicsView(scene)
        view.show()
        scene.addItem(item)
        self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)

        item.startEdit()
        view.setFocus()
        item.setFocus()
        self.app.processEvents()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(5, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)

        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)
        item.set_export_effect_render(True)
        try:
            self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)
        finally:
            item.set_export_effect_render(False)
            view.close()

    def test_cache_generation_preview_isolation_and_reshape_rebuild(self):
        stack = TextEffectStack(effects=(StrokeEffect(
            width=0.12, paint=LinearGradientPaint()
        ),))
        preview = TextEffectStack(effects=(StrokeEffect(
            width=0.24, paint=LinearGradientPaint(angle=120.0)
        ),))
        item = self._item(stack, self._partial_mask())
        renderer = item.effect_renderer
        key = renderer._effect_cache_input_key()
        self.assertIsInstance(key[1], int)
        self.assertNotIn(item.blk.text_alpha_mask, key)

        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render:
            self._render(item)
            self._render(item)
            self.assertEqual(render.call_count, 0)
            item.set_text_effects(preview, preview=True)
            self.assertEqual(render.call_count, 1)
            committed = renderer._effect_raster_state
            item.set_text_alpha_mask(self._erase_all())
            self.assertEqual(render.call_count, 2)
            self.assertTrue(committed.cache_dirty)
            item.clear_text_effect_preview()
            self.assertEqual(render.call_count, 3)

            item.startReshape()
            item.set_text_alpha_mask(self._partial_mask())
            item.repaint_background()
            self.assertEqual(render.call_count, 3)
            item.endReshape()
            self.assertEqual(render.call_count, 4)

    def test_mask_preview_reuses_expensive_stroke_shadow_composite(self):
        stack = TextEffectStack(effects=(
            ShadowEffect(offset=(0.18, 0.12), blur=0.08, spread=0.04),
            StrokeEffect(width=0.12, paint=LinearGradientPaint(stops=(
                GradientStop(0.0, (255, 0, 0), 0.4),
                GradientStop(1.0, (0, 0, 255), 1.0),
            ))),
        ))
        item = self._item(stack, self._partial_mask())
        renderer = item.effect_renderer
        renderer._effect_raster_state.pre_mask_cache.clear()
        masks = (
            TextAlphaMask(strokes=(
                AlphaBrushStroke('erase', 18, ((20, 20),)),
            )),
            TextAlphaMask(strokes=(
                AlphaBrushStroke('erase', 18, ((20, 20), (80, 60))),
                AlphaBrushStroke('restore', 6, ((40, 35),)),
            )),
        )
        with patch.object(
            renderer,
            '_render_pre_mask_effect_surface',
            wraps=renderer._render_pre_mask_effect_surface,
        ) as upstream:
            item.set_text_alpha_mask(masks[0], preview=True)
            self.assertEqual(upstream.call_count, 1)
            item.set_text_alpha_mask(masks[1], preview=True)
            self.assertEqual(upstream.call_count, 1)
        self.assertEqual(item.blk.text_alpha_mask, self._partial_mask())

    def test_mask_preview_reuses_pre_mask_tile_for_same_visible_region(self):
        stack = TextEffectStack(effects=(
            ShadowEffect(offset=(0.18, 0.12), blur=0.08),
            StrokeEffect(width=0.12, paint=LinearGradientPaint(angle=45.0)),
        ))
        item = self._item(stack, self._partial_mask())
        renderer = item.effect_renderer
        first = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 18, ((20, 20),)),
        ))
        second = TextAlphaMask(strokes=(
            AlphaBrushStroke('erase', 18, ((20, 20), (60, 40))),
        ))
        item.set_text_alpha_mask(first, preview=True)
        bounds = renderer.boundingRect()
        visible = QRectF(bounds.left(), bounds.top(), 48, 48)
        target = renderer._new_effect_pixmap(1.0, bounds)
        painter = QPainter(target)
        painter.translate(-bounds.topLeft())
        try:
            with patch.object(
                renderer,
                '_render_pre_mask_effect_surface',
                wraps=renderer._render_pre_mask_effect_surface,
            ) as upstream:
                renderer._draw_tiled_effects(
                    painter,
                    EffectRasterPlan('tiles', 1.0, 0, 0, 128),
                    visible,
                )
                calls = upstream.call_count
                self.assertGreater(calls, 0)
                item.set_text_alpha_mask(second, preview=True)
                renderer._draw_tiled_effects(
                    painter,
                    EffectRasterPlan('tiles', 1.0, 0, 0, 128),
                    visible,
                )
                self.assertEqual(upstream.call_count, calls)
        finally:
            painter.end()

    def _assert_canonical_cache_not_preview_owned(
        self, item: TextBlkItem, scratch
    ) -> None:
        renderer = item.effect_renderer
        committed = renderer._effect_raster_state
        self.assertIsNot(committed, scratch)
        if committed is not None and committed.cache_input_key is not None:
            self.assertEqual(
                committed.cache_input_key[0],
                item.blk.fontformat.text_effects.effects,
            )
            self.assertGreaterEqual(committed.cache_input_key[1], 0)

    def _assert_active_scratch_is_current(self, item: TextBlkItem) -> None:
        renderer = item.effect_renderer
        scratch = renderer._preview_effect_raster_state
        self.assertIs(renderer._peek_raster_state(), scratch)
        self.assertIsNotNone(scratch)
        self.assertFalse(scratch.cache_dirty)
        self.assertEqual(
            scratch.cache_input_key, renderer._effect_cache_input_key()
        )

    def test_effect_mask_preview_overlap_commit_orders_stay_isolated(self):
        canonical = TextEffectStack(effects=(StrokeEffect(width=0.10),))
        effect_preview = TextEffectStack(effects=(
            ShadowEffect(blur=0.06, offset=(0.12, 0.08)),
            StrokeEffect(width=0.22),
        ))
        mask_preview = self._erase_all()

        mask_first = self._item(canonical, self._partial_mask())
        mask_first.set_text_transform(TextTransformStack((SineTextTransform(),)))
        self._render(mask_first)
        renderer = mask_first.effect_renderer
        mask_first.set_text_alpha_mask(mask_preview, preview=True)
        retained = renderer.geometry_controller._retained_effect_preview_surface
        self.assertIsNotNone(retained)
        mask_first.set_text_effects(effect_preview, preview=True)
        scratch = renderer._preview_effect_raster_state
        mask_first.set_text_alpha_mask(mask_preview)
        self._assert_canonical_cache_not_preview_owned(mask_first, scratch)
        self.assertEqual(mask_first.effective_text_effects(), effect_preview)
        self.assertEqual(mask_first.effective_text_alpha_mask(), mask_preview)
        self._assert_active_scratch_is_current(mask_first)
        self.assertIsNone(
            renderer.geometry_controller._retained_effect_preview_surface
        )
        mask_first.clear_text_effect_preview()
        self.assertEqual(mask_first.effective_text_effects(), canonical)

        effect_first = self._item(canonical, self._partial_mask())
        renderer = effect_first.effect_renderer
        effect_first.set_text_effects(effect_preview, preview=True)
        effect_first.set_text_alpha_mask(mask_preview, preview=True)
        scratch = renderer._preview_effect_raster_state
        effect_first.set_text_effects(effect_preview)
        self._assert_canonical_cache_not_preview_owned(effect_first, scratch)
        self.assertEqual(effect_first.effective_text_effects(), effect_preview)
        self.assertEqual(effect_first.effective_text_alpha_mask(), mask_preview)
        self._assert_active_scratch_is_current(effect_first)
        effect_first.clear_text_alpha_mask_preview()
        self.assertEqual(
            effect_first.effective_text_alpha_mask(), self._partial_mask()
        )

    def test_effect_mask_preview_overlap_cancel_paths_keep_remaining_preview(self):
        canonical = TextEffectStack(effects=(StrokeEffect(width=0.10),))
        effect_preview = TextEffectStack(effects=(StrokeEffect(width=0.24),))
        mask_preview = self._erase_all()

        item = self._item(canonical, self._partial_mask())
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        self._render(item)
        renderer = item.effect_renderer
        item.set_text_alpha_mask(mask_preview, preview=True)
        item.set_text_effects(effect_preview, preview=True)
        item.clear_text_alpha_mask_preview()
        self.assertEqual(item.effective_text_effects(), effect_preview)
        self.assertEqual(item.effective_text_alpha_mask(), self._partial_mask())
        self._assert_active_scratch_is_current(item)
        self.assertIsNotNone(
            renderer.geometry_controller._retained_effect_preview_surface
        )
        item.clear_text_effect_preview()
        self.assertIsNone(
            renderer.geometry_controller._retained_effect_preview_surface
        )

        item = self._item(canonical, self._partial_mask())
        renderer = item.effect_renderer
        item.set_text_effects(effect_preview, preview=True)
        item.set_text_alpha_mask(mask_preview, preview=True)
        item.clear_text_effect_preview()
        self.assertEqual(item.effective_text_effects(), canonical)
        self.assertEqual(item.effective_text_alpha_mask(), mask_preview)
        self._assert_active_scratch_is_current(item)
        scratch = renderer._preview_effect_raster_state
        item.set_text_alpha_mask(mask_preview)
        self.assertIs(renderer._effect_raster_state, scratch)

    def test_overall_opacity_overlap_keeps_mask_scratch_and_upstream(self):
        canonical = TextEffectStack(effects=(StrokeEffect(width=0.12),))
        opacity_preview = replace(canonical, overall_opacity=0.45)
        item = self._item(canonical, self._partial_mask())
        renderer = item.effect_renderer
        item.set_text_alpha_mask(self._erase_all(), preview=True)
        scratch = renderer._preview_effect_raster_state
        pre_mask = dict(scratch.pre_mask_cache)
        with patch.object(
            renderer,
            '_render_pre_mask_effect_surface',
            wraps=renderer._render_pre_mask_effect_surface,
        ) as upstream:
            item.set_text_effects(opacity_preview, preview=True)
            item.clear_text_effect_preview()
            self.assertIs(renderer._preview_effect_raster_state, scratch)
            self.assertEqual(scratch.pre_mask_cache, pre_mask)
            self.assertEqual(upstream.call_count, 0)

            item.set_text_effects(opacity_preview, preview=True)
            item.set_text_effects(opacity_preview)
            self.assertIs(renderer._preview_effect_raster_state, scratch)
            self.assertEqual(scratch.pre_mask_cache, pre_mask)
            self.assertEqual(upstream.call_count, 0)
        self.assertAlmostEqual(item.opacity(), 0.45)

    def test_reshape_omits_mask_but_strict_export_does_not(self):
        item = self._item(mask=self._erase_all())
        self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)

        item.startReshape()
        self.assertGreater(np.count_nonzero(self._render(item)[..., 3]), 0)
        item.set_export_effect_render(True)
        try:
            self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)
        finally:
            item.set_export_effect_render(False)
            item.endReshape()
        self.assertEqual(np.count_nonzero(self._render(item)[..., 3]), 0)

    def test_interactive_mask_failure_is_visible_and_export_is_strict(self):
        mask = self._erase_all()
        failure = RuntimeError('mock alpha mask bridge failure')
        item = self._item(mask=mask)
        renderer = item.effect_renderer
        self.assertTrue(renderer._completed_foreground_ready())
        with patch(
            'ballontranslator.ui.text_engine.effect_renderer.'
            'render_text_alpha_mask',
            side_effect=failure,
        ):
            renderer.repaint_background(2.0)
            pixels = self._render(item)
        self.assertGreater(np.count_nonzero(pixels[..., 3]), 0)
        self.assertFalse(renderer._completed_foreground_ready())

        hollow = self._item(TextEffectStack(effects=(
            StrokeEffect(
                width=0.2,
                paint=SolidPaint((0, 0, 255)),
                position='center',
            ),
            HollowEffect(),
        )))
        with patch(
            'ballontranslator.ui.text_engine.effect_renderer.'
            'render_text_alpha_mask',
            side_effect=failure,
        ):
            hollow.set_text_alpha_mask(mask)
            hollow_pixels = self._render(hollow)
        visible = hollow_pixels[..., 3] > 100
        self.assertGreater(np.count_nonzero(visible), 0)
        self.assertFalse(np.any(
            (hollow_pixels[..., 0] > 180)
            & (hollow_pixels[..., 1] < 80)
            & visible
        ))

        renderer.set_export_effect_render(True)
        try:
            with patch(
                'ballontranslator.ui.text_engine.effect_renderer.'
                'render_text_alpha_mask',
                side_effect=failure,
            ):
                self._render(item)
            self.assertIsInstance(
                item.export_effect_error, EffectRasterAllocationError
            )
        finally:
            renderer.set_export_effect_render(False)

    def test_canvas_export_selects_mask_only_items_for_strict_rendering(self):
        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(
            inpainted_array=np.zeros((220, 360, 3), dtype=np.uint8),
            inpainted_valid=False,
        )
        canvas.baseLayer.setRect(QRectF(0, 0, 360, 220))
        masked = self._item(mask=self._partial_mask())
        neutral = self._item(mask=TextAlphaMask())
        masked.setParentItem(canvas.textLayer)
        neutral.setParentItem(canvas.textLayer)
        try:
            with patch.object(
                masked,
                'set_export_effect_render',
                wraps=masked.set_export_effect_render,
            ) as masked_export, patch.object(
                neutral,
                'set_export_effect_render',
                wraps=neutral.set_export_effect_render,
            ) as neutral_export:
                image = canvas.render_result_img()
            self.assertFalse(image.isNull())
            self.assertEqual(
                [call.args[0] for call in masked_export.call_args_list],
                [True, False],
            )
            neutral_export.assert_not_called()
        finally:
            canvas.deleteLater()
            self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
