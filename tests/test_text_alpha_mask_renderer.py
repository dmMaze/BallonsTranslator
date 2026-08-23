import os
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
    HollowEffect,
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
                    color=(0, 255, 0), offset=(-0.35, 0), blur=0.04
                ),
                StrokeEffect(width=0.18, paint=SolidPaint((0, 0, 255))),
            )),
            TextEffectStack(effects=(
                ShadowEffect(
                    shadow_type='long', color=(0, 255, 0), offset=(-0.5, 0.2)
                ),
                StrokeEffect(width=0.18, paint=SolidPaint((0, 0, 255))),
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
                StrokeEffect(width=0.12),
                ShadowEffect(
                    shadow_type='inner', offset=(0.08, 0.04), blur=0.06
                ),
            )),
            TextEffectStack(effects=(
                ShadowEffect(shadow_type='long', offset=(0.30, 0.22)),
                StrokeEffect(width=0.12),
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
        item = self._item(vertical=True, text='東京12')
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

        projective = self._item(text='Projective mask')
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
        stack = TextEffectStack(effects=(StrokeEffect(width=0.12),))
        preview = TextEffectStack(effects=(StrokeEffect(width=0.24),))
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
            StrokeEffect(width=0.2, paint=SolidPaint((0, 0, 255))),
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
