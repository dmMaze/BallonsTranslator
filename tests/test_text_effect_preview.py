import json
import math
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage, QPainter, QTextCursor
from qtpy.QtWidgets import QApplication, QGraphicsScene

try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.editing.commands import (
    SetTextEffectStackCommand,
)
from ballontranslator.ui.text_engine.effects.edit_session import (
    TextEffectEditSession,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.fontformat import (
    ProjectiveTextTransform,
    SineTextTransform,
    TextTransformStack,
)
from ballontranslator.utils import config as C
from ballontranslator.utils.proj_imgtrans import TextBlkEncoder
from ballontranslator.utils.text_effects import (
    GlowEffect,
    HollowEffect,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextFillEffect,
    TextEffectStack,
    TEXT_EFFECT_BLEND_MODES,
)
from ballontranslator.utils.textblock import TextBlock


class _UndoCanvas:
    def __init__(self) -> None:
        self.stack = QUndoStack()

    def push_undo_command(self, command) -> None:
        self.stack.push(command)


class TextEffectPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _stack(
        width=0.12,
        color=(20, 40, 60),
        opacity=1.0,
        position='center',
    ):
        return TextEffectStack(
            overall_opacity=opacity,
            effects=(
                StrokeEffect(
                    width=width,
                    paint=SolidPaint(color),
                    position=position,
                ),
            ),
        )

    @classmethod
    def _item(cls, vertical=False, stack=None):
        block = TextBlock([0, 0, 320, 180])
        block._bounding_rect = [0, 0, 320, 180]
        block.translation = 'Effect preview'
        block.vertical = vertical
        if stack is not None:
            block.fontformat.text_effects = stack
        return TextBlkItem(block, 1)

    @staticmethod
    def _render_scene(scene, scale=1.0):
        width = math.ceil(420 * scale)
        height = math.ceil(260 * scale)
        image = QImage(
            width, height, QImage.Format.Format_ARGB32_Premultiplied
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        scene.render(
            painter,
            QRectF(0, 0, width, height),
            QRectF(-30, -30, 420, 260),
        )
        painter.end()
        return image

    @staticmethod
    def _alpha_bounds(image):
        alpha = pixmap2ndarray(image, keep_alpha=True)[..., 3]
        rows, columns = np.nonzero(alpha)
        return (
            columns.min(), rows.min(), columns.max(), rows.max()
        )

    def test_preview_is_live_but_not_model_or_serialization_state(self):
        canonical = self._stack()
        preview = self._stack(
            0.25,
            (120, 80, 40),
            opacity=0.45,
            position='inside',
        )
        item = self._item(stack=canonical)
        before_json = json.dumps(item.blk, cls=TextBlkEncoder, sort_keys=True)
        renderer = item.effect_renderer
        committed_state = renderer._effect_raster_state
        self.assertIsNotNone(committed_state)
        committed_key = committed_state.background_pixmap.cacheKey()

        self.assertTrue(item.set_text_effects(preview, preview=True))
        self.assertEqual(item.blk.fontformat.text_effects, canonical)
        self.assertEqual(item.effective_text_effects(), preview)
        self.assertAlmostEqual(item.opacity(), 0.45)
        self.assertEqual(
            json.dumps(item.blk, cls=TextBlkEncoder, sort_keys=True),
            before_json,
        )
        self.assertIs(renderer._effect_raster_state, committed_state)
        self.assertIsNotNone(renderer._preview_effect_raster_state)
        self.assertNotEqual(
            renderer._preview_effect_raster_state.background_pixmap.cacheKey(),
            committed_key,
        )

        self.assertTrue(item.clear_text_effect_preview())
        self.assertEqual(item.effective_text_effects(), canonical)
        self.assertAlmostEqual(item.opacity(), 1.0)
        self.assertIsNone(renderer._preview_effect_raster_state)
        self.assertEqual(renderer.background_pixmap.cacheKey(), committed_key)
        self.assertFalse(item.clear_text_effect_preview())

    def test_distinct_render_copy_is_mirrored_only_on_commit(self):
        canonical = self._stack()
        target = self._stack(0.26, (80, 90, 100), opacity=0.6)
        item = self._item(stack=canonical)
        model_format = item.blk.fontformat
        render_format = model_format.deepcopy()
        item.fontformat = render_format

        self.assertIs(
            item.effect_renderer.canonical_text_effects(),
            model_format.text_effects,
        )
        item.set_text_effects(target, preview=True)
        self.assertEqual(model_format.text_effects, canonical)
        self.assertEqual(render_format.text_effects, canonical)

        item.set_text_effects(target)
        self.assertEqual(model_format.text_effects, target)
        self.assertEqual(render_format.text_effects, target)
        self.assertFalse(item.effect_renderer.has_preview())

    def test_opacity_only_preview_reuses_committed_effect_pixels(self):
        canonical = self._stack()
        item = self._item(stack=canonical)
        renderer = item.effect_renderer
        committed = renderer._effect_raster_state
        preview = TextEffectStack(0.3, canonical.effects)

        item.set_text_effects(preview, preview=True)

        self.assertIs(renderer._effect_raster_state, committed)
        self.assertIsNone(renderer._preview_effect_raster_state)
        self.assertAlmostEqual(item.opacity(), 0.3)
        item.clear_text_effect_preview()
        self.assertIs(renderer._effect_raster_state, committed)

        committed_key = committed.background_pixmap.cacheKey()
        item.set_text_effects(self._stack(0.24), preview=True)
        self.assertIsNotNone(renderer._preview_effect_raster_state)
        item.set_text_effects(preview, preview=True)
        self.assertIsNone(renderer._preview_effect_raster_state)
        self.assertIs(renderer._effect_raster_state, committed)
        self.assertEqual(committed.background_pixmap.cacheKey(), committed_key)
        self.assertAlmostEqual(item.opacity(), 0.3)

    def test_preview_cancel_rebuilds_parked_cache_after_text_change(self):
        stroke = self._stack().effects[0]
        canonical = TextEffectStack(effects=(
            ShadowEffect(blur=0.12, opacity=0.8),
            stroke,
        ))
        item = self._item(stack=canonical)
        renderer = item.effect_renderer
        committed = renderer._effect_raster_state
        old_pixmap_key = committed.background_pixmap.cacheKey()

        preview = TextEffectStack(effects=(
            canonical.effects[0],
            StrokeEffect(width=0.24),
        ))
        item.set_text_effects(preview, preview=True)
        item.setPlainText('Changed while previewing')
        renderer.repaint_background()

        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render_effect:
            item.clear_text_effect_preview()

        self.assertEqual(render_effect.call_count, 1)
        self.assertIs(renderer._effect_raster_state, committed)
        self.assertNotEqual(
            committed.background_pixmap.cacheKey(), old_pixmap_key
        )
        self.assertEqual(
            committed.cache_input_key,
            renderer._effect_cache_input_key(canonical),
        )

    def test_equal_preview_clears_and_neutral_transition_releases_cache(self):
        item = self._item()
        active = self._stack()
        self.assertFalse(item.set_text_effects(TextEffectStack()))
        self.assertIsNone(item.effect_renderer._effect_raster_state)

        self.assertTrue(item.set_text_effects(active))
        self.assertIsNotNone(item.effect_renderer._effect_raster_state)
        self.assertTrue(item.set_text_effects(self._stack(0.2), preview=True))
        self.assertTrue(item.set_text_effects(active, preview=True))
        self.assertFalse(item.effect_renderer.has_preview())

        self.assertTrue(item.set_text_effects(TextEffectStack()))
        self.assertIsNone(item.effect_renderer._effect_raster_state)

    def test_strokes_paint_back_to_front_with_per_card_opacity(self):
        stack = TextEffectStack(effects=(
            StrokeEffect(
                width=0.1,
                opacity=0.25,
                paint=SolidPaint((255, 0, 0)),
                position='center',
            ),
            StrokeEffect(
                width=0.3,
                opacity=0.75,
                paint=SolidPaint((0, 0, 255)),
                position='center',
            ),
        ))
        item = self._item(stack=stack)
        renderer = item.effect_renderer
        image = QImage(4, 4, QImage.Format.Format_ARGB32_Premultiplied)
        painter = QPainter(image)
        observed = []

        renderer._paint_strokes(
            painter,
            lambda: observed.append((
                renderer._stroke_width(),
                renderer.stroke_qcolor.getRgb()[:3],
                painter.opacity(),
            )),
        )
        painter.end()

        self.assertEqual(
            observed,
            [(0.3, (0, 0, 255), 0.75), (0.1, (255, 0, 0), 0.25)],
        )

    def test_edit_session_handles_blend_and_fill_opacity_preview_commit(self):
        effects = (
            StrokeEffect(),
            ShadowEffect(),
            GlowEffect(),
            TextFillEffect(),
        )
        owner = SimpleNamespace(text_effects=TextEffectStack(effects=effects))
        session = TextEffectEditSession(SimpleNamespace(global_format=owner))

        for index, blend_mode in enumerate(TEXT_EFFECT_BLEND_MODES[:4]):
            updated = session._with_value(
                owner.text_effects, index, 'blend_mode', blend_mode
            )
            self.assertEqual(updated.effects[index].blend_mode, blend_mode)

        session.preview_value(3, 'opacity', 0.25)
        self.assertEqual(owner.text_effects.effects[3].opacity, 0.25)
        self.assertTrue(session.cancel_preview())
        self.assertEqual(owner.text_effects.effects[3].opacity, 1.0)
        self.assertTrue(
            session.commit_value(3, 'blend_mode', 'linear_dodge')
        )
        self.assertEqual(
            owner.text_effects.effects[3].blend_mode, 'linear_dodge'
        )

    def test_add_text_fill_is_repeatable_and_applied_last(self):
        old_fill = TextFillEffect(paint=SolidPaint((10, 20, 30)))
        stroke = StrokeEffect()
        owner = SimpleNamespace(text_effects=TextEffectStack(
            effects=(stroke, old_fill)
        ))
        session = TextEffectEditSession(SimpleNamespace(global_format=owner))

        self.assertTrue(session.add_effect('text_fill'))

        self.assertIsInstance(owner.text_effects.effects[0], TextFillEffect)
        self.assertEqual(owner.text_effects.effects[1:], (stroke, old_fill))
        self.assertTrue(session.add_effect('text_fill'))
        self.assertEqual(sum(
            isinstance(effect, TextFillEffect)
            for effect in owner.text_effects.effects
        ), 3)

    def test_one_session_commit_is_one_canvas_undo_without_document_history(self):
        before = self._stack()
        after = self._stack(0.28, (90, 100, 110), opacity=0.7)
        items = [self._item(False, before), self._item(True, before)]
        document_steps = tuple(
            item.document().availableUndoSteps() for item in items
        )
        canvas = _UndoCanvas()
        host = SimpleNamespace(update_text_style_label=lambda: None)
        session = TextEffectEditSession(host)
        old_canvas = getattr(SW, 'canvas', None)
        SW.canvas = canvas
        try:
            session.replace_targets(items)
            self.assertTrue(session.preview_states((after, after)))
            scratches = tuple(
                item.effect_renderer._preview_effect_raster_state
                for item in items
            )
            self.assertTrue(all(
                scratch.background_pixmap_scale == 0.5
                for scratch in scratches
            ))
            with patch.object(
                items[0],
                'clear_text_effect_preview',
                wraps=items[0].clear_text_effect_preview,
            ) as explicit_clear:
                self.assertTrue(session.commit_states())
            explicit_clear.assert_not_called()
            self.assertEqual(canvas.stack.count(), 1)
            self.assertTrue(all(
                item.blk.fontformat.text_effects == after for item in items
            ))
            self.assertTrue(all(
                not item.effect_renderer.has_preview() for item in items
            ))
            self.assertTrue(all(
                item.effect_renderer._effect_raster_state is not scratch
                for item, scratch in zip(items, scratches)
            ))
            self.assertEqual(
                tuple(item.document().availableUndoSteps() for item in items),
                document_steps,
            )

            canvas.stack.undo()
            self.assertTrue(all(
                item.blk.fontformat.text_effects == before for item in items
            ))
            canvas.stack.redo()
            self.assertTrue(all(
                item.blk.fontformat.text_effects == after for item in items
            ))
            self.assertIsNone(SetTextEffectStackCommand.create(
                items, (after, after), (after, after)
            ))
        finally:
            SW.canvas = old_canvas

    def test_page_and_scene_lifecycle_release_effect_targets(self):
        item = self._item(stack=self._stack())
        target = self._stack(0.2)
        session = TextEffectEditSession(SimpleNamespace())
        session.replace_targets([item])
        session.preview_states((target,))
        session.resolve_for_save()
        self.assertEqual(session.items, [item])
        self.assertIsNone(session.preview_before)

        session.preview_states((target,))
        session.resolve_for_page_change()
        self.assertEqual(session.items, [])
        self.assertIsNone(session.preview_before)
        self.assertFalse(item.effect_renderer.has_preview())

        session.replace_targets([item])
        session.preview_states((target,))
        item_ref = weakref.ref(item)
        session.cancel_for_scene_change()
        self.assertEqual(session.items, [])
        self.assertIsNone(session.preview_before)
        self.assertIs(item_ref(), item)

    def test_undo_refresh_keeps_active_owner_copy_in_sync(self):
        before = self._stack()
        after = self._stack(0.27, (110, 120, 130))
        item = self._item(stack=before)
        active_copy = item.blk.fontformat.deepcopy()
        host = SimpleNamespace(
            textblk_item=item,
            update_text_style_label=lambda: None,
        )
        session = TextEffectEditSession(host)
        canvas = _UndoCanvas()
        old_canvas = getattr(SW, 'canvas', None)
        old_active = C.active_format
        SW.canvas = canvas
        C.active_format = active_copy
        try:
            session.replace_targets([item])
            session.preview_states((after,))
            session.commit_states()
            self.assertEqual(active_copy.text_effects, after)

            # Selection teardown merges the local owner copy back to the item.
            item.blk.fontformat.merge(active_copy)
            self.assertEqual(item.blk.fontformat.text_effects, after)

            canvas.stack.undo()
            self.assertEqual(active_copy.text_effects, before)
            item.blk.fontformat.merge(active_copy)
            self.assertEqual(item.blk.fontformat.text_effects, before)
        finally:
            C.active_format = old_active
            SW.canvas = old_canvas

    def test_reshape_omits_effect_work_then_rebuilds_once(self):
        canonical = self._stack()
        item = self._item(stack=canonical)
        renderer = item.effect_renderer
        original = renderer._render_effect_surface

        with patch.object(
            renderer, '_render_effect_surface', wraps=original
        ) as render:
            item.startReshape()
            item.setRect(QRectF(0, 0, 300, 170))
            item.repaint_background()
            item.setRect(QRectF(0, 0, 290, 160))
            item.repaint_background()
            self.assertEqual(render.call_count, 0)
            self.assertEqual(item.blk.fontformat.text_effects, canonical)
            item.endReshape()
            self.assertEqual(render.call_count, 1)

    def test_effect_preview_does_not_replace_nonlinear_committed_surface(self):
        canonical = self._stack()
        item = self._item(stack=canonical)
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        scene = QGraphicsScene()
        scene.addItem(item)
        self._render_scene(scene)
        surface = item.geometry_controller.surface_renderer
        committed_pixmap = surface.cached_pixmap
        committed_key = surface.cached_key
        self.assertIsNotNone(committed_pixmap)

        item.set_text_effects(self._stack(0.22), preview=True)
        self._render_scene(scene)
        retained = (
            item.geometry_controller._retained_effect_preview_surface
        )
        self.assertIsNotNone(retained)
        self.assertIs(retained[0], committed_pixmap)
        self.assertEqual(retained[1], committed_key)
        self.assertIsNone(surface.cached_pixmap)

        item.clear_text_effect_preview()
        self.assertIs(surface.cached_pixmap, committed_pixmap)
        self.assertEqual(
            surface.cached_key[:-2],
            item.geometry_controller.surface_cache_key(),
        )
        self._render_scene(scene)
        self.assertIs(surface.cached_pixmap, committed_pixmap)
        self.assertEqual(
            surface.cached_key[:-2],
            item.geometry_controller.surface_cache_key(),
        )

    def test_selection_change_rejects_retained_nonlinear_pixels(self):
        item = self._item(stack=self._stack())
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        scene = QGraphicsScene()
        scene.addItem(item)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        self._render_scene(scene)
        surface = item.geometry_controller.surface_renderer
        old_pixmap = surface.cached_pixmap
        old_remap = surface.cached_remap
        self.assertIsNotNone(old_pixmap)

        item.set_text_effects(self._stack(0.22), preview=True)
        cursor = item.textCursor()
        cursor.clearSelection()
        item.setTextCursor(cursor)
        item.clear_text_effect_preview()

        self.assertIsNone(surface.cached_pixmap)
        self.assertIs(surface.cached_remap, old_remap)
        with patch.object(
            surface, '_capture_source', wraps=surface._capture_source
        ) as capture:
            self._render_scene(scene)
        self.assertEqual(capture.call_count, 1)
        self.assertIsNot(surface.cached_pixmap, old_pixmap)
        item.endEdit()

    def test_effect_preview_is_half_resolution_but_keeps_logical_size(self):
        canonical = TextEffectStack(effects=(
            StrokeEffect(width=0.10, position='outside'),
            HollowEffect(),
        ))
        target = TextEffectStack(effects=(
            StrokeEffect(width=0.23, position='outside'),
            HollowEffect(),
        ))
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._item(vertical, canonical)
                scene = QGraphicsScene()
                scene.addItem(item)

                item.set_text_effects(target, preview=True)
                renderer = item.effect_renderer
                scratch = renderer._preview_effect_raster_state
                bounds = renderer.boundingRect()
                self.assertEqual(scratch.background_pixmap_scale, 0.5)
                self.assertEqual(
                    scratch.background_pixmap.width(),
                    math.ceil(bounds.width() * 0.5),
                )
                self.assertEqual(
                    scratch.background_pixmap.height(),
                    math.ceil(bounds.height() * 0.5),
                )
                self.assertEqual(
                    scratch.background_pixmap.devicePixelRatioF(), 1.0
                )
                preview_bounds = self._alpha_bounds(
                    self._render_scene(scene)
                )

                item.set_text_effects(target)
                committed_bounds = self._alpha_bounds(
                    self._render_scene(scene)
                )
                for preview_edge, committed_edge in zip(
                    preview_bounds, committed_bounds
                ):
                    self.assertLessEqual(
                        abs(preview_edge - committed_edge), 2
                    )

    def test_effect_preview_never_rebuilds_for_high_tier_scene_paints(self):
        canonical = self._stack()
        target = self._stack(
            0.23, (70, 80, 90), position='outside'
        )
        transforms = (
            TextTransformStack(),
            TextTransformStack((
                ProjectiveTextTransform(horizontal_scale=1.1),
            )),
            TextTransformStack((SineTextTransform(),)),
        )
        for transform in transforms:
            with self.subTest(transform=transform):
                item = self._item(stack=canonical)
                if not transform.is_neutral():
                    item.set_text_transform(transform)
                scene = QGraphicsScene()
                scene.addItem(item)
                renderer = item.effect_renderer
                with patch.object(
                    renderer,
                    '_render_effect_surface',
                    wraps=renderer._render_effect_surface,
                ) as render:
                    item.set_text_effects(target, preview=True)
                    scratch = renderer._preview_effect_raster_state
                    self.assertEqual(render.call_count, 1)
                    self._render_scene(scene, 2.0)
                    self._render_scene(scene, 4.0)
                    self.assertEqual(render.call_count, 1)
                self.assertEqual(scratch.background_pixmap_scale, 0.5)

    def test_commit_and_export_rerender_low_quality_preview_exactly(self):
        canonical = self._stack()
        target = self._stack(
            0.23, (70, 80, 90), position='outside'
        )
        item = self._item(stack=canonical)
        renderer = item.effect_renderer
        with patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render:
            item.set_text_effects(target, preview=True)
            scratch = renderer._preview_effect_raster_state
            item.set_text_effects(target)
            persistent = renderer._effect_raster_state
            self.assertIsNot(persistent, scratch)
            self.assertEqual(persistent.background_pixmap_scale, 1.0)

            renderer.repaint_background(4.0)
            self.assertEqual(persistent.background_pixmap_scale, 4.0)
            renderer.set_export_effect_render(True)
            try:
                renderer.repaint_background(4.0)
                exported = renderer._export_effect_raster_state
                self.assertIsNot(exported, persistent)
                self.assertEqual(exported.background_pixmap_scale, 4.0)
            finally:
                renderer.set_export_effect_render(False)

        self.assertEqual(
            [call.args[1] for call in render.call_args_list],
            [0.5, 1.0, 4.0, 4.0],
        )
        self.assertIs(renderer._effect_raster_state, persistent)

    def test_commit_never_promotes_low_quality_nonlinear_preview(self):
        canonical = self._stack()
        target = self._stack(0.23)
        item = self._item(stack=canonical)
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        scene = QGraphicsScene()
        scene.addItem(item)
        self._render_scene(scene)
        surface = item.geometry_controller.surface_renderer

        item.set_text_effects(target, preview=True)
        self._render_scene(scene)
        self.assertIsNone(surface.cached_pixmap)

        item.set_text_effects(target)
        self.assertIsNone(surface.cached_pixmap)
        with patch.object(
            surface, '_capture_source', wraps=surface._capture_source
        ) as capture:
            self._render_scene(scene)
        self.assertEqual(capture.call_count, 1)
        self.assertIsNotNone(surface.cached_pixmap)

    def test_future_non_stroke_entry_is_not_compiled_as_stroke(self):
        item = self._item(stack=self._stack())
        stack = object.__new__(TextEffectStack)
        object.__setattr__(stack, 'overall_opacity', 1.0)
        future_effect = SimpleNamespace(is_neutral=lambda: False)
        stroke = StrokeEffect(width=0.2)
        object.__setattr__(stack, 'effects', (future_effect, stroke))

        self.assertEqual(item.effect_renderer._active_strokes(stack), (stroke,))

    def test_effect_only_canvas_export_enables_strict_effect_rendering(self):
        canvas = Canvas()
        canvas.imgtrans_proj = SimpleNamespace(
            inpainted_array=np.zeros((220, 360, 3), dtype=np.uint8),
            inpainted_valid=False,
        )
        canvas.baseLayer.setRect(QRectF(0, 0, 360, 220))
        active = self._item(stack=self._stack())
        shadow = self._item(stack=TextEffectStack(effects=(
            ShadowEffect(blur=0.12, opacity=0.8),
        )))
        neutral = self._item()
        active.setParentItem(canvas.textLayer)
        shadow.setParentItem(canvas.textLayer)
        neutral.setParentItem(canvas.textLayer)
        try:
            with patch.object(
                active,
                'set_export_effect_render',
                wraps=active.set_export_effect_render,
            ) as active_export, patch.object(
                shadow,
                'set_export_effect_render',
                wraps=shadow.set_export_effect_render,
            ) as shadow_export, patch.object(
                neutral,
                'set_export_effect_render',
                wraps=neutral.set_export_effect_render,
            ) as neutral_export:
                image = canvas.render_result_img()
            self.assertFalse(image.isNull())
            self.assertEqual(
                [call.args[0] for call in active_export.call_args_list],
                [True, False],
            )
            self.assertEqual(
                [call.args[0] for call in shadow_export.call_args_list],
                [True, False],
            )
            neutral_export.assert_not_called()
        finally:
            canvas.deleteLater()
            self.app.processEvents()

    def test_strict_export_overrides_preview_and_reshape_degradation(self):
        canonical = self._stack()
        preview = self._stack(0.24, position='inside')
        item = self._item(stack=canonical)
        item.set_text_transform(TextTransformStack((SineTextTransform(),)))
        scene = QGraphicsScene()
        scene.addItem(item)
        self._render_scene(scene)
        item.set_text_effects(preview, preview=True)
        item.startReshape()
        renderer = item.effect_renderer
        surface = item.geometry_controller.surface_renderer
        committed_state = renderer._effect_raster_state
        preview_state = renderer._preview_effect_raster_state

        renderer.set_export_effect_render(True)
        export_state = renderer._export_effect_raster_state
        try:
            with patch.object(
                renderer,
                '_render_effect_surface',
                wraps=renderer._render_effect_surface,
            ) as effect_render, patch.object(
                surface, 'paint', wraps=surface.paint
            ) as surface_paint:
                self._render_scene(scene)
            self.assertGreater(effect_render.call_count, 0)
            kwargs = surface_paint.call_args.kwargs
            self.assertFalse(kwargs['cache_allowed'])
            self.assertIsNone(kwargs['maximum_scale'])
            self.assertTrue(kwargs['high_quality'])
            self.assertIsNotNone(export_state.background_pixmap)
        finally:
            renderer.set_export_effect_render(False)
            item.reshaping = False
        self.assertIs(renderer._effect_raster_state, committed_state)
        self.assertIs(renderer._preview_effect_raster_state, preview_state)
        self.assertIsNone(renderer._export_effect_raster_state)


if __name__ == '__main__':
    unittest.main()
