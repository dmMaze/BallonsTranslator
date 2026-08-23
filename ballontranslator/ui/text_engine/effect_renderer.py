"""Typed effects, block alpha masks, legacy Gradient, and text rendering."""

import math
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextLayout,
)
from qtpy.QtWidgets import QStyle, QStyleOptionGraphicsItem, QWidget

from ballontranslator.utils.fontformat import FontFormat, pt2px
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from ballontranslator.utils.text_effects import (
    HollowEffect,
    ShadowEffect,
    StrokeEffect,
    TextEffectStack,
    hollow_effect,
    primary_stroke,
)
from ..misc import ndarray2pixmap, pixmap2ndarray
from .horizontal_layout import HorizontalTextDocumentLayout
from .vertical_layout import VerticalTextDocumentLayout
from .rendering.alpha_mask import render_text_alpha_mask
from .rendering.glyph import (
    GLYPH_DILATED_STROKE_FORMAT_PROPERTY,
    GLYPH_STROKE_FORMAT_PROPERTY,
)
from .rendering.shadow import render_shadow_rgba
from .rendering.raster import (
    EFFECT_CACHE_MAX_BYTES,
    EFFECT_CACHE_MAX_DIMENSION,
    EFFECT_CACHE_MAX_PIXELS,
    EFFECT_CACHE_MAX_SCALE,
    EFFECT_RASTER_FAILURES,
    EFFECT_RASTER_GUARD,
    EFFECT_TILE_MAX_EDGE,
    RASTER_BOUNDARY_FAILURES,
    EffectRasterAllocationError,
    EffectRasterPlan,
    plan_effect_raster,
    quality_raster_request,
)


GRADIENT_LAYOUT_FORMAT_PROPERTY = 0x100000 + 1238
STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY = 0x100000 + 1241
_STROKE_ALIGNMENT_RANGE_LENGTH = 0x7FFFFFFF
# Glyph Slant writes vector paths into effect pixmaps, not native text.
_VECTOR_EFFECT_RENDER_HINTS = (
    QPainter.RenderHint.Antialiasing
    | QPainter.RenderHint.TextAntialiasing
)


class _EffectRasterState:
    """Allocate raster/cache state only after an effect needs it.

    >>> _EffectRasterState().cache_generation
    0
    """

    def __init__(self) -> None:
        self.cache_generation = 0
        self.cache_rendered_generation = -1
        self.cache_dirty = False
        self.tile_cache = {}
        self.allocation_warning_generation = -1
        self.export_error = None
        self.in_graphics_paint = False
        self.capturing_surface = False
        self.surface_raster_error = None
        self.force_tiles = False
        self.direct_stroke = False
        self.background_pixmap = None
        self.background_pixmap_scale = None
        self.cache_input_key = None


class _EffectRasterField:
    """Descriptor keeping raster-only fields lazy at existing call sites."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance._raster_state(), self.name)

    def __set__(self, instance, value):
        setattr(instance._raster_state(), self.name, value)


class TextEffectRenderer:
    """Own all effect cache state and transformed effect rendering.

    >>> hasattr(TextEffectRenderer, 'repaint_background')
    True
    """

    cache_generation = _EffectRasterField()
    cache_rendered_generation = _EffectRasterField()
    cache_dirty = _EffectRasterField()
    tile_cache = _EffectRasterField()
    allocation_warning_generation = _EffectRasterField()
    export_error = _EffectRasterField()
    in_graphics_paint = _EffectRasterField()
    capturing_surface = _EffectRasterField()
    surface_raster_error = _EffectRasterField()
    force_tiles = _EffectRasterField()
    direct_stroke = _EffectRasterField()

    def __init__(self, item) -> None:
        self.item = item
        self._effect_raster_state = None
        self._preview_effect_raster_state = None
        self._export_effect_raster_state = None
        self._export_active = False
        self._mask_generation = 0
        self.preview = None
        self._render_stroke = None
        self._outline_only_stroke = False
        self.refreshing_gradient_geometry = False
        self.refreshing_effect_padding = False
        self.has_transient_gradient_ranges = False

    def _raster_state(self) -> _EffectRasterState:
        if self._export_active:
            state = self._export_effect_raster_state
            if state is None:
                state = _EffectRasterState()
                self._export_effect_raster_state = state
            return state
        preview = self._uses_preview_cache_namespace()
        state = (
            self._preview_effect_raster_state
            if preview
            else self._effect_raster_state
        )
        if state is None:
            state = _EffectRasterState()
            if preview:
                self._preview_effect_raster_state = state
            else:
                self._effect_raster_state = state
        return state

    def _peek_raster_state(self) -> Optional[_EffectRasterState]:
        if self._export_active:
            return self._export_effect_raster_state
        if self._uses_preview_cache_namespace():
            return self._preview_effect_raster_state
        return self._effect_raster_state

    def _drop_active_raster_state(self) -> None:
        if self._export_active:
            self._export_effect_raster_state = None
            return
        if self._uses_preview_cache_namespace():
            self._preview_effect_raster_state = None
        else:
            self._effect_raster_state = None

    @property
    def background_pixmap(self):
        state = self._peek_raster_state()
        return None if state is None else state.background_pixmap

    @background_pixmap.setter
    def background_pixmap(self, pixmap) -> None:
        state = self._peek_raster_state()
        if state is None and pixmap is None:
            return
        self._raster_state().background_pixmap = pixmap

    @property
    def background_pixmap_scale(self):
        state = self._peek_raster_state()
        return None if state is None else state.background_pixmap_scale

    @background_pixmap_scale.setter
    def background_pixmap_scale(self, scale) -> None:
        state = self._peek_raster_state()
        if state is None and scale is None:
            return
        self._raster_state().background_pixmap_scale = scale

    def surface_cache_state(self) -> Tuple[Tuple[str, int], bool]:
        """Return settled final-warp inputs without allocating effect state."""
        stale = self._invalidate_stale_active_raster_state()
        if stale and not self._export_active and any(self._effect_flags()):
            # The nonlinear cache key includes the completed effect pixmap.
            # Settle it before geometry snapshots that key.
            self.repaint_background()
        export = self._export_active
        preview = self._uses_preview_cache_namespace()
        state = self._peek_raster_state()
        if state is None:
            namespace = 'export' if export else (
                'preview' if preview else 'committed'
            )
            return (namespace, 0), export
        return (
            (
                'export'
                if export
                else ('preview' if preview else 'committed'),
                state.cache_generation,
            ),
            export,
        )

    @property
    def export_render(self) -> bool:
        return self._export_active

    def canonical_text_effects(self) -> TextEffectStack:
        return self.item.blk.fontformat.text_effects

    def effective_text_effects(self) -> TextEffectStack:
        return (
            self.preview
            if self.preview is not None
            else self.canonical_text_effects()
        )

    def has_preview(self) -> bool:
        return self.preview is not None

    def uses_preview_surface(self) -> bool:
        """Return whether preview changes source-surface pixels or geometry."""
        return self._uses_preview_cache_namespace()

    def has_active_effects(self) -> bool:
        return self.effective_text_effects().has_active_effects

    def has_raster_effects(self) -> bool:
        """Return whether strict export must own the complete effect output."""
        return any(self._effect_flags()) or self._renders_completed_foreground()

    def has_generated_effect_layers(self) -> bool:
        """Return whether font/geometry changes invalidate generated layers."""
        return any(self._effect_flags())

    def surface_semantic_state(self) -> tuple:
        """Return effect values that change completed source-surface pixels."""
        return (
            self.effective_text_effects().effects,
            self._mask_generation,
        )

    def _active_text_alpha_mask(self) -> Optional[TextAlphaMask]:
        block = getattr(self.item, 'blk', None)
        mask = None if block is None else block.text_alpha_mask
        return mask if mask is not None and not mask.is_neutral() else None

    def _mask_requires_surface(self) -> bool:
        return (
            self._active_text_alpha_mask() is not None
            and not self._hollow_enabled()
        )

    def _uses_preview_cache_namespace(self) -> bool:
        return (
            self.preview is not None
            and self.preview.effects
            != self.canonical_text_effects().effects
        )

    def _active_strokes(
        self, stack: Optional[TextEffectStack] = None
    ) -> Tuple[StrokeEffect, ...]:
        active = self.effective_text_effects() if stack is None else stack
        return tuple(
            effect
            for effect in active.effects
            if isinstance(effect, StrokeEffect) and not effect.is_neutral()
        )

    def _active_shadows(
        self,
        shadow_type: Optional[str] = None,
        stack: Optional[TextEffectStack] = None,
    ) -> Tuple[ShadowEffect, ...]:
        active = self.effective_text_effects() if stack is None else stack
        return tuple(
            effect
            for effect in active.effects
            if isinstance(effect, ShadowEffect)
            and not effect.is_neutral()
            and (shadow_type is None or effect.shadow_type == shadow_type)
        )

    def _compiled_shadows(
        self, shadow_type: Optional[str] = None
    ) -> Tuple[ShadowEffect, ...]:
        shadows = self._active_shadows(shadow_type)
        if not self._hollow_enabled():
            return shadows
        return tuple(
            shadow
            for shadow in shadows
            if shadow.shadow_type != 'inner'
        )

    def _hollow_enabled(
        self, stack: Optional[TextEffectStack] = None
    ) -> bool:
        active = self.effective_text_effects() if stack is None else stack
        hollow = hollow_effect(active)
        return hollow is not None and not hollow.is_neutral()

    def _effect_cache_input_key(
        self, stack: Optional[TextEffectStack] = None
    ) -> tuple:
        active = self.effective_text_effects() if stack is None else stack
        rect = self.boundingRect()
        layout_generation = getattr(self.layout, 'layout_generation', 0)
        layout_render_key = (
            None
            if self.geometry_controller.layout_renderer is None
            else self.geometry_controller.layout_renderer.render_cache_key()
        )
        return (
            active.effects,
            self._mask_generation,
            self.document().revision(),
            layout_generation,
            layout_render_key,
            self.geometry_controller.effective(),
            self.fontformat.vertical,
            (
                rect.x(), rect.y(), rect.width(), rect.height()
            ),
        )

    @staticmethod
    def _effect_cache_semantic_key(cache_key: tuple) -> tuple:
        layout_render_key = cache_key[4]
        if isinstance(layout_render_key, tuple) and layout_render_key:
            layout_render_key = layout_render_key[1:]
        return (
            cache_key[0],
            cache_key[1],
            cache_key[2],
            layout_render_key,
        ) + cache_key[5:]

    def _promotable_preview_state(
        self, stack: TextEffectStack
    ) -> Optional[_EffectRasterState]:
        state = self._preview_effect_raster_state
        if (
            state is None
            or state.background_pixmap is None
            or state.cache_dirty
            or state.cache_rendered_generation != state.cache_generation
            or state.cache_input_key != self._effect_cache_input_key(stack)
        ):
            return None
        rect = self.boundingRect()
        plan = plan_effect_raster(
            rect.width(), rect.height(), quality_raster_request(1.0)
        )
        if (
            plan.mode != 'full'
            or state.background_pixmap_scale != plan.tier
        ):
            return None
        return state

    def _invalidate_stale_active_raster_state(self) -> bool:
        state = self._peek_raster_state()
        if (
            state is not None
            and (not self.pre_editing or self._export_active)
            and not state.cache_dirty
            and state.cache_rendered_generation == state.cache_generation
            and state.cache_input_key != self._effect_cache_input_key()
        ):
            self._mark_effect_cache_dirty()
            return True
        return False

    def _current_stroke(self) -> Optional[StrokeEffect]:
        if self._render_stroke is not None:
            return self._render_stroke
        return primary_stroke(self.effective_text_effects())

    def _stroke_width(self) -> float:
        stroke = self._current_stroke()
        return 0.0 if stroke is None else stroke.width

    def _paint_strokes(
        self, painter: QPainter, paint: Callable[[], None]
    ) -> None:
        previous = self._render_stroke
        try:
            # The first card is topmost, so paint semantic order back-to-front.
            for stroke in reversed(self._active_strokes()):
                self._render_stroke = stroke
                painter.save()
                try:
                    painter.setOpacity(painter.opacity() * stroke.opacity)
                    paint()
                finally:
                    painter.restore()
        finally:
            self._render_stroke = previous

    @property
    def fontformat(self):
        return self.item.fontformat

    @property
    def layout(self):
        return self.item.layout

    @property
    def geometry_controller(self):
        return self.item.geometry_controller

    @property
    def repainting(self):
        return self.item.repainting

    @repainting.setter
    def repainting(self, value):
        # Formatting and effect rendering share this reentrancy guard.
        self.item.repainting = value

    @property
    def reshaping(self):
        return self.item.reshaping

    @property
    def pre_editing(self):
        return self.item.pre_editing

    @property
    def stroke_qcolor(self):
        stroke = self._current_stroke()
        if stroke is None:
            return self.item.stroke_qcolor
        return QColor(*stroke.paint.color)

    @property
    def idx(self):
        return self.item.idx

    def document(self):
        return self.item.document()

    def boundingRect(self):
        if self.geometry_controller.uses_surface_warp():
            return self.geometry_controller.source_paint_rect()
        return self.item.boundingRect()

    def logical_unpadded_rect(self):
        return self.item.logical_unpadded_rect()

    def padding(self):
        return self.item.padding()

    def setPadding(self, padding):
        return self.item.setPadding(padding)

    def update(self):
        self.item.update()

    def _text_transform_is_neutral(self):
        # A final surface warp still consumes source-local effects exactly
        # once. Active effects around Glyph Slant must keep the
        # transform-aware source path so their silhouette stays slanted.
        if self.geometry_controller.uses_surface_warp():
            return not (
                self._has_layout_distortion()
                and any(self._effect_flags())
            )
        return self.item._text_transform_is_neutral()

    def _has_layout_distortion(self) -> bool:
        return self.geometry_controller.has_layout_distortion()

    def clear_cached_surface(self) -> None:
        self.background_pixmap = None
        self.background_pixmap_scale = None

    def requires_no_item_cache(self) -> bool:
        """Let the effect raster cache see the actual paint-device scale."""
        return any(self._effect_flags())

    def release_caches(self) -> None:
        """Release every item-owned raster cache before page removal."""
        for state in (
            self._effect_raster_state,
            self._preview_effect_raster_state,
            self._export_effect_raster_state,
        ):
            if state is not None:
                state.tile_cache.clear()
        self._effect_raster_state = None
        self._preview_effect_raster_state = None
        self._export_effect_raster_state = None
        self._export_active = False

    def _apply_effective_opacity(self) -> None:
        self.item._set_effective_opacity(
            self.effective_text_effects().overall_opacity
        )

    def _sync_legacy_primary_stroke_view(self) -> None:
        stroke = primary_stroke(self.effective_text_effects())
        if stroke is not None:
            self.item.stroke_qcolor = QColor(*stroke.paint.color)

    @staticmethod
    def _invalidate_raster_state(state: Optional[_EffectRasterState]) -> None:
        if state is None:
            return
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_rendered_generation = -1
        state.cache_input_key = None
        state.tile_cache.clear()
        state.background_pixmap = None
        state.background_pixmap_scale = None

    def _finish_effect_transition(self, repaint: bool) -> None:
        self._apply_effective_opacity()
        self._sync_legacy_primary_stroke_view()
        was_repainting = self.repainting
        self.repainting = True
        try:
            self._sync_native_stroke_alignment()
        finally:
            self.repainting = was_repainting
        self._update_effect_padding()
        self.item.refresh_cache_policy()
        if repaint and not self.reshaping:
            self.repaint_background()
        self.item.update()

    def set_text_effects(
        self, stack: TextEffectStack, preview: bool = False
    ) -> bool:
        """Apply a complete preview or committed stack at the item boundary.

        >>> isinstance(TextEffectStack(), TextEffectStack)
        True
        """
        if not isinstance(stack, TextEffectStack):
            raise TypeError('live text effects require TextEffectStack')
        canonical = self.canonical_text_effects()
        effective_before = self.effective_text_effects()
        preview_before = self.preview

        if preview:
            if stack == canonical:
                return self.clear_text_effect_preview()
            if preview_before == stack:
                return False
            self.preview = stack
            effects_changed = effective_before.effects != stack.effects
            if effects_changed:
                self._preview_effect_raster_state = None
                if stack.effects != canonical.effects:
                    self.geometry_controller.retain_effect_preview_surface()
                    self._mark_effect_cache_dirty()
                else:
                    # Returning to canonical effect pixels keeps the complete
                    # preview alive only for its native overall opacity.
                    self._finish_effect_transition(False)
                    self.geometry_controller.restore_effect_preview_surface()
                    return True
            self._finish_effect_transition(effects_changed)
            return True

        model_format = self.item.blk.fontformat
        render_format = self.item.fontformat
        canonical_changed = canonical != stack
        render_format_changed = (
            render_format is not model_format
            and render_format.text_effects != stack
        )
        if (
            not canonical_changed
            and not render_format_changed
            and preview_before is None
        ):
            self._apply_effective_opacity()
            return False
        effects_changed = canonical.effects != stack.effects
        promoted_state = (
            self._promotable_preview_state(stack)
            if effects_changed and preview_before == stack
            else None
        )
        committed_generation = (
            0
            if self._effect_raster_state is None
            else self._effect_raster_state.cache_generation
        )
        if canonical_changed:
            model_format.text_effects = stack
        if render_format_changed:
            render_format.text_effects = stack
        self.preview = None
        self._preview_effect_raster_state = None
        self.geometry_controller.invalidate_effect_preview_surface()
        if effects_changed:
            if promoted_state is None:
                self._mark_effect_cache_dirty()
            else:
                promoted_state.cache_generation = committed_generation + 1
                promoted_state.cache_rendered_generation = (
                    promoted_state.cache_generation
                )
                promoted_state.tile_cache.clear()
                self._effect_raster_state = promoted_state
        self._finish_effect_transition(
            effects_changed and promoted_state is None
        )
        if promoted_state is not None:
            current_key = self._effect_cache_input_key(stack)
            if promoted_state.cache_input_key != current_key:
                self._mark_effect_cache_dirty()
                self.repaint_background()
            else:
                promoted_state.cache_input_key = current_key
        return (
            canonical_changed
            or render_format_changed
            or effective_before != stack
        )

    def _on_text_alpha_mask_changed(self) -> None:
        """Invalidate every raster namespace after a committed mask replace."""
        self._mask_generation += 1
        for state in (
            self._effect_raster_state,
            self._preview_effect_raster_state,
            self._export_effect_raster_state,
        ):
            self._invalidate_raster_state(state)
        self.geometry_controller.invalidate_surface_cache()
        self.item.refresh_cache_policy()
        if not self.reshaping:
            self.repaint_background()
        self.item.update()

    def clear_text_effect_preview(self) -> bool:
        if self.preview is None:
            return False
        preview = self.preview
        self.preview = None
        self._preview_effect_raster_state = None
        effects_changed = preview.effects != self.canonical_text_effects().effects
        self._finish_effect_transition(False)
        state = self._effect_raster_state
        current_key = self._effect_cache_input_key()
        if (
            state is not None
            and not state.cache_dirty
            and state.cache_rendered_generation == state.cache_generation
            and state.cache_input_key is not None
            and self._effect_cache_semantic_key(state.cache_input_key)
            == self._effect_cache_semantic_key(current_key)
        ):
            # Preview padding advances layout-only generations. Re-key only
            # after all pixel-bearing inputs return to the canonical values.
            state.cache_input_key = current_key
        needs_repaint = bool(
            effects_changed
            and any(self._effect_flags())
            and (
                state is None
                or state.cache_dirty
                or state.cache_rendered_generation
                != state.cache_generation
                or state.cache_input_key != current_key
            )
        )
        if needs_repaint and not self.reshaping:
            self.repaint_background()
        self.geometry_controller.restore_effect_preview_surface()
        return True

    def begin_reshape(self) -> None:
        """Omit effects during pointer motion and retire old geometry caches."""
        self._invalidate_raster_state(self._effect_raster_state)
        self._invalidate_raster_state(self._preview_effect_raster_state)
        self._invalidate_raster_state(self._export_effect_raster_state)
        self.geometry_controller.invalidate_effect_preview_surface()

    def end_reshape(self) -> None:
        """Rebuild only the effective namespace after geometry settles."""
        self.repaint_background()

    def paint_item(self, painter: QPainter, option, widget: QWidget, base_paint) -> None:
        """Paint effects around the host item's normal text pass."""
        if (
            (self.reshaping and not self._export_active)
            or (
                not any(self._effect_flags())
                and not self._renders_completed_foreground()
            )
        ):
            option.state = QStyle.State_None
            base_paint(painter, option, widget)
            return

        # Effects must be composited before the normal fill. DestinationOver
        # against an already opaque scene would discard them.
        was_in_graphics_paint = self.in_graphics_paint
        self.in_graphics_paint = True
        try:
            interaction_option = QStyleOptionGraphicsItem(option)
            if any(self._effect_flags()):
                self._draw_effects(painter, option.exposedRect)
            replace_foreground = self._hollow_enabled() or (
                (
                    bool(self._active_shadows('inner'))
                    or self._active_text_alpha_mask() is not None
                )
                and (
                    self.export_render
                    or self._completed_foreground_ready()
                )
            )
            if replace_foreground:
                self._paint_effect_interaction(
                    painter, interaction_option, widget, base_paint
                )
            else:
                option.state = QStyle.State_None
                base_paint(painter, option, widget)
        finally:
            self.in_graphics_paint = was_in_graphics_paint

    def _renders_completed_foreground(self) -> bool:
        return (
            self._hollow_enabled()
            or bool(self._active_shadows('inner'))
            or self._mask_requires_surface()
        )

    def _completed_foreground_ready(self) -> bool:
        state = self._peek_raster_state()
        return bool(
            state is not None
            and not state.cache_dirty
            and state.cache_rendered_generation == state.cache_generation
        )

    def _paint_effect_interaction(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget,
        base_paint,
    ) -> None:
        """Paint only selection/caret feedback over a completed foreground.

        The native pass keeps Qt's caret/IME state current. Canonical glyph
        alpha is removed from a transient layer so only editing feedback is
        composited over the completed cached surface.

        >>> hasattr(TextEffectRenderer, '_paint_effect_interaction')
        True
        """
        if self.export_render or not self.item.isEditing():
            return
        rect = self._visible_effect_rect(painter, option.exposedRect)
        if rect.isEmpty():
            return
        requested_scale = self._paint_device_scale(painter)
        plan = plan_effect_raster(
            rect.width(),
            rect.height(),
            quality_raster_request(requested_scale),
        )
        try:
            if plan.mode != 'full':
                raise EffectRasterAllocationError(
                    'interaction surface exceeds bounded raster policy'
                )
            interaction = self._new_effect_pixmap(plan.tier, rect)
            interaction_painter = QPainter(interaction)
            if not interaction_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin effect interaction painter'
                )
            try:
                interaction_painter.translate(-rect.topLeft())
                base_paint(
                    interaction_painter,
                    QStyleOptionGraphicsItem(option),
                    widget,
                )
                interaction_painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationOut
                )
                self._paint_live_layout(
                    interaction_painter, self._effect_paint_context()
                )
            finally:
                interaction_painter.end()
            painter.drawPixmap(rect.topLeft(), interaction)
        except RASTER_BOUNDARY_FAILURES:
            painter.save()
            try:
                painter.setOpacity(0.0)
                base_paint(painter, option, widget)
            finally:
                painter.restore()

    def finalize_neutral_cache(self) -> None:
        """Invalidate transformed pixels after neutral restoration."""
        self._refresh_gradient_geometry()
        state = self._peek_raster_state()
        if state is not None:
            state.tile_cache.clear()
            state.force_tiles = False
            state.direct_stroke = False
            state.cache_dirty = True
            state.cache_rendered_generation = -1
        self.clear_cached_surface()
        self.item.update()
        if not any(self._effect_flags()):
            self._drop_active_raster_state()

    def _effect_paint_context(self):
        context = QAbstractTextDocumentLayout.PaintContext()
        context.cursorPosition = -1
        context.selections = []
        return context

    def _paint_live_layout(self, painter: QPainter, context=None):
        layout = self.document().documentLayout()
        if context is None:
            context = self._effect_paint_context()
        layout.draw(painter, context)

    def _stroke_paint_context(self):
        context = self._effect_paint_context()
        doc = self.document()
        selections = []
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                char_format = fragment.charFormat()
                point_size = char_format.fontPointSize()
                if point_size <= 0:
                    point_size = char_format.font().pointSizeF()
                if point_size <= 0:
                    point_size = doc.defaultFont().pointSizeF()

                pen = QPen(
                    self.stroke_qcolor,
                    pt2px(point_size) * self._stroke_width(),
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                effect_format = QTextCharFormat()
                effect_format.setProperty(
                    GLYPH_STROKE_FORMAT_PROPERTY, True
                )
                # The later normal fill restores glyph interiors. Keeping this
                # pass opaque also avoids bindings that suppress textOutline
                # when the selection foreground itself is transparent.
                foreground = QColor(self.stroke_qcolor)
                if self._outline_only_stroke:
                    foreground.setAlpha(1)
                effect_format.setForeground(foreground)
                effect_format.setTextOutline(pen)

                selection = QAbstractTextDocumentLayout.Selection()
                selection.cursor = QTextCursor(doc)
                selection.cursor.setPosition(fragment.position())
                selection.cursor.setPosition(
                    fragment.position() + fragment.length(),
                    QTextCursor.MoveMode.KeepAnchor,
                )
                selection.format = effect_format
                selections.append(selection)
                it += 1
            block = block.next()
        context.selections = selections
        return context

    def _stroke_outset(self) -> float:
        strokes = self._active_strokes()
        if not strokes:
            return 0.0
        return (
            self.layout.max_font_size(to_px=True)
            * max(stroke.width for stroke in strokes)
            / 2
        )

    def _sync_native_stroke_alignment(self) -> None:
        """Keep fill and stroke on Qt's same native glyph raster path."""
        if self.layout is None:
            return
        enabled = bool(self._active_strokes())
        changed = False
        alignment_format = None
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            formats = list(layout.formats())
            tagged = [
                entry
                for entry in formats
                if bool(entry.format.property(
                    STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY
                ))
            ]
            if enabled == bool(tagged):
                block = block.next()
                continue
            formats = [
                entry
                for entry in formats
                if not bool(entry.format.property(
                    STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY
                ))
            ]
            if enabled:
                if alignment_format is None:
                    alignment_format = QTextCharFormat()
                    alignment_format.setProperty(
                        STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY, True
                    )
                    # A styled outline selects Qt's path-backed glyph
                    # rasterizer; transparent zero width paints no pixels.
                    alignment_format.setTextOutline(QPen(
                        QColor(0, 0, 0, 0),
                        0.0,
                        Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap,
                        Qt.PenJoinStyle.RoundJoin,
                    ))
                entry = QTextLayout.FormatRange()
                entry.start = 0
                entry.length = _STROKE_ALIGNMENT_RANGE_LENGTH
                entry.format = alignment_format
                formats.append(entry)
            layout.setFormats(formats)
            changed = True
            block = block.next()
        if changed:
            # setFormats invalidates QTextLine objects but changes no document
            # content or geometry; rebuild once after all blocks are updated.
            self.layout.reLayout()

    def _new_effect_pixmap(
        self,
        render_scale: float = 1.0,
        surface_rect: QRectF = None,
    ) -> QPixmap:
        rect = self.boundingRect() if surface_rect is None else surface_rect
        pixel_width = max(1, math.ceil(rect.width() * render_scale))
        pixel_height = max(1, math.ceil(rect.height() * render_scale))
        if (
            pixel_width > EFFECT_CACHE_MAX_DIMENSION
            or pixel_height > EFFECT_CACHE_MAX_DIMENSION
            or pixel_width * pixel_height > EFFECT_CACHE_MAX_PIXELS
            or pixel_width * pixel_height * 4 > EFFECT_CACHE_MAX_BYTES
        ):
            raise EffectRasterAllocationError(
                f'effect surface {pixel_width}x{pixel_height} exceeds policy'
            )
        try:
            pixmap = QPixmap(pixel_width, pixel_height)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                f'unable to allocate effect surface '
                f'{pixel_width}x{pixel_height}'
            ) from error
        if pixmap.isNull():
            raise EffectRasterAllocationError(
                f'unable to allocate effect surface {pixel_width}x{pixel_height}'
            )
        try:
            pixmap.setDevicePixelRatio(render_scale)
            pixmap.fill(Qt.GlobalColor.transparent)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                f'unable to initialize effect surface '
                f'{pixel_width}x{pixel_height}'
            ) from error
        return pixmap

    def _paint_cloned_document_stroke(self, painter: QPainter) -> None:
        """Paint stroke through the BASE cloned-document path."""
        # Qt's native clone preserves UserProperty values and avoids a full
        # HTML serialization/parse cycle on every effect refresh.
        doc = self.document().clone()
        doc.setUndoRedoEnabled(False)
        doc.setDocumentMargin(self.layout.effectPadding())
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        stroke_pen = QPen(
            self.stroke_qcolor,
            0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                char_format = fragment.charFormat()
                stroke_pen.setWidthF(
                    pt2px(char_format.fontPointSize())
                    * self._stroke_width()
                )
                cursor.setPosition(fragment.position())
                cursor.setPosition(
                    fragment.position() + fragment.length(),
                    QTextCursor.MoveMode.KeepAnchor,
                )
                char_format.setTextOutline(stroke_pen)
                if self._outline_only_stroke:
                    foreground = QColor(self.stroke_qcolor)
                    foreground.setAlpha(1)
                    char_format.setForeground(foreground)
                # Path-painted glyph extensions consume this flag. Ruby and
                # emphasis derive half-width native outlines in temporary docs.
                char_format.setProperty(
                    GLYPH_DILATED_STROKE_FORMAT_PROPERTY, True
                )
                cursor.mergeCharFormat(char_format)
                it += 1
            block = block.next()

        layout = (
            VerticalTextDocumentLayout(doc, self.fontformat)
            if self.fontformat.vertical
            else HorizontalTextDocumentLayout(doc, self.fontformat)
        )
        layout._draw_offset = self.layout._draw_offset
        layout._is_painting_stroke = True
        layout.setMaxSize(self.layout.max_width, self.layout.max_height, False)
        doc.setDocumentLayout(layout)
        layout.relayout_on_changed = False
        doc.drawContents(painter)

    def _paint_vertical_stroke(
        self,
        painter: QPainter,
        render_scale: float = 1.0,
        surface_rect: QRectF = None,
    ):
        """Stroke vertical glyphs per rich-text fragment on every binding."""
        stroke_alpha = None
        rgba = None
        stroke_context = self._stroke_paint_context()
        selections_by_radius = {}
        for selection in stroke_context.selections:
            logical_radius = selection.format.textOutline().widthF() / 2
            selections_by_radius.setdefault(logical_radius, []).append(selection)

        for logical_radius, selections in selections_by_radius.items():
            rect = self.boundingRect() if surface_rect is None else surface_rect
            source = self._new_effect_pixmap(render_scale, rect)
            source_painter = QPainter(source)
            if not source_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin vertical stroke source painter'
                )
            try:
                source_painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
                source_painter.translate(-rect.topLeft())
                fragment_context = self._effect_paint_context()
                fragment_context.selections = selections
                self.geometry_controller.draw_layout_selection_mask(
                    source_painter,
                    fragment_context,
                    include_annotations=False,
                )
            finally:
                source_painter.end()

            try:
                rgba = pixmap2ndarray(source, keep_alpha=True)
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to access vertical stroke source pixels'
                ) from error
            if rgba is None:
                raise EffectRasterAllocationError(
                    'unable to access vertical stroke source pixels'
                )
            alpha = rgba[..., 3]
            radius = math.ceil(logical_radius * render_scale)
            if radius > 0:
                diameter = radius * 2 + 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (diameter, diameter)
                )
                alpha = cv2.dilate(alpha, kernel)
            if stroke_alpha is None:
                stroke_alpha = alpha
            else:
                np.maximum(stroke_alpha, alpha, out=stroke_alpha)

        if stroke_alpha is None or rgba is None:
            return
        stroke = np.empty_like(rgba)
        stroke[..., 0] = self.stroke_qcolor.red()
        stroke[..., 1] = self.stroke_qcolor.green()
        stroke[..., 2] = self.stroke_qcolor.blue()
        stroke[..., 3] = stroke_alpha
        try:
            stroke_pixmap = ndarray2pixmap(stroke)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                'unable to allocate vertical stroke result'
            ) from error
        if stroke_pixmap is None or stroke_pixmap.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate vertical stroke result'
            )
        stroke_pixmap.setDevicePixelRatio(render_scale)
        painter.drawPixmap(rect.topLeft(), stroke_pixmap)
        # Half-font annotations own native outlines, not the base mask's
        # full-font morphology radius.
        self.geometry_controller.draw_layout_annotations(
            painter, stroke_context
        )

    def paint_stroke(
        self,
        painter: QPainter,
        render_scale: float = 1.0,
        surface_rect: QRectF = None,
    ):
        if self._text_transform_is_neutral():
            self._paint_cloned_document_stroke(painter)
            return
        active_layout = self.document().documentLayout()
        if (
            isinstance(active_layout, VerticalTextDocumentLayout)
            and self._has_layout_distortion()
        ):
            self._paint_vertical_stroke(painter, render_scale, surface_rect)
            return
        self._paint_source_local_stroke(painter)

    def _paint_source_local_stroke(self, painter: QPainter):
        # Native box transforms map the completed source surface. Only an
        # attached glyph renderer changes the source glyph geometry itself.
        if self._has_layout_distortion():
            self._paint_live_layout(painter, self._stroke_paint_context())
            return
        self._paint_cloned_document_stroke(painter)

    def _shadow_metrics(
        self, shadow: ShadowEffect
    ) -> Tuple[float, float, float, float]:
        font_size = self.layout.max_font_size(to_px=True)
        return (
            shadow.blur * font_size,
            shadow.spread * font_size,
            shadow.offset[0] * font_size,
            shadow.offset[1] * font_size,
        )

    def _shadowed_bounds(
        self, source_bounds: QRectF, shadow: ShadowEffect
    ) -> QRectF:
        blur, spread, xoffset, yoffset = self._shadow_metrics(shadow)
        if shadow.shadow_type == 'long':
            return source_bounds.united(
                source_bounds.translated(xoffset, yoffset)
            )
        return source_bounds.translated(xoffset, yoffset).adjusted(
            -blur - spread,
            -blur - spread,
            blur + spread,
            blur + spread,
        )

    def _logical_ink_bounds(self) -> QRectF:
        if self.document().isEmpty() or not self._has_layout_distortion():
            return QRectF()
        return self.geometry_controller.layout_ink_bounds()

    def _effect_padding(self) -> float:
        paint_stroke, _paint_non_stroke = self._effect_flags()
        layout_distorted = self._has_layout_distortion()
        if not layout_distorted:
            return self._conservative_effect_padding()
        ink_bounds = self._logical_ink_bounds()
        if ink_bounds.isEmpty():
            return 0.0
        stroke_outset = self._stroke_outset()
        logical_rect = self.logical_unpadded_rect()
        effect_bounds = ink_bounds.adjusted(
            -stroke_outset if paint_stroke else 0.0,
            -stroke_outset if paint_stroke else 0.0,
            stroke_outset if paint_stroke else 0.0,
            stroke_outset if paint_stroke else 0.0,
        )
        shadow_source_bounds = QRectF(effect_bounds)
        for shadow in self._compiled_shadows():
            if shadow.shadow_type != 'inner':
                effect_bounds = effect_bounds.united(
                    self._shadowed_bounds(shadow_source_bounds, shadow)
                )
        effect_bounds = effect_bounds.adjusted(
            -EFFECT_RASTER_GUARD,
            -EFFECT_RASTER_GUARD,
            EFFECT_RASTER_GUARD,
            EFFECT_RASTER_GUARD,
        )
        return max(
            0.0,
            logical_rect.left() - effect_bounds.left(),
            effect_bounds.right() - logical_rect.right(),
            logical_rect.top() - effect_bounds.top(),
            effect_bounds.bottom() - logical_rect.bottom(),
        )

    def _conservative_effect_padding(self) -> float:
        """Return cheap symmetric padding for non-distorting glyph paths."""
        if self.layout is None:
            return 0.0
        max_font_size = max(0.0, self.layout.max_font_size(to_px=True))
        stroke_outset = 0.0
        active_strokes = self._active_strokes()
        if active_strokes:
            stroke_outset = (
                max_font_size
                * (max(stroke.width for stroke in active_strokes) + 0.05)
                / 2
            )
        padding = stroke_outset
        exterior_padding = None
        for shadow in self._compiled_shadows():
            if shadow.shadow_type == 'inner':
                continue
            blur, spread, xoffset, yoffset = self._shadow_metrics(shadow)
            shadow_padding = (
                stroke_outset
                + (0.0 if shadow.shadow_type == 'long' else blur + spread)
                + max(abs(xoffset), abs(yoffset))
            )
            exterior_padding = (
                shadow_padding
                if exterior_padding is None
                else max(exterior_padding, shadow_padding)
            )
        if exterior_padding is not None:
            padding = max(
                padding, exterior_padding + EFFECT_RASTER_GUARD
            )
        return padding

    def _commit_effect_padding(
        self,
        padding: float,
    ) -> bool:
        changed = (
            self.setPadding(padding)
            if self.padding() != padding
            else False
        )
        if changed and self.fontformat.gradient_enabled:
            was_repainting = self.repainting
            self.repainting = True
            try:
                self._refresh_gradient_geometry()
            finally:
                self.repainting = was_repainting
        return changed

    def _update_effect_padding(self):
        if self.refreshing_effect_padding or self.layout is None:
            return False
        self.refreshing_effect_padding = True
        try:
            padding = self._effect_padding()
            # QTextLayout stores coordinates at 26.6 fixed-point precision.
            # Round outward so relayout and undo cycles converge.
            if padding > 0.0:
                layout_units = math.nextafter(padding * 64.0, -math.inf)
                padding = math.ceil(layout_units) / 64.0
            return self._commit_effect_padding(padding)
        finally:
            self.refreshing_effect_padding = False

    def _effect_flags(self) -> Tuple[bool, bool]:
        """Return active Stroke and generated completed-surface flags."""
        return (
            bool(self._active_strokes()),
            bool(self._compiled_shadows()) or self._mask_requires_surface(),
        )

    def _effect_tile_overlap(self) -> float:
        overlap = self._stroke_outset() + EFFECT_RASTER_GUARD
        for shadow in self._compiled_shadows():
            blur, spread, xoffset, yoffset = self._shadow_metrics(shadow)
            if shadow.shadow_type == 'long':
                reach = max(abs(xoffset), abs(yoffset))
            else:
                reach = blur + spread + max(abs(xoffset), abs(yoffset))
            overlap = max(
                overlap,
                reach + self._stroke_outset() + EFFECT_RASTER_GUARD,
            )
        return overlap

    def _warn_effect_allocation_once(self, error: Exception):
        if self.allocation_warning_generation == self.cache_generation:
            return
        self.allocation_warning_generation = self.cache_generation
        LOGGER.warning(
            'Text effect raster allocation failed for item %s; '
            'using the bounded interactive fallback for this frame: %s',
            self.idx,
            error,
        )

    def _on_glyph_raster_failure(
        self, error: Exception, effect_pass: bool = False
    ):
        """Bridge renderer degradation into item/export failure policy."""
        failure = EffectRasterAllocationError(str(error))
        self._warn_effect_allocation_once(failure)
        if self.capturing_surface:
            self.surface_raster_error = failure
        if effect_pass:
            self.cache_dirty = True
        if self.capturing_surface:
            return
        if self.export_render:
            if self.in_graphics_paint:
                self.export_error = failure
            else:
                raise failure from error

    def set_export_effect_render(self, enabled: bool):
        """Make effect allocation failures fatal during a render transaction."""
        enabled = bool(enabled)
        if enabled:
            self._export_active = True
            self._export_effect_raster_state = _EffectRasterState()
            return
        self._export_active = False
        self._export_effect_raster_state = None

    def _raise_or_defer_export_effect_error(self, error: Exception) -> bool:
        """Raise at a Python boundary or defer across Qt's paint callback.

        PyQt treats an exception escaping a virtual ``QGraphicsItem.paint``
        callback as fatal. Canvas checks the deferred error immediately after
        ``QGraphicsScene.render`` and raises before returning its image.
        """
        if not self.export_render:
            return False
        failure = EffectRasterAllocationError(str(error))
        if self.in_graphics_paint:
            self.export_error = failure
            return True
        raise failure from error

    def _render_effect_surface(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        target_stroke: bool = True,
    ) -> QPixmap:
        """Render fixed effect phases from one canonical glyph alpha.

        >>> hasattr(TextEffectRenderer, '_render_effect_surface')
        True
        """
        paint_stroke, _paint_non_stroke = self._effect_flags()
        hollow = self._hollow_enabled()
        alpha_mask = self._active_text_alpha_mask()
        exterior = tuple(
            shadow
            for shadow in self._active_shadows()
            if shadow.shadow_type != 'inner'
        )
        interior = self._active_shadows('inner') if not hollow else ()
        target_map = self._new_effect_pixmap(render_scale, surface_rect)
        canonical = None
        canonical_alpha = None
        completed_foreground = alpha_mask is not None and not hollow
        if (
            interior
            or completed_foreground
            or (exterior and not paint_stroke)
            or (hollow and exterior)
        ):
            canonical = self._capture_effect_source(
                surface_rect, render_scale, include_strokes=False
            )
            canonical_alpha = self._pixmap_alpha(canonical)
        exterior_alpha = canonical_alpha
        if exterior and paint_stroke:
            silhouette = self._capture_effect_source(
                surface_rect, render_scale, include_strokes=True
            )
            exterior_alpha = self._pixmap_alpha(silhouette)

        try:
            target_painter = QPainter(target_map)
            if not target_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin effect target painter'
                )
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to begin effect target painter'
            ) from error

        previous_capture = self.capturing_surface
        previous_raster_error = self.surface_raster_error
        self.capturing_surface = True
        self.surface_raster_error = None
        try:
            target_painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            # First card in a phase is topmost, so paint each phase backwards.
            for shadow in reversed(exterior):
                assert exterior_alpha is not None
                target_painter.drawPixmap(
                    QPointF(),
                    self._shadow_pixmap(
                        exterior_alpha, shadow, render_scale
                    ),
                )

            if hollow and exterior:
                assert canonical is not None
                target_painter.save()
                try:
                    target_painter.setCompositionMode(
                        QPainter.CompositionMode.CompositionMode_DestinationOut
                    )
                    target_painter.drawPixmap(QPointF(), canonical)
                finally:
                    target_painter.restore()

            target_painter.translate(-surface_rect.topLeft())
            if paint_stroke and target_stroke:
                if hollow:
                    self._paint_hollow_strokes(
                        target_painter,
                        surface_rect,
                        render_scale,
                    )
                else:
                    self._paint_strokes(
                        target_painter,
                        lambda: self.paint_stroke(
                            target_painter, render_scale, surface_rect
                        ),
                    )

            if interior:
                assert canonical is not None
                assert canonical_alpha is not None
                target_painter.drawPixmap(surface_rect.topLeft(), canonical)
                target_painter.translate(surface_rect.topLeft())
                for shadow in reversed(interior):
                    target_painter.drawPixmap(
                        QPointF(),
                        self._shadow_pixmap(
                            canonical_alpha, shadow, render_scale
                        ),
                    )
                target_painter.translate(-surface_rect.topLeft())
            elif completed_foreground:
                assert canonical is not None
                target_painter.drawPixmap(surface_rect.topLeft(), canonical)

            if alpha_mask is not None:
                self._apply_text_alpha_mask(
                    target_painter,
                    alpha_mask,
                    surface_rect,
                    render_scale,
                )
            if self.surface_raster_error is not None:
                raise self.surface_raster_error
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render typed effect surface'
            ) from error
        finally:
            end_error = None
            try:
                target_painter.end()
            except RASTER_BOUNDARY_FAILURES as error:
                end_error = error
            self.capturing_surface = previous_capture
            self.surface_raster_error = previous_raster_error
            if end_error is not None:
                raise EffectRasterAllocationError(
                    'unable to finish typed effect painter'
                ) from end_error
        return target_map

    def _apply_text_alpha_mask(
        self,
        painter: QPainter,
        mask: TextAlphaMask,
        surface_rect: QRectF,
        render_scale: float,
    ) -> None:
        """Clip a complete Normal composite with the block-owned alpha mask.

        >>> hasattr(TextEffectRenderer, '_apply_text_alpha_mask')
        True
        """
        try:
            alpha = render_text_alpha_mask(
                mask,
                surface_rect,
                self.logical_unpadded_rect().topLeft(),
                render_scale,
            )
            painter.save()
            try:
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationIn
                )
                painter.drawImage(surface_rect.topLeft(), alpha)
            finally:
                painter.restore()
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to apply text alpha mask'
            ) from error

    def _paint_hollow_strokes(
        self,
        painter: QPainter,
        surface_rect: QRectF,
        render_scale: float,
    ) -> None:
        """Paint centered Stroke bands without their opaque glyph fill.

        >>> hasattr(TextEffectRenderer, '_paint_hollow_strokes')
        True
        """
        previous = self._render_stroke
        previous_outline_only = self._outline_only_stroke
        self._outline_only_stroke = True
        try:
            for stroke in reversed(self._active_strokes()):
                self._render_stroke = stroke
                try:
                    layer = self._new_effect_pixmap(
                        render_scale, surface_rect
                    )
                    layer_painter = QPainter(layer)
                    if not layer_painter.isActive():
                        raise EffectRasterAllocationError(
                            'unable to begin Hollow Stroke painter'
                        )
                    try:
                        layer_painter.setRenderHints(
                            _VECTOR_EFFECT_RENDER_HINTS
                        )
                        layer_painter.translate(-surface_rect.topLeft())
                        self.paint_stroke(
                            layer_painter, render_scale, surface_rect
                        )
                    finally:
                        layer_painter.end()

                    rgba = pixmap2ndarray(layer, keep_alpha=True)
                    if rgba is None:
                        raise EffectRasterAllocationError(
                            'unable to access Hollow Stroke pixels'
                        )
                    # The fragment-correct outline pass uses alpha 1 only to
                    # keep Qt from suppressing textOutline. Remove that
                    # sentinel before this layer becomes persistent pixels.
                    rgba[..., 3][rgba[..., 3] <= 1] = 0
                    if stroke.opacity != 1.0:
                        product = rgba[..., 3].astype(np.uint16)
                        product *= int(round(stroke.opacity * 255))
                        product += 127
                        product //= 255
                        rgba[..., 3] = product.astype(np.uint8)
                    band = ndarray2pixmap(rgba)
                    if band is None or band.isNull():
                        raise EffectRasterAllocationError(
                            'unable to allocate Hollow Stroke band'
                        )
                    band.setDevicePixelRatio(render_scale)
                    painter.drawPixmap(surface_rect.topLeft(), band)
                except RASTER_BOUNDARY_FAILURES as error:
                    raise EffectRasterAllocationError(
                        'unable to render Hollow Stroke band'
                    ) from error
        finally:
            self._outline_only_stroke = previous_outline_only
            self._render_stroke = previous

    def _capture_effect_source(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        include_strokes: bool,
    ) -> QPixmap:
        """Capture the canonical glyph alpha, optionally with Stroke coverage.

        >>> hasattr(TextEffectRenderer, '_capture_effect_source')
        True
        """
        source = self._new_effect_pixmap(render_scale, surface_rect)
        try:
            painter = QPainter(source)
            if not painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin effect source painter'
                )
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to begin effect source painter'
            ) from error
        previous_capture = self.capturing_surface
        previous_raster_error = self.surface_raster_error
        self.capturing_surface = True
        self.surface_raster_error = None
        try:
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            painter.translate(-surface_rect.topLeft())
            self._paint_live_layout(painter, self._effect_paint_context())
            if include_strokes:
                self._paint_strokes(
                    painter,
                    lambda: self.paint_stroke(
                        painter, render_scale, surface_rect
                    ),
                )
            if self.surface_raster_error is not None:
                raise self.surface_raster_error
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render effect source surface'
            ) from error
        finally:
            end_error = None
            try:
                painter.end()
            except RASTER_BOUNDARY_FAILURES as error:
                end_error = error
            self.capturing_surface = previous_capture
            self.surface_raster_error = previous_raster_error
            if end_error is not None:
                raise EffectRasterAllocationError(
                    'unable to finish effect source painter'
                ) from end_error
        return source

    @staticmethod
    def _pixmap_alpha(pixmap: QPixmap) -> np.ndarray:
        try:
            rgba = pixmap2ndarray(pixmap, keep_alpha=True)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                'unable to access text effect source pixels'
            ) from error
        if rgba is None:
            raise EffectRasterAllocationError(
                'unable to access text effect source pixels'
            )
        return rgba[..., 3].copy()

    def _shadow_pixmap(
        self,
        source_alpha: np.ndarray,
        shadow: ShadowEffect,
        render_scale: float,
    ) -> QPixmap:
        blur, spread, xoffset, yoffset = self._shadow_metrics(shadow)
        try:
            rgba = render_shadow_rgba(
                source_alpha,
                shadow.shadow_type,
                shadow.color,
                shadow.opacity,
                (
                    xoffset * render_scale,
                    yoffset * render_scale,
                ),
                max(0, int(round(blur * render_scale))),
                max(0, int(round(spread * render_scale))),
            )
            pixmap = ndarray2pixmap(rgba)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                f'unable to allocate typed shadow surface: {error}'
            ) from error
        if pixmap is None or pixmap.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate typed shadow surface'
            )
        pixmap.setDevicePixelRatio(render_scale)
        return pixmap

    def repaint_background(self, render_scale: float = 1.0) -> None:
        self.item.refresh_cache_policy()
        empty = self.document().isEmpty()
        if (
            self.repainting
            or (self.reshaping and not self._export_active)
            or (self.pre_editing and not self._export_active)
        ):
            # Avoid reshape/reentrant work. During IME, reuse the preedit-free
            # cache because PaintContext cannot exclude active preedit glyphs.
            return

        self.repainting = True
        try:
            self._sync_native_stroke_alignment()
        finally:
            self.repainting = False
        self._update_effect_padding()

        paint_stroke, paint_non_stroke = self._effect_flags()
        if not paint_non_stroke and not paint_stroke or empty:
            changed = self.background_pixmap is not None
            self.background_pixmap = None
            self.background_pixmap_scale = None
            state = self._peek_raster_state()
            if state is not None:
                state.tile_cache.clear()
            self._drop_active_raster_state()
            if changed:
                self.item.update()
            return

        self.tile_cache.clear()
        self.repainting = True
        try:
            br = self.boundingRect()
            plan = plan_effect_raster(
                br.width(),
                br.height(),
                quality_raster_request(render_scale),
            )
            if plan.mode == 'tiles':
                self.background_pixmap = None
                self.background_pixmap_scale = None
                self.direct_stroke = False
                # Visible tiles are intentionally deferred until QPainter's
                # exposed/clip rectangle is available.
                return
            try:
                target_map = self._render_effect_surface(br, plan.tier)
            except EFFECT_RASTER_FAILURES as error:
                # A higher tier may fail despite satisfying the deterministic
                # caps. Retry the smallest full tier before degrading.
                retry = plan_effect_raster(br.width(), br.height(), 1.0)
                if plan.tier != 1.0 and retry.mode == 'full':
                    try:
                        target_map = self._render_effect_surface(br, 1.0)
                        plan = retry
                    except EFFECT_RASTER_FAILURES as retry_error:
                        error = retry_error
                        target_map = None
                else:
                    target_map = None
                if target_map is None:
                    self.background_pixmap = None
                    self.background_pixmap_scale = None
                    self.cache_dirty = True
                    self.cache_rendered_generation = -1
                    if self.export_render:
                        # A policy-valid full allocation can still fail at
                        # runtime. Export gets one bounded visible-tile retry
                        # before the transaction is failed.
                        self.direct_stroke = False
                        self.force_tiles = True
                        return
                    self.direct_stroke = paint_stroke
                    self._warn_effect_allocation_once(error)
                    return

            self.background_pixmap = target_map
            self.background_pixmap_scale = plan.tier
            self.direct_stroke = False
            self.force_tiles = False
            self.cache_dirty = False
            self.cache_rendered_generation = self.cache_generation
            self._raster_state().cache_input_key = (
                self._effect_cache_input_key()
            )
        finally:
            self.repainting = False
        self.item.update()


    def _mark_effect_cache_dirty(self) -> None:
        state = self._raster_state()
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_input_key = None
        state.tile_cache.clear()
        # Never combine a previous glyph silhouette with a new fill angle.
        self.background_pixmap = None
        self.background_pixmap_scale = None

    def _visible_effect_rect(
        self, painter: QPainter, exposed_rect: QRectF = None
    ) -> QRectF:
        visible = QRectF(self.boundingRect())
        if exposed_rect is not None and not exposed_rect.isEmpty():
            visible = visible.intersected(exposed_rect)
        if painter.hasClipping():
            clip = painter.clipBoundingRect()
            if not clip.isEmpty():
                visible = visible.intersected(clip)
        return visible

    def _draw_tiled_effects(
        self,
        painter: QPainter,
        plan: EffectRasterPlan,
        exposed_rect: QRectF = None,
    ) -> None:
        br = self.boundingRect()
        visible = self._visible_effect_rect(painter, exposed_rect)
        if visible.isEmpty():
            return

        paint_stroke, paint_non_stroke = self._effect_flags()
        stroke_overlap = self._stroke_outset() + EFFECT_RASTER_GUARD
        vector_stroke_direct = (
            paint_stroke
            and not paint_non_stroke
            # The vector fallback cannot apply the block-wide alpha mask.
            and self._active_text_alpha_mask() is None
            and 2 * math.ceil(stroke_overlap * plan.tier)
            >= plan.tile_edge
        )
        target_overlap = (
            EFFECT_RASTER_GUARD
            if vector_stroke_direct
            else self._effect_tile_overlap()
        )
        if vector_stroke_direct:
            self.tile_cache.clear()
            self.direct_stroke = True
            self.cache_dirty = False
            self.cache_rendered_generation = self.cache_generation
            self._raster_state().cache_input_key = (
                self._effect_cache_input_key()
            )
            self.force_tiles = False
            return
        overlap_px = math.ceil(target_overlap * plan.tier)
        surface_overlap = overlap_px / plan.tier
        core_edge_px = plan.tile_edge - 2 * overlap_px
        if core_edge_px < 1:
            error = EffectRasterAllocationError(
                'effect overlap exceeds bounded tile surface'
            )
            if self._raise_or_defer_export_effect_error(error):
                return
            self._warn_effect_allocation_once(error)
            self.direct_stroke = paint_stroke
            return
        core_edge = core_edge_px / plan.tier

        first_x = max(
            0, int(math.floor((visible.left() - br.left()) / core_edge))
        )
        first_y = max(
            0, int(math.floor((visible.top() - br.top()) / core_edge))
        )
        last_x = max(
            first_x,
            int(
                math.floor(
                    (math.nextafter(visible.right(), -math.inf) - br.left())
                    / core_edge
                )
            ),
        )
        last_y = max(
            first_y,
            int(
                math.floor(
                    (math.nextafter(visible.bottom(), -math.inf) - br.top())
                    / core_edge
                )
            ),
        )

        active_keys = set()
        staging_pixmap = None
        staging_painter = None
        tile_painter = painter
        raster_failure = None
        try:
            if not self.export_render:
                staging_plan = plan_effect_raster(
                    visible.width(), visible.height(), plan.tier
                )
                if (
                    staging_plan.mode != 'full'
                    or staging_plan.tier != plan.tier
                ):
                    raise EffectRasterAllocationError(
                        'visible effect staging surface exceeds policy'
                    )
                staging_pixmap = self._new_effect_pixmap(
                    plan.tier, visible
                )
                staging_painter = QPainter(staging_pixmap)
                if not staging_painter.isActive():
                    raise EffectRasterAllocationError(
                        'unable to begin visible effect staging painter'
                    )
                staging_painter.translate(-visible.topLeft())
                tile_painter = staging_painter
            tile_painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform
            )
            for tile_y in range(first_y, last_y + 1):
                for tile_x in range(first_x, last_x + 1):
                    core = QRectF(
                        br.left() + tile_x * core_edge,
                        br.top() + tile_y * core_edge,
                        core_edge,
                        core_edge,
                    ).intersected(br)
                    if core.isEmpty():
                        continue
                    surface = core.adjusted(
                        -surface_overlap,
                        -surface_overlap,
                        surface_overlap,
                        surface_overlap,
                    ).intersected(br)
                    key = (
                        self.cache_generation,
                        plan.tier,
                        tile_x,
                        tile_y,
                        round(surface.left(), 6),
                        round(surface.top(), 6),
                        round(surface.width(), 6),
                        round(surface.height(), 6),
                        vector_stroke_direct,
                    )
                    active_keys.add(key)
                    cached = self.tile_cache.get(key)
                    if cached is None:
                        pixmap = self._render_effect_surface(
                            surface,
                            plan.tier,
                            target_stroke=not vector_stroke_direct,
                        )
                        cached = (QRectF(surface), pixmap)
                        self.tile_cache[key] = cached
                        while len(self.tile_cache) > 2:
                            oldest = next(iter(self.tile_cache))
                            if oldest == key and len(self.tile_cache) > 1:
                                oldest = next(
                                    candidate
                                    for candidate in self.tile_cache
                                    if candidate != key
                                )
                            self.tile_cache.pop(oldest, None)
                    tile_painter.save()
                    try:
                        tile_painter.setClipRect(
                            core, Qt.ClipOperation.IntersectClip
                        )
                        tile_painter.drawPixmap(
                            cached[0].topLeft(), cached[1]
                        )
                    finally:
                        tile_painter.restore()
        except RASTER_BOUNDARY_FAILURES as error:
            raster_failure = (
                error
                if isinstance(error, EFFECT_RASTER_FAILURES)
                else EffectRasterAllocationError(
                    'unable to render tiled effect surface'
                )
            )
            if raster_failure is not error:
                raster_failure.__cause__ = error
        finally:
            if staging_painter is not None:
                try:
                    if staging_painter.isActive():
                        staging_painter.end()
                except RASTER_BOUNDARY_FAILURES as error:
                    if raster_failure is None:
                        raster_failure = EffectRasterAllocationError(
                            'unable to finish tiled effect painter'
                        )
                        raster_failure.__cause__ = error

        if raster_failure is not None:
            self.tile_cache.clear()
            self.direct_stroke = paint_stroke
            self.cache_dirty = True
            self.cache_rendered_generation = -1
            if self._raise_or_defer_export_effect_error(raster_failure):
                return
            self._warn_effect_allocation_once(raster_failure)
            return

        if staging_pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(visible.topLeft(), staging_pixmap)

        # Retain no cache from a viewport that is no longer exposed.
        for key in list(self.tile_cache):
            if key not in active_keys:
                self.tile_cache.pop(key, None)

        self.direct_stroke = vector_stroke_direct
        self.cache_dirty = False
        self.cache_rendered_generation = self.cache_generation
        self._raster_state().cache_input_key = (
            self._effect_cache_input_key()
        )
        self.force_tiles = False

    def _draw_direct_stroke(self, painter: QPainter):
        if not self._effect_flags()[0]:
            return
        # This path intentionally avoids every intermediate raster allocation.
        # The custom glyph renderer still consumes outline selections, while a
        # native box transform keeps the unclipped cloned-document stroke.
        previous = self._outline_only_stroke
        self._outline_only_stroke = self._hollow_enabled()
        try:
            self._paint_strokes(
                painter, lambda: self._paint_source_local_stroke(painter)
            )
        finally:
            self._outline_only_stroke = previous

    def _draw_effects(
        self, painter: QPainter, exposed_rect: QRectF = None
    ) -> None:
        painter.save()
        try:
            paint_stroke, paint_non_stroke = self._effect_flags()
            if not paint_stroke and not paint_non_stroke:
                return
            # A preview can park committed pixels while content or another
            # effect changes. Validate semantics at the final reuse boundary
            # so cancellation cannot revive that stale surface.
            self._invalidate_stale_active_raster_state()
            br = self.boundingRect()
            requested_scale = self._paint_device_scale(painter)
            plan = plan_effect_raster(
                br.width(),
                br.height(),
                quality_raster_request(requested_scale),
            )
            if self.force_tiles:
                plan = EffectRasterPlan(
                    'tiles', 1.0, 0, 0, EFFECT_TILE_MAX_EDGE
                )
            stale = (
                self.cache_rendered_generation
                != self.cache_generation
            )
            if plan.mode == 'full':
                if (
                    (not self.pre_editing or self._export_active)
                    and (
                        self.background_pixmap is None
                        or self.background_pixmap_scale != plan.tier
                        or self.cache_dirty
                        or stale
                    )
                ):
                    self.repaint_background(requested_scale)
                if self.force_tiles:
                    tile_plan = EffectRasterPlan(
                        'tiles', 1.0, 0, 0, EFFECT_TILE_MAX_EDGE
                    )
                    self._draw_tiled_effects(
                        painter, tile_plan, exposed_rect
                    )
                    if self.direct_stroke:
                        self._draw_direct_stroke(painter)
                    return
                if (
                    self.background_pixmap is not None
                    and self.background_pixmap_scale == plan.tier
                    and self.cache_rendered_generation
                    == self.cache_generation
                ):
                    painter.setRenderHint(
                        QPainter.RenderHint.SmoothPixmapTransform
                    )
                    painter.drawPixmap(br.topLeft(), self.background_pixmap)
                elif self.direct_stroke:
                    self._draw_direct_stroke(painter)
            else:
                # A previous ordinary-size fast cache must never be stretched
                # over a new huge local surface.
                self.background_pixmap = None
                self.background_pixmap_scale = None
                self._draw_tiled_effects(painter, plan, exposed_rect)
                if self.direct_stroke:
                    self._draw_direct_stroke(painter)
        finally:
            painter.restore()

    @staticmethod
    def _paint_device_scale(painter: QPainter) -> float:
        transform = painter.deviceTransform()
        a, b = transform.m11(), transform.m21()
        c, d = transform.m12(), transform.m22()
        trace = a * a + b * b + c * c + d * d
        determinant_squared = (a * d - b * c) ** 2
        discriminant = max(0.0, trace * trace - 4 * determinant_squared)
        scale = math.sqrt((trace + math.sqrt(discriminant)) / 2)
        if scale <= 0:
            return 1.0
        return min(max(1.0, scale), EFFECT_CACHE_MAX_SCALE)

    def _refresh_gradient_geometry(self):
        """Refresh the block-local gradient as non-document layout state."""
        if self.refreshing_gradient_geometry:
            return
        neutral = self._text_transform_is_neutral()
        if neutral and not self.has_transient_gradient_ranges:
            return
        self.refreshing_gradient_geometry = True
        gradient_format = None
        if not neutral and self.fontformat.gradient_enabled:
            gradient_format = QTextCharFormat()
            gradient_format.setForeground(self.get_text_gradient())
            gradient_format.setProperty(GRADIENT_LAYOUT_FORMAT_PROPERTY, True)
        try:
            formats_changed = False
            transient_present = False
            block = self.document().firstBlock()
            while block.isValid():
                layout = block.layout()
                old_ranges = layout.formats()
                ranges = []
                removed_transient = False
                for format_range in old_ranges:
                    if bool(
                        format_range.format.property(
                            GRADIENT_LAYOUT_FORMAT_PROPERTY
                        )
                    ):
                        removed_transient = True
                    else:
                        ranges.append(format_range)
                text_length = block.length() - 1
                add_transient = gradient_format is not None and text_length > 0
                if add_transient:
                    transient_present = True
                    format_range = QTextLayout.FormatRange()
                    format_range.start = 0
                    format_range.length = text_length
                    format_range.format = gradient_format
                    ranges.append(format_range)
                if removed_transient or add_transient:
                    layout.setFormats(ranges)
                    formats_changed = True
                block = block.next()
            if formats_changed:
                # setFormats invalidates QTextLine objects. Rebuild them through
                # the attached custom layout; this changes no document state.
                self.layout.reLayout()
                self.update()
            self.has_transient_gradient_ranges = transient_present
        finally:
            self.refreshing_gradient_geometry = False

    def get_text_gradient(
        self,
        fontformat: FontFormat = None,
        *,
        persistent: bool = False,
    ):
        gradient = QLinearGradient()
        if fontformat is None:
            fontformat = self.fontformat
        angle = fontformat.gradient_angle
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)

        # Set gradient points with size adjustment
        if persistent and not self._text_transform_is_neutral():
            # The document foreground is the neutral fallback underneath the
            # active layout-only range. Use the current non-distorting padding
            # so removing the range reveals the same gradient coordinates.
            logical_rect = self.logical_unpadded_rect()
            neutral_padding = self._conservative_effect_padding()
            rect = QRectF(
                0.0,
                0.0,
                logical_rect.width() + neutral_padding * 2,
                logical_rect.height() + neutral_padding * 2,
            )
        else:
            rect = (
                self.boundingRect()
                if self._text_transform_is_neutral()
                else self.logical_unpadded_rect()
            )
        center = rect.center()
        radius = max(rect.width(), rect.height()) * fontformat.gradient_size
        gradient.setStart(center.x() - dx * radius, center.y() - dy * radius)
        gradient.setFinalStop(center.x() + dx * radius, center.y() + dy * radius)

        # Set gradient colors
        start_color = QColor(*fontformat.gradient_start_color)
        end_color = QColor(*fontformat.gradient_end_color)
        gradient.setColorAt(0, start_color)
        gradient.setColorAt(1, end_color)
        return gradient
