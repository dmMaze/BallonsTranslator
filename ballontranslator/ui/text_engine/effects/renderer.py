"""Typed effects, block alpha masks, and completed text rendering."""

import math
from typing import Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
from qtpy.QtCore import QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextLayout,
)
from qtpy.QtWidgets import QStyle, QStyleOptionGraphicsItem, QWidget

from ballontranslator.utils.fontformat import SYNTHETIC_BOLD_OFFSET_MAX, pt2px
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    HollowEffect,
    ImageEffect,
    LinearGradientPaint,
    ShadowEffect,
    StrokeEffect,
    TextFillEffect,
    TextEffect,
    TextEffectStack,
    TexturePaint,
    effect_phase,
    effect_paint_fallback_color,
    hollow_effect,
    primary_stroke,
)
from ...misc import ndarray2pixmap, pixmap2ndarray
from ..horizontal_layout import HorizontalTextDocumentLayout
from ..vertical_layout import VerticalTextDocumentLayout
from .alpha_mask import render_text_alpha_mask
from .blend import CUSTOM_BLEND_MODES, composite_custom_blend_rgba
from .paint import (
    colorize_effect_paint_rgba,
    colorize_texture_paint_rgba,
)
from ..rendering.glyph import (
    GLYPH_DILATED_STROKE_FORMAT_PROPERTY,
    GLYPH_FEEDBACK_ONLY_FORMAT_PROPERTY,
    GLYPH_STROKE_FORMAT_PROPERTY,
)
from .filters import (
    FilterContext,
    FilterRuntime,
    get_filter_registry,
)
from .shadow import render_glow_alpha, render_shadow_alpha
from .limits import limit_effect_radii
from ..rendering.morphology import dilate_alpha_disc
from ..rendering.raster import (
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


STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY = 0x100000 + 1241
_STROKE_ALIGNMENT_RANGE_LENGTH = 0x7FFFFFFF
# Glyph Slant writes vector paths into effect pixmaps, not native text.
_VECTOR_EFFECT_RENDER_HINTS = (
    QPainter.RenderHint.Antialiasing
    | QPainter.RenderHint.TextAntialiasing
)
_BLEND_COMPOSITION_MODES = {
    'normal': QPainter.CompositionMode.CompositionMode_SourceOver,
    'darken': QPainter.CompositionMode.CompositionMode_Darken,
    'multiply': QPainter.CompositionMode.CompositionMode_Multiply,
    'color_burn': QPainter.CompositionMode.CompositionMode_ColorBurn,
    'lighten': QPainter.CompositionMode.CompositionMode_Lighten,
    'screen': QPainter.CompositionMode.CompositionMode_Screen,
    'color_dodge': QPainter.CompositionMode.CompositionMode_ColorDodge,
}
_FILTER_HALO_MAX_PIXELS = 512
_FILTER_WARNING_LIMIT = 64
_FilterExecutionPlan = Tuple[
    Tuple[int, FilterEffect, FilterRuntime, int], ...
]


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
        # Mask previews only change the final alpha. Keep at most two complete
        # pre-mask surfaces so visible full/tile output can be derived cheaply.
        self.pre_mask_cache = {}
        # Filter previews reuse the fixed base and generated prefix below the
        # bottom Filter. The two entries mirror the existing full/tile bound.
        self.pre_filter_cache = {}
        # Effect paint does not change the canonical glyph pixels. Retain at
        # most the same two full/tile source captures across paint previews.
        self.effect_source_cache = {}
        # Stroke paint and opacity consume, but do not change, native outline
        # coverage. Keep the same bounded full/tile working set.
        self.positioned_stroke_coverage_cache: Dict[
            tuple, np.ndarray
        ] = {}


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

    _NO_MASK_PREVIEW = object()

    def __init__(self, item) -> None:
        self.item = item
        self._effect_raster_state = None
        self._preview_effect_raster_state = None
        self._export_effect_raster_state = None
        self._export_active = False
        self._mask_generation = 0
        self._mask_preview_generation = 0
        self._mask_preview = self._NO_MASK_PREVIEW
        self._mask_preview_changes_pixels = False
        self.preview = None
        self.faster_preview = False
        self._render_stroke = None
        self._outline_only_stroke = False
        self._native_stroke_alignment = False
        self.refreshing_effect_padding = False
        self._verified_export_assets: Set[
            Tuple[object, RasterAssetRef]
        ] = set()
        self._filter_warnings: Set[tuple] = set()
        self._radius_limit_cache = None

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

    def _bounded_text_effects(self, stack: TextEffectStack) -> TextEffectStack:
        """Keep at least half a tile for its core, at the settled 1x tier."""
        if self.layout is None:
            return stack
        font_size = self.layout.max_font_size(to_px=True)
        reach = max(
            0.0,
            (EFFECT_TILE_MAX_EDGE - 2) / 4.0
            - EFFECT_RASTER_GUARD - self._synthetic_bold_outset(),
        )
        key = (stack, font_size, reach)
        if self._radius_limit_cache is None or self._radius_limit_cache[0] != key:
            self._radius_limit_cache = (key, limit_effect_radii(stack, font_size, reach))
        return self._radius_limit_cache[1]

    def has_preview(self) -> bool:
        return self.preview is not None or self.has_text_alpha_mask_preview()

    def uses_preview_surface(self) -> bool:
        """Return whether preview changes source-surface pixels or geometry."""
        return self._uses_preview_cache_namespace()

    def uses_faster_preview_surface(self) -> bool:
        """Return whether an effect-stack preview selected the 0.5x path."""
        return self.faster_preview and self._effect_preview_changes_pixels()

    def has_active_effects(self) -> bool:
        return self.effective_text_effects().has_active_effects

    def has_synthetic_bold(self) -> bool:
        return any(value > 0.0 for value in self._synthetic_bold_ratios())

    def has_raster_effects(self) -> bool:
        """Return whether strict export must own the complete effect output."""
        return (
            any(self._effect_flags())
            or self._renders_completed_foreground()
            or bool(self._active_image_effects(
                self.canonical_text_effects(), suppress_editing=False
            ))
            or any(
                isinstance(effect, FilterEffect) and effect.enabled
                for effect in self.canonical_text_effects().effects
            )
        )

    def has_generated_effect_layers(self) -> bool:
        """Return whether font/geometry changes invalidate generated layers."""
        return any(self._effect_flags())

    def surface_semantic_state(self) -> tuple:
        """Return effect values that change completed source-surface pixels."""
        return (
            self._surface_effect_values(self.effective_text_effects()),
            self.fontformat.synthetic_bold,
            self._synthetic_bold_ratios(),
            self._effective_mask_generation(),
        )

    def _surface_effect_values(
        self, stack: TextEffectStack
    ) -> Tuple[TextEffect, ...]:
        """Return stack values after native-edit Image suppression."""
        stack = self._bounded_text_effects(stack)
        if not self.item.isEditing() or self.export_render:
            return stack.effects
        return tuple(
            effect
            for effect in stack.effects
            if not isinstance(effect, ImageEffect)
        )

    def _active_image_effects(
        self,
        stack: Optional[TextEffectStack] = None,
        *,
        suppress_editing: bool = True,
    ) -> Tuple[ImageEffect, ...]:
        active = self.effective_text_effects() if stack is None else stack
        if (
            suppress_editing
            and self.item.isEditing()
            and not self.export_render
        ):
            return ()
        return tuple(
            effect
            for effect in active.effects
            if isinstance(effect, ImageEffect) and not effect.is_neutral()
        )

    def canonical_text_alpha_mask(self) -> Optional[TextAlphaMask]:
        block = getattr(self.item, 'blk', None)
        return None if block is None else block.text_alpha_mask

    def effective_text_alpha_mask(self) -> Optional[TextAlphaMask]:
        """Return preview mask interactively and canonical mask for export."""
        if self._export_active or not self.has_text_alpha_mask_preview():
            return self.canonical_text_alpha_mask()
        return self._mask_preview

    def has_text_alpha_mask_preview(self) -> bool:
        return self._mask_preview is not self._NO_MASK_PREVIEW

    def _effect_preview_changes_pixels(self) -> bool:
        return bool(
            self.preview is not None
            and self._bounded_text_effects(self.preview).effects
            != self._bounded_text_effects(self.canonical_text_effects()).effects
        )

    def _effective_mask_generation(self) -> int:
        if (
            not self._export_active
            and self._mask_preview_changes_pixels
        ):
            return -self._mask_preview_generation - 1
        return self._mask_generation

    def _active_text_alpha_mask(self) -> Optional[TextAlphaMask]:
        mask = self.effective_text_alpha_mask()
        return mask if mask is not None and not mask.is_neutral() else None

    def _mask_requires_surface(self) -> bool:
        return (
            self._active_text_alpha_mask() is not None
            and not self._hollow_enabled()
        )

    def _uses_preview_cache_namespace(self) -> bool:
        return (
            self._mask_preview_changes_pixels
            or self._effect_preview_changes_pixels()
        )

    def _active_strokes(
        self, stack: Optional[TextEffectStack] = None
    ) -> Tuple[StrokeEffect, ...]:
        active = self.effective_text_effects() if stack is None else stack
        active = self._bounded_text_effects(active)
        return tuple(
            effect
            for effect in active.effects
            if isinstance(effect, StrokeEffect) and not effect.is_neutral()
        )

    def _active_text_fills(
        self, stack: Optional[TextEffectStack] = None
    ) -> Tuple[TextFillEffect, ...]:
        active = self.effective_text_effects() if stack is None else stack
        return tuple(
            effect
            for effect in reversed(active.effects)
            if isinstance(effect, TextFillEffect)
            and not effect.is_neutral()
        )

    def _active_filters(
        self, stack: Optional[TextEffectStack] = None
    ) -> Tuple[Tuple[int, FilterEffect], ...]:
        active = self.effective_text_effects() if stack is None else stack
        return tuple(
            (index, effect)
            for index, effect in enumerate(active.effects)
            if isinstance(effect, FilterEffect) and effect.enabled
        )

    def _ordered_surface_nodes(
        self,
        *,
        target_stroke: bool = True,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
        strict_assets: bool = True,
    ) -> Tuple[Tuple[int, TextEffect], ...]:
        """Return visible stack nodes in bottom-to-top execution order.

        Text Fill is the permanent canonical face and Hollow is a structural
        modifier. Generated layers retain their canonical geometry/source;
        only their composition relative to Filters follows global card order.

        >>> hasattr(TextEffectRenderer, '_ordered_surface_nodes')
        True
        """
        if image_rasters is None:
            image_rasters = {}
        hollow = self._hollow_enabled()
        editing = self.item.isEditing() and not self.export_render
        nodes = []
        for index, effect in reversed(tuple(enumerate(
            self._bounded_text_effects(self.effective_text_effects()).effects
        ))):
            if isinstance(effect, ImageEffect):
                if editing or effect.is_neutral():
                    continue
                nodes.append((index, effect))
                continue
            if isinstance(effect, FilterEffect):
                if effect.enabled:
                    nodes.append((index, effect))
                continue
            if isinstance(effect, StrokeEffect):
                if (
                    target_stroke
                    and not effect.is_neutral()
                ):
                    nodes.append((index, effect))
                continue
            if isinstance(effect, (ShadowEffect, GlowEffect)):
                if effect.is_neutral():
                    continue
                if hollow and effect_phase(effect) == 'interior':
                    continue
                nodes.append((index, effect))
        return tuple(
            node
            for node in nodes
            if not isinstance(node[1], ImageEffect)
            or self._image_raster(
                node[1], image_rasters, strict_export=strict_assets
            ) is not None
        )

    def _retained_strokes(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> Tuple[StrokeEffect, ...]:
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        return tuple(
            effect
            for _index, effect in retained
            if isinstance(effect, StrokeEffect)
        )

    def _retained_phase_effects(
        self,
        phase: str,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> Tuple[TextEffect, ...]:
        if phase not in {'exterior', 'interior'}:
            raise ValueError('generated phase must be exterior or interior')
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        return tuple(
            effect
            for _index, effect in retained
            if isinstance(effect, (ShadowEffect, GlowEffect))
            and effect_phase(effect) == phase
        )

    def _stroke_sources_for_nodes(
        self,
        nodes: Tuple[Tuple[int, TextEffect], ...],
    ) -> Tuple[StrokeEffect, ...]:
        """Return painted Strokes plus canonical exterior dependencies."""
        if self._retained_phase_effects('exterior', nodes):
            return self._active_strokes()
        return self._retained_strokes(nodes)

    @staticmethod
    def _retained_filter_indices(
        nodes: Tuple[Tuple[int, TextEffect], ...],
    ) -> frozenset[int]:
        return frozenset(
            index
            for index, effect in nodes
            if isinstance(effect, FilterEffect)
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
            self._surface_effect_values(active),
            self._effective_mask_generation(),
            self.document().revision(),
            layout_generation,
            layout_render_key,
            self.geometry_controller.effective(),
            self.fontformat.vertical,
            (
                rect.x(), rect.y(), rect.width(), rect.height()
            ),
            (self.fontformat.synthetic_bold, self._synthetic_bold_ratios()),
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

    @staticmethod
    def _effect_cache_key_without_mask(cache_key: tuple) -> tuple:
        return cache_key[:1] + cache_key[2:]

    @staticmethod
    def _effect_cache_key_before_bottom_filter(
        cache_key: tuple,
        ordered_nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> tuple:
        """Describe the reusable base and layers below the bottom Filter.

        Filter values themselves are deliberately excluded so parameter-only
        previews can reuse this prefix. Their positions still choose the
        prefix boundary, which prevents a reorder across a generated layer
        from reusing the wrong pixels.

        >>> callable(TextEffectRenderer._effect_cache_key_before_bottom_filter)
        True
        """
        effects = cache_key[0]
        if ordered_nodes is None:
            filter_indices = tuple(
                index
                for index, effect in enumerate(effects)
                if isinstance(effect, FilterEffect) and effect.enabled
            )
            boundary = max(filter_indices) if filter_indices else -1
            structural = tuple(
                effect
                for effect in effects
                if isinstance(
                    effect, (HollowEffect, TextFillEffect, ImageEffect)
                )
            )
            generated_below = tuple(
                effect
                for index, effect in enumerate(effects)
                if index > boundary
                and isinstance(
                    effect, (StrokeEffect, ShadowEffect, GlowEffect)
                )
            )
        else:
            first_filter = next(
                (
                    position
                    for position, (_index, effect)
                    in enumerate(ordered_nodes)
                    if isinstance(effect, FilterEffect)
                ),
                len(ordered_nodes),
            )
            prefix = ordered_nodes[:first_filter]
            boundary = (
                -1
                if first_filter == len(ordered_nodes)
                else ordered_nodes[first_filter][0]
            )
            structural = (
                tuple(
                    effect
                    for effect in effects
                    if isinstance(effect, (HollowEffect, TextFillEffect))
                ),
                prefix,
            )
            generated_below = tuple(
                effect
                for _index, effect in prefix
                if isinstance(effect, (StrokeEffect, ShadowEffect, GlowEffect))
            )
        cache_filter_indices = tuple(
            index
            for index, effect in enumerate(effects)
            if isinstance(effect, FilterEffect) and effect.enabled
        )
        cache_boundary = (
            max(cache_filter_indices) if cache_filter_indices else -1
        )
        exterior_indices = tuple(
            index
            for index, effect in enumerate(effects)
            if index > cache_boundary
            and isinstance(effect, (ShadowEffect, GlowEffect))
            and not effect.is_neutral()
            and effect_phase(effect) == 'exterior'
        )
        exterior_stroke_source = tuple(
            effect
            for index, effect in enumerate(effects)
            if exterior_indices
            and index > min(exterior_indices)
            and isinstance(effect, StrokeEffect)
            and not effect.is_neutral()
        )
        canonical_stroke_alignment = any(
            isinstance(effect, StrokeEffect) and not effect.is_neutral()
            for effect in effects
        )
        return (
            (
                structural,
                generated_below,
                exterior_stroke_source,
                canonical_stroke_alignment,
                boundary,
            ),
        ) + cache_key[2:]

    def _pre_filter_cache_key(
        self,
        surface_rect: QRectF,
        render_scale: float,
        target_stroke: bool,
        nodes: Tuple[Tuple[int, TextEffect], ...],
    ) -> tuple:
        return (
            self._effect_cache_key_before_bottom_filter(
                self._effect_cache_input_key(), nodes
            ),
            float(render_scale),
            round(surface_rect.left(), 6),
            round(surface_rect.top(), 6),
            round(surface_rect.width(), 6),
            round(surface_rect.height(), 6),
            bool(target_stroke),
        )

    def _pre_mask_cache_key(
        self,
        surface_rect: QRectF,
        render_scale: float,
        target_stroke: bool,
        skipped_filters: frozenset[int],
    ) -> tuple:
        return (
            self._effect_cache_key_without_mask(
                self._effect_cache_input_key()
            ),
            float(render_scale),
            round(surface_rect.left(), 6),
            round(surface_rect.top(), 6),
            round(surface_rect.width(), 6),
            round(surface_rect.height(), 6),
            bool(target_stroke),
            tuple(sorted(skipped_filters)),
        )

    def _effect_source_cache_key(
        self,
        surface_rect: QRectF,
        render_scale: float,
    ) -> tuple:
        """Describe only inputs that can change canonical source pixels.

        >>> callable(TextEffectRenderer._effect_source_cache_key)
        True
        """
        document = self.document()
        layout = self.layout
        layout_renderer = self.geometry_controller.layout_renderer
        layout_render_key = (
            None
            if layout_renderer is None
            else layout_renderer.render_cache_key()
        )
        logical_rect = self.logical_unpadded_rect()
        source_rect = self.boundingRect()
        return (
            document.revision(),
            getattr(layout, 'layout_generation', 0),
            layout_render_key,
            self.geometry_controller.effective(),
            self.fontformat.vertical,
            (self.fontformat.synthetic_bold, self._synthetic_bold_ratios()),
            self._native_stroke_alignment,
            (
                logical_rect.x(), logical_rect.y(),
                logical_rect.width(), logical_rect.height(),
            ),
            (
                source_rect.x(), source_rect.y(),
                source_rect.width(), source_rect.height(),
            ),
            (
                surface_rect.x(), surface_rect.y(),
                surface_rect.width(), surface_rect.height(),
            ),
            float(render_scale),
        )

    @staticmethod
    def _copy_source_caches(
        source: Optional[_EffectRasterState],
        target: _EffectRasterState,
    ) -> None:
        if source is not None:
            target.effect_source_cache.update(source.effect_source_cache)
            target.positioned_stroke_coverage_cache.update(
                source.positioned_stroke_coverage_cache
            )

    @staticmethod
    def _copy_pre_mask_cache(
        source: Optional[_EffectRasterState],
        target: _EffectRasterState,
    ) -> None:
        if source is not None:
            target.pre_mask_cache.update(source.pre_mask_cache)

    @staticmethod
    def _copy_pre_filter_cache(
        source: Optional[_EffectRasterState],
        target: _EffectRasterState,
    ) -> None:
        if source is not None:
            target.pre_filter_cache.update(source.pre_filter_cache)

    def _promotable_preview_state(
        self, stack: TextEffectStack
    ) -> Optional[_EffectRasterState]:
        state = self._preview_effect_raster_state
        if (
            self.faster_preview
            or state is None
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
            or state.background_pixmap_scale < plan.tier
        ):
            return None
        return state

    def _promotable_mask_preview_state(
        self, mask: Optional[TextAlphaMask]
    ) -> Optional[_EffectRasterState]:
        state = self._preview_effect_raster_state
        if (
            not self._mask_preview_changes_pixels
            or self._mask_preview != mask
            or state is None
            or state.background_pixmap is None
            or state.cache_dirty
            or state.cache_rendered_generation != state.cache_generation
            or state.cache_input_key != self._effect_cache_input_key()
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

    def _raster_request(self, requested_scale: float) -> float:
        if (
            self.faster_preview
            and not self._export_active
            and self._effect_preview_changes_pixels()
        ):
            return 0.5
        return quality_raster_request(requested_scale)

    def set_faster_preview(self, enabled: bool) -> bool:
        """Choose the existing half-resolution live effect preview path."""
        enabled = bool(enabled)
        if self.faster_preview == enabled:
            return False
        self.faster_preview = enabled
        if self._effect_preview_changes_pixels():
            self._mark_effect_cache_dirty()
            self.item.update()
        return True

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
        return primary_stroke(self._bounded_text_effects(self.effective_text_effects()))

    def _stroke_width(self) -> float:
        stroke = self._current_stroke()
        if stroke is None:
            return 0.0
        # Position clips the same historical native outline; it does not
        # redefine the saved width as an outside-only radius.
        return stroke.width

    def _all_strokes_vector_compatible(
        self,
        strokes: Optional[Tuple[StrokeEffect, ...]] = None,
    ) -> bool:
        active = self._retained_strokes() if strokes is None else strokes
        return all(
            stroke.position == 'center'
            and stroke.blend_mode == 'normal'
            and not isinstance(stroke.paint, LinearGradientPaint)
            for stroke in active
        )

    def _has_inside_strokes(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> bool:
        return any(
            stroke.position == 'inside'
            for stroke in self._retained_strokes(nodes)
        )

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
        return QColor(*effect_paint_fallback_color(stroke.paint))

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

    def requires_no_item_cache(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> bool:
        """Let the effect raster cache see the actual paint-device scale."""
        return any(self._effect_flags(nodes))

    def release_caches(self) -> None:
        """Release every item-owned raster cache before page removal."""
        self._radius_limit_cache = None
        for state in (
            self._effect_raster_state,
            self._preview_effect_raster_state,
            self._export_effect_raster_state,
        ):
            if state is not None:
                state.tile_cache.clear()
                state.pre_mask_cache.clear()
                state.pre_filter_cache.clear()
                state.effect_source_cache.clear()
                state.positioned_stroke_coverage_cache.clear()
        self._effect_raster_state = None
        self._preview_effect_raster_state = None
        self._export_effect_raster_state = None
        self._export_active = False
        self._mask_preview = self._NO_MASK_PREVIEW
        self._mask_preview_changes_pixels = False
        self._verified_export_assets.clear()
        self._filter_warnings.clear()

    def project_assets_changed(self) -> None:
        """Invalidate project raster output after attachment/file recovery."""
        text_fills = self._active_text_fills()
        if (
            not any(
                isinstance(text_fill.paint, TexturePaint)
                for text_fill in text_fills
            )
            and not self._active_image_effects(suppress_editing=False)
        ):
            return
        self._verified_export_assets.clear()
        for state in (
            self._effect_raster_state,
            self._preview_effect_raster_state,
            self._export_effect_raster_state,
        ):
            self._invalidate_project_raster_state(state)
        self.repaint_background()

    def _apply_effective_opacity(self) -> None:
        self.item._set_effective_opacity(
            self.effective_text_effects().overall_opacity
        )

    def _sync_legacy_primary_stroke_view(self) -> None:
        stroke = primary_stroke(self.effective_text_effects())
        if stroke is not None:
            self.item.stroke_qcolor = QColor(
                *effect_paint_fallback_color(stroke.paint)
            )

    @staticmethod
    def _invalidate_raster_state(state: Optional[_EffectRasterState]) -> None:
        if state is None:
            return
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_rendered_generation = -1
        state.cache_input_key = None
        state.tile_cache.clear()
        state.pre_mask_cache.clear()
        state.pre_filter_cache.clear()
        state.effect_source_cache.clear()
        state.positioned_stroke_coverage_cache.clear()
        state.background_pixmap = None
        state.background_pixmap_scale = None

    @staticmethod
    def _invalidate_mask_raster_state(
        state: Optional[_EffectRasterState],
    ) -> None:
        if state is None:
            return
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_rendered_generation = -1
        state.cache_input_key = None
        state.tile_cache.clear()
        state.background_pixmap = None
        state.background_pixmap_scale = None

    @staticmethod
    def _invalidate_project_raster_state(
        state: Optional[_EffectRasterState],
    ) -> None:
        """Drop asset-derived output while retaining canonical text pixels."""
        if state is None:
            return
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_rendered_generation = -1
        state.cache_input_key = None
        state.tile_cache.clear()
        state.pre_mask_cache.clear()
        state.pre_filter_cache.clear()
        state.background_pixmap = None
        state.background_pixmap_scale = None

    def _finish_effect_transition(self, repaint: bool) -> None:
        self._apply_effective_opacity()
        self._sync_legacy_primary_stroke_view()
        image_rasters: Dict[RasterAssetRef, Optional[np.ndarray]] = {}
        nodes = self._ordered_surface_nodes(
            image_rasters=image_rasters,
            strict_assets=self.export_render,
        )
        was_repainting = self.repainting
        self.repainting = True
        try:
            self._sync_native_stroke_alignment(nodes)
        finally:
            self.repainting = was_repainting
        self._update_effect_padding(nodes)
        self.item.refresh_cache_policy(nodes)
        if repaint and not self.reshaping:
            self.repaint_background(
                nodes=nodes,
                image_rasters=image_rasters,
                geometry_prepared=True,
            )
        self.item.update()

    def synthetic_bold_changed(self) -> None:
        self._invalidate_raster_state(self._effect_raster_state)
        self._invalidate_raster_state(self._preview_effect_raster_state)
        self._invalidate_raster_state(self._export_effect_raster_state)
        self._update_effect_padding()
        if (
            self.has_active_effects()
            or self._active_text_alpha_mask() is not None
        ):
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
            had_pixel_preview = self._uses_preview_cache_namespace()
            source_state = (
                None if self._export_active else self._peek_raster_state()
            )
            self.preview = stack
            effects_changed = (
                self._bounded_text_effects(effective_before).effects
                != self._bounded_text_effects(stack).effects
            )
            if effects_changed:
                if stack.effects != canonical.effects:
                    if not had_pixel_preview:
                        preview_state = _EffectRasterState()
                        self._copy_source_caches(
                            source_state, preview_state
                        )
                        self._copy_pre_filter_cache(
                            source_state, preview_state
                        )
                        self._preview_effect_raster_state = preview_state
                        self.geometry_controller.retain_effect_preview_surface()
                    self._mark_effect_cache_dirty()
                elif self._mask_preview_changes_pixels:
                    # The mask scratch remains active, but its upstream effect
                    # phases must return to the canonical stack.
                    self._mark_effect_cache_dirty()
                else:
                    # Returning to canonical effect pixels keeps the complete
                    # preview alive only for its native overall opacity.
                    self._preview_effect_raster_state = None
                    self._finish_effect_transition(False)
                    self.geometry_controller.restore_effect_preview_surface()
                    return True
            self._finish_effect_transition(
                effects_changed and self.faster_preview
            )
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
        effects_changed = (
            self._bounded_text_effects(canonical).effects
            != self._bounded_text_effects(stack).effects
        )
        mask_preview_active = self._mask_preview_changes_pixels
        scratch_state = self._preview_effect_raster_state
        scratch_current = bool(
            scratch_state is not None
            and not scratch_state.cache_dirty
            and scratch_state.cache_rendered_generation
            == scratch_state.cache_generation
            and scratch_state.cache_input_key == self._effect_cache_input_key()
        )
        promoted_state = (
            self._promotable_preview_state(stack)
            if (
                effects_changed
                and preview_before == stack
                and not mask_preview_active
            )
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
        if not mask_preview_active:
            self._preview_effect_raster_state = None
            self.geometry_controller.invalidate_effect_preview_surface()
        if effects_changed:
            if mask_preview_active:
                # Scratch may contain the exact committed effect stack plus a
                # live mask. It remains preview-owned until the mask settles.
                self._invalidate_raster_state(self._effect_raster_state)
                self._invalidate_raster_state(self._export_effect_raster_state)
                self.geometry_controller.invalidate_effect_preview_surface()
                if not (
                    preview_before == stack
                    and scratch_current
                    and scratch_state is self._preview_effect_raster_state
                ):
                    self._mark_effect_cache_dirty()
            elif promoted_state is None:
                self._mark_effect_cache_dirty()
            else:
                promoted_state.cache_generation = committed_generation + 1
                promoted_state.cache_rendered_generation = (
                    promoted_state.cache_generation
                )
                promoted_state.tile_cache.clear()
                self._effect_raster_state = promoted_state
        self._finish_effect_transition(
            effects_changed
            and promoted_state is None
            and preview_before != stack
            and not (
                mask_preview_active
                and preview_before == stack
                and scratch_current
            )
        )
        if (
            mask_preview_active
            and effects_changed
            and preview_before == stack
            and scratch_current
        ):
            current_key = self._effect_cache_input_key()
            if scratch_state is not None:
                scratch_state.cache_input_key = current_key
        elif promoted_state is not None:
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

    def set_text_alpha_mask(
        self,
        mask: Optional[TextAlphaMask],
        *,
        preview: bool = False,
    ) -> bool:
        """Replace one committed mask or its complete transient preview.

        >>> hasattr(TextEffectRenderer, 'clear_text_alpha_mask_preview')
        True
        """
        if mask is not None and not isinstance(mask, TextAlphaMask):
            raise TypeError('live text alpha mask requires TextAlphaMask or None')
        canonical = self.canonical_text_alpha_mask()
        effective_before = self.effective_text_alpha_mask()

        if preview:
            if mask == canonical:
                return self.clear_text_alpha_mask_preview()
            if self.has_text_alpha_mask_preview() and self._mask_preview == mask:
                return False
            first_preview = not self._mask_preview_changes_pixels
            had_pixel_preview = self._uses_preview_cache_namespace()
            source_state = (
                None if self._export_active else self._peek_raster_state()
            )
            self._mask_preview = mask
            self._mask_preview_changes_pixels = True
            self._mask_preview_generation += 1
            if first_preview and not had_pixel_preview:
                preview_state = _EffectRasterState()
                self._copy_pre_mask_cache(source_state, preview_state)
                self._copy_pre_filter_cache(source_state, preview_state)
                self._copy_source_caches(source_state, preview_state)
                self._preview_effect_raster_state = preview_state
                self.geometry_controller.retain_effect_preview_surface()
            self._mark_mask_cache_dirty()
            self.item.refresh_cache_policy()
            if not self.reshaping:
                self.repaint_background()
            self.item.update()
            return True

        canonical_changed = canonical != mask
        had_preview = self._mask_preview_changes_pixels
        if not canonical_changed and not had_preview:
            return False
        preview_matches_commit = bool(had_preview and self._mask_preview == mask)
        effect_preview_active = self._effect_preview_changes_pixels()
        scratch_state = self._preview_effect_raster_state
        scratch_current = bool(
            scratch_state is not None
            and not scratch_state.cache_dirty
            and scratch_state.cache_rendered_generation
            == scratch_state.cache_generation
            and scratch_state.cache_input_key == self._effect_cache_input_key()
        )
        promoted_state = (
            self._promotable_mask_preview_state(mask)
            if canonical_changed and had_preview and not effect_preview_active
            else None
        )
        preview_key = (
            None if promoted_state is None else promoted_state.cache_input_key
        )
        committed_generation = (
            0
            if self._effect_raster_state is None
            else self._effect_raster_state.cache_generation
        )
        if canonical_changed:
            self.item.blk.text_alpha_mask = mask
            self._mask_generation += 1
        self._mask_preview = self._NO_MASK_PREVIEW
        self._mask_preview_changes_pixels = False
        self._invalidate_raster_state(self._export_effect_raster_state)

        if effect_preview_active:
            # The scratch remains preview-owned. Never publish pixels that
            # include a transient effect stack into the canonical namespace.
            if canonical_changed:
                self._invalidate_mask_raster_state(self._effect_raster_state)
                self.geometry_controller.invalidate_effect_preview_surface()
            current_key = self._effect_cache_input_key()
            can_rekey_scratch = bool(
                canonical_changed
                and preview_matches_commit
                and scratch_current
                and scratch_state is self._preview_effect_raster_state
            )
            if can_rekey_scratch and scratch_state is not None:
                scratch_state.cache_input_key = current_key
            else:
                self._mark_mask_cache_dirty()
                if not self.reshaping:
                    self.repaint_background()
            self.item.refresh_cache_policy()
            self.item.update()
            return canonical_changed or effective_before != mask

        self._preview_effect_raster_state = None
        self.geometry_controller.invalidate_effect_preview_surface()

        current_key = self._effect_cache_input_key()
        can_promote = bool(
            promoted_state is not None
            and preview_key is not None
            and self._effect_cache_key_without_mask(preview_key)
            == self._effect_cache_key_without_mask(current_key)
        )
        if can_promote:
            promoted_state.cache_generation = committed_generation + 1
            promoted_state.cache_rendered_generation = (
                promoted_state.cache_generation
            )
            promoted_state.cache_input_key = current_key
            promoted_state.tile_cache.clear()
            self._effect_raster_state = promoted_state
        elif canonical_changed:
            self._invalidate_mask_raster_state(self._effect_raster_state)

        self.geometry_controller.invalidate_surface_cache()
        self.item.refresh_cache_policy()
        if canonical_changed and not can_promote and not self.reshaping:
            self.repaint_background()
        self.item.update()
        return canonical_changed or effective_before != mask

    def clear_text_alpha_mask_preview(self) -> bool:
        if not self.has_text_alpha_mask_preview():
            return False
        changed_pixels = self._mask_preview_changes_pixels
        effect_preview_active = self._effect_preview_changes_pixels()
        self._mask_preview = self._NO_MASK_PREVIEW
        self._mask_preview_changes_pixels = False
        if effect_preview_active:
            self._mark_mask_cache_dirty()
            self.item.refresh_cache_policy()
            if not self.reshaping:
                self.repaint_background()
            self.item.update()
            return True
        self._preview_effect_raster_state = None
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
            state.cache_input_key = current_key
        needs_repaint = bool(
            changed_pixels
            and (any(self._effect_flags()) or self._renders_completed_foreground())
            and (
                state is None
                or state.cache_dirty
                or state.cache_rendered_generation != state.cache_generation
                or state.cache_input_key != current_key
            )
        )
        self.item.refresh_cache_policy()
        if needs_repaint and not self.reshaping:
            self.repaint_background()
        self.geometry_controller.restore_effect_preview_surface()
        self.item.update()
        return True

    def clear_text_effect_preview(self) -> bool:
        if self.preview is None:
            return False
        preview = self.preview
        mask_preview_active = self._mask_preview_changes_pixels
        self.preview = None
        effects_changed = (
            self._bounded_text_effects(preview).effects
            != self._bounded_text_effects(self.canonical_text_effects()).effects
        )
        if effects_changed:
            self._preview_effect_raster_state = None
        elif not mask_preview_active:
            self._preview_effect_raster_state = None
        self._finish_effect_transition(False)
        if effects_changed and mask_preview_active:
            self._mark_effect_cache_dirty()
            if not self.reshaping:
                self.repaint_background()
            return True
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
        if not mask_preview_active:
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
        if self.reshaping and not self._export_active:
            option.state = QStyle.State_None
            base_paint(painter, option, widget)
            return
        if (
            not self.has_active_effects()
            and self._active_text_alpha_mask() is None
        ):
            if self.has_synthetic_bold():
                interaction_option = QStyleOptionGraphicsItem(option)
                if self._draw_cached_synthetic_bold(painter):
                    self._paint_effect_interaction(
                        painter, interaction_option, widget, base_paint
                    )
                    return
                self._paint_synthetic_bold(painter)
            option.state = QStyle.State_None
            base_paint(painter, option, widget)
            return

        # Effects must be composited before the normal fill. DestinationOver
        # against an already opaque scene would discard them.
        was_in_graphics_paint = self.in_graphics_paint
        self.in_graphics_paint = True
        try:
            image_rasters: Dict[
                RasterAssetRef, Optional[np.ndarray]
            ] = {}
            nodes = self._ordered_surface_nodes(
                image_rasters=image_rasters,
                strict_assets=self.export_render,
            )
            flags = self._effect_flags(nodes)
            renders_foreground = self._renders_completed_foreground(nodes)
            if not any(flags) and not renders_foreground:
                option.state = QStyle.State_None
                base_paint(painter, option, widget)
                return
            interaction_option = QStyleOptionGraphicsItem(option)
            if any(flags):
                self._draw_effects(
                    painter,
                    option.exposedRect,
                    nodes=nodes,
                    image_rasters=image_rasters,
                    flags=flags,
                )
            replace_foreground = self._hollow_enabled() or (
                renders_foreground
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

    def _renders_completed_foreground(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> bool:
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        return (
            self._hollow_enabled()
            or bool(self._retained_phase_effects('interior', retained))
            or self._mask_requires_surface()
            or self._has_inside_strokes(retained)
            or bool(self._active_text_fills())
            or any(
                isinstance(effect, (ImageEffect, FilterEffect))
                for _index, effect in retained
            )
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

        A zero-opacity native pass keeps Qt's caret/IME state current and
        captures its selection formats. Ordinary foreground is then muted
        while those selections are replayed over the completed surface; the
        geometry owner paints the deferred caret last.

        >>> hasattr(TextEffectRenderer, '_paint_effect_interaction')
        True
        """
        if self.export_render or not self.item.isEditing():
            return
        layout = self.item.layout
        previous_defer_cursor = layout.defer_cursor_paint
        previous_observer = layout.paint_context_observer
        deferred_cursor_position = -1
        captured_context: Optional[
            QAbstractTextDocumentLayout.PaintContext
        ] = None

        def capture_context(
            context: QAbstractTextDocumentLayout.PaintContext,
        ) -> None:
            nonlocal captured_context
            if previous_observer is not None:
                previous_observer(context)
            # A caret is painted separately. Avoid a second full layout pass
            # unless Qt supplied selection feedback or active IME preedit ink.
            if context.selections or self.pre_editing:
                captured_context = self._editing_feedback_context(context)

        layout.defer_cursor_paint = True
        layout.paint_context_observer = capture_context
        try:
            painter.save()
            try:
                painter.setOpacity(0.0)
                base_paint(painter, option, widget)
                deferred_cursor_position = layout.deferred_cursor_position
            finally:
                painter.restore()
            layout.paint_context_observer = None
            if captured_context is not None:
                self._paint_live_layout(painter, captured_context)
        finally:
            layout.deferred_cursor_position = deferred_cursor_position
            layout.defer_cursor_paint = previous_defer_cursor
            layout.paint_context_observer = previous_observer
        if not self.geometry_controller.uses_surface_warp():
            self.geometry_controller.paint_deferred_cursor(
                painter, None, export_render=False
            )

    def _editing_feedback_context(
        self,
        context: QAbstractTextDocumentLayout.PaintContext,
    ) -> QAbstractTextDocumentLayout.PaintContext:
        """Keep Qt selections while suppressing ordinary foreground paint.

        >>> callable(TextEffectRenderer._editing_feedback_context)
        True
        """
        feedback = QAbstractTextDocumentLayout.PaintContext()
        feedback.clip = QRectF(context.clip)
        feedback.cursorPosition = -1
        feedback.palette = context.palette

        muted = QAbstractTextDocumentLayout.Selection()
        muted.cursor = QTextCursor(self.document())
        muted.cursor.select(QTextCursor.SelectionType.Document)
        muted_format = QTextCharFormat()
        transparent = QColor(0, 0, 0, 0)
        muted_format.setForeground(transparent)
        muted_format.setBackground(transparent)
        muted_format.setTextOutline(QPen(Qt.PenStyle.NoPen))
        muted_format.setUnderlineColor(transparent)
        feedback_base_format = QTextCharFormat(muted_format)
        muted_format.setProperty(GLYPH_FEEDBACK_ONLY_FORMAT_PROPERTY, True)
        muted.format = muted_format
        feedback_selections = [muted]
        for selection in context.selections:
            char_format = selection.format
            copied = QAbstractTextDocumentLayout.Selection()
            copied.cursor = QTextCursor(selection.cursor)
            feedback_format = QTextCharFormat(feedback_base_format)
            feedback_format.merge(char_format)
            if (
                char_format.foreground().style()
                != Qt.BrushStyle.NoBrush
                and not char_format.underlineColor().isValid()
            ):
                feedback_format.setUnderlineColor(
                    char_format.foreground().color()
                )
            copied.format = feedback_format
            feedback_selections.append(copied)
        feedback.selections = feedback_selections
        return feedback

    def finalize_neutral_cache(self) -> None:
        """Invalidate transformed pixels after neutral restoration."""
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

    def _synthetic_bold_uniform_context(
        self, offset_ratio: float
    ) -> QAbstractTextDocumentLayout.PaintContext:
        """Build per-fragment outlines that preserve rich foreground paint."""
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
                    char_format.foreground(),
                    pt2px(point_size) * offset_ratio * 2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                    Qt.PenJoinStyle.RoundJoin,
                )
                effect_format = QTextCharFormat()
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

    def _synthetic_bold_ratios(self) -> Tuple[float, float]:
        if self.fontformat.synthetic_bold == 'none':
            return 0.0, 0.0
        values = self.fontformat.synthetic_bold_offset
        return (
            min(max(float(values[0]), 0.0), SYNTHETIC_BOLD_OFFSET_MAX),
            min(max(float(values[1]), 0.0), SYNTHETIC_BOLD_OFFSET_MAX),
        )

    def _synthetic_bold_outsets(self) -> Tuple[float, float]:
        x_ratio, y_ratio = self._synthetic_bold_ratios()
        font_size = self.layout.max_font_size(to_px=True)
        return font_size * x_ratio, font_size * y_ratio

    def _synthetic_bold_outset(self) -> float:
        return max(self._synthetic_bold_outsets())

    def _dilate_synthetic_bold_alpha(
        self,
        alpha: np.ndarray,
        render_scale: float,
    ) -> np.ndarray:
        """Expand Stroke coverage on the configured synthetic-bold axes.

        >>> hasattr(TextEffectRenderer, '_dilate_synthetic_bold_alpha')
        True
        """
        x_outset, y_outset = self._synthetic_bold_outsets()
        x_radius = math.ceil(x_outset * render_scale)
        y_radius = math.ceil(y_outset * render_scale)
        if x_radius <= 0 and y_radius <= 0:
            return alpha
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE
            if self.fontformat.synthetic_bold == 'ellipse'
            and x_radius > 0 and y_radius > 0
            else cv2.MORPH_RECT,
            (x_radius * 2 + 1, y_radius * 2 + 1),
        )
        return cv2.dilate(alpha, kernel, borderType=cv2.BORDER_CONSTANT)

    def _anisotropic_synthetic_bold_offsets(
        self,
    ) -> Tuple[Tuple[float, float], ...]:
        """Return translations whose union expands ink on requested axes."""
        x_radius, y_radius = self._synthetic_bold_outsets()
        x_steps = max(1, math.ceil(x_radius)) if x_radius > 0 else 0
        y_steps = max(1, math.ceil(y_radius)) if y_radius > 0 else 0
        x_offsets = (
            tuple(
                x_radius * step / x_steps
                for step in range(-x_steps, x_steps + 1)
            )
            if x_steps else (0.0,)
        )
        y_offsets = (
            tuple(
                y_radius * step / y_steps
                for step in range(-y_steps, y_steps + 1)
            )
            if y_steps else (0.0,)
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE
            if self.fontformat.synthetic_bold == 'ellipse'
            and x_steps > 0 and y_steps > 0
            else cv2.MORPH_RECT,
            (2 * x_steps + 1, 2 * y_steps + 1),
        )
        return tuple(
            (x_offset, y_offset)
            for y_index, y_offset in enumerate(y_offsets)
            for x_index, x_offset in enumerate(x_offsets)
            if kernel[y_index, x_index]
        )

    def _paint_synthetic_bold(self, painter: QPainter) -> None:
        """Expand glyph ink without changing the selected font or metrics."""
        x_ratio, y_ratio = self._synthetic_bold_ratios()
        if x_ratio <= 0.0 and y_ratio <= 0.0:
            return
        if (
            self.fontformat.synthetic_bold == 'ellipse'
            and math.isclose(x_ratio, y_ratio, abs_tol=1e-12)
        ):
            self._paint_live_layout(
                painter, self._synthetic_bold_uniform_context(x_ratio)
            )
            return
        context = self._effect_paint_context()
        for x_offset, y_offset in self._anisotropic_synthetic_bold_offsets():
            painter.save()
            try:
                painter.translate(x_offset, y_offset)
                self._paint_live_layout(painter, context)
            finally:
                painter.restore()

    def _draw_cached_synthetic_bold(self, painter: QPainter) -> bool:
        """Draw a tight cached glyph surface without rasterizing its text box."""
        ink_bounds = self.geometry_controller.layout_ink_bounds()
        if ink_bounds.isEmpty():
            ink_bounds = self._native_text_line_bounds()
        if ink_bounds.isEmpty():
            return False
        outset = self._synthetic_bold_outset() + EFFECT_RASTER_GUARD
        surface_rect = ink_bounds.adjusted(-outset, -outset, outset, outset)
        requested_scale = self._paint_device_scale(painter)
        plan = plan_effect_raster(
            surface_rect.width(),
            surface_rect.height(),
            self._raster_request(requested_scale),
        )
        if plan.mode != 'full':
            return False
        try:
            pixmap, _alpha = self._cached_effect_source(
                surface_rect, plan.tier, needs_alpha=False
            )
            self._draw_surface_pixmap(
                painter, surface_rect, pixmap, plan.tier
            )
        except EFFECT_RASTER_FAILURES as error:
            self._warn_effect_allocation_once(error)
            return False
        return True

    def _native_text_line_bounds(self) -> QRectF:
        """Return tight native line bounds when no transform tracks ink."""
        bounds = QRectF()
        block = self.document().firstBlock()
        while block.isValid():
            layout = block.layout()
            layout_position = layout.position()
            for index in range(layout.lineCount()):
                line_bounds = layout.lineAt(index).naturalTextRect()
                line_bounds.translate(layout_position)
                bounds = (
                    line_bounds
                    if bounds.isEmpty()
                    else bounds.united(line_bounds)
                )
            block = block.next()
        if not bounds.isEmpty():
            bounds.translate(self.logical_unpadded_rect().topLeft())
        return bounds

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

    def _stroke_outset(
        self,
        strokes: Optional[Tuple[StrokeEffect, ...]] = None,
    ) -> float:
        """Return the maximum visible Stroke reach outside glyph alpha."""
        strokes = self._retained_strokes() if strokes is None else strokes
        if not strokes:
            return 0.0
        font_size = self.layout.max_font_size(to_px=True)
        return max(
            font_size
            * stroke.width
            * (
                0.0
                if stroke.position == 'inside'
                else 0.5
            )
            for stroke in strokes
        )

    def _stroke_generation_reach(
        self,
        strokes: Optional[Tuple[StrokeEffect, ...]] = None,
    ) -> float:
        """Return the halo needed to generate positioned Stroke tiles."""
        strokes = self._retained_strokes() if strokes is None else strokes
        if not strokes:
            return 0.0
        font_size = self.layout.max_font_size(to_px=True)
        return max(
            font_size
            * stroke.width
            * 0.5
            for stroke in strokes
        )

    def _sync_native_stroke_alignment(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> None:
        """Keep fill and stroke on Qt's same native glyph raster path."""
        if self.layout is None:
            self._native_stroke_alignment = False
            return
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        enabled = bool(self._stroke_sources_for_nodes(retained))
        self._native_stroke_alignment = enabled
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
            if render_scale >= 1.0:
                pixmap.setDevicePixelRatio(render_scale)
            pixmap.fill(Qt.GlobalColor.transparent)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                f'unable to initialize effect surface '
                f'{pixel_width}x{pixel_height}'
            ) from error
        return pixmap

    @staticmethod
    def _prepare_effect_surface_painter(
        painter: QPainter, render_scale: float
    ) -> None:
        """Map logical item coordinates onto a sub-unit preview surface."""
        if render_scale < 1.0:
            painter.scale(render_scale, render_scale)

    def _begin_effect_layer_painter(
        self,
        target: QPixmap,
        surface_rect: QRectF,
        render_scale: float,
    ) -> QPainter:
        """Begin one painter in the shared item-local surface space."""
        painter: Optional[QPainter] = None
        try:
            painter = QPainter(target)
            if not painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin text-effect layer painter'
                )
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            self._prepare_effect_surface_painter(painter, render_scale)
            painter.translate(-surface_rect.topLeft())
            return painter
        except RASTER_BOUNDARY_FAILURES as error:
            if painter is not None and painter.isActive():
                try:
                    painter.end()
                except RASTER_BOUNDARY_FAILURES:
                    pass
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to prepare text-effect layer painter'
            ) from error

    @staticmethod
    def _custom_blend_surface_pixmaps(
        destination: QPixmap,
        source: QPixmap,
        blend_mode: str,
        render_scale: float,
    ) -> QPixmap:
        """Bridge one non-native blend without changing surface coordinates."""
        destination_rgba = pixmap2ndarray(destination, keep_alpha=True)
        source_rgba = pixmap2ndarray(source, keep_alpha=True)
        if destination_rgba is None or source_rgba is None:
            raise EffectRasterAllocationError(
                'unable to read text-effect blend layers'
            )
        result = ndarray2pixmap(composite_custom_blend_rgba(
            destination_rgba, source_rgba, blend_mode
        ))
        if result is None or result.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate blended text-effect surface'
            )
        if render_scale >= 1.0:
            result.setDevicePixelRatio(render_scale)
        return result

    @staticmethod
    def _draw_surface_pixmap(
        painter: QPainter,
        destination: QRectF,
        pixmap: QPixmap,
        render_scale: float,
    ) -> None:
        """Draw physical surface pixels into their explicit logical bounds."""
        if render_scale < 1.0:
            painter.drawPixmap(destination, pixmap, QRectF(pixmap.rect()))
        else:
            painter.drawPixmap(destination.topLeft(), pixmap)

    def _paint_cloned_document_stroke(self, painter: QPainter) -> None:
        """Paint stroke through the BASE cloned-document path."""
        # Qt's native clone preserves UserProperty values and avoids a full
        # HTML serialization/parse cycle on every effect refresh.
        doc = self.document().clone()
        doc.setUndoRedoEnabled(False)
        doc.setDocumentMargin(self.layout.effectPadding())
        cursor = QTextCursor(doc)
        fragments: list[tuple[int, int, QTextCharFormat]] = []
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                fragments.append((
                    fragment.position(),
                    fragment.length(),
                    fragment.charFormat(),
                ))
                it += 1
            block = block.next()

        stroke_pen = QPen(
            self.stroke_qcolor,
            0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        # Applying a format invalidates QTextBlock iterators. Snapshot every
        # fragment before editing so native Qt never advances a stale iterator.
        for position, length, char_format in fragments:
            stroke_pen.setWidthF(
                pt2px(char_format.fontPointSize())
                * self._stroke_width()
            )
            cursor.setPosition(position)
            cursor.setPosition(
                position + length,
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
                self._prepare_effect_surface_painter(
                    source_painter, render_scale
                )
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
                alpha = dilate_alpha_disc(alpha, radius)
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
        if render_scale >= 1.0:
            stroke_pixmap.setDevicePixelRatio(render_scale)
        self._draw_surface_pixmap(
            painter, rect, stroke_pixmap, render_scale
        )
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
    ) -> None:
        for x_offset, y_offset in self._anisotropic_synthetic_bold_offsets():
            painter.save()
            try:
                painter.translate(x_offset, y_offset)
                self._paint_stroke_core(
                    painter, render_scale, surface_rect
                )
            finally:
                painter.restore()

    def _paint_stroke_core(
        self,
        painter: QPainter,
        render_scale: float = 1.0,
        surface_rect: QRectF = None,
    ) -> None:
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
        distance = shadow.distance * font_size
        radians = math.radians(shadow.angle)
        return (
            shadow.blur * font_size,
            shadow.spread * font_size,
            math.cos(radians) * distance,
            math.sin(radians) * distance,
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

    def _glow_metrics(self, glow: GlowEffect) -> Tuple[float, float]:
        font_size = self.layout.max_font_size(to_px=True)
        return glow.size * font_size, glow.spread * font_size

    def _exterior_effect_bounds(
        self, source_bounds: QRectF, effect: TextEffect
    ) -> QRectF:
        if isinstance(effect, ShadowEffect):
            return self._shadowed_bounds(source_bounds, effect)
        if isinstance(effect, GlowEffect) and effect.glow_type == 'outer':
            size, spread = self._glow_metrics(effect)
            return source_bounds.adjusted(
                -size - spread,
                -size - spread,
                size + spread,
                size + spread,
            )
        raise TypeError('exterior bounds require Shadow or Outer Glow')

    def _logical_ink_bounds(self) -> QRectF:
        if self.document().isEmpty() or not self._has_layout_distortion():
            return QRectF()
        return self.geometry_controller.layout_ink_bounds()

    def _effect_padding(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> float:
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        layout_distorted = self._has_layout_distortion()
        if not layout_distorted:
            return self._conservative_effect_padding(retained)
        ink_bounds = self._logical_ink_bounds()
        logical_rect = self.logical_unpadded_rect()
        retained_strokes = self._retained_strokes(retained)
        painted_stroke_outset = self._stroke_outset(retained_strokes)
        source_stroke_outset = self._stroke_outset(
            self._stroke_sources_for_nodes(retained)
        )
        synthetic_outset = self._synthetic_bold_outset()
        painted_stroke_bounds = ink_bounds.adjusted(
            -painted_stroke_outset - synthetic_outset,
            -painted_stroke_outset - synthetic_outset,
            painted_stroke_outset + synthetic_outset,
            painted_stroke_outset + synthetic_outset,
        )
        exterior_source_bounds = ink_bounds.adjusted(
            -source_stroke_outset - synthetic_outset,
            -source_stroke_outset - synthetic_outset,
            source_stroke_outset + synthetic_outset,
            source_stroke_outset + synthetic_outset,
        )
        filter_expansion = self._filter_expansion_by_index(retained)
        effect_bounds = ink_bounds.adjusted(
            -synthetic_outset,
            -synthetic_outset,
            synthetic_outset,
            synthetic_outset,
        )
        exterior = False
        for index, effect in retained:
            if isinstance(effect, ImageEffect):
                effect_bounds = effect_bounds.united(logical_rect)
            elif isinstance(effect, StrokeEffect):
                if not ink_bounds.isEmpty():
                    effect_bounds = effect_bounds.united(
                        painted_stroke_bounds
                    )
            elif (
                isinstance(effect, (ShadowEffect, GlowEffect))
                and effect_phase(effect) == 'exterior'
            ):
                if not ink_bounds.isEmpty():
                    exterior = True
                    effect_bounds = effect_bounds.united(
                        self._exterior_effect_bounds(
                            exterior_source_bounds, effect
                        )
                    )
            elif isinstance(effect, FilterEffect):
                expansion = filter_expansion.get(index, 0.0)
                if expansion > 0.0 and not effect_bounds.isEmpty():
                    effect_bounds = effect_bounds.adjusted(
                        -expansion, -expansion, expansion, expansion
                    )
        if effect_bounds.isEmpty():
            return 0.0
        if painted_stroke_outset > 0.0 or exterior:
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

    def _conservative_effect_padding(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> float:
        """Return cheap symmetric padding for non-distorting glyph paths."""
        if self.layout is None:
            return 0.0
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        max_font_size = max(0.0, self.layout.max_font_size(to_px=True))
        stroke_outset = 0.0
        active_strokes = self._stroke_sources_for_nodes(retained)
        if active_strokes:
            stroke_outset = max(
                max_font_size
                * (stroke.width + 0.05)
                * (
                    0.0
                    if stroke.position == 'inside'
                    else 0.5
                )
                for stroke in active_strokes
            )
        synthetic_outset = self._synthetic_bold_outset()
        padding = stroke_outset + synthetic_outset
        exterior_padding = None
        for effect in self._retained_phase_effects('exterior', retained):
            if isinstance(effect, ShadowEffect):
                blur, spread, xoffset, yoffset = self._shadow_metrics(effect)
                effect_padding = (
                    stroke_outset
                    + synthetic_outset
                    + (
                        0.0
                        if effect.shadow_type == 'long'
                        else blur + spread
                    )
                    + max(abs(xoffset), abs(yoffset))
                )
            else:
                size, spread = self._glow_metrics(effect)
                effect_padding = (
                    stroke_outset + synthetic_outset + size + spread
                )
            exterior_padding = (
                effect_padding
                if exterior_padding is None
                else max(exterior_padding, effect_padding)
            )
        if exterior_padding is not None:
            padding = max(
                padding, exterior_padding + EFFECT_RASTER_GUARD
            )
        return padding + self._filter_expansion_padding(retained)

    def _filter_expansion_padding(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> float:
        """Return conservative logical padding for declared alpha growers."""
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        return sum(self._filter_expansion_by_index(retained).values())

    def _filter_expansion_by_index(
        self,
        nodes: Tuple[Tuple[int, TextEffect], ...],
    ) -> Dict[int, float]:
        """Return each retained alpha grower's worst logical halo."""
        included_filters = self._retained_filter_indices(nodes)
        expansion: Dict[int, float] = {}
        # Interactive effect previews render at 0.5x. Account for physical
        # halo rounding there as well as in settled 1x+ rendering.
        for scale in (0.5, 1.0):
            for index, _effect, runtime, halo in self._filter_execution_plan(
                scale, included_filters=included_filters
            ):
                if getattr(runtime.spec, 'expands_alpha', False):
                    expansion[index] = max(
                        expansion.get(index, 0.0), halo / scale
                    )
        return expansion

    def _commit_effect_padding(
        self,
        padding: float,
    ) -> bool:
        return (
            self.setPadding(padding)
            if self.padding() != padding
            else False
        )

    def _update_effect_padding(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ):
        if self.refreshing_effect_padding or self.layout is None:
            return False
        self.refreshing_effect_padding = True
        try:
            padding = self._effect_padding(nodes)
            # QTextLayout stores coordinates at 26.6 fixed-point precision.
            # Round outward so relayout and undo cycles converge.
            if padding > 0.0:
                layout_units = math.nextafter(padding * 64.0, -math.inf)
                padding = math.ceil(layout_units) / 64.0
            return self._commit_effect_padding(padding)
        finally:
            self.refreshing_effect_padding = False

    def _effect_flags(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> Tuple[bool, bool]:
        """Return active Stroke and generated completed-surface flags."""
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        strokes = self._retained_strokes(retained)
        exterior = self._retained_phase_effects('exterior', retained)
        interior = self._retained_phase_effects('interior', retained)
        images = any(
            isinstance(effect, ImageEffect)
            for _index, effect in retained
        )
        filters = any(
            isinstance(effect, FilterEffect)
            for _index, effect in retained
        )
        return (
            bool(strokes),
            bool(exterior)
            or bool(interior)
            or self._mask_requires_surface()
            or (
                not self._hollow_enabled()
                and bool(self._active_text_fills())
            )
            or images
            or filters
            or any(
                stroke.position != 'center'
                or stroke.blend_mode != 'normal'
                or isinstance(stroke.paint, LinearGradientPaint)
                for stroke in strokes
            ),
        )

    def _effect_tile_overlap(
        self,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
    ) -> float:
        retained = (
            self._ordered_surface_nodes(strict_assets=False)
            if nodes is None
            else nodes
        )
        stroke_reach = self._stroke_generation_reach(
            self._stroke_sources_for_nodes(retained)
        )
        overlap = stroke_reach + EFFECT_RASTER_GUARD
        for effect in (
            self._retained_phase_effects('exterior', retained)
            + self._retained_phase_effects('interior', retained)
        ):
            if isinstance(effect, ShadowEffect):
                blur, spread, xoffset, yoffset = self._shadow_metrics(effect)
                if effect.shadow_type == 'long':
                    reach = max(abs(xoffset), abs(yoffset))
                else:
                    reach = (
                        blur + spread + max(abs(xoffset), abs(yoffset))
                    )
                source_reach = (
                    stroke_reach
                    if effect.shadow_type != 'inner'
                    else 0.0
                )
            else:
                size, spread = self._glow_metrics(effect)
                reach = size + spread
                source_reach = (
                    stroke_reach if effect.glow_type == 'outer' else 0.0
                )
            overlap = max(
                overlap,
                reach + source_reach + EFFECT_RASTER_GUARD,
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
        self._verified_export_assets.clear()
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

    def _filter_failure(
        self,
        effect: FilterEffect,
        stage: str,
        error: Exception,
    ) -> None:
        failure = EffectRasterAllocationError(
            f'text filter {effect.filter_id} {stage} failed: {error}'
        )
        if self.export_render:
            self._raise_or_defer_export_effect_error(failure)
            return
        warning_key = effect.filter_id, effect.schema_version, stage
        if (
            warning_key in self._filter_warnings
            or len(self._filter_warnings) >= _FILTER_WARNING_LIMIT
        ):
            return
        self._filter_warnings.add(warning_key)
        LOGGER.warning(
            'Text filter %s bypassed for item %s during %s: %s',
            effect.filter_id,
            self.idx,
            stage,
            error,
        )

    def _filter_execution_plan(
        self,
        render_scale: float,
        skipped_filters: frozenset[int] = frozenset(),
        included_filters: Optional[frozenset[int]] = None,
    ) -> _FilterExecutionPlan:
        """Resolve active filters bottom-to-top and validate bounded halos."""
        resolved = []
        for index, effect in reversed(self._active_filters()):
            if index in skipped_filters or (
                included_filters is not None
                and index not in included_filters
            ):
                continue
            try:
                runtime = get_filter_registry().resolve(effect)
                raw_halo = runtime.tile_halo(runtime.params, render_scale)
                if (
                    isinstance(raw_halo, bool)
                    or not isinstance(raw_halo, (int, float))
                    or not math.isfinite(float(raw_halo))
                    or raw_halo < 0
                ):
                    raise ValueError('tile_halo must return a nonnegative number')
                halo = int(math.ceil(float(raw_halo)))
                if halo > _FILTER_HALO_MAX_PIXELS:
                    raise ValueError(
                        f'tile halo exceeds {_FILTER_HALO_MAX_PIXELS} pixels'
                    )
            except Exception as error:
                self._filter_failure(effect, 'resolution', error)
                continue
            resolved.append((index, effect, runtime, halo))
        return tuple(resolved)

    def _apply_filter_chain(
        self,
        source: QPixmap,
        surface_rect: QRectF,
        render_scale: float,
        skipped_filters: frozenset[int],
        filter_plan: Optional[_FilterExecutionPlan] = None,
    ) -> QPixmap:
        plan = (
            self._filter_execution_plan(render_scale, skipped_filters)
            if filter_plan is None
            else filter_plan
        )
        if not plan:
            return source
        try:
            rgba = pixmap2ndarray(source, keep_alpha=True)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                'unable to access pre-filter surface pixels'
            ) from error
        if rgba is None:
            raise EffectRasterAllocationError(
                'unable to access pre-filter surface pixels'
            )
        current = np.ascontiguousarray(rgba)
        logical = self.logical_unpadded_rect()
        context = FilterContext(
            render_scale=float(render_scale),
            origin_x=int(round(
                (surface_rect.left() - logical.left()) * render_scale
            )),
            origin_y=int(round(
                (surface_rect.top() - logical.top()) * render_scale
            )),
            strict_export=self.export_render,
        )
        adopted = False
        for _index, effect, runtime, halo in plan:
            candidate = current.copy()
            alpha_before = current[:, :, 3]
            try:
                result = runtime.apply(candidate, runtime.params, context)
                if (
                    not isinstance(result, np.ndarray)
                    or result.shape != current.shape
                    or result.dtype != np.uint8
                    or not result.flags.c_contiguous
                ):
                    raise ValueError(
                        'apply must return contiguous same-shaped RGBA8'
                    )
                expanded = result[:, :, 3] > alpha_before
                if np.any(expanded):
                    if not getattr(runtime.spec, 'expands_alpha', False):
                        raise ValueError('filter expanded the source alpha')
                    source_support = (alpha_before > 0).astype(np.uint8)
                    if halo > 0:
                        allowed_support = cv2.dilate(
                            source_support,
                            np.ones(
                                (halo * 2 + 1, halo * 2 + 1),
                                dtype=np.uint8,
                            ),
                            borderType=cv2.BORDER_CONSTANT,
                        )
                    else:
                        allowed_support = source_support
                    if np.any(expanded & (allowed_support == 0)):
                        raise ValueError(
                            'filter expanded alpha beyond its tile halo'
                        )
            except Exception as error:
                self._filter_failure(effect, 'apply', error)
                continue
            current = result
            adopted = True
        if not adopted:
            return source
        try:
            filtered = ndarray2pixmap(current)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                'unable to allocate filtered surface'
            ) from error
        if filtered is None or filtered.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate filtered surface'
            )
        if render_scale >= 1.0:
            filtered.setDevicePixelRatio(render_scale)
        return filtered

    def _render_effect_surface(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        target_stroke: bool = True,
        skipped_filters: frozenset[int] = frozenset(),
        filter_plan: Optional[_FilterExecutionPlan] = None,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> QPixmap:
        """Render or reuse the ordered stack, then apply the final block mask.

        >>> hasattr(TextEffectRenderer, '_render_effect_surface')
        True
        """
        state = self._raster_state()
        pre_mask_key = self._pre_mask_cache_key(
            surface_rect, render_scale, target_stroke, skipped_filters
        )
        target_map = state.pre_mask_cache.get(pre_mask_key)
        if target_map is None:
            target_map = self._render_pre_mask_effect_surface(
                surface_rect,
                render_scale,
                target_stroke=target_stroke,
                skipped_filters=skipped_filters,
                filter_plan=filter_plan,
                nodes=nodes,
                image_rasters=image_rasters,
            )
            state.pre_mask_cache[pre_mask_key] = target_map
            while len(state.pre_mask_cache) > 2:
                state.pre_mask_cache.pop(next(iter(state.pre_mask_cache)))

        alpha_mask = self._active_text_alpha_mask()
        if alpha_mask is None:
            return target_map

        masked_map = self._new_effect_pixmap(render_scale, surface_rect)
        painter = None
        try:
            painter = QPainter(masked_map)
            if not painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin masked effect painter'
                )
            self._prepare_effect_surface_painter(painter, render_scale)
            local_rect = QRectF(
                0.0, 0.0, surface_rect.width(), surface_rect.height()
            )
            self._draw_surface_pixmap(
                painter, local_rect, target_map, render_scale
            )
            painter.translate(-surface_rect.topLeft())
            self._apply_text_alpha_mask(
                painter,
                alpha_mask,
                surface_rect,
                render_scale,
            )
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render masked effect surface'
            ) from error
        finally:
            try:
                if painter is not None and painter.isActive():
                    painter.end()
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to finish masked effect painter'
                ) from error
        return masked_map

    def _render_pre_mask_effect_surface(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        target_stroke: bool = True,
        skipped_filters: frozenset[int] = frozenset(),
        filter_plan: Optional[_FilterExecutionPlan] = None,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> QPixmap:
        """Compose ordered generated-layer batches and Filter chains.

        >>> hasattr(TextEffectRenderer, '_render_pre_mask_effect_surface')
        True
        """
        image_rasters = {} if image_rasters is None else image_rasters
        if nodes is None:
            nodes = self._ordered_surface_nodes(
                target_stroke=target_stroke,
                image_rasters=image_rasters,
            )
        first_filter = next(
            (
                position
                for position, (_index, effect) in enumerate(nodes)
                if isinstance(effect, FilterEffect)
            ),
            None,
        )
        if first_filter is None:
            return self._render_pre_filter_effect_surface(
                surface_rect,
                render_scale,
                target_stroke=target_stroke,
                nodes=nodes,
                image_rasters=image_rasters,
            )
        state = self._raster_state()
        key = self._pre_filter_cache_key(
            surface_rect, render_scale, target_stroke, nodes
        )
        upstream = state.pre_filter_cache.get(key)
        if upstream is None:
            upstream = self._render_pre_filter_effect_surface(
                surface_rect,
                render_scale,
                target_stroke=target_stroke,
                nodes=nodes,
                image_rasters=image_rasters,
            )
            state.pre_filter_cache[key] = upstream
            while len(state.pre_filter_cache) > 2:
                state.pre_filter_cache.pop(next(iter(state.pre_filter_cache)))
        return self._compose_ordered_surface_nodes(
            upstream,
            nodes[first_filter:],
            surface_rect,
            render_scale,
            skipped_filters=skipped_filters,
            filter_plan=filter_plan,
            image_rasters=image_rasters,
        )

    def _render_pre_filter_effect_surface(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        target_stroke: bool = True,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> QPixmap:
        """Render the fixed base and layers below the bottom active Filter.

        >>> hasattr(TextEffectRenderer, '_render_pre_filter_effect_surface')
        True
        """
        if nodes is None:
            image_rasters = {} if image_rasters is None else image_rasters
            nodes = self._ordered_surface_nodes(
                target_stroke=target_stroke,
                image_rasters=image_rasters,
            )
        first_filter = next(
            (
                position
                for position, (_index, effect) in enumerate(nodes)
                if isinstance(effect, FilterEffect)
            ),
            len(nodes),
        )
        target = self._render_effect_base(surface_rect, render_scale)
        return self._composite_generated_layer_batch(
            target,
            nodes[:first_filter],
            surface_rect,
            render_scale,
            _source_is_fresh_base=True,
            image_rasters=image_rasters,
        )

    def _render_effect_base(
        self,
        surface_rect: QRectF,
        render_scale: float,
    ) -> QPixmap:
        """Render the structural canonical Text Fill base.

        >>> hasattr(TextEffectRenderer, '_render_effect_base')
        True
        """
        target = self._new_effect_pixmap(render_scale, surface_rect)
        hollow = self._hollow_enabled()
        canonical = None
        if not hollow:
            canonical, _canonical_alpha = self._cached_effect_source(
                surface_rect, render_scale, needs_alpha=False
            )

        painter = QPainter(target)
        if not painter.isActive():
            raise EffectRasterAllocationError(
                'unable to begin effect base painter'
            )
        previous_capture = self.capturing_surface
        previous_raster_error = self.surface_raster_error
        self.capturing_surface = True
        self.surface_raster_error = None
        try:
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            self._prepare_effect_surface_painter(painter, render_scale)
            painter.translate(-surface_rect.topLeft())
            if canonical is not None:
                text_fill_group = self._text_fill_group_pixmap(
                    canonical,
                    surface_rect,
                    render_scale,
                    self._active_text_fills(),
                )
                self._draw_surface_pixmap(
                    painter,
                    surface_rect,
                    canonical if text_fill_group is None else text_fill_group,
                    render_scale,
                )
            if self.surface_raster_error is not None:
                raise self.surface_raster_error
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render effect base surface'
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
                    'unable to finish effect base painter'
                ) from end_error
        return target

    def _compose_ordered_surface_nodes(
        self,
        source: QPixmap,
        nodes: Tuple[Tuple[int, TextEffect], ...],
        surface_rect: QRectF,
        render_scale: float,
        *,
        skipped_filters: frozenset[int],
        filter_plan: Optional[_FilterExecutionPlan],
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> QPixmap:
        """Alternate the minimum contiguous generated and Filter segments.

        >>> hasattr(TextEffectRenderer, '_compose_ordered_surface_nodes')
        True
        """
        included_filters = frozenset(
            index
            for index, effect in nodes
            if isinstance(effect, FilterEffect)
        )
        plan = (
            self._filter_execution_plan(
                render_scale, skipped_filters, included_filters
            )
            if filter_plan is None
            else filter_plan
        )
        plan_by_index = {entry[0]: entry for entry in plan}
        target = source
        position = 0
        while position < len(nodes):
            is_filter = isinstance(nodes[position][1], FilterEffect)
            end = position + 1
            while (
                end < len(nodes)
                and isinstance(nodes[end][1], FilterEffect) == is_filter
            ):
                end += 1
            segment = nodes[position:end]
            if is_filter:
                segment_plan = tuple(
                    plan_by_index[index]
                    for index, _effect in segment
                    if index in plan_by_index
                )
                target = self._apply_filter_chain(
                    target,
                    surface_rect,
                    render_scale,
                    skipped_filters,
                    segment_plan,
                )
            else:
                target = self._composite_generated_layer_batch(
                    target,
                    segment,
                    surface_rect,
                    render_scale,
                    image_rasters=image_rasters,
                )
            position = end
        return target

    def _composite_generated_layer_batch(
        self,
        source: QPixmap,
        nodes: Tuple[Tuple[int, TextEffect], ...],
        surface_rect: QRectF,
        render_scale: float,
        *,
        _source_is_fresh_base: bool = False,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> QPixmap:
        """Source-over one contiguous canonical generated-layer batch.

        >>> hasattr(TextEffectRenderer, '_composite_generated_layer_batch')
        True
        """
        if not nodes:
            return source
        generated_nodes = tuple(
            (index, effect)
            for index, effect in nodes
            if isinstance(effect, (StrokeEffect, ShadowEffect, GlowEffect))
        )
        needs_canonical_alpha = bool(generated_nodes)
        if generated_nodes:
            canonical, canonical_alpha = self._cached_effect_source(
                surface_rect,
                render_scale,
                needs_alpha=needs_canonical_alpha,
            )
        else:
            canonical = None
            canonical_alpha = None
        positioned_stroke_bands: Dict[StrokeEffect, QPixmap] = {}
        try:
            exterior_alphas = (
                self._ordered_exterior_source_alphas(
                    canonical,
                    canonical_alpha,
                    nodes,
                    surface_rect,
                    render_scale,
                    positioned_stroke_bands,
                )
                if canonical is not None and canonical_alpha is not None
                else {}
            )
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render ordered Stroke source silhouette'
            ) from error

        # The prefix owner just allocated this base and no cache observes it
        # yet. Upper batches may receive cached/filter output and must detach.
        target = source if _source_is_fresh_base else QPixmap(source)
        painter: Optional[QPainter] = None
        previous_capture = self.capturing_surface
        previous_raster_error = self.surface_raster_error
        self.capturing_surface = True
        self.surface_raster_error = None
        try:
            painter = self._begin_effect_layer_painter(
                target, surface_rect, render_scale
            )
            for index, effect in nodes:
                if isinstance(effect, ImageEffect):
                    rgba = self._image_raster(effect, image_rasters)
                    if rgba is not None:
                        self._paint_image_effect(
                            painter,
                            effect,
                            rgba,
                            surface_rect,
                            render_scale,
                        )
                    continue
                if isinstance(effect, StrokeEffect):
                    assert canonical is not None
                    layer = self._stroke_layer_pixmap(
                        effect,
                        surface_rect,
                        render_scale,
                        canonical_alpha,
                        positioned_stroke_bands,
                    )
                else:
                    assert canonical is not None
                    source_alpha = (
                        exterior_alphas[index]
                        if effect_phase(effect) == 'exterior'
                        else canonical_alpha
                    )
                    assert source_alpha is not None
                    layer = self._generated_effect_pixmap(
                        source_alpha,
                        effect,
                        surface_rect,
                        render_scale,
                        canonical_alpha,
                    )
                if effect.blend_mode in CUSTOM_BLEND_MODES:
                    painter.end()
                    target = self._custom_blend_surface_pixmaps(
                        target, layer, effect.blend_mode, render_scale
                    )
                    painter = self._begin_effect_layer_painter(
                        target, surface_rect, render_scale
                    )
                else:
                    painter.setCompositionMode(
                        _BLEND_COMPOSITION_MODES[effect.blend_mode]
                    )
                    self._draw_surface_pixmap(
                        painter, surface_rect, layer, render_scale
                    )
            if self.surface_raster_error is not None:
                raise self.surface_raster_error
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to composite generated text-effect layers'
            ) from error
        finally:
            end_error = None
            try:
                if painter is not None and painter.isActive():
                    painter.end()
            except RASTER_BOUNDARY_FAILURES as error:
                end_error = error
            self.capturing_surface = previous_capture
            self.surface_raster_error = previous_raster_error
            if end_error is not None:
                raise EffectRasterAllocationError(
                    'unable to finish generated-layer painter'
                ) from end_error
        return target

    def _stroke_layer_pixmap(
        self,
        stroke: StrokeEffect,
        surface_rect: QRectF,
        render_scale: float,
        canonical_alpha: Optional[np.ndarray],
        positioned_stroke_bands: Dict[StrokeEffect, QPixmap],
    ) -> QPixmap:
        """Return one Stroke layer without consulting sibling visual order.

        >>> hasattr(TextEffectRenderer, '_stroke_layer_pixmap')
        True
        """
        previous = self._render_stroke
        self._render_stroke = stroke
        try:
            band = positioned_stroke_bands.get(stroke)
            if band is None:
                band = self._positioned_stroke_band(
                    surface_rect,
                    render_scale,
                    stroke,
                    canonical_alpha,
                )
                positioned_stroke_bands[stroke] = band
            return band
        finally:
            self._render_stroke = previous

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
                if render_scale < 1.0:
                    painter.drawImage(
                        surface_rect,
                        alpha,
                        QRectF(alpha.rect()),
                    )
                else:
                    painter.drawImage(surface_rect.topLeft(), alpha)
            finally:
                painter.restore()
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to apply text alpha mask'
            ) from error

    @staticmethod
    def _positioned_stroke_coverage_cache_key(
        source_key: tuple,
        stroke: StrokeEffect,
    ) -> tuple:
        """Key Stroke geometry without its downstream paint or opacity.

        >>> first = StrokeEffect(width=0.2, opacity=0.25)
        >>> second = StrokeEffect(
        ...     width=0.2, opacity=0.75,
        ...     paint=LinearGradientPaint(angle=95.0)
        ... )
        >>> TextEffectRenderer._positioned_stroke_coverage_cache_key(
        ...     ('source',), first
        ... ) == TextEffectRenderer._positioned_stroke_coverage_cache_key(
        ...     ('source',), second
        ... )
        True
        """
        return source_key, float(stroke.width), stroke.position

    def _positioned_stroke_coverage(
        self,
        surface_rect: QRectF,
        render_scale: float,
        stroke: StrokeEffect,
        canonical_alpha: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return immutable native outline alpha clipped to its position.

        >>> hasattr(TextEffectRenderer, '_positioned_stroke_coverage')
        True
        """
        state = self._raster_state()
        key = self._positioned_stroke_coverage_cache_key(
            self._effect_source_cache_key(surface_rect, render_scale),
            stroke,
        )
        cached = state.positioned_stroke_coverage_cache.get(key)
        if cached is not None:
            return cached

        previous = self._render_stroke
        previous_outline_only = self._outline_only_stroke
        self._render_stroke = stroke
        self._outline_only_stroke = True
        try:
            layer = self._new_effect_pixmap(render_scale, surface_rect)
            layer_painter = QPainter(layer)
            if not layer_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin positioned Stroke painter'
                )
            try:
                layer_painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
                self._prepare_effect_surface_painter(
                    layer_painter, render_scale
                )
                layer_painter.translate(-surface_rect.topLeft())
                layer_painter.save()
                try:
                    self._paint_stroke_core(
                        layer_painter, render_scale, surface_rect
                    )
                finally:
                    layer_painter.restore()
            finally:
                layer_painter.end()

            rgba = pixmap2ndarray(layer, keep_alpha=True)
            if rgba is None:
                raise EffectRasterAllocationError(
                    'unable to access positioned Stroke pixels'
                )
            # Alpha 1 keeps Qt from suppressing textOutline. It is a capture
            # sentinel, not visible foreground in the persistent band.
            alpha = rgba[..., 3]
            alpha[alpha <= 1] = 0
            # Stroke geometry is already rasterized. Expanding its coverage
            # here replaces the quadratic grid of translated pixmap draws.
            alpha = self._dilate_synthetic_bold_alpha(alpha, render_scale)
            if stroke.position != 'center':
                if canonical_alpha is None:
                    raise EffectRasterAllocationError(
                        'positioned Stroke requires canonical glyph alpha'
                    )
                coverage = (
                    canonical_alpha
                    if stroke.position == 'inside'
                    else 255 - canonical_alpha
                )
                product = alpha.astype(np.uint16)
                product *= coverage.astype(np.uint16)
                product += 127
                product //= 255
                alpha = product.astype(np.uint8)
            alpha = np.ascontiguousarray(alpha)
            alpha.setflags(write=False)
            state.positioned_stroke_coverage_cache[key] = alpha
            while len(state.positioned_stroke_coverage_cache) > 2:
                state.positioned_stroke_coverage_cache.pop(
                    next(iter(state.positioned_stroke_coverage_cache))
                )
            return alpha
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render positioned Stroke coverage'
            ) from error
        finally:
            self._outline_only_stroke = previous_outline_only
            self._render_stroke = previous

    def _positioned_stroke_band(
        self,
        surface_rect: QRectF,
        render_scale: float,
        stroke: StrokeEffect,
        canonical_alpha: Optional[np.ndarray],
    ) -> QPixmap:
        """Apply one Stroke's paint and opacity to geometric coverage.

        >>> hasattr(TextEffectRenderer, '_positioned_stroke_band')
        True
        """
        try:
            alpha = self._positioned_stroke_coverage(
                surface_rect,
                render_scale,
                stroke,
                canonical_alpha,
            )
            if stroke.position == 'center' and not self._hollow_enabled():
                if canonical_alpha is None:
                    raise EffectRasterAllocationError(
                        'Center Stroke requires canonical glyph alpha'
                    )
                product = alpha.astype(np.uint16)
                product *= (255 - canonical_alpha).astype(np.uint16)
                product += 127
                product //= 255
                alpha = product.astype(np.uint8)
            if stroke.opacity != 1.0:
                product = alpha.astype(np.uint16)
                product *= int(round(stroke.opacity * 255))
                product += 127
                product //= 255
                alpha = product.astype(np.uint8)
            rgba = np.empty(alpha.shape + (4,), dtype=np.uint8)
            rgba[..., 3] = alpha
            colorize_effect_paint_rgba(
                stroke.paint,
                rgba,
                surface_rect,
                self.logical_unpadded_rect(),
                render_scale,
            )
            band = ndarray2pixmap(rgba)
            if band is None or band.isNull():
                raise EffectRasterAllocationError(
                    'unable to allocate positioned Stroke band'
                )
            if render_scale >= 1.0:
                band.setDevicePixelRatio(render_scale)
            return band
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render positioned Stroke band'
            ) from error

    def _paint_positioned_strokes(
        self,
        painter: QPainter,
        surface_rect: QRectF,
        render_scale: float,
        canonical_alpha: Optional[np.ndarray],
        positions: Tuple[str, ...],
        positioned_stroke_bands: Optional[
            Dict[StrokeEffect, QPixmap]
        ] = None,
        strokes: Optional[Tuple[StrokeEffect, ...]] = None,
    ) -> None:
        """Paint selected Stroke positions back-to-front.

        >>> hasattr(TextEffectRenderer, '_paint_positioned_strokes')
        True
        """
        previous = self._render_stroke
        try:
            paint_order = (
                tuple(reversed(self._active_strokes()))
                if strokes is None
                else strokes
            )
            for stroke in paint_order:
                if stroke.position not in positions:
                    continue
                self._render_stroke = stroke
                band = (
                    None
                    if positioned_stroke_bands is None
                    else positioned_stroke_bands.get(stroke)
                )
                if band is None:
                    band = self._positioned_stroke_band(
                        surface_rect,
                        render_scale,
                        stroke,
                        canonical_alpha,
                    )
                    if positioned_stroke_bands is not None:
                        positioned_stroke_bands[stroke] = band
                self._draw_surface_pixmap(
                    painter,
                    surface_rect,
                    band,
                    render_scale,
                )
        finally:
            self._render_stroke = previous

    def _paint_stroke_silhouette(
        self,
        silhouette: QPixmap,
        canonical_alpha: np.ndarray,
        strokes: Tuple[StrokeEffect, ...],
        surface_rect: QRectF,
        render_scale: float,
        positioned_stroke_bands: Dict[StrokeEffect, QPixmap],
    ) -> None:
        """Extend a canonical silhouette with Strokes in application order."""
        if not strokes:
            return
        painter = QPainter(silhouette)
        if not painter.isActive():
            raise EffectRasterAllocationError(
                'unable to begin Stroke silhouette painter'
            )
        try:
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            self._prepare_effect_surface_painter(painter, render_scale)
            painter.translate(-surface_rect.topLeft())
            self._paint_positioned_strokes(
                painter,
                surface_rect,
                render_scale,
                canonical_alpha,
                ('center', 'inside', 'outside'),
                positioned_stroke_bands,
                strokes,
            )
        finally:
            painter.end()

    def _capture_effect_source(
        self,
        surface_rect: QRectF,
        render_scale: float,
    ) -> QPixmap:
        """Capture the canonical glyph pixels once for compiled phases.

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
            self._prepare_effect_surface_painter(painter, render_scale)
            painter.translate(-surface_rect.topLeft())
            x_ratio, y_ratio = self._synthetic_bold_ratios()
            anisotropic = (
                (x_ratio > 0.0 or y_ratio > 0.0)
                and (
                    self.fontformat.synthetic_bold == 'rect'
                    or not math.isclose(x_ratio, y_ratio, abs_tol=1e-12)
                )
            )
            if self.fontformat.synthetic_bold == 'rect' and anisotropic:
                canonical = self._capture_rectangular_bold_source(
                    surface_rect, render_scale
                )
                self._draw_surface_pixmap(
                    painter, surface_rect, canonical, render_scale
                )
            elif anisotropic:
                canonical = self._capture_plain_effect_source(
                    surface_rect, render_scale
                )
                for x_offset, y_offset in (
                    self._anisotropic_synthetic_bold_offsets()
                ):
                    painter.save()
                    try:
                        painter.translate(x_offset, y_offset)
                        self._draw_surface_pixmap(
                            painter,
                            surface_rect,
                            canonical,
                            render_scale,
                        )
                    finally:
                        painter.restore()
            else:
                self._paint_synthetic_bold(painter)
                self._paint_live_layout(
                    painter, self._effect_paint_context()
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

    def _capture_rectangular_bold_source(
        self, surface_rect: QRectF, render_scale: float
    ) -> QPixmap:
        """Expand rectangular ink in two separable passes.

        >>> callable(TextEffectRenderer._capture_rectangular_bold_source)
        True
        """
        source = self._capture_plain_effect_source(surface_rect, render_scale)
        for axis, radius in enumerate(self._synthetic_bold_outsets()):
            if radius <= 0.0:
                continue
            expanded = self._new_effect_pixmap(render_scale, surface_rect)
            painter = QPainter(expanded)
            if not painter.isActive():
                raise EffectRasterAllocationError('unable to expand bold source')
            try:
                self._prepare_effect_surface_painter(painter, render_scale)
                painter.translate(-surface_rect.topLeft())
                steps = max(1, math.ceil(radius))
                for step in range(-steps, steps + 1):
                    offset = radius * step / steps
                    target = surface_rect.translated(
                        offset if axis == 0 else 0.0,
                        offset if axis == 1 else 0.0,
                    )
                    self._draw_surface_pixmap(painter, target, source, render_scale)
            finally:
                painter.end()
            source = expanded
        return source

    def _capture_plain_effect_source(
        self,
        surface_rect: QRectF,
        render_scale: float,
    ) -> QPixmap:
        """Capture one glyph layout pass for cheap anisotropic replication."""
        source = self._new_effect_pixmap(render_scale, surface_rect)
        painter = QPainter(source)
        if not painter.isActive():
            raise EffectRasterAllocationError(
                'unable to begin plain effect source painter'
            )
        try:
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            self._prepare_effect_surface_painter(painter, render_scale)
            painter.translate(-surface_rect.topLeft())
            self._paint_live_layout(painter, self._effect_paint_context())
        finally:
            painter.end()
        return source

    def capture_plain_logical_rgba(
        self,
        width: int,
        height: int,
        offset_x: float,
        offset_y: float,
    ) -> np.ndarray:
        """Render the untransformed document foreground without effects.

        ``offset`` places the logical rectangle inside an integer page crop.
        The active Glyph Slant delegate and transient effect foreground state
        are restored even when Qt painting fails.

        >>> callable(TextEffectRenderer.capture_plain_logical_rgba)
        True
        """
        if width <= 0 or height <= 0:
            raise ValueError('plain text capture requires a positive size')
        image = QImage(
            width, height, QImage.Format.Format_ARGB32_Premultiplied
        )
        if image.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate plain text capture'
            )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        if not painter.isActive():
            raise EffectRasterAllocationError(
                'unable to begin plain text capture'
            )
        logical = self.logical_unpadded_rect()
        layout = self.item.layout
        previous_delegate = layout.render_delegate
        previous_stroke = self._render_stroke
        previous_outline = self._outline_only_stroke
        previous_alignment = self._native_stroke_alignment
        previous_deferred_cursor = layout.deferred_cursor_position
        try:
            layout.render_delegate = None
            self._render_stroke = None
            self._outline_only_stroke = False
            self._native_stroke_alignment = False
            painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
            painter.translate(
                float(offset_x) - logical.x(),
                float(offset_y) - logical.y(),
            )
            self._paint_live_layout(painter, self._effect_paint_context())
        finally:
            layout.render_delegate = previous_delegate
            self._render_stroke = previous_stroke
            self._outline_only_stroke = previous_outline
            self._native_stroke_alignment = previous_alignment
            layout.deferred_cursor_position = previous_deferred_cursor
            painter.end()
        rgba = pixmap2ndarray(image, keep_alpha=True)
        if rgba is None:
            raise EffectRasterAllocationError(
                'unable to read plain text capture'
            )
        return rgba

    def _cached_effect_source(
        self,
        surface_rect: QRectF,
        render_scale: float,
        *,
        needs_alpha: bool,
    ) -> Tuple[QPixmap, Optional[np.ndarray]]:
        """Reuse paint-independent canonical glyph pixels and alpha.

        >>> hasattr(TextEffectRenderer, '_cached_effect_source')
        True
        """
        state = self._raster_state()
        key = self._effect_source_cache_key(surface_rect, render_scale)
        cached = state.effect_source_cache.get(key)
        if cached is None:
            canonical = self._capture_effect_source(
                surface_rect, render_scale
            )
            canonical_alpha = (
                self._pixmap_alpha(canonical) if needs_alpha else None
            )
            cached = (canonical, canonical_alpha)
            state.effect_source_cache[key] = cached
            while len(state.effect_source_cache) > 2:
                state.effect_source_cache.pop(
                    next(iter(state.effect_source_cache))
                )
        elif needs_alpha and cached[1] is None:
            cached = (cached[0], self._pixmap_alpha(cached[0]))
            state.effect_source_cache[key] = cached
        return cached

    def _project_raster(
        self,
        asset: RasterAssetRef,
        label: str,
        *,
        strict_export: bool = True,
    ) -> Optional[np.ndarray]:
        """Read one project-cached raster, verifying once per strict export."""
        scene = self.item.scene()
        project = None if scene is None else getattr(scene, 'imgtrans_proj', None)
        if project is None:
            if self.export_render and strict_export:
                if self._raise_or_defer_export_effect_error(
                    EffectRasterAllocationError(
                        f'strict export cannot resolve the {label}'
                    )
                ):
                    return None
            return None
        key = (getattr(project, 'load_identity', None), asset)
        strict = (
            self.export_render
            and strict_export
            and key not in self._verified_export_assets
        )
        try:
            image = project.load_raster_asset(
                asset, strict=strict, premultiplied=True
            )
            if image is not None and (
                not isinstance(image, np.ndarray)
                or image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] != 4
            ):
                raise ValueError('project raster cache did not return RGBA8')
            if image is None and self.export_render and strict_export:
                if self._raise_or_defer_export_effect_error(
                    EffectRasterAllocationError(
                        f'unable to decode Raster asset: {asset.path}'
                    )
                ):
                    return None
        except (OSError,) + RASTER_BOUNDARY_FAILURES as error:
            if self.export_render and strict_export:
                if self._raise_or_defer_export_effect_error(error):
                    return None
            LOGGER.warning('Unable to load %s: %s', label, error)
            image = None
        if strict and image is not None:
            self._verified_export_assets.add(key)
        return image

    def _image_raster(
        self,
        effect: ImageEffect,
        resolved: Optional[Dict[RasterAssetRef, Optional[np.ndarray]]],
        *,
        strict_export: bool = True,
    ) -> Optional[np.ndarray]:
        """Resolve an Image at most once within one surface composite."""
        assert effect.asset is not None
        if resolved is not None and effect.asset in resolved:
            return resolved[effect.asset]
        rgba = self._project_raster(
            effect.asset,
            'Image effect',
            strict_export=strict_export,
        )
        if resolved is not None:
            resolved[effect.asset] = rgba
        return rgba

    def _paint_image_effect(
        self,
        painter: QPainter,
        effect: ImageEffect,
        rgba: np.ndarray,
        surface_rect: QRectF,
        render_scale: float,
    ) -> None:
        """Map one RGBA8 asset with tile-stable bilinear coordinates."""
        assert effect.asset is not None
        logical_rect = self.logical_unpadded_rect()
        surface_width = max(
            1, math.ceil(surface_rect.width() * render_scale)
        )
        surface_height = max(
            1, math.ceil(surface_rect.height() * render_scale)
        )
        left = max(0, math.ceil(
            (logical_rect.left() - surface_rect.left()) * render_scale - 0.5
        ))
        top = max(0, math.ceil(
            (logical_rect.top() - surface_rect.top()) * render_scale - 0.5
        ))
        right = min(surface_width, math.ceil(
            (logical_rect.right() - surface_rect.left()) * render_scale - 0.5
        ))
        bottom = min(surface_height, math.ceil(
            (logical_rect.bottom() - surface_rect.top()) * render_scale - 0.5
        ))
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return
        mapped_rect = QRectF(
            surface_rect.left() + left / render_scale,
            surface_rect.top() + top / render_scale,
            width / render_scale,
            height / render_scale,
        )
        mapped = np.empty((height, width, 4), dtype=np.uint8)
        mapped[..., 3] = 255
        colorize_texture_paint_rgba(
            TexturePaint(effect.asset),
            mapped,
            rgba,
            mapped_rect,
            logical_rect,
            render_scale,
            texture_premultiplied=True,
        )
        image = QImage(
            mapped.data,
            width,
            height,
            mapped.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        if image.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate Image pixels'
            )
        if render_scale >= 1.0:
            image.setDevicePixelRatio(render_scale)
        painter.save()
        try:
            painter.setCompositionMode(
                {
                    'foreground': (
                        QPainter.CompositionMode.CompositionMode_SourceOver
                    ),
                    'background': (
                        QPainter.CompositionMode.CompositionMode_DestinationOver
                    ),
                }[effect.mode]
            )
            painter.setClipRect(logical_rect)
            if render_scale < 1.0:
                painter.drawImage(mapped_rect, image, QRectF(image.rect()))
            else:
                painter.drawImage(mapped_rect.topLeft(), image)
        finally:
            painter.restore()

    def _text_fill_group_pixmap(
        self,
        canonical: QPixmap,
        surface_rect: QRectF,
        render_scale: float,
        text_fills: Tuple[TextFillEffect, ...],
    ) -> Optional[QPixmap]:
        """Compose renderable Text Fills over a transparent face surface.

        >>> hasattr(TextEffectRenderer, '_text_fill_group_pixmap')
        True
        """
        if not text_fills:
            return None
        painter = None
        try:
            target = None
            rgba = None
            for text_fill in text_fills:
                texture = (
                    self._project_raster(
                        text_fill.paint.asset, 'Text Fill texture'
                    )
                    if (
                        isinstance(text_fill.paint, TexturePaint)
                        and text_fill.paint.asset is not None
                    )
                    else None
                )
                if (
                    isinstance(text_fill.paint, TexturePaint)
                    and texture is None
                ):
                    continue
                # Compose paint alpha first, then apply glyph coverage once so
                # repeated Fills cannot thicken antialiased face edges.
                if rgba is None:
                    rgba = np.empty(
                        (canonical.height(), canonical.width(), 4),
                        dtype=np.uint8,
                    )
                rgba[..., 3] = 255
                if isinstance(text_fill.paint, TexturePaint):
                    assert texture is not None
                    colorize_texture_paint_rgba(
                        text_fill.paint,
                        rgba,
                        texture,
                        surface_rect,
                        self.logical_unpadded_rect(),
                        render_scale,
                        texture_premultiplied=True,
                    )
                else:
                    colorize_effect_paint_rgba(
                        text_fill.paint,
                        rgba,
                        surface_rect,
                        self.logical_unpadded_rect(),
                        render_scale,
                    )
                if text_fill.opacity != 1.0:
                    product = rgba[..., 3].astype(np.uint16)
                    product *= int(round(text_fill.opacity * 255.0))
                    product += 127
                    product //= 255
                    rgba[..., 3] = product.astype(np.uint8)
                layer = ndarray2pixmap(rgba)
                if layer is None or layer.isNull():
                    raise EffectRasterAllocationError(
                        'unable to allocate Text Fill layer'
                    )
                if render_scale >= 1.0:
                    layer.setDevicePixelRatio(render_scale)
                if target is None:
                    # Every blend mode is source identity over transparency.
                    # Source-copy into an alpha-capable surface so the final
                    # canonical clip can still reduce an opaque first layer.
                    target = self._new_effect_pixmap(
                        render_scale, surface_rect
                    )
                    painter = self._begin_effect_layer_painter(
                        target, surface_rect, render_scale
                    )
                    painter.setCompositionMode(
                        QPainter.CompositionMode.CompositionMode_Source
                    )
                    self._draw_surface_pixmap(
                        painter, surface_rect, layer, render_scale
                    )
                    continue
                if text_fill.blend_mode in CUSTOM_BLEND_MODES:
                    painter.end()
                    target = self._custom_blend_surface_pixmaps(
                        target, layer, text_fill.blend_mode, render_scale
                    )
                    painter = self._begin_effect_layer_painter(
                        target, surface_rect, render_scale
                    )
                else:
                    painter.setCompositionMode(
                        _BLEND_COMPOSITION_MODES[text_fill.blend_mode]
                    )
                    self._draw_surface_pixmap(
                        painter, surface_rect, layer, render_scale
                    )
            if target is not None:
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationIn
                )
                self._draw_surface_pixmap(
                    painter, surface_rect, canonical, render_scale
                )
            return target
        except RASTER_BOUNDARY_FAILURES as error:
            if isinstance(error, EffectRasterAllocationError):
                raise
            raise EffectRasterAllocationError(
                'unable to render Text Fill'
            ) from error
        finally:
            if painter is not None and painter.isActive():
                painter.end()

    def _ordered_exterior_source_alphas(
        self,
        canonical: QPixmap,
        canonical_alpha: np.ndarray,
        nodes: Tuple[Tuple[int, TextEffect], ...],
        surface_rect: QRectF,
        render_scale: float,
        positioned_stroke_bands: Dict[StrokeEffect, QPixmap],
    ) -> Dict[int, np.ndarray]:
        """Map exterior nodes to canonical plus preceding Stroke alpha.

        The working silhouette grows monotonically through the card order, so
        each Stroke band is composited at most once per generated-layer batch.

        >>> hasattr(TextEffectRenderer, '_ordered_exterior_source_alphas')
        True
        """
        active_effects = self._bounded_text_effects(self.effective_text_effects()).effects
        ordered_strokes = tuple(
            (index, effect)
            for index, effect in reversed(tuple(enumerate(active_effects)))
            if isinstance(effect, StrokeEffect) and not effect.is_neutral()
        )
        sources: Dict[int, np.ndarray] = {}
        silhouette: Optional[QPixmap] = None
        painted_count = 0
        source_alpha = canonical_alpha
        for index, effect in nodes:
            if not isinstance(effect, (ShadowEffect, GlowEffect)) or (
                effect_phase(effect) != 'exterior'
            ):
                continue
            previous_count = painted_count
            while (
                painted_count < len(ordered_strokes)
                and ordered_strokes[painted_count][0] > index
            ):
                painted_count += 1
            stroke_slice = ordered_strokes[previous_count:painted_count]
            new_strokes = tuple(
                stroke for _stroke_index, stroke in stroke_slice
            )
            if new_strokes:
                if silhouette is None:
                    silhouette = QPixmap(canonical)
                self._paint_stroke_silhouette(
                    silhouette,
                    canonical_alpha,
                    new_strokes,
                    surface_rect,
                    render_scale,
                    positioned_stroke_bands,
                )
                source_alpha = self._pixmap_alpha(silhouette)
            sources[index] = source_alpha
        return sources

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
        surface_rect: QRectF,
        render_scale: float,
        canonical_alpha: Optional[np.ndarray] = None,
    ) -> QPixmap:
        """Render Shadow alpha while protecting only the canonical face.

        >>> hasattr(TextEffectRenderer, '_shadow_pixmap')
        True
        """
        blur, spread, xoffset, yoffset = self._shadow_metrics(shadow)
        if canonical_alpha is None:
            canonical_alpha = source_alpha
        try:
            alpha = render_shadow_alpha(
                source_alpha,
                shadow.shadow_type,
                shadow.opacity,
                (
                    xoffset * render_scale,
                    yoffset * render_scale,
                ),
                max(0, int(round(blur * render_scale))),
                max(0, int(round(spread * render_scale))),
                canonical_alpha,
            )
            rgba = np.empty(source_alpha.shape + (4,), dtype=np.uint8)
            rgba[..., 3] = alpha
            colorize_effect_paint_rgba(
                shadow.paint,
                rgba,
                surface_rect,
                self.logical_unpadded_rect(),
                render_scale,
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
        if render_scale >= 1.0:
            pixmap.setDevicePixelRatio(render_scale)
        return pixmap

    def _glow_pixmap(
        self,
        source_alpha: np.ndarray,
        glow: GlowEffect,
        surface_rect: QRectF,
        render_scale: float,
    ) -> QPixmap:
        """Render one typed Glow from the phase's shared source alpha.

        >>> hasattr(TextEffectRenderer, '_glow_pixmap')
        True
        """
        size, spread = self._glow_metrics(glow)
        try:
            alpha = render_glow_alpha(
                source_alpha,
                glow.glow_type,
                max(0, int(round(size * render_scale))),
                max(0, int(round(spread * render_scale))),
            )
            if glow.opacity != 1.0:
                product = alpha.astype(np.uint16)
                product *= int(round(glow.opacity * 255.0))
                product += 127
                product //= 255
                alpha = product.astype(np.uint8)
            rgba = np.empty(source_alpha.shape + (4,), dtype=np.uint8)
            rgba[..., 3] = alpha
            colorize_effect_paint_rgba(
                glow.paint,
                rgba,
                surface_rect,
                self.logical_unpadded_rect(),
                render_scale,
            )
            pixmap = ndarray2pixmap(rgba)
        except RASTER_BOUNDARY_FAILURES as error:
            raise EffectRasterAllocationError(
                f'unable to allocate typed Glow surface: {error}'
            ) from error
        if pixmap is None or pixmap.isNull():
            raise EffectRasterAllocationError(
                'unable to allocate typed Glow surface'
            )
        if render_scale >= 1.0:
            pixmap.setDevicePixelRatio(render_scale)
        return pixmap

    def _generated_effect_pixmap(
        self,
        source_alpha: np.ndarray,
        effect: TextEffect,
        surface_rect: QRectF,
        render_scale: float,
        canonical_alpha: Optional[np.ndarray],
    ) -> QPixmap:
        """Render one generated node from canonical geometry inputs.

        >>> hasattr(TextEffectRenderer, '_generated_effect_pixmap')
        True
        """
        if isinstance(effect, ShadowEffect):
            return self._shadow_pixmap(
                source_alpha,
                effect,
                surface_rect,
                render_scale,
                canonical_alpha,
            )
        if isinstance(effect, GlowEffect):
            return self._glow_pixmap(
                source_alpha, effect, surface_rect, render_scale
            )
        raise TypeError('generated effect must be Shadow or Glow')

    def repaint_background(
        self,
        render_scale: float = 1.0,
        *,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
        geometry_prepared: bool = False,
    ) -> None:
        if (
            self.repainting
            or (self.reshaping and not self._export_active)
            or (self.pre_editing and not self._export_active)
        ):
            # Avoid reshape/reentrant work. During IME, reuse the preedit-free
            # cache because PaintContext cannot exclude active preedit glyphs.
            return

        planned_here = nodes is None
        if image_rasters is None:
            image_rasters = {}
        retained = (
            self._ordered_surface_nodes(
                image_rasters=image_rasters,
                strict_assets=self.export_render,
            )
            if planned_here
            else nodes
        )
        if planned_here:
            self.item.refresh_cache_policy(retained)
        empty = self.document().isEmpty()

        # Immediate transitions already prepared this exact immutable plan.
        if not geometry_prepared:
            self.repainting = True
            try:
                self._sync_native_stroke_alignment(retained)
            finally:
                self.repainting = False
            self._update_effect_padding(retained)

        paint_stroke, paint_non_stroke = self._effect_flags(retained)
        if (
            not paint_non_stroke and not paint_stroke
            or (
                empty
                and not any(
                    isinstance(effect, (ImageEffect, FilterEffect))
                    for _index, effect in retained
                )
            )
        ):
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
                self._raster_request(render_scale),
            )
            if plan.mode == 'tiles':
                self.background_pixmap = None
                self.background_pixmap_scale = None
                self.direct_stroke = False
                # Visible tiles are intentionally deferred until QPainter's
                # exposed/clip rectangle is available.
                return
            render_kwargs = {
                'nodes': retained,
                'image_rasters': image_rasters,
            }
            try:
                target_map = self._render_effect_surface(
                    br, plan.tier, **render_kwargs
                )
            except EFFECT_RASTER_FAILURES as error:
                # A higher tier may fail despite satisfying the deterministic
                # caps. Retry the smallest full tier before degrading.
                retry = plan_effect_raster(br.width(), br.height(), 1.0)
                if plan.tier > 1.0 and retry.mode == 'full':
                    try:
                        target_map = self._render_effect_surface(
                            br, 1.0, **render_kwargs
                        )
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
                    self.direct_stroke = (
                        paint_stroke
                        and self._all_strokes_vector_compatible(
                            self._retained_strokes(retained)
                        )
                    )
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
        state.pre_mask_cache.clear()
        # Completed effect pixels contain the previous paint parameters.
        self.background_pixmap = None
        self.background_pixmap_scale = None

    def _mark_mask_cache_dirty(self) -> None:
        """Invalidate final alpha while retaining matching upstream pixels."""
        state = self._raster_state()
        state.cache_generation += 1
        state.cache_dirty = True
        state.cache_input_key = None
        state.tile_cache.clear()
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
        *,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
    ) -> None:
        br = self.boundingRect()
        visible = self._visible_effect_rect(painter, exposed_rect)
        if visible.isEmpty():
            return

        if image_rasters is None:
            image_rasters = {}
        if nodes is None:
            nodes = self._ordered_surface_nodes(
                image_rasters=image_rasters,
                strict_assets=self.export_render,
            )
        retained_strokes = self._retained_strokes(nodes)
        paint_stroke, paint_non_stroke = self._effect_flags(nodes)
        stroke_overlap = (
            self._stroke_generation_reach(retained_strokes)
            + EFFECT_RASTER_GUARD
        )
        vector_stroke_direct = (
            paint_stroke
            and not paint_non_stroke
            and self._all_strokes_vector_compatible(retained_strokes)
            # The vector fallback cannot apply the block-wide alpha mask.
            and self._active_text_alpha_mask() is None
            and 2 * math.ceil(stroke_overlap * plan.tier)
            >= plan.tile_edge
        )
        target_overlap = (
            EFFECT_RASTER_GUARD
            if vector_stroke_direct
            else self._effect_tile_overlap(nodes)
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
        included_filters = self._retained_filter_indices(nodes)
        filter_plan = self._filter_execution_plan(
            plan.tier, included_filters=included_filters
        )
        skipped_filters = set(included_filters) - {
            index for index, _effect, _runtime, _halo in filter_plan
        }
        overlap_px = math.ceil(target_overlap * plan.tier)
        allocated_filter_plan = []
        for entry in filter_plan:
            index, effect, _runtime, halo = entry
            proposed_overlap = overlap_px + halo
            if plan.tile_edge - 2 * proposed_overlap < 1:
                self._filter_failure(
                    effect,
                    'tile halo',
                    EffectRasterAllocationError(
                        'cumulative filter halo leaves no tile core'
                    ),
                )
                skipped_filters.add(index)
                continue
            overlap_px = proposed_overlap
            allocated_filter_plan.append(entry)
        filter_plan = tuple(allocated_filter_plan)
        skipped_filters = frozenset(skipped_filters)
        surface_overlap = overlap_px / plan.tier
        core_edge_px = plan.tile_edge - 2 * overlap_px
        if core_edge_px < 1:
            error = EffectRasterAllocationError(
                'effect overlap exceeds bounded tile surface'
            )
            if self._raise_or_defer_export_effect_error(error):
                return
            self._warn_effect_allocation_once(error)
            self.direct_stroke = (
                paint_stroke
                and self._all_strokes_vector_compatible(retained_strokes)
            )
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
        cached_bytes = sum(
            pixmap.width() * pixmap.height() * 4
            for _rect, pixmap in self.tile_cache.values()
        )
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
                    staging_plan.mode == 'full'
                    and staging_plan.tier == plan.tier
                ):
                    staging_pixmap = self._new_effect_pixmap(
                        plan.tier, visible
                    )
                    staging_painter = QPainter(staging_pixmap)
                    if not staging_painter.isActive():
                        raise EffectRasterAllocationError(
                            'unable to begin visible effect staging painter'
                        )
                    self._prepare_effect_surface_painter(
                        staging_painter, plan.tier
                    )
                    staging_painter.translate(-visible.topLeft())
                    staging_painter.setCompositionMode(
                        QPainter.CompositionMode.CompositionMode_Source
                    )
                    tile_painter = staging_painter
                # Oversized views use per-tile clips without a full staging allocation.
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
                        tuple(sorted(skipped_filters)),
                    )
                    active_keys.add(key)
                    cached = self.tile_cache.get(key)
                    if cached is None:
                        pixmap = self._render_effect_surface(
                            surface,
                            plan.tier,
                            target_stroke=not vector_stroke_direct,
                            skipped_filters=skipped_filters,
                            filter_plan=filter_plan,
                            nodes=nodes,
                            image_rasters=image_rasters,
                        )
                        # Cache only the core and interpolation border; halos are temporary.
                        scale_x = plan.tier if plan.tier >= 1.0 else pixmap.width() / surface.width()
                        scale_y = plan.tier if plan.tier >= 1.0 else pixmap.height() / surface.height()
                        crop = QRectF(
                            (core.left() - surface.left()) * scale_x - 1,
                            (core.top() - surface.top()) * scale_y - 1,
                            core.width() * scale_x + 2,
                            core.height() * scale_y + 2,
                        ).toAlignedRect().intersected(pixmap.rect())
                        kept = pixmap.copy(crop)
                        if kept.isNull():
                            raise EffectRasterAllocationError('unable to retain effect tile core')
                        cached = (QRectF(
                            surface.left() + crop.x() / scale_x,
                            surface.top() + crop.y() / scale_y,
                            crop.width() / scale_x,
                            crop.height() / scale_y,
                        ), kept)
                        self.tile_cache[key] = cached
                        cached_bytes += kept.width() * kept.height() * 4
                        while cached_bytes > EFFECT_CACHE_MAX_BYTES and len(self.tile_cache) > 1:
                            oldest = next(iter(self.tile_cache))
                            if oldest == key and len(self.tile_cache) > 1:
                                oldest = next(
                                    candidate
                                    for candidate in self.tile_cache
                                    if candidate != key
                                )
                            _old_rect, old_pixmap = self.tile_cache.pop(oldest)
                            cached_bytes -= old_pixmap.width() * old_pixmap.height() * 4
                    tile_painter.save()
                    try:
                        tile_painter.setClipRect(
                            core, Qt.ClipOperation.IntersectClip
                        )
                        self._draw_surface_pixmap(
                            tile_painter, cached[0], cached[1], plan.tier
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
            self.direct_stroke = (
                paint_stroke
                and self._all_strokes_vector_compatible(retained_strokes)
            )
            self.cache_dirty = True
            self.cache_rendered_generation = -1
            if self._raise_or_defer_export_effect_error(raster_failure):
                return
            self._warn_effect_allocation_once(raster_failure)
            return

        if staging_pixmap is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self._draw_surface_pixmap(
                painter, visible, staging_pixmap, plan.tier
            )

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

    def _draw_direct_stroke(self, painter: QPainter) -> None:
        if (
            not self._effect_flags()[0]
            or not self._all_strokes_vector_compatible()
        ):
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
        self,
        painter: QPainter,
        exposed_rect: QRectF = None,
        *,
        nodes: Optional[Tuple[Tuple[int, TextEffect], ...]] = None,
        image_rasters: Optional[
            Dict[RasterAssetRef, Optional[np.ndarray]]
        ] = None,
        flags: Optional[Tuple[bool, bool]] = None,
    ) -> None:
        painter.save()
        try:
            if image_rasters is None:
                image_rasters = {}
            retained = (
                self._ordered_surface_nodes(
                    image_rasters=image_rasters,
                    strict_assets=self.export_render,
                )
                if nodes is None
                else nodes
            )
            paint_stroke, paint_non_stroke = (
                self._effect_flags(retained) if flags is None else flags
            )
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
                self._raster_request(requested_scale),
            )
            if self.force_tiles:
                plan = EffectRasterPlan(
                    'tiles', min(1.0, plan.tier), 0, 0,
                    EFFECT_TILE_MAX_EDGE,
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
                    self.repaint_background(
                        requested_scale,
                        nodes=retained,
                        image_rasters=image_rasters,
                    )
                if self.force_tiles:
                    tile_plan = EffectRasterPlan(
                        'tiles', min(1.0, plan.tier), 0, 0,
                        EFFECT_TILE_MAX_EDGE,
                    )
                    self._draw_tiled_effects(
                        painter,
                        tile_plan,
                        exposed_rect,
                        nodes=retained,
                        image_rasters=image_rasters,
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
                    self._draw_surface_pixmap(
                        painter, br, self.background_pixmap, plan.tier
                    )
                elif self.direct_stroke:
                    self._draw_direct_stroke(painter)
            else:
                # A previous ordinary-size fast cache must never be stretched
                # over a new huge local surface.
                self.background_pixmap = None
                self.background_pixmap_scale = None
                self._draw_tiled_effects(
                    painter,
                    plan,
                    exposed_rect,
                    nodes=retained,
                    image_rasters=image_rasters,
                )
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
