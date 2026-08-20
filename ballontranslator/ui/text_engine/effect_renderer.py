"""Stroke, shadow, gradient, and transformed-effect rendering."""

import math
from typing import Tuple

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
from qtpy.QtWidgets import QStyle, QWidget

from ballontranslator.utils.fontformat import FontFormat, pt2px
from ballontranslator.utils.logger import logger as LOGGER
from ..misc import ndarray2pixmap, pixmap2ndarray
from .horizontal_layout import HorizontalTextDocumentLayout
from .vertical_layout import VerticalTextDocumentLayout
from .rendering.glyph import (
    GLYPH_DILATED_STROKE_FORMAT_PROPERTY,
    GLYPH_STROKE_FORMAT_PROPERTY,
)
from .rendering.shadow import apply_shadow_effect
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
        self.export_render = False
        self.export_error = None
        self.in_graphics_paint = False
        self.capturing_surface = False
        self.surface_raster_error = None
        self.force_tiles = False
        self.direct_stroke = False


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
    export_render = _EffectRasterField()
    export_error = _EffectRasterField()
    in_graphics_paint = _EffectRasterField()
    capturing_surface = _EffectRasterField()
    surface_raster_error = _EffectRasterField()
    force_tiles = _EffectRasterField()
    direct_stroke = _EffectRasterField()

    def __init__(self, item) -> None:
        self.item = item
        self.background_pixmap = None
        self.background_pixmap_scale = None
        self._effect_raster_state = None
        self.refreshing_gradient_geometry = False
        self.refreshing_effect_padding = False
        self.has_transient_gradient_ranges = False

    def _raster_state(self) -> _EffectRasterState:
        state = self._effect_raster_state
        if state is None:
            state = _EffectRasterState()
            self._effect_raster_state = state
        return state

    def surface_cache_state(self) -> Tuple[int, bool]:
        """Return final-warp cache inputs without allocating effect state."""
        state = self._effect_raster_state
        if state is None:
            return 0, False
        return state.cache_generation, state.export_render

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
        return self.item.stroke_qcolor

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
        self.clear_cached_surface()
        state = self._effect_raster_state
        if state is not None:
            state.tile_cache.clear()
        self._effect_raster_state = None

    def paint_item(self, painter: QPainter, option, widget: QWidget, base_paint) -> None:
        """Paint effects around the host item's normal text pass."""
        if not any(self._effect_flags()):
            option.state = QStyle.State_None
            base_paint(painter, option, widget)
            return

        # Effects must be composited before the normal fill. DestinationOver
        # against an already opaque scene would discard them.
        was_in_graphics_paint = self.in_graphics_paint
        self.in_graphics_paint = True
        try:
            self._draw_effects(painter, option.exposedRect)
            option.state = QStyle.State_None
            base_paint(painter, option, widget)
        finally:
            self.in_graphics_paint = was_in_graphics_paint

    def finalize_neutral_cache(self) -> None:
        """Invalidate transformed pixels after neutral restoration."""
        self._refresh_gradient_geometry()
        state = self._effect_raster_state
        if state is not None:
            state.tile_cache.clear()
            state.force_tiles = False
            state.direct_stroke = False
            state.cache_dirty = True
            state.cache_rendered_generation = -1
        self.clear_cached_surface()
        self.item.update()
        if not any(self._effect_flags()):
            self._effect_raster_state = None

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
                    pt2px(point_size) * self.fontformat.stroke_width,
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
                effect_format.setForeground(self.stroke_qcolor)
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
        if self.fontformat.stroke_width <= 0:
            return 0.0
        return (
            self.layout.max_font_size(to_px=True)
            * self.fontformat.stroke_width
            / 2
        )

    def _sync_native_stroke_alignment(self) -> None:
        """Keep fill and stroke on Qt's same native glyph raster path."""
        if self.layout is None:
            return
        enabled = self.fontformat.stroke_width > 0
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
                    * self.fontformat.stroke_width
                )
                cursor.setPosition(fragment.position())
                cursor.setPosition(
                    fragment.position() + fragment.length(),
                    QTextCursor.MoveMode.KeepAnchor,
                )
                char_format.setTextOutline(stroke_pen)
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

    def _shadow_metrics(self):
        font_size = self.layout.max_font_size(to_px=True)
        radius = max(0.0, self.fontformat.shadow_radius * font_size)
        xoffset = self.fontformat.shadow_offset[0] * font_size
        yoffset = self.fontformat.shadow_offset[1] * font_size
        return radius, xoffset, yoffset

    def _logical_ink_bounds(self) -> QRectF:
        if self.document().isEmpty() or not self._has_layout_distortion():
            return QRectF()
        return self.geometry_controller.layout_ink_bounds()

    def _effect_padding(self) -> float:
        paint_stroke, paint_shadow = self._effect_flags()
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
        if paint_shadow:
            radius, xoffset, yoffset = self._shadow_metrics()
            shadow_bounds = effect_bounds.translated(xoffset, yoffset).adjusted(
                -radius, -radius, radius, radius
            )
            effect_bounds = effect_bounds.united(shadow_bounds)
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
        if self.fontformat.stroke_width > 0:
            stroke_outset = (
                max_font_size * (self.fontformat.stroke_width + 0.05) / 2
            )
        padding = stroke_outset
        if (
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0
        ):
            radius = self.fontformat.shadow_radius * max_font_size
            xoffset = abs(self.fontformat.shadow_offset[0] * max_font_size)
            yoffset = abs(self.fontformat.shadow_offset[1] * max_font_size)
            padding = max(
                padding,
                stroke_outset + radius + max(xoffset, yoffset),
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
        return (
            self.fontformat.stroke_width > 0,
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0,
        )

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
            self.export_error = None
            self.force_tiles = False
        else:
            self.force_tiles = False
        self.export_render = enabled

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
        shadow_rect: QRectF = None,
        shadow_scale: float = None,
        target_stroke: bool = True,
    ) -> QPixmap:
        """Render one bounded effect surface in item-local coordinates."""
        paint_stroke, paint_shadow = self._effect_flags()
        target_map = self._new_effect_pixmap(render_scale, surface_rect)

        if paint_shadow:
            shadow_rect = QRectF(surface_rect if shadow_rect is None else shadow_rect)
            shadow_scale = render_scale if shadow_scale is None else shadow_scale
            silhouette = self._new_effect_pixmap(shadow_scale, shadow_rect)
            try:
                silhouette_painter = QPainter(silhouette)
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to begin shadow silhouette painter'
                ) from error
            if not silhouette_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin shadow silhouette painter'
                )
            previous_capture = self.capturing_surface
            previous_raster_error = self.surface_raster_error
            self.capturing_surface = True
            self.surface_raster_error = None
            try:
                silhouette_painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
                silhouette_painter.translate(-shadow_rect.topLeft())
                self._paint_live_layout(
                    silhouette_painter, self._effect_paint_context()
                )
                if paint_stroke:
                    self.paint_stroke(
                        silhouette_painter, shadow_scale, shadow_rect
                    )
                if self.surface_raster_error is not None:
                    raise self.surface_raster_error
            finally:
                silhouette_painter.end()
                self.capturing_surface = previous_capture
                self.surface_raster_error = previous_raster_error

            radius, xoffset, yoffset = self._shadow_metrics()
            try:
                shadow_source = pixmap2ndarray(
                    silhouette, keep_alpha=True
                )
                if shadow_source is None:
                    raise EffectRasterAllocationError(
                        'unable to access shadow silhouette pixels'
                    )
                shadow_map, _ = apply_shadow_effect(
                    shadow_source,
                    self.fontformat.shadow_color,
                    self.fontformat.shadow_strength,
                    max(0, int(round(radius * shadow_scale))),
                )
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to allocate blurred shadow surface: '
                    f'{error}'
                ) from error
            if shadow_map is None or shadow_map.isNull():
                raise EffectRasterAllocationError(
                    'unable to allocate blurred shadow surface'
                )
            try:
                shadow_map.setDevicePixelRatio(shadow_scale)
                target_painter = QPainter(target_map)
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to begin effect target painter'
                ) from error
            if not target_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin effect target painter'
                )
            try:
                target_painter.setRenderHint(
                    QPainter.RenderHint.SmoothPixmapTransform
                )
                target_painter.drawPixmap(
                    shadow_rect.topLeft()
                    - surface_rect.topLeft()
                    + QPointF(xoffset, yoffset),
                    shadow_map,
                )
            finally:
                target_painter.end()

        if paint_stroke and target_stroke:
            try:
                stroke_painter = QPainter(target_map)
            except RASTER_BOUNDARY_FAILURES as error:
                raise EffectRasterAllocationError(
                    'unable to begin stroke target painter'
                ) from error
            if not stroke_painter.isActive():
                raise EffectRasterAllocationError(
                    'unable to begin stroke target painter'
                )
            previous_capture = self.capturing_surface
            previous_raster_error = self.surface_raster_error
            self.capturing_surface = True
            self.surface_raster_error = None
            try:
                stroke_painter.setRenderHints(_VECTOR_EFFECT_RENDER_HINTS)
                stroke_painter.translate(-surface_rect.topLeft())
                self.paint_stroke(
                    stroke_painter, render_scale, surface_rect
                )
                if self.surface_raster_error is not None:
                    raise self.surface_raster_error
            finally:
                stroke_painter.end()
                self.capturing_surface = previous_capture
                self.surface_raster_error = previous_raster_error
        return target_map

    def repaint_background(self, render_scale: float = 1.0):
        self.item.refresh_cache_policy()
        empty = self.document().isEmpty()
        if self.repainting or self.reshaping or self.pre_editing:
            # Avoid reshape/reentrant work. During IME, reuse the preedit-free
            # cache because PaintContext cannot exclude active preedit glyphs.
            return

        self.repainting = True
        try:
            self._sync_native_stroke_alignment()
        finally:
            self.repainting = False
        self._update_effect_padding()

        paint_stroke, paint_shadow = self._effect_flags()
        if not paint_shadow and not paint_stroke or empty:
            changed = self.background_pixmap is not None
            self.background_pixmap = None
            self.background_pixmap_scale = None
            state = self._effect_raster_state
            if state is not None:
                state.tile_cache.clear()
            self._effect_raster_state = None
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
        finally:
            self.repainting = False
        self.item.update()


    def _mark_effect_cache_dirty(self):
        self.cache_generation += 1
        self.cache_dirty = True
        self.tile_cache.clear()
        # Never combine a previous glyph silhouette with a new fill angle.
        self.background_pixmap = None
        self.background_pixmap_scale = None

    def _tile_shadow_scale(
        self, shadow_rect: QRectF, requested_scale: float
    ) -> float:
        """Bound a shadow-only context while preserving vector stroke tier."""
        width = max(shadow_rect.width(), 1.0)
        height = max(shadow_rect.height(), 1.0)
        scale = min(
            requested_scale,
            EFFECT_TILE_MAX_EDGE / width,
            EFFECT_TILE_MAX_EDGE / height,
            EFFECT_CACHE_MAX_DIMENSION / width,
            EFFECT_CACHE_MAX_DIMENSION / height,
            math.sqrt(EFFECT_CACHE_MAX_PIXELS / (width * height)),
            math.sqrt((EFFECT_CACHE_MAX_BYTES / 4) / (width * height)),
        )
        # QPixmap accepts a fractional DPR. The one-pixel floor keeps even an
        # extreme blur context representable without an unbounded allocation.
        return max(scale, 1.0 / max(width, height))

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
    ):
        br = self.boundingRect()
        visible = self._visible_effect_rect(painter, exposed_rect)
        if visible.isEmpty():
            return

        paint_stroke, paint_shadow = self._effect_flags()
        stroke_overlap = (
            self._stroke_outset() + EFFECT_RASTER_GUARD
            if paint_stroke
            else EFFECT_RASTER_GUARD
        )
        vector_stroke_direct = (
            paint_stroke
            and 2 * math.ceil(stroke_overlap * plan.tier)
            >= plan.tile_edge
        )
        target_overlap = (
            EFFECT_RASTER_GUARD
            if vector_stroke_direct
            else stroke_overlap
        )
        if vector_stroke_direct and not paint_shadow:
            self.tile_cache.clear()
            self.direct_stroke = True
            self.cache_dirty = False
            self.cache_rendered_generation = self.cache_generation
            self.force_tiles = False
            return
        overlap_px = math.ceil(target_overlap * plan.tier)
        core_edge_px = plan.tile_edge - 2 * overlap_px
        if core_edge_px < 1:
            error = EffectRasterAllocationError(
                'stroke overlap exceeds bounded tile surface'
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
                        -target_overlap,
                        -target_overlap,
                        target_overlap,
                        target_overlap,
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
                        shadow_rect = None
                        shadow_scale = None
                        if paint_shadow:
                            radius, xoffset, yoffset = self._shadow_metrics()
                            shadow_rect = (
                                core.translated(-xoffset, -yoffset)
                                .adjusted(
                                    -radius - stroke_overlap,
                                    -radius - stroke_overlap,
                                    radius + stroke_overlap,
                                    radius + stroke_overlap,
                                )
                                .intersected(br)
                            )
                            shadow_scale = self._tile_shadow_scale(
                                shadow_rect, plan.tier
                            )
                        pixmap = self._render_effect_surface(
                            surface,
                            plan.tier,
                            shadow_rect=shadow_rect,
                            shadow_scale=shadow_scale,
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
        except EFFECT_RASTER_FAILURES as error:
            self.tile_cache.clear()
            self.direct_stroke = paint_stroke
            if self._raise_or_defer_export_effect_error(error):
                return
            self._warn_effect_allocation_once(error)
            return
        finally:
            if staging_painter is not None and staging_painter.isActive():
                staging_painter.end()

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
        self.force_tiles = False

    def _draw_direct_stroke(self, painter: QPainter):
        if not self._effect_flags()[0]:
            return
        # This path intentionally avoids every intermediate raster allocation.
        # The custom glyph renderer still consumes outline selections, while a
        # native box transform keeps the unclipped cloned-document stroke.
        self._paint_source_local_stroke(painter)

    def _draw_effects(
        self, painter: QPainter, exposed_rect: QRectF = None
    ):
        painter.save()
        try:
            paint_stroke, paint_shadow = self._effect_flags()
            if not paint_stroke and not paint_shadow:
                return
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
                    not self.pre_editing
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
