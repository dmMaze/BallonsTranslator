import math, re
from contextlib import contextmanager
import cv2
import numpy as np
from typing import List, NamedTuple, Optional, Union, Tuple

from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QStyle, QGraphicsSceneMouseEvent
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, Signal, QSizeF
from qtpy.QtGui import (QGradient, QKeyEvent, QFont, QTextCursor, QImage, QPixmap, QPainterPath, QTextDocument,
                       QInputMethodEvent, QPainter, QPen, QColor, QTextCharFormat, QLinearGradient,
                       QBrush, QPalette, QAbstractTextDocumentLayout, QPolygonF, QTextLayout)

from ballontranslator.utils.textblock import TextBlock, FontFormat, TextAlignment, LineSpacingType
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.fontformat import (
    FontFormat,
    TextTransform,
    normalize_text_transform,
    px2pt,
    pt2px,
)
from .misc import td_pattern, table_pattern, pixmap2ndarray, ndarray2pixmap
from .scene_textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout, SceneTextLayout
from .text_glyph_renderer import (
    GLYPH_STROKE_FORMAT_PROPERTY,
    GlyphRasterAllocationError,
)
from .text_graphical_effect import apply_shadow_effect
from .text_transform import compensated_text_transform_matrix, rect_polygon

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)
# At the minimum cache DPR (1), one logical pixel contains the final
# antialias/kernel sample and one further pixel guarantees a transparent
# texture-sampling border. Higher DPR tiers receive at least the same two
# device rows. These are raster invariants, not transform-dependent padding.
EFFECT_ANTIALIAS_GUARD = 1.0
EFFECT_CLEAR_BORDER_GUARD = 1.0
EFFECT_RASTER_GUARD = EFFECT_ANTIALIAS_GUARD + EFFECT_CLEAR_BORDER_GUARD
EFFECT_CACHE_MAX_SCALE = 8.0
EFFECT_CACHE_MAX_PIXELS = 4_194_304
EFFECT_CACHE_MAX_DIMENSION = 8192
EFFECT_CACHE_MAX_BYTES = 32 * 1024 * 1024
EFFECT_TILE_MAX_EDGE = 2048
# QTextLayout additional formats are derived paint state, not document state.
# The marker lets us replace only our block-gradient override while preserving
# any unrelated highlighter ranges attached to the same live layout.
GRADIENT_LAYOUT_FORMAT_PROPERTY = 0x100000 + 1238


class EffectRasterPlan(NamedTuple):
    mode: str
    tier: float
    pixel_width: int
    pixel_height: int
    tile_edge: int


class EffectRasterAllocationError(RuntimeError):
    pass


EFFECT_RASTER_FAILURES = (
    EffectRasterAllocationError,
    GlyphRasterAllocationError,
    MemoryError,
    OverflowError,
    cv2.error,
)
RASTER_BRIDGE_FAILURES = (
    RuntimeError,
    ValueError,
    TypeError,
    BufferError,
)
RASTER_BOUNDARY_FAILURES = (
    EFFECT_RASTER_FAILURES + RASTER_BRIDGE_FAILURES
)


def plan_effect_raster(
    width: float,
    height: float,
    requested_scale: float,
) -> EffectRasterPlan:
    """Choose a bounded full-surface tier or visible-tile plan.

    >>> plan_effect_raster(100, 80, 99).tier
    8.0
    >>> plan_effect_raster(10000, 10000, 8).mode
    'tiles'
    """
    width = max(0.0, float(width))
    height = max(0.0, float(height))
    requested_scale = max(1.0, min(float(requested_scale), EFFECT_CACHE_MAX_SCALE))
    for tier in (8.0, 4.0, 2.0, 1.0):
        if tier > requested_scale:
            continue
        pixel_width = max(1, math.ceil(width * tier))
        pixel_height = max(1, math.ceil(height * tier))
        pixels = pixel_width * pixel_height
        if (
            pixel_width <= EFFECT_CACHE_MAX_DIMENSION
            and pixel_height <= EFFECT_CACHE_MAX_DIMENSION
            and pixels <= EFFECT_CACHE_MAX_PIXELS
            and pixels * 4 <= EFFECT_CACHE_MAX_BYTES
        ):
            return EffectRasterPlan(
                'full', tier, pixel_width, pixel_height, 0
            )
    tile_edge = min(
        EFFECT_TILE_MAX_EDGE,
        EFFECT_CACHE_MAX_DIMENSION,
        int(math.sqrt(EFFECT_CACHE_MAX_PIXELS)),
    )
    return EffectRasterPlan('tiles', 1.0, 0, 0, tile_edge)


class TextBlkItem(QGraphicsTextItem):

    begin_edit = Signal(int)
    end_edit = Signal(int)
    hover_enter = Signal(int)
    hover_move = Signal(int)
    moved = Signal()
    moving = Signal(QGraphicsTextItem)
    rotated = Signal(float)
    reshaped = Signal(QGraphicsTextItem)
    leftbutton_pressed = Signal(int)
    doc_size_changed = Signal(int)
    pasted = Signal(int)
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int, bool)
    propagate_user_edited = Signal(int, str, bool)

    def __init__(self, blk: TextBlock = None, idx: int = 0, set_format=True, show_rect=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._installing_text_transform = False
        self._text_transform_update_depth = 1
        self._text_transform_update_dirty = False
        self.pre_editing = False
        self.blk: TextBlock = None
        self.fontformat: FontFormat = None
        self._text_transform_preview: Optional[TextTransform] = None
        self._text_transform_entry_padding: Optional[float] = None
        self._effect_cache_generation = 0
        self._effect_cache_rendered_generation = -1
        self._effect_cache_dirty = False
        self._effect_tile_cache = {}
        self._effect_allocation_warning_generation = -1
        self._export_effect_render = False
        self._export_effect_error = None
        self._in_graphics_paint = False
        self._capturing_effect_surface = False
        self._effect_surface_raster_error = None
        self._force_effect_tiles = False
        self._effect_direct_stroke = False
        self._refreshing_gradient_geometry = False
        self.repainting = False
        self.reshaping = False
        self.under_ctrl = False
        self.draw_rect = show_rect
        self._display_rect: QRectF = QRectF(0, 0, 1, 1)
        self.old_ffmt_values = None
        
        self.idx = idx
        
        self.background_pixmap: QPixmap = None
        self._background_pixmap_scale = None
        self.stroke_qcolor = QColor(0, 0, 0)
        self.oldPos = QPointF()
        self.oldRect = QRectF()
        self.repaint_on_changed = True

        self.is_formatting = False
        self.old_undo_steps = 0
        self.in_redo_undo = False
        self.change_from: int = 0
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_text = ''
        self.block_change_signal = False

        self.layout: Union[VerticalTextDocumentLayout, HorizontalTextDocumentLayout] = None
        # Qt meta-properties can bypass Python setter overrides. Geometry
        # notifications keep rotation and origin changes on one code path.
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.document().setDocumentMargin(0)
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self._request_text_transform_update()
        self._text_transform_update_depth = 0
        self._flush_text_transform_update()

    def inputMethodEvent(self, e: QInputMethodEvent):
        if self.pre_editing == False:
            cursor = self.textCursor()
            self.input_method_from = cursor.selectionStart()
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            self.pre_editing = True
        super().inputMethodEvent(e)
        
    def is_editting(self):
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def on_content_changed(self):
        if (self.hasFocus() or self.is_formatting) and not self.pre_editing and not self.block_change_signal:   
            # self.content_changed.emit(self)
            if not self.in_redo_undo:
                undo_steps = self.document().availableUndoSteps()
                new_steps = undo_steps - self.old_undo_steps
                joint_previous = new_steps == 0

                if not self.is_formatting:
                    change_from = self.change_from
                    added_text = ''
                    if self.input_method_from != -1:
                        added_text = self.input_method_text
                        change_from = self.input_method_from
                        self.input_method_from = -1

                    elif self.change_added > 0:
                        cursor = self.textCursor()
                        cursor.setPosition(change_from)
                        cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor)
                        added_text = cursor.selectedText()

                    self.propagate_user_edited.emit(change_from, added_text, joint_previous)
                    self.change_added = 0

                if new_steps > 0:
                    self.old_undo_steps = undo_steps
                    self.push_undo_stack.emit(new_steps, self.is_formatting)

        if not (self.hasFocus() and self.pre_editing):
            if self.fontformat.gradient_enabled:
                self._refresh_gradient_geometry()
            if self.repaint_on_changed:
                if not self.repainting:
                    self.repaint_background()
            self.update()

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

    def _paint_neutral_stroke(self, painter: QPainter):
        """Paint stroke through the BASE cloned-document path."""
        doc = QTextDocument()
        doc.setUndoRedoEnabled(False)
        doc.setDocumentMargin(self.layout.documentMargin())
        doc.setDefaultFont(self.document().defaultFont())
        doc.setHtml(self.document().toHtml())
        doc.setDefaultTextOption(self.document().defaultTextOption())
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        stroke_pen = QPen(
            self.stroke_qcolor,
            0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        letter_spacing = self.fontformat.letter_spacing * 100
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
                if letter_spacing != 100 and not self.fontformat.vertical:
                    char_format.setFontLetterSpacingType(
                        QFont.SpacingType.PercentageSpacing
                    )
                    char_format.setFontLetterSpacing(letter_spacing)
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

    def _paint_legacy_vertical_stroke(self, painter: QPainter):
        """Use the upstream cloned-document stroke path when Glyph Slant is neutral.

        The live glyph-mask renderer below is required for non-zero Glyph
        Slant.  Keeping it out of the neutral path prevents this feature PR
        from also carrying the independent vertical rich-text stroke rewrite.
        """
        doc = QTextDocument()
        doc.setUndoRedoEnabled(False)
        doc.setDocumentMargin(self.layout.documentMargin())
        doc.setDefaultFont(self.document().defaultFont())
        doc.setHtml(self.document().toHtml())
        doc.setDefaultTextOption(self.document().defaultTextOption())
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
                cursor.mergeCharFormat(char_format)
                it += 1
            block = block.next()

        layout = VerticalTextDocumentLayout(doc, self.fontformat)
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
                source_painter.translate(-rect.topLeft())
                fragment_context = self._effect_paint_context()
                fragment_context.selections = selections
                self.layout.draw_glyph_selection_mask(
                    source_painter, fragment_context
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

    def paint_stroke(
        self,
        painter: QPainter,
        render_scale: float = 1.0,
        surface_rect: QRectF = None,
    ):
        if self._text_transform_is_neutral():
            self._paint_neutral_stroke(painter)
            return
        active_layout = self.document().documentLayout()
        if isinstance(active_layout, VerticalTextDocumentLayout):
            if self._effective_text_transform().glyph_slant_angle == 0.0:
                self._paint_legacy_vertical_stroke(painter)
                return
            self._paint_vertical_stroke(painter, render_scale, surface_rect)
            return
        self._paint_live_layout(painter, self._stroke_paint_context())

    def _shadow_metrics(self):
        font_size = self.layout.max_font_size(to_px=True)
        radius = max(0.0, self.fontformat.shadow_radius * font_size)
        xoffset = self.fontformat.shadow_offset[0] * font_size
        yoffset = self.fontformat.shadow_offset[1] * font_size
        return radius, xoffset, yoffset

    def _logical_ink_bounds(self) -> QRectF:
        logical_rect = self.logical_unpadded_rect()
        if self.document().isEmpty():
            return QRectF()

        # Non-zero glyph slant has an exact vector envelope derived from the
        # same live glyph runs and orientation transforms as paint.  Avoid the
        # legacy expanding scratch-image loop for this path.
        if self._effective_text_transform().glyph_slant_angle != 0.0:
            return self.layout.glyphInkBounds()

        # QTextLayout.boundingRect() includes bookkeeping lines and does not
        # include every custom vertical rotation/offset. Paint the attached
        # live layout into an expanding scratch envelope and measure its actual
        # alpha instead. No document, layout, or format is cloned. Expansion
        # stops only after all four raster borders are clear, so arbitrary
        # combining-mark overhang is measured rather than hidden by a fixed
        # padding guess. Align the logical origin to an integer pixel so the
        # result is independent of the current fractional document margin.
        font_guard = max(
            1,
            math.ceil(
                self.layout.max_font_size(to_px=True)
                + EFFECT_RASTER_GUARD
            ),
        )

        def bounded_vector_fallback(error: Exception) -> QRectF:
            self._warn_effect_allocation_once(error)
            bounds = self.layout.glyphInkBounds()
            return bounds.translated(-logical_rect.topLeft())

        while True:
            pixel_width = max(
                1, math.ceil(logical_rect.width()) + font_guard * 2
            )
            pixel_height = max(
                1, math.ceil(logical_rect.height()) + font_guard * 2
            )
            pixels = pixel_width * pixel_height
            if (
                pixel_width > EFFECT_CACHE_MAX_DIMENSION
                or pixel_height > EFFECT_CACHE_MAX_DIMENSION
                or pixels > EFFECT_CACHE_MAX_PIXELS
                or pixels * 4 > EFFECT_CACHE_MAX_BYTES
            ):
                return bounded_vector_fallback(
                    EffectRasterAllocationError(
                        'text ink measurement surface exceeds policy'
                    )
                )
            try:
                image = QImage(
                    pixel_width,
                    pixel_height,
                    QImage.Format.Format_ARGB32,
                )
            except RASTER_BOUNDARY_FAILURES as error:
                return bounded_vector_fallback(error)
            if image.isNull():
                return bounded_vector_fallback(
                    EffectRasterAllocationError(
                        'unable to allocate text ink measurement image'
                    )
                )
            try:
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
            except RASTER_BOUNDARY_FAILURES as error:
                return bounded_vector_fallback(error)
            if not painter.isActive():
                return bounded_vector_fallback(
                    EffectRasterAllocationError(
                        'unable to begin text ink measurement painter'
                    )
                )
            paint_error = None
            try:
                painter.translate(
                    font_guard - logical_rect.left(),
                    font_guard - logical_rect.top(),
                )
                self._paint_live_layout(painter, self._effect_paint_context())
            except EFFECT_RASTER_FAILURES as error:
                paint_error = error
            finally:
                painter.end()
            if paint_error is not None:
                return bounded_vector_fallback(paint_error)
            try:
                rgba = pixmap2ndarray(image, keep_alpha=True)
            except RASTER_BOUNDARY_FAILURES as error:
                return bounded_vector_fallback(error)
            if rgba is None:
                return bounded_vector_fallback(
                    EffectRasterAllocationError(
                        'unable to access text ink measurement pixels'
                    )
                )
            alpha = rgba[..., 3]
            ys, xs = np.nonzero(alpha)
            if len(xs) == 0:
                return QRectF()
            if (
                xs.min() == 0
                or ys.min() == 0
                or xs.max() == image.width() - 1
                or ys.max() == image.height() - 1
            ):
                font_guard *= 2
                continue
            return QRectF(
                xs.min() - font_guard,
                ys.min() - font_guard,
                xs.max() - xs.min() + 1,
                ys.max() - ys.min() + 1,
            )

    def _effect_padding(self) -> float:
        paint_stroke = self.fontformat.stroke_width > 0
        paint_shadow = (
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0
        )
        glyph_slanted = (
            self._effective_text_transform().glyph_slant_angle != 0.0
        )
        if not paint_stroke and not paint_shadow and not glyph_slanted:
            return 0.0
        ink_bounds = self._logical_ink_bounds()
        if ink_bounds.isEmpty():
            return 0.0
        stroke_outset = self._stroke_outset()
        logical_rect = (
            self.logical_unpadded_rect()
            if glyph_slanted
            else QRectF(QPointF(), self.logical_unpadded_rect().size())
        )
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

    def _neutral_effect_padding_floor(self) -> float:
        """Return the grow-only effect padding requested by the BASE path."""
        if self.layout is None:
            return 0.0
        max_font_size = self.layout.max_font_size(to_px=True)
        padding = 0.0
        if self.fontformat.shadow_radius > 0:
            padding = max(padding, max_font_size)
        if self.fontformat.stroke_width > 0:
            padding = max(
                padding,
                max_font_size * (self.fontformat.stroke_width + 0.05) / 2,
            )
        return padding

    def _commit_effect_padding(
        self,
        padding: float,
        *,
        allow_neutral_shrink: bool = False,
    ) -> bool:
        changed = (
            self.setPadding(
                padding,
                allow_neutral_shrink=allow_neutral_shrink,
            )
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
        if self._text_transform_entry_padding is not None:
            # Preserve the BASE grow-only high-water mark if an effect or font
            # change raises it while any text transform is active. The
            # transformed envelope may later shrink before neutral restore.
            self._text_transform_entry_padding = max(
                self._text_transform_entry_padding,
                self._neutral_effect_padding_floor(),
            )
        padding = self._effect_padding()
        # QTextLayout stores coordinates at 26.6 fixed-point precision. Use the
        # same grid as the canonical envelope and round outward so repeated
        # relayout/undo cycles converge without ever undersizing the effects.
        if padding > 0.0:
            layout_units = math.nextafter(padding * 64.0, -math.inf)
            padding = math.ceil(layout_units) / 64.0
        return self._commit_effect_padding(padding)

    def _finalize_neutral_text_transform(
        self,
        was_visual_neutral: bool,
        target: TextTransform,
    ) -> bool:
        neutral = TextTransform(1.0, 1.0, 0.0, 0.0)
        if was_visual_neutral or target != neutral:
            return False
        entry_padding = self._text_transform_entry_padding
        if entry_padding is None:
            # Loaded or externally merged active state may not have an
            # in-session neutral entry point.
            entry_padding = self._neutral_effect_padding_floor()
        padding = max(
            entry_padding,
            self._neutral_effect_padding_floor(),
        )
        self._text_transform_entry_padding = None
        self._commit_effect_padding(
            padding,
            allow_neutral_shrink=True,
        )

        # Padding can already equal the neutral target, so cleanup must be
        # driven by the transform transition rather than by a margin change.
        self._refresh_gradient_geometry()
        self._effect_tile_cache.clear()
        self._force_effect_tiles = False
        self._effect_direct_stroke = False
        self._effect_cache_dirty = False
        # The rebuilt BASE-neutral pixmap is not an active-transform effect
        # surface. Keep the active cache generation stale so a later Box-only
        # re-entry cannot reuse the neutral pixmap at the same raster tier.
        self._effect_cache_rendered_generation = -1
        if any(self._effect_flags()):
            self.repaint_background()
        else:
            self.background_pixmap = None
            self._background_pixmap_scale = None
        self.update()
        return True

    def _effect_flags(self) -> Tuple[bool, bool]:
        return (
            self.fontformat.stroke_width > 0,
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0,
        )

    def _warn_effect_allocation_once(self, error: Exception):
        if self._effect_allocation_warning_generation == self._effect_cache_generation:
            return
        self._effect_allocation_warning_generation = self._effect_cache_generation
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
        if self._capturing_effect_surface:
            self._effect_surface_raster_error = failure
        if effect_pass:
            self._effect_cache_dirty = True
        if self._capturing_effect_surface:
            return
        if self._export_effect_render:
            if self._in_graphics_paint:
                self._export_effect_error = failure
            else:
                raise failure from error

    def set_export_effect_render(self, enabled: bool):
        """Make effect allocation failures fatal during a render transaction."""
        enabled = bool(enabled)
        if enabled:
            self._export_effect_error = None
            self._force_effect_tiles = False
        else:
            self._force_effect_tiles = False
        self._export_effect_render = enabled

    @property
    def export_effect_error(self):
        return self._export_effect_error

    def _raise_or_defer_export_effect_error(self, error: Exception) -> bool:
        """Raise at a Python boundary or defer across Qt's paint callback.

        PyQt treats an exception escaping a virtual ``QGraphicsItem.paint``
        callback as fatal. Canvas checks the deferred error immediately after
        ``QGraphicsScene.render`` and raises before returning its image.
        """
        if not self._export_effect_render:
            return False
        failure = EffectRasterAllocationError(str(error))
        if self._in_graphics_paint:
            self._export_effect_error = failure
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
            previous_capture = self._capturing_effect_surface
            previous_raster_error = self._effect_surface_raster_error
            self._capturing_effect_surface = True
            self._effect_surface_raster_error = None
            try:
                silhouette_painter.translate(-shadow_rect.topLeft())
                self._paint_live_layout(
                    silhouette_painter, self._effect_paint_context()
                )
                if paint_stroke:
                    self.paint_stroke(
                        silhouette_painter, shadow_scale, shadow_rect
                    )
                if self._effect_surface_raster_error is not None:
                    raise self._effect_surface_raster_error
            finally:
                silhouette_painter.end()
                self._capturing_effect_surface = previous_capture
                self._effect_surface_raster_error = previous_raster_error

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
            previous_capture = self._capturing_effect_surface
            previous_raster_error = self._effect_surface_raster_error
            self._capturing_effect_surface = True
            self._effect_surface_raster_error = None
            try:
                stroke_painter.translate(-surface_rect.topLeft())
                self.paint_stroke(
                    stroke_painter, render_scale, surface_rect
                )
                if self._effect_surface_raster_error is not None:
                    raise self._effect_surface_raster_error
            finally:
                stroke_painter.end()
                self._capturing_effect_surface = previous_capture
                self._effect_surface_raster_error = previous_raster_error
        return target_map

    def _repaint_neutral_background(self):
        """Rebuild effects with the BASE pixmap and composition path."""
        empty = self.document().isEmpty()
        if self.repainting or self.reshaping:
            return

        paint_stroke = self.fontformat.stroke_width > 0
        paint_shadow = (
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0
        )
        if (not paint_shadow and not paint_stroke) or empty:
            self.background_pixmap = None
            self._background_pixmap_scale = None
            return

        self.repainting = True
        try:
            font_size = self.layout.max_font_size(to_px=True)
            target_map = QPixmap(self.boundingRect().size().toSize())
            target_map.fill(Qt.GlobalColor.transparent)
            painter = QPainter(target_map)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            if paint_stroke:
                self._paint_neutral_stroke(painter)
            else:
                self.document().drawContents(painter)

            if paint_shadow:
                radius = int(round(self.fontformat.shadow_radius * font_size))
                xoffset = int(self.fontformat.shadow_offset[0] * font_size)
                yoffset = int(self.fontformat.shadow_offset[1] * font_size)
                shadow_map, _ = apply_shadow_effect(
                    target_map,
                    self.fontformat.shadow_color,
                    self.fontformat.shadow_strength,
                    radius,
                )
                composition = painter.compositionMode()
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationOver
                )
                painter.drawPixmap(xoffset, yoffset, shadow_map)
                painter.setCompositionMode(composition)

            painter.end()
            self.background_pixmap = target_map
            self._background_pixmap_scale = 1.0
        finally:
            self.repainting = False

    def repaint_background(self, render_scale: float = 1.0):
        if self._text_transform_is_neutral():
            self._repaint_neutral_background()
            return
        empty = self.document().isEmpty()
        if self.repainting or self.reshaping or self.pre_editing:
            # Avoid reshape/reentrant work. During IME, reuse the preedit-free
            # cache because PaintContext cannot exclude active preedit glyphs.
            return

        self._update_effect_padding()

        paint_stroke, paint_shadow = self._effect_flags()
        if not paint_shadow and not paint_stroke or empty:
            self.background_pixmap = None
            self._background_pixmap_scale = None
            self._effect_tile_cache.clear()
            self._effect_direct_stroke = False
            self._force_effect_tiles = False
            self._effect_cache_dirty = False
            self._effect_cache_rendered_generation = self._effect_cache_generation
            return

        self._effect_tile_cache.clear()
        self.repainting = True
        try:
            br = self.boundingRect()
            plan = plan_effect_raster(
                br.width(), br.height(), render_scale
            )
            if plan.mode == 'tiles':
                self.background_pixmap = None
                self._background_pixmap_scale = None
                self._effect_direct_stroke = False
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
                    self._background_pixmap_scale = None
                    if self._export_effect_render:
                        # A policy-valid full allocation can still fail at
                        # runtime. Export gets one bounded visible-tile retry
                        # before the transaction is failed.
                        self._effect_direct_stroke = False
                        self._force_effect_tiles = True
                        return
                    self._effect_direct_stroke = paint_stroke
                    self._warn_effect_allocation_once(error)
                    return

            self.background_pixmap = target_map
            self._background_pixmap_scale = plan.tier
            self._effect_direct_stroke = False
            self._force_effect_tiles = False
            self._effect_cache_dirty = False
            self._effect_cache_rendered_generation = self._effect_cache_generation
        finally:
            self.repainting = False
        
    def docSizeChanged(self):
        self.setCenterTransform()
        self.doc_size_changed.emit(self.idx)

    def initTextBlock(self, blk: TextBlock = None, set_format=True):
        self.blk = blk
        self.fontformat = blk.fontformat
        if blk is None:
            xyxy = [0, 0, 0, 0]
            blk = TextBlock(xyxy)
            blk.lines = [xyxy]
            bx1, by1, bx2, by2 = xyxy
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        self._text_transform_preview = None

        self.setVertical(blk.vertical)
        self.setRect(blk.bounding_rect(), update_blk_rect=False)

        try:
            block_angle = self._validated_rotation_angle(blk.angle)
        except ValueError as error:
            try:
                LOGGER.warning(
                    f'Reset invalid TextBlock rotation to 0 during load: '
                    f'{error}'
                )
            except Exception:
                pass
            block_angle = 0.0
        blk.angle = block_angle
        
        if block_angle != 0:
            self.setRotation(block_angle)
        
        set_char_fmt = False
        if blk.translation:
            set_char_fmt = True

        font_fmt = blk.fontformat
        self._text_transform_entry_padding = None
        if set_format:
            self.set_fontformat(font_fmt, set_char_format=set_char_fmt, set_stroke_width=False, set_effect=False)

        if not blk.rich_text:
            if blk.translation:
                self.setPlainText(blk.translation)
        else:
            self.setHtml(blk.rich_text)
            self.setLetterSpacing(font_fmt.letter_spacing, repaint_background=False)
            cursor = self.textCursor()
            cursor.clearSelection()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            cfmt = cursor.charFormat()
            cursor.setCharFormat(cfmt)
            cursor.setBlockCharFormat(cfmt)
            self.setTextCursor(cursor)
        if self.fontformat.gradient_enabled:
            self.setGradientEnabled(True)
        self.setShadow(font_fmt, repaint=False)
        self.setStrokeWidth(font_fmt.stroke_width, repaint_background=False)
        if not self._text_transform_is_neutral():
            # Loaded active transforms have no in-session neutral entry point.
            # Seed the fallback after effects/text are initialized so later
            # style changes cannot erase the BASE neutral minimum.
            floor = self._neutral_effect_padding_floor()
            self._text_transform_entry_padding = max(
                self._text_transform_entry_padding or 0.0,
                floor,
            )
        self.setCenterTransform()
        self.repaint_background()

    def _canonical_text_transform(self) -> TextTransform:
        return normalize_text_transform(*self.blk.fontformat.text_transform)

    def _effective_text_transform(self) -> TextTransform:
        if self._text_transform_preview is not None:
            return self._text_transform_preview
        return self._canonical_text_transform()

    def _text_transform_is_neutral(self) -> bool:
        return self._effective_text_transform() == TextTransform(
            1.0, 1.0, 0.0, 0.0
        )

    def _visual_text_transform_is_neutral(self) -> bool:
        return (
            self.transform().isIdentity()
            and (self.layout is None or self.layout.glyph_slant_angle == 0.0)
        )

    @contextmanager
    def _text_transform_update(self):
        """Batch cache and input-method work across nested Qt changes."""
        self._text_transform_update_depth += 1
        try:
            yield
        finally:
            self._text_transform_update_depth -= 1
            if self._text_transform_update_depth == 0:
                self._flush_text_transform_update()

    def _request_text_transform_update(self) -> None:
        self._text_transform_update_dirty = True
        if self._text_transform_update_depth == 0:
            self._flush_text_transform_update()

    def _flush_text_transform_update(self) -> None:
        if not self._text_transform_update_dirty:
            return
        self._text_transform_update_dirty = False
        self.refresh_cache_policy()
        if self.is_editting():
            self.updateMicroFocus()

    def _compensated_text_transform(
        self,
        values: TextTransform,
        *,
        angle: Optional[float] = None,
        box_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ):
        """Build the derived Qt base transform for the current item state.

        ``transformations()`` must stay empty: arbitrary QGraphicsTransform
        entries add another non-commuting stage that this conjugation does not
        model.
        """
        if angle is None:
            angle = self.rotation()
        if box_pivot is None:
            box_pivot = self.logical_unpadded_rect().center()
        if rotation_pivot is None:
            rotation_pivot = self.transformOriginPoint()
        return compensated_text_transform_matrix(
            values.horizontal_scale,
            values.vertical_scale,
            values.slant_angle,
            box_pivot,
            angle,
            rotation_pivot,
        )

    def _install_compensated_text_transform(
        self,
        values: TextTransform,
        *,
        angle: Optional[float] = None,
        box_pivot: Optional[QPointF] = None,
        rotation_pivot: Optional[QPointF] = None,
    ) -> bool:
        """Install the derived Qt base transform without lifecycle effects."""
        matrix = self._compensated_text_transform(
            values,
            angle=angle,
            box_pivot=box_pivot,
            rotation_pivot=rotation_pivot,
        )
        if self.transform() == matrix:
            return False
        self._installing_text_transform = True
        try:
            self.setTransform(matrix, combine=False)
        finally:
            self._installing_text_transform = False
        return True

    @staticmethod
    def _finite_point(point: QPointF) -> bool:
        return math.isfinite(point.x()) and math.isfinite(point.y())

    @staticmethod
    def _validated_rotation_angle(angle) -> float:
        if isinstance(angle, bool):
            raise ValueError('rotation angle must be a finite number')
        try:
            angle = float(angle)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                'rotation angle must be a finite number'
            ) from error
        if not math.isfinite(angle):
            raise ValueError('rotation angle must be a finite number')
        return angle

    def _report_rejected_item_change(self, change, error) -> None:
        try:
            LOGGER.warning(
                f'Rejected unsafe TextBlkItem graphics change {change}: '
                f'{error}'
            )
        except Exception:
            # Logging must never turn a rejected Qt virtual callback into an
            # exception crossing the C++/Python boundary.
            pass

    def _item_change(self, change, value):
        if getattr(self, '_installing_text_transform', False):
            return super().itemChange(change, value)

        if change in (
            QGraphicsItem.GraphicsItemChange.ItemRotationChange,
            QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointChange,
        ):
            candidate = super().itemChange(change, value)
            try:
                if change == QGraphicsItem.GraphicsItemChange.ItemRotationChange:
                    angle = float(candidate)
                    if not math.isfinite(angle):
                        raise ValueError('rotation angle must be finite')
                    rotation_pivot = self.transformOriginPoint()
                else:
                    rotation_pivot = QPointF(candidate)
                    if not self._finite_point(rotation_pivot):
                        raise ValueError(
                            'transform origin coordinates must be finite'
                        )
                    angle = self.rotation()

                if self.blk is not None:
                    # Validate every derived input while Qt can still reject
                    # the property write by returning the current value.
                    self._compensated_text_transform(
                        self._effective_text_transform(),
                        angle=angle,
                        box_pivot=self.logical_unpadded_rect().center(),
                        rotation_pivot=rotation_pivot,
                    )
            except Exception as error:
                self._report_rejected_item_change(change, error)
                if change == QGraphicsItem.GraphicsItemChange.ItemRotationChange:
                    return self.rotation()
                return QPointF(self.transformOriginPoint())
            return candidate

        if change in (
            QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformOriginPointHasChanged,
        ) and self.blk is not None:
            # At HasChanged the Qt property already contains its final value.
            # Installing here makes nested transform notifications and the
            # later public rotationChanged signal observe one coherent map.
            with self._text_transform_update():
                self._install_compensated_text_transform(
                    self._effective_text_transform(),
                    angle=self.rotation(),
                    box_pivot=self.logical_unpadded_rect().center(),
                    rotation_pivot=self.transformOriginPoint(),
                )
                result = super().itemChange(change, value)
                self._request_text_transform_update()
            return result

        result = super().itemChange(change, value)
        if change in (
            QGraphicsItem.GraphicsItemChange.ItemScaleHasChanged,
            QGraphicsItem.GraphicsItemChange.ItemTransformHasChanged,
        ):
            self._request_text_transform_update()
        return result

    def itemChange(self, change, value):
        """Keep all exceptions inside Qt's C++ virtual-call boundary."""
        try:
            return self._item_change(change, value)
        except Exception as error:
            self._report_rejected_item_change(change, error)
            try:
                return super().itemChange(change, value)
            except Exception:
                return value

    def refresh_cache_policy(self) -> bool:
        """Apply the sole QGraphicsItem cache policy for live text items."""
        transform = self._effective_text_transform()
        active_box_transform = (
            transform.horizontal_scale != 1.0
            or transform.vertical_scale != 1.0
            or transform.slant_angle != 0.0
        )
        use_no_cache = (
            self.is_editting()
            or active_box_transform
        )
        cache_mode = (
            QGraphicsItem.CacheMode.NoCache
            if use_no_cache
            else QGraphicsItem.CacheMode.DeviceCoordinateCache
        )
        if self.cacheMode() == cache_mode:
            return False
        self.setCacheMode(cache_mode)
        return True

    def _mark_effect_cache_dirty(self):
        self._effect_cache_generation += 1
        self._effect_cache_dirty = True
        self._effect_tile_cache.clear()
        # Never combine a previous glyph silhouette with a new fill angle.
        self.background_pixmap = None
        self._background_pixmap_scale = None

    def _apply_text_transform(self, values: TextTransform) -> bool:
        with self._text_transform_update():
            changed = self._install_compensated_text_transform(values)
            if changed:
                self._request_text_transform_update()
        return changed

    def _apply_glyph_slant(self, angle: float) -> Tuple[bool, bool]:
        if self.layout is None:
            return False, False
        if not self.layout.setGlyphSlantAngle(angle):
            return False, False
        self._mark_effect_cache_dirty()
        padding_changed = self._update_effect_padding()
        self.refresh_cache_policy()
        self.update()
        return True, padding_changed

    def _reconcile_active_text_transform_state(
        self,
        was_visual_neutral: bool,
        target: TextTransform,
        glyph_changed: bool,
        glyph_padding_changed: bool,
    ) -> bool:
        neutral = TextTransform(1.0, 1.0, 0.0, 0.0)
        if not was_visual_neutral or target == neutral:
            return False
        # Box-only transforms do not change the glyph silhouette and can have
        # no raster effects to enter _draw_effects(). Reconcile geometry here
        # so retained BASE padding and the active gradient range cannot leak
        # across a neutral-to-active transition.
        padding_changed = glyph_padding_changed
        if not glyph_changed:
            padding_changed = self._update_effect_padding()
        if self.fontformat.gradient_enabled and not padding_changed:
            self._refresh_gradient_geometry()
        return padding_changed

    def set_text_transform(
        self,
        horizontal_scale: float = None,
        vertical_scale: float = None,
        slant_angle: float = None,
        glyph_slant_angle: float = None,
        *,
        preview: bool = False,
    ) -> bool:
        """Apply the canonical item-local transform, optionally as a preview.

        ``preview`` is transient item state; only a committed call writes the
        existing ``TextBlock.fontformat`` owner.
        """
        raw_canonical = self.blk.fontformat.text_transform
        canonical = normalize_text_transform(*raw_canonical)
        current = self._effective_text_transform()
        base = current if preview else canonical
        target = normalize_text_transform(
            base[0] if horizontal_scale is None else horizontal_scale,
            base[1] if vertical_scale is None else vertical_scale,
            base[2] if slant_angle is None else slant_angle,
            base[3] if glyph_slant_angle is None else glyph_slant_angle,
        )
        was_visual_neutral = self._visual_text_transform_is_neutral()
        if (
            was_visual_neutral
            and target != TextTransform(1.0, 1.0, 0.0, 0.0)
            and self._text_transform_entry_padding is None
        ):
            self._text_transform_entry_padding = self.padding()

        if preview:
            if target == current:
                return False
            self._text_transform_preview = None if target == canonical else target
            glyph_changed, glyph_padding_changed = self._apply_glyph_slant(
                target.glyph_slant_angle
            )
            active_state_changed = self._reconcile_active_text_transform_state(
                was_visual_neutral,
                target,
                glyph_changed,
                glyph_padding_changed,
            )
            box_changed = self._apply_text_transform(target)
            finalized = self._finalize_neutral_text_transform(
                was_visual_neutral,
                target,
            )
            return (
                glyph_changed
                or active_state_changed
                or box_changed
                or finalized
            )

        model_format = self.blk.fontformat
        render_format = self.fontformat
        model_changed = raw_canonical != target
        render_format_changed = (
            render_format is not None
            and render_format is not model_format
            and render_format.text_transform != target
        )
        if model_changed:
            (
                model_format.horizontal_scale,
                model_format.vertical_scale,
                model_format.slant_angle,
                model_format.glyph_slant_angle,
            ) = target
        if render_format_changed:
            # Selection changes can detach the render/UI format cache from the
            # canonical TextBlock owner. Keep its quartet coherent before a
            # neutral stroke/effect surface is rebuilt during Undo/Redo.
            (
                render_format.horizontal_scale,
                render_format.vertical_scale,
                render_format.slant_angle,
                render_format.glyph_slant_angle,
            ) = target
        self._text_transform_preview = None
        glyph_changed, glyph_padding_changed = self._apply_glyph_slant(
            target.glyph_slant_angle
        )
        active_state_changed = self._reconcile_active_text_transform_state(
            was_visual_neutral,
            target,
            glyph_changed,
            glyph_padding_changed,
        )
        visual_changed = self._apply_text_transform(target)
        finalized = self._finalize_neutral_text_transform(
            was_visual_neutral,
            target,
        )
        return (
            model_changed
            or render_format_changed
            or glyph_changed
            or active_state_changed
            or visual_changed
            or finalized
        )

    def clear_text_transform_preview(self) -> bool:
        if self._text_transform_preview is None:
            return False
        was_visual_neutral = self._visual_text_transform_is_neutral()
        self._text_transform_preview = None
        target = self._canonical_text_transform()
        if (
            was_visual_neutral
            and target != TextTransform(1.0, 1.0, 0.0, 0.0)
            and self._text_transform_entry_padding is None
        ):
            self._text_transform_entry_padding = self.padding()
        glyph_changed, glyph_padding_changed = self._apply_glyph_slant(
            target.glyph_slant_angle
        )
        active_state_changed = self._reconcile_active_text_transform_state(
            was_visual_neutral,
            target,
            glyph_changed,
            glyph_padding_changed,
        )
        box_changed = self._apply_text_transform(target)
        finalized = self._finalize_neutral_text_transform(
            was_visual_neutral,
            target,
        )
        return (
            glyph_changed
            or active_state_changed
            or box_changed
            or finalized
        )

    def setCenterTransform(self) -> bool:
        center = self.logical_unpadded_rect().center()
        with self._text_transform_update():
            origin_changed = self.transformOriginPoint() != center
            if origin_changed:
                self.setTransformOriginPoint(center)
            transform_changed = self._install_compensated_text_transform(
                self._effective_text_transform(),
                box_pivot=center,
                rotation_pivot=self.transformOriginPoint(),
            )
            if transform_changed:
                self._request_text_transform_update()
        return origin_changed or transform_changed

    def logical_unpadded_rect(self) -> QRectF:
        """Return the untransformed, effect-free block rect in item coordinates."""
        return self.unpadRect(self.boundingRect())

    def visual_polygon_in_scene(self) -> QPolygonF:
        """Return the exact transformed logical block polygon in scene space."""
        return QPolygonF(
            [self.mapToScene(point) for point in rect_polygon(self.logical_unpadded_rect())]
        )

    def visual_bounds_in_scene(self) -> QRectF:
        return self.visual_polygon_in_scene().boundingRect()

    def rect(self) -> QRectF:
        return QRectF(self.pos(), self.boundingRect().size())

    def logical_position(self) -> QPointF:
        """Return the persistent logical rectangle's absolute top-left."""
        return self.absBoundingRect(qrect=True).topLeft()

    def set_logical_position(self, point: QPointF) -> bool:
        """Move the logical top-left independently of paint padding."""
        point = QPointF(point)
        delta = point - self.logical_position()
        if delta.isNull():
            return False
        self.setPos(self.pos() + delta)
        self.blk._bounding_rect = self.absBoundingRect()
        return True

    def startReshape(self):
        self.oldRect = self.absBoundingRect(qrect=True)
        self.reshaping = True
        # disable background repainting to avoid heavy redrawing in the whole process
        self.background_pixmap = None
        self._background_pixmap_scale = None

    def endReshape(self):
        self.reshaped.emit(self)
        self.reshaping = False
        self.repaint_background()

    def padRect(self, rect: QRectF) -> QRectF:
        p = self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)
    
    def unpadRect(self, rect: QRectF) -> QRectF:
        p = -self.padding()
        P = p * 2
        return QRectF(rect.x() - p, rect.y() - p, rect.width() + P, rect.height() + P)

    def setRect(self, rect: Union[List, QRectF], padding=True, repaint=True, update_blk_rect=True) -> None:
        old_logical_rect = self.logical_unpadded_rect()
        if isinstance(rect, List):
            rect = QRectF(*rect)
        if padding:
            rect = self.padRect(rect)
        self.setPos(rect.topLeft())
        self.prepareGeometryChange()
        self._display_rect = rect
        self.layout.setMaxSize(rect.width(), rect.height())
        self.setCenterTransform()
        if (
            self.fontformat.gradient_enabled
            and not self.repainting
            and self.logical_unpadded_rect() != old_logical_rect
        ):
            self._refresh_gradient_geometry()
        if repaint:
            self.repaint_background()

        if update_blk_rect:
            self.blk._bounding_rect = self.absBoundingRect()

    def documentSize(self):
        return self.layout.documentSize()

    def boundingRect(self) -> QRectF:
        br = super().boundingRect()
        if self._display_rect is not None:
            br.setHeight(self._display_rect.height())
            br.setWidth(self._display_rect.width())
        return br

    def padding(self) -> float:
        if self.layout is None:
            return 0.0
        return self.layout.documentMargin()

    def setPadding(self, p: float, *, allow_neutral_shrink: bool = False):
        p = max(0.0, float(p))
        _p = self.padding()
        if self._text_transform_is_neutral() and not allow_neutral_shrink:
            if _p >= p:
                return False
            absolute_rect = self.absBoundingRect(qrect=True)
            self.layout.relayout_on_changed = False
            self.layout.updateDocumentMargin(p)
            self.layout.relayout_on_changed = True
            self.setRect(absolute_rect, repaint=False)
            return True
        if _p == p:
            return False
        abr = self.absBoundingRect(qrect=True)
        was_repainting = self.repainting
        self.repainting = True
        signals_were_blocked = self.layout.blockSignals(True)
        try:
            # The document margin participates in boundingRect(); notify the
            # scene before mutating it while preserving the absolute logical
            # rectangle captured above.
            self.prepareGeometryChange()
            self.layout.relayout_on_changed = False
            self.layout.updateDocumentMargin(p)
            self.layout.relayout_on_changed = True
            self.setRect(
                abr, repaint=False, update_blk_rect=False
            )
        finally:
            self.layout.relayout_on_changed = True
            self.layout.blockSignals(signals_were_blocked)
            self.repainting = was_repainting
        return True

    def absBoundingRect(self, max_h=None, max_w=None, qrect=False) -> Union[List, QRectF]:
        # This remains the logical, untransformed persistence/layout rectangle.
        br = self.logical_unpadded_rect()
        w, h = br.width(), br.height()
        pos = self.pos()
        x = pos.x() + br.x()
        y = pos.y() + br.y()
        if max_h is not None:
            y = min(max(0, y), max_h)
            y1 = y + h
            h = min(max_h, y1) - y
        if max_w is not None:
            x = min(max(0, x), max_w)
            x1 = x + w
            w = min(max_w, x1) - x
        if qrect:
            return QRectF(x, y, w, h)
        return [int(round(x)), int(round(y)), math.ceil(w), math.ceil(h)]

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRect(
            self.boundingRect()
            if self._text_transform_is_neutral()
            else self.logical_unpadded_rect()
        )
        return path

    def setScale(self, scale: float) -> None:
        if self._text_transform_is_neutral():
            self.setTransformOriginPoint(0, 0)
            super().setScale(scale)
            self.setCenterTransform()
            return
        with self._text_transform_update():
            self.setCenterTransform()
            super().setScale(scale)

    def setRotation(self, angle: float) -> None:
        # Qt meta-property writes bypass this Python override; itemChange() is
        # the authoritative compensation and finalization path.
        super().setRotation(angle)

    @property
    def angle(self) -> int:
        return self.blk.angle

    def toTextBlock(self) -> TextBlock:
        raise NotImplementedError

    def setAngle(self, angle: int):
        angle = self._validated_rotation_angle(angle)

        with self._text_transform_update():
            self.setCenterTransform()
            # Preview/meta-property paths intentionally do not mutate the
            # model, so the live Qt property is the authoritative comparison.
            if self.rotation() != angle:
                self.setRotation(angle)
            if self.rotation() != angle:
                raise RuntimeError('rotation change was rejected')
            self.blk.angle = angle

    def setVertical(self, vertical: bool):

        is_editing = self.is_editting()
        preserve_selection_direction = not self._text_transform_is_neutral()
        if is_editing:
            cursor = self.textCursor()
            cursor_pos = (cursor.position(), cursor.anchor().__pos__())

        valid_layout = True
        doc = self.document()
        if self.layout is not None:
            document_margin = self.layout.documentMargin()
            if isinstance(self.layout, VerticalTextDocumentLayout) == vertical:
                if self.fontformat is not None:
                    self.fontformat.vertical = vertical
                return
            self.layout.size_enlarged.disconnect(self.on_document_enlarged)
            self.layout.documentSizeChanged.disconnect(self.docSizeChanged)
        else:
            valid_layout = False
            document_margin = 0.0
            doc.contentsChanged.connect(self.on_content_changed)
            doc.contentsChange.connect(self.on_content_changing)

        if valid_layout:
            rect = self.rect() if self.layout is not None else None
        
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        doc.documentLayout().blockSignals(True)

        # Preserve BASE writing-mode letter-spacing semantics. Glyph slant is
        # layout-only and must not rewrite the document's spacing behavior.
        reset_spacing_val = 1 if vertical else self.fontformat.letter_spacing
        cursor = QTextCursor(doc)
        cursor.joinPreviousEditBlock()
        char_fmt = QTextCharFormat()
        char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
        char_fmt.setFontLetterSpacing(reset_spacing_val * 100)
        cursor.select(QTextCursor.SelectionType.Document)
        self.set_cursor_cfmt(cursor, char_fmt, True)
        cursor.endEditBlock()

        # QTextCursor formatting emits contentsChanged synchronously while the
        # old layout is still attached. Keep the writing-mode flag aligned with
        # that layout until the formatting transaction has finished, otherwise
        # effect repaint can enter the new vertical-only stroke path through an
        # old horizontal layout.
        if self.fontformat is not None:
            self.fontformat.vertical = vertical

        if vertical:
            layout = VerticalTextDocumentLayout(doc, self.fontformat)
        else:
            layout = HorizontalTextDocumentLayout(doc, self.fontformat)
        self.layout = layout
        layout.glyph_raster_failure_handler = (
            self._on_glyph_raster_failure
        )
        doc.setDocumentLayout(layout)
        layout.setGlyphSlantAngle(self._effective_text_transform().glyph_slant_angle)
        layout.updateDocumentMargin(document_margin)
        layout.size_enlarged.connect(self.on_document_enlarged)
        layout.documentSizeChanged.connect(self.docSizeChanged)
        
        if valid_layout:
            layout.setMaxSize(rect.width(), rect.height())
            self.setCenterTransform()
            self.repaint_background()
        self.doc_size_changed.emit(self.idx)

        if is_editing:
            self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
            self.setFocus()
            cursor = QTextCursor(doc)
            position, anchor = cursor_pos
            if preserve_selection_direction:
                cursor.setPosition(anchor)
                cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
            else:
                cursor.setPosition(min(position, anchor))
                cursor.setPosition(
                    max(position, anchor), QTextCursor.MoveMode.KeepAnchor
                )
            self.setTextCursor(cursor)
        if self.fontformat.gradient_enabled:
            self._refresh_gradient_geometry()

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            if self.hasFocus():
                self.change_from = from_
                self.change_added = added

    def keyPressEvent(self, e: QKeyEvent) -> None:

        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.undo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_Y:
                e.accept()
                self.redo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_V:
                if self.isEditing():
                    e.accept()
                    self.pasted.emit(self.idx)
                    return
        elif e.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.redo_signal.emit()
                return
        elif e.key() == Qt.Key.Key_Return:
            e.accept()
            self.textCursor().insertText('\n')
            return
        return super().keyPressEvent(e)

    def undo(self) -> None:
        self.in_redo_undo = True
        self.document().undo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

    def redo(self) -> None:
        self.in_redo_undo = True
        self.document().redo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_document_enlarged(self):
        size = self.documentSize()
        self.set_size(size.width(), size.height())

    def get_scale(self) -> float:
        tl = self.topLevelItem()
        if tl is not None:
            return tl.scale()
        else:
            return self.scale()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        if self._text_transform_is_neutral():
            editing = self.is_editting()
            if editing and self.background_pixmap is not None:
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawPixmap(
                    self.boundingRect().toRect(), self.background_pixmap
                )
                painter.restore()

            option.state = QStyle.State_None
            super().paint(painter, option, widget)

            if not editing and self.background_pixmap is not None:
                painter.save()
                painter.setCompositionMode(
                    QPainter.CompositionMode.CompositionMode_DestinationOver
                )
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawPixmap(
                    self.boundingRect().toRect(), self.background_pixmap
                )
                painter.restore()
            return

        # Effects must be composited inside the item before its normal fill.
        # DestinationOver against an already opaque scene would discard them.
        was_in_graphics_paint = self._in_graphics_paint
        self._in_graphics_paint = True
        try:
            self._draw_effects(painter, option.exposedRect)
            option.state = QStyle.State_None
            super().paint(painter, option, widget)
        finally:
            self._in_graphics_paint = was_in_graphics_paint

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
            self._effect_tile_cache.clear()
            self._effect_direct_stroke = True
            self._effect_cache_dirty = False
            self._effect_cache_rendered_generation = self._effect_cache_generation
            self._force_effect_tiles = False
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
            self._effect_direct_stroke = paint_stroke
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
            if not self._export_effect_render:
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
                        self._effect_cache_generation,
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
                    cached = self._effect_tile_cache.get(key)
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
                        self._effect_tile_cache[key] = cached
                        while len(self._effect_tile_cache) > 2:
                            oldest = next(iter(self._effect_tile_cache))
                            if oldest == key and len(self._effect_tile_cache) > 1:
                                oldest = next(
                                    candidate
                                    for candidate in self._effect_tile_cache
                                    if candidate != key
                                )
                            self._effect_tile_cache.pop(oldest, None)
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
            self._effect_tile_cache.clear()
            self._effect_direct_stroke = paint_stroke
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
        for key in list(self._effect_tile_cache):
            if key not in active_keys:
                self._effect_tile_cache.pop(key, None)

        self._effect_direct_stroke = vector_stroke_direct
        self._effect_cache_dirty = False
        self._effect_cache_rendered_generation = self._effect_cache_generation
        self._force_effect_tiles = False

    def _draw_direct_stroke(self, painter: QPainter):
        if not self._effect_flags()[0]:
            return
        # This path intentionally avoids every intermediate allocation. The
        # attached layout consumes the same per-fragment outline selections,
        # preserving vector geometry while shadow is omitted for this frame.
        self._paint_live_layout(painter, self._stroke_paint_context())

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
                br.width(), br.height(), requested_scale
            )
            if self._force_effect_tiles:
                plan = EffectRasterPlan(
                    'tiles', 1.0, 0, 0, EFFECT_TILE_MAX_EDGE
                )
            stale = (
                self._effect_cache_rendered_generation
                != self._effect_cache_generation
            )
            if plan.mode == 'full':
                if (
                    not self.pre_editing
                    and (
                        self.background_pixmap is None
                        or self._background_pixmap_scale != plan.tier
                        or self._effect_cache_dirty
                        or stale
                    )
                ):
                    self.repaint_background(requested_scale)
                if self._force_effect_tiles:
                    tile_plan = EffectRasterPlan(
                        'tiles', 1.0, 0, 0, EFFECT_TILE_MAX_EDGE
                    )
                    self._draw_tiled_effects(
                        painter, tile_plan, exposed_rect
                    )
                    if self._effect_direct_stroke:
                        self._draw_direct_stroke(painter)
                    return
                if (
                    self.background_pixmap is not None
                    and self._background_pixmap_scale == plan.tier
                    and self._effect_cache_rendered_generation
                    == self._effect_cache_generation
                ):
                    painter.setRenderHint(
                        QPainter.RenderHint.SmoothPixmapTransform
                    )
                    painter.drawPixmap(br.topLeft(), self.background_pixmap)
                elif self._effect_direct_stroke:
                    self._draw_direct_stroke(painter)
            else:
                # A previous ordinary-size fast cache must never be stretched
                # over a new huge local surface.
                self.background_pixmap = None
                self._background_pixmap_scale = None
                self._draw_tiled_effects(painter, plan, exposed_rect)
                if self._effect_direct_stroke:
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


    def startEdit(self, pos: QPointF = None) -> None:
        self.pre_editing = False
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.refresh_cache_policy()
        self.setFocus()
        self.begin_edit.emit(self.idx)
        if pos is not None:
            hit = self.layout.hitTest(pos, None)
            cursor = self.textCursor()
            cursor.setPosition(hit)
            self.setTextCursor(cursor)

    def endEdit(self, keep_focus=True) -> None:
        self.end_edit.emit(self.idx)
        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.refresh_cache_policy()
        if keep_focus:
            self.setFocus()

    def isEditing(self) -> bool:
        return self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction

    def isMultiFontSize(self) -> bool:
        doc = self.document()
        block = doc.firstBlock()
        if block.isValid():
            it = block.begin()
            if it.atEnd():
                firstFontSize = block.charFormat().fontPointSize()
            else:
                # empty blocks causes frozen for pyside==6.8.1
                # also randomly freezes pyqt==6.6.1 https://github.com/dmMaze/BallonsTranslator/issues/736
                firstFontSize = it.fragment().charFormat().fontPointSize()
        else:
            return False
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                font_size = fragment.charFormat().fontPointSize()
                if not firstFontSize == font_size:
                    return True
                it += 1
            block = block.next()
        return False
    
    def minFontSize(self, to_px=True):
        doc = self.document()
        block = doc.firstBlock()
        min_font_size = self.textCursor().charFormat().fontPointSize()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                font_size = fragment.charFormat().fontPointSize()
                min_font_size = min(min_font_size, font_size)
                it += 1
            block = block.next()
        if to_px:
            min_font_size = pt2px(min_font_size)
        return min_font_size

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if not self.isEditing():
            self.startEdit(pos=event.pos())
        else:
            super().mouseDoubleClickEvent(event)
        
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)  
        if self.textInteractionFlags() != Qt.TextInteractionFlag.TextEditorInteraction:
            self.moving.emit(self)

    # QT 5.15.x causing segmentation fault 
    def contextMenuEvent(self, event):
        return super().contextMenuEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.oldPos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.oldPos != self.pos():
                self.moved.emit()
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_move.emit(self.idx)
        return super().hoverMoveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_enter.emit(self.idx)
        return super().hoverEnterEvent(event)

    def toPixmap(self) -> QPixmap:
        pixmap = QPixmap(self.boundingRect().size().toSize())
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        doc = self.document()
        doc.drawContents(painter)
        painter.end()
        return pixmap

    def toHtml(self) -> str:
        html = super().toHtml()
        tables = table_pattern.findall(html)
        if tables:
            _, td = td_pattern.findall(html)[0]
            html = tables[0] + td + '</body></html>'

        return html.replace('>\n<', '><')

    def get_fontformat(self) -> FontFormat:
        fmt = self.textCursor().charFormat()
        font = fmt.font()
        color = fmt.foreground().color()
        fontformat = self.fontformat.deepcopy()
        fontformat.frgb = [color.red(), color.green(), color.blue()]
        fontformat.font_weight = font.weight()
        fontformat.font_family = font.family()
        if self.isEditing():
            fontformat.font_size = pt2px(font.pointSizeF())
        else:
            fontformat.font_size = self.minFontSize()
        fontformat.bold = font.bold()
        fontformat.underline = font.underline()
        fontformat.italic = font.italic()
        # Preserve gradient settings
        fontformat.gradient_enabled = self.fontformat.gradient_enabled
        fontformat.gradient_start_color = self.fontformat.gradient_start_color
        fontformat.gradient_end_color = self.fontformat.gradient_end_color
        fontformat.gradient_angle = self.fontformat.gradient_angle
        fontformat.gradient_size = self.fontformat.gradient_size
        # Selection changes can detach the render/UI format cache from the
        # persistent TextBlock owner.  The canonical quartet must always win
        # when producing a save/undo format snapshot.
        (
            fontformat.horizontal_scale,
            fontformat.vertical_scale,
            fontformat.slant_angle,
            fontformat.glyph_slant_angle,
        ) = self.blk.fontformat.text_transform
        return fontformat

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False, set_stroke_width=True, set_effect=True):
        self.repainting = True
        if self.fontformat.vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        format = cursor.charFormat()
        font = self.document().defaultFont()

        font.setFamily(ffmat.font_family)
        font.setPointSizeF(ffmat.size_pt)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)

        fweight = ffmat.font_weight
        if fweight is  None:
            fweight = font.weight()
            ffmat.font_weight = fweight
        font.setBold(ffmat.bold)

        self.document().setDefaultFont(font)
        format.setFont(font)
        if ffmat.gradient_enabled:
            gradient = self.get_text_gradient(ffmat, persistent=True)
            format.setForeground(gradient)
        else:
            format.setForeground(QColor(*ffmat.foreground_color()))
        if not ffmat.bold:
            format.setFontWeight(fweight)
        format.setFontItalic(ffmat.italic)
        format.setFontUnderline(ffmat.underline)
        if not ffmat.vertical:
            format.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            format.setFontLetterSpacing(ffmat.letter_spacing * 100)
        cursor.setCharFormat(format)
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.setBlockCharFormat(format)
        if set_char_format:
            cursor.setCharFormat(format)
        cursor.clearSelection()
        # https://stackoverflow.com/questions/37160039/set-default-character-format-in-qtextdocument
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        self.stroke_qcolor = QColor(*ffmat.stroke_color())

        if set_effect:
            self.setShadow(ffmat, repaint=False)
        if set_stroke_width:
            self.setStrokeWidth(ffmat.stroke_width, repaint_background=False)
        self.setOpacity(ffmat.opacity)
        
        alignment_qt_flag = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][ffmat.alignment]
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(alignment_qt_flag)
        doc.setDefaultTextOption(op)
        
        if ffmat.vertical:
            self.setLetterSpacing(ffmat.letter_spacing)
        self.setLineSpacing(ffmat.line_spacing)
        
        # Preserve gradient properties
        self.fontformat.gradient_enabled = ffmat.gradient_enabled
        self.fontformat.gradient_start_color = ffmat.gradient_start_color
        self.fontformat.gradient_end_color = ffmat.gradient_end_color
        self.fontformat.gradient_angle = ffmat.gradient_angle
        self.fontformat.gradient_size = ffmat.gradient_size
        
        self.fontformat.merge(ffmat)
        self.set_text_transform(*self.fontformat.text_transform)

        self.repainting = False
        if self.fontformat.gradient_enabled:
            self._refresh_gradient_geometry()
            self.update()
        if set_effect or set_stroke_width:
            self.repaint_background()

    def updateBlkFormat(self):
        fmt = self.get_fontformat()
        self.blk.fontformat.merge(fmt)

    def set_cursor_cfmt(self, cursor: QTextCursor, cfmt: QTextCharFormat, merge_char: bool = False):
        doc_is_empty = self.document().isEmpty()
        if merge_char:
            self.block_change_signal = True
            cursor.mergeCharFormat(cfmt)
            self.block_change_signal = False
        cursor.mergeBlockCharFormat(cfmt)
        cursor.clearSelection()
        self.setTextCursor(cursor)
        if doc_is_empty:
            self.document().setDefaultFont(cursor.blockCharFormat().font())

    def _before_set_ffmt(self, set_selected: bool, restore_cursor: bool):
        self.is_formatting = True
        cursor = self.textCursor()

        cursor_pos = None
        if restore_cursor:
            cursor_pos = (cursor.position(), cursor.anchor().__pos__()) if restore_cursor else None

        if set_selected:
            has_set_all = not cursor.hasSelection()
            if has_set_all:
                cursor.select(QTextCursor.SelectionType.Document)
        else:
            has_set_all = False
            cursor = QTextCursor(self.document())
            cursor.select(QTextCursor.SelectionType.Document)

        cursor.beginEditBlock()
        return cursor, dict(cursor_pos=cursor_pos, has_set_all=has_set_all)

    def _after_set_ffmt(self, cursor: QTextCursor, repaint_background: bool, restore_cursor: bool, cursor_pos: Tuple, has_set_all: bool):
        
        if restore_cursor:
            if cursor_pos is not None:
                pos1, pos2 = cursor_pos
                if has_set_all:
                    cursor.setPosition(pos1)
                else:
                    cursor.setPosition(min(pos1, pos2))
                    cursor.setPosition(max(pos1, pos2), QTextCursor.MoveMode.KeepAnchor)
                self.setTextCursor(cursor)

        if repaint_background:
            self.repaint_background()

        cursor.endEditBlock()
        self.is_formatting = False

    def setFontFamily(self, value: str, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        self.layout.relayout_on_changed = False
        self._doc_set_font_family(value, cursor)
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def _doc_set_font_family(self, value: str, cursor: QTextCursor):
        doc = self.document()
        lastpos = doc.rootFrame().lastPosition()
        if cursor.selectionStart() == 0 and \
            cursor.selectionEnd() == lastpos:
            font = doc.defaultFont()
            font.setFamily(value)
            doc.setDefaultFont(font)

        sel_start = cursor.selectionStart()
        sel_end = cursor.selectionEnd()
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                
                frag_start = fragment.position()
                frag_end = frag_start + fragment.length()
                pos2 = min(frag_end, sel_end)
                pos1 = max(frag_start, sel_start)
                if pos1 < pos2:
                    cfmt = fragment.charFormat()
                    under_line = cfmt.fontUnderline()
                    cfont = cfmt.font()
                    font = QFont(value, cfont.pointSize(), cfont.weight(), cfont.italic())
                    font.setPointSizeF(cfont.pointSizeF())
                    font.setBold(font.bold())
                    font.setWordSpacing(cfont.wordSpacing())
                    font.setLetterSpacing(cfont.letterSpacingType(), cfont.letterSpacing())
                    cfmt.setFont(font)
                    cfmt.setFontUnderline(under_line)
                    cursor.setPosition(pos1)
                    cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                    cursor.setCharFormat(cfmt)
                it += 1
            block = block.next()

        cfmt = cursor.charFormat()
        cfmt.setFontFamily(value)
        self.set_cursor_cfmt(cursor, cfmt)

    def setFontWeight(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontWeight(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setFontItalic(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontItalic(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setFontUnderline(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontUnderline(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setGradientEnabled(self, value: bool, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.fontformat.gradient_enabled = value
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        if value:
            gradient = self.get_text_gradient(persistent=True)
            cfmt.setForeground(gradient)
        else:
            cfmt.setForeground(QColor(*[int(c) for c in self.fontformat.frgb]))

        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)
        self._refresh_gradient_geometry()

    def _refresh_gradient_geometry(self):
        """Refresh the block-local gradient as non-document layout state."""
        if self._refreshing_gradient_geometry:
            return
        neutral = self._text_transform_is_neutral()
        if neutral:
            block = self.document().firstBlock()
            has_transient_range = False
            while block.isValid() and not has_transient_range:
                has_transient_range = any(
                    bool(
                        format_range.format.property(
                            GRADIENT_LAYOUT_FORMAT_PROPERTY
                        )
                    )
                    for format_range in block.layout().formats()
                )
                block = block.next()
            if not has_transient_range:
                return
        self._refreshing_gradient_geometry = True
        gradient_format = None
        if not neutral and self.fontformat.gradient_enabled:
            gradient_format = QTextCharFormat()
            gradient_format.setForeground(self.get_text_gradient())
            gradient_format.setProperty(GRADIENT_LAYOUT_FORMAT_PROPERTY, True)
        try:
            formats_changed = False
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
        finally:
            self._refreshing_gradient_geometry = False

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
            # The document foreground is the BASE-neutral fallback underneath
            # the active layout-only gradient range. Reconstruct the neutral
            # entry rectangle so removing that range cannot reveal coordinates
            # derived from an active Box transform or its exact effect padding.
            logical_rect = self.logical_unpadded_rect()
            entry_padding = self._text_transform_entry_padding or 0.0
            rect = QRectF(
                0.0,
                0.0,
                logical_rect.width() + entry_padding * 2,
                logical_rect.height() + entry_padding * 2,
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

    def setLineSpacing(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing = value
        self.layout.setLineSpacing(value)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.is_formatting = False

    def setLineSpacingType(self, value: int, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing_type = value
        self.layout.setLineSpacingType(value)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.is_formatting = False

    def setLetterSpacing(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False, force=False):
        self.is_formatting = True
        self.fontformat.letter_spacing = value
        if self.fontformat.vertical:
            self.layout.setLetterSpacing(value)
        else:
            cursor = QTextCursor(self.document())
            char_fmt = QTextCharFormat()
            char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
            char_fmt.setFontLetterSpacing(value * 100)
            cursor.select(QTextCursor.SelectionType.Document)
            self.set_cursor_cfmt(cursor, char_fmt, True)

        if repaint_background:
            self.repaint_background()
            self.update()

        self.is_formatting = False

    def setFontColor(self, value: Tuple, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, force=False):
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setForeground(QColor(*value))
        self.set_cursor_cfmt(cursor, cfmt, True)
        self._after_set_ffmt(cursor, repaint_background=repaint_background, restore_cursor=restore_cursor, **after_kwargs)

    def setStrokeColor(self, scolor, **kwargs):
        self.stroke_qcolor = scolor if isinstance(scolor, QColor) else QColor(*scolor)
        self.fontformat.srgb = [self.stroke_qcolor.red(), self.stroke_qcolor.green(), self.stroke_qcolor.blue()]
        self.repaint_background()
        self.update()

    def setStrokeWidth(self, stroke_width: float, padding=True, repaint_background=True, restore_cursor=False, **kwargs):
        
        cursor, after_kwargs = self._before_set_ffmt(set_selected=False, restore_cursor=restore_cursor)

        self.fontformat.stroke_width = stroke_width
        if padding:
            if self._text_transform_is_neutral():
                if stroke_width > 0:
                    effect_padding = (
                        self.layout.max_font_size(to_px=True)
                        * (stroke_width + 0.05)
                        / 2
                    )
                    self.setPadding(effect_padding)
            else:
                self._update_effect_padding()

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)
        if repaint_background:
            self.update()

    def setRelFontSize(self, value: float, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, clip_size: bool = False, **kwargs):
        self.layout.relayout_on_changed = False
        _, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        doc = self.document()
        cursor = QTextCursor(doc)
        block = doc.firstBlock()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                old_font_size = fragment.charFormat().fontPointSize()
                new_font_size = round(old_font_size * value,2)
                cfmt = fragment.charFormat()
                cfmt.setFontPointSize(new_font_size)
                pos1 = fragment.position()
                pos2 = pos1 + fragment.length()
                cursor.setPosition(pos1)
                cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                cursor.mergeCharFormat(cfmt)
                it += 1
            block = block.next()
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        if not self._text_transform_is_neutral() and (
            self.fontformat.stroke_width > 0
            or (
                self.fontformat.shadow_radius > 0
                and self.fontformat.shadow_strength > 0
            )
        ):
            repaint_background = True
        if clip_size:
            self.squeezeBoundingRect(True, repaint=False)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)
        

    def setFontSize(self, value: float, repaint_background: bool = False, set_selected: bool = False, restore_cursor: bool = False, clip_size: bool = False, **kwargs):
        '''
        value should be point size
        '''
        
        cursor, after_kwargs = self._before_set_ffmt(set_selected=set_selected, restore_cursor=restore_cursor)
        self.layout.relayout_on_changed = False
        if self._text_transform_is_neutral():
            if self.fontformat.stroke_width != 0:
                repaint_background = True
            if repaint_background:
                fs = pt2px(max(self.layout.max_font_size(), value))
                self.layout.relayout_on_changed = False
                if self.fontformat.stroke_width > 0:
                    self.setPadding(
                        fs * (self.fontformat.stroke_width + 0.05) / 2
                    )
                self.layout.relayout_on_changed = True
        elif self.fontformat.stroke_width > 0 or (
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0
        ):
            repaint_background = True
        cfmt = QTextCharFormat()
        cfmt.setFontPointSize(value)
        self.set_cursor_cfmt(cursor, cfmt, True)
        self.layout.relayout_on_changed = True
        self.layout.reLayoutEverything()
        if clip_size:
            self.squeezeBoundingRect(cond_on_alignment=True)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def setAlignment(self, value, restore_cursor=False, repaint_background=True, *args, **kwargs):
        cursor, after_kwargs = self._before_set_ffmt(set_selected=False, restore_cursor=restore_cursor)
        if isinstance(value, int):
            qt_align_flag = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][value]
        doc = self.document()
        op = doc.defaultTextOption()
        op.setAlignment(qt_align_flag)
        doc.setDefaultTextOption(op)
        if repaint_background:
            self.repaint_background()
            self.update()
        self.fontformat.alignment = value
        self._after_set_ffmt(cursor, repaint_background=False, restore_cursor=restore_cursor, **after_kwargs)

    def get_char_fmts(self) -> List[QTextCharFormat]:
        cursor = self.textCursor()
        
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        char_fmts = []
        while True:
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
            cursor.clearSelection()
            char_fmts.append(cursor.charFormat())
            if cursor.atEnd():
                break
        return char_fmts

    def setShadow(self, fmt: FontFormat, repaint=True):
        self.fontformat.shadow_radius = fmt.shadow_radius
        self.fontformat.shadow_strength = fmt.shadow_strength
        self.fontformat.shadow_color = fmt.shadow_color
        self.fontformat.shadow_offset = fmt.shadow_offset
        if self._text_transform_is_neutral():
            if self.fontformat.shadow_radius > 0:
                self.setPadding(self.layout.max_font_size(to_px=True))
        else:
            self._update_effect_padding()
        if repaint:
            self.repaint_background()

    def setBGAttribute(self, attr_name: str, value, repaint=True):
        setattr(self.fontformat, attr_name, value)
        if not self._text_transform_is_neutral():
            self._update_effect_padding()
        if repaint:
            self.repaint_background()
            self.update()

    def setGradientAttribute(self, attr_name: str, value):
        self.old_ffmt_values = {}
        self.old_ffmt_values[attr_name] = self.fontformat[attr_name]
        setattr(self.fontformat, attr_name, value)
        self.setGradientEnabled(self.fontformat.gradient_enabled)
        self.old_ffmt_values = None

    def setOpacity(self, opacity: float):
        super().setOpacity(opacity)
        self.fontformat.opacity = opacity

    def setPlainTextAndKeepUndoStack(self, text: str):
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)

    def squeezeBoundingRect(self, cond_on_alignment: bool = False, repaint=True):
        mh, mw = self.layout.minSize()
        if mh == 0 or mw == 0:
            return
        br = self.absBoundingRect(qrect=True)
        br_w, br_h = br.width(), br.height()

        if self.fontformat.vertical:
            if cond_on_alignment:
                mh = br.height()
        else:
            if cond_on_alignment:
                mw = br.width()

        if np.abs(br_w - mw) > 0.001 or np.abs(br_h - mh) > 0.001:
            P = self.padding() * 2
            mh += P
            mw += P
            self.set_size(mw, mh, set_layout_maxsize=True, set_blk_size=True)
            if self._text_transform_is_neutral() and self.under_ctrl:
                self.doc_size_changed.emit(self.idx)
            if repaint:
                self.repaint_background()

    def _size_alignment_anchor(self, rect: QRectF) -> QPointF:
        """Return the semantic anchor preserved by automatic resizing."""
        if (
            self.fontformat.vertical
            or self.fontformat.alignment == TextAlignment.Right
        ):
            return rect.topRight()
        if self.fontformat.alignment == TextAlignment.Left:
            return rect.topLeft()
        return rect.center()

    def scene_scale_factor(self):
        """Return the legacy scene scale used by neutral resize positioning."""
        scale = 1
        if hasattr(self.scene(), 'scale_factor'):
            scale = self.scene().scale_factor
        return scale

    def set_size(
        self,
        w: float,
        h: float,
        set_layout_maxsize=False,
        set_blk_size=True,
    ):
        """Resize with affine anchor correction only for active box transforms.

        Neutral box geometry deliberately follows the BASE implementation.
        Active affine geometry blocks the layout signal until the display rect,
        transform, anchor, and document size agree.
        """
        transform = self._effective_text_transform()
        active_box_transform = (
            transform.horizontal_scale != 1.0
            or transform.vertical_scale != 1.0
            or transform.slant_angle != 0.0
        )
        if not active_box_transform:
            if set_layout_maxsize:
                self.layout.setMaxSize(w, h)

            old_w = self._display_rect.width()
            old_h = self._display_rect.height()
            old_center = self.sceneBoundingRect().center()
            self._display_rect.setWidth(w)
            self._display_rect.setHeight(h)
            self.setCenterTransform()
            pos_shift = old_center - self.sceneBoundingRect().center()
            pos_shift = pos_shift / self.scene_scale_factor()

            align_center = align_top_left = align_top_right = False
            if self.fontformat.vertical:
                align_top_right = True
            else:
                alignment = self.fontformat.alignment
                if alignment == TextAlignment.Left:
                    align_top_left = True
                elif alignment == TextAlignment.Right:
                    align_top_right = True
                else:
                    align_center = True

            if not align_center:
                dw, dh = (w - old_w) / 2, (h - old_h) / 2
                if align_top_right:
                    dw = -dw
                rad = -np.deg2rad(self.rotation())
                c, s = np.cos(rad), np.sin(rad)
                dx = c * dw + s * dh
                dy = -s * dw + c * dh
                pos_shift = pos_shift + QPointF(dx, dy)

            self.setPos(self.pos() + pos_shift)
            if self.blk is not None and set_blk_size:
                self.blk._bounding_rect = self.absBoundingRect()
            return

        if self.transformations():
            raise RuntimeError(
                'TextBlkItem requires an empty QGraphicsTransform list'
            )
        old_rect = self.logical_unpadded_rect()
        old_anchor_parent = self.mapToParent(
            self._size_alignment_anchor(old_rect)
        )

        # Both the custom display rect and a synchronous QTextDocument relayout
        # can change boundingRect(), so notify the scene before either mutation.
        self.prepareGeometryChange()
        signals_were_blocked = None
        final_size = None
        if set_layout_maxsize:
            signals_were_blocked = self.layout.blockSignals(True)
        try:
            if set_layout_maxsize:
                self.layout.setMaxSize(w, h)
                final_size = QSizeF(self.layout.documentSize())
                w = final_size.width()
                h = final_size.height()

            with self._text_transform_update():
                self._display_rect.setWidth(w)
                self._display_rect.setHeight(h)
                self.setCenterTransform()
                new_rect = self.logical_unpadded_rect()
                new_anchor_parent = self.mapToParent(
                    self._size_alignment_anchor(new_rect)
                )
                self.setPos(
                    self.pos() + old_anchor_parent - new_anchor_parent
                )

            if self.blk is not None and set_blk_size:
                self.blk._bounding_rect = self.absBoundingRect()
        finally:
            if set_layout_maxsize:
                self.layout.blockSignals(signals_were_blocked)

        if set_layout_maxsize and not signals_were_blocked:
            self.layout.documentSizeChanged.emit(QSizeF(final_size))
