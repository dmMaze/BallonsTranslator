import math, re
import cv2
import numpy as np
from typing import List, Optional, Union, Tuple

from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QStyle, QGraphicsSceneMouseEvent
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, Signal, QSizeF
from qtpy.QtGui import (QGradient, QKeyEvent, QFont, QTextCursor, QImage, QPixmap, QPainterPath,
                       QInputMethodEvent, QPainter, QPen, QColor, QTextCharFormat, QLinearGradient,
                       QBrush, QPalette, QAbstractTextDocumentLayout, QPolygonF, QTextLayout,
                       QTextOption)

from ballontranslator.utils.textblock import TextBlock, FontFormat, TextAlignment, LineSpacingType
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly, rotate_polygons
from ballontranslator.utils.fontformat import FontFormat, normalize_text_transform, px2pt, pt2px
from .misc import td_pattern, table_pattern, pixmap2ndarray, ndarray2pixmap
from .scene_textlayout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout, SceneTextLayout
from .text_graphical_effect import apply_shadow_effect
from .text_transform import rect_polygon, text_transform_matrix

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)
# At the minimum cache DPR (1), one logical pixel contains the final
# antialias/kernel sample and one further pixel guarantees a transparent
# texture-sampling border. Higher DPR tiers receive at least the same two
# device rows. These are raster invariants, not transform-dependent padding.
EFFECT_ANTIALIAS_GUARD = 1.0
EFFECT_CLEAR_BORDER_GUARD = 1.0
EFFECT_RASTER_GUARD = EFFECT_ANTIALIAS_GUARD + EFFECT_CLEAR_BORDER_GUARD
# QTextLayout additional formats are derived paint state, not document state.
# The marker lets us replace only our block-gradient override while preserving
# any unrelated highlighter ranges attached to the same live layout.
GRADIENT_LAYOUT_FORMAT_PROPERTY = 0x100000 + 1238


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
        self.pre_editing = False
        self.blk: TextBlock = None
        self.fontformat: FontFormat = None
        self._text_transform_preview: Optional[Tuple[float, float, float]] = None
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
        self.document().setDocumentMargin(0)
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self._update_text_transform_cache_mode()

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

    def _new_effect_pixmap(self, render_scale: float = 1.0) -> QPixmap:
        rect = self.boundingRect()
        pixmap = QPixmap(
            max(1, math.ceil(rect.width() * render_scale)),
            max(1, math.ceil(rect.height() * render_scale)),
        )
        pixmap.setDevicePixelRatio(render_scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        return pixmap

    def _paint_vertical_stroke(
        self, painter: QPainter, render_scale: float = 1.0
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
            source = self._new_effect_pixmap(render_scale)
            source_painter = QPainter(source)
            try:
                source_painter.translate(-self.boundingRect().topLeft())
                fragment_context = self._effect_paint_context()
                fragment_context.selections = selections
                self.layout.draw_glyph_selection_mask(
                    source_painter, fragment_context
                )
            finally:
                source_painter.end()

            rgba = pixmap2ndarray(source, keep_alpha=True)
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
        stroke_pixmap = ndarray2pixmap(stroke)
        stroke_pixmap.setDevicePixelRatio(render_scale)
        painter.drawPixmap(self.boundingRect().topLeft(), stroke_pixmap)

    def paint_stroke(self, painter: QPainter, render_scale: float = 1.0):
        if self.fontformat.vertical:
            self._paint_vertical_stroke(painter, render_scale)
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
        while True:
            image = QImage(
                max(1, math.ceil(logical_rect.width()) + font_guard * 2),
                max(1, math.ceil(logical_rect.height()) + font_guard * 2),
                QImage.Format.Format_ARGB32,
            )
            if image.isNull():
                raise RuntimeError('unable to allocate text ink measurement image')
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            try:
                painter.translate(
                    font_guard - logical_rect.left(),
                    font_guard - logical_rect.top(),
                )
                self._paint_live_layout(painter, self._effect_paint_context())
            finally:
                painter.end()

            alpha = pixmap2ndarray(image, keep_alpha=True)[..., 3]
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
        if not paint_stroke and not paint_shadow:
            return 0.0
        ink_bounds = self._logical_ink_bounds()
        if ink_bounds.isEmpty():
            return 0.0
        stroke_outset = self._stroke_outset()
        logical_size = self.logical_unpadded_rect().size()
        logical_rect = QRectF(QPointF(), logical_size)
        effect_bounds = ink_bounds.adjusted(
            -stroke_outset,
            -stroke_outset,
            stroke_outset,
            stroke_outset,
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

    def _update_effect_padding(self):
        padding = self._effect_padding()
        # QTextLayout stores coordinates at 26.6 fixed-point precision. Use the
        # same grid as the canonical envelope and round outward so repeated
        # relayout/undo cycles converge without ever undersizing the effects.
        if padding > 0.0:
            layout_units = math.nextafter(padding * 64.0, -math.inf)
            padding = math.ceil(layout_units) / 64.0
        changed = self.setPadding(padding) if self.padding() != padding else False
        if changed and self.fontformat.gradient_enabled:
            was_repainting = self.repainting
            self.repainting = True
            try:
                self._refresh_gradient_geometry()
            finally:
                self.repainting = was_repainting
        return changed

    def repaint_background(self, render_scale: float = 1.0):
        empty = self.document().isEmpty()
        if self.repainting or self.reshaping or self.pre_editing:
            # Avoid reshape/reentrant work. During IME, reuse the preedit-free
            # cache because PaintContext cannot exclude active preedit glyphs.
            return

        self._update_effect_padding()

        paint_stroke = self.fontformat.stroke_width > 0
        paint_shadow = self.fontformat.shadow_radius > 0 and self.fontformat.shadow_strength > 0
        if not paint_shadow and not paint_stroke or empty:
            self.background_pixmap = None
            self._background_pixmap_scale = None
            return
        
        self.repainting = True
        try:
            target_map = self._new_effect_pixmap(render_scale)

            if paint_shadow:
                silhouette = self._new_effect_pixmap(render_scale)
                silhouette_painter = QPainter(silhouette)
                try:
                    silhouette_painter.translate(-self.boundingRect().topLeft())
                    self._paint_live_layout(
                        silhouette_painter, self._effect_paint_context()
                    )
                    if paint_stroke:
                        self.paint_stroke(silhouette_painter, render_scale)
                finally:
                    silhouette_painter.end()

                radius, xoffset, yoffset = self._shadow_metrics()
                shadow_map, _ = apply_shadow_effect(
                    silhouette,
                    self.fontformat.shadow_color,
                    self.fontformat.shadow_strength,
                    max(0, int(round(radius * render_scale))),
                )
                shadow_map.setDevicePixelRatio(render_scale)
                target_painter = QPainter(target_map)
                try:
                    target_painter.drawPixmap(
                        QPointF(xoffset, yoffset), shadow_map
                    )
                finally:
                    target_painter.end()

            if paint_stroke:
                stroke_painter = QPainter(target_map)
                try:
                    stroke_painter.translate(-self.boundingRect().topLeft())
                    self.paint_stroke(stroke_painter, render_scale)
                finally:
                    stroke_painter.end()

            self.background_pixmap = target_map
            self._background_pixmap_scale = render_scale
        finally:
            self.repainting = False
        
    def docSizeChanged(self):
        self.setCenterTransform()
        self.doc_size_changed.emit(self.idx)

    def initTextBlock(self, blk: TextBlock = None, set_format=True):
        if blk is None:
            xyxy = [0, 0, 0, 0]
            blk = TextBlock(xyxy)
            blk.lines = [xyxy]
            bx1, by1, bx2, by2 = xyxy
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        self.blk = blk
        self.fontformat = blk.fontformat
        self._text_transform_preview = None
        self.setVertical(blk.vertical)
        self.setRect(blk.bounding_rect(), update_blk_rect=False)
        
        if blk.angle != 0:
            self.setRotation(blk.angle)
        
        set_char_fmt = False
        if blk.translation:
            set_char_fmt = True

        font_fmt = blk.fontformat
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
        self.setCenterTransform()
        self.repaint_background()

    def _canonical_text_transform(self) -> Tuple[float, float, float]:
        return normalize_text_transform(*self.blk.fontformat.text_transform)

    def _effective_text_transform(self) -> Tuple[float, float, float]:
        if self._text_transform_preview is not None:
            return self._text_transform_preview
        return self._canonical_text_transform()

    def _update_text_transform_cache_mode(self) -> bool:
        use_no_cache = (
            self.is_editting()
            or not self.transform().isIdentity()
            or self.rotation() != 0
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

    def _apply_text_transform(self, values: Tuple[float, float, float]) -> bool:
        matrix = text_transform_matrix(
            *values,
            pivot=self.logical_unpadded_rect().center(),
        )
        changed = self.transform() != matrix
        if changed:
            self.setTransform(matrix, combine=False)
        self._update_text_transform_cache_mode()
        if changed and self.is_editting():
            self.updateMicroFocus()
        return changed

    def set_text_transform(
        self,
        horizontal_scale: float = None,
        vertical_scale: float = None,
        slant_angle: float = None,
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
        )

        if preview:
            if target == current:
                return False
            self._text_transform_preview = None if target == canonical else target
            return self._apply_text_transform(target)

        model_changed = raw_canonical != target
        if model_changed:
            fontformat = self.blk.fontformat
            (
                fontformat.horizontal_scale,
                fontformat.vertical_scale,
                fontformat.slant_angle,
            ) = target
        self._text_transform_preview = None
        visual_changed = self._apply_text_transform(target)
        return model_changed or visual_changed

    def clear_text_transform_preview(self) -> bool:
        if self._text_transform_preview is None:
            return False
        self._text_transform_preview = None
        return self._apply_text_transform(self._canonical_text_transform())

    def setCenterTransform(self) -> bool:
        center = self.logical_unpadded_rect().center()
        origin_changed = self.transformOriginPoint() != center
        if origin_changed:
            self.setTransformOriginPoint(center)
        transform_changed = self._apply_text_transform(
            self._effective_text_transform()
        )
        if origin_changed and not transform_changed and self.is_editting():
            self.updateMicroFocus()
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

    def setPadding(self, p: float):
        p = max(0.0, float(p))
        _p = self.padding()
        if _p == p:
            return False
        abr = self.absBoundingRect(qrect=True)
        was_repainting = self.repainting
        self.repainting = True
        try:
            self.layout.relayout_on_changed = False
            self.layout.updateDocumentMargin(p)
            self.layout.relayout_on_changed = True
            self.setRect(abr, repaint=False)
        finally:
            self.layout.relayout_on_changed = True
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
        br = self.boundingRect()
        path.addRect(br)
        return path

    def setScale(self, scale: float) -> None:
        self.setTransformOriginPoint(0, 0)
        super().setScale(scale)
        self.setCenterTransform()

    def setRotation(self, angle: float) -> None:
        if self.rotation() == angle:
            return
        super().setRotation(angle)
        self._update_text_transform_cache_mode()
        if self.is_editting():
            self.updateMicroFocus()

    @property
    def angle(self) -> int:
        return self.blk.angle

    def toTextBlock(self) -> TextBlock:
        raise NotImplementedError

    def setAngle(self, angle: int):
        self.setCenterTransform()
        if self.blk.angle != angle:
            self.setRotation(angle)
        self.blk.angle = angle

    def setVertical(self, vertical: bool):

        is_editing = self.is_editting()
        if is_editing:
            cursor = self.textCursor()
            cursor_pos = (cursor.position(), cursor.anchor())

        if self.fontformat is not None:
            self.fontformat.vertical = vertical

        valid_layout = True
        doc = self.document()
        if self.layout is not None:
            document_margin = self.layout.documentMargin()
            if isinstance(self.layout, VerticalTextDocumentLayout) == vertical:
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
        if vertical:
            layout = VerticalTextDocumentLayout(doc, self.fontformat)
        else:
            layout = HorizontalTextDocumentLayout(doc, self.fontformat)

        self.layout = layout
        doc.setDocumentLayout(layout)
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
            cursor.setPosition(anchor)
            cursor.setPosition(position, QTextCursor.MoveMode.KeepAnchor)
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
        # Effects must be composited inside the item before its normal fill.
        # DestinationOver against an already opaque scene would discard them.
        self._draw_effects(painter)
        option.state = QStyle.State_None
        super().paint(painter, option, widget)
        self._draw_item_guides(painter)

    def _draw_effects(self, painter: QPainter):
        br = self.boundingRect()
        painter.save()
        if self.background_pixmap is not None:
            render_scale = self._paint_device_scale(painter)
            if (
                not self.pre_editing
                and self._background_pixmap_scale != render_scale
            ):
                self.repaint_background(render_scale)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(br.topLeft(), self.background_pixmap)
        painter.restore()

    def _draw_item_guides(self, painter: QPainter):
        br = self.boundingRect()
        painter.save()
        draw_rect = self.draw_rect and not self.under_ctrl
        if self.isSelected() and not self.is_editting():
            pen = QPen(TEXTRECT_SELECTED_COLOR, 3.5 / self.get_scale(), Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.unpadRect(br))
        elif draw_rect:
            pen = QPen(TEXTRECT_SHOW_COLOR, 3 / self.get_scale(), Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.unpadRect(br))
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
        # Reuse a stable cache tier while always rasterizing at or above the
        # requested device resolution.
        return 2 ** math.ceil(math.log2(scale))


    def startEdit(self, pos: QPointF = None) -> None:
        self.pre_editing = False
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
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
        self._update_text_transform_cache_mode()
        if keep_focus:
            self.setFocus()
        else:
            self.clearFocus()

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
        return fontformat

    def _whole_fontformat_document_values(self, ffmat: FontFormat):
        """Build the exact document values used by whole-format apply."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        char_format = QTextCharFormat(cursor.charFormat())
        document = self.document()
        font = QFont(document.defaultFont())

        font.setFamily(ffmat.font_family)
        font.setPointSizeF(ffmat.size_pt)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        font.setStyleStrategy(
            QFont.StyleStrategy.PreferAntialias
            | QFont.StyleStrategy.NoSubpixelAntialias
        )
        font_weight = ffmat.font_weight
        if font_weight is None:
            font_weight = font.weight()
        font.setBold(ffmat.bold)

        char_format.setFont(font)
        if ffmat.gradient_enabled:
            char_format.setForeground(self.get_text_gradient(ffmat))
        else:
            char_format.setForeground(QColor(*ffmat.foreground_color()))
        if not ffmat.bold:
            char_format.setFontWeight(font_weight)
        char_format.setFontItalic(ffmat.italic)
        char_format.setFontUnderline(ffmat.underline)
        if not ffmat.vertical:
            char_format.setFontLetterSpacingType(
                QFont.SpacingType.PercentageSpacing
            )
            char_format.setFontLetterSpacing(ffmat.letter_spacing * 100)

        option = QTextOption(document.defaultTextOption())
        option.setAlignment(
            (
                Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignCenter,
                Qt.AlignmentFlag.AlignRight,
            )[ffmat.alignment]
        )
        return font, char_format, option

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False, set_stroke_width=True, set_effect=True):
        self.repainting = True
        if self.fontformat.vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)

        if ffmat.font_weight is None:
            ffmat.font_weight = self.document().defaultFont().weight()

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        font, format, option = self._whole_fontformat_document_values(ffmat)
        self.document().setDefaultFont(font)
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
        
        doc = self.document()
        doc.setDefaultTextOption(option)
        
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
            gradient = self.get_text_gradient()
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
        self._refreshing_gradient_geometry = True
        gradient_format = QTextCharFormat()
        gradient_format.setForeground(self.get_text_gradient())
        gradient_format.setProperty(GRADIENT_LAYOUT_FORMAT_PROPERTY, True)
        try:
            block = self.document().firstBlock()
            while block.isValid():
                layout = block.layout()
                ranges = [
                    format_range
                    for format_range in layout.formats()
                    if not bool(
                        format_range.format.property(
                            GRADIENT_LAYOUT_FORMAT_PROPERTY
                        )
                    )
                ]
                text_length = block.length() - 1
                if self.fontformat.gradient_enabled and text_length > 0:
                    format_range = QTextLayout.FormatRange()
                    format_range.start = 0
                    format_range.length = text_length
                    format_range.format = gradient_format
                    ranges.append(format_range)
                layout.setFormats(ranges)
                block = block.next()
            # setFormats invalidates QTextLine objects. Rebuild them through
            # the attached custom layout; this changes no document state.
            self.layout.reLayout()
            self.update()
        finally:
            self._refreshing_gradient_geometry = False

    def get_text_gradient(self, fontformat: FontFormat = None):
        gradient = QLinearGradient()
        if fontformat is None:
            fontformat = self.fontformat
        angle = fontformat.gradient_angle
        rad = math.radians(angle)
        dx = math.cos(rad)
        dy = math.sin(rad)
        
        # Set gradient points with size adjustment
        rect = self.logical_unpadded_rect()
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
        if self.fontformat.stroke_width > 0 or (
            self.fontformat.shadow_radius > 0
            and self.fontformat.shadow_strength > 0
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
        if self.fontformat.stroke_width > 0 or (
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
        self._update_effect_padding()
        if repaint:
            self.repaint_background()

    def setBGAttribute(self, attr_name: str, value, repaint=True):
        setattr(self.fontformat, attr_name, value)
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
            if self.under_ctrl:
                self.doc_size_changed.emit(self.idx)
            if repaint:
                self.repaint_background()

    def scene_scale_factor(self):
        scale = 1
        if hasattr(self.scene(), 'scale_factor'):
            scale = self.scene().scale_factor
        return scale
            
    def set_size(self, w: float, h: float, set_layout_maxsize=False, set_blk_size=True):
        '''
        rotation invariant
        '''

        if set_layout_maxsize:
            self.layout.setMaxSize(w, h)

        old_w = self._display_rect.width()
        old_h = self._display_rect.height()

        oc = self.sceneBoundingRect().center()
        self._display_rect.setWidth(w)
        self._display_rect.setHeight(h)
        self.setCenterTransform()
        pos_shift = oc - self.sceneBoundingRect().center()
        pos_shift = pos_shift / self.scene_scale_factor()
        
        align_c = align_tl = align_tr = False
        if self.fontformat.vertical:
            align_tr = True
        else:
            alignment = self.fontformat.alignment
            if alignment == TextAlignment.Left:
                align_tl = True
            elif alignment == TextAlignment.Right:
                align_tr = True
            else:
                align_c = True

        if align_c:
            pass
        else:
            dw, dh = (w - old_w) / 2, (h - old_h) / 2
            if align_tr:
                dw = -dw
            rad = -np.deg2rad(self.rotation())
            c, s = np.cos(rad), np.sin(rad)
            dx = c * dw + s * dh
            dy = -s * dw + c * dh
            pos_shift = pos_shift + QPointF(dx, dy)

        self.setPos(self.pos() + pos_shift)
        if self.blk is not None and set_blk_size:
            self.blk._bounding_rect = self.absBoundingRect()
