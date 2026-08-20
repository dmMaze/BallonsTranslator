import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from qtpy import QT6
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QWidget,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneHoverEvent,
    QGraphicsTextItem,
    QStyleOptionGraphicsItem,
    QGraphicsSceneMouseEvent,
)
from qtpy.QtCore import Qt, QRect, QRectF, QPoint, QPointF, QMimeData, Signal
from qtpy.QtGui import (QKeyEvent, QFont, QTextCursor,
                       QInputMethodEvent, QPainter, QColor, QTextCharFormat,
                       QBrush, QFontMetrics, QPen,
                       QTextBlockFormat)

from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly
from ballontranslator.utils.fontformat import (
    FontFormat,
    FontWeight,
    LineSpacingType,
    TextTransformStack,
    font_weight_from_qt,
    font_weight_to_qt,
    pt2px,
)
from .font_family import (
    font_family_for_project,
    qfont_with_family,
)
from .editing.context_menu import create_text_edit_context_menu
from ..misc import td_pattern, table_pattern
from .horizontal_layout import HorizontalTextDocumentLayout
from .vertical_layout import VerticalTextDocumentLayout
from .effect_renderer import TextEffectRenderer
from .geometry import TextItemGeometryController
from .annotations import (
    AnnotationProperty,
    LIGATURE_COMMON,
    LIGATURE_CONTEXTUAL,
    LIGATURE_DISCRETIONARY,
    TEXT_COMBINE_ALL,
    apply_emphasis,
    apply_ligature_axis,
    apply_oldstyle_nums,
    apply_line_spacing,
    apply_letter_spacing,
    apply_ruby,
    apply_text_combine_upright,
    canonical_letter_spacing,
    create_rich_text_mime,
    emphasis_values,
    insert_rich_text_mime,
    ligature_axis_value,
    letter_spacing_value,
    line_spacing_values,
    load_rich_text_html,
    oldstyle_nums_value,
    prepare_ruby_insertion,
    remove_ruby,
    ruby_container_for_cursor,
    ruby_containers_intersecting_cursor,
    set_document_letter_spacing_writing_mode,
    set_ligature_axes,
    set_oldstyle_nums,
    sync_native_ligature_shaping,
    text_combine_upright_values,
    to_rich_text_html,
    validated_line_spacing,
)

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)


class _OrderBadgeItem(QGraphicsItem):
    """Paint a fixed-size badge outside its parent's text geometry.

    >>> _OrderBadgeItem.HORIZONTAL_PADDING
    4
    """

    HORIZONTAL_PADDING = 4
    VERTICAL_PADDING = 2

    def __init__(self, parent: QGraphicsItem) -> None:
        super().__init__(parent)
        self._font = QFont()
        self._font.setBold(True)
        self._font.setPixelSize(11)
        self._text = ''
        self._bounds = QRectF()
        self._selected = False
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True,
        )
        # Keep the badge out of the parent's cached paint surface.
        self.setCacheMode(QGraphicsItem.CacheMode.NoCache)
        self.setZValue(100.0)
        self.hide()

    def boundingRect(self) -> QRectF:
        return QRectF(self._bounds)

    def set_number(self, number: int) -> None:
        text = str(max(1, int(number)))
        if self._text == text:
            return
        metrics = QFontMetrics(self._font)
        width = metrics.horizontalAdvance(text) + 2 * self.HORIZONTAL_PADDING
        height = metrics.height() + 2 * self.VERTICAL_PADDING
        self.prepareGeometryChange()
        self._text = text
        # The bottom-left corner remains attached to the block's top-left.
        self._bounds = QRectF(0, -height, width, height)
        self.update()

    def paint(
        self,
        painter: QPainter,
        _option: QStyleOptionGraphicsItem,
        _widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        try:
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(
                TEXTRECT_SELECTED_COLOR
                if self._selected
                else TEXTRECT_SHOW_COLOR
            )
            painter.drawRoundedRect(self._bounds, 3, 3)
            painter.setPen(Qt.GlobalColor.white)
            painter.setFont(self._font)
            painter.drawText(
                self._bounds,
                Qt.AlignmentFlag.AlignCenter,
                self._text,
            )
        finally:
            painter.restore()

    def set_selected(self, selected: bool) -> None:
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.update()


class TextBlkItem(QGraphicsTextItem):

    begin_edit = Signal(int)
    end_edit = Signal(int)
    hover_enter = Signal(int)
    move_interaction_finished = Signal()
    moving = Signal(QGraphicsTextItem)
    rotated = Signal(float)
    reshaped = Signal(QGraphicsTextItem)
    leftbutton_pressed = Signal(int)
    pasted = Signal(int)
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int, bool)
    propagate_user_edited = Signal(int, int, str, bool)
    visual_geometry_changed = Signal()
    inline_format_changed = Signal()

    def __init__(self, blk: TextBlock = None, idx: int = 0, set_format=True, show_rect=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_controller = TextItemGeometryController(self)
        self.effect_renderer = TextEffectRenderer(self)
        self.pre_editing = False
        self.blk: TextBlock = None
        self.fontformat: FontFormat = None
        self.repainting = False
        self.reshaping = False
        self.under_ctrl = False
        self.draw_rect = show_rect
        self._ui_guide_suppressed = False
        self._order_badge_visible = False
        self._order_number_override: Optional[int] = None
        self._order_badge_item: Optional[_OrderBadgeItem] = None
        self.old_ffmt_values = None
        
        self.idx = idx
        
        self.stroke_qcolor = QColor(0, 0, 0)
        self._old_pos = QPointF()
        self._old_rect = QRectF()
        self.repaint_on_changed = True

        self.is_formatting = False
        self.old_undo_steps = 0
        self.in_redo_undo = False
        self.change_from: int = 0
        self.change_removed: int = 0
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_removed = 0
        self.input_method_text = ''
        self.block_change_signal = False
        self._vertical_navigation_y: Optional[float] = None

        self.layout: Union[VerticalTextDocumentLayout, HorizontalTextDocumentLayout] = None
        self.document().setDocumentMargin(0)
        self.initTextBlock(blk, set_format=set_format)
        self.setBoundingRegionGranularity(0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.geometry_controller.finish_initialization()
        self._order_badge_item = _OrderBadgeItem(self)
        self.visual_geometry_changed.connect(self._sync_order_badge)
        self._sync_order_badge()

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        self._vertical_navigation_y = None
        if not self.pre_editing:
            cursor = self.textCursor()
            self.input_method_from = cursor.selectionStart()
            self.input_method_removed = (
                cursor.selectionEnd() - cursor.selectionStart()
            )
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            self.pre_editing = True
        replacement_length = e.replacementLength()
        replacement_start = None
        replacement_end = None
        if replacement_length > 0:
            cursor = self.textCursor()
            document_end = max(0, self.document().characterCount() - 1)
            replacement_start = max(
                0,
                min(
                    document_end,
                    cursor.position() + e.replacementStart(),
                ),
            )
            replacement_end = min(
                document_end, replacement_start + replacement_length
            )
            self.input_method_from = replacement_start
            self.input_method_removed = replacement_end - replacement_start
        if e.commitString():
            cursor = self.textCursor()
            cursor.beginEditBlock()
            prepare_cursor = QTextCursor(cursor)
            if replacement_start is not None:
                prepare_cursor.setPosition(replacement_start)
                prepare_cursor.setPosition(
                    replacement_end, QTextCursor.MoveMode.KeepAnchor
                )
            prepare_ruby_insertion(prepare_cursor, e.commitString())
            if replacement_length == 0:
                cursor = prepare_cursor
            super().setTextCursor(cursor)
            try:
                super().inputMethodEvent(e)
            finally:
                cursor.endEditBlock()
        else:
            super().inputMethodEvent(e)
        if (
            e.preeditString() == ''
            and not e.commitString()
            and replacement_length == 0
        ):
            self.input_method_from = -1
            self.input_method_removed = 0
            self.input_method_text = ''
        # Preedit text and attributes live in QTextLayout, so they need an
        # explicit surface invalidation even when the document revision does
        # not change. The next paint is cached until another IME event.
        self.geometry_controller.invalidate_surface_cache()
        self._update_nonlinear_editing_ui()

    def setTextCursor(self, cursor: QTextCursor) -> None:
        self._vertical_navigation_y = None
        super().setTextCursor(cursor)
        self._emit_inline_format_changed()
        self._update_nonlinear_editing_ui()

    def _move_cursor_across_vertical_column(
        self,
        horizontal_direction: int,
        keep_anchor: bool,
    ) -> None:
        """Move the active caret to the adjacent vertical text column.

        >>> callable(TextBlkItem._move_cursor_across_vertical_column)
        True
        """
        cursor = self.textCursor()
        if self._vertical_navigation_y is None:
            caret = self.layout.source_cursor_rect(cursor.position())
            if caret.isEmpty():
                return
            self._vertical_navigation_y = caret.center().y()
        target = self.layout.adjacent_column_cursor_position(
            cursor.position(),
            horizontal_direction,
            self._vertical_navigation_y,
        )
        if target is None:
            return
        move_mode = (
            QTextCursor.MoveMode.KeepAnchor
            if keep_anchor
            else QTextCursor.MoveMode.MoveAnchor
        )
        cursor.setPosition(target, move_mode)
        # This is a cursor-only operation; bypass the public setter so the
        # preferred row survives consecutive Left/Right key presses.
        super().setTextCursor(cursor)
        self._emit_inline_format_changed()
        self._update_nonlinear_editing_ui()

    def _emit_inline_format_changed(self) -> None:
        self.inline_format_changed.emit()

    def _update_nonlinear_editing_ui(self) -> None:
        controller = getattr(self, 'geometry_controller', None)
        # Qt's source-local dirty rectangles cannot cover warped UI pixels.
        if (
            controller is not None
            and self.isEditing()
            and controller.uses_surface_warp()
        ):
            self.update()
        
    def on_content_changed(self):
        self.geometry_controller.invalidate_surface_cache()
        if (self.hasFocus() or self.is_formatting) and not self.pre_editing and not self.block_change_signal:   
            # self.content_changed.emit(self)
            if not self.in_redo_undo:
                undo_steps = self.document().availableUndoSteps()
                new_steps = undo_steps - self.old_undo_steps
                joint_previous = new_steps == 0

                if not self.is_formatting:
                    change_from = self.change_from
                    removed = self.change_removed
                    added_text = ''
                    if self.input_method_from != -1:
                        added_text = self.input_method_text
                        change_from = self.input_method_from
                        removed = self.input_method_removed
                        self.input_method_from = -1
                        self.input_method_removed = 0

                    elif self.change_added > 0:
                        cursor = QTextCursor(self.document())
                        cursor.setPosition(change_from)
                        # QTextLayout range changes can make Qt report the
                        # terminal document separator as part of a full-range
                        # edit. It is not a selectable character.
                        selection_end = min(
                            change_from + self.change_added,
                            self.document().characterCount() - 1,
                        )
                        cursor.setPosition(
                            selection_end,
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                        added_text = cursor.selectedText()

                    if removed > 0 or added_text:
                        self.propagate_user_edited.emit(
                            change_from,
                            removed,
                            added_text,
                            joint_previous,
                        )
                self.change_added = 0
                self.change_removed = 0

                if new_steps > 0:
                    self.old_undo_steps = undo_steps
                    self.push_undo_stack.emit(new_steps, self.is_formatting)

        if not (self.hasFocus() and self.pre_editing):
            # Text edits can change glyph overhang, effect extents, and the
            # logical gradient envelope without changing the FontFormat.
            padding_changed = self._update_effect_padding()
            if self.fontformat.gradient_enabled and not padding_changed:
                self._refresh_gradient_geometry()
            if self.repaint_on_changed:
                if not self.repainting:
                    self.repaint_background()
            self.update()

    def repaint_background(self, render_scale: float = 1.0):
        return self.effect_renderer.repaint_background(render_scale)

    def set_export_effect_render(self, enabled: bool):
        self.effect_renderer.set_export_effect_render(enabled)

    @property
    def export_effect_error(self) -> Optional[Exception]:
        return self.effect_renderer.export_error

    def _update_effect_padding(self):
        return self.effect_renderer._update_effect_padding()

    def _refresh_gradient_geometry(self):
        self.effect_renderer._refresh_gradient_geometry()

    def get_text_gradient(self, fontformat=None, persistent=False):
        return self.effect_renderer.get_text_gradient(
            fontformat,
            persistent=persistent,
        )

    def docSizeChanged(self):
        # A padding change routes through setRect(), which synchronizes the
        # origin after updating the display rectangle. If padding is unchanged,
        # line reflow can still move the logical center, so sync it directly.
        if not self._update_effect_padding():
            self.geometry_controller.sync_origin()

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
        self.geometry_controller.bind_model()

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
            self.load_rich_text_html(blk.rich_text)
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
        self.repaint_background()

    def _effective_text_transform(self) -> TextTransformStack:
        return self.geometry_controller.effective()

    def _text_transform_is_neutral(self) -> bool:
        return self.geometry_controller.is_neutral()

    def itemChange(self, change, value):
        controller = getattr(self, 'geometry_controller', None)
        if controller is None:
            return super().itemChange(change, value)
        result = controller.item_change(change, value, super().itemChange)
        if (
            change
            == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged
            and self._order_badge_item is not None
        ):
            self._order_badge_item.set_selected(bool(value))
        elif (
            change
            == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and self._order_badge_item is not None
            and self._order_badge_item.parentItem() is not self
        ):
            self._sync_order_badge()
        return result

    def refresh_cache_policy(self) -> bool:
        """Apply the sole QGraphicsItem cache policy for live text items."""
        use_no_cache = (
            self.isEditing()
            or self.geometry_controller.requires_no_cache()
            or self.geometry_controller.has_layout_distortion()
            or self.effect_renderer.requires_no_item_cache()
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


    def set_text_transform(
        self,
        state: TextTransformStack,
        *,
        preview: bool = False,
    ) -> bool:
        effective_before = self.geometry_controller.effective()
        changed = self.geometry_controller.set(state, preview=preview)
        # A neutral Grid division edit changes controller topology without
        # changing the compiled text geometry or requiring a full repaint.
        if self.geometry_controller.effective() != effective_before:
            self.visual_geometry_changed.emit()
        return changed

    def clear_text_transform_preview(self) -> bool:
        effective_before = self.geometry_controller.effective()
        changed = self.geometry_controller.clear_preview()
        if self.geometry_controller.effective() != effective_before:
            self.visual_geometry_changed.emit()
        return changed

    def logical_unpadded_rect(self) -> QRectF:
        """Return the untransformed, effect-free block rect in item coordinates."""
        return self.geometry_controller.logical_rect()

    def visual_bounds_in_scene(self) -> QRectF:
        return self.geometry_controller.visual_bounds_in_scene()

    def rect(self) -> QRectF:
        return QRectF(
            self.pos(), self.geometry_controller.source_rect().size()
        )

    def logical_position(self) -> QPointF:
        """Return the persistent logical rectangle's absolute top-left."""
        return self.geometry_controller.logical_position()

    def _sync_block_xyxy(self) -> None:
        if self.blk is None:
            return
        self.blk._bounding_rect = self.absBoundingRect()
        self.blk.sync_xyxy_from_bounding_rect()

    def set_logical_position(self, point: QPointF) -> bool:
        """Move the logical top-left independently of paint padding."""
        changed = self.geometry_controller.set_logical_position(point)
        # A mouse drag reaches the undo command at its final position, so this
        # must sync even when the setter itself observes a zero delta.
        self._sync_block_xyxy()
        if changed:
            self.visual_geometry_changed.emit()
        return changed

    def startReshape(self):
        self._old_rect = self.absBoundingRect(qrect=True)
        self.reshaping = True
        # disable background repainting to avoid heavy redrawing in the whole process
        self.effect_renderer.clear_cached_surface()

    def endReshape(self):
        self.reshaped.emit(self)
        self.reshaping = False
        self.repaint_background()

    def setRect(
        self,
        rect: Union[List, QRectF],
        padding: bool = True,
        repaint: bool = True,
        update_blk_rect: bool = True,
        *,
        notify: bool = True,
    ) -> None:
        self.geometry_controller.set_rect(
            rect,
            padding=padding,
            repaint=repaint,
            update_blk_rect=update_blk_rect,
        )
        if update_blk_rect:
            self._sync_block_xyxy()
        if notify:
            self.visual_geometry_changed.emit()

    def documentSize(self):
        return self.layout.documentSize()

    def boundingRect(self) -> QRectF:
        controller = getattr(self, 'geometry_controller', None)
        base_rect = super().boundingRect()
        return base_rect if controller is None else controller.bounding_rect(base_rect)

    def padding(self) -> float:
        if self.layout is None:
            return 0.0
        return self.layout.effectPadding()

    def setPadding(self, p: float):
        p = max(0.0, float(p))
        _p = self.padding()
        if _p == p:
            return False
        abr = self.absBoundingRect(qrect=True)
        was_repainting = self.repainting
        self.repainting = True
        signals_were_blocked = self.layout.blockSignals(True)
        try:
            # Effect padding participates in boundingRect(); preserve the
            # logical rectangle while notifying the scene of the size change.
            self.prepareGeometryChange()
            self.layout.relayout_on_changed = False
            self.layout.setEffectPadding(p)
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
        return self.geometry_controller.absolute_rect(max_h, max_w, qrect)

    def shape(self):
        return self.geometry_controller.shape()

    def contains(self, point: QPointF) -> bool:
        return self.geometry_controller.contains(point)

    def inputMethodQuery(self, query):
        value = super().inputMethodQuery(query)
        if (
            query == Qt.InputMethodQuery.ImCursorRectangle
            and self.layout is not None
        ):
            cursor_rect = self.layout.source_cursor_rect(
                self.textCursor().position()
            )
            if cursor_rect is not None:
                value = cursor_rect
        mapper = self.geometry_controller.visual_mapper
        if mapper is None:
            return value
        if isinstance(value, (QPointF, QPoint)):
            return mapper.forward_point(QPointF(value))
        if isinstance(value, (QRectF, QRect)):
            return mapper.map_rect_path(QRectF(value)).boundingRect()
        return value

    def setScale(self, scale: float) -> None:
        previous = self.scale()
        if self._text_transform_is_neutral():
            self.setTransformOriginPoint(0, 0)
            super().setScale(scale)
            self.geometry_controller.sync_origin()
        else:
            with self.geometry_controller.update_transaction():
                super().setScale(scale)
        if self.scale() != previous:
            self.visual_geometry_changed.emit()

    def setRotation(self, angle: float) -> None:
        # Qt meta-property writes bypass this Python override; itemChange() is
        # the authoritative compensation and finalization path.
        previous = self.rotation()
        super().setRotation(angle)
        if self.rotation() != previous:
            self.visual_geometry_changed.emit()

    @property
    def angle(self) -> int:
        return self.blk.angle

    def setAngle(self, angle: int):
        with self.geometry_controller.update_transaction():
            # Preview/meta-property paths intentionally do not mutate the
            # model, so the live Qt property is the authoritative comparison.
            if self.rotation() != angle:
                self.setRotation(angle)
            self.blk.angle = angle
            self._sync_block_xyxy()

    def setVertical(self, vertical: bool) -> None:

        is_editing = self.isEditing()
        preserve_selection_direction = not self._text_transform_is_neutral()
        if is_editing:
            cursor = self.textCursor()
            cursor_pos = (cursor.position(), cursor.anchor().__pos__())
            insertion_format = (
                None
                if cursor.hasSelection()
                else QTextCharFormat(cursor.charFormat())
            )

        valid_layout = True
        doc = self.document()
        if self.layout is not None:
            effect_padding = self.layout.effectPadding()
            if isinstance(self.layout, VerticalTextDocumentLayout) == vertical:
                if self.fontformat is not None:
                    self.fontformat.vertical = vertical
                return
            self.layout.size_enlarged.disconnect(self.on_document_enlarged)
            self.layout.documentSizeChanged.disconnect(self.docSizeChanged)
        else:
            valid_layout = False
            effect_padding = 0.0
            doc.contentsChanged.connect(self.on_content_changed)
            doc.contentsChange.connect(self.on_content_changing)

        if valid_layout:
            rect = self.rect() if self.layout is not None else None
        
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        doc.documentLayout().blockSignals(True)

        controller = self.geometry_controller
        with controller.defer_compilation():
            block_change_signal = self.block_change_signal
            was_repainting = self.repainting
            was_relayout_on_changed = (
                self.layout.relayout_on_changed if valid_layout else None
            )
            self.block_change_signal = True
            self.repainting = True
            if valid_layout:
                # This layout is about to be replaced; formatting it again is
                # pure transition overhead.
                self.layout.relayout_on_changed = False
            try:
                set_document_letter_spacing_writing_mode(
                    doc,
                    vertical=vertical,
                    fallback=self.fontformat.letter_spacing,
                )
            finally:
                self.block_change_signal = block_change_signal
                self.repainting = was_repainting
                if valid_layout:
                    self.layout.relayout_on_changed = was_relayout_on_changed

            # QTextCursor formatting emits contentsChanged synchronously while
            # the old layout is still attached. Keep the writing-mode flag
            # aligned with that layout until the formatting transaction has
            # finished, otherwise effect repaint can enter the new vertical-only
            # stroke path through an old horizontal layout.
            if self.fontformat is not None:
                self.fontformat.vertical = vertical

            # Vertical alignment moves settled columns without touching the
            # document. Synchronize Qt's paragraph option only when a writing
            # mode switch is already rebuilding the layout.
            option = doc.defaultTextOption()
            option.setAlignment((
                Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignCenter,
                Qt.AlignmentFlag.AlignRight,
            )[int(self.fontformat.alignment)])
            if self.layout is None:
                doc.setDefaultTextOption(option)
            else:
                relayout_on_changed = self.layout.relayout_on_changed
                self.layout.relayout_on_changed = False
                try:
                    doc.setDefaultTextOption(option)
                finally:
                    self.layout.relayout_on_changed = relayout_on_changed

            # QTextDocument owns its document layout and can delete the old
            # QObject synchronously in setDocumentLayout(). The glyph renderer
            # is bound to that exact layout, so release it before crossing the
            # ownership boundary; initialize_layout() attaches a fresh one.
            controller.detach_layout_renderer()
            if vertical:
                layout = VerticalTextDocumentLayout(doc, self.fontformat)
            else:
                layout = HorizontalTextDocumentLayout(doc, self.fontformat)
            if valid_layout:
                # setDocumentLayout() immediately announces the existing
                # document. Defer that provisional zero-size layout and run
                # one settled pass after the final size and renderer are set.
                layout.relayout_on_changed = False
            self.layout = layout
            doc.setDocumentLayout(layout)
            layout.setEffectPadding(effect_padding)
            controller.initialize_layout(
                persistent_cache=controller.preview is None,
            )
            layout.size_enlarged.connect(self.on_document_enlarged)
            layout.documentSizeChanged.connect(self.docSizeChanged)

            if valid_layout:
                layout.setMaxSize(
                    rect.width(), rect.height(), relayout=False
                )
                layout.relayout_on_changed = True
                layout.reLayoutEverything()
                controller.refresh_compiled_geometry()
                self.repaint_background()
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
            if insertion_format is not None:
                spacing = letter_spacing_value(
                    insertion_format,
                    self.fontformat.letter_spacing,
                )
                insertion_format.setProperty(
                    AnnotationProperty.LETTER_SPACING,
                    spacing,
                )
                sync_native_ligature_shaping(
                    insertion_format,
                    vertical=vertical,
                    letter_spacing_fallback=spacing,
                )
                cursor.setCharFormat(insertion_format)
            self.setTextCursor(cursor)
        if self.fontformat.gradient_enabled:
            self._refresh_gradient_geometry()
        if valid_layout:
            self.visual_geometry_changed.emit()

    def setStandardVerticalRomanAlignment(
        self,
        enabled: bool,
        repaint_background: bool = True,
    ) -> None:
        """Set the item-wide Roman orientation used by vertical layout."""
        enabled = bool(enabled)
        if self.fontformat.standard_vertical_roman_alignment == enabled:
            return
        vertical_layout = isinstance(
            self.layout, VerticalTextDocumentLayout
        )
        if vertical_layout:
            # Orientation changes can expose ink outside the logical box.
            self.prepareGeometryChange()
        self.fontformat.standard_vertical_roman_alignment = enabled
        if not vertical_layout:
            return

        self.layout.reLayout()
        self.geometry_controller.flush_deferred_compilation()
        if repaint_background:
            self.repaint_background()
        self.update()
        self.visual_geometry_changed.emit()

    def refreshVerticalLayout(
        self,
        repaint_background: bool = True,
    ) -> None:
        """Refresh derived vertical geometry after a global setting change."""
        if not isinstance(self.layout, VerticalTextDocumentLayout):
            return

        # Punctuation and orientation changes can expose ink outside the box.
        self.prepareGeometryChange()
        self.layout.reLayout()
        self.geometry_controller.flush_deferred_compilation()
        if repaint_background:
            self.repaint_background()
        self.update()
        self.visual_geometry_changed.emit()

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            if self.hasFocus():
                self.change_from = from_
                self.change_removed = removed
                self.change_added = added

    def keyPressEvent(self, e: QKeyEvent) -> None:

        vertical_column_navigation = (
            self.isEditing()
            and isinstance(self.layout, VerticalTextDocumentLayout)
            and e.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right)
            and e.modifiers() in (
                Qt.KeyboardModifier.NoModifier,
                Qt.KeyboardModifier.ShiftModifier,
            )
        )
        if vertical_column_navigation:
            self._move_cursor_across_vertical_column(
                -1 if e.key() == Qt.Key.Key_Left else 1,
                e.modifiers() == Qt.KeyboardModifier.ShiftModifier,
            )
            e.accept()
            return
        self._vertical_navigation_y = None

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
            elif e.key() in (Qt.Key.Key_C, Qt.Key.Key_X):
                cursor = self.textCursor()
                if cursor.hasSelection():
                    self._copy_selected_text()
                    if e.key() == Qt.Key.Key_X:
                        cursor.removeSelectedText()
                        self.setTextCursor(cursor)
                e.accept()
                return
        elif e.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.redo_signal.emit()
                return
        elif e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            e.accept()
            cursor = self.textCursor()
            cursor.beginEditBlock()
            try:
                prepare_ruby_insertion(cursor, '\n')
                cursor.insertText('\n')
            finally:
                cursor.endEditBlock()
            self.setTextCursor(cursor)
            return
        elif e.text() and e.text().isprintable():
            e.accept()
            cursor = self.textCursor()
            cursor.beginEditBlock()
            try:
                prepare_ruby_insertion(cursor, e.text())
                cursor.insertText(e.text())
            finally:
                cursor.endEditBlock()
            self.setTextCursor(cursor)
            return
        super().keyPressEvent(e)
        self._emit_inline_format_changed()
        self._update_nonlinear_editing_ui()

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

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        self.geometry_controller.paint_item(
            painter,
            option,
            widget,
            super().paint,
        )
        self._paint_ui_guide(painter)

    def order_number(self) -> int:
        """Return the one-based order currently shown by the canvas guide."""
        if self._order_number_override is not None:
            return self._order_number_override
        return self.idx + 1

    @property
    def order_badge_visible(self) -> bool:
        return self._order_badge_visible

    def set_order_badge_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._order_badge_visible == visible:
            return
        self._order_badge_visible = visible
        self._sync_order_badge()

    def set_order_badge_layer(
        self,
        layer: Optional[QGraphicsItem],
    ) -> None:
        """Place the badge in a shared overlay, or rejoin it to this item."""
        badge = self._order_badge_item
        if badge is None:
            return
        parent = self if layer is None else layer
        if badge.parentItem() is parent:
            if layer is not None:
                self._sync_order_badge()
            return
        badge.hide()
        badge.setParentItem(parent)
        self.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges,
            layer is not None,
        )
        if layer is not None:
            self._sync_order_badge()

    def refresh_order_badge(self) -> None:
        """Refresh the badge after the item's persistent index changes."""
        self._sync_order_badge()

    def _sync_order_badge(self) -> None:
        badge = self._order_badge_item
        if badge is None:
            return
        visible = (
            not self._ui_guide_suppressed
            and not self.isEditing()
            and (
                self._order_badge_visible
                or self._order_number_override is not None
            )
        )
        if visible:
            badge.set_number(self.order_number())
            outline = self.geometry_controller.visual_outline_in_item()
            visible = not outline.isEmpty()
            if visible:
                anchor = outline.boundingRect().topLeft()
                parent = badge.parentItem()
                badge.setPos(
                    anchor
                    if parent is self or parent is None
                    else self.mapToItem(parent, anchor)
                )
        badge.setVisible(visible)

    def set_order_number_override(self, order_number: Optional[int]) -> None:
        """Set a transient order preview without changing project state."""
        if order_number is not None:
            order_number = max(1, int(order_number))
        if self._order_number_override == order_number:
            return
        self._order_number_override = order_number
        self._sync_order_badge()

    def _paint_ui_guide(self, painter: QPainter) -> None:
        """Paint selection and block guides outside cached effect surfaces."""
        if (
            self._ui_guide_suppressed
            or self.isEditing()
        ):
            return
        selected = self.isSelected()
        draw_rect = self.draw_rect and not self.under_ctrl
        if not selected and not draw_rect:
            return
        outline = self.geometry_controller.visual_outline_in_item()
        if outline.isEmpty():
            return
        painter.save()
        try:
            pen = QPen(
                TEXTRECT_SELECTED_COLOR if selected else TEXTRECT_SHOW_COLOR,
                3.5 if selected else 3.0,
                Qt.PenStyle.DashLine if selected else Qt.PenStyle.SolidLine,
            )
            pen.setCosmetic(True)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.setPen(pen)
            painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
            painter.drawPath(outline)
        finally:
            painter.restore()

    def set_ui_guide_suppressed(self, suppressed: bool) -> None:
        suppressed = bool(suppressed)
        if self._ui_guide_suppressed == suppressed:
            return
        self._ui_guide_suppressed = suppressed
        self._sync_order_badge()
        self.update()


    def startEdit(self, pos: QPointF = None) -> None:
        self.pre_editing = False
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.refresh_cache_policy()
        self._sync_order_badge()
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
        self._sync_order_badge()
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
        self._vertical_navigation_y = None
        if not self.isEditing():
            self.startEdit(pos=event.pos())
        else:
            super().mouseDoubleClickEvent(event)
        self._emit_inline_format_changed()
        self._update_nonlinear_editing_ui()
        
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)  
        if self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction:
            self._emit_inline_format_changed()
            self._update_nonlinear_editing_ui()
        else:
            self.moving.emit(self)

    def _copy_selected_text(self) -> None:
        cursor = self.textCursor()
        if not cursor.hasSelection():
            return
        QApplication.clipboard().setMimeData(
            create_rich_text_mime(
                cursor,
                line_spacing_fallback=self.fontformat.line_spacing,
                line_spacing_type_fallback=self.fontformat.line_spacing_type,
            )
        )

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:
        if not self.isEditing():
            return super().contextMenuEvent(event)
        event.accept()
        self.show_editing_context_menu(event.screenPos())

    def show_editing_context_menu(
        self,
        screen_pos: QPoint,
        parent: Optional[QWidget] = None,
    ) -> None:
        cursor = self.textCursor()
        has_selection = cursor.hasSelection()
        menu, quick_insert_actions = create_text_edit_context_menu(
            parent,
            has_selection=has_selection,
            can_undo=self.document().isUndoAvailable(),
            can_redo=self.document().isRedoAvailable(),
        )

        action = menu.exec(screen_pos)
        operation = action.data() if action is not None else None
        if action in quick_insert_actions:
            self.insert_plain_text_at_cursor(operation)
        elif operation == 'undo':
            self.undo_signal.emit()
        elif operation == 'redo':
            self.redo_signal.emit()
        elif operation == 'cut':
            self._copy_selected_text()
            cursor.removeSelectedText()
            self.setTextCursor(cursor)
        elif operation == 'copy':
            self._copy_selected_text()
        elif operation == 'paste':
            self.pasted.emit(self.idx)
        elif operation == 'delete':
            cursor.removeSelectedText()
            self.setTextCursor(cursor)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._vertical_navigation_y = None
        if event.button() == Qt.MouseButton.LeftButton:
            if self.isEditing():
                self.geometry_controller.begin_input_mapping()
            self._old_pos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        result = super().mousePressEvent(event)
        self._emit_inline_format_changed()
        self._update_nonlinear_editing_ui()
        return result

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            # The manager owns the multi-item movement snapshot. Finish every
            # press/release interaction so a click without movement cannot keep
            # the clicked items alive after their scene is replaced.
            self.move_interaction_finished.emit()
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.geometry_controller.end_input_mapping()
            self._emit_inline_format_changed()
            self._update_nonlinear_editing_ui()

    def dragEnterEvent(self, event) -> None:
        self.geometry_controller.begin_input_mapping()
        super().dragEnterEvent(event)

    def dragLeaveEvent(self, event) -> None:
        try:
            super().dragLeaveEvent(event)
        finally:
            self.geometry_controller.end_input_mapping()

    def dropEvent(self, event) -> None:
        try:
            super().dropEvent(event)
        finally:
            self.geometry_controller.end_input_mapping()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.hover_enter.emit(self.idx)
        return super().hoverEnterEvent(event)

    def toHtml(self) -> str:
        html = super().toHtml()
        tables = table_pattern.findall(html)
        if tables:
            _, td = td_pattern.findall(html)[0]
            html = tables[0] + td + '</body></html>'

        html = html.replace('>\n<', '><')
        return to_rich_text_html(
            self.document(),
            html,
            line_spacing_fallback=self.fontformat.line_spacing,
            line_spacing_type_fallback=self.fontformat.line_spacing_type,
        )

    def load_rich_text_html(self, html: str) -> None:
        """Restore ordinary Qt HTML plus application-owned annotations."""
        block_change_signal = self.block_change_signal
        self.block_change_signal = True
        try:
            load_rich_text_html(
                self.document(),
                html,
                letter_spacing_fallback=self.fontformat.letter_spacing,
                vertical=self.fontformat.vertical,
            )
        finally:
            self.block_change_signal = block_change_signal

    def insert_from_mime_data(self, mime: QMimeData) -> bool:
        cursor = self.textCursor()
        inserted = insert_rich_text_mime(
            cursor,
            mime,
            vertical=self.fontformat.vertical,
        )
        if inserted:
            self.setTextCursor(cursor)
        return inserted

    def insert_plain_text_at_cursor(self, text: str) -> None:
        """Insert clipboard text with Ruby's boundary inheritance rules."""
        cursor = self.textCursor()
        cursor.beginEditBlock()
        try:
            prepare_ruby_insertion(cursor, text)
            cursor.insertText(text)
        finally:
            cursor.endEditBlock()
        self.setTextCursor(cursor)

    def get_fontformat(self) -> FontFormat:
        fmt = self._active_char_format()
        font = fmt.font()
        color = fmt.foreground().color()
        fontformat = self.fontformat.deepcopy()
        fontformat.frgb = [color.red(), color.green(), color.blue()]
        fontformat.font_weight = font_weight_from_qt(font.weight())
        fontformat.font_family = font_family_for_project(font.family())
        if self.isEditing():
            fontformat.font_size = pt2px(font.pointSizeF())
        else:
            fontformat.font_size = self.minFontSize()
        fontformat.underline = font.underline()
        fontformat.italic = font.italic()
        fontformat.letter_spacing = self.letter_spacing_value()
        if self.document().isEmpty():
            for axis in (
                LIGATURE_COMMON,
                LIGATURE_DISCRETIONARY,
                LIGATURE_CONTEXTUAL,
            ):
                setattr(
                    fontformat,
                    f'ligature_{axis}',
                    self.ligature_axis_value(axis),
                )
            fontformat.oldstyle_nums = self.oldstyle_nums_value()
        (
            fontformat.line_spacing,
            fontformat.line_spacing_type,
        ) = self.line_spacing_values()
        return fontformat

    def set_fontformat(self, ffmat: FontFormat, set_char_format=False, set_stroke_width=True, set_effect=True):
        self.repainting = True
        if self.fontformat.vertical != ffmat.vertical:
            self.setVertical(ffmat.vertical)
        if (
            self.fontformat.standard_vertical_roman_alignment
            != ffmat.standard_vertical_roman_alignment
        ):
            self.setStandardVerticalRomanAlignment(
                ffmat.standard_vertical_roman_alignment,
                repaint_background=False,
            )

        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        format = cursor.charFormat()
        font = qfont_with_family(
            self.document().defaultFont(),
            ffmat.font_family,
        )
        font.setPointSizeF(ffmat.size_pt)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias)

        fweight = QFont.Weight(
            font_weight_to_qt(ffmat.font_weight, qt6=QT6)
        )
        font.setWeight(fweight)

        self.document().setDefaultFont(font)
        format.setFont(font)
        if ffmat.gradient_enabled:
            gradient = self.get_text_gradient(ffmat, persistent=True)
            format.setForeground(gradient)
        else:
            format.setForeground(QColor(*ffmat.foreground_color()))
        format.setFontWeight(fweight)
        format.setFontItalic(ffmat.italic)
        format.setFontUnderline(ffmat.underline)
        format.setProperty(
            AnnotationProperty.LETTER_SPACING,
            ffmat.letter_spacing,
        )
        set_ligature_axes(
            format,
            {
                axis: getattr(ffmat, f'ligature_{axis}')
                for axis in (
                    LIGATURE_COMMON,
                    LIGATURE_DISCRETIONARY,
                    LIGATURE_CONTEXTUAL,
                )
            },
            vertical=ffmat.vertical,
        )
        set_oldstyle_nums(format, ffmat.oldstyle_nums)
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
        
        self.setAlignment(ffmat.alignment, repaint_background=False)
        
        if set_char_format:
            self._set_line_spacing_pair(
                ffmat.line_spacing,
                ffmat.line_spacing_type,
                whole_item=True,
            )
        else:
            # Rich HTML may already own different paragraph pairs. Update only
            # the compatibility default unless this is a whole-style apply.
            fallback_changed = (
                self.layout.line_spacing != ffmat.line_spacing
                or self.layout.linespacing_type != ffmat.line_spacing_type
            )
            self.layout.line_spacing = ffmat.line_spacing
            self.layout.linespacing_type = ffmat.line_spacing_type
            if fallback_changed:
                self.layout.reLayout()
        
        # Preserve gradient properties
        self.fontformat.gradient_enabled = ffmat.gradient_enabled
        self.fontformat.gradient_start_color = ffmat.gradient_start_color
        self.fontformat.gradient_end_color = ffmat.gradient_end_color
        self.fontformat.gradient_angle = ffmat.gradient_angle
        self.fontformat.gradient_size = ffmat.gradient_size
        
        # Apply while the canonical model still contains the previous
        # transform; merging first would skip live geometry recompilation.
        self.set_text_transform(ffmat.text_transform)
        self.fontformat.merge(ffmat)

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
                    # Restore the original active end as well as the range.
                    # Selection direction controls Qt's insertion format.
                    cursor.setPosition(pos2)
                    cursor.setPosition(
                        pos1, QTextCursor.MoveMode.KeepAnchor
                    )
                self.setTextCursor(cursor)

        cursor.endEditBlock()
        self.geometry_controller.flush_deferred_compilation()
        if repaint_background:
            self.repaint_background()

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
            doc.setDefaultFont(qfont_with_family(doc.defaultFont(), value))

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
                    cfmt.setFont(qfont_with_family(cfmt.font(), value))
                    cfmt.setFontUnderline(under_line)
                    cursor.setPosition(pos1)
                    cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                    cursor.setCharFormat(cfmt)
                it += 1
            block = block.next()

        cfmt = cursor.charFormat()
        cfmt.setFont(qfont_with_family(cfmt.font(), value))
        self.set_cursor_cfmt(cursor, cfmt)

    def setFontWeight(
        self,
        value: FontWeight,
        repaint_background: bool = True,
        set_selected: bool = False,
        restore_cursor: bool = False,
    ) -> None:
        cursor, after_kwargs = self._before_set_ffmt(set_selected, restore_cursor)
        cfmt = QTextCharFormat()
        cfmt.setFontWeight(
            QFont.Weight(font_weight_to_qt(value, qt6=QT6))
        )
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

    def _active_char_format(self) -> QTextCharFormat:
        """Return a direction-independent format for panel synchronization."""
        cursor = self.textCursor()
        if cursor.hasSelection():
            # charFormat() samples immediately before the active cursor end.
            # Use the selection end so forward and backward selections agree.
            cursor.setPosition(cursor.selectionEnd())
        return cursor.charFormat()

    def emphasis_values(self) -> tuple[str, str]:
        return emphasis_values(self._active_char_format())

    def letter_spacing_value(self) -> float:
        return letter_spacing_value(
            self._active_char_format(),
            self.fontformat.letter_spacing,
        )

    def ligature_axis_value(self, axis: str) -> str:
        return ligature_axis_value(self._active_char_format(), axis)

    def oldstyle_nums_value(self) -> str:
        return oldstyle_nums_value(self._active_char_format())

    def _active_block_format(self) -> QTextBlockFormat:
        cursor = self.textCursor()
        if cursor.hasSelection():
            block = self.document().findBlock(cursor.selectionEnd() - 1)
        else:
            block = cursor.block()
        return block.blockFormat()

    def line_spacing_values(self) -> tuple[float, LineSpacingType]:
        """Return the item default or active paragraph spacing pair."""
        if not self.isEditing():
            return (
                self.fontformat.line_spacing,
                LineSpacingType(self.fontformat.line_spacing_type),
            )
        return line_spacing_values(
            self._active_block_format(),
            self.fontformat.line_spacing,
            self.fontformat.line_spacing_type,
        )

    def _apply_text_format(
        self,
        apply_format: Callable[[QTextCursor], None],
        *,
        select_document: bool = False,
    ) -> None:
        """Run one selection/caret formatting transaction."""
        cursor = self.textCursor()
        restore_cursor = not self.isEditing() or select_document
        cursor_position = cursor.position()
        cursor_anchor = cursor.anchor()
        if not self.isEditing() or select_document:
            cursor.select(QTextCursor.SelectionType.Document)
        self.is_formatting = True
        try:
            cursor.beginEditBlock()
            try:
                apply_format(cursor)
            finally:
                cursor.endEditBlock()
                if restore_cursor:
                    cursor.setPosition(cursor_anchor)
                    if cursor_position != cursor_anchor:
                        cursor.setPosition(
                            cursor_position,
                            QTextCursor.MoveMode.KeepAnchor,
                        )
                self.setTextCursor(cursor)
            self.geometry_controller.flush_deferred_compilation()
        finally:
            self.is_formatting = False

    def setEmphasis(self, style: str, position: str) -> None:
        """Apply emphasis to a selection or the active insertion format."""
        self._apply_text_format(
            lambda cursor: apply_emphasis(cursor, style, position)
        )

    def setLigatureAxis(self, axis: str, state: str) -> None:
        """Apply one ligature axis to the active text range."""
        self._apply_text_format(
            lambda cursor: apply_ligature_axis(
                cursor,
                axis,
                state,
                vertical=self.fontformat.vertical,
            )
        )

    def setOldstyleNums(self, state: str) -> None:
        """Apply oldstyle figures to the active text range."""
        self._apply_text_format(
            lambda cursor: apply_oldstyle_nums(cursor, state)
        )

    def tate_chu_yoko_enabled(self) -> bool:
        value, _group_id = text_combine_upright_values(
            self._active_char_format()
        )
        return value == TEXT_COMBINE_ALL

    def setTateChuYoko(self, enabled: bool) -> None:
        """Combine one selected run, or change the insertion format."""
        self._apply_text_format(
            lambda cursor: apply_text_combine_upright(
                cursor,
                enabled,
                vertical=self.fontformat.vertical,
            )
        )

    def ruby_editor_values(
        self,
    ) -> tuple[str, str, str, bool]:
        """Return the current Ruby editor values for the Advanced panel."""
        cursor = self.textCursor()
        container = ruby_container_for_cursor(cursor)
        if container is None:
            return (
                'group',
                '',
                'over',
                bool(ruby_containers_intersecting_cursor(cursor)),
            )
        text = (
            container.units[0].text
            if container.ruby_type == 'group'
            else ' '.join(unit.text for unit in container.units)
        )
        return (
            container.ruby_type,
            text,
            container.position,
            True,
        )

    def setRuby(self, ruby_type: str, text: str, position: str) -> None:
        """Apply or update Ruby at the current text selection/caret."""
        cursor = self.textCursor()
        self.is_formatting = True
        try:
            apply_ruby(cursor, ruby_type, text, position)
            self.setTextCursor(cursor)
            self.geometry_controller.flush_deferred_compilation()
        finally:
            self.is_formatting = False

    def removeRuby(self) -> bool:
        """Remove Ruby containers intersecting the active cursor."""
        cursor = self.textCursor()
        self.is_formatting = True
        try:
            removed = remove_ruby(cursor)
            if removed:
                self.setTextCursor(cursor)
                self.geometry_controller.flush_deferred_compilation()
            return removed
        finally:
            self.is_formatting = False

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


    def _set_line_spacing_pair(
        self,
        value: float,
        spacing_type: int,
        *,
        whole_item: bool = False,
    ) -> None:
        canonical_value, canonical_type = validated_line_spacing(
            value, spacing_type
        )
        update_item_default = whole_item or not self.isEditing()
        if update_item_default:
            self.old_ffmt_values = {
                'line_spacing': self.fontformat.line_spacing,
                'line_spacing_type': self.fontformat.line_spacing_type,
            }
            self.fontformat.line_spacing = canonical_value
            self.fontformat.line_spacing_type = int(canonical_type)
            # Paragraph formats drive settled layout; these remain the
            # compatibility defaults for unformatted legacy paragraphs.
            self.layout.line_spacing = canonical_value
            self.layout.linespacing_type = canonical_type
        previous_block_change = self.block_change_signal
        if whole_item:
            # ApplyFontformatCommand already owns this whole-item transaction.
            self.block_change_signal = True
        try:
            self._apply_text_format(
                lambda cursor: apply_line_spacing(
                    cursor, canonical_value, canonical_type
                ),
                select_document=whole_item,
            )
        finally:
            self.block_change_signal = previous_block_change
            self.old_ffmt_values = None

    def setLineSpacing(self, value: float) -> None:
        _current_value, spacing_type = self.line_spacing_values()
        self._set_line_spacing_pair(value, spacing_type)

    def setLineSpacingType(self, value: int) -> None:
        line_spacing, _current_type = self.line_spacing_values()
        self._set_line_spacing_pair(line_spacing, value)

    def setLetterSpacing(self, value: float) -> None:
        canonical_value = canonical_letter_spacing(value)
        if canonical_value is None:
            raise ValueError(f'unsupported letter spacing: {value!r}')
        value = canonical_value
        update_item_default = not self.isEditing()
        height_growth = 0.0
        if isinstance(self.layout, VerticalTextDocumentLayout):
            cursor = self.textCursor()
            if update_item_default:
                selection_start = 0
                selection_end = self.document().characterCount() - 1
            elif cursor.hasSelection():
                selection_start = cursor.selectionStart()
                selection_end = cursor.selectionEnd()
            else:
                selection_start = selection_end = cursor.position()
            height_growth = self.layout.spacing_change_height_growth(
                selection_start,
                selection_end,
                value,
            )
        if height_growth > 1e-6:
            source_rect = self.geometry_controller.source_rect()
            self.set_size(
                source_rect.width(),
                source_rect.height() + height_growth,
                set_layout_maxsize=True,
            )
        if update_item_default:
            self.old_ffmt_values = {
                'letter_spacing': self.fontformat.letter_spacing
            }
            self.fontformat.letter_spacing = value
            self.layout.letter_spacing = value
        try:
            self._apply_text_format(
                lambda cursor: apply_letter_spacing(
                    cursor,
                    value,
                    vertical=self.fontformat.vertical,
                )
            )
        finally:
            self.old_ffmt_values = None

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
        self._update_effect_padding()
        if (
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
        self._update_effect_padding()
        if clip_size:
            self.squeezeBoundingRect(cond_on_alignment=True)

        self._after_set_ffmt(cursor, repaint_background, restore_cursor, **after_kwargs)

    def _set_alignment_state(self, value: int) -> bool:
        value = int(value)
        state_changed = self.fontformat.alignment != value
        vertical_layout = isinstance(self.layout, VerticalTextDocumentLayout)
        if vertical_layout:
            if not state_changed:
                return False
            self.prepareGeometryChange()
            self.fontformat.alignment = value
            self.layout.apply_alignment()
            return True

        qt_align_flag = (
            Qt.AlignmentFlag.AlignLeft,
            Qt.AlignmentFlag.AlignCenter,
            Qt.AlignmentFlag.AlignRight,
        )[value]
        doc = self.document()
        option = doc.defaultTextOption()
        option_changed = option.alignment() != qt_align_flag
        if not option_changed and not state_changed:
            return False

        # Alignment can move slanted-glyph ink beyond the logical rectangle in
        # either writing mode, so notify the scene before changing the layout.
        self.prepareGeometryChange()
        self.fontformat.alignment = value
        option.setAlignment(qt_align_flag)
        doc.setDefaultTextOption(option)
        return True

    def setAlignment(self, value, restore_cursor=False, repaint_background=True, *args, **kwargs):
        if not self._set_alignment_state(value):
            return

        self.geometry_controller.invalidate_surface_cache()
        if repaint_background:
            self.repaint_background()
        else:
            # A caller may defer the expensive effect redraw, but visible ink
            # still needs correct bounds after slanted glyphs move.
            self._update_effect_padding()
        self.update()

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
            if repaint:
                self.repaint_background()

    def set_size(
        self,
        w: float,
        h: float,
        set_layout_maxsize=False,
        set_blk_size=True,
    ) -> None:
        self.geometry_controller.resize(
            w,
            h,
            set_layout_maxsize=set_layout_maxsize,
            set_blk_size=set_blk_size,
        )
        self.visual_geometry_changed.emit()
