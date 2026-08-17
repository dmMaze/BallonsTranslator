import numpy as np
from typing import List, Optional, Tuple, Union

from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QGraphicsTextItem, QStyleOptionGraphicsItem, QGraphicsSceneMouseEvent
from qtpy.QtCore import Qt, QRect, QRectF, QPoint, QPointF, Signal
from qtpy.QtGui import (QKeyEvent, QFont, QTextCursor,
                       QInputMethodEvent, QPainter, QColor, QTextCharFormat,
                       QBrush, QPen, QFontMetrics)

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly
from ballontranslator.utils.fontformat import (
    FontFormat,
    TextTransformStack,
    pt2px,
)
from ..misc import td_pattern, table_pattern
from .layout import VerticalTextDocumentLayout, HorizontalTextDocumentLayout
from .effect_renderer import TextEffectRenderer
from .geometry import TextItemGeometryController

TEXTRECT_SHOW_COLOR = QColor(30, 147, 229, 170)
TEXTRECT_SELECTED_COLOR = QColor(248, 64, 147, 170)


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
    propagate_user_edited = Signal(int, str, bool)
    visual_geometry_changed = Signal()

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
        self.old_ffmt_values = None
        
        self.idx = idx
        self._reorder_seq = -1   # reading-order badge override during path reorder
        
        self.stroke_qcolor = QColor(0, 0, 0)
        self._old_pos = QPointF()
        self._old_rect = QRectF()
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
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.geometry_controller.finish_initialization()

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
        # Preedit text and attributes live in QTextLayout, so they need an
        # explicit surface invalidation even when the document revision does
        # not change. The next paint is cached until another IME event.
        self.geometry_controller.invalidate_surface_cache()
        self._update_nonlinear_editing_ui()

    def setTextCursor(self, cursor: QTextCursor) -> None:
        super().setTextCursor(cursor)
        self._update_nonlinear_editing_ui()

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
                    added_text = ''
                    if self.input_method_from != -1:
                        added_text = self.input_method_text
                        change_from = self.input_method_from
                        self.input_method_from = -1

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

                    self.propagate_user_edited.emit(change_from, added_text, joint_previous)
                    self.change_added = 0

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
        self.repaint_background()

    def _effective_text_transform(self) -> TextTransformStack:
        return self.geometry_controller.effective()

    def _text_transform_is_neutral(self) -> bool:
        return self.geometry_controller.is_neutral()

    def itemChange(self, change, value):
        controller = getattr(self, 'geometry_controller', None)
        if controller is None:
            return super().itemChange(change, value)
        return controller.item_change(change, value, super().itemChange)

    def refresh_cache_policy(self) -> bool:
        """Apply the sole QGraphicsItem cache policy for live text items."""
        use_no_cache = (
            self.isEditing()
            or self.geometry_controller.requires_no_cache()
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

    def set_logical_position(self, point: QPointF) -> bool:
        """Move the logical top-left independently of paint padding."""
        changed = self.geometry_controller.set_logical_position(point)
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

    def setVertical(self, vertical: bool):

        is_editing = self.isEditing()
        preserve_selection_direction = not self._text_transform_is_neutral()
        if is_editing:
            cursor = self.textCursor()
            cursor_pos = (cursor.position(), cursor.anchor().__pos__())

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

        # Preserve BASE writing-mode letter-spacing semantics. Glyph slant is
        # layout-only and must not rewrite the document's spacing behavior.
        reset_spacing_val = 1 if vertical else self.fontformat.letter_spacing
        cursor = QTextCursor(doc)
        cursor.joinPreviousEditBlock()
        char_fmt = QTextCharFormat()
        char_fmt.setFontLetterSpacingType(QFont.SpacingType.PercentageSpacing)
        char_fmt.setFontLetterSpacing(reset_spacing_val * 100)
        cursor.select(QTextCursor.SelectionType.Document)
        controller = self.geometry_controller
        with controller.defer_compilation():
            self.set_cursor_cfmt(cursor, char_fmt, True)
            cursor.endEditBlock()

            # QTextCursor formatting emits contentsChanged synchronously while
            # the old layout is still attached. Keep the writing-mode flag
            # aligned with that layout until the formatting transaction has
            # finished, otherwise effect repaint can enter the new vertical-only
            # stroke path through an old horizontal layout.
            if self.fontformat is not None:
                self.fontformat.vertical = vertical

            if vertical:
                layout = VerticalTextDocumentLayout(doc, self.fontformat)
            else:
                layout = HorizontalTextDocumentLayout(doc, self.fontformat)
            self.layout = layout
            doc.setDocumentLayout(layout)
            layout.setEffectPadding(effect_padding)
            controller.initialize_layout(
                persistent_cache=controller.preview is None,
            )
            layout.size_enlarged.connect(self.on_document_enlarged)
            layout.documentSizeChanged.connect(self.docSizeChanged)

            if valid_layout:
                layout.setMaxSize(rect.width(), rect.height())
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
            self.setTextCursor(cursor)
        if self.fontformat.gradient_enabled:
            self._refresh_gradient_geometry()
        if valid_layout:
            self.visual_geometry_changed.emit()

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
        super().keyPressEvent(e)
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
        self._paint_seq_badge(painter)

    def _paint_seq_badge(self, painter: QPainter) -> None:
        """Draw the reading-order number badge at the content top-left corner."""
        if self._ui_guide_suppressed or not pcfg.show_seq_badge:
            return
        outline = self.geometry_controller.visual_outline_in_item()
        content_rect = (
            outline.boundingRect() if not outline.isEmpty() else self.boundingRect()
        )
        top_level = self.topLevelItem()
        scale = top_level.scale() if top_level is not None else 1.0
        scale = scale if scale > 0 else 1.0
        font_size = max(6, int(11 / scale))
        font = QFont()
        font.setBold(True)
        font.setPixelSize(font_size)
        if self._reorder_seq >= 0:
            seq_text = str(self._reorder_seq + 1)
        else:
            seq_text = str(self.idx + 1)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(seq_text) + 8
        text_h = fm.height() + 4
        badge_rect = QRectF(
            content_rect.x(),
            content_rect.y(),
            text_w,
            text_h,
        )
        painter.save()
        if self.isSelected():
            from ballontranslator.ui.custom_widget.helper import themeColor

            bg = QColor(themeColor())
            bg.setAlpha(200)
        else:
            bg = QColor(0, 0, 0, 170)
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 3, 3)
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(font)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, seq_text)
        painter.restore()

    def _paint_ui_guide(self, painter: QPainter) -> None:
        """Paint selection/block guides outside cached effect surfaces."""
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
        self.update()


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
        self._update_nonlinear_editing_ui()
        
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        super().mouseMoveEvent(event)  
        if self.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction:
            self._update_nonlinear_editing_ui()
        else:
            self.moving.emit(self)

    # QT 5.15.x causing segmentation fault 
    def contextMenuEvent(self, event):
        return super().contextMenuEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.isEditing():
                self.geometry_controller.begin_input_mapping()
            self._old_pos = self.pos()
            self.leftbutton_pressed.emit(self.idx)
        result = super().mousePressEvent(event)
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
        # persistent TextBlock owner. Canonical transform state must win when
        # producing a save/undo format snapshot.
        fontformat.text_transform = self.blk.fontformat.text_transform
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
                    cursor.setPosition(min(pos1, pos2))
                    cursor.setPosition(max(pos1, pos2), QTextCursor.MoveMode.KeepAnchor)
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


    def setLineSpacing(self, value: float, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing = value
        self.layout.setLineSpacing(value)
        self.geometry_controller.flush_deferred_compilation()
        if repaint_background:
            self.repaint_background()
            self.update()
        self.is_formatting = False

    def setLineSpacingType(self, value: int, repaint_background: bool = True, set_selected: bool = False, restore_cursor: bool = False):
        self.is_formatting = True
        self.fontformat.line_spacing_type = value
        self.layout.setLineSpacingType(value)
        self.geometry_controller.flush_deferred_compilation()
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

        self.geometry_controller.flush_deferred_compilation()
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
            if repaint:
                self.repaint_background()

    def set_size(
        self,
        w: float,
        h: float,
        set_layout_maxsize=False,
        set_blk_size=True,
    ):
        self.geometry_controller.resize(
            w,
            h,
            set_layout_maxsize=set_layout_maxsize,
            set_blk_size=set_blk_size,
        )
        self.visual_geometry_changed.emit()
