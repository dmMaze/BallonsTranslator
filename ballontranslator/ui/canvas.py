import numpy as np
from typing import List, Union, Tuple

from qtpy.QtWidgets import QMenu, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsItem, QScrollBar, QGraphicsPixmapItem, QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent, QRubberBand
from qtpy.QtCore import Qt, QDateTime, QRectF, QPointF, QPoint, Signal, QSizeF, QEvent
from qtpy.QtGui import QPixmap, QHideEvent, QKeyEvent, QWheelEvent, QResizeEvent, QPainter, QPen, QPainterPath, QCursor

try:
    from qtpy.QtWidgets import QUndoStack, QUndoCommand
except:
    from qtpy.QtGui import QUndoStack, QUndoCommand

from .misc import ndarray2pixmap
from .imgtrans_proj import ProjImgTrans
from .textitem import TextBlkItem, TextBlock
from .texteditshapecontrol import TextBlkShapeControl
from .stylewidgets import FadeLabel
from .image_edit import ImageEditMode, DrawingLayer, StrokeImgItem
from .page_search_widget import PageSearchWidget
from . import constants as C

CANVAS_SCALE_MAX = 3.0
CANVAS_SCALE_MIN = 0.1
CANVAS_SCALE_SPEED = 0.1

class CustomGV(QGraphicsView):
    ctrl_pressed = False
    scale_up_signal = Signal()
    scale_down_signal = Signal()
    view_resized = Signal()
    hide_canvas = Signal()
    ctrl_released = Signal()
    canvas: QGraphicsScene = None

    def wheelEvent(self, event : QWheelEvent) -> None:
        # qgraphicsview always scroll content according to wheelevent
        # which is not desired when scaling img
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_up_signal.emit()
            else:
                self.scale_down_signal.emit()
            return
        return super().wheelEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_pressed = False
            self.ctrl_released.emit()
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Control:
            self.ctrl_pressed = True

        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_V:
                # self.ctrlv_pressed.emit(e)
                if self.canvas.handle_ctrlv():
                    e.accept()
                    return
            if e.key() == Qt.Key.Key_C:
                if self.canvas.handle_ctrlc():
                    e.accept()
                    return

        return super().keyPressEvent(e)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.view_resized.emit()
        return super().resizeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self.hide_canvas.emit()
        return super().hideEvent(event)

    def enterEvent(self, event: QEvent) -> None:
        self.setFocus()
        return super().enterEvent(event)


class Canvas(QGraphicsScene):

    scalefactor_changed = Signal()
    end_create_textblock = Signal(QRectF)
    paste2selected_textitems = Signal()
    end_create_rect = Signal(QRectF, int)
    finish_painting = Signal(StrokeImgItem)
    finish_erasing = Signal(StrokeImgItem)
    delete_textblks = Signal(int)
    copy_textblks = Signal(QPointF)
    paste_textblks = Signal(QPointF)
    format_textblks = Signal()
    layout_textblks = Signal()

    run_blktrans = Signal(int)

    begin_scale_tool = Signal(QPointF)
    scale_tool = Signal(QPointF)
    end_scale_tool = Signal()
    canvas_undostack_changed = Signal()
    
    imgtrans_proj: ProjImgTrans = None
    painting_pen = QPen()
    painting_shape = 0
    erasing_pen = QPen()
    image_edit_mode = ImageEditMode.NONE
    alt_pressed = False
    scale_tool_mode = False

    projstate_unsaved = False
    proj_savestate_changed = Signal(bool)
    textstack_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.
        self.mask_transparency = 0
        self.original_transparency = 0
        self.textblock_mode = False
        self.creating_textblock = False
        self.create_block_origin: QPointF = None
        self.editing_textblkitem: TextBlkItem = None

        self.gv = CustomGV(self)
        self.gv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.gv.scale_down_signal.connect(self.scaleDown)
        self.gv.scale_up_signal.connect(self.scaleUp)
        self.gv.view_resized.connect(self.onViewResized)
        self.gv.hide_canvas.connect(self.on_hide_canvas)
        self.gv.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.gv.canvas = self
        
        if not C.FLAG_QT6:
            # mitigate https://bugreports.qt.io/browse/QTBUG-93417
            # produce blurred result, saving imgs remain unaffected
            self.gv.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.search_widget = PageSearchWidget(self.gv)
        self.search_widget.hide()
        
        self.ctrl_relesed = self.gv.ctrl_released
        self.vscroll_bar = self.gv.verticalScrollBar()
        self.hscroll_bar = self.gv.horizontalScrollBar()
        # self.default_cursor = self.gv.cursor()
        self.rubber_band = self.addWidget(QRubberBand(QRubberBand.Shape.Rectangle))
        self.rubber_band.hide()
        self.rubber_band_origin = None

        self.draw_undo_stack = QUndoStack(self)
        self.draw_undo_stack.indexChanged.connect(self.on_drawstack_changed)
        self.text_undo_stack = QUndoStack(self)
        self.text_undo_stack.indexChanged.connect(self.on_textstack_changed)
        self.saved_drawundo_step = 0
        self.saved_textundo_step = 0

        self.scaleFactorLabel = FadeLabel(self.gv)
        self.scaleFactorLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scaleFactorLabel.setText('100%')
        self.scaleFactorLabel.gv = self.gv

        self.txtblkShapeControl = TextBlkShapeControl(self.gv)
        
        self.baseLayer = QGraphicsRectItem()
        pen = QPen()
        pen.setColor(Qt.GlobalColor.transparent)
        self.baseLayer.setPen(pen)
        
        self.imgLayer = QGraphicsPixmapItem()
        self.imgLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.inpaintLayer = QGraphicsPixmapItem()
        self.inpaintLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.maskLayer = QGraphicsPixmapItem()
        self.drawingLayer = DrawingLayer()
        self.drawingLayer.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.textLayer = QGraphicsPixmapItem()
        
        self.addItem(self.baseLayer)
        self.inpaintLayer.setParentItem(self.baseLayer)
        self.imgLayer.setParentItem(self.baseLayer)
        self.maskLayer.setParentItem(self.baseLayer)
        self.drawingLayer.setParentItem(self.baseLayer)
        self.textLayer.setParentItem(self.baseLayer)
        self.txtblkShapeControl.setParentItem(self.textLayer)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)
        self.selectionChanged.connect(self.on_selection_changed)     
        self.stroke_img_item: StrokeImgItem = None
        self.erase_img_key = None

        self.editor_index = 0 # 0: drawing 1: text editor
        self.mid_btn_pressed = False
        self.pan_initial_pos = QPoint(0, 0)

        self.saved_textundo_step = 0
        self.saved_drawundo_step = 0

        self.clipboard_blks: List[TextBlock] = []

    def textEditMode(self) -> bool:
        return self.editor_index == 1

    def drawMode(self) -> bool:
        return self.editor_index == 0

    def scaleUp(self):
        self.scaleImage(1 + CANVAS_SCALE_SPEED)

    def scaleDown(self):
        self.scaleImage(1 - CANVAS_SCALE_SPEED)

    def setImageLayer(self):
        if not self.imgtrans_proj.img_valid:
            return
        pixmap = ndarray2pixmap(self.imgtrans_proj.img_array)
        self.imgLayer.setPixmap(pixmap)
        
        im_rect = self.imgLayer.pixmap().rect()
        self.baseLayer.setRect(QRectF(im_rect))
        if im_rect != self.sceneRect():
            self.setSceneRect(0, 0, im_rect.width(), im_rect.height())
        self.scaleImage(1)
        self.imgLayer.setOpacity(self.original_transparency)

    def setMaskLayer(self):
        if not self.imgtrans_proj.mask_valid:
            return
        pixmap = ndarray2pixmap(self.imgtrans_proj.mask_array)
        self.maskLayer.setPixmap(pixmap)
        self.maskLayer.setOpacity(self.mask_transparency)

    def setMaskTransparency(self, transparency: float):
        self.maskLayer.setOpacity(transparency)
        self.mask_transparency = transparency
        if transparency == 0:
            self.maskLayer.setVisible(False)
        else:
            self.maskLayer.setVisible(True)

    def setOriginalTransparency(self, transparency: float):
        self.imgLayer.setOpacity(transparency)
        self.original_transparency = transparency
        if transparency == 0:
            self.imgLayer.hide()
        else:
            self.imgLayer.show()

    def adjustScrollBar(self, scrollBar: QScrollBar, factor: float):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

    def scaleImage(self, factor: float):
        if not self.gv.isVisible() or not self.imgtrans_proj.img_valid:
            return
        s_f = self.scale_factor * factor
        s_f = np.clip(s_f, CANVAS_SCALE_MIN, CANVAS_SCALE_MAX)

        sbr = self.imgLayer.sceneBoundingRect()
        self.old_size = sbr.size()
        scale_changed = self.scale_factor != s_f
        self.scale_factor = s_f
        self.baseLayer.setScale(self.scale_factor)
        self.txtblkShapeControl.updateScale(self.scale_factor)

        if scale_changed:
            self.adjustScrollBar(self.gv.horizontalScrollBar(), factor)
            self.adjustScrollBar(self.gv.verticalScrollBar(), factor)
            self.scalefactor_changed.emit()
        self.setSceneRect(0, 0, self.imgLayer.sceneBoundingRect().width(), self.imgLayer.sceneBoundingRect().height())

    def onViewResized(self):
        gv_w, gv_h = self.gv.geometry().width(), self.gv.geometry().height()

        x = gv_w - self.scaleFactorLabel.width()
        y = gv_h - self.scaleFactorLabel.height()
        pos_new = (QPointF(x, y) / 2).toPoint()
        if self.scaleFactorLabel.pos() != pos_new:
            self.scaleFactorLabel.move(pos_new)
        
        x = gv_w - self.search_widget.width()
        pos = self.search_widget.pos()
        pos.setX(x-30)
        self.search_widget.move(pos)
        
    def onScaleFactorChanged(self):
        self.scaleFactorLabel.setText(f'{self.scale_factor*100:2.0f}%')
        self.scaleFactorLabel.raise_()
        self.scaleFactorLabel.startFadeAnimation()

    def on_selection_changed(self):
        if self.txtblkShapeControl.isVisible():
            blk_item = self.txtblkShapeControl.blk_item
            if blk_item is not None and blk_item.isEditing():
                blk_item.endEdit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.editing_textblkitem is not None:
            return super().keyPressEvent(event)
        if event.key() == Qt.Key.Key_Alt:
            self.alt_pressed = True
        return super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Alt:
            self.alt_pressed = False
        return super().keyReleaseEvent(event)

    def addStrokeImageItem(self, pos: QPointF, pen: QPen, erasing: bool = False):
        if self.stroke_img_item is not None:
            self.stroke_img_item.startNewPoint(pos)
        else:
            self.stroke_img_item = StrokeImgItem(pen, pos, self.imgLayer.pixmap().size(), shape=self.painting_shape)
            if not erasing:
                self.stroke_img_item.setParentItem(self.baseLayer)
            else:
                self.erase_img_key = str(QDateTime.currentMSecsSinceEpoch())
                compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
                self.drawingLayer.addQImage(0, 0, self.stroke_img_item._img, compose_mode, self.erase_img_key)

    def startCreateTextblock(self, pos: QPointF, hide_control: bool = False):
        pos = pos / self.scale_factor
        self.creating_textblock = True
        self.create_block_origin = pos
        self.gv.setCursor(Qt.CursorShape.CrossCursor)
        self.txtblkShapeControl.setBlkItem(None)
        self.txtblkShapeControl.setPos(0, 0)
        self.txtblkShapeControl.setRotation(0)
        self.txtblkShapeControl.setRect(QRectF(pos, QSizeF(1, 1)))
        if hide_control:
            self.txtblkShapeControl.hideControls()
        self.txtblkShapeControl.show()

    def endCreateTextblock(self, btn=0):
        self.creating_textblock = False
        self.gv.setCursor(Qt.CursorShape.ArrowCursor)
        self.txtblkShapeControl.hide()
        if self.creating_normal_rect:
            self.end_create_rect.emit(self.txtblkShapeControl.rect(), btn)
            self.txtblkShapeControl.showControls()
        else:
            self.end_create_textblock.emit(self.txtblkShapeControl.rect())

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.mid_btn_pressed:
            new_pos = event.screenPos()
            delta_pos = new_pos - self.pan_initial_pos
            self.pan_initial_pos = new_pos
            self.hscroll_bar.setValue(int(self.hscroll_bar.value() - delta_pos.x()))
            self.vscroll_bar.setValue(int(self.vscroll_bar.value() - delta_pos.y()))
            
        elif self.creating_textblock:
            self.txtblkShapeControl.setRect(QRectF(self.create_block_origin, event.scenePos() / self.scale_factor).normalized())
        
        elif self.stroke_img_item is not None:
            if self.stroke_img_item.is_painting:
                pos = self.imgLayer.mapFromScene(event.scenePos())
                if self.erase_img_key is None:
                    # painting
                    self.stroke_img_item.lineTo(pos)
                else:
                    rect = self.stroke_img_item.lineTo(pos, update=False)
                    if rect is not None:
                        self.drawingLayer.update(rect)
        
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        
        elif self.rubber_band.isVisible() and self.rubber_band_origin is not None:
            self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, event.scenePos()).normalized())
            sel_path = QPainterPath(self.rubber_band_origin)
            sel_path.addRect(self.rubber_band.geometry())
            if C.FLAG_QT6:
                self.setSelectionArea(sel_path, deviceTransform=self.gv.viewportTransform())
            else:
                self.setSelectionArea(sel_path, Qt.ItemSelectionMode.IntersectsItemBoundingRect, self.gv.viewportTransform())
        
        return super().mouseMoveEvent(event)

    def selected_text_items(self) -> List[TextBlkItem]:
        sel_titem = []
        selitems = self.selectedItems()
        for sel in selitems:
            if isinstance(sel, TextBlkItem):
                sel_titem.append(sel)
        return sel_titem

    def handle_ctrlv(self) -> bool:
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False

        if len(self.selected_text_items()) > 0:
            self.paste2selected_textitems.emit()
        else:
            self.paste_textblks.emit(self.scene_cursor_pos())

        return True

    def handle_ctrlc(self):
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False

        if len(self.selected_text_items()) > 0:
            self.copy_textblks.emit(self.scene_cursor_pos())
        
        return True

    def scene_cursor_pos(self):
        origin = self.gv.mapFromGlobal(QCursor.pos())
        return self.gv.mapToScene(origin)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = True
            self.pan_initial_pos = event.screenPos()
            return

        if self.textblock_mode and len(self.selectedItems()) == 0:
            if btn == Qt.MouseButton.RightButton:
                return self.startCreateTextblock(event.scenePos())
        elif self.creating_normal_rect:
            if btn == Qt.MouseButton.RightButton or btn == Qt.MouseButton.LeftButton:
                return self.startCreateTextblock(event.scenePos(), hide_control=True)

        elif btn == Qt.MouseButton.LeftButton:
            # user is drawing using the pen/inpainting tool
            if self.alt_pressed:
                self.scale_tool_mode = True
                self.begin_scale_tool.emit(event.scenePos())
            elif self.painting:
                self.addStrokeImageItem(self.imgLayer.mapFromScene(event.scenePos()), self.painting_pen)

        elif btn == Qt.MouseButton.RightButton:
            # user is drawing using eraser
            if self.painting:
                erasing = self.image_edit_mode == ImageEditMode.PenTool
                self.addStrokeImageItem(self.imgLayer.mapFromScene(event.scenePos()), self.erasing_pen, erasing)
            else:   # rubber band selection
                self.rubber_band_origin = event.scenePos()
                self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, self.rubber_band_origin).normalized())
                self.rubber_band.show()
                self.rubber_band.setZValue(1)

        return super().mousePressEvent(event)

    @property
    def creating_normal_rect(self):
        return self.image_edit_mode == ImageEditMode.RectTool and self.editor_index == 0

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = False
        elif self.creating_textblock:
            btn = 0 if btn == Qt.MouseButton.LeftButton else 1
            return self.endCreateTextblock(btn=btn)
        elif btn == Qt.MouseButton.RightButton:
            if self.stroke_img_item is not None:
                self.finish_erasing.emit(self.stroke_img_item)
            if self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band_origin = None
        elif btn == Qt.MouseButton.LeftButton:
            if self.stroke_img_item is not None:
                self.finish_painting.emit(self.stroke_img_item)
            elif self.scale_tool_mode:
                self.scale_tool_mode = False
                self.end_scale_tool.emit()
        return super().mouseReleaseEvent(event)

    def updateCanvas(self):
        self.editing_textblkitem = None
        self.stroke_img_item = None
        self.erase_img_key = None
        self.txtblkShapeControl.setBlkItem(None)
        self.mid_btn_pressed = False
        self.search_widget.reInitialize()

        self.clearSelection()
        self.setProjSaveState(False)
        self.setImageLayer()
        self.setInpaintLayer()
        self.setMaskLayer()
        self.setDrawingLayer()

    def setInpaintLayer(self):
        if not self.imgtrans_proj.inpainted_valid:
            return
        pixmap = ndarray2pixmap(self.imgtrans_proj.inpainted_array)
        self.inpaintLayer.setPixmap(pixmap)

        pixmap.fill(Qt.GlobalColor.transparent)
        self.textLayer.setPixmap(pixmap)

    def setDrawingLayer(self, img: Union[QPixmap, np.ndarray] = None):
        
        self.drawingLayer.clearAllDrawings()

        if not self.imgtrans_proj.img_valid:
            return
        if img is None:
            drawing_map = self.imgLayer.pixmap().copy()
            drawing_map.fill(Qt.GlobalColor.transparent)
        elif not isinstance(img, QPixmap):
            drawing_map = ndarray2pixmap(img)
        else:
            drawing_map = img
        self.drawingLayer.setPixmap(drawing_map)

    def setPaintMode(self, painting: bool):
        if painting:
            self.editing_textblkitem = None
            self.textblock_mode = False
            self.maskLayer.setVisible(True)
        else:
            self.maskLayer.setVisible(False)
            # self.gv.setCursor(self.default_cursor)
            self.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.image_edit_mode = ImageEditMode.NONE

    @property
    def painting(self):
        return self.image_edit_mode == ImageEditMode.PenTool or self.image_edit_mode == ImageEditMode.InpaintTool

    def setMaskTransparencyBySlider(self, slider_value: int):
        self.setMaskTransparency(slider_value / 100)

    def setOriginalTransparencyBySlider(self, slider_value: int):
        self.setOriginalTransparency(slider_value / 100)

    def setTextBlockMode(self, mode: bool):
        self.textblock_mode = mode

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        if self.textEditMode():
            menu = QMenu()
            copy_act = menu.addAction(self.tr("Copy"))
            paste_act = menu.addAction(self.tr("Paste"))
            delete_act = menu.addAction(self.tr("Delete"))
            delete_recover_act = menu.addAction(self.tr("Delete and Recover removed text"))
            menu.addSeparator()
            format_act = menu.addAction(self.tr("Apply font formatting"))
            layout_act = menu.addAction(self.tr("Auto layout"))
            menu.addSeparator()
            translate_act = menu.addAction(self.tr("translate"))
            ocr_act = menu.addAction(self.tr("OCR"))
            ocr_translate_act = menu.addAction(self.tr("OCR and translate"))
            ocr_translate_inpaint_act = menu.addAction(self.tr("OCR, translate and inpaint"))

            rst = menu.exec_(event.screenPos())
            
            if rst == delete_act:
                self.delete_textblks.emit(0)
            elif rst == delete_recover_act:
                self.delete_textblks.emit(1)
            elif rst == copy_act:
                self.copy_textblks.emit(event.scenePos())
            elif rst == paste_act:
                self.paste_textblks.emit(event.scenePos())
            elif rst == format_act:
                self.format_textblks.emit()
            elif rst == layout_act:
                self.layout_textblks.emit()
            elif rst == translate_act:
                self.run_blktrans.emit(-1)
            elif rst == ocr_act:
                self.run_blktrans.emit(0)
            elif rst == ocr_translate_act:
                self.run_blktrans.emit(1)
            elif rst == ocr_translate_inpaint_act:
                self.run_blktrans.emit(2)
    
    def on_hide_canvas(self):
        self.clear_states()

    def on_activation_changed(self):
        self.clear_states()

    def clear_states(self):
        self.alt_pressed = False
        self.scale_tool_mode = False
        self.creating_textblock = False
        self.create_block_origin = None
        self.editing_textblkitem = None
        self.gv.ctrl_pressed = False
        if self.stroke_img_item is not None:
            self.removeItem(self.stroke_img_item)

    def setProjSaveState(self, un_saved: bool):
        if un_saved == self.projstate_unsaved:
            return
        else:
            self.projstate_unsaved = un_saved
            self.proj_savestate_changed.emit(un_saved)

    def removeItem(self, item: QGraphicsItem) -> None:
        super().removeItem(item)
        if isinstance(item, StrokeImgItem):
            item.setParentItem(None)
            self.stroke_img_item = None
            self.erase_img_key = None

    def get_active_undostack(self) -> QUndoStack:
        if self.textEditMode():
            return self.text_undo_stack
        elif self.drawMode():
            return self.draw_undo_stack
        return None

    def push_undo_command(self, command: QUndoCommand):
        undo_stack = self.get_active_undostack()
        if undo_stack is not None:
            undo_stack.push(command)

    def on_drawstack_changed(self, index: int):
        if index != self.saved_drawundo_step:
            self.setProjSaveState(True)
        elif self.text_undo_stack.index() == self.saved_textundo_step:
            self.setProjSaveState(False)

    def on_textstack_changed(self, index: int):
        if index != self.saved_textundo_step:
            self.setProjSaveState(True)
        elif self.draw_undo_stack.index() == self.saved_drawundo_step:
            self.setProjSaveState(False)
        self.textstack_changed.emit()

    def redo_textedit(self):
        self.text_undo_stack.redo()

    def undo_textedit(self):
        self.text_undo_stack.undo()

    def redo(self):
        undo_stack = self.get_active_undostack()
        if undo_stack is not None:
            undo_stack.redo()
            if undo_stack == self.text_undo_stack:
                self.txtblkShapeControl.updateBoundingRect()

    def undo(self):
        undo_stack = self.get_active_undostack()
        if undo_stack is not None:
            undo_stack.undo()
            if undo_stack == self.text_undo_stack:
                self.txtblkShapeControl.updateBoundingRect()

    def clear_undostack(self, update_saved_step=False):
        if update_saved_step:
            self.saved_drawundo_step = 0
            self.saved_textundo_step = 0
        self.draw_undo_stack.clear()
        self.text_undo_stack.clear()

    def clear_text_stack(self):
        self.text_undo_stack.clear()

    def clear_draw_stack(self):
        self.draw_undo_stack.clear()

    def update_saved_undostep(self):
        self.saved_drawundo_step = self.draw_undo_stack.index()
        self.saved_textundo_step = self.text_undo_stack.index()

    def text_change_unsaved(self) -> bool:
        return self.saved_textundo_step != self.text_undo_stack.index()

    def draw_change_unsaved(self) -> bool:
        return self.saved_drawundo_step != self.draw_undo_stack.index()

    def prepareClose(self):
        self.blockSignals(True)
        self.disconnect()
        self.text_undo_stack.disconnect()
        self.draw_undo_stack.disconnect()

