import numpy as np
from typing import List, Union, Tuple

from qtpy.QtWidgets import QMenu, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsItem, QScrollBar, QGraphicsPixmapItem, QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent, QRubberBand
from qtpy.QtCore import Qt, QRect, QRectF, QPointF, QPoint, Signal, QSizeF, QObject, QEvent
from qtpy.QtGui import QPixmap, QHideEvent, QKeyEvent, QWheelEvent, QResizeEvent, QKeySequence, QPainter, QPen, QPainterPath

try:
    from qtpy.QtWidgets import QUndoStack

except:
    from qtpy.QtGui import QUndoStack

from .misc import ndarray2pixmap, ProjImgTrans
from .textitem import TextBlkItem, TextBlock
from .texteditshapecontrol import TextBlkShapeControl
from .stylewidgets import FadeLabel
from .image_edit import StrokeItem, StrokeItem, ImageEditMode

CANVAS_SCALE_MAX = 3.0
CANVAS_SCALE_MIN = 0.1
CANVAS_SCALE_SPEED = 0.1

class CustomGV(QGraphicsView):
    do_scale = True
    ctrl_pressed = False
    scale_up_signal = Signal()
    scale_down_signal = Signal()
    view_resized = Signal()
    hide_canvas = Signal()
    ctrl_released = Signal()

    def wheelEvent(self, event : QWheelEvent) -> None:
        # qgraphicsview always scroll content according to wheelevent
        # which is not desired when scaling img
        if self.ctrl_pressed:
            if self.do_scale:
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Control:
            self.ctrl_pressed = True
        return super().keyPressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.view_resized.emit()
        return super().resizeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self.leftbtn_pressed = False
        self.do_scale = True
        self.ctrl_pressed = False
        self.hide_canvas.emit()
        return super().hideEvent(event)

    def enterEvent(self, event: QEvent) -> None:
        self.setFocus()
        return super().enterEvent(event)


class Canvas(QGraphicsScene):

    scalefactor_changed = Signal()
    end_create_textblock = Signal(QRectF)
    end_create_rect = Signal(QRectF, int)
    finish_painting = Signal(StrokeItem)
    finish_erasing = Signal(StrokeItem)
    delete_textblks = Signal()
    format_textblks = Signal()
    layout_textblks = Signal()

    begin_scale_tool = Signal(QPointF)
    scale_tool = Signal(QPointF)
    end_scale_tool = Signal()
    canvas_undostack_changed = Signal()
    
    imgtrans_proj: ProjImgTrans = None
    painting_pen = QPen()
    image_edit_mode = ImageEditMode.NONE
    alt_pressed = False
    scale_tool_mode = False

    projstate_unsaved = False
    proj_savestate_changed = Signal(bool)

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
        self.gv.setRenderHint(QPainter.Antialiasing)
        self.ctrl_relesed = self.gv.ctrl_released

        self.default_cursor = self.gv.cursor()
        self.rubber_band = self.addWidget(QRubberBand(QRubberBand.Shape.Rectangle))
        self.rubber_band.hide()
        self.rubber_band_origin = None

        self.undoStack = QUndoStack(self)
        self.undoStack.indexChanged.connect(self.on_undostack_changed)
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
        self.maskLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.drawingLayer = QGraphicsPixmapItem()
        self.drawingLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.textLayer = QGraphicsPixmapItem()
        self.textLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        self.addItem(self.baseLayer)
        self.inpaintLayer.setParentItem(self.baseLayer)
        self.imgLayer.setParentItem(self.baseLayer)
        self.maskLayer.setParentItem(self.baseLayer)
        self.drawingLayer.setParentItem(self.baseLayer)
        self.textLayer.setParentItem(self.baseLayer)
        self.txtblkShapeControl.setParentItem(self.textLayer)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)
        self.selectionChanged.connect(self.on_selection_changed)     
        self.stroke_path_item: StrokeItem = None

        self.editor_index = 0 # 0: drawing 1: text editor

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
        self.scale_factor = s_f
        self.baseLayer.setScale(self.scale_factor)
        self.txtblkShapeControl.updateScale(self.scale_factor)

        self.adjustScrollBar(self.gv.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.gv.verticalScrollBar(), factor)
        self.setSceneRect(0, 0, self.imgLayer.sceneBoundingRect().width(), self.imgLayer.sceneBoundingRect().height())
        self.scalefactor_changed.emit()

    def onViewResized(self):
        x = self.gv.geometry().width() - self.scaleFactorLabel.width()
        y = self.gv.geometry().height() - self.scaleFactorLabel.height()
        pos_new = (QPointF(x, y) / 2).toPoint()
        if self.scaleFactorLabel.pos() != pos_new:
            self.scaleFactorLabel.move(pos_new)
        
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
        if event == QKeySequence.Undo:
            self.undoStack.undo()
            self.txtblkShapeControl.updateBoundingRect()
        elif event == QKeySequence.Redo:
            self.undoStack.redo()
            self.txtblkShapeControl.updateBoundingRect()
        elif event.key() == Qt.Key.Key_Alt:
            self.alt_pressed = True
        return super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Alt:
            self.alt_pressed = False
        return super().keyReleaseEvent(event)

    def addStrokeItem(self, item: StrokeItem):
        self.addItem(item)
        item.setPen(self.painting_pen)
        item.setParentItem(self.drawingLayer)

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
        if self.creating_textblock:
            self.txtblkShapeControl.setRect(QRectF(self.create_block_origin, event.scenePos() / self.scale_factor).normalized())
        elif self.stroke_path_item is not None:
            self.stroke_path_item.addNewPoint(self.imgLayer.mapFromScene(event.scenePos()))
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        elif self.rubber_band.isVisible() and self.rubber_band_origin is not None:
            self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, event.scenePos()).normalized())
            sel_path = QPainterPath(self.rubber_band_origin)
            sel_path.addRect(self.rubber_band.geometry())
            self.setSelectionArea(sel_path, Qt.ItemSelectionMode.IntersectsItemBoundingRect, self.gv.viewportTransform())
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.textblock_mode and len(self.selectedItems()) == 0:
            if event.button() == Qt.MouseButton.RightButton:
                return self.startCreateTextblock(event.scenePos())
        elif self.creating_normal_rect:
            return self.startCreateTextblock(event.scenePos(), hide_control=True)

        elif event.button() == Qt.MouseButton.LeftButton:
            # user is drawing using the pen/inpainting tool
            if self.alt_pressed:
                self.scale_tool_mode = True
                self.begin_scale_tool.emit(event.scenePos())
            elif self.painting and self.stroke_path_item is None:
                self.stroke_path_item = StrokeItem(self.imgLayer.mapFromScene(event.scenePos()))
                self.addStrokeItem(self.stroke_path_item)

        elif event.button() == Qt.MouseButton.RightButton:
            # user is drawing using eraser
            if self.painting and self.stroke_path_item is None:
                self.stroke_path_item = StrokeItem(self.imgLayer.mapFromScene(event.scenePos()))
                self.addStrokeItem(self.stroke_path_item)
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
        if self.creating_textblock:
            btn = 0 if event.button() == Qt.MouseButton.LeftButton else 1
            return self.endCreateTextblock(btn=btn)
        elif event.button() == Qt.MouseButton.RightButton:
            if self.stroke_path_item is not None:
                self.finish_erasing.emit(self.stroke_path_item)
            if self.rubber_band.isVisible():
                self.rubber_band.hide()
                self.rubber_band_origin = None
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.stroke_path_item is not None:
                self.finish_painting.emit(self.stroke_path_item)
            elif self.scale_tool_mode:
                self.scale_tool_mode = False
                self.end_scale_tool.emit()
            self.stroke_path_item = None
        return super().mouseReleaseEvent(event)

    def updateCanvas(self):
        self.clearSelection()
        self.setProjSaveState(False)
        self.editing_textblkitem = None
        self.txtblkShapeControl.setBlkItem(None)
        self.stroke_path_item
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
        
        ditems = self.get_drawings(visible=False)
        for item in ditems:
            self.removeItem(item)

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
            self.gv.setCursor(self.default_cursor)
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
        if len(self.selectedItems()) != 0:
            menu = QMenu()
            delete_act = menu.addAction(self.tr("Delete"))
            format_act = menu.addAction(self.tr("Apply font formatting"))
            layout_act = menu.addAction(self.tr("Auto layout"))
            rst = menu.exec_(event.screenPos())
            if rst == delete_act:
                self.delete_textblks.emit()
            elif rst == format_act:
                self.format_textblks.emit()
            elif rst == layout_act:
                self.layout_textblks.emit()
    
    def on_hide_canvas(self):
        self.alt_pressed = False
        self.scale_tool_mode = False
        self.creating_textblock = False
        self.create_block_origin = None
        self.editing_textblkitem = None
        if self.stroke_path_item is not None:
            self.removeItem(self.stroke_path_item)
            self.stroke_path_item = None

    def on_undostack_changed(self):
        if self.undoStack.count() != 0:
            self.setProjSaveState(True)

    def setProjSaveState(self, un_saved: bool):
        if un_saved == self.projstate_unsaved:
            return
        else:
            self.projstate_unsaved = un_saved
            self.proj_savestate_changed.emit(un_saved)

    def removeItem(self, item: QGraphicsItem) -> None:
        if item == self.stroke_path_item:
            self.stroke_path_item = None
        return super().removeItem(item)

    def get_drawings(self, visible=False) -> List[QGraphicsItem]:
        ditems = self.drawingLayer.childItems()
        if visible:
            ditems = [item for item in ditems if item.isVisible()]
        return ditems