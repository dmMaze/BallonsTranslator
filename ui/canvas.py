
import numpy as np
from PyQt5.QtWidgets import QMenu, QGraphicsPathItem, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QGraphicsItem, QWidget, QGraphicsSceneHoverEvent, QLabel, QSizePolicy, QScrollBar, QListView, QGraphicsSceneWheelEvent, QGraphicsTextItem, QGraphicsPixmapItem, QStyle, QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent, QUndoGroup, QUndoStack, QUndoView
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, QPoint, pyqtSignal, QSizeF, QObject, QEvent
from PyQt5.QtGui import QPixmap, QHideEvent, QMouseEvent, QKeyEvent, QWheelEvent, QResizeEvent, QKeySequence, QPainter, QTextFrame, QTransform, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetrics, QTextOption, QFocusEvent, QPen, QColor, QTextFormat, QPainterPath

from typing import List, Union, Tuple
from .misc import ndarray2pixmap, pixmap2ndarray, qrgb2bgr, ProjImgTrans

from .textitem import TextBlkItem, TextBlock
from .texteditshapecontrol import TextBlkShapeControl
from .stylewidgets import FadeLabel
from .image_edit import StrokeItem, PenStrokeItem, PenStrokeCommand, ImageEditMode

CANVAS_MODE_TEXTEDIT = 1
CANVAS_SCALE_MAX = 3.0
CANVAS_SCALE_MIN = 0.1
CANVAS_SCALE_SPEED = 0.1

PROJ_NAME = 'proj.json'
TST_HTML = '<html><p>xxxxxx</p><p style="font-size:15pt">AAAAA</p><p>、测试、</p><br /><p style="font-size:20pt">测测测测测试</p></html>'

class CustomGV(QGraphicsView):
    do_scale = True
    ctrl_pressed = False
    scale_up_signal = pyqtSignal()
    scale_down_signal = pyqtSignal()
    view_resized = pyqtSignal()
    hide_canvas = pyqtSignal()
    ctrl_released = pyqtSignal()

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

    def resizeTool(self) -> bool:
        return self.alt_pressed and self.leftbtn_pressed


class Canvas(QGraphicsScene):

    scalefactor_changed = pyqtSignal()
    end_create_textblock = pyqtSignal(QRectF)
    delete_textblks = pyqtSignal()
    finish_painting = pyqtSignal(StrokeItem)
    finish_erasing = pyqtSignal(StrokeItem)

    begin_scale_tool = pyqtSignal(QPointF)
    scale_tool = pyqtSignal(QPointF)
    end_scale_tool = pyqtSignal()
    canvas_undostack_changed = pyqtSignal()
    
    imgtrans_proj: ProjImgTrans = None
    painting = False
    painting_pen = QPen()
    image_edit_mode = None
    alt_pressed = False
    scale_tool_mode = False

    projstate_unsaved = False
    proj_savestate_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.
        self.mask_transparency = 0
        self.original_transparency = 0
        self.textblock_mode = False
        self.creating_textblock = False
        self.create_block_origin: QPointF = None
        self.editing_textblkitem: TextBlkItem = None
        self.hovering_textblkitem: TextBlkItem = None

        self.gv = CustomGV(self)
        self.gv.setAlignment(Qt.AlignCenter)
        self.gv.setDragMode(QGraphicsView.ScrollHandDrag)
        self.gv.scale_down_signal.connect(self.scaleDown)
        self.gv.scale_up_signal.connect(self.scaleUp)
        self.gv.view_resized.connect(self.onViewResized)
        self.gv.hide_canvas.connect(self.on_hide_canvas)
        self.gv.setRenderHint(QPainter.Antialiasing)
        self.ctrl_relesed = self.gv.ctrl_released

        self.default_cursor = self.gv.cursor()

        self.undoStack = QUndoStack(self)
        self.undoStack.indexChanged.connect(self.on_undostack_changed)
        self.scaleFactorLabel = FadeLabel(self.gv)
        self.scaleFactorLabel.setAlignment(Qt.AlignCenter)
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
        
        # self.addItem(self.imgLayer)
        self.addItem(self.baseLayer)
        self.inpaintLayer.setParentItem(self.baseLayer)
        self.imgLayer.setParentItem(self.baseLayer)
        self.maskLayer.setParentItem(self.baseLayer)
        self.drawingLayer.setParentItem(self.baseLayer)
        self.addItem(self.txtblkShapeControl)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)        
        self.stroke_path_item: StrokeItem = None

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
        s_f = self.scale_factor * factor
        s_f = np.clip(s_f, CANVAS_SCALE_MIN, CANVAS_SCALE_MAX)

        sbr = self.imgLayer.sceneBoundingRect()
        self.old_size = sbr.size()
        self.scale_factor = s_f
        # self.imgLayer.setScale(self.scale_factor)
        self.baseLayer.setScale(self.scale_factor)

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

    def startCreateTextblock(self, pos: QPointF):
        self.creating_textblock = True
        self.create_block_origin = pos
        self.gv.viewport().setCursor(Qt.CrossCursor)
        self.txtblkShapeControl.setBlkItem(None)
        self.txtblkShapeControl.setPos(0, 0)
        self.txtblkShapeControl.setRotation(0)
        self.txtblkShapeControl.setRect(QRectF(pos, QSizeF(1, 1)))
        self.txtblkShapeControl.show()

    def endCreateTextblock(self):
        self.creating_textblock = False
        self.gv.viewport().setCursor(Qt.ArrowCursor)
        self.txtblkShapeControl.hide()
        self.end_create_textblock.emit(self.txtblkShapeControl.rect())

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.creating_textblock:
            epos, opos = event.scenePos(), self.create_block_origin
            w, h = epos.x() - opos.x(), epos.y() - opos.y()
            if w < 0:
                x = epos.x()
                w = -w
            else:
                x = opos.x()
            if h < 0:
                y = epos.y()
                h = -h
            else:
                y = opos.y()
            self.txtblkShapeControl.setRect(QRectF(x, y, w, h))
        elif self.stroke_path_item is not None:
            self.stroke_path_item.addNewPoint(self.imgLayer.mapFromScene(event.scenePos()))
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.textblock_mode:
            if event.button() == Qt.RightButton:
                if self.hovering_textblkitem is None:
                    return self.startCreateTextblock(event.scenePos())
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.alt_pressed:
                self.scale_tool_mode = True
                self.begin_scale_tool.emit(event.scenePos())
            elif self.painting:
                self.stroke_path_item = PenStrokeItem(self.imgLayer.mapFromScene(event.scenePos()))
                self.addStrokeItem(self.stroke_path_item)
                
        elif event.button() == Qt.MouseButton.RightButton:
            if self.painting:
                self.stroke_path_item = PenStrokeItem(self.imgLayer.mapFromScene(event.scenePos()))
                self.addStrokeItem(self.stroke_path_item)

        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.RightButton:
            if self.creating_textblock:
                return self.endCreateTextblock()
            elif self.stroke_path_item is not None:
                self.finish_erasing.emit(self.stroke_path_item)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.stroke_path_item is not None:
                self.finish_painting.emit(self.stroke_path_item)
            elif self.scale_tool_mode:
                self.scale_tool_mode = False
                self.end_scale_tool.emit()
            self.stroke_path_item = None
        return super().mouseReleaseEvent(event)

    def updateCanvas(self):
        self.setProjSaveState(False)
        self.editing_textblkitem = None
        self.hovering_textblkitem = None
        self.txtblkShapeControl.setBlkItem(None)
        self.setImageLayer()
        self.setInpaintLayer()
        self.setMaskLayer()
        self.setDrawingLayer()

    def setInpaintLayer(self):
        if not self.imgtrans_proj.inpainted_valid:
            return
        pixmap = ndarray2pixmap(self.imgtrans_proj.inpainted_array)
        self.inpaintLayer.setPixmap(pixmap)

    def setDrawingLayer(self, img: Union[QPixmap, np.ndarray] = None):
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
            self.hovering_textblkitem = None
            self.textblock_mode = False
            self.maskLayer.setVisible(True)
        else:
            self.maskLayer.setVisible(False)
            self.gv.setCursor(self.default_cursor)
            self.gv.setDragMode(QGraphicsView.ScrollHandDrag)
            self.painting = False

    def setMaskTransparencyBySlider(self, slider_value: int):
        self.setMaskTransparency(slider_value / 100)

    def setOriginalTransparencyBySlider(self, slider_value: int):
        self.setOriginalTransparency(slider_value / 100)

    def setTextBlockMode(self, mode: bool):
        self.textblock_mode = mode

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        if self.hovering_textblkitem or self.editing_textblkitem:
            menu = QMenu()
            delete_act = menu.addAction(self.tr("Delete"))
            rst = menu.exec_(event.screenPos())
            blk_item = self.txtblkShapeControl.blk_item
            selected = self.selectedItems()
            if rst == delete_act:
                if selected:
                    self.delete_textblks.emit()
                elif blk_item:
                    blk_item.delete()
    
    def on_hide_canvas(self):
        self.alt_pressed = False
        self.scale_tool_mode = False
        self.textblock_mode = False
        self.creating_textblock = False
        self.create_block_origin = None
        self.editing_textblkitem = None
        self.hovering_textblkitem = None
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

    

    
        

    