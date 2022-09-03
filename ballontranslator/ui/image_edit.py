import numpy as np

from qtpy.QtCore import QRectF, Qt, QPointF, QSize, QPoint, QDateTime
from qtpy.QtWidgets import QStyleOptionGraphicsItem, QGraphicsPixmapItem, QWidget, QGraphicsPathItem, QGraphicsItem
from qtpy.QtGui import QPen, QColor, QPainterPath, QCursor, QPainter, QPixmap, QImage, QBrush

try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .misc import DrawPanelConfig, pixmap2ndarray, ndarray2pixmap
from utils.io_utils import imread, imwrite

SIZE_MAX = 2147483647

class ImageEditMode:
    NONE = 0
    HandTool = 0
    InpaintTool = 1
    PenTool = 2
    RectTool = 3


class StrokeImgItem(QGraphicsItem):
    def __init__(self, pen: QPen, point: QPointF, size: QSize, format: QImage.Format = QImage.Format.Format_ARGB32, ):
        self._img = QImage(size, format)
        self._img.fill(Qt.GlobalColor.transparent)
        self.pen = pen
        self.painter = QPainter()
        self.painter.setPen(pen)
        self.setBoundingRegionGranularity(0)
        self.cur_point = point
        self.drawLine(point, point)
        self._br = QRectF(0, 0, size.width(), size.height())

    def boundingRect(self) -> QRectF:
        return self._br

    def drawLine(self, new_pnt: QPointF):
        self.painter.begin(self._img)
        self.painter.drawLine(self.cur_point, new_pnt)
        self.painter.end()
        self.cur_point = new_pnt

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.drawImage(0, 0, self._img)
    

class StrokeItem(QGraphicsPathItem):
    def __init__(self, origin_point: QPointF, erasing: bool = False):
        super().__init__()
        # self.stroke = QPainterPath(QPointF(0, 0))
        self.stroke = QPainterPath(QPointF(origin_point))
        self.erasing = erasing
        self.last_point = origin_point
        self.setPath(self.stroke)
        self.setBoundingRegionGranularity(0)
        self.clip_offset = QPointF(0, 0)

    def addNewPoint(self, new_point: QPointF):
        if new_point != self.last_point:
            self.stroke.lineTo(new_point)
            self.setPath(self.stroke)
            self.last_point = new_point

    def addStroke(self, stroke: QPainterPath):
        self.stroke.addPath(stroke)
        self.setPath(self.stroke)

    def isEmpty(self):
        return self.stroke.isEmpty()

    def convertToPixmapItem(self, convert_mask=False, remove_stroke=True, target_layer: QGraphicsPixmapItem = None) -> QGraphicsPixmapItem:
        if target_layer is None:
            target_layer = self.parentItem()

        pixmap = self.getSubimg(convert_mask, return_pixmap=True)
        if pixmap is None:
            self.scene().removeItem(self)
            return None, None, None
        # pixmap = ndarray2pixmap(img_array)
        pixmap_item = QGraphicsPixmapItem(pixmap)

        pixmap_item.setParentItem(target_layer)
        pos = self.subBlockPos()
        pixmap_item.setPos(pos.x(), pos.y())
        if self.scene() is not None:
            if remove_stroke:
                self.scene().removeItem(self)
            else:
                self.setZValue(3)
        return pixmap_item

    def convertToQImg(self, convert_mask=False) -> QImage:
        img_array = self.getSubimg(convert_mask)
        if img_array is None:
            return None
        qimg = ndarray2pixmap(img_array, return_qimg=True)
        return qimg

    def originOffset(self) -> QPointF:
        thickness = self.pen().widthF() / 2
        return QPointF(thickness, thickness) - self.stroke.boundingRect().topLeft() - self.clip_offset

    def subBlockPos(self) -> QPoint:
        pos = self.pos() - self.originOffset()
        pos.setX(int(round(max(0, pos.x()))))
        pos.setY(int(round(max(0, pos.y()))))
        return pos.toPoint()

    def getSubimg(self, convert_mask=False, return_pixmap=False) -> np.ndarray:
        if self.isEmpty():
            return None

        origin_offset = self.originOffset()
        parent_layer = self.parentItem()
        while parent_layer.parentItem() is not None:
            if isinstance(parent_layer, QGraphicsPixmapItem):
                layer_size = parent_layer.pixmap().size()
            parent_layer = parent_layer.parentItem()

        scale_factor = parent_layer.scale()
        # layer_size = parent_layer.pixmap().size()
        max_width, max_height = layer_size.width(), layer_size.height()
        scene_br = self.sceneBoundingRect()
        stroke_size = scene_br.size()
        stroke_size.setHeight(stroke_size.height() / scale_factor)
        stroke_size.setWidth(stroke_size.width() / scale_factor)
        stroke_size = stroke_size.toSize()

        lt = self.pos() - origin_offset
        xywh = [lt.x(), lt.y(), scene_br.width() / scale_factor, scene_br.height() / scale_factor]
        xyxy = np.array(xywh)
        xyxy[[2, 3]] += xyxy[[0, 1]]

        xyxy = xyxy.astype(np.int)            
        xyxy_clip = xyxy.copy()
        xyxy_clip[[0, 2]] = np.clip(xyxy[[0, 2]], 0, max_width - 1)
        xyxy_clip[[1, 3]] = np.clip(xyxy[[1, 3]], 0, max_height - 1)
        # clipped stroke is empty
        if xyxy_clip[0] >= xyxy_clip[2] or xyxy_clip[1] >= xyxy_clip[3]:
            return None

        pixmap = QPixmap(stroke_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.translate(self.originOffset())
        painter.setPen(self.pen())
        painter.setBrush(self.brush())
        painter.drawPath(self.stroke)
        painter.end()

        stroke_clip = xyxy_clip - xyxy
        stroke_clip[2] += stroke_size.width()
        stroke_clip[3] += stroke_size.height()
        self.clip_offset = QPointF(stroke_clip[0], stroke_clip[1])

        if return_pixmap:
            return pixmap
            
        imgarray = pixmap2ndarray(pixmap, keep_alpha=True)
        imgarray = imgarray[stroke_clip[1]: stroke_clip[3], stroke_clip[0]: stroke_clip[2]]
        # print(imgarray.shape, stroke_clip)
        
        if convert_mask:
            mask = imgarray[..., -1]
            mask[mask > 0] = 255
            imgarray[..., :] = mask[..., np.newaxis]
            return mask
            
        return imgarray


class PenCursor(QCursor):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.thickness = 2

    def updatePenCursor(self, size: int, color: QColor):
        
        pen = QPen(color, self.thickness, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        if size < 20:
            pen.setWidth(3)
            pen.setStyle(Qt.PenStyle.SolidLine)
        cur_pixmap = QPixmap(QSize(int(size), int(size)))
        cur_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cur_pixmap)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawEllipse(self.thickness, self.thickness, size-2*self.thickness, size-2*self.thickness)
        painter.end()


class PixmapItem(QGraphicsPixmapItem):
    def __init__(self, border_pen: QPen, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_pen = border_pen

    def paint(self, painter: QPainter, option: 'QStyleOptionGraphicsItem', widget: QWidget) -> None:
        pen = painter.pen()
        painter.setPen(self.border_pen)
        painter.drawRect(self.boundingRect())
        painter.setPen(pen)
        return super().paint(painter, option, widget)


class DrawingLayer(QGraphicsPixmapItem):

    def __init__(self):
        super().__init__()
        self.qimg_dict = {}
        self.drawing_items_info = {}

    def addQImage(self, x: int, y: int, qimg: QImage, compose_mode, key: str):
        self.qimg_dict[key] = qimg
        self.drawing_items_info[key] = {'pos': [x, y], 'compose': compose_mode}
        self.update()

    def removeQImage(self, key: str):
        if key in self.qimg_dict:
            self.qimg_dict.pop(key)
            # self.drawing_items_pos.pop(key)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget):
        pixmap = self.pixmap()
        p = QPainter()
        p.begin(pixmap)
        for key in self.qimg_dict:
            item = self.qimg_dict[key]
            info = self.drawing_items_info[key]
            if isinstance(item, QImage):
                p.setCompositionMode(info['compose'])
                p.drawImage(info['pos'][0], info['pos'][1], item)
        p.end()
        painter.drawPixmap(self.offset(), pixmap)

