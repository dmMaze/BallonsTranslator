from typing import Tuple, List, Union
import numpy as np
import cv2
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
        super().__init__()
        self._img = QImage(size, format)
        self._img.fill(Qt.GlobalColor.transparent)
        self.pen = pen
        self._d = d = self.pen.widthF()
        self._r = d / 2
        self.clipped_rect = None

        self.painter = QPainter(self._img)
        self.painter.setPen(pen)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        self.setBoundingRegionGranularity(0)
        self.cur_point = point
        self._br = QRectF(0, 0, size.width(), size.height())
        self.is_painting = True
        self.lineTo(point)

    def finishPainting(self):
        self.painter.end()
        self.is_painting = False

    def clip(self, mask_only=False, format=QImage.Format.Format_ARGB32_Premultiplied) -> Tuple[List, np.ndarray, QImage]:
        img_array = pixmap2ndarray(self._img, True)
        ar = cv2.boundingRect(cv2.findNonZero(img_array[..., -1]))
        img_array = img_array[ar[1]: ar[1] + ar[3], ar[0]: ar[0] + ar[2]]
        if not (ar[2] > 0 and ar[3] > 0):
            return None, None, None
        if mask_only:
            img_array = img_array[..., -1]
            img_array[img_array > 0] = 255
        return ar, img_array, self._img.copy(*ar).convertToFormat(format)

    def startNewPoint(self, pos: QPointF):
        self.is_painting = True
        self.painter.begin(self._img)
        self.painter.setPen(self.pen)
        self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        self.cur_point = pos
        self.lineTo(pos)

    def boundingRect(self) -> QRectF:
        return self._br

    def lineTo(self, new_pnt: QPointF, update=True) -> QRectF:
        delta = self.cur_point - new_pnt
        delta_w, delta_h = abs(delta.x()),  abs(delta.y())
        rect = None
        if delta_w + delta_h > 2:
            min_x = min(self.cur_point.x(), new_pnt.x()) - self._r
            min_y = min(self.cur_point.y(), new_pnt.y()) - self._r
            delta_w += self._d
            delta_h += self._d
            rect = QRectF(min_x, min_y, delta_w, delta_h)
            self.painter.drawLine(self.cur_point, new_pnt)
            self.cur_point = new_pnt
            if update:
                self.update(rect)
        return rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.drawImage(0, 0, self._img)


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
        self.drawed_pixmap = None

    def addQImage(self, x: int, y: int, qimg: QImage, compose_mode, key: str):
        self.qimg_dict[key] = qimg
        self.drawing_items_info[key] = {'pos': [x, y], 'compose': compose_mode}
        self.update()

    def removeQImage(self, key: str):
        if key in self.qimg_dict:
            self.qimg_dict.pop(key)
            self.drawing_items_info.pop(key)

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
        self.drawed_pixmap = pixmap

    def get_drawed_pixmap(self, format=QImage.Format.Format_ARGB32) -> QPixmap:
        pixmap = self.pixmap() if self.drawed_pixmap is None else self.drawed_pixmap
        return pixmap

    def drawed(self) -> bool:
        return len(self.qimg_dict) > 0

    def clearAllDrawings(self):
        self.qimg_dict.clear()
        self.drawing_items_info.clear()

