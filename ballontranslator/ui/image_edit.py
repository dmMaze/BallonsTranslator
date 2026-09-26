from typing import Tuple, List, Union
import numpy as np
import cv2

from qtpy.QtCore import QRectF, Qt, QPointF, QSize
from qtpy.QtWidgets import QStyleOptionGraphicsItem, QGraphicsPixmapItem, QWidget, QGraphicsItem
from qtpy.QtGui import QPen, QPainter, QPainterPath, QPixmap, QImage, QBrush

from .misc import pixmap2ndarray

SIZE_MAX = 2147483647

class ImageEditMode:
    NONE = 0
    HandTool = 0
    InpaintTool = 1
    PenTool = 2
    RectTool = 3
    ShapeFillTool = 4


def shape_fill_path(rect: QRectF, shape: str) -> QPainterPath:
    """Share the shape geometry between the drag preview and saved pixels.

    >>> shape_fill_path(QRectF(0, 0, 10, 20), 'ellipse').boundingRect().height()
    20.0
    """
    path = QPainterPath()
    if shape == 'rectangle':
        path.addRect(rect.normalized())
    elif shape == 'ellipse':
        path.addEllipse(rect.normalized())
    else:
        raise ValueError(f'Unknown fill shape: {shape!r}')
    return path

class PenShape:
    Circle = 0
    Rectangle = 1
    MagicWand = 2
    Triangle = 3

class StrokeImgItem(QGraphicsItem):
    def __init__(self, pen: QPen, point: QPointF, size: QSize, format: QImage.Format = QImage.Format.Format_ARGB32, shape=PenShape.Circle):
        super().__init__()
        self._img = QImage(size, format)
        self._img.fill(Qt.GlobalColor.transparent)
        pen = QPen(pen)
        if shape == PenShape.Rectangle:
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            
        self.pen = pen
        self._d = d = pen.widthF()
        self._d_rect = d // 32
        self._r = d / 2
        self.clipped_rect = None
        self.shape = shape
        self._line_to = [self._line_to_circle, self._line_to_rectangle][shape]

        self.painter = QPainter(self._img)
        self.painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        if shape != PenShape.Circle:
            pen.setWidthF(0)
        self.painter.setPen(pen)
        self.painter.setBrush(pen.color())
        
        self.setBoundingRegionGranularity(0)
        self.cur_point = point
        self._br = QRectF(0, 0, size.width(), size.height())
        self.is_painting = True

        min_x = self.cur_point.x() - self._r
        min_y = self.cur_point.y() - self._r
        if shape == PenShape.Circle:
            self._line_to(self.cur_point, None)
        else:
            self._line_to(self.cur_point, None)
        rect = QRectF(min_x, min_y, self._d, self._d)
        self.init_rect = rect
        self.update(rect)

    def finishPainting(self) -> None:
        if self.painter.isActive():
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

    def _line_to_circle(self, pnt1: QPointF, pnt2: QPointF):
        if pnt2 is not None:
            self.painter.drawLine(pnt1, pnt2)
        else:
            pen = QPen(self.pen)
            pen.setWidthF(0)
            self.painter.setPen(pen)
            self.painter.setBrush(self.pen.color())
            rect = QRectF(pnt1.x() - self._r, pnt1.y() - self._r, self._d, self._d)
            self.painter.drawEllipse(rect)
            self.painter.setPen(self.pen)

    def _line_to_rectangle(self, pnt1: QPointF, pnt2: QPointF):
        shape_rect = QRectF(pnt1.x() - self._r, pnt1.y() - self._r, self._d, self._d)
        self.painter.drawRect(shape_rect)

    def lineTo(self, new_pnt: QPointF, update=True) -> QRectF:
        delta = self.cur_point - new_pnt
        delta_w, delta_h = abs(delta.x()),  abs(delta.y())
        rect = None
        if delta_w + delta_h > 1:
            min_x = min(self.cur_point.x(), new_pnt.x()) - self._r
            min_y = min(self.cur_point.y(), new_pnt.y()) - self._r
            delta_w += self._d
            delta_h += self._d
            rect = QRectF(min_x, min_y, delta_w, delta_h)
            self._line_to(self.cur_point, new_pnt)
            self.cur_point = new_pnt
            if update:
                self.update(rect)
        return rect

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.drawImage(0, 0, self._img)


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

    def __init__(self) -> None:
        super().__init__()
        self.qimg_dict = {}
        self.drawing_items_info = {}
        self._drawn_pixmap: QPixmap | None = None

    def addQImage(
        self, x: int, y: int, qimg: QImage,
        compose_mode: QPainter.CompositionMode, key: str,
    ) -> None:
        self.qimg_dict[key] = qimg
        self.drawing_items_info[key] = {'pos': [x, y], 'compose': compose_mode}
        self.updateDrawing()

    def removeQImage(self, key: str) -> None:
        if key in self.qimg_dict:
            self.qimg_dict.pop(key)
            self.drawing_items_info.pop(key)
            self.updateDrawing()

    def updateDrawing(self, rect: QRectF = QRectF()) -> None:
        # Live erasing mutates a stored QImage in place between mouse moves.
        self._drawn_pixmap = None
        self.update(rect)

    def setPixmap(self, pixmap: QPixmap) -> None:
        self._drawn_pixmap = None
        super().setPixmap(pixmap)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        painter.drawPixmap(self.offset(), self.get_drawed_pixmap())

    def get_drawed_pixmap(self) -> QPixmap:
        """Share a current composite between display and saving until pixels change.

        >>> DrawingLayer.get_drawed_pixmap.__annotations__['return'] is QPixmap
        True
        """
        if self._drawn_pixmap is not None:
            return self._drawn_pixmap
        pixmap = self.pixmap()
        if pixmap.isNull() or not self.qimg_dict:
            return pixmap
        painter = QPainter(pixmap)
        try:
            for key, item in self.qimg_dict.items():
                info = self.drawing_items_info[key]
                if isinstance(item, QImage):
                    painter.setCompositionMode(info['compose'])
                    painter.drawImage(info['pos'][0], info['pos'][1], item)
        finally:
            painter.end()
        self._drawn_pixmap = pixmap
        return pixmap

    def drawed(self) -> bool:
        return len(self.qimg_dict) > 0

    def clearAllDrawings(self) -> None:
        self.qimg_dict.clear()
        self.drawing_items_info.clear()
        self.updateDrawing()
