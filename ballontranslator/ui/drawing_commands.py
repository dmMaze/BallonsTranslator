from qtpy.QtCore import Signal, Qt, QPointF, QSize, QLineF, QDateTime, QRectF, QPoint
from qtpy.QtGui import QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics, QImage
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from typing import Union, Tuple, List
import numpy as np
from utils.logger import logger

from .image_edit import ImageEditMode, PixmapItem, DrawingLayer, StrokeImgItem
from .canvas import Canvas


class StrokeItemUndoCommand(QUndoCommand):
    def __init__(self, target_layer: DrawingLayer, rect: Tuple[int], qimg: QImage, erasing=False):
        super().__init__()
        self.qimg = qimg
        self.x = rect[0]
        self.y = rect[1]
        self.target_layer = target_layer
        self.key = str(QDateTime.currentMSecsSinceEpoch())
        if erasing:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
        else:
            self.compose_mode = QPainter.CompositionMode.CompositionMode_SourceOver
        
    def undo(self):
        if self.qimg is not None:
            self.target_layer.removeQImage(self.key)
            self.target_layer.update()

    def redo(self):
        if self.qimg is not None:
            self.target_layer.addQImage(self.x, self.y, self.qimg, self.compose_mode, self.key)
            self.target_layer.scene().update()


class InpaintUndoCommand(QUndoCommand):
    def __init__(self, canvas: Canvas, inpainted: np.ndarray, mask: np.ndarray, inpaint_rect: List[int]):
        super().__init__()
        self.canvas = canvas
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        self.undo_img = np.copy(img_view)
        self.undo_mask = np.copy(mask_view)
        self.redo_img = inpainted
        self.redo_mask = mask
        self.inpaint_rect = inpaint_rect

    def redo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.redo_img
        mask_view[:] = self.redo_mask
        self.canvas.setInpaintLayer()
        self.canvas.setMaskLayer()

    def undo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.undo_img
        mask_view[:] = self.undo_mask
        self.canvas.setInpaintLayer()
        self.canvas.setMaskLayer()