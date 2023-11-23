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
from .canvas import Canvas, TextBlkItem
from .textedit_area import TransPairWidget


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
        self.canvas.updateLayers()

    def undo(self) -> None:
        inpaint_rect = self.inpaint_rect
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        img_view[:] = self.undo_img
        mask_view[:] = self.undo_mask
        self.canvas.updateLayers()


class EmptyCommand(QUndoCommand):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
    

class RunBlkTransCommand(QUndoCommand):
    def __init__(self, canvas: Canvas, blkitems: List[TextBlkItem], transpairw_list: List[TransPairWidget],  mode: int):
        super().__init__()

        self.empty_command = None
        if mode > 1:
            self.empty_command = EmptyCommand()
            canvas.draw_undo_stack.push(self.empty_command)

        self.op_counter = -1
        self.blkitems = blkitems
        self.transpairw_list = transpairw_list

        if mode < 3:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if mode != 0:
                    trs = blkitem.blk.translation
                    transpairw.e_trans.setPlainTextAndKeepUndoStack(trs)
                    blkitem.setPlainTextAndKeepUndoStack(trs)
                blkitem.blk.rich_text = ''
                if mode >= 0:
                    transpairw.e_source.setPlainTextAndKeepUndoStack(blkitem.blk.get_text())

        self.canvas = canvas
        self.mode = mode
        if mode > 1:
            self.undo_img_list = []
            self.undo_mask_list = []
            self.redo_img_list = []
            self.redo_mask_list = []
            self.inpaint_rect_lst = []
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            self.num_inpainted = 0
            for item in self.blkitems:
                inpainted_dict = item.blk.region_inpaint_dict
                item.blk.region_inpaint_dict = None
                if inpainted_dict is None:
                    self.undo_img_list.append(None)
                    self.undo_mask_list.append(None)
                    self.redo_mask_list.append(None)
                    self.redo_img_list.append(None)
                    self.inpaint_rect_lst.append(None)
                else:
                    inpaint_rect = inpainted_dict['inpaint_rect']
                    img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                    mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                    self.undo_img_list.append(np.copy(img_view))
                    self.undo_mask_list.append(np.copy(mask_view))
                    self.redo_img_list.append(inpainted_dict['inpainted'])
                    self.redo_mask_list.append(inpainted_dict['mask'])
                    self.inpaint_rect_lst.append(inpaint_rect)
                    self.num_inpainted += 1

    def redo(self) -> None:

        if self.empty_command is not None:
            self.empty_command.redo()

        if self.mode > 1 and self.num_inpainted > 0:
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for inpaint_rect, redo_img, redo_mask in zip(self.inpaint_rect_lst, self.redo_img_list, self.redo_mask_list):
                if inpaint_rect is None:
                    continue
                img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                img_view[:] = redo_img
                mask_view[:] = redo_mask
            self.canvas.updateLayers()

        if self.op_counter < 0:
            self.op_counter += 1
            return

        if self.mode < 3:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if self.mode != 0:
                    transpairw.e_trans.redo()
                    blkitem.redo()
                if self.mode >= 0:
                    transpairw.e_source.redo()

    def undo(self) -> None:

        if self.empty_command is not None:
            self.empty_command.undo()

        if self.mode > 1 and self.num_inpainted > 0:
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for inpaint_rect, undo_img, undo_mask in zip(self.inpaint_rect_lst, self.undo_img_list, self.undo_mask_list):
                if inpaint_rect is None:
                    continue
                img_view = img_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                mask_view = mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
                img_view[:] = undo_img
                mask_view[:] = undo_mask
            self.canvas.updateLayers()

        if self.mode < 3:
            for blkitem, transpairw in zip(self.blkitems, self.transpairw_list):
                if self.mode != 0:
                    transpairw.e_trans.undo()
                    blkitem.undo()
                if self.mode >= 0:
                    transpairw.e_source.undo()