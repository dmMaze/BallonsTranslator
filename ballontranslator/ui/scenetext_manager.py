
from typing import List, Union
import numpy as np

from PyQt5.QtWidgets import QApplication, QUndoCommand
from PyQt5.QtCore import pyqtSignal, QObject, QRectF, Qt
from PyQt5.QtGui import QTextCursor
from .imgtranspanel import TransPairWidget
from .textitem import TextBlkItem, TextBlock, xywh2xyxypoly, rotate_polygons
from .canvas import Canvas
from .imgtranspanel import TextPanel, TextEditListScrollArea, SourceTextEdit, TransTextEdit
from .texteditshapecontrol import TextBlkShapeControl


class MoveItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, parent=None):
        super(MoveItemCommand, self).__init__(parent)
        self.item = item
        self.oldPos = item.oldPos
        self.newPos = item.pos()

    def redo(self):
        self.item.setPos(self.newPos)

    def undo(self):
        self.item.setPos(self.oldPos)

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.newPos = item.pos()
        return True

# according to https://doc.qt.io/qt-5/qundocommand.html
# some of following commands are done twice after initialization
class ReshapeItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, parent=None):
        super(ReshapeItemCommand, self).__init__(parent)
        self.item = item
        self.oldRect = item.oldRect
        self.newRect = item.rect()

    def redo(self):
        self.item.setRect(self.newRect)

    def undo(self):
        self.item.setRect(self.oldRect)

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.newRect = item.rect()
        return True

class RotateItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, new_angle: float):
        super(RotateItemCommand, self).__init__()
        self.item = item
        self.old_angle = item.rotation()
        self.new_angle = new_angle

    def redo(self):
        self.item.setRotation(self.new_angle)
        self.item.blk.angle = self.new_angle

    def undo(self):
        self.item.setRotation(self.old_angle)
        self.item.blk.angle = self.old_angle

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.new_angle = item.angle
        return True

class OrientationItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, ctrl):
        super(OrientationItemCommand, self).__init__()
        self.item = item
        self.ctrl: SceneTextManager = ctrl
        self.oldVertical = item.is_vertical
        self.newVertical = self.ctrl.fontformat.vertical

    def redo(self):
        self.item.setVertical(self.newVertical)
        self.ctrl.formatpanel.verticalChecker.setChecked(self.newVertical)

    def undo(self):
        self.item.setVertical(self.oldVertical)
        self.ctrl.formatpanel.verticalChecker.setChecked(self.oldVertical)

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.newVertical = command.newVertical
        self.oldVertical = command.oldVertical
        return True

class DeleteItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, ctrl, parent=None):
        super().__init__(parent)
        self.item = item
        self.p_widget = ctrl.pairwidget_list[item.idx]
        self.ctrl: SceneTextManager = ctrl

    def redo(self):
        self.ctrl.deleteTextblkItem(self.item)

    def undo(self):
        self.ctrl.recoverTextblkItem(self.item, self.p_widget)

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.item = item
        self.p_widget = command.p_widget
        return True

class CreateItemCommand(QUndoCommand):
    def __init__(self, blk_item: TextBlkItem, ctrl, parent=None):
        super().__init__(parent)
        self.blk_item = blk_item
        self.ctrl: SceneTextManager = ctrl

    def redo(self):
        self.ctrl.addTextBlock(self.blk_item)
        self.ctrl.txtblkShapeControl.setBlkItem(self.blk_item)

    def undo(self):
        self.ctrl.deleteTextblkItem(self.blk_item)

    def mergeWith(self, command: QUndoCommand):
        blk_item = command.blk_item
        if self.blk_item != blk_item:
            return False
        self.blk_item = blk_item
        return True

class DeleteItemListCommand(QUndoCommand):
    def __init__(self, blk_list: TextBlkItem, ctrl, parent=None):
        super().__init__(parent)
        self.blk_list = []
        self.pwidget_list = []
        self.ctrl: SceneTextManager = ctrl
        for blkitem in blk_list:
            if isinstance(blkitem, TextBlkItem):
                self.blk_list.append(blkitem)
                self.pwidget_list.append(ctrl.pairwidget_list[blkitem.idx])

    def redo(self):
        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)

    def undo(self):
        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)

    def mergeWith(self, command: QUndoCommand):
        blk_list = command.blk_list
        if self.blk_list != blk_list:
            return False
        return True


class SceneTextManager(QObject):
    def __init__(self, 
                 app: QApplication,
                 canvas: Canvas, 
                 textpanel: TextPanel, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app     
        self.canvas = canvas
        self.canvas.scalefactor_changed.connect(self.adjustSceneTextRect)
        self.canvas.end_create_textblock.connect(self.onEndCreateTextBlock)
        self.canvas.delete_textblks.connect(self.onDeleteTextBlks)
        self.canvasUndoStack = self.canvas.undoStack
        self.txtblkShapeControl = canvas.txtblkShapeControl
        self.textpanel = textpanel

        self.textEditList = textpanel.textEditList
        self.formatpanel = textpanel.formatpanel

        self.imgtrans_proj = self.canvas.imgtrans_proj
        self.textblk_item_list: List[TextBlkItem] = []
        self.pairwidget_list: List[TransPairWidget] = []

        self.editing = False
        self.hovering_transwidget : TransTextEdit = None

        self.prev_blkitem: TextBlkItem = None

    def setTextEditMode(self, edit: bool = False):
        self.editing = edit
        if edit:
            self.textpanel.show()
            for blk_item in self.textblk_item_list:
                blk_item.show()
        else:
            self.txtblkShapeControl.setBlkItem(None)
            self.textpanel.hide()
            for blk_item in self.textblk_item_list:
                blk_item.hide()

    def adjustSceneTextRect(self):
        new_size = self.canvas.imgLayer.sceneBoundingRect().size()
        scale_factor = new_size.width() / self.canvas.old_size.width()
        for blk_item in self.textblk_item_list:
            rel_pos = blk_item.scenePos() * scale_factor
            blk_item.setScale(self.canvas.scale_factor)
            blk_item.setPos(blk_item.pos() + rel_pos - blk_item.scenePos())
        self.txtblkShapeControl.updateBoundingRect()

    def clearTextList(self):
        self.txtblkShapeControl.setBlkItem(None)
        for blkitem in self.textblk_item_list:
            self.canvas.removeItem(blkitem)
        self.textblk_item_list.clear()
        for textwidget in self.pairwidget_list:
            self.textEditList.removeWidget(textwidget)
        self.pairwidget_list.clear()

    def updateTextList(self):
        self.txtblkShapeControl.setBlkItem(None)
        self.clearTextList()
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            blk_item = self.addTextBlock(textblock)
            if not self.editing:
                blk_item.hide()

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
        self.addTextBlkItem(blk_item)
        rel_pos = blk_item.scenePos() * self.canvas.scale_factor
        blk_item.setScale(self.canvas.scale_factor)
        blk_item.setPos(blk_item.pos() + rel_pos - blk_item.scenePos())

        pair_widget = TransPairWidget(blk, len(self.pairwidget_list))
        self.pairwidget_list.append(pair_widget)
        self.textEditList.addPairWidget(pair_widget)
        pair_widget.e_source.setPlainText(blk_item.blk.get_text())
        pair_widget.e_source.user_edited.connect(self.on_srcwidget_edited)
        pair_widget.e_trans.setPlainText(blk_item.toPlainText())
        pair_widget.e_trans.hover_enter.connect(self.onTransWidgetHoverEnter)
        pair_widget.e_trans.content_change.connect(self.onTransWidgetContentchange)
        return blk_item

    def addTextBlkItem(self, textblk_item: TextBlkItem) -> TextBlkItem:
        self.textblk_item_list.append(textblk_item)
        self.canvas.addItem(textblk_item)
        textblk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
        textblk_item.end_edit.connect(self.onTextBlkItemEndEdit)
        textblk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
        textblk_item.hover_leave.connect(self.onTextBlkItemHoverLeave)
        textblk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
        textblk_item.moving.connect(self.onTextBlkItemMoving)
        textblk_item.moved.connect(self.onTextBlkItemMoved)
        textblk_item.reshaped.connect(self.onTextBlkItemReshaped)
        textblk_item.rotated.connect(self.onTextBlkItemRotated)
        textblk_item.to_delete.connect(self.onDeleteTextBlkItem)
        textblk_item.content_changed.connect(self.onTextBlkItemContentChanged)
        textblk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
        return textblk_item

    def deleteTextblkItem(self, blkitem: TextBlkItem):
        self.canvas.removeItem(blkitem)
        self.textblk_item_list.remove(blkitem)
        pwidget = self.pairwidget_list.pop(blkitem.idx)
        self.textEditList.removeWidget(pwidget)
        self.updateTextBlkItemIdx()
        self.txtblkShapeControl.setBlkItem(None)

    def deleteTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.canvas.removeItem(blkitem)
            self.textblk_item_list.remove(blkitem)
            self.pairwidget_list.remove(p_widget)
            self.textEditList.removeWidget(p_widget)
        self.updateTextBlkItemIdx()
        self.txtblkShapeControl.setBlkItem(None)

    def recoverTextblkItem(self, blkitem: TextBlkItem, p_widget: TransPairWidget):
        blkitem.idx = len(self.textblk_item_list)
        p_widget.idx = len(self.pairwidget_list)
        self.textblk_item_list.append(blkitem)
        self.canvas.addItem(blkitem)
        self.pairwidget_list.append(p_widget)
        self.textEditList.addPairWidget(p_widget)

    def recoverTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.recoverTextblkItem(blkitem, p_widget)

    def onTextBlkItemContentChanged(self, blk_item: TextBlkItem):
        if blk_item.hasFocus():
            trans_widget = self.pairwidget_list[blk_item.idx].e_trans
            if not trans_widget.hasFocus():
                trans_widget.setText(blk_item.toPlainText())
            self.canvas.setProjSaveState(True)
            

    def onTextBlkItemSizeChanged(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        if not self.txtblkShapeControl.reshaping:
            if self.txtblkShapeControl.blk_item == blk_item:
                self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemBeginEdit(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        self.canvas.editing_textblkitem = blk_item
        self.formatpanel.set_textblk_item(blk_item)
        self.txtblkShapeControl.setCursor(Qt.CursorShape.IBeamCursor)

    def onLeftbuttonPressed(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)

    def onTextBlkItemEndEdit(self, blk_id: int):
        blkitem = self.textblk_item_list[blk_id]
        self.canvas.editing_textblkitem = None
        self.formatpanel.set_textblk_item(None)
        self.txtblkShapeControl.setCursor(Qt.CursorShape.SizeAllCursor)

    def savePrevBlkItem(self, blkitem: TextBlkItem):
        self.prev_blkitem = blkitem
        self.prev_textCursor = QTextCursor(self.prev_blkitem.textCursor())

    def is_editting(self):
        blk_item = self.txtblkShapeControl.blk_item
        return blk_item is not None and blk_item.is_editting()

    def onTextBlkItemHoverEnter(self, blk_id: int):
        if self.is_editting():
            return
        blk_item = self.textblk_item_list[blk_id]
        if not blk_item.hasFocus():
            self.txtblkShapeControl.setBlkItem(blk_item)
        if self.hovering_transwidget is not None:
            self.hovering_transwidget.setHoverEffect(False)
        self.hovering_transwidget = self.pairwidget_list[blk_id].e_trans
        self.hovering_transwidget.setHoverEffect(True)
        self.textpanel.textEditList.ensureWidgetVisible(self.hovering_transwidget)
        self.canvas.hovering_textblkitem = blk_item

    def onTextBlkItemHoverLeave(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.canvas.hovering_textblkitem = None

    def onTextBlkItemMoving(self, item: TextBlkItem):
        self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemMoved(self, item: TextBlkItem):
        self.canvasUndoStack.push(MoveItemCommand(item))
        
    def onTextBlkItemReshaped(self, item: TextBlkItem):
        self.canvasUndoStack.push(ReshapeItemCommand(item))

    def onTextBlkItemRotated(self, new_angle: float):
        blk_item = self.txtblkShapeControl.blk_item
        if blk_item:
            self.canvasUndoStack.push(RotateItemCommand(blk_item, new_angle))

    def onDeleteTextBlkItem(self, item: TextBlkItem):
        self.canvasUndoStack.push(DeleteItemCommand(item, self))

    def onDeleteTextBlks(self):
        selections = self.canvas.selectedItems()
        self.canvasUndoStack.push(DeleteItemListCommand(selections, self))

    def onEndCreateTextBlock(self, rect: QRectF):
        scale_f = self.canvas.scale_factor
        if rect.width() > 1 and rect.height() > 1:
            xyxy = np.array([rect.x(), rect.y(), rect.right(), rect.bottom()])        
            xyxy = np.round(xyxy / scale_f).astype(np.int)
            block = TextBlock(xyxy)
            xywh = np.copy(xyxy)
            xywh[[2, 3]] -= xywh[[0, 1]]
            block.lines = xywh2xyxypoly(np.array([xywh])).reshape(-1, 4, 2).tolist()
            blk_item = TextBlkItem(block, len(self.textblk_item_list), set_format=False, show_rect=True)
            blk_item.set_fontformat(self.formatpanel.global_format)
            self.canvasUndoStack.push(CreateItemCommand(blk_item, self))

    def onRotateTextBlkItem(self, item: TextBlock):
        self.canvasUndoStack.push(RotateItemCommand(item))
    
    def onTransWidgetHoverEnter(self, idx: int):
        if self.is_editting():
            return
        blk_item = self.textblk_item_list[idx]
        self.canvas.gv.ensureVisible(blk_item)
        self.txtblkShapeControl.setBlkItem(blk_item)

    def onTransWidgetContentchange(self, idx: int, text: str):
        blk_item = self.textblk_item_list[idx]
        blk_item.setTextInteractionFlags(Qt.NoTextInteraction)
        blk_item.setPlainText(text)
        self.canvas.setProjSaveState(True)

    def on_srcwidget_edited(self):
        self.canvas.setProjSaveState(True)
        pass

    def updateTextBlkItemIdx(self):
        for ii, blk_item in enumerate(self.textblk_item_list):
            blk_item.idx = ii
            self.pairwidget_list[ii].updateIndex(ii)

    def updateTextBlkList(self):
        cbl = self.imgtrans_proj.current_block_list()
        if cbl is None:
            return
        cbl.clear()
        for blk_item, trans_pair in zip(self.textblk_item_list, self.pairwidget_list):
            if not blk_item.document().isEmpty():
                blk_item.blk.rich_text = blk_item.toHtml()
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            br = blk_item.boundingRect()
            w, h = br.width(), br.height()
            sc = blk_item.sceneBoundingRect().center()
            x = sc.x() / blk_item.scale() - w / 2
            y = sc.y() / blk_item.scale() - h / 2
            xywh = [x, y, w, h]
            blk_item.blk._bounding_rect = xywh
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def updateTranslation(self):
        for blk_item, transwidget in zip(self.textblk_item_list, self.pairwidget_list):
            transwidget.e_trans.setPlainText(blk_item.blk.translation)
            blk_item.setPlainText(blk_item.blk.translation)
            # blk_item.update()

    def showTextblkItemRect(self, draw_rect: bool):
        for blk_item in self.textblk_item_list:
            blk_item.draw_rect = draw_rect
            blk_item.update()
