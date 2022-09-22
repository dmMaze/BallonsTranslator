
from typing import List, Union, Tuple
import numpy as np

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QObject, QRectF, Qt, Signal
from qtpy.QtGui import QTextCursor, QFontMetrics, QFont, QTextCharFormat, QInputMethodEvent
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .textitem import TextBlkItem, TextBlock, xywh2xyxypoly
from .canvas import Canvas
from .textedit_area import TextPanel, TransTextEdit, SourceTextEdit, TransPairWidget, TextEditCommand
from .fontformatpanel import set_textblk_fontsize
from .misc import FontFormat, ProgramConfig, pt2px
from .texteditshapecontrol import TextBlkShapeControl

from utils.imgproc_utils import extract_ballon_region
from utils.text_processing import seg_text, is_cjk
from utils.text_layout import layout_text

class MoveBlkItemsCommand(QUndoCommand):
    def __init__(self, items: List[TextBlkItem], shape_ctrl: TextBlkShapeControl):
        super(MoveBlkItemsCommand, self).__init__()
        self.items = items
        self.old_pos_lst = []
        self.new_pos_lst = []
        self.shape_ctrl = shape_ctrl
        for item in items:
            self.old_pos_lst.append(item.oldPos)
            self.new_pos_lst.append(item.pos())
            item.oldPos = item.pos()

    def redo(self):
        for item, new_pos in zip(self.items, self.new_pos_lst):
            item.setPos(new_pos)
            if self.shape_ctrl.blk_item == item and self.shape_ctrl.pos() != new_pos:
                self.shape_ctrl.setPos(new_pos)

    def undo(self):
        for item, old_pos in zip(self.items, self.old_pos_lst):
            item.setPos(old_pos)
            if self.shape_ctrl.blk_item == item and self.shape_ctrl.pos() != old_pos:
                self.shape_ctrl.setPos(old_pos)

    def mergeWith(self, command: QUndoCommand):
        if command.old_pos_lst == self.old_pos_lst:
            return True
        return False


class ApplyFontformatCommand(QUndoCommand):
    def __init__(self, items: List[TextBlkItem], fontformat: FontFormat):
        super(ApplyFontformatCommand, self).__init__()
        self.items = items
        self.old_html_lst = []
        self.old_rect_lst = []
        self.old_fmt_lst = []
        self.new_fmt = fontformat
        for item in items:
            self.old_html_lst.append(item.toHtml())
            self.old_fmt_lst.append(item.get_fontformat())
            self.old_rect_lst.append(item.absBoundingRect())

    def redo(self):
        for item in self.items:
            item.set_fontformat(self.new_fmt, set_char_format=True)

    def undo(self):
        for rect, item, html, fmt in zip(self.old_rect_lst, self.items, self.old_html_lst, self.old_fmt_lst):
            item.setHtml(html)
            item.set_fontformat(fmt)
            item.setRect(rect)

    def mergeWith(self, command: QUndoCommand):
        if command.new_fmt == self.new_fmt:
            return True
        return False


class ApplyEffectCommand(QUndoCommand):
    def __init__(self, items: List[TextBlkItem], fontformat: FontFormat):
        super(ApplyEffectCommand, self).__init__()
        self.items = items
        self.old_fmt_lst: List[FontFormat] = []
        self.new_fmt = fontformat
        for item in items:
            self.old_fmt_lst.append(item.get_fontformat())

    def redo(self):
        for item in self.items:
            item.update_effect(self.new_fmt)
            item.update()

    def undo(self):
        for item, fmt in zip(self.items, self.old_fmt_lst):
            item.update_effect(fmt)
            item.update()

class ReshapeItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem, parent=None):
        super(ReshapeItemCommand, self).__init__(parent)
        self.item = item
        self.oldRect = item.oldRect
        self.newRect = item.absBoundingRect()

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
    def __init__(self, item: TextBlkItem, new_angle: float, shape_ctrl: TextBlkShapeControl):
        super(RotateItemCommand, self).__init__()
        self.item = item
        self.old_angle = item.rotation()
        self.new_angle = new_angle
        self.shape_ctrl = shape_ctrl

    def redo(self):
        self.item.setRotation(self.new_angle)
        self.item.blk.angle = self.new_angle
        if self.shape_ctrl.blk_item == self.item and self.shape_ctrl.rotation() != self.new_angle:
            self.shape_ctrl.setRotation(self.new_angle)

    def undo(self):
        self.item.setRotation(self.old_angle)
        self.item.blk.angle = self.old_angle
        if self.shape_ctrl.blk_item == self.item and self.shape_ctrl.rotation() != self.old_angle:
            self.shape_ctrl.setRotation(self.old_angle)

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


class DeleteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = []
        self.pwidget_list: List[TransPairWidget] = []
        self.ctrl: SceneTextManager = ctrl
        self.sw = self.ctrl.canvas.search_widget

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list = []
        self.highlighter_list = []
        self.old_counter_sum = self.sw.counter_sum
        self.sw_changed = False
        
        for blkitem in blk_list:
            if not isinstance(blkitem, TextBlkItem):
                continue
            self.blk_list.append(blkitem)
            pw: TransPairWidget = ctrl.pairwidget_list[blkitem.idx]
            self.pwidget_list.append(pw)

            rst_idx = self.sw.get_result_edit_index(pw.e_trans)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_start(-1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

            rst_idx = self.sw.get_result_edit_index(pw.e_source)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_start(-1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

        self.new_counter_sum = self.sw.counter_sum
        if self.sw_changed:
            if self.sw.counter_sum > 0:
                idx = self.sw.get_result_edit_index(self.sw.current_edit)
                if self.sw.current_cursor is not None and idx != -1:
                    self.sw.result_pos = self.sw.highlighter_list[idx].matched_map[self.sw.current_cursor.position()]
                    if idx > 0:
                        self.sw.result_pos += sum(self.sw.search_counter_list[: idx])
                    self.sw.updateCounterText()
                else:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
            else:
                self.sw.setCurrentEditor(None)

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        if self.sw_changed:
            self.sw.counter_sum = self.new_counter_sum
            cursor_removed = False
            for edit in self.search_rstedit_list:
                idx = self.sw.get_result_edit_index(edit)
                if idx != -1:
                    self.sw.search_rstedit_list.pop(idx)
                    self.sw.search_counter_list.pop(idx)
                    self.sw.highlighter_list.pop(idx)
                if edit == self.sw.current_edit:
                    cursor_removed = True
            if cursor_removed:
                if self.sw.counter_sum > 0:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
                else:
                    self.sw.setCurrentEditor(None)


    def undo(self):
        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)
        if self.sw_changed:
            self.sw.counter_sum = self.old_counter_sum
            self.sw.search_rstedit_list += self.search_rstedit_list
            self.sw.search_counter_list += self.search_counter_list
            self.sw.highlighter_list += self.highlighter_list
            self.sw.updateCounterText()


class AutoLayoutCommand(QUndoCommand):
    def __init__(self, items: List[TextBlkItem], old_rect_lst: List, old_html_lst: List, trans_widget_lst: List[TransTextEdit]):
        super(AutoLayoutCommand, self).__init__()
        self.items = items
        self.old_html_lst = old_html_lst
        self.old_rect_lst = old_rect_lst
        self.trans_widget_lst = trans_widget_lst
        self.new_rect_lst = []
        self.new_html_lst = []
        for item in items:
            self.new_html_lst.append(item.toHtml())
            self.new_rect_lst.append(item.absBoundingRect())
        self.counter = 0

    def redo(self):
        self.counter += 1
        if self.counter <= 1:
            return
        for item, trans_widget, html, rect  in zip(self.items, self.trans_widget_lst, self.new_html_lst, self.new_rect_lst):
            trans_widget.setPlainText(item.toPlainText())
            item.setPlainText('')
            item.setRect(rect, repaint=False)
            item.setHtml(html)
            if item.letter_spacing != 1:
                item.setLetterSpacing(item.letter_spacing, force=True)
            

    def undo(self):
        for item, trans_widget, html, rect  in zip(self.items, self.trans_widget_lst, self.old_html_lst, self.old_rect_lst):
            trans_widget.setPlainText(item.toPlainText())
            item.setPlainText('')
            item.setRect(rect, repaint=False)
            item.setHtml(html)
            if item.letter_spacing != 1:
                item.setLetterSpacing(item.letter_spacing, force=True)


class TextItemEditCommand(QUndoCommand):
    def __init__(self, blkitem: TextBlkItem, trans_edit: TransTextEdit, num_steps: int):
        super(TextItemEditCommand, self).__init__()
        self.op_counter = 0
        self.edit = trans_edit
        self.blkitem = blkitem
        self.num_steps = min(num_steps, 2)
        if blkitem.input_method_from == -1:
            self.num_steps = 1
        else:
            blkitem.input_method_from = -1

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        for _ in range(self.num_steps):
            self.blkitem.redo()
        if self.edit is not None:
            self.edit.redo()

    def undo(self):
        for _ in range(self.num_steps):
            self.blkitem.undo()
        if self.edit is not None:
            self.edit.undo()


class SceneTextManager(QObject):
    new_textblk = Signal(int)
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
        self.canvas.delete_textblks.connect(self.onDeleteBlkItems)
        self.canvas.format_textblks.connect(self.onFormatTextblks)
        self.canvas.layout_textblks.connect(self.onAutoLayoutTextblks)
        self.canvasUndoStack = self.canvas.undoStack
        self.txtblkShapeControl = canvas.txtblkShapeControl
        self.textpanel = textpanel

        self.textEditList = textpanel.textEditList
        self.formatpanel = textpanel.formatpanel
        self.formatpanel.effect_panel.apply.connect(self.on_apply_effect)
        self.formatpanel.global_format_changed.connect(self.onGlobalFormatChanged)

        self.imgtrans_proj = self.canvas.imgtrans_proj
        self.textblk_item_list: List[TextBlkItem] = []
        self.pairwidget_list: List[TransPairWidget] = []

        self.editing_flag = False
        self.auto_textlayout_flag = False
        self.hovering_transwidget : TransTextEdit = None

        self.prev_blkitem: TextBlkItem = None

        self.config: ProgramConfig = None

    def setTextEditMode(self, edit: bool = False):
        self.editing_flag = edit
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
        self.txtblkShapeControl.updateBoundingRect()

    def clearSceneTextitems(self):
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        for blkitem in self.textblk_item_list:
            self.canvas.removeItem(blkitem)
        self.textblk_item_list.clear()
        for textwidget in self.pairwidget_list:
            self.textEditList.removeWidget(textwidget)
        self.pairwidget_list.clear()

    def updateSceneTextitems(self):
        self.txtblkShapeControl.setBlkItem(None)
        self.clearSceneTextitems()
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            blk_item = self.addTextBlock(textblock)
            if not self.editing_flag:
                blk_item.hide()
        if self.auto_textlayout_flag:
            self.updateTextBlkList()

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            translation = ''
            if self.auto_textlayout_flag and not blk.vertical:
                translation = blk.translation
                blk.translation = ''
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
            if translation:
                blk.translation = translation
                self.layout_textblk(blk_item, text=translation)
        self.addTextBlkItem(blk_item)

        pair_widget = TransPairWidget(blk, len(self.pairwidget_list))
        self.pairwidget_list.append(pair_widget)
        self.textEditList.addPairWidget(pair_widget)
        pair_widget.e_source.setPlainText(blk_item.blk.get_text())
        pair_widget.e_source.user_edited.connect(self.on_srcwidget_edited)
        pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)

        pair_widget.e_trans.setPlainText(blk_item.toPlainText())
        pair_widget.e_trans.focus_in.connect(self.onTransWidgetHoverEnter)
        pair_widget.e_trans.user_edited_verbose.connect(self.onTransWidgetUserEditedVerbose)
        pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)

        self.new_textblk.emit(blk_item.idx)
        return blk_item

    def addTextBlkItem(self, textblk_item: TextBlkItem) -> TextBlkItem:
        self.textblk_item_list.append(textblk_item)
        textblk_item.setParentItem(self.canvas.textLayer)
        textblk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
        textblk_item.end_edit.connect(self.onTextBlkItemEndEdit)
        textblk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
        textblk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
        textblk_item.moving.connect(self.onTextBlkItemMoving)
        textblk_item.moved.connect(self.onTextBlkItemMoved)
        textblk_item.reshaped.connect(self.onTextBlkItemReshaped)
        textblk_item.rotated.connect(self.onTextBlkItemRotated)
        textblk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
        textblk_item.undo_signal.connect(self.on_textedit_undo)
        textblk_item.redo_signal.connect(self.on_textedit_redo)
        textblk_item.user_edited_verbose.connect(self.onTextItemUserEditedVerbose)
        textblk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
        textblk_item.pasted.connect(self.onBlkitemPaste)
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
        # recovered order is different from before
        blkitem.idx = len(self.textblk_item_list)
        p_widget.idx = len(self.pairwidget_list)
        self.textblk_item_list.append(blkitem)
        blkitem.setParentItem(self.canvas.textLayer)
        self.pairwidget_list.append(p_widget)
        self.textEditList.addPairWidget(p_widget)
        self.updateTextBlkItemIdx()

    def recoverTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.recoverTextblkItem(blkitem, p_widget)
            if self.txtblkShapeControl.blk_item is not None and blkitem.isSelected():
                blkitem.setSelected(False)
        

    def onTextBlkItemSizeChanged(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        if not self.txtblkShapeControl.reshaping:
            if self.txtblkShapeControl.blk_item == blk_item:
                self.txtblkShapeControl.updateBoundingRect()

    def onBlkitemPaste(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        text = self.app.clipboard().text()
        cursor = blk_item.textCursor()
        cursor.insertText(text)

    def onTextBlkItemBeginEdit(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        self.canvas.editing_textblkitem = blk_item
        self.formatpanel.set_textblk_item(blk_item)
        self.txtblkShapeControl.setCursor(Qt.CursorShape.IBeamCursor)
        e_trans = self.pairwidget_list[blk_item.idx].e_trans
        self.changeHoveringWidget(e_trans)

    def changeHoveringWidget(self, edit: SourceTextEdit):
        if self.hovering_transwidget != edit:
            if self.hovering_transwidget is not None:
                self.hovering_transwidget.setHoverEffect(False)
            self.hovering_transwidget = edit
            if edit is not None:
                self.textEditList.ensureWidgetVisible(edit)
                edit.setHoverEffect(True)

    def onLeftbuttonPressed(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        selections: List[TextBlkItem] = self.canvas.selectedItems()
        if len(selections) > 1:
            for item in selections:
                item.oldPos = item.pos()

    def onTextBlkItemEndEdit(self, blk_id: int):
        self.canvas.editing_textblkitem = None
        self.formatpanel.set_textblk_item(None)
        self.txtblkShapeControl.setCursor(Qt.CursorShape.SizeAllCursor)

    def editingTextItem(self) -> TextBlkItem:
        if self.txtblkShapeControl.isVisible() and self.canvas.editing_textblkitem is not None:
            return self.canvas.editing_textblkitem
        return None

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
        self.changeHoveringWidget(self.pairwidget_list[blk_id].e_trans)

    def onTextBlkItemMoving(self, item: TextBlkItem):
        self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemMoved(self):
        selected_blks = self.get_selected_blkitems()
        if len(selected_blks) > 0:
            self.canvasUndoStack.push(MoveBlkItemsCommand(selected_blks, self.txtblkShapeControl))
        
    def onTextBlkItemReshaped(self, item: TextBlkItem):
        self.canvasUndoStack.push(ReshapeItemCommand(item))

    def onTextBlkItemRotated(self, new_angle: float):
        blk_item = self.txtblkShapeControl.blk_item
        if blk_item:
            self.canvasUndoStack.push(RotateItemCommand(blk_item, new_angle, self.txtblkShapeControl))

    def onDeleteBlkItems(self):
        selected_blks = self.get_selected_blkitems()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)
        if len(selected_blks) > 0:
            self.canvasUndoStack.push(DeleteBlkItemsCommand(selected_blks, self))

    def onFormatTextblks(self):
        self.apply_fontformat(self.formatpanel.global_format)

    def onAutoLayoutTextblks(self):
        selected_blks = self.get_selected_blkitems()
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        selected_blks = [blk for blk in selected_blks if not blk.is_vertical]
        if len(selected_blks) > 0:
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect())
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                self.layout_textblk(blkitem)

            self.canvasUndoStack.push(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def layout_textblk(self, blkitem: TextBlkItem, text: str = None, mask: np.ndarray = None, bounding_rect: List = None, region_rect: List = None):
        
        img = self.imgtrans_proj.img_array
        if img is None:
            return

        blk_font = blkitem.font()
        fmt = blkitem.get_fontformat()
        blk_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, fmt.letter_spacing * 100)
        text_size_func = lambda text: get_text_size(QFontMetrics(blk_font), text)
        
        src_is_cjk = is_cjk(self.config.dl.translate_source)
        tgt_is_cjk = is_cjk(self.config.dl.translate_target)

        if mask is None:
            bounding_rect = blkitem.absBoundingRect()
            if tgt_is_cjk:
                max_enlarge_ratio = 2.5
            else:
                max_enlarge_ratio = 3
            enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, max_enlarge_ratio)
            mask, ballon_area, mask_xyxy, region_rect = extract_ballon_region(img, bounding_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True)
        else:
            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]
        region_x, region_y, region_w, region_h = region_rect

        restore_charfmts = False
        if text is None:
            text = blkitem.toPlainText()
            restore_charfmts = True

        if self.config.let_uppercase_flag:
            text = text.upper()
        
        words, delimiter = seg_text(text, self.config.dl.translate_target)
        if len(words) == 0:
            return

        wl_list = get_words_length_list(QFontMetrics(blk_font), words)
        text_w, text_h = text_size_func(text)
        text_area = text_w * text_h
        line_height = int(round(fmt.line_spacing * text_h))
        delimiter_len = text_size_func(delimiter)[0]
 
        adaptive_fntsize = False
        if self.auto_textlayout_flag and self.config.let_fntsize_flag == 0:
            if not tgt_is_cjk:
                adaptive_fntsize = True
            
        resize_ratio = 1
        if adaptive_fntsize:
            area_ratio = ballon_area / text_area
            ballon_area_thresh = 1.7
            downscale_constraint = 0.6
            # downscale the font size if textarea exceeds the balloon_area / ballon_area_thresh
            # or the longest word exceeds the region_width
            resize_ratio = np.clip(min(area_ratio / ballon_area_thresh, max(wl_list) / region_rect[2], blkitem.blk.font_size / line_height), downscale_constraint, 1.0) 

        max_central_width = np.inf
        if tgt_is_cjk:
            if ballon_area / text_area > 2:
                if blkitem.blk.text:
                    _, _, brw, brh = blkitem.blk.bounding_rect()
                    br_area = brw * brh
                    if src_is_cjk:
                        resize_ratio = np.sqrt(region_h * region_w / br_area)
                    else:
                        resize_ratio = np.clip(max(np.sqrt(br_area / text_area) * 0.8, np.sqrt(ballon_area / text_area ) * 0.7), 1, 1.1)
                    if len(blkitem.blk) > 1:
                        normalized_width_list = blkitem.blk.normalizd_width_list()
                        max_central_width = max(normalized_width_list)
                else:
                    resize_ratio = 1.1
            else:
                if ballon_area / text_area < 1.5:   # default eng->cjk font_size = 1.1 * detected_size, because detected eng bboxes are a bit small
                    # print(1.8 * text_area / ballon_area)
                    resize_ratio = max(ballon_area / 1.5 / text_area, 0.5)
                    

        if resize_ratio != 1:
            new_font_size = blk_font.pointSizeF() * resize_ratio
            blk_font.setPointSizeF(new_font_size)
            wl_list = (np.array(wl_list, np.float64) * resize_ratio).astype(np.int32).tolist()
            line_height = int(line_height * resize_ratio)
            text_w = int(text_w * resize_ratio)
            delimiter_len = int(delimiter_len * resize_ratio)

        if max_central_width != np.inf:
            max_central_width = max(int(max_central_width * text_w), 0.75 * region_rect[2])

        padding = pt2px(blk_font.pointSize()) + 20   # dummpy padding variable
        if fmt.alignment == 1:
            if len(blkitem.blk) > 0:
                centroid = blkitem.blk.center().astype(np.int64).tolist()
                centroid[0] -= mask_xyxy[0]
                centroid[1] -= mask_xyxy[1]
            else:
                centroid = [bounding_rect[2] // 2, bounding_rect[3] // 2]
        else:
            max_central_width = np.inf
            centroid = [0, 0]
            abs_centroid = [bounding_rect[0], bounding_rect[1]]
            if len(blkitem.blk) > 0:
                blkitem.blk.lines[0]
                abs_centroid = blkitem.blk.lines[0][0]
                centroid[0] = int(abs_centroid[0] - mask_xyxy[0])
                centroid[1] = int(abs_centroid[1] - mask_xyxy[1])

        new_text, xywh = layout_text(mask, mask_xyxy, centroid, words, wl_list, delimiter, delimiter_len, blkitem.blk.angle, line_height, fmt.alignment, fmt.vertical, 0, padding, max_central_width)

        # font size post adjustment
        post_resize_ratio = 1
        if adaptive_fntsize:
            downscale_constraint = 0.5
            w = xywh[2] - padding * 2
            post_resize_ratio = np.clip(max(region_rect[2] / w, downscale_constraint), 0, 1)
            resize_ratio *= post_resize_ratio

        if tgt_is_cjk:
            resize_ratio = 1
            post_resize_ratio = 1 / resize_ratio

        if post_resize_ratio != 1:
            cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
            w, h = xywh[2] * post_resize_ratio, xywh[3] * post_resize_ratio
            xywh = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

        if resize_ratio != 1:
            new_font_size = blkitem.font().pointSizeF() * resize_ratio
            blkitem.textCursor().clearSelection()
            set_textblk_fontsize(blkitem, new_font_size)


        scale = blkitem.scale()
        if scale != 1 and not fmt.alignment == 0:
            xywh = (np.array(xywh, np.float64) * scale).astype(np.int32).tolist()

        if fmt.alignment == 0:
            x_shift = (scale - 1) * xywh[2] // 2 + xywh[0] * scale
            y_shift = (scale - 1) * xywh[3] // 2 + xywh[1] * scale
            xywh[0] = int(abs_centroid[0] * scale) + x_shift
            xywh[1] = int(abs_centroid[1] * scale)  + y_shift

        if restore_charfmts:
            char_fmts = blkitem.get_char_fmts()        
        
        blkitem.setRect(xywh, repaint=False)
        blkitem.setPlainText(new_text)
        if len(self.pairwidget_list) > blkitem.idx:
            self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
        if restore_charfmts:
            self.restore_charfmts(blkitem, text, new_text, char_fmts)
    
    def restore_charfmts(self, blkitem: TextBlkItem, text: str, new_text: str, char_fmts: List[QTextCharFormat]):
        cursor = blkitem.textCursor()
        cpos = 0
        num_text = len(new_text)
        num_fmt = len(char_fmts)
        blkitem.layout.relayout_on_changed = False
        blkitem.repaint_on_changed = False
        for fmt_i in range(num_fmt):
            fmt = char_fmts[fmt_i]
            ori_char = text[fmt_i].strip()
            if ori_char == '':
                continue
            else:
                if cursor.atEnd():   
                    break
                matched = False
                while cpos < num_text:
                    if new_text[cpos] == ori_char:
                        matched = True
                        break
                    cpos += 1
                if matched:
                    cursor.clearSelection()
                    cursor.setPosition(cpos)
                    cursor.setPosition(cpos+1, QTextCursor.MoveMode.KeepAnchor)
                    cursor.setCharFormat(fmt)
                    cursor.setBlockCharFormat(fmt)
                    cpos += 1
        blkitem.repaint_on_changed = True
        blkitem.layout.relayout_on_changed = True
        blkitem.layout.reLayout()
        blkitem.repaint_background()


    def onEndCreateTextBlock(self, rect: QRectF):
        if rect.width() > 1 and rect.height() > 1:
            xyxy = np.array([rect.x(), rect.y(), rect.right(), rect.bottom()])        
            xyxy = np.round(xyxy).astype(np.int)
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

    def on_textedit_redo(self):
        self.canvasUndoStack.redo()

    def on_textedit_undo(self):
        self.canvasUndoStack.undo()

    def on_push_textitem_undostack(self, num_steps: int, is_formatting: bool):
        blkitem: TextBlkItem = self.sender()
        e_trans = self.pairwidget_list[blkitem.idx].e_trans if not is_formatting else None
        self.canvasUndoStack.push(TextItemEditCommand(blkitem, e_trans, num_steps))

    def on_push_edit_stack(self, num_steps: int):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        blkitem = self.textblk_item_list[edit.idx] if type(edit) == TransTextEdit else None
        self.canvasUndoStack.push(TextEditCommand(edit, num_steps, blkitem))

    def onTextItemUserEditedVerbose(self, pos: int, added_text: str, input_method_used: bool):
        blk_item: TextBlkItem = self.sender()
        idx = blk_item.idx
        edit = self.pairwidget_list[idx].e_trans

        ori_count = edit.document().characterCount()
        new_count = blk_item.document().characterCount()
        removed = ori_count + len(added_text) - new_count

        cursor = edit.textCursor()
        cursor.setPosition(pos)
        if removed > 0:
            cursor.setPosition(pos + removed, QTextCursor.MoveMode.KeepAnchor)
        if input_method_used:
            cursor.beginEditBlock()
        cursor.insertText((added_text))
        if input_method_used:
            cursor.endEditBlock()
        self.canvas.setProjSaveState(True)

    def onTransWidgetUserEditedVerbose(self, pos: int, added_text: str, input_method_used: bool):
        edit: TransTextEdit = self.sender()
        idx = edit.idx
        blk_item = self.textblk_item_list[idx]
        if blk_item.isEditing():
            blk_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)

        ori_count = blk_item.document().characterCount()
        new_count = edit.document().characterCount()
        removed = ori_count + len(added_text) - new_count

        cursor = blk_item.textCursor()
        cursor.setPosition(pos)
        if removed > 0:
            cursor.setPosition(pos + removed, QTextCursor.MoveMode.KeepAnchor)
        if input_method_used:
            cursor.beginEditBlock()
        cursor.insertText((added_text))
        if input_method_used:
            cursor.endEditBlock()
        self.canvas.setProjSaveState(True)

    def onGlobalFormatChanged(self):
        # if not isinstance(self.app.focusWidget(), PaintQSlider):
        self.apply_fontformat(self.formatpanel.global_format)

    def apply_fontformat(self, fontformat: FontFormat):
        selected_blks = self.get_selected_blkitems()
        if len(selected_blks) > 0:
            self.canvasUndoStack.push(ApplyFontformatCommand(selected_blks, fontformat))

    def on_apply_effect(self):
        format = self.formatpanel.active_format
        selected_blks = self.get_selected_blkitems()
        if len(selected_blks) > 0:
            self.canvasUndoStack.push(ApplyEffectCommand(selected_blks, format))

    def get_selected_blkitems(self) -> List[TextBlkItem]:
        selections = self.canvas.selectedItems()
        selected_blks = []
        for selection in selections:
            if isinstance(selection, TextBlkItem):
                selected_blks.append(selection)
        return selected_blks

    def on_srcwidget_edited(self):
        self.canvas.setProjSaveState(True)

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
                blk_item.blk.translation = blk_item.toPlainText()
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            blk_item.blk._bounding_rect = blk_item.absBoundingRect()
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def updateTranslation(self):
        for blk_item, transwidget in zip(self.textblk_item_list, self.pairwidget_list):
            transwidget.e_trans.setPlainText(blk_item.blk.translation)
            blk_item.setPlainText(blk_item.blk.translation)

    def showTextblkItemRect(self, draw_rect: bool):
        for blk_item in self.textblk_item_list:
            blk_item.draw_rect = draw_rect
            blk_item.update()

    def set_blkitems_selection(self, selected: bool, blk_items: List[TextBlkItem] = None):
        if blk_items is None:
            blk_items = self.textblk_item_list
        for blk_item in blk_items:
            blk_item.setSelected(selected)

    def on_ensure_textitem_svisible(self):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        self.changeHoveringWidget(edit)
        self.canvas.gv.ensureVisible(self.textblk_item_list[edit.idx])
        self.txtblkShapeControl.setBlkItem(self.textblk_item_list[edit.idx])

def get_text_size(fm: QFontMetrics, text: str) -> Tuple[int, int]:
    brt = fm.tightBoundingRect(text)
    br = fm.boundingRect(text)
    return br.width(), brt.height()
    
def get_words_length_list(fm: QFontMetrics, words: List[str]) -> List[int]:
    return [fm.tightBoundingRect(word).width() for word in words]