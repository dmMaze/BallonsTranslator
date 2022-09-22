from typing import List, Union, Tuple

from qtpy.QtCore import QObject, QRectF, Qt, Signal
from qtpy.QtGui import QTextCursor
try:
    from qtpy.QtWidgets import QUndoCommand, QUndoStack
except:
    from qtpy.QtGui import QUndoCommand, QUndoStack

from .textitem import TextBlkItem
from .textedit_area import TransTextEdit, SourceTextEdit
from .misc import FontFormat
from .texteditshapecontrol import TextBlkShapeControl
from .page_search_widget import PageSearchWidget


def propagate_user_edit(src_edit: Union[TransTextEdit, TextBlkItem], target_edit: Union[TransTextEdit, TextBlkItem], pos: int, added_text: str, input_method_used: bool):

    ori_count = target_edit.document().characterCount()
    new_count = src_edit.document().characterCount()
    removed = ori_count + len(added_text) - new_count

    cursor = target_edit.textCursor()
    if len(added_text) > 0:
        if removed > 0:
            cursor.setPosition(pos + removed, QTextCursor.MoveMode.KeepAnchor)
        if input_method_used:
            cursor.beginEditBlock()
        cursor.insertText((added_text))
        if input_method_used:
            cursor.endEditBlock()
    elif removed > 0:
        if removed == 1:
            cursor.setPosition(pos + removed - 1)
            cursor.deleteChar()
        else:
            cursor.setPosition(pos)
            cursor.setPosition(pos + removed, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()


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
    def __init__(self, item: TextBlkItem):
        super(ReshapeItemCommand, self).__init__()
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


class TextEditCommand(QUndoCommand):
    def __init__(self, edit: Union[SourceTextEdit, TransTextEdit], num_steps: int, blkitem: TextBlkItem) -> None:
        super().__init__()
        self.edit = edit
        self.blkitem = blkitem
        self.op_counter = 0
        self.num_steps = min(num_steps, 2)
        if edit.input_method_from == -1:
            self.num_steps = 1
        else:
            edit.input_method_from = -1

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        for _ in range(self.num_steps):
            self.edit.redo()
        if self.blkitem is not None:
            self.blkitem.document().redo()

    def undo(self):
        for _ in range(self.num_steps):
            self.edit.undo()
        if self.blkitem is not None:
            self.blkitem.document().undo()


class PageReplaceOneCommand(QUndoCommand):
    def __init__(self, se: PageSearchWidget, parent=None):
        super(PageReplaceOneCommand, self).__init__(parent)
        self.op_counter = 0
        self.sw = se
        self.reptxt = self.sw.replace_editor.toPlainText()
        self.repl_len = len(self.reptxt)
        
        self.sel_start = self.sw.current_cursor.selectionStart()
        self.oritxt = self.sw.current_cursor.selectedText()
        self.ori_len = len(self.oritxt)
        self.edit: Union[SourceTextEdit, TransTextEdit] = self.sw.current_edit
        self.edit_is_src = type(self.edit) == SourceTextEdit
        self.blkitem = self.sw.textblk_item_list[self.sw.current_edit.idx]

        if self.sw.current_edit is not None and self.sw.isVisible():
            move = self.sw.move_cursor(1)
            if move == 0:
                self.sw.result_pos = min(self.sw.counter_sum - 1, self.sw.result_pos + 1)
            else:
                self.sw.result_pos = 0

        if not self.edit_is_src:
            cursor = self.blkitem.textCursor()
            cursor.setPosition(self.sel_start)
            cursor.setPosition(self.sel_start+self.ori_len, QTextCursor.MoveMode.KeepAnchor)
            cursor.beginEditBlock()
            cursor.insertText(self.reptxt)
            cursor.endEditBlock()

        self.rep_cursor = self.edit.textCursor()
        self.rep_cursor.setPosition(self.sel_start)
        self.rep_cursor.setPosition(self.sel_start+self.ori_len, QTextCursor.MoveMode.KeepAnchor)
        self.rep_cursor.insertText(self.reptxt)
        self.edit.updateUndoSteps()

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        if self.sw.current_edit is not None and self.sw.isVisible():
            move = self.sw.move_cursor(1)
            if move == 0:
                self.sw.result_pos = min(self.sw.counter_sum - 1, self.sw.result_pos + 1)
            else:
                self.sw.result_pos = 0

        if not self.edit_is_src:
            self.blkitem.document().redo()
        self.edit.redo()

    def undo(self):
        if not self.edit_is_src:
            self.blkitem.document().undo()
        self.edit.undo()

        if self.sw.current_edit is not None and self.sw.isVisible():
            move = self.sw.move_cursor(-1)
            if move == 0:
                self.sw.result_pos = max(self.sw.result_pos - 1, 0)
            else:
                self.sw.result_pos = self.sw.counter_sum - 1
            self.sw.updateCounterText()

        self.edit.user_edited.emit()


class PageReplaceAllCommand(QUndoCommand):

    def __init__(self, search_widget: PageSearchWidget) -> None:
        super().__init__()
        self.op_counter = 0
        self.sw = search_widget

        self.rstedit_list: List[SourceTextEdit] = []
        self.blkitem_list: List[TextBlkItem] = []
        curpos_list = []
        for edit, highlighter in zip(self.sw.search_rstedit_list, self.sw.highlighter_list):
            self.rstedit_list.append(edit)
            curpos_list.append(list(highlighter.matched_map.keys()))

        text = self.sw.search_editor.toPlainText()
        len_text = len(text)
        replace = self.sw.replace_editor.toPlainText()
        len_delta = len(replace) - len_text
        for edit, curpos_lst in zip(self.rstedit_list, curpos_list):
            redo_blk = type(edit) == TransTextEdit
            if redo_blk:
                blkitem = self.sw.textblk_item_list[edit.idx]
                self.blkitem_list.append(blkitem)
                sel_list = []

            cursor = edit.textCursor()
            cursor.clearSelection()
            cursor.setPosition(0)
            cursor.beginEditBlock()
            for ii, sel_start in enumerate(curpos_lst):
                sel_start += len_delta * ii - len_text
                cursor.setPosition(sel_start)
                sel_end = sel_start+len_text
                cursor.setPosition(sel_end, QTextCursor.MoveMode.KeepAnchor)
                cursor.insertText(replace)
                if redo_blk:
                    sel_list.append([sel_start, sel_end])
            cursor.endEditBlock()
            edit.updateUndoSteps()

            if redo_blk:
                cursor = blkitem.textCursor()
                cursor.beginEditBlock()
                for sel in sel_list:
                    cursor.setPosition(sel[0])
                    cursor.setPosition(sel[1], QTextCursor.MoveMode.KeepAnchor)
                    cursor.insertText(replace)
                cursor.endEditBlock()

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        for edit in self.rstedit_list:
            edit.redo()
        for blkitem in self.blkitem_list:
            blkitem.document().redo()

    def undo(self):
        for edit in self.rstedit_list:
            edit.undo()
        for blkitem in self.blkitem_list:
            blkitem.document().undo()