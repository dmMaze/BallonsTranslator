from typing import Callable, List, Optional, Sequence, Tuple, Union

from qtpy.QtGui import QTextCursor
from qtpy.QtCore import QPointF
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .textitem import TextBlkItem, TextBlock
from .textedit_area import TransTextEdit, SourceTextEdit
from ballontranslator.utils.fontformat import (
    FontFormat,
    TextTransform,
    coerce_text_transform,
)
import ballontranslator.utils.config as C
from .misc import doc_replace, doc_replace_no_shift
from .texteditshapecontrol import TextBlkShapeControl
from .page_search_widget import PageSearchWidget, Matched
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from .scene_textlayout import PUNSET_HALF


def propagate_user_edit(src_edit: Union[TransTextEdit, TextBlkItem], target_edit: Union[TransTextEdit, TextBlkItem], pos: int, added_text: str, joint_previous: bool = False):
    ori_count = target_edit.document().characterCount()
    new_count = src_edit.document().characterCount()
    removed = ori_count + len(added_text) - new_count

    cursor = target_edit.textCursor()
    cursor.setPosition(pos)
    if joint_previous:
        cursor.joinPreviousEditBlock()
    else:
        cursor.beginEditBlock()
    if removed > 0:
        cursor.setPosition(pos + removed, QTextCursor.MoveMode.KeepAnchor)
    cursor.insertText(added_text)
    cursor.endEditBlock()
    target_edit.old_undo_steps = target_edit.document().availableUndoSteps()


class SetTextTransformCommand(QUndoCommand):
    """Atomically apply canonical text transforms to one or more items."""

    def __init__(
        self,
        items: Sequence[TextBlkItem],
        before: Sequence[TextTransform],
        after: Sequence[TextTransform],
        refresh_callback: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.items = tuple(items)
        if len(self.items) != len(before) or len(self.items) != len(after):
            raise ValueError("items, before, and after must have the same length")
        self.before = tuple(coerce_text_transform(values) for values in before)
        self.after = tuple(coerce_text_transform(values) for values in after)
        self.refresh_callback = refresh_callback

    @classmethod
    def create(
        cls,
        items: Sequence[TextBlkItem],
        before: Sequence[TextTransform],
        after: Sequence[TextTransform],
        refresh_callback: Optional[Callable[[], None]] = None,
    ) -> Optional["SetTextTransformCommand"]:
        """Build a command, or return ``None`` for a normalized no-op."""
        command = cls(items, before, after, refresh_callback)
        return None if command.before == command.after else command

    def _apply(self, transforms: Sequence[TextTransform]):
        for item, transform in zip(self.items, transforms):
            item.set_text_transform(transform, preview=False)
        if self.refresh_callback is not None:
            self.refresh_callback()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class MoveBlkItemsCommand(QUndoCommand):
    def __init__(
        self,
        items: List[TextBlkItem],
        before_positions: Optional[Sequence[QPointF]] = None,
        after_positions: Optional[Sequence[QPointF]] = None,
    ):
        super(MoveBlkItemsCommand, self).__init__()
        self.items = list(items)
        self.old_pos_lst: List[QPointF] = []
        self.new_pos_lst: List[QPointF] = []
        if before_positions is not None and len(before_positions) != len(self.items):
            raise ValueError('items and before_positions must have the same length')
        if after_positions is not None and len(after_positions) != len(self.items):
            raise ValueError('items and after_positions must have the same length')
        for index, item in enumerate(self.items):
            logical = item.logical_position()
            logical_offset = logical - item.pos()
            before = (
                QPointF(before_positions[index])
                if before_positions is not None
                else item.oldPos + logical_offset
            )
            after = (
                QPointF(after_positions[index])
                if after_positions is not None
                else logical
            )
            self.old_pos_lst.append(before)
            self.new_pos_lst.append(after)
            item.oldPos = item.pos()
        if self.old_pos_lst == self.new_pos_lst:
            self.setObsolete(True)

    def _apply(self, positions: Sequence[QPointF]):
        for item, position in zip(self.items, positions):
            item.set_logical_position(position)
            item.oldPos = item.pos()

    def redo(self):
        self._apply(self.new_pos_lst)

    def undo(self):
        self._apply(self.old_pos_lst)


class ApplyFontformatCommand(QUndoCommand):
    """Apply one captured whole-format value to the selected text items."""

    def __init__(
        self,
        items: List[TextBlkItem],
        trans_widget_lst: List[TransTextEdit],
        fontformat: FontFormat,
    ):
        super(ApplyFontformatCommand, self).__init__()
        self.items = items
        self.old_html_lst = []
        self.old_rect_lst = []
        self.old_fmt_lst = []
        # Redo must replay the format that was applied when the command was
        # created, even if the live global/preset format changes afterwards.
        self.new_fmt = fontformat.deepcopy()
        self.trans_widget_lst = trans_widget_lst
        for item in items:
            self.old_html_lst.append(item.toHtml())
            # get_fontformat() deep-copies FontFormat, including its complete
            # typed transform, so whole-format undo restores it as well.
            self.old_fmt_lst.append(item.get_fontformat())
            self.old_rect_lst.append(item.absBoundingRect(qrect=True))

    def redo(self):
        for item, edit in zip(self.items, self.trans_widget_lst):
            item.set_fontformat(self.new_fmt, set_char_format=True)
            edit.document().clearUndoRedoStacks()

    def undo(self):
        for rect, item, html, fmt, edit in zip(
            self.old_rect_lst,
            self.items,
            self.old_html_lst,
            self.old_fmt_lst,
            self.trans_widget_lst,
        ):
            item.setHtml(html)
            item.set_fontformat(fmt)
            item.setRect(rect)
            edit.document().clearUndoRedoStacks()

    
class ReshapeItemCommand(QUndoCommand):
    def __init__(self, item: TextBlkItem):
        super(ReshapeItemCommand, self).__init__()
        self.item = item
        self.oldRect = item.oldRect
        self.newRect = item.absBoundingRect(qrect=True)
        self.idx = -1

    def redo(self):
        if self.idx < 0:
            self.idx += 1
            return
        self.item.setRect(self.newRect)

    def undo(self):
        self.item.setRect(self.oldRect)

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.newRect = item.absBoundingRect(qrect=True)
        return True


class RotateItemCommand(QUndoCommand):
    def __init__(
        self,
        item: Union[TextBlkItem, List[TextBlkItem]],
        new_angle: float = None,
    ):
        super(RotateItemCommand, self).__init__()
        self.items = item if isinstance(item, list) else [item]
        self.items = [item for item in self.items if item is not None]
        self.item = self.items[0] if len(self.items) > 0 else None
        self.old_angles = [item.rotation() for item in self.items]
        if new_angle is None and self.item is not None:
            new_angle = self.item.angle
        self.new_angle = new_angle

    def redo(self):
        for item in self.items:
            item.setAngle(self.new_angle)

    def undo(self):
        for item, old_angle in zip(self.items, self.old_angles):
            item.setAngle(old_angle)

    def mergeWith(self, command: QUndoCommand):
        if not isinstance(command, RotateItemCommand):
            return False
        if self.items != command.items:
            return False
        self.new_angle = command.new_angle
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
            self.new_rect_lst.append(item.absBoundingRect(qrect=True))
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
            if item.fontformat.letter_spacing != 1:
                item.setLetterSpacing(item.fontformat.letter_spacing, force=True)
            
    def undo(self):
        for item, trans_widget, html, rect  in zip(self.items, self.trans_widget_lst, self.old_html_lst, self.old_rect_lst):
            trans_widget.setPlainText(item.toPlainText())
            item.setPlainText('')
            item.setRect(rect, repaint=False)
            item.setHtml(html)
            if item.fontformat.letter_spacing != 1:
                item.setLetterSpacing(item.fontformat.letter_spacing, force=True)


class SqueezeCommand(QUndoCommand):
    def __init__(self, blkitem_lst: List[TextBlkItem], ctrl: TextBlkShapeControl):
        super(SqueezeCommand, self).__init__()
        self.blkitem_lst = blkitem_lst
        self.old_rect_lst = []
        self.ctrl = ctrl
        for item in blkitem_lst:
            self.old_rect_lst.append(item.absBoundingRect(qrect=True))
    
    def redo(self):
        for blk in self.blkitem_lst:
            blk.squeezeBoundingRect()

    def undo(self):
        for blk, rect in zip(self.blkitem_lst, self.old_rect_lst):
            blk.setRect(rect, repaint=True)
            if blk.under_ctrl:
                self.ctrl.updateBoundingRect()

class ResetAngleCommand(QUndoCommand):
    def __init__(self, blkitem_lst: List[TextBlkItem]):
        super(ResetAngleCommand, self).__init__()
        self.blkitem_lst = blkitem_lst
        self.angle_lst = []
        blkitem_lst = []
        for blk in self.blkitem_lst:
            rotation = blk.rotation()
            if rotation != 0:
                self.angle_lst.append(rotation)
                blkitem_lst.append(blk)
        self.blkitem_lst = blkitem_lst
    
    def redo(self):
        for blk in self.blkitem_lst:
            blk.setAngle(0)

    def undo(self):
        for blk, angle in zip(self.blkitem_lst, self.angle_lst):
            blk.setAngle(angle)

class TextItemEditCommand(QUndoCommand):
    def __init__(self, blkitem: TextBlkItem, trans_edit: TransTextEdit, num_steps: int, formatpanel=None):
        super(TextItemEditCommand, self).__init__()
        self.op_counter = 0
        self.edit = trans_edit
        self.blkitem = blkitem
        self.num_steps = num_steps
        self.is_formatting = blkitem.is_formatting
        self.old_ffmt_values = self.new_ffmt_values = None
        if blkitem.is_formatting and blkitem.old_ffmt_values is not None:
            self.old_ffmt_values = blkitem.old_ffmt_values.copy()
            self.new_ffmt_values = self.old_ffmt_values.copy()
            for k in self.new_ffmt_values:
                self.new_ffmt_values[k] = getattr(blkitem.fontformat, k)
        self.formatpanel = formatpanel

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        
        self.blkitem.repaint_on_changed = False
        if self.new_ffmt_values is not None:
            for k, v in self.new_ffmt_values.items():
                self.blkitem.fontformat[k] = v
        self.blkitem.redo()
        self.blkitem.repaint_on_changed = True
        if self.num_steps > 0:
            self.blkitem.repaint_background()

        if self.is_formatting and self.blkitem == self.formatpanel.textblk_item:
            multi_size = not self.blkitem.isEditing() and self.blkitem.isMultiFontSize()
            self.formatpanel.set_active_format(self.blkitem.get_fontformat(), multi_size)

        if self.edit is not None and not self.is_formatting:
            self.edit.redo()

    def undo(self):
        self.blkitem.repaint_on_changed = False
        if self.old_ffmt_values is not None:
            for k, v in self.old_ffmt_values.items():
                self.blkitem.fontformat[k] = v
        self.blkitem.undo()
        self.blkitem.repaint_on_changed = True
        if self.num_steps > 0:
            self.blkitem.repaint_background()

        if self.is_formatting and self.blkitem == self.formatpanel.textblk_item:
            multi_size = not self.blkitem.isEditing() and self.blkitem.isMultiFontSize()
            self.formatpanel.set_active_format(self.blkitem.get_fontformat(), multi_size)

        if self.edit is not None:
            self.edit.undo()


class TextEditCommand(QUndoCommand):
    def __init__(self, edit: Union[SourceTextEdit, TransTextEdit], num_steps: int, blkitem: TextBlkItem) -> None:
        super().__init__()
        # TODO: remove it for transtextedit
        self.edit = edit
        self.blkitem = blkitem
        self.op_counter = 0
        self.num_steps = num_steps

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        self.edit.redo()
        if self.blkitem is not None:
            self.blkitem.redo()

    def undo(self):
        self.edit.undo()
        if self.blkitem is not None:
            self.blkitem.undo()


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
            self.blkitem.redo()
        self.edit.redo()

    def undo(self):
        if not self.edit_is_src:
            self.blkitem.undo()
        self.sw.update_cursor_on_insert = False
        self.edit.undo()
        self.sw.update_cursor_on_insert = True
        if self.sw.current_edit is not None and self.sw.isVisible():
            move = self.sw.move_cursor(-1)
            if move == 0:
                self.sw.result_pos = max(self.sw.result_pos - 1, 0)
            else:
                self.sw.result_pos = self.sw.counter_sum - 1
            self.sw.updateCounterText()


class PageReplaceAllCommand(QUndoCommand):

    def __init__(self, search_widget: PageSearchWidget) -> None:
        super().__init__()
        self.op_counter = 0
        self.sw = search_widget

        self.rstedit_list: List[SourceTextEdit] = []
        self.blkitem_list: List[TextBlkItem] = []
        curpos_list: List[List[Matched]] = []
        for edit, highlighter in zip(self.sw.search_rstedit_list, self.sw.highlighter_list):
            self.rstedit_list.append(edit)
            curpos_list.append(list(highlighter.matched_map.values()))

        replace = self.sw.replace_editor.toPlainText()
        for edit, curpos_lst in zip(self.rstedit_list, curpos_list):
            redo_blk = type(edit) == TransTextEdit
            if redo_blk:
                blkitem = self.sw.textblk_item_list[edit.idx]
                self.blkitem_list.append(blkitem)
            span_list = [[matched.start, matched.end] for matched in curpos_lst]
            sel_list = doc_replace(edit.document(), span_list, replace)
            if redo_blk:
                doc_replace_no_shift(blkitem.document(), sel_list, replace)
                blkitem.updateUndoSteps()

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        for edit in self.rstedit_list:
            edit.redo()
        for blkitem in self.blkitem_list:
            blkitem.redo()

    def undo(self):
        for edit in self.rstedit_list:
            edit.undo()
        for blkitem in self.blkitem_list:
            blkitem.undo()


class GlobalRepalceAllCommand(QUndoCommand):
    def __init__(self, sceneitem_list: dict, background_list: dict, target_text: str, proj: ProjImgTrans) -> None:
        super().__init__()
        self.op_counter = -1
        self.target_text = target_text
        self.proj = proj
        self.trans_list = sceneitem_list['trans']
        self.src_list = sceneitem_list['src']
        self.btrans_list = background_list['trans']
        self.bsrc_list = background_list['src']

        for trans_dict in self.trans_list:
            edit: TransTextEdit = trans_dict['edit']
            item: TextBlkItem = trans_dict['item']
            matched_map = trans_dict['matched_map']
            sel_list = doc_replace(edit.document(), matched_map, target_text)

            doc_replace_no_shift(item.document(), sel_list, target_text)
            item.updateUndoSteps()
            item.updateUndoSteps()

            trans_dict.pop('matched_map')

        for src_dict in self.src_list:
            edit: SourceTextEdit = src_dict['edit']
            edit.setPlainTextAndKeepUndoStack(src_dict['replace'])
            edit.updateUndoSteps()
            src_dict.pop('replace')

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return

        for trans_dict in self.trans_list:
            edit: TransTextEdit = trans_dict['edit']
            item: TextBlkItem = trans_dict['item']
            edit.redo()
            item.redo()

        for src_dict in self.src_list:
            edit: SourceTextEdit = src_dict['edit']
            edit.redo()

        for trans_dict in self.btrans_list:
            blk: TextBlock = self.proj.pages[trans_dict['pagename']][trans_dict['idx']]
            blk.translation = trans_dict['replace']
            blk.rich_text = trans_dict['replace_html']

        for src_dict in self.bsrc_list:
            blk: TextBlock = self.proj.pages[src_dict['pagename']][src_dict['idx']]
            blk.text = src_dict['replace']

    def undo(self):
        for trans_dict in self.trans_list:
            edit: TransTextEdit = trans_dict['edit']
            item: TextBlkItem = trans_dict['item']
            edit.undo()
            item.undo()

        for src_dict in self.src_list:
            edit: SourceTextEdit = src_dict['edit']
            edit.undo()

        for trans_dict in self.btrans_list:
            blk: TextBlock = self.proj.pages[trans_dict['pagename']][trans_dict['idx']]
            blk.translation = trans_dict['ori']
            blk.rich_text = trans_dict['ori_html']

        for src_dict in self.src_list:
            blk: TextBlock = self.proj.pages[src_dict['pagename']][src_dict['idx']]
            blk.text = src_dict['ori']


class MultiPasteCommand(QUndoCommand):
    def __init__(self, text_list: Union[str, List], blkitems: List[TextBlkItem], etrans: List[TransTextEdit]) -> None:
        super().__init__()
        self.op_counter = -1
        self.blkitems = blkitems
        self.etrans = etrans

        if len(blkitems) > 0:
            if isinstance(text_list, str):
                text_list = [text_list] * len(blkitems)

        for blkitem, etran, text in zip(self.blkitems, self.etrans, text_list):
            etran.setPlainTextAndKeepUndoStack(text)
            blkitem.setPlainTextAndKeepUndoStack(text)

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        for blkitem, etran in zip(self.blkitems, self.etrans):
            blkitem.redo()
            etran.redo()

    def undo(self):
        for blkitem, etran in zip(self.blkitems, self.etrans):
            blkitem.undo()
            etran.undo()
