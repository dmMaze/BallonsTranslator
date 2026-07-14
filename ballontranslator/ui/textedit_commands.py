from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

from qtpy.QtGui import QColor, QFont, QTextCharFormat, QTextCursor, QTextOption
from qtpy.QtCore import QPointF, QRectF
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .textitem import TextBlkItem, TextBlock
from .textedit_area import TransTextEdit, SourceTextEdit
from ballontranslator.utils.fontformat import FontFormat, normalize_text_transform
import ballontranslator.utils.config as C
from .misc import doc_replace, doc_replace_no_shift
from .texteditshapecontrol import TextBlkShapeControl
from .page_search_widget import PageSearchWidget, Matched
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from .scene_textlayout import PUNSET_HALF


def utf16_code_unit_length(text: str) -> int:
    """Return the number of UTF-16 code units used by Qt text positions.

    Python indexes Unicode code points, while ``QTextCursor`` indexes UTF-16
    code units. Combining characters still occupy their own cursor position,
    and supplementary characters occupy two.

    >>> utf16_code_unit_length('')
    0
    >>> utf16_code_unit_length('A\U0001f600e\u0301')
    5
    """
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def propagate_user_edit(src_edit: Union[TransTextEdit, TextBlkItem], target_edit: Union[TransTextEdit, TextBlkItem], pos: int, added_text: str, joint_previous: bool = False):
    ori_count = target_edit.document().characterCount()
    new_count = src_edit.document().characterCount()
    removed = ori_count + utf16_code_unit_length(added_text) - new_count

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
        before: Sequence[Tuple[float, float, float]],
        after: Sequence[Tuple[float, float, float]],
        refresh_callback: Optional[Callable[[], None]] = None,
    ):
        super().__init__()
        self.items = tuple(items)
        if len(self.items) != len(before) or len(self.items) != len(after):
            raise ValueError("items, before, and after must have the same length")
        self.before = tuple(normalize_text_transform(*values) for values in before)
        self.after = tuple(normalize_text_transform(*values) for values in after)
        self.refresh_callback = refresh_callback

    @classmethod
    def create(
        cls,
        items: Sequence[TextBlkItem],
        before: Sequence[Tuple[float, float, float]],
        after: Sequence[Tuple[float, float, float]],
        refresh_callback: Optional[Callable[[], None]] = None,
    ) -> Optional["SetTextTransformCommand"]:
        """Build a command, or return ``None`` for a normalized no-op."""
        command = cls(items, before, after, refresh_callback)
        return None if command.before == command.after else command

    def _apply(self, transforms: Sequence[Tuple[float, float, float]]):
        for item, transform in zip(self.items, transforms):
            item.set_text_transform(*transform, preview=False)
        if self.refresh_callback is not None:
            self.refresh_callback()

    def redo(self):
        self._apply(self.after)

    def undo(self):
        self._apply(self.before)


class MoveBlkItemsCommand(QUndoCommand):
    def __init__(self, items: List[TextBlkItem], shape_ctrl: TextBlkShapeControl):
        super(MoveBlkItemsCommand, self).__init__()
        self.items = items
        self.old_pos_lst: List[QPointF] = []
        self.new_pos_lst: List[QPointF] = []
        self.shape_ctrl = shape_ctrl
        for item in items:
            padding = item.padding()
            padding = QPointF(padding, padding)
            self.old_pos_lst.append(item.oldPos + padding)
            self.new_pos_lst.append(item.pos() + padding)
            item.oldPos = item.pos()

    def redo(self):
        for item, new_pos in zip(self.items, self.new_pos_lst):
            padding = item.padding()
            padding = QPointF(padding, padding)
            item.setPos(new_pos - padding)
            if self.shape_ctrl.blk_item == item and self.shape_ctrl.pos() != new_pos:
                self.shape_ctrl.setPos(new_pos)

    def undo(self):
        for item, old_pos in zip(self.items, self.old_pos_lst):
            padding = item.padding()
            padding = QPointF(padding, padding)
            item.setPos(old_pos - padding)
            if self.shape_ctrl.blk_item == item and self.shape_ctrl.pos() != old_pos:
                self.shape_ctrl.setPos(old_pos)


class _FontformatItemState(NamedTuple):
    html: str
    fontformat: FontFormat
    logical_rect: QRectF
    item_position: QPointF
    cursor_position: int
    cursor_anchor: int
    default_font: QFont
    default_text_option: QTextOption
    cursor_char_format: QTextCharFormat
    cursor_block_char_format: QTextCharFormat


class ApplyFontformatCommand(QUndoCommand):
    """Apply a whole format to every selected item as one deterministic edit."""

    def __init__(
        self,
        items: List[TextBlkItem],
        trans_widget_lst: List[TransTextEdit],
        fontformat: FontFormat,
        shape_ctrl: Optional[TextBlkShapeControl] = None,
    ):
        super(ApplyFontformatCommand, self).__init__()
        if len(items) != len(trans_widget_lst):
            raise ValueError("items and trans_widget_lst must have the same length")
        self.items = tuple(items)
        self.trans_widget_lst = tuple(trans_widget_lst)
        self.new_fmt = fontformat.deepcopy()
        self.shape_ctrl = shape_ctrl
        self.before = tuple(self._capture_state(item) for item in self.items)
        self.after: Optional[Tuple[_FontformatItemState, ...]] = None
        self.document_undo_floors: Optional[Tuple[int, ...]] = None

    @staticmethod
    def _capture_state(item: TextBlkItem) -> _FontformatItemState:
        cursor = item.textCursor()
        return _FontformatItemState(
            item.toHtml(),
            item.fontformat.deepcopy(),
            QRectF(item.absBoundingRect(qrect=True)),
            QPointF(item.pos()),
            cursor.position(),
            cursor.anchor(),
            QFont(item.document().defaultFont()),
            QTextOption(item.document().defaultTextOption()),
            QTextCharFormat(cursor.charFormat()),
            QTextCharFormat(cursor.blockCharFormat()),
        )

    @staticmethod
    def _restore_cursor(
        item: TextBlkItem,
        state: _FontformatItemState,
        restore_empty_formats: bool = False,
    ):
        # Starting at the anchor preserves a reverse selection's orientation.
        cursor = QTextCursor(item.document())
        cursor.setPosition(state.cursor_anchor)
        cursor.setPosition(
            state.cursor_position,
            QTextCursor.MoveMode.KeepAnchor,
        )
        if restore_empty_formats and item.document().isEmpty():
            # Empty-document insertion and paragraph formats are cursor state,
            # not fragments. Qt does not restore them with document.undo(),
            # but setting them on an empty cursor is history-neutral.
            cursor.setBlockCharFormat(
                QTextCharFormat(state.cursor_block_char_format)
            )
            cursor.setCharFormat(QTextCharFormat(state.cursor_char_format))
        item.setTextCursor(cursor)

    def _restore_state(
        self,
        item: TextBlkItem,
        state: _FontformatItemState,
        document_undo_floor: int,
    ):
        was_in_redo_undo = item.in_redo_undo
        was_repainting = item.repainting
        item.in_redo_undo = True
        item.repainting = True
        try:
            # A QTextCursor edit block may account for several low-level
            # availableUndoSteps(), while document-wide properties can form a
            # separate top-level operation. Undo only operations above the
            # exact pre-command floor so an earlier TextItemEditCommand stays
            # paired with its translation editor.
            while item.document().availableUndoSteps() > document_undo_floor:
                undo_steps = item.document().availableUndoSteps()
                item.document().undo()
                if item.document().availableUndoSteps() >= undo_steps:
                    break
            item.fontformat.merge(state.fontformat)
            item.setVertical(state.fontformat.vertical)
            document = item.document()
            document.setDefaultFont(QFont(state.default_font))
            document.setDefaultTextOption(
                QTextOption(state.default_text_option)
            )
            item.layout.setLineSpacingType(state.fontformat.line_spacing_type)
            item.layout.setLineSpacing(state.fontformat.line_spacing)
            if state.fontformat.vertical:
                item.layout.setLetterSpacing(state.fontformat.letter_spacing)
            item.stroke_qcolor = QColor(*state.fontformat.stroke_color())
            item.setOpacity(state.fontformat.opacity)
            item.set_text_transform(*state.fontformat.text_transform)
            item.setRect(QRectF(state.logical_rect), repaint=False)
            item.blk._bounding_rect = item.absBoundingRect()
            self._restore_cursor(item, state, restore_empty_formats=True)
        finally:
            item.repainting = was_repainting
            item.in_redo_undo = was_in_redo_undo
        item.repaint_background()

    def _apply_new_format(self):
        document_undo_floors = []
        for item, before in zip(self.items, self.before):
            document = item.document()
            document_undo_floors.append(document.availableUndoSteps())
            was_in_redo_undo = item.in_redo_undo
            was_repainting = item.repainting
            item.in_redo_undo = True
            item.repainting = True
            try:
                history_cursor = QTextCursor(document)
                history_cursor.beginEditBlock()
                try:
                    item.set_fontformat(
                        self.new_fmt.deepcopy(),
                        set_char_format=True,
                    )
                    item.repainting = False
                    item.repaint_background()
                    self._restore_cursor(item, before)
                finally:
                    history_cursor.endEditBlock()
            finally:
                item.in_redo_undo = was_in_redo_undo
                item.repainting = was_repainting
                self._restore_cursor(item, before)
        if self.document_undo_floors is None:
            self.document_undo_floors = tuple(document_undo_floors)

    @staticmethod
    def _document_matches_target(
        item: TextBlkItem,
        target: FontFormat,
    ) -> bool:
        document = item.document()
        expected_font, expected_format, expected_option = (
            item._whole_fontformat_document_values(target.deepcopy())
        )
        if (
            document.defaultFont() != expected_font
            or document.defaultTextOption().alignment()
            != expected_option.alignment()
        ):
            return False

        def comparable(char_format: QTextCharFormat) -> QTextCharFormat:
            result = QTextCharFormat(char_format)
            if target.gradient_enabled:
                # Current gradient coordinates are derived QTextLayout state;
                # stale historical brush coordinates in HTML are irrelevant
                # to whole-format no-op detection.
                result.setForeground(QColor())
            return result

        expected_comparable = comparable(expected_format)
        block = item.document().firstBlock()
        while block.isValid():
            it = block.begin()
            has_fragment = False
            while not it.atEnd():
                has_fragment = True
                if comparable(it.fragment().charFormat()) != expected_comparable:
                    return False
                it += 1
            if not has_fragment:
                insertion_cursor = QTextCursor(block)
                if (
                    comparable(insertion_cursor.charFormat())
                    != expected_comparable
                ):
                    return False
            block = block.next()
        return (
            item.layout.line_spacing == target.line_spacing
            and item.layout.linespacing_type == target.line_spacing_type
            and (
                not target.vertical
                or item.layout.letter_spacing == target.letter_spacing
            )
            and item.opacity() == target.opacity
        )

    def _is_noop(self) -> bool:
        return all(
            item._text_transform_preview is None
            and state.fontformat == self.new_fmt
            # A live gradient is a QTextLayout override. QTextCharFormat's
            # foreground().color() is therefore not a meaningful model value
            # while that override is enabled; the exact document comparator
            # below validates every persisted whole-format document value.
            and (
                self.new_fmt.gradient_enabled
                or item.get_fontformat() == self.new_fmt
            )
            and self._document_matches_target(item, self.new_fmt)
            for item, state in zip(self.items, self.before)
        )

    def _finish(self):
        for item in self.items:
            item.updateUndoSteps()
        if (
            self.shape_ctrl is not None
            and self.shape_ctrl.blk_item in self.items
        ):
            self.shape_ctrl.updateBoundingRect()

    def redo(self):
        if self.after is None:
            if self._is_noop():
                self.after = self.before
                self.document_undo_floors = tuple(
                    item.document().availableUndoSteps()
                    for item in self.items
                )
                self.setObsolete(True)
                return
            self._apply_new_format()
            self.after = tuple(self._capture_state(item) for item in self.items)
        else:
            # Reapply the canonical target. QTextDocument can replay the
            # grouped character formats, but layout swaps and document-wide
            # defaults are not part of that cursor edit block.
            self._apply_new_format()
        self._finish()

    def undo(self):
        for item, state, document_undo_floor in zip(
            self.items, self.before, self.document_undo_floors
        ):
            self._restore_state(
                item,
                state,
                document_undo_floor,
            )
        self._finish()

    
class ReshapeItemCommand(QUndoCommand):
    def __init__(
        self,
        item: TextBlkItem,
        shape_ctrl: Optional[TextBlkShapeControl] = None,
    ):
        super(ReshapeItemCommand, self).__init__()
        self.item = item
        self.shape_ctrl = shape_ctrl
        self.oldRect = item.oldRect
        self.newRect = item.absBoundingRect(qrect=True)
        self.idx = -1

    def _refresh_shape_control(self):
        if self.shape_ctrl is not None and self.shape_ctrl.blk_item is self.item:
            self.shape_ctrl.updateBoundingRect()

    def redo(self):
        if self.idx < 0:
            self.idx += 1
            return
        self.item.setRect(self.newRect)
        self._refresh_shape_control()

    def undo(self):
        self.item.setRect(self.oldRect)
        self._refresh_shape_control()

    def mergeWith(self, command: QUndoCommand):
        item = command.item
        if self.item != item:
            return False
        self.newRect = item.absBoundingRect(qrect=True)
        return True


class RotateItemCommand(QUndoCommand):
    def __init__(self, item: Union[TextBlkItem, List[TextBlkItem]], new_angle: float = None, shape_ctrl: TextBlkShapeControl = None):
        super(RotateItemCommand, self).__init__()
        self.items = item if isinstance(item, list) else [item]
        self.items = [item for item in self.items if item is not None]
        self.item = self.items[0] if len(self.items) > 0 else None
        self.old_angles = [item.rotation() for item in self.items]
        if new_angle is None and self.item is not None:
            new_angle = self.item.angle
        self.new_angle = new_angle
        self.shape_ctrl = shape_ctrl

    def redo(self):
        for item in self.items:
            item.setAngle(self.new_angle)
            if self.shape_ctrl is not None and self.shape_ctrl.blk_item == item and self.shape_ctrl.rotation() != self.new_angle:
                self.shape_ctrl.setRotation(self.new_angle)

    def undo(self):
        for item, old_angle in zip(self.items, self.old_angles):
            item.setAngle(old_angle)
            if self.shape_ctrl is not None and self.shape_ctrl.blk_item == item and self.shape_ctrl.rotation() != old_angle:
                self.shape_ctrl.setRotation(old_angle)

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
    def __init__(self, blkitem_lst: List[TextBlkItem], ctrl: TextBlkShapeControl):
        super(ResetAngleCommand, self).__init__()
        self.blkitem_lst = blkitem_lst
        self.angle_lst = []
        self.ctrl = ctrl
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
            if self.ctrl.blk_item == blk:
                self.ctrl.setAngle(0)

    def undo(self):
        for blk, angle in zip(self.blkitem_lst, self.angle_lst):
            blk.setAngle(angle)
            if self.ctrl.blk_item == blk:
                self.ctrl.setAngle(angle)

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
