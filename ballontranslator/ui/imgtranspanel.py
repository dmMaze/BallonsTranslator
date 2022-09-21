from typing import List, Union

from qtpy.QtWidgets import QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QFrame, QApplication
from qtpy.QtCore import Signal, Qt, QSize, QEvent
from qtpy.QtGui import QColor, QFocusEvent, QInputMethodEvent, QKeyEvent, QTextCursor
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .stylewidgets import Widget, SeparatorWidget
from .textitem import TextBlock, TextBlkItem
from .fontformatpanel import FontFormatPanel


class SourceTextEdit(QTextEdit):
    hover_enter = Signal(int)
    hover_leave = Signal(int)
    focus_in = Signal(int)
    user_edited = Signal()
    user_edited_verbose = Signal(int, str, bool)
    ensure_scene_visible = Signal()
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int)
    text_changed = Signal()

    def __init__(self, idx, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.idx = idx
        self.pre_editing = False
        self.setMinimumHeight(50)
        self.document().contentsChanged.connect(self.on_content_changed)
        self.document().documentLayout().documentSizeChanged.connect(self.adjustSize)
        self.document().contentsChange.connect(self.on_content_changing)
        self.setAcceptRichText(False)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)
        self.old_undo_steps = self.document().availableUndoSteps()
        self.in_redo_undo = False
        self.change_from: int = 0
        self.change_removed: int = 0
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_text = ''
        self.text_content_changed = False
        self.highlighting = False

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            self.text_content_changed = True
        if self.hasFocus() and not self.pre_editing:
            self.change_from = from_
            self.change_added = added
    
    def adjustSize(self):
        h = self.document().documentLayout().documentSize().toSize().height()
        self.setFixedHeight(max(h, 50))

    def on_content_changed(self):
        if self.text_content_changed:
            self.text_content_changed = False
            if not self.highlighting:
                self.text_changed.emit()
        if self.hasFocus() and not self.pre_editing and not self.highlighting:
            self.user_edited.emit()

            if not self.in_redo_undo:
                change_from = self.change_from
                added_text = ''
                input_method_used = False
                if self.input_method_from != -1:
                    added_text = self.input_method_text
                    change_from = self.input_method_from
                    input_method_used = True
        
                elif self.change_added > 0:
                    len_text = len(self.toPlainText())
                    cursor = self.textCursor()
                    
                    if self.change_added >  len_text:
                        self.change_added = 1
                        change_from = self.textCursor().position() - 1
                        input_method_used = True
                    cursor.setPosition(change_from)
                    cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor)
                    
                    added_text = cursor.selectedText()
                self.user_edited_verbose.emit(change_from, added_text, input_method_used)
                
                undo_steps = self.document().availableUndoSteps()
                new_steps = undo_steps - self.old_undo_steps
                if new_steps > 0:
                    self.old_undo_steps = undo_steps
                    self.push_undo_stack.emit(new_steps)

    def setHoverEffect(self, hover: bool):
        try:
            if hover:
                se = QGraphicsDropShadowEffect()
                se.setBlurRadius(12)
                se.setOffset(0, 0)
                se.setColor(QColor(30, 147, 229))
                self.setGraphicsEffect(se)
            else:
                self.setGraphicsEffect(None)
        except RuntimeError:
            pass

    def enterEvent(self, event: QEvent) -> None:
        self.setHoverEffect(True)
        self.hover_enter.emit(self.idx)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.setHoverEffect(False)
        self.hover_leave.emit(self.idx)
        return super().leaveEvent(event)

    def focusInEvent(self, event: QFocusEvent) -> None:
        self.setHoverEffect(True)
        self.focus_in.emit(self.idx)
        self.pre_editing = False
        return super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.setHoverEffect(False)
        return super().focusOutEvent(event)

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            if self.pre_editing is False:
                cursor = self.textCursor()
                self.input_method_from = cursor.selectionStart()
            self.pre_editing = True
        self.input_method_event = e
        super().inputMethodEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.undo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_Y:
                e.accept()
                self.redo_signal.emit()
                return
        return super().keyPressEvent(e)

    def undo(self) -> None:
        self.in_redo_undo = True
        self.document().undo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

    def redo(self) -> None:
        self.in_redo_undo = True
        self.document().redo()
        self.in_redo_undo = False
        self.old_undo_steps = self.document().availableUndoSteps()

        
class TransTextEdit(SourceTextEdit):
    pass


class TransPairWidget(Widget):
    def __init__(self, textblock: TextBlock = None, idx: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_source = SourceTextEdit(idx, self)
        self.e_trans = TransTextEdit(idx, self)
        self.textblock = textblock
        self.idx = idx
        vlayout = QVBoxLayout(self)
        vlayout.setAlignment(Qt.AlignTop)
        vlayout.addWidget(self.e_source)
        vlayout.addWidget(self.e_trans)
        vlayout.addWidget(SeparatorWidget(self))
        vlayout.setSpacing(14)

    def updateIndex(self, idx):
        self.idx = idx
        self.e_source.idx = idx
        self.e_trans.idx = idx

class TextEditListScrollArea(QScrollArea):
    textblock_list: List[TextBlock] = []
    pairwidget_list: List[TransPairWidget] = []
    remove_textblock = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scrollContent = QFrame()
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignTop)
        vlayout.setSpacing(0)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.vlayout = vlayout
        
    def addPairWidget(self, pairwidget: TransPairWidget):
        self.vlayout.addWidget(pairwidget)
        pairwidget.setVisible(True)

    def removeWidget(self, widget: TransPairWidget):
        widget.setVisible(False)
        self.vlayout.removeWidget(widget)


class TextPanel(Widget):
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textEditList = TextEditListScrollArea(self)
        self.activePair: TransPairWidget = None
        self.formatpanel = FontFormatPanel(app, self)
        layout.addWidget(self.formatpanel)
        layout.addWidget(self.textEditList)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


class TextEditCommand(QUndoCommand):
    def __init__(self, edit: Union[SourceTextEdit, TransTextEdit], blkitem: TextBlkItem, num_steps: int) -> None:
        super().__init__()
        self.edit = edit
        self.blkitem = blkitem
        self.op_counter = 0
        self.change_from = edit.change_from
        self.change_added = edit.change_added
        self.change_removed = edit.change_removed
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
        self.blkitem.document().redo()

    def undo(self):
        for _ in range(self.num_steps):
            self.edit.undo()
        self.blkitem.document().undo()

