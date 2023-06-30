from typing import List, Union

from qtpy.QtWidgets import QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QFrame, QApplication, QHBoxLayout 
from qtpy.QtCore import Signal, Qt, QSize, QEvent, QPoint
from qtpy.QtGui import QColor, QFocusEvent, QInputMethodEvent, QKeyEvent, QTextCursor, QMouseEvent, QKeySequence
import keyboard

from .stylewidgets import Widget, SeparatorWidget, ClickableLabel
from .textitem import TextBlock
from .fontformatpanel import FontFormatPanel
from .misc import ProgramConfig
import webbrowser

class SelectTextMiniMenu(Widget):

    block_current_editor = Signal(bool)

    def __init__(self, app: QApplication, config: ProgramConfig, parent=None, *args, **kwargs) -> None:
        super().__init__(parent=parent, *args, **kwargs)
        self.app = app
        self.config = config
        self.search_internet_btn = ClickableLabel(parent=self)
        self.search_internet_btn.setObjectName("SearchInternet")
        self.search_internet_btn.setToolTip(self.tr("Search selected text on Internet"))
        self.search_internet_btn.clicked.connect(self.on_search_internet)
        self.saladict_btn = ClickableLabel(parent=self)
        self.saladict_btn.setObjectName("SalaDict")
        self.saladict_btn.clicked.connect(self.on_saladict)
        self.saladict_btn.setToolTip(self.tr("Look up selected text in SalaDict, see installation guide in configpanel"))
        layout = QHBoxLayout(self)
        layout.addWidget(self.saladict_btn)
        layout.addWidget(self.search_internet_btn)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.selected_text = ''

    def on_search_internet(self):
        browser = webbrowser.get()
        browser.open_new(self.config.search_url + self.selected_text)
        self.hide()

    def on_saladict(self):
        self.app.clipboard().setText(self.selected_text)
        self.block_current_editor.emit(True)
        keyboard.press(self.config.saladict_shortcut)
        keyboard.release(self.config.saladict_shortcut)
        self.block_current_editor.emit(False)
        self.hide()


class SourceTextEdit(QTextEdit):
    hover_enter = Signal(int)
    hover_leave = Signal(int)
    focus_in = Signal(int)
    propagate_user_edited = Signal(int, str, bool)
    ensure_scene_visible = Signal()
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int)
    text_changed = Signal()
    show_select_menu = Signal(QPoint, str)
    focus_out = Signal()

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
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_text = ''
        self.text_content_changed = False
        self.highlighting = False
        self.paste_flag = False

        self.selected_text = ''
        self.cursorPositionChanged.connect(self.on_cursorpos_changed)

        self.cursor_coord = None
        self.block_all_input = False

        self.in_acts = False

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        acts = menu.actions()
        self.in_acts = True
        rst = menu.exec_(event.globalPos())

        # future actions orders changes could break these comparsion
        self.paste_flag = rst == acts[5]
        if self.paste_flag or rst == acts[3] or rst == acts[6]:
            self.handle_content_change()
        self.in_acts = False

    def on_cursorpos_changed(self) -> None:
        cursor = self.textCursor()
        if cursor.hasSelection():
            self.selected_text = cursor.selectedText()
            crect = self.cursorRect()
            if cursor.selectionStart() == cursor.position():
                self.cursor_coord = crect.bottomLeft()
            else:
                self.cursor_coord = crect.bottomRight()
        else:
            if self.cursor_coord is not None:
                self.show_select_menu.emit(QPoint(), '')
            self.cursor_coord = None

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        super().mouseReleaseEvent(e)
        if e.button() == Qt.MouseButton.LeftButton:
            if self.hasFocus():
                if self.cursor_coord is not None:
                    pos = self.mapToGlobal(self.cursor_coord)
                    sel_text = self.selected_text
                    self.show_select_menu.emit(pos, sel_text)

    def block_all_signals(self, block: bool):
        self.blockSignals(block)
        self.document().blockSignals(block)

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            self.text_content_changed = True
            if self.hasFocus():
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
                
        if self.hasFocus() and not self.pre_editing and not self.highlighting and not self.in_acts:
            self.handle_content_change()

    def handle_content_change(self):
        if not self.in_redo_undo:
            
            change_from = self.change_from
            added_text = ''
            input_method_used = False
            
            if self.paste_flag:
                self.paste_flag = False
                cursor = self.textCursor()
                cursor.setPosition(change_from)
                cursor.setPosition(self.textCursor().position(), QTextCursor.MoveMode.KeepAnchor)
                added_text = cursor.selectedText()
            
            else:
                if self.input_method_from != -1:
                    added_text = self.input_method_text
                    change_from = self.input_method_from
                    input_method_used = True
                elif self.change_added > 0:
                    text = self.toPlainText()
                    len_text = len(text)
                    cursor = self.textCursor()
                    
                    if self.change_added >  len_text or change_from + self.change_added > len_text:
                        self.change_added = 1
                        change_from = self.textCursor().position() - 1
                        cursor.setPosition(change_from)
                        cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor)
                        added_text = cursor.selectedText()
                        if added_text == '…' or added_text == '—':
                                self.change_added = 2
                                change_from -= 1
                    cursor.setPosition(change_from)
                    cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor) 
                    added_text = cursor.selectedText()

            self.propagate_user_edited.emit(change_from, added_text, input_method_used)
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
        self.focus_out.emit()
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
        super().inputMethodEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if self.block_all_input:
            e.setAccepted(True)
            return

        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.undo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_Y:
                e.accept()
                self.redo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_V:
                self.paste_flag = True
                return super().keyPressEvent(e)
        elif e.key() == Qt.Key.Key_Return:
            e.accept()
            self.textCursor().insertText('\n')
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

    def setPlainTextAndKeepUndoStack(self, text: str):
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.SelectionType.Document)
        cursor.insertText(text)

        
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
        layout.setContentsMargins(0, 0, 5, 0)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

