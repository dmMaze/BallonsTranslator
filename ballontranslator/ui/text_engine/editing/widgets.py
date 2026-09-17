import threading
import traceback
from typing import List

from qtpy.QtWidgets import QStackedWidget, QSizePolicy, QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QApplication, QHBoxLayout, QLabel, QLineEdit, QWidget, QPushButton, QFrame
from qtpy.QtCore import Signal, Qt, QEvent, QPoint, QSize, QTimer
from qtpy.QtGui import QContextMenuEvent, QIntValidator, QColor, QFocusEvent, QInputMethodEvent, QKeyEvent, QKeySequence, QTextCursor, QMouseEvent, QPainter, QPen
import numpy as np

from ...custom_widget import ScrollBar, Widget, SeparatorWidget
from ..item import TextBlock
from .context_menu import create_text_edit_context_menu
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ...spellcheck import (
    SpellCheckHighlighter,
    SpellCheckManager,
    iter_spellcheck_words,
)


STYLE_TRANSPAIR_CHECKED = (
    "background-color: rgba(30, 147, 229, 20%);"
    # Checked reinforces state with an accent border — same color the
    # editors' focus border and the drag drop indicator use.
    "border: 1px solid rgb(30, 147, 229);"
)
# Accent used by the row-drag drop indicator — same color the old
# top/bottom border hint (removed with the QDrag implementation) showed.
DRAG_INDICATOR_COLOR = QColor(30, 147, 229)


class FloatingSuggestionLabel(QWidget):
    def __init__(self, editor):
        super().__init__(editor, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.editor = editor
        self.setObjectName("suggestion_popup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Horizontal scroll area for suggestions only
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(28)
        self.scroll_area.wheelEvent = self.wheelEvent
        
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.scroll_content.setStyleSheet("background: transparent;")
        
        self.buttons_layout = QHBoxLayout(self.scroll_content)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setSpacing(0)
        
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)
        
        # Add static "Add to Dict" button outside scroll area
        self.add_dict_btn = QPushButton(self)
        self.add_dict_btn.setObjectName("add_to_dict")
        self.add_dict_btn.clicked.connect(self.add_to_dict)
        self.add_dict_btn.setFixedHeight(28)
        self.add_dict_btn.setMouseTracking(True)
        self.add_dict_btn.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.main_layout.addWidget(self.add_dict_btn)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        # Hide the suggestion popup automatically when application focus changes
        app = QApplication.instance()
        if app:
            app.focusChanged.connect(self.on_focus_changed)
            app.installEventFilter(self)
            
        self.hide()

    def eventFilter(self, watched, event):
        app = QApplication.instance()
        if watched is not app:
            return super().eventFilter(watched, event)
        try:
            if event.type() == QEvent.Type.ApplicationDeactivate:
                self.hide()
        except RuntimeError:
            pass
        return super().eventFilter(watched, event)

    def on_focus_changed(self, old_widget, new_widget):
        try:
            if not new_widget:
                # Application lost focus (user switched to another application)
                self.hide()
            elif new_widget is not self.editor and not self.isAncestorOf(new_widget):
                # User focused another widget within our application
                self.hide()
        except RuntimeError:
            # Widget or editor might be partially deleted/garbage collected during teardown
            pass

    def wheelEvent(self, event):
        scrollbar = self.scroll_area.horizontalScrollBar()
        if scrollbar:
            delta = event.angleDelta().y() or event.angleDelta().x()
            scrollbar.setValue(scrollbar.value() - delta // 2)
            event.accept()

    def set_suggestions(self, cursor, word, suggestions):
        self.cursor = cursor
        self.word = word

        is_dark = pcfg.darkmode
        border_color = "rgba(255, 255, 255, 12%)" if is_dark else "rgba(0, 0, 0, 12%)"
        hover_bg = "rgba(255, 255, 255, 16%)" if is_dark else "rgba(0, 0, 0, 10%)"
        
        while self.buttons_layout.count() > 0:
            item = self.buttons_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
                
        for i, sug in enumerate(suggestions):
            btn = QPushButton(sug, self.scroll_content)
            btn.setProperty('suggestion', sug)
            btn.clicked.connect(self._apply_clicked_suggestion)
            self.buttons_layout.addWidget(btn)
            
            # Stylize borders and round corners so they form a single seamless block
            border_right = "none" if i == len(suggestions) - 1 else f"1px solid {border_color}"
            left_radius = "4px" if i == 0 else "0px"
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    border-right: {border_right};
                    border-top-left-radius: {left_radius};
                    border-bottom-left-radius: {left_radius};
                    border-top-right-radius: 0px;
                    border-bottom-right-radius: 0px;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-color: {hover_bg};
                }}
            """)

        # Update static Add button
        self.add_dict_btn.setText(f"+ {self.tr('Add')}")
        left_radius = "4px" if len(suggestions) == 0 else "0px"
        border_left = "none" if len(suggestions) == 0 else f"1px solid {border_color}"
        self.add_dict_btn.setStyleSheet(f"""
            QPushButton#add_to_dict {{
                border: none;
                border-left: {border_left};
                border-top-left-radius: {left_radius};
                border-bottom-left-radius: {left_radius};
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                background-color: transparent;
                color: #4caf50;
            }}
            QPushButton#add_to_dict:hover {{
                background-color: #4caf50;
                color: #ffffff;
            }}
        """)
            
        self.scroll_content.adjustSize()
        suggestions_width = self.scroll_content.width()
        
        self.add_dict_btn.adjustSize()
        add_btn_width = self.add_dict_btn.width()
        self.add_dict_btn.setFixedWidth(add_btn_width)
        
        # Calculate horizontal sizes
        scroll_area_width = min(suggestions_width, 180) if len(suggestions) > 0 else 0
        self.scroll_area.setFixedWidth(scroll_area_width)
        
        if len(suggestions) == 0:
            self.scroll_area.hide()
        else:
            self.scroll_area.show()
            
        popup_width = scroll_area_width + add_btn_width
        self.setFixedSize(popup_width, 28)

    def _apply_clicked_suggestion(self, _checked: bool = False) -> None:
        button = self.sender()
        if isinstance(button, QPushButton):
            suggestion = button.property('suggestion')
            if suggestion is not None:
                self.apply_suggestion(str(suggestion))
        
    def apply_suggestion(self, replacement):
        self.editor._replace_word(self.cursor, replacement)
        self.hide()

    def add_to_dict(self):
        SpellCheckManager.get_instance().add_to_dictionary(self.word)
        self.hide()


class SourceTextEdit(QTextEdit):
    is_source_text_edit = True

    hover_enter = Signal(int)
    hover_leave = Signal(int)
    focus_in = Signal(int)
    propagate_user_edited = Signal(int, int, str, bool)
    ensure_scene_visible = Signal()
    redo_signal = Signal()
    undo_signal = Signal()
    push_undo_stack = Signal(int)
    text_changed = Signal()
    focus_out = Signal(int)
    suggestions_ready = Signal(object, str, list)

    def __init__(self, idx, parent, fold=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.idx = idx
        self.pre_editing = False
        self.setStyleSheet(r"QScrollBar:horizontal {height: 5px;}")
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
        self.input_method_removed = 0
        self.input_method_text = ''
        self.text_content_changed = False
        self.highlighting = False
        self.paste_flag = False
        self.paste_from: int = 0
        self.paste_removed: int = 0

        self.in_acts = False

        self.min_height = 45
        self.setFold(fold)
        self.spell_highlighter = SpellCheckHighlighter(self)
        self.selectionChanged.connect(self.on_selection_changed)
        self.suggestion_popup = None
        self.suggestions_ready.connect(self.show_suggestions_popup)
        self.current_suggestion_word = None

    def setFold(self, fold: bool):
        if fold:
            self.min_height = 35
            self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        else:
            self.min_height = 45
            self.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
            

    def _replace_word(self, cursor, replacement):
        self.setFocus()
        tc = self.textCursor()
        tc.setPosition(cursor.selectionStart())
        tc.setPosition(cursor.selectionEnd(), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(tc)
        
        tc.beginEditBlock()
        tc.insertText(replacement)
        tc.endEditBlock()
        
        self.handle_content_change()

    def _spellcheck_is_enabled(self) -> bool:
        return (
            getattr(pcfg, 'spellcheck_enabled', True)
            and (
                not self.is_source_text_edit
                or getattr(pcfg, 'spellcheck_on_source_enabled', False)
            )
        )

    def _dismiss_suggestions(self) -> None:
        if self.suggestion_popup is not None:
            self.suggestion_popup.hide()
        self.current_suggestion_word = None

    def on_selection_changed(self) -> None:
        try:
            # Do not scan selections or touch the optional dependency while the
            # relevant spellcheck path is disabled.
            if not self._spellcheck_is_enabled():
                self._dismiss_suggestions()
                return

            cursor = self.textCursor()
            if not cursor.hasSelection():
                self._dismiss_suggestions()
                return

            selected_text = cursor.selectedText()
            word = next(iter_spellcheck_words(selected_text), None)
            if word is None or word[2] != selected_text:
                self._dismiss_suggestions()
                return

            manager = SpellCheckManager.get_instance()
            if not manager.is_available() or manager.is_correct(selected_text):
                self._dismiss_suggestions()
                return

            self.current_suggestion_word = selected_text

            # Fetch suggestions in background to keep the UI thread responsive.
            def fetch_bg(c, w):
                try:
                    sugs = manager.get_suggestions(w)
                    self.suggestions_ready.emit(c, w, sugs)
                except Exception:
                    pass

            thread = threading.Thread(
                target=fetch_bg,
                args=(cursor, selected_text),
                daemon=True,
            )
            thread.start()
        except Exception:
            traceback.print_exc()

    def show_suggestions_popup(
        self,
        cursor: QTextCursor,
        word: str,
        suggestions: List[str],
    ) -> None:
        try:
            if not self._spellcheck_is_enabled():
                self._dismiss_suggestions()
                return
            if getattr(self, 'current_suggestion_word', None) != word:
                return
                
            if not self.suggestion_popup:
                self.suggestion_popup = FloatingSuggestionLabel(self)
            
            self.suggestion_popup.set_suggestions(cursor, word, suggestions)
            self.suggestion_popup.adjustSize()
                
            rect = self.cursorRect()
            
            # Position above the selection inside the editor viewport
            px = rect.left() + (rect.width() - self.suggestion_popup.width()) // 2
            py = rect.top() - self.suggestion_popup.height() - 4
            
            # Guard boundaries
            # If it goes off the top of the viewport, show it below the cursor
            if py < 0:
                py = rect.bottom() + 4
                
            px = max(5, min(px, self.viewport().width() - self.suggestion_popup.width() - 5))
            
            # Map viewport local coordinates to global screen coordinates
            global_pos = self.viewport().mapToGlobal(QPoint(px, py))
            
            self.suggestion_popup.move(global_pos)
            self.suggestion_popup.show()
        except Exception:
            traceback.print_exc()

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        acts = menu.actions()

        self._capture_paste_selection()
        self.in_acts = True
        rst = menu.exec_(event.globalPos())

        # future actions orders changes could break these comparsion
        self.paste_flag = rst == acts[5]
        if self.paste_flag or rst == acts[3] or rst == acts[6]:
            self.handle_content_change()
        self.in_acts = False

    def block_all_signals(self, block: bool):
        self.blockSignals(block)
        self.document().blockSignals(block)

    def updateUndoSteps(self):
        self.old_undo_steps = self.document().availableUndoSteps()

    def _capture_paste_selection(self) -> None:
        cursor = self.textCursor()
        self.paste_from = cursor.selectionStart()
        self.paste_removed = cursor.selectionEnd() - self.paste_from

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            self.text_content_changed = True
            if self.hasFocus():
                self.change_from = from_
                self.change_removed = removed
                self.change_added = added

    def adjustSize(self):
        h = self.document().documentLayout().documentSize().toSize().height()
        self.setFixedHeight(max(h, self.min_height))

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
            removed = self.change_removed
            added_text = ''
            
            if self.paste_flag:
                self.paste_flag = False
                # Qt can report a whole-block replacement when pasting at 0.
                change_from = self.paste_from
                removed = self.paste_removed
                cursor = self.textCursor()
                cursor.setPosition(change_from)
                cursor.setPosition(self.textCursor().position(), QTextCursor.MoveMode.KeepAnchor)
                added_text = cursor.selectedText()
            
            else:
                if self.input_method_from != -1:
                    added_text = self.input_method_text
                    change_from = self.input_method_from
                    removed = self.input_method_removed
                    self.input_method_from = -1
                    self.input_method_removed = 0
                elif self.change_added > 0:
                    cursor = self.textCursor()
                    cursor.setPosition(change_from)
                    cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor) 
                    added_text = cursor.selectedText()

            undo_steps = self.document().availableUndoSteps()
            new_steps = undo_steps - self.old_undo_steps
            joint_previous = new_steps == 0
            if removed > 0 or added_text:
                self.propagate_user_edited.emit(
                    change_from,
                    removed,
                    added_text,
                    joint_previous,
                )
            self.change_added = 0
            self.change_removed = 0

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
        self.focus_out.emit(self.idx)
        self._dismiss_suggestions()
        return super().focusOutEvent(event)

    def wheelEvent(self, event) -> None:
        self._dismiss_suggestions()
        return super().wheelEvent(event)

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        if not self.pre_editing:
            cursor = self.textCursor()
            self.input_method_from = cursor.selectionStart()
            self.input_method_removed = (
                cursor.selectionEnd() - cursor.selectionStart()
            )
        if e.replacementLength() > 0:
            cursor = self.textCursor()
            document_end = max(0, self.document().characterCount() - 1)
            replacement_start = max(
                0,
                min(
                    document_end,
                    cursor.position() + e.replacementStart(),
                ),
            )
            replacement_end = min(
                document_end,
                replacement_start + e.replacementLength(),
            )
            self.input_method_from = replacement_start
            self.input_method_removed = replacement_end - replacement_start
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            self.pre_editing = True
        super().inputMethodEvent(e)
        if (
            e.preeditString() == ''
            and not e.commitString()
            and e.replacementLength() == 0
        ):
            self.input_method_from = -1
            self.input_method_removed = 0
            self.input_method_text = ''

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.matches(QKeySequence.StandardKey.Paste):
            self._capture_paste_selection()
            self.paste_flag = True
            try:
                return super().keyPressEvent(e)
            finally:
                self.paste_flag = False
        if e.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.undo_signal.emit()
                return
            elif e.key() == Qt.Key.Key_Y:
                e.accept()
                self.redo_signal.emit()
                return
        elif e.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if e.key() == Qt.Key.Key_Z:
                e.accept()
                self.redo_signal.emit()
                return
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
    is_source_text_edit = False

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        cursor = self.textCursor()
        menu, quick_insert_actions = create_text_edit_context_menu(
            self,
            has_selection=cursor.hasSelection(),
            can_undo=self.document().isUndoAvailable(),
            can_redo=self.document().isRedoAvailable(),
        )

        self.in_acts = True
        changed = False
        try:
            action = menu.exec(event.globalPos())
            operation = action.data() if action is not None else None
            if action in quick_insert_actions:
                self.insertPlainText(operation)
                changed = True
            elif operation == 'undo':
                self.undo_signal.emit()
            elif operation == 'redo':
                self.redo_signal.emit()
            elif operation == 'cut':
                self.cut()
                changed = True
            elif operation == 'copy':
                self.copy()
            elif operation == 'paste':
                self._capture_paste_selection()
                self.paste_flag = True
                self.paste()
                changed = True
            elif operation == 'delete':
                cursor.removeSelectedText()
                self.setTextCursor(cursor)
                changed = True

            if changed:
                self.handle_content_change()
        finally:
            self.in_acts = False
        event.accept()


class RowIndexEditor(QLineEdit):

    focus_out = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setValidator(QIntValidator())
        self.setReadOnly(True)
        self.setTextMargins(0, 0, 0, 0)

    def focusOutEvent(self, e: QFocusEvent) -> None:
        super().focusOutEvent(e)
        self.focus_out.emit()

    def minimumSizeHint(self):
        size = super().minimumSizeHint()
        return QSize(1, size.height())
    
    def sizeHint(self):
        size = super().sizeHint()
        return QSize(1, size.height())
    

class RowIndexLabel(QStackedWidget):

    submmit_idx = Signal(int)

    def __init__(self, text: str = None, parent=None):
        super().__init__(parent=parent)
        self.lineedit = RowIndexEditor(parent=self)
        self.lineedit.focus_out.connect(self.on_lineedit_focusout)

        self.show_label = QLabel(self)
        self.text = self.show_label.text

        self.addWidget(self.show_label)
        self.addWidget(self.lineedit)
        self.setCurrentIndex(0)

        if text is not None:
            self.setText(text)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def setText(self, text):
        if isinstance(text, int):
            text = str(text)
        self.show_label.setText(text)
        self.lineedit.setText(text)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        super().keyPressEvent(e)

        key = e.key()
        if key == Qt.Key.Key_Return:
            self.try_update_idx()

    def try_update_idx(self):
        idx_str = self.lineedit.text().strip()
        if not idx_str:
            return
        if self.text() == idx_str:
            return
        idx = int(idx_str)
        self.lineedit.setReadOnly(True)
        self.submmit_idx.emit(idx)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        self.startEdit()
        return super().mouseDoubleClickEvent(e)

    def startEdit(self) -> None:
        self.setCurrentIndex(1)
        self.lineedit.setReadOnly(False)
        self.lineedit.setFocus()

    def on_lineedit_focusout(self):
        edited = not self.lineedit.isReadOnly()
        self.lineedit.setReadOnly(True)
        self.setCurrentIndex(0)
        if edited:
            self.try_update_idx()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        e.ignore()
        return super().mousePressEvent(e)
 

class TransPairWidget(Widget):

    check_state_changed = Signal(object, bool, bool)
    idx_edited = Signal(int, int)

    def __init__(self, idx: int = None, fold: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_source = SourceTextEdit(idx, self, fold)
        self.e_trans = TransTextEdit(idx, self, fold)
        self.idx_label = RowIndexLabel(idx, self)
        self.idx_label.setText(str(idx + 1).zfill(2))   # showed index start from 1!
        self.submmit_idx = self.idx_label.submmit_idx.connect(self.on_idx_edited)
        self.idx = idx
        self.checked = False
        vlayout = QVBoxLayout()
        vlayout.setAlignment(Qt.AlignTop)
        vlayout.addWidget(self.e_source)
        vlayout.addWidget(self.e_trans)
        vlayout.addWidget(SeparatorWidget(self))
        spacing = 7
        vlayout.setSpacing(spacing)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContentsMargins(0, 0, 0, 0)
        vlayout.setContentsMargins(0, spacing, spacing, spacing)

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.idx_label)
        hlayout.addLayout(vlayout)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(spacing)

    def on_idx_edited(self, new_idx: int):
        new_idx -= 1
        self.idx_edited.emit(self.idx, new_idx)

    def _set_checked_state(self, checked: bool):
        """
        this wont emit state_change signal and take care of the style
        """
        if self.checked != checked:
            self.checked = checked
            if checked:
                self.setStyleSheet('TransPairWidget{' + f'{STYLE_TRANSPAIR_CHECKED}' + '}')
            else:
                self.setStyleSheet("")

    def update_checkstate_by_mousevent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            modifiers = e.modifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier and modifiers & Qt.KeyboardModifier.ControlModifier:
                shift_pressed = ctrl_pressed = True
            else:
                shift_pressed = modifiers == Qt.KeyboardModifier.ShiftModifier
                ctrl_pressed = modifiers == Qt.KeyboardModifier.ControlModifier
            self.check_state_changed.emit(self, shift_pressed, ctrl_pressed)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if not self.checked:
            self.update_checkstate_by_mousevent(e)
        return super().mousePressEvent(e)

    def updateIndex(self, idx: int):
        if self.idx != idx:
            self.idx = idx
            self.idx_label.setText(str(idx + 1).zfill(2))
            self.e_source.idx = idx
            self.e_trans.idx = idx


class _DragGapFrame(QFrame):
    """Drop-slot indicator: translucent accent fill + dashed accent border.

    QSS ``border-style: dashed`` renders thin, short segments that are hard
    to notice, so the frame paints itself with QPainter: pen width, dash
    lengths and radius are all controllable."""

    PEN_WIDTH = 3
    # In pen-width units: at 3px pen → 9px dash, 5.4px gap
    DASH_PATTERN = (3.0, 1.8)
    RADIUS = 6
    FILL_ALPHA = 46

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = DRAG_INDICATOR_COLOR
        inset = self.PEN_WIDTH // 2 + 1
        rect = self.rect().adjusted(inset, inset, -inset, -inset)
        fill = QColor(accent)
        fill.setAlpha(self.FILL_ALPHA)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(rect, self.RADIUS, self.RADIUS)
        pen = QPen(accent, self.PEN_WIDTH)
        pen.setDashPattern(list(self.DASH_PATTERN))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, self.RADIUS, self.RADIUS)


class TextEditListScrollArea(QScrollArea):

    textblock_list: List[TextBlock] = []
    pairwidget_list: List[TransPairWidget] = []
    remove_textblock = Signal()
    selection_changed = Signal()   # this signal could only emit in on_widget_checkstate_changed, i.e. via user op
    rearrange_blks = Signal(object)
    textpanel_contextmenu_requested = Signal(QPoint, bool)
    focus_out = Signal()

    # Class-level default: Qt re-enters eventFilter during
    # QScrollArea.setWidget (before __init__ instance attributes exist),
    # so reading _drag_active there must not raise.
    _drag_active = False

    # Pile folding: while dragging a multi-selection, card i sags i*PILE_PEEK
    # below the pile top so every card's top strip (with its index) stays
    # readable; the gap slot is sized to the folded pile height.
    PILE_PEEK = 18
    # Opacity (0-255, ~15%) of the dim overlay covering non-dragged content.
    DIM_ALPHA = 38

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # ── Grab-style row drag state (see begin_rows_drag) ──
        # Must be initialized before setWidget: eventFilter is re-entered
        # during construction.
        self._drag_pws: List[TransPairWidget] = []   # dragged group (sorted by orig idx)
        self._rest: List[TransPairWidget] = []       # non-dragged rows (original order)
        self._rest_y = {}                            # target y of the yielding rows
        self._gap_slot = 0                           # drop slot (insert before rest[i])
        self._spacing = 0
        self._base_y = self._base_x = self._card_w = 0
        self._gap_h = 0
        self._drag_cursor_vp_y = 0.0
        self._pile_offsets: List[int] = []           # pile fold offsets (pile top = 0)
        self._drag_dim: QWidget = None
        self._gap_frame: QFrame = None
        self._auto_timer: QTimer = None
        self._auto_speed = 0

        self.scrollContent = Widget(parent=self)
        self.setWidget(self.scrollContent)

        # ScrollBar(Qt.Orientation.Horizontal, self)
        ScrollBar(Qt.Orientation.Vertical, self)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        vlayout = QVBoxLayout(self.scrollContent)
        vlayout.setContentsMargins(0, 0, 3, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.setSpacing(0)
        vlayout.addStretch(1)
        self.setWidgetResizable(True)
        self.vlayout = vlayout
        self.checked_list: List[TransPairWidget] = []
        self.sel_anchor_widget: TransPairWidget = None
        self.dragStartPosition = None

        self.source_visible = True
        self.trans_visible = True

        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if self._drag_active:
            if e.button() == Qt.MouseButton.LeftButton:
                self._finish_drag()
            return  # swallow other-button releases while dragging (context menu etc.)
        if e.button() == Qt.MouseButton.RightButton:
            pos = self.mapToGlobal(e.position()).toPoint()
            self.textpanel_contextmenu_requested.emit(pos, True)
        self.dragStartPosition = None
        super().mouseReleaseEvent(e)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if self._drag_active:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.dragStartPosition = e.pos()
        return super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._drag_active:
            self._drag_cursor_vp_y = e.position().y()
            self._update_drag_frame()
            return
        if self.sel_anchor_widget is not None and self.dragStartPosition is not None:
            if (e.pos() - self.dragStartPosition).manhattanLength() < QApplication.startDragDistance():
                return
            self.dragStartPosition = None
            self.begin_rows_drag(e.position().y())

        return super().mouseMoveEvent(e)

    def wheelEvent(self, e) -> None:
        super().wheelEvent(e)
        if self._drag_active:
            # Wheel keeps scrolling normally, but content coordinates changed:
            # re-sync the dragged pile and the yielding arrangement.
            self._update_drag_frame()

    # ── Grab-style row drag with live yielding ─────────────────
    # Differences from the old native-QDrag flow: the mouse is grabbed on
    # the viewport (hover never leaks into the editors), the dragged group
    # keeps its real rendering and folds into a pile that follows the
    # cursor, non-dragged content sits under a dim overlay, a self-drawn
    # dashed frame marks the drop slot, and the remaining rows yield in
    # real time. The block list order only lands through rearrange_blks on
    # release — dragging moves UI only, never data.

    def begin_rows_drag(self, cursor_vp_y: float) -> None:
        """Start a row drag. *cursor_vp_y* is the viewport y that triggered it
        (the pile anchors there)."""
        if self._drag_active:
            return
        n = len(self.pairwidget_list)
        drags = sorted(self.checked_list, key=lambda w: w.idx)
        if n < 2 or not drags or len(drags) == n:
            return
        self._drag_active = True
        self._drag_pws = drags
        self._rest = [w for w in self.pairwidget_list if w not in drags]
        self._gap_slot = drags[0].idx
        self._spacing = self.vlayout.spacing()
        self._base_y = min(w.y() for w in self.pairwidget_list)
        self._base_x = self._rest[0].x()
        self._card_w = self._rest[0].width()
        # Multi-selection folds into a pile: card i sags i*PILE_PEEK; the gap
        # slot height follows the folded pile height.
        self._pile_offsets = [i * self.PILE_PEEK for i in range(len(drags))]
        self._gap_h = max(
            off + w.height() for off, w in zip(self._pile_offsets, drags)
        )
        pile_top = int(cursor_vp_y + self.verticalScrollBar().value())
        self._drag_cursor_vp_y = cursor_vp_y

        QApplication.instance().installEventFilter(self)
        # Primary deactivation hook is the signal: the ApplicationDeactivate
        # event is not reliably delivered to app-level filters on Windows
        # (BallonsTranslator-lite 2026-08-18 lesson, same workaround the
        # pie menu there uses); the filter branch below is a fallback.
        QApplication.instance().applicationStateChanged.connect(
            self._on_app_state_changed
        )
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        self.viewport().grabMouse()

        # Layout takeover: rows leave the layout and get positioned manually.
        # LayoutRequest only sees the stretch, so it never touches manually
        # positioned rows; the minimum-height floor keeps widgetResizable
        # from collapsing scrollContent (which would kill the scrollbar).
        self.scrollContent.setMinimumHeight(self.scrollContent.height())
        for w in self.pairwidget_list:
            self.vlayout.removeWidget(w)

        # Dim overlay over non-dragged content (WA_StyledBackground lets a
        # plain QWidget take a QSS background color); the dragged pile and
        # the indicator frame float above it at full native resolution.
        self._drag_dim = QWidget(self.scrollContent)
        self._drag_dim.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._drag_dim.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._drag_dim.setStyleSheet(
            f"background-color: rgba(0, 0, 0, {self.DIM_ALPHA});"
        )
        self._drag_dim.setGeometry(
            0, 0, self.scrollContent.width(), self.scrollContent.height()
        )
        self._drag_dim.show()

        # Drop indicator frame (geometry refreshed by _apply_arrangement)
        self._gap_frame = _DragGapFrame(self.scrollContent)
        self._gap_frame.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )

        # Solid checked tint while piling: the 20%-alpha selection background
        # would let text of lower cards bleed through the stack.
        r, g, b = (
            DRAG_INDICATOR_COLOR.red(),
            DRAG_INDICATOR_COLOR.green(),
            DRAG_INDICATOR_COLOR.blue(),
        )
        wr, wg, wb = shared.WIDGET_BACKGROUND_COLOR[:3]
        mix = lambda t, s: int(round(t * 0.2 + s * 0.8))
        solid = QColor(mix(r, wr), mix(g, wg), mix(b, wb)).name()
        solid_style = (
            f'TransPairWidget{{background-color: {solid};'
            'border: 1px solid rgb(30, 147, 229);}'
        )
        for w in drags:
            w.setStyleSheet(solid_style)

        # Dragged group keeps its real widgets: jump to the cursor anchor,
        # z-order rear-card-on-top so every top strip stays visible.
        for w, off in zip(drags, self._pile_offsets):
            w.move(w.x(), pile_top + off)
        self._apply_arrangement()
        self._gap_frame.show()
        self._drag_dim.raise_()
        self._gap_frame.raise_()
        for w in drags:
            w.raise_()

    def _arrange_targets(self):
        """Yielding arrangement: rest rows stack in order, the dragged group's
        gap slot inserts before rest row ``_gap_slot``. Returns (target y per
        row, gap top y)."""
        ys = {}
        y = self._base_y
        gap_top = self._base_y
        placed = False
        for i, w in enumerate(self._rest):
            if i == self._gap_slot:
                gap_top = y
                y += self._gap_h + self._spacing
                placed = True
            ys[w] = y
            y += w.height() + self._spacing
        if not placed:
            gap_top = y  # gap at the end: below the last row
        return ys, gap_top

    def _apply_arrangement(self):
        ys, gap_top = self._arrange_targets()
        self._rest_y = ys
        for w, ty in ys.items():
            w.move(w.x(), ty)
        if self._gap_frame is not None:
            self._gap_frame.setGeometry(
                self._base_x, gap_top, self._card_w, self._gap_h
            )

    def _update_gap(self, y_cursor: int):
        """Yield decision (neighbor midpoints): cursor crossing the midpoint
        of the row above the gap moves the gap up one slot, the row below
        moves it down. Computed from the arrangement targets (not animated
        positions); the while loop absorbs one fast event skipping rows."""
        while True:
            if self._gap_slot > 0:
                above = self._rest[self._gap_slot - 1]
                if y_cursor < self._rest_y[above] + above.height() / 2:
                    self._gap_slot -= 1
                    self._apply_arrangement()
                    continue
            if self._gap_slot < len(self._rest):
                below = self._rest[self._gap_slot]
                if y_cursor > self._rest_y[below] + below.height() / 2:
                    self._gap_slot += 1
                    self._apply_arrangement()
                    continue
            break

    def _update_drag_frame(self):
        sb = self.verticalScrollBar()
        y_content = int(self._drag_cursor_vp_y + sb.value())
        for w, off in zip(self._drag_pws, self._pile_offsets):
            w.move(w.x(), y_content + off)
        self._update_gap(y_content)
        # Viewport edge auto-scroll (within 30px of top/bottom, faster when closer)
        vp_h = self.viewport().height()
        edge = 30
        speed = 0
        if self._drag_cursor_vp_y < edge:
            speed = -max(3, int((edge - self._drag_cursor_vp_y) * 0.25))
        elif self._drag_cursor_vp_y > vp_h - edge:
            speed = max(3, int((self._drag_cursor_vp_y - (vp_h - edge)) * 0.25))
        if speed:
            self._auto_speed = speed
            if self._auto_timer is None:
                t = QTimer(self)
                t.timeout.connect(self._auto_scroll_tick)
                self._auto_timer = t
                t.start(16)
        elif self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer.deleteLater()
            self._auto_timer = None

    def _auto_scroll_tick(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.value() + self._auto_speed)
        self._update_drag_frame()

    def _restore_layout(self, order: List[TransPairWidget]):
        for i, w in enumerate(order):
            self.vlayout.insertWidget(i, w)
        self.scrollContent.setMinimumHeight(0)

    def _teardown_drag(self):
        """Tear-down shared by finish and cancel: release grab, remove filter,
        clear overlays."""
        self._drag_active = False
        QApplication.instance().removeEventFilter(self)
        try:
            QApplication.instance().applicationStateChanged.disconnect(
                self._on_app_state_changed
            )
        except (TypeError, RuntimeError):
            pass
        try:
            self.viewport().releaseMouse()
        except RuntimeError:
            pass
        self.viewport().unsetCursor()
        if self._auto_timer is not None:
            self._auto_timer.stop()
            self._auto_timer.deleteLater()
            self._auto_timer = None
        for w in (self._drag_dim, self._gap_frame):
            if w is not None:
                w.hide()
                w.deleteLater()
        self._drag_dim = None
        self._gap_frame = None
        for w in self._drag_pws:
            if w.checked:
                w.setStyleSheet('TransPairWidget{' + f'{STYLE_TRANSPAIR_CHECKED}' + '}')

    def _finish_drag(self):
        """Drop: snapshot current positions → restore layout in the new order
        and activate it synchronously → land the reorder through
        rearrange_blks (what you see is the final state)."""
        if not self._drag_active:
            return
        self._teardown_drag()
        new_order = (
            self._rest[: self._gap_slot] + self._drag_pws + self._rest[self._gap_slot:]
        )
        self._restore_layout(new_order)
        self.vlayout.activate()
        self._drag_pws = []
        self._rest = []
        self._emit_rearrange_from_perm([w.idx for w in new_order])

    def _cancel_drag(self):
        if not self._drag_active:
            return
        self._teardown_drag()
        self._restore_layout(self.pairwidget_list)  # original order, original slots
        self.vlayout.activate()
        self._drag_pws = []
        self._rest = []

    def clearDrag(self):
        """External clear request (e.g. focus moved away): cancel an active drag."""
        if self._drag_active:
            self._cancel_drag()

    def _on_app_state_changed(self, state) -> None:
        """App-wide deactivation (screenshot overlay, window switch) cancels
        the drag and restores the rows."""
        if self._drag_active and state != Qt.ApplicationState.ApplicationActive:
            self._cancel_drag()

    def eventFilter(self, obj, event) -> bool:
        if self._drag_active:
            t = event.type()
            if t == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._cancel_drag()
                return True
            if t == QEvent.Type.ApplicationDeactivate:
                # Another window took over (e.g. a screenshot overlay): the
                # frozen pile would linger and still follow wheel scrolling,
                # then a stray click after refocusing would land the stale
                # position. Cancel on focus loss, like the old QDrag flow did.
                self._cancel_drag()
                return False
            if t in (QEvent.Type.HoverEnter, QEvent.Type.HoverMove):
                # Swallow hovers inside the list while dragging so editors
                # don't light up (mouse grab should isolate this already).
                w = obj if isinstance(obj, QWidget) else None
                while w is not None and w is not self.scrollContent:
                    w = w.parentWidget()
                if w is self.scrollContent:
                    return True
        return super().eventFilter(obj, event)

    def _emit_rearrange_from_perm(self, result_list):
        """Compute (drags_ori, drags_tgt) from a permutation list (each entry = old idx
        at that position), emit rearrange_blks so on_rearrange_blks -> RearrangeBlksCommand
        runs through the same path as drag-drop. Items unchanged are filtered out, so
        unchanged blocks stay out of tgt_ids (updateTextBlkItemIdx won't touch them).
        """
        drags_ori, drags_tgt = [], []
        for ii, idx in enumerate(result_list):
            if ii != idx:
                drags_ori.append(idx)
                drags_tgt.append(ii)
        if drags_ori:
            self.rearrange_blks.emit((drags_ori, drags_tgt))


    def on_idx_edited(self, src_idx: int, tgt_idx: int):
        src_idx_ori = tgt_idx
        tgt_idx = max(min(tgt_idx, len(self.pairwidget_list) - 1), 0)
        if src_idx_ori != tgt_idx:
            self.pairwidget_list[src_idx].idx_label.setText(str(src_idx + 1).zfill(2))
        if src_idx == tgt_idx:
            return
        ids_ori, ids_tgt = [src_idx], [tgt_idx]
        
        if src_idx < tgt_idx:
            for idx in range(src_idx+1, tgt_idx+1):
                ids_ori.append(idx)
                ids_tgt.append(idx-1)
        else:
            for idx in range(tgt_idx, src_idx):
                ids_ori.append(idx)
                ids_tgt.append(idx+1)
        self.rearrange_blks.emit((ids_ori, ids_tgt, (tgt_idx, src_idx)))

    def addPairWidget(self, pairwidget: TransPairWidget):
        self.vlayout.insertWidget(pairwidget.idx, pairwidget)
        pairwidget.check_state_changed.connect(self.on_widget_checkstate_changed)
        pairwidget.e_trans.setVisible(self.trans_visible)
        pairwidget.e_source.setVisible(self.source_visible)
        pairwidget.setVisible(True)

    def insertPairWidget(self, pairwidget: TransPairWidget, idx: int):
        self.vlayout.insertWidget(idx, pairwidget)
        pairwidget.e_trans.setVisible(self.trans_visible)
        pairwidget.e_source.setVisible(self.source_visible)
        pairwidget.setVisible(True)

    def on_widget_checkstate_changed(
        self,
        pwc: TransPairWidget,
        shift_pressed: bool,
        ctrl_pressed: bool,
    ) -> None:
        if self._drag_active:
            return
        
        idx = pwc.idx
        if shift_pressed:
            checked = True
        else:
            checked = not pwc.checked
        pwc._set_checked_state(checked)

        num_sel = len(self.checked_list)
        old_idx_list = [pw.idx for pw in self.checked_list]
        old_idx_set = set(old_idx_list)
        new_check_list = []
        if shift_pressed:
            if num_sel == 0:
                new_check_list.append(idx)
            else:
                tgt_w = self.pairwidget_list[idx]
                if ctrl_pressed:
                    sel_min, sel_max = min(old_idx_list[0], tgt_w.idx), max(old_idx_list[-1], tgt_w.idx)
                else:
                    anchor = self.sel_anchor_widget or self.checked_list[0]
                    self.sel_anchor_widget = anchor
                    sel_min, sel_max = min(anchor.idx, tgt_w.idx), max(anchor.idx, tgt_w.idx)
                new_check_list = list(range(sel_min, sel_max + 1))
        elif ctrl_pressed:
            new_check_set = set(old_idx_list)
            if idx in new_check_set:
                new_check_set.remove(idx)
                if self.sel_anchor_widget is not None and self.sel_anchor_widget.idx == idx:
                    self.sel_anchor_widget = None
            elif checked:
                new_check_set.add(idx)
            new_check_list = list(new_check_set)
            new_check_list.sort()
            if checked:
                self.sel_anchor_widget = self.pairwidget_list[idx]
        else:
            if num_sel > 2:
                if idx in old_idx_set:
                    old_idx_set.remove(idx)
                    checked = True
            if checked:
                new_check_list.append(idx)
        
        new_check_set = set(new_check_list)
        check_changed = False
        for oidx in old_idx_set:
            if oidx not in new_check_set:
                self.pairwidget_list[oidx]._set_checked_state(False)
                check_changed = True

        self.checked_list.clear()
        for nidx in new_check_list:
            pw = self.pairwidget_list[nidx]
            if nidx not in old_idx_set:
                check_changed = True
                pw._set_checked_state(True)
            self.checked_list.append(pw)
            
        num_new = len(new_check_list)
        if num_new == 0:
            self.sel_anchor_widget = None
        elif num_new == 1 or self.sel_anchor_widget is None:
            self.sel_anchor_widget = self.checked_list[0]
        if check_changed:
            self.selection_changed.emit()
            if pwc.checked:
                pwc.e_trans.focus_in.emit(pwc.idx)

    def set_selected_list(self, selection_indices: List[int]) -> None:
        self.clearDrag()

        old_sel_set = {pw.idx for pw in self.checked_list}
        new_sel_set = set(selection_indices)
        if old_sel_set == new_sel_set:
            if not self.checked_list:
                self.sel_anchor_widget = None
            elif self.sel_anchor_widget not in self.checked_list:
                self.sel_anchor_widget = self.checked_list[0]
            return

        to_remove = old_sel_set.difference(new_sel_set)
        to_add = new_sel_set.difference(old_sel_set)
        if (
            self.sel_anchor_widget is not None
            and self.sel_anchor_widget.idx not in new_sel_set
        ):
            self.sel_anchor_widget = None

        for idx in sorted(to_remove):
            pw = self.pairwidget_list[idx]
            pw._set_checked_state(False)
            self.checked_list.remove(pw)

        for idx in sorted(to_add):
            pw = self.pairwidget_list[idx]
            pw._set_checked_state(True)
            self.checked_list.append(pw)
        self.checked_list.sort(key=lambda pw: pw.idx)
        if self.checked_list and self.sel_anchor_widget is None:
            self.sel_anchor_widget = self.checked_list[0]

    def clearAllSelected(self, emit_signal=True):
        self.sel_anchor_widget = None
        if len(self.checked_list) > 0:
            for w in self.checked_list:
                w._set_checked_state(False)
            self.checked_list.clear()
            if emit_signal:
                self.selection_changed.emit()

    def removeWidget(
        self,
        widget: TransPairWidget,
        remove_checked: bool = True,
    ) -> None:
        widget.setVisible(False)
        if remove_checked:
            if self.sel_anchor_widget is not None and self.sel_anchor_widget.idx == widget.idx:
                self.sel_anchor_widget = None
            if widget in self.checked_list:
                widget._set_checked_state(False)
                self.checked_list.remove(widget)
            if self.sel_anchor_widget is None and self.checked_list:
                self.sel_anchor_widget = self.checked_list[0]
        self.vlayout.removeWidget(widget)
    
    def focusOutEvent(self, e: QFocusEvent) -> None:
        self.focus_out.emit()
        super().focusOutEvent(e)
    
    def setFoldTextarea(self, fold: bool):
        for pw in self.pairwidget_list:
            pw.e_trans.setFold(fold)
            pw.e_source.setFold(fold)

    def setSourceVisible(self, show: bool):
        self.source_visible = show
        for pw in self.pairwidget_list:
            pw.e_source.setVisible(show)

    def setTransVisible(self, show: bool):
        self.trans_visible = show
        for pw in self.pairwidget_list:
            pw.e_trans.setVisible(show)
