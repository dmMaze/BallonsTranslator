from typing import List, Union

from qtpy.QtWidgets import QStackedWidget, QSizePolicy, QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QApplication, QHBoxLayout, QLabel, QLineEdit, QWidget, QPushButton
from qtpy.QtCore import Signal, Qt, QMimeData, QEvent, QPoint, QSize
from qtpy.QtGui import QIntValidator, QColor, QFocusEvent, QInputMethodEvent, QDragEnterEvent, QDropEvent, QKeyEvent, QTextCursor, QMouseEvent, QDrag, QPixmap
import numpy as np

from .custom_widget import ScrollBar, Widget, SeparatorWidget
from .textitem import TextBlock


STYLE_TRANSPAIR_CHECKED = "background-color: rgba(30, 147, 229, 20%);"
STYLE_TRANSPAIR_BOTTOM = "border-width: 5px; border-bottom-style: solid; border-color: rgb(30, 147, 229);"
STYLE_TRANSPAIR_TOP = "border-width: 5px; border-top-style: solid; border-color: rgb(30, 147, 229);"


class FloatingSuggestionLabel(QWidget):
    def __init__(self, editor):
        super().__init__(editor.viewport())
        self.editor = editor
        self.setObjectName("suggestion_popup")
        
        # Segmented tab style: dark background, border, zero margins/padding between buttons
        self.setStyleSheet("""
            QWidget#suggestion_popup {
                background-color: #2b2b2d;
                border: 1px solid #45474a;
                border-radius: 4px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)
        
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Horizontal scroll area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setFixedHeight(28)
        self.scroll_area.wheelEvent = self.scroll_area_wheel_event
        
        self.scroll_content = QWidget()
        self.scroll_content.setObjectName("scroll_content")
        self.scroll_content.setStyleSheet("background: transparent;")
        
        self.buttons_layout = QHBoxLayout(self.scroll_content)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setSpacing(0)
        
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(6)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 2)
        self.setGraphicsEffect(shadow)
        
        self.hide()

    def scroll_area_wheel_event(self, event):
        scrollbar = self.scroll_area.horizontalScrollBar()
        if scrollbar:
            delta = event.angleDelta().y() or event.angleDelta().x()
            scrollbar.setValue(scrollbar.value() - delta // 2)
            event.accept()

    def set_suggestions(self, cursor, word, suggestions):
        self.cursor = cursor
        self.word = word
        
        while self.buttons_layout.count() > 0:
            item = self.buttons_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
                
        for i, sug in enumerate(suggestions):
            btn = QPushButton(sug, self.scroll_content)
            btn.clicked.connect(lambda checked=False, s=sug: self.apply_suggestion(s))
            self.buttons_layout.addWidget(btn)
            
            # Stylize borders and round corners so they form a single seamless block
            border_right = "1px solid #45474a" if i < len(suggestions) - 1 else "none"
            left_radius = "4px" if i == 0 else "0px"
            right_radius = "4px" if i == len(suggestions) - 1 else "0px"
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: #3b3d40;
                    color: #ffffff;
                    border: none;
                    border-right: {border_right};
                    border-top-left-radius: {left_radius};
                    border-bottom-left-radius: {left_radius};
                    border-top-right-radius: {right_radius};
                    border-bottom-right-radius: {right_radius};
                    padding: 0px 12px;
                    height: 28px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                    font-size: 11px;
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: #1e93e5;
                    color: #ffffff;
                }}
            """)
            
        self.scroll_content.adjustSize()
        total_width = self.scroll_content.width()
        
        # Max width is 250px, if more we scroll
        popup_width = min(total_width, 250)
        self.scroll_area.setFixedWidth(popup_width)
        self.setFixedSize(popup_width, 28)
        
    def apply_suggestion(self, replacement):
        self.editor._replace_word(self.cursor, replacement)
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
    focus_out = Signal(int)

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
        self.change_added: int = 0
        self.input_method_from = -1
        self.input_method_text = ''
        self.text_content_changed = False
        self.highlighting = False
        self.paste_flag = False

        self.in_acts = False

        self.min_height = 45
        self.setFold(fold)
        from ballontranslator.utils.spellcheck import SpellCheckHighlighter
        self.spell_highlighter = SpellCheckHighlighter(self.document())
        self.selectionChanged.connect(self.on_selection_changed)
        self.suggestion_popup = None

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
        tc.beginEditBlock()
        tc.setPosition(cursor.selectionStart())
        tc.setPosition(cursor.selectionEnd(), QTextCursor.MoveMode.KeepAnchor)
        tc.insertText(replacement)
        tc.endEditBlock()

    def on_selection_changed(self):
        try:
            from ballontranslator.utils.spellcheck import SpellCheckManager
            import re

            manager = SpellCheckManager.get_instance()
            if not manager.is_available():
                return
            from ballontranslator.utils.config import pcfg
            if not getattr(pcfg, 'spellcheck_enabled', True):
                return

            cursor = self.textCursor()
            if not cursor.hasSelection():
                if hasattr(self, 'suggestion_popup') and self.suggestion_popup:
                    self.suggestion_popup.hide()
                return

            selected_text = cursor.selectedText().strip()
            # Only suggest for a single word
            if len(selected_text) > 1 and re.match(r'^[a-zA-Zа-яА-ЯёЁ]+$', selected_text):
                if not manager.is_correct(selected_text):
                    suggestions = manager.get_suggestions(selected_text)
                    if suggestions:
                        if not self.suggestion_popup:
                            self.suggestion_popup = FloatingSuggestionLabel(self)
                        
                        self.suggestion_popup.set_suggestions(cursor, selected_text, suggestions)
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
                        
                        self.suggestion_popup.move(px, py)
                        self.suggestion_popup.show()
                        return

            if hasattr(self, 'suggestion_popup') and self.suggestion_popup:
                self.suggestion_popup.hide()
        except Exception as e:
            import traceback
            traceback.print_exc()

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        acts = menu.actions()

        try:
            # Spell suggestions integration
            from ballontranslator.utils.spellcheck import SpellCheckManager
            import re
            from qtpy.QtWidgets import QAction
            from qtpy.QtGui import QDesktopServices
            from qtpy.QtCore import QUrl

            manager = SpellCheckManager.get_instance()
            pos = event.pos()
            # Handle keyboard-triggered menu (where pos is negative)
            if pos.x() < 0 or pos.y() < 0:
                cursor = self.textCursor()
            else:
                cursor = self.cursorForPosition(pos)
                
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
            selected_word = cursor.selectedText().strip()

            if selected_word and len(selected_word) > 1:
                first_act = acts[0] if acts else None

                # Add Search Online action
                search_act = QAction(self.tr("Search Online"), menu)
                search_act.triggered.connect(lambda checked, w=selected_word: QDesktopServices.openUrl(QUrl(f"https://translate.google.com/?sl=auto&tl=ru&text={w}")))
                if first_act:
                    menu.insertAction(first_act, search_act)
                    menu.insertSeparator(first_act)
                else:
                    menu.addAction(search_act)

                # Check spelling suggestions
                is_misspelled = False
                if manager.is_available() and re.match(r'^[a-zA-Zа-яА-ЯёЁ]+$', selected_word):
                    is_misspelled = not manager.is_correct(selected_word)

                if is_misspelled:
                    suggestions = manager.get_suggestions(selected_word)

                    # Add suggestions
                    if suggestions:
                        for sug in suggestions:
                            sug_act = QAction(sug, menu)
                            sug_act.triggered.connect(lambda checked, s=sug, c=cursor: self._replace_word(c, s))
                            font = sug_act.font()
                            font.setBold(True)
                            sug_act.setFont(font)
                            if first_act:
                                menu.insertAction(first_act, sug_act)
                            else:
                                menu.addAction(sug_act)
                    else:
                        no_sug_act = QAction(self.tr("No spelling suggestions"), menu)
                        no_sug_act.setEnabled(False)
                        if first_act:
                            menu.insertAction(first_act, no_sug_act)
                        else:
                            menu.addAction(no_sug_act)

                    # Add to Dictionary action
                    add_dict_act = QAction(self.tr("Add to Dictionary"), menu)
                    add_dict_act.triggered.connect(lambda checked, w=selected_word: manager.add_to_dictionary(w))
                    if first_act:
                        menu.insertAction(first_act, add_dict_act)
                        menu.insertSeparator(first_act)
                    else:
                        menu.addAction(add_dict_act)
        except Exception as e:
            import traceback
            traceback.print_exc()

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

    def on_content_changing(self, from_: int, removed: int, added: int):
        if not self.pre_editing:
            self.text_content_changed = True
            if self.hasFocus():
                self.change_from = from_
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
            added_text = ''
            
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
                    self.input_method_from = -1
                elif self.change_added > 0:
                    cursor = self.textCursor()
                    cursor.setPosition(change_from)
                    cursor.setPosition(change_from + self.change_added, QTextCursor.MoveMode.KeepAnchor) 
                    added_text = cursor.selectedText()

            undo_steps = self.document().availableUndoSteps()
            new_steps = undo_steps - self.old_undo_steps
            joint_previous = new_steps == 0
            self.propagate_user_edited.emit(change_from, added_text, joint_previous)
            self.change_added = 0

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
        if hasattr(self, 'suggestion_popup') and self.suggestion_popup:
            self.suggestion_popup.hide()
        return super().focusOutEvent(event)

    def wheelEvent(self, event) -> None:
        if hasattr(self, 'suggestion_popup') and self.suggestion_popup:
            self.suggestion_popup.hide()
        return super().wheelEvent(event)

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        if self.pre_editing is False:
            cursor = self.textCursor()
            self.input_method_from = cursor.selectionStart()
        if e.preeditString() == '':
            self.pre_editing = False
            self.input_method_text = e.commitString()
        else:
            self.pre_editing = True
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
            elif e.key() == Qt.Key.Key_V:
                self.paste_flag = True
                return super().keyPressEvent(e)
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
    pass


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
    drag_move = Signal(int)
    idx_edited = Signal(int, int)
    pw_drop = Signal()

    def __init__(self, textblock: TextBlock = None, idx: int = None, fold: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_source = SourceTextEdit(idx, self, fold)
        self.e_trans = TransTextEdit(idx, self, fold)
        self.idx_label = RowIndexLabel(idx, self)
        self.idx_label.setText(str(idx + 1).zfill(2))   # showed index start from 1!
        self.submmit_idx = self.idx_label.submmit_idx.connect(self.on_idx_edited)
        self.textblock = textblock
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

        self.setAcceptDrops(True)

    def on_idx_edited(self, new_idx: int):
        new_idx -= 1
        self.idx_edited.emit(self.idx, new_idx)

    def dragEnterEvent(self, e: QDragEnterEvent) -> None:
        if isinstance(e.source(), TransPairWidget):
            e.accept()
        return super().dragEnterEvent(e)
    
    def handle_drag(self, pos: QPoint):
        y = pos.y()
        to_pos = self.idx
        if y > self.size().height() / 2:
            to_pos += 1
        self.drag_move.emit(to_pos)
    
    def dragMoveEvent(self, e: QDragEnterEvent) -> None:
        if isinstance(e.source(), TransPairWidget):
            e.accept()
            self.handle_drag(e.position())

        return super().dragMoveEvent(e)

    def dropEvent(self, e: QDropEvent) -> None:
        if isinstance(e.source(), TransPairWidget):
            e.acceptProposedAction()
            self.pw_drop.emit()

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


class TextEditListScrollArea(QScrollArea):

    textblock_list: List[TextBlock] = []
    pairwidget_list: List[TransPairWidget] = []
    remove_textblock = Signal()
    selection_changed = Signal()   # this signal could only emit in on_widget_checkstate_changed, i.e. via user op
    rearrange_blks = Signal(object)
    textpanel_contextmenu_requested = Signal(QPoint, bool)
    focus_out = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
        self.drag: QDrag = None
        self.dragStartPosition = None

        self.source_visible = True
        self.trans_visible = True

        self.drag_to_pos: int = -1

        self.setSizePolicy(self.sizePolicy().horizontalPolicy(), QSizePolicy.Policy.Expanding)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.RightButton:
            pos = self.mapToGlobal(e.position()).toPoint()
            self.textpanel_contextmenu_requested.emit(pos, True)
        super().mouseReleaseEvent(e)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.dragStartPosition = e.pos()
        return super().mousePressEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.drag is None and self.sel_anchor_widget is not None and self.dragStartPosition is not None:
            if (e.pos() - self.dragStartPosition).manhattanLength() < QApplication.startDragDistance():
                return
            self.dragStartPosition = None
            w = self.sel_anchor_widget
            drag = self.drag = QDrag(w)
            mime = QMimeData()
            drag.setMimeData(mime)
            pixmap = QPixmap(w.size())
            w.render(pixmap)
            drag.setPixmap(pixmap)
            ac = drag.exec(Qt.DropAction.MoveAction)
            self.drag = None
            if self.drag_to_pos != -1:
                self.set_drag_style(self.drag_to_pos, True)
                self.drag_to_pos = -1
            pass

        return super().mouseMoveEvent(e)
    
    def set_drag_style(self, pos: int, clear_style: bool = False):
        if pos == len(self.pairwidget_list):
            pos -= 1
            style = STYLE_TRANSPAIR_BOTTOM
        else:
            style = STYLE_TRANSPAIR_TOP
        if clear_style:
            style = ""
        pw = self.pairwidget_list[pos]
        if pw.checked:
            style += STYLE_TRANSPAIR_CHECKED
        style = "TransPairWidget{" + style + "}"
        pw.setStyleSheet(style)
    
    def clearDrag(self):
        self.drag_to_pos = -1
        if self.drag is not None:
            try:
                self.drag.cancel()
            except RuntimeError:
                pass
            self.drag = None
    
    def handle_drag_pos(self, to_pos: int):
        if self.drag_to_pos != to_pos:
            if self.drag_to_pos is not None:
                self.set_drag_style(self.drag_to_pos, True)
            self.drag_to_pos = to_pos
            self.set_drag_style(to_pos)

    def on_pw_dropped(self):
        if self.drag_to_pos != -1:
            to_pos = self.drag_to_pos
            self.drag_to_pos = -1
            self.drag = None
            self.set_drag_style(to_pos, True)
            num_pw = len(self.pairwidget_list)
            num_drags = len(self.checked_list)
            if num_pw < 2 or num_drags == num_pw:
                return
            
            tgt_pos = to_pos
            drags = []
            for pw in self.checked_list:
                if pw.idx < tgt_pos:
                    tgt_pos -= 1
                drags.append(pw.idx)
            new_pos = np.arange(num_drags, dtype=np.int32) + tgt_pos
            drags = np.array(drags).astype(np.int32)
            new_maps = np.where(drags != new_pos)
            if len(new_maps) == 0:
                return

            drags_ori, drags_tgt = drags[new_maps], new_pos[new_maps]
            result_list = list(range(len(self.pairwidget_list)))
            to_insert = []
            for ii, src_idx in enumerate(drags_ori):
                pos = src_idx - ii
                to_insert.append(result_list.pop(pos))
            for ii, tgt_idx in enumerate(drags_tgt):
                result_list.insert(tgt_idx, to_insert[ii])
            drags_ori, drags_tgt = [], []
            for ii, idx in enumerate(result_list):
                if ii != idx:
                    drags_ori.append(idx)
                    drags_tgt.append(ii)

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

    def on_widget_checkstate_changed(self, pwc: TransPairWidget, shift_pressed: bool, ctrl_pressed: bool):
        if self.drag is not None:
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
                    sel_min, sel_max = min(self.sel_anchor_widget.idx, tgt_w.idx), max(self.sel_anchor_widget.idx, tgt_w.idx)
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

    def set_selected_list(self, selection_indices: List):
        self.clearDrag()

        old_sel_set, new_sel_set = set([pw.idx for pw in self.checked_list]), set(selection_indices)
        to_remove = old_sel_set.difference(new_sel_set)
        to_add = new_sel_set.difference(old_sel_set)
        self.sel_anchor_widget = None

        for idx in to_remove:
            pw = self.pairwidget_list[idx]
            pw._set_checked_state(False)
            self.checked_list.remove(pw)

        for ii, idx in enumerate(to_add):
            pw = self.pairwidget_list[idx]
            pw._set_checked_state(True)
            self.checked_list.append(pw)
            if ii == 0:
                self.sel_anchor_widget = pw

    def clearAllSelected(self, emit_signal=True):
        self.sel_anchor_widget = None
        if len(self.checked_list) > 0:
            for w in self.checked_list:
                w._set_checked_state(False)
            self.checked_list.clear()
            if emit_signal:
                self.selection_changed.emit()

    def removeWidget(self, widget: TransPairWidget, remove_checked: bool = True):
        widget.setVisible(False)
        if remove_checked:
            if self.sel_anchor_widget is not None and self.sel_anchor_widget.idx == widget.idx:
                self.sel_anchor_widget = None
            if widget in self.checked_list:
                widget._set_checked_state(False)
                self.checked_list.remove(widget)
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
