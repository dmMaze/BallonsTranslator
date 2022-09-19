from qtpy.QtWidgets import QHBoxLayout, QComboBox, QTextEdit, QLabel, QTreeView, QPlainTextEdit, QCheckBox, QMessageBox, QVBoxLayout, QStyle, QSlider, QProxyStyle, QStyle,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QTimer, QEasingCurve, QPointF, QRect, Signal
from qtpy.QtGui import QKeyEvent, QTextDocument, QTextCursor, QHideEvent, QInputMethodEvent, QFontMetrics, QColor, QShowEvent
from typing import List, Union, Tuple, Dict

from .misc import ProgramConfig
from .stylewidgets import Widget, ClickableLabel
from .textitem import TextBlkItem
from .imgtranspanel import TransPairWidget, SourceTextEdit, TransTextEdit

HIGHLIGHT_COLOR = QColor(30, 147, 229, 60)
CURRENT_TEXT_COLOR = QColor(244, 249, 28)

class SearchEditor(QPlainTextEdit):
    height_changed = Signal()
    commit = Signal()
    enter_pressed = Signal()
    shift_enter_pressed = Signal()
    def __init__(self, parent: QWidget = None, original_height: int = 32, commit_latency: int = -1, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.original_height = original_height
        self.commit_latency = commit_latency
        if commit_latency > 0:
            self.commit_timer = QTimer(self)
            self.commit_timer.timeout.connect(self.on_commit_timer_timeout)
        else:
            self.commit_timer = None
        self.pre_editing = False
        self.setFixedHeight(original_height)
        self.document().documentLayout().documentSizeChanged.connect(self.adjustSize)
        self.textChanged.connect(self.on_text_changed)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, True)

    def adjustSize(self):
        fm = QFontMetrics(self.font())
        h = fm.height() * self.document().size().height() * 1.05
        h += self.document().documentMargin() * 2
        if self.geometry().height() != h:
            self.setFixedHeight(max(h, self.original_height))
            self.height_changed.emit()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Return:
            if self.commit_timer is not None:
                self.commit_timer.stop()
            e.setAccepted(True)
            if e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                self.shift_enter_pressed.emit()
            else:
                self.enter_pressed.emit()
            return
        return super().keyPressEvent(e)

    def on_text_changed(self):
        if self.commit_timer is not None:
            if not self.pre_editing:
                self.commit_timer.stop()
                self.commit_timer.start(self.commit_latency)
        elif not self.pre_editing:
            self.commit.emit()

    def on_commit_timer_timeout(self):
        self.commit_timer.stop()
        self.commit.emit()

    def hideEvent(self, e: QHideEvent) -> None:
        if self.commit_timer is not None:
            self.commit_timer.stop()
        return super().hideEvent(e)

    def inputMethodEvent(self, e: QInputMethodEvent) -> None:
        if e.preeditString() == '':
            self.pre_editing = False
            if self.commit_timer is not None:
                self.commit_timer.start(self.commit_latency)
        else:
            if self.commit_timer is not None:
                self.commit_timer.stop()
            self.pre_editing = True
        return super().inputMethodEvent(e)


class SearchWidget(Widget):

    search = Signal()
    replace = Signal()
    reinit = Signal()

    def __init__(self, parent: QWidget = None, is_floating=True, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.is_floating = is_floating
        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list: List[int] = []
        self.search_cursorpos_map: List[Dict] = []
        self.counter_sum = 0
        self.pairwidget_list: List[TransPairWidget] = []
        self.textblk_item_list: List[TextBlkItem] = []

        self.current_edit: SourceTextEdit = None
        self.current_cursor: QTextCursor = None
        self.result_pos = 0

        self.search_editor = SearchEditor(self, commit_latency=-1)
        self.search_editor.setPlaceholderText(self.tr('Find'))
        self.search_editor.height_changed.connect(self.on_editor_height_changed)
        
        self.no_result_str = self.tr('No result')
        self.result_counter_label = QLabel(self.no_result_str)
        self.result_counter_label.setMaximumHeight(32)
        self.prev_match_btn = ClickableLabel(None, self)
        self.prev_match_btn.setObjectName('PrevMatchBtn')
        self.prev_match_btn.clicked.connect(self.on_prev_search_result)
        self.prev_match_btn.setToolTip(self.tr('Previous Match (Shift+Enter)'))

        self.next_match_btn = ClickableLabel(None, self)
        self.next_match_btn.setObjectName('NextMatchBtn')
        self.next_match_btn.clicked.connect(self.on_next_search_result)
        self.next_match_btn.setToolTip(self.tr('Next Match (Enter)'))

        self.case_sensitive_toggle = QCheckBox(self)
        self.case_sensitive_toggle.setObjectName('CaseSensitiveToggle')
        self.case_sensitive_toggle.setToolTip(self.tr('Match Case'))
        self.case_sensitive_toggle.clicked.connect(self.on_case_clicked)

        self.whole_word_toggle = QCheckBox(self)
        self.whole_word_toggle.setObjectName('WholeWordToggle')
        self.whole_word_toggle.setToolTip(self.tr('Match Whole Word'))
        self.whole_word_toggle.clicked.connect(self.on_whole_word_clicked)

        self.range_combobox = QComboBox(self)
        self.range_combobox.addItems([self.tr('Translation'), self.tr('Source'), self.tr('All')])
        self.range_combobox.currentIndexChanged.connect(self.on_range_changed)
        self.range_label = QLabel(self)
        self.range_label.setText(self.tr('Range'))

        self.replace_editor = SearchEditor(self)
        self.replace_editor.setPlaceholderText(self.tr('Replace'))
        self.replace_btn = ClickableLabel(None, self)
        self.replace_btn.setObjectName(self.tr('ReplaceBtn'))
        self.replace_btn.clicked.connect(self.on_replace_btn_clicked)
        self.replace_all_btn = ClickableLabel(None, self)
        self.replace_all_btn.setObjectName(self.tr('ReplaceAllBtn'))
        self.replace_all_btn.clicked.connect(self.on_replaceall_btn_clicked)

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        hlayout_bar1_0.addWidget(self.result_counter_label)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(10)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
        hlayout_bar1_1.addWidget(self.whole_word_toggle)
        hlayout_bar1_1.addWidget(self.prev_match_btn)
        hlayout_bar1_1.addWidget(self.next_match_btn)
        hlayout_bar1_1.setAlignment(hlayout_bar1_1.alignment() | Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_1.setSpacing(5)

        hlayout_bar1 = QHBoxLayout()
        hlayout_bar1.addLayout(hlayout_bar1_0)
        hlayout_bar1.addLayout(hlayout_bar1_1)
        
        self.replace_layout = hlayout_bar2 = QHBoxLayout()
        hlayout_bar2.addWidget(self.replace_editor)
        hlayout_bar2.addWidget(self.replace_btn)
        hlayout_bar2.addWidget(self.replace_all_btn)
        hlayout_bar2.addStretch()
        hlayout_bar2.addWidget(self.range_label)
        hlayout_bar2.addWidget(self.range_combobox)
        hlayout_bar2.setSpacing(5)

        vlayout = QVBoxLayout(self)
        vlayout.addLayout(hlayout_bar1)
        vlayout.addLayout(hlayout_bar2)

        if self.is_floating:
            self.search_editor.commit.connect(self.on_commit_search)
            self.close_btn = ClickableLabel(None, self)
            self.close_btn.setObjectName('SearchCloseBtn')
            self.close_btn.setToolTip(self.tr('Close (Escape)'))
            self.close_btn.clicked.connect(self.on_close_button_clicked)
            hlayout_bar1_1.addWidget(self.close_btn)
            e = QGraphicsDropShadowEffect(self)
            e.setOffset(0, 0)
            e.setBlurRadius(35)
            self.setGraphicsEffect(e)
            self.setFixedWidth(480)
            self.search_editor.setFixedWidth(200)
            self.replace_editor.setFixedWidth(200)
            self.search_editor.enter_pressed.connect(self.on_next_search_result)
            self.search_editor.shift_enter_pressed.connect(self.on_prev_search_result)

        self.adjustSize()
        self.config: ProgramConfig = None

    def on_close_button_clicked(self):
        self.hide()

    def hideEvent(self, e: QHideEvent) -> None:
        self.clean_highted()
        self.clearSearchResult()
        return super().hideEvent(e)

    def showEvent(self, e: QShowEvent) -> None:
        self.search_editor.setFocus()
        cursor = self.search_editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        self.search_editor.setTextCursor(cursor)
        text = self.search_editor.toPlainText()
        if text != '':
            self.on_commit_search()
        return super().showEvent(e)

    def on_editor_height_changed(self):
        self.adjustSize()

    def adjustSize(self) -> None:
        tgt_size = self.search_editor.height() + self.replace_editor.height() + 30
        self.setFixedHeight(tgt_size)

    def setReplaceWidgetsVisibility(self, visible: bool):
        self.replace_editor.setVisible(visible)
        self.replace_all_btn.setVisible(visible)
        self.replace_btn.setVisible(visible)

    def clean_highted(self):
        for e in self.search_rstedit_list:
            self.clean_editor_highted(e)

    def clean_editor_highted(self, e: QTextEdit):
        e.blockSignals(True)
        e.textCursor().beginEditBlock()
        cursor = QTextCursor(e.document())
        cursor.select(QTextCursor.SelectionType.Document)
        cf = cursor.charFormat()
        cf.setBackground(Qt.GlobalColor.transparent)
        cursor.setCharFormat(cf)
        e.textCursor().endEditBlock()
        e.blockSignals(False)

    def clearSearchResult(self):
        for rst in self.search_rstedit_list:
            rst.textChanged.disconnect(self.on_rst_text_changed)
        self.search_rstedit_list.clear()
        self.search_counter_list.clear()
        self.search_cursorpos_map.clear()
        self.current_edit = None
        self.current_cursor = None

    def on_rst_text_changed(self):
        edit: SourceTextEdit = self.sender()
        if edit.pre_editing:
            return
        idx = self.get_result_edit_index(edit)
        if idx < 0:
            return
        self.clean_editor_highted(edit)
        counter, pos_map = self._find_page_text(edit, self.search_editor.toPlainText(), self.get_find_flag())
        delta_count = counter - self.search_counter_list[idx]
        self.counter_sum += delta_count
        
        is_current_edit = False
        before_current = False
        if edit == self.current_edit:
            is_current_edit = True
        elif self.current_edit is not None and self.current_edit_index() > idx:
            before_current = True

        if counter > 0:
            self.search_counter_list[idx] = counter
            self.search_cursorpos_map[idx] = pos_map
            if is_current_edit:
                text = self.search_editor.toPlainText()
                if self.current_cursor.selectedText() != text:
                    new_cursor: QTextCursor = edit.document().find(self.search_editor.toPlainText(), self.current_cursor, self.get_find_flag() | QTextDocument.FindFlag.FindBackward)
                    if new_cursor.isNull():
                        self.setCurrentEditor(self.current_edit)
                    else:
                        self.current_cursor = new_cursor
                        self.result_pos = pos_map[new_cursor.position()]
                        if idx > 0:
                            self.result_pos += sum(self.search_counter_list[ :idx])
                        self.highlight_current_text()
                else:
                    self.result_pos = pos_map[self.current_cursor.position()]
                    if idx > 0:
                        self.result_pos += sum(self.search_counter_list[ :idx])
                    self.highlight_current_text()
            elif before_current:
                self.result_pos += delta_count
        else:
            edit = self.search_rstedit_list.pop(idx)
            edit.textChanged.disconnect(self.on_rst_text_changed)
            self.search_cursorpos_map.pop(idx)
            if len(self.search_rstedit_list) == 0:
                self.clearSearchResult()
            elif self.current_edit is not None:
                if is_current_edit:
                    if idx >= len(self.search_rstedit_list):
                        self.setCurrentEditor(self.search_rstedit_list[0])
                    else:
                        self.setCurrentEditor(self.search_rstedit_list[idx])
                elif before_current:
                    self.result_pos += delta_count
        self.updateCounterText()

    def reInitialize(self):
        self.clearSearchResult()
        self.reinit.emit()

    def page_search(self, update_cursor=True):

        self.clean_highted()
        self.clearSearchResult()

        if not self.isVisible():
            return

        text = self.search_editor.toPlainText()
        if text == '':
            return

        search_range = self.range_combobox.currentIndex()
        search_src = search_range == 1
        search_trans = search_range == 0

        find_flag = self.get_find_flag()
        if search_src:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_source, text, find_flag)
        elif search_trans:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_trans, text, find_flag)
        else:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_source, text, find_flag)
                self.find_page_text(pw.e_trans, text, find_flag)

        if len(self.search_counter_list) > 0:
            self.counter_sum = sum(self.search_counter_list)
        else:
            self.counter_sum = 0

        if update_cursor:
            if len(self.search_rstedit_list) > 0:
                self.current_edit = self.search_rstedit_list[0]
                self.updateCurrentCursor()
            self.updateCounterText()

    def get_find_flag(self) -> QTextDocument.FindFlag:
        find_flag = QTextDocument.FindFlags()
        if self.case_sensitive_toggle.isChecked():
            find_flag |= QTextDocument.FindFlag.FindCaseSensitively
        if self.whole_word_toggle.isChecked():
            find_flag |= QTextDocument.FindFlag.FindWholeWords
        return find_flag

    def _find_page_text(self, text_edit: QTextEdit, text: str, find_flag: QTextDocument.FindFlag, highlight=True) -> Tuple[int, Dict]:
        text_edit.blockSignals(True)
        doc = text_edit.document()
        text_cursor = text_edit.textCursor()
        text_cursor.beginEditBlock()
        cursor = QTextCursor(doc)
        found_counter = 0
        pos_map = {}
        while True:
            cursor: QTextCursor = doc.find(text, cursor, options=find_flag)
            if cursor.isNull():
                break
            pos_map[cursor.position()] = found_counter
            found_counter += 1
            if highlight:
                cf = cursor.charFormat()
                cf.setBackground(HIGHLIGHT_COLOR)
                cursor.mergeCharFormat(cf)
        text_cursor.endEditBlock()
        text_edit.blockSignals(False)
        return found_counter, pos_map

    def find_page_text(self, text_edit: QTextEdit, text: str, find_flag):
        found_counter, pos_map = self._find_page_text(text_edit, text, find_flag)
        if found_counter > 0:
            self.search_rstedit_list.append(text_edit)
            text_edit.textChanged.connect(self.on_rst_text_changed)
            self.search_counter_list.append(found_counter)
            self.search_cursorpos_map.append(pos_map)

    def get_result_edit_index(self, result: SourceTextEdit) -> int:
        try:
            return self.search_rstedit_list.index(result)
        except ValueError:
            return -1

    def current_edit_index(self) -> int:
        if self.current_edit is None:
            return -1
        return self.get_result_edit_index(self.current_edit)

    def setCurrentEditor(self, edit: SourceTextEdit):
        
        if type(edit) == SourceTextEdit and self.range_combobox.currentIndex() == 0 \
            or type(edit) == TransPairWidget and self.range_combobox.currentIndex() == 1:
            edit = None
        self.current_edit = edit

        if edit is None:
            if len(self.search_rstedit_list) > 0:
                self.current_edit = self.search_rstedit_list[0]

        if self.current_edit is not None:
            self.updateCurrentCursor()
            idx = self.current_edit_index()
            pos_map = self.search_cursorpos_map[idx]
            self.result_pos = pos_map[self.current_cursor.position()]
            if idx > 0:
                self.result_pos += sum(self.search_counter_list[ :idx])
        else:
            self.current_cursor = None
        
        self.updateCounterText()
        self.highlight_current_text()

    def updateCurrentCursor(self, intro_cursor=False, backward=False):
        cursor = self.current_edit.textCursor()
        text = self.search_editor.toPlainText()
        if intro_cursor or cursor.selectedText() != text:
            cursor.clearSelection()
            
        doc = self.current_edit.document()
        find_flag = self.get_find_flag()
        if not cursor.hasSelection():
            if backward:
                cursor.movePosition(QTextCursor.MoveOperation.End)
                find_flag |= QTextDocument.FindFlag.FindBackward
            else:
                cursor.movePosition(QTextCursor.MoveOperation.Start)
            cursor: QTextCursor = doc.find(text, cursor, options=find_flag)
        else:
            sel_start = cursor.selectionStart()
            cursor: QTextCursor = doc.find(text, sel_start, options=find_flag)

        idx = self.current_edit_index()
        pos_map = self.search_cursorpos_map[idx]
        c_pos = cursor.position()
        if c_pos not in pos_map:
            find_flag |= QTextDocument.FindFlag.FindBackward
            for k, val in reversed(pos_map.items()):
                if k < c_pos:
                    text = self.search_editor.toPlainText()
                    cursor.setPosition(k-len(text))
                    cursor.setPosition(k, QTextCursor.MoveMode.KeepAnchor)
                    break

        self.current_cursor = cursor

    def updateCounterText(self):
        if self.current_cursor is None or len(self.search_rstedit_list) == 0:
            self.result_counter_label.setText(self.no_result_str)
        else:
            self.result_counter_label.setText(f'{self.result_pos + 1} of {self.counter_sum}')

    def move_cursor(self, step: int = 1) -> int:
        old_cursor = self.current_cursor
        old_edit = self.current_edit
        doc = self.current_edit.document()
        text = self.search_editor.toPlainText()

        find_flag = self.get_find_flag()
        len_text = len(text)
        if step < 0:
            find_flag |= QTextDocument.FindFlag.FindBackward
            new_cursor = self.current_cursor
            while not new_cursor.isNull() and \
                self.current_cursor.position() - new_cursor.position() < len_text:
                new_cursor = doc.find(text, new_cursor, find_flag)
        else:
            new_cursor: QTextCursor = doc.find(text, self.current_cursor, find_flag)
        if new_cursor.isNull():
            idx = self.current_edit_index() + step
            if idx >= len(self.search_rstedit_list) or idx < 0:
                return step     # return step value if next move will be out of page
            self.current_edit = self.search_rstedit_list[idx]
            self.updateCurrentCursor(intro_cursor=True, backward=step < 0)
        else:
            self.current_cursor = new_cursor

        old_edit.blockSignals(True)
        cf = old_cursor.charFormat()
        cf.setBackground(HIGHLIGHT_COLOR)
        old_cursor.setCharFormat(cf)
        old_edit.blockSignals(False)

        self.highlight_current_text()
        return 0

    def highlight_current_text(self):
        if self.current_edit is None or not self.current_cursor.hasSelection():
            return
        self.current_edit.blockSignals(True)
        cursor = self.current_edit.textCursor()
        if cursor.hasSelection():
            cursor.clearSelection()
        cursor.setPosition(self.current_cursor.position())
        self.current_edit.setTextCursor(cursor)
        
        cf = self.current_cursor.charFormat()
        cf.setBackground(CURRENT_TEXT_COLOR)
        self.current_cursor.setCharFormat(cf)
        self.current_edit.blockSignals(False)
        self.current_edit.setFocus()
        self.current_edit.ensure_visible.emit()

    def on_next_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(1)
        if move == 0:
            self.result_pos = min(self.result_pos + 1, self.counter_sum - 1)
            self.updateCounterText()

    def on_prev_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(-1)
        if move == 0:
            self.result_pos = max(self.result_pos - 1, 0)
            self.updateCounterText()

    def on_whole_word_clicked(self):
        self.config.fsearch_whole_word = self.whole_word_toggle.isChecked()
        self.page_search()

    def on_case_clicked(self):
        self.config.fsearch_case = self.case_sensitive_toggle.isChecked()
        self.page_search()

    def on_range_changed(self):
        self.config.fsearch_range = self.range_combobox.currentIndex()
        self.page_search()
    
    def set_config(self, config: ProgramConfig):
        self.config = config
        self.whole_word_toggle.setChecked(config.fsearch_whole_word)
        self.case_sensitive_toggle.setChecked(config.fsearch_case)
        self.range_combobox.setCurrentIndex(config.fsearch_range)

    def on_commit_search(self):
        self.page_search()

    def on_replaceall_btn_clicked(self):
        pass

    def on_replace_btn_clicked(self):
        pass

    def on_new_textblk(self, idx: int):
        if self.isVisible():
            pair_widget = self.pairwidget_list[idx]
            pair_widget.e_trans.textChanged.connect(self.on_nonrst_edit_text_changed)
            pair_widget.e_source.textChanged.connect(self.on_nonrst_edit_text_changed)

    def on_nonrst_edit_text_changed(self):
        edit: SourceTextEdit = self.sender()
        if not self.isVisible() or edit.pre_editing or edit in self.search_rstedit_list:
            return

        if type(edit) == SourceTextEdit and self.range_combobox.currentIndex() == 0 \
            or type(edit) == TransPairWidget and self.range_combobox.currentIndex() == 1:
            return
        
        text = self.search_editor.toPlainText()
        find_flag = self.get_find_flag()
        found_counter, pos_map = self._find_page_text(edit, text, find_flag)
        if found_counter > 0:
            current_idx = self.current_edit_index()
            insert_idx = 0
            for e in self.search_rstedit_list:
                if e.idx < edit.idx:
                    insert_idx += 1
                elif e.idx == edit.idx:
                    if type(edit) == TransTextEdit:
                        insert_idx += 1
            if current_idx >= insert_idx:
                self.result_pos += found_counter
            self.search_cursorpos_map.insert(insert_idx, pos_map)
            edit.textChanged.connect(self.on_rst_text_changed)
            self.search_counter_list.insert(insert_idx, found_counter)
            self.search_rstedit_list.insert(insert_idx, edit)
            self.counter_sum += found_counter
            
            self.updateCounterText()