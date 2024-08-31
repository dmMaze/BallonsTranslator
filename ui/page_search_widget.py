from qtpy.QtWidgets import QHBoxLayout, QComboBox, QTextEdit, QLabel, QPlainTextEdit, QCheckBox, QVBoxLayout,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QKeyEvent, QTextCursor, QHideEvent, QInputMethodEvent, QFontMetrics, QColor, QShowEvent, QSyntaxHighlighter, QTextCharFormat

from typing import List, Union, Tuple, Dict
import re

from utils.config import pcfg
from .custom_widget import Widget, ClickableLabel
from .textitem import TextBlkItem
from .textedit_area import TransPairWidget, SourceTextEdit, TransTextEdit

SEARCHRST_HIGHLIGHT_COLOR = QColor(30, 147, 229, 60)
CURRENT_TEXT_COLOR = QColor(244, 249, 28)


class Matched:
    def __init__(self, local_no: int, start: int, end: int) -> None:
        self.local_no = local_no
        self.start = start
        self.end = end


def match_text(pattern: re.Pattern, text: str) -> Tuple[int, Dict]:
    found_counter = 0
    match_map = {}
    rst_iter = pattern.finditer(text)
    for rst in rst_iter:
        span = rst.span()
        match_map[span[1]] = Matched(found_counter, span[0], span[1])
        found_counter += 1
    return found_counter, match_map


class HighlightMatched(QSyntaxHighlighter):

    def __init__(self, edit: SourceTextEdit, matched_map: dict = None):
        super().__init__(edit.document())
        
        self.case_sensitive = False
        self.whole_word = False
        if matched_map is None:
            self.matched_map: Dict = {}
        else:
            self.matched_map = matched_map
        self.current_start = -1
        self.edit = edit

    def setEditor(self, edit: SourceTextEdit):
        old_edit = self.edit
        if old_edit is not None:
            old_edit.highlighting = True
            old_edit.block_all_signals(True)
        if edit is not None:
            edit.highlighting = True
            self.setDocument(edit.document())
            edit.highlighting = False
        else:
            self.setDocument(None)
        self.edit = edit
        if old_edit is not None:
            old_edit.highlighting = False
            old_edit.block_all_signals(False)

    def set_matched_map(self, matched_map: dict):
        self.matched_map = matched_map
        self.rehighlight()

    def rehighlight(self) -> None:
        if self.edit is not None:
            self.edit.highlighting = True
        super().rehighlight()
        if self.edit is not None:
            self.edit.highlighting = False

    def set_current_span(self, start: int, end: int):
        self.current_start = start
        self.current_end = end
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        if self.edit is None:
            return
        self.edit.highlighting = True
        fmt = QTextCharFormat()
        fmt.setBackground(SEARCHRST_HIGHLIGHT_COLOR)
        block = self.currentBlock()
        block_start = block.position()
        block_end = block_start + block.length()
        matched: Matched
        for match_end, matched in self.matched_map.items():
            match_start = matched.start
            intersect_start = max(match_start, block_start)
            intersect_end = min(match_end, block_end)
            length = intersect_end - intersect_start
            if length > 0:
                self.setFormat(intersect_start - block_start, length, fmt)

        if self.current_start >= 0:
            intersect_start = max(self.current_start, block_start)
            intersect_end = min(self.current_end, block_end)
            length = intersect_end - intersect_start
            if length > 0:
                fmt.setBackground(CURRENT_TEXT_COLOR)
                self.setFormat(intersect_start - block_start, length, fmt)
        self.edit.highlighting = False


class SearchEditor(QPlainTextEdit):
    height_changed = Signal()
    commit = Signal()
    enter_pressed = Signal()
    shift_enter_pressed = Signal()
    def __init__(self, parent: QWidget = None, original_height: int = 32, commit_latency: int = -1, shift_enter_prev: bool = True, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.original_height = original_height
        self.commit_latency = commit_latency
        self.shift_enter_prev = shift_enter_prev
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
        h = int(h)
        if self.geometry().height() != h:
            self.setFixedHeight(max(h, self.original_height))
            self.height_changed.emit()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Return:
            if self.commit_timer is not None:
                self.commit_timer.stop()
            if e.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                if self.shift_enter_prev:
                    e.setAccepted(True)
                    self.shift_enter_pressed.emit()
                    return
            else:
                e.setAccepted(True)
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


class PageSearchWidget(Widget):

    search = Signal()
    replace_all = Signal()
    replace_one = Signal()

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent)

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list: List[int] = []
        self.highlighter_list: List[HighlightMatched] = []
        self.counter_sum = 0
        self.pairwidget_list: List[TransPairWidget] = []
        self.textblk_item_list: List[TextBlkItem] = []

        self.current_edit: SourceTextEdit = None
        self.current_cursor: QTextCursor = None
        self.current_highlighter: HighlightMatched = None
        self.result_pos = 0
        self.update_cursor_on_insert = True

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

        self.regex_toggle = QCheckBox(self)
        self.regex_toggle.setObjectName('RegexToggle')
        self.regex_toggle.setToolTip(self.tr('Use Regular Expression'))
        self.regex_toggle.clicked.connect(self.on_regex_clicked)

        self.range_combobox = QComboBox(self)
        self.range_combobox.addItems([self.tr('Translation'), self.tr('Source'), self.tr('All')])
        self.range_combobox.currentIndexChanged.connect(self.on_range_changed)
        self.range_label = QLabel(self)
        self.range_label.setText(self.tr('Range'))

        self.replace_editor = SearchEditor(self)
        self.replace_editor.setPlaceholderText(self.tr('Replace'))
        self.replace_btn = ClickableLabel(None, self)
        self.replace_btn.setObjectName('ReplaceBtn')
        self.replace_btn.clicked.connect(self.on_replace_btn_clicked)
        self.replace_btn.setToolTip(self.tr('Replace'))
        self.replace_all_btn = ClickableLabel(None, self)
        self.replace_all_btn.setObjectName('ReplaceAllBtn')
        self.replace_all_btn.clicked.connect(self.on_replaceall_btn_clicked)
        self.replace_all_btn.setToolTip(self.tr('Replace All'))

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        hlayout_bar1_0.addWidget(self.result_counter_label)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(10)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
        hlayout_bar1_1.addWidget(self.whole_word_toggle)
        hlayout_bar1_1.addWidget(self.regex_toggle)
        hlayout_bar1_1.addWidget(self.prev_match_btn)
        hlayout_bar1_1.addWidget(self.next_match_btn)
        hlayout_bar1_1.setAlignment(hlayout_bar1_1.alignment() | Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_1.setSpacing(5)

        hlayout_bar1 = QHBoxLayout()
        hlayout_bar1.addLayout(hlayout_bar1_0)
        hlayout_bar1.addLayout(hlayout_bar1_1)
        
        hlayout_bar2 = QHBoxLayout()
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
        self.setFixedWidth(520)
        self.search_editor.setFixedWidth(200)
        self.replace_editor.setFixedWidth(200)
        self.search_editor.enter_pressed.connect(self.on_next_search_result)
        self.search_editor.shift_enter_pressed.connect(self.on_prev_search_result)

        self.adjustSize()
        

    def on_close_button_clicked(self):
        self.hide()

    def hideEvent(self, e: QHideEvent) -> None:
        self.clean_highlighted()
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

    def clean_highlighted(self):
        for ii, e in enumerate(self.search_rstedit_list):
            self.highlighter_list[ii].setEditor(None)

    def clearSearchResult(self):
        for rst, hightlighter in zip(self.search_rstedit_list, self.highlighter_list):
            rst.text_changed.disconnect(self.on_rst_text_changed)
            hightlighter.setDocument(None)
        self.search_rstedit_list.clear()
        self.search_counter_list.clear()
        self.highlighter_list.clear()

        self.current_edit = None
        self.current_highlighter = None
        self.current_cursor = None
        self.updateCounterText()

    def on_rst_text_changed(self):
        edit: SourceTextEdit = self.sender()
        if edit.pre_editing or edit.highlighting:
            return

        idx = self.get_result_edit_index(edit)
        if idx < 0:
            return

        highlighter = self.highlighter_list[idx]
        counter, matched_map = self._match_text(edit.toPlainText())

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
            if is_current_edit:
                cursor_end = self.current_cursor.selectionEnd() 
                if cursor_end not in matched_map:
                    matched = self.get_prev_match(cursor_end)
                    if matched is None:
                        self.setCurrentEditor(self.current_edit)
                    else:
                        self.current_cursor.setPosition(matched.start)
                        self.current_cursor.setPosition(matched.end, QTextCursor.MoveMode.KeepAnchor)
                        self.result_pos = matched_map[matched.end].local_no
                        if idx > 0:
                            self.result_pos += sum(self.search_counter_list[ :idx])
                        self.highlight_current_text()
                else:
                    self.result_pos = matched_map[cursor_end].local_no
                    if idx > 0:
                        self.result_pos += sum(self.search_counter_list[ :idx])
                    self.highlight_current_text()
            elif before_current:
                self.result_pos += delta_count
            highlighter.set_matched_map(matched_map)
        else:
            edit = self.search_rstedit_list.pop(idx)
            self.search_counter_list.pop(idx)
            edit.text_changed.disconnect(self.on_rst_text_changed)
            highlighter = self.highlighter_list.pop(idx)
            highlighter.setEditor(None)
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

    def page_search(self, update_cursor=True):

        self.clean_highlighted()
        self.clearSearchResult()

        if not self.isVisible():
            return

        text = self.search_editor.toPlainText()
        if text == '':
            self.updateCounterText()
            return

        search_range = self.range_combobox.currentIndex()
        search_src = search_range == 1
        search_trans = search_range == 0

        if search_src:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_source)
        elif search_trans:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_trans)
        else:
            for pw in self.pairwidget_list:
                self.find_page_text(pw.e_source)
                self.find_page_text(pw.e_trans)

        if len(self.search_counter_list) > 0:
            self.counter_sum = sum(self.search_counter_list)
        else:
            self.counter_sum = 0

        if update_cursor:
            if len(self.search_rstedit_list) > 0:
                self.setCurrentEditor(self.search_rstedit_list[0])
            else:
                self.updateCounterText()

    def get_regex_pattern(self) -> re.Pattern:
        target_text = self.search_editor.toPlainText()
        regexr = target_text
        if target_text == '':
            return None

        flag = re.DOTALL
        if not self.case_sensitive_toggle.isChecked():
            flag |= re.IGNORECASE
        if not self.regex_toggle.isChecked():
            regexr = re.escape(regexr)
        if self.whole_word_toggle.isChecked():
            regexr = r'\b' + target_text + r'\b'

        return re.compile(regexr, flag)

    def find_page_text(self, text_edit: QTextEdit):
        found_counter, pos_map = self._match_text(text_edit.toPlainText())
        if found_counter > 0:
            self.search_rstedit_list.append(text_edit)
            self.search_counter_list.append(found_counter)
            self.highlighter_list.append(HighlightMatched(text_edit, pos_map))
            text_edit.text_changed.connect(self.on_rst_text_changed)

    def _match_text(self, text: str) -> Tuple[int, Dict]:
        try:
            return match_text(self.get_regex_pattern(), text)
        except re.error:
            return 0, {}

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

        old_idx = self.current_edit_index()
        self.current_edit = edit

        if edit is None:
            if len(self.search_rstedit_list) > 0:
                self.current_edit = self.search_rstedit_list[0]
                self.current_highlighter = self.highlighter_list[0]

        if self.current_edit is not None:
            idx = self.current_edit_index()
            self.current_highlighter = self.highlighter_list[idx]
            self.updateCurrentCursor()
            matched_map = self.current_highlighter.matched_map
            matched: Matched = matched_map[self.current_cursor.selectionEnd()]
            self.result_pos = matched.local_no
            if idx > 0:
                self.result_pos += sum(self.search_counter_list[ :idx])
        else:
            self.current_cursor = None
            self.current_highlighter = None
        
        self.updateCounterText()
        self.highlight_current_text(old_idx)

    def updateCurrentCursor(self, intro_cursor=False, backward=False):
        cursor = self.current_edit.textCursor()
        text = self.search_editor.toPlainText()
        if intro_cursor or cursor.selectedText() != text:
            cursor.clearSelection()
            
        matched_map = self.current_highlighter.matched_map
        matched: Matched
        
        if not cursor.hasSelection():
            if backward:
                matched: Matched = matched_map[list(matched_map.keys())[-1]]
            else:
                matched: Matched = matched_map[list(matched_map.keys())[0]]
            cursor.setPosition(matched.start)
            cursor.setPosition(matched.end, QTextCursor.MoveMode.KeepAnchor)
        else:
            sel_start = cursor.selectionStart()
            for _, matched in matched_map.items():
                if matched.start >= sel_start:
                    cursor.setPosition(matched.start)
                    cursor.setPosition(matched.end, QTextCursor.MoveMode.KeepAnchor)
                    break

        c_pos = cursor.position()
        if c_pos not in matched_map:
            for k, matched in reversed(matched_map.items()):
                if k < c_pos:
                    cursor.setPosition(matched.start)
                    cursor.setPosition(matched.end, QTextCursor.MoveMode.KeepAnchor)
                    break

        if cursor is not None:
            if cursor.selectionEnd() not in self.current_highlighter.matched_map:
                for k, matched in self.current_highlighter.matched_map.items():
                    cursor.setPosition(matched.start)
                    cursor.setPosition(matched.end, QTextCursor.MoveMode.KeepAnchor)
                    break

        self.current_cursor = cursor

    def updateCounterText(self):
        if self.current_cursor is None or len(self.search_rstedit_list) == 0:
            self.result_counter_label.setText(self.no_result_str)
        else:
            self.result_counter_label.setText(f'{self.result_pos + 1} of {self.counter_sum}')

    def clean_current_selection(self):
        cursor = self.current_edit.textCursor()
        if cursor.hasSelection():
            cursor.clearSelection()
            self.current_edit.setTextCursor(cursor)

    def get_next_match(self, cursor_sel_start: int) -> Matched:
        if self.current_highlighter is None:
            return None
        matched: Matched
        for _, matched in self.current_highlighter.matched_map.items():
            if matched.start > cursor_sel_start:
                return matched
        return None

    def get_prev_match(self, cursor_sel_end: int) -> Matched:
        if self.current_highlighter is None:
            return None
        matched: Matched
        for _, matched in reversed(self.current_highlighter.matched_map.items()):
            if matched.end < cursor_sel_end:
                return matched
        return None

    def move_cursor(self, step: int = 1) -> int:
        cursor_reset = 0
        self.clean_current_selection()
        if step < 0:
            moved_matched = self.get_prev_match(self.current_cursor.selectionEnd())
        else:
            moved_matched = self.get_next_match(self.current_cursor.selectionStart())

        old_idx = -1
        if moved_matched is None:
            old_idx = self.current_edit_index()
            idx = old_idx + step
            # return step value if next move will be out of page
            num_rstedit = len(self.search_rstedit_list)
            if idx >= num_rstedit:
                cursor_reset = step
                idx = 0
            elif idx < 0:
                cursor_reset = step
                idx = num_rstedit - 1
            self.current_edit = self.search_rstedit_list[idx]
            self.current_highlighter = self.highlighter_list[idx]
            self.updateCurrentCursor(intro_cursor=True, backward=step < 0)
        else:
            self.current_cursor.setPosition(moved_matched.start)
            self.current_cursor.setPosition(moved_matched.end, QTextCursor.MoveMode.KeepAnchor)

        self.highlight_current_text(old_idx)
        return cursor_reset

    def highlight_current_text(self, old_idx: int = -1):
        if self.current_edit is None or not self.current_cursor.hasSelection():
            return

        idx = self.current_edit_index()
        if idx != -1:
            self.highlighter_list[idx].set_current_span(self.current_cursor.selectionStart(), self.current_cursor.selectionEnd())

        if old_idx != -1 and old_idx != idx:
            self.highlighter_list[old_idx].set_current_span(-1, -1)

        if self.isVisible():
            self.current_edit.ensure_scene_visible.emit()

    def on_next_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(1)
        if move == 0:
            self.result_pos = min(self.result_pos + 1, self.counter_sum - 1)
        else:
            self.result_pos = 0
        self.updateCounterText()

    def on_prev_search_result(self):
        if self.current_cursor is None:
            return
        move = self.move_cursor(-1)
        if move == 0:
            self.result_pos = max(self.result_pos - 1, 0)
        else:
            self.result_pos = self.counter_sum - 1
        self.updateCounterText()

    def on_whole_word_clicked(self):
        pcfg.fsearch_whole_word = self.whole_word_toggle.isChecked()
        self.page_search()

    def on_regex_clicked(self):
        pcfg.fsearch_regex = self.regex_toggle.isChecked()
        self.page_search()

    def on_case_clicked(self):
        pcfg.fsearch_case = self.case_sensitive_toggle.isChecked()
        self.page_search()

    def on_range_changed(self):
        pcfg.fsearch_range = self.range_combobox.currentIndex()
        self.page_search()

    def on_commit_search(self):
        self.page_search()
        self.highlight_current_text()

    def on_replaceall_btn_clicked(self):
        if self.counter_sum > 0:
            self.replace_all.emit()

    def on_replace_btn_clicked(self):
        if self.current_cursor is not None:
            self.replace_one.emit()

    def on_new_textblk(self, idx: int):
        if self.isVisible():
            pair_widget = self.pairwidget_list[idx]
            pair_widget.e_trans.text_changed.connect(self.on_nonrst_edit_text_changed)
            pair_widget.e_source.text_changed.connect(self.on_nonrst_edit_text_changed)

    def on_nonrst_edit_text_changed(self):
        edit: SourceTextEdit = self.sender()
        if not self.isVisible() or edit.pre_editing or edit in self.search_rstedit_list:
            return

        if type(edit) == SourceTextEdit and self.range_combobox.currentIndex() == 0 \
            or type(edit) == TransPairWidget and self.range_combobox.currentIndex() == 1:
            return

        text = self.search_editor.toPlainText()
        if text == '':
            return

        found_counter, match_map = self._match_text(edit.toPlainText())
        if found_counter > 0:
            current_idx = self.current_edit_index()
            insert_idx = 0
            for e in self.search_rstedit_list:
                if e.idx < edit.idx:
                    insert_idx += 1
                elif e.idx == edit.idx:
                    if type(edit) == TransTextEdit:
                        insert_idx += 1

            self.search_counter_list.insert(insert_idx, found_counter)
            self.search_rstedit_list.insert(insert_idx, edit)
            self.highlighter_list.insert(insert_idx, HighlightMatched(edit, match_map))
            edit.text_changed.connect(self.on_rst_text_changed)
            self.counter_sum += found_counter

            if current_idx != -1 and current_idx >= insert_idx:
                self.result_pos += found_counter
                self.updateCounterText()
            else:
                if self.update_cursor_on_insert:
                    self.result_pos = 0
                    self.setCurrentEditor(edit)
                else:
                    self.updateCounterText()