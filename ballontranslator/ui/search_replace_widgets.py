from qtpy.QtWidgets import QHBoxLayout, QComboBox, QTextEdit, QLabel, QTreeView, QPlainTextEdit, QCheckBox, QMessageBox, QVBoxLayout, QStyle, QSlider, QProxyStyle, QStyle,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QTimer, QEasingCurve, QPointF, QRect, Signal
from qtpy.QtGui import QKeyEvent, QTextDocument, QTextCursor, QHideEvent, QInputMethodEvent, QFontMetrics, QColor
from typing import List, Union, Tuple

from .stylewidgets import Widget, ClickableLabel
from .textitem import TextBlkItem
from .imgtranspanel import TransPairWidget

class SearchEditor(QPlainTextEdit):
    height_changed = Signal()
    commit = Signal()
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
            self.commit.emit()
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
        self.search_rst_list: List[QTextEdit] = []
        self.pairwidget_list: List[TransPairWidget] = []
        self.textblk_item_list: List[TextBlkItem] = []

        self.search_editor = SearchEditor(self, commit_latency=-1)
        
        self.search_editor.setPlaceholderText(self.tr('Find'))
        self.search_editor.height_changed.connect(self.on_editor_height_changed)
        self.result_counter = QLabel(self.tr('No result'))
        self.result_counter.setMaximumHeight(32)
        self.prev_match_btn = ClickableLabel(None, self)
        self.prev_match_btn.setObjectName('PrevMatchBtn')
        self.next_match_btn = ClickableLabel(None, self)
        self.next_match_btn.setObjectName('NextMatchBtn')
        self.case_sensitive_toggle = QCheckBox(self)
        self.case_sensitive_toggle.setObjectName('CaseSensitiveToggle')
        self.range_combobox = QComboBox(self)
        self.range_combobox.addItems([self.tr('Translation'), self.tr('Source'), self.tr('All')])
        self.range_label = QLabel(self)
        self.range_label.setText(self.tr('Range'))

        self.replace_editor = SearchEditor(self)
        self.replace_editor.setPlaceholderText(self.tr('Replace'))
        self.replace_btn = ClickableLabel(None, self)
        self.replace_btn.setObjectName(self.tr('ReplaceBtn'))
        self.replace_all_btn = ClickableLabel(None, self)
        self.replace_all_btn.setObjectName(self.tr('ReplaceAllBtn'))

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        hlayout_bar1_0.addWidget(self.result_counter)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(10)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
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
            self.search_editor.commit.connect(self.on_page_search)

            self.close_btn = ClickableLabel(None, self)
            self.close_btn.setObjectName('SearchCloseBtn')
            self.close_btn.clicked.connect(self.on_close_button_clicked)
            hlayout_bar1_1.addWidget(self.close_btn)
            e = QGraphicsDropShadowEffect(self)
            e.setOffset(0, 0)
            e.setBlurRadius(35)
            self.setGraphicsEffect(e)
            self.setFixedWidth(480)
            self.search_editor.setFixedWidth(200)
            self.replace_editor.setFixedWidth(200)

        self.adjustSize()

    def on_close_button_clicked(self):
        self.hide()

    def hideEvent(self, e: QHideEvent) -> None:
        return super().hideEvent(e)

    def on_editor_height_changed(self):
        self.adjustSize()

    def adjustSize(self) -> None:
        tgt_size = self.search_editor.height() + self.replace_editor.height() + 30
        self.setFixedHeight(tgt_size)

    def setReplaceWidgetsVisibility(self, visible: bool):
        self.replace_editor.setVisible(visible)
        self.replace_all_btn.setVisible(visible)
        self.replace_btn.setVisible(visible)

    def reInitialize(self):
        self.search_rst_list = []
        self.reinit.emit()

    def on_page_search(self):

        for e in self.search_rst_list:
            e.textCursor().beginEditBlock()
            cursor = QTextCursor(e.document())
            cursor.select(QTextCursor.SelectionType.Document)
            cf = cursor.charFormat()
            cf.setBackground(Qt.GlobalColor.transparent)
            cursor.setCharFormat(cf)
            e.textCursor().endEditBlock()
        self.search_rst_list = []

        text = self.search_editor.toPlainText()
        if text == '':
            return

        search_range = self.range_combobox.currentIndex()
        search_src = search_range == 1
        search_trans = search_range == 0
        
        find_flag = QTextDocument.FindFlag.FindBackward
        find_flag = QTextDocument.FindFlags()
        if self.case_sensitive_toggle.isChecked():
            find_flag |= QTextDocument.FindFlag.FindCaseSensitively

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

    def find_page_text(self, text_edit: QTextEdit, text: str, find_flag):
        doc = text_edit.document()
        text_edit.textCursor().beginEditBlock()
        cursor = QTextCursor(doc)
        found = False
        while True:
            cursor: QTextCursor = doc.find(text, cursor, options=find_flag)
            if cursor.isNull():
                break
            found = True
            cf = cursor.charFormat()
            cf.setBackground(QColor(30, 147, 229, 60))
            cursor.mergeCharFormat(cf)
        if found:
            self.search_rst_list.append(text_edit)
        text_edit.textCursor().endEditBlock()