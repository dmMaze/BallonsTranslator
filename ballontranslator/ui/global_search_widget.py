from qtpy.QtWidgets import QHBoxLayout, QComboBox, QSizePolicy, QLabel, QTreeView, QCheckBox, QMessageBox, QVBoxLayout, QStyle, QSlider, QStyle,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QItemSelection, QSize, Signal, QUrl, QThread
from qtpy.QtGui import QKeyEvent, QTextCursor, QStandardItemModel, QStandardItem, QFontMetrics, QColor, QShowEvent, QSyntaxHighlighter, QTextCharFormat
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from typing import List, Union, Tuple, Dict

from utils.logger import logger as LOGGER
from .page_search_widget import SearchEditor, HighlightMatched
from .misc import ProgramConfig
from .stylewidgets import Widget, ClickableLabel
from .textitem import TextBlkItem, TextBlock
from .textedit_area import TransPairWidget, SourceTextEdit, TransTextEdit
from .imgtrans_proj import ProjImgTrans
from .io_thread import ThreadBase

import re

class PageSearchThead(ThreadBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj: ProjImgTrans = None

    def search_proj(self, proj: ProjImgTrans, target: str, match_case: bool, whole_word: bool, search_range: int):
        if self.isRunning():
            LOGGER.warn('Terminate a running search thread')
            self.terminate()
        self.job = lambda : self._search_proj(proj, target, match_case, whole_word, search_range)
        self.start()

    def _search_proj(self, proj: ProjImgTrans, target: str, match_case: bool, whole_word: bool, search_range: int):
        pass


SEARCH_RESULT_FONTSIZE = 12


class SearchResultItem(QStandardItem):
    def __init__(self, text: str, span: Tuple[int, int], blk_idx: int, pagename: str, is_src: bool):
        super().__init__()
        self.text = text
        self.start = span[0]
        self.end = span[1]
        self.is_src = is_src
        self.blk_idx = blk_idx
        self.pagename = pagename

        font = self.font()
        font.setPointSizeF(SEARCH_RESULT_FONTSIZE)
        self.setFont(font)

        self.setText(text)
        self.setEditable(False)

    def setBold(self, bold: bool):
        font = self.font()
        font.setBold(bold)
        self.setFont(font)


class PageSeachResultItem(QStandardItem):
    def __init__(self, pagename: str):
        super().__init__()
        self.pagename = pagename
        self.setText(pagename)
        self.matched_span_list = []
        self.matched_blkidx_list = []
        self.matched_issrc_list = []
        font = self.font()
        font.setPointSizeF(SEARCH_RESULT_FONTSIZE)
        self.setFont(font)

    def addResult(self, matched_span: Tuple[int, int], text: str, blk_idx: int, is_src: bool) -> SearchResultItem:
        self.matched_span_list.append(matched_span)
        self.matched_blkidx_list.append(blk_idx)
        self.matched_issrc_list.append(is_src)
        rstitem = SearchResultItem(text, matched_span, blk_idx, self.pagename, is_src)
        self.appendRow(rstitem)
        return rstitem



def gen_searchitem_list(span_list: List[int], text: str, blk_idx: int, pagename: str, is_src: bool) -> List[SearchResultItem]:
    sr_list = []
    for span in span_list:
        sr_list.append(SearchResultItem(text, span, blk_idx, pagename, is_src))
    return sr_list

def match_blk(pattern: re.Pattern, blk: TextBlock, match_src: bool) -> Tuple[List[Tuple], int]:
    if match_src:
        rst_iter = pattern.finditer(blk.get_text())
    else:
        rst_iter = pattern.finditer(blk.translation)
    rst_span_list = []
    match_counter = 0
    for rst in rst_iter:
        rst_span_list.append(rst.span())
        match_counter += 1
    return rst_span_list, match_counter


class SearchResultModel(QStandardItemModel):
    # https://stackoverflow.com/questions/32229314/pyqt-how-can-i-set-row-heights-of-qtreeview
    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.SizeHintRole:
            size = QSize()
            item = self.itemFromIndex(index)
            size.setHeight(item.font().pointSize()+14)
            return size
        else:
            return super().data(index, role)


class SearchResultTree(QTreeView):

    result_item_clicked = Signal(str, int, bool)

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        sm = SearchResultModel()
        self.sm = sm
        self.root_item = sm.invisibleRootItem()
        self.setModel(sm)
        font = self.font()
        font.setPointSizeF(SEARCH_RESULT_FONTSIZE)
        self.setFont(font)
        self.setUniformRowHeights(True)
        self.selected: SearchResultItem = None
        self.last_selected: SearchResultItem = None
        self.setHeaderHidden(True)
        self.expandAll()

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        selected_indexes = selected.indexes()
        if len(selected_indexes) > 0: 
            sel: SearchResultItem = self.sm.itemFromIndex(selected_indexes[0])
            if isinstance(sel, SearchResultItem):
                self.result_item_clicked.emit(sel.pagename, sel.blk_idx, sel.is_src)
        super().selectionChanged(selected, deselected)

    def addPage(self, pagename: str) -> PageSeachResultItem:
        prst = PageSeachResultItem(pagename)
        self.root_item.appendRow(prst)
        return prst

    def clearPages(self):
        rc = self.root_item.rowCount()
        if rc > 0:
            self.root_item.removeRows(0, rc)
        


class GlobalSearchWidget(Widget):

    search = Signal()
    replace_all = Signal()
    req_update_pagetext = Signal()

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent)
        self.config: ProgramConfig = None
        self.imgtrans_proj: ProjImgTrans = None

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list: List[int] = []
        self.highlighter_list: List[HighlightMatched] = []
        self.counter_sum = 0
        self.pairwidget_list: List[TransPairWidget] = []
        self.textblk_item_list: List[TextBlkItem] = []

        self.current_edit: SourceTextEdit = None
        self.current_cursor: QTextCursor = None
        self.result_pos = 0

        self.search_editor = SearchEditor(self, commit_latency=-1)
        self.search_editor.setPlaceholderText(self.tr('Find'))
        self.search_editor.enter_pressed.connect(self.commit_search)
        
        self.no_result_str = self.tr('No results found. ')
        self.result_str = self.tr(' results')
        self.result_counter_label = QLabel(self.no_result_str)
        self.result_counter_label.setMaximumHeight(32)

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
        self.range_label.setText(self.tr(' in'))

        self.replace_editor = SearchEditor(self)
        self.replace_editor.setPlaceholderText(self.tr('Replace'))

        self.search_tree = SearchResultTree(self)

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(10)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
        hlayout_bar1_1.addWidget(self.whole_word_toggle)
        hlayout_bar1_1.addWidget(self.regex_toggle)
        hlayout_bar1_1.setAlignment(hlayout_bar1_1.alignment() | Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_1.setSpacing(5)

        hlayout_bar1 = QHBoxLayout()
        hlayout_bar1.addLayout(hlayout_bar1_0)
        hlayout_bar1.addLayout(hlayout_bar1_1)
        
        hlayout_bar2 = QHBoxLayout()
        hlayout_bar2.addWidget(self.replace_editor)
        hlayout_bar2.addWidget(self.range_label)
        hlayout_bar2.addWidget(self.range_combobox)
        hlayout_bar2.setSpacing(5)

        vlayout = QVBoxLayout(self)
        vlayout.addLayout(hlayout_bar1)
        vlayout.addLayout(hlayout_bar2)
        vlayout.addWidget(self.result_counter_label)
        vlayout.addWidget(self.search_tree)
        vlayout.setStretchFactor(self.search_tree, 10)
        vlayout.setSpacing(7)

    def on_whole_word_clicked(self):
        self.config.gsearch_whole_word = self.whole_word_toggle.isChecked()
        self.commit_search()

    def on_regex_clicked(self):
        self.config.gsearch_regex = self.regex_toggle.isChecked()
        self.commit_search()

    def on_case_clicked(self):
        self.config.gsearch_case = self.case_sensitive_toggle.isChecked()
        self.commit_search()

    def on_range_changed(self):
        self.config.gsearch_range = self.range_combobox.currentIndex()
        self.commit_search()
    
    def set_config(self, config: ProgramConfig):
        self.config = config
        self.whole_word_toggle.setChecked(config.gsearch_whole_word)
        self.case_sensitive_toggle.setChecked(config.gsearch_case)
        self.regex_toggle.setChecked(config.gsearch_regex)
        self.range_combobox.setCurrentIndex(config.gsearch_range)

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

    def commit_search(self):
        self.search_tree.clearPages()
        pattern = self.get_regex_pattern()
        if pattern is None:
            return

        self.req_update_pagetext.emit()

        proj_match_counter = 0
        match_src = True if self.range_combobox.currentIndex() != 0 else False
        match_trans = True if self.range_combobox.currentIndex() != 1 else False
        
        for pagename, page in self.imgtrans_proj.pages.items():
            page_match_counter = 0
            page_rstitem_list = []
            blk: TextBlock
            for ii, blk in enumerate(page):
                if match_src: 
                    rst_span_list, match_counter = match_blk(pattern, blk, match_src=True)
                    if match_counter > 0:
                        page_rstitem_list += gen_searchitem_list(rst_span_list, blk.get_text(), ii, pagename, is_src=True)
                        page_match_counter += match_counter
                if match_trans:
                    rst_span_list, match_counter = match_blk(pattern, blk, match_src=False)
                    if match_counter > 0:
                        page_rstitem_list += gen_searchitem_list(rst_span_list, blk.translation, ii, pagename, is_src=False)
                        page_match_counter += match_counter
            if page_match_counter > 0:
                proj_match_counter += page_match_counter
                pageitem = self.search_tree.addPage(pagename)
                pageitem.appendRows(page_rstitem_list)

        self.search_tree.expandAll()


    

    def on_replaceall_btn_clicked(self):
        pass

    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        size.setWidth(360)
        return size