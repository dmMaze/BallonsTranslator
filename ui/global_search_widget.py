from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QComboBox, QStyledItemDelegate, QLabel, QTreeView, QCheckBox, QStyleOptionViewItem, QVBoxLayout, QStyle, QMessageBox, QStyle,  QApplication, QWidget
from qtpy.QtCore import Qt, QItemSelection, QSize, Signal, QUrl, QModelIndex, QRectF
from qtpy.QtGui import QFont, QPainter, QTextCursor, QStandardItemModel, QStandardItem, QAbstractTextDocumentLayout, QColor, QPalette, QTextDocument, QTextCharFormat

from typing import List, Union, Tuple, Dict
import re, time
import os.path as osp

from utils.logger import logger as LOGGER
from .page_search_widget import SearchEditor, HighlightMatched, SEARCHRST_HIGHLIGHT_COLOR
from .misc import ProgramConfig, doc_replace, doc_replace_no_shift
from .stylewidgets import Widget, NoBorderPushBtn, ProgressMessageBox
from .textitem import TextBlkItem, TextBlock
from .textedit_area import TransPairWidget, SourceTextEdit, TransTextEdit
from .imgtrans_proj import ProjImgTrans
from .io_thread import ThreadBase
from . import constants as C

SEARCHRST_FONTSIZE = 10.3

class HTMLDelegate( QStyledItemDelegate ):
    def __init__( self ):
        super().__init__()
        self.doc = QTextDocument()
        self.doc.setUndoRedoEnabled(False)

    def paint(self, painter, option, index):
        
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        painter.save()
        self.doc.setDefaultFont(options.font)
        self.doc.setHtml(options.text)
        
        options.text = ''
        
        painter.translate(options.rect.left(), options.rect.top())

        clip = QRectF(0, 0, options.rect.width(), options.rect.height())
        painter.setClipRect(clip)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        ctx.clip = clip
        ctx.palette.setColor(QPalette.ColorRole.Text, QColor(*C.FOREGROUND_FONTCOLOR))
        self.doc.documentLayout().draw(painter, ctx)
        painter.restore()
        style = QApplication.style() if options.widget is None else options.widget.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, options, painter)


def get_rstitem_renderhtml(text: str, span: Tuple[int, int], font: QFont = None) -> str:
    if text == '':
        return text
    doc = QTextDocument()
    if font is None:
        font = doc.defaultFont()
    font.setPointSizeF(SEARCHRST_FONTSIZE)
    doc.setDefaultFont(font)
    doc.setPlainText(text.replace('\n', ' '))
    cursor = QTextCursor(doc)
    cursor.setPosition(span[0])
    cursor.setPosition(span[1], QTextCursor.MoveMode.KeepAnchor)
    cfmt = QTextCharFormat()
    cfmt.setBackground(SEARCHRST_HIGHLIGHT_COLOR)
    cursor.setCharFormat(cfmt)
    html = doc.toHtml()
    cleaned_html = re.findall(r'<body(.*?)>(.*?)</body>', html, re.DOTALL)
    if len(cleaned_html) > 0:
        cleaned_html = cleaned_html[0]
        return f'<body{cleaned_html[0]}>{cleaned_html[1]}</body>'
    else:
        return ''

class SearchResultItem(QStandardItem):
    def __init__(self, text: str, span: Tuple[int, int], blk_idx: int, pagename: str, is_src: bool):
        super().__init__()
        self.text = text

        self.start = span[0]
        self.end = span[1]
        self.is_src = is_src
        self.blk_idx = blk_idx
        self.pagename = pagename
        self.setText(get_rstitem_renderhtml(text, span, font=self.font()))
        self.setEditable(False)


class PageSeachResultItem(QStandardItem):
    def __init__(self, pagename: str, result_counter: int, blkid2match: dict):
        super().__init__()
        self.setData(result_counter, Qt.ItemDataRole.UserRole)
        self.pagename = pagename
        self.setText(str(result_counter) + ' - ' + pagename)
        self.blkid2match = blkid2match
        font = self.font()
        font.setPointSizeF(SEARCHRST_FONTSIZE)
        self.setFont(font)
        self.setEditable(False)


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

    result_item_clicked = Signal(str, int, bool, int, int)

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        sm = SearchResultModel()
        self.sm = sm
        self.setItemDelegate(HTMLDelegate())
        self.root_item = sm.invisibleRootItem()
        self.setModel(sm)
        font = self.font()
        font.setPointSizeF(SEARCHRST_FONTSIZE)
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
                self.result_item_clicked.emit(sel.pagename, sel.blk_idx, sel.is_src, sel.start, sel.end)
        super().selectionChanged(selected, deselected)

    def addPage(self, pagename: str, num_result: int, blkid2match: dict) -> PageSeachResultItem:
        prst = PageSeachResultItem(pagename, num_result, blkid2match)
        self.root_item.appendRow(prst)
        return prst

    def clearPages(self):
        rc = self.root_item.rowCount()
        if rc > 0:
            self.root_item.removeRows(0, rc)

    def rowCount(self):
        return self.root_item.rowCount()
        

class GlobalReplaceThead(ThreadBase):

    finished = Signal()


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.srt: SearchResultTree = None
        self.pairwidget_list: List[TransPairWidget] = None
        self.textblk_item_list: List[TextBlkItem] = None
        self.proj: ProjImgTrans = None
        self.progress_bar = ProgressMessageBox('task')
        self.progress_bar.setTaskName(self.tr('Replace...'))
        self.searched_pattern: re.Pattern = None
        self.finished.connect(self.on_finished)

    def replace(self, target: str):
        msg = QMessageBox()
        msg.setText(self.tr('Replace all occurrences?'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.job = lambda : self._search_proj(target)
            self.progress_bar.updateTaskProgress(0)
            self.progress_bar.show()
            self.start()

    def _search_proj(self, target: str):
        row_count = self.srt.rowCount()
        doc = QTextDocument()
        doc.setUndoRedoEnabled(False)
        sceneitem_list = {'src': [], 'trans': []}
        background_list = {'src': [], 'trans': []}
        self.target_text = target
        
        for ii in range(row_count):
            page_rst_item: PageSeachResultItem = self.srt.sm.item(ii, 0)
            self.progress_bar.updateTaskProgress(int(ii / row_count * 100))
            if page_rst_item.pagename == self.proj.current_img:
                for idx in page_rst_item.blkid2match['src']:
                    src = self.pairwidget_list[idx].e_source
                    sceneitem_list['src'].append({
                        'edit': src, 
                        'replace': self.searched_pattern.sub(target, src.toPlainText())
                    })
                for idx, rstitem_list in page_rst_item.blkid2match['trans'].items():
                    edit = self.pairwidget_list[idx].e_trans
                    item = self.textblk_item_list[idx]
                    
                    sceneitem_list['trans'].append({
                        'edit': edit, 
                        'item': item,
                        'matched_map': [[rstitem.start, rstitem.end] for rstitem in rstitem_list]
                    })
                    
            else:
                for idx in page_rst_item.blkid2match['src']:
                    blk: TextBlock = self.proj.pages[page_rst_item.pagename][idx]
                    text = blk.get_text()
                    replace = self.searched_pattern.sub(target, text)
                    background_list['src'].append({
                        'ori': text, 
                        'replace': replace,
                        'pagename': page_rst_item.pagename,
                        'idx': idx
                    })
                    blk.text = replace
                
                for idx, rstitem_list in page_rst_item.blkid2match['trans'].items():
                    blk: TextBlock = self.proj.pages[page_rst_item.pagename][idx]
                    ori = blk.translation
                    replace = ''
                    ori_html = blk.rich_text
                    replace_html = ''
                    if blk.rich_text:
                        ori_html = blk.rich_text
                        doc.setHtml(blk.rich_text)
                        span_list = [[rstitem.start, rstitem.end] for rstitem in rstitem_list]
                        doc_replace(doc, span_list, target)
                        replace_html = doc.toHtml()
                        replace = doc.toPlainText()
                    else:
                        replace = self.searched_pattern.sub(target, ori)
                    blk.translation = replace
                    blk.rich_text = replace_html
                    background_list['trans'].append({
                        'ori': ori, 
                        'replace': replace,
                        'ori_html': ori_html,
                        'replace_html': replace_html,
                        'pagename': page_rst_item.pagename,
                        'idx': idx
                    })

        self.sceneitem_list = sceneitem_list
        self.background_list = background_list
        self.finished.emit()

    def on_finished(self):
        self.progress_bar.hide()

    def handleRunTimeException(self, msg: str, detail: str = None, verbose: str = ''):
        super().handleRunTimeException(msg, detail, verbose)
        self.progress_bar.hide()


class GlobalSearchWidget(Widget):

    search = Signal()
    replace_all = Signal()
    req_update_pagetext = Signal()
    req_move_page = Signal(str, bool)

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
        self.doc_edited_str = self.tr('Document changed. Press Enter to re-search.')
        self.search_rst_str = self.tr('Found results: ')
        self.result_label = QLabel(self.no_result_str)
        self.result_label.setMaximumHeight(32)

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
        self.replace_btn = NoBorderPushBtn(self.tr('Replace All'))
        self.replace_btn.clicked.connect(self.on_replace)
        self.replace_rerender_btn = NoBorderPushBtn(self.tr('Replace All and Re-render all pages'))
        self.replace_rerender_btn.clicked.connect(self.on_replace_rerender)
        self.replace_thread = GlobalReplaceThead()

        sp = self.replace_rerender_btn.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
        self.replace_rerender_btn.setSizePolicy(sp)

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
        vlayout.addWidget(self.result_label)
        vlayout.addWidget(self.search_tree)
        vlayout.addWidget(self.replace_btn)
        vlayout.addWidget(self.replace_rerender_btn)
        vlayout.setStretchFactor(self.search_tree, 10)
        vlayout.setSpacing(7)

        self.progress_bar = ProgressMessageBox('task')
        self.progress_bar.setTaskName(self.tr('Replace...'))
        self.progress_bar.hide()

    def setupReplaceThread(self, pairwidget_list: List[TransPairWidget], textblk_item_list: List[TextBlkItem]):
        self.pairwidget_list = self.replace_thread.pairwidget_list = pairwidget_list
        self.textblk_item_list = self.replace_thread.textblk_item_list = textblk_item_list
        self.replace_thread.srt = self.search_tree
        self.replace_thread.proj = self.imgtrans_proj

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
        
        try: 
            return re.compile(regexr, flag)
        except re.error:
            return None

    def commit_search(self):
        self.search_tree.clearPages()
        pattern = self.get_regex_pattern()
        if pattern is None:
            self.replace_thread.searched_pattern = None
            return

        self.req_update_pagetext.emit()
        self.counter_sum = 0

        match_src = self.range_combobox.currentIndex() != 0
        match_trans = self.range_combobox.currentIndex() != 1
        
        for pagename, page in self.imgtrans_proj.pages.items():
            page_match_counter = 0
            page_rstitem_list = []
            blkid2match = {'src': {}, 'trans': {}}
            blk: TextBlock
            for ii, blk in enumerate(page):
                if match_src: 
                    rst_span_list, match_counter = match_blk(pattern, blk, match_src=True)
                    if match_counter > 0:
                        rstitem_list = gen_searchitem_list(rst_span_list, blk.get_text(), ii, pagename, is_src=True)
                        blkid2match['src'][ii] = rstitem_list
                        page_rstitem_list += rstitem_list
                        page_match_counter += match_counter
                if match_trans:
                    rst_span_list, match_counter = match_blk(pattern, blk, match_src=False)
                    if match_counter > 0:
                        rstitem_list = gen_searchitem_list(rst_span_list, blk.translation, ii, pagename, is_src=False)
                        blkid2match['trans'][ii] = rstitem_list
                        page_rstitem_list += rstitem_list
                        page_match_counter += match_counter
            if page_match_counter > 0:
                self.counter_sum += page_match_counter
                pageitem = self.search_tree.addPage(pagename, page_match_counter, blkid2match)
                pageitem.appendRows(page_rstitem_list)

        self.search_tree.expandAll()
        self.updateResultText()
        self.replace_thread.searched_pattern = pattern

    def updateResultText(self):
        if self.counter_sum > 0:
            self.result_label.setText(self.search_rst_str + str(self.counter_sum))
        else:
            self.result_label.setText(self.no_result_str)

    def on_replace(self):
        if self.counter_sum < 1:
            return
        self.replace_thread.replace(self.replace_editor.toPlainText())

    def on_replace_rerender(self):
        if self.counter_sum < 1:
            return
        pattern = self.replace_thread.searched_pattern
        if pattern is None:
            return

        msg = QMessageBox()
        msg.setText(self.tr('Replace all occurrences re-render all pages? It can\'t be undone.'))
        
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.No:
            return

        self.num_pages = self.search_tree.rowCount()
        self.fin_page_counter = 0
        self.page_set = set()
        rerender_pages = []
        for ii in range(self.num_pages):
            pagename = self.search_tree.sm.item(ii, 0).pagename
            self.page_set.add(pagename)
            if pagename == self.imgtrans_proj.current_img:
                rerender_pages.insert(0, [pagename, ii])
            else:
                rerender_pages.append([pagename, ii])
        self.progress_bar.updateTaskProgress(0)
        self.progress_bar.show()
        target = self.replace_editor.toPlainText()

        replace_src = self.range_combobox.currentIndex() != 0
        replace_trans = self.range_combobox.currentIndex() != 1
        
        for pagename, page_row in rerender_pages:
            self.req_move_page.emit(pagename, False)
            page_rst_item: PageSeachResultItem = self.search_tree.sm.item(page_row, 0)

            if replace_src:
                for idx in page_rst_item.blkid2match['src']:
                    src = self.replace_thread.pairwidget_list[idx].e_source
                    src.setPlainText(pattern.sub(target, src.toPlainText()))

            if replace_trans:
                for idx, rstitem_list in page_rst_item.blkid2match['trans'].items():
                    item = self.textblk_item_list[idx]
                    span_list = [[rstitem.start, rstitem.end] for rstitem in rstitem_list]
                    doc_replace(item.document(), span_list, target)
        
        if len(rerender_pages) > 0:
            self.req_move_page.emit(pagename, True)
            self.set_document_edited()

    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        size.setWidth(360)
        return size

    def set_document_edited(self):
        if self.counter_sum > 0:
            self.search_tree.clearPages()
            self.result_label.setText(self.doc_edited_str)
            self.counter_sum = 0

    def on_img_writed(self, pagename: str):
        if not self.progress_bar.isVisible():
            return
        if pagename not in self.page_set:
            return
        else:
            self.page_set.remove(pagename)
            self.fin_page_counter += 1
            if self.fin_page_counter == self.num_pages:
                self.progress_bar.hide()
            else:
                self.progress_bar.updateTaskProgress(int(self.fin_page_counter / self.num_pages * 100))
