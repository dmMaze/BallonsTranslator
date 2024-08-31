import re, traceback

from qtpy.QtWidgets import QHeaderView, QTableView, QWidget, QVBoxLayout, QDialog
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QStandardItem, QStandardItemModel
from typing import List, Dict

from utils.logger import logger as LOGGER
from utils.fontformat import FontFormat
from .custom_widget import NoBorderPushBtn

class KeywordSubWidget(QDialog):

    hide_signal = Signal()
    load_preset = Signal(FontFormat)

    def __init__(self, title: str, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent=parent, *args, **kwargs)
        self.setWindowTitle(title)
        self.setModal(True)
        self.sublist: List[Dict] = []

        self.submodel = QStandardItemModel()
        self.submodel.setHorizontalHeaderLabels([
            self.tr("Keyword"),
            self.tr("Substitution"),
            self.tr("Use regex"),
            self.tr("Case sensitive")
        ])

        self.subtable = table = QTableView(self)
        table.setModel(self.submodel)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch) 

        self.newbtn = NoBorderPushBtn(self.tr("New"), self)
        self.newbtn.clicked.connect(self.on_new_subpair)
        self.delbtn = NoBorderPushBtn(self.tr("Delete"), self)
        self.delbtn.clicked.connect(self.on_del_selected)
        layout = QVBoxLayout(self)
        layout.addWidget(table)
        layout.addWidget(self.newbtn)
        layout.addWidget(self.delbtn)

        self.submodel.itemChanged.connect(self.on_item_changed)
        self.changing_rows = False

        self.setMinimumWidth(700)

    def loadCfgSublist(self, sublist: List):
        self.sublist = sublist
        for sub in sublist:
            self.add_subpair(**sub, save2sublist=False)

    def on_new_subpair(self):
        self.add_subpair()

    def add_subpair(self, keyword: str = '', sub: str = '', use_reg: bool = False, case_sens: bool = True, save2sublist=True):
        self.changing_rows = True

        row = self.submodel.rowCount()
        kitem = QStandardItem(keyword)
        sitem = QStandardItem(sub)
        ritem = QStandardItem()
        ritem.setCheckable(True)
        ritem.setCheckState(Qt.CheckState.Checked if use_reg else Qt.CheckState.Unchecked)
        ritem.setEditable(False)
        citem = QStandardItem()
        citem.setCheckable(True)
        citem.setCheckState(Qt.CheckState.Checked if case_sens else Qt.CheckState.Unchecked)
        citem.setEditable(False)

        self.submodel.setItem(row, 0, kitem)
        self.submodel.setItem(row, 1, sitem)
        self.submodel.setItem(row, 2, ritem)
        self.submodel.setItem(row, 3, citem)

        if save2sublist:
            self.sublist.append({'keyword': keyword, 'sub': sub, 'use_reg': use_reg, 'case_sens': case_sens})
        self.changing_rows = False

    def delete_subpairs(self, del_ids: List[int]):
        self.changing_rows = True
        del_ids.sort(reverse=True)
        for idx in del_ids:
            self.sublist.pop(idx)
            self.submodel.removeRow(idx)
        self.changing_rows = False
        pass

    def on_del_selected(self):
        sel_ids = self.subtable.selectedIndexes()
        delist = set()
        for idx in sel_ids:
            delist.add(self.submodel.itemFromIndex(idx).row())
        delist = list(delist)
        self.delete_subpairs(delist)

    def on_item_changed(self, item: QStandardItem):
        if self.changing_rows:
            return

        row, col = item.row(), item.column()
        subpair = self.sublist[row]
        if col == 0:
            subpair['keyword'] = item.text()
        elif col == 1:
            subpair['sub'] = item.text()
        elif col == 2:
            subpair['use_reg'] = item.checkState() == Qt.CheckState.Checked
        elif col == 3:
            subpair['case_sens'] = item.checkState() == Qt.CheckState.Checked

    def sub_text(self, text: str) -> str:
        for ii, subpair in enumerate(self.sublist):
            k = subpair['keyword']
            if k == '':
                continue
            
            regexr = k
            flag = re.DOTALL
            if not subpair['case_sens']:
                flag |= re.IGNORECASE
            if not subpair['use_reg']:
                regexr = re.escape(regexr)
            try: 
                text = re.sub(regexr, subpair['sub'], text)
            except Exception as e:
                LOGGER.error(f'Invalid regex expression {regexr} at {ii+1}:')
                LOGGER.error(traceback.format_exc())
                continue

        return text