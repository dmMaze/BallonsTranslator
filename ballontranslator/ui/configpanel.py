from typing import List, Union, Tuple
import json

from qtpy.QtWidgets import QLayout, QHBoxLayout, QVBoxLayout, QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox, QSplitter, QScrollArea, QGroupBox, QLineEdit
from qtpy.QtCore import Qt, QModelIndex, Signal, QSize
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QColor, QPalette
from qtpy import QtCore

from utils.logger import logger as LOGGER
from .stylewidgets import Widget, ConfigComboBox
from .misc import ProgramConfig, DLModuleConfig
from .constants import CONFIG_PATH, CONFIG_FONTSIZE_CONTENT, CONFIG_FONTSIZE_HEADER, CONFIG_FONTSIZE_TABLE, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN
from .dlconfig_parse_widgets import InpaintConfigPanel, TextDetectConfigPanel, TranslatorConfigPanel, OCRConfigPanel

class ConfigTextLabel(QLabel):
    def __init__(self, text: str, fontsize: int, font_weight: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText(text)
        font = self.font()
        if font_weight is not None:
            font.setWeight(font_weight)
        font.setPointSize(fontsize)
        self.setFont(font)

    def setActiveBackground(self):
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(30, 147, 229, 51))
        self.setPalette(pal)


class ConfigSubBlock(Widget):
    pressed = Signal(int, int)
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout=True) -> None:
        super().__init__()
        self.idx0: int = None
        self.idx1: int = None
        if vertical_layout:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)
        self.name = name
        if name is not None:
            textlabel = ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal)
            layout.addWidget(textlabel)
        if discription is not None:
            layout.addWidget(ConfigTextLabel(discription, CONFIG_FONTSIZE_CONTENT-2))
        if isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
        self.setContentsMargins(24, 6, 24, 6)

    def setIdx(self, idx0: int, idx1: int) -> None:
        self.idx0 = idx0
        self.idx1 = idx1

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        self.pressed.emit(self.idx0, self.idx1)
        return super().enterEvent(a0)


class ConfigBlock(Widget):
    sublock_pressed = Signal(int, int)

    def __init__(self, header: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.header = ConfigTextLabel(header, CONFIG_FONTSIZE_HEADER)
        self.vlayout = QVBoxLayout(self)
        self.vlayout.addWidget(self.header)
        self.setContentsMargins(24, 24, 24, 24)
        self.label_list = []
        self.subblock_list = []
        self.index: int = 0

    def setIndex(self, index: int):
        self.index = index

    def addLineEdit(self, name: str = None, discription: str = None, vertical_layout: bool = False):
        le = QLineEdit()
        le.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        le.setFixedHeight(45)
        sublock = ConfigSubBlock(le, name, discription, vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.addSublock(sublock)
        sublock.layout().setSpacing(20)
        return le

    def addTextLabel(self, text: str = None):
        label = ConfigTextLabel(text, CONFIG_FONTSIZE_HEADER)
        self.vlayout.addWidget(label)
        self.label_list.append(label)

    def addSublock(self, sublock: ConfigSubBlock):
        self.vlayout.addWidget(sublock)
        sublock.setIdx(self.index, len(self.label_list)-1)
        sublock.pressed.connect(lambda idx0, idx1: self.sublock_pressed.emit(idx0, idx1))
        self.subblock_list.append(sublock)

    def addCombobox(self, sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None) -> ConfigComboBox:
        combox = ConfigComboBox()
        combox.addItems(sel)
        if target_block is None:
            sublock = ConfigSubBlock(combox, name, discription, vertical_layout=vertical_layout)
            sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
            sublock.layout().setSpacing(20)
            self.addSublock(sublock)
            return combox, sublock
        else:
            layout = target_block.layout()
            layout.addSpacing(20)
            layout.addWidget(ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal))
            layout.addWidget(combox)
            return combox, target_block

    def addBlockWidget(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout: bool = False) -> ConfigSubBlock:
        sublock = ConfigSubBlock(widget, name, discription, vertical_layout)
        self.addSublock(sublock)
        return sublock

    def addCheckBox(self, name: str, discription: str = None) -> QCheckBox:
        checkbox = QCheckBox()
        if discription is not None:
            font = checkbox.font()
            font.setPointSizeF(CONFIG_FONTSIZE_CONTENT * 0.8)
            checkbox.setFont(font)
            checkbox.setText(discription)
            vertical_layout = True
        else:
            vertical_layout = False
        sublock = ConfigSubBlock(checkbox, name, vertical_layout=vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.addSublock(sublock)
        return checkbox

    def getSubBlockbyIdx(self, idx: int) -> ConfigSubBlock:
        return self.subblock_list[idx]


class ConfigContent(QScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_block_list: List[ConfigBlock] = []
        self.scrollContent = QGroupBox()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(32, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.vlayout = vlayout
        self.active_label: ConfigTextLabel = None

    def addConfigBlock(self, block: ConfigBlock):
        self.vlayout.addWidget(block)
        self.config_block_list.append(block)

    def setActiveLabel(self, idx0: int, idx1: int):
        if self.active_label is not None:
            self.deactiveLabel()
        block = self.config_block_list[idx0]
        if idx1 >= 0:
            self.active_label = block.label_list[idx1]
        else:
            self.active_label = block.header
        self.active_label.setActiveBackground()
        # self.ensureWidgetVisible(self.active_label)

    def deactiveLabel(self):
        if self.active_label is not None:
            self.active_label.setAutoFillBackground(False)
            self.active_label = None


class TableItem(QStandardItem):
    def __init__(self, text, fontsize):
        super().__init__()
        font = self.font()
        font.setPointSizeF(fontsize)
        self.setFont(font)
        self.setText(text)
        self.setEditable(False)

    def setBold(self, bold: bool):
        font = self.font()
        font.setBold(bold)
        self.setFont(font)


class TreeModel(QStandardItemModel):
    # https://stackoverflow.com/questions/32229314/pyqt-how-can-i-set-row-heights-of-qtreeview
    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.SizeHintRole:
            size = QSize()
            item = self.itemFromIndex(index)
            size.setHeight(item.font().pointSize()+40)
            return size
        else:
            return super().data(index, role)


class ConfigTable(QTreeView):
    tableitem_pressed = Signal(int, int)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        treeModel = TreeModel()
        self.tm = treeModel
        self.setModel(treeModel)
        self.selected: TableItem = None
        self.last_selected: TableItem = None
        self.setHeaderHidden(True)
        self.setMinimumWidth(400)
        self.expandAll()

    def addHeader(self, header: str) -> TableItem:
        rootNode = self.model().invisibleRootItem()
        ti = TableItem(header, CONFIG_FONTSIZE_TABLE)
        rootNode.appendRow(ti)
        return ti

    def selectionChanged(self, selected: QtCore.QItemSelection, deselected: QtCore.QItemSelection) -> None:
        dis = deselected.indexes()
        sel = selected.indexes()
        model = self.model()
        self.last_selected = model.itemFromIndex(dis[0]) \
            if len(dis) > 0 else None
        
        self.selected = model.itemFromIndex(sel[0]) \
            if len(sel) > 0 else None
        for i in deselected.indexes():
            self.model().itemFromIndex(i).setBold(False)
        
        index = self.currentIndex()
        if index.isValid():
            self.model().itemFromIndex(index).setBold(True)
        super().selectionChanged(selected, deselected)

    def setCurrentItem(self, idx0, idx1):
        index = self.tm.item(idx0, 0).child(idx1).index()
        self.setCurrentIndex(index)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.selected is not None:
            parent = self.selected.parent()
            if parent is None:
                idx1 = -1
                idx0 = self.selected.row()
            else:
                idx1 = self.selected.row()
                idx0 = parent.row()
            self.tableitem_pressed.emit(idx0, idx1)


class GeneralPanel(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


        layout = QVBoxLayout(self)
        


class ConfigPanel(Widget):

    save_config = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        try:
            with open(CONFIG_PATH, 'r', encoding='utf8') as f:
                config_dict = json.loads(f.read())
            self.config = ProgramConfig(**config_dict)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            self.config = ProgramConfig()

        self.configTable = ConfigTable()
        self.configTable.tableitem_pressed.connect(self.onTableItemPressed)
        self.configContent = ConfigContent()
        dlConfigPanel, dltableitem = self.addConfigBlock(self.tr('DL Module'))
        generalConfigPanel, generalTableItem = self.addConfigBlock(self.tr('General'))
        
        label_text_det = self.tr('Text Detection')
        label_text_ocr = self.tr('OCR')
        label_inpaint = self.tr('Inpaint')
        label_translator = self.tr('Translator')
        label_startup = self.tr('Startup')
        # label_sources = self.tr('Sources')
        label_lettering = self.tr('Lettering')
    
        dltableitem.appendRows([
            TableItem(label_text_det, CONFIG_FONTSIZE_TABLE),
            TableItem(label_text_ocr, CONFIG_FONTSIZE_TABLE),
            TableItem(label_inpaint, CONFIG_FONTSIZE_TABLE),
            TableItem(label_translator, CONFIG_FONTSIZE_TABLE),
        ])
        generalTableItem.appendRows([
            TableItem(label_startup, CONFIG_FONTSIZE_TABLE),
            # TableItem(label_sources, CONFIG_FONTSIZE_TABLE),
            TableItem(label_lettering, CONFIG_FONTSIZE_TABLE)
        ])

        dlConfigPanel.addTextLabel(label_text_det)
        self.detect_config_panel = TextDetectConfigPanel(self.tr('Detector'))
        self.detect_sub_block = dlConfigPanel.addBlockWidget(self.detect_config_panel)
        
        dlConfigPanel.addTextLabel(label_text_ocr)
        self.ocr_config_panel = OCRConfigPanel(self.tr('OCR'))
        self.ocr_sub_block = dlConfigPanel.addBlockWidget(self.ocr_config_panel)

        dlConfigPanel.addTextLabel(label_inpaint)
        self.inpaint_config_panel = InpaintConfigPanel(self.tr('Inpainter'))
        self.inpaint_sub_block = dlConfigPanel.addBlockWidget(self.inpaint_config_panel)

        dlConfigPanel.addTextLabel(label_translator)
        self.trans_config_panel = TranslatorConfigPanel(label_translator)
        self.trans_sub_block = dlConfigPanel.addBlockWidget(self.trans_config_panel)

        generalConfigPanel.addTextLabel(label_startup)
        self.open_on_startup_checker = generalConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)

        # generalConfigPanel.addTextLabel(label_sources)
        # src_manual_str = self.tr('manual')
        # src_nhentai_str = self.tr('nhentai')
        # self.src_choice_combox = generalConfigPanel.addCombobox([src_manual_str, src_nhentai_str], self.tr('source'))
        # self.src_choice_combox.currentIndexChanged.connect(self.on_source_flag_changed)
        # self.src_link_textbox = generalConfigPanel.addLineEdit('source url')
        # self.src_link_textbox.textChanged.connect(self.on_source_link_changed)
        # self.src_force_download_checker = generalConfigPanel.addCheckBox(self.tr('Force download/redownload'))
        # self.src_force_download_checker.stateChanged.connect(self.on_source_force_download_changed)

        generalConfigPanel.addTextLabel(label_lettering)
        dec_program_str = self.tr('decide by program')
        use_global_str = self.tr('use global setting')
        
        self.let_fntsize_combox, letblk_0 = generalConfigPanel.addCombobox([dec_program_str, use_global_str], self.tr('font size'))
        self.let_fntsize_combox.currentIndexChanged.connect(self.on_fntsize_flag_changed)
        self.let_fntstroke_combox, _ = generalConfigPanel.addCombobox([dec_program_str, use_global_str], self.tr('stroke'), target_block=letblk_0)
        self.let_fntstroke_combox.currentIndexChanged.connect(self.on_fntstroke_flag_changed)
        self.let_fntcolor_combox, letblk_1 = generalConfigPanel.addCombobox([dec_program_str, use_global_str], self.tr('font & stroke color'))
        self.let_fntcolor_combox.currentIndexChanged.connect(self.on_fontcolor_flag_changed)
        self.let_alignment_combox, _ = generalConfigPanel.addCombobox([dec_program_str, use_global_str], self.tr('alignment'), target_block=letblk_1)
        self.let_alignment_combox.currentIndexChanged.connect(self.on_alignment_flag_changed)

        self.let_autolayout_checker = generalConfigPanel.addCheckBox(self.tr('Auto layout'), 
                discription=self.tr('Split translation into multi-lines according to the extracted balloon region. The font size will be adaptively resized if it is set to \"decide by program.\"'))
        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)
        self.let_uppercase_checker = generalConfigPanel.addCheckBox(self.tr('To uppercase'))
        self.let_uppercase_checker.stateChanged.connect(self.on_uppercase_changed)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.configTable)
        splitter.addWidget(self.configContent)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        hlayout = QHBoxLayout(self)

        hlayout.addWidget(splitter)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(96, 0, 0, 0)

    def addConfigBlock(self, header: str) -> Tuple[ConfigBlock, TableItem]:
        cb = ConfigBlock(header)
        cb.sublock_pressed.connect(self.onSublockPressed)
        self.configContent.addConfigBlock(cb)
        cb.setIndex(len(self.configContent.config_block_list)-1)
        ti = self.configTable.addHeader(header)
        return cb, ti

    def onSublockPressed(self, idx0, idx1):
        self.configTable.setCurrentItem(idx0, idx1)
        self.configContent.deactiveLabel()

    def onTableItemPressed(self, idx0, idx1):
        self.configContent.setActiveLabel(idx0, idx1)
        cb: ConfigBlock = self.configContent.config_block_list[idx0]
        self.configContent.ensureWidgetVisible(cb.getSubBlockbyIdx(idx1))

    def on_open_onstartup_changed(self):
        self.config.open_recent_on_startup = self.open_on_startup_checker.isChecked()

    def on_fntsize_flag_changed(self):
        self.config.let_fntsize_flag = self.let_fntsize_combox.currentIndex()

    def on_fntstroke_flag_changed(self):
        self.config.let_fntstroke_flag = self.let_fntstroke_combox.currentIndex()

    def on_autolayout_changed(self):
        self.config.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def on_uppercase_changed(self):
        self.config.let_uppercase_flag = self.let_uppercase_checker.isChecked()

    def on_fontcolor_flag_changed(self):
        self.config.let_fntcolor_flag = self.let_fntcolor_combox.currentIndex()

    def on_alignment_flag_changed(self):
        self.config.let_alignment_flag = self.let_alignment_combox.currentIndex()

    def on_source_flag_changed(self):
        self.config.src_choice_flag = self.src_choice_combox.currentIndex()

    def on_source_link_changed(self):
        self.config.src_link_flag = self.src_link_textbox.text()

    def on_source_force_download_changed(self):
        self.config.src_force_download_flag = self.src_force_download_checker.isChecked()

    def focusOnTranslator(self):
        idx0, idx1 = self.trans_sub_block.idx0, self.trans_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def showEvent(self, e) -> None:
        self.inpaint_sub_block.layout().addWidget(self.inpaint_config_panel)
        return super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.inpaint_sub_block.layout().removeWidget(self.inpaint_config_panel)
        return super().hideEvent(e)
        
    