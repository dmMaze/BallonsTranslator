from typing import List, Union, Tuple

from qtpy.QtWidgets import QKeySequenceEdit, QLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox, QSplitter, QScrollArea, QGroupBox, QLineEdit
from qtpy.QtCore import Qt, Signal, QSize, QEvent, QItemSelection
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QColor, QPalette, QIntValidator, QValidator, QFocusEvent
from qtpy import API

from utils import shared as C

# nuitka seems to require import QtCore explicitly 
if C.FLAG_QT6:
    if API == 'pyside6':
        from PySide6 import QtCore
    else:
        from PyQt6 import QtCore
else:
    from PyQt5 import QtCore

from .stylewidgets import Widget, ConfigComboBox
from utils.config import pcfg
from utils.shared import CONFIG_FONTSIZE_CONTENT, CONFIG_FONTSIZE_HEADER, CONFIG_FONTSIZE_TABLE, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN
from .dlconfig_parse_widgets import InpaintConfigPanel, TextDetectConfigPanel, TranslatorConfigPanel, OCRConfigPanel

class CustomIntValidator(QIntValidator):

    def __init__(self, bottom: int, top: int, ndigits: int = None, parent = None):
        super().__init__(bottom=bottom, top=top, parent=parent)
        self.ndigits = ndigits

    def validate(self, s: str, pos: int) -> object:
        if not s.isnumeric():
            if s != '':
                return (QValidator.State.Invalid, s, pos)
            else:
                return (QValidator.State.Intermediate, s, pos)
            
        s_ori = s
        d = int(s)
        s = str(d)
        if len(s) != len(s_ori):
            pos -= len(s_ori) - len(s)
        if len(s) > self.ndigits:
            ndel = len(s) - self.ndigits
            s = s[ndel:]
            pos -= ndel
        else:
            if d > self.top():
                if s[-1] == '0':
                    d = self.top()
                else:
                    d = d % self.top()
            d = max(d, self.bottom())
            s = str(d)
        return (QValidator.State.Acceptable, s, pos)
    
class PercentageLineEdit(QLineEdit):

    finish_edited = Signal(str)

    def __init__(self, default_value: str = '100', parent=None) -> None:
        super().__init__(default_value, parent=parent)
        validator = CustomIntValidator(0, 100, 3)
        self.setValidator(validator)
        self.textEdited.connect(self.on_text_edited)
        self._edited = False

    def on_text_edited(self):
        self._edited = True

    def focusOutEvent(self, e: QFocusEvent) -> None:
        if self._edited:
            text = self.text()
            if not text.isnumeric():
                text = '100'
                self.setText(text)
            self.finish_edited.emit(text)

        return super().focusOutEvent(e)



class ConfigTextLabel(QLabel):
    def __init__(self, text: str, fontsize: int, font_weight: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText(text)
        font = self.font()
        if font_weight is not None:
            font.setWeight(font_weight)
        font.setPointSizeF(fontsize)
        self.setFont(font)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setOpenExternalLinks(True)

    def setActiveBackground(self):
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(30, 147, 229, 51))
        self.setPalette(pal)


class ConfigSubBlock(Widget):
    pressed = Signal(int, int)
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout=True, insert_stretch: bool = False, content_margins = (24, 6, 24, 6)) -> None:
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
            self.name_label = textlabel
            layout.addWidget(textlabel)
        if discription is not None:
            layout.addWidget(ConfigTextLabel(discription, CONFIG_FONTSIZE_CONTENT-2))
        if insert_stretch:
            layout.insertStretch(-1)
        if isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
        self.widget = widget
        self.setContentsMargins(*content_margins)

    def setIdx(self, idx0: int, idx1: int) -> None:
        self.idx0 = idx0
        self.idx1 = idx1

    def enterEvent(self, e: QEvent) -> None:
        self.pressed.emit(self.idx0, self.idx1)
        return super().enterEvent(e)
    

def combobox_with_label(sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True, parent: QWidget = None, insert_stretch: bool = False) -> Tuple[ConfigComboBox, QWidget]:
    combox = ConfigComboBox(fix_size=fix_size, scrollWidget=parent)
    combox.addItems(sel)
    if target_block is None:
        sublock = ConfigSubBlock(combox, name, discription, vertical_layout=vertical_layout, insert_stretch=insert_stretch)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().setSpacing(20)
        return combox, sublock
    else:
        layout = target_block.layout()
        layout.addSpacing(20)
        layout.addWidget(ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal))
        layout.addWidget(combox)
        return combox, target_block


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
        return le, sublock

    def addTextLabel(self, text: str = None):
        label = ConfigTextLabel(text, CONFIG_FONTSIZE_HEADER)
        self.vlayout.addWidget(label)
        self.label_list.append(label)

    def addSublock(self, sublock: ConfigSubBlock):
        self.vlayout.addWidget(sublock)
        sublock.setIdx(self.index, len(self.label_list)-1)
        sublock.pressed.connect(lambda idx0, idx1: self.sublock_pressed.emit(idx0, idx1))
        self.subblock_list.append(sublock)

    def addCombobox(self, sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True) -> Tuple[ConfigComboBox, QWidget]:
        combox, sublock = combobox_with_label(sel, name, discription, vertical_layout, target_block, fix_size, parent=self)
        if target_block is None:
            self.addSublock(sublock)
        return combox, sublock

    def addBlockWidget(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout: bool = False) -> ConfigSubBlock:
        sublock = ConfigSubBlock(widget, name, discription, vertical_layout)
        self.addSublock(sublock)
        return sublock

    def addCheckBox(self, name: str, discription: str = None, sublock: ConfigSubBlock = None) -> QCheckBox:
        checkbox = QCheckBox()
        if discription is not None:
            font = checkbox.font()
            font.setPointSizeF(CONFIG_FONTSIZE_CONTENT * 0.8)
            checkbox.setFont(font)
            checkbox.setText(discription)
            vertical_layout = True
        else:
            vertical_layout = False
        if sublock is None:
            sublock = ConfigSubBlock(checkbox, name, vertical_layout=vertical_layout)
            if vertical_layout is False:
                sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
            self.addSublock(sublock)
        else:
            sublock.layout().addWidget(checkbox)
        return checkbox, sublock

    def getSubBlockbyIdx(self, idx: int) -> ConfigSubBlock:
        return self.subblock_list[idx]


class ConfigContent(QScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_block_list: List[ConfigBlock] = []
        self.scrollContent = Widget()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.setContentsMargins(0, 0, 0, 0)
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
        if C.USE_PYSIDE6:
            self.ensureWidgetVisible(self.active_label, ymargin=self.active_label.height() * 7)
        else:
            self.ensureWidgetVisible(self.active_label, yMargin=self.active_label.height() * 7)

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
            size.setHeight(item.font().pointSize()+20)
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
        self.setMinimumWidth(260)

    def addHeader(self, header: str) -> TableItem:
        rootNode = self.model().invisibleRootItem()
        ti = TableItem(header, CONFIG_FONTSIZE_TABLE)
        rootNode.appendRow(ti)
        return ti

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
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


class ConfigPanel(Widget):

    save_config = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
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
        label_lettering = self.tr('Lettering')
        label_save = self.tr('Save')
        label_saladict = self.tr('SalaDict')
    
        dltableitem.appendRows([
            TableItem(label_text_det, CONFIG_FONTSIZE_TABLE),
            TableItem(label_text_ocr, CONFIG_FONTSIZE_TABLE),
            TableItem(label_inpaint, CONFIG_FONTSIZE_TABLE),
            TableItem(label_translator, CONFIG_FONTSIZE_TABLE),
        ])
        generalTableItem.appendRows([
            TableItem(label_startup, CONFIG_FONTSIZE_TABLE),
            TableItem(label_lettering, CONFIG_FONTSIZE_TABLE),
            TableItem(label_save, CONFIG_FONTSIZE_TABLE),
            TableItem(label_saladict, CONFIG_FONTSIZE_TABLE),
        ])

        dlConfigPanel.addTextLabel(label_text_det)
        self.detect_config_panel = TextDetectConfigPanel(self.tr('Detector'), scrollWidget=self)
        self.detect_sub_block = dlConfigPanel.addBlockWidget(self.detect_config_panel)
        
        dlConfigPanel.addTextLabel(label_text_ocr)
        self.ocr_config_panel = OCRConfigPanel(self.tr('OCR'), scrollWidget=self)
        self.ocr_sub_block = dlConfigPanel.addBlockWidget(self.ocr_config_panel)

        dlConfigPanel.addTextLabel(label_inpaint)
        self.inpaint_config_panel = InpaintConfigPanel(self.tr('Inpainter'), scrollWidget=self)
        self.inpaint_sub_block = dlConfigPanel.addBlockWidget(self.inpaint_config_panel)

        dlConfigPanel.addTextLabel(label_translator)
        self.trans_config_panel = TranslatorConfigPanel(label_translator, scrollWidget=self)
        self.trans_sub_block = dlConfigPanel.addBlockWidget(self.trans_config_panel)

        generalConfigPanel.addTextLabel(label_startup)
        self.open_on_startup_checker, _ = generalConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)

        generalConfigPanel.addTextLabel(label_lettering)
        dec_program_str = self.tr('decide by program')
        use_global_str = self.tr('use global setting')

        global_fntfmt_widget = QWidget()
        global_fntfmt_layout = QGridLayout(global_fntfmt_widget)
        global_fntfmt_layout.setSpacing(0)
        global_fntfmt_widget.setContentsMargins(0, 0, 0, 0)

        b = generalConfigPanel.addBlockWidget(global_fntfmt_widget)
        b.layout().setContentsMargins(0, 0, 0, 0)
        b.setContentsMargins(0, 0, 0, 0)
        self.let_fntsize_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Size'), parent=self, insert_stretch=True)
        global_fntfmt_layout.addWidget(sublock, 0, 0)

        self.let_fntsize_combox.currentIndexChanged.connect(self.on_fntsize_flag_changed)
        self.let_fntstroke_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Size'), parent=self, insert_stretch=True)
        self.let_fntstroke_combox.currentIndexChanged.connect(self.on_fntstroke_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 0, 1)
        
        self.let_fntcolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Color'), parent=self, insert_stretch=True)
        self.let_fntcolor_combox.currentIndexChanged.connect(self.on_fontcolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 0)
        self.let_fnt_scolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Color'), parent=self, insert_stretch=True)
        self.let_fnt_scolor_combox.currentIndexChanged.connect(self.on_font_scolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 1)

        self.let_effect_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Effect'), parent=self, insert_stretch=True)
        self.let_effect_combox.currentIndexChanged.connect(self.on_effect_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 0)
        self.let_alignment_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Alignment'), parent=self, insert_stretch=True)
        self.let_alignment_combox.currentIndexChanged.connect(self.on_alignment_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 1)
        global_fntfmt_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding), 0, 2)

        self.let_autolayout_checker, sublock = generalConfigPanel.addCheckBox(self.tr('Auto layout'), 
                discription=self.tr('Split translation into multi-lines according to the extracted balloon region.'))
        self.let_autolayout_adaptive_fntsize_checker, _ = generalConfigPanel.addCheckBox(None, self.tr('Adjust font size adaptively if it is set to \"decide by program.\"'), sublock=sublock)
        self.let_autolayout_adaptive_fntsize_checker.stateChanged.connect(self.on_adaptive_fntsize_changed)

        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)
        self.let_uppercase_checker, _ = generalConfigPanel.addCheckBox(self.tr('To uppercase'))
        self.let_uppercase_checker.stateChanged.connect(self.on_uppercase_changed)

        generalConfigPanel.addTextLabel(label_save)
        self.rst_imgformat_combobox, imsave_sublock = generalConfigPanel.addCombobox(['PNG', 'JPG', 'WEBP'], self.tr('Result image format'))
        self.rst_imgformat_combobox.currentIndexChanged.connect(self.on_rst_imgformat_changed)
        self.rst_imgquality_edit = PercentageLineEdit('100')
        self.rst_imgquality_edit.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.rst_imgquality_edit.finish_edited.connect(self.on_edit_quality_changed)
        sublock = ConfigSubBlock(self.rst_imgquality_edit, self.tr('Quality'), vertical_layout=False)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().insertStretch(-1)
        imsave_sublock.layout().addWidget(sublock)

        generalConfigPanel.addTextLabel(label_saladict)

        sublock = ConfigSubBlock(ConfigTextLabel(self.tr("<a href=\"https://github.com/dmMaze/BallonsTranslator/tree/master/doc/saladict.md\">Installation guide</a>"), CONFIG_FONTSIZE_CONTENT - 2), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)

        self.selectext_minimenu_checker, _ = generalConfigPanel.addCheckBox(self.tr('Show mini menu when selecting text.'))
        self.selectext_minimenu_checker.stateChanged.connect(self.on_selectext_minimenu_changed)
        self.saladict_shortcut = QKeySequenceEdit("ALT+W", self)
        self.saladict_shortcut.keySequenceChanged.connect(self.on_saladict_shortcut_changed)
        self.saladict_shortcut.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)

        sublock = ConfigSubBlock(self.saladict_shortcut, self.tr("Shortcut"), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)
        self.searchurl_combobox, _ = generalConfigPanel.addCombobox(["https://www.google.com/search?q=", "https://www.bing.com/search?q=", "https://duckduckgo.com/?q=", "https://yandex.com/search/?text=", "http://www.baidu.com/s?wd=", "https://search.yahoo.com/search;?p=", "https://www.urbandictionary.com/define.php?term="], self.tr("Search Engines"), fix_size=False)
        self.searchurl_combobox.setEditable(True)
        self.searchurl_combobox.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.searchurl_combobox.currentTextChanged.connect(self.on_searchurl_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.configTable)
        splitter.addWidget(self.configContent)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        hlayout = QHBoxLayout(self)

        hlayout.addWidget(splitter)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.configTable.expandAll()

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

    def on_open_onstartup_changed(self):
        pcfg.open_recent_on_startup = self.open_on_startup_checker.isChecked()

    def on_fntsize_flag_changed(self):
        pcfg.let_fntsize_flag = self.let_fntsize_combox.currentIndex()

    def on_fntstroke_flag_changed(self):
        pcfg.let_fntstroke_flag = self.let_fntstroke_combox.currentIndex()

    def on_autolayout_changed(self):
        pcfg.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def on_adaptive_fntsize_changed(self):
        pcfg.let_autolayout_adaptive_fntsz = self.let_autolayout_adaptive_fntsize_checker.isChecked()

    def on_uppercase_changed(self):
        pcfg.let_uppercase_flag = self.let_uppercase_checker.isChecked()

    def on_rst_imgformat_changed(self):
        pcfg.imgsave_ext = '.' + self.rst_imgformat_combobox.currentText().lower()

    def on_edit_quality_changed(self, value: str):
        pcfg.imgsave_quality = int(value)

    def on_selectext_minimenu_changed(self):
        pcfg.textselect_mini_menu = self.selectext_minimenu_checker.isChecked()

    def on_saladict_shortcut_changed(self):
        kstr = self.saladict_shortcut.keySequence().toString()
        if kstr:
            pcfg.saladict_shortcut = self.saladict_shortcut.keySequence().toString()

    def on_searchurl_changed(self):
        url = self.searchurl_combobox.currentText()
        pcfg.search_url = url

    def on_fontcolor_flag_changed(self):
        pcfg.let_fntcolor_flag = self.let_fntcolor_combox.currentIndex()

    def on_font_scolor_flag_changed(self):
        pcfg.let_fnt_scolor_flag = self.let_fnt_scolor_combox.currentIndex()

    def on_alignment_flag_changed(self):
        pcfg.let_alignment_flag = self.let_alignment_combox.currentIndex()

    def on_effect_flag_changed(self):
        pcfg.let_fnteffect_flag = self.let_effect_combox.currentIndex()

    def focusOnTranslator(self):
        idx0, idx1 = self.trans_sub_block.idx0, self.trans_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnInpaint(self):
        idx0, idx1 = self.inpaint_sub_block.idx0, self.inpaint_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def hideEvent(self, e) -> None:
        self.save_config.emit()
        return super().hideEvent(e)
        
    def setupConfig(self):
        self.blockSignals(True)

        if pcfg.open_recent_on_startup:
            self.open_on_startup_checker.setChecked(True)

        self.let_effect_combox.setCurrentIndex(pcfg.let_fnteffect_flag)
        self.let_fntsize_combox.setCurrentIndex(pcfg.let_fntsize_flag)
        self.let_fntstroke_combox.setCurrentIndex(pcfg.let_fntstroke_flag)
        self.let_fntcolor_combox.setCurrentIndex(pcfg.let_fntcolor_flag)
        self.let_fnt_scolor_combox.setCurrentIndex(pcfg.let_fnt_scolor_flag)
        self.let_alignment_combox.setCurrentIndex(pcfg.let_alignment_flag)
        self.let_autolayout_checker.setChecked(pcfg.let_autolayout_flag)
        self.let_autolayout_adaptive_fntsize_checker.setChecked(pcfg.let_autolayout_adaptive_fntsz)
        self.selectext_minimenu_checker.setChecked(pcfg.textselect_mini_menu)
        self.let_uppercase_checker.setChecked(pcfg.let_uppercase_flag)
        self.saladict_shortcut.setKeySequence(pcfg.saladict_shortcut)
        self.searchurl_combobox.setCurrentText(pcfg.search_url)
        self.ocr_config_panel.restoreEmptyOCRChecker.setChecked(pcfg.restore_ocr_empty)
        self.rst_imgformat_combobox.setCurrentText(pcfg.imgsave_ext.replace('.', '').upper())
        self.rst_imgquality_edit.setText(str(pcfg.imgsave_quality))

        self.blockSignals(False)