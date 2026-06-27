from typing import List, Union, Tuple

from qtpy.QtWidgets import QApplication, QPushButton, QLayout, QGridLayout, QHBoxLayout, QVBoxLayout, QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox, QSplitter, QScrollArea, QLineEdit, QDialog, QStackedWidget, QMessageBox
from qtpy.QtCore import Qt, Signal, QSize, QEvent, QItemSelection
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QIntValidator, QValidator, QFocusEvent

from .custom_widget import ConfigComboBox, Widget
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.version import APP_VERSION
from ballontranslator.utils.network_mirrors import (
    HUGGINGFACE_MIRROR_OPTIONS,
    PYPI_MIRROR_OPTIONS,
    display_options,
    mirror_from_display,
    mirror_to_display,
)
from ballontranslator.utils.shared import CONFIG_FONTSIZE_CONTENT, CONFIG_FONTSIZE_TABLE, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN
from .module_parse_widgets import InpaintConfigPanel, TextDetectConfigPanel, TranslatorConfigPanel, OCRConfigPanel

LAYOUT_SET_MINIMUM_SIZE = getattr(getattr(QLayout, 'SizeConstraint', QLayout), 'SetMinimumSize')
PUSHBTN_FIXED_HEIGHT = 32
SECTION_ALIASES = {
    'startup': 'application',
    'save': 'application',
}
PRESERVE_ACTIVE_WIDGET_CLASS_NAMES = {
    'FrameLessMessageBox',
    'ImgtransProgressMessageBox',
    'KeywordSubWidget',
    'MessageBox',
    'ProgressMessageBox',
}

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
        validator = CustomIntValidator(0, 101, 3)
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


class ConfigSubBlock(Widget):
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, 
    vertical_layout=True, insert_stretch: bool = False, content_margins = (0, 0, 0, 0), fnt_size=None) -> None:
        super().__init__()
        if vertical_layout:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)

        if fnt_size is None:
            fnt_size = CONFIG_FONTSIZE_CONTENT
            if discription is not None:
                fnt_size = CONFIG_FONTSIZE_CONTENT-2
        if name is not None:
            textlabel = ConfigTextLabel(name, fnt_size, QFont.Weight.Normal)
            layout.addWidget(textlabel)
        if discription is not None:
            layout.addWidget(ConfigTextLabel(discription, fnt_size))
        if insert_stretch:
            layout.insertStretch(-1)
        if isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
        self.widget = widget
        self.setContentsMargins(*content_margins)


def combobox_with_label(sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True, parent: QWidget = None, insert_stretch: bool = False) -> Tuple[ConfigComboBox, QWidget]:
    combox = ConfigComboBox(fix_size=fix_size, scrollWidget=parent)
    combox.addItems(sel)
    if target_block is None:
        sublock = ConfigSubBlock(combox, name, discription, vertical_layout=vertical_layout, insert_stretch=insert_stretch, fnt_size=CONFIG_FONTSIZE_TABLE-2)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().setSpacing(12)
        return combox, sublock
    else:
        layout = target_block.layout()
        layout.addSpacing(12)
        layout.addWidget(ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal))
        layout.addWidget(combox)
        return combox, target_block
    
def checkbox_with_label(name: str, discription: str = None, target_block: QWidget = None):
    checkbox = QCheckBox()
    if discription is not None:
        font = checkbox.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT * 0.8)
        checkbox.setFont(font)
        checkbox.setText(discription)
        vertical_layout = True
    else:
        vertical_layout = False

    if target_block is None:
        sublock = ConfigSubBlock(checkbox, name, vertical_layout=vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        target_block = sublock
    return checkbox, target_block
    


class ConfigBlock(Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(0)
        self.vlayout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def addLineEdit(self, name: str = None, discription: str = None, vertical_layout: bool = False):
        le = QLineEdit()
        le.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        le.setFixedHeight(30)
        sublock = ConfigSubBlock(le, name, discription, vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.addSublock(sublock)
        sublock.layout().setSpacing(12)
        return le, sublock

    def addSublock(self, sublock: ConfigSubBlock):
        self.vlayout.addWidget(sublock)

    def addCombobox(self, sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True) -> Tuple[ConfigComboBox, QWidget]:
        combox, sublock = combobox_with_label(sel, name, discription, vertical_layout, target_block, fix_size, parent=self)
        if target_block is None:
            self.addSublock(sublock)
        return combox, sublock

    def addBlockWidget(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout: bool = False) -> ConfigSubBlock:
        sublock = ConfigSubBlock(widget, name, discription, vertical_layout)
        self.addSublock(sublock)
        return sublock

    def addCheckBox(self, name: str, discription: str = None, target_block: ConfigSubBlock = None) -> QCheckBox:
        checkbox, sublock = checkbox_with_label(name, discription, target_block)
        if target_block is None:
            self.addSublock(sublock)
        return checkbox, sublock


class ConfigContent(QStackedWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config_block_list: List[ConfigBlock] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.section_index = {}

    def addConfigBlock(self, block: ConfigBlock, section_key: str):
        scroll_area = QScrollArea()
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidgetResizable(True)
        scroll_area.setContentsMargins(0, 0, 0, 0)
        scroll_content = Widget()
        scroll_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        scroll_layout = QHBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll_layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.addWidget(block, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        self.addWidget(scroll_area)
        self.section_index[section_key] = self.count() - 1
        self.config_block_list.append(block)

    def showSection(self, section_key: str):
        index = self.section_index.get(section_key)
        if index is not None:
            self.setCurrentIndex(index)

    def wheelEvent(self, event) -> None:
        widget = self.currentWidget()
        if widget is not None:
            return widget.wheelEvent(event)
        return super().wheelEvent(event)


class TableItem(QStandardItem):
    def __init__(self, text, fontsize, section_key: str = None):
        super().__init__()
        font = self.font()
        font.setPointSizeF(fontsize)
        self.setFont(font)
        self.setText(text)
        self.setEditable(False)
        if section_key is not None:
            self.setData(section_key, Qt.ItemDataRole.UserRole)

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
            size.setHeight(item.font().pointSize() + 14)
            return size
        else:
            return super().data(index, role)


class ConfigTable(QTreeView):
    section_pressed = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        treeModel = TreeModel()
        self.setModel(treeModel)
        self.selected: TableItem = None
        self.setHeaderHidden(True)
        self.setMinimumWidth(190)
        self.setMaximumWidth(240)
        self.section_items = {}

    def addHeader(self, header: str) -> TableItem:
        rootNode = self.model().invisibleRootItem()
        ti = TableItem(header, CONFIG_FONTSIZE_TABLE)
        ti.setSelectable(False)
        rootNode.appendRow(ti)
        return ti

    def addSection(self, parent: TableItem, text: str, section_key: str) -> TableItem:
        item = TableItem(text, CONFIG_FONTSIZE_TABLE, section_key)
        parent.appendRow(item)
        self.section_items[section_key] = item
        return item

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        sel = selected.indexes()
        model = self.model()

        self.selected = model.itemFromIndex(sel[0]) \
            if len(sel) > 0 else None
        for i in deselected.indexes():
            self.model().itemFromIndex(i).setBold(False)
        
        index = self.currentIndex()
        if index.isValid():
            self.model().itemFromIndex(index).setBold(True)
            section_key = self.model().itemFromIndex(index).data(Qt.ItemDataRole.UserRole)
            if section_key is not None:
                self.section_pressed.emit(section_key)
        super().selectionChanged(selected, deselected)

    def setCurrentSection(self, section_key: str):
        item = self.section_items.get(section_key)
        if item is not None and self.currentIndex() != item.index():
            self.setCurrentIndex(item.index())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.selected is not None:
            section_key = self.selected.data(Qt.ItemDataRole.UserRole)
            if section_key is not None:
                self.section_pressed.emit(section_key)


from qtpy.QtCore import QThread

class DictDownloadThread(QThread):
    progress = Signal(int, int)
    finished = Signal(bool, str)

    def __init__(self, url, save_path):
        super().__init__()
        self.url = url
        self.save_path = save_path
        import threading
        self.cancel_event = threading.Event()

    def run(self):
        try:
            import os
            from ballontranslator.utils.download_util import download_url_to_file

            def progress_callback(payload):
                if payload.get('event') == 'file_progress':
                    self.progress.emit(payload.get('downloaded', 0), payload.get('total', 0))

            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            download_url_to_file(
                self.url,
                self.save_path,
                progress_callback=progress_callback,
                cancel_event=self.cancel_event
            )
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))

    def cancel(self):
        self.cancel_event.set()


from qtpy.QtWidgets import QListWidget, QInputDialog, QHBoxLayout, QVBoxLayout, QPushButton, QLineEdit, QDialog, QLabel, QWidget

class WordListItemWidget(QWidget):
    def __init__(self, word, on_delete_callback, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.label = QLabel(word, self)
        self.label.setStyleSheet("color: #ffffff; font-family: 'Segoe UI', Arial; font-size: 12px;")

        self.delete_btn = QPushButton("🗑", self)
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff4d4d;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff4d4d;
                color: #ffffff;
                border-radius: 3px;
            }
        """)
        self.delete_btn.clicked.connect(lambda: on_delete_callback(word))
        self.delete_btn.setVisible(False)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.delete_btn)

    def enterEvent(self, event):
        self.delete_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.delete_btn.setVisible(False)
        super().leaveEvent(event)


class AddWordItemWidget(QWidget):
    def __init__(self, on_add_callback, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText(self.tr("Add new word..."))
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2d;
                color: #ffffff;
                border: 1px solid #45474a;
                border-radius: 3px;
                padding: 2px 6px;
                font-family: 'Segoe UI', Arial;
                font-size: 12px;
            }
        """)
        self.input_field.returnPressed.connect(self.trigger_add)

        self.add_btn = QPushButton("+", self)
        self.add_btn.setFixedSize(24, 24)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #1e93e5;
                color: #ffffff;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1a7abf;
            }
        """)
        self.add_btn.clicked.connect(self.trigger_add)

        layout.addWidget(self.input_field)
        layout.addWidget(self.add_btn)
        self.on_add_callback = on_add_callback

    def trigger_add(self):
        word = self.input_field.text().strip().lower()
        if word:
            self.on_add_callback(word)
            self.input_field.clear()
            self.input_field.setFocus()


class DictionaryManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Custom Dictionary Manager"))
        self.resize(360, 480)

        from ballontranslator.utils.spellcheck import SpellCheckManager
        self.manager = SpellCheckManager.get_instance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1f;
                border: 1px solid #45474a;
                border-radius: 4px;
            }
            QListWidget::item {
                background-color: transparent;
                border-bottom: 1px solid #2b2b2d;
            }
            QListWidget::item:hover {
                background-color: #2b2b2d;
            }
        """)
        layout.addWidget(self.list_widget)

        self.close_btn = QPushButton(self.tr("Close"), self)
        self.close_btn.setFixedHeight(32)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b3d40;
                color: #ffffff;
                border: 1px solid #45474a;
                border-radius: 4px;
                font-family: 'Segoe UI', Arial;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e93e5;
            }
        """)
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

        self.populate_list()

    def populate_list(self):
        self.list_widget.clear()
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt

        for word in sorted(self.manager.custom_words):
            item = QListWidgetItem(self.list_widget)
            widget = WordListItemWidget(word, self.delete_word, self)
            item.setSizeHint(widget.sizeHint())
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        input_item = QListWidgetItem(self.list_widget)
        input_item.setFlags(input_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        input_widget = AddWordItemWidget(self.add_word, self)
        input_item.setSizeHint(input_widget.sizeHint())
        self.list_widget.addItem(input_item)
        self.list_widget.setItemWidget(input_item, input_widget)

    def add_word(self, word):
        if word and word not in self.manager.custom_words:
            self.manager.add_to_dictionary(word)
            self.populate_list()

    def delete_word(self, word):
        self.manager.custom_words.discard(word)
        self.manager._save_custom_dictionary()
        self.manager.notify_config_changed()
        self.populate_list()


class ConfigPanel(QDialog):

    save_config = Signal()
    unload_models = Signal()
    prepare_selected_modules = Signal()
    check_update = Signal()
    reload_textstyle = Signal(bool)
    show_only_custom_font = Signal(bool)

    dictionary_urls = {
        "Arabic (ar)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ar/ar.dic",
        "Belarusian (be_BY)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/be_BY/be_BY.dic",
        "Bulgarian (bg_BG)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/bg_BG/bg_BG.dic",
        "Bosnian (bs_BA)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/bs_BA/bs_BA.dic",
        "Catalan (ca)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ca/ca.dic",
        "Czech (cs_CZ)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/cs_CZ/cs_CZ.dic",
        "Danish (da_DK)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/da_DK/da_DK.dic",
        "German (de_DE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/de/de_DE.dic",
        "Greek (el_GR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/el_GR/el_GR.dic",
        "English (en_US)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic",
        "English (en_GB)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_GB.dic",
        "Spanish (es_ES)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/es/es_ES.dic",
        "Estonian (et_EE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/et_EE/et_EE.dic",
        "Persian (fa_IR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fa_IR/fa_IR.dic",
        "French (fr_FR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/fr_FR.dic",
        "Croatian (hr_HR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/hr_HR/hr_HR.dic",
        "Hungarian (hu_HU)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/hu_HU/hu_HU.dic",
        "Indonesian (id)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/id/id.dic",
        "Icelandic (is)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/is/is.dic",
        "Italian (it_IT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/it_IT/it_IT.dic",
        "Korean (ko_KR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ko_KR/ko_KR.dic",
        "Lithuanian (lt_LT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/lt_LT/lt_LT.dic",
        "Latvian (lv_LV)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/lv_LV/lv_LV.dic",
        "Dutch (nl_NL)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/nl_NL/nl_NL.dic",
        "Polish (pl_PL)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pl_PL/pl_PL.dic",
        "Portuguese (pt_BR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_BR/pt_BR.dic",
        "Portuguese (pt_PT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_PT/pt_PT.dic",
        "Romanian (ro_RO)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ro/ro_RO.dic",
        "Russian (ru_RU)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ru_RU/ru_RU.dic",
        "Russian (Large - all inflections) (ru_RU)": "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt",
        "Slovak (sk_SK)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sk_SK/sk_SK.dic",
        "Slovenian (sl_SI)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sl_SI/sl_SI.dic",
        "Swedish (sv_SE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sv_SE/sv_SE.dic",
        "Turkish (tr_TR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/tr_TR/tr_TR.dic",
        "Ukrainian (uk_UA)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/uk_UA/uk_UA.dic",
        "Vietnamese (vi)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/vi/vi.dic"
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._outside_click_filter_installed = False
        self.setObjectName("ConfigPanel")
        self.setWindowTitle(self.tr('Settings'))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setSizeGripEnabled(True)
        self.resize(900, 640)
        self.setMinimumSize(720, 520)
        self.configTable = ConfigTable()
        self.configTable.section_pressed.connect(self.showSection)
        self.configContent = ConfigContent()
        moduleTableItem = self.configTable.addHeader(self.tr('Modules'))
        generalTableItem = self.configTable.addHeader(self.tr('General'))
        
        label_modules = self.tr('Module Actions')
        label_text_det = self.tr('Detector')
        label_text_ocr = self.tr('OCR')
        label_inpaint = self.tr('Inpainter')
        label_translator = self.tr('Translator')
        label_application = self.tr('Application')
        label_typesetting = self.tr('Typesetting')
        label_spellcheck = self.tr('Spell Checker')

        moduleConfigPanel = self.addConfigBlock(label_modules, moduleTableItem, 'modules')
        dlConfigPanel = self.addConfigBlock(label_text_det, moduleTableItem, 'detector')
        ocrConfigPanel = self.addConfigBlock(label_text_ocr, moduleTableItem, 'ocr')
        inpaintConfigPanel = self.addConfigBlock(label_inpaint, moduleTableItem, 'inpainter')
        translatorConfigPanel = self.addConfigBlock(label_translator, moduleTableItem, 'translator')
        applicationConfigPanel = self.addConfigBlock(label_application, generalTableItem, 'application')
        typesettingConfigPanel = self.addConfigBlock(label_typesetting, generalTableItem, 'typesetting')
        spellcheckConfigPanel = self.addConfigBlock(label_spellcheck, generalTableItem, 'spellcheck')
        
        self.empty_runcache_checker, empty_runcache_subblock = checkbox_with_label(self.tr('Empty cache after RUN'), discription=self.tr('Empty cache after RUN to save memory.'))
        moduleConfigPanel.vlayout.addWidget(empty_runcache_subblock)
        self.empty_runcache_checker.stateChanged.connect(self.on_runcache_changed)
        self.package_auto_install_checker, msublock = checkbox_with_label(
            self.tr('Auto install missing packages'),
            discription=self.tr('Install missing Python packages automatically when a selected module requires them.'),
        )
        self.package_auto_install_checker.stateChanged.connect(self.on_package_auto_install_changed)
        moduleConfigPanel.vlayout.addWidget(msublock)
        module_actions = QWidget()
        module_actions_layout = QHBoxLayout(module_actions)
        module_actions_layout.setContentsMargins(0, 0, 0, 0)
        module_actions_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.prepare_modules_btn = QPushButton(parent=self)
        self.prepare_modules_btn.setText(self.tr('Prepare Selected Modules'))
        self.prepare_modules_btn.clicked.connect(self.prepare_selected_modules)
        self.prepare_modules_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)
        module_actions_layout.addWidget(self.prepare_modules_btn)
        self.unload_model_btn = QPushButton(parent=self)
        self.unload_model_btn.setText(self.tr('Unload All Models'))
        self.unload_model_btn.clicked.connect(self.unload_models)
        self.unload_model_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)
        module_actions_layout.addWidget(self.unload_model_btn)
        moduleConfigPanel.addBlockWidget(module_actions)

        self.detect_config_panel = TextDetectConfigPanel(self.tr('Detector'), scrollWidget=self)
        self.detect_config_panel.module_label.hide()
        self.detect_sub_block = dlConfigPanel.addBlockWidget(self.detect_config_panel)
        self.detect_config_panel.keep_existing_checker.clicked.connect(self.on_keepline_clicked)

        self.ocr_config_panel = OCRConfigPanel(self.tr('OCR'), scrollWidget=self)
        self.ocr_config_panel.module_label.hide()
        self.ocr_sub_block = ocrConfigPanel.addBlockWidget(self.ocr_config_panel)

        self.inpaint_config_panel = InpaintConfigPanel(self.tr('Inpainter'), scrollWidget=self)
        self.inpaint_config_panel.module_label.hide()
        self.inpaint_sub_block = inpaintConfigPanel.addBlockWidget(self.inpaint_config_panel)
        self.inpaint_config_panel.filter_mask_by_bboxes_checker.clicked.connect(self.on_filter_mask_by_bboxes_clicked)

        self.trans_config_panel = TranslatorConfigPanel(label_translator, scrollWidget=self)
        self.trans_config_panel.module_label.hide()
        self.trans_sub_block = translatorConfigPanel.addBlockWidget(self.trans_config_panel)

        self.open_on_startup_checker, _ = applicationConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)

        self.check_update_on_startup_checker, _ = applicationConfigPanel.addCheckBox(self.tr('Check update on startup'))
        self.check_update_on_startup_checker.stateChanged.connect(self.on_check_update_onstartup_changed)

        self.spellcheck_checker, _ = spellcheckConfigPanel.addCheckBox(self.tr('Enable Spell Checker'))
        self.spellcheck_checker.stateChanged.connect(self.on_spellcheck_changed)

        # Dictionary Words Manager Button
        self.manage_words_btn = QPushButton(parent=self)
        self.manage_words_btn.setText(self.tr("Dictionary Words..."))
        self.manage_words_btn.clicked.connect(self.open_words_manager)
        self.manage_words_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)
        spellcheckConfigPanel.addBlockWidget(self.manage_words_btn)

        # Repository Dictionaries List
        repo_layout = QVBoxLayout()
        repo_label = ConfigTextLabel(self.tr("Repository Dictionaries"), CONFIG_FONTSIZE_CONTENT, QFont.Weight.Bold)
        repo_layout.addWidget(repo_label)

        self.repo_dicts_list = QListWidget(self)
        self.repo_dicts_list.setFixedHeight(150)
        self.repo_dicts_list.itemChanged.connect(self.on_repo_dict_item_changed)
        repo_layout.addWidget(self.repo_dicts_list)

        spellcheckConfigPanel.addBlockWidget(repo_layout)

        # External Dictionaries List
        ext_layout = QVBoxLayout()
        ext_label = ConfigTextLabel(self.tr("External Dictionaries"), CONFIG_FONTSIZE_CONTENT, QFont.Weight.Bold)
        ext_layout.addWidget(ext_label)

        self.external_dicts_list = QListWidget(self)
        self.external_dicts_list.setFixedHeight(120)
        ext_layout.addWidget(self.external_dicts_list)

        ext_btns_layout = QHBoxLayout()
        self.add_ext_btn = QPushButton(self.tr("Add Dictionary..."), self)
        self.add_ext_btn.clicked.connect(self.add_external_dictionary)
        self.add_ext_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)
        self.remove_ext_btn = QPushButton(self.tr("Remove Selected"), self)
        self.remove_ext_btn.clicked.connect(self.remove_external_dictionary)
        self.remove_ext_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)

        ext_btns_layout.addWidget(self.add_ext_btn)
        ext_btns_layout.addWidget(self.remove_ext_btn)
        ext_layout.addLayout(ext_btns_layout)

        self.spellcheck_subblock = spellcheckConfigPanel.addBlockWidget(ext_layout)

        update_status_widget = QWidget()
        update_status_layout = QHBoxLayout(update_status_widget)
        update_status_layout.setContentsMargins(0, 0, 0, 0)
        update_status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.current_version_label = ConfigTextLabel(
            self.tr('Current version: ') + APP_VERSION,
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        self.latest_version_label = ConfigTextLabel(
            self.tr('Latest version: ') + self.tr('Not checked'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        self.check_update_btn = QPushButton(parent=self)
        self.check_update_btn.setText(self.tr('Check update'))
        self.check_update_btn.clicked.connect(self.check_update)
        self.check_update_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)

        update_status_layout.addWidget(self.check_update_btn)
        update_status_layout.addSpacing(24)
        update_status_layout.addWidget(self.current_version_label)
        update_status_layout.addSpacing(24)
        update_status_layout.addWidget(self.latest_version_label)

        applicationConfigPanel.addBlockWidget(update_status_widget)

        none_label = self.tr('None')
        self.huggingface_mirror_combobox, _ = applicationConfigPanel.addCombobox(
            display_options(HUGGINGFACE_MIRROR_OPTIONS, none_label=none_label),
            self.tr('Huggingface Mirrors'),
            fix_size=False,
        )
        self.huggingface_mirror_combobox.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        self.huggingface_mirror_combobox.currentTextChanged.connect(self.on_huggingface_mirror_changed)
        self.pypi_mirror_combobox, _ = applicationConfigPanel.addCombobox(
            display_options(PYPI_MIRROR_OPTIONS, none_label=none_label),
            self.tr('PyPI Mirrors'),
            fix_size=False,
        )
        self.pypi_mirror_combobox.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        self.pypi_mirror_combobox.currentTextChanged.connect(self.on_pypi_mirror_changed)

        dec_program_str = self.tr('decide by program')
        use_global_str = self.tr('use global setting')

        global_fntfmt_widget = Widget()
        global_fntfmt_layout = QGridLayout(global_fntfmt_widget)
        global_fntfmt_layout.setSpacing(0)
        global_fntfmt_widget.setContentsMargins(0, 0, 0, 0)

        b = typesettingConfigPanel.addBlockWidget(global_fntfmt_widget)
        b.layout().setContentsMargins(0, 0, 0, 0)
        b.setContentsMargins(0, 0, 0, 0)
        self.let_fntsize_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Size'), parent=self, insert_stretch=True)
        global_fntfmt_layout.addWidget(sublock, 0, 0)

        self.let_fntsize_combox.activated.connect(self.on_fntsize_flag_changed)
        self.let_fntstroke_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Size'), parent=self, insert_stretch=True)
        self.let_fntstroke_combox.activated.connect(self.on_fntstroke_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 0, 1)
        
        self.let_fntcolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Color'), parent=self, insert_stretch=True)
        self.let_fntcolor_combox.activated.connect(self.on_fontcolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 0)
        self.let_fnt_scolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Color'), parent=self, insert_stretch=True)
        self.let_fnt_scolor_combox.activated.connect(self.on_font_scolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 1)

        self.let_effect_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Effect'), parent=self, insert_stretch=True)
        self.let_effect_combox.activated.connect(self.on_effect_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 0)
        self.let_alignment_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Alignment'), parent=self, insert_stretch=True)
        self.let_alignment_combox.activated.connect(self.on_alignment_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 1)

        self.let_writing_mode_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Writing-mode'), parent=self, insert_stretch=True)
        self.let_writing_mode_combox.activated.connect(self.on_writing_mode_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 3, 0)
        self.let_family_combox, sublock = combobox_with_label([self.tr('Keep existing'), self.tr('Always use global setting')], self.tr('Font Family'), parent=self, insert_stretch=True)
        self.let_family_combox.activated.connect(self.on_family_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 3, 1)

        global_fntfmt_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding), 0, 2)

        self.let_autolayout_checker, sublock = typesettingConfigPanel.addCheckBox(self.tr('Auto layout'),
                discription=self.tr('Split translation into multi-lines according to the extracted balloon region.'))

        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)
        self.let_uppercase_checker, _ = typesettingConfigPanel.addCheckBox(self.tr('To uppercase'))
        self.let_uppercase_checker.stateChanged.connect(self.on_uppercase_changed)

        self.let_textstyle_indep_checker, _ = typesettingConfigPanel.addCheckBox(self.tr('Independent text styles for each projects'))
        self.let_textstyle_indep_checker.stateChanged.connect(self.on_textstyle_indep_changed)

        self.let_show_only_custom_fonts, sublock = typesettingConfigPanel.addCheckBox(self.tr("Show only custom fonts"))
        self.let_show_only_custom_fonts.stateChanged.connect(self.on_show_only_custom_fonts)

        self.rst_imgformat_combobox, imsave_sublock = applicationConfigPanel.addCombobox(['PNG', 'JPG', 'WEBP', 'JXL'], self.tr('Result image format'))
        self.rst_imgformat_combobox.activated.connect(self.on_rst_imgformat_changed)
        self.rst_imgquality_edit = PercentageLineEdit('100')
        self.rst_imgquality_edit.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.rst_imgquality_edit.finish_edited.connect(self.on_edit_quality_changed)

        sublock = ConfigSubBlock(self.rst_imgquality_edit, self.tr('Quality'), vertical_layout=False)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().insertStretch(-1)
        imsave_sublock.layout().addWidget(sublock)

        self.intermediate_imgformat_combobox, intermediate_imsave_sublock = applicationConfigPanel.addCombobox(['PNG', 'JXL'], self.tr('Intermediate image format'))
        self.intermediate_imgformat_combobox.activated.connect(self.on_intermediate_imgformat_changed)

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
        self.showSection('application')

    def on_runcache_changed(self):
        pcfg.module.empty_runcache = self.empty_runcache_checker.isChecked()

    def on_package_auto_install_changed(self):
        pcfg.package_manager.auto_install_missing_packages = self.package_auto_install_checker.isChecked()

    def on_keepline_clicked(self):
        pcfg.module.keep_exist_textlines = self.detect_config_panel.keep_existing_checker.isChecked()

    def on_filter_mask_by_bboxes_clicked(self):
        pcfg.module.filter_mask_by_bboxes = self.inpaint_config_panel.filter_mask_by_bboxes_checker.isChecked()

    def addConfigBlock(self, header: str, parent_item: TableItem, section_key: str) -> ConfigBlock:
        cb = ConfigBlock(parent=self)
        self.configContent.addConfigBlock(cb, section_key)
        self.configTable.addSection(parent_item, header, section_key)
        return cb

    def showConfigDialog(self, section_key: str = None):
        if section_key is not None:
            self.showSection(section_key)
        elif self.configContent.currentIndex() < 0:
            self.showSection('application')
        self._installOutsideClickFilter()
        self.show()
        self.raise_()
        self.activateWindow()

    def showSection(self, section_key: str):
        section_key = SECTION_ALIASES.get(section_key, section_key)
        self.configContent.showSection(section_key)
        self.configTable.setCurrentSection(section_key)

    def on_open_onstartup_changed(self):
        pcfg.open_recent_on_startup = self.open_on_startup_checker.isChecked()

    def on_check_update_onstartup_changed(self):
        pcfg.check_update_on_startup = self.check_update_on_startup_checker.isChecked()

    def on_spellcheck_changed(self):
        enabled = self.spellcheck_checker.isChecked()
        pcfg.spellcheck_enabled = enabled
        self.manage_words_btn.setEnabled(enabled)
        self.repo_dicts_list.setEnabled(enabled)
        self.external_dicts_list.setEnabled(enabled)
        self.add_ext_btn.setEnabled(enabled)
        self.remove_ext_btn.setEnabled(enabled)
        from ballontranslator.utils.spellcheck import SpellCheckManager
        SpellCheckManager.get_instance().notify_config_changed()
        self.save_config.emit()

    def open_words_manager(self):
        dialog = DictionaryManagerDialog(self)
        dialog.exec_()

    def on_repo_dict_item_changed(self, item):
        self.repo_dicts_list.blockSignals(True)
        try:
            from qtpy.QtCore import Qt
            import os
            from ballontranslator.utils.shared import PROGRAM_PATH

            url, filename = item.data(Qt.ItemDataRole.UserRole)
            save_path = os.path.join(PROGRAM_PATH, 'data', 'dictionaries', filename)

            if item.checkState() == Qt.CheckState.Checked:
                if not os.path.exists(save_path):
                    success = self.download_repo_dict_sync(url, filename)
                    if not success:
                        item.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        lang_name = ""
                        for k, v in self.dictionary_urls.items():
                            if v == url:
                                lang_name = k
                                break
                        if lang_name:
                            item.setText(self.tr(lang_name) + self.tr(" (Installed local)"))

            enabled_files = []
            for idx in range(self.repo_dicts_list.count()):
                it = self.repo_dicts_list.item(idx)
                if it.checkState() == Qt.CheckState.Checked:
                    _, fname = it.data(Qt.ItemDataRole.UserRole)
                    enabled_files.append(fname)

            pcfg.spellcheck_repo_dicts = ",".join(enabled_files)
            from ballontranslator.utils.spellcheck import SpellCheckManager
            SpellCheckManager.get_instance().notify_config_changed()
            self.save_config.emit()
        finally:
            self.repo_dicts_list.blockSignals(False)

    def download_repo_dict_sync(self, url, filename):
        import os
        from ballontranslator.utils.shared import PROGRAM_PATH
        from qtpy.QtWidgets import QProgressDialog
        from qtpy.QtCore import Qt

        save_path = os.path.join(PROGRAM_PATH, 'data', 'dictionaries', filename)

        progress = QProgressDialog(self.tr("Downloading dictionary..."), self.tr("Cancel"), 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        thread = DictDownloadThread(url, save_path)
        download_success = [False]

        def on_progress(downloaded, total):
            if total > 0:
                percent = int(downloaded * 100 / total)
                progress.setValue(percent)

        def on_finished(success, err):
            progress.close()
            if success:
                download_success[0] = True
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.information(self, self.tr("Download Complete"), self.tr("Dictionary downloaded successfully!"))
            else:
                if "cancelled" not in err.lower():
                    from qtpy.QtWidgets import QMessageBox
                    QMessageBox.warning(self, self.tr("Download Failed"), self.tr("Failed to download dictionary: ") + err)

        thread.progress.connect(on_progress)
        thread.finished.connect(on_finished)
        progress.canceled.connect(thread.cancel)

        self.active_download_thread = thread
        thread.start()
        progress.exec_()
        thread.wait()
        return download_success[0]

    def add_external_dictionary(self):
        from qtpy.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Dictionary File"),
            "",
            self.tr("Dictionary files (*.txt *.dic)")
        )
        if path:
            self.external_dicts_list.addItem(path)
            self.update_external_dicts_config()

    def remove_external_dictionary(self):
        selected_items = self.external_dicts_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.external_dicts_list.takeItem(self.external_dicts_list.row(item))
        self.update_external_dicts_config()

    def update_external_dicts_config(self):
        paths = []
        for idx in range(self.external_dicts_list.count()):
            paths.append(self.external_dicts_list.item(idx).text())
        pcfg.spellcheck_external_dict_path = ";".join(paths)
        from ballontranslator.utils.spellcheck import SpellCheckManager
        SpellCheckManager.get_instance().notify_config_changed()
        self.save_config.emit()

    def setLatestVersion(self, version: str):
        self.latest_version_label.setText(self.tr('Latest version: ') + version)

    def setUpdateChecking(self, checking: bool):
        self.check_update_btn.setEnabled(not checking)

    def on_huggingface_mirror_changed(self):
        pcfg.mirrors.huggingface = mirror_from_display(
            self.huggingface_mirror_combobox.currentText(),
            none_label=self.tr('None'),
        )

    def on_pypi_mirror_changed(self):
        pcfg.mirrors.pypi = mirror_from_display(
            self.pypi_mirror_combobox.currentText(),
            none_label=self.tr('None'),
        )

    def on_fntsize_flag_changed(self):
        pcfg.let_fntsize_flag = self.let_fntsize_combox.currentIndex()

    def on_fntstroke_flag_changed(self):
        pcfg.let_fntstroke_flag = self.let_fntstroke_combox.currentIndex()

    def on_autolayout_changed(self):
        pcfg.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def on_uppercase_changed(self):
        pcfg.let_uppercase_flag = self.let_uppercase_checker.isChecked()

    def on_textstyle_indep_changed(self):
        pcfg.let_textstyle_indep_flag = self.let_textstyle_indep_checker.isChecked()
        self.reload_textstyle.emit(pcfg.let_textstyle_indep_flag)

    def on_rst_imgformat_changed(self):
        pcfg.imgsave_ext = '.' + self.rst_imgformat_combobox.currentText().lower()

    def on_intermediate_imgformat_changed(self):
        pcfg.intermediate_imgsave_ext = '.' + self.intermediate_imgformat_combobox.currentText().lower()

    def on_edit_quality_changed(self, value: str):
        pcfg.imgsave_quality = int(value)

    def on_fontcolor_flag_changed(self):
        pcfg.let_fntcolor_flag = self.let_fntcolor_combox.currentIndex()

    def on_font_scolor_flag_changed(self):
        pcfg.let_fnt_scolor_flag = self.let_fnt_scolor_combox.currentIndex()

    def on_alignment_flag_changed(self):
        pcfg.let_alignment_flag = self.let_alignment_combox.currentIndex()

    def on_writing_mode_flag_changed(self):
        pcfg.let_writing_mode_flag = self.let_writing_mode_combox.currentIndex()

    def on_family_flag_changed(self):
        pcfg.let_family_flag = self.let_family_combox.currentIndex()

    def on_effect_flag_changed(self):
        pcfg.let_fnteffect_flag = self.let_effect_combox.currentIndex()

    def on_show_only_custom_fonts(self):
        pcfg.let_show_only_custom_fonts_flag = self.let_show_only_custom_fonts.isChecked()
        self.show_only_custom_font.emit(pcfg.let_show_only_custom_fonts_flag)

    def focusOnTranslator(self):
        self.showConfigDialog('translator')

    def focusOnInpaint(self):
        self.showConfigDialog('inpainter')

    def focusOnDetect(self):
        self.showConfigDialog('detector')

    def focusOnOCR(self):
        self.showConfigDialog('ocr')

    def hideEvent(self, e) -> None:
        self._removeOutsideClickFilter()
        self.save_config.emit()
        return super().hideEvent(e)

    def _installOutsideClickFilter(self):
        if self._outside_click_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._outside_click_filter_installed = True

    def _removeOutsideClickFilter(self):
        if not self._outside_click_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._outside_click_filter_installed = False

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.MouseButtonPress and self.isVisible():
            if (
                isinstance(watched, QWidget)
                and QApplication.activePopupWidget() is None
                and not self._widgetInsidePanel(watched)
                and not self._activeWidgetInWhitelist()
            ):
                self.hide()
        return super().eventFilter(watched, event)

    def _widgetInsidePanel(self, widget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = widget.parentWidget()
        return False

    def _activeWidgetInWhitelist(self) -> bool:
        return any(
            self._widgetInWhitelist(widget)
            for widget in (
                QApplication.activeWindow(),
                QApplication.activeModalWidget(),
                QApplication.focusWidget(),
            )
        )

    def _widgetInWhitelist(self, widget) -> bool:
        while widget is not None:
            if self._isWhitelistedWidget(widget):
                return True
            window = widget.window()
            if window is not widget and self._isWhitelistedWidget(window):
                return True
            widget = widget.parentWidget()
        return False

    def _isWhitelistedWidget(self, widget) -> bool:
        return (
            isinstance(widget, QMessageBox)
            or widget.__class__.__name__ in PRESERVE_ACTIVE_WIDGET_CLASS_NAMES
        )

    def setupConfig(self):
        self.blockSignals(True)
        import os
        from ballontranslator.utils import shared

        if pcfg.open_recent_on_startup:
            self.open_on_startup_checker.setChecked(True)
        self.check_update_on_startup_checker.setChecked(pcfg.check_update_on_startup)
        
        # Setup repository dictionaries
        active_repos = pcfg.spellcheck_repo_dicts.split(',')
        active_repos = [x.strip() for x in active_repos if x.strip()]
        
        self.repo_dicts_list.blockSignals(True)
        self.repo_dicts_list.clear()
        from qtpy.QtWidgets import QListWidgetItem
        from qtpy.QtCore import Qt
        
        for lang_name, url in self.dictionary_urls.items():
            filename = url.split('/')[-1]
            dict_path = os.path.join(shared.PROGRAM_PATH, 'data', 'dictionaries', filename)

            display_text = self.tr(lang_name)
            if os.path.exists(dict_path):
                display_text += self.tr(" (Installed local)")

            item = QListWidgetItem(display_text, self.repo_dicts_list)
            item.setData(Qt.ItemDataRole.UserRole, (url, filename))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            if filename in active_repos:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.repo_dicts_list.blockSignals(False)

        # Setup external dictionaries
        self.external_dicts_list.clear()
        ext_paths = pcfg.spellcheck_external_dict_path.split(';')
        for p in ext_paths:
            p = p.strip()
            if p:
                self.external_dicts_list.addItem(p)

        self.spellcheck_checker.setChecked(pcfg.spellcheck_enabled)
        self.manage_words_btn.setEnabled(pcfg.spellcheck_enabled)
        self.repo_dicts_list.setEnabled(pcfg.spellcheck_enabled)
        self.external_dicts_list.setEnabled(pcfg.spellcheck_enabled)
        self.add_ext_btn.setEnabled(pcfg.spellcheck_enabled)
        self.remove_ext_btn.setEnabled(pcfg.spellcheck_enabled)
        self.huggingface_mirror_combobox.setCurrentText(mirror_to_display(
            pcfg.mirrors.huggingface,
            none_label=self.tr('None'),
        ))
        self.pypi_mirror_combobox.setCurrentText(mirror_to_display(
            pcfg.mirrors.pypi,
            none_label=self.tr('None'),
        ))

        self.detect_config_panel.keep_existing_checker.setChecked(pcfg.module.keep_exist_textlines)
        self.inpaint_config_panel.filter_mask_by_bboxes_checker.setChecked(pcfg.module.filter_mask_by_bboxes)
        self.let_effect_combox.setCurrentIndex(pcfg.let_fnteffect_flag)
        self.let_fntsize_combox.setCurrentIndex(pcfg.let_fntsize_flag)
        self.let_fntstroke_combox.setCurrentIndex(pcfg.let_fntstroke_flag)
        self.let_fntcolor_combox.setCurrentIndex(pcfg.let_fntcolor_flag)
        self.let_fnt_scolor_combox.setCurrentIndex(pcfg.let_fnt_scolor_flag)
        self.let_alignment_combox.setCurrentIndex(pcfg.let_alignment_flag)
        self.let_family_combox.setCurrentIndex(pcfg.let_family_flag)
        self.let_writing_mode_combox.setCurrentIndex(pcfg.let_writing_mode_flag)
        self.let_autolayout_checker.setChecked(pcfg.let_autolayout_flag)
        self.let_uppercase_checker.setChecked(pcfg.let_uppercase_flag)
        self.let_textstyle_indep_checker.setChecked(pcfg.let_textstyle_indep_flag)
        self.ocr_config_panel.restoreEmptyOCRChecker.setChecked(pcfg.restore_ocr_empty)
        self.rst_imgformat_combobox.setCurrentText(pcfg.imgsave_ext.replace('.', '').upper())
        self.intermediate_imgformat_combobox.setCurrentText(pcfg.intermediate_imgsave_ext.replace('.', '').upper())
        self.rst_imgquality_edit.setText(str(pcfg.imgsave_quality))
        self.empty_runcache_checker.setChecked(pcfg.module.empty_runcache)
        self.package_auto_install_checker.setChecked(pcfg.package_manager.auto_install_missing_packages)
        self.let_show_only_custom_fonts.setChecked(pcfg.let_show_only_custom_fonts_flag)

        self.blockSignals(False)
