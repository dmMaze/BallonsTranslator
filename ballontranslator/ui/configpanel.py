import os
from typing import List, Optional, Union, Tuple

from qtpy.QtWidgets import (
    QApplication, QPushButton, QLayout, QGridLayout, QHBoxLayout, QVBoxLayout,
    QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox,
    QSplitter, QScrollArea, QLineEdit, QStackedWidget, QMessageBox,
    QListWidget, QSpinBox, QProgressDialog, QFileDialog, QListWidgetItem,
    QDialog, QAbstractItemView, QButtonGroup, QRadioButton,
    QFrame,
)
from qtpy.QtCore import Qt, Signal, QSize, QItemSelection, QTimer
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QIntValidator, QValidator, QFocusEvent

from .custom_widget import ConfigComboBox, NoBorderPushBtn, ScrollBar, Widget
from ballontranslator.utils import shared
from ballontranslator.utils.config import OCRTextPostprocess, pcfg
from ballontranslator.utils.version import APP_VERSION
from ballontranslator.utils.network_mirrors import (
    HUGGINGFACE_MIRROR_OPTIONS,
    PYPI_MIRROR_OPTIONS,
    display_options,
    mirror_from_display,
    mirror_to_display,
)
from ballontranslator.utils.shared import (
    CONFIG_COMBOBOX_LONG,
    CONFIG_COMBOBOX_MIDEAN,
    CONFIG_COMBOBOX_SHORT,
    CONFIG_CONTENT_MARGIN,
    CONFIG_CONTENT_MARGINS,
    CONFIG_FONTSIZE_CONTENT,
    CONFIG_FONTSIZE_TABLE,
    LEGACY_FONTS,
    ON_MACOS,
    ON_WINDOWS,
    PROGRAM_PATH,
    TITLEBAR_HEIGHT,
)
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.modules.lazy_registry import probe_torch_package
from .llm_profile_widgets import LLMProfilesWidget
from .framelesswindow import (
    DialogCloseButton,
    FramelessWindow,
    OutsideClickFramelessMixin,
)
from ballontranslator.ui.spellcheck import DICTIONARY_URLS, SpellCheckManager, DictionaryManagerDialog, DictDownloadThread


LAYOUT_SET_MINIMUM_SIZE = getattr(getattr(QLayout, 'SizeConstraint', QLayout), 'SetMinimumSize')
PUSHBTN_FIXED_HEIGHT = 32
SECTION_ALIASES = {
    'startup': 'application',
    'save': 'application',
    'modules': 'pipeline',
}
PRESERVE_ACTIVE_WIDGET_CLASS_NAMES = {
    'FontExcludeDialog',
    'FrameLessMessageBox',
    'ImgtransProgressMessageBox',
    'KeywordSubWidget',
    'MessageBox',
    'ProgressMessageBox',
    'TorchInstallHelperDialog',
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


class FontExcludeDialog(OutsideClickFramelessMixin, QDialog):
    """Dialog for selecting which fonts to exclude from the font list."""

    def __init__(self, parent: QWidget = None) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        super().__init__(
            parent,
            window_type.Dialog | window_type.FramelessWindowHint,
        )
        self.setObjectName('FontExcludeDialog')
        self.setWindowTitle(self.tr("Font Exclusion"))
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(640, 440)
        self.resize(760, 540)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setAttribute(widget_attribute.WA_TranslucentBackground)
        self.setAttribute(widget_attribute.WA_DeleteOnClose)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 5, 5)

        surface = QFrame(self)
        surface.setObjectName('FontExcludeSurface')
        root_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(14)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('FontExcludeTitleBar')
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(self.tr('Font Exclusion'), self.title_bar)
        self.title_label.setObjectName('FontExcludeTitle')
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.reject)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        # Search bar
        self.search_edit = QLineEdit(surface)
        self.search_edit.setObjectName('FontExcludeSearch')
        self.search_edit.setPlaceholderText(self.tr("Filter Fonts"))
        self.search_edit.textChanged.connect(self._filter_lists)
        layout.addWidget(self.search_edit)

        # Side-by-side list widgets
        lists_layout = QHBoxLayout()

        # Available fonts list
        left_layout = QVBoxLayout()
        left_layout.setSpacing(6)
        available_title = QLabel(self.tr("Available Fonts"), surface)
        available_title.setObjectName('FontExcludeListTitle')
        left_layout.addWidget(available_title)
        self.available_list = QListWidget(surface)
        self.available_list.setObjectName('FontExcludeAvailableList')
        self._configure_font_list(self.available_list)
        left_layout.addWidget(self.available_list)
        lists_layout.addLayout(left_layout)

        # Center buttons
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()
        self.hide_btn = QPushButton(">", surface)
        self.hide_btn.setObjectName('FontExcludeMoveButton')
        self.hide_btn.setFixedWidth(34)
        self.hide_btn.setToolTip(self.tr("Hide selected fonts"))
        self.hide_btn.clicked.connect(self._hide_fonts)
        btn_layout.addWidget(self.hide_btn)
        self.show_btn = QPushButton("<", surface)
        self.show_btn.setObjectName('FontExcludeMoveButton')
        self.show_btn.setFixedWidth(34)
        self.show_btn.setToolTip(self.tr("Show selected fonts"))
        self.show_btn.clicked.connect(self._show_fonts)
        btn_layout.addWidget(self.show_btn)
        btn_layout.addStretch()
        lists_layout.addLayout(btn_layout)

        # Excluded fonts list
        right_layout = QVBoxLayout()
        right_layout.setSpacing(6)
        excluded_title = QLabel(self.tr("Hidden Fonts"), surface)
        excluded_title.setObjectName('FontExcludeListTitle')
        right_layout.addWidget(excluded_title)
        self.excluded_list = QListWidget(surface)
        self.excluded_list.setObjectName('FontExcludeHiddenList')
        self._configure_font_list(self.excluded_list)
        right_layout.addWidget(self.excluded_list)
        lists_layout.addLayout(right_layout)

        layout.addLayout(lists_layout)

        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 4, 0, 0)
        action_layout.setSpacing(8)

        self.legacy_btn = QPushButton(
            self.tr("Hide Legacy Fonts"),
            surface,
        )
        self.legacy_btn.setObjectName('FontExcludeLegacyButton')
        self.legacy_btn.clicked.connect(self._on_add_legacy_fonts)
        action_layout.addWidget(self.legacy_btn)
        action_layout.addStretch()

        self.cancel_button = QPushButton(self.tr('Cancel'), surface)
        self.cancel_button.setObjectName('FontExcludeSecondaryButton')
        self.cancel_button.clicked.connect(self.reject)
        action_layout.addWidget(self.cancel_button)
        self.ok_button = QPushButton(self.tr('OK'), surface)
        self.ok_button.setObjectName('FontExcludePrimaryButton')
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.accept)
        action_layout.addWidget(self.ok_button)
        layout.addLayout(action_layout)

        # Populate lists
        self._populate_lists()

    @staticmethod
    def _configure_font_list(list_widget: QListWidget) -> None:
        list_widget.setUniformItemSizes(True)
        list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )

    def _add_font_item(self, list_widget: QListWidget, font_name: str) -> None:
        """Add a font name to a list widget.

        Legacy fonts get a "[Legacy]" suffix. The original font name is stored
        in ``Qt.UserRole``.
        """
        is_legacy = font_name in LEGACY_FONTS
        display = f"{font_name} [{self.tr('Legacy')}]" if is_legacy else font_name
        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, font_name)
        list_widget.addItem(item)

    def _dismiss_transient_window(self) -> None:
        self.reject()

    @staticmethod
    def _real_name(item: QListWidgetItem) -> str:
        """Return the original font name stored in UserRole."""
        name = item.data(Qt.ItemDataRole.UserRole)
        return name if name else item.text()

    def _populate_lists(self) -> None:
        self.available_list.clear()
        self.excluded_list.clear()

        for font in shared.get_filtered_font_list(shared.FONT_FAMILIES, pcfg.excluded_fonts):
            self._add_font_item(self.available_list, font)

        for font in sorted(pcfg.excluded_fonts, key=str.casefold):
            self._add_font_item(self.excluded_list, font)

    def _filter_lists(self) -> None:
        text = self.search_edit.text().casefold()
        for list_widget in (self.available_list, self.excluded_list):
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                hidden = bool(text) and text not in self._real_name(item).casefold()
                if item.isHidden() != hidden:
                    item.setHidden(hidden)
                if hidden:
                    item.setSelected(False)

    def _move_selected(
        self,
        source: QListWidget,
        target: QListWidget,
    ) -> None:
        for item in source.selectedItems():
            if not item.isHidden():
                target.addItem(source.takeItem(source.row(item)))
        target.sortItems(Qt.SortOrder.AscendingOrder)
        self._filter_lists()

    def _hide_fonts(self) -> None:
        self._move_selected(self.available_list, self.excluded_list)

    def _show_fonts(self) -> None:
        self._move_selected(self.excluded_list, self.available_list)

    def _on_add_legacy_fonts(self) -> None:
        """Detect legacy Windows fonts and add them to the hidden list automatically."""
        # Fonts that exist on this system AND are legacy
        exist_legacy = shared.FONT_FAMILIES & LEGACY_FONTS
        already_excluded = {
            self._real_name(self.excluded_list.item(i))
            for i in range(self.excluded_list.count())
        }
        to_add = sorted(exist_legacy - already_excluded)

        if not to_add:
            QMessageBox.information(
                self,
                self.tr("Legacy Fonts"),
                self.tr("No additional legacy fonts detected on this system."),
            )
            return

        for font_name in to_add:
            for i in range(self.available_list.count()):
                if self._real_name(self.available_list.item(i)) == font_name:
                    self.excluded_list.addItem(self.available_list.takeItem(i))
                    break
            else:
                self._add_font_item(self.excluded_list, font_name)

        self.excluded_list.sortItems(Qt.SortOrder.AscendingOrder)
        self._filter_lists()

        QMessageBox.information(
            self,
            self.tr("Legacy Fonts"),
            self.tr(
                "Added {count} legacy font(s) to the hidden list:\n\n{fonts}"
            ).replace("{count}", str(len(to_add))).replace("{fonts}", "\n".join(to_add)),
        )

    def get_excluded_fonts(self) -> List[str]:
        return sorted(
            {
                self._real_name(self.excluded_list.item(i))
                for i in range(self.excluded_list.count())
            },
            key=str.casefold,
        )


class ConfigSubBlock(Widget):
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None,
    vertical_layout=True, insert_stretch: bool = False, content_margins = (0, 0, 0, 0), fnt_size=None,
    tooltip: str = None) -> None:
        super().__init__()
        if vertical_layout:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)
        layout.setContentsMargins(*content_margins)

        tooltip = tooltip or discription
        self.name_label = None
        self.description_label = None
        if tooltip is None and isinstance(widget, QWidget):
            tooltip = widget.toolTip()
        if fnt_size is None:
            fnt_size = CONFIG_FONTSIZE_CONTENT
            if discription is not None:
                fnt_size = CONFIG_FONTSIZE_CONTENT-2
        if name is not None:
            textlabel = ConfigTextLabel(name, fnt_size, QFont.Weight.Normal)
            self.name_label = textlabel
            if tooltip:
                textlabel.setToolTip(tooltip)
            layout.addWidget(textlabel)
        if discription is not None:
            description_label = ConfigTextLabel(discription, fnt_size)
            self.description_label = description_label
            if tooltip:
                description_label.setToolTip(tooltip)
            layout.addWidget(description_label)
        if insert_stretch:
            layout.insertStretch(-1)
        if isinstance(widget, QWidget):
            if tooltip and not widget.toolTip():
                widget.setToolTip(tooltip)
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
        self.widget = widget


def combobox_with_label(sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True, parent: QWidget = None, insert_stretch: bool = False) -> Tuple[ConfigComboBox, QWidget]:
    combox = ConfigComboBox(fix_size=fix_size, scrollWidget=parent)
    combox.addItems(sel)
    if discription:
        combox.setToolTip(discription)
    if target_block is None:
        sublock = ConfigSubBlock(
            combox,
            name,
            discription,
            vertical_layout=vertical_layout,
            insert_stretch=insert_stretch,
            fnt_size=CONFIG_FONTSIZE_CONTENT,
        )
        for label in (sublock.name_label, sublock.description_label):
            if label is not None:
                font = label.font()
                font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
                label.setFont(font)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().setSpacing(12)
        return combox, sublock
    else:
        layout = target_block.layout()
        layout.addSpacing(12)
        textlabel = ConfigTextLabel(name, CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal)
        font = textlabel.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        textlabel.setFont(font)
        if discription:
            textlabel.setToolTip(discription)
        layout.addWidget(textlabel)
        layout.addWidget(combox)
        return combox, target_block
    
def checkbox_with_label(
    name: str,
    discription: Optional[str] = None,
    target_block: Optional[ConfigSubBlock] = None,
) -> Tuple[QCheckBox, ConfigSubBlock]:
    checkbox = QCheckBox()
    checkbox.setObjectName('ConfigCheckBox')
    if discription is not None:
        checkbox.setToolTip(discription)

    if target_block is None:
        sublock = ConfigSubBlock(
            checkbox,
            name,
            vertical_layout=False,
            tooltip=discription,
        )
        font = sublock.name_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        sublock.name_label.setFont(font)
        sublock.layout().removeWidget(checkbox)
        sublock.layout().insertWidget(0, checkbox)
        sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        target_block = sublock
    return checkbox, target_block
    


class ConfigBlock(Widget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setContentsMargins(0, 0, 0, 0)
        self.vlayout.setSpacing(CONFIG_CONTENT_MARGIN)
        self.vlayout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        self.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

    def addLineEdit(
        self,
        name: Optional[str] = None,
        discription: Optional[str] = None,
        vertical_layout: bool = False,
    ) -> Tuple[QLineEdit, ConfigSubBlock]:
        le = QLineEdit()
        le.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        le.setFixedHeight(30)
        sublock = ConfigSubBlock(le, name, discription, vertical_layout)
        if sublock.name_label is not None:
            font = sublock.name_label.font()
            font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
            sublock.name_label.setFont(font)
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

    def addCheckBox(
        self,
        name: str,
        discription: Optional[str] = None,
        target_block: Optional[ConfigSubBlock] = None,
    ) -> Tuple[QCheckBox, ConfigSubBlock]:
        checkbox, sublock = checkbox_with_label(name, discription, target_block)
        if target_block is None:
            self.addSublock(sublock)
        return checkbox, sublock


class ConfigContent(QStackedWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('ConfigContent')
        self.config_block_list: List[ConfigBlock] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.section_index = {}

    def addConfigBlock(self, block: ConfigBlock, section_key: str):
        scroll_area = QScrollArea()
        scroll_area.setObjectName('ConfigContentScrollArea')
        scroll_area.viewport().setObjectName('ConfigContentViewport')
        fadeout_scrollbar = section_key != 'llm_profile'
        scroll_area.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, scroll_area, fadeout=fadeout_scrollbar, hover_style=True)
        scroll_area.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, scroll_area, fadeout=fadeout_scrollbar, hover_style=True)
        scroll_area.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidgetResizable(True)
        scroll_area.setContentsMargins(0, 0, 0, 0)
        scroll_content = Widget()
        scroll_content.setObjectName('ConfigContentScrollContent')
        scroll_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        scroll_layout = QHBoxLayout(scroll_content)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        scroll_layout.setSizeConstraint(LAYOUT_SET_MINIMUM_SIZE)
        scroll_layout.setContentsMargins(*CONFIG_CONTENT_MARGINS)
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

    def scrollWidgetToTop(self, section_key: str, widget: QWidget):
        index = self.section_index.get(section_key)
        if index is None:
            return
        scroll_area = self.widget(index)

        def scroll_to_widget():
            scroll_content = scroll_area.widget()
            top = widget.mapTo(scroll_content, widget.rect().topLeft()).y()
            scroll_area.verticalScrollBar().setValue(max(0, top - 12))

        QTimer.singleShot(0, scroll_to_widget)

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


class ConfigPanel(OutsideClickFramelessMixin, FramelessWindow):
    """Non-modal frameless settings window.

    >>> issubclass(ConfigPanel, FramelessWindow)
    True
    """

    save_config = Signal()
    unload_models = Signal()
    prepare_selected_modules = Signal()
    reinstall_torch = Signal()
    check_update = Signal()
    reload_textstyle = Signal(bool)
    font_list_changed = Signal(bool)
    compact_vertical_punctuation_changed = Signal(bool)
    apply_auto_tate_chu_yoko_requested = Signal()
    show_pre_MT_keyword_window = Signal()
    show_MT_keyword_window = Signal()
    show_OCR_keyword_window = Signal()

    dictionary_urls = DICTIONARY_URLS


    def __init__(self, parent: QWidget = None) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        # Establish the owned top-level type before frameless initialization.
        super().__init__(parent, window_type.Dialog)
        self.font_exclude_dialog: Optional[FontExcludeDialog] = None
        self.setObjectName("ConfigPanel")
        # QNSWindow composites a transparent outer inset as an opaque black band.
        opaque_frame = ON_WINDOWS or ON_MACOS
        self.setProperty('opaqueFrame', opaque_frame)
        self.setProperty('nativeFrame', ON_WINDOWS)
        self.setWindowTitle(self.tr('Settings'))
        self.setWindowModality(Qt.WindowModality.NonModal)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        if not opaque_frame:
            self.setAttribute(widget_attribute.WA_TranslucentBackground)
        self.setAttribute(widget_attribute.WA_StyledBackground)
        if ON_MACOS:
            self.windowEffect.removeShadowEffect(self.winId())
        self.resize(900, 720)
        self.setMinimumSize(720, 520)
        self.configTable = ConfigTable()
        self.configTable.section_pressed.connect(self.showSection)
        self.configContent = ConfigContent()
        moduleTableItem = self.configTable.addHeader(self.tr('Modules'))
        generalTableItem = self.configTable.addHeader(self.tr('General'))
        
        label_pipeline = self.tr('Pipeline')
        label_llm_profile = self.tr('LLM Profile')
        label_application = self.tr('Application')
        label_typesetting = self.tr('Typesetting')
        label_spellcheck = self.tr('Spell Checker')

        pipelineConfigPanel = self.addConfigBlock(label_pipeline, moduleTableItem, 'pipeline')
        llmProfileConfigPanel = self.addConfigBlock(label_llm_profile, moduleTableItem, 'llm_profile')
        applicationConfigPanel = self.addConfigBlock(label_application, generalTableItem, 'application')
        typesettingConfigPanel = self.addConfigBlock(label_typesetting, generalTableItem, 'typesetting')
        spellcheckConfigPanel = self.addConfigBlock(label_spellcheck, generalTableItem, 'spellcheck')
        
        pipeline_options = QWidget()
        pipeline_options.setObjectName('PipelineModuleOptions')
        pipeline_options.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pipeline_options_layout = QVBoxLayout(pipeline_options)
        pipeline_options_layout.setContentsMargins(0, 0, 0, 0)
        pipeline_options_layout.setSpacing(CONFIG_CONTENT_MARGIN)

        torch_status_row = QWidget()
        torch_status_row.setObjectName('ConfigInlineRow')
        torch_status_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        torch_status_layout = QHBoxLayout(torch_status_row)
        torch_status_layout.setContentsMargins(0, 0, 0, 0)
        torch_status_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        torch_label = QLabel(self.tr('Torch'))
        torch_label.setObjectName('TorchInfoLabel')
        torch_status_layout.addWidget(torch_label)
        self.torch_status_label = QLabel()
        self.torch_status_label.setObjectName('TorchInfoLabel')
        torch_status_layout.addWidget(self.torch_status_label)
        torch_status_layout.addStretch()
        self.reinstall_torch_btn = QPushButton(parent=self)
        self.reinstall_torch_btn.clicked.connect(self.reinstall_torch)
        torch_status_layout.addWidget(self.reinstall_torch_btn)
        pipeline_options_layout.addWidget(torch_status_row)
        self.refreshTorchStatus()

        self.empty_runcache_checker = QCheckBox(self.tr('Empty cache after RUN'))
        self.empty_runcache_checker.setObjectName('PipelineModuleActionCheckBox')
        self.empty_runcache_checker.setToolTip(
            self.tr('Empty cache after RUN to save memory.')
        )
        pipeline_options_layout.addWidget(self.empty_runcache_checker)
        self.empty_runcache_checker.stateChanged.connect(self.on_runcache_changed)
        self.package_auto_install_checker = QCheckBox(
            self.tr('Auto install missing packages')
        )
        self.package_auto_install_checker.setObjectName(
            'PipelineModuleActionCheckBox'
        )
        self.package_auto_install_checker.setToolTip(
            self.tr(
                'Install missing Python packages automatically when a selected '
                'module requires them.'
            )
        )
        self.package_auto_install_checker.stateChanged.connect(self.on_package_auto_install_changed)
        pipeline_options_layout.addWidget(self.package_auto_install_checker)

        module_actions = QWidget()
        module_actions.setObjectName('ConfigInlineRow')
        module_actions.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        pipeline_options_layout.addWidget(module_actions)

        self.replaceOCRkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for source text'),
            self,
        )
        self.replaceOCRkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        pipeline_options_layout.addWidget(self.replaceOCRkeywordBtn)
        self.replacePreMTkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for machine translation source text'),
            self,
        )
        self.replacePreMTkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replacePreMTkeywordBtn.clicked.connect(
            self.show_pre_MT_keyword_window
        )
        pipeline_options_layout.addWidget(self.replacePreMTkeywordBtn)
        self.replaceMTkeywordBtn = NoBorderPushBtn(
            self.tr('Keyword substitution for machine translation'),
            self,
        )
        self.replaceMTkeywordBtn.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        pipeline_options_layout.addWidget(self.replaceMTkeywordBtn)
        pipelineConfigPanel.vlayout.addWidget(pipeline_options)
        self.llm_profiles_panel = LLMProfilesWidget(scrollWidget=self)
        llmProfileConfigPanel.addBlockWidget(self.llm_profiles_panel)

        update_status_widget = QWidget()
        update_status_widget.setObjectName('ConfigInlineRow')
        update_status_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        for label in (self.current_version_label, self.latest_version_label):
            font = label.font()
            font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
            label.setFont(font)
        self.check_update_btn = QPushButton(parent=self)
        self.check_update_btn.setText(self.tr('Check update'))
        self.check_update_btn.clicked.connect(self.check_update)
        self.check_update_btn.setFixedHeight(PUSHBTN_FIXED_HEIGHT)

        update_status_layout.addWidget(self.current_version_label)
        update_status_layout.addSpacing(24)
        update_status_layout.addWidget(self.latest_version_label)
        update_status_layout.addSpacing(24)
        update_status_layout.addWidget(self.check_update_btn)
        applicationConfigPanel.addBlockWidget(update_status_widget)

        self.open_on_startup_checker, _ = applicationConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)

        self.check_update_on_startup_checker, _ = applicationConfigPanel.addCheckBox(self.tr('Check update on startup'))
        self.check_update_on_startup_checker.stateChanged.connect(self.on_check_update_onstartup_changed)

        self.spellcheck_checker, _ = spellcheckConfigPanel.addCheckBox(self.tr('Enable'))
        self.spellcheck_checker.stateChanged.connect(self.on_spellcheck_changed)

        self.spellcheck_on_source_checker, _ = spellcheckConfigPanel.addCheckBox(self.tr('Apply for source text'))
        self.spellcheck_on_source_checker.stateChanged.connect(self.on_spellcheck_on_source_changed)

        # Edit Distance Spinbox
        self.spellcheck_distance_spin = QSpinBox(self)
        self.spellcheck_distance_spin.setObjectName('SpellCheckDistanceSpin')
        self.spellcheck_distance_spin.setRange(1, 4)
        self.spellcheck_distance_spin.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.spellcheck_distance_spin.setToolTip(self.tr("Higher value, slower analysis"))
        self.spellcheck_distance_spin.valueChanged.connect(self.on_spellcheck_distance_changed)

        dist_layout = QHBoxLayout()
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(12)
        dist_label = ConfigTextLabel(self.tr("Edit Distance"), CONFIG_FONTSIZE_CONTENT, QFont.Weight.Normal)
        dist_label.setToolTip(self.tr("Higher value, slower analysis"))
        dist_layout.addWidget(dist_label)
        dist_layout.addWidget(self.spellcheck_distance_spin)
        dist_layout.insertStretch(-1)

        dist_block = QVBoxLayout()
        dist_block.setContentsMargins(0, 0, 0, 0)
        dist_block.setSpacing(4)
        dist_block.addLayout(dist_layout)

        spellcheckConfigPanel.addBlockWidget(dist_block)

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
        self.repo_dicts_list.setObjectName('SpellCheckDictionaryList')
        self.repo_dicts_list.setFixedHeight(150)
        self.repo_dicts_list.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, self.repo_dicts_list, hover_style=True)
        self.repo_dicts_list.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, self.repo_dicts_list, hover_style=True)
        self.repo_dicts_list.itemChanged.connect(self.on_repo_dict_item_changed)
        repo_layout.addWidget(self.repo_dicts_list)

        spellcheckConfigPanel.addBlockWidget(repo_layout)

        # External Dictionaries List
        ext_layout = QVBoxLayout()
        ext_label = ConfigTextLabel(self.tr("External Dictionaries"), CONFIG_FONTSIZE_CONTENT, QFont.Weight.Bold)
        ext_layout.addWidget(ext_label)

        self.external_dicts_list = QListWidget(self)
        self.external_dicts_list.setObjectName('SpellCheckDictionaryList')
        self.external_dicts_list.setFixedHeight(120)
        self.external_dicts_list.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, self.external_dicts_list, hover_style=True)
        self.external_dicts_list.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, self.external_dicts_list, hover_style=True)
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
        global_fntfmt_layout.setContentsMargins(0, 0, 0, 0)
        global_fntfmt_layout.setHorizontalSpacing(CONFIG_CONTENT_MARGIN)
        global_fntfmt_layout.setVerticalSpacing(CONFIG_CONTENT_MARGIN)
        global_fntfmt_widget.setContentsMargins(0, 0, 0, 0)

        global_fntfmt_group = QVBoxLayout()
        global_fntfmt_group.setContentsMargins(0, 0, 0, 0)
        global_fntfmt_group.setSpacing(CONFIG_CONTENT_MARGIN)
        global_fntfmt_group.addWidget(ConfigTextLabel(
            self.tr('Pipeline Font Formatting'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        ))
        global_fntfmt_group.addWidget(global_fntfmt_widget)
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

        self.quick_insert_characters_edit, _ = typesettingConfigPanel.addLineEdit(
            self.tr('Quick insert characters')
        )
        self.quick_insert_characters_edit.setText(pcfg.quick_insert_characters)
        self.quick_insert_characters_edit.textChanged.connect(
            self.on_quick_insert_characters_changed
        )

        self.exclude_fonts_btn = QPushButton(self.tr('Hide Unused Fonts'), parent=self)
        self.exclude_fonts_btn.clicked.connect(self.show_font_exclusion_dialog)
        font_exclusion_block = typesettingConfigPanel.addBlockWidget(
            self.exclude_fonts_btn, self.tr('Font Exclusion'),
        )
        font = font_exclusion_block.name_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        font_exclusion_block.name_label.setFont(font)
        font_exclusion_block.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        font_exclusion_block.layout().setSpacing(12)

        letter_case_row = Widget()
        letter_case_row.setObjectName('ConfigInlineRow')
        letter_case_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        letter_case_layout = QHBoxLayout(letter_case_row)
        letter_case_layout.setContentsMargins(0, 0, 0, 0)
        letter_case_layout.setSpacing(12)
        letter_case_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        letter_case_label = ConfigTextLabel(
            self.tr('Letter Case'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        font = letter_case_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        letter_case_label.setFont(font)
        letter_case_label.setToolTip(self.tr(
            'Choose how translated text letter case is adjusted after keyword substitution.'
        ))
        letter_case_layout.addWidget(letter_case_label)

        self.let_letter_case_group = QButtonGroup(letter_case_row)
        self.let_letter_case_buttons = {}
        letter_case_options = (
            (
                self.tr('None'),
                OCRTextPostprocess.NONE,
                self.tr('Keep translated text letter case unchanged.'),
            ),
            (
                self.tr('Capitalize'),
                OCRTextPostprocess.CAPITALIZE,
                self.tr(
                    'Lowercase translated text, then capitalize the first letter of each sentence.'
                ),
            ),
            (
                self.tr('Uppercase'),
                OCRTextPostprocess.UPPERCASE,
                self.tr('Convert translated text to uppercase.'),
            ),
        )
        for text, mode, tooltip in letter_case_options:
            button = QRadioButton(text, letter_case_row)
            button.setObjectName('ConfigLetterCaseOption')
            button.setProperty('letterCaseMode', mode)
            button.setToolTip(tooltip)
            font = button.font()
            font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
            button.setFont(font)
            button.setChecked(pcfg.let_letter_case == mode)
            button.toggled.connect(self.on_letter_case_changed)
            self.let_letter_case_group.addButton(button)
            self.let_letter_case_buttons[mode] = button
            letter_case_layout.addWidget(button)
        letter_case_layout.addStretch()
        typesettingConfigPanel.addBlockWidget(letter_case_row)

        self.let_autolayout_checker, sublock = typesettingConfigPanel.addCheckBox(
            self.tr('Auto layout'),
            discription=self.tr(
                'Split translation into multi-lines according to the extracted balloon region.'
            ),
        )
        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)

        self.let_textstyle_indep_checker, _ = typesettingConfigPanel.addCheckBox(self.tr('Independent text styles for each projects'))
        self.let_textstyle_indep_checker.stateChanged.connect(self.on_textstyle_indep_changed)

        self.let_show_only_custom_fonts, sublock = typesettingConfigPanel.addCheckBox(self.tr("Show only custom fonts"))
        self.let_show_only_custom_fonts.stateChanged.connect(self.on_show_only_custom_fonts)

        font_format_block = typesettingConfigPanel.addBlockWidget(global_fntfmt_group)
        font_format_block.layout().setContentsMargins(0, 0, 0, 0)
        font_format_block.setContentsMargins(0, 0, 0, 0)

        vertical_layout_group = QVBoxLayout()
        vertical_layout_group.setContentsMargins(0, 0, 0, 0)
        vertical_layout_group.setSpacing(CONFIG_CONTENT_MARGIN)
        vertical_layout_group.addWidget(ConfigTextLabel(
            self.tr('Vertical Text Layout'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        ))

        (
            self.compact_vertical_punctuation_checker,
            compact_vertical_punctuation_row,
        ) = checkbox_with_label(
            self.tr('Compact punctuation spacing'),
            discription=self.tr(
                'Remove extra spacing around punctuation in vertical text.'
            ),
        )
        self.compact_vertical_punctuation_checker.setChecked(
            pcfg.compact_vertical_punctuation_spacing
        )
        self.compact_vertical_punctuation_checker.toggled.connect(
            self.on_compact_vertical_punctuation_changed
        )
        vertical_layout_group.addWidget(compact_vertical_punctuation_row)

        auto_tcy_group = Widget()
        auto_tcy_group.setObjectName('ConfigInlineRow')
        auto_tcy_group.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        auto_tcy_layout = QVBoxLayout(auto_tcy_group)
        auto_tcy_layout.setContentsMargins(0, 0, 0, 0)
        auto_tcy_layout.setSpacing(12)

        auto_tcy_header = Widget()
        auto_tcy_header.setObjectName('ConfigInlineRow')
        auto_tcy_header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        auto_tcy_header_layout = QHBoxLayout(auto_tcy_header)
        auto_tcy_header_layout.setContentsMargins(0, 0, 0, 0)
        auto_tcy_header_layout.setSpacing(8)
        auto_tcy_header_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.auto_tate_chu_yoko_checker = QCheckBox(auto_tcy_header)
        self.auto_tate_chu_yoko_checker.setObjectName('ConfigCheckBox')
        auto_tcy_title = ConfigTextLabel(
            self.tr('Automatic Tate-chu-yoko'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
            parent=auto_tcy_header,
        )
        font = auto_tcy_title.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        auto_tcy_title.setFont(font)
        auto_tcy_tooltip = self.tr(
            'Automatically combine matching character runs into one upright horizontal unit in vertical text.'
        )
        self.auto_tate_chu_yoko_checker.setToolTip(auto_tcy_tooltip)
        auto_tcy_title.setToolTip(auto_tcy_tooltip)
        auto_tcy_header_layout.addWidget(self.auto_tate_chu_yoko_checker)
        auto_tcy_header_layout.addWidget(auto_tcy_title)
        self.auto_tate_chu_yoko_apply_btn = QPushButton(
            self.tr('Apply'),
            auto_tcy_header,
        )
        auto_tcy_header_layout.addWidget(self.auto_tate_chu_yoko_apply_btn)
        auto_tcy_header_layout.addStretch()
        auto_tcy_layout.addWidget(auto_tcy_header)

        self.auto_tate_chu_yoko_options = Widget()
        self.auto_tate_chu_yoko_options.setObjectName('ConfigInlineRow')
        self.auto_tate_chu_yoko_options.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        auto_tcy_options_layout = QVBoxLayout(self.auto_tate_chu_yoko_options)
        auto_tcy_options_layout.setContentsMargins(24, 0, 0, 0)
        auto_tcy_options_layout.setSpacing(12)

        max_length_row = Widget()
        max_length_row.setObjectName('ConfigInlineRow')
        max_length_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        max_length_layout = QHBoxLayout(max_length_row)
        max_length_layout.setContentsMargins(0, 0, 0, 0)
        max_length_layout.setSpacing(12)
        max_length_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        max_length_label = ConfigTextLabel(
            self.tr('Maximum Run Length'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        font = max_length_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        max_length_label.setFont(font)
        max_length_tooltip = self.tr(
            'Maximum number of consecutive matching characters to combine.'
        )
        max_length_label.setToolTip(max_length_tooltip)
        self.auto_tate_chu_yoko_max_length = QSpinBox(max_length_row)
        self.auto_tate_chu_yoko_max_length.setObjectName(
            'AutoTateChuYokoMaxLength'
        )
        self.auto_tate_chu_yoko_max_length.setRange(1, 99)
        self.auto_tate_chu_yoko_max_length.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.auto_tate_chu_yoko_max_length.setToolTip(max_length_tooltip)
        max_length_layout.addWidget(max_length_label)
        max_length_layout.addWidget(self.auto_tate_chu_yoko_max_length)
        max_length_layout.addStretch()

        category_row = Widget()
        category_row.setObjectName('ConfigInlineRow')
        category_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        category_layout = QHBoxLayout(category_row)
        category_layout.setContentsMargins(0, 0, 0, 0)
        category_layout.setSpacing(8)
        category_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.auto_tate_chu_yoko_numbers = QCheckBox(category_row)
        self.auto_tate_chu_yoko_numbers.setObjectName('ConfigCheckBox')
        numbers_label = ConfigTextLabel(
            self.tr('Numbers'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        font = numbers_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        numbers_label.setFont(font)
        numbers_tooltip = self.tr(
            'Include consecutive digits from 0 to 9 in automatic runs.'
        )
        self.auto_tate_chu_yoko_numbers.setToolTip(numbers_tooltip)
        numbers_label.setToolTip(numbers_tooltip)
        category_layout.addWidget(self.auto_tate_chu_yoko_numbers)
        category_layout.addWidget(numbers_label)
        category_layout.addSpacing(16)
        self.auto_tate_chu_yoko_letters = QCheckBox(category_row)
        self.auto_tate_chu_yoko_letters.setObjectName('ConfigCheckBox')
        letters_label = ConfigTextLabel(
            self.tr('Letters'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        font = letters_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        letters_label.setFont(font)
        letters_tooltip = self.tr(
            'Include consecutive Latin letters from A to Z and a to z in automatic runs.'
        )
        self.auto_tate_chu_yoko_letters.setToolTip(letters_tooltip)
        letters_label.setToolTip(letters_tooltip)
        category_layout.addWidget(self.auto_tate_chu_yoko_letters)
        category_layout.addWidget(letters_label)
        category_layout.addStretch()

        additional_chars_row = Widget()
        additional_chars_row.setObjectName('ConfigInlineRow')
        additional_chars_row.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        additional_chars_layout = QHBoxLayout(additional_chars_row)
        additional_chars_layout.setContentsMargins(0, 0, 0, 0)
        additional_chars_layout.setSpacing(12)
        additional_chars_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        additional_chars_label = ConfigTextLabel(
            self.tr('Additional Characters'),
            CONFIG_FONTSIZE_CONTENT,
            QFont.Weight.Normal,
        )
        font = additional_chars_label.font()
        font.setPixelSize(CONFIG_FONTSIZE_CONTENT)
        additional_chars_label.setFont(font)
        additional_chars_tooltip = self.tr(
            'Other characters that can participate in an automatic run.'
        )
        additional_chars_label.setToolTip(additional_chars_tooltip)
        self.auto_tate_chu_yoko_additional_chars = QLineEdit(
            additional_chars_row
        )
        self.auto_tate_chu_yoko_additional_chars.setFixedWidth(
            CONFIG_COMBOBOX_SHORT
        )
        self.auto_tate_chu_yoko_additional_chars.setToolTip(
            additional_chars_tooltip
        )
        additional_chars_layout.addWidget(additional_chars_label)
        additional_chars_layout.addWidget(
            self.auto_tate_chu_yoko_additional_chars
        )
        additional_chars_layout.addStretch()
        auto_tcy_options_layout.addWidget(category_row)
        auto_tcy_options_layout.addWidget(additional_chars_row)
        auto_tcy_options_layout.addWidget(max_length_row)
        auto_tcy_layout.addWidget(self.auto_tate_chu_yoko_options)
        vertical_layout_group.addWidget(auto_tcy_group)

        auto_tcy = pcfg.auto_tate_chu_yoko
        self.auto_tate_chu_yoko_checker.setChecked(auto_tcy.enabled)
        self.auto_tate_chu_yoko_max_length.setValue(auto_tcy.max_length)
        self.auto_tate_chu_yoko_numbers.setChecked(auto_tcy.include_numbers)
        self.auto_tate_chu_yoko_letters.setChecked(auto_tcy.include_letters)
        self.auto_tate_chu_yoko_additional_chars.setText(
            auto_tcy.additional_chars
        )
        self.auto_tate_chu_yoko_options.setVisible(auto_tcy.enabled)
        self.auto_tate_chu_yoko_apply_btn.setVisible(auto_tcy.enabled)

        self.auto_tate_chu_yoko_checker.toggled.connect(
            self.on_auto_tate_chu_yoko_changed
        )
        self.auto_tate_chu_yoko_apply_btn.clicked.connect(
            self.on_apply_auto_tate_chu_yoko_clicked
        )
        self.auto_tate_chu_yoko_max_length.valueChanged.connect(
            self.on_auto_tate_chu_yoko_max_length_changed
        )
        self.auto_tate_chu_yoko_numbers.toggled.connect(
            self.on_auto_tate_chu_yoko_numbers_changed
        )
        self.auto_tate_chu_yoko_letters.toggled.connect(
            self.on_auto_tate_chu_yoko_letters_changed
        )
        self.auto_tate_chu_yoko_additional_chars.textChanged.connect(
            self.on_auto_tate_chu_yoko_additional_chars_changed
        )

        vertical_layout_block = typesettingConfigPanel.addBlockWidget(
            vertical_layout_group
        )
        vertical_layout_block.layout().setContentsMargins(0, 0, 0, 0)
        vertical_layout_block.setContentsMargins(0, 0, 0, 0)

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

        root_layout = QVBoxLayout(self)
        margin = 0 if opaque_frame else 5
        root_layout.setContentsMargins(margin, margin, margin, margin)

        surface = QFrame(self)
        surface.setObjectName('ConfigPanelSurface')
        root_layout.addWidget(surface)

        window_layout = QVBoxLayout(surface)
        window_layout.setSpacing(0)
        window_layout.setContentsMargins(6, 0, 6, 6)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('ConfigPanelTitleBar')
        self.title_bar.setFixedHeight(TITLEBAR_HEIGHT)
        title_layout = QGridLayout(self.title_bar)
        title_layout.setContentsMargins(6, 1, 6, 1)
        title_layout.setSpacing(0)
        title_layout.setColumnMinimumWidth(0, 46)
        title_layout.setColumnStretch(1, 1)

        self.title_label = QLabel(self.tr('Settings'), self.title_bar)
        self.title_label.setObjectName('ConfigPanelTitle')
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(self.title_label, 0, 1)

        self.close_button = QPushButton(self.title_bar)
        self.close_button.setObjectName('closeBtn')
        self.close_button.setToolTip(self.tr('Close'))
        self.close_button.setAccessibleName(self.tr('Close'))
        self.close_button.clicked.connect(self.hide)
        title_layout.addWidget(self.close_button, 0, 2)

        window_layout.addWidget(self.title_bar)
        window_layout.addWidget(splitter, 1)

        self.configTable.expandAll()
        self.showSection('application')

    def on_runcache_changed(self):
        pcfg.module.empty_runcache = self.empty_runcache_checker.isChecked()

    def on_package_auto_install_changed(self):
        pcfg.package_manager.auto_install_missing_packages = self.package_auto_install_checker.isChecked()

    def refreshTorchStatus(self) -> None:
        version, device = probe_torch_package()
        if version is not None:
            status = self.tr(
                'Installed ({version}, {device})'
            ).format(version=version, device=device)
        else:
            status = self.tr('Not installed')
        self.torch_status_label.setText(status)
        self.reinstall_torch_btn.setText(
            self.tr('Reinstall Torch') if version is not None else self.tr('Install Torch')
        )
        self.reinstall_torch_btn.setEnabled(True)

    def setTorchInstalling(self) -> None:
        self.torch_status_label.setText(self.tr('Installing...'))
        self.reinstall_torch_btn.setEnabled(False)

    def addConfigBlock(self, header: str, parent_item: TableItem, section_key: str) -> ConfigBlock:
        cb = ConfigBlock(parent=self)
        self.configContent.addConfigBlock(cb, section_key)
        self.configTable.addSection(parent_item, header, section_key)
        return cb

    def showConfigDialog(self, section_key: str = None):
        self.refreshTorchStatus()
        if section_key is not None:
            self.showSection(section_key)
        elif self.configContent.currentIndex() < 0:
            self.showSection('application')
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
        if enabled:
            manager = SpellCheckManager.get_instance()
            if not manager.is_available():
                # Uncheck immediately to prevent UI state drift while prompting
                self.spellcheck_checker.blockSignals(True)
                self.spellcheck_checker.setChecked(False)
                self.spellcheck_checker.blockSignals(False)

                if manager.install_pyspellchecker(self):
                    self.spellcheck_checker.blockSignals(True)
                    self.spellcheck_checker.setChecked(True)
                    self.spellcheck_checker.blockSignals(False)
                    enabled = True
                else:
                    return

        pcfg.spellcheck_enabled = enabled
        self.spellcheck_on_source_checker.setEnabled(enabled)
        self.manage_words_btn.setEnabled(enabled)
        self.repo_dicts_list.setEnabled(enabled)
        self.external_dicts_list.setEnabled(enabled)
        self.add_ext_btn.setEnabled(enabled)
        self.remove_ext_btn.setEnabled(enabled)
        self.spellcheck_distance_spin.setEnabled(enabled)
        SpellCheckManager.get_instance().notify_config_changed()
        self.save_config.emit()

    def on_spellcheck_on_source_changed(self):
        enabled = self.spellcheck_on_source_checker.isChecked()
        pcfg.spellcheck_on_source_enabled = enabled
        SpellCheckManager.get_instance().notify_config_changed()
        self.save_config.emit()

    def on_spellcheck_distance_changed(self):
        pcfg.spellcheck_distance = self.spellcheck_distance_spin.value()
        SpellCheckManager.get_instance().notify_config_changed()
        self.save_config.emit()

    def open_words_manager(self) -> None:
        dialog = DictionaryManagerDialog(self)
        try:
            dialog.exec_()
        finally:
            dialog.deleteLater()

    def on_repo_dict_item_changed(self, item):
        url, filename = item.data(Qt.ItemDataRole.UserRole)
        save_path = os.path.join(PROGRAM_PATH, 'data', 'dictionaries', filename)

        LOGGER.info(f"on_repo_dict_item_changed: item={item.text()}, checkState={item.checkState()}")

        if item.checkState() == Qt.CheckState.Checked:
            if not os.path.exists(save_path):
                # Uncheck immediately so it doesn't look enabled while downloading
                self.repo_dicts_list.blockSignals(True)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.repo_dicts_list.blockSignals(False)
                
                # Start async download
                self.download_repo_dict_async(item, url, filename)
                return

        self.save_repo_dicts_config()

    def save_repo_dicts_config(self):
        self.repo_dicts_list.blockSignals(True)
        try:
            enabled_files = []
            for idx in range(self.repo_dicts_list.count()):
                it = self.repo_dicts_list.item(idx)
                if it.checkState() == Qt.CheckState.Checked:
                    _, fname = it.data(Qt.ItemDataRole.UserRole)
                    enabled_files.append(fname)

            pcfg.spellcheck_repo_dicts = ",".join(enabled_files)
            SpellCheckManager.get_instance().notify_config_changed()
            self.save_config.emit()
        finally:
            self.repo_dicts_list.blockSignals(False)

    def download_repo_dict_async(self, item, url, filename):
        save_path = os.path.join(PROGRAM_PATH, 'data', 'dictionaries', filename)

        progress = QProgressDialog(self.tr("Downloading dictionary..."), self.tr("Cancel"), 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        thread = DictDownloadThread(url, save_path)

        def on_progress(downloaded, total):
            if total > 0:
                percent = int(downloaded * 100 / total)
                progress.setValue(percent)

        def on_finished(success, err):
            progress.close()
            if success:
                LOGGER.info(f"download_repo_dict_async: successful download of {filename}")
                self.repo_dicts_list.blockSignals(True)
                try:
                    item.setCheckState(Qt.CheckState.Checked)
                    # Remove " (Installed local)" if it exists, then append it
                    clean_text = item.text().replace(self.tr(" - Installed"), "")
                    item.setText(clean_text + self.tr(" - Installed"))
                finally:
                    self.repo_dicts_list.blockSignals(False)
                
                self.save_repo_dicts_config()
                
                QMessageBox.information(self, self.tr("Download Complete"), self.tr("Dictionary downloaded successfully!"))
            else:
                LOGGER.warning(f"download_repo_dict_async: download failed for {filename}: {err}")
                if "cancelled" not in err.lower():
                    QMessageBox.warning(self, self.tr("Download Failed"), self.tr("Failed to download dictionary: ") + err)

        thread.progress.connect(on_progress)
        thread.finished.connect(on_finished)
        progress.canceled.connect(thread.cancel)

        self.active_download_thread = thread
        thread.start()
        progress.exec_()
        thread.wait()

    def add_external_dictionary(self):
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

    def on_quick_insert_characters_changed(self, text: str) -> None:
        pcfg.quick_insert_characters = text

    def on_letter_case_changed(self, checked: bool) -> None:
        button = self.sender()
        if checked and isinstance(button, QRadioButton):
            pcfg.let_letter_case = button.property('letterCaseMode')

    def on_auto_tate_chu_yoko_changed(self, enabled: bool) -> None:
        pcfg.auto_tate_chu_yoko.enabled = enabled
        self.auto_tate_chu_yoko_options.setVisible(enabled)
        self.auto_tate_chu_yoko_apply_btn.setVisible(enabled)

    def on_apply_auto_tate_chu_yoko_clicked(self) -> None:
        self.apply_auto_tate_chu_yoko_requested.emit()

    def on_compact_vertical_punctuation_changed(
        self,
        enabled: bool,
    ) -> None:
        pcfg.compact_vertical_punctuation_spacing = enabled
        self.compact_vertical_punctuation_changed.emit(enabled)

    def on_auto_tate_chu_yoko_max_length_changed(self, value: int) -> None:
        pcfg.auto_tate_chu_yoko.max_length = value

    def on_auto_tate_chu_yoko_numbers_changed(self, checked: bool) -> None:
        pcfg.auto_tate_chu_yoko.include_numbers = checked

    def on_auto_tate_chu_yoko_letters_changed(self, checked: bool) -> None:
        pcfg.auto_tate_chu_yoko.include_letters = checked

    def on_auto_tate_chu_yoko_additional_chars_changed(
        self,
        text: str,
    ) -> None:
        pcfg.auto_tate_chu_yoko.additional_chars = text

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

    def on_show_only_custom_fonts(self) -> None:
        pcfg.let_show_only_custom_fonts_flag = self.let_show_only_custom_fonts.isChecked()
        self.font_list_changed.emit(pcfg.let_show_only_custom_fonts_flag)

    def show_font_exclusion_dialog(self) -> None:
        dialog = self.font_exclude_dialog
        if dialog is not None:
            dialog.raise_()
            dialog.activateWindow()
            return

        dialog = FontExcludeDialog(self.parentWidget() or self)
        dialog.accepted.connect(self._apply_pending_font_exclusions)
        dialog.finished.connect(self._clear_font_exclude_dialog)
        self.font_exclude_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _apply_pending_font_exclusions(self) -> None:
        dialog = self.font_exclude_dialog
        if dialog is not None:
            self._apply_font_exclusions(dialog.get_excluded_fonts())

    def _clear_font_exclude_dialog(self, _result: int) -> None:
        self.font_exclude_dialog = None

    def _apply_font_exclusions(self, excluded_fonts: List[str]) -> None:
        if excluded_fonts == pcfg.excluded_fonts:
            return
        pcfg.excluded_fonts = excluded_fonts
        self.font_list_changed.emit(pcfg.let_show_only_custom_fonts_flag)
        self.save_config.emit()

    def focusOnLLMProfile(self, profile_id: str, expand_details: bool = True, target: str = 'api_key'):
        self.showConfigDialog('llm_profile')
        self.llm_profiles_panel.focusProfileControl(profile_id, target=target, expand_details=expand_details)

    def hideEvent(self, e) -> None:
        if hasattr(self, 'llm_profiles_panel'):
            self.llm_profiles_panel.collapseProfiles()
        self.save_config.emit()
        return super().hideEvent(e)

    def _preserve_on_outside_click(self) -> bool:
        return self._activeWidgetInWhitelist()

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

        if pcfg.open_recent_on_startup:
            self.open_on_startup_checker.setChecked(True)
        self.check_update_on_startup_checker.setChecked(pcfg.check_update_on_startup)
        
        # Setup repository dictionaries
        active_repos = pcfg.spellcheck_repo_dicts.split(',')
        active_repos = [x.strip() for x in active_repos if x.strip()]
        
        self.repo_dicts_list.blockSignals(True)
        self.repo_dicts_list.clear()
        
        for lang_name, url in self.dictionary_urls.items():
            filename = url.split('/')[-1]
            dict_path = os.path.join(PROGRAM_PATH, 'data', 'dictionaries', filename)

            display_text = self.tr(lang_name)
            exists = os.path.exists(dict_path)
            if exists:
                display_text += self.tr(" - Installed")

            item = QListWidgetItem(display_text, self.repo_dicts_list)
            item.setData(Qt.ItemDataRole.UserRole, (url, filename))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)

            if filename in active_repos and exists:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
        self.repo_dicts_list.blockSignals(False)

        # Synchronize config to remove any manually deleted dictionaries
        enabled_files = []
        for idx in range(self.repo_dicts_list.count()):
            it = self.repo_dicts_list.item(idx)
            if it.checkState() == Qt.CheckState.Checked:
                _, fname = it.data(Qt.ItemDataRole.UserRole)
                enabled_files.append(fname)
        pcfg.spellcheck_repo_dicts = ",".join(enabled_files)

        # Setup external dictionaries
        self.external_dicts_list.clear()
        ext_paths = pcfg.spellcheck_external_dict_path.split(';')
        for p in ext_paths:
            p = p.strip()
            if p:
                self.external_dicts_list.addItem(p)

        self.spellcheck_checker.blockSignals(True)
        if pcfg.spellcheck_enabled and not SpellCheckManager.get_instance().is_available():
            pcfg.spellcheck_enabled = False
        self.spellcheck_checker.setChecked(pcfg.spellcheck_enabled)
        self.spellcheck_checker.blockSignals(False)
        self.spellcheck_on_source_checker.setChecked(getattr(pcfg, 'spellcheck_on_source_enabled', False))
        self.spellcheck_on_source_checker.setEnabled(pcfg.spellcheck_enabled)
        self.spellcheck_distance_spin.setValue(getattr(pcfg, 'spellcheck_distance', 1))
        self.spellcheck_distance_spin.setEnabled(pcfg.spellcheck_enabled)
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

        self.let_effect_combox.setCurrentIndex(pcfg.let_fnteffect_flag)
        self.let_fntsize_combox.setCurrentIndex(pcfg.let_fntsize_flag)
        self.let_fntstroke_combox.setCurrentIndex(pcfg.let_fntstroke_flag)
        self.let_fntcolor_combox.setCurrentIndex(pcfg.let_fntcolor_flag)
        self.let_fnt_scolor_combox.setCurrentIndex(pcfg.let_fnt_scolor_flag)
        self.let_alignment_combox.setCurrentIndex(pcfg.let_alignment_flag)
        self.let_family_combox.setCurrentIndex(pcfg.let_family_flag)
        self.let_writing_mode_combox.setCurrentIndex(pcfg.let_writing_mode_flag)
        self.let_autolayout_checker.setChecked(pcfg.let_autolayout_flag)
        self.quick_insert_characters_edit.setText(pcfg.quick_insert_characters)
        self.let_letter_case_buttons[pcfg.let_letter_case].setChecked(True)
        self.compact_vertical_punctuation_checker.setChecked(
            pcfg.compact_vertical_punctuation_spacing
        )
        auto_tcy = pcfg.auto_tate_chu_yoko
        self.auto_tate_chu_yoko_checker.setChecked(auto_tcy.enabled)
        self.auto_tate_chu_yoko_max_length.setValue(auto_tcy.max_length)
        self.auto_tate_chu_yoko_numbers.setChecked(auto_tcy.include_numbers)
        self.auto_tate_chu_yoko_letters.setChecked(auto_tcy.include_letters)
        self.auto_tate_chu_yoko_additional_chars.setText(
            auto_tcy.additional_chars
        )
        self.auto_tate_chu_yoko_options.setVisible(auto_tcy.enabled)
        self.auto_tate_chu_yoko_apply_btn.setVisible(auto_tcy.enabled)
        self.let_textstyle_indep_checker.setChecked(pcfg.let_textstyle_indep_flag)
        self.rst_imgformat_combobox.setCurrentText(pcfg.imgsave_ext.replace('.', '').upper())
        self.intermediate_imgformat_combobox.setCurrentText(pcfg.intermediate_imgsave_ext.replace('.', '').upper())
        self.rst_imgquality_edit.setText(str(pcfg.imgsave_quality))
        self.empty_runcache_checker.setChecked(pcfg.module.empty_runcache)
        self.package_auto_install_checker.setChecked(pcfg.package_manager.auto_install_missing_packages)
        self.let_show_only_custom_fonts.setChecked(pcfg.let_show_only_custom_fonts_flag)

        self.blockSignals(False)
