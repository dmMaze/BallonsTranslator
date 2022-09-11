from qtpy.QtWidgets import QMenu, QAbstractItemView, QListWidget, QListWidgetItem, QWidget, QGridLayout, QPushButton, QVBoxLayout
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QContextMenuEvent, QShowEvent, QHideEvent
from typing import List, Union

from .misc import FontFormat
from .stylewidgets import Widget

def mutate_dict_key(adict: dict, old_key: Union[str, int], new_key: str):
    # https://stackoverflow.com/questions/12150872/change-key-in-ordereddict-without-losing-order
    key_list = list(adict.keys())
    if isinstance(old_key, int):
        old_key = key_list[old_key]
    
    for key in key_list:
        value = adict.pop(key)
        adict[new_key if old_key == key else key] = value


class PresetListWidget(QListWidget):

    load = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.presets = {}
        self.current_fmt = {}
        self.last_editing_item = None
        self.default_preset_name = self.tr('preset')
        self.itemDoubleClicked.connect(self.on_edit_item)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.itemDelegate().commitData.connect(self.on_commit_data)

    def on_edit_item(self):
        self.last_editing_item = self.currentItem()

    def on_commit_data(self, editor: QWidget):
        item = self.last_editing_item
        text = editor.text()
        founds = self.findItems(text, Qt.MatchFlag.MatchExactly)
        
        if len(founds) > 1:
            text = self.handle_duplicate_name(text)
            item.setText(text)
        mutate_dict_key(self.presets, self.row(item), text)

    def removeItems(self, items: List[QListWidgetItem]):
        for item in items:
            key = item.text()
            if key in self.presets:
                self.presets.pop(key)
            self.takeItem(self.row(item))

    def contextMenuEvent(self, e: QContextMenuEvent) -> None:
        menu = QMenu()
        delete_act = menu.addAction(self.tr('Delete'))
        new_act = menu.addAction(self.tr('New preset'))
        load_act = menu.addAction(self.tr('Load preset'))
        rst = menu.exec_(e.globalPos())
        if rst == delete_act:
            self.removeItems(self.selectedItems())
        elif rst == new_act:
            self.add_new_preset()
        elif rst == load_act:
            self.load.emit()

        return super().contextMenuEvent(e)

    def handle_duplicate_name(self, name: str, preset_num=0) -> str:
        dd_name = name
        while True:
            if not dd_name in self.presets:
                break
            preset_num += 1
            dd_name = name + '_' + str(preset_num).zfill(3)
        return dd_name

    def add_new_preset(self, preset_name: str = None):
        if preset_name is None:
            preset_name = self.default_preset_name + '_' + str(self.count() + 1).zfill(3)
            preset_name = self.handle_duplicate_name(preset_name)
        item = QListWidgetItem(preset_name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.addItem(item)
        self.presets[preset_name] = self.current_fmt.copy()
        self.editItem(item)

    def addItem(self, item: QListWidgetItem) -> None:
        font = item.font()
        font.setPointSizeF(12)
        item.setFont(font)
        return super().addItem(item)

    def editItem(self, item: QListWidgetItem) -> None:
        self.last_editing_item = item
        return super().editItem(item)


class PresetPanel(Widget):

    hide_signal = Signal()
    load_preset = Signal(FontFormat)

    def __init__(self, parent: QWidget = None, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        self.setWindowTitle(self.tr('Text Style Presets'))
        self.list_widget = PresetListWidget(self)
        self.new_btn = QPushButton(self.tr('New'))
        self.new_tip = self.tr('Create new preset: ')
        self.new_btn.clicked.connect(self.on_new_clicked)
        self.delete_btn = QPushButton(self.tr('Delete'), self)
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        self.load_btn = QPushButton(self.tr('Load'), self)
        self.load_btn.setToolTip(self.tr('Load preset as global format'))
        self.load_btn.clicked.connect(self.on_load_clicked)
        self.exit_btn = QPushButton(self.tr('Exit'), self)
        self.exit_btn.clicked.connect(self.on_exit_clicked)

        self.editing_item: QListWidgetItem = None
        self.global_fmt_str = ''
        
        layout = QGridLayout()
        layout.addWidget(self.new_btn, 0, 0)
        layout.addWidget(self.delete_btn, 0, 1)
        layout.addWidget(self.load_btn, 1, 0)
        layout.addWidget(self.exit_btn, 1, 1)

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(self.list_widget)
        vlayout.addLayout(layout)

    def updateCurrentFontFormat(self, fmt: Union[FontFormat, dict], fmtname: str):
        self.new_btn.setToolTip(self.new_tip + fmtname)
        if isinstance(fmt, FontFormat):
            fmt = vars(fmt)
        self.list_widget.current_fmt = fmt

    def on_new_clicked(self):
        self.list_widget.add_new_preset()

    def on_delete_clicked(self):
        self.list_widget.removeItems(self.list_widget.selectedItems())

    def on_load_clicked(self):
        sel = self.list_widget.selectedItems()
        if len(sel) > 0:
            sel = sel[0]
            preset = self.list_widget.presets[sel.text()]
            preset = FontFormat(**preset)
            self.list_widget.current_fmt = vars(preset)
            self.new_btn.setToolTip(self.new_tip + self.global_fmt_str)
            self.load_preset.emit(preset)

    def on_exit_clicked(self):
        self.hide()

    def initPresets(self, presets: dict):
        self.list_widget.presets = presets
        for key in presets:
            item = QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self.list_widget.addItem(item)

    def showEvent(self, e: QShowEvent) -> None:
        return super().showEvent(e)

    def hideEvent(self, e: QHideEvent) -> None:
        self.hide_signal.emit()
        return super().hideEvent(e)