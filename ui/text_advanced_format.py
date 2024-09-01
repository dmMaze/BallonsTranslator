from typing import Any, Callable

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QComboBox, QLabel, QHBoxLayout
from qtpy.QtCore import Signal, Qt, QRectF

from .custom_widget import PanelGroupBox, PanelArea, ComboBox
from utils.fontformat import FontFormat


class TextAdvancedFormatPanel(PanelArea):

    param_changed = Signal(str, object)

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None

        self.linespacing_type_combobox = ComboBox(
            parent=self,
            options=[
                self.tr("Proportional"),
                self.tr("Distance")
            ]
        )
        self.linespacing_type_combobox.activated.connect(
            lambda:  self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex)
        )
        linespacing_type_label = QLabel(self.tr('Line Spacing Type: '))
        linespacing_type_layout = QHBoxLayout()
        linespacing_type_layout.addWidget(linespacing_type_label)
        linespacing_type_layout.addWidget(self.linespacing_type_combobox)
        # linespacing_type_layout.addStretch()

        vlayout = QVBoxLayout()
        vlayout.addLayout(linespacing_type_layout)
        self.setContentLayout(vlayout)

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)

    def on_format_changed(self, format_name: str, get_format: Callable):
        self.param_changed.emit(format_name, get_format())