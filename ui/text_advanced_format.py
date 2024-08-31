from qtpy.QtWidgets import QVBoxLayout, QPushButton, QComboBox
from qtpy.QtCore import Signal, Qt, QRectF

from .custom_widget import PanelGroupBox, PanelArea, ComboBox
from utils.fontformat import FontFormat


class TextAdvancedFormatPanel(PanelArea):

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None

        self.linespacing_combobox = ComboBox(
            parent=self,
            options=[
                self.tr("Proportional"),
                self.tr("Distance")
            ]
        )

        vlayout = QVBoxLayout()
        vlayout.addWidget(QPushButton('rrrr'))
        self.setContentLayout(vlayout)

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format