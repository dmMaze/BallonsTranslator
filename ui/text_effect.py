from qtpy.QtWidgets import QVBoxLayout, QPushButton
from qtpy.QtCore import Signal, Qt, QRectF

from .custom_widget import PanelGroupBox, PanelArea
from utils.fontformat import FontFormat

class TextEffectPanel(PanelArea):

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str):
        super().__init__(panel_name, config_name, config_expand_name)

        # self.flayout = FlowLayout(self.scrollContent)
        # # margin = 7
        # # self.flayout.setVerticalSpacing(7)
        # # self.flayout.setHorizontalSpacing(7)
        # # self.flayout.setContentsMargins(margin, margin, margin, margin)

        gradient_group = PanelGroupBox(title=self.tr('Gradient'))
        gradient_group.setAutoFillBackground(True)
        glayout = QVBoxLayout(gradient_group)
        glayout.addWidget(QPushButton('ttt'))

        vlayout = QVBoxLayout()
        vlayout.addWidget(gradient_group)
        self.setContentLayout(vlayout)

        self.active_format: FontFormat = None

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format