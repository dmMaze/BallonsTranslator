from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout

from .scrollbar import ScrollBar
from .combobox import ComboBox, ConfigComboBox, ParamComboBox, SizeComboBox, SmallComboBox, SmallSizeComboBox
from .widget import Widget, SeparatorWidget
from .view_panel import PanelGroupBox, PanelArea, PanelAreaContent, ViewWidget, ExpandLabel
from .message import MessageBox, TaskProgressBar, FrameLessMessageBox, ProgressMessageBox, ImgtransProgressMessageBox
from .flow_layout import FlowLayout
from .label import FadeLabel, SmallColorPickerLabel, ColorPickerLabel, ConfigClickableLabel, ClickableLabel, CheckableLabel, TextCheckerLabel, ParamNameLabel, SmallParamLabel, SizeControlLabel, SmallSizeControlLabel
from .slider import PaintQSlider
from .helper import isDarkTheme, themeColor
from .push_button import NoBorderPushBtn
from .checkbox import QFontChecker, AlignmentChecker


def combobox_with_label(param_name: str = None, size='small', options=None, parent=None, scrollWidget=None, label_alignment=None, vertical_layout=False, editable=False, label=False):
    combobox_cls = SmallComboBox if size == 'small' else ComboBox
    combobox = combobox_cls(options=options, parent=parent, scrollWidget=scrollWidget)
    combobox.setEditable(editable)
    if label is None:
        label_cls = SmallParamLabel if size == 'small' else ParamNameLabel
        label = label_cls(param_name=param_name, alignment=label_alignment)
    if vertical_layout:
        layout = QVBoxLayout()
    else:
        layout = QHBoxLayout()
    layout.addWidget(label)
    layout.addWidget(combobox)
    return combobox, label, layout