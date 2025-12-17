from typing import Any, Callable

from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QPushButton, QGroupBox, QLabel, QHBoxLayout
from qtpy.QtCore import Signal, Qt

from .custom_widget import SmallColorPickerLabel, SmallParamLabel, PanelArea, SmallSizeControlLabel, SmallSizeComboBox, SmallParamLabel, SmallSizeComboBox, SmallComboBox, TextCheckerLabel
from utils.fontformat import FontFormat
from utils.config import pcfg


class TextShadowGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None, title=None):
        super().__init__(title=title)
        self.on_param_changed = on_param_changed

        self.xoffset_box = SmallSizeComboBox([-2, 2], 'shadow_xoffset', self)
        self.xoffset_box.setToolTip(self.tr("Set X offset"))
        self.xoffset_box.param_changed.connect(self.on_offset_changed)
        self.xoffset_label = SmallSizeControlLabel(self, direction=1, text='X', alignment=Qt.AlignmentFlag.AlignCenter)
        self.xoffset_label.size_ctrl_changed.connect(self.xoffset_box.changeByDelta)
        self.xoffset_label.btn_released.connect(self.on_offset_changed)
        xoffset_layout = QHBoxLayout()
        xoffset_layout.addWidget(self.xoffset_label)
        xoffset_layout.addWidget(self.xoffset_box)

        self.yoffset_box = SmallSizeComboBox([-2, 2], 'shadow_yoffset', self)
        self.yoffset_box.setToolTip(self.tr("Set Y offset"))
        self.yoffset_box.param_changed.connect(self.on_offset_changed)
        self.yoffset_label = SmallSizeControlLabel(self, direction=1, text='Y', alignment=Qt.AlignmentFlag.AlignCenter)
        self.yoffset_label.size_ctrl_changed.connect(self.yoffset_box.changeByDelta)
        self.yoffset_label.btn_released.connect(self.on_offset_changed)
        yoffset_layout = QHBoxLayout()
        yoffset_layout.addWidget(self.yoffset_label)
        yoffset_layout.addWidget(self.yoffset_box)

        self.color_label = SmallColorPickerLabel(self, param_name='shadow_color')

        self.strength_box = SmallSizeComboBox([0, 3], 'shadow_strength', self)
        self.strength_box.setToolTip(self.tr("Set Shadow Strength"))
        self.strength_box.param_changed.connect(self.on_param_changed)
        self.strength_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Strength'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.strength_label.size_ctrl_changed.connect(lambda x : self.strength_box.changeByDelta(x, multiplier=0.03))
        self.strength_label.btn_released.connect(lambda : self.on_param_changed('shadow_strength', self.strength_box.value()))
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_box)

        self.radius_box = SmallSizeComboBox([0, 2], 'shadow_radius', self)
        self.radius_box.setToolTip(self.tr("Set Shadow Radius"))
        self.radius_box.param_changed.connect(self.on_param_changed)
        self.radius_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Radius'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.radius_label.size_ctrl_changed.connect(self.radius_box.changeByDelta)
        self.radius_label.btn_released.connect(lambda : self.on_param_changed('shadow_radius', self.radius_box.value()))
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(self.radius_label)
        radius_layout.addWidget(self.radius_box)

        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(self.color_label)
        hlayout2.addLayout(strength_layout)
        hlayout2.addLayout(radius_layout)

        yoffset_layout = QHBoxLayout()
        yoffset_layout.addWidget(self.yoffset_label)
        yoffset_layout.addWidget(self.yoffset_box)

        offset_label = SmallParamLabel(self.tr('Offset'))
        offset_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        offset_row = QHBoxLayout()
        offset_row.addWidget(offset_label)
        offset_row.addLayout(xoffset_layout)
        offset_row.addLayout(yoffset_layout)

        layout = QVBoxLayout(self)
        layout.addLayout(offset_row)
        layout.addLayout(hlayout2)

    def on_offset_changed(self, *args, **kwargs):
        self.on_param_changed('shadow_offset', [self.xoffset_box.value(), self.yoffset_box.value()])


class TextGradientGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None):
        super().__init__()
        self.setTitle(self.tr('Gradient'))
        self.on_param_changed = on_param_changed

        self.start_picker = SmallColorPickerLabel(self, param_name='gradient_start_color')
        start_picker_label = SmallParamLabel(self.tr('Start Color'), alignment=Qt.AlignmentFlag.AlignCenter)
        start_picker_layout = QHBoxLayout()
        start_picker_layout.addWidget(start_picker_label)
        start_picker_layout.addWidget(self.start_picker)

        self.end_picker = SmallColorPickerLabel(self, param_name='gradient_end_color')
        end_picker_label = SmallParamLabel(self.tr('End Color'), alignment=Qt.AlignmentFlag.AlignCenter)
        end_picker_layout = QHBoxLayout()
        end_picker_layout.addWidget(end_picker_label)
        end_picker_layout.addWidget(self.end_picker)

        self.enable_checker = TextCheckerLabel(self.tr('Enable'))
        self.enable_checker.checkStateChanged.connect(lambda checked: self.on_param_changed('gradient_enabled', checked))

        self.angle_box = SmallSizeComboBox([0, 359], 'gradient_angle', self)
        self.angle_box.setToolTip(self.tr("Set Gradient Angle"))
        self.angle_box.param_changed.connect(self.on_param_changed)
        self.angle_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Angle'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.angle_label.size_ctrl_changed.connect(lambda x : self.angle_box.changeByDelta(x, multiplier=1))
        self.angle_label.btn_released.connect(lambda : self.on_param_changed('gradient_angle', self.angle_box.value()))
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(self.angle_label)
        angle_layout.addWidget(self.angle_box)

        self.size_box = SmallSizeComboBox([0.5, 2], 'gradient_size', self)
        self.size_box.setToolTip(self.tr("Set Gradient Size"))
        self.size_box.param_changed.connect(self.on_param_changed)
        self.size_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Size'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.size_label.size_ctrl_changed.connect(lambda x : self.size_box.changeByDelta(x, multiplier=0.02))
        self.size_label.btn_released.connect(lambda : self.on_param_changed('gradient_size', self.size_box.value()))
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_box)

        hlayout1 = QHBoxLayout()
        hlayout1.addLayout(start_picker_layout)
        hlayout1.addLayout(end_picker_layout)
        hlayout1.addWidget(self.enable_checker)
        # hlayout1.addStretch(-1)

        hlayout2 = QHBoxLayout()
        hlayout2.addLayout(angle_layout)
        hlayout2.addLayout(size_layout)

        layout = QVBoxLayout(self)
        layout.addLayout(hlayout1)
        layout.addLayout(hlayout2)


class TextAdvancedFormatPanel(PanelArea):

    param_changed = Signal(str, object)
    warp_edit_toggled = Signal(bool)
    text_eraser_toggled = Signal(bool)
    clear_text_mask_clicked = Signal()
    warp_preset_clicked = Signal(str)
    reset_angle_clicked = Signal()
    squeeze_clicked = Signal()

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str, on_format_changed: Callable):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None
        self.on_format_changed = on_format_changed

        self.linespacing_type_combobox = SmallComboBox(
            parent=self,
            options=[
                self.tr("Proportional"),
                self.tr("Distance")
            ]
        )
        self.linespacing_type_combobox.activated.connect(self.on_linespacing_type_changed)
        linespacing_type_label = SmallParamLabel(self.tr('Line Spacing Type'))
        linespacing_type_layout = QHBoxLayout()
        linespacing_type_layout.addWidget(linespacing_type_label)
        linespacing_type_layout.addWidget(self.linespacing_type_combobox)

        self.opacity_box = SmallSizeComboBox([0, 1], 'opacity', self, init_value=1.)
        self.opacity_box.setToolTip(self.tr("Set Text Opacity"))
        self.opacity_box.param_changed.connect(self.on_format_changed)
        self.opacity_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Opacity'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.opacity_label.size_ctrl_changed.connect(self.opacity_box.changeByDelta)
        self.opacity_label.btn_released.connect(lambda : self.on_format_changed('opacity', self.opacity_box.value()))
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_box)

        # self.tate_chu_yoko_checker = QFontChecker()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.scrollContent.after_resized.connect(self.adjuset_size)

        self.shadow_group = TextShadowGroup(self.on_format_changed, title=self.tr('Shadow'))
        self.shadow_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.gradient_group = TextGradientGroup(self.on_format_changed)
        self.gradient_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.tools_group = QGroupBox(self.tr('Text Tools'))
        self.free_transform_toggle = TextCheckerLabel(self.tr('Free Transform'))
        self.free_transform_toggle.checkStateChanged.connect(self.warp_edit_toggled.emit)
        self.text_eraser_toggle = TextCheckerLabel(self.tr('Text Eraser'))
        self.text_eraser_toggle.checkStateChanged.connect(self.text_eraser_toggled.emit)

        self.clear_text_mask_btn = QPushButton(self.tr('Clear Text Mask'))
        self.clear_text_mask_btn.clicked.connect(self.clear_text_mask_clicked.emit)
        self.reset_angle_btn = QPushButton(self.tr('Reset Angle'))
        self.reset_angle_btn.clicked.connect(self.reset_angle_clicked.emit)
        self.squeeze_btn = QPushButton(self.tr('Squeeze'))
        self.squeeze_btn.clicked.connect(self.squeeze_clicked.emit)

        self.warp_reset_btn = QPushButton(self.tr('Reset Warp'))
        self.warp_reset_btn.clicked.connect(lambda: self.warp_preset_clicked.emit('reset'))
        self.warp_arc_up_btn = QPushButton(self.tr('Arc Up'))
        self.warp_arc_up_btn.clicked.connect(lambda: self.warp_preset_clicked.emit('arc_up'))
        self.warp_arc_down_btn = QPushButton(self.tr('Arc Down'))
        self.warp_arc_down_btn.clicked.connect(lambda: self.warp_preset_clicked.emit('arc_down'))
        self.warp_arch_btn = QPushButton(self.tr('Arch'))
        self.warp_arch_btn.clicked.connect(lambda: self.warp_preset_clicked.emit('arch'))
        self.warp_flag_btn = QPushButton(self.tr('Flag'))
        self.warp_flag_btn.clicked.connect(lambda: self.warp_preset_clicked.emit('flag'))

        toggles_layout = QHBoxLayout()
        toggles_layout.addWidget(self.free_transform_toggle)
        toggles_layout.addWidget(self.text_eraser_toggle)

        transform_layout = QHBoxLayout()
        transform_layout.addWidget(self.reset_angle_btn)
        transform_layout.addWidget(self.squeeze_btn)
        transform_layout.addWidget(self.clear_text_mask_btn)

        warp_layout = QHBoxLayout()
        warp_layout.addWidget(self.warp_reset_btn)
        warp_layout.addWidget(self.warp_arc_up_btn)
        warp_layout.addWidget(self.warp_arc_down_btn)
        warp_layout.addWidget(self.warp_arch_btn)
        warp_layout.addWidget(self.warp_flag_btn)

        tools_layout = QVBoxLayout(self.tools_group)
        tools_layout.addLayout(toggles_layout)
        tools_layout.addLayout(transform_layout)
        tools_layout.addLayout(warp_layout)
        self.tools_group.setVisible(bool(getattr(pcfg, 'move_text_tools_to_sidebar', False)))

        hlayout = QHBoxLayout()
        hlayout.addLayout(linespacing_type_layout)
        hlayout.addLayout(opacity_layout)
        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.addWidget(self.shadow_group)
        vlayout.addWidget(self.gradient_group)
        vlayout.addWidget(self.tools_group)

        self.setContentLayout(vlayout)
        self.vlayout = vlayout

    def set_text_tools_visible(self, visible: bool):
        self.tools_group.setVisible(bool(visible))
        self.adjuset_size()

    def set_warp_edit_mode_checked(self, checked: bool):
        self.free_transform_toggle.setCheckState(bool(checked))

    def set_text_eraser_checked(self, checked: bool):
        self.text_eraser_toggle.setCheckState(bool(checked))

    def adjuset_size(self):
        TEXT_ADVANCED_PANEL_MAXH = 300
        self.setFixedHeight(min(TEXT_ADVANCED_PANEL_MAXH, self.scrollContent.height()))

    def on_linespacing_type_changed(self):
        self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex())

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)

        self.shadow_group.color_label.setPickerColor(font_format.shadow_color)
        self.shadow_group.strength_box.setValue(font_format.shadow_strength)
        self.shadow_group.radius_box.setValue(font_format.shadow_radius)
        self.shadow_group.xoffset_box.setValue(font_format.shadow_offset[0])
        self.shadow_group.yoffset_box.setValue(font_format.shadow_offset[1])

        self.gradient_group.size_box.setValue(font_format.gradient_size)
        self.gradient_group.angle_box.setValue(font_format.gradient_angle)
        self.gradient_group.enable_checker.setCheckState(font_format.gradient_enabled)
        self.gradient_group.start_picker.setPickerColor(font_format.gradient_start_color)
        self.gradient_group.end_picker.setPickerColor(font_format.gradient_end_color)
        # self.tate_chu_yoko_checker.setChecked(font_format.font)
