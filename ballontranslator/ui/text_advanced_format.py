import math
from typing import Any, Callable

from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QPushButton, QGroupBox, QLabel, QHBoxLayout, QLineEdit, QWidget
from qtpy.QtCore import QEvent, Signal, Qt
from qtpy.QtGui import QKeyEvent

from .custom_widget import SmallColorPickerLabel, SmallParamLabel, PanelArea, SmallSizeControlLabel, SmallSizeComboBox, SmallParamLabel, SmallSizeComboBox, SmallComboBox, TextCheckerLabel
from ballontranslator.utils.fontformat import FontFormat


class TransformDragLabel(SmallSizeControlLabel):
    drag_started = Signal()
    drag_canceled = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self.drag_started.emit()
        return super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape and self.mouse_pressed:
            self.mouse_pressed = False
            self.drag_canceled.emit()
            event.accept()
            return
        return super().keyPressEvent(event)


class CommittedTransformControl(QWidget):
    """One Advanced-panel-only committed numeric transform editor."""

    IDLE = 'IDLE'
    PENDING_TEXT = 'PENDING_TEXT'
    DRAG_PREVIEW = 'DRAG_PREVIEW'

    commit_requested = Signal(str, float)
    preview_requested = Signal(str, float)
    drag_commit_requested = Signal(str, float)
    preview_canceled = Signal(str)

    def __init__(self, title: str, param_name: str, percentage: bool, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.percentage = percentage
        self.state = self.IDLE
        self._model_value = None
        self._drag_delta = 0.0

        self.label = TransformDragLabel(
            self,
            direction=0,
            text=title,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.editor = QLineEdit(self)
        self.editor.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.editor.setMinimumWidth(72)
        self.editor.textEdited.connect(self._on_text_edited)
        self.editor.returnPressed.connect(self.commit_pending)
        self.editor.installEventFilter(self)

        self.label.drag_started.connect(self._start_drag)
        self.label.size_ctrl_changed.connect(self._move_drag)
        self.label.btn_released.connect(self._finish_drag)
        self.label.drag_canceled.connect(self.cancel_preview)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.editor)

    def _canonical_to_display(self, value: float) -> float:
        return value * 100.0 if self.percentage else value

    def _display_to_canonical(self, value: float) -> float:
        return value / 100.0 if self.percentage else value

    def _format(self, canonical_value: float) -> str:
        if self.percentage:
            return f'{self._canonical_to_display(canonical_value):.1f}%'
        return f'{canonical_value:.1f}\N{DEGREE SIGN}'

    def _parse(self, text: str) -> float:
        text = text.strip()
        if self.percentage:
            if text.endswith('%'):
                text = text[:-1].strip()
            value = float(text)
            if not math.isfinite(value) or not 10.0 <= value <= 400.0:
                raise ValueError
            return self._display_to_canonical(value)
        if text.endswith('\N{DEGREE SIGN}'):
            text = text[:-1].strip()
        value = float(text)
        if not math.isfinite(value) or not -45.0 <= value <= 45.0:
            raise ValueError
        return value

    def _restore_display(self):
        self.editor.setText(
            '\N{EM DASH}'
            if self._model_value is None
            else self._format(self._model_value)
        )

    def set_model_value(self, canonical_value):
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._model_value = canonical_value
        self._restore_display()

    def _on_text_edited(self):
        if self.state != self.DRAG_PREVIEW:
            self.state = self.PENDING_TEXT

    def commit_pending(self):
        if self.state != self.PENDING_TEXT:
            return False
        try:
            canonical_value = self._parse(self.editor.text())
        except (TypeError, ValueError):
            self.state = self.IDLE
            self._restore_display()
            return False
        self.state = self.IDLE
        self._model_value = canonical_value
        self._restore_display()
        self.commit_requested.emit(self.param_name, canonical_value)
        return True

    def cancel_pending(self):
        if self.state == self.PENDING_TEXT:
            self.state = self.IDLE
            self._restore_display()

    def eventFilter(self, watched, event):
        if watched is self.editor:
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self.cancel_pending()
                event.accept()
                return True
            if event.type() == QEvent.Type.FocusOut:
                self.commit_pending()
        return super().eventFilter(watched, event)

    def _start_drag(self):
        self.commit_pending()
        self.state = self.DRAG_PREVIEW
        self._drag_delta = 0.0

    def _move_drag(self, delta: int):
        if self.state != self.DRAG_PREVIEW:
            self._start_drag()
        self._drag_delta += float(delta)
        if self._model_value is None:
            suffix = '%' if self.percentage else '\N{DEGREE SIGN}'
            self.editor.setText(
                f'\N{GREEK CAPITAL LETTER DELTA} {self._drag_delta:+.1f}{suffix}'
            )
        else:
            canonical_delta = self._display_to_canonical(self._drag_delta)
            minimum, maximum = ((0.1, 4.0) if self.percentage else (-45.0, 45.0))
            preview_value = min(max(self._model_value + canonical_delta, minimum), maximum)
            self.editor.setText(self._format(preview_value))
        self.preview_requested.emit(
            self.param_name,
            self._display_to_canonical(self._drag_delta),
        )

    def _finish_drag(self):
        if self.state != self.DRAG_PREVIEW:
            return
        delta = self._drag_delta
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._restore_display()
        if delta == 0.0:
            self.preview_canceled.emit(self.param_name)
        else:
            self.drag_commit_requested.emit(
                self.param_name,
                self._display_to_canonical(delta),
            )

    def cancel_preview(self):
        if self.state != self.DRAG_PREVIEW:
            return
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._restore_display()
        self.preview_canceled.emit(self.param_name)


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
    transform_commit_requested = Signal(str, float)
    transform_preview_requested = Signal(str, float)
    transform_drag_commit_requested = Signal(str, float)
    transform_preview_canceled = Signal(str)

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str, on_format_changed: Callable):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None
        self.on_format_changed = on_format_changed

        self.horizontal_scale_control = CommittedTransformControl(
            self.tr('Horizontal Scale'), 'horizontal_scale', True, self
        )
        self.vertical_scale_control = CommittedTransformControl(
            self.tr('Vertical Scale'), 'vertical_scale', True, self
        )
        self.slant_angle_control = CommittedTransformControl(
            self.tr('Slant Angle'), 'slant_angle', False, self
        )
        self.transform_controls = {
            'horizontal_scale': self.horizontal_scale_control,
            'vertical_scale': self.vertical_scale_control,
            'slant_angle': self.slant_angle_control,
        }
        for control in self.transform_controls.values():
            control.commit_requested.connect(self.transform_commit_requested.emit)
            control.preview_requested.connect(self.transform_preview_requested.emit)
            control.drag_commit_requested.connect(
                self.transform_drag_commit_requested.emit
            )
            control.preview_canceled.connect(self.transform_preview_canceled.emit)

        transform_layout = QHBoxLayout()
        transform_layout.addWidget(self.horizontal_scale_control)
        transform_layout.addWidget(self.vertical_scale_control)
        transform_layout.addWidget(self.slant_angle_control)

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

        hlayout = QHBoxLayout()
        hlayout.addLayout(linespacing_type_layout)
        hlayout.addLayout(opacity_layout)
        vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout)
        vlayout.addLayout(transform_layout)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.addWidget(self.shadow_group)
        vlayout.addWidget(self.gradient_group)

        self.setContentLayout(vlayout)
        self.vlayout = vlayout

    def adjuset_size(self):
        TEXT_ADVANCED_PANEL_MAXH = 300
        self.setFixedHeight(min(TEXT_ADVANCED_PANEL_MAXH, self.scrollContent.height()))

    def on_linespacing_type_changed(self):
        self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex())

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)
        for name, control in self.transform_controls.items():
            control.set_model_value(getattr(font_format, name))

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

    def set_transform_items(self, items):
        for name, control in self.transform_controls.items():
            values = [getattr(item.blk.fontformat, name) for item in items]
            common = values[0] if values and all(value == values[0] for value in values) else None
            control.set_model_value(common)

    def finish_pending_transform_edits(self):
        for control in self.transform_controls.values():
            control.commit_pending()

    def cancel_transform_previews(self):
        for control in self.transform_controls.values():
            control.cancel_preview()
