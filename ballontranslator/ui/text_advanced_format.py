from typing import Callable

from qtpy.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import QEvent, QPoint, QRect, QSize, QTimer, Signal, Qt

from .custom_widget import (
    PanelArea,
    SmallColorPickerLabel,
    SmallComboBox,
    SmallParamLabel,
    SmallSizeComboBox,
    SmallSizeControlLabel,
    TextCheckerLabel,
)
from .custom_widget.scrollbar import ScrollBar
from .adaptive_wrap_layout import AdaptiveWrapLayout
from .text_transform_controls import CommittedTransformControl
from .text_transform_variants import TEXT_TRANSFORM_VARIANTS
from ballontranslator.utils.fontformat import FontFormat

def _word_wrap_label(label: QLabel):
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return label


def _atomic_unit(parent: QWidget, *widgets: QWidget):
    unit = QWidget(parent)
    unit.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    layout = QHBoxLayout(unit)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return unit


class TextShadowGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None, title=None):
        super().__init__(title=title)
        self.on_param_changed = on_param_changed
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.xoffset_box = SmallSizeComboBox([-2, 2], 'shadow_xoffset', self)
        self.xoffset_box.setToolTip(self.tr("Set X offset"))
        self.xoffset_box.param_changed.connect(self.on_offset_changed)
        self.xoffset_label = SmallSizeControlLabel(
            self,
            direction=1,
            text='X',
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.xoffset_label.size_ctrl_changed.connect(self.xoffset_box.changeByDelta)
        self.xoffset_label.btn_released.connect(self.on_offset_changed)

        self.yoffset_box = SmallSizeComboBox([-2, 2], 'shadow_yoffset', self)
        self.yoffset_box.setToolTip(self.tr("Set Y offset"))
        self.yoffset_box.param_changed.connect(self.on_offset_changed)
        self.yoffset_label = SmallSizeControlLabel(
            self,
            direction=1,
            text='Y',
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self.yoffset_label.size_ctrl_changed.connect(self.yoffset_box.changeByDelta)
        self.yoffset_label.btn_released.connect(self.on_offset_changed)

        self.color_label = SmallColorPickerLabel(self, param_name='shadow_color')
        self.color_name_label = _word_wrap_label(
            SmallParamLabel(self.tr('Color'), parent=self)
        )

        self.strength_box = SmallSizeComboBox([0, 3], 'shadow_strength', self)
        self.strength_box.setToolTip(self.tr("Set Shadow Strength"))
        self.strength_box.param_changed.connect(self.on_param_changed)
        self.strength_label = SmallSizeControlLabel(
            self,
            direction=1,
            text=self.tr('Strength'),
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.strength_label)
        self.strength_label.size_ctrl_changed.connect(
            lambda x: self.strength_box.changeByDelta(x, multiplier=0.03)
        )
        self.strength_label.btn_released.connect(
            lambda: self.on_param_changed(
                'shadow_strength', self.strength_box.value()
            )
        )

        self.radius_box = SmallSizeComboBox([0, 2], 'shadow_radius', self)
        self.radius_box.setToolTip(self.tr("Set Shadow Radius"))
        self.radius_box.param_changed.connect(self.on_param_changed)
        self.radius_label = SmallSizeControlLabel(
            self,
            direction=1,
            text=self.tr('Radius'),
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.radius_label)
        self.radius_label.size_ctrl_changed.connect(self.radius_box.changeByDelta)
        self.radius_label.btn_released.connect(
            lambda: self.on_param_changed('shadow_radius', self.radius_box.value())
        )

        self.offset_label = _word_wrap_label(
            SmallParamLabel(self.tr('Offset'), parent=self)
        )
        self.offset_unit = _atomic_unit(
            self,
            self.offset_label,
            self.xoffset_label,
            self.xoffset_box,
            self.yoffset_label,
            self.yoffset_box,
        )
        self.color_unit = _atomic_unit(
            self, self.color_name_label, self.color_label
        )
        self.strength_unit = _atomic_unit(
            self, self.strength_label, self.strength_box
        )
        self.radius_unit = _atomic_unit(
            self, self.radius_label, self.radius_box
        )
        self.atomic_units = (
            self.offset_unit,
            self.color_unit,
            self.strength_unit,
            self.radius_unit,
        )

        self.adaptive_layout = AdaptiveWrapLayout(self)
        for unit in self.atomic_units:
            self.adaptive_layout.addWidget(unit)

    def on_offset_changed(self, *args, **kwargs):
        self.on_param_changed(
            'shadow_offset',
            [self.xoffset_box.value(), self.yoffset_box.value()],
        )


class TextGradientGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None):
        super().__init__()
        self.setTitle(self.tr('Gradient'))
        self.on_param_changed = on_param_changed
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.start_picker = SmallColorPickerLabel(self, param_name='gradient_start_color')
        self.start_picker_label = _word_wrap_label(
            SmallParamLabel(
                self.tr('Start Color'),
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                parent=self,
            )
        )

        self.end_picker = SmallColorPickerLabel(self, param_name='gradient_end_color')
        self.end_picker_label = _word_wrap_label(
            SmallParamLabel(
                self.tr('End Color'),
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                parent=self,
            )
        )

        self.enable_checker = TextCheckerLabel(self.tr('Enable'), parent=self)
        self.enable_checker.setWordWrap(True)
        self.enable_checker.checkStateChanged.connect(
            lambda checked: self.on_param_changed('gradient_enabled', checked)
        )

        self.angle_box = SmallSizeComboBox([0, 359], 'gradient_angle', self)
        self.angle_box.setToolTip(self.tr("Set Gradient Angle"))
        self.angle_box.param_changed.connect(self.on_param_changed)
        self.angle_label = SmallSizeControlLabel(
            self,
            direction=1,
            text=self.tr('Angle'),
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.angle_label)
        self.angle_label.size_ctrl_changed.connect(
            lambda x: self.angle_box.changeByDelta(x, multiplier=1)
        )
        self.angle_label.btn_released.connect(
            lambda: self.on_param_changed('gradient_angle', self.angle_box.value())
        )

        self.size_box = SmallSizeComboBox([0.5, 2], 'gradient_size', self)
        self.size_box.setToolTip(self.tr("Set Gradient Size"))
        self.size_box.param_changed.connect(self.on_param_changed)
        self.size_label = SmallSizeControlLabel(
            self,
            direction=1,
            text=self.tr('Size'),
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.size_label)
        self.size_label.size_ctrl_changed.connect(
            lambda x: self.size_box.changeByDelta(x, multiplier=0.02)
        )
        self.size_label.btn_released.connect(
            lambda: self.on_param_changed('gradient_size', self.size_box.value())
        )

        self.start_color_unit = _atomic_unit(
            self, self.start_picker_label, self.start_picker
        )
        self.end_color_unit = _atomic_unit(
            self, self.end_picker_label, self.end_picker
        )
        self.enable_unit = _atomic_unit(self, self.enable_checker)
        self.angle_unit = _atomic_unit(self, self.angle_label, self.angle_box)
        self.size_unit = _atomic_unit(self, self.size_label, self.size_box)
        self.atomic_units = (
            self.start_color_unit,
            self.end_color_unit,
            self.enable_unit,
            self.angle_unit,
            self.size_unit,
        )

        self.adaptive_layout = AdaptiveWrapLayout(self)
        for unit in self.atomic_units:
            self.adaptive_layout.addWidget(unit)


class TextAdvancedFormatPanel(PanelArea):

    param_changed = Signal(str, object)
    transform_commit_requested = Signal(str, float)
    transform_preview_requested = Signal(str, float)
    transform_drag_commit_requested = Signal(str, float)
    transform_preview_canceled = Signal(str)
    transform_type_change_requested = Signal(str)

    def __init__(
        self,
        panel_name: str,
        config_name: str,
        config_expand_name: str,
        on_format_changed: Callable,
    ):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None
        self.on_format_changed = on_format_changed
        self._last_content_width = None
        self._last_content_height = None
        self._last_height_cap = None
        self._updating_responsive_geometry = False

        # PanelArea installs overlay scrollbars and forces the native bars off.
        # This panel needs a width-consuming native vertical bar so its content
        # width and height-for-width calculation agree exactly.
        self._inherited_overlay_scrollbars = tuple(self.findChildren(ScrollBar))
        for scrollbar in self._inherited_overlay_scrollbars:
            scrollbar.setForceHidden(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.scrollContent.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self._geometry_timer = QTimer(self)
        self._geometry_timer.setSingleShot(True)
        self._geometry_timer.timeout.connect(self._update_responsive_geometry)
        self._responsive_event_targets = (self.scrollContent, self.viewport())
        for target in self._responsive_event_targets:
            target.installEventFilter(self)
        self.scrollContent.after_resized.connect(
            self._schedule_geometry_update
        )

        self.transform_group = QGroupBox(
            self.tr('Transform'), self.scrollContent
        )
        self.transform_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.transform_variants = TEXT_TRANSFORM_VARIANTS
        self.transform_types = tuple(
            variant.transform_type for variant in self.transform_variants
        )
        self.transform_effect_selector = SmallComboBox(
            parent=self.transform_group,
            options=[variant.label() for variant in self.transform_variants],
        )
        self.transform_effect_selector.activated.connect(
            self._on_transform_type_activated
        )
        self.transform_effect_label = _word_wrap_label(
            SmallParamLabel(self.tr('Effect'), parent=self.transform_group)
        )
        self.transform_effect_unit = _atomic_unit(
            self.transform_group,
            self.transform_effect_label,
            self.transform_effect_selector,
        )

        transform_controls = {}
        self._transform_control_names_by_type = {}
        for variant in self.transform_variants:
            control_names = []
            for spec in variant.controls:
                control = transform_controls.get(spec.attribute_name)
                if control is None:
                    control = CommittedTransformControl(
                        spec.label(),
                        spec.attribute_name,
                        spec.factor,
                        spec.minimum,
                        spec.maximum,
                        spec.suffix,
                        1.0,
                        self.transform_group,
                    )
                    setattr(self, spec.name, control)
                    transform_controls[spec.attribute_name] = control
                control_names.append(spec.attribute_name)
            self._transform_control_names_by_type[variant.transform_type] = frozenset(
                control_names
            )
        self.transform_controls = transform_controls
        for control in self.transform_controls.values():
            control.commit_requested.connect(self.transform_commit_requested.emit)
            control.preview_requested.connect(self.transform_preview_requested.emit)
            control.drag_commit_requested.connect(
                self.transform_drag_commit_requested.emit
            )
            control.preview_canceled.connect(self.transform_preview_canceled.emit)

        self.transform_layout = AdaptiveWrapLayout(self.transform_group)
        self.transform_layout.addWidget(self.transform_effect_unit)
        for control in self.transform_controls.values():
            self.transform_layout.addWidget(control)
        self._set_transform_controls_visible(None)

        self.top_section = QWidget(self.scrollContent)
        self.top_section.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.linespacing_type_combobox = SmallComboBox(
            parent=self.top_section,
            options=[
                self.tr("Proportional"),
                self.tr("Distance")
            ]
        )
        self.linespacing_type_combobox.activated.connect(
            self.on_linespacing_type_changed
        )
        self.linespacing_type_label = _word_wrap_label(
            SmallParamLabel(
                self.tr('Line Spacing Type'), parent=self.top_section
            )
        )
        self.linespacing_type_unit = _atomic_unit(
            self.top_section,
            self.linespacing_type_label,
            self.linespacing_type_combobox,
        )

        self.opacity_box = SmallSizeComboBox(
            [0, 1], 'opacity', self.top_section, init_value=1.
        )
        self.opacity_box.setToolTip(self.tr("Set Text Opacity"))
        self.opacity_box.param_changed.connect(self.on_format_changed)
        self.opacity_label = SmallSizeControlLabel(
            self.top_section,
            direction=1,
            text=self.tr('Opacity'),
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.opacity_label)
        self.opacity_label.size_ctrl_changed.connect(self.opacity_box.changeByDelta)
        self.opacity_label.btn_released.connect(
            lambda: self.on_format_changed('opacity', self.opacity_box.value())
        )
        self.opacity_unit = _atomic_unit(
            self.top_section, self.opacity_label, self.opacity_box
        )
        self.top_atomic_units = (
            self.linespacing_type_unit,
            self.opacity_unit,
        )
        self.top_layout = AdaptiveWrapLayout(self.top_section)
        for unit in self.top_atomic_units:
            self.top_layout.addWidget(unit)

        self.shadow_group = TextShadowGroup(
            self.on_format_changed, title=self.tr('Shadow')
        )

        self.gradient_group = TextGradientGroup(self.on_format_changed)
        vlayout = QVBoxLayout()
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        # Preserve the current panel's section order.
        vlayout.addWidget(self.top_section)
        vlayout.addWidget(self.transform_group)
        vlayout.addWidget(self.shadow_group)
        vlayout.addWidget(self.gradient_group)

        self.setContentLayout(vlayout)
        self.vlayout = vlayout
        self.setMaximumHeight(self._panel_height_cap())
        self._schedule_geometry_update()

    @staticmethod
    def _event_types(*names):
        event_types = []
        namespace = getattr(QEvent, 'Type', QEvent)
        for name in names:
            event_type = getattr(namespace, name, None)
            if event_type is not None:
                event_types.append(event_type)
        return tuple(event_types)

    def _schedule_geometry_update(self, *_args):
        if (
            not hasattr(self, '_geometry_timer')
            or self._updating_responsive_geometry
            or self._geometry_timer.isActive()
            or self._responsive_geometry_is_current()
        ):
            return
        self._geometry_timer.start(0)

    def _responsive_geometry_is_current(self):
        if (
            self._last_content_width is None
            or self.scrollContent.layout() is None
        ):
            return False
        width = self._effective_content_width()
        return (
            width == self._last_content_width
            and self._content_preferred_height(width)
            == self._last_content_height
            and self._panel_height_cap() == self._last_height_cap
        )

    def eventFilter(self, watched, event):
        if (
            hasattr(self, '_responsive_event_targets')
            and watched in self._responsive_event_targets
        ):
            if event.type() in self._event_types(
                'Resize',
                'Show',
                'LayoutRequest',
                'FontChange',
                'ApplicationFontChange',
                'StyleChange',
                'ScreenChangeInternal',
                'DevicePixelRatioChange',
            ):
                self._schedule_geometry_update()
        return super().eventFilter(watched, event)

    def event(self, event):
        event_type = event.type()
        result = super().event(event)
        if hasattr(self, '_geometry_timer') and event_type in self._event_types(
            'LayoutRequest',
            'FontChange',
            'ApplicationFontChange',
            'StyleChange',
            'ScreenChangeInternal',
            'DevicePixelRatioChange',
        ):
            self._schedule_geometry_update()
        return result

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_geometry_update()

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_geometry_update()

    def _panel_height_cap(self):
        return max(240, 18 * self.fontMetrics().lineSpacing())

    def _scrollbar_extent(self):
        pixel_metrics = getattr(QStyle, 'PixelMetric', QStyle)
        metric = getattr(pixel_metrics, 'PM_ScrollBarExtent')
        return max(0, self.style().pixelMetric(metric))

    def _effective_content_width(self):
        # viewport().width() already excludes the frame and a visible native
        # vertical scrollbar; use an outer-width fallback before first polish.
        width = self.viewport().width()
        if width <= 0:
            width = self.width() - 2 * self.frameWidth()
            if self.verticalScrollBar().isVisible():
                width -= self._scrollbar_extent()
        return max(1, width)

    def _content_preferred_height(self, width):
        layout = self.scrollContent.layout()
        if layout is None:
            return 0
        if layout.hasHeightForWidth():
            return max(0, layout.heightForWidth(width))
        return max(0, layout.sizeHint().height())

    def _minimum_panel_width(self):
        if not hasattr(self, 'top_layout'):
            return super().minimumSizeHint().width()
        atomic_width = max(
            layout.minimumSize().width()
            for layout in (
                self.top_layout,
                self.transform_layout,
                self.shadow_group.adaptive_layout,
                self.gradient_group.adaptive_layout,
            )
        )
        left, _top, right, _bottom = self.vlayout.getContentsMargins()
        return (
            atomic_width
            + left
            + right
            + 2 * self.frameWidth()
            + self._scrollbar_extent()
        )

    def sizeHint(self):
        base_hint = super().sizeHint()
        if not hasattr(self, 'vlayout'):
            return base_hint
        width = self._effective_content_width()
        preferred_height = (
            self._content_preferred_height(width) + 2 * self.frameWidth()
        )
        return QSize(
            max(base_hint.width(), self._minimum_panel_width()),
            min(preferred_height, self._panel_height_cap()),
        )

    def minimumSizeHint(self):
        if not hasattr(self, 'vlayout'):
            return super().minimumSizeHint()
        hint = self.sizeHint()
        minimum_height_cap = max(96, 6 * self.fontMetrics().lineSpacing())
        return QSize(
            self._minimum_panel_width(),
            min(hint.height(), minimum_height_cap),
        )

    def _focus_outside_viewport(self, widget):
        top_left = widget.mapTo(self.viewport(), QPoint(0, 0))
        focus_rect = QRect(top_left, widget.size())
        return not self.viewport().rect().intersects(focus_rect)

    def _update_responsive_geometry(self):
        if self._updating_responsive_geometry:
            return
        self._updating_responsive_geometry = True
        try:
            self._apply_responsive_geometry()
        finally:
            self._updating_responsive_geometry = False

    def _apply_responsive_geometry(self):
        if self.scrollContent.layout() is None:
            return
        scrollbar = self.verticalScrollBar()
        old_scroll_position = scrollbar.value()
        focus_widget = QApplication.focusWidget()
        focus_in_content = (
            focus_widget is not None
            and (
                focus_widget is self.scrollContent
                or self.scrollContent.isAncestorOf(focus_widget)
            )
        )

        content_width = self._effective_content_width()
        preferred_height = self._content_preferred_height(content_width)
        height_cap = self._panel_height_cap()
        width_changed = (
            self._last_content_width is not None
            and content_width != self._last_content_width
        )
        geometry_changed = (
            content_width != self._last_content_width
            or preferred_height != self._last_content_height
        )

        if self.scrollContent.minimumWidth() != content_width:
            self.scrollContent.setMinimumWidth(content_width)
        if self.scrollContent.maximumWidth() != content_width:
            self.scrollContent.setMaximumWidth(content_width)
        if self.scrollContent.minimumHeight() != preferred_height:
            self.scrollContent.setMinimumHeight(preferred_height)
        target_height = max(preferred_height, self.viewport().height())
        if self.scrollContent.size() != QSize(content_width, target_height):
            self.scrollContent.resize(content_width, target_height)
        self.scrollContent.layout().activate()

        if height_cap != self._last_height_cap:
            self.setMaximumHeight(height_cap)
            self._last_height_cap = height_cap
            geometry_changed = True
        self._last_content_width = content_width
        self._last_content_height = preferred_height

        if width_changed:
            scrollbar.setValue(old_scroll_position)
        if (
            width_changed
            and focus_in_content
            and self._focus_outside_viewport(focus_widget)
        ):
            self.ensureWidgetVisible(focus_widget, 0, 0)
        if geometry_changed:
            self.scrollContent.updateGeometry()
            self.updateGeometry()
            self.view_widget.updateGeometry()

    def on_linespacing_type_changed(self):
        self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex())

    def _set_transform_controls_visible(self, transform_type):
        visible_names = self._transform_control_names_by_type.get(
            transform_type, ()
        )
        changed = any(
            (not control.isHidden()) != (name in visible_names)
            for name, control in self.transform_controls.items()
        )
        for name, control in self.transform_controls.items():
            control.setVisible(name in visible_names)
        if changed:
            self.transform_layout.invalidate()
            self.transform_group.updateGeometry()
            self._schedule_geometry_update()

    def _set_transform_values(self, transforms):
        transform_types = [transform.transform_type for transform in transforms]
        common_type = (
            transform_types[0]
            if transform_types
            and all(value == transform_types[0] for value in transform_types)
            else None
        )
        selector_index = (
            self.transform_types.index(common_type)
            if common_type in self.transform_types
            else -1
        )
        self.transform_effect_selector.setCurrentIndex(selector_index)

        visible_names = self._transform_control_names_by_type.get(
            common_type, ()
        )
        self._set_transform_controls_visible(common_type)
        for name, control in self.transform_controls.items():
            if name not in visible_names:
                control.set_model_value(None)
                continue
            values = [getattr(transform, name) for transform in transforms]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common)

    def _on_transform_type_activated(self, index: int):
        if index < 0 or index >= len(self.transform_types):
            return
        for control in self.transform_controls.values():
            control.cancel_pending()
            control.cancel_preview()
        transform_type = self.transform_types[index]
        self._set_transform_controls_visible(transform_type)
        self.transform_type_change_requested.emit(transform_type)

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)
        self._set_transform_values([font_format.text_transform])

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
        self._set_transform_values(
            [item.blk.fontformat.text_transform for item in items]
        )

    def set_transform(self, transform):
        self._set_transform_values([transform])

    def finish_pending_transform_edits(self):
        for control in self.transform_controls.values():
            control.commit_pending()
