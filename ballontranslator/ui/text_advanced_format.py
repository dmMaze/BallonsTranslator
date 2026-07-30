from typing import Callable

from qtpy.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QStyle,
    QToolButton,
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
from .text_transform_controls import (
    CommittedTransformControl,
    TransformParameterPanel,
)
from .text_transform_variants import (
    GLYPH_SLANT_CONTROL,
    TEXT_TRANSFORM_VARIANTS,
)
from ballontranslator.utils.fontformat import FontFormat, TextTransformState

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
    transform_commit_requested = Signal(int, str, float)
    transform_preview_requested = Signal(int, str, float)
    transform_drag_commit_requested = Signal(int, str, float)
    transform_preview_canceled = Signal(int, str)
    transform_add_requested = Signal(str)
    transform_remove_requested = Signal(int)
    transform_move_requested = Signal(int, int)

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
        glyph = GLYPH_SLANT_CONTROL
        self.glyph_slant_control = CommittedTransformControl(
            glyph.label(),
            glyph.attribute_name,
            glyph.factor,
            glyph.minimum,
            glyph.maximum,
            glyph.suffix,
            1.0,
            self.transform_group,
        )
        setattr(self, glyph.name, self.glyph_slant_control)
        self.glyph_slant_control.commit_requested.connect(
            lambda name, value:
            self.transform_commit_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.preview_requested.connect(
            lambda name, value:
            self.transform_preview_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.drag_commit_requested.connect(
            lambda name, value:
            self.transform_drag_commit_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.preview_canceled.connect(
            lambda name: self.transform_preview_canceled.emit(-1, name)
        )

        self.add_transform_button = QToolButton(self.transform_group)
        self.add_transform_button.setObjectName('AddTextTransformButton')
        self.add_transform_button.setText(self.tr('Add Transform'))
        self.add_transform_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.add_transform_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        add_menu = QMenu(self.add_transform_button)
        for variant in self.transform_variants:
            action = add_menu.addAction(variant.label())
            action.triggered.connect(
                lambda _checked=False, transform_type=variant.transform_type:
                self.transform_add_requested.emit(transform_type)
            )
        self.add_transform_button.setMenu(add_menu)

        self.transform_mixed_label = QLabel(
            self.tr('Mixed'), self.transform_group
        )
        self.transform_mixed_label.setObjectName('TextTransformMixedLabel')
        self.transform_mixed_label.setVisible(False)

        self.transform_rows = QWidget(self.transform_group)
        self.transform_rows.setObjectName('TextTransformRows')
        self.transform_rows_layout = QVBoxLayout(self.transform_rows)
        self.transform_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.transform_rows_layout.setSpacing(6)
        self.transform_panels = []
        self._transform_panel_types = ()

        self.transform_layout = QVBoxLayout(self.transform_group)
        self.transform_layout.setContentsMargins(8, 8, 8, 8)
        self.transform_layout.setSpacing(6)
        self.transform_layout.addWidget(self.glyph_slant_control)
        self.transform_layout.addWidget(
            self.add_transform_button, alignment=Qt.AlignmentFlag.AlignLeft
        )
        self.transform_layout.addWidget(self.transform_mixed_label)
        self.transform_layout.addWidget(self.transform_rows)

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

    def _clear_transform_panels(self):
        for panel in self.transform_panels:
            self.transform_rows_layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self.transform_panels = []
        self._transform_panel_types = ()

    def _rebuild_transform_panels(self, transform_types):
        transform_types = tuple(transform_types)
        if transform_types == self._transform_panel_types:
            return
        self._clear_transform_panels()
        variants = {
            variant.transform_type: variant
            for variant in self.transform_variants
        }
        for index, transform_type in enumerate(transform_types):
            panel = TransformParameterPanel(
                index, variants[transform_type], self.transform_rows
            )
            panel.commit_requested.connect(
                self.transform_commit_requested.emit
            )
            panel.preview_requested.connect(
                self.transform_preview_requested.emit
            )
            panel.drag_commit_requested.connect(
                self.transform_drag_commit_requested.emit
            )
            panel.preview_canceled.connect(
                self.transform_preview_canceled.emit
            )
            panel.remove_requested.connect(
                self.transform_remove_requested.emit
            )
            panel.move_requested.connect(self.transform_move_requested.emit)
            self.transform_rows_layout.addWidget(panel)
            self.transform_panels.append(panel)
        self._transform_panel_types = transform_types
        count = len(self.transform_panels)
        for index, panel in enumerate(self.transform_panels):
            panel.set_index(index)
            panel.set_move_enabled(index > 0, index + 1 < count)
        self.transform_group.updateGeometry()
        self._schedule_geometry_update()

    def _set_transform_states(self, states):
        states = [
            state
            if isinstance(state, TextTransformState)
            else TextTransformState(
                state.text_transform, state.glyph_slant_angle
            )
            for state in states
        ]
        glyph_values = [state.glyph_slant_angle for state in states]
        common_glyph = (
            glyph_values[0]
            if glyph_values
            and all(value == glyph_values[0] for value in glyph_values)
            else None
        )
        self.glyph_slant_control.set_model_value(common_glyph)

        sequences = [
            tuple(transform.transform_type for transform in state.stack)
            for state in states
        ]
        common_sequence = (
            sequences[0]
            if sequences
            and all(sequence == sequences[0] for sequence in sequences)
            else None
        )
        mixed = common_sequence is None
        self.transform_mixed_label.setVisible(mixed)
        self.transform_rows.setVisible(not mixed)
        if mixed:
            self._rebuild_transform_panels(())
            return
        self._rebuild_transform_panels(common_sequence)
        for index, panel in enumerate(self.transform_panels):
            panel.set_values([state.stack[index] for state in states])

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)
        self._set_transform_states([font_format])

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
        self._set_transform_states(
            [
                TextTransformState(
                    item.blk.fontformat.text_transform,
                    item.blk.fontformat.glyph_slant_angle,
                )
                for item in items
            ]
        )

    def set_transform(self, state):
        self._set_transform_states([state])

    def iter_transform_controls(self):
        yield self.glyph_slant_control
        for panel in self.transform_panels:
            yield from panel.iter_controls()

    def cancel_pending_transform_edits(self):
        for control in self.iter_transform_controls():
            control.cancel_pending()

    def cancel_transform_previews(self):
        for control in self.iter_transform_controls():
            control.cancel_preview()

    def finish_pending_transform_edits(self):
        for control in self.iter_transform_controls():
            control.commit_pending()
