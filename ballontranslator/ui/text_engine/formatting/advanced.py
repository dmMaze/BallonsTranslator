from pathlib import Path
from typing import Callable

from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import (
    QEvent,
    QPoint,
    QRect,
    QSignalBlocker,
    QSize,
    QTimer,
    Signal,
    Qt,
)

from ...custom_widget import (
    PanelArea,
    SmallColorPickerLabel,
    SmallComboBox,
    SmallParamLabel,
    SmallSizeComboBox,
    SmallSizeControlLabel,
)
from ...adaptive_wrap_layout import AdaptiveWrapLayout
from ...icon_rendering import render_svg_pixmap
from ballontranslator.utils import shared
from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import FontFormat
from ..annotations import (
    FONT_FEATURES_AVAILABLE,
    LIGATURE_AXIS_VALUES,
    LIGATURE_COMMON,
    LIGATURE_CONTEXTUAL,
    LIGATURE_DISCRETIONARY,
    OLDSTYLE_NUMS,
)

def _word_wrap_label(label: QLabel):
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return label


def _atomic_unit(parent: QWidget, *widgets: QWidget):
    unit = QWidget(parent)
    unit.setObjectName('TextAdvancedFormatUnit')
    unit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    layout = QHBoxLayout(unit)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        if isinstance(widget, QComboBox):
            widget.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                widget.sizePolicy().verticalPolicy(),
            )
        layout.addWidget(widget)
    return unit


def _compact_unit(unit: QWidget, *boxes: QComboBox) -> None:
    unit.setSizePolicy(
        QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
    )
    unit.layout().setSpacing(0)
    for box in boxes:
        box.setFixedWidth(38)


def _adaptive_row(parent: QWidget, *units: QWidget):
    row = QWidget(parent)
    row.setObjectName('TextAdvancedFormatUnit')
    row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    layout = AdaptiveWrapLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for unit in units:
        layout.addWidget(unit)
    return row, layout


def _set_svg_label(
    label: QLabel,
    filename: str,
    width: int = 22,
    height: int = 22,
) -> None:
    path = str(Path(shared.PROGRAM_PATH) / 'resources' / 'icons' / filename)
    label.setFixedSize(width, height)
    label.setPixmap(
        render_svg_pixmap(
            path, width, height, label.devicePixelRatioF()
        )
    )


def _gradient_angle_for_ui(clockwise_angle: float) -> float:
    """Convert persisted clockwise degrees to signed CW UI degrees.

    >>> _gradient_angle_for_ui(90)
    90.0
    >>> _gradient_angle_for_ui(330)
    -30.0
    """
    signed = (float(clockwise_angle) + 180.0) % 360.0 - 180.0
    return 180.0 if signed == -180.0 else signed


def _gradient_angle_from_ui(signed_clockwise_angle: float) -> float:
    """Convert signed CW UI degrees to persisted clockwise degrees.

    >>> _gradient_angle_from_ui(30)
    30.0
    >>> _gradient_angle_from_ui(-90)
    270.0
    """
    return float(signed_clockwise_angle) % 360.0


class TextGradientGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None):
        super().__init__('')
        self.setObjectName('TextGradientGroup')
        self.setTitle(self.tr('Gradient'))
        self.on_param_changed = on_param_changed
        self.enable_checker = QCheckBox(self)
        self.enable_checker.setObjectName('EffectTitleChecker')
        self.enable_checker.setToolTip(self.tr('Enable'))
        self.enable_checker.toggled.connect(self._on_enabled_changed)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.start_picker = SmallColorPickerLabel(self, param_name='gradient_start_color')
        self.end_picker = SmallColorPickerLabel(self, param_name='gradient_end_color')
        self.start_picker.dialog_color_provider = self._start_color
        self.end_picker.dialog_color_provider = self._end_color
        self.color_arrow = QLabel('→', self)
        self.color_arrow.setObjectName('GradientColorArrow')
        self.color_arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_arrow.setText('')
        _set_svg_label(
            self.color_arrow, 'gradient_transition.svg', 24, 14
        )

        self.angle_box = SmallSizeComboBox([-180, 180], 'gradient_angle', self)
        self.angle_box.setToolTip(self.tr("Set Gradient Angle"))
        self.angle_box.param_changed.connect(self._on_angle_changed)
        self.angle_label = SmallSizeControlLabel(
            self,
            direction=1,
            text='',
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        self.angle_label.setObjectName('fontAngleLabel')
        self.angle_label.setToolTip(self.tr('Angle'))
        _set_svg_label(self.angle_label, 'rotation.svg')
        _word_wrap_label(self.angle_label)
        self.angle_label.size_ctrl_changed.connect(self._change_angle)
        self.angle_label.btn_released.connect(self._apply_angle)
        self.angle_label.reset_requested.connect(self._reset_angle)

        self.size_box = SmallSizeComboBox([0.5, 2], 'gradient_size', self)
        self.size_box.setToolTip(self.tr("Set Gradient Size"))
        self.size_box.param_changed.connect(self.on_param_changed)
        self.size_label = SmallSizeControlLabel(
            self,
            direction=1,
            text=self.tr('Size'),
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        _word_wrap_label(self.size_label)
        self.size_label.size_ctrl_changed.connect(self._change_size)
        self.size_label.btn_released.connect(self._apply_size)
        self.size_label.reset_requested.connect(self._reset_size)

        self.color_unit = _atomic_unit(
            self,
            self.start_picker,
            self.color_arrow,
            self.end_picker,
        )
        self.color_unit.layout().setSpacing(1)
        self.angle_unit = _atomic_unit(self, self.angle_label, self.angle_box)
        self.size_unit = _atomic_unit(self, self.size_label, self.size_box)
        self.atomic_units = (
            self.color_unit,
            self.angle_unit,
            self.size_unit,
        )

        self.color_row, self.color_layout = _adaptive_row(
            self,
            self.color_unit,
            self.angle_unit,
            self.size_unit,
        )
        self.adaptive_layout = QVBoxLayout(self)
        self.adaptive_layout.addWidget(self.color_row)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        hint = self.enable_checker.sizeHint()
        self.enable_checker.setGeometry(
            8,
            0,
            hint.width(),
            hint.height(),
        )

    def _on_enabled_changed(self, enabled: bool) -> None:
        self.on_param_changed('gradient_enabled', enabled)

    @staticmethod
    def _start_color():
        return getattr(C.active_format, 'gradient_start_color', None)

    @staticmethod
    def _end_color():
        return getattr(C.active_format, 'gradient_end_color', None)

    def _change_angle(self, delta: int) -> None:
        self.angle_box.changeByDelta(-delta, multiplier=1)

    def _apply_angle(self) -> None:
        self._on_angle_changed('gradient_angle', self.angle_box.value())

    def _on_angle_changed(self, param_name: str, value: float) -> None:
        self.on_param_changed(param_name, _gradient_angle_from_ui(value))

    def _reset_angle(self) -> None:
        if self.angle_box.value() == 0.0:
            return
        with QSignalBlocker(self.angle_box):
            self.angle_box.setValue(0.0)
        self._on_angle_changed('gradient_angle', 0.0)

    def _change_size(self, delta: int) -> None:
        self.size_box.changeByDelta(delta, multiplier=0.02)

    def _apply_size(self) -> None:
        self.on_param_changed('gradient_size', self.size_box.value())

    def _reset_size(self) -> None:
        if self.size_box.value() == 1.0:
            return
        with QSignalBlocker(self.size_box):
            self.size_box.setValue(1.0)
        self.on_param_changed('gradient_size', 1.0)

    def set_effect_enabled(self, enabled: bool) -> None:
        with QSignalBlocker(self.enable_checker):
            self.enable_checker.setChecked(enabled)


class RubyFuriganaGroup(QGroupBox):
    """Selection-owned group/mono Ruby editor."""

    apply_requested = Signal(str, str, str)
    remove_requested = Signal()

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setObjectName('RubyFuriganaGroup')
        self.setTitle(self.tr('Ruby / Furigana'))
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.type_combobox = SmallComboBox(parent=self)
        self.type_combobox.addItem(self.tr('Group'), 'group')
        self.type_combobox.addItem(self.tr('Mono'), 'mono')
        self.type_label = _word_wrap_label(
            SmallParamLabel(self.tr('Type'), parent=self)
        )

        self.text_edit = QLineEdit(self)
        self.text_edit.setPlaceholderText(self.tr('Ruby text'))
        self.text_edit.setToolTip(
            self.tr('For Mono Ruby, separate readings with whitespace')
        )
        self.text_edit.returnPressed.connect(self._emit_apply)
        self.text_label = _word_wrap_label(
            SmallParamLabel(self.tr('Reading'), parent=self)
        )

        self.position_combobox = SmallComboBox(parent=self)
        self.position_combobox.addItem(self.tr('Over / Right'), 'over')
        self.position_combobox.addItem(self.tr('Under / Left'), 'under')
        self.position_label = _word_wrap_label(
            SmallParamLabel(self.tr('Position'), parent=self)
        )

        self.apply_button = QPushButton(self.tr('Apply'), self)
        self.apply_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self.apply_button.clicked.connect(self._emit_apply)
        self.remove_button = QPushButton(self.tr('Remove'), self)
        self.remove_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        self.remove_button.clicked.connect(self.remove_requested.emit)
        self.remove_button.setEnabled(False)

        type_unit = _atomic_unit(
            self, self.type_label, self.type_combobox
        )
        position_unit = _atomic_unit(
            self, self.position_label, self.position_combobox
        )
        selector_row, _ = _adaptive_row(
            self, type_unit, position_unit
        )
        text_unit = _atomic_unit(
            self,
            self.text_label,
            self.text_edit,
            self.apply_button,
            self.remove_button,
        )
        self.adaptive_layout = QVBoxLayout(self)
        self.adaptive_layout.addWidget(selector_row)
        self.adaptive_layout.addWidget(text_unit)

    def _emit_apply(self) -> None:
        self.apply_requested.emit(
            str(self.type_combobox.currentData()),
            self.text_edit.text(),
            str(self.position_combobox.currentData()),
        )

    def set_state(
        self,
        ruby_type: str,
        text: str,
        position: str,
        editable: bool,
    ) -> None:
        for combobox, value in (
            (self.type_combobox, ruby_type),
            (self.position_combobox, position),
        ):
            index = combobox.findData(value)
            if index >= 0:
                combobox.setCurrentIndex(index)
        if self.text_edit.text() != text:
            self.text_edit.setText(text)
        self.remove_button.setEnabled(editable)

    def set_error(self, message: str) -> None:
        QMessageBox.warning(
            self,
            self.tr('Ruby / Furigana'),
            message,
        )


class TextAdvancedFormatPanel(PanelArea):

    param_changed = Signal(str, object)
    ligature_axis_changed = Signal(str, str)
    ruby_apply_requested = Signal(str, str, str)
    ruby_remove_requested = Signal()

    def __init__(
        self,
        panel_name: str,
        config_name: str,
        config_expand_name: str,
        on_format_changed: Callable,
    ):
        super().__init__(panel_name, config_name, config_expand_name)

        self.on_format_changed = on_format_changed
        self._last_content_width = None
        self._last_content_height = None
        self._last_height_cap = None
        self._updating_responsive_geometry = False

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

        self.top_section = QWidget(self.scrollContent)
        self.top_section.setObjectName('TextAdvancedFormatTopSection')
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

        self.ligature_group = QGroupBox(
            self.tr('Ligature'), self.scrollContent
        )
        self.ligature_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        ligature_specs = [(
            LIGATURE_COMMON,
            self.tr('Common'),
            self.tr(
                'Set common ligatures for the selected text'
            ),
        )]
        if FONT_FEATURES_AVAILABLE:
            ligature_specs.extend((
                (
                    LIGATURE_DISCRETIONARY,
                    self.tr('Discretionary'),
                    self.tr(
                        'Set font-specific optional ligatures for the '
                        'selected text'
                    ),
                ),
                (
                    OLDSTYLE_NUMS,
                    self.tr('Oldstyle'),
                    self.tr(
                        'Set oldstyle numerals for the selected text'
                    ),
                ),
                (
                    LIGATURE_CONTEXTUAL,
                    self.tr('Contextual'),
                    self.tr(
                        'Set contextual alternate glyphs for the '
                        'selected text'
                    ),
                ),
            ))
        self.ligature_comboboxes = {}
        ligature_units = []
        for axis, label, tooltip in ligature_specs:
            combo = SmallComboBox(parent=self.ligature_group)
            for option, value in zip(
                (self.tr('Default'), self.tr('On'), self.tr('Off')),
                LIGATURE_AXIS_VALUES,
            ):
                combo.addItem(option, value)
            combo.setProperty('ligature-axis', axis)
            combo.setToolTip(tooltip)
            combo.activated.connect(self.on_ligature_axis_changed)
            self.ligature_comboboxes[axis] = combo
            ligature_units.append(_atomic_unit(
                self.ligature_group,
                _word_wrap_label(
                    SmallParamLabel(label, parent=self.ligature_group)
                ),
                combo,
            ))
        unit_rows = (
            (ligature_units[:2], ligature_units[2:])
            if FONT_FEATURES_AVAILABLE
            else (ligature_units,)
        )
        rows_and_layouts = [
            _adaptive_row(
                self.ligature_group,
                *units,
            )
            for units in unit_rows
        ]
        self.ligature_rows = [row for row, _layout in rows_and_layouts]
        self.ligature_layouts = [layout for _row, layout in rows_and_layouts]
        self.ligature_group_layout = QVBoxLayout(self.ligature_group)
        for row in self.ligature_rows:
            self.ligature_group_layout.addWidget(row)

        self.top_atomic_units = (self.linespacing_type_unit,)
        self.top_layout = AdaptiveWrapLayout(self.top_section)
        for unit in self.top_atomic_units:
            self.top_layout.addWidget(unit)

        self.ruby_group = RubyFuriganaGroup(self.scrollContent)
        self.ruby_group.apply_requested.connect(
            self.ruby_apply_requested.emit
        )
        self.ruby_group.remove_requested.connect(
            self.ruby_remove_requested.emit
        )
        self.gradient_group = TextGradientGroup(self.on_format_changed)
        vlayout = QVBoxLayout()
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        vlayout.addWidget(self.top_section)
        vlayout.addWidget(self.ligature_group)
        vlayout.addWidget(self.gradient_group)
        vlayout.addWidget(self.ruby_group)

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

    def _effective_content_width(self):
        # PanelArea's styled scrollbar overlays the viewport without consuming
        # content width.
        width = self.viewport().width()
        if width <= 0:
            width = self.width() - 2 * self.frameWidth()
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
                *self.ligature_layouts,
                self.ruby_group.adaptive_layout,
                self.gradient_group.adaptive_layout,
            )
        )
        left, _top, right, _bottom = self.vlayout.getContentsMargins()
        return (
            atomic_width
            + left
            + right
            + 2 * self.frameWidth()
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

    def on_linespacing_type_changed(self) -> None:
        self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex())

    def on_ligature_axis_changed(self, _index: int) -> None:
        combo = self.sender()
        axis = str(combo.property('ligature-axis'))
        self.ligature_axis_changed.emit(axis, str(combo.currentData()))

    def set_active_format(self, font_format: FontFormat) -> None:
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)

        self.gradient_group.size_box.setValue(font_format.gradient_size)
        self.gradient_group.angle_box.setValue(
            _gradient_angle_for_ui(font_format.gradient_angle)
        )
        self.gradient_group.set_effect_enabled(font_format.gradient_enabled)
        self.gradient_group.start_picker.setPickerColor(font_format.gradient_start_color)
        self.gradient_group.end_picker.setPickerColor(font_format.gradient_end_color)

    def set_line_spacing_type(self, spacing_type: int) -> None:
        if not self.linespacing_type_combobox.hasFocus():
            self.linespacing_type_combobox.setCurrentIndex(int(spacing_type))

    def set_ligature_axis(self, axis: str, value: str) -> None:
        combo = self.ligature_comboboxes.get(axis)
        if combo is None:
            return
        index = combo.findData(value)
        if index < 0 or combo.hasFocus():
            return
        combo.setCurrentIndex(index)

    def set_ruby_state(
        self,
        ruby_type: str,
        text: str,
        position: str,
        editable: bool,
    ) -> None:
        self.ruby_group.set_state(ruby_type, text, position, editable)
