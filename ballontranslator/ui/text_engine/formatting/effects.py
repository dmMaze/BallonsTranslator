"""Expandable controls for item-wide text effects."""

from typing import Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QTimer, Signal, QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from ballontranslator.utils.text_effects import (
    HollowEffect,
    SHADOW_BLUR_LIMIT,
    SHADOW_OFFSET_LIMIT,
    SHADOW_SPREAD_LIMIT,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
    effect_phase,
)

from ...custom_widget import ColorPickerLabel, PanelArea
from ...icon_rendering import render_svg_pixmap
from ...misc import themed_icon_path
from ..transforms.controls import CommittedTransformControl

if TYPE_CHECKING:
    from ..alpha_mask_edit_session import TextAlphaMaskEditSession
    from ..item import TextBlkItem


class EffectVisibilityButton(QToolButton):
    """Compact enabled, disabled, or mixed visibility control.

    >>> EffectVisibilityButton.__name__
    'EffectVisibilityButton'
    """

    visibility_requested = Signal(bool)

    def __init__(
        self,
        show_tooltip: str,
        hide_tooltip: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._show_tooltip = show_tooltip
        self._hide_tooltip = hide_tooltip
        self._visibility: Optional[bool] = None
        self.setObjectName('TextEffectVisibilityButton')
        self.setFixedSize(18, 18)
        self.setIconSize(QSize(14, 14))
        self.clicked.connect(self._on_clicked)
        self.set_visibility(None)

    def set_visibility(self, visible: Optional[bool]) -> None:
        self._visibility = visible
        if visible is True:
            icon_name = 'text-effect-visibility-open.svg'
            description = self._hide_tooltip
        elif visible is False:
            icon_name = 'text-effect-visibility-closed.svg'
            description = self._show_tooltip
        else:
            icon_name = 'text-effect-visibility-mixed.svg'
            description = self._show_tooltip
        self.setIcon(QIcon(themed_icon_path(icon_name)))
        self.setToolTip(description)
        self.setAccessibleName(description)

    def _on_clicked(self) -> None:
        self.visibility_requested.emit(self._visibility is not True)


def _effect_icon_label(
    icon_name: str,
    parent: QWidget,
) -> QLabel:
    label = QLabel(parent)
    label.setObjectName('TextEffectParameterIcon')
    label.setFixedSize(16, 16)
    label.setPixmap(render_svg_pixmap(
        themed_icon_path(icon_name),
        16,
        16,
        parent.devicePixelRatioF(),
    ))
    return label


def _effect_action_widget(
    parent: QWidget,
    buttons: Sequence[QToolButton],
) -> QWidget:
    widget = QWidget(parent)
    widget.setObjectName('TextEffectPanelActions')
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for button in buttons:
        layout.addWidget(button)
    return widget


class EffectNumericControl(CommittedTransformControl):
    """Reuse the committed numeric editor with typed-text preview signals.

    >>> issubclass(EffectNumericControl, CommittedTransformControl)
    True
    """

    value_preview_requested = Signal(str, object)
    value_preview_canceled = Signal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('TextEffectControl')
        self.label.setObjectName('TextEffectParamLabel')
        self.editor.setObjectName('TextEffectParamEditor')

    def _on_text_edited(self) -> None:
        super()._on_text_edited()
        try:
            value = self._parse(self.editor.text())
        except (TypeError, ValueError):
            return
        self.value_preview_requested.emit(self.param_name, value)

    def commit_pending(self) -> bool:
        was_pending = self.state == self.PENDING_TEXT
        committed = super().commit_pending()
        if was_pending and not committed:
            self.value_preview_canceled.emit(self.param_name)
        return committed

    def cancel_pending(self) -> None:
        was_pending = self.state == self.PENDING_TEXT
        super().cancel_pending()
        if was_pending:
            self.value_preview_canceled.emit(self.param_name)


class StrokeEffectCard(QFrame):
    """One Stroke at its complete-stack semantic index.

    >>> StrokeEffectCard.__name__
    'StrokeEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-stroke.svg', self
        )
        self.title_label = QLabel(self.tr('Stroke'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Stroke'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')

        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Stroke'), self.tr('Hide Stroke'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.visibility_button,
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(action_widget)

        self.width_control = EffectNumericControl(
            self.tr('Width'), 'width', 1.0, 0.0, 10.0, '', 0.01,
            self, decimals=2,
        )
        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        for control in (self.width_control, self.opacity_control):
            control.editor.setProperty('cardEditor', True)
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(
                self._on_value_preview
            )
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(
                self._on_parameter_commit
            )
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        color_label = QLabel(self.tr('Color'), self)
        color_label.setObjectName('TextEffectParamLabel')
        self.color_picker = ColorPickerLabel(self, param_name='paint')
        self.color_picker.setObjectName('TextEffectColorPicker')
        self.color_picker.setFixedSize(22, 22)
        self.color_picker.setToolTip(self.tr('Stroke Color'))
        self.color_picker.changingColor.connect(
            self._on_color_dialog_opened
        )
        self.color_picker.colorChanged.connect(self._on_color_changed)
        self.color_picker.apply_color.connect(self._on_apply_color)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(color_label)
        color_row.addWidget(self.color_picker)
        color_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)
        controls.addWidget(self.width_control, 0, 0)
        controls.addLayout(color_row, 0, 1)
        controls.addWidget(self.opacity_control, 1, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(controls)

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, strokes: Sequence[StrokeEffect]) -> None:
        enabled_values = [stroke.enabled for stroke in strokes]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)

        for name, control in (
            ('width', self.width_control),
            ('opacity', self.opacity_control),
        ):
            values = [getattr(stroke, name) for stroke in strokes]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)

        colors = [stroke.paint.color for stroke in strokes]
        common_color = (
            colors[0]
            if colors and all(color == colors[0] for color in colors)
            else None
        )
        if common_color is None:
            self.color_picker.color = None
            self.color_picker.setStyleSheet('')
            self.color_picker.setToolTip(self.tr('Mixed'))
        else:
            self.color_picker.setPickerColor(common_color)
            self.color_picker.setToolTip(self.tr('Stroke Color'))

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (self.width_control, self.opacity_control)

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _on_color_dialog_opened(self) -> None:
        self.color_dialog_active_changed.emit(True)

    def _on_color_changed(self, accepted: bool) -> None:
        self.color_dialog_active_changed.emit(False)
        if accepted:
            self.value_commit_requested.emit(
                self.index,
                'paint',
                SolidPaint(self.color_picker.rgb()),
            )

    def _on_apply_color(self, _name: str, color: Tuple[int, int, int]) -> None:
        self.value_commit_requested.emit(
            self.index, 'paint', SolidPaint(color)
        )


class ShadowEffectCard(QFrame):
    """Edit one typed Shadow at its complete-stack index.

    >>> ShadowEffectCard.__name__
    'ShadowEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-shadow.svg', self
        )
        self.title_label = QLabel(self.tr('Shadow'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Shadow'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')

        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Shadow'), self.tr('Hide Shadow'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.visibility_button,
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(action_widget)

        type_label = QLabel(self.tr('Type'), self)
        type_label.setObjectName('TextEffectParamLabel')
        self.type_selector = QComboBox(self)
        self.type_selector.setObjectName('TextEffectTypeSelector')
        self.type_selector.setPlaceholderText(self.tr('Mixed'))
        for label, value in (
            (self.tr('Drop'), 'drop'),
            (self.tr('Inner'), 'inner'),
            (self.tr('Long / Extrude'), 'long'),
        ):
            self.type_selector.addItem(label, value)
        self.type_selector.currentIndexChanged.connect(
            self._on_type_changed
        )
        type_row = QHBoxLayout()
        type_row.setContentsMargins(0, 0, 0, 0)
        type_row.addWidget(type_label)
        type_row.addWidget(self.type_selector)

        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        self.offset_x_control = EffectNumericControl(
            self.tr('X Offset'), 'offset_x', 1.0,
            -SHADOW_OFFSET_LIMIT, SHADOW_OFFSET_LIMIT, '', 0.01,
            self, decimals=2,
        )
        self.offset_y_control = EffectNumericControl(
            self.tr('Y Offset'), 'offset_y', 1.0,
            -SHADOW_OFFSET_LIMIT, SHADOW_OFFSET_LIMIT, '', 0.01,
            self, decimals=2,
        )
        self.blur_control = EffectNumericControl(
            self.tr('Blur'), 'blur', 1.0, 0.0,
            SHADOW_BLUR_LIMIT, '', 0.01,
            self, decimals=2,
        )
        self.spread_control = EffectNumericControl(
            self.tr('Spread'), 'spread', 1.0, 0.0,
            SHADOW_SPREAD_LIMIT, '', 0.01,
            self, decimals=2,
        )
        for control in self.iter_controls():
            control.editor.setProperty('cardEditor', True)
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(self._on_parameter_commit)
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        color_label = QLabel(self.tr('Color'), self)
        color_label.setObjectName('TextEffectParamLabel')
        self.color_picker = ColorPickerLabel(self, param_name='color')
        self.color_picker.setObjectName('TextEffectColorPicker')
        self.color_picker.setFixedSize(22, 22)
        self.color_picker.setToolTip(self.tr('Shadow Color'))
        self.color_picker.changingColor.connect(
            self._on_color_dialog_opened
        )
        self.color_picker.colorChanged.connect(self._on_color_changed)
        self.color_picker.apply_color.connect(self._on_apply_color)
        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.addWidget(color_label)
        color_row.addWidget(self.color_picker)
        color_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)
        controls.addLayout(type_row, 0, 0)
        controls.addLayout(color_row, 0, 1)
        controls.addWidget(self.opacity_control, 1, 0, 1, 2)
        controls.addWidget(self.offset_x_control, 2, 0)
        controls.addWidget(self.offset_y_control, 2, 1)
        controls.addWidget(self.blur_control, 3, 0)
        controls.addWidget(self.spread_control, 3, 1)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(controls)

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, shadows: Sequence[ShadowEffect]) -> None:
        enabled_values = [shadow.enabled for shadow in shadows]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)

        types = [shadow.shadow_type for shadow in shadows]
        common_type = (
            types[0]
            if types and all(value == types[0] for value in types)
            else None
        )
        with QSignalBlocker(self.type_selector):
            self.type_selector.setCurrentIndex(
                -1 if common_type is None
                else self.type_selector.findData(common_type)
            )
        show_soft_controls = common_type != 'long'
        self.blur_control.setVisible(show_soft_controls)
        self.spread_control.setVisible(show_soft_controls)
        if common_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        elif common_type is None:
            self.spread_control.label.setText(self.tr('Spread / Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('offset_x', self.offset_x_control),
            ('offset_y', self.offset_y_control),
            ('blur', self.blur_control),
            ('spread', self.spread_control),
        ):
            values = [
                shadow.offset[0]
                if name == 'offset_x'
                else shadow.offset[1]
                if name == 'offset_y'
                else getattr(shadow, name)
                for shadow in shadows
            ]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)

        colors = [shadow.color for shadow in shadows]
        common_color = (
            colors[0]
            if colors and all(color == colors[0] for color in colors)
            else None
        )
        if common_color is None:
            self.color_picker.color = None
            self.color_picker.setStyleSheet('')
            self.color_picker.setToolTip(self.tr('Mixed'))
        else:
            self.color_picker.setPickerColor(common_color)
            self.color_picker.setToolTip(self.tr('Shadow Color'))

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            self.opacity_control,
            self.offset_x_control,
            self.offset_y_control,
            self.blur_control,
            self.spread_control,
        )

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'shadow_type',
                self.type_selector.itemData(combo_index),
            )

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _on_color_dialog_opened(self) -> None:
        self.color_dialog_active_changed.emit(True)

    def _on_color_changed(self, accepted: bool) -> None:
        self.color_dialog_active_changed.emit(False)
        if accepted:
            self.value_commit_requested.emit(
                self.index, 'color', self.color_picker.rgb()
            )

    def _on_apply_color(self, _name: str, color: Tuple[int, int, int]) -> None:
        self.value_commit_requested.emit(self.index, 'color', color)


class HollowEffectCard(QFrame):
    """Edit the single structural Hollow effect.

    >>> HollowEffectCard.__name__
    'HollowEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    remove_requested = Signal(int)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.title_icon_label = _effect_icon_label(
            'text-effect-hollow.svg', self
        )
        self.title_label = QLabel(self.tr('Hollow'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.delete_button = QToolButton(self)
        self.delete_button.setObjectName('TextEffectCloseButton')
        self.delete_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.delete_button.setToolTip(self.tr('Delete Hollow'))
        self.delete_button.setAccessibleName(self.tr('Delete Hollow'))
        self.delete_button.setFixedSize(18, 18)
        self.delete_button.clicked.connect(self._on_delete_clicked)

        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Hollow'), self.tr('Hide Hollow'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self, (self.visibility_button, self.delete_button)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        layout.addWidget(self.title_icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(action_widget)

    def set_values(self, hollows: Sequence[HollowEffect]) -> None:
        values = [hollow.enabled for hollow in hollows]
        common = (
            values[0]
            if values and all(value == values[0] for value in values)
            else None
        )
        self.visibility_button.set_visibility(common)

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return ()

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_delete_clicked(self) -> None:
        self.remove_requested.emit(self.index)


class AlphaMaskCard(QFrame):
    """Pinned controls for the selected TextBlock-owned mask.

    >>> AlphaMaskCard.__name__
    'AlphaMaskCard'
    """

    enabled_requested = Signal(bool)
    mode_changed = Signal(str)
    diameter_changed = Signal(float)
    clear_requested = Signal()
    remove_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('TextAlphaMaskCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-alpha-mask.svg', self
        )
        self.title_label = QLabel(self.tr('Alpha Mask'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Alpha Mask'), self.tr('Hide Alpha Mask'), self
        )
        self.visibility_button.visibility_requested.connect(
            self.enabled_requested.emit
        )
        self.remove_button = QToolButton(self)
        self.remove_button.setObjectName('TextEffectCloseButton')
        self.remove_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.remove_button.setToolTip(self.tr('Remove Alpha Mask'))
        self.remove_button.setAccessibleName(self.tr('Remove Alpha Mask'))
        self.remove_button.setFixedSize(18, 18)
        self.remove_button.clicked.connect(self.remove_requested.emit)

        action_widget = _effect_action_widget(
            self, (self.visibility_button, self.remove_button)
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(action_widget)

        mode_label = QLabel(self.tr('Mode'), self)
        mode_label.setObjectName('TextEffectParamLabel')
        self.mode_selector = QComboBox(self)
        self.mode_selector.setObjectName('TextAlphaMaskModeSelector')
        self.mode_selector.addItem(self.tr('Erase'), 'erase')
        self.mode_selector.addItem(self.tr('Restore'), 'restore')
        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_changed
        )

        size_label = QLabel(self.tr('Size'), self)
        size_label.setObjectName('TextEffectParamLabel')
        self.diameter_editor = QDoubleSpinBox(self)
        self.diameter_editor.setObjectName('TextAlphaMaskSizeEditor')
        self.diameter_editor.setRange(1.0, 500.0)
        self.diameter_editor.setDecimals(1)
        self.diameter_editor.setSingleStep(1.0)
        self.diameter_editor.setSuffix(self.tr(' px'))
        self.diameter_editor.valueChanged.connect(
            self.diameter_changed.emit
        )

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(4)
        controls.addWidget(mode_label, 0, 0)
        controls.addWidget(self.mode_selector, 0, 1)
        controls.addWidget(size_label, 1, 0)
        controls.addWidget(self.diameter_editor, 1, 1)
        controls.setColumnStretch(1, 1)

        self.clear_button = QToolButton(self)
        self.clear_button.setObjectName('TextAlphaMaskClearButton')
        self.clear_button.setText(self.tr('Clear'))
        self.clear_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.clear_button.clicked.connect(self.clear_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header)
        layout.addLayout(controls)
        layout.addWidget(
            self.clear_button, alignment=Qt.AlignmentFlag.AlignLeft
        )

    def set_values(
        self,
        mask: TextAlphaMask,
        mode: str,
        diameter: float,
    ) -> None:
        blockers = (
            QSignalBlocker(self.mode_selector),
            QSignalBlocker(self.diameter_editor),
        )
        self.visibility_button.set_visibility(mask.enabled)
        index = self.mode_selector.findData(mode)
        self.mode_selector.setCurrentIndex(max(0, index))
        self.diameter_editor.setValue(diameter)
        del blockers

    def _on_mode_changed(self, index: int) -> None:
        mode = self.mode_selector.itemData(index)
        if mode in {'erase', 'restore'}:
            self.mode_changed.emit(mode)


class TextEffectPanel(PanelArea):
    """Own Overall Opacity and typed effect cards.

    >>> TextEffectPanel.__name__
    'TextEffectPanel'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    add_effect_requested = Signal(str)
    remove_effect_requested = Signal(int)
    move_effect_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)
    mask_edit_requested = Signal(bool)
    mask_enabled_requested = Signal(bool)
    mask_mode_changed = Signal(str)
    mask_diameter_changed = Signal(float)
    mask_clear_requested = Signal()
    mask_remove_requested = Signal()

    MAX_CONTENT_HEIGHT = 480

    def __init__(
        self,
        panel_name: str,
        config_name: str,
        config_expand_name: str,
    ) -> None:
        super().__init__(panel_name, config_name, config_expand_name)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.setMaximumHeight(self.MAX_CONTENT_HEIGHT)

        self.overall_opacity_control = EffectNumericControl(
            self.tr('Overall Opacity'),
            'overall_opacity',
            100.0,
            0.0,
            1.0,
            '%',
            1.0,
            self.scrollContent,
            decimals=1,
        )
        self.overall_opacity_control.commit_requested.connect(
            self._on_overall_commit
        )
        self.overall_opacity_control.value_preview_requested.connect(
            self._on_overall_value_preview
        )
        self.overall_opacity_control.preview_requested.connect(
            self._on_overall_parameter_preview
        )
        self.overall_opacity_control.drag_commit_requested.connect(
            self._on_overall_parameter_commit
        )
        self.overall_opacity_control.preview_canceled.connect(
            self._on_overall_preview_canceled
        )
        self.overall_opacity_control.value_preview_canceled.connect(
            self._on_overall_preview_canceled
        )

        self.mask_brush_button = QToolButton(self.scrollContent)
        self.mask_brush_button.setObjectName('TextEffectBrushButton')
        self.mask_brush_button.setIcon(
            QIcon(themed_icon_path('drawingtools_pen.svg'))
        )
        self.mask_brush_button.setFixedSize(26, 26)
        self.mask_brush_button.setCheckable(True)
        self.mask_brush_button.setEnabled(False)
        self.mask_brush_button.setToolTip(
            self.tr('Select one text block in text edit mode.')
        )
        self.mask_brush_button.setAccessibleName(
            self.tr('Alpha Mask Brush')
        )
        self.mask_brush_button.clicked.connect(
            self._on_mask_brush_clicked
        )

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        top_row.addWidget(self.overall_opacity_control)
        top_row.addWidget(self.mask_brush_button)

        self.add_effect_button = QToolButton(self.scrollContent)
        self.add_effect_button.setObjectName('AddTextEffectButton')
        self.add_effect_button.setText(self.tr('Add'))
        self.add_effect_button.setToolTip(self.tr('Add Effect'))
        self.add_effect_button.setAccessibleName(self.tr('Add Effect'))
        self.add_effect_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.add_effect_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        add_menu = QMenu(self.add_effect_button)
        add_menu.setObjectName('TextEffectAddMenu')
        self.add_effect_actions = {}
        for label, effect_type, icon_name in (
            (self.tr('Stroke'), 'stroke', 'text-effect-stroke.svg'),
            (self.tr('Shadow'), 'shadow', 'text-effect-shadow.svg'),
            (self.tr('Hollow'), 'hollow', 'text-effect-hollow.svg'),
        ):
            action = add_menu.addAction(
                QIcon(themed_icon_path(icon_name)), label
            )
            action.setData(effect_type)
            action.triggered.connect(self._on_add_effect_triggered)
            self.add_effect_actions[effect_type] = action
        self.add_effect_button.setMenu(add_menu)

        self.mixed_label = QLabel(self.tr('Mixed'), self.scrollContent)
        self.mixed_label.setObjectName('TextEffectMixedLabel')
        self.mixed_label.setVisible(False)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.effect_cards = []
        self.stroke_cards = []
        self.shadow_cards = []
        self.hollow_card = None
        self._effect_types = None
        self.alpha_mask_card = None
        self._mask_items = ()
        self._alpha_mask_session = None
        self.mask_card_layout = QVBoxLayout()
        self.mask_card_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(top_row)
        layout.addLayout(self.mask_card_layout)
        layout.addWidget(
            self.add_effect_button,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )
        layout.addWidget(self.mixed_label)
        layout.addLayout(self.cards_layout)
        self.setContentLayout(layout)
        self.content_layout = layout
        self.scrollContent.after_resized.connect(self._sync_content_height)
        self._sync_content_height()
        QTimer.singleShot(0, self._sync_content_height)

    def set_alpha_mask_session(
        self, session: "TextAlphaMaskEditSession"
    ) -> None:
        self._alpha_mask_session = session
        self.refresh_alpha_mask_state()

    def _set_alpha_mask_card(self, present: bool) -> None:
        if present and self.alpha_mask_card is None:
            card = AlphaMaskCard(self.scrollContent)
            card.enabled_requested.connect(
                self.mask_enabled_requested.emit
            )
            card.mode_changed.connect(self.mask_mode_changed.emit)
            card.diameter_changed.connect(
                self.mask_diameter_changed.emit
            )
            card.clear_requested.connect(self.mask_clear_requested.emit)
            card.remove_requested.connect(self.mask_remove_requested.emit)
            self.mask_card_layout.addWidget(card)
            self.alpha_mask_card = card
        elif not present and self.alpha_mask_card is not None:
            card = self.alpha_mask_card
            self.alpha_mask_card = None
            self.mask_card_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()

    def refresh_alpha_mask_state(self) -> None:
        session = self._alpha_mask_session
        item = self._mask_items[0] if len(self._mask_items) == 1 else None
        if item is not None and session is not None:
            try:
                attached = item.scene() is session.canvas
            except RuntimeError:
                attached = False
            if not attached:
                self._mask_items = ()
                item = None
        mask = None if item is None else item.blk.text_alpha_mask
        self._set_alpha_mask_card(mask is not None)
        eligible = bool(session is not None and session.can_activate(item))
        active = bool(
            session is not None and session.active and session.target is item
        )
        self.mask_brush_button.setEnabled(eligible)
        blocker = QSignalBlocker(self.mask_brush_button)
        self.mask_brush_button.setChecked(active)
        del blocker
        self.mask_brush_button.setToolTip(
            self.tr('Edit Alpha Mask')
            if eligible
            else self.tr('Select one text block in text edit mode.')
        )
        if self.alpha_mask_card is not None and mask is not None:
            self.alpha_mask_card.setEnabled(eligible)
            self.alpha_mask_card.set_values(
                mask,
                session.mode if session is not None else 'erase',
                session.diameter if session is not None else 24.0,
            )
        self._sync_content_height()

    def _clear_effect_cards(self) -> None:
        for card in self.effect_cards:
            self.cards_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.effect_cards = []
        self.stroke_cards = []
        self.shadow_cards = []
        self.hollow_card = None

    def _rebuild_effect_cards(self, effect_types: Sequence[str]) -> None:
        effect_types = tuple(effect_types)
        if effect_types == self._effect_types:
            return
        self._clear_effect_cards()
        self._effect_types = effect_types
        for index, effect_type in enumerate(effect_types):
            if effect_type == 'stroke':
                card = StrokeEffectCard(index, self.scrollContent)
                self.stroke_cards.append(card)
            elif effect_type == 'shadow':
                card = ShadowEffectCard(index, self.scrollContent)
                self.shadow_cards.append(card)
            elif effect_type == 'hollow':
                card = HollowEffectCard(index, self.scrollContent)
                self.hollow_card = card
            else:
                continue
            card.value_commit_requested.connect(
                self.value_commit_requested.emit
            )
            if not isinstance(card, HollowEffectCard):
                card.value_preview_requested.connect(
                    self.value_preview_requested.emit
                )
                card.parameter_preview_requested.connect(
                    self.parameter_preview_requested.emit
                )
                card.parameter_commit_requested.connect(
                    self.parameter_commit_requested.emit
                )
                card.preview_canceled.connect(self.preview_canceled.emit)
                card.move_requested.connect(self.move_effect_requested.emit)
                card.color_dialog_active_changed.connect(
                    self.color_dialog_active_changed.emit
                )
            card.remove_requested.connect(self.remove_effect_requested.emit)
            self.cards_layout.addWidget(card)
            self.effect_cards.append(card)

    @staticmethod
    def _effect_sequence(stack: TextEffectStack) -> Tuple[str, ...]:
        return tuple(effect.effect_type for effect in stack.effects)

    def _set_effect_states(
        self, states: Sequence[TextEffectStack]
    ) -> None:
        states = tuple(states)
        if not states or any(
            not isinstance(state, TextEffectStack) for state in states
        ):
            raise TypeError('effect panel requires TextEffectStack values')

        opacity_values = [state.overall_opacity for state in states]
        common_opacity = (
            opacity_values[0]
            if all(value == opacity_values[0] for value in opacity_values)
            else None
        )
        self.overall_opacity_control.set_model_value(
            common_opacity, opacity_values
        )

        sequences = [self._effect_sequence(state) for state in states]
        common_sequence = (
            sequences[0]
            if all(sequence == sequences[0] for sequence in sequences)
            else None
        )
        mixed = common_sequence is None
        self.mixed_label.setVisible(mixed)
        self.add_effect_button.setEnabled(not mixed)
        if mixed:
            self._rebuild_effect_cards(())
        else:
            self._rebuild_effect_cards(common_sequence)
            phase_sequences = [
                tuple(effect_phase(effect) for effect in state.effects)
                for state in states
            ]
            phases_match = all(
                sequence == phase_sequences[0]
                for sequence in phase_sequences
            )
            for card in self.effect_cards:
                values = [state.effects[card.index] for state in states]
                card.set_values(values)
                if isinstance(card, (StrokeEffectCard, ShadowEffectCard)):
                    if phases_match:
                        phase = phase_sequences[0][card.index]
                        phase_indices = [
                            index
                            for index, value in enumerate(phase_sequences[0])
                            if value == phase
                        ]
                        position = phase_indices.index(card.index)
                        card.set_move_enabled(
                            position > 0,
                            position + 1 < len(phase_indices),
                        )
                    else:
                        card.set_move_enabled(False, False)
        self.add_effect_actions['hollow'].setEnabled(
            not mixed and common_sequence is not None
            and 'hollow' not in common_sequence
        )
        self._sync_content_height()

    def set_active_format(self, font_format: FontFormat) -> None:
        self._mask_items = ()
        self._set_effect_states([font_format.text_effects])
        self.refresh_alpha_mask_state()

    def set_effect_items(self, items: Sequence["TextBlkItem"]) -> None:
        self._mask_items = tuple(items)
        self._set_effect_states(
            [item.blk.fontformat.text_effects for item in items]
        )
        self.refresh_alpha_mask_state()

    def set_alpha_mask_items(self, items: Sequence["TextBlkItem"]) -> None:
        """Refresh only the TextBlock-owned mask target boundary."""
        self._mask_items = tuple(items)
        self.refresh_alpha_mask_state()

    def iter_controls(self) -> Iterator[EffectNumericControl]:
        yield self.overall_opacity_control
        for card in self.effect_cards:
            yield from card.iter_controls()

    def finish_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.commit_pending()

    def cancel_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.cancel_pending()

    def cancel_effect_previews(self) -> None:
        for control in self.iter_controls():
            control.cancel_preview()

    def _sync_content_height(self) -> None:
        if not hasattr(self, 'content_layout'):
            return
        content_height = self.content_layout.sizeHint().height()
        self.scrollContent.setMinimumHeight(content_height)
        self.setMinimumHeight(min(content_height, self.MAX_CONTENT_HEIGHT))
        self.scrollContent.updateGeometry()
        self.updateGeometry()
        self.view_widget.updateGeometry()

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if not hasattr(self, 'content_layout'):
            return hint
        hint.setHeight(min(
            self.content_layout.sizeHint().height(),
            self.MAX_CONTENT_HEIGHT,
        ))
        return hint

    def _on_overall_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(-1, name, value)

    def _on_overall_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(-1, name, value)

    def _on_overall_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(-1, name, delta)

    def _on_overall_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(-1, name, delta)

    def _on_overall_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(-1, name)

    def _on_add_effect_triggered(self, _checked: bool = False) -> None:
        action = self.sender()
        if action is not None and action.data() in {
            'stroke', 'shadow', 'hollow'
        }:
            self.add_effect_requested.emit(action.data())

    def _on_mask_brush_clicked(self, checked: bool) -> None:
        self.mask_edit_requested.emit(checked)
