"""Expandable controls for item-wide text effects."""

from typing import Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import QEvent, QRectF, QSignalBlocker, QTimer, Signal, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPaintEvent, QPainter
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QColorDialog,
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
    EffectPaint,
    GlowEffect,
    GradientOverlayEffect,
    LinearGradientPaint,
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

from ...custom_widget import PanelArea
from ...custom_widget.combobox import BottomBorderComboBox
from ...icon_rendering import render_svg_pixmap
from ...misc import themed_icon_path
from ..transforms.controls import CommittedTransformControl, TransformDragLabel
from ..rendering.effect_paint import paint_effect_paint_preview
from .gradient_editor import InlineLinearGradientEditor

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
        self.setIconSize(QSize(16, 16))
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


class _EffectCard(QFrame):
    """Keep pointer-only action icons keyboard reachable."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hovered = False
        self._keyboard_focused_action: Optional[QToolButton] = None
        self._hover_actions: Tuple[Tuple[QToolButton, QIcon], ...] = ()

    def set_hover_actions(
        self, buttons: Sequence[QToolButton]
    ) -> None:
        self._hover_actions = tuple(
            (button, button.icon()) for button in buttons
        )
        for button, _icon in self._hover_actions:
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.installEventFilter(self)
        self._sync_action_icons()

    def _sync_action_icons(self) -> None:
        visible = self._hovered or self._keyboard_focused_action is not None
        for button, icon in self._hover_actions:
            button.setIcon(icon if visible else QIcon())

    def eventFilter(self, watched: QWidget, event: QEvent) -> bool:
        if any(watched is button for button, _icon in self._hover_actions):
            if event.type() == QEvent.Type.FocusIn:
                keyboard_reasons = {
                    Qt.FocusReason.TabFocusReason,
                    Qt.FocusReason.BacktabFocusReason,
                    Qt.FocusReason.ShortcutFocusReason,
                }
                self._keyboard_focused_action = (
                    watched if event.reason() in keyboard_reasons else None
                )
                self._sync_action_icons()
            elif (
                event.type() == QEvent.Type.FocusOut
                and watched is self._keyboard_focused_action
            ):
                self._keyboard_focused_action = None
                self._sync_action_icons()
        return super().eventFilter(watched, event)

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self._sync_action_icons()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self._sync_action_icons()
        super().leaveEvent(event)


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
    parent: _EffectCard,
    buttons: Sequence[QToolButton],
) -> QWidget:
    widget = QWidget(parent)
    widget.setObjectName('TextEffectPanelActions')
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for button in buttons:
        button.setFixedSize(18, 18)
        icon_size = (
            12 if button.objectName() == 'TextEffectCloseButton' else 16
        )
        button.setIconSize(QSize(icon_size, icon_size))
        layout.addWidget(button)
    widget.setFixedWidth(18 * len(buttons) + 4 * max(0, len(buttons) - 1))
    parent.set_hover_actions(buttons)
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
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.label.setObjectName('TextEffectParamLabel')
        self.label.setWordWrap(False)
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.editor.setObjectName('TextEffectParamEditor')
        self.editor.setProperty('cardEditor', True)
        self.editor.setMinimumWidth(0)
        self.editor.setMaximumWidth(16777215)
        self.editor.setFixedHeight(22)
        self.editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.layout().setSpacing(8)
        self.layout().setStretch(0, 0)
        self.layout().setStretch(1, 1)

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


class EffectPaintButton(QToolButton):
    """Compact solid swatch or rendered linear-gradient strip.

    >>> issubclass(EffectPaintButton, QToolButton)
    True
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paint: Optional[EffectPaint] = None
        self._mixed = False
        self.setObjectName('TextEffectPaintButton')
        self.setMinimumHeight(24)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_paint(
        self,
        paint: Optional[EffectPaint],
        mixed: bool = False,
        editable: bool = True,
        description: Optional[str] = None,
    ) -> None:
        self._paint = paint
        self._mixed = bool(mixed)
        self.setIcon(QIcon())
        if mixed:
            self.setText(self.tr('Mixed'))
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            if description is None:
                if editable:
                    description = self.tr('Choose Shared Stroke Color')
                elif isinstance(paint, LinearGradientPaint):
                    description = self.tr('Mixed Gradient Paint')
                else:
                    description = self.tr('Mixed Stroke Paint')
            self.setToolTip(description)
            self.setAccessibleName(description)
            self.setEnabled(editable)
            self.update()
            return
        if paint is None:
            raise ValueError('non-mixed effect paint button requires paint')
        self.setText('')
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        if description is None:
            description = (
                self.tr('Edit Gradient')
                if isinstance(paint, LinearGradientPaint)
                else self.tr('Choose Stroke Color')
            )
        self.setToolTip(description)
        self.setAccessibleName(description)
        self.setEnabled(True)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._paint is None or self._mixed:
            return
        rect = QRectF(self.contentsRect()).adjusted(4.0, 3.0, -4.0, -3.0)
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        painter = QPainter(self)
        paint_effect_paint_preview(
            painter,
            rect,
            self._paint,
            self.palette(),
            self.devicePixelRatioF(),
        )


class StrokeEffectCard(_EffectCard):
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
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
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
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        self.position_selector = BottomBorderComboBox(self)
        self.position_selector.setObjectName('TextEffectParamEditor')
        self.position_selector.setPlaceholderText(self.tr('Mixed'))
        self.position_selector.setAccessibleName(self.tr('Stroke Position'))
        for label, value in (
            (self.tr('Inside'), 'inside'),
            (self.tr('Center'), 'center'),
            (self.tr('Outside'), 'outside'),
        ):
            self.position_selector.addItem(label, value)
        self.position_selector.currentIndexChanged.connect(
            self._on_position_changed
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.position_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.width_control = EffectNumericControl(
            self.tr('Width'), 'width', 1.0, 0.0, 10.0, '', 0.01,
            self, decimals=2,
        )
        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(self)
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Stroke Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(self.tr('Gradient'), 'linear_gradient')
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        for control in (self.width_control, self.opacity_control):
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

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[EffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.width_control, 0, 0)
        controls.addWidget(self.opacity_control, 0, 1)
        controls.addLayout(paint_row, 1, 0, 1, 2)
        controls.addWidget(self.gradient_editor, 2, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
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

        positions = [stroke.position for stroke in strokes]
        common_position = (
            positions[0]
            if positions
            and all(position == positions[0] for position in positions)
            else None
        )
        with QSignalBlocker(self.position_selector):
            self.position_selector.setCurrentIndex(
                -1
                if common_position is None
                else self.position_selector.findData(common_position)
            )

        paints = [stroke.paint for stroke in strokes]
        common_paint_type = (
            paints[0].paint_type
            if paints
            and all(
                value.paint_type == paints[0].paint_type
                for value in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1
                if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )

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

        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=(common_paint_type == 'solid') if mixed_paint else True,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (self.width_control, self.opacity_control)

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_position_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'position',
                self.position_selector.itemData(combo_index),
            )

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
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

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Stroke Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class ShadowEffectCard(_EffectCard):
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
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
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
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        self.type_selector = BottomBorderComboBox(self)
        self.type_selector.setObjectName('TextEffectParamEditor')
        self.type_selector.setPlaceholderText(self.tr('Mixed'))
        self.type_selector.setAccessibleName(self.tr('Shadow Type'))
        for label, value in (
            (self.tr('Drop'), 'drop'),
            (self.tr('Inner'), 'inner'),
            (self.tr('Long / Extrude'), 'long'),
        ):
            self.type_selector.addItem(label, value)
        self.type_selector.currentIndexChanged.connect(
            self._on_type_changed
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.type_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

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
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(self._on_parameter_commit)
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(self)
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Shadow Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(
            self.tr('Gradient'), 'linear_gradient'
        )
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[EffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.opacity_control, 0, 0)
        controls.addWidget(self.blur_control, 0, 1)
        controls.addWidget(self.offset_x_control, 1, 0)
        controls.addWidget(self.offset_y_control, 1, 1)
        controls.addWidget(self.spread_control, 2, 0)
        controls.addLayout(paint_row, 3, 0, 1, 2)
        controls.addWidget(self.gradient_editor, 4, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
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

        paints = [shadow.paint for shadow in shadows]
        common_paint_type = (
            paints[0].paint_type
            if paints and all(
                paint.paint_type == paints[0].paint_type for paint in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1 if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        editable = common_paint_type == 'solid' if mixed_paint else True
        if mixed_paint:
            if editable:
                description = self.tr('Choose Shared Shadow Color')
            elif common_paint_type == 'linear_gradient':
                description = self.tr('Mixed Shadow Gradient Paint')
            else:
                description = self.tr('Mixed Shadow Paint')
        else:
            description = (
                self.tr('Edit Shadow Gradient')
                if isinstance(common_paint, LinearGradientPaint)
                else self.tr('Choose Shadow Color')
            )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=editable,
            description=description,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = self.gradient_editor.isHidden() == show_gradient
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

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

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
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

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Shadow Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class GlowEffectCard(_EffectCard):
    """Edit one typed Glow at its complete-stack index.

    >>> GlowEffectCard.__name__
    'GlowEffectCard'
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
            'text-effect-glow.svg', self
        )
        self.title_label = QLabel(self.tr('Glow'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Glow'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Glow'), self.tr('Hide Glow'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )
        self.type_selector = BottomBorderComboBox(self)
        self.type_selector.setObjectName('TextEffectParamEditor')
        self.type_selector.setPlaceholderText(self.tr('Mixed'))
        self.type_selector.setAccessibleName(self.tr('Glow Type'))
        self.type_selector.addItem(self.tr('Outer'), 'outer')
        self.type_selector.addItem(self.tr('Inner'), 'inner')
        self.type_selector.currentIndexChanged.connect(
            self._on_type_changed
        )
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.type_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        self.size_control = EffectNumericControl(
            self.tr('Size'), 'size', 1.0, 0.0,
            SHADOW_BLUR_LIMIT, '', 0.01, self, decimals=2,
        )
        self.spread_control = EffectNumericControl(
            self.tr('Spread'), 'spread', 1.0, 0.0,
            SHADOW_SPREAD_LIMIT, '', 0.01, self, decimals=2,
        )
        for control in self.iter_controls():
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(
                self._on_parameter_commit
            )
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(self)
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Glow Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(
            self.tr('Gradient'), 'linear_gradient'
        )
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[EffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.opacity_control, 0, 0)
        controls.addWidget(self.size_control, 0, 1)
        controls.addWidget(self.spread_control, 1, 0)
        controls.addLayout(paint_row, 2, 0, 1, 2)
        controls.addWidget(self.gradient_editor, 3, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
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

    def set_values(self, glows: Sequence[GlowEffect]) -> None:
        enabled_values = [glow.enabled for glow in glows]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)

        types = [glow.glow_type for glow in glows]
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
        if common_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        elif common_type is None:
            self.spread_control.label.setText(self.tr('Spread / Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('size', self.size_control),
            ('spread', self.spread_control),
        ):
            values = [getattr(glow, name) for glow in glows]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)

        paints = [glow.paint for glow in glows]
        common_paint_type = (
            paints[0].paint_type
            if paints and all(
                paint.paint_type == paints[0].paint_type for paint in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1 if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        editable = (
            common_paint_type == 'solid' if mixed_paint else True
        )
        if mixed_paint:
            if editable:
                description = self.tr('Choose Shared Glow Color')
            elif common_paint_type == 'linear_gradient':
                description = self.tr('Mixed Glow Gradient Paint')
            else:
                description = self.tr('Mixed Glow Paint')
        else:
            description = (
                self.tr('Edit Glow Gradient')
                if isinstance(common_paint, LinearGradientPaint)
                else self.tr('Choose Glow Color')
            )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=editable,
            description=description,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            self.opacity_control,
            self.size_control,
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
                'glow_type',
                self.type_selector.itemData(combo_index),
            )

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
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

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Glow Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class GradientOverlayEffectCard(_EffectCard):
    """Edit the single foreground Gradient effect.

    >>> GradientOverlayEffectCard.__name__
    'GradientOverlayEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
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
            'text-effect-gradient.svg', self
        )
        self.title_label = QLabel(self.tr('Gradient'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Gradient'),
            self.tr('Hide Gradient'),
            self,
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        self.delete_button = QToolButton(self)
        self.delete_button.setObjectName('TextEffectCloseButton')
        self.delete_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.delete_button.setToolTip(self.tr('Delete Gradient'))
        self.delete_button.setAccessibleName(
            self.tr('Delete Gradient')
        )
        self.delete_button.setFixedSize(18, 18)
        self.delete_button.clicked.connect(self._on_delete_clicked)

        action_widget = _effect_action_widget(self, (self.delete_button,))
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addWidget(self.gradient_editor)

    def set_values(
        self, overlays: Sequence[GradientOverlayEffect]
    ) -> None:
        enabled_values = [overlay.enabled for overlay in overlays]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        paints = [overlay.paint for overlay in overlays]
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        paint = common_paint or paints[0]
        self.gradient_editor.set_paint(
            paint, editable=common_paint is not None
        )

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return ()

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_delete_clicked(self) -> None:
        self.remove_requested.emit(self.index)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class AlphaMaskCard(_EffectCard):
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
        self.title_label = QLabel(self.tr('Eraser'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Eraser'), self.tr('Hide Eraser'), self
        )
        self.visibility_button.visibility_requested.connect(
            self.enabled_requested.emit
        )
        self.remove_button = QToolButton(self)
        self.remove_button.setObjectName('TextEffectCloseButton')
        self.remove_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.remove_button.setToolTip(self.tr('Remove Eraser'))
        self.remove_button.setAccessibleName(self.tr('Remove Eraser'))
        self.remove_button.setFixedSize(18, 18)
        self.remove_button.clicked.connect(self.remove_requested.emit)

        action_widget = _effect_action_widget(self, (self.remove_button,))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        mode_label = QLabel(self.tr('Mode'), self)
        mode_label.setObjectName('TextEffectParamLabel')
        mode_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.mode_selector = BottomBorderComboBox(self)
        self.mode_selector.setObjectName('TextEffectParamEditor')
        self.mode_selector.addItem(self.tr('Erase'), 'erase')
        self.mode_selector.addItem(self.tr('Restore'), 'restore')
        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_changed
        )
        mode_widget = QWidget(self)
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_selector, 1)

        self.size_label = TransformDragLabel(
            self,
            direction=0,
            text=self.tr('Size'),
            alignment=(
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
            ),
        )
        self.size_label.setObjectName('TextEffectParamLabel')
        self.diameter_editor = QDoubleSpinBox(self)
        self.diameter_editor.setObjectName('TextEffectParamEditor')
        self.diameter_editor.setRange(1.0, 500.0)
        self.diameter_editor.setDecimals(1)
        self.diameter_editor.setSingleStep(1.0)
        self.diameter_editor.setSuffix(self.tr(' px'))
        button_symbols = getattr(
            QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox
        )
        self.diameter_editor.setButtonSymbols(button_symbols.NoButtons)
        self.diameter_editor.valueChanged.connect(
            self.diameter_changed.emit
        )
        self._diameter_drag_start: Optional[float] = None
        self.size_label.drag_started.connect(self._begin_diameter_drag)
        self.size_label.size_ctrl_changed.connect(
            self._change_diameter_by_drag
        )
        self.size_label.btn_released.connect(self._finish_diameter_drag)
        self.size_label.drag_canceled.connect(self._cancel_diameter_drag)
        size_widget = QWidget(self)
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(8)
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.diameter_editor, 1)

        self.clear_button = QToolButton(self)
        self.clear_button.setObjectName('TextAlphaMaskClearButton')
        self.clear_button.setText(self.tr('Clear'))
        self.clear_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.clear_button.clicked.connect(self.clear_requested.emit)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(mode_widget, 0, 0)
        controls.addWidget(size_widget, 0, 1)
        controls.addWidget(self.clear_button, 0, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)

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

    def _begin_diameter_drag(self) -> None:
        self._diameter_drag_start = self.diameter_editor.value()

    def _change_diameter_by_drag(self, delta: int) -> None:
        self.diameter_editor.setValue(
            self.diameter_editor.value()
            + delta * self.diameter_editor.singleStep()
        )

    def _finish_diameter_drag(self) -> None:
        self._diameter_drag_start = None

    def _cancel_diameter_drag(self) -> None:
        if self._diameter_drag_start is None:
            return
        value = self._diameter_drag_start
        self._diameter_drag_start = None
        self.diameter_editor.setValue(value)


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
    hollow_enabled_requested = Signal(bool)
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
        self.scrollContent.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.setMaximumHeight(self.MAX_CONTENT_HEIGHT)

        self.overall_opacity_control = EffectNumericControl(
            self.tr('Opacity'),
            'overall_opacity',
            100.0,
            0.0,
            1.0,
            '%',
            1.0,
            self.scrollContent,
            decimals=1,
        )
        overall_opacity_hint = self.tr(
            'Overall opacity of the text and all effects'
        )
        self.overall_opacity_control.label.setToolTip(overall_opacity_hint)
        self.overall_opacity_control.editor.setToolTip(overall_opacity_hint)
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
            QIcon(themed_icon_path('text-effect-alpha-mask.svg'))
        )
        self.mask_brush_button.setIconSize(QSize(16, 16))
        self.mask_brush_button.setFixedSize(26, 26)
        self.mask_brush_button.setCheckable(True)
        self.mask_brush_button.setEnabled(False)
        self.mask_brush_button.setToolTip(
            self.tr('Select one text block in text edit mode.')
        )
        self.mask_brush_button.setAccessibleName(
            self.tr('Text Eraser')
        )
        self.mask_brush_button.clicked.connect(
            self._on_mask_brush_clicked
        )

        self.hollow_toggle_button = QToolButton(self.scrollContent)
        self.hollow_toggle_button.setObjectName('TextEffectHollowButton')
        self.hollow_toggle_button.setIcon(
            QIcon(themed_icon_path('text-effect-hollow.svg'))
        )
        self.hollow_toggle_button.setIconSize(QSize(16, 16))
        self.hollow_toggle_button.setFixedSize(26, 26)
        self.hollow_toggle_button.setCheckable(True)
        self.hollow_toggle_button.setProperty('mixed', False)
        self.hollow_toggle_button.clicked.connect(
            self._on_hollow_toggled
        )
        self._set_hollow_toggle_state(False)

        self.add_effect_button = QToolButton(self.scrollContent)
        self.add_effect_button.setObjectName('AddTextEffectButton')
        self.add_effect_button.setText(self.tr('Add'))
        self.add_effect_button.setToolTip(self.tr('Add Effect'))
        self.add_effect_button.setAccessibleName(self.tr('Add Effect'))
        self.add_effect_button.setFixedSize(72, 26)
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
            (self.tr('Glow'), 'glow', 'text-effect-glow.svg'),
            (
                self.tr('Gradient'),
                'gradient_overlay',
                'text-effect-gradient.svg',
            ),
        ):
            action = add_menu.addAction(
                QIcon(themed_icon_path(icon_name)), label
            )
            action.setData(effect_type)
            action.triggered.connect(self._on_add_effect_triggered)
            self.add_effect_actions[effect_type] = action
        self.add_effect_button.setMenu(add_menu)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self.add_effect_button)
        top_row.addWidget(self.mask_brush_button)
        top_row.addWidget(self.hollow_toggle_button)
        top_row.addStretch()
        top_row.addWidget(self.overall_opacity_control)

        self.mixed_label = QLabel(self.tr('Mixed'), self.scrollContent)
        self.mixed_label.setObjectName('TextEffectMixedLabel')
        self.mixed_label.setVisible(False)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.effect_cards = []
        self.stroke_cards = []
        self.shadow_cards = []
        self.glow_cards = []
        self.gradient_overlay_card = None
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
            card.show()
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
            self.tr('Edit Text Eraser')
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
        self.glow_cards = []
        self.gradient_overlay_card = None

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
            elif effect_type == 'glow':
                card = GlowEffectCard(index, self.scrollContent)
                self.glow_cards.append(card)
            elif effect_type == 'hollow':
                continue
            elif effect_type == 'gradient_overlay':
                card = GradientOverlayEffectCard(index, self.scrollContent)
                self.gradient_overlay_card = card
            else:
                continue
            card.value_commit_requested.connect(
                self.value_commit_requested.emit
            )
            if isinstance(
                card,
                (
                    StrokeEffectCard,
                    ShadowEffectCard,
                    GlowEffectCard,
                    GradientOverlayEffectCard,
                ),
            ):
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
                card.color_dialog_active_changed.connect(
                    self.color_dialog_active_changed.emit
                )
            if isinstance(
                card, (StrokeEffectCard, ShadowEffectCard, GlowEffectCard)
            ):
                card.move_requested.connect(self.move_effect_requested.emit)
            card.remove_requested.connect(self.remove_effect_requested.emit)
            self.cards_layout.addWidget(card)
            card.show()
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

        hollow_values = [
            next(
                (
                    effect.enabled
                    for effect in state.effects
                    if isinstance(effect, HollowEffect)
                ),
                False,
            )
            for state in states
        ]
        common_hollow = (
            hollow_values[0]
            if all(value == hollow_values[0] for value in hollow_values)
            else None
        )
        self._set_hollow_toggle_state(common_hollow)

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
            gradient_visibility_changed = False
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
                gradient_editor = getattr(card, 'gradient_editor', None)
                gradient_was_hidden = (
                    gradient_editor.isHidden()
                    if isinstance(
                        card,
                        (
                            StrokeEffectCard,
                            ShadowEffectCard,
                            GlowEffectCard,
                        ),
                    )
                    else None
                )
                card.set_values(values)
                if (
                    gradient_was_hidden is not None
                    and gradient_editor.isHidden() != gradient_was_hidden
                ):
                    gradient_visibility_changed = True
                if isinstance(
                    card,
                    (StrokeEffectCard, ShadowEffectCard, GlowEffectCard),
                ):
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
            if gradient_visibility_changed:
                self.cards_layout.invalidate()
                self.content_layout.invalidate()
        self.add_effect_actions['gradient_overlay'].setEnabled(
            not mixed and common_sequence is not None
            and 'gradient_overlay' not in common_sequence
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

    def iter_gradient_editors(self) -> Iterator[InlineLinearGradientEditor]:
        for card in self.effect_cards:
            editor = getattr(card, 'gradient_editor', None)
            if isinstance(editor, InlineLinearGradientEditor):
                yield editor

    def finish_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.commit_pending()
        for editor in tuple(self.iter_gradient_editors()):
            editor.commit_pending()

    def cancel_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.cancel_pending()
        for editor in tuple(self.iter_gradient_editors()):
            editor.cancel_pending()

    def cancel_effect_previews(self) -> None:
        for control in self.iter_controls():
            control.cancel_preview()
        for editor in tuple(self.iter_gradient_editors()):
            editor.cancel_pending()

    def _sync_content_height(self) -> None:
        if not hasattr(self, 'content_layout'):
            return
        self._sync_scroll_content_height(self.content_layout)

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
            'stroke', 'shadow', 'glow', 'gradient_overlay'
        }:
            self.add_effect_requested.emit(action.data())

    def _set_hollow_toggle_state(
        self, enabled: Optional[bool]
    ) -> None:
        mixed = enabled is None
        blocker = QSignalBlocker(self.hollow_toggle_button)
        self.hollow_toggle_button.setChecked(enabled is True)
        del blocker
        if self.hollow_toggle_button.property('mixed') != mixed:
            self.hollow_toggle_button.setProperty('mixed', mixed)
            style = self.hollow_toggle_button.style()
            style.unpolish(self.hollow_toggle_button)
            style.polish(self.hollow_toggle_button)
        if mixed:
            description = self.tr('Enable Hollow for All Selected Text')
        elif enabled is True:
            description = self.tr('Disable Hollow')
        else:
            description = self.tr('Enable Hollow')
        self.hollow_toggle_button.setToolTip(description)
        self.hollow_toggle_button.setAccessibleName(description)

    def _on_hollow_toggled(self, enabled: bool) -> None:
        self.hollow_enabled_requested.emit(enabled)

    def _on_mask_brush_clicked(self, checked: bool) -> None:
        self.mask_edit_requested.emit(checked)
