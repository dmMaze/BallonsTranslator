"""Expandable controls for item-wide text effects."""

from typing import Iterator, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QTimer, Signal, QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.text_effects import (
    SolidPaint,
    StrokeEffect,
    TextEffectStack,
)

from ...custom_widget import ColorPickerLabel, PanelArea
from ...misc import themed_icon_path
from ..transforms.controls import CommittedTransformControl

if TYPE_CHECKING:
    from ..item import TextBlkItem


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

        self.enabled_checkbox = QCheckBox(self.tr('Enabled'), self)
        self.enabled_checkbox.setObjectName('TextEffectEnabledCheckBox')
        self.enabled_checkbox.clicked.connect(self._on_enabled_clicked)

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

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(5)
        header.addWidget(self.enabled_checkbox)
        header.addWidget(self.title_label)
        header.addWidget(self.move_up_button)
        header.addWidget(self.move_down_button)
        header.addWidget(self.delete_button)

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

    def set_index(self, index: int) -> None:
        self.index = int(index)

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
        with QSignalBlocker(self.enabled_checkbox):
            self.enabled_checkbox.setTristate(enabled is None)
            if enabled is None:
                self.enabled_checkbox.setCheckState(
                    Qt.CheckState.PartiallyChecked
                )
            else:
                self.enabled_checkbox.setChecked(enabled)

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


class TextEffectPanel(PanelArea):
    """Own Overall Opacity and repeatable solid Stroke cards.

    >>> TextEffectPanel.__name__
    'TextEffectPanel'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    add_stroke_requested = Signal()
    remove_stroke_requested = Signal(int)
    move_stroke_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

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
        self.mask_brush_button.setEnabled(False)
        self.mask_brush_button.setToolTip(
            self.tr('Alpha mask brush is not available yet.')
        )
        self.mask_brush_button.setAccessibleName(
            self.tr('Alpha Mask Brush')
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
        stroke_action = add_menu.addAction(self.tr('Stroke'))
        stroke_action.setData('stroke')
        stroke_action.triggered.connect(self._on_add_effect_triggered)
        self.add_effect_button.setMenu(add_menu)

        self.mixed_label = QLabel(self.tr('Mixed'), self.scrollContent)
        self.mixed_label.setObjectName('TextEffectMixedLabel')
        self.mixed_label.setVisible(False)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.stroke_cards = []
        self._effect_types = None

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(top_row)
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

    def _clear_stroke_cards(self) -> None:
        for card in self.stroke_cards:
            self.cards_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.stroke_cards = []

    def _rebuild_stroke_cards(self, effect_types: Sequence[str]) -> None:
        effect_types = tuple(effect_types)
        if effect_types == self._effect_types:
            return
        self._clear_stroke_cards()
        self._effect_types = effect_types
        for index, effect_type in enumerate(effect_types):
            if effect_type != 'stroke':
                continue
            card = StrokeEffectCard(index, self.scrollContent)
            card.value_commit_requested.connect(
                self.value_commit_requested.emit
            )
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
            card.remove_requested.connect(self.remove_stroke_requested.emit)
            card.move_requested.connect(self.move_stroke_requested.emit)
            card.color_dialog_active_changed.connect(
                self.color_dialog_active_changed.emit
            )
            self.cards_layout.addWidget(card)
            self.stroke_cards.append(card)
        for position, card in enumerate(self.stroke_cards):
            card.set_move_enabled(
                position > 0, position + 1 < len(self.stroke_cards)
            )

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
            self._rebuild_stroke_cards(())
        else:
            self._rebuild_stroke_cards(common_sequence)
            for card in self.stroke_cards:
                card.set_values(
                    [state.effects[card.index] for state in states]
                )
        self._sync_content_height()

    def set_active_format(self, font_format: FontFormat) -> None:
        self._set_effect_states([font_format.text_effects])

    def set_effect_items(self, items: Sequence["TextBlkItem"]) -> None:
        self._set_effect_states(
            [item.blk.fontformat.text_effects for item in items]
        )

    def iter_controls(self) -> Iterator[EffectNumericControl]:
        yield self.overall_opacity_control
        for card in self.stroke_cards:
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
        if action is not None and action.data() == 'stroke':
            self.add_stroke_requested.emit()
