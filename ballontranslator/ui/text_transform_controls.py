"""Committed numeric editors for text-transform variants."""

import math

from qtpy.QtCore import QEvent, Signal, Qt
from qtpy.QtGui import QIcon, QKeyEvent
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .adaptive_wrap_layout import AdaptiveWrapLayout
from .custom_widget import SmallSizeControlLabel
from .misc import themed_icon_path


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

    def abort_drag_session(self):
        self.mouse_pressed = False

    def event(self, event):
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and self.mouse_pressed
            and event.key() == Qt.Key.Key_Escape
        ):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape and self.mouse_pressed:
            self.mouse_pressed = False
            self.drag_canceled.emit()
            event.accept()
            return
        return super().keyPressEvent(event)


class _TransformValueEdit(QLineEdit):
    """Line edit with the Advanced-panel logical width contract."""

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(84)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(64)
        return hint


class CommittedTransformControl(QWidget):
    """One Advanced-panel-only committed numeric transform editor."""

    IDLE = 'IDLE'
    PENDING_TEXT = 'PENDING_TEXT'
    DRAG_PREVIEW = 'DRAG_PREVIEW'

    commit_requested = Signal(str, float)
    preview_requested = Signal(str, float)
    drag_commit_requested = Signal(str, float)
    preview_canceled = Signal(str)

    def __init__(
        self,
        title: str,
        param_name: str,
        display_factor: float,
        canonical_minimum: float,
        canonical_maximum: float,
        suffix: str,
        drag_step: float,
        parent=None,
    ):
        super().__init__(parent)
        if display_factor == 0 or not math.isfinite(display_factor):
            raise ValueError('display_factor must be finite and non-zero')
        if canonical_minimum > canonical_maximum:
            raise ValueError('canonical minimum must not exceed maximum')
        if drag_step <= 0 or not math.isfinite(drag_step):
            raise ValueError('drag_step must be finite and positive')
        self.param_name = param_name
        self.display_factor = float(display_factor)
        self.canonical_minimum = float(canonical_minimum)
        self.canonical_maximum = float(canonical_maximum)
        self.suffix = suffix
        self.drag_step = float(drag_step)
        self.state = self.IDLE
        self._model_value = None
        self._drag_delta = 0.0

        self.label = TransformDragLabel(
            self,
            direction=0,
            text=title,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        self.label.setWordWrap(True)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.editor = _TransformValueEdit(self)
        self.editor.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.editor.setMinimumWidth(64)
        self.editor.setMaximumWidth(84)
        self.editor.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
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
        return value * self.display_factor

    def _display_to_canonical(self, value: float) -> float:
        return value / self.display_factor

    def _format(self, canonical_value: float) -> str:
        return f'{self._canonical_to_display(canonical_value):.1f}{self.suffix}'

    def _parse(self, text: str) -> float:
        text = text.strip()
        if self.suffix and text.endswith(self.suffix):
            text = text[:-len(self.suffix)].strip()
        canonical_value = self._display_to_canonical(float(text))
        if (
            not math.isfinite(canonical_value)
            or not self.canonical_minimum
            <= canonical_value
            <= self.canonical_maximum
        ):
            raise ValueError
        return 0.0 if canonical_value == 0.0 else canonical_value

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
            if (
                event.type() == QEvent.Type.ShortcutOverride
                and event.key() == Qt.Key.Key_Escape
                and self.state == self.PENDING_TEXT
            ):
                event.accept()
                return True
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
        self._drag_delta += float(delta) * self.drag_step
        if self._model_value is None:
            self.editor.setText(
                f'\N{GREEK CAPITAL LETTER DELTA} '
                f'{self._drag_delta:+.1f}{self.suffix}'
            )
        else:
            canonical_delta = self._display_to_canonical(self._drag_delta)
            preview_value = min(
                max(
                    self._model_value + canonical_delta,
                    self.canonical_minimum,
                ),
                self.canonical_maximum,
            )
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
        self.label.abort_drag_session()
        if self.state != self.DRAG_PREVIEW:
            return
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._restore_display()
        self.preview_canceled.emit(self.param_name)


class TransformParameterPanel(QFrame):
    """One indexed transform operation with independently owned controls.

    >>> TransformParameterPanel.__name__
    'TransformParameterPanel'
    """

    commit_requested = Signal(int, str, float)
    preview_requested = Signal(int, str, float)
    drag_commit_requested = Signal(int, str, float)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)

    def __init__(self, index, variant, parent=None):
        super().__init__(parent)
        self.index = int(index)
        self.variant = variant
        self._hovered = False
        self.setObjectName('TextTransformParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        title = QLabel(variant.label(), self)
        title.setObjectName('TextTransformParameterTitle')
        title.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        standard = getattr(QStyle, 'StandardPixmap', QStyle)
        self.move_up_button = QToolButton(self)
        self.move_up_button.setObjectName('TextTransformMoveButton')
        self.move_up_button.setIcon(
            self.style().standardIcon(standard.SP_ArrowUp)
        )
        self.move_up_button.setToolTip(self.tr('Move Up'))
        self.move_up_button.setAccessibleName(self.tr('Move Up'))
        self.move_up_button.clicked.connect(
            lambda: self.move_requested.emit(self.index, -1)
        )

        self.move_down_button = QToolButton(self)
        self.move_down_button.setObjectName('TextTransformMoveButton')
        self.move_down_button.setIcon(
            self.style().standardIcon(standard.SP_ArrowDown)
        )
        self.move_down_button.setToolTip(self.tr('Move Down'))
        self.move_down_button.setAccessibleName(self.tr('Move Down'))
        self.move_down_button.clicked.connect(
            lambda: self.move_requested.emit(self.index, 1)
        )

        self.close_button = QToolButton(self)
        self.close_button.setObjectName('TextTransformCloseButton')
        self.close_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.close_button.setToolTip(self.tr('Delete Transform'))
        self.close_button.setAccessibleName(self.tr('Delete Transform'))
        self.close_button.clicked.connect(
            lambda: self.remove_requested.emit(self.index)
        )

        action_widget = QWidget(self)
        action_widget.setObjectName('TextTransformPanelActions')
        action_widget.setFixedWidth(66)
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.close_button,
        ):
            button.setFixedSize(18, 18)
            action_layout.addWidget(button)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(title)
        header_layout.addWidget(action_widget)

        self.controls = {}
        controls_widget = QWidget(self)
        controls_widget.setObjectName('TextTransformPanelControls')
        controls_layout = AdaptiveWrapLayout(controls_widget)
        for spec in variant.controls:
            control = CommittedTransformControl(
                spec.label(),
                spec.attribute_name,
                spec.factor,
                spec.minimum,
                spec.maximum,
                spec.suffix,
                1.0,
                controls_widget,
            )
            control.commit_requested.connect(
                lambda name, value, self=self:
                self.commit_requested.emit(self.index, name, value)
            )
            control.preview_requested.connect(
                lambda name, value, self=self:
                self.preview_requested.emit(self.index, name, value)
            )
            control.drag_commit_requested.connect(
                lambda name, value, self=self:
                self.drag_commit_requested.emit(self.index, name, value)
            )
            control.preview_canceled.connect(
                lambda name, self=self:
                self.preview_canceled.emit(self.index, name)
            )
            self.controls[spec.attribute_name] = control
            controls_layout.addWidget(control)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(header_layout)
        layout.addWidget(controls_widget)

        self._sync_action_visibility()

    def set_index(self, index: int) -> None:
        self.index = int(index)

    def set_move_enabled(self, can_move_up: bool, can_move_down: bool) -> None:
        self.move_up_button.setEnabled(can_move_up)
        self.move_down_button.setEnabled(can_move_down)

    def set_values(self, transforms) -> None:
        for name, control in self.controls.items():
            values = [getattr(transform, name) for transform in transforms]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common)

    def iter_controls(self):
        return self.controls.values()

    def cancel_pending(self) -> None:
        for control in self.controls.values():
            control.cancel_pending()

    def cancel_previews(self) -> None:
        for control in self.controls.values():
            control.cancel_preview()

    def finish_pending(self) -> None:
        for control in self.controls.values():
            control.commit_pending()

    def _sync_action_visibility(self) -> None:
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.close_button,
        ):
            button.setVisible(self._hovered)

    def enterEvent(self, event):
        self._hovered = True
        self._sync_action_visibility()
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._sync_action_visibility()
        return super().leaveEvent(event)
