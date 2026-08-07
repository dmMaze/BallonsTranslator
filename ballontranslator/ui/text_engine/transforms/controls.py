"""Committed numeric editors for text-transform variants."""

import math

from qtpy.QtCore import QEvent, QPoint, QRect, QSize, Signal, Qt
from qtpy.QtGui import QColor, QIcon, QKeyEvent, QPainter
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.fontformat import TEXT_TRANSFORM_PRECISION

from ...custom_widget import SmallSizeControlLabel
from ...icon_rendering import render_svg_pixmap
from ...misc import themed_icon_path


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
            super().mousePressEvent(event)
            # This label owns the drag. Letting QLabel ignore the press makes
            # it bubble to the card, which immediately toggles selection off.
            event.accept()
            return
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
    """Line edit with the transform-panel logical width contract."""

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(56)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(56)
        return hint


class _TransformIntegerEdit(_TransformValueEdit):
    """Text editor with the same compact SVG steppers as page ranges.

    >>> _TransformIntegerEdit.__name__
    '_TransformIntegerEdit'
    """

    step_requested = Signal(int)
    ICON_SIZE = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty('integerStepper', True)
        self.setMouseTracking(True)
        self._hover_button = ''

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setWidth(80)
        return hint

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        hint.setWidth(80)
        return hint

    def _button_rects(self):
        button_size = 16
        right = self.width() - 4
        y = (self.height() - button_size) // 2
        up_rect = QRect(right - button_size, y, button_size, button_size)
        down_rect = QRect(
            up_rect.left() - button_size - 1,
            y,
            button_size,
            button_size,
        )
        return up_rect, down_rect

    @staticmethod
    def _event_pos(event) -> QPoint:
        if hasattr(event, 'position'):
            return event.position().toPoint()
        return event.pos()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        up_rect, down_rect = self._button_rects()
        for name, rect, icon_name in (
            ('down', down_rect, 'chevron-down.svg'),
            ('up', up_rect, 'chevron-up.svg'),
        ):
            if self._hover_button == name and self.isEnabled():
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(30, 147, 229, 32))
                painter.drawRoundedRect(rect, 3, 3)
            pixmap = render_svg_pixmap(
                themed_icon_path(icon_name),
                self.ICON_SIZE,
                self.ICON_SIZE,
                self.devicePixelRatioF(),
            )
            painter.drawPixmap(
                rect.center().x() - self.ICON_SIZE // 2,
                rect.center().y() - self.ICON_SIZE // 2,
                pixmap,
            )
        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self._event_pos(event)
            up_rect, down_rect = self._button_rects()
            if up_rect.contains(pos) or down_rect.contains(pos):
                self.step_requested.emit(1 if up_rect.contains(pos) else -1)
                event.accept()
                return
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self._event_pos(event)
        up_rect, down_rect = self._button_rects()
        hovered = (
            'up' if up_rect.contains(pos)
            else 'down' if down_rect.contains(pos)
            else ''
        )
        if hovered != self._hover_button:
            self._hover_button = hovered
            self.update()
        return super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_button:
            self._hover_button = ''
            self.update()
        return super().leaveEvent(event)

class CommittedTransformControl(QWidget):
    """One committed numeric transform editor."""

    IDLE = 'IDLE'
    PENDING_TEXT = 'PENDING_TEXT'
    DRAG_PREVIEW = 'DRAG_PREVIEW'

    commit_requested = Signal(str, object)
    preview_requested = Signal(str, object)
    drag_commit_requested = Signal(str, object)
    preview_canceled = Signal(str)
    user_interacted = Signal()

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
        decimals: int = 1,
    ):
        super().__init__(parent)
        self.setObjectName('TextTransformControl')
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
        self.decimals = max(0, int(decimals))
        self.state = self.IDLE
        self._model_value = None
        self._model_values = ()
        self._drag_delta = 0.0
        self._drag_remainder = 0.0

        self.label = TransformDragLabel(
            self,
            direction=0,
            text=title,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        self.label.setObjectName('TextTransformParamLabel')
        self.label.setWordWrap(True)
        self.label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.editor = (
            _TransformIntegerEdit(self)
            if self.decimals == 0
            else _TransformValueEdit(self)
        )
        self.editor.setObjectName('TextTransformParamEditor')
        self.editor.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.editor.setFixedSize(80 if self.decimals == 0 else 56, 22)
        self.editor.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        self.editor.textEdited.connect(self._on_text_edited)
        self.editor.returnPressed.connect(self.commit_pending)
        self.editor.installEventFilter(self)
        if isinstance(self.editor, _TransformIntegerEdit):
            self.editor.step_requested.connect(self._step_integer)

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

    def _display_to_canonical(self, value: float) -> int | float:
        canonical_value = float(value) / self.display_factor
        if self.decimals == 0:
            return int(canonical_value)
        canonical_value = round(canonical_value, TEXT_TRANSFORM_PRECISION)
        return 0.0 if canonical_value == 0.0 else canonical_value

    def _format(self, canonical_value: float) -> str:
        return (
            f'{self._canonical_to_display(canonical_value):.{self.decimals}f}'
            f'{self.suffix}'
        )

    def _parse(self, text: str) -> int | float:
        text = text.strip()
        if self.suffix and text.endswith(self.suffix):
            text = text[:-len(self.suffix)].strip()
        display_value = float(text)
        unrounded_value = display_value / self.display_factor
        if (
            not math.isfinite(unrounded_value)
            or (
                self.decimals == 0
                and not unrounded_value.is_integer()
            )
        ):
            raise ValueError
        canonical_value = self._display_to_canonical(display_value)
        if (
            not self.canonical_minimum
            <= canonical_value
            <= self.canonical_maximum
        ):
            raise ValueError
        if self.decimals == 0:
            return canonical_value
        return 0.0 if canonical_value == 0.0 else canonical_value

    def _restore_display(self):
        self.editor.setText(
            '\N{EM DASH}'
            if self._model_value is None
            else self._format(self._model_value)
        )

    def set_model_value(self, canonical_value, model_values=None):
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._drag_remainder = 0.0
        self._model_value = canonical_value
        self._model_values = (
            (() if canonical_value is None else (canonical_value,))
            if model_values is None
            else tuple(model_values)
        )
        self._restore_display()

    def _on_text_edited(self):
        self.user_interacted.emit()
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
        self._model_values = (canonical_value,)
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
            if (
                event.type() == QEvent.Type.KeyPress
                and event.key() == Qt.Key.Key_Escape
            ):
                self.cancel_pending()
                event.accept()
                return True
            if event.type() == QEvent.Type.FocusOut:
                self.commit_pending()
        return super().eventFilter(watched, event)

    def _start_drag(self):
        self.user_interacted.emit()
        self.commit_pending()
        self.state = self.DRAG_PREVIEW
        self._drag_delta = 0.0
        self._drag_remainder = 0.0

    def _drag_limits(self):
        if not self._model_values:
            return None
        canonical_minimum = max(
            self.canonical_minimum - value for value in self._model_values
        )
        canonical_maximum = min(
            self.canonical_maximum - value for value in self._model_values
        )
        limits = (
            self._canonical_to_display(canonical_minimum),
            self._canonical_to_display(canonical_maximum),
        )
        return min(limits), max(limits)

    def _move_drag(self, delta: int):
        if self.state != self.DRAG_PREVIEW:
            self._start_drag()
        movement = float(delta) * self.drag_step
        limits = self._drag_limits()
        if limits is not None and (
            (self._drag_delta <= limits[0] and movement < 0.0)
            or (self._drag_delta >= limits[1] and movement > 0.0)
        ):
            # Discard outward overshoot so reversing responds immediately.
            self._drag_remainder = 0.0
            movement = 0.0
        if self.decimals == 0:
            self._drag_remainder += movement
            whole_steps = math.trunc(self._drag_remainder)
            self._drag_remainder -= whole_steps
            candidate = self._drag_delta + whole_steps
        else:
            candidate = self._drag_delta + movement
        if limits is not None:
            clamped = min(max(candidate, limits[0]), limits[1])
            if clamped != candidate:
                self._drag_remainder = 0.0
            candidate = clamped
        self._drag_delta = candidate
        canonical_delta = self._display_to_canonical(self._drag_delta)
        if self._model_value is None:
            self.editor.setText(
                f'\N{GREEK CAPITAL LETTER DELTA} '
                f'{self._drag_delta:+.1f}{self.suffix}'
            )
        else:
            preview_value = self._model_value + canonical_delta
            self.editor.setText(self._format(preview_value))
        self.preview_requested.emit(
            self.param_name,
            canonical_delta,
        )

    def _finish_drag(self):
        if self.state != self.DRAG_PREVIEW:
            return
        delta = self._drag_delta
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._drag_remainder = 0.0
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
        self._drag_remainder = 0.0
        self._restore_display()
        self.preview_canceled.emit(self.param_name)

    def _step_integer(self, direction: int):
        self.user_interacted.emit()
        canonical_step = self._display_to_canonical(
            1.0 if direction > 0 else -1.0
        )
        if self.state == self.PENDING_TEXT:
            try:
                canonical_value = self._parse(self.editor.text())
            except (TypeError, ValueError):
                self.cancel_pending()
                return
            canonical_value = min(
                max(canonical_value + canonical_step, self.canonical_minimum),
                self.canonical_maximum,
            )
            self.state = self.IDLE
            self._model_value = canonical_value
            self._model_values = (canonical_value,)
            self._restore_display()
            self.commit_requested.emit(self.param_name, canonical_value)
            return
        if not self._model_values:
            return
        display_delta = self._canonical_to_display(canonical_step)
        limits = self._drag_limits()
        if limits is not None:
            display_delta = min(max(display_delta, limits[0]), limits[1])
        if display_delta:
            canonical_delta = self._display_to_canonical(display_delta)
            self.preview_requested.emit(self.param_name, canonical_delta)
            self.drag_commit_requested.emit(
                self.param_name,
                canonical_delta,
            )


class CommittedTransformChoiceControl(QWidget):
    """One immediately committed transform choice."""

    commit_requested = Signal(str, object)
    user_interacted = Signal()

    def __init__(self, title, param_name, choices, parent=None):
        super().__init__(parent)
        self.setObjectName('TextTransformControl')
        self.param_name = param_name
        self.choices = tuple(choices)
        self.label = QLabel(title, self)
        self.label.setObjectName('TextTransformParamLabel')
        self.label.setWordWrap(True)
        self.combobox = QComboBox(self)
        self.combobox.setObjectName('TextTransformParamEditor')
        for value, label in self.choices:
            self.combobox.addItem(label(), value)
        self.combobox.activated.connect(self._commit_index)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        layout.addWidget(self.combobox)

    def _commit_index(self, index):
        self.user_interacted.emit()
        self.commit_requested.emit(
            self.param_name, self.combobox.itemData(index)
        )

    def set_model_value(self, value):
        index = self.combobox.findData(value)
        self.combobox.setCurrentIndex(index)

    def cancel_pending(self):
        pass

    def cancel_preview(self):
        pass

    def commit_pending(self):
        return False


class TransformParameterPanel(QFrame):
    """One indexed transform operation with independently owned controls.

    >>> TransformParameterPanel.__name__
    'TransformParameterPanel'
    """

    commit_requested = Signal(int, str, object)
    preview_requested = Signal(int, str, object)
    drag_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    card_clicked = Signal(int)
    selected = Signal(int)

    def __init__(self, index, variant, parent=None):
        super().__init__(parent)
        self.index = int(index)
        self._hovered = False
        self._selected = False
        self.setObjectName('TextTransformParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = QLabel(self)
        self.title_icon_label.setObjectName('TextTransformParameterIcon')
        self.title_icon_label.setFixedSize(16, 16)
        self.title_icon_label.setPixmap(render_svg_pixmap(
            themed_icon_path(variant.icon_name),
            16,
            16,
            self.devicePixelRatioF(),
        ))

        self.title_label = QLabel(variant.label(), self)
        self.title_label.setObjectName('TextTransformParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.move_up_button = QToolButton(self)
        self.move_up_button.setObjectName('TextTransformMoveButton')
        self.move_up_button.setIcon(
            QIcon(themed_icon_path('chevron-up.svg'))
        )
        self.move_up_button.setToolTip(self.tr('Move Up'))
        self.move_up_button.setAccessibleName(self.tr('Move Up'))
        self.move_up_button.clicked.connect(
            lambda: self.move_requested.emit(self.index, -1)
        )

        self.move_down_button = QToolButton(self)
        self.move_down_button.setObjectName('TextTransformMoveButton')
        self.move_down_button.setIcon(
            QIcon(themed_icon_path('chevron-down.svg'))
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
        self.action_widget = action_widget
        action_layout = QHBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(4)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.close_button,
        ):
            button.setFixedSize(18, 18)
            button.setIconSize(QSize(12, 12))
            action_layout.addWidget(button)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_layout.addWidget(self.title_icon_label)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(action_widget)

        self.controls = {}
        controls_widget = QWidget(self)
        controls_widget.setObjectName('TextTransformPanelControls')
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        grouped_controls = {}
        for spec in variant.controls:
            if spec.choices:
                control = CommittedTransformChoiceControl(
                    spec.label(),
                    spec.attribute_name,
                    spec.choices,
                    controls_widget,
                )
            else:
                control = CommittedTransformControl(
                    spec.label(),
                    spec.attribute_name,
                    spec.factor,
                    spec.minimum,
                    spec.maximum,
                    spec.suffix,
                    0.125 if spec.decimals == 0 else 1.0,
                    controls_widget,
                    decimals=spec.decimals,
                )
                if spec.shortcut is not None:
                    shortcut = spec.shortcut()
                    control.label.setToolTip(shortcut)
                    control.editor.setToolTip(shortcut)
            control.layout().setSpacing(8)
            control.layout().setStretch(0, 1)
            control.layout().setStretch(1, 2)
            control.label.setWordWrap(False)
            control.label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            editor = (
                control.combobox
                if isinstance(control, CommittedTransformChoiceControl)
                else control.editor
            )
            editor.setProperty('cardEditor', True)
            editor.setMinimumWidth(0)
            editor.setMaximumWidth(16777215)
            editor.setFixedHeight(22)
            editor.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            control.commit_requested.connect(
                lambda name, value, self=self:
                self.commit_requested.emit(self.index, name, value)
            )
            control.user_interacted.connect(
                lambda self=self: self.selected.emit(self.index)
            )
            if isinstance(control, CommittedTransformControl):
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
            section = spec.section() if spec.section is not None else None
            grouped_controls.setdefault(section, []).append((spec, control))

        section_order = ([None] if None in grouped_controls else []) + [
            section for section in grouped_controls if section is not None
        ]
        self.section_labels = []
        self.control_grids = []
        for section in section_order:
            if section is not None:
                section_label = QLabel(section, controls_widget)
                section_label.setObjectName('TextTransformSectionTitle')
                controls_layout.addWidget(section_label)
                self.section_labels.append(section_label)
            grid = QGridLayout()
            grid.setContentsMargins(4 if section is not None else 0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)
            section_controls = grouped_controls[section]
            column_count = max(
                spec.section_columns for spec, _control in section_controls
            )
            for column in range(column_count):
                grid.setColumnStretch(column, 1)
            for control_index, (_spec, control) in enumerate(section_controls):
                control.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
                grid.addWidget(
                    control,
                    control_index // column_count,
                    control_index % column_count,
                )
            controls_layout.addLayout(grid)
            self.control_grids.append(grid)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 12, 8)
        layout.setSpacing(6)
        layout.addLayout(header_layout)
        layout.addWidget(controls_widget)

        self._sync_action_visibility()

    def set_index(self, index: int) -> None:
        self.index = int(index)

    def set_move_enabled(self, can_move_up: bool, can_move_down: bool) -> None:
        self.move_up_button.setEnabled(can_move_up)
        self.move_down_button.setEnabled(can_move_down)

    def set_selected(self, selected: bool) -> None:
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty('selected', selected)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_values(self, transforms) -> None:
        for name, control in self.controls.items():
            values = [getattr(transform, name) for transform in transforms]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            if isinstance(control, CommittedTransformControl):
                control.set_model_value(common, values)
            else:
                control.set_model_value(common)

    def iter_controls(self):
        return self.controls.values()

    def cancel_pending(self) -> None:
        for control in self.controls.values():
            control.cancel_pending()

    def _sync_action_visibility(self) -> None:
        self.action_widget.setVisible(self._hovered)
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.card_clicked.emit(self.index)
        return super().mousePressEvent(event)
