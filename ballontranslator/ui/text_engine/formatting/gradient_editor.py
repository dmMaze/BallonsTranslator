"""Inline controls for immutable linear-gradient effect paints."""

from dataclasses import replace
from typing import Optional, Tuple

from qtpy.QtCore import QEvent, QPointF, QRectF, QSignalBlocker, Signal, Qt
from qtpy.QtGui import (
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
)
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QColorDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.text_effects import GradientStop, LinearGradientPaint

from ..rendering.effect_paint import paint_effect_paint_preview


def _mouse_position(event: QMouseEvent) -> QPointF:
    if hasattr(event, 'position'):
        return event.position()
    return event.localPos()


class GradientStopBar(QWidget):
    """Render and edit the ordered stops of one linear gradient.

    >>> GradientStopBar.__name__
    'GradientStopBar'
    """

    paint_previewed = Signal(object)
    paint_commit_requested = Signal(object)
    paint_preview_canceled = Signal()
    selection_changed = Signal(int)

    HANDLE_RADIUS = 5.0

    def __init__(
        self,
        paint: LinearGradientPaint,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._paint = paint
        self._selected_index = 0
        self._mixed = False
        self._dragging = False
        self._drag_start_paint: Optional[LinearGradientPaint] = None
        self.setMinimumHeight(42)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(self.tr('Gradient Stops'))
        self._edit_tooltip = self.tr(
            'Click the strip to add a stop; drag a stop to move it'
        )
        self.setToolTip(self._edit_tooltip)

    @property
    def paint(self) -> LinearGradientPaint:
        return self._paint

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def interaction_active(self) -> bool:
        return getattr(self, '_drag_start_paint', None) is not None

    def set_paint(
        self,
        paint: LinearGradientPaint,
        selected_index: Optional[int] = None,
    ) -> None:
        self._paint = paint
        if selected_index is not None:
            self._selected_index = max(
                0, min(int(selected_index), len(paint.stops) - 1)
            )
        else:
            self._selected_index = min(
                self._selected_index, len(paint.stops) - 1
            )
        self.update()

    def set_mixed(self, mixed: bool) -> None:
        self._mixed = bool(mixed)
        self.setAccessibleName(
            self.tr('Mixed Gradient Stops')
            if self._mixed else self.tr('Gradient Stops')
        )
        self.setToolTip(
            self.tr('Mixed Gradient') if self._mixed else self._edit_tooltip
        )
        self.update()

    def end_interaction(self) -> None:
        """Forget an unfinished pointer gesture after its owner resolves it."""
        self._drag_start_paint = None
        self._dragging = False

    def select_stop(self, index: int) -> None:
        index = max(0, min(int(index), len(self._paint.stops) - 1))
        if index == self._selected_index:
            return
        self._selected_index = index
        self.selection_changed.emit(index)
        self.update()

    def add_stop(self, position: float) -> bool:
        """Insert an interpolated stop, up to the persisted 32-stop limit.

        >>> callable(GradientStopBar.add_stop)
        True
        """
        if len(self._paint.stops) >= 32:
            return False
        position = max(0.0, min(float(position), 1.0))
        stops = self._paint.stops
        right_index = next(
            (
                index
                for index, stop in enumerate(stops)
                if stop.position >= position
            ),
            len(stops) - 1,
        )
        left_index = max(0, right_index - 1)
        left = stops[left_index]
        right = stops[right_index]
        distance = right.position - left.position
        ratio = (
            0.0 if distance <= 0.0
            else (position - left.position) / distance
        )
        ratio = max(0.0, min(ratio, 1.0))
        color = tuple(
            round(start + (end - start) * ratio)
            for start, end in zip(left.color, right.color)
        )
        opacity = left.opacity + (right.opacity - left.opacity) * ratio
        insert_at = next(
            (
                index
                for index, stop in enumerate(stops)
                if stop.position > position
            ),
            len(stops),
        )
        updated = list(stops)
        updated.insert(insert_at, GradientStop(position, color, opacity))
        self._selected_index = insert_at
        self._set_stops(tuple(updated))
        self.selection_changed.emit(insert_at)
        return True

    def move_selected(self, position: float) -> None:
        index = self._selected_index
        stops = self._paint.stops
        minimum = stops[index - 1].position if index > 0 else 0.0
        maximum = (
            stops[index + 1].position
            if index + 1 < len(stops) else 1.0
        )
        position = max(minimum, min(float(position), maximum))
        updated = list(stops)
        updated[index] = replace(updated[index], position=position)
        self._set_stops(tuple(updated))

    def remove_selected(self) -> bool:
        if len(self._paint.stops) <= 2:
            return False
        updated = list(self._paint.stops)
        del updated[self._selected_index]
        self._selected_index = min(self._selected_index, len(updated) - 1)
        self._set_stops(tuple(updated))
        self.selection_changed.emit(self._selected_index)
        return True

    def _set_stops(self, stops: Tuple[GradientStop, ...]) -> None:
        paint = replace(self._paint, stops=stops)
        if paint == self._paint:
            return
        self._paint = paint
        self.paint_previewed.emit(self._paint)
        self.update()

    def _strip_rect(self) -> QRectF:
        return QRectF(7.0, 6.0, max(1.0, self.width() - 14.0), 18.0)

    def _position_from_x(self, x: float) -> float:
        rect = self._strip_rect()
        return max(0.0, min((x - rect.left()) / rect.width(), 1.0))

    def _handle_center(self, index: int) -> QPointF:
        rect = self._strip_rect()
        return QPointF(
            rect.left() + self._paint.stops[index].position * rect.width(),
            rect.bottom() + self.HANDLE_RADIUS + 2.0,
        )

    def _hit_handle(self, point: QPointF) -> Optional[int]:
        nearest = None
        nearest_distance = (self.HANDLE_RADIUS + 3.0) ** 2
        for index in range(len(self._paint.stops)):
            delta = point - self._handle_center(index)
            distance = delta.x() ** 2 + delta.y() ** 2
            if distance <= nearest_distance:
                nearest = index
                nearest_distance = distance
        return nearest

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = _mouse_position(event)
        index = self._hit_handle(point)
        if index is None and not self._strip_rect().contains(point):
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._drag_start_paint = self._paint
        if index is None:
            if not self.add_stop(self._position_from_x(point.x())):
                self._drag_start_paint = None
                return
        else:
            self.select_stop(index)
        self._dragging = True
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self.move_selected(self._position_from_x(_mouse_position(event).x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragging and event.button() == Qt.MouseButton.LeftButton:
            self.move_selected(self._position_from_x(_mouse_position(event).x()))
            before = self._drag_start_paint
            self._drag_start_paint = None
            self._dragging = False
            if before is not None and self._paint != before:
                self.paint_commit_requested.emit(self._paint)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self._drag_start_paint is not None:
            before = self._drag_start_paint
            changed = self._paint != before
            self._paint = before
            self._drag_start_paint = None
            self._dragging = False
            self.update()
            if changed:
                self.paint_preview_canceled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and self.interaction_active
            and event.key() == Qt.Key.Key_Escape
        ):
            event.accept()
            return True
        return super().event(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        strip = self._strip_rect()
        paint_effect_paint_preview(
            painter,
            strip,
            self._paint,
            self.palette(),
            self.devicePixelRatioF(),
        )
        painter.setPen(QPen(self.palette().mid().color(), 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(strip, 3.0, 3.0)
        if self._mixed:
            overlay = self.palette().window().color()
            overlay.setAlpha(180)
            painter.fillRect(strip, overlay)
            painter.setPen(self.palette().windowText().color())
            painter.drawText(strip, Qt.AlignmentFlag.AlignCenter, self.tr('Mixed'))
        for index, stop in enumerate(self._paint.stops):
            center = self._handle_center(index)
            painter.setPen(QPen(
                self.palette().highlight().color()
                if index == self._selected_index
                else self.palette().windowText().color(),
                2.0 if index == self._selected_index else 1.0,
            ))
            color = QColor(*stop.color)
            color.setAlphaF(stop.opacity)
            painter.setBrush(color)
            painter.drawEllipse(center, self.HANDLE_RADIUS, self.HANDLE_RADIUS)


class GradientValueEditor(QDoubleSpinBox):
    """Spin box that distinguishes live text preview from one commit."""

    step_committed = Signal()
    preview_canceled = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._preview_pending = False
        self._text_edit_pending = False
        self.lineEdit().textEdited.connect(self._on_text_edited)
        self.editingFinished.connect(self._on_editing_finished)

    def set_preview_pending(self, pending: bool) -> None:
        self._preview_pending = bool(pending)

    def _on_text_edited(self, _text: str) -> None:
        self._text_edit_pending = True

    def _on_editing_finished(self) -> None:
        self._text_edit_pending = False

    def resolve_text_edit(self) -> None:
        if not self._text_edit_pending:
            return
        self._text_edit_pending = False
        blocker = QSignalBlocker(self.lineEdit())
        self.lineEdit().setText(
            self.prefix() + self.textFromValue(self.value()) + self.suffix()
        )
        del blocker

    def stepBy(self, steps: int) -> None:
        before = self.value()
        self._text_edit_pending = False
        super().stepBy(steps)
        if self.value() != before:
            self.step_committed.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() == Qt.Key.Key_Escape
            and (self._preview_pending or self._text_edit_pending)
        ):
            self.resolve_text_edit()
            self.preview_canceled.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and (
                getattr(self, '_preview_pending', False)
                or getattr(self, '_text_edit_pending', False)
            )
            and event.key() == Qt.Key.Key_Escape
        ):
            event.accept()
            return True
        return super().event(event)


class GradientStopColorButton(QToolButton):
    """Small paint-only swatch for the selected stop color."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._color = QColor(0, 0, 0)
        self.setObjectName('GradientStopColorPicker')
        self.setFixedSize(26, 22)

    def set_color(self, color: Tuple[int, int, int]) -> None:
        self._color = QColor(*color)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(self.contentsRect().adjusted(3, 3, -3, -3), self._color)


class InlineLinearGradientEditor(QWidget):
    """Edit one gradient inline and publish through the card edit session.

    >>> issubclass(InlineLinearGradientEditor, QWidget)
    True
    """

    paint_previewed = Signal(object)
    paint_commit_requested = Signal(object)
    paint_preview_canceled = Signal()
    color_dialog_active_changed = Signal(bool)

    def __init__(
        self,
        paint: LinearGradientPaint,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._paint = paint
        self._edit_before: Optional[LinearGradientPaint] = None
        self.setObjectName('InlineLinearGradientEditor')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.stop_bar = GradientStopBar(paint, self)
        self.stop_bar.paint_previewed.connect(self._on_stop_paint_preview)
        self.stop_bar.paint_commit_requested.connect(self._commit_current)
        self.stop_bar.paint_preview_canceled.connect(self.cancel_pending)
        self.stop_bar.selection_changed.connect(self._sync_selected_stop)

        self.add_stop_button = QToolButton(self)
        self.add_stop_button.setText(self.tr('Add'))
        self.add_stop_button.setToolTip(self.tr('Add Stop'))
        self.add_stop_button.setAccessibleName(self.tr('Add Stop'))
        self.add_stop_button.clicked.connect(self._on_add_stop)
        self.remove_stop_button = QToolButton(self)
        self.remove_stop_button.setText(self.tr('Remove'))
        self.remove_stop_button.setToolTip(self.tr('Remove Stop'))
        self.remove_stop_button.setAccessibleName(self.tr('Remove Stop'))
        self.remove_stop_button.clicked.connect(self._on_remove_stop)

        strip_row = QHBoxLayout()
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.setSpacing(4)
        strip_row.addWidget(self.stop_bar, 1)
        strip_actions = QVBoxLayout()
        strip_actions.setContentsMargins(0, 3, 0, 3)
        strip_actions.setSpacing(3)
        strip_actions.addWidget(self.add_stop_button)
        strip_actions.addWidget(self.remove_stop_button)
        strip_row.addLayout(strip_actions)

        self.stop_color_picker = GradientStopColorButton(self)
        self.stop_color_picker.setToolTip(self.tr('Stop Color'))
        self.stop_color_picker.setAccessibleName(self.tr('Stop Color'))
        self.stop_color_picker.clicked.connect(self._choose_stop_color)
        self.stop_opacity_editor = self._spinbox(
            0.0, 100.0, 1.0, '%', decimals=0
        )
        self.stop_position_editor = self._spinbox(
            0.0, 100.0, 0.1, '%', decimals=1
        )
        self.stop_opacity_editor.valueChanged.connect(
            self._on_stop_opacity_preview
        )
        self.stop_position_editor.valueChanged.connect(
            self._on_stop_position_preview
        )
        self.stop_opacity_editor.editingFinished.connect(self._commit_current)
        self.stop_position_editor.editingFinished.connect(self._commit_current)

        stop_grid = QGridLayout()
        stop_grid.setContentsMargins(0, 0, 0, 0)
        stop_grid.setHorizontalSpacing(8)
        stop_grid.setVerticalSpacing(4)
        stop_grid.addWidget(QLabel(self.tr('Color'), self), 0, 0)
        stop_grid.addWidget(self.stop_color_picker, 0, 1)
        stop_grid.addWidget(QLabel(self.tr('Opacity'), self), 0, 2)
        stop_grid.addWidget(self.stop_opacity_editor, 0, 3)
        stop_grid.addWidget(QLabel(self.tr('Position'), self), 1, 0)
        stop_grid.addWidget(self.stop_position_editor, 1, 1)

        self.angle_editor = self._spinbox(
            0.0, 359.9, 1.0, self.tr('°'), decimals=1
        )
        self.scale_editor = self._spinbox(
            10.0, 400.0, 1.0, '%', decimals=0
        )
        self.angle_editor.valueChanged.connect(self._on_angle_preview)
        self.scale_editor.valueChanged.connect(self._on_scale_preview)
        self.angle_editor.editingFinished.connect(self._commit_current)
        self.scale_editor.editingFinished.connect(self._commit_current)
        self.flip_button = QToolButton(self)
        self.flip_button.setText(self.tr('Flip'))
        self.flip_button.setToolTip(self.tr('Flip Gradient'))
        self.flip_button.setAccessibleName(self.tr('Flip Gradient'))
        self.flip_button.clicked.connect(self._on_flip)

        geometry_row = QHBoxLayout()
        geometry_row.setContentsMargins(0, 0, 0, 0)
        geometry_row.setSpacing(4)
        geometry_row.addWidget(QLabel(self.tr('Angle'), self))
        geometry_row.addWidget(self.angle_editor)
        geometry_row.addWidget(QLabel(self.tr('Scale'), self))
        geometry_row.addWidget(self.scale_editor)
        geometry_row.addWidget(self.flip_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(strip_row)
        layout.addLayout(stop_grid)
        layout.addLayout(geometry_row)

        for editor in self._editors():
            editor.step_committed.connect(self._commit_current)
            editor.preview_canceled.connect(self.cancel_pending)
        self._sync_all_controls()

    @property
    def paint(self) -> LinearGradientPaint:
        return self._paint

    def _spinbox(
        self,
        minimum: float,
        maximum: float,
        step: float,
        suffix: str,
        *,
        decimals: int,
    ) -> GradientValueEditor:
        editor = GradientValueEditor(self)
        editor.setObjectName('TextEffectParamEditor')
        editor.setRange(minimum, maximum)
        editor.setDecimals(decimals)
        editor.setSingleStep(step)
        editor.setSuffix(suffix)
        button_symbols = getattr(
            QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox
        )
        editor.setButtonSymbols(button_symbols.NoButtons)
        # Reserve the no-button line-edit chrome plus a small suffix margin
        # without making the complete effect card wider than the side panel.
        editor.setMinimumWidth(max(54, editor.minimumSizeHint().width() + 4))
        editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        return editor

    def _editors(self) -> Tuple[GradientValueEditor, ...]:
        return (
            self.stop_opacity_editor,
            self.stop_position_editor,
            self.angle_editor,
            self.scale_editor,
        )

    def set_paint(
        self,
        paint: LinearGradientPaint,
        *,
        editable: bool = True,
    ) -> None:
        self._paint = paint
        self._edit_before = None
        self.stop_bar.end_interaction()
        self.stop_bar.set_paint(paint)
        self.stop_bar.set_mixed(not editable)
        self.stop_bar.setEnabled(editable)
        self.add_stop_button.setEnabled(editable and len(paint.stops) < 32)
        self.remove_stop_button.setEnabled(editable and len(paint.stops) > 2)
        self.stop_color_picker.setEnabled(editable)
        self.flip_button.setEnabled(editable)
        for editor in self._editors():
            editor.resolve_text_edit()
            editor.set_preview_pending(False)
            editor.setEnabled(editable)
        self._sync_all_controls()

    def _selected_stop(self) -> GradientStop:
        return self._paint.stops[self.stop_bar.selected_index]

    def _sync_all_controls(self) -> None:
        blockers = (
            QSignalBlocker(self.angle_editor),
            QSignalBlocker(self.scale_editor),
        )
        self.angle_editor.setValue(self._paint.angle)
        self.scale_editor.setValue(self._paint.scale * 100.0)
        del blockers
        self._sync_selected_stop(self.stop_bar.selected_index)

    def _sync_selected_stop(self, _index: int) -> None:
        stop = self._selected_stop()
        index = self.stop_bar.selected_index
        stops = self._paint.stops
        minimum_position = (
            stops[index - 1].position * 100.0 if index > 0 else 0.0
        )
        maximum_position = (
            stops[index + 1].position * 100.0
            if index + 1 < len(stops) else 100.0
        )
        blockers = (
            QSignalBlocker(self.stop_opacity_editor),
            QSignalBlocker(self.stop_position_editor),
        )
        self.stop_color_picker.set_color(stop.color)
        self.stop_opacity_editor.setValue(stop.opacity * 100.0)
        self.stop_position_editor.setRange(
            minimum_position, maximum_position
        )
        self.stop_position_editor.setValue(stop.position * 100.0)
        self.add_stop_button.setEnabled(
            self.stop_bar.isEnabled() and len(self._paint.stops) < 32
        )
        self.remove_stop_button.setEnabled(
            self.stop_bar.isEnabled() and len(self._paint.stops) > 2
        )
        del blockers

    def _begin_edit(self) -> None:
        if self._edit_before is None:
            self._edit_before = self._paint
            for editor in self._editors():
                editor.set_preview_pending(True)

    def _publish_preview(self, paint: LinearGradientPaint) -> None:
        if paint == self._paint:
            return
        self._begin_edit()
        self._paint = paint
        self.stop_bar.set_paint(paint)
        self.paint_previewed.emit(paint)

    def _on_stop_paint_preview(self, paint: LinearGradientPaint) -> None:
        if paint == self._paint:
            return
        self._begin_edit()
        self._paint = paint
        self._sync_selected_stop(self.stop_bar.selected_index)
        self.paint_previewed.emit(paint)

    def _commit_current(self, *_args: object) -> bool:
        before = self._edit_before
        self._edit_before = None
        for editor in self._editors():
            editor.set_preview_pending(False)
        if before is None or before == self._paint:
            return False
        self.paint_commit_requested.emit(self._paint)
        return True

    def commit_pending(self) -> bool:
        self.stop_bar.end_interaction()
        for editor in self._editors():
            editor.resolve_text_edit()
        return self._commit_current()

    def cancel_pending(self, *_args: object) -> bool:
        self.stop_bar.end_interaction()
        before = self._edit_before
        self._edit_before = None
        for editor in self._editors():
            editor.resolve_text_edit()
            editor.set_preview_pending(False)
        if before is None or before == self._paint:
            return False
        self._paint = before
        self.stop_bar.set_paint(before)
        self._sync_all_controls()
        self.paint_preview_canceled.emit()
        return True

    def _on_add_stop(self) -> None:
        stops = self._paint.stops
        _, left, right = max(
            (
                (following.position - current.position, current, following)
                for current, following in zip(stops, stops[1:])
            ),
            key=lambda entry: entry[0],
        )
        self._begin_edit()
        if self.stop_bar.add_stop((left.position + right.position) / 2.0):
            self._commit_current()
        else:
            self._edit_before = None

    def _on_remove_stop(self) -> None:
        self._begin_edit()
        if self.stop_bar.remove_selected():
            self._commit_current()
        else:
            self._edit_before = None

    def _on_stop_opacity_preview(self, value: float) -> None:
        stop = self._selected_stop()
        self._publish_preview(replace(
            self._paint,
            stops=self._replaced_selected_stop(
                replace(stop, opacity=value / 100.0)
            ),
        ))

    def _on_stop_position_preview(self, value: float) -> None:
        index = self.stop_bar.selected_index
        stops = self._paint.stops
        minimum = stops[index - 1].position if index > 0 else 0.0
        maximum = stops[index + 1].position if index + 1 < len(stops) else 1.0
        stop = replace(
            stops[index],
            position=max(minimum, min(value / 100.0, maximum)),
        )
        self._publish_preview(replace(
            self._paint, stops=self._replaced_selected_stop(stop)
        ))

    def _replaced_selected_stop(
        self, stop: GradientStop
    ) -> Tuple[GradientStop, ...]:
        stops = list(self._paint.stops)
        stops[self.stop_bar.selected_index] = stop
        return tuple(stops)

    def _on_angle_preview(self, value: float) -> None:
        self._publish_preview(replace(self._paint, angle=value))

    def _on_scale_preview(self, value: float) -> None:
        self._publish_preview(replace(self._paint, scale=value / 100.0))

    def _on_flip(self) -> None:
        self._begin_edit()
        self._publish_preview(replace(self._paint, angle=self._paint.angle + 180.0))
        blocker = QSignalBlocker(self.angle_editor)
        self.angle_editor.setValue(self._paint.angle)
        del blocker
        self._commit_current()

    def _choose_stop_color(self) -> None:
        self._begin_edit()
        dialog = QColorDialog(QColor(*self._selected_stop().color), self.window())
        dialog.currentColorChanged.connect(self._on_stop_color_preview)
        dialog.accepted.connect(self._on_stop_color_accepted)
        dialog.rejected.connect(self._on_stop_color_rejected)
        self.color_dialog_active_changed.emit(True)
        try:
            dialog.exec_()
        finally:
            self.color_dialog_active_changed.emit(False)
            dialog.deleteLater()

    def _on_stop_color_preview(self, color: QColor) -> None:
        if not color.isValid():
            return
        stop = replace(
            self._selected_stop(),
            color=(color.red(), color.green(), color.blue()),
        )
        self._publish_preview(replace(
            self._paint, stops=self._replaced_selected_stop(stop)
        ))
        self.stop_color_picker.set_color(stop.color)

    def _on_stop_color_accepted(self) -> None:
        self._commit_current()

    def _on_stop_color_rejected(self) -> None:
        self.cancel_pending()
