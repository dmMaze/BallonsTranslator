"""Staged editor widgets for immutable linear-gradient effect paints."""

from dataclasses import replace
from typing import Optional, Tuple

from qtpy.QtCore import QPointF, QRectF, QSignalBlocker, Signal, Qt
from qtpy.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPen
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.text_effects import (
    GradientStop,
    LinearGradientPaint,
)

from ...custom_widget import ColorPickerLabel
from ...framelesswindow import DialogCloseButton, OutsideClickFramelessMixin
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

    paint_changed = Signal(object)
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
        self._dragging = False
        self.setMinimumHeight(42)
        self.setMouseTracking(True)
        self.setAccessibleName(self.tr('Gradient Stops'))
        self.setToolTip(
            self.tr('Click the strip to add a stop; drag a stop to move it')
        )

    @property
    def paint(self) -> LinearGradientPaint:
        return self._paint

    @property
    def selected_index(self) -> int:
        return self._selected_index

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
            (index for index, stop in enumerate(stops)
             if stop.position >= position),
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
            (index for index, stop in enumerate(stops)
             if stop.position > position),
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

    def replace_selected(
        self,
        *,
        color: Optional[Tuple[int, int, int]] = None,
        opacity: Optional[float] = None,
        position: Optional[float] = None,
    ) -> None:
        index = self._selected_index
        current = self._paint.stops[index]
        if position is None:
            new_position = current.position
        else:
            minimum = (
                self._paint.stops[index - 1].position if index > 0 else 0.0
            )
            maximum = (
                self._paint.stops[index + 1].position
                if index + 1 < len(self._paint.stops) else 1.0
            )
            new_position = max(minimum, min(float(position), maximum))
        updated = list(self._paint.stops)
        updated[index] = replace(
            current,
            color=current.color if color is None else color,
            opacity=current.opacity if opacity is None else opacity,
            position=new_position,
        )
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
        self._paint = replace(self._paint, stops=stops)
        self.paint_changed.emit(self._paint)
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
        if index is None:
            if not self._strip_rect().contains(point):
                super().mousePressEvent(event)
                return
            if not self.add_stop(self._position_from_x(point.x())):
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
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

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
        for index, stop in enumerate(self._paint.stops):
            center = self._handle_center(index)
            painter.setPen(
                QPen(
                    self.palette().highlight().color()
                    if index == self._selected_index
                    else self.palette().windowText().color(),
                    2.0 if index == self._selected_index else 1.0,
                )
            )
            color = QColor(*stop.color)
            color.setAlphaF(stop.opacity)
            painter.setBrush(color)
            painter.drawEllipse(center, self.HANDLE_RADIUS, self.HANDLE_RADIUS)


class LinearGradientEditorDialog(OutsideClickFramelessMixin, QDialog):
    """Stage one immutable linear gradient with live preview signals.

    >>> issubclass(LinearGradientEditorDialog, QDialog)
    True
    """

    paint_previewed = Signal(object)

    def __init__(
        self,
        paint: LinearGradientPaint,
        parent: Optional[QWidget] = None,
    ) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        super().__init__(
            parent,
            window_type.Dialog | window_type.FramelessWindowHint,
        )
        self._paint = paint
        self.setObjectName('LinearGradientEditorDialog')
        self.setWindowTitle(self.tr('Gradient'))
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(360)

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 5, 5, 5)
        surface = QFrame(self)
        surface.setObjectName('LinearGradientEditorSurface')
        root.addWidget(surface)
        layout = QVBoxLayout(surface)
        layout.setContentsMargins(16, 12, 16, 16)
        layout.setSpacing(10)

        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('LinearGradientEditorTitleBar')
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel(self.tr('Gradient'), self.title_bar)
        title.setObjectName('LinearGradientEditorTitle')
        title_layout.addWidget(title)
        title_layout.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.reject)
        title_layout.addWidget(self.close_button)
        layout.addWidget(self.title_bar)

        self.stop_bar = GradientStopBar(paint, surface)
        self.stop_bar.paint_changed.connect(self._on_stop_paint_changed)
        self.stop_bar.selection_changed.connect(self._sync_selected_stop)
        layout.addWidget(self.stop_bar)

        self.stop_color_picker = ColorPickerLabel(surface, param_name='stop')
        self.stop_color_picker.setObjectName('GradientStopColorPicker')
        self.stop_color_picker.setFixedSize(26, 22)
        self.stop_color_picker.setToolTip(self.tr('Stop Color'))
        self.stop_color_picker.setAccessibleName(self.tr('Stop Color'))
        self.stop_color_picker.colorChanged.connect(
            self._on_stop_color_changed
        )
        self.stop_color_picker.apply_color.connect(
            self._on_apply_stop_color
        )
        self.stop_opacity_editor = self._spinbox(0.0, 100.0, 1.0, '%')
        self.stop_position_editor = self._spinbox(0.0, 100.0, 0.1, '%')
        self.stop_opacity_editor.valueChanged.connect(
            self._on_stop_opacity_changed
        )
        self.stop_position_editor.valueChanged.connect(
            self._on_stop_position_changed
        )
        self.remove_stop_button = QToolButton(surface)
        self.remove_stop_button.setText(self.tr('Remove'))
        self.remove_stop_button.setToolTip(self.tr('Remove Stop'))
        self.remove_stop_button.setAccessibleName(self.tr('Remove Stop'))
        self.remove_stop_button.clicked.connect(self._on_remove_stop)

        stop_grid = QGridLayout()
        stop_grid.setContentsMargins(0, 0, 0, 0)
        stop_grid.setHorizontalSpacing(8)
        stop_grid.addWidget(QLabel(self.tr('Color'), surface), 0, 0)
        stop_grid.addWidget(self.stop_color_picker, 0, 1)
        stop_grid.addWidget(QLabel(self.tr('Opacity'), surface), 0, 2)
        stop_grid.addWidget(self.stop_opacity_editor, 0, 3)
        stop_grid.addWidget(QLabel(self.tr('Position'), surface), 1, 0)
        stop_grid.addWidget(self.stop_position_editor, 1, 1)
        stop_grid.addWidget(self.remove_stop_button, 1, 3)
        layout.addLayout(stop_grid)

        self.angle_editor = self._spinbox(0.0, 359.99, 1.0, self.tr('°'))
        self.scale_editor = self._spinbox(10.0, 400.0, 1.0, '%')
        self.angle_editor.valueChanged.connect(self._on_angle_changed)
        self.scale_editor.valueChanged.connect(self._on_scale_changed)
        self.flip_button = QToolButton(surface)
        self.flip_button.setText(self.tr('Flip'))
        self.flip_button.setToolTip(self.tr('Flip Gradient'))
        self.flip_button.setAccessibleName(self.tr('Flip Gradient'))
        self.flip_button.clicked.connect(self._on_flip)

        geometry_row = QHBoxLayout()
        geometry_row.setContentsMargins(0, 0, 0, 0)
        geometry_row.addWidget(QLabel(self.tr('Angle'), surface))
        geometry_row.addWidget(self.angle_editor)
        geometry_row.addWidget(QLabel(self.tr('Scale'), surface))
        geometry_row.addWidget(self.scale_editor)
        geometry_row.addWidget(self.flip_button)
        layout.addLayout(geometry_row)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel_button = QPushButton(self.tr('Cancel'), surface)
        cancel_button.clicked.connect(self.reject)
        actions.addWidget(cancel_button)
        accept_button = QPushButton(self.tr('OK'), surface)
        accept_button.setDefault(True)
        accept_button.clicked.connect(self.accept)
        actions.addWidget(accept_button)
        layout.addLayout(actions)
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
    ) -> QDoubleSpinBox:
        editor = QDoubleSpinBox(self)
        editor.setRange(minimum, maximum)
        editor.setDecimals(2)
        editor.setSingleStep(step)
        editor.setSuffix(suffix)
        return editor

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
        blockers = (
            QSignalBlocker(self.stop_opacity_editor),
            QSignalBlocker(self.stop_position_editor),
        )
        self.stop_color_picker.setPickerColor(stop.color)
        self.stop_opacity_editor.setValue(stop.opacity * 100.0)
        self.stop_position_editor.setValue(stop.position * 100.0)
        self.remove_stop_button.setEnabled(len(self._paint.stops) > 2)
        del blockers

    def _publish(self, paint: LinearGradientPaint) -> None:
        self._paint = paint
        self.stop_bar.set_paint(paint)
        self.paint_previewed.emit(paint)

    def _on_stop_paint_changed(self, paint: LinearGradientPaint) -> None:
        self._paint = paint
        self._sync_selected_stop(self.stop_bar.selected_index)
        self.paint_previewed.emit(paint)

    def _on_stop_color_changed(self, accepted: bool) -> None:
        if not accepted:
            return
        self.stop_bar.replace_selected(
            color=tuple(self.stop_color_picker.rgb())
        )

    def _on_apply_stop_color(
        self, _name: str, color: Tuple[int, int, int]
    ) -> None:
        self.stop_bar.replace_selected(color=color)

    def _on_stop_opacity_changed(self, value: float) -> None:
        self.stop_bar.replace_selected(opacity=value / 100.0)

    def _on_stop_position_changed(self, value: float) -> None:
        self.stop_bar.replace_selected(position=value / 100.0)

    def _on_remove_stop(self) -> None:
        self.stop_bar.remove_selected()

    def _on_angle_changed(self, value: float) -> None:
        self._publish(replace(self._paint, angle=value))

    def _on_scale_changed(self, value: float) -> None:
        self._publish(replace(self._paint, scale=value / 100.0))

    def _on_flip(self) -> None:
        self._publish(replace(self._paint, angle=self._paint.angle + 180.0))
        blocker = QSignalBlocker(self.angle_editor)
        self.angle_editor.setValue(self._paint.angle)
        del blocker

    def _dismiss_transient_window(self) -> None:
        self.reject()

    def _preserve_on_outside_click(self) -> bool:
        active_modal = QApplication.activeModalWidget()
        parent = self.parentWidget()
        parent_window = parent.window() if parent is not None else None
        return active_modal not in (None, self, parent_window)
