import math
from typing import Callable, Sequence

from qtpy.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import QEvent, QPoint, QRect, QSize, QTimer, Signal, Qt
from qtpy.QtGui import QKeyEvent

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
from ballontranslator.utils.fontformat import (
    FontFormat,
    TEXT_TRANSFORM_BOX_SLANT_MAX,
    TEXT_TRANSFORM_BOX_SLANT_MIN,
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    TEXT_TRANSFORM_SCALE_MAX,
    TEXT_TRANSFORM_SCALE_MIN,
)


def _pack_preferred_widths(
    preferred_widths: Sequence[int],
    available_width: int,
    spacing: int,
):
    """Return greedy rows of indexes without splitting an atomic item.

    An over-wide item occupies a row by itself; geometry assignment later gives
    it the available width instead of allowing horizontal overflow.

    >>> _pack_preferred_widths([40, 30, 50], 75, 5)
    [(0, 1), (2,)]
    >>> _pack_preferred_widths([100, 20], 60, 5)
    [(0,), (1,)]
    """
    available_width = max(0, int(available_width))
    spacing = max(0, int(spacing))
    rows = []
    row = []
    used_width = 0
    for index, preferred_width in enumerate(preferred_widths):
        preferred_width = max(0, int(preferred_width))
        next_width = preferred_width if not row else used_width + spacing + preferred_width
        if row and next_width > available_width:
            rows.append(tuple(row))
            row = [index]
            used_width = preferred_width
        else:
            row.append(index)
            used_width = next_width
    if row:
        rows.append(tuple(row))
    return rows


class AdaptiveWrapLayout(QLayout):
    """Panel-local height-for-width layout for indivisible control units.

    The layout owns only ``QLayoutItem`` objects and changes only their
    geometries. It never reparents, recreates, or delays movement of widgets.

    >>> _pack_preferred_widths([30, 30, 30], 65, 5)
    [(0, 1), (2,)]
    """

    def __init__(self, parent=None, horizontal_spacing=-1, vertical_spacing=-1):
        super().__init__(parent)
        self._items = []
        self._horizontal_spacing = horizontal_spacing
        self._vertical_spacing = vertical_spacing

    def addItem(self, item: QLayoutItem):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def hasHeightForWidth(self):
        return True

    def _style_spacing(self, horizontal):
        explicit = (
            self._horizontal_spacing if horizontal else self._vertical_spacing
        )
        if explicit >= 0:
            return explicit
        inherited = self.spacing()
        if inherited >= 0:
            return inherited
        parent = self.parentWidget()
        if parent is not None:
            metric_name = (
                'PM_LayoutHorizontalSpacing'
                if horizontal
                else 'PM_LayoutVerticalSpacing'
            )
            pixel_metrics = getattr(QStyle, 'PixelMetric', QStyle)
            metric = getattr(pixel_metrics, metric_name)
            value = parent.style().pixelMetric(metric)
            if value >= 0:
                return value
        return 6

    def horizontalSpacing(self):
        return self._style_spacing(True)

    def verticalSpacing(self):
        return self._style_spacing(False)

    @staticmethod
    def _item_height(item, width):
        if item.hasHeightForWidth():
            height = item.heightForWidth(width)
        else:
            height = item.sizeHint().height()
        return max(item.minimumSize().height(), height)

    def _visible_items(self):
        return [item for item in self._items if not item.isEmpty()]

    def _do_layout(self, rect: QRect, test_only: bool):
        left, top, right, bottom = self.getContentsMargins()
        content_x = rect.x() + left
        content_y = rect.y() + top
        available_width = max(0, rect.width() - left - right)
        items = self._visible_items()
        if not items:
            return top + bottom

        horizontal_spacing = self.horizontalSpacing()
        vertical_spacing = self.verticalSpacing()
        preferred_widths = [
            max(item.minimumSize().width(), item.sizeHint().width())
            for item in items
        ]
        rows = _pack_preferred_widths(
            preferred_widths, available_width, horizontal_spacing
        )

        y = content_y
        for row_index, row in enumerate(rows):
            widths = [min(preferred_widths[index], available_width) for index in row]
            heights = [
                self._item_height(items[index], width)
                for index, width in zip(row, widths)
            ]
            row_height = max(heights, default=0)
            x = content_x
            for index, width in zip(row, widths):
                if not test_only:
                    items[index].setGeometry(QRect(x, y, width, row_height))
                x += width + horizontal_spacing
            y += row_height
            if row_index + 1 < len(rows):
                y += vertical_spacing
        return (y - rect.y()) + bottom

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, max(0, width), 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def minimumSize(self):
        items = self._visible_items()
        left, top, right, bottom = self.getContentsMargins()
        width = max((item.minimumSize().width() for item in items), default=0)
        width += left + right
        return QSize(width, self.heightForWidth(width))

    def sizeHint(self):
        items = self._visible_items()
        left, _top, right, _bottom = self.getContentsMargins()
        spacing = self.horizontalSpacing()
        width = sum(
            max(item.minimumSize().width(), item.sizeHint().width())
            for item in items
        )
        if items:
            width += spacing * (len(items) - 1)
        width += left + right
        return QSize(width, self.heightForWidth(width))


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
        # Qt still delivers the matching move/release events for the physical
        # press. Clear the label-owned latch so those moves cannot restart a
        # preview after an external transaction boundary canceled it.
        self.mouse_pressed = False

    def event(self, event):
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and self.mouse_pressed
            and event.key() == Qt.Key.Key_Escape
        ):
            # The active gesture owns Escape. Prevent an ancestor window
            # shortcut from consuming it before keyPressEvent can cancel.
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
        _word_wrap_label(self.label)
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
        display_value = float(text)
        canonical_value = self._display_to_canonical(display_value)
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
        # A control refresh can already have changed state to IDLE while its
        # label still owns the physical press, so abort before the state check.
        self.label.abort_drag_session()
        if self.state != self.DRAG_PREVIEW:
            return
        self.state = self.IDLE
        self._drag_delta = 0.0
        self._restore_display()
        self.preview_canceled.emit(self.param_name)


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

        self.transform_section = QWidget(self.scrollContent)
        self.transform_section.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.horizontal_scale_control = CommittedTransformControl(
            self.tr('Horizontal Scale'),
            'horizontal_scale',
            100.0,
            TEXT_TRANSFORM_SCALE_MIN,
            TEXT_TRANSFORM_SCALE_MAX,
            '%',
            1.0,
            self.transform_section,
        )
        self.vertical_scale_control = CommittedTransformControl(
            self.tr('Vertical Scale'),
            'vertical_scale',
            100.0,
            TEXT_TRANSFORM_SCALE_MIN,
            TEXT_TRANSFORM_SCALE_MAX,
            '%',
            1.0,
            self.transform_section,
        )
        self.slant_angle_control = CommittedTransformControl(
            self.tr('Box Slant'),
            'slant_angle',
            1.0,
            TEXT_TRANSFORM_BOX_SLANT_MIN,
            TEXT_TRANSFORM_BOX_SLANT_MAX,
            '\N{DEGREE SIGN}',
            1.0,
            self.transform_section,
        )
        self.glyph_slant_angle_control = CommittedTransformControl(
            self.tr('Glyph Slant'),
            'glyph_slant_angle',
            1.0,
            TEXT_TRANSFORM_GLYPH_SLANT_MIN,
            TEXT_TRANSFORM_GLYPH_SLANT_MAX,
            '\N{DEGREE SIGN}',
            1.0,
            self.transform_section,
        )
        self.transform_controls = {
            'horizontal_scale': self.horizontal_scale_control,
            'vertical_scale': self.vertical_scale_control,
            'slant_angle': self.slant_angle_control,
            'glyph_slant_angle': self.glyph_slant_angle_control,
        }
        for control in self.transform_controls.values():
            control.commit_requested.connect(self.transform_commit_requested.emit)
            control.preview_requested.connect(self.transform_preview_requested.emit)
            control.drag_commit_requested.connect(
                self.transform_drag_commit_requested.emit
            )
            control.preview_canceled.connect(self.transform_preview_canceled.emit)

        self.transform_layout = AdaptiveWrapLayout(self.transform_section)
        for control in self.transform_controls.values():
            self.transform_layout.addWidget(control)

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
        vlayout.addWidget(self.transform_section)
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

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)
        for name, control in self.transform_controls.items():
            control.set_model_value(getattr(font_format, name))

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
        for name, control in self.transform_controls.items():
            values = [getattr(item.blk.fontformat, name) for item in items]
            common = values[0] if values and all(value == values[0] for value in values) else None
            control.set_model_value(common)

    def finish_pending_transform_edits(self):
        for control in self.transform_controls.values():
            control.commit_pending()
