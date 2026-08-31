import math
from typing import Callable, List, Optional

from qtpy.QtWidgets import (
    QComboBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)
from qtpy.QtCore import QEvent, QSignalBlocker, QSize, Signal, Qt
from qtpy.QtGui import (
    QDoubleValidator,
    QPaintEvent,
    QPainter,
    QPalette,
    QValidator,
)

from ballontranslator.utils.shared import CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from .push_button import NoBorderPushBtn
from ..icon_rendering import render_svg_pixmap
from ..misc import themed_icon_path


class ComboBox(QComboBox):

    # https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox
    def __init__(self, parent: QWidget = None, scrollWidget: QWidget = None, options: List[str] = None) -> None:
        super().__init__(parent)
        self.scrollWidget = scrollWidget
        if options is not None:
            self.addItems(options)

    def setScrollWidget(self, scrollWidget: QWidget):
        self.scrollWidget = scrollWidget

    def wheelEvent(self, *args, **kwargs):
        if self.scrollWidget is None or self.hasFocus():
            return super().wheelEvent(*args, **kwargs)
        else:
            return self.scrollWidget.wheelEvent(*args, **kwargs)
        

class SmallComboBox(ComboBox):
    pass


class BottomBorderComboBox(QComboBox):
    """Combo box with the app's compact bottom-border selector treatment.

    >>> BottomBorderComboBox.__name__
    'BottomBorderComboBox'
    """

    ARROW_SIZE = 12

    def __init__(
        self,
        parent: QWidget = None,
        *,
        text_alignment: Optional[Qt.AlignmentFlag] = None,
    ) -> None:
        super().__init__(parent)
        self._text_alignment = text_alignment
        self._width_sample_text: Optional[str] = None
        self.setProperty('bottomBorderSelector', True)

    def setWidthSampleText(self, text: str) -> None:
        """Prefer room for ``text`` while retaining normal shrink behavior."""
        self._width_sample_text = text
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        if not self._width_sample_text:
            return size
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        contents = QSize(
            option.fontMetrics.horizontalAdvance(self._width_sample_text),
            option.fontMetrics.height(),
        )
        reference = self.style().sizeFromContents(
            QStyle.ContentsType.CT_ComboBox,
            option,
            contents,
            self,
        )
        size.setWidth(max(size.width(), reference.width()))
        return size

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._text_alignment is None:
            super().paintEvent(event)
            painter = QPainter(self)
        else:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
            current_text = option.currentText
            option.currentText = ''
            painter = QStylePainter(self)
            painter.drawComplexControl(
                QStyle.ComplexControl.CC_ComboBox, option
            )
            painter.drawControl(
                QStyle.ControlElement.CE_ComboBoxLabel, option
            )
            text_rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_ComboBox,
                option,
                QStyle.SubControl.SC_ComboBoxEditField,
                self,
            ).adjusted(2, 0, -2, 0)
            color_group = (
                QPalette.ColorGroup.Active
                if self.isEnabled()
                else QPalette.ColorGroup.Disabled
            )
            color_role = (
                QPalette.ColorRole.PlaceholderText
                if self.currentIndex() < 0
                else QPalette.ColorRole.Text
            )
            painter.setPen(option.palette.color(color_group, color_role))
            painter.drawText(
                text_rect,
                self._text_alignment | Qt.AlignmentFlag.AlignVCenter,
                option.fontMetrics.elidedText(
                    current_text,
                    Qt.TextElideMode.ElideRight,
                    max(0, text_rect.width()),
                ),
            )
        pixmap = render_svg_pixmap(
            themed_icon_path('chevron-down.svg'),
            self.ARROW_SIZE,
            self.ARROW_SIZE,
            self.devicePixelRatioF(),
        )
        x = self.width() - self.ARROW_SIZE - 4
        y = (self.height() - self.ARROW_SIZE) // 2
        painter.drawPixmap(x, y, pixmap)
        painter.end()


class ConfigComboBox(ComboBox):

    def __init__(self, fix_size=True, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(scrollWidget, *args, **kwargs)
        self.fix_size = fix_size
        self.adjustSize()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def addItems(self, texts: List[str]) -> None:
        super().addItems(texts)
        self.adjustSize()

    def adjustSize(self) -> None:
        super().adjustSize()
        width = self.minimumSizeHint().width()
        if width < CONFIG_COMBOBOX_SHORT:
            width = CONFIG_COMBOBOX_SHORT
        elif width < CONFIG_COMBOBOX_MIDEAN:
            width = CONFIG_COMBOBOX_MIDEAN
        else:
            width = CONFIG_COMBOBOX_LONG
        if self.fix_size:
            self.setFixedWidth(width)
        else:
            self.setMaximumWidth(width)


class ParamComboBox(ComboBox):
    paramwidget_edited = Signal(str, str)
    flushbtn_clicked = Signal()
    pathbtn_clicked = Signal()
    def __init__(self, param_key: str, options: List[str], size=CONFIG_COMBOBOX_SHORT, scrollWidget: QWidget = None, flush_btn: bool = False, path_selector: bool = False, *args, **kwargs) -> None:
        super().__init__(scrollWidget=scrollWidget, *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size)
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        options = [str(opt) for opt in options]
        self.addItems(options)
        self.currentTextChanged.connect(self.on_select_changed)
        
        if flush_btn:
            self.flush_btn = NoBorderPushBtn(self.tr('Flush'))
            self.flush_btn.clicked.connect(self.flushbtn_clicked)
        if path_selector:
            self.path_select_btn = NoBorderPushBtn(self.tr('Select Path'))
            self.path_select_btn.clicked.connect(self.pathbtn_clicked)

    def on_select_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.currentText())


class SizeComboBox(QComboBox):
    
    param_changed = Signal(str, float)
    pending_edit_started = Signal()
    def __init__(
        self,
        val_range: List = None,
        param_name: str = '',
        parent=None,
        init_value=None,
        *args,
        defer_text_changes: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.param_name = param_name
        self._defer_text_changes = bool(defer_text_changes)
        self._pending_text_change = False
        self._programmatic_change = False
        self._committing_pending = False
        self._committed_text = ''
        self._committed_value = None
        self.setEditable(True)
        self.editTextChanged.connect(self.on_text_changed)
        self.activated.connect(self.on_current_index_changed)
        self.min_val = val_range[0]
        self.max_val = val_range[1]
        validator = QDoubleValidator()
        if val_range is not None:
            validator.setTop(val_range[1])
            validator.setBottom(val_range[0])
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.setValidator(validator)
        self._value = 0
        editor = self.lineEdit()
        if editor is not None:
            editor.returnPressed.connect(self.commit_pending)
            editor.editingFinished.connect(self.commit_pending)
            editor.installEventFilter(self)
        if init_value is not None:
            self.setValue(init_value)

    def set_defer_text_changes(self, enabled: bool) -> None:
        """Toggle whether typed values wait for an explicit commit."""
        enabled = bool(enabled)
        if enabled == self._defer_text_changes:
            return
        if not enabled:
            self.cancel_pending()
        self._defer_text_changes = enabled
        if enabled:
            self._remember_current_display()

    def _editor_has_focus(self) -> bool:
        editor = self.lineEdit()
        return self.hasFocus() or (
            editor is not None and editor.hasFocus()
        )

    def _remember_current_display(self) -> None:
        self._committed_text = self.currentText()
        try:
            value = float(self._committed_text)
        except (TypeError, ValueError):
            self._committed_value = None
            return
        if math.isfinite(value) and self.min_val <= value <= self.max_val:
            self._value = value
            self._committed_value = value
        else:
            self._committed_value = None

    def setCurrentText(self, text: str) -> None:
        if not self._defer_text_changes:
            return super().setCurrentText(text)
        self._programmatic_change = True
        try:
            super().setCurrentText(text)
        finally:
            self._programmatic_change = False
        if self._defer_text_changes:
            self._pending_text_change = False
            self._remember_current_display()

    def addItem(self, text: str, userData=None) -> None:
        super().addItem(text, userData)
        if self._defer_text_changes and not self._pending_text_change:
            self._remember_current_display()

    def addItems(self, texts: List[str]) -> None:
        super().addItems(texts)
        if self._defer_text_changes and not self._pending_text_change:
            self._remember_current_display()

    def _restore_committed_display(self) -> None:
        blocker = QSignalBlocker(self)
        self._programmatic_change = True
        try:
            super().setCurrentText(self._committed_text)
        finally:
            self._programmatic_change = False
            del blocker

    def _parse_deferred_value(self) -> float:
        text = self.currentText().strip()
        if not text:
            raise ValueError
        validator = self.validator()
        if validator is not None:
            validation = validator.validate(text, 0)
            state = validation[0] if isinstance(validation, tuple) else validation
            if state != QValidator.State.Acceptable:
                raise ValueError
        value = float(text)
        if (
            not math.isfinite(value)
            or not self.min_val <= value <= self.max_val
        ):
            raise ValueError
        return value

    def _commit_value(self, value: float) -> bool:
        changed = (
            self._committed_value is None
            or value != self._committed_value
        )
        self._pending_text_change = False
        if not changed:
            self._restore_committed_display()
            return False
        self._value = value
        self._committed_value = value
        self._committed_text = self.currentText()
        self.param_changed.emit(self.param_name, value)
        return True

    @property
    def has_pending_text(self) -> bool:
        return self._pending_text_change

    @property
    def committing_pending(self) -> bool:
        return self._committing_pending

    def _emit_current_value(self) -> bool:
        try:
            value = self._parse_deferred_value()
        except (TypeError, ValueError):
            self._restore_committed_display()
            return False
        return self._commit_value(value)

    def on_text_changed(self):
        if self._programmatic_change:
            return
        if self._defer_text_changes:
            if self._editor_has_focus():
                if not self._pending_text_change:
                    self._pending_text_change = True
                    self.pending_edit_started.emit()
        elif self.hasFocus():
            self.param_changed.emit(self.param_name, self.value())

    def on_current_index_changed(self):
        if self.hasFocus() or self.view().isVisible():
            if self._defer_text_changes:
                if self._pending_text_change:
                    self.commit_pending()
                else:
                    self._emit_current_value()
            else:
                self.param_changed.emit(self.param_name, self.value())

    def commit_pending(self) -> bool:
        if not self._defer_text_changes or not self._pending_text_change:
            return False
        try:
            value = self._parse_deferred_value()
        except (TypeError, ValueError):
            self._pending_text_change = False
            self._restore_committed_display()
            return False
        self._committing_pending = True
        try:
            return self._commit_value(value)
        finally:
            self._committing_pending = False

    def cancel_pending(self) -> bool:
        if not self._defer_text_changes or not self._pending_text_change:
            return False
        self._pending_text_change = False
        self._restore_committed_display()
        return True

    def eventFilter(self, watched, event):
        if watched is self.lineEdit() and self._defer_text_changes:
            if (
                event.type() == QEvent.Type.ShortcutOverride
                and event.key() == Qt.Key.Key_Escape
                and self._pending_text_change
            ):
                event.accept()
                return True
            if (
                event.type() == QEvent.Type.KeyPress
                and event.key() in (
                    Qt.Key.Key_Return,
                    Qt.Key.Key_Enter,
                )
                and self._pending_text_change
            ):
                self.commit_pending()
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

    def focusOutEvent(self, event) -> None:
        if self._defer_text_changes:
            self.commit_pending()
        return super().focusOutEvent(event)

    def value(self) -> float:
        txt = self.currentText()
        try:
            val = float(txt)
            self._value = val
            return val
        except:
            return self._value

    def setValue(self, value: float):
        value = min(self.max_val, max(self.min_val, value))
        self.setCurrentText(str(round(value, 2)))

    def changeByDelta(self, delta: float, multiplier = 0.01):
        if isinstance(multiplier, Callable):
            multiplier = multiplier()
        self.setValue(self.value() + delta * multiplier)


class SmallSizeComboBox(SizeComboBox):
    pass
