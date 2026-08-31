from typing import Optional

from qtpy.QtWidgets import QAbstractSpinBox, QSlider, QSpinBox, QWidget
from qtpy.QtCore import Qt, QPropertyAnimation, QRectF, QSignalBlocker, Signal, QPoint, Property
from qtpy.QtGui import QFontMetrics, QMouseEvent, QPainter, QResizeEvent, QColor

from .helper import themeColor, borderColor, widgetBackgroundColor
from ballontranslator.utils import shared

class SliderHandle(QWidget):
    """ Slider handle """

    pressed = Signal()
    released = Signal()

    def __init__(self, parent: QSlider):
        super().__init__(parent=parent)
        self.setFixedSize(22, 22)
        self._radius = 5
        self.radiusAni = QPropertyAnimation(self, b'radius', self)
        self.radiusAni.setDuration(100)

    @Property(int)
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        self.update()

    def enterEvent(self, e):
        self._startAni(6)

    def leaveEvent(self, e):
        self._startAni(5)

    def mousePressEvent(self, e):
        self._startAni(4)
        self.pressed.emit()

    def mouseReleaseEvent(self, e):
        self._startAni(6)
        self.released.emit()

    def _startAni(self, radius):
        self.radiusAni.stop()
        self.radiusAni.setStartValue(self.radius)
        self.radiusAni.setEndValue(radius)
        self.radiusAni.start()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # draw outer circle
        painter.setPen(borderColor())
        painter.setBrush(widgetBackgroundColor())
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))

        # draw innert circle
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(themeColor())
        painter.drawEllipse(QPoint(11, 11), self.radius, self.radius)


class Slider(QSlider):
    """ A slider can be clicked

    modified from https://github.com/zhiyiYo/PyQt-Fluent-Widgets

    Constructors
    ------------
    * Slider(`parent`: QWidget = None)
    * Slider(`orient`: Qt.Orientation, `parent`: QWidget = None)
    """

    clicked = Signal(int)

    def __init__(self, orientation: Qt.Orientation, parent: QWidget = None):
        super().__init__(orientation, parent=parent)
        self.hovering = False
        self._postInit()

    def _postInit(self):
        self.handle = SliderHandle(self)
        self._pressedPos = QPoint()
        self.setOrientation(self.orientation())

        self.handle.pressed.connect(self.sliderPressed)
        self.handle.released.connect(self.sliderReleased)
        self.valueChanged.connect(self._adjustHandlePos)

    def setOrientation(self, orientation: Qt.Orientation) -> None:
        super().setOrientation(orientation)
        if orientation == Qt.Orientation.Horizontal:
            self.setMinimumHeight(22)
        else:
            self.setMinimumWidth(22)

    def mousePressEvent(self, e: QMouseEvent):
        self._pressedPos = e.pos()
        self.setValue(self._posToValue(e.pos()))
        self.clicked.emit(self.value())

    def mouseMoveEvent(self, e: QMouseEvent):
        self.setValue(self._posToValue(e.pos()))
        self._pressedPos = e.pos()
        self.sliderMoved.emit(self.value())

    @property
    def grooveLength(self) -> int:
        return self._track_span() - self.handle.width()

    def _track_span(self) -> int:
        if self.orientation() == Qt.Orientation.Horizontal:
            return self.width()
        return self.height()

    def _adjustHandlePos(self):
        total = max(self.maximum() - self.minimum(), 1)
        delta = int((self.value() - self.minimum()) / total * self.grooveLength)

        if self.orientation() == Qt.Orientation.Vertical:
            self.handle.move(0, delta)
        else:
            self.handle.move(delta, 0)

    def _posToValue(self, pos: QPoint):
        pd = self.handle.width() / 2
        gs = max(self.grooveLength, 1)
        v = pos.x() if self.orientation() == Qt.Orientation.Horizontal else pos.y()
        return int((v - pd) / gs * (self.maximum() - self.minimum()) + self.minimum())

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.grooveColor())

        if self.orientation() == Qt.Orientation.Horizontal:
            self._drawHorizonGroove(painter)
        else:
            self._drawVerticalGroove(painter)

        if hasattr(self, 'draw_content') and self.hovering:
            # its a bad idea to display text like this, but I leave it as it is for now
            
            painter.setPen(QColor(*shared.SLIDERHANDLE_COLOR,255))
            font = painter.font()
            font.setPointSizeF(8)
            fm = QFontMetrics(font)
            painter.setFont(font)

            dy = self.height() - fm.height() + fm.descent()
            if getattr(self, 'value_editor', None) is None:
                value_str = str(self.value())
                value_w = fm.boundingRect(value_str).width()
                dx = self.width() - value_w if (
                    self.orientation() == Qt.Orientation.Horizontal
                ) else 0
                painter.drawText(dx, dy, value_str)

            if self.draw_content is not None:
                painter.drawText(0, dy, self.draw_content)
                

    def _drawHorizonGroove(self, painter: QPainter):
        w, r = self._track_span(), self.handle.width() / 2
        painter.drawRoundedRect(QRectF(r, r-2, w-r*2, 4), 2, 2)

        if self.maximum() - self.minimum() == 0:
            return

        painter.setBrush(themeColor())
        aw = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) * (w - r*2)
        painter.drawRoundedRect(QRectF(r, r-2, aw, 4), 2, 2)

    def _drawVerticalGroove(self, painter: QPainter):
        h, r = self._track_span(), self.handle.width() / 2
        painter.drawRoundedRect(QRectF(r-2, r, 4, h-2*r), 2, 2)

        if self.maximum() - self.minimum() == 0:
            return

        painter.setBrush(themeColor())
        ah = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) * (h - r*2)
        painter.drawRoundedRect(QRectF(r-2, r, 4, ah), 2, 2)

    def grooveColor(self) -> QColor:
        return borderColor()

    def resizeEvent(self, e):
        self._adjustHandlePos()

    def enterEvent(self, event) -> None:
        self.hovering = True
        self.update()
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.hovering = False
        self.update()
        return super().leaveEvent(event)


class PaintQSlider(Slider):
    """Paint slider with an optional editable value at its right edge.

    >>> hasattr(PaintQSlider, 'enableValueEditor')
    True
    """

    mouse_released = Signal()
    valueEdited = Signal(int)

    VALUE_EDITOR_WIDTH = 60
    VALUE_EDITOR_HEIGHT = 22
    VALUE_EDITOR_GAP = 8

    def __init__(self, draw_content=None, orientation=Qt.Orientation.Horizontal, *args, **kwargs) -> None:
        self.value_editor: Optional[QSpinBox] = None
        super().__init__(orientation, *args, **kwargs)
        self.draw_content = draw_content
        self.pressed: bool = False
        self.valueChanged.connect(self._sync_value_editor)

    def enableValueEditor(self, suffix: str = '') -> QSpinBox:
        """Replace the hover value with a persistent inline editor."""
        if self.value_editor is not None:
            self.value_editor.setSuffix(suffix)
            return self.value_editor

        editor = QSpinBox(self)
        editor.setProperty('paintSliderValueEditor', True)
        editor.setRange(self.minimum(), self.maximum())
        editor.setValue(self.value())
        editor.setSuffix(suffix)
        editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_symbols = getattr(
            QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox
        )
        editor.setButtonSymbols(button_symbols.NoButtons)
        editor.setKeyboardTracking(False)
        editor.setFixedSize(
            self.VALUE_EDITOR_WIDTH,
            self.VALUE_EDITOR_HEIGHT,
        )
        editor.valueChanged.connect(self._set_value_from_editor)
        self.rangeChanged.connect(editor.setRange)
        self.value_editor = editor
        editor.show()
        self._position_value_editor()
        self._adjustHandlePos()
        self.update()
        return editor

    def _track_span(self) -> int:
        span = super()._track_span()
        if (
            self.value_editor is not None
            and self.orientation() == Qt.Orientation.Horizontal
        ):
            span -= self.VALUE_EDITOR_WIDTH + self.VALUE_EDITOR_GAP
        return max(self.handle.width(), span)

    def _sync_value_editor(self, value: int) -> None:
        if self.value_editor is not None:
            with QSignalBlocker(self.value_editor):
                self.value_editor.setValue(value)
        if self.hasFocus():
            self.valueEdited.emit(value)

    def _set_value_from_editor(self, value: int) -> None:
        self.setValue(value)
        if self.value_editor is not None and self.value_editor.hasFocus():
            self.valueEdited.emit(value)

    def _position_value_editor(self) -> None:
        if self.value_editor is None:
            return
        self.value_editor.move(
            self.width() - self.value_editor.width(),
            max(0, (self.height() - self.value_editor.height()) // 2),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = False
            self.mouse_released.emit()
        return super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_value_editor()
