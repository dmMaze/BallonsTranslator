from qtpy.QtWidgets import QWidget, QStyle, QSlider, QStyle, QStyleOptionSlider
from qtpy.QtCore import  Qt, QPropertyAnimation, QRect, QRectF, Signal, QPoint, Property
from qtpy.QtGui import QFontMetrics, QMouseEvent, QPainter, QFontMetrics, QColor

from .helper import isDarkTheme, themeColor
from utils import shared as C


def slider_subcontrol_rect(r: QRect, widget: QWidget):
    if widget.orientation() == Qt.Orientation.Horizontal:
        y = widget.height() // 4
        h = y * 2
        r = QRect(r.x(), y, r.width(), h)
    else:
        x = widget.width() // 4
        w = x * 2
        r = QRect(x, r.y(), w, r.height())

    # seems a bit dumb, otherwise the handle is buggy
    if r.height() < r.width():
        r.setHeight(r.width())
    else:
        r.setWidth(r.height())
    return r


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
        isDark = isDarkTheme()
        painter.setPen(QColor(0, 0, 0, 90 if isDark else 25))
        painter.setBrush(QColor(69, 69, 69) if isDark else QColor(225, 228, 235))
        painter.drawEllipse(self.rect().adjusted(1, 1, -1, -1))

        # draw innert circle
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
    def grooveLength(self):
        l = self.width() if self.orientation() == Qt.Orientation.Horizontal else self.height()
        return l - self.handle.width()

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
        painter.setBrush(QColor(255, 255, 255, 115) if isDarkTheme() else QColor(0, 0, 0, 100))

        if self.orientation() == Qt.Orientation.Horizontal:
            self._drawHorizonGroove(painter)
        else:
            self._drawVerticalGroove(painter)

        if hasattr(self, 'draw_content') and self.hovering:
            # its a bad idea to display text like this, but I leave it as it is for now
            
            option = QStyleOptionSlider()
            self.initStyleOption(option)

            rect = self.style().subControlRect(
                QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
            rect = slider_subcontrol_rect(rect, self)
            
            value = self.value()
            value_str = str(value)
                
            painter.setPen(QColor(*C.SLIDERHANDLE_COLOR,255))
            font = painter.font()
            font.setPointSizeF(8)
            fm = QFontMetrics(font)
            painter.setFont(font)

            is_hor = self.orientation() == Qt.Orientation.Horizontal
            if is_hor: 
                value_w = fm.boundingRect(value_str).width()
                dx = self.width() - value_w
            else:
                dx = dy = 0

            dy = self.height() - fm.height() + fm.descent()
            painter.drawText(dx, dy, value_str)

            if self.draw_content is not None:
                painter.drawText(0, dy, self.draw_content, )
                

    def _drawHorizonGroove(self, painter: QPainter):
        w, r = self.width(), self.handle.width() / 2
        painter.drawRoundedRect(QRectF(r, r-2, w-r*2, 4), 2, 2)

        if self.maximum() - self.minimum() == 0:
            return

        painter.setBrush(themeColor())
        aw = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) * (w - r*2)
        painter.drawRoundedRect(QRectF(r, r-2, aw, 4), 2, 2)

    def _drawVerticalGroove(self, painter: QPainter):
        h, r = self.height(), self.handle.width() / 2
        painter.drawRoundedRect(QRectF(r-2, r, 4, h-2*r), 2, 2)

        if self.maximum() - self.minimum() == 0:
            return

        painter.setBrush(themeColor())
        ah = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) * (h - r*2)
        painter.drawRoundedRect(QRectF(r-2, r, 4, ah), 2, 2)

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

    mouse_released = Signal()

    def __init__(self, draw_content = None, orientation=Qt.Orientation.Horizontal, *args, **kwargs):
        super().__init__(orientation, *args, **kwargs)
        self.draw_content = draw_content
        self.pressed: bool = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = False
            self.mouse_released.emit()
        return super().mouseReleaseEvent(event)
