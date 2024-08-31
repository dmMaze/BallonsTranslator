from qtpy.QtWidgets import QApplication, QAbstractScrollArea, QGraphicsOpacityEffect, QWidget, QVBoxLayout, QHBoxLayout
from qtpy.QtCore import QEvent, Qt, QPropertyAnimation, QTimer, Signal, QPoint, Property, QAbstractAnimation
from qtpy.QtGui import QMouseEvent, QPainter, QColor

class ScrollBarGroove(QWidget):
    """ Scroll bar groove """

    def __init__(self, orient: Qt.Orientation, parent):
        super().__init__(parent=parent)
        if orient == Qt.Vertical:
            self.setFixedWidth(12)
            self.setLayout(QVBoxLayout(self))
            self.layout().addStretch(1)
            self.layout().setContentsMargins(0, 3, 0, 3)
        else:
            self.setFixedHeight(12)
            self.setLayout(QHBoxLayout(self))
            self.layout().addStretch(1)
            self.layout().setContentsMargins(3, 0, 3, 0)

        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b'opacity', self)
        self.setGraphicsEffect(self.opacityEffect)
        self.opacityEffect.setOpacity(0)

    def fadeIn(self):
        self.opacityAni.setEndValue(1)
        self.opacityAni.setDuration(150)
        self.opacityAni.start()

    def fadeOut(self):
        self.opacityAni.setEndValue(0)
        self.opacityAni.setDuration(150)
        self.opacityAni.start()

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        painter.setBrush(QColor(0, 0, 0, 30))
        painter.drawRoundedRect(self.rect(), 6, 6)


# ScrollBarHandle, ScrollBar and FlowLayout are modified from https://github.com/zhiyiYo/PyQt-Fluent-Widgets/blob/master/qfluentwidgets/components/widgets/scroll_bar.py

class ScrollBarHandle(QWidget):
    """ Scroll bar handle """

    def __init__(self, orient: Qt.Orientation, parent=None, fadeout: bool = False):
        super().__init__(parent)
        self.orient = orient

        if fadeout:
            self.effect = effect = QGraphicsOpacityEffect(self, opacity=1.0)
            self.setGraphicsEffect(effect)
            self.fadeAnimation = QPropertyAnimation(
                self,
                propertyName=b"opacity",
                targetObject=effect,
                duration=300,
                startValue=1.0,
                endValue=0.,
            )
            # self.fadeAnimation.setEasingCurve(QEasingCurve.Type.InQuint)
            self.fadeAnimation.finished.connect(self.hide)
            fixsize = 5
            self.anime_timer = QTimer(self)
            self.anime_timer.setSingleShot(True)
            self.anime_timer.timeout.connect(self.start_fade_animation)
        else:
            fixsize = 3
         
        if orient == Qt.Vertical:
            self.setFixedWidth(fixsize)
        else:
            self.setFixedHeight(fixsize)
        
        self.fadeout = fadeout

    def start_fade_animation(self):
        self.show()
        if self.fadeAnimation.state() == QAbstractAnimation.State.Running:
            self.fadeAnimation.stop()
        self.fadeAnimation.start()

    def prepareFadeout(self):
        self.anime_timer.stop()
        self.anime_timer.start(700)
        if self.isHidden():
            self.show()
        if self.fadeAnimation.state() == QAbstractAnimation.State.Running:
            self.fadeAnimation.stop()
        if self.effect.opacity() != 1.:
            self.effect.setOpacity(1.)

    def stopFadeout(self):
        if self.fadeAnimation.state() == QAbstractAnimation.State.Running:
            self.fadeAnimation.stop()
        self.anime_timer.stop()
        self.show()
        if self.effect.opacity() != 1.:
            self.effect.setOpacity(1.)

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        r = self.width() / 2 if self.orient == Qt.Vertical else self.height() / 2
        c = QColor(0, 0, 0, 90)
        painter.setBrush(c)
        painter.drawRoundedRect(self.rect(), r, r)


class ScrollBar(QWidget):
    """ Fluent scroll bar """

    rangeChanged = Signal(tuple)
    valueChanged = Signal(int)
    sliderPressed = Signal()
    sliderReleased = Signal()
    sliderMoved = Signal()

    def __init__(self, orient: Qt.Orientation, parent: QAbstractScrollArea, fadeout: bool = False):
        super().__init__(parent)
        self.groove = ScrollBarGroove(orient, self)
        self.handle = ScrollBarHandle(orient, self, fadeout)
        self.timer = QTimer(self)
        self.scroll_area = parent
        self.fadeout = fadeout

        self._orientation = orient
        self._singleStep = 1
        self._pageStep = 50
        self._padding = 0

        self._minimum = 0
        self._maximum = 0
        self._value = 0

        self._isPressed = False
        self.isEnter = False
        self._isExpanded = False
        self._pressedPos = QPoint()
        self._isForceHidden = False

        if orient == Qt.Vertical:
            self.partnerBar = parent.verticalScrollBar()
            QAbstractScrollArea.setVerticalScrollBarPolicy(parent, Qt.ScrollBarAlwaysOff)
        else:
            self.partnerBar = parent.horizontalScrollBar()
            QAbstractScrollArea.setHorizontalScrollBarPolicy(parent, Qt.ScrollBarAlwaysOff)

        self.__initWidget(parent)

    def __initWidget(self, parent):
        self.groove.opacityAni.valueChanged.connect(self._onOpacityAniValueChanged)

        self.partnerBar.rangeChanged.connect(self.setRange)
        self.partnerBar.valueChanged.connect(self._onValueChanged)
        self.valueChanged.connect(self.partnerBar.setValue)

        parent.installEventFilter(self)

        self.setRange(self.partnerBar.minimum(), self.partnerBar.maximum())
        self.setVisible(self.maximum() > 0 and not self._isForceHidden)
        self._adjustPos(self.parent().size())

    def _onPageUp(self):
        self.setValue(self.value() - self.pageStep())

    def _onPageDown(self):
        self.setValue(self.value() + self.pageStep())

    def _onValueChanged(self, value):
        self.val = value
        if self.fadeout and not self.isEnter:
            self.handle.prepareFadeout()

    def value(self):
        return self._value

    @Property(int, notify=valueChanged)
    def val(self):
        return self._value

    @val.setter
    def val(self, value: int):
        if value == self.value():
            return

        value = max(self.minimum(), min(value, self.maximum()))
        self._value = value
        self.valueChanged.emit(value)

        # adjust the position of handle
        self._adjustHandlePos()

    def minimum(self):
        return self._minimum

    def maximum(self):
        return self._maximum

    def orientation(self):
        return self._orientation

    def pageStep(self):
        return self._pageStep

    def singleStep(self):
        return self._singleStep

    def isSliderDown(self):
        return self._isPressed

    def setValue(self, value: int):
        self.val = value

    def setMinimum(self, min: int):
        if min == self.minimum():
            return

        self._minimum = min
        self.rangeChanged.emit((min, self.maximum()))

    def setMaximum(self, max: int):
        if max == self.maximum():
            return

        self._maximum = max
        self.rangeChanged.emit((self.minimum(), max))

    def setRange(self, min: int, max: int):
        if min > max or (min == self.minimum() and max == self.maximum()):
            return

        self.setMinimum(min)
        self.setMaximum(max)

        self._adjustHandleSize()
        self._adjustHandlePos()
        self.setVisible(max > 0 and not self._isForceHidden)

        self.rangeChanged.emit((min, max))

    def setPageStep(self, step: int):
        if step >= 1:
            self._pageStep = step

    def setSingleStep(self, step: int):
        if step >= 1:
            self._singleStep = step

    def setSliderDown(self, isDown: bool):
        self._isPressed = True
        if isDown:
            self.sliderPressed.emit()
        else:
            self.sliderReleased.emit()

    def expand(self):
        """ expand scroll bar """
        if self._isExpanded or not self.isEnter:
            return

        self._isExpanded = True
        self.groove.fadeIn()

    def collapse(self):
        """ collapse scroll bar """
        if not self._isExpanded or self.isEnter:
            return

        self._isExpanded = False
        self.groove.fadeOut()

    def enterEvent(self, e):
        self.isEnter = True
        self.timer.stop()
        self.timer.singleShot(200, self.expand)
        if self.fadeout:
            self.handle.stopFadeout()

    def leaveEvent(self, e):
        self.isEnter = False
        self.timer.stop()
        self.timer.singleShot(200, self.collapse)
        if self.fadeout:
            self.handle.prepareFadeout()

    def eventFilter(self, obj, e: QEvent):
        if obj is not self.parent():
            return super().eventFilter(obj, e)

        # adjust the position of slider
        if e.type() == QEvent.Resize:
            self._adjustPos(e.size())

        return super().eventFilter(obj, e)

    def resizeEvent(self, e):
        self.groove.resize(self.size())

    def mousePressEvent(self, e: QMouseEvent):
        super().mousePressEvent(e)
        self._isPressed = True
        self._pressedPos = e.pos()

        if self.childAt(e.pos()) is self.handle or not self._isSlideResion(e.pos()):
            return

        if self.orientation() == Qt.Vertical:
            if e.pos().y() > self.handle.geometry().bottom():
                value = e.pos().y() - self.handle.height() - self._padding
            else:
                value = e.pos().y() - self._padding
        else:
            if e.pos().x() > self.handle.geometry().right():
                value = e.pos().x() - self.handle.width() - self._padding
            else:
                value = e.pos().x() - self._padding

        self.setValue(int(value / max(self._slideLength(), 1) * self.maximum()))
        self.sliderPressed.emit()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self._isPressed = False
        self.sliderReleased.emit()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self.orientation() == Qt.Vertical:
            dv = e.pos().y() - self._pressedPos.y()
        else:
            dv = e.pos().x() - self._pressedPos.x()

        # don't use `self.setValue()`, because it could be reimplemented
        dv = int(dv / max(self._slideLength(), 1) * (self.maximum() - self.minimum()))
        ScrollBar.setValue(self, self.value() + dv)

        self._pressedPos = e.pos()
        self.sliderMoved.emit()

    def _adjustPos(self, size):
        if self.orientation() == Qt.Vertical:
            self.resize(12, size.height() - 2)
            self.move(size.width() - 13, 1)
        else:
            self.resize(size.width() - 2, 12)
            self.move(1, size.height() - 13)

    def _adjustHandleSize(self):
        p = self.parent()
        if self.orientation() == Qt.Vertical:
            total = self.maximum() - self.minimum() + p.height()
            s = int(self._grooveLength() * p.height() / max(total, 1))
            self.handle.setFixedHeight(max(30, s))
        else:
            total = self.maximum() - self.minimum() + p.width()
            s = int(self._grooveLength() * p.width() / max(total, 1))
            self.handle.setFixedWidth(max(30, s))

    def _adjustHandlePos(self):
        total = max(self.maximum() - self.minimum(), 1)
        delta = int(self.value() / total * self._slideLength())

        if self.orientation() == Qt.Vertical:
            x = self.width() - self.handle.width() - 3
            self.handle.move(x, self._padding + delta)
        else:
            y = self.height() - self.handle.height() - 3
            self.handle.move(self._padding + delta, y)

    def _grooveLength(self):
        if self.orientation() == Qt.Vertical:
            return self.height() - 2 * self._padding

        return self.width() - 2 * self._padding

    def _slideLength(self):
        if self.orientation() == Qt.Vertical:
            return self._grooveLength() - self.handle.height()

        return self._grooveLength() - self.handle.width()

    def _isSlideResion(self, pos: QPoint):
        if self.orientation() == Qt.Vertical:
            return self._padding <= pos.y() <= self.height() - self._padding

        return self._padding <= pos.x() <= self.width() - self._padding

    def _onOpacityAniValueChanged(self):
        if not self.fadeout:
            opacity = self.groove.opacityEffect.opacity()
            if self.orientation() == Qt.Vertical:
                self.handle.setFixedWidth(int(3 + opacity * 3))
            else:
                self.handle.setFixedHeight(int(3 + opacity * 3))

        self._adjustHandlePos()

    def setForceHidden(self, isHidden: bool):
        """ whether to force the scrollbar to be hidden """
        self._isForceHidden = isHidden
        self.setVisible(self.maximum() > 0 and not isHidden)

    def wheelEvent(self, e):
        QApplication.sendEvent(self.parent().viewport(), e)
