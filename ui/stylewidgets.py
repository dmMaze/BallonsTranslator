from qtpy.QtWidgets import QApplication, QAbstractScrollArea, QGraphicsOpacityEffect, QFrame, QWidget, QComboBox, QLabel, QVBoxLayout, QStyle, QSlider, QHBoxLayout, QStyle, QStyleOptionSlider, QColorDialog, QPushButton, QSizePolicy
from qtpy.QtCore import QEvent, Qt, QPropertyAnimation, QEasingCurve, QTimer, QSize, QRect, QRectF, Signal, QPoint, Property, QAbstractAnimation
from qtpy.QtGui import QIcon, QFontMetrics, QMouseEvent, QWheelEvent, QPainter, QFontMetrics, QColor
from typing import List, Union, Tuple

from utils.shared import CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT
from utils import shared as C
from utils.config import pcfg


def isDarkTheme():
    return pcfg.darkmode

def themeColor():
    return QColor(30, 147, 229, 127)


class Widget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


# class ViewWidgetMetaclass(type, Widget):
#     def __call__(cls, *args, **kwargs):
#         """Called when you call MyNewClass() """
#         widget = type.__call__(cls, *args, **kwargs)
#         C.register_view_widget(widget)
#         return widget


class ViewWidget(Widget):

    config_name: str = ''
    config_expand_name: str = ''
    action_name: str = ''
    view_hide_btn_clicked = Signal(str)
    expend_changed = Signal()

    def __init__(self, content_widget: Widget, text=None, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        
        self.title_label = ExpandLabel(text, self)
        self.title_label.hidelabel.clicked.connect(self.on_view_hide_btn_clicked)
        self.content_widget = content_widget

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.title_label.clicked.connect(self.set_expend_area)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def on_view_hide_btn_clicked(self):
        self.view_hide_btn_clicked.emit(self.config_name)
    
    def register_view_widget(self, config_name: str, config_expand_name: str, action_name: str):
        self.config_name = config_name
        self.config_expand_name = config_expand_name
        self.action_name = action_name
        C.register_view_widget(self)

    def set_expend_area(self, expend: bool = None, set_config: bool = True):
        if expend is None:
            return self.set_expend_area(self.title_label.expanded)
        if self.title_label.expanded != expend:
            self.title_label.setExpand(expend)
        self.content_widget.setVisible(expend)
        if set_config:
            setattr(pcfg, self.config_expand_name, expend)

    def setTitle(self, text: str):
        self.title_label.textlabel.setText(text)

    def elidedText(self, text: str):
        fm = QFontMetrics(self.title_label.font())
        return fm.elidedText(text, Qt.TextElideMode.ElideRight, self.content_widget.width() - 40)

    def title(self) -> str:
        return self.title_label.textlabel.text()


class HidePanelButton(QPushButton):
    pass


CHEVRON_SIZE = 20
def chevron_down():
    return QIcon(r'icons/chevron-down.svg').pixmap(CHEVRON_SIZE, CHEVRON_SIZE, mode=QIcon.Mode.Normal)

def chevron_right():
    return QIcon(r'icons/chevron-right.svg').pixmap(CHEVRON_SIZE, CHEVRON_SIZE, mode=QIcon.Mode.Normal)


class ExpandLabel(Widget):

    clicked = Signal()

    def __init__(self, text=None, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.textlabel = QLabel(self)
        self.textlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        font = self.textlabel.font()
        if C.ON_MACOS:
            font.setPointSize(13)
        else:
            font.setPointSizeF(10)
        self.textlabel.setFont(font)
        self.arrowlabel = QLabel(self)
        self.arrowlabel.setFixedSize(CHEVRON_SIZE, CHEVRON_SIZE)
        self.arrowlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.hidelabel = HidePanelButton(self)
        self.hidelabel.setVisible(False)

        if text is not None:
            self.textlabel.setText(text)
        layout = QHBoxLayout(self)
        layout.addWidget(self.arrowlabel)
        layout.addWidget(self.textlabel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addStretch(-1)
        layout.addWidget(self.hidelabel)
    
        self.expanded = True
        self.setExpand(True)
        self.setFixedHeight(26)

    def enterEvent(self, event) -> None:
        self.hidelabel.setVisible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.hidelabel.setVisible(False)
        return super().leaveEvent(event)

    def setExpand(self, expand: bool):
        self.expanded = expand
        if expand:
            self.arrowlabel.setPixmap(chevron_down())
        else:
            self.arrowlabel.setPixmap(chevron_right())

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.setExpand(not self.expanded)
            pcfg.expand_tstyle_panel = self.expanded
            self.clicked.emit()
        return super().mousePressEvent(e)


class FadeLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # https://stackoverflow.com/questions/57828052/qpropertyanimation-not-working-with-window-opacity
        effect = QGraphicsOpacityEffect(self, opacity=1.0)
        self.setGraphicsEffect(effect)
        self.fadeAnimation = QPropertyAnimation(
            self,
            propertyName=b"opacity",
            targetObject=effect,
            duration=1200,
            startValue=1.0,
            endValue=0.,
        )
        self.fadeAnimation.setEasingCurve(QEasingCurve.Type.InQuint)
        self.fadeAnimation.finished.connect(self.hide)
        self.setHidden(True)
        self.gv = None

    def startFadeAnimation(self):
        self.show()
        self.fadeAnimation.stop()
        self.fadeAnimation.start()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.gv is not None:
            self.gv.wheelEvent(event)
        return super().wheelEvent(event)


class SeparatorWidget(QFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class ColorPicker(QLabel):
    colorChanged = Signal(bool)
    changingColor = Signal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.color: QColor = None

    def mousePressEvent(self, event):
        self.changingColor.emit()
        color = QColorDialog.getColor()
        is_valid = color.isValid()
        if is_valid:
            self.setPickerColor(color)
        self.colorChanged.emit(is_valid)

    def setPickerColor(self, color: Union[QColor, List, Tuple]):
        if not isinstance(color, QColor):
            color = QColor(*color)
        self.color = color
        r, g, b, a = color.getRgb()
        rgba = f'rgba({r}, {g}, {b}, {a})'
        self.setStyleSheet("background-color: " + rgba)

    def rgb(self) -> List:
        color = self.color
        return (color.red(), color.green(), color.blue())

    def rgba(self) -> List:
        color = self.color
        return (color.red(), color.green(), color.blue(), color.alpha())

    
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


class CustomComboBox(QComboBox):
    # https://stackoverflow.com/questions/3241830/qt-how-to-disable-mouse-scrolling-of-qcombobox
    def __init__(self, scrollWidget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        self.scrollWidget=scrollWidget
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.scroll_sel = True

    def setScrollSelectionEnabled(self, enable: bool):
        self.scroll_sel = enable

    def wheelEvent(self, *args, **kwargs):
        if self.scroll_sel and (self.scrollWidget is None or self.hasFocus()):
            return super().wheelEvent(*args, **kwargs)
        else:
            return self.scrollWidget.wheelEvent(*args, **kwargs)
        

class ConfigComboBox(CustomComboBox):

    def __init__(self, fix_size=True, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(scrollWidget, *args, **kwargs)
        self.fix_size = fix_size
        self.adjustSize()

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


class ClickableLabel(QLabel):

    clicked = Signal()

    def __init__(self, text=None, parent=None, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        if text is not None:
            self.setText(text)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(e)

    
class CheckableLabel(QLabel):

    checkStateChanged = Signal(bool)

    def __init__(self, checked_text: str, unchecked_text: str, default_checked: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checked_text = checked_text
        self.unchecked_text = unchecked_text
        self.checked = default_checked
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if default_checked:
            self.setText(checked_text)
        else:
            self.setText(unchecked_text)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self.checked)
            self.checkStateChanged.emit(self.checked)
        return super().mousePressEvent(e)

    def setChecked(self, checked: bool):
        self.checked = checked
        if checked:
            self.setText(self.checked_text)
        else:
            self.setText(self.unchecked_text)


class NoBorderPushBtn(QPushButton):
    pass


class TextChecker(QLabel):
    checkStateChanged = Signal(bool)
    def __init__(self, text: str, checked: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setText(text)
        self.setCheckState(checked)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def setCheckState(self, checked: bool):
        self.checked = checked
        if checked:
            self.setStyleSheet("QLabel { background-color: rgb(30, 147, 229); color: white; }")
        else:
            self.setStyleSheet("")

    def isChecked(self):
        return self.checked

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCheckState(not self.checked)
            self.checkStateChanged.emit(self.checked)


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
