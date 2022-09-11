from qtpy.QtWidgets import QGraphicsOpacityEffect, QFrame, QWidget, QComboBox, QLabel, QSizePolicy, QDialog, QProgressBar, QMessageBox, QVBoxLayout, QStyle, QSlider, QProxyStyle, QStyle, QStyleOptionSlider, QColorDialog
from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPointF, QRect, Signal
from qtpy.QtGui import QFontMetrics, QMouseEvent, QShowEvent, QWheelEvent, QPainter, QFontMetrics, QColor
from typing import List, Union, Tuple

from .constants import CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT, HORSLIDER_FIXHEIGHT


class Widget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


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
        self.fadeAnimation.setEasingCurve(QEasingCurve.InQuint)
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


class TaskProgressBar(Widget):
    def __init__(self, task_name: str, description: str = '', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.progressbar = QProgressBar(self)
        self.progressbar.setTextVisible(False)
        self.textlabel = QLabel(self)
        self.description = description
        self.text_len = 100
        layout = QVBoxLayout(self)
        layout.addWidget(self.textlabel)
        layout.addWidget(self.progressbar)
        self.updateProgress(0)

    def updateProgress(self, progress: int, msg: str = ''):
        self.progressbar.setValue(progress)
        if self.description:
            msg = self.description + msg
        if len(msg) > self.text_len - 3:
            msg = msg[:self.text_len - 3] + '...'
        elif len(msg) < self.text_len:
            msg = msg + ' ' * (self.text_len - len(msg))
        self.textlabel.setText(msg)
        self.progressbar.setValue(progress)


class FrameLessMessageBox(QMessageBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        

class ProgressMessageBox(QDialog):
    showed = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setModal(True)
        
        self.detect_bar = TaskProgressBar('detect', self.tr('Detecting: '), self)
        self.ocr_bar = TaskProgressBar('ocr', self.tr('OCR: '), self)
        self.inpaint_bar = TaskProgressBar('inpaint', self.tr('Inpainting: '), self)
        self.translate_bar = TaskProgressBar('translate', self.tr('Translating: '), self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.detect_bar)
        layout.addWidget(self.ocr_bar)
        layout.addWidget(self.inpaint_bar)
        layout.addWidget(self.translate_bar)
        layout.setSpacing(0)
        layout.setContentsMargins(20, 10, 20, 30)

        self.setStyleSheet("""
            QWidget {
                font-size: 13pt;
                /* border-style: none; */
                color: #5d5d5f;
                background-color: #ebeef5;

            }
            Widget {
                background-color: #ebeef5;
            }
            QProgressBar {
                border: 0px;
                text-align: center;
                max-height: 3px;
                background-color: #e1e4eb;
            }
            QProgressBar::chunk {
                background-color: rgb(30, 147, 229);
            }
        """)

    def updateDetectProgress(self, value: int, msg: str = ''):
        self.detect_bar.updateProgress(value, msg)

    def updateOCRProgress(self, value: int, msg: str = ''):
        self.ocr_bar.updateProgress(value, msg)

    def updateInpaintProgress(self, value: int, msg: str = ''):
        self.inpaint_bar.updateProgress(value, msg)

    def updateTranslateProgress(self, value: int, msg: str = ''):
        self.translate_bar.updateProgress(value, msg)
    
    def zero_progress(self):
        self.updateDetectProgress(0)
        self.updateOCRProgress(0)
        self.updateInpaintProgress(0)
        self.updateTranslateProgress(0)

    def showEvent(self, e: QShowEvent) -> None:
        self.showed.emit()
        return super().showEvent(e)


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
        return [color.red(), color.green(), color.blue()]

    def rgba(self) -> List:
        color = self.color
        return [color.red(), color.green(), color.blue(), color.alpha()]


class SliderProxyStyle(QProxyStyle):

    def subControlRect(self, cc, opt, sc, widget):
        r = super().subControlRect(cc, opt, sc, widget)
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


class PaintQSlider(QSlider):

    mouse_released = Signal()

    def __init__(self, draw_content, orientation=Qt.Orientation.Horizontal, *args, **kwargs):
        super(PaintQSlider, self).__init__(orientation, *args, **kwargs)
        self.draw_content = draw_content
        self.pressed: bool = False
        self.setStyle(SliderProxyStyle())
        if orientation == Qt.Orientation.Horizontal:
            self.setFixedHeight(HORSLIDER_FIXHEIGHT)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = True
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressed = False
            self.mouse_released.emit()
        return super().mouseReleaseEvent(event)

    def paintEvent(self, _):
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 中间圆圈的位置
        rect = self.style().subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)
        
        # 画中间白色线条
        painter.setPen(QColor(85,85,96))
        painter.setBrush(QColor(85,85,96))
        if self.orientation() == Qt.Orientation.Horizontal:
            y = self.height() / 2
            painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
        else:
            x = self.width() / 2
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        # 画圆
        painter.setPen(Qt.NoPen)
        if option.state & QStyle.State_MouseOver:  # 双重圆
            # 半透明大圆
            r = rect.height() / 2
            painter.setBrush(QColor(85,85,96,100))
            painter.drawRoundedRect(rect, r, r)
            # 实心小圆(上下左右偏移4)
            rect = rect.adjusted(4, 4, -4, -4)
            r = rect.height() / 2
            painter.setBrush(QColor(85,85,96,255))
            painter.drawRoundedRect(rect, r, r)
            if self.draw_content is not None:
                painter.setPen(QColor(85,85,96,255))
                font = painter.font()
                font.setPointSize(8)
                fm = QFontMetrics(font)
                painter.setFont(font)
                draw_content = self.draw_content.replace("value", str(self.value()))
                textw = fm.width(draw_content)

                if self.orientation() == Qt.Orientation.Horizontal:  # 在上方绘制文字
                    x, y = rect.x() - textw/2 + rect.width()/2, rect.y() - rect.height()
                    x = min(max(0, x), self.width()-textw)
                    # x = rect.x()
                else:  # 在左侧绘制文字
                    x, y = rect.x() - rect.width(), rect.y()
                painter.drawText(
                    x, y-10, textw, rect.height()+20,
                    Qt.AlignmentFlag.AlignCenter, self.draw_content.replace("value", str(self.value()))
                )

        else:  # 实心圆
            rect = rect.adjusted(4, 4, -4, -4)
            r = rect.height() / 2
            painter.setBrush(QColor(85,85,96,200))
            painter.drawRoundedRect(rect, r, r)


class ConfigComboBox(QComboBox):

    def __init__(self, fix_size=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fix_size = fix_size
        self.adjustSize()

    def addItems(self, texts: List[str]) -> None:
        super().addItems(texts)
        self.adjustSize()

    def adjustSize(self) -> None:
        super().adjustSize()
        width = self.minimumSizeHint().width() + 100
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

    def __init__(self, text=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if text is not None:
            self.setText(text)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(e)          


