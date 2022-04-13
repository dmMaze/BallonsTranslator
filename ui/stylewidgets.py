from PyQt5.QtWidgets import QGraphicsOpacityEffect, QFrame, QWidget, QComboBox, QLabel, QSizePolicy, QDialog, QProgressBar, QMessageBox, QMenu, QSpacerItem, QPushButton, QHBoxLayout, QVBoxLayout, QToolButton, QSplitter, QStylePainter, QStyleOption, QStyle, QSlider, QProxyStyle, QStyle, QStyleOptionSlider, QColorDialog
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPointF, QRect, pyqtSignal, QSizeF, QObject, QEvent
from PyQt5.QtGui import QFontMetrics, QMouseEvent, QShowEvent, QWheelEvent, QResizeEvent, QKeySequence, QPainter, QTextFrame, QTransform, QTextBlock, QAbstractTextDocumentLayout, QTextLayout, QFont, QFontMetrics, QColor, QTextFormat, QTextCursor, QTextCharFormat, QTextDocument
from typing import List, Union, Tuple

from .constants import CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, CONFIG_COMBOBOX_SHORT

class Widget(QWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.style_option = QStyleOption()
        self.style_painter = QStylePainter()

    # https://wiki.qt.io/How_to_Change_the_Background_Color_of_QWidget#Using_Style_Sheet
    def paintEvent(self, event):
        self.style_option.initFrom(self)
        self.style_painter.begin(self)
        self.style_painter.drawPrimitive(QStyle.PE_Widget, self.style_option)
        self.style_painter.end()
        super().paintEvent(event)

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
    showed = pyqtSignal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setModal(True)
        layout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.detect_bar = TaskProgressBar('detect', self.tr('Detecting: '), self)
        self.ocr_bar = TaskProgressBar('ocr', self.tr('OCR: '), self)
        self.inpaint_bar = TaskProgressBar('inpaint', self.tr('Inpainting: '), self)
        self.translate_bar = TaskProgressBar('translate', self.tr('Translating: '), self)

        layout.addWidget(self.detect_bar)
        layout.addWidget(self.ocr_bar)
        layout.addWidget(self.inpaint_bar)
        layout.addWidget(self.translate_bar)

        layout.setSpacing(0)
        layout.setContentsMargins(20, 10, 20, 30)

        self.setStyleSheet("""
            QWidget {
                font-family: "Arial";
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

class SliderStyle(QProxyStyle):

    def subControlRect(self, control, option, subControl, widget=None):
        rect = super(SliderStyle, self).subControlRect(
            control, option, subControl, widget)
        if subControl == QStyle.SC_SliderHandle:
            if option.orientation == Qt.Horizontal:
                # 高度1/3
                radius = int(widget.height() / 2)
                offset = int(radius / 2)
                # if option.state & QStyle.State_MouseOver:
                x = min(rect.x() - offset, widget.width() - radius)
                x = x if x >= 0 else 0
                # else:
                #     radius = offset
                #     x = min(rect.x(), widget.width() - radius)
                rect = QRect(x, int((rect.height() - radius) / 2),
                             radius, radius)
            else:
                # 宽度1/3
                radius = int(widget.width() / 2)
                offset = int(radius / 2)
                # if option.state & QStyle.State_MouseOver:
                y = min(rect.y() - offset, widget.height() - radius)
                y = y if y >= 0 else 0
                # else:
                    # radius = offset
                    # y = min(rect.y(), widget.height() - radius)
                rect = QRect(int((rect.width() - radius) / 2),
                             y, radius, radius)
            return rect
        return rect


class ColorPicker(QLabel):
    colorChanged = pyqtSignal(bool)
    changingColor = pyqtSignal()
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


class PaintQSlider(QSlider):

    def __init__(self, draw_content, *args, **kwargs):
        super(PaintQSlider, self).__init__(*args, **kwargs)
        # 设置代理样式,主要用于计算和解决鼠标点击区域
        self.setStyle(SliderStyle())
        self.draw_content = draw_content
        self.pressed: bool = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.pressed = True
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.pressed = False
        return super().mouseReleaseEvent(event)

    def paintEvent(self, _):
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 中间圆圈的位置
        rect = self.style().subControlRect(
            QStyle.CC_Slider, option, QStyle.SC_SliderHandle, self)

        # 画中间白色线条
        painter.setPen(QColor(85,85,96))
        painter.setBrush(QColor(85,85,96))
        if self.orientation() == Qt.Horizontal:
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
            # 绘制文字
            if self.draw_content is not None:
                painter.setPen(QColor(85,85,96,255))
                font = painter.font()
                font.setPointSize(8)
                fm = QFontMetrics(font)
                painter.setFont(font)
                draw_content = self.draw_content.replace("value", str(self.value()))
                textw = fm.width(draw_content)

                if self.orientation() == Qt.Horizontal:  # 在上方绘制文字
                    x, y = rect.x() - textw/2 + rect.width()/2, rect.y() - rect.height() - 2
                    x = min(max(0, x), self.width()-textw)
                    # x = rect.x()
                else:  # 在左侧绘制文字
                    x, y = rect.x() - rect.width() - 2, rect.y()
                painter.drawText(
                    x, y-10, textw, rect.height()+20,
                    Qt.AlignCenter, self.draw_content.replace("value", str(self.value()))
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