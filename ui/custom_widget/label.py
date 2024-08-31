from qtpy.QtWidgets import QGraphicsOpacityEffect, QLabel, QColorDialog
from qtpy.QtCore import  Qt, QPropertyAnimation, QEasingCurve, Signal
from qtpy.QtGui import QMouseEvent, QWheelEvent, QColor
from typing import List, Union, Tuple

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


class ColorPickerLabel(QLabel):
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


class TextCheckerLabel(QLabel):
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


