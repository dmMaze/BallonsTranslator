from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QComboBox, QApplication, QPushButton, QCheckBox, QLabel
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor, QTextCharFormat, QDoubleValidator, QMouseEvent, QFont, QTextCursor, QFocusEvent, QKeyEvent


class EffectBtn(QLabel):

    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text = self.tr('Effect')
        self.setText(self.text)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        return super().mousePressEvent(e)