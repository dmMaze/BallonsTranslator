
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QFrame, QFontComboBox, QColorDialog, QComboBox, QApplication, QPushButton, QRadioButton, QCheckBox
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QEvent, QObject
from PyQt5.QtGui import QColor, QFocusEvent, QIntValidator, QMouseEvent, QFont, QTextCursor
from .stylewidgets import Widget, SeparatorWidget, PaintQSlider

from typing import List
from .textitem import TextBlock, TextBlkItem
from .fontformatpanel import FontFormatPanel



class SourceTextEdit(QTextEdit):
    hover_enter = pyqtSignal(int)
    hover_leave = pyqtSignal(int)
    user_edited = pyqtSignal()
    def __init__(self, idx, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.idx = idx
        self.setMinimumHeight(75)
        self.document().contentsChanged.connect(self.on_content_changed)
        self.document().documentLayout().documentSizeChanged.connect(self.adjustSize)

    def adjustSize(self):
        h = self.document().documentLayout().documentSize().toSize().height()
        self.setFixedHeight(max(h, 75))

    def on_content_changed(self):
        if self.hasFocus():
            self.user_edited.emit()

    def setHoverEffect(self, hover: bool):
        try:
            if hover:
                se = QGraphicsDropShadowEffect()
                se.setBlurRadius(12)
                se.setOffset(0, 0)
                se.setColor(QColor(30, 147, 229))
                self.setGraphicsEffect(se)
            else:
                self.setGraphicsEffect(None)
        except RuntimeError:
            pass

    def enterEvent(self, event: QEvent) -> None:
        self.setHoverEffect(True)
        self.hover_enter.emit(self.idx)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.setHoverEffect(False)
        self.hover_leave.emit(self.idx)
        return super().leaveEvent(event)

    def focusInEvent(self, event: QFocusEvent) -> None:
        self.setHoverEffect(True)
        return super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.setHoverEffect(False)
        return super().focusOutEvent(event)
        
class TransTextEdit(SourceTextEdit):
    content_change = pyqtSignal(int, str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.document().contentsChange.connect(self.onContentChange)

    def onContentChange(self, pos: int, delete: int, add: int):
        if self.hasFocus():
            text = self.toPlainText()
            self.content_change.emit(self.idx, text)

class TransPairWidget(Widget):
    def __init__(self, textblock: TextBlock = None, idx: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.e_source = SourceTextEdit(idx, self)
        self.e_trans = TransTextEdit(idx, self)
        self.textblock = textblock
        self.idx = idx
        vlayout = QVBoxLayout(self)
        vlayout.setAlignment(Qt.AlignTop)
        vlayout.addWidget(self.e_source)
        vlayout.addWidget(self.e_trans)
        vlayout.addWidget(SeparatorWidget(self))
        vlayout.setSpacing(20)

    def updateIndex(self, idx):
        self.idx = idx
        self.e_source.idx = idx
        self.e_trans.idx = idx

class TextEditListScrollArea(QScrollArea):
    textblock_list: List[TextBlock] = []
    pairwidget_list: List[TransPairWidget] = []
    remove_textblock = pyqtSignal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scrollContent = QFrame()
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignTop)
        vlayout.setSpacing(0)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.vlayout = vlayout
        
    def addPairWidget(self, pairwidget):
        
        self.vlayout.addWidget(pairwidget)
        pairwidget.setVisible(True)

    def removeWidget(self, widget: TransPairWidget):
        widget.setVisible(False)
        self.vlayout.removeWidget(widget)


class TextPanel(Widget):
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textEditList = TextEditListScrollArea(self)
        self.activePair: TransPairWidget = None
        self.formatpanel = FontFormatPanel(app, self)
        layout.addWidget(self.formatpanel)
        layout.addWidget(self.textEditList)
        layout.setContentsMargins(10, 0, 10, 0)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignCenter)

    
        
        