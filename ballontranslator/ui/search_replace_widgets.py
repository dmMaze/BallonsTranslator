from qtpy.QtWidgets import QHBoxLayout, QPushButton, QTextEdit, QLabel, QTreeView, QDialog, QCheckBox, QMessageBox, QVBoxLayout, QStyle, QSlider, QProxyStyle, QStyle,  QGraphicsDropShadowEffect, QWidget
from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve, QPointF, QRect, Signal
from qtpy.QtGui import QFontMetrics, QMouseEvent, QShowEvent, QWheelEvent, QPainter, QFontMetrics, QColor
from typing import List, Union, Tuple

from .stylewidgets import Widget, ClickableLabel

class SearchEditor(QTextEdit):
    height_changed = Signal()
    def __init__(self, parent: QWidget = None, original_height: int = 32, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.original_height = original_height
        self.setFixedHeight(original_height)
        self.document().documentLayout().documentSizeChanged.connect(self.adjustSize)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def adjustSize(self):
        h = self.document().documentLayout().documentSize().toSize().height()
        _h = self.geometry().height()
        if _h != h:
            self.setFixedHeight(max(h, self.original_height))
            self.height_changed.emit()

class SearchWidget(Widget):

    def __init__(self, parent: QWidget = None, is_floating=True, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.is_floating = is_floating

        self.search_editor = SearchEditor(self)
        self.search_editor.setAcceptRichText(False)
        self.search_editor.setPlaceholderText(self.tr('Find'))
        self.search_editor.height_changed.connect(self.on_editor_height_changed)
        self.result_counter = QLabel(self.tr('No result'))
        self.result_counter.setMaximumHeight(32)
        self.prev_match_btn = ClickableLabel(None, self)
        self.prev_match_btn.setObjectName('PrevMatchBtn')
        self.next_match_btn = ClickableLabel(None, self)
        self.next_match_btn.setObjectName('NextMatchBtn')
        self.case_sensitive_toggle = QCheckBox(self)
        self.case_sensitive_toggle.setObjectName('CaseSensitiveToggle')

        self.replace_editor = SearchEditor(self)
        self.replace_editor.setAcceptRichText(False)
        self.replace_editor.setPlaceholderText(self.tr('Replace'))
        self.replace_btn = ClickableLabel(None, self)
        self.replace_btn.setObjectName(self.tr('ReplaceBtn'))
        self.replace_all_btn = ClickableLabel(None, self)
        self.replace_all_btn.setObjectName(self.tr('ReplaceAllBtn'))

        self.replace_toggle = QCheckBox(self)
        self.replace_toggle.setObjectName('ReplaceToggle')
        self.replace_toggle.stateChanged.connect(self.on_replace_statechanged)
        self.replace_toggle.setToolTip(self.tr('Toggle Replace'))

        hlayout_bar1_0 = QHBoxLayout()
        hlayout_bar1_0.addWidget(self.search_editor)
        hlayout_bar1_0.addWidget(self.result_counter)
        hlayout_bar1_0.setAlignment(Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_0.setSpacing(5)

        hlayout_bar1_1 = QHBoxLayout()
        hlayout_bar1_1.addWidget(self.case_sensitive_toggle)
        hlayout_bar1_1.addWidget(self.prev_match_btn)
        hlayout_bar1_1.addWidget(self.next_match_btn)
        hlayout_bar1_1.setAlignment(hlayout_bar1_1.alignment() | Qt.AlignmentFlag.AlignTop)
        hlayout_bar1_1.setSpacing(5)

        hlayout_bar1 = QHBoxLayout()
        hlayout_bar1.addLayout(hlayout_bar1_0)
        hlayout_bar1.addLayout(hlayout_bar1_1)
        

        self.replace_layout = hlayout_bar2 = QHBoxLayout()
        hlayout_bar2.addWidget(self.replace_editor)
        hlayout_bar2.addWidget(self.replace_btn)
        hlayout_bar2.addWidget(self.replace_all_btn)
        hlayout_bar2.setSpacing(5)
        hlayout_bar2.addStretch()

        self.content_layout = vlayout = QVBoxLayout()
        vlayout.addLayout(hlayout_bar1)
        vlayout.addLayout(hlayout_bar2)

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.replace_toggle)
        hlayout.addLayout(vlayout)
        hlayout.setSpacing(0)

        if self.is_floating:
            self.close_btn = ClickableLabel(None, self)
            self.close_btn.setObjectName('SearchCloseBtn')
            self.close_btn.clicked.connect(self.on_close_button_clicked)
            hlayout_bar1_1.addWidget(self.close_btn)
            e = QGraphicsDropShadowEffect(self)
            e.setOffset(5, 5)
            e.setBlurRadius(25)
            self.setGraphicsEffect(e)
            self.setFixedWidth(460)
            self.search_editor.setFixedWidth(200)
            self.search_editor.setSizeAdjustPolicy(32)
            self.replace_editor.setFixedWidth(200)
            

    def on_close_button_clicked(self):
        self.hide()

    def on_editor_height_changed(self):
        tgt_size = self.search_editor.height() + 20
        if not self.replace_editor.isHidden():
            tgt_size += self.replace_editor.height() + 10
        self.setFixedHeight(tgt_size)

    def setReplaceWidgetsVisibility(self, visible: bool):
        self.replace_editor.setVisible(visible)
        self.replace_all_btn.setVisible(visible)
        self.replace_btn.setVisible(visible)

    def on_replace_statechanged(self):
        if self.replace_toggle.isChecked():
             self.setReplaceWidgetsVisibility(True)
        else:
            self.setReplaceWidgetsVisibility(False)
        self.on_editor_height_changed()