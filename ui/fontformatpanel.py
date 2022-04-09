import functools
from typing import List, Tuple, Union

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QTextEdit, QScrollArea, QGraphicsDropShadowEffect, QVBoxLayout, QFrame, QFontComboBox, QColorDialog, QComboBox, QApplication, QPushButton, QRadioButton, QCheckBox
from PyQt5.QtCore import pyqtSignal, Qt, QSize, QEvent, QObject
from PyQt5.QtGui import QColor, QTextCharFormat, QIntValidator, QMouseEvent, QFont, QTextCursor, QTextFormat
from .stylewidgets import Widget, ColorPicker, PaintQSlider

from .misc import FontFormat, set_html_color
from .textitem import TextBlkItem, TextBlock
# restore text cursor status after formatting
def restore_textcursor(formatting_func):

    @functools.wraps(formatting_func)
    def wrapper(blkitem: TextBlkItem, *args, **kwargs):
        if blkitem is None:
            return
        stroke_width_before = blkitem.stroke_width
        cursor = blkitem.textCursor()
        set_all = not cursor.hasSelection()
        pos1 = cursor.position()
        pos2 = cursor.anchor().__pos__()
        if set_all:
            cursor.select(QTextCursor.Document)

        formatting_func(blkitem, cursor, *args, **kwargs)
        
        if not set_all:
            cursor.setPosition(min(pos1, pos2))
            cursor.setPosition(max(pos1, pos2), QTextCursor.KeepAnchor)
        else:
            cursor.setPosition(pos1)
        blkitem.setTextCursor(cursor)
        if blkitem.stroke_width != stroke_width_before:
            blkitem.repaint_background()
    return wrapper

@restore_textcursor
def set_textblk_color(blkitem: TextBlkItem, cursor: QTextCursor, rgb: List):
    fraghtml = cursor.selection().toHtml()
    cursor.insertHtml(set_html_color(fraghtml, rgb))
    

@restore_textcursor
def set_textblk_fontsize(blkitem: TextBlkItem, cursor: QTextCursor, fontsize):
    format = QTextCharFormat()
    format.setFontPointSize(fontsize)
    cursor.mergeCharFormat(format)
    doc = blkitem.document()
    lastpos = doc.rootFrame().lastPosition()
    if cursor.selectionStart() == 0 and \
        cursor.selectionEnd() == lastpos:
        font = doc.defaultFont()
        font.setPointSizeF(fontsize)
        doc.setDefaultFont(font)
    cursor.mergeBlockCharFormat(format)
    doc.documentLayout().reLayout()

@restore_textcursor
def set_textblk_weight(blkitem, cursor: QTextCursor, weight):
    format = QTextCharFormat()
    format.setFontWeight(weight)
    cursor.mergeCharFormat(format)

@restore_textcursor
def set_textblk_italic(blkitem, cursor: QTextCursor, italic: bool):
    format = QTextCharFormat()
    format.setFontItalic(italic)
    cursor.mergeCharFormat(format)

@restore_textcursor
def set_textblk_underline(blkitem, cursor: QTextCursor, underline: bool):
    format = QTextCharFormat()
    format.setFontUnderline(underline)
    cursor.mergeCharFormat(format)

@restore_textcursor
def set_textblk_alignment(blkitem: TextBlkItem, cursor: QTextCursor, alignment: int):
    alignment = [Qt.AlignLeft, Qt.AlignCenter, Qt.AlignRight][alignment]
    blkitem.setAlignment(alignment)

@restore_textcursor
def set_textblk_strokewidth(blkitem: TextBlkItem, cursor: QTextCursor, stroke_width: int):
    blkitem.setStrokeWidth(stroke_width)

@restore_textcursor
def set_textblk_strokecolor(blkitem: TextBlkItem, cursor: QTextCursor, stroke_color: List):
    blkitem.setStrokeColor(stroke_color)

@restore_textcursor
def set_textblk_family(blkitem: TextBlkItem, cursor: QTextCursor, family: str):
    format = cursor.blockCharFormat()
    format.setFontFamily(family)
    cursor.setBlockCharFormat(format)
    doc = blkitem.document()
    lastpos = doc.rootFrame().lastPosition()
    if cursor.selectionStart() == 0 and \
        cursor.selectionEnd() == lastpos:
        font = doc.defaultFont()
        font.setFamily(family)
        doc.setDefaultFont(font)
    cursor.mergeCharFormat(format)

@restore_textcursor
def set_textblk_linespacing(blkitem: TextBlkItem, cursor: QTextCursor, line_spacing: float):
    blkitem.setLineSpacing(line_spacing)

class IncrementalBtn(QPushButton):
    pass

class QFontChecker(QCheckBox):
    pass

class AlignmentChecker(QCheckBox):
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.isChecked():
            return event.accept()
        return super().mousePressEvent(event)

class AlignmentBtnGroup(QFrame):
    set_alignment = pyqtSignal(int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignLeftChecker = AlignmentChecker(self)
        self.alignLeftChecker.clicked.connect(self.alignBtnPressed)
        self.alignCenterChecker = AlignmentChecker(self)
        self.alignCenterChecker.clicked.connect(self.alignBtnPressed)
        self.alignRightChecker = AlignmentChecker(self)
        self.alignRightChecker.clicked.connect(self.alignBtnPressed)
        self.alignLeftChecker.setObjectName("AlignLeftChecker")
        self.alignRightChecker.setObjectName("AlignRightChecker")
        self.alignCenterChecker.setObjectName("AlignCenterChecker")

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.alignLeftChecker)
        hlayout.addWidget(self.alignCenterChecker)
        hlayout.addWidget(self.alignRightChecker)
        hlayout.setSpacing(0)

    def alignBtnPressed(self):
        btn = self.sender()
        if btn == self.alignLeftChecker:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.set_alignment.emit(0)
        elif btn == self.alignRightChecker:
            self.alignRightChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignLeftChecker.setChecked(False)
            self.set_alignment.emit(2)
        else:
            self.alignCenterChecker.setChecked(True)
            self.alignLeftChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.set_alignment.emit(1)
        # btn.setChecked(True)
    
    def setAlignment(self, alignment: int):
        if alignment == 0:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
        elif alignment == 1:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(True)
            self.alignRightChecker.setChecked(False)
        else:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(True)

class FormatGroupBtn(QFrame):
    set_bold = pyqtSignal(bool)
    set_italic = pyqtSignal(bool)
    set_underline = pyqtSignal(bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boldBtn = QFontChecker(self)
        self.boldBtn.setObjectName("FontBoldChecker")
        self.boldBtn.clicked.connect(self.setBold)
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.italicBtn.clicked.connect(self.setItalic)
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.underlineBtn.clicked.connect(self.setUnderline)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.boldBtn)
        hlayout.addWidget(self.italicBtn)
        hlayout.addWidget(self.underlineBtn)
        hlayout.setSpacing(0)

    def setBold(self):
        self.set_bold.emit(self.boldBtn.isChecked())

    def setItalic(self):
        self.set_italic.emit(self.italicBtn.isChecked())

    def setUnderline(self):
        self.set_underline.emit(self.underlineBtn.isChecked())
    
class FontSizeBox(QFrame):
    fontsize_changed = pyqtSignal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.fcombobox = QComboBox(self)
        self.fcombobox.setFixedWidth(200)
        self.fcombobox.setEditable(True)
        self.fcombobox.setObjectName("FontSizeComboBox")
        self.fcombobox.editTextChanged.connect(self.fontsize_changed)
        self.fcombobox.addItems([
            "5", "5.5", "6.5", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", '22', "26", "28", 
            "36", "48", "56", "72"
        ])
        validator = QIntValidator()
        validator.setTop(1000)
        validator.setBottom(1)
        self.fcombobox.setValidator(validator)

        hlayout = QHBoxLayout(self)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.upBtn)
        vlayout.addWidget(self.downBtn)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        hlayout.addLayout(vlayout)
        hlayout.addWidget(self.fcombobox)
        hlayout.setSpacing(3)

        self.btn_clicked = False

    def getFontSize(self):
        return float(self.fcombobox.currentText())

    def onUpBtnClicked(self):
        self.btn_clicked = True
        size = self.getFontSize()
        newsize = int(round(size * 1.25))
        if newsize == size:
            newsize += 1
        newsize = min(1000, newsize)
        if newsize != size:
            self.fcombobox.setCurrentText(str(newsize))
        
    def onDownBtnClicked(self):
        self.btn_clicked = True
        size = self.getFontSize()
        newsize = int(round(size * 0.75))
        if newsize == size:
            newsize -= 1
        newsize = max(1, newsize)
        if newsize != size:
            self.fcombobox.setCurrentText(str(newsize))
        
    def isActive(self):
        active = self.btn_clicked or self.fcombobox.hasFocus()
        self.btn_clicked = False
        return active

class FontFormatPanel(Widget):
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    active_format: FontFormat = None
    global_format: FontFormat = None
    restoring_textblk: bool = False
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignTop)
        self.familybox = QFontComboBox(self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.currentFontChanged.connect(self.onfontFamilyChanged)

        self.fontsizebox = FontSizeBox(self)
        self.fontsizebox.setToolTip(self.tr("Font Size"))
        self.fontsizebox.setObjectName("FontSizeBox")
        self.fontsizebox.fcombobox.editTextChanged.connect(self.onfontSizeChanged)
        
        self.colorPicker = ColorPicker(self)
        self.colorPicker.setObjectName("FontColorPicker")
        self.colorPicker.setToolTip(self.tr("Change font color"))
        self.colorPicker.changingColor.connect(self.changingColor)
        self.colorPicker.colorChanged.connect(self.onColorChanged)

        self.alignBtnGroup = AlignmentBtnGroup(self)
        self.alignBtnGroup.set_alignment.connect(self.onAlignmentChanged)

        self.formatBtnGroup = FormatGroupBtn(self)
        self.formatBtnGroup.set_bold.connect(self.onfontBoldChanged)
        self.formatBtnGroup.set_italic.connect(self.onfontItalicChanged)
        self.formatBtnGroup.set_underline.connect(self.onfontUnderlineChanged)

        self.verticalChecker = QFontChecker(self)
        self.verticalChecker.setObjectName("FontVerticalChecker")
        self.verticalChecker.clicked.connect(self.onOrientationChanged)

        self.strokeColorPicker = ColorPicker(self)
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onStrokeColorChanged)
        self.strokeColorPicker.setObjectName("StrokeColorPicker")
        
        self.strokeWidthSlider = PaintQSlider(self.tr("Stroke width: ") + 'value%', Qt.Horizontal)
        self.strokeWidthSlider.setFixedHeight(50)
        self.strokeWidthSlider.setRange(0, 100)
        self.strokeWidthSlider.valueChanged.connect(self.onSrokeWidthChanged)

        self.lineSpacingSlider = PaintQSlider(self.tr("line spacing: ") + 'value%', Qt.Horizontal)
        self.lineSpacingSlider.setFixedHeight(50)
        self.lineSpacingSlider.setRange(0, 300)
        self.lineSpacingSlider.valueChanged.connect(self.onLinespacingChanged)

        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.setSpacing(10)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.setSpacing(10)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignLeft)
        hl3.addWidget(self.lineSpacingSlider)
        hl3.addWidget(self.strokeColorPicker)
        hl3.addWidget(self.strokeWidthSlider)
        hl3.setContentsMargins(5, 5, 5, 5)
        hl3.setSpacing(20)

        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.setContentsMargins(10, 10, 10, 10)
        self.setFixedWidth(520)

        self.focusOnColorDialog = False
        self.active_format = self.global_format

    def restoreTextBlkItem(self):
        blkitem = self.textblk_item
        self.restoring_textblk = True
        if blkitem:
            blkitem.startEdit()
            blkitem.setTextCursor(self.text_cursor)
            blkitem.scene().gv.setFocus(True)
        self.restoring_textblk = False
        return blkitem

    def changingColor(self):
        self.focusOnColorDialog = True

    def onColorChanged(self, is_valid=True):
        self.active_format.frgb = self.colorPicker.rgb()
        self.focusOnColorDialog = False
        self.restoreTextBlkItem()
        if is_valid:
            set_textblk_color(self.textblk_item, self.active_format.frgb)

    def onStrokeColorChanged(self, is_valid=True):
        self.active_format.srgb = self.strokeColorPicker.rgb()
        self.focusOnColorDialog = False
        self.restoreTextBlkItem()
        if is_valid:
            set_textblk_strokecolor(self.textblk_item, self.active_format.srgb)

    def onfontSizeChanged(self):
        if self.fontsizebox.isActive():
            self.active_format.size = self.fontsizebox.getFontSize()
            self.restoreTextBlkItem()
            set_textblk_fontsize(self.textblk_item, self.active_format.size)

    def onfontFamilyChanged(self):
        if self.familybox.hasFocus():
            self.active_format.family = self.familybox.currentText()
            self.restoreTextBlkItem()
            set_textblk_family(self.textblk_item, self.active_format.family)

    def onfontBoldChanged(self, checked: bool):
        if checked:
            self.active_format.weight = QFont.Bold
            self.active_format.bold = True
        else:
            self.active_format.weight = QFont.Normal
            self.active_format.bold = False
        self.restoreTextBlkItem()
        set_textblk_weight(self.textblk_item, self.active_format.weight)
        
    def onfontUnderlineChanged(self, checked: bool):
        self.active_format.underline = checked
        self.restoreTextBlkItem()
        set_textblk_underline(self.textblk_item, self.active_format.underline)

    def onfontItalicChanged(self, checked: bool):
        self.active_format.italic = checked
        self.restoreTextBlkItem()
        set_textblk_italic(self.textblk_item, self.active_format.italic)

    def onAlignmentChanged(self, alignment):
        self.active_format.alignment = alignment
        set_textblk_alignment(self.textblk_item, self.active_format.alignment)
        self.restoreTextBlkItem()
            
    def onOrientationChanged(self):
        self.active_format.vertical = self.verticalChecker.isChecked()
        self.restoreTextBlkItem()
        if self.textblk_item is not None:
            self.textblk_item.setVertical(self.active_format.vertical)

    def onSrokeWidthChanged(self):
        if self.strokeWidthSlider.pressed:
            self.active_format.stroke_width = self.strokeWidthSlider.value() / 100
            self.restoreTextBlkItem()
            set_textblk_strokewidth(self.textblk_item, self.active_format.stroke_width)

    def onLinespacingChanged(self):
        if self.lineSpacingSlider.pressed:
            self.active_format.line_spacing = self.lineSpacingSlider.value() / 100
            self.restoreTextBlkItem()
            set_textblk_linespacing(self.textblk_item, self.active_format.line_spacing)
            
    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.fontsizebox.fcombobox.setCurrentText(str(int(font_format.size)))
        self.familybox.setCurrentText(font_format.family)
        self.colorPicker.setPickerColor(font_format.frgb)
        self.strokeColorPicker.setPickerColor(font_format.srgb)
        self.strokeWidthSlider.setValue(font_format.stroke_width * 100)
        self.lineSpacingSlider.setValue(font_format.line_spacing * 100)
        self.verticalChecker.setChecked(font_format.vertical)
        self.formatBtnGroup.boldBtn.setChecked(font_format.bold)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.alignBtnGroup.setAlignment(font_format.alignment)

    def set_textblk_item(self, textblk_item: TextBlkItem = None):
        if textblk_item is None:
            # if self.textblk_item:
            focus_w = self.app.focusWidget()
            focus_p = None if focus_w is None else focus_w.parentWidget()
            focus_on_fmtoptions = False
            if self.focusOnColorDialog:
                focus_on_fmtoptions = True
            elif focus_p:
                if focus_p == self or\
                    focus_p.parentWidget() == self:
                    focus_on_fmtoptions = True
            if focus_on_fmtoptions:
                self.text_cursor = QTextCursor(self.textblk_item.textCursor())
            else:
                self.textblk_item = None
                self.text_cursor = None
                self.set_active_format(self.global_format)
            return

        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                self.textblk_item = textblk_item
                self.text_cursor = QTextCursor(self.textblk_item.textCursor())
                self.set_active_format(blk_fmt)