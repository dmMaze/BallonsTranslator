import functools
from typing import List, Tuple, Union

from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QComboBox, QApplication, QPushButton, QCheckBox, QLabel
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QColor, QTextCharFormat, QDoubleValidator, QMouseEvent, QFont, QTextCursor, QFocusEvent, QKeyEvent

from .stylewidgets import Widget, ColorPicker
from .misc import FontFormat, set_html_color
from .textitem import TextBlkItem
from .canvas import Canvas
from .constants import CONFIG_FONTSIZE_CONTENT, WIDGET_SPACING_CLOSE
from . import constants as C

from utils.logger import logger as LOGGER


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
            cursor.select(QTextCursor.SelectionType.Document)

        formatting_func(blkitem, cursor, *args, **kwargs)
        
        if not set_all:
            cursor.setPosition(min(pos1, pos2))
            cursor.setPosition(max(pos1, pos2), QTextCursor.MoveMode.KeepAnchor)
        else:
            cursor.setPosition(pos1)
        blkitem.setTextCursor(cursor)
        if blkitem.stroke_width != stroke_width_before:
            blkitem.repaint_background()
    return wrapper

@restore_textcursor
def set_textblk_color(blkitem: TextBlkItem, cursor: QTextCursor, rgb: List):
    if not blkitem.document().isEmpty():
        fraghtml = cursor.selection().toHtml()
        cursor.insertHtml(set_html_color(fraghtml, rgb))
    else:
        fmt = cursor.charFormat()
        fmt.setForeground(QColor(*rgb))
        cursor.setCharFormat(fmt)
    
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
    blkitem.layout.reLayout()

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
    alignment = [Qt.AlignmentFlag.AlignLeft, Qt.AlignmentFlag.AlignCenter, Qt.AlignmentFlag.AlignRight][alignment]
    blkitem.setAlignment(alignment)

@restore_textcursor
def set_textblk_strokewidth(blkitem: TextBlkItem, cursor: QTextCursor, stroke_width: int):
    blkitem.setStrokeWidth(stroke_width)

@restore_textcursor
def set_textblk_strokecolor(blkitem: TextBlkItem, cursor: QTextCursor, stroke_color: List):
    blkitem.setStrokeColor(stroke_color)

@restore_textcursor
def set_textblk_family(blkitem: TextBlkItem, cursor: QTextCursor, family: str):

    doc = blkitem.document()
    lastpos = doc.rootFrame().lastPosition()
    if cursor.selectionStart() == 0 and \
        cursor.selectionEnd() == lastpos:
        font = doc.defaultFont()
        font.setFamily(family)
        doc.setDefaultFont(font)

    sel_start = cursor.selectionStart()
    sel_end = cursor.selectionEnd()
    block = doc.firstBlock()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            
            frag_start = fragment.position()
            frag_end = frag_start + fragment.length()
            pos2 = min(frag_end, sel_end)
            pos1 = max(frag_start, sel_start)
            if pos1 < pos2:
                cfmt = fragment.charFormat()
                under_line = cfmt.fontUnderline()
                cfont = cfmt.font()
                font = QFont(family, cfont.pointSizeF(), cfont.weight(), cfont.italic())
                font.setBold(font.bold())
                font.setWordSpacing(cfont.wordSpacing())
                font.setLetterSpacing(cfont.letterSpacingType(), cfont.letterSpacing())
                cfmt.setFont(font)
                cfmt.setFontUnderline(under_line)
                cursor.setPosition(pos1)
                cursor.setPosition(pos2, QTextCursor.MoveMode.KeepAnchor)
                cursor.setCharFormat(cfmt)
            it += 1
        block = block.next()

@restore_textcursor
def set_textblk_linespacing(blkitem: TextBlkItem, cursor: QTextCursor, line_spacing: float):
    blkitem.setLineSpacing(line_spacing)

@restore_textcursor
def set_textblk_letterspacing(blkitem: TextBlkItem, cursor: QTextCursor, letter_spacing: float):
    blkitem.setLetterSpacing(letter_spacing)


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
    set_alignment = Signal(int)
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
    set_bold = Signal(bool)
    set_italic = Signal(bool)
    set_underline = Signal(bool)
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
    
class SizeComboBox(QComboBox):
    
    apply_change = Signal(float)
    def __init__(self, val_range: List = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.text_changed_by_user = False
        self.editTextChanged.connect(self.on_text_changed)
        self.currentIndexChanged.connect(self.on_current_index_changed)
        self.setEditable(True)
        self.min_val = val_range[0]
        self.max_val = val_range[1]
        validator = QDoubleValidator()
        if val_range is not None:
            validator.setTop(val_range[1])
            validator.setBottom(val_range[0])
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

        self.setValidator(validator)
        self.lineEdit().setValidator(validator)
        self._value = 0

    def keyPressEvent(self, e: QKeyEvent) -> None:
        key = e.key()
        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            self.check_change()
        super().keyPressEvent(e)

    def focusInEvent(self, e: QFocusEvent) -> None:
        super().focusInEvent(e)
        self.text_changed_by_user = False

    def on_text_changed(self):
        if self.hasFocus():
            self.text_changed_by_user = True

    def on_current_index_changed(self):
        if self.hasFocus():
            self.check_change()

    def value(self) -> float:
        txt = self.currentText()
        try:
            val = float(txt)
            self._value = val
            return val
        except:
            LOGGER.warning(f'SizeComboBox invalid input: {txt}, return {self._value}')
            return self._value

    def setValue(self, value: float):
        value = min(self.max_val, max(self.min_val, value))
        self.setCurrentText(str(round(value, 2)))

    def check_change(self):
        if self.text_changed_by_user:
            self.text_changed_by_user = False
            self.apply_change.emit(self.value())


class FontSizeBox(QFrame):
    apply_fontsize = Signal(float)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.fcombobox = SizeComboBox([0, 10000], self)
        self.fcombobox.addItems([
            "5", "5.5", "6.5", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", '22', "26", "28", 
            "36", "48", "56", "72"
        ])
        self.fcombobox.apply_change.connect(self.on_fbox_apply_change)
        validator = QDoubleValidator()
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

    def getFontSize(self) -> float:
        return self.fcombobox.value()

    def onUpBtnClicked(self):
        size = self.getFontSize()
        newsize = int(round(size * 1.25))
        if newsize == size:
            newsize += 1
        newsize = min(1000, newsize)
        if newsize != size:
            self.apply_fontsize.emit(newsize)
            self.fcombobox.setCurrentText(str(newsize))
        
    def onDownBtnClicked(self):
        size = self.getFontSize()
        newsize = int(round(size * 0.75))
        if newsize == size:
            newsize -= 1
        newsize = max(1, newsize)
        if newsize != size:
            self.apply_fontsize.emit(newsize)
            self.fcombobox.text_changed_by_user = False
            self.fcombobox.setCurrentText(str(newsize))

    def on_fbox_apply_change(self, value: float):
        self.apply_fontsize.emit(value)

class SizeControlLabel(QLabel):
    
    btn_released = Signal()
    size_ctrl_changed = Signal(int)
    
    def __init__(self, parent=None, direction=0, text=''):
        super().__init__(parent)
        if text:
            self.setText(text)
        if direction == 0:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.cur_pos = 0
        self.direction = direction
        self.mouse_pressed = False

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = True
            if C.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            self.cur_pos = g_pos.x() if self.direction == 0 else g_pos.y()
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.mouse_pressed = False
            self.btn_released.emit()
        return super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.mouse_pressed:
            if C.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            if self.direction == 0:
                new_pos = g_pos.x()
                self.size_ctrl_changed.emit(new_pos - self.cur_pos)
            else:
                new_pos = g_pos.y()
                self.size_ctrl_changed.emit(self.cur_pos - new_pos)
            self.cur_pos = new_pos
        return super().mouseMoveEvent(e)

class FontFormatPanel(Widget):
    
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    active_format: FontFormat = None
    global_format: FontFormat = None
    restoring_textblk: bool = False
    
    global_format_changed = Signal()

    def __init__(self, app: QApplication, canvas: Canvas, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app
        self.canvas = canvas

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
        self.fontsizebox.fcombobox.setToolTip("Change font size")
        self.fontsizebox.apply_fontsize.connect(self.onApplyFontsize)
        self.fontsizebox.fcombobox.editTextChanged.connect(self.onSizeEditorChanged)

        self.lineSpacingLabel = SizeControlLabel(self, direction=1)
        self.lineSpacingLabel.setObjectName("lineSpacingLabel")
        self.lineSpacingLabel.size_ctrl_changed.connect(self.onLineSpacingCtrlChanged)
        self.lineSpacingLabel.btn_released.connect(self.onLineSpacingCtrlReleased)

        self.lineSpacingBox = SizeComboBox([0, 10], self)
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.apply_change.connect(self.update_line_spacing)
        self.lineSpacingBox.editTextChanged.connect(self.onLineSpacingEditorChanged)
        
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

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        font = self.fontStrokeLabel.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT * 0.95)
        self.fontStrokeLabel.setFont(font)
        self.fontStrokeLabel.size_ctrl_changed.connect(self.onStrokeCtrlChanged)
        self.fontStrokeLabel.btn_released.connect(self.onStrokeCtrlReleased)
        
        self.strokeColorPicker = ColorPicker(self)
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onStrokeColorChanged)
        self.strokeColorPicker.setObjectName("StrokeColorPicker")

        self.strokeWidthBox = SizeComboBox([0, 10], self)
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.apply_change.connect(self.update_stroke_width)
        self.strokeWidthBox.editTextChanged.connect(self.onStrokeWidthEditorChanged)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(WIDGET_SPACING_CLOSE)

        self.letterSpacingLabel = SizeControlLabel(self, direction=0)
        self.letterSpacingLabel.setObjectName("letterSpacingLabel")
        self.letterSpacingLabel.size_ctrl_changed.connect(self.onLetterSpacingCtrlChanged)
        self.letterSpacingLabel.btn_released.connect(self.onLetterSpacingCtrlReleased)

        self.letterSpacingBox = SizeComboBox([0, 10], self)
        self.letterSpacingBox.addItems(["0.0"])
        self.letterSpacingBox.setToolTip(self.tr("Change letter spacing"))
        self.letterSpacingBox.setMinimumWidth(self.letterSpacingBox.height() * 2.5)
        self.letterSpacingBox.apply_change.connect(self.update_letter_spacing)
        self.letterSpacingBox.editTextChanged.connect(self.onLetterSpacingEditorChanged)

        lettersp_hlayout = QHBoxLayout()
        lettersp_hlayout.addWidget(self.letterSpacingLabel)
        lettersp_hlayout.addWidget(self.letterSpacingBox)
        lettersp_hlayout.setSpacing(WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.fontfmtLabel = QLabel(self)
        font = self.fontfmtLabel.font()
        font.setPointSizeF(CONFIG_FONTSIZE_CONTENT * 0.7)
        self.fontfmtLabel.setText(self.global_fontfmt_str)
        self.fontfmtLabel.setFont(font)

        hl0 = QHBoxLayout()
        hl0.addStretch(1)
        hl0.addWidget(self.fontfmtLabel)
        hl0.addStretch(1)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.addWidget(self.lineSpacingLabel)
        hl1.addWidget(self.lineSpacingBox)
        hl1.setSpacing(10)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.setSpacing(10)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.setContentsMargins(5, 5, 5, 5)
        hl3.setSpacing(20)

        self.vlayout.addLayout(hl0)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.setContentsMargins(10, 10, 10, 10)
        self.setFixedWidth(520)

        self.focusOnColorDialog = False
        self.active_format = self.global_format

    def restoreTextBlkItem(self):
        if self.active_format == self.global_format:
            self.global_format_changed.emit()

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

    def onApplyFontsize(self, font_size: float):
        self.active_format.size = font_size
        self.restoreTextBlkItem()
        set_textblk_fontsize(self.textblk_item, self.active_format.size)

    def onSizeEditorChanged(self):
        if self.fontsizebox.fcombobox.hasFocus() and self.active_format == self.global_format:
            self.global_format.size = self.fontsizebox.getFontSize()

    def onStrokeWidthEditorChanged(self):
        if self.strokeWidthBox.hasFocus() and self.active_format == self.global_format:
            self.global_format.stroke_width = self.strokeWidthBox.value()

    def onLineSpacingEditorChanged(self):
        if self.lineSpacingBox.hasFocus() and self.active_format == self.global_format:
            self.global_format.line_spacing = self.lineSpacingBox.value()

    def onLetterSpacingEditorChanged(self):
        if self.letterSpacingBox.hasFocus() and self.active_format == self.global_format:
            self.global_format.letter_spacing = self.letterSpacingBox.value()

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

    def onStrokeCtrlChanged(self, delta: int):
        self.strokeWidthBox.setValue(self.strokeWidthBox.value() + delta * 0.01)

    def onStrokeCtrlReleased(self):
        self.update_stroke_width(self.strokeWidthBox.value())

    def onLetterSpacingCtrlChanged(self, delta: int):
        self.letterSpacingBox.setValue(self.letterSpacingBox.value() + delta * 0.01)

    def onLetterSpacingCtrlReleased(self):
        self.update_letter_spacing(self.letterSpacingBox.value())

    def onLineSpacingCtrlChanged(self, delta: int):
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * 0.01)

    def onLineSpacingCtrlReleased(self):
        self.update_line_spacing(self.lineSpacingBox.value())

    def update_stroke_width(self, value: float):
        self.active_format.stroke_width = value
        self.restoreTextBlkItem()
        set_textblk_strokewidth(self.textblk_item, self.active_format.stroke_width)

    def update_letter_spacing(self, value: float):
        self.active_format.letter_spacing = value
        self.restoreTextBlkItem()
        set_textblk_letterspacing(self.textblk_item, value)

    def update_line_spacing(self, value: float):
        self.active_format.line_spacing = value
        self.restoreTextBlkItem()
        set_textblk_linespacing(self.textblk_item, self.active_format.line_spacing)
            
    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.fontsizebox.fcombobox.setCurrentText(str(int(font_format.size)))
        self.familybox.setCurrentText(font_format.family)
        self.colorPicker.setPickerColor(font_format.frgb)
        self.strokeColorPicker.setPickerColor(font_format.srgb)
        self.strokeWidthBox.setValue(font_format.stroke_width)
        self.lineSpacingBox.setValue(font_format.line_spacing)
        self.letterSpacingBox.setValue(font_format.letter_spacing)
        self.verticalChecker.setChecked(font_format.vertical)
        self.formatBtnGroup.boldBtn.setChecked(font_format.bold)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.alignBtnGroup.setAlignment(font_format.alignment)

    def set_textblk_item(self, textblk_item: TextBlkItem = None):
        if textblk_item is None:
            focus_w = self.app.focusWidget()
            focus_p = None if focus_w is None else focus_w.parentWidget()
            focus_on_fmtoptions = False
            if self.focusOnColorDialog:
                focus_on_fmtoptions = True
            elif focus_p:
                if focus_p == self or focus_p.parentWidget() == self:
                    focus_on_fmtoptions = True
            if not focus_on_fmtoptions:
                self.textblk_item = None
                self.set_active_format(self.global_format)
                self.fontfmtLabel.setText(self.global_fontfmt_str)
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                self.textblk_item = textblk_item
                self.set_active_format(blk_fmt)
                self.fontfmtLabel.setText(f'TextBlock #{textblk_item.idx}')