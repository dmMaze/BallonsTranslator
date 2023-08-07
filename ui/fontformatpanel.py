import copy

from qtpy.QtWidgets import QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QApplication, QPushButton, QCheckBox, QLabel
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QMouseEvent, QTextCursor

from .stylewidgets import Widget, ColorPicker, ClickableLabel, CheckableLabel, TextChecker
from .misc import FontFormat
from .textitem import TextBlkItem
from .text_graphical_effect import TextEffectPanel
from .combobox import SizeComboBox
from . import constants as C
from . import funcmaps as FM


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
    param_changed = Signal(str, int)
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
            self.param_changed.emit('alignment', 0)
        elif btn == self.alignRightChecker:
            self.alignRightChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignLeftChecker.setChecked(False)
            self.param_changed.emit('alignment', 2)
        else:
            self.alignCenterChecker.setChecked(True)
            self.alignLeftChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 1)
    
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
    param_changed = Signal(str, bool)
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
        self.param_changed.emit('bold', self.boldBtn.isChecked())

    def setItalic(self):
        self.param_changed.emit('italic', self.italicBtn.isChecked())

    def setUnderline(self):
        self.param_changed.emit('underline', self.underlineBtn.isChecked())
    

class FontSizeBox(QFrame):
    param_changed = Signal(str, float)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.fcombobox = SizeComboBox([1, 1000], 'size', self)
        self.fcombobox.addItems([
            "5", "5.5", "6.5", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", '22', "26", "28", 
            "36", "48", "56", "72"
        ])
        self.fcombobox.param_changed.connect(self.param_changed)

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
            self.param_changed.emit('size', newsize)
            self.fcombobox.setCurrentText(str(newsize))
        
    def onDownBtnClicked(self):
        size = self.getFontSize()
        newsize = int(round(size * 0.75))
        if newsize == size:
            newsize -= 1
        newsize = max(1, newsize)
        if newsize != size:
            self.param_changed.emit('size', newsize)
            self.fcombobox.text_changed_by_user = False
            self.fcombobox.setCurrentText(str(newsize))



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
    

class FontFamilyComboBox(QFontComboBox):
    param_changed = Signal(str, object)
    def __init__(self, emit_if_focused=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentFontChanged.connect(self.on_fontfamily_changed)
        self.emit_if_focused = emit_if_focused

    def on_fontfamily_changed(self):
        if self.emit_if_focused and not self.hasFocus():
            return
        self.param_changed.emit('family', self.currentText())


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
        self.familybox = FontFamilyComboBox(emit_if_focused=True, parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_param_changed)

        self.fontsizebox = FontSizeBox(self)
        self.fontsizebox.setToolTip(self.tr("Font Size"))
        self.fontsizebox.setObjectName("FontSizeBox")
        self.fontsizebox.fcombobox.setToolTip(self.tr("Change font size"))
        self.fontsizebox.param_changed.connect(self.on_param_changed)

        self.lineSpacingLabel = SizeControlLabel(self, direction=1)
        self.lineSpacingLabel.setObjectName("lineSpacingLabel")
        self.lineSpacingLabel.size_ctrl_changed.connect(self.onLineSpacingCtrlChanged)
        self.lineSpacingLabel.btn_released.connect(lambda : self.on_param_changed('line_spacing', self.lineSpacingBox.value()))

        self.lineSpacingBox = SizeComboBox([0, 10], 'line_spacing', self)
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.param_changed.connect(self.on_param_changed)
        self.lineSpacingBox.editTextChanged.connect(self.onLineSpacingEditorChanged)
        
        self.colorPicker = ColorPicker(self)
        self.colorPicker.setObjectName("FontColorPicker")
        self.colorPicker.setToolTip(self.tr("Change font color"))
        self.colorPicker.changingColor.connect(self.changingColor)
        self.colorPicker.colorChanged.connect(self.onColorChanged)

        self.alignBtnGroup = AlignmentBtnGroup(self)
        self.alignBtnGroup.param_changed.connect(self.on_param_changed)

        self.formatBtnGroup = FormatGroupBtn(self)
        self.formatBtnGroup.param_changed.connect(self.on_param_changed)

        self.verticalChecker = QFontChecker(self)
        self.verticalChecker.setObjectName("FontVerticalChecker")
        self.verticalChecker.clicked.connect(lambda : self.on_param_changed('vertical', self.verticalChecker.isChecked()))

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        font = self.fontStrokeLabel.font()
        font.setPointSizeF(C.CONFIG_FONTSIZE_CONTENT * 0.95)
        self.fontStrokeLabel.setFont(font)
        self.fontStrokeLabel.size_ctrl_changed.connect(self.onStrokeCtrlChanged)
        self.fontStrokeLabel.btn_released.connect(lambda : self.on_param_changed('stroke_width', self.strokeWidthBox.value()))
        
        self.strokeColorPicker = ColorPicker(self)
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onStrokeColorChanged)
        self.strokeColorPicker.setObjectName("StrokeColorPicker")

        self.strokeWidthBox = SizeComboBox([0, 10], 'stroke_width', self)
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.param_changed.connect(self.on_param_changed)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(C.WIDGET_SPACING_CLOSE)

        self.letterSpacingLabel = SizeControlLabel(self, direction=0)
        self.letterSpacingLabel.setObjectName("letterSpacingLabel")
        self.letterSpacingLabel.size_ctrl_changed.connect(self.onLetterSpacingCtrlChanged)
        self.letterSpacingLabel.btn_released.connect(lambda : self.on_param_changed('letter_spacing', self.letterSpacingBox.value()))

        self.letterSpacingBox = SizeComboBox([0, 10], "letter_spacing", self)
        self.letterSpacingBox.addItems(["0.0"])
        self.letterSpacingBox.setToolTip(self.tr("Change letter spacing"))
        self.letterSpacingBox.setMinimumWidth(int(self.letterSpacingBox.height() * 2.5))
        self.letterSpacingBox.param_changed.connect(self.on_param_changed)

        lettersp_hlayout = QHBoxLayout()
        lettersp_hlayout.addWidget(self.letterSpacingLabel)
        lettersp_hlayout.addWidget(self.letterSpacingBox)
        lettersp_hlayout.setSpacing(C.WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.fontfmtLabel = ClickableLabel(self.global_fontfmt_str, self)
        font = self.fontfmtLabel.font()
        font.setPointSizeF(C.CONFIG_FONTSIZE_CONTENT * 0.75)
        self.fontfmtLabel.setFont(font)

        self.effectBtn = ClickableLabel(self.tr("Effect"), self)
        self.effectBtn.clicked.connect(self.on_effectbtn_clicked)
        self.effect_panel = TextEffectPanel()
        self.effect_panel.hide()

        self.foldTextBtn = CheckableLabel(self.tr("Unfold"), self.tr("Fold"), False)
        self.sourceBtn = TextChecker(self.tr("Source"))
        self.transBtn = TextChecker(self.tr("Translation"))

        FONTFORMAT_SPACING = 6

        hl0 = QHBoxLayout()
        hl0.addStretch(1)
        hl0.addWidget(self.fontfmtLabel)
        hl0.addStretch(1)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.addWidget(self.lineSpacingLabel)
        hl1.addWidget(self.lineSpacingBox)
        hl1.setSpacing(4)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.setSpacing(FONTFORMAT_SPACING)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.addWidget(self.effectBtn)
        hl3.setContentsMargins(3, 3, 3, 3)
        hl3.setSpacing(13)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setSpacing(0)

        self.vlayout.addLayout(hl0)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.setContentsMargins(7, 7, 7, 0)
        self.setFixedWidth(C.TEXTEDIT_FIXWIDTH)

        self.focusOnColorDialog = False
        self.active_format = self.global_format

    def global_mode(self):
        return id(self.active_format) == id(self.global_format)

    def on_param_changed(self, param_name: str, value):
        func = FM.handle_ffmt_change.get(param_name)
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True)
        else:
            func(param_name, value, self.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True)

    def changingColor(self):
        self.focusOnColorDialog = True

    def onColorChanged(self, is_valid=True):
        self.focusOnColorDialog = False
        if is_valid:
            frgb = self.colorPicker.rgb()
            self.on_param_changed('frgb', frgb)

    def onStrokeColorChanged(self, is_valid=True):
        self.focusOnColorDialog = False
        if is_valid:
            srgb = self.strokeColorPicker.rgb()
            self.on_param_changed('srgb', srgb)

    def onLineSpacingEditorChanged(self):
        if self.lineSpacingBox.hasFocus() and self.active_format == self.global_format:
            self.global_format.line_spacing = self.lineSpacingBox.value()

    def onStrokeCtrlChanged(self, delta: int):
        self.strokeWidthBox.setValue(self.strokeWidthBox.value() + delta * 0.01)

    def onLetterSpacingCtrlChanged(self, delta: int):
        self.letterSpacingBox.setValue(self.letterSpacingBox.value() + delta * 0.01)

    def onLineSpacingCtrlChanged(self, delta: int):
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * 0.01)
            
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

    def on_effectbtn_clicked(self):
        self.effect_panel.active_fontfmt = self.active_format
        self.effect_panel.fontfmt = copy.deepcopy(self.active_format)
        self.effect_panel.updatePanels()
        self.effect_panel.show()

    def on_load_preset(self, preset: FontFormat):
        self.global_format = preset
        if self.textblk_item is not None:
            if self.textblk_item.isEditing():
                self.textblk_item.endEdit()
            self.set_textblk_item(None)
                
        self.set_active_format(preset)
        self.fontfmtLabel.setText(self.global_fontfmt_str)
        