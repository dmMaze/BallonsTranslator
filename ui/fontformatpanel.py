import copy
import sys
from typing import List

from qtpy.QtWidgets import QComboBox, QMenu, QMessageBox, QStackedLayout, QGraphicsDropShadowEffect, QLineEdit, QScrollArea, QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QApplication, QPushButton, QCheckBox, QLabel
from qtpy.QtCore import Signal, Qt, QRectF
from qtpy.QtGui import QDoubleValidator, QFocusEvent, QMouseEvent, QTextCursor, QFontMetrics, QIcon, QColor, QPixmap, QPainter, QContextMenuEvent, QKeyEvent


from utils.fontformat import FontFormat
from utils import shared
from utils.config import pcfg, save_text_styles, text_styles
from utils import config as C
from .stylewidgets import Widget, ColorPicker, ClickableLabel, CheckableLabel, TextChecker, FlowLayout, ScrollBar
from .textitem import TextBlkItem
from .text_graphical_effect import TextEffectPanel
from . import funcmaps as FM


class LineEdit(QLineEdit):

    return_pressed_wochange = Signal()

    def __init__(self, content: str = None, parent = None):
        super().__init__(content, parent)
        self.textChanged.connect(self.on_text_changed)
        self._text_changed = False
        self.editingFinished.connect(self.on_editing_finished)
        self.returnPressed.connect(self.on_return_pressed)

    def on_text_changed(self):
        self._text_changed = True

    def on_editing_finished(self):
        self._text_changed = False

    def focusOutEvent(self, e: QFocusEvent) -> None:
        self._text_changed = False
        return super().focusOutEvent(e)

    def on_return_pressed(self):
        if not self._text_changed:
            self.return_pressed_wochange.emit()


class SizeComboBox(QComboBox):
    
    param_changed = Signal(str, float)
    def __init__(self, val_range: List = None, param_name: str = '', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        lineedit = LineEdit(parent=self)
        lineedit.return_pressed_wochange.connect(self.apply_size)
        self.setLineEdit(lineedit)
        self.text_changed_by_user = False
        self.param_name = param_name
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

    def apply_size(self):
        self.param_changed.emit(self.param_name, self.value())

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
            self.check_change()

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
            return self._value

    def setValue(self, value: float):
        value = min(self.max_val, max(self.min_val, value))
        self.setCurrentText(str(round(value, 2)))

    def check_change(self):
        if self.text_changed_by_user:
            self.text_changed_by_user = False
            self.param_changed.emit(self.param_name, self.value())


class IncrementalBtn(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(13, 13)
        
class QFontChecker(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sys.platform == 'darwin':
            self.setStyleSheet("min-width: 45px")

class AlignmentChecker(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sys.platform == 'darwin':
            self.setStyleSheet("min-width: 15px")

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
        hlayout.setContentsMargins(0, 0, 0, 0)

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
            if shared.FLAG_QT6:
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
            if shared.FLAG_QT6:
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
        self.lineedit = lineedit = LineEdit(parent=self)
        lineedit.return_pressed_wochange.connect(self.apply_fontfamily)
        self.setLineEdit(lineedit)
        self.emit_if_focused = emit_if_focused
        self._current_font = self.currentFont().family()
        
    def apply_fontfamily(self):
        ffamily = self.currentFont().family()
        if ffamily in shared.FONT_FAMILIES:
            self.param_changed.emit('family', ffamily)
            self._current_font = ffamily

    def on_fontfamily_changed(self):
        # if not self.hasFocus():
        # #     self._current_font = self.currentFont().family()
        # #     self.lineedit._text_changed = False
        #     if self.emit_if_focused and not self.hasFocus():
        #         return

        # ffamily = self.currentFont().family()
        # if self._current_font != ffamily:
        self.apply_fontfamily()
            

CHEVRON_SIZE = 20
def chevron_down():
    return QIcon(r'icons/chevron-down.svg').pixmap(CHEVRON_SIZE, CHEVRON_SIZE, mode=QIcon.Mode.Normal)

def chevron_right():
    return QIcon(r'icons/chevron-right.svg').pixmap(CHEVRON_SIZE, CHEVRON_SIZE, mode=QIcon.Mode.Normal)


class StyleLabel(QLineEdit):

    edit_finished = Signal()

    def __init__(self, style_name: str = None, parent = None):
        super().__init__(parent=parent)

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0); border: none")
        self.setTextMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)

        self.editingFinished.connect(self.edit_finished)
        self.setEnabled(False)
        
        if style_name is not None:
            self.setText(style_name)

        self.resizeToContent()
        self.edit_finished.connect(self.resizeToContent)

    def focusOutEvent(self, e) -> None:
        super().focusOutEvent(e)
        self.edit_finished.emit()

    def resizeToContent(self):
        fm = QFontMetrics(self.font())
        text = self.text()
        w = fm.boundingRect(text).width() + 5

        self.setFixedWidth(max(w, 32))


class ArrowLeftButton(QPushButton):
    pass


class ArrowRightButton(QPushButton):
    pass

class DeleteStyleButton(QPushButton):
    pass


class TextStyleLabel(Widget):

    style_name_edited = Signal()
    delete_btn_clicked = Signal()
    stylelabel_activated = Signal(bool)
    apply_fontfmt = Signal(FontFormat)

    def __init__(self, style_name: str = '', parent: Widget = None, fontfmt: FontFormat = None, active_stylename_edited: Signal = None):
        super().__init__(parent=parent)
        self._double_clicked = False
        self.active = False
        if fontfmt is None:
            if C.active_format is None:
                self.fontfmt = FontFormat()
            else:
                self.fontfmt = C.active_format.copy()
            self.fontfmt._style_name = style_name
        else:
            self.fontfmt = fontfmt
            style_name = fontfmt._style_name

        # following subwidgets must have parents, otherwise they kinda of pop up when creating it
        self.active_stylename_edited = active_stylename_edited
        self.stylelabel = StyleLabel(style_name, parent=self)
        self.stylelabel.edit_finished.connect(self.on_style_name_edited)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)

        self.setToolTip(self.tr('Click to set as Global format. Double click to edit name.'))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        BTN_SIZE = 14
        self.colorw = colorw = QLabel(parent=self)
        self.colorw.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colorw.setStyleSheet("border-radius: 7px; border: none; background-color: rgba(0, 0, 0, 0);")
        d = int(BTN_SIZE * 2)
        self.colorw.setFixedSize(d, d)
        
        self.apply_btn = ArrowLeftButton(parent=self)
        self.apply_btn.setFixedSize(d, BTN_SIZE)
        self.apply_btn.setToolTip(self.tr('Apply Text Style'))
        self.apply_btn.clicked.connect(self.on_applybtn_clicked)
        self.update_btn = ArrowRightButton(parent=self)
        self.update_btn.setFixedSize(d, BTN_SIZE)
        self.update_btn.clicked.connect(self.on_updatebtn_clicked)
        self.update_btn.setToolTip(self.tr('Update from active style'))
        applyw = Widget(parent=self)
        applyw.setStyleSheet("border-radius: 7px; border: none")
        applylayout = QVBoxLayout(applyw)
        applylayout.setSpacing(0)
        applylayout.setContentsMargins(0, 0, 0, 0)
        applylayout.addWidget(self.apply_btn)
        applylayout.addWidget(self.update_btn)

        self.leftstack = QStackedLayout()
        self.leftstack.setContentsMargins(0, 0, 0, 0)
        self.leftstack.addWidget(colorw)
        self.leftstack.addWidget(applyw)

        self.delete_btn = DeleteStyleButton(parent=self)
        dsize = BTN_SIZE // 3 * 2
        self.delete_btn.setFixedSize(dsize, dsize)
        self.delete_btn.setToolTip(self.tr("Delete Style"))
        self.delete_btn.clicked.connect(self.on_delete_btn_clicked)
        self.delete_btn.setStyleSheet("border: none")
        
        hlayout = QHBoxLayout(self)
        hlayout.setContentsMargins(0, 0, 3, 0)
        hlayout.setSpacing(0)
        hlayout.addLayout(self.leftstack)
        hlayout.addWidget(self.stylelabel)
        hlayout.addWidget(self.delete_btn)

        self.updatePreview()

    def on_delete_btn_clicked(self, *args, **kwargs):
        self.delete_btn_clicked.emit()

    def on_updatebtn_clicked(self, *args, **kwargs):
        self.update_style()

    def on_applybtn_clicked(self, *args, **kwargs):
        self.apply_fontfmt.emit(self.fontfmt)

    def update_style(self, fontfmt: FontFormat = None):
        if fontfmt is None:
            fontfmt = C.active_format
        if fontfmt is None:
            return
        updated_keys = self.fontfmt.merge(fontfmt)
        if len(updated_keys) > 0:
            save_text_styles()
        
        preview_keys = {'family', 'frgb', 'srgb', 'stroke_width'}
        for k in updated_keys:
            if k in preview_keys:
                self.updatePreview()
                break
            
    def setActive(self, active: bool):
        self.active = active
        if active:
            self.setStyleSheet("border: 2px solid rgb(30, 147, 229)")
        else:
            self.setStyleSheet("")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self._double_clicked:
                self._double_clicked = False
            else:
                active = not self.active
                self.setActive(active)
                self.stylelabel_activated.emit(active)
        return super().mouseReleaseEvent(event)

    def updatePreview(self):
        font = self.stylelabel.font()
        font.setFamily(self.fontfmt.family)
        self.stylelabel.setFont(font)

        d = int(self.colorw.width() * 0.66)
        radius = d / 2
        pixmap = QPixmap(d, d)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)

        draw_rect, draw_radius = QRectF(0, 0, d, d), radius
        if self.fontfmt.stroke_width > 0:
            r, g, b = self.fontfmt.srgb
            color = QColor(r, g, b, 255)
            painter.setBrush(color)
            painter.drawRoundedRect(draw_rect, draw_radius, draw_radius)
            draw_radius = draw_radius * 0.66
            offset = d / 2 - draw_radius
            draw_rect = QRectF(offset, offset, draw_radius*2, draw_radius*2)

        r, g, b = self.fontfmt.frgb
        color = QColor(r, g, b, 255)
        painter.setBrush(color)
        painter.drawRoundedRect(draw_rect, draw_radius, draw_radius)
        painter.end()
        self.colorw.setPixmap(pixmap)

        self.stylelabel.resizeToContent()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        self._double_clicked = True
        self.startEdit()
        return super().mouseDoubleClickEvent(event)
    
    def startEdit(self, select_all=False):
        self.stylelabel.setEnabled(True)
        self.stylelabel.setFocus()
        self.setCursor(Qt.CursorShape.IBeamCursor)
        if select_all:
            self.stylelabel.selectAll()

    def setHoverEffect(self, hover: bool):
        try:
            if hover:
                se = QGraphicsDropShadowEffect()
                se.setBlurRadius(6)
                se.setOffset(0, 0)
                se.setColor(QColor(30, 147, 229))
                self.setGraphicsEffect(se)
            else:
                self.setGraphicsEffect(None)
        except RuntimeError:
            pass

    def enterEvent(self, event) -> None:
        self.setHoverEffect(True)
        self.leftstack.setCurrentIndex(1)
        self.delete_btn.setStyleSheet("image: url(icons/titlebar_close.svg); border: none")
        return super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        self.setHoverEffect(False)
        self.leftstack.setCurrentIndex(0)
        self.delete_btn.setStyleSheet("image: \"none\"; border: none")
        return super().leaveEvent(event)
    
    def on_style_name_edited(self):
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stylelabel.setEnabled(False)
        new_name = self.stylelabel.text()
        if self.fontfmt._style_name != new_name:
            self.fontfmt._style_name = new_name
            save_text_styles()

        if self.active and self.active_stylename_edited is not None:
            self.active_stylename_edited.emit()

        self._double_clicked = False



class TextAreaStyleButton(QPushButton):
    pass



class TextStyleArea(QScrollArea):

    entered = False
    active_text_style_label_changed = Signal()
    apply_fontfmt = Signal(FontFormat)
    active_stylename_edited = Signal()
    export_style = Signal()
    import_style = Signal()

    def __init__(self, parent: Widget = None):
        super().__init__(parent)

        self.active_text_style_label: TextStyleLabel = None
        self.scrollContent = Widget()
        self.scrollContent.setObjectName("TextStyleAreaContent")
        self.setWidget(self.scrollContent)
        self.flayout = FlowLayout(self.scrollContent)
        # margin = 7
        # self.flayout.setVerticalSpacing(7)
        # self.flayout.setHorizontalSpacing(7)
        # self.flayout.setContentsMargins(margin, margin, margin, margin)
        self.setWidgetResizable(True)
        self.default_preset_name = self.tr('Style')
        
        self.new_btn = TextAreaStyleButton()
        self.new_btn.setObjectName("NewTextStyleButton")
        self.new_btn.setToolTip(self.tr("New Text Style"))
        self.new_btn.clicked.connect(self.on_newbtn_clicked)

        self.clear_btn = TextAreaStyleButton()
        self.clear_btn.setObjectName("ClearTextStyleButton")
        self.clear_btn.setToolTip(self.tr("Remove All"))
        self.clear_btn.clicked.connect(self.on_clearbtn_clicked)

        self.flayout.addWidget(self.new_btn)
        self.flayout.addWidget(self.clear_btn)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        ScrollBar(Qt.Orientation.Vertical, self)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ScrollBar(Qt.Orientation.Horizontal, self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def on_newbtn_clicked(self, clicked = None):
        textstylelabel = self.new_textstyle_label()
        textstylelabel.startEdit(select_all=True)
        self.resizeToContent()

    def on_clearbtn_clicked(self, clicked = None):
        msg = QMessageBox()
        msg.setText(self.tr('Remove all styles?'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.clearStyles()

    def count(self):
        return self.flayout.count() - 2
    
    def isEmpty(self):
        return self.count() < 1

    def new_textstyle_label(self, preset_name: str = None):
        if preset_name is None:
            sno = str(self.count() + 1)
            if len(sno) < 2:
                preset_name = self.default_preset_name + ' ' + sno
            else:
                preset_name = self.default_preset_name + sno
        textstylelabel = TextStyleLabel(preset_name, active_stylename_edited=self.active_stylename_edited)
        textstylelabel.stylelabel_activated.connect(self.on_stylelabel_activated)
        textstylelabel.delete_btn_clicked.connect(self.on_deletebtn_clicked)
        textstylelabel.apply_fontfmt.connect(self.apply_fontfmt)
        self.flayout.insertWidget(self.count(), textstylelabel)
        text_styles.append(textstylelabel.fontfmt)
        save_text_styles()
        return textstylelabel

    def resizeToContent(self):
        TEXTSTYLEAREA_MAXH = 200
        self.setFixedHeight(min(TEXTSTYLEAREA_MAXH, self.flayout.heightForWidth(self.width())))

    def resizeEvent(self, e):
        self.resizeToContent()
        return super().resizeEvent(e)
    
    def showNewBtn(self):
        if not self.new_btn.isVisible():
            self.new_btn.show()
            self.clear_btn.show()
            self.resizeToContent()

    def hideNewBtn(self):
        if self.new_btn.isVisible():
            self.new_btn.hide()
            self.clear_btn.hide()
            self.resizeToContent()

    def updateNewBtnVisibility(self):
        if self.isEmpty() or self.entered:
            self.showNewBtn()
        else:
            self.hideNewBtn()

    def enterEvent(self, event) -> None:
        self.entered = True
        self.showNewBtn()
        return super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        self.entered = False
        if not self.isEmpty():
            self.hideNewBtn()
        return super().leaveEvent(event)

    def _clear_styles(self):
        self.active_text_style_label = None
        for _ in range(self.count()):
            w: TextStyleLabel = self.flayout.takeAt(0)
            if w is not None:
                if w.active:
                    w.setActive(False)
                    self.active_text_style_label_changed.emit()
                w.deleteLater()

    def _add_style_label(self, fontfmt: FontFormat):
        textstylelabel = TextStyleLabel(fontfmt=fontfmt, active_stylename_edited=self.active_stylename_edited)
        textstylelabel.delete_btn_clicked.connect(self.on_deletebtn_clicked)
        textstylelabel.stylelabel_activated.connect(self.on_stylelabel_activated)
        textstylelabel.apply_fontfmt.connect(self.apply_fontfmt)
        self.flayout.insertWidget(self.count(), textstylelabel)

    def on_deletebtn_clicked(self):
        w: TextStyleLabel = self.sender()
        self.removeStyleLabel(w)

    def on_stylelabel_activated(self, active: bool):
        if self.active_text_style_label is not None:
            self.active_text_style_label.setActive(False)
            self.active_text_style_label = None
        if active:
            self.active_text_style_label = self.sender()
        self.active_text_style_label_changed.emit()

    def clearStyles(self):
        if self.isEmpty():
            return
        self._clear_styles()
        self.updateNewBtnVisibility()
        text_styles.clear()
        save_text_styles()

    def removeStyleLabel(self, w: TextStyleLabel):
        for i, item in enumerate(self.flayout._items):
            if item.widget() is w:
                if w is self.active_text_style_label:
                    w.setActive(False)
                    self.active_text_style_label = None
                    self.active_text_style_label_changed.emit()
                self.flayout.takeAt(i)
                self.flayout.update()
                self.updateNewBtnVisibility()
                text_styles.pop(i)
                save_text_styles()
                w.deleteLater()
                self.resizeToContent()
                break
        
    def initStyles(self, styles: List[FontFormat]):
        assert self.isEmpty()
        for style in styles:
            self._add_style_label(style)
        if not self.isEmpty():
            self.new_btn.hide()
            self.clear_btn.hide()
            self.resizeToContent()

    def setStyles(self, styles: List[FontFormat], save_styles = False):
        self._clear_styles()
        for style in styles:
            self._add_style_label(style)
        
        self.updateNewBtnVisibility()
        self.resizeToContent()
        if save_styles:
            save_text_styles()

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()

        new_act = menu.addAction(self.tr('New Text Style'))
        removeall_act = menu.addAction(self.tr('Remove all'))
        menu.addSeparator()
        import_act = menu.addAction(self.tr('Import Text Styles'))
        export_act = menu.addAction(self.tr('Export Text Styles'))
        
        rst = menu.exec_(e.globalPos())

        if rst == new_act:
            self.on_newbtn_clicked()
        elif rst == removeall_act:
            self.on_clearbtn_clicked()
        elif rst == import_act:
            self.import_style.emit()
        elif rst == export_act:
            self.export_style.emit()

        return super().contextMenuEvent(e)


class ExpandLabel(Widget):

    clicked = Signal()

    def __init__(self, text=None, parent=None, expanded=False, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.textlabel = QLabel(self)
        self.textlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        font = self.textlabel.font()
        if shared.ON_MACOS:
            font.setPointSize(13)
        else:
            font.setPointSizeF(10)
        self.textlabel.setFont(font)
        self.arrowlabel = QLabel(self)
        self.arrowlabel.setFixedSize(CHEVRON_SIZE, CHEVRON_SIZE)
        self.arrowlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        if text is not None:
            self.textlabel.setText(text)
        layout = QHBoxLayout(self)
        layout.addWidget(self.arrowlabel)
        layout.addWidget(self.textlabel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addStretch(-1)
    
        self.expanded = False
        self.setExpand(expanded)
        self.setFixedHeight(26)

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


class TextStylePanel(Widget):

    def __init__(self, text=None, parent=None, expanded=True, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        
        self.title_label = ExpandLabel(text, self, expanded=expanded)
        self.style_area = TextStyleArea(self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.style_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if not expanded:
            self.style_area.hide()
        
        self.title_label.clicked.connect(self.on_title_label_clicked)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def expand(self):
        if not self.title_label.expanded:
            self.title_label.setExpand(True)
        if self.style_area.isHidden():
            self.style_area.show()

    def on_title_label_clicked(self):
        if self.title_label.expanded:
            self.style_area.show()
        else:
            self.style_area.hide()

    def setTitle(self, text: str):
        self.title_label.textlabel.setText(text)

    def elidedText(self, text: str):
        fm = QFontMetrics(self.title_label.font())
        return fm.elidedText(text, Qt.TextElideMode.ElideRight, self.style_area.width() - 40)

    def title(self) -> str:
        return self.title_label.textlabel.text()


class FontFormatPanel(Widget):
    
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    global_format: FontFormat = None
    restoring_textblk: bool = False

    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.familybox = FontFamilyComboBox(emit_if_focused=True, parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_param_changed)
        self.familybox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

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
        font.setPointSizeF(shared.CONFIG_FONTSIZE_CONTENT * 0.95)
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
        stroke_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

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
        lettersp_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.textstyle_panel = TextStylePanel(self.global_fontfmt_str, parent=self, expanded=pcfg.expand_tstyle_panel)
        self.textstyle_panel.style_area.active_text_style_label_changed.connect(self.on_active_textstyle_label_changed)
        self.textstyle_panel.style_area.active_stylename_edited.connect(self.on_active_stylename_edited)

        self.effectBtn = ClickableLabel(self.tr("Effect"), self)
        self.effectBtn.clicked.connect(self.on_effectbtn_clicked)
        self.effect_panel = TextEffectPanel(update_text_style_label=self.update_text_style_label)
        self.effect_panel.hide()

        self.foldTextBtn = CheckableLabel(self.tr("Unfold"), self.tr("Fold"), False)
        self.sourceBtn = TextChecker(self.tr("Source"))
        self.transBtn = TextChecker(self.tr("Translation"))

        FONTFORMAT_SPACING = 6

        vl0 = QVBoxLayout()
        vl0.addWidget(self.textstyle_panel)
        vl0.setSpacing(0)
        vl0.setContentsMargins(0, 0, 0, 0)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.addWidget(self.lineSpacingLabel)
        hl1.addWidget(self.lineSpacingBox)
        hl1.setSpacing(4)
        hl1.setContentsMargins(0, 12, 0, 0)
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
        hl3.setContentsMargins(3, 0, 3, 0)
        hl3.setSpacing(13)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setContentsMargins(0, 12, 0, 0)
        hl4.setSpacing(0)

        self.vlayout.addLayout(vl0)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.setContentsMargins(7, 0, 7, 0)
        self.vlayout.setSpacing(0)

        self.focusOnColorDialog = False
        C.active_format = self.global_format

    def global_mode(self):
        return id(C.active_format) == id(self.global_format)
    
    def active_text_style_label(self):
        return self.textstyle_panel.style_area.active_text_style_label

    def on_param_changed(self, param_name: str, value):
        func = FM.handle_ffmt_change.get(param_name)
        func_kwargs = {}
        if param_name == 'size':
            func_kwargs['clip_size'] = True
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True, **func_kwargs)
            self.update_text_style_label()
        else:
            func(param_name, value, C.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True, **func_kwargs)

    def update_text_style_label(self):
        if self.global_mode():
            active_text_style_label = self.active_text_style_label()
            if active_text_style_label is not None:
                active_text_style_label.update_style(self.global_format)

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
        if self.lineSpacingBox.hasFocus() and C.active_format == self.global_format:
            self.global_format.line_spacing = self.lineSpacingBox.value()

    def onStrokeCtrlChanged(self, delta: int):
        self.strokeWidthBox.setValue(self.strokeWidthBox.value() + delta * 0.01)

    def onLetterSpacingCtrlChanged(self, delta: int):
        self.letterSpacingBox.setValue(self.letterSpacingBox.value() + delta * 0.01)

    def onLineSpacingCtrlChanged(self, delta: int):
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * 0.01)
            
    def set_active_format(self, font_format: FontFormat):
        C.active_format = font_format
        self.familybox.blockSignals(True)
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
        self.familybox.blockSignals(False)

    def set_globalfmt_title(self):
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is None:
            self.textstyle_panel.setTitle(self.global_fontfmt_str)
        else:
            title = self.global_fontfmt_str + ' - ' + active_text_style_label.fontfmt._style_name
            valid_title = self.textstyle_panel.elidedText(title)
            self.textstyle_panel.setTitle(valid_title)

    def on_active_textstyle_label_changed(self):
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is not None:
            updated_keys = self.global_format.merge(active_text_style_label.fontfmt)
            if self.global_mode() and len(updated_keys) > 0:
                self.set_active_format(self.global_format)
            self.set_globalfmt_title()
        else:
            if self.global_mode():
                self.set_globalfmt_title()

    def on_active_stylename_edited(self):
        if self.global_mode():
            self.set_globalfmt_title()

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
                self.set_globalfmt_title()
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                self.textblk_item = textblk_item
                self.set_active_format(blk_fmt)
                self.textstyle_panel.setTitle(f'TextBlock #{textblk_item.idx}')

    def on_effectbtn_clicked(self):
        self.effect_panel.active_fontfmt = C.active_format
        self.effect_panel.fontfmt = copy.deepcopy(C.active_format)
        self.effect_panel.updatePanels()
        self.effect_panel.show()