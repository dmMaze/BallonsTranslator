from typing import List

from qtpy.QtWidgets import QMenu, QMessageBox, QStackedLayout, QGraphicsDropShadowEffect, QLineEdit, QSizePolicy, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from qtpy.QtCore import Signal, Qt, QRectF
from qtpy.QtGui import QMouseEvent, QFontMetrics, QColor, QPixmap, QPainter, QContextMenuEvent


from utils.fontformat import FontFormat
from utils.config import save_text_styles, text_styles
from utils import config as C
from .custom_widget import PanelArea, Widget, FlowLayout


class ArrowLeftButton(QPushButton):
    pass


class ArrowRightButton(QPushButton):
    pass


class DeleteStyleButton(QPushButton):
    pass


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


class TextStylePresetPanel(PanelArea):

    entered = False
    active_text_style_label_changed = Signal()
    apply_fontfmt = Signal(FontFormat)
    active_stylename_edited = Signal()
    export_style = Signal()
    import_style = Signal()

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_text_style_label: TextStyleLabel = None
        self.flayout = FlowLayout()
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
        self.setContentLayout(self.flayout)

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
