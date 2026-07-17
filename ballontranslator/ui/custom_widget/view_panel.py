from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
)
from qtpy.QtCore import Qt, Signal, QEvent
from qtpy.QtGui import QFontMetrics, QMouseEvent

from .scrollbar import ScrollBar
from .widget import Widget
from ..icon_rendering import render_svg_pixmap
from ..misc import themed_icon_path
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg

CHEVRON_SIZE = 12
CHEVRON_SIZE_SMALL = 14


def _chevron_pixmap(filename: str, size: int, device_pixel_ratio: float):
    return render_svg_pixmap(
        themed_icon_path(filename),
        size,
        size,
        device_pixel_ratio,
    )



class HidePanelButton(QPushButton):
    pass


class ExpandLabel(Widget):

    clicked = Signal()

    def __init__(self, text=None, parent=None, size_type='normal', *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self.size_type = size_type
        self.textlabel = QLabel(self)
        self.textlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.arrowlabel = QLabel(self)
        self.arrowlabel.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        font = self.textlabel.font()
        if size_type == 'normal':
            if shared.ON_MACOS:
                font.setPointSize(13)
            else:
                font.setPointSizeF(10)
            self.setFixedHeight(26)
            self._chevron_size = CHEVRON_SIZE
        elif size_type == 'small':
            if shared.ON_MACOS:
                font.setPointSize(10)
            else:
                font.setPointSizeF(8)
            self.setFixedHeight(20)
            self._chevron_size = CHEVRON_SIZE_SMALL
        else:
            raise
        self.arrowlabel.setFixedSize(self._chevron_size, self._chevron_size)
            
        self.textlabel.setFont(font)
        self.hidelabel = HidePanelButton(self)
        self.hidelabel.setVisible(False)

        if text is not None:
            self.textlabel.setText(text)
        layout = QHBoxLayout(self)
        layout.addWidget(self.arrowlabel)
        layout.addSpacing(3)
        layout.addWidget(self.textlabel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch(-1)
        layout.addWidget(self.hidelabel)
    
        self.expanded = True
        self.setExpand(True)

    def enterEvent(self, event) -> None:
        self.hidelabel.setVisible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.hidelabel.setVisible(False)
        return super().leaveEvent(event)

    def setExpand(self, expand: bool):
        self.expanded = expand
        icon_name = 'chevron-down.svg' if expand else 'chevron-right.svg'
        self.arrowlabel.setPixmap(
            _chevron_pixmap(
                icon_name,
                self._chevron_size,
                self.arrowlabel.devicePixelRatioF(),
            )
        )

    def changeEvent(self, event: QEvent) -> None:
        if event.type() == QEvent.Type.StyleChange:
            self.setExpand(self.expanded)
        return super().changeEvent(event)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if e.button() == Qt.MouseButton.LeftButton:
            self.setExpand(not self.expanded)
            pcfg.expand_tstyle_panel = self.expanded
            self.clicked.emit()
        return super().mousePressEvent(e)



class PanelArea(QScrollArea):
    def __init__(self, panel_name: str, config_name: str, config_expand_name: str, action_name: str = None):
        super().__init__()
        self.scrollContent = PanelAreaContent()
        self.scrollContent.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setWidget(self.scrollContent)
        self.setWidgetResizable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        ScrollBar(Qt.Orientation.Vertical, self)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ScrollBar(Qt.Orientation.Horizontal, self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.view_widget = ViewWidget(self, panel_name)
        self.view_hide_btn_clicked = self.view_widget.view_hide_btn_clicked
        self.expand_changed = self.view_widget.expend_changed
        self.title = self.view_widget.title
        self.setTitle = self.view_widget.setTitle
        self.elidedText = self.view_widget.elidedText
        self.set_expend_area = self.view_widget.set_expend_area

        if action_name is None:
            action_name = panel_name
        self.view_widget.register_view_widget(
            config_name=config_name, 
            config_expand_name=config_expand_name,
            action_name=action_name
        )

    def setContentLayout(self, layout):
        self.scrollContent.setLayout(layout)


class PanelGroupBox(QGroupBox):
    pass


class PanelAreaContent(Widget):

    after_resized = Signal()
    
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.after_resized.emit()


class ViewWidget(Widget):

    config_name: str = ''
    config_expand_name: str = ''
    action_name: str = ''
    view_hide_btn_clicked = Signal(str)
    expend_changed = Signal()

    def __init__(self, content_widget: Widget, panel_name: str = None, parent=None, title_size_type='normal', *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        
        self.title_label = ExpandLabel(panel_name, self, size_type=title_size_type)
        self.title_label.hidelabel.clicked.connect(self.on_view_hide_btn_clicked)
        self.content_widget = content_widget

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.content_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.title_label.clicked.connect(self.set_expend_area)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

    def on_view_hide_btn_clicked(self):
        self.view_hide_btn_clicked.emit(self.config_name)
    
    def register_view_widget(self, config_name: str, config_expand_name: str, action_name: str):
        self.config_name = config_name
        self.config_expand_name = config_expand_name
        self.action_name = action_name
        shared.register_view_widget(self)

    def set_expend_area(self, expend: bool = None, set_config: bool = True):
        if expend is None:
            return self.set_expend_area(self.title_label.expanded)
        if self.title_label.expanded != expend:
            self.title_label.setExpand(expend)
        self.content_widget.setVisible(expend)
        if set_config:
            setattr(pcfg, self.config_expand_name, expend)

    def setTitle(self, text: str):
        self.title_label.textlabel.setText(text)

    def elidedText(self, text: str):
        fm = QFontMetrics(self.title_label.font())
        return fm.elidedText(text, Qt.TextElideMode.ElideRight, self.content_widget.width() - 40)

    def title(self) -> str:
        return self.title_label.textlabel.text()
