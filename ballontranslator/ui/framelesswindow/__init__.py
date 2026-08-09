# modified from https://github.com/zhiyiYo/PyQt-Frameless-Window

from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPalette, QPen
from qtpy.QtWidgets import QAbstractButton, QApplication, QWidget

from ballontranslator.utils import shared

if not shared.FLAG_QT6:

    from .fw_qt5 import FramelessMoveResize
    from .fw_qt5 import FramelessWindow

else:
    from .fw_qt6 import FramelessMoveResize
    from .fw_qt6 import FramelessWindow


class DialogCloseButton(QAbstractButton):
    """Small title-bar button that paints its own close glyph.

    >>> DialogCloseButton.__name__
    'DialogCloseButton'
    """

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setObjectName('DialogCloseButton')
        self.setFixedSize(26, 26)
        self.setToolTip(self.tr('Close'))
        self.setAccessibleName(self.tr('Close'))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter).Antialiasing
        painter.setRenderHint(render_hint)

        hovered = self.underMouse() or self.isDown()
        color_role = getattr(QPalette, 'ColorRole', QPalette)
        if hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor('#E81123'))
            painter.drawRoundedRect(self.rect(), 6, 6)

        role = color_role.HighlightedText if hovered else color_role.WindowText
        color = self.palette().color(role)
        if not hovered:
            color.setAlpha(210)
        pen = QPen(color)
        pen.setWidthF(1.6)
        pen.setCapStyle(getattr(getattr(Qt, 'PenCapStyle', Qt), 'RoundCap'))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        inset = 8.5
        end = self.width() - inset
        painter.drawLine(QPointF(inset, inset), QPointF(end, end))
        painter.drawLine(QPointF(end, inset), QPointF(inset, end))


class OutsideClickFramelessMixin:
    """Provide centered, draggable, outside-click-closing window behavior.

    Place this mixin before the widget base class. Subclasses must expose
    ``title_bar`` and optionally ``close_button`` attributes. Override
    ``_dismiss_transient_window`` when hiding is not the desired close
    behavior.

    >>> OutsideClickFramelessMixin._preserve_on_outside_click(None)
    False
    """

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._install_outside_click_filter()
        parent = self.parentWidget()
        if parent is not None:
            geometry = self.frameGeometry()
            geometry.moveCenter(parent.window().frameGeometry().center())
            self.move(geometry.topLeft())

    def hideEvent(self, event) -> None:
        self._remove_outside_click_filter()
        super().hideEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._dismiss_transient_window()
            event.accept()
            return
        super().keyPressEvent(event)

    def _install_outside_click_filter(self) -> None:
        if getattr(self, '_outside_click_filter_installed', False):
            return
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._outside_click_filter_installed = True

    def _remove_outside_click_filter(self) -> None:
        if not getattr(self, '_outside_click_filter_installed', False):
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._outside_click_filter_installed = False

    def eventFilter(self, watched, event):
        if (
            not self.isVisible()
            or not isinstance(watched, QWidget)
            or not isinstance(event, QMouseEvent)
        ):
            return QWidget.eventFilter(self, watched, event)

        event_type = event.type()
        inside_window = self._widget_inside_window(watched)
        if (
            event_type == QEvent.Type.MouseButtonPress
            and QApplication.activePopupWidget() is None
            and not inside_window
            and not self._preserve_on_outside_click()
        ):
            self._dismiss_transient_window()

        handled = super().eventFilter(watched, event)
        if handled:
            return True
        if (
            inside_window
            and event_type == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
            and self._can_drag_title(watched)
        ):
            FramelessMoveResize.startSystemMove(
                self,
                self._global_mouse_pos(event),
            )
            return True
        return handled

    def _dismiss_transient_window(self) -> None:
        self.hide()

    def _preserve_on_outside_click(self) -> bool:
        return False

    @staticmethod
    def _global_mouse_pos(event: QMouseEvent):
        if hasattr(event, 'globalPosition'):
            return event.globalPosition().toPoint()
        return event.globalPos()

    def _can_drag_title(self, watched: QWidget) -> bool:
        close_button = getattr(self, 'close_button', None)
        if close_button is not None and (
            watched is close_button or close_button.isAncestorOf(watched)
        ):
            return False
        return watched is self.title_bar or self.title_bar.isAncestorOf(watched)

    def _widget_inside_window(self, widget: QWidget) -> bool:
        while widget is not None:
            if widget is self:
                return True
            widget = widget.parentWidget()
        return False
