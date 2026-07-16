import sys

from qtpy.QtCore import QEvent, QObject, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPen, QRegion
from qtpy.QtWidgets import QMenu, QWidget

from ballontranslator.utils import shared


class _MenuBorderOverlay(QWidget):
    """Paint the Windows menu border independently of its binary mask."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('MenuBorderOverlay')
        attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
        self.setAttribute(attr_enum.WA_TransparentForMouseEvents, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        render_hint = getattr(QPainter, 'RenderHint', QPainter)
        painter.setRenderHint(render_hint.Antialiasing, True)
        brush_style = getattr(Qt, 'BrushStyle', Qt)
        painter.setBrush(brush_style.NoBrush)
        painter.setPen(QPen(QColor(*shared.BORDER_COLOR), 1))
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.drawRoundedRect(rect, 7, 7)


class MenuStyleFilter(QObject):
    """Apply shared popup geometry and text-based checked markers to menus."""

    def eventFilter(self, watched, event):
        if not isinstance(watched, QMenu):
            return False
        event_type = event.type()
        if event_type == QEvent.Type.Polish:
            attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
            watched.setAttribute(attr_enum.WA_TranslucentBackground, True)
            if sys.platform == 'darwin':
                # Use the stylesheet border instead of macOS's native popup
                # frame, which is black and leaves an outer gap around menus.
                window_type = getattr(Qt, 'WindowType', Qt)
                watched.setWindowFlag(window_type.NoDropShadowWindowHint, True)
            elif sys.platform == 'win32':
                watched.setProperty('windowsBorderOverlay', True)
                if not hasattr(watched, '_menu_border_overlay'):
                    watched._menu_border_overlay = _MenuBorderOverlay(watched)
        elif event_type in (QEvent.Type.Show, QEvent.Type.Resize):
            if sys.platform == 'win32':
                rect = QRectF(watched.rect()).adjusted(0, 0, -1, -1)
                path = QPainterPath()
                path.addRoundedRect(rect, 8, 8)
                window = watched.windowHandle()
                if window is not None:
                    window.setMask(QRegion(path.toFillPolygon().toPolygon()))
                overlay = getattr(watched, '_menu_border_overlay', None)
                if overlay is not None:
                    overlay.setGeometry(watched.rect().adjusted(1, 1, -1, -1))
                    overlay.show()
                    overlay.raise_()
                    overlay.update()
            elif sys.platform != 'darwin':
                path = QPainterPath()
                path.addRoundedRect(QRectF(watched.rect()), 8, 8)
                watched.setMask(QRegion(path.toFillPolygon().toPolygon()))
        if event_type == QEvent.Type.Show:
            self._sync_checked_action_text(watched)
        return super().eventFilter(watched, event)

    @staticmethod
    def _sync_checked_action_text(menu: QMenu):
        for action in menu.actions():
            if not action.isCheckable():
                continue
            base_text = action.property('_menuBaseText')
            if base_text is None:
                base_text = action.text()
                action.setProperty('_menuBaseText', base_text)
            action.setText(base_text + ('\t\u2713' if action.isChecked() else ''))
