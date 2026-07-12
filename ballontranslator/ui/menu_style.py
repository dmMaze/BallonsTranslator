import sys

from qtpy.QtCore import QEvent, QObject, QRectF, Qt
from qtpy.QtGui import QPainterPath, QRegion
from qtpy.QtWidgets import QMenu


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
        elif event_type in (QEvent.Type.Show, QEvent.Type.Resize):
            if sys.platform != 'darwin':
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
