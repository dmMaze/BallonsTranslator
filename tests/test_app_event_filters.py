import os
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QApplication, QDialog, QTextEdit
from qtpy import API_NAME


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.textedit_area import FloatingSuggestionLabel

if API_NAME in ('PyQt6', 'PySide6'):
    from ballontranslator.ui.framelesswindow.fw_qt6.linux_frameless_window import LinuxFramelessWindow
else:
    LinuxFramelessWindow = None


class TypeSensitiveEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type.User)
        self.type_requested = False

    def type(self):
        self.type_requested = True
        raise RecursionError('event type should not be requested')


class MinimalConfigPanel(ConfigPanel):
    def __init__(self):
        QDialog.__init__(self)
        self._outside_click_filter_installed = False

    def _widgetInsidePanel(self, widget) -> bool:
        return False

    def _activeWidgetInWhitelist(self) -> bool:
        return False


class AppEventFilterOrderingTest(unittest.TestCase):
    def test_config_panel_ignores_non_widget_events_before_type(self):
        panel = MinimalConfigPanel()
        panel.show()
        event = TypeSensitiveEvent()
        try:
            panel.eventFilter(QObject(), event)
        finally:
            panel.close()
            panel.deleteLater()

        self.assertFalse(event.type_requested)

    def test_suggestion_popup_ignores_non_app_events_before_type(self):
        editor = QTextEdit()
        popup = FloatingSuggestionLabel(editor)
        event = TypeSensitiveEvent()
        try:
            popup.eventFilter(editor, event)
        finally:
            QApplication.instance().removeEventFilter(popup)
            popup.deleteLater()
            editor.deleteLater()

        self.assertFalse(event.type_requested)

    @unittest.skipIf(LinuxFramelessWindow is None, 'Qt6 frameless filter is not active')
    def test_linux_frameless_window_ignores_non_mouse_events_before_type(self):
        window = LinuxFramelessWindow()
        event = TypeSensitiveEvent()
        try:
            window.eventFilter(QObject(), event)
        finally:
            QApplication.instance().removeEventFilter(window)
            window.deleteLater()

        self.assertFalse(event.type_requested)


if __name__ == '__main__':
    unittest.main()
