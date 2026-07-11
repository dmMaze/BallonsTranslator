import os
import sys
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QEvent, QPointF, Qt
from qtpy.QtGui import QColor, QMouseEvent
from qtpy.QtWidgets import QApplication, QColorDialog, QDialog, QLabel, QMenu, QTextEdit, QWidget
from qtpy import API_NAME


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.custom_widget.label import ColorPickerLabel
from ballontranslator.ui.menu_style import MenuStyleFilter
from ballontranslator.ui.textedit_area import FloatingSuggestionLabel

if API_NAME in ('PyQt6', 'PySide6'):
    from ballontranslator.ui.framelesswindow.fw_qt6.linux_frameless_window import LinuxFramelessWindow, LinuxMoveResize
else:
    LinuxFramelessWindow = None
    LinuxMoveResize = None


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

    def test_color_picker_uses_window_parent_for_dialog(self):
        parent = QWidget()
        label = ColorPickerLabel(parent)
        calls = []
        original_get_color = QColorDialog.getColor

        def fake_get_color(initial, dialog_parent, *args, **kwargs):
            calls.append((initial, dialog_parent))
            return QColor(1, 2, 3)

        QColorDialog.getColor = staticmethod(fake_get_color)
        try:
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(1, 1),
                QPointF(1, 1),
                QPointF(1, 1),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            label.mousePressEvent(event)
        finally:
            QColorDialog.getColor = original_get_color
            label.deleteLater()
            parent.deleteLater()

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1], parent.window())
        self.assertEqual(label.rgb(), (1, 2, 3))

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

    @unittest.skipIf(LinuxFramelessWindow is None, 'Qt6 frameless filter is not active')
    def test_linux_frameless_window_starts_resize_on_border_press(self):
        window = LinuxFramelessWindow()
        calls = []
        original_resize = LinuxMoveResize.__dict__['starSystemResize']

        def fake_resize(cls, resize_window, global_pos, edges):
            calls.append((resize_window, global_pos.toPoint(), edges))

        LinuxMoveResize.starSystemResize = classmethod(fake_resize)
        try:
            window.move(100, 100)
            window.resize(300, 200)
            window.show()
            event = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(2, 50),
                QPointF(2, 50),
                QPointF(102, 150),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            handled = window.eventFilter(window, event)
        finally:
            LinuxMoveResize.starSystemResize = original_resize
            QApplication.instance().removeEventFilter(window)
            window.deleteLater()

        self.assertTrue(handled)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], window)
        self.assertTrue(calls[0][2] & Qt.Edge.LeftEdge)


class MenuStyleFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = MenuStyleFilter(QApplication.instance())

    def test_ignores_non_menu_before_reading_event(self):
        class ExplodingEvent:
            def type(self):
                raise AssertionError('irrelevant objects must not read event details')
        self.assertFalse(self.filter.eventFilter(QLabel(), ExplodingEvent()))

    def test_show_and_resize_apply_rounded_mask(self):
        menu = QMenu()
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Polish))
        self.assertTrue(menu.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
        self.assertTrue(menu.mask().isEmpty())
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Resize))
        self.assertFalse(menu.mask().isEmpty())
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Show))
        self.assertFalse(menu.mask().isEmpty())

    def test_checked_marker_is_idempotent_and_tracks_uncheck(self):
        menu = QMenu()
        action = menu.addAction('Dark Mode')
        original_text = action.text()
        action.setCheckable(True)
        action.setChecked(True)
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Show))
        checked_text = action.text()
        self.assertNotEqual(checked_text, original_text)
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Show))
        self.assertEqual(action.text(), checked_text)
        action.setChecked(False)
        self.filter.eventFilter(menu, QEvent(QEvent.Type.Show))
        self.assertEqual(action.text(), original_text)


if __name__ == '__main__':
    unittest.main()
