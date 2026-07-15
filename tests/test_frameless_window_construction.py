import os
import sys
import unittest
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QWidget
from qtpy import API_NAME

from ballontranslator.utils import shared

shared.FLAG_QT6 = API_NAME in ('PyQt6', 'PySide6')


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.framelesswindow import FramelessWindow


class _ConstructionStopped(Exception):
    pass


class FramelessWindowConstructionTest(unittest.TestCase):
    def test_config_panel_starts_as_an_owned_dialog(self):
        parent = QWidget()
        observed = {}

        class ProbeConfigPanel(ConfigPanel):
            def updateFrameless(self):
                observed['panel'] = self
                observed['flags'] = self.windowFlags()
                observed['is_window'] = self.isWindow()
                raise _ConstructionStopped

        try:
            with self.assertRaises(_ConstructionStopped):
                ProbeConfigPanel(parent)

            window_type = getattr(Qt, 'WindowType', Qt)
            self.assertTrue(issubclass(ConfigPanel, FramelessWindow))
            self.assertTrue(observed['flags'] & window_type.Dialog)
            self.assertTrue(observed['is_window'])
        finally:
            panel = observed.get('panel')
            if panel is not None:
                panel.deleteLater()
            parent.deleteLater()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows native-window regression')
    def test_owned_frameless_window_defers_native_effects_until_show(self):
        from ballontranslator.ui.framelesswindow.win_window_effect import WindowsWindowEffect

        parent = QWidget()
        window_type = getattr(Qt, 'WindowType', Qt)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        parent.setWindowFlags(window_type.Window)
        parent.winId()

        try:
            with mock.patch.object(WindowsWindowEffect, 'addWindowAnimation') as animation, \
                    mock.patch.object(WindowsWindowEffect, 'addShadowEffect') as shadow:
                owned = FramelessWindow(parent, window_type.Dialog)
                sibling = QWidget(parent)

                self.assertTrue(owned.windowFlags() & window_type.Dialog)
                self.assertFalse(animation.called)
                self.assertFalse(shadow.called)
                self.assertFalse(sibling.testAttribute(widget_attribute.WA_NativeWindow))

                owned.show()
                QApplication.processEvents()

                animation.assert_called_once()
                shadow.assert_called_once()
                self.assertFalse(sibling.testAttribute(widget_attribute.WA_NativeWindow))
        finally:
            if 'owned' in locals():
                owned.close()
                owned.deleteLater()
            if 'sibling' in locals():
                sibling.deleteLater()
            parent.deleteLater()


if __name__ == '__main__':
    unittest.main()
