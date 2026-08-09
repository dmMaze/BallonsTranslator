import gc
import os
import sys
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QColor, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QLabel,
    QMenu,
    QTextEdit,
    QWidget,
)
from qtpy import API_NAME

from ballontranslator.utils import shared

shared.FLAG_QT6 = API_NAME in ('PyQt6', 'PySide6')
application_attribute = getattr(Qt, 'ApplicationAttribute', Qt)
QApplication.setAttribute(
    application_attribute.AA_DontCreateNativeWidgetSiblings,
    True,
)


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

from ballontranslator.ui.configpanel import ConfigPanel, FontExcludeDialog
from ballontranslator.ui.custom_widget.label import ColorPickerLabel
from ballontranslator.ui.icon_rendering import render_svg_pixmap
from ballontranslator.ui.menu_style import DropDownStyleFilter, MenuStyleFilter
from ballontranslator.ui.module_tool_button import ModuleSelectionWidget
from ballontranslator.ui.spellcheck import AddWordItemWidget, WordListItemWidget
from ballontranslator.ui.text_engine.editing.widgets import FloatingSuggestionLabel

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


class MinimalFontExcludeDialog(FontExcludeDialog):
    def __init__(self):
        QDialog.__init__(self)


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

    def test_font_exclude_dialog_ignores_non_widget_events_before_type(self):
        dialog = MinimalFontExcludeDialog()
        dialog.show()
        event = TypeSensitiveEvent()
        try:
            dialog.eventFilter(QObject(), event)
        finally:
            dialog.close()
            dialog.deleteLater()

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
            window.close()
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

    def test_linux_show_and_resize_apply_rounded_widget_mask(self):
        menu = QMenu()
        with patch('ballontranslator.ui.menu_style.sys.platform', 'linux'):
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

    def test_transient_style_records_follow_widget_destruction(self):
        dropdown_filter = DropDownStyleFilter(QApplication.instance())
        combo = QComboBox()
        dropdown_filter._style_view(combo, combo.view())
        self.assertEqual(len(dropdown_filter._delegates), 1)

        menu = QMenu()
        self.filter._menu_border_overlay(menu, create=True)
        self.assertEqual(len(self.filter._menu_border_overlays), 1)

        combo.deleteLater()
        menu.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()

        self.assertEqual(dropdown_filter._delegates, {})
        self.assertEqual(self.filter._menu_border_overlays, {})


class DynamicCallbackLifecycleTest(unittest.TestCase):
    def test_module_menu_action_selects_its_stored_value(self):
        widget = ModuleSelectionWidget('Module', 'translate.svg')
        widget.selector.addItems(['First', 'Second'])
        widget.selector.setCurrentText('First')

        widget.rebuildMenu()
        second_action = next(
            action for action in widget.menu.actions()
            if action.data() == 'Second'
        )
        second_action.trigger()

        self.assertEqual(widget.selector.currentText(), 'Second')
        widget.deleteLater()

    def test_replaced_suggestion_button_is_collectable_and_still_dispatches(self):
        class Editor(QTextEdit):
            def __init__(self):
                super().__init__()
                self.replacements = []

            def _replace_word(self, cursor, replacement):
                self.replacements.append((cursor, replacement))

        editor = Editor()
        popup = FloatingSuggestionLabel(editor)
        popup.set_suggestions('cursor', 'word', ['first'])
        old_button = popup.buttons_layout.itemAt(0).widget()
        old_button_ref = weakref.ref(old_button)

        popup.set_suggestions('cursor', 'word', ['second'])
        del old_button
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QApplication.processEvents()
        gc.collect()
        self.assertIsNone(old_button_ref())

        new_button = popup.buttons_layout.itemAt(0).widget()
        new_button.click()
        self.assertEqual(editor.replacements, [('cursor', 'second')])

        QApplication.instance().removeEventFilter(popup)
        popup.deleteLater()
        editor.deleteLater()

    def test_dictionary_rows_emit_values_without_callback_wrappers(self):
        deleted = []
        row = WordListItemWidget('Hello')
        row.delete_requested.connect(deleted.append)
        row.delete_btn.click()
        self.assertEqual(deleted, ['Hello'])

        added = []
        add_row = AddWordItemWidget()
        add_row.word_added.connect(added.append)
        add_row.input_field.setText('  NeW Word  ')
        QTest.keyClick(add_row.input_field, Qt.Key.Key_Return)
        self.assertEqual(added, ['new word'])
        self.assertEqual(add_row.input_field.text(), '')

        row.deleteLater()
        add_row.deleteLater()



if __name__ == '__main__':
    unittest.main()
