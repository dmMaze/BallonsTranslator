import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QFocusEvent, QKeyEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget

from ballontranslator.ui.custom_widget import SizeComboBox, SizeControlLabel


class SizeControlLabelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_mouse_press_blurs_editor_before_drag_updates(self) -> None:
        host = QWidget()
        self.addCleanup(host.deleteLater)
        layout = QHBoxLayout(host)
        editor = SizeComboBox([0, 10], 'stroke_width', host)
        editor.setValue(1)
        label = SizeControlLabel(host, text='Stroke')
        layout.addWidget(editor)
        layout.addWidget(label)
        label.size_ctrl_changed.connect(editor.changeByDelta)
        host.show()
        host.activateWindow()
        editor.setFocus()
        self.app.processEvents()
        self.assertTrue(editor.hasFocus())

        live_edits = []
        commits = []
        editor.param_changed.connect(
            lambda name, value: live_edits.append((name, value))
        )
        label.btn_released.connect(lambda: commits.append(editor.value()))
        QTest.mousePress(label, Qt.MouseButton.LeftButton)
        self.app.processEvents()

        self.assertTrue(label.hasFocus())
        self.assertFalse(editor.hasFocus())
        label.size_ctrl_changed.emit(5)
        self.assertEqual(live_edits, [])
        self.assertAlmostEqual(editor.value(), 1.05)
        QTest.mouseRelease(label, Qt.MouseButton.LeftButton)
        self.assertEqual(commits, [1.05])

    def test_deferred_typed_value_commits_once_and_restores_invalid_input(self):
        host = QWidget()
        self.addCleanup(host.deleteLater)
        layout = QHBoxLayout(host)
        editor = SizeComboBox(
            [1, 1000],
            'font_size',
            host,
            defer_text_changes=True,
        )
        editor.setValue(80)
        layout.addWidget(editor)
        host.show()
        editor.lineEdit().setFocus()
        self.app.processEvents()

        changes = []
        editor.param_changed.connect(
            lambda name, value: changes.append((name, value))
        )

        editor.lineEdit().setText('800')
        self.assertEqual(changes, [])
        QApplication.sendEvent(
            editor.lineEdit(),
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        # Qt may still deliver these signals in a different order; the
        # committed state must make them no-ops.
        editor.lineEdit().returnPressed.emit()
        editor.lineEdit().editingFinished.emit()
        editor.activated.emit(editor.currentIndex())
        self.assertEqual(changes, [('font_size', 800.0)])

        editor.lineEdit().setText('1001')
        QApplication.sendEvent(
            editor.lineEdit(),
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Return,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(changes, [('font_size', 800.0)])
        self.assertEqual(editor.currentText(), '800')

        editor.lineEdit().setText('900')
        QApplication.sendEvent(
            editor.lineEdit(),
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        self.assertEqual(changes, [('font_size', 800.0)])
        self.assertEqual(editor.currentText(), '800')

        editor.lineEdit().setText('850')
        QApplication.sendEvent(
            editor.lineEdit(),
            QFocusEvent(QEvent.Type.FocusOut),
        )
        self.assertEqual(changes, [('font_size', 800.0), ('font_size', 850.0)])


if __name__ == '__main__':
    unittest.main()
