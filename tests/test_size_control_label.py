import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
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


if __name__ == '__main__':
    unittest.main()
