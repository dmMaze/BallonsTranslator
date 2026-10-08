import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.drawingpanel import RectPanel


class DrawingBrushSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_box_dilate_can_be_typed_and_updates_the_slider(self) -> None:
        panel = RectPanel()
        panel.show()
        panel.dilateSpinBox.setFocus()
        panel.dilateSpinBox.selectAll()

        QTest.keyClicks(panel.dilateSpinBox, '17')
        QTest.keyClick(panel.dilateSpinBox, Qt.Key.Key_Return)

        self.assertEqual(panel.dilateSpinBox.value(), 17)
        self.assertEqual(panel.dilate_slider.value(), 17)
        self.assertFalse(panel.dilate_slider.show_hover_value)


if __name__ == '__main__':
    unittest.main()
