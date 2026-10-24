import os
import sys
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel


class SliderRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_drawing_panel_constructs_without_slot_exceptions(self) -> None:
        errors = Mock()
        with patch.object(sys, 'excepthook', errors):
            canvas = Canvas()
            panel = DrawingPanel(canvas)
            try:
                self.app.processEvents()
                for controls in (panel.penConfigPanel, panel.inpaintConfigPanel):
                    self.assertGreaterEqual(controls.thicknessSlider.value(), 1)
            finally:
                panel.deleteLater()
                canvas.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        errors.assert_not_called()

    def test_brush_range_clips_then_aligns_handle_without_slot_exceptions(self) -> None:
        errors = Mock()
        with patch.object(sys, 'excepthook', errors):
            canvas = Canvas()
            panel = DrawingPanel(canvas)
            slider = panel.penConfigPanel.thicknessSlider
            try:
                slider.resize(240, 22)
                for value, minimum, maximum, expected in (
                    (1, 25, 1000, 25),
                    (1000, 25, 50, 50),
                    (25, -25, 50, 25),
                    (-10, 1, 1000, 1),
                ):
                    with self.subTest(value=value, range=(minimum, maximum)):
                        slider.setValue(value)
                        slider.setRange(minimum, maximum)
                        self.assertEqual(slider.value(), expected)
                        if expected == minimum:
                            self.assertEqual(slider.handle.x(), 0)
                        elif expected == maximum:
                            self.assertEqual(slider.handle.x(), slider.grooveLength)
                        QTest.mouseClick(slider, Qt.MouseButton.LeftButton,
                                         pos=slider.handle.geometry().center())
                        self.assertLessEqual(abs(slider.value() - expected), 1)
            finally:
                panel.deleteLater()
                canvas.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        errors.assert_not_called()
