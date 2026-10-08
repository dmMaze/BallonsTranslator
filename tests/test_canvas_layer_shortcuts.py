import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QSlider

from ballontranslator.ui.canvas import Canvas


class CanvasLayerShortcutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = Canvas()
        self.canvas.editor_index = 0
        self.canvas.editing_layer_opacity_slider = QSlider()
        self.canvas.originallayer_trans_slider = QSlider()
        for slider in (
            self.canvas.editing_layer_opacity_slider,
            self.canvas.originallayer_trans_slider,
        ):
            slider.setRange(0, 100)
        self.canvas.editing_layer_opacity_slider.valueChanged.connect(
            self.canvas.setEditingLayerOpacityBySlider
        )
        self.update_count = 0
        self.canvas.updateLayers = self._record_update

    def _record_update(self) -> None:
        self.update_count += 1

    def _press(self, key: Qt.Key) -> None:
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            key,
            Qt.KeyboardModifier.NoModifier,
        )
        self.canvas.keyPressEvent(event)

    def test_number_key_blends_layers_in_paint_mode(self) -> None:
        self._press(Qt.Key.Key_3)

        self.assertEqual(self.canvas.editing_layer_opacity_slider.value(), 30)
        self.assertEqual(self.canvas.originallayer_trans_slider.value(), 70)
        self.assertEqual(self.canvas.textLayer.opacity(), 0.3)
        self.assertEqual(self.update_count, 1)

    def test_zero_toggles_original_and_inpainted_in_paint_mode(self) -> None:
        self.canvas.editing_layer_opacity_slider.setValue(100)
        self._press(Qt.Key.Key_0)

        self.assertEqual(self.canvas.editing_layer_opacity_slider.value(), 0)
        self.assertEqual(self.canvas.originallayer_trans_slider.value(), 100)

        self._press(Qt.Key.Key_0)

        self.assertEqual(self.canvas.editing_layer_opacity_slider.value(), 100)
        self.assertEqual(self.canvas.originallayer_trans_slider.value(), 0)


if __name__ == '__main__':
    unittest.main()
