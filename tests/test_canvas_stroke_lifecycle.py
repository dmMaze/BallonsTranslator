import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPen
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.image_edit import StrokeImgItem


def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()


class CanvasStrokeLifecycleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.canvas = Canvas()
        self.canvas.imgtrans_proj = SimpleNamespace(
            img_valid=False,
            inpainted_valid=False,
        )
        self.canvas.baseLayer.setRect(QRectF(0, 0, 64, 64))

    def tearDown(self) -> None:
        self.canvas.deleteLater()
        _APP.processEvents()

    def _start_stroke(self, erasing: bool = False) -> StrokeImgItem:
        self.canvas.addStrokeImageItem(
            QPointF(10, 10),
            QPen(),
            erasing=erasing,
        )
        stroke = self.canvas.stroke_img_item
        self.assertTrue(stroke.painter.isActive())
        return stroke

    def test_cleanup_ends_active_stroke_before_releasing_its_image(self) -> None:
        stroke = self._start_stroke()
        self.canvas.on_activation_changed()

        self.assertFalse(stroke.painter.isActive())
        self.assertIsNone(self.canvas.stroke_img_item)

        stroke = self._start_stroke(erasing=True)
        erase_img_key = self.canvas.erase_img_key
        self.assertIn(erase_img_key, self.canvas.drawingLayer.qimg_dict)
        self.canvas.on_hide_canvas()

        self.assertFalse(stroke.painter.isActive())
        self.assertNotIn(erase_img_key, self.canvas.drawingLayer.qimg_dict)

        stroke = self._start_stroke()
        self.canvas.updateCanvas()

        self.assertFalse(stroke.painter.isActive())
        self.assertIsNone(self.canvas.stroke_img_item)


if __name__ == '__main__':
    unittest.main()
