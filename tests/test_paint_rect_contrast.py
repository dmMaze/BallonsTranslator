import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QColor, QImage, QPainter
from qtpy.QtWidgets import QApplication, QGraphicsView

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.text_engine.shape_control import TextBlkShapeControl


class PaintRectContrastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_only_paint_box_creation_enables_contrast_outline(self) -> None:
        canvas = Canvas()

        canvas.startCreateTextblock(QPointF(10, 10), hide_control=True)
        self.assertTrue(canvas.txtblkShapeControl._contrast_outline)
        canvas.clear_states()
        self.assertFalse(canvas.txtblkShapeControl._contrast_outline)

        canvas.startCreateTextblock(QPointF(10, 10), hide_control=False)
        self.assertFalse(canvas.txtblkShapeControl._contrast_outline)

    def test_finishing_paint_box_resets_outline(self) -> None:
        canvas = Canvas()
        canvas.startCreateTextblock(QPointF(10, 10), hide_control=True)
        canvas.endCreateTextblock()
        self.assertFalse(canvas.txtblkShapeControl._contrast_outline)

    def test_resetting_interaction_restores_normal_outline(self) -> None:
        view = QGraphicsView()
        control = TextBlkShapeControl(view)
        control.setContrastOutline(True)
        control.resetInteraction()
        self.assertFalse(control._contrast_outline)

    def test_paint_box_outline_contains_black_and_white_dashes(self) -> None:
        control = TextBlkShapeControl(QGraphicsView())
        control.setRect(QRectF(5, 5, 100, 40))
        control.setContrastOutline(True)
        image = QImage(120, 60, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        control.paint(painter, None)
        painter.end()

        colors = [
            QColor.fromRgba(image.pixel(x, y))
            for y in range(image.height())
            for x in range(image.width())
        ]
        self.assertTrue(any(
            color.alpha() > 200 and color.red() > 220
            for color in colors
        ))
        self.assertTrue(any(
            color.alpha() > 200 and color.red() < 35
            for color in colors
        ))


if __name__ == '__main__':
    unittest.main()
