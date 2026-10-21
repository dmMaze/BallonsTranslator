"""Whole-page fit remembers and restores the zoom and offset."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QRectF
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.page_fit import (
    PageFitController,
    make_magnifier_icon,
)


class FakeBar:
    def __init__(self, value: int = 0) -> None:
        self._value = value

    def value(self) -> int:
        return self._value

    def setValue(self, value: int) -> None:
        self._value = int(value)


class FakeView:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.hbar = FakeBar(120)
        self.vbar = FakeBar(340)
        self.centered = None

    def viewport(self):
        return self

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def centerOn(self, point) -> None:
        self.centered = point

    def horizontalScrollBar(self):
        return self.hbar

    def verticalScrollBar(self):
        return self.vbar


class FakeCanvas:
    def __init__(self) -> None:
        self.scale_factor = 1.0
        self.gv = FakeView(573, 658)
        self.factors = []

    def sceneRect(self):
        return QRectF(0, 0, 3000, 4000)

    def scaleImage(self, factor: float) -> None:
        self.factors.append(factor)
        self.scale_factor *= factor


class PageFitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_fit_scales_the_page_into_the_viewport(self) -> None:
        canvas = FakeCanvas()
        controller = PageFitController(canvas)
        self.assertTrue(controller.fit())
        expected = min(573 / 3000, 658 / 4000)
        self.assertAlmostEqual(canvas.scale_factor, expected, places=6)
        self.assertTrue(controller.fitted)
        self.assertIsNotNone(canvas.gv.centered)

    def test_restore_returns_zoom_and_offset(self) -> None:
        canvas = FakeCanvas()
        controller = PageFitController(canvas)
        controller.fit()
        self.assertTrue(controller.restore())
        self.assertAlmostEqual(canvas.scale_factor, 1.0, places=6)
        self.assertEqual(canvas.gv.hbar.value(), 120)
        self.assertEqual(canvas.gv.vbar.value(), 340)
        self.assertFalse(controller.fitted)

    def test_toggle_alternates_between_the_two_states(self) -> None:
        canvas = FakeCanvas()
        controller = PageFitController(canvas)
        controller.toggle()
        self.assertTrue(controller.fitted)
        controller.toggle()
        self.assertFalse(controller.fitted)
        self.assertAlmostEqual(canvas.scale_factor, 1.0, places=6)

    def test_degenerate_page_is_ignored(self) -> None:
        canvas = FakeCanvas()
        canvas.sceneRect = lambda: QRectF(0, 0, 0, 0)
        controller = PageFitController(canvas)
        self.assertFalse(controller.fit())
        self.assertFalse(controller.fitted)

    def test_magnifier_icon_renders(self) -> None:
        icon = make_magnifier_icon(64, outline=3)
        self.assertFalse(icon.pixmap(64, 64).isNull())


if __name__ == '__main__':
    unittest.main()
