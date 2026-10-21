"""Peek modes hide and restore exactly the layers above the peeked one."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene

from ballontranslator.ui.layer_peek import (
    LayerPeekButton,
    PeekController,
    PeekMode,
    PeekResizeWatcher,
    make_peek_icon,
    peek_key_map,
)


class FakeCanvas(QGraphicsScene):
    def __init__(self) -> None:
        super().__init__()
        self.inpaintLayer = QGraphicsPixmapItem()
        self.addItem(self.inpaintLayer)
        pixmap = QPixmap(4, 4)
        pixmap.fill(Qt.GlobalColor.white)
        self.inpaintLayer.setPixmap(pixmap)
        array = np.zeros((4, 4, 3), dtype=np.uint8)
        array[:, :, 0] = 200
        self.img_array = array


class FakeItem:
    def __init__(self, visible: bool = True) -> None:
        self.visible = visible

    def isVisible(self) -> bool:
        return self.visible

    def setVisible(self, visible: bool) -> None:
        self.visible = visible


class PeekControllerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.canvas = FakeCanvas()
        self.items = [FakeItem(), FakeItem(False)]
        self.controller = PeekController(
            self.canvas, lambda: self.items
        )
        self.original_key = self.canvas.inpaintLayer.pixmap().cacheKey()

    def test_original_shows_the_source_page_and_hides_text(self) -> None:
        self.controller.set_mode(PeekMode.ORIGINAL)
        # The base normally paints the inpainted page, so the peek swaps in
        # the untouched array instead of hiding the layer (which showed blank).
        self.assertIsNot(
            self.canvas.inpaintLayer.pixmap().cacheKey(), self.original_key
        )
        self.assertFalse(self.items[0].visible)

    def test_inpainted_hides_text_only(self) -> None:
        self.controller.set_mode(PeekMode.INPAINTED)
        self.assertTrue(self.canvas.inpaintLayer.isVisible())
        self.assertFalse(self.items[0].visible)

    def test_returning_to_current_restores_exactly(self) -> None:
        self.controller.set_mode(PeekMode.ORIGINAL)
        self.controller.set_mode(PeekMode.CURRENT)
        self.assertEqual(
            self.canvas.inpaintLayer.pixmap().cacheKey(), self.original_key
        )
        self.assertTrue(self.items[0].visible)
        self.assertFalse(self.items[1].visible, 'a hidden item stays hidden')

    def test_reset_is_safe_when_already_current(self) -> None:
        self.controller.reset()
        self.assertEqual(self.controller.mode, PeekMode.CURRENT)

    def test_switching_between_modes_does_not_lose_the_original_state(
        self,
    ) -> None:
        self.controller.set_mode(PeekMode.ORIGINAL)
        self.controller.set_mode(PeekMode.INPAINTED)
        self.controller.set_mode(PeekMode.CURRENT)
        self.assertEqual(
            self.canvas.inpaintLayer.pixmap().cacheKey(), self.original_key
        )
        self.assertTrue(self.items[0].visible)
        self.assertFalse(self.items[1].visible)


class PeekWidgetsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_each_mode_has_its_own_icon(self) -> None:
        icons = [make_peek_icon(mode) for mode in PeekMode]
        keys = {icon.pixmap(16, 16).cacheKey() for icon in icons}
        self.assertEqual(len(keys), 3)

    def test_holding_the_button_peeks_and_releasing_returns(self) -> None:
        canvas = FakeCanvas()
        canvas.img_array = [[255]]
        controller = PeekController(canvas, lambda: [])
        from qtpy.QtWidgets import QWidget
        parent = QWidget()
        button = LayerPeekButton(controller, parent)
        self.assertEqual(controller.mode, PeekMode.CURRENT)

        button.begin_peek(PeekMode.ORIGINAL)
        self.assertEqual(controller.mode, PeekMode.ORIGINAL)
        button.end_peek()
        self.assertEqual(controller.mode, PeekMode.CURRENT)

        button.begin_peek(PeekMode.INPAINTED)
        self.assertEqual(controller.mode, PeekMode.INPAINTED)
        button.end_peek()
        self.assertEqual(controller.mode, PeekMode.CURRENT)

    def test_resize_watcher_reports_view_changes(self) -> None:
        from qtpy.QtCore import QSize
        from qtpy.QtGui import QResizeEvent
        from qtpy.QtWidgets import QWidget
        calls = []
        watcher = PeekResizeWatcher(lambda: calls.append(1))
        widget = QWidget()
        widget.installEventFilter(watcher)
        # A hidden widget may not emit a resize; post it explicitly.
        QApplication.sendEvent(
            widget, QResizeEvent(QSize(120, 90), QSize(10, 10))
        )
        self.assertTrue(calls)

    def test_default_shortcuts_map_to_the_two_peek_modes(self) -> None:
        keys = peek_key_map('Tab', '`')
        self.assertEqual(len(keys), 2)
        self.assertIn(PeekMode.ORIGINAL, keys.values())
        self.assertIn(PeekMode.INPAINTED, keys.values())


if __name__ == '__main__':
    unittest.main()
