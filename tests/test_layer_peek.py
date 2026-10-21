"""Peek modes hide and restore exactly the layers above the peeked one."""

import os
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene

from ballontranslator.ui.layer_peek import (
    LayerPeekButton,
    PeekController,
    PeekMode,
    make_peek_icon,
    peek_key_map,
)


class FakeCanvas(QGraphicsScene):
    def __init__(self) -> None:
        super().__init__()
        self.inpaintLayer = QGraphicsPixmapItem()
        self.addItem(self.inpaintLayer)


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

    def test_original_hides_inpaint_and_text(self) -> None:
        self.controller.set_mode(PeekMode.ORIGINAL)
        self.assertFalse(self.canvas.inpaintLayer.isVisible())
        self.assertFalse(self.items[0].visible)

    def test_inpainted_hides_text_only(self) -> None:
        self.controller.set_mode(PeekMode.INPAINTED)
        self.assertTrue(self.canvas.inpaintLayer.isVisible())
        self.assertFalse(self.items[0].visible)

    def test_returning_to_current_restores_exactly(self) -> None:
        self.controller.set_mode(PeekMode.ORIGINAL)
        self.controller.set_mode(PeekMode.CURRENT)
        self.assertTrue(self.canvas.inpaintLayer.isVisible())
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
        self.assertTrue(self.canvas.inpaintLayer.isVisible())
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

    def test_button_cycles_through_the_three_states(self) -> None:
        canvas = FakeCanvas()
        controller = PeekController(canvas, lambda: [])
        from qtpy.QtWidgets import QWidget
        parent = QWidget()
        button = LayerPeekButton(controller, parent)
        self.assertEqual(controller.mode, PeekMode.CURRENT)
        button._cycle()
        self.assertEqual(controller.mode, PeekMode.ORIGINAL)
        button._cycle()
        self.assertEqual(controller.mode, PeekMode.INPAINTED)
        button._cycle()
        self.assertEqual(controller.mode, PeekMode.CURRENT)

    def test_default_shortcuts_map_to_the_two_peek_modes(self) -> None:
        keys = peek_key_map('Tab', '`')
        self.assertEqual(len(keys), 2)
        self.assertIn(PeekMode.ORIGINAL, keys.values())
        self.assertIn(PeekMode.INPAINTED, keys.values())


if __name__ == '__main__':
    unittest.main()
