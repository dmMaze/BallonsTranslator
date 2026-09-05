import os
import sys
import unittest

import numpy as np

from ballontranslator.ui.image_edit import PenShape
from ballontranslator.utils.config import DrawPanelConfig
from ballontranslator.utils.imgproc_utils import MagicWandFillMode, magic_wand_mask


class MagicWandMaskTests(unittest.TestCase):

    def test_selects_connected_region_only(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[2:6, 2:6] = 255
        img[0, 0] = 255
        mask = magic_wand_mask(img, (3, 3), 0, 0)
        self.assertEqual(int(mask[3, 3]), 255)
        self.assertEqual(int(mask[2, 2]), 255)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[1, 1]), 0)

    def test_positive_range_expands_and_negative_shrinks(self):
        img = np.zeros((9, 9, 3), dtype=np.uint8)
        img[3:6, 3:6] = 200
        base = magic_wand_mask(img, (4, 4), 0, 0)
        expanded = magic_wand_mask(img, (4, 4), 0, 1)
        shrunk = magic_wand_mask(img, (4, 4), 0, -1)
        self.assertGreater(int(expanded.sum()), int(base.sum()))
        self.assertLess(int(shrunk.sum()), int(base.sum()))

    def test_out_of_bounds_seed_returns_empty_mask(self):
        img = np.zeros((3, 3, 3), dtype=np.uint8)
        mask = magic_wand_mask(img, (9, 9), 32, 0)
        self.assertEqual(mask.shape, (3, 3))
        self.assertEqual(int(mask.max()), 0)

    def test_rgba_uses_rgb_channels(self):
        img = np.zeros((4, 4, 4), dtype=np.uint8)
        img[1:3, 1:3, :3] = 255
        img[..., 3] = 128
        mask = magic_wand_mask(img, (1, 1), 0, 0)
        self.assertEqual(int(mask[1, 1]), 255)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_interior_is_holes_inside_a_ring(self):
        img = np.zeros((9, 9, 3), dtype=np.uint8)
        img[2:7, 2:7] = 255
        img[3:6, 3:6] = 0
        selection = magic_wand_mask(
            img, (2, 2), 0, 0, MagicWandFillMode.Selection
        )
        interior = magic_wand_mask(
            img, (2, 2), 0, 0, MagicWandFillMode.Interior
        )
        both = magic_wand_mask(
            img, (2, 2), 0, 0, MagicWandFillMode.SelectionAndInterior
        )
        self.assertEqual(int(selection[2, 2]), 255)
        self.assertEqual(int(selection[4, 4]), 0)
        self.assertEqual(int(interior[2, 2]), 0)
        self.assertEqual(int(interior[4, 4]), 255)
        self.assertEqual(int(both[2, 2]), 255)
        self.assertEqual(int(both[4, 4]), 255)

    def test_solid_blob_has_no_interior(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[2:6, 2:6] = 255
        interior = magic_wand_mask(
            img, (3, 3), 0, 0, MagicWandFillMode.Interior
        )
        self.assertEqual(int(interior.max()), 0)

    def test_range_applies_after_fill_mode(self):
        img = np.zeros((11, 11, 3), dtype=np.uint8)
        img[2:9, 2:9] = 255
        img[4:7, 4:7] = 0
        interior = magic_wand_mask(
            img, (2, 2), 0, 0, MagicWandFillMode.Interior
        )
        expanded = magic_wand_mask(
            img, (2, 2), 0, 1, MagicWandFillMode.Interior
        )
        self.assertEqual(int(interior[4, 4]), 255)
        self.assertEqual(int(interior[3, 4]), 0)
        self.assertEqual(int(expanded[3, 4]), 255)


class MagicWandPreviewOverlayTests(unittest.TestCase):

    def test_preview_uses_pale_blue_on_selected_pixels(self):
        from ballontranslator.utils.imgproc_utils import magic_wand_preview_overlay

        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 2:4] = 255
        overlay, x, y = magic_wand_preview_overlay(mask)
        self.assertEqual((x, y), (2, 1))
        self.assertEqual(overlay.shape, (2, 2, 4))
        self.assertEqual(tuple(overlay[0, 0].tolist()), (150, 210, 255, 96))
        self.assertEqual(tuple(overlay[0, 1].tolist()), (150, 210, 255, 96))

    def test_empty_mask_has_no_preview(self):
        from ballontranslator.utils.imgproc_utils import magic_wand_preview_overlay

        self.assertIsNone(magic_wand_preview_overlay(np.zeros((3, 3), dtype=np.uint8)))


class DrawPanelMagicWandConfigTests(unittest.TestCase):

    def test_invalid_shape_and_range_are_clamped(self):
        cfg = DrawPanelConfig(
            inpainter_shape=9,
            magicwand_tolerance=400,
            magicwand_range=-80,
            magicwand_fill_mode=9,
        )
        self.assertEqual(cfg.inpainter_shape, 0)
        self.assertEqual(cfg.magicwand_tolerance, 255)
        self.assertEqual(cfg.magicwand_range, -50)
        self.assertEqual(cfg.magicwand_fill_mode, 0)


class InpaintMagicWandPanelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        cls.app = app

    def test_magic_wand_hides_thickness_and_shows_sliders(self):
        from ballontranslator.ui.drawingpanel import InpaintPanel

        panel = InpaintPanel()
        self.assertTrue(panel.thickness_row.isVisibleTo(panel))
        self.assertFalse(panel.tolerance_row.isVisibleTo(panel))
        self.assertFalse(panel.range_row.isVisibleTo(panel))
        self.assertFalse(panel.fill_mode_row.isVisibleTo(panel))

        panel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        panel.sync_shape_controls()
        self.assertFalse(panel.thickness_row.isVisibleTo(panel))
        self.assertTrue(panel.tolerance_row.isVisibleTo(panel))
        self.assertTrue(panel.range_row.isVisibleTo(panel))
        self.assertTrue(panel.fill_mode_row.isVisibleTo(panel))
        self.assertEqual(panel.fillModeCombobox.count(), 3)
        self.assertEqual(panel.toleranceSlider.value(), 32)
        self.assertEqual(panel.rangeSlider.value(), 0)
        self.assertEqual(panel.shapeCombobox.itemText(0), panel.tr('Circle Brush'))
        self.assertEqual(panel.shapeCombobox.itemText(1), panel.tr('Rectangle Brush'))
        self.assertEqual(panel.shapeCombobox.itemText(2), panel.tr('Magic Wand'))
        self.assertEqual(panel.fill_mode, MagicWandFillMode.Selection)
        panel.set_fill_mode(MagicWandFillMode.Interior)
        self.assertEqual(panel.fill_mode, MagicWandFillMode.Interior)
        panel.deleteLater()

    def test_magic_wand_cursor_is_a_custom_bitmap(self):
        from qtpy.QtCore import Qt

        from ballontranslator.ui.cursor import magic_wand_cursor

        cursor = magic_wand_cursor()
        self.assertEqual(cursor.shape(), Qt.CursorShape.BitmapCursor)
        self.assertFalse(cursor.pixmap().isNull())
        self.assertEqual(cursor.hotSpot().x(), 7)
        self.assertEqual(cursor.hotSpot().y(), 7)

    def test_leaving_magic_wand_keeps_viewport_mouse_tracking(self):
        from ballontranslator.ui.canvas import Canvas

        canvas = Canvas()
        viewport = canvas.gv.viewport()
        viewport.setMouseTracking(True)
        canvas.set_magic_wand_hover_tracking(True)
        self.assertTrue(canvas._magic_wand_hover_enabled)
        canvas.set_magic_wand_hover_tracking(False)
        self.assertFalse(canvas._magic_wand_hover_enabled)
        self.assertTrue(viewport.hasMouseTracking())
        canvas.deleteLater()

    def test_magic_wand_inpaint_restores_paint_mode(self):
        from ballontranslator.ui.canvas import Canvas
        from ballontranslator.ui.drawingpanel import DrawingPanel
        from ballontranslator.ui.image_edit import ImageEditMode

        canvas = Canvas()
        panel = DrawingPanel(canvas)
        panel.inpaintTool.setChecked(True)
        panel.on_use_inpainttool()
        self.assertEqual(canvas.image_edit_mode, ImageEditMode.InpaintTool)

        canvas.image_edit_mode = ImageEditMode.NONE
        panel.clearInpaintItems()
        self.assertEqual(canvas.image_edit_mode, ImageEditMode.InpaintTool)

        canvas.image_edit_mode = ImageEditMode.NONE
        panel.setInpaintShape(PenShape.Rectangle)
        self.assertEqual(canvas.image_edit_mode, ImageEditMode.InpaintTool)
        self.assertEqual(canvas.painting_shape, PenShape.Rectangle)

        panel.deleteLater()
        canvas.deleteLater()

    def test_magic_wand_hover_emits_on_viewport_move(self):
        from qtpy.QtCore import QEvent, QPointF, QRectF, Qt
        from qtpy.QtGui import QMouseEvent
        from qtpy.QtTest import QSignalSpy
        from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget

        from ballontranslator.ui.canvas import Canvas
        from ballontranslator.ui.drawingpanel import DrawingPanel

        canvas = Canvas()
        panel = DrawingPanel(canvas)
        window = QWidget()
        layout = QHBoxLayout(window)
        layout.addWidget(canvas.gv)
        layout.addWidget(panel)
        canvas.setSceneRect(0, 0, 300, 300)
        canvas.baseLayer.setRect(QRectF(0, 0, 300, 300))
        window.resize(700, 400)
        window.show()
        self.app.processEvents()

        panel.inpaintTool.setChecked(True)
        panel.on_use_inpainttool()
        panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        self.app.processEvents()
        self.assertTrue(canvas._magic_wand_hover_enabled)

        spy = QSignalSpy(canvas.magic_wand_hover)
        pos = canvas.gv.mapFromScene(QPointF(80, 80))
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(pos),
            QPointF(pos),
            QPointF(canvas.gv.viewport().mapToGlobal(pos)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(canvas.gv.viewport(), move)
        self.app.processEvents()
        self.assertGreater(len(spy), 0)

        window.close()
        panel.deleteLater()
        canvas.deleteLater()


if __name__ == '__main__':
    unittest.main()
