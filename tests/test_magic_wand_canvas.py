import gc
import os
import threading
import time
import unittest
import weakref
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QWidget

from ballontranslator.ui import drawingpanel
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawing_commands import InpaintUndoCommand
from ballontranslator.ui.image_edit import ImageEditMode, PenShape
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class MagicWandCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel.copy()
        self.canvas = Canvas()
        self.project = ProjImgTrans()
        self.project.img_array = np.full((100, 100, 3), 240, np.uint8)
        self.project.inpainted_array = np.full((100, 100, 3), 180, np.uint8)
        self.project.inpainted_array[20:80, 20:80] = 60
        self.project.inpainted_array[40:60, 40:60] = 180
        self.project.mask_array = np.full((100, 100), 255, np.uint8)
        self.canvas.imgtrans_proj = self.project
        self.panel = drawingpanel.DrawingPanel(self.canvas)
        self.panel.module_manager = Mock()
        self.window = QWidget()
        layout = QHBoxLayout(self.window)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.window.resize(800, 500)
        self.window.show()
        self.canvas.updateCanvas()
        self.canvas.setPaintMode(True)
        self.panel.setCurrentToolByName('inpaint')
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        self.app.processEvents()

    def tearDown(self) -> None:
        if self.window is not None:
            self.window.close()
            self.window.deleteLater()
            self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config

    def click(self, button=Qt.MouseButton.LeftButton, modifiers=Qt.KeyboardModifier.NoModifier) -> None:
        QTest.mouseClick(
            self.canvas.gv.viewport(), button, modifiers,
            self.canvas.gv.mapFromScene(QPointF(25, 25)),
        )

    def wait_for_preview(self) -> None:
        deadline = time.monotonic() + 3
        while (
            not self.panel.magic_wand_preview_item.isVisible()
            or self.panel.magic_wand_preview_item.pixmap().isNull()
        ):
            self.assertLess(time.monotonic(), deadline, 'Preview did not complete.')
            QTest.qWait(10)

    def test_preview_is_excluded_from_export_and_tracks_undo(self) -> None:
        before = pixmap2ndarray(self.canvas.render_result_img())
        self.panel.on_magic_wand_hover(QPointF(25, 25))
        self.wait_for_preview()
        np.testing.assert_array_equal(before, pixmap2ndarray(self.canvas.render_result_img()))

        self.canvas.push_undo_command(InpaintUndoCommand(
            self.canvas, np.full((100, 100, 3), 240, np.uint8),
            self.project.mask_array.copy(), [0, 0, 100, 100],
        ))
        self.assertFalse(self.panel.magic_wand_preview_item.isVisible())
        self.wait_for_preview()
        overlay = pixmap2ndarray(self.panel.magic_wand_preview_item.pixmap())
        self.assertEqual(np.count_nonzero(overlay[..., 3]), 10000)
        self.canvas.draw_undo_stack.undo()
        self.assertFalse(self.panel.magic_wand_preview_item.isVisible())
        self.wait_for_preview()
        overlay = pixmap2ndarray(self.panel.magic_wand_preview_item.pixmap())
        self.assertEqual(np.count_nonzero(overlay[..., 3]), 3200)

    def test_restore_preserves_unselected_pixels_and_mask_through_undo(self) -> None:
        before = self.project.inpainted_array.copy()
        expected_mask = self.project.mask_array.copy()
        expected_mask[20:80, 20:80] = 0
        expected_mask[40:60, 40:60] = 255
        expected_image = before.copy()
        expected_image[expected_mask == 0] = 240
        self.click(Qt.MouseButton.RightButton)
        np.testing.assert_array_equal(self.project.mask_array, expected_mask)
        np.testing.assert_array_equal(self.project.inpainted_array, expected_image)
        self.canvas.draw_undo_stack.undo()
        np.testing.assert_array_equal(self.project.inpainted_array, before)
        self.assertTrue(np.all(self.project.mask_array == 255))
        self.canvas.draw_undo_stack.redo()
        np.testing.assert_array_equal(self.project.mask_array, expected_mask)
        np.testing.assert_array_equal(self.project.inpainted_array, expected_image)

    def test_pending_and_hidden_tools_do_not_submit_edits(self) -> None:
        submit = self.panel.module_manager.canvas_inpaint
        self.click()
        request = submit.call_args.args[0]
        submit.reset_mock()
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Rectangle)
        self.click()
        submit.assert_not_called()
        self.panel.on_inpaint_finished({**request, 'inpainted': request['img'].copy()})
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.InpaintTool)
        self.canvas.setPaintMode(False)
        self.panel.hide()
        self.click()
        submit.assert_not_called()
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.NONE)

        self.panel.show()
        self.panel.setCurrentToolByName('rect')
        self.canvas.image_edit_mode = ImageEditMode.NONE
        self.panel.on_rect_deletebtn_clicked()
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.RectTool)
        self.panel.setCurrentToolByName('hand')
        self.panel.on_inpaint_failed()
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.HandTool)

    def test_alt_click_and_switching_a_ctrl_stroke_to_wand(self) -> None:
        submit = self.panel.module_manager.canvas_inpaint
        self.click(modifiers=Qt.KeyboardModifier.AltModifier)
        submit.assert_not_called()
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Circle)
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.click(modifiers=Qt.KeyboardModifier.ControlModifier)
        self.assertIsNotNone(self.panel.inpaint_stroke)
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        submit.assert_not_called()
        self.assertIsNone(self.canvas.stroke_img_item)
        self.click()
        self.assertIsNotNone(submit.call_args)

    def test_pending_preview_keeps_keys_responsive_and_discards_old_image(self) -> None:
        release = threading.Event()
        calculate = drawingpanel._magic_wand_preview

        def held_preview(img: np.ndarray, seed: tuple) -> tuple:
            if not release.wait(2):
                raise TimeoutError('GUI did not release the worker.')
            return calculate(img, seed)

        with patch.object(drawingpanel, '_magic_wand_preview', held_preview):
            try:
                self.panel.on_magic_wand_hover(QPointF(25, 25))
                self.panel._apply_magic_wand_hover_preview()
                self.canvas.gv.setFocus()
                before = self.canvas.order_badges_visible
                QTest.keyClick(self.canvas.gv, Qt.Key.Key_N)
                self.assertNotEqual(before, self.canvas.order_badges_visible)
                self.project.inpainted_array[:] = 240
                self.canvas.updateLayers()
            finally:
                release.set()
            self.wait_for_preview()
        overlay = pixmap2ndarray(self.panel.magic_wand_preview_item.pixmap())
        self.assertEqual(np.count_nonzero(overlay[..., 3]), 10000)

    def test_destroying_panel_releases_wrapper_while_worker_is_pending(self) -> None:
        release = threading.Event()

        def held_preview(img: np.ndarray, seed: tuple) -> None:
            release.wait(2)

        with patch.object(drawingpanel, '_magic_wand_preview', held_preview):
            try:
                self.panel.on_magic_wand_hover(QPointF(25, 25))
                self.panel._apply_magic_wand_hover_preview()
                panel_ref = weakref.ref(self.panel)
                self.window.close()
                self.window.deleteLater()
                self.canvas.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.window = self.panel = self.canvas = None
                gc.collect()
                self.assertIsNone(panel_ref())
            finally:
                release.set()


if __name__ == '__main__':
    unittest.main()
