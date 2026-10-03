import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QKeySequence, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QShortcut, QWidget

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.image_edit import PenShape
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


SHIFT = Qt.KeyboardModifier.ShiftModifier
CTRL = Qt.KeyboardModifier.ControlModifier
LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton


class _CanvasWindow(QWidget):
    shortcutEscape = MainWindow.shortcutEscape


class BrushLineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel
        pcfg.drawpanel = DrawPanelConfig(pentool_color=[20, 40, 80, 128], pentool_width=6,
                                       inpainter_width=6)
        self.canvas = Canvas()
        self.project = ProjImgTrans()
        self.project.img_array = np.full((120, 120, 3), 240, np.uint8)
        self.project.inpainted_array = np.full((120, 120, 3), 180, np.uint8)
        self.project.mask_array = np.full((120, 120), 255, np.uint8)
        self.canvas.imgtrans_proj = self.project
        self.panel = DrawingPanel(self.canvas)
        self.panel.module_manager = Mock()
        self.panel.set_config(pcfg.drawpanel)
        self.window = _CanvasWindow()
        self.window.canvas = self.canvas
        self.escape_shortcut = QShortcut(QKeySequence('Escape'), self.window)
        self.escape_shortcut.activated.connect(self.window.shortcutEscape)
        layout = QHBoxLayout(self.window)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.window.resize(850, 550)
        self.window.show()
        self.canvas.updateCanvas()
        self.canvas.setPaintMode(True)
        self.panel.setCurrentToolByName('pen')
        self.app.processEvents()

    def tearDown(self) -> None:
        self.canvas.clear_states()
        self.window.close()
        self.window.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config

    def view_pos(self, x: float, y: float) -> QPoint:
        return self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(QPointF(x, y)))

    def click(self, x: float, y: float, modifiers=Qt.KeyboardModifier.NoModifier, button=LEFT) -> None:
        QTest.mouseClick(self.canvas.gv.viewport(), button, modifiers, self.view_pos(x, y))

    def move(self, x: float, y: float, modifiers=Qt.KeyboardModifier.NoModifier,
             buttons=Qt.MouseButton.NoButton) -> None:
        viewport = self.canvas.gv.viewport()
        pos = self.view_pos(x, y)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
                            Qt.MouseButton.NoButton, buttons, modifiers)
        self.app.sendEvent(viewport, event)

    def pixels(self) -> np.ndarray:
        return pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())

    def test_shift_click_connects_only_from_last_stroke_with_undo(self) -> None:
        self.click(20, 20)
        first = self.pixels()
        self.click(80, 80, SHIFT)
        connected = self.pixels()
        self.assertEqual(connected[50, 50, 3], 128)
        self.assertEqual(connected[50, 30, 3], 0)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 2)
        self.canvas.undo()
        np.testing.assert_array_equal(self.pixels(), first)
        self.canvas.redo()
        np.testing.assert_array_equal(self.pixels(), connected)
        self.click(90, 20)
        self.assertEqual(self.pixels()[50, 90, 3], 0)

    def test_zoomed_pen_preview_matches_committed_drawing(self) -> None:
        viewport = self.canvas.gv.viewport()
        with patch.object(pcfg, 'original_transparency', 0), patch.object(pcfg, 'mask_transparency', 0):
            self.canvas.updateLayers()
            self.canvas.scaleImage(3)
            self.canvas.scaleFactorLabel.hide()
            QTest.mousePress(viewport, LEFT, pos=self.view_pos(30, 30))
            self.move(65, 65, buttons=LEFT)
            preview = pixmap2ndarray(viewport.grab())
            QTest.mouseRelease(viewport, LEFT, pos=self.view_pos(65, 65))
            committed = pixmap2ndarray(viewport.grab())
            np.testing.assert_allclose(preview, committed, atol=1)
            saved = self.pixels()
            self.canvas.undo()
            self.canvas.redo()
            np.testing.assert_array_equal(self.pixels(), saved)
            np.testing.assert_array_equal(pixmap2ndarray(viewport.grab()), committed)

    def test_ctrl_shift_chains_axis_aligned_square_at_zoom(self) -> None:
        self.canvas.scaleImage(2)
        self.click(20, 20)
        for x, y in ((80, 24), (76, 80), (20, 76), (24, 20)):
            self.click(x, y, CTRL | SHIFT)
        alpha = self.pixels()[..., 3]
        for x, y in ((50, 20), (80, 50), (50, 80), (20, 50)):
            self.assertEqual(alpha[y, x], 128)
        self.assertEqual(alpha[50, 50], 0)
        self.assertEqual(alpha[24, 50], 0)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 5)

    def test_rectangle_brush_segments_have_no_gaps_in_any_direction(self) -> None:
        self.panel.penConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Rectangle)
        for start, end in (((20, 20), (100, 100)), ((100, 20), (20, 100)),
                           ((100, 100), (20, 20)), ((20, 100), (100, 20))):
            self.canvas.setDrawingLayer()
            self.click(*start)
            self.click(*end, SHIFT)
            alpha = self.pixels()[..., 3]
            for fraction in np.linspace(0.1, 0.9, 9):
                x, y = np.rint(np.array(start) * (1 - fraction) + np.array(end) * fraction).astype(int)
                self.assertEqual(alpha[y, x], 128)
            self.assertEqual(alpha[10, 60], 0)

    def test_preview_tracks_modifiers_and_never_changes_export(self) -> None:
        self.click(20, 20)
        before = pixmap2ndarray(self.canvas.render_result_img())
        self.move(90, 50, SHIFT)
        self.assertTrue(self.canvas.brush_line_preview.isVisible())
        self.assertEqual(self.canvas.brush_line_preview.line().p2(), QPointF(90, 50))
        self.move(90, 50, SHIFT | CTRL)
        self.assertEqual(self.canvas.brush_line_preview.line().p2(), QPointF(90, 20))
        np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), before)
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertFalse(self.canvas.brush_line_preview.isVisible())
        self.move(90, 50, SHIFT)
        self.app.sendEvent(self.canvas.gv.viewport(), QEvent(QEvent.Type.Leave))
        self.assertFalse(self.canvas.brush_line_preview.isVisible())

    def test_stationary_pointer_preview_updates_on_modifier_keys(self) -> None:
        self.click(20, 20)
        QTest.mouseMove(self.canvas.gv.viewport(), self.view_pos(90, 50))
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertTrue(self.canvas.brush_line_preview.isVisible())
        self.assertEqual(self.canvas.brush_line_preview.line().p2(), QPointF(90, 50))
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control, SHIFT)
        self.assertEqual(self.canvas.brush_line_preview.line().p2(), QPointF(90, 20))
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control, SHIFT)
        self.assertEqual(self.canvas.brush_line_preview.line().p2(), QPointF(90, 50))
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertFalse(self.canvas.brush_line_preview.isVisible())

    def test_modifier_keys_outside_view_do_not_restore_line_preview(self) -> None:
        self.click(20, 20)
        QTest.mouseMove(self.panel, QPoint(10, 10))
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Shift)
        self.assertFalse(self.canvas.brush_line_preview.isVisible())
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Shift)

    def test_shift_line_starts_at_freehand_endpoint(self) -> None:
        viewport = self.canvas.gv.viewport()
        QTest.mousePress(viewport, LEFT, pos=self.view_pos(20, 20))
        self.move(60, 20, buttons=LEFT)
        QTest.mouseRelease(viewport, LEFT, pos=self.view_pos(60, 20))
        self.click(60, 80, SHIFT)
        self.assertEqual(self.pixels()[50, 60, 3], 128)
        self.assertEqual(self.pixels()[50, 40, 3], 0)

    def test_inpaint_anchor_survives_completion_and_submits_line_mask(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        self.click(20, 20)
        submit = self.panel.module_manager.canvas_inpaint
        first = submit.call_args.args[0]
        self.panel.on_inpaint_finished({**first, 'inpainted': first['img'].copy()})
        self.click(90, 90, SHIFT)
        request = submit.call_args.args[0]
        x, y, _, _ = request['inpaint_rect']
        self.assertEqual(request['mask'][50-y, 50-x], 255)
        self.assertEqual(request['mask'][50-y, 30-x], 0)
        self.assertEqual(submit.call_count, 2)
        self.assertFalse(np.any(self.pixels()[..., 3]))

    def test_inpaint_ctrl_accumulates_square_and_release_submits_once(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.Rectangle)
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.click(20, 20, CTRL)
        for x, y in ((80, 24), (76, 80), (20, 76), (24, 20)):
            self.click(x, y, CTRL | SHIFT)
        submit = self.panel.module_manager.canvas_inpaint
        submit.assert_not_called()
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.assertEqual(submit.call_count, 1)
        request = submit.call_args.args[0]
        x, y, _, _ = request['inpaint_rect']
        for px, py in ((50, 20), (80, 50), (50, 80), (20, 50)):
            self.assertEqual(request['mask'][py-y, px-x], 255)
        self.assertEqual(request['mask'][50-y, 50-x], 0)

    def test_releasing_ctrl_mid_segment_waits_for_mouse_release(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.click(20, 20, CTRL)
        viewport = self.canvas.gv.viewport()
        QTest.mousePress(viewport, LEFT, CTRL | SHIFT, self.view_pos(80, 25))
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        QTest.mouseRelease(viewport, LEFT, SHIFT, self.view_pos(80, 25))
        self.panel.module_manager.canvas_inpaint.assert_called_once()

    def test_switching_brush_tools_preserves_held_ctrl_accumulation(self) -> None:
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.panel.setCurrentToolByName('inpaint')
        self.click(20, 20, CTRL)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.click(80, 25, CTRL | SHIFT)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.panel.module_manager.canvas_inpaint.assert_called_once()

    def test_late_other_button_release_does_not_erase_pending_inpaint_mask(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        self.click(20, 20, CTRL)
        viewport = self.canvas.gv.viewport()
        pos = self.view_pos(80, 25)
        QTest.mousePress(viewport, LEFT, CTRL | SHIFT, pos)
        QTest.mousePress(viewport, RIGHT, CTRL | SHIFT, pos)
        QTest.mouseRelease(viewport, LEFT, CTRL | SHIFT, pos)
        QTest.mouseRelease(viewport, RIGHT, CTRL | SHIFT, pos)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 0)
        self.assertTrue(np.all(self.project.inpainted_array == 180))
        self.assertTrue(np.all(self.project.mask_array == 255))
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.panel.module_manager.canvas_inpaint.assert_called_once()

    def test_cancel_and_tool_page_history_changes_forget_anchor(self) -> None:
        for cancel in (lambda: QTest.keyClick(self.canvas.gv, Qt.Key.Key_Escape),
                       lambda: self.panel.setCurrentToolByName('shape'),
                       lambda: self.panel.setCurrentToolByName('inpaint'),
                       self.panel.hide, self.canvas.on_activation_changed,
                       self.canvas.updateCanvas, self.canvas.undo,
                       lambda: self.canvas.setPaintMode(False)):
            self.panel.show()
            self.canvas.setPaintMode(True)
            self.panel.setCurrentToolByName('pen')
            self.canvas.setDrawingLayer()
            self.click(20, 20)
            self.move(80, 80, SHIFT)
            cancel()
            self.assertFalse(self.canvas.brush_line_preview.isVisible())
            self.panel.show()
            self.canvas.setPaintMode(True)
            self.panel.setCurrentToolByName('pen')
            self.click(80, 80, SHIFT)
            self.assertEqual(self.pixels()[50, 50, 3], 0)

    def test_segment_ignores_drag_and_other_button_release_and_escape_discards_it(self) -> None:
        self.click(20, 20)
        before = self.pixels()
        viewport = self.canvas.gv.viewport()
        QTest.mousePress(viewport, LEFT, SHIFT, self.view_pos(80, 20))
        self.move(80, 80, SHIFT, LEFT)
        QTest.mousePress(viewport, RIGHT, SHIFT, self.view_pos(80, 80))
        QTest.mouseRelease(viewport, RIGHT, SHIFT, self.view_pos(80, 80))
        self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        QTest.keyClick(self.canvas.gv, Qt.Key.Key_Escape)
        QTest.mouseRelease(viewport, LEFT, SHIFT, self.view_pos(80, 80))
        self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        self.assertIsNone(self.canvas.stroke_img_item)
        np.testing.assert_array_equal(self.pixels(), before)

    def test_tool_switch_discards_held_line_and_pending_inpaint_without_late_commit(self) -> None:
        for source, target in (('pen', 'inpaint'), ('inpaint', 'pen'), ('pen', 'hand'),
                               ('inpaint', 'rect'), ('pen', 'shape')):
            self.panel.setCurrentToolByName(source)
            QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
            self.click(20, 20, CTRL)
            before = self.pixels()
            commands = self.canvas.draw_undo_stack.count()
            viewport = self.canvas.gv.viewport()
            QTest.mousePress(viewport, LEFT, CTRL | SHIFT, self.view_pos(80, 25))
            stroke = self.canvas.stroke_img_item
            self.panel.setCurrentToolByName(target)
            self.assertFalse(stroke.painter.isActive())
            self.assertIsNone(self.canvas.stroke_img_item)
            self.assertIsNone(self.panel.inpaint_stroke)
            QTest.mouseRelease(viewport, LEFT, CTRL | SHIFT, self.view_pos(80, 25))
            QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
            self.assertEqual(self.canvas.draw_undo_stack.count(), commands)
            np.testing.assert_array_equal(self.pixels(), before)
        self.panel.module_manager.canvas_inpaint.assert_not_called()

    def test_eraser_lines_do_not_connect_to_paint_anchor(self) -> None:
        self.panel.fill_shape(QRectF(0, 0, 120, 120))
        self.click(20, 20)
        self.click(90, 90, SHIFT, RIGHT)
        self.assertEqual(self.pixels()[50, 50, 3], 255)
        self.click(20, 90, CTRL | SHIFT, RIGHT)
        self.assertEqual(self.pixels()[90, 50, 3], 0)
        self.canvas.undo()
        self.assertEqual(self.pixels()[90, 50, 3], 255)

    def test_magic_wand_keeps_click_selection_and_clears_brush_anchor(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        self.click(20, 20, button=RIGHT)
        self.panel.inpaintConfigPanel.shapeCombobox.setCurrentIndex(PenShape.MagicWand)
        self.move(90, 90, SHIFT)
        self.assertFalse(self.canvas.brush_line_preview.isVisible())
        self.click(90, 90, SHIFT)
        self.panel.module_manager.canvas_inpaint.assert_called_once()


if __name__ == '__main__':
    unittest.main()
