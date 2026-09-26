import json
import os
import unittest
from dataclasses import asdict
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt
from qtpy.QtGui import QColor, QKeySequence, QMouseEvent
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QShortcut, QWidget

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.image_edit import ImageEditMode
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class _CanvasWindow(QWidget):
    shortcutEscape = MainWindow.shortcutEscape


class ShapeFillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel
        pcfg.drawpanel = DrawPanelConfig(shape_fill_color='#e02846')
        self.canvas = Canvas()
        self.project = ProjImgTrans()
        self.project.img_array = np.full((100, 100, 3), 240, np.uint8)
        self.project.inpainted_array = np.full((100, 100, 3), 180, np.uint8)
        self.project.mask_array = np.full((100, 100), 127, np.uint8)
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
        self.window.resize(800, 500)
        self.window.show()
        self.canvas.updateCanvas()
        self.canvas.setPaintMode(True)
        self.panel.setCurrentToolByName('shape')
        self.app.processEvents()

    def tearDown(self) -> None:
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.window.close()
        self.window.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config

    def move_pointer(self, pos: QPointF, button: Qt.MouseButton) -> None:
        viewport = self.canvas.gv.viewport()
        pos = self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(pos))
        move = QMouseEvent(
            QEvent.Type.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
            Qt.MouseButton.NoButton, button,
            Qt.KeyboardModifier.NoModifier,
        )
        self.app.sendEvent(viewport, move)

    def drag(
        self, start: QPointF, end: QPointF, release: bool = True,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> None:
        viewport = self.canvas.gv.viewport()
        start = self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(start))
        QTest.mousePress(viewport, button, pos=start)
        self.move_pointer(end, button)
        if release:
            end = self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(end))
            QTest.mouseRelease(viewport, button, pos=end)

    def rendered(self) -> np.ndarray:
        return pixmap2ndarray(self.canvas.render_result_img())

    def test_rectangle_drag_undo_redo_and_source_preservation(self) -> None:
        before = self.rendered()
        self.canvas.scaleImage(2)
        self.drag(QPointF(80, 60), QPointF(20, 20))
        expected = before.copy()
        expected[20:60, 20:80] = (224, 40, 70, 255)
        np.testing.assert_array_equal(self.rendered(), expected)
        self.assertTrue(self.canvas.projstate_unsaved)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        self.assertEqual(self.canvas.text_undo_stack.count(), 0)
        self.canvas.undo()
        np.testing.assert_array_equal(self.rendered(), before)
        self.canvas.redo()
        np.testing.assert_array_equal(self.rendered(), expected)
        self.assertTrue(np.all(self.project.img_array == 240))
        self.assertTrue(np.all(self.project.inpainted_array == 180))
        self.assertTrue(np.all(self.project.mask_array == 127))

    def test_ellipse_and_clipping_keep_original_shape(self) -> None:
        self.panel.shapePanel.shapeCombobox.setCurrentIndex(1)
        self.drag(QPointF(20, 20), QPointF(80, 80))
        pixels = self.rendered()
        np.testing.assert_array_equal(pixels[50, 50], (224, 40, 70, 255))
        np.testing.assert_array_equal(pixels[20, 20], (180, 180, 180, 255))
        self.canvas.undo()
        self.drag(QPointF(-40, 20), QPointF(40, 80))
        pixels = self.rendered()
        # This is the right half of the original ellipse, not a squeezed one.
        np.testing.assert_array_equal(pixels[50, 0], (224, 40, 70, 255))
        np.testing.assert_array_equal(pixels[20, 39], (180, 180, 180, 255))
        np.testing.assert_array_equal(pixels[50, 40], (180, 180, 180, 255))
        self.canvas.undo()
        self.panel.shapePanel.shapeCombobox.setCurrentIndex(0)
        self.drag(QPointF(80, 80), QPointF(120, 120))
        np.testing.assert_array_equal(self.rendered()[99, 99], (224, 40, 70, 255))

    def test_preview_is_excluded_from_export_and_cancelled_on_state_changes(self) -> None:
        before = self.rendered()
        escape_activations = QSignalSpy(self.escape_shortcut.activated)
        for cancel in (
            lambda: QTest.keyClick(self.canvas.gv, Qt.Key.Key_Escape),
            lambda: self.panel.setCurrentToolByName('pen'),
            self.panel.hide,
            self.canvas.updateCanvas,
            lambda: self.canvas.setPaintMode(False),
            self.canvas.clearToolStates,
            self.canvas.on_activation_changed,
        ):
            self.panel.show()
            self.canvas.setPaintMode(True)
            self.panel.setCurrentToolByName('shape')
            self.drag(QPointF(20, 20), QPointF(80, 80), release=False)
            self.assertTrue(self.canvas.shape_fill_preview.isVisible())
            np.testing.assert_array_equal(self.rendered(), before)
            cancel()
            self.assertFalse(self.canvas.shape_fill_preview.isVisible())
            QTest.mouseRelease(self.canvas.gv.viewport(), Qt.MouseButton.LeftButton)
            self.assertEqual(self.canvas.draw_undo_stack.count(), 0)
        self.assertEqual(len(escape_activations), 1)

    def test_other_mouse_buttons_do_not_finish_left_drag(self) -> None:
        self.drag(QPointF(20, 20), QPointF(60, 60), release=False)
        viewport = self.canvas.gv.viewport()
        pos = self.canvas.gv.mapFromScene(QPointF(60, 60))
        for button in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            QTest.mousePress(viewport, button, pos=pos)
            QTest.mouseRelease(viewport, button, pos=pos)
            self.assertTrue(self.canvas.shape_fill_preview.isVisible())
            self.assertEqual(self.canvas.draw_undo_stack.count(), 0)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=pos)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        np.testing.assert_array_equal(self.rendered()[40, 40], (224, 40, 70, 255))

    def test_undo_and_redo_cancel_pending_fill(self) -> None:
        self.drag(QPointF(20, 20), QPointF(40, 40))
        saved = self.rendered()
        for history in (self.canvas.undo, self.canvas.redo):
            self.drag(QPointF(60, 60), QPointF(80, 80), release=False)
            history()
            self.assertFalse(self.canvas.shape_fill_preview.isVisible())
            QTest.mouseRelease(self.canvas.gv.viewport(), Qt.MouseButton.LeftButton)
            self.assertEqual(self.canvas.draw_undo_stack.count(), 1)
        np.testing.assert_array_equal(self.rendered(), saved)

    def test_switching_held_brush_or_rect_to_fill_cancels_previous_gesture(self) -> None:
        for tool in ('pen', 'inpaint', 'rect'):
            self.panel.setCurrentToolByName(tool)
            self.drag(QPointF(10, 10), QPointF(20, 20), release=False)
            stroke = self.canvas.stroke_img_item
            self.panel.setCurrentToolByName('shape')
            if stroke is not None:
                self.assertFalse(stroke.painter.isActive())
            self.assertIsNone(self.canvas.stroke_img_item)
            QTest.mouseRelease(self.canvas.gv.viewport(), Qt.MouseButton.LeftButton)
            self.assertEqual(self.canvas.draw_undo_stack.count(), 0)
        self.drag(QPointF(20, 20), QPointF(80, 80))
        np.testing.assert_array_equal(self.rendered()[50, 50], (224, 40, 70, 255))

    def test_crosshair_allocation_is_independent_of_brush_width_and_zoom(self) -> None:
        normal = self.panel.get_pen_cursor(pen_size=30, draw_shape=False)
        self.canvas.scaleImage(10)
        large = self.panel.get_pen_cursor(pen_size=1000, draw_shape=False)
        self.assertEqual(normal.pixmap().size(), large.pixmap().size())

    def test_empty_outside_and_right_click_do_not_draw(self) -> None:
        before = self.rendered()
        self.drag(QPointF(20, 20), QPointF(20, 20))
        self.drag(QPointF(-30, -30), QPointF(-10, -10))
        QTest.mouseClick(self.canvas.gv.viewport(), Qt.MouseButton.RightButton)
        self.assertEqual(self.canvas.draw_undo_stack.count(), 0)
        np.testing.assert_array_equal(self.rendered(), before)

    def test_saved_drawing_follows_edits_before_repaint(self) -> None:
        self.drag(QPointF(20, 20), QPointF(80, 80))
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        np.testing.assert_array_equal(pixels[50, 50], (224, 40, 70, 255))
        self.canvas.undo()
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        self.assertFalse(np.any(pixels[..., 3]))
        self.canvas.redo()
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        np.testing.assert_array_equal(pixels[50, 50], (224, 40, 70, 255))

    def test_view_repaints_fill_and_history_without_export(self) -> None:
        viewport = self.canvas.gv.viewport()
        pos = self.canvas.gv.mapFromScene(QPointF(40, 40))
        before = viewport.grab().toImage().pixelColor(pos)
        self.drag(QPointF(20, 20), QPointF(80, 80))
        QTest.qWait(20)
        self.assertEqual(viewport.grab().toImage().pixelColor(pos), QColor('#e02846'))
        self.canvas.undo()
        QTest.qWait(20)
        self.assertEqual(viewport.grab().toImage().pixelColor(pos), before)
        self.canvas.redo()
        QTest.qWait(20)
        self.assertEqual(viewport.grab().toImage().pixelColor(pos), QColor('#e02846'))

    def test_switching_from_pending_brush_discards_inpaint_preview(self) -> None:
        self.panel.setCurrentToolByName('inpaint')
        QTest.keyPress(self.canvas.gv, Qt.Key.Key_Control)
        pos = self.canvas.gv.mapFromScene(QPointF(50, 50))
        QTest.mouseClick(
            self.canvas.gv.viewport(), Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier, pos,
        )
        self.assertIsNotNone(self.panel.inpaint_stroke)
        self.panel.setCurrentToolByName('shape')
        QTest.keyRelease(self.canvas.gv, Qt.Key.Key_Control)
        self.assertIsNone(self.panel.inpaint_stroke)
        self.assertIsNone(self.canvas.stroke_img_item)
        self.drag(QPointF(20, 20), QPointF(80, 80))
        np.testing.assert_array_equal(self.rendered()[50, 50], (224, 40, 70, 255))

    def test_pen_eraser_removes_fill_and_undo_restores_it(self) -> None:
        self.drag(QPointF(20, 20), QPointF(80, 80))
        before = self.rendered()
        self.panel.setCurrentToolByName('pen')
        QTest.mouseClick(
            self.canvas.gv.viewport(), Qt.MouseButton.RightButton,
            pos=self.canvas.gv.mapFromScene(QPointF(50, 50)),
        )
        np.testing.assert_array_equal(self.rendered()[50, 50], (180, 180, 180, 255))
        self.canvas.undo()
        np.testing.assert_array_equal(self.rendered(), before)

    def test_saved_drawing_tracks_live_eraser_and_page_replacement(self) -> None:
        self.drag(QPointF(20, 20), QPointF(80, 80))
        before = self.rendered()
        self.panel.setCurrentToolByName('pen')
        self.panel.setPenToolWidth(10)
        self.drag(
            QPointF(30, 30), QPointF(40, 40), release=False,
            button=Qt.MouseButton.RightButton,
        )
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        self.assertEqual(pixels[40, 40, 3], 0)
        self.assertEqual(pixels[70, 70, 3], 255)
        self.move_pointer(QPointF(70, 70), Qt.MouseButton.RightButton)
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        self.assertEqual(pixels[70, 70, 3], 0)
        self.canvas.clear_states()
        QTest.mouseRelease(self.canvas.gv.viewport(), Qt.MouseButton.RightButton)
        np.testing.assert_array_equal(self.rendered(), before)
        self.canvas.setDrawingLayer(np.full((100, 100, 4), (1, 2, 3, 255), np.uint8))
        pixels = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        np.testing.assert_array_equal(pixels[50, 50], (1, 2, 3, 255))

    def test_color_picker_and_saved_tool_settings(self) -> None:
        picker = self.panel.shapePanel.colorPicker
        with patch('qtpy.QtWidgets.QColorDialog.getColor', return_value=QColor('#12ab34')):
            QTest.mouseClick(picker, Qt.MouseButton.LeftButton)
        self.panel.shapePanel.shapeCombobox.setCurrentIndex(1)
        config = DrawPanelConfig(**json.loads(json.dumps(asdict(pcfg.drawpanel))))
        self.assertEqual(config.shape_fill_color, '#12ab34')
        self.assertEqual(config.shape_fill_shape, 'ellipse')
        self.assertEqual(config.current_tool, ImageEditMode.ShapeFillTool)
        with patch('qtpy.QtWidgets.QColorDialog.getColor', return_value=QColor()):
            QTest.mouseClick(picker, Qt.MouseButton.LeftButton)
        self.assertEqual(pcfg.drawpanel.shape_fill_color, '#12ab34')
        self.panel.setCurrentToolByName('rect')
        self.panel.set_config(config)
        self.assertEqual(self.canvas.image_edit_mode, ImageEditMode.ShapeFillTool)
        self.drag(QPointF(20, 20), QPointF(80, 80))
        np.testing.assert_array_equal(self.rendered()[50, 50], (18, 171, 52, 255))

    def test_old_and_invalid_config_preserve_other_settings(self) -> None:
        old = DrawPanelConfig(pentool_width=42)
        self.assertEqual((old.shape_fill_shape, old.shape_fill_color), ('rectangle', '#ffffff'))
        for shape, color in ((None, None), ([], []), ('triangle', '#12345g')):
            with self.assertLogs('BallonTranslator', level='WARNING'):
                config = DrawPanelConfig(
                    shape_fill_shape=shape, shape_fill_color=color,
                    pentool_width=42,
                )
            self.assertEqual(config.shape_fill_shape, 'rectangle')
            self.assertEqual(config.shape_fill_color, '#ffffff')
            self.assertEqual(config.pentool_width, 42)


if __name__ == '__main__':
    unittest.main()
