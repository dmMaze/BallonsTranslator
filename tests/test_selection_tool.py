import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt
from qtpy.QtGui import QKeySequence, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QHBoxLayout, QShortcut, QWidget

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class _SelectionWindow(QWidget):
    shortcutEscape = MainWindow.shortcutEscape


class SelectionToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel
        pcfg.drawpanel = DrawPanelConfig(rectool_method=2, recttool_dilate_ksize=0)
        self.canvas = Canvas()
        self.project = ProjImgTrans()
        self.project.img_array = np.full((100, 100, 3), 10, np.uint8)
        self.project.inpainted_array = np.full((100, 100, 3), 80, np.uint8)
        self.project.mask_array = np.full((100, 100), 255, np.uint8)
        self.canvas.imgtrans_proj = self.project
        self.panel = DrawingPanel(self.canvas)
        self.panel.module_manager = Mock()
        self.panel.set_config(pcfg.drawpanel)
        self.window = _SelectionWindow()
        self.window.canvas = self.canvas
        self.escape_shortcut = QShortcut(QKeySequence('Escape'), self.window)
        self.escape_shortcut.activated.connect(self.window.shortcutEscape)
        layout = QHBoxLayout(self.window)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        self.window.resize(900, 600)
        self.window.show()
        self.canvas.updateCanvas()
        self.canvas.setPaintMode(True)
        self.panel.rectTool.click()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config

    def pointer(self, x: float, y: float):
        return self.canvas.gv.mapFromScene(self.canvas.baseLayer.mapToScene(QPointF(x, y)))

    def move(self, x: float, y: float, button: Qt.MouseButton = Qt.MouseButton.NoButton) -> None:
        viewport = self.canvas.gv.viewport()
        pos = self.pointer(x, y)
        event = QMouseEvent(QEvent.Type.MouseMove, QPointF(pos), QPointF(viewport.mapToGlobal(pos)),
                            Qt.MouseButton.NoButton, button, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, event)

    def shape(self, shape: str) -> None:
        combo = self.panel.rectPanel.shapeComboBox
        combo.setCurrentIndex(combo.findData(shape))
        self.assertEqual(pcfg.drawpanel.selection_shape, shape)

    def select(self, shape: str, button: Qt.MouseButton = Qt.MouseButton.LeftButton) -> None:
        self.shape(shape)
        viewport = self.canvas.gv.viewport()
        if shape == 'polygon':
            for x, y in ((20, 20), (80, 20), (20, 80)):
                QTest.mouseClick(viewport, button, pos=self.pointer(x, y))
            QTest.keyClick(self.canvas.gv, Qt.Key.Key_Return)
        else:
            QTest.mousePress(viewport, button, pos=self.pointer(20, 20))
            if shape == 'lasso':
                self.move(80, 20, button)
                self.move(20, 80, button)
            else:
                self.move(80, 80, button)
            end = (20, 80) if shape == 'lasso' else (80, 80)
            QTest.mouseRelease(viewport, button, pos=self.pointer(*end))

    def test_all_shapes_stage_page_coordinates_at_different_zoom(self) -> None:
        for zoom in (0.5, 1.0, 2.0):
            self.canvas.scaleImage(zoom / self.canvas.scale_factor)
            for shape in ('rectangle', 'lasso', 'polygon'):
                with self.subTest(zoom=zoom, shape=shape):
                    self.panel._finish_inpaint()
                    self.select(shape)
                    request = self.panel.rect_inpaint_dict
                    self.assertIsNotNone(request)
                    self.assertEqual(request['inpaint_rect'], [20, 20, 80, 80])
                    self.assertEqual(request['mask'][5, 5], 255)
                    self.assertEqual(request['mask'][55, 55], 255 if shape == 'rectangle' else 0)
                    np.testing.assert_array_equal(request['mask'], request['selection_mask'])
                    self.assertFalse(self.canvas.selection_preview.isVisible())
                    self.assertFalse(self.canvas.creating_textblock)

    def test_methods_share_dilation_and_clip_both_sides_of_morphology(self) -> None:
        self.project.inpainted_array[35:45, 35:45] = 0
        for shape in ('rectangle', 'lasso', 'polygon'):
            for method in (0, 1, 2):
                with self.subTest(shape=shape, method=method):
                    self.panel._finish_inpaint()
                    controls = self.panel.rectPanel
                    controls.methodComboBox.setCurrentIndex(method)
                    controls.methodComboBox.activated.emit(method)
                    controls.dilate_slider.setValue(3)
                    self.select(shape)
                    request = self.panel.rect_inpaint_dict
                    self.assertIsNotNone(request)
                    self.assertFalse(np.any(request['mask'][request['selection_mask'] == 0]))

                    # Pre-morphology clipping is shared by all mask methods.
                    if method == 2 and shape != 'rectangle':
                        outside_seed = 255 - request['selection_mask']
                        algorithm = Mock(return_value=(outside_seed, outside_seed,
                            {'need_inpaint': True, 'bground_rgb': [255, 255, 255]}))
                        with patch('ballontranslator.ui.drawingpanel.get_maskseg_method', return_value=algorithm):
                            self.panel._update_rect_mask()
                        self.assertFalse(np.any(request['mask']))

    def test_restore_and_undo_only_touch_selected_pixels(self) -> None:
        for shape in ('rectangle', 'lasso', 'polygon'):
            with self.subTest(shape=shape):
                self.panel._finish_inpaint()
                before_image = self.project.inpainted_array.copy()
                before_mask = self.project.mask_array.copy()
                self.select(shape, Qt.MouseButton.RightButton)
                self.assertEqual(self.project.inpainted_array[25, 25, 0], 10)
                self.assertEqual(self.project.mask_array[25, 25], 0)
                if shape != 'rectangle':
                    self.assertEqual(self.project.inpainted_array[75, 75, 0], 80)
                    self.assertEqual(self.project.mask_array[75, 75], 255)
                after_image = self.project.inpainted_array.copy()
                after_mask = self.project.mask_array.copy()
                self.canvas.undo()
                np.testing.assert_array_equal(self.project.inpainted_array, before_image)
                np.testing.assert_array_equal(self.project.mask_array, before_mask)
                self.canvas.redo()
                np.testing.assert_array_equal(self.project.inpainted_array, after_image)
                np.testing.assert_array_equal(self.project.mask_array, after_mask)
                self.canvas.undo()

    def test_unmasked_backend_result_is_composited_inside_selection(self) -> None:
        for shape in ('lasso', 'polygon'):
            with self.subTest(shape=shape):
                self.panel._finish_inpaint()
                self.panel.rectPanel.use_mask_checker.setChecked(False)
                self.select(shape)
                request = self.panel.rect_inpaint_dict
                self.panel.inpaintRect(request)
                self.assertFalse(self.panel.rectPanel.shapeComboBox.isEnabled())
                self.panel.on_inpaint_finished({**request, 'inpainted': np.full_like(request['img'], 150),
                                               'mask': np.full_like(request['mask'], 255)})
                self.assertEqual(self.project.inpainted_array[25, 25, 0], 150)
                self.assertEqual(self.project.inpainted_array[75, 75, 0], 80)
                self.assertTrue(self.panel.rectPanel.shapeComboBox.isEnabled())
                self.canvas.undo()
                np.testing.assert_array_equal(self.project.inpainted_array, 80)

    def test_erosion_preserves_existing_mask_outside_selection(self) -> None:
        self.panel.rectPanel.dilate_slider.setValue(-3)
        self.select('polygon')
        request = self.panel.rect_inpaint_dict
        selected = request['selection_mask'] > 0
        expected_mask = np.full((60, 60), 255, np.uint8)
        expected_mask[selected] = request['mask'][selected]
        self.panel.runInpaint(request)
        self.panel.on_inpaint_finished({**request, 'inpainted': np.full_like(request['img'], 150)})
        np.testing.assert_array_equal(self.project.mask_array[20:80, 20:80], expected_mask)
        removed = selected & (expected_mask == 0)
        self.assertTrue(np.any(removed))
        np.testing.assert_array_equal(self.project.inpainted_array[20:80, 20:80][removed], 10)
        self.assertEqual(self.project.inpainted_array[75, 75, 0], 80)
        self.canvas.undo()
        np.testing.assert_array_equal(self.project.inpainted_array, 80)
        np.testing.assert_array_equal(self.project.mask_array, 255)

    def test_empty_erosion_restores_selected_mask_without_inference(self) -> None:
        self.project.mask_array.fill(0)
        self.project.mask_array[25, 25] = self.project.mask_array[75, 75] = 255
        self.panel.rectPanel.dilate_slider.setValue(-3)
        self.select('lasso')
        request = self.panel.rect_inpaint_dict
        self.assertFalse(np.any(request['mask']))
        self.panel.inpaintRect(request)
        self.panel.module_manager.canvas_inpaint.assert_not_called()
        self.assertEqual(self.project.inpainted_array[25, 25, 0], 10)
        self.assertEqual(self.project.mask_array[25, 25], 0)
        self.assertEqual(self.project.inpainted_array[75, 75, 0], 80)
        self.assertEqual(self.project.mask_array[75, 75], 255)

    def test_polygon_backspace_double_click_and_close_at_first_vertex(self) -> None:
        for finish in ('double-click', 'first-vertex'):
            with self.subTest(finish=finish):
                self.panel._finish_inpaint()
                self.shape('polygon')
                viewport = self.canvas.gv.viewport()
                for x, y in ((20, 20), (80, 20), (80, 80)):
                    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(x, y))
                QTest.keyClick(self.canvas.gv, Qt.Key.Key_Backspace)
                self.assertIsNone(self.panel.rect_inpaint_dict)
                if finish == 'double-click':
                    QTest.mouseDClick(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(20, 80))
                else:
                    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(20, 80))
                    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(20, 20))
                self.assertIsNotNone(self.panel.rect_inpaint_dict)
                self.assertEqual(self.panel.rect_inpaint_dict['mask'][55, 55], 0)

    def test_foreign_button_release_does_not_commit_selection(self) -> None:
        for shape in ('rectangle', 'lasso'):
            self.panel._finish_inpaint()
            self.shape(shape)
            viewport = self.canvas.gv.viewport()
            QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(20, 20))
            self.move(80, 20, Qt.MouseButton.LeftButton)
            self.move(20, 80, Qt.MouseButton.LeftButton)
            QTest.mouseRelease(viewport, Qt.MouseButton.RightButton, pos=self.pointer(20, 80))
            self.assertIsNone(self.panel.rect_inpaint_dict)
            QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=self.pointer(80, 80))
            self.assertIsNotNone(self.panel.rect_inpaint_dict)

    def test_cancellation_boundaries_and_export_exclude_draft(self) -> None:
        boundaries = (
            lambda: QTest.keyClick(self.canvas.gv, Qt.Key.Key_Escape),
            lambda: self.panel.setCurrentToolByName('pen'),
            self.panel.handle_page_changed, self.canvas.updateCanvas, self.canvas.undo, self.canvas.redo,
            self.canvas.on_activation_changed, self.canvas.on_hide_canvas,
            lambda: self.shape('rectangle'), lambda: self.canvas.setPaintMode(False),
        )
        for cancel in boundaries:
            self.canvas.setPaintMode(True)
            self.panel.rectTool.setChecked(True)
            self.panel.on_use_recttool()
            self.shape('polygon')
            before = pixmap2ndarray(self.canvas.render_result_img())
            for x, y in ((20, 20), (80, 20), (20, 80)):
                QTest.mouseClick(self.canvas.gv.viewport(), Qt.MouseButton.LeftButton, pos=self.pointer(x, y))
            self.assertTrue(self.canvas.selection_preview.isVisible())
            np.testing.assert_array_equal(pixmap2ndarray(self.canvas.render_result_img()), before)
            cancel()
            self.assertFalse(self.canvas.selection_preview.isVisible())
            QTest.keyClick(self.canvas.gv, Qt.Key.Key_Return)
            self.assertIsNone(self.panel.rect_inpaint_dict)
            np.testing.assert_array_equal(self.project.inpainted_array, 80)
            np.testing.assert_array_equal(self.project.mask_array, 255)


if __name__ == '__main__':
    unittest.main()
