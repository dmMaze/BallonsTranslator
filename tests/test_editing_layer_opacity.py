import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, QRectF
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.io_thread import ImgSaveThread
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import BottomBar
from ballontranslator.ui.misc import pixmap2ndarray
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.io_utils import imread, imwrite
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class EditingLayerOpacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.old_config = pcfg.drawpanel
        self.old_original = pcfg.original_transparency
        self.old_mask = pcfg.mask_transparency
        pcfg.drawpanel = DrawPanelConfig(shape_fill_color='#e02846')
        pcfg.original_transparency = pcfg.mask_transparency = 0
        self.window = QMainWindow()
        central = QWidget(self.window)
        self.window.setCentralWidget(central)
        self.canvas = Canvas()
        project = ProjImgTrans()
        project.img_array = np.full((120, 120, 3), 230, np.uint8)
        project.inpainted_array = project.img_array.copy()
        project.mask_array = np.zeros((120, 120), np.uint8)
        self.canvas.imgtrans_proj = project
        self.panel = DrawingPanel(self.canvas)
        self.panel.set_config(pcfg.drawpanel)
        self.bar = BottomBar(self.window)
        self.canvas.editing_layer_opacity_slider = self.bar.editingLayerSlider
        self.canvas.originallayer_trans_slider = self.bar.originalSlider
        self.bar.editingLayerSlider.valueChanged.connect(self.canvas.setEditingLayerOpacityBySlider)
        self.bar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        layout = QVBoxLayout(central)
        layout.addWidget(self.canvas.gv)
        layout.addWidget(self.panel)
        layout.addWidget(self.bar)
        self.window.show()
        self.canvas.updateCanvas()
        self.text_item = TextBlkItem(TextBlock(
            [10, 5, 110, 50], _bounding_rect=[10, 5, 100, 45], translation='Text',
        ), 0)
        self.canvas.attach_text_item(self.text_item)
        self.canvas.setPaintMode(True)
        self.panel.setCurrentToolByName('pen')
        self.panel.setPenToolWidth(14)
        self.panel.setPenToolColor([20, 30, 240, 128])
        self.canvas.addStrokeImageItem(QPointF(100, 80), self.canvas.painting_pen)
        self.panel.on_finish_painting(self.canvas.stroke_img_item)
        self.panel.setCurrentToolByName('shape')
        self.panel.fill_shape(QRectF(10, 60, 70, 40))
        self.app.processEvents()

    def tearDown(self) -> None:
        self.canvas.clear_states()
        self.text_item.geometry_controller.release_render_resources()
        self.window.close()
        self.window.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        pcfg.drawpanel = self.old_config
        pcfg.original_transparency = self.old_original
        pcfg.mask_transparency = self.old_mask

    def test_slider_dims_both_layers_without_changing_drawing_pixels(self) -> None:
        drawing = pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap())
        self.assertEqual(drawing[80, 100, 3], 128)
        viewport = self.canvas.gv.viewport()
        pos = self.canvas.gv.mapFromScene(QPointF(40, 80))
        for value in (50, 0, 100):
            self.bar.editingLayerSlider.setValue(value)
            self.assertEqual(self.canvas.textLayer.opacity(), value / 100)
            self.assertEqual(self.canvas.drawingLayer.opacity(), value / 100)
            QTest.qWait(20)
            pixel = viewport.grab().toImage().pixelColor(pos)
            expected = np.array([224, 40, 70]) * (value / 100) + 230 * (1 - value / 100)
            np.testing.assert_allclose(pixel.getRgb()[:3], expected, atol=1)
            np.testing.assert_array_equal(
                pixmap2ndarray(self.canvas.drawingLayer.get_drawed_pixmap()), drawing,
            )

    def test_export_ignores_review_opacity_and_restores_it(self) -> None:
        expected = pixmap2ndarray(self.canvas.render_result_img())
        self.assertTrue(np.any(expected[:50, :, :3] < 230))
        for value in (25, 0):
            self.bar.editingLayerSlider.setValue(value)
            np.testing.assert_array_equal(
                pixmap2ndarray(self.canvas.render_result_img()), expected,
            )
            self.assertEqual(self.canvas.textLayer.opacity(), value / 100)
            self.assertEqual(self.canvas.drawingLayer.opacity(), value / 100)
            self.assertEqual(self.bar.editingLayerSlider.value(), value)
        with patch.object(self.canvas, 'render', side_effect=RuntimeError('export failed')):
            with self.assertRaisesRegex(RuntimeError, 'export failed'):
                self.canvas.render_result_img()
        self.assertEqual(self.canvas.textLayer.opacity(), 0)
        self.assertEqual(self.canvas.drawingLayer.opacity(), 0)

    def test_drawing_previews_follow_opacity_but_inpaint_mask_does_not(self) -> None:
        self.bar.editingLayerSlider.setValue(25)
        self.assertEqual(self.canvas.shape_fill_preview.effectiveOpacity(), 0.25)
        for tool, expected in (('pen', 0.25), ('inpaint', 1.0)):
            self.panel.setCurrentToolByName(tool)
            self.canvas.addStrokeImageItem(QPointF(40, 80), self.canvas.painting_pen)
            self.assertEqual(self.canvas.stroke_img_item.effectiveOpacity(), expected)
            self.canvas.removeItem(self.canvas.stroke_img_item)

    def test_existing_numeric_opacity_toggle_updates_both_layers(self) -> None:
        self.canvas.editor_index = 1
        for expected in (0, 1):
            self.canvas.set_active_layer_transparency(0)
            self.assertEqual(self.canvas.textLayer.opacity(), expected)
            self.assertEqual(self.canvas.drawingLayer.opacity(), expected)
            self.assertEqual(self.bar.editingLayerSlider.value(), expected * 100)

    def test_zero_review_opacity_keeps_saved_images_and_reload_at_full_strength(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(pcfg, 'intermediate_imgsave_ext', '.png'),
            patch.object(pcfg, 'imgsave_ext', '.png'),
            patch.object(pcfg, 'imgtrans_textblock', False),
        ):
            project = self.canvas.imgtrans_proj
            imwrite(os.path.join(directory, 'page.png'), project.img_array)
            project.load(directory)
            expected = pixmap2ndarray(self.canvas.render_result_img(), keep_alpha=False)
            writer = ImgSaveThread()
            # Drain the production save queue synchronously for deterministic IO.
            writer.job = writer._save_img
            owner = SimpleNamespace(
                canvas=self.canvas, imgtrans_proj=project,
                st_manager=Mock(txtblkShapeControl=self.canvas.txtblkShapeControl),
                rightComicTransStackPanel=self.window, bottomBar=self.bar,
                imsave_thread=writer,
            )
            self.bar.editingLayerSlider.setValue(0)
            MainWindow.saveCurrentPage(owner, update_scene_text=False)
            writer._save_img()
            reloaded = ProjImgTrans(directory)
            reloaded.set_current_img('page.png')
            np.testing.assert_array_equal(imread(reloaded.get_result_path('page.png')), expected)
            np.testing.assert_array_equal(reloaded.inpainted_array[80, 40], expected[80, 40])
            np.testing.assert_array_equal(reloaded.inpainted_array[80, 100], expected[80, 100])
            self.assertTrue(np.all(reloaded.img_array == 230))
            self.assertEqual(self.canvas.textLayer.opacity(), 0)
            self.assertEqual(self.canvas.drawingLayer.opacity(), 0)
            self.assertEqual(self.bar.editingLayerSlider.value(), 0)


if __name__ == '__main__':
    unittest.main()
