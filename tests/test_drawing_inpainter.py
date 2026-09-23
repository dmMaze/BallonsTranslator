import copy
import gc
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, QRectF, Qt
from qtpy.QtGui import QHelpEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QToolTip

from ballontranslator.modules import codex
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.llm_modality import LLM_MODALITY_IMAGE
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.ui.module_tool_button import ModuleSelectionWidget
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.llm_profiles import default_codex_profile, default_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class DrawingInpainterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.saved_module = copy.deepcopy(pcfg.module.__dict__)
        self.draw_patch = patch.object(pcfg, 'drawpanel', DrawPanelConfig(inpainter='opencv-tela'))
        self.draw_patch.start()
        self.api = default_profile('OpenAI')
        self.api.image_model = 'run-image'
        self.api.image_model_options = ['run-image', 'draw-image', 'another-image']
        pcfg.module.llm_profiles = [self.api, default_codex_profile()]
        pcfg.module.inpainter = 'lama_large_512px'
        pcfg.module.inpaint_llm_id = 'openai'
        self.canvas = Canvas()
        self.panel = DrawingPanel(self.canvas)
        self.panel.setInpainterOptions(['opencv-tela', 'lama_large_512px', 'LLMInpaint'])
        self.brush, self.rect = self.panel._inpainter_selector_rows()
        self.run = ModuleSelectionWidget('Inpaint', 'drawingtools_inpaint.svg', LLM_MODALITY_IMAGE, parent=self.panel)
        self.run.selector.addItems(['opencv-tela', 'lama_large_512px', 'LLMInpaint'])
        self.run.setSelectedValue(pcfg.module.inpainter)
        self.manager = ModuleManager(ProjImgTrans(), self.panel)
        self.canvas.imgtrans_proj = self.manager.imgtrans_proj
        self.run.selector.currentTextChanged.connect(self.manager.selectInpainter)

    def tearDown(self) -> None:
        self.panel.deleteLater()
        self.canvas.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()
        pcfg.module.__dict__.update(self.saved_module)
        self.draw_patch.stop()

    @staticmethod
    def choose(menu, value: str | tuple[str, str, str]) -> None:
        menu.rebuildMenu()
        actions = list(menu.actions())
        for action in menu.actions():
            if action.menu() is not None:
                actions.extend(action.menu().actions())
        next(action for action in actions if action.data() == value).trigger()

    def test_draw_tools_share_native_and_llm_choices_without_changing_run(self) -> None:
        with patch.object(codex.account, '_load', side_effect=AssertionError('Credential IO')), \
                patch.object(self.manager, '_prepare_modules_then', side_effect=AssertionError('Module loading')):
            self.choose(self.brush.menu, 'lama_large_512px')
            self.assertEqual(self.rect.selector.currentText(), 'lama_large_512px')
            self.choose(self.rect.menu, ('openai', 'image_model', 'draw-image'))
            for row in (self.brush, self.rect):
                self.assertEqual(row.selector.currentText(), 'LLMInpaint')
                self.assertEqual(row.tool_button.text(), 'draw-image')
                self.assertFalse(row.prompt_panel.isHidden())
            self.assertEqual(pcfg.drawpanel.inpaint_llm_id, 'openai')
            self.assertEqual(pcfg.drawpanel.inpaint_llm_model, 'draw-image')
            self.assertEqual(pcfg.module.inpainter, 'lama_large_512px')
            self.assertEqual(pcfg.module.inpaint_llm_id, 'openai')
            self.assertEqual(self.api.image_model, 'run-image')
            codex_model = pcfg.module.llm_profiles[1].image_model
            self.choose(self.brush.menu, ('codex', 'image_model', codex_model))
            self.assertEqual(self.rect.menu.selectedProfileId(), 'codex')
            self.assertEqual(pcfg.module.inpaint_llm_id, 'openai')

    def test_run_model_and_module_changes_preserve_draw_choices_and_checkmark(self) -> None:
        self.choose(self.brush.menu, ('openai', 'image_model', 'draw-image'))
        self.choose(self.run.menu, ('openai', 'image_model', 'another-image'))
        self.panel.refreshInpainterSelection()
        self.assertEqual(self.api.image_model, 'another-image')
        self.assertEqual(pcfg.module.inpainter, 'LLMInpaint')
        self.assertEqual(self.brush.tool_button.text(), 'draw-image')
        self.brush.menu.rebuildMenu()
        profile_menu = next(action.menu() for action in self.brush.menu.actions() if action.menu() is not None)
        choices = {action.data(): action.text() for action in profile_menu.actions() if isinstance(action.data(), tuple)}
        self.assertIn('\u2713', choices[('openai', 'image_model', 'draw-image')])
        self.assertNotIn('\u2713', choices[('openai', 'image_model', 'another-image')])
        self.choose(self.run.menu, 'opencv-tela')
        window = SimpleNamespace(bottomBar=SimpleNamespace(inpaint_selector=self.run), drawingPanel=self.panel,
                                 configPanel=SimpleNamespace(llm_profiles_panel=SimpleNamespace(refreshSelectionBorders=Mock())))
        MainWindow.on_module_selection_changed(window, 'inpainter', 'opencv-tela')
        self.assertEqual(pcfg.module.inpainter, 'opencv-tela')
        self.assertEqual(pcfg.drawpanel.inpainter, 'LLMInpaint')
        self.assertEqual(self.rect.tool_button.text(), 'draw-image')

    def test_gpt_image_combinations_keep_drawing_and_run_choices_independent(self) -> None:
        self.api.image_model_options = ['gpt-image-2']
        self.api.vision_model_options = ['gpt-draw', 'gpt-run', 'other-vision-model']
        self.choose(self.brush.menu, ('openai', 'image_model', 'gpt-draw → gpt-image-2'))
        self.choose(self.run.menu, ('openai', 'image_model', 'gpt-run → gpt-image-2'))
        self.panel.refreshInpainterSelection()
        self.assertEqual(pcfg.drawpanel.inpaint_llm_model, 'gpt-draw → gpt-image-2')
        self.assertEqual(self.api.image_model, 'gpt-run → gpt-image-2')
        self.assertEqual(self.api.image_model_options, ['gpt-image-2'])
        self.assertIn('gpt-draw', self.rect.tool_button.text())

    def test_prompt_editors_share_raw_text_preserve_undo_and_hide_for_native(self) -> None:
        self.assertTrue(self.brush.prompt_panel.isHidden())
        self.choose(self.brush.menu, ('openai', 'image_model', 'draw-image'))
        original_prompt = self.api.image_prompt
        prompt = '  Repair the background.\nKeep the frame.  '
        self.brush.prompt_edit.insertPlainText(prompt)
        self.assertEqual(pcfg.drawpanel.inpaint_prompt_override, prompt)
        self.assertEqual(self.rect.prompt_edit.toPlainText(), prompt)
        self.assertTrue(self.brush.prompt_edit.document().isUndoAvailable())
        self.panel.refreshInpainterSelection()
        self.brush.prompt_edit.undo()
        self.assertEqual(pcfg.drawpanel.inpaint_prompt_override, '')
        self.assertEqual(self.rect.prompt_edit.toPlainText(), '')
        self.rect.prompt_edit.insertPlainText('  \n ')
        self.assertEqual(self.brush.prompt_edit.toPlainText(), '  \n ')
        self.choose(self.rect.menu, 'opencv-tela')
        self.assertTrue(self.brush.prompt_panel.isHidden())
        self.assertTrue(self.rect.prompt_panel.isHidden())
        self.assertEqual(pcfg.drawpanel.inpaint_prompt_override, '  \n ')
        self.assertEqual(self.api.image_prompt, original_prompt)

    def test_settings_open_the_draw_module_without_changing_run_selection(self) -> None:
        self.choose(self.brush.menu, ('openai', 'image_model', 'draw-image'))
        window = SimpleNamespace(show_module_param_dialog=Mock())
        MainWindow.to_drawing_inpaint_config(window)
        window.show_module_param_dialog.assert_called_once_with('inpainter', 'LLMInpaint')
        self.assertEqual(pcfg.module.inpainter, 'lama_large_512px')

    def test_rectangle_mask_toggle_updates_staged_request_independently_of_run(self) -> None:
        project = self.canvas.imgtrans_proj
        project.inpainted_array = np.full((8, 8, 3), 80, np.uint8)
        project.mask_array = np.zeros((8, 8), np.uint8)
        project.mask_array[3, 3] = 255
        self.panel.rectTool.click()
        controls = self.panel.rectPanel
        self.assertTrue(controls.use_mask_checker.isChecked())
        controls.dilate_slider.setValue(0)
        controls.methodComboBox.setCurrentIndex(2)
        controls.methodComboBox.activated.emit(2)
        pcfg.module.inpainter = 'LLMInpaint'
        self.panel.on_end_create_rect(QRectF(1, 1, 5, 5), 0)
        request = self.panel.rect_inpaint_dict
        expected = project.mask_array[1:6, 1:6].copy()
        np.testing.assert_array_equal(request['mask'], expected)
        self.assertTrue(request['use_mask'])
        controls.use_mask_checker.click()
        self.assertFalse(pcfg.drawpanel.rectool_use_mask)
        self.assertFalse(request['use_mask'])
        self.assertFalse(controls.methodComboBox.isEnabled())
        np.testing.assert_array_equal(request['mask'], np.full((5, 5), 255, np.uint8))
        controls.use_mask_checker.click()
        np.testing.assert_array_equal(request['mask'], expected)
        self.assertTrue(controls.methodComboBox.isEnabled())
        self.choose(self.rect.menu, ('openai', 'image_model', 'draw-image'))
        pcfg.module.inpainter = 'opencv-tela'
        self.panel.on_end_create_rect(QRectF(1, 1, 5, 5), 0)
        np.testing.assert_array_equal(self.panel.rect_inpaint_dict['mask'], expected)

    def test_disabled_rectangle_method_explains_how_to_enable_it(self) -> None:
        pcfg.drawpanel.rectool_use_mask = False
        pcfg.drawpanel.rectool_method = 1
        pcfg.drawpanel.recttool_dilate_ksize = 0
        self.panel.set_config(pcfg.drawpanel)
        self.panel.show()
        self.panel.rectTool.click()
        controls = self.panel.rectPanel
        combo = controls.methodComboBox
        QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
        self.assertFalse(combo.view().isVisible())
        self.assertEqual(combo.currentIndex(), 1)
        self.assertFalse(controls.dilate_slider.isEnabled())
        position = combo.rect().center()
        self.app.sendEvent(combo, QHelpEvent(QEvent.Type.ToolTip, position, combo.mapToGlobal(position)))
        try:
            self.assertEqual(QToolTip.text(), controls.tr('Enable Use mask to change the mask method and dilation.'))
        finally:
            QToolTip.hideText()
        controls.use_mask_checker.click()
        self.assertTrue(combo.isEnabled())
        self.assertTrue(controls.dilate_slider.isEnabled())
        self.assertEqual(combo.toolTip(), '')
        QTest.mouseClick(combo, Qt.MouseButton.LeftButton)
        self.assertTrue(combo.view().isVisible())
        QTest.keyClick(combo.view(), Qt.Key.Key_End)
        QTest.keyClick(combo.view(), Qt.Key.Key_Return)
        self.assertEqual(combo.currentIndex(), 2)
        self.assertEqual(pcfg.drawpanel.rectool_method, 2)

    def test_unmasked_rectangle_skips_segmentation_and_restores_image_and_mask_on_undo(self) -> None:
        project = self.canvas.imgtrans_proj
        project.inpainted_array = np.full((8, 8, 3), 80, np.uint8)
        project.mask_array = np.zeros((8, 8), np.uint8)
        original_image = project.inpainted_array.copy()
        original_mask = project.mask_array.copy()
        self.panel.rectTool.click()
        self.panel.rectPanel.use_mask_checker.setChecked(False)
        self.panel.rectPanel.autoChecker.setChecked(True)
        with patch('ballontranslator.ui.drawingpanel.get_maskseg_method', side_effect=AssertionError('No segmentation')), \
                patch.object(self.panel, 'runInpaint') as auto_run:
            self.panel.on_end_create_rect(QRectF(1, 1, 5, 5), 0)
        request = self.panel.rect_inpaint_dict
        auto_run.assert_called_once_with(inpaint_dict=request)
        request['need_inpaint'] = False
        with patch.object(pcfg.module, 'check_need_inpaint', True), patch.object(self.panel, 'runInpaint') as run:
            self.panel.inpaintRect(request)
        run.assert_called_once_with(inpaint_dict=request)
        result = {**request, 'inpainted': np.full_like(request['img'], 123)}
        with patch.object(self.canvas, 'updateLayers'):
            self.panel.on_inpaint_finished(result)
            np.testing.assert_array_equal(project.inpainted_array[1:6, 1:6], result['inpainted'])
            np.testing.assert_array_equal(project.mask_array[1:6, 1:6], np.full((5, 5), 255, np.uint8))
            self.canvas.draw_undo_stack.undo()
            np.testing.assert_array_equal(project.inpainted_array, original_image)
            np.testing.assert_array_equal(project.mask_array, original_mask)
            self.canvas.draw_undo_stack.redo()
            np.testing.assert_array_equal(project.inpainted_array[1:6, 1:6], result['inpainted'])
            np.testing.assert_array_equal(project.mask_array[1:6, 1:6], np.full((5, 5), 255, np.uint8))

    def test_masked_llm_rectangle_runs_model_instead_of_background_fill(self) -> None:
        self.choose(self.rect.menu, ('openai', 'image_model', 'draw-image'))
        request = {'img': np.zeros((3, 3, 3), np.uint8), 'mask': np.ones((3, 3), np.uint8),
                   'use_mask': True, 'need_inpaint': False, 'bground_rgb': [255, 255, 255],
                   'ballon_mask': np.ones((3, 3), np.uint8), 'inpaint_rect': [0, 0, 3, 3]}
        with patch.object(pcfg.module, 'check_need_inpaint', True), patch.object(self.panel, 'runInpaint') as run:
            self.panel.inpaintRect(request)
        run.assert_called_once_with(inpaint_dict=request)

    def test_native_background_fill_logs_without_starting_a_worker(self) -> None:
        project = self.canvas.imgtrans_proj
        project.inpainted_array = np.zeros((3, 3, 3), np.uint8)
        project.mask_array = np.zeros((3, 3), np.uint8)
        request = {'img': project.inpainted_array.copy(), 'mask': np.full((3, 3), 255, np.uint8),
                   'use_mask': True, 'need_inpaint': False, 'bground_rgb': [255, 255, 255],
                   'ballon_mask': np.ones((3, 3), np.uint8), 'inpaint_rect': [0, 0, 3, 3]}
        with patch.object(pcfg.module, 'check_need_inpaint', True), \
                patch.object(self.panel, 'runInpaint', side_effect=AssertionError('Unexpected worker')), \
                patch.object(self.canvas, 'updateLayers'), self.assertLogs('BallonTranslator', level='INFO') as logs:
            self.panel.inpaintRect(request)
        self.assertIn('Draw inpaint completed: background fill', '\n'.join(logs.output))
        np.testing.assert_array_equal(project.inpainted_array, np.full((3, 3, 3), 255, np.uint8))


if __name__ == '__main__':
    unittest.main()
