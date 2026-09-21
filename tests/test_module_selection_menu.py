import copy
import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent, Qt, Signal
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.modules import (
    GET_VALID_INPAINTERS, GET_VALID_OCR, GET_VALID_TEXTDETECTORS,
    GET_VALID_TRANSLATORS,
)
from ballontranslator.ui.llm_modality import (
    LLM_MODALITY_IMAGE, LLM_MODALITY_TEXT, LLM_MODALITY_VISION,
)
from ballontranslator.ui.llm_profile_widgets import LLMProfilesWidget
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.ui.module_tool_button import ModuleSelectionWidget
from ballontranslator.ui.run_pipeline_dialog import RunPipelineDialog
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, default_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans


class SelectionWindow(QWidget):
    """Exercise the real window routing without starting inference workers."""

    llm_profile_selection_changed = Signal()
    on_run_module_selected = MainWindow.on_run_module_selected
    on_run_llm_profile_selected = MainWindow.on_run_llm_profile_selected
    on_module_selection_changed = MainWindow.on_module_selection_changed
    setTranslatorSelectionFromMetadata = MainWindow.setTranslatorSelectionFromMetadata
    on_llm_profile_changed = MainWindow.on_llm_profile_changed
    on_ocr_llm_profile_changed = MainWindow.on_ocr_llm_profile_changed
    on_inpaint_llm_profile_changed = MainWindow.on_inpaint_llm_profile_changed
    on_trans_src_changed = MainWindow.on_trans_src_changed
    on_trans_tgt_changed = MainWindow.on_trans_tgt_changed

    def __init__(self) -> None:
        super().__init__()
        self.imgtrans_proj = ProjImgTrans()
        self.module_manager = ModuleManager(self.imgtrans_proj, self)
        self.module_manager.translate_thread = SimpleNamespace(translator=None)
        self.configPanel = SimpleNamespace(llm_profiles_panel=LLMProfilesWidget(parent=self))
        self.drawingPanel = SimpleNamespace(setInpainter=Mock())
        self.show_module_param_dialog = Mock()
        self.bottomBar = SimpleNamespace()
        for attr, role, modality, options, setter, profile_slot in (
            ('textdet_selector', 'textdetector', '', GET_VALID_TEXTDETECTORS(), self.module_manager.selectTextDetector, None),
            ('ocr_selector', 'ocr', LLM_MODALITY_VISION, GET_VALID_OCR(), self.module_manager.selectOCR, self.on_ocr_llm_profile_changed),
            ('trans_selector', 'translator', LLM_MODALITY_TEXT, GET_VALID_TRANSLATORS(), self.module_manager.selectTranslator, self.on_llm_profile_changed),
            ('inpaint_selector', 'inpainter', LLM_MODALITY_IMAGE, GET_VALID_INPAINTERS(), self.module_manager.selectInpainter, self.on_inpaint_llm_profile_changed),
        ):
            widget = ModuleSelectionWidget(role, 'text.svg', modality, parent=self)
            widget.selector.addItems(options)
            widget.setSelectedValue(getattr(pcfg.module, role))
            widget.selector.currentTextChanged.connect(setter)
            if profile_slot is not None:
                widget.llm_profile_changed.connect(profile_slot)
            setattr(self.bottomBar, attr, widget)
        self.module_manager.module_selection_changed.connect(self.on_module_selection_changed)


class ModuleSelectionMenuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.saved = copy.deepcopy(pcfg.module.__dict__)
        profile = default_profile('OpenAI')
        profile.image_model = 'image-one'
        profile.image_model_options = ['image-one', 'image-two']
        pcfg.module.llm_profiles = [profile]
        pcfg.module.translator = next(name for name in GET_VALID_TRANSLATORS() if name != 'LLMTranslator')
        self.window = SelectionWindow()
        self.save_patch = patch('ballontranslator.ui.run_pipeline_dialog.save_config')
        self.save_patch.start()

    def tearDown(self) -> None:
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()
        pcfg.module.__dict__.update(self.saved)
        self.save_patch.stop()

    @staticmethod
    def choose(menu, data: tuple) -> None:
        menu.rebuildMenu()
        profile_menu = next(action.menu() for action in menu.actions() if action.menu() is not None)
        next(action for action in profile_menu.actions() if action.data() == data).trigger()

    def test_run_and_bottom_bar_share_live_profile_and_module_selection(self) -> None:
        def interact(dialog: RunPipelineDialog) -> int:
            for role, bottom_name, llm_key, field, value, profile_id_attr in (
                ('translator', 'trans_selector', 'LLMTranslator', 'model', 'gpt-5.4', 'translator_llm_id'),
                ('ocr', 'ocr_selector', 'LLMOCR', 'vision_model', 'gpt-4o', 'ocr_llm_id'),
                ('inpainter', 'inpaint_selector', 'LLMInpaint', 'image_model', 'image-two', 'inpaint_llm_id'),
            ):
                with self.subTest(role=role):
                    activator = next(a for a in dialog.module_activators if a.module_type == role)
                    bottom = getattr(self.window.bottomBar, bottom_name)
                    self.choose(activator.menu, ('openai', field, value))
                    self.assertEqual(getattr(pcfg.module, role), llm_key)
                    self.assertEqual(getattr(pcfg.module, profile_id_attr), 'openai')
                    self.assertEqual(bottom.selector.currentText(), llm_key)
                    self.assertIn(value, bottom.tool_btn.text())
                    self.assertIn(value, activator.selector.text())
                    self.assertEqual(
                        getattr(self.window.configPanel.llm_profiles_panel.rows['openai'].profile, field), value,
                    )

                    other = next(option for option in getattr(pcfg.module.llm_profiles[0], activator.menu.model_options_attr) if option != value)
                    self.choose(bottom.menu, ('openai', field, other))
                    self.assertIn(other, activator.selector.text())
                    activator.menu.rebuildMenu()
                    profile_action = next(action for action in activator.menu.actions() if action.menu() is not None)
                    self.assertIn('\u2713', profile_action.text())
                    selected = next(action for action in profile_action.menu().actions() if action.data() == ('openai', field, other))
                    self.assertIn('\u2713', selected.text())

                    if role == 'translator':
                        self.choose(activator.menu, ('openai', 'thinking_level', 'high'))
                        self.assertTrue(bottom.tool_btn.text().endswith(' high'))
                        self.assertTrue(activator.selector.text().endswith(' high'))
                    elif role == 'ocr':
                        self.choose(activator.menu, ('openai', 'vision_detail_level', 'high'))
                        self.assertEqual(pcfg.module.llm_profiles[0].vision_detail_level, 'high')

                    normal = next(bottom.selector.itemText(i) for i in range(bottom.selector.count()) if bottom.selector.itemText(i) != llm_key)
                    bottom.selector.setCurrentText(normal)
                    self.assertEqual(activator.module_combo.currentText(), normal)
                    self.assertEqual(activator.selector.text(), normal)
            return 0

        with patch.object(RunPipelineDialog, 'exec_', interact):
            MainWindow.run_imgtrans(self.window)

    def test_profiles_are_filtered_and_languages_remain_bottom_bar_only(self) -> None:
        pcfg.module.llm_profiles.append(LLMProfile(
            id='text-only', name='Text only', model='text-model', model_options=['text-model'],
        ))
        dialog = RunPipelineDialog(self.window)
        for activator in dialog.module_activators:
            activator.menu.rebuildMenu()
            titles = [action.menu().title() for action in activator.menu.actions() if action.menu() is not None]
            expected = ['OpenAI', 'Text only'] if activator.module_type == 'translator' else ['OpenAI']
            if activator.module_type == 'textdetector':
                expected = []
            self.assertEqual(titles, expected)
        bottom = self.window.bottomBar.trans_selector
        bottom.src_selector.addItem('Japanese')
        bottom.tgt_selector.addItem('English')
        bottom.menu.aboutToShow.emit()
        titles = [action.menu().title() for action in bottom.menu.actions() if action.menu() is not None]
        self.assertIn('Source - Japanese', titles)
        self.assertIn('Target - English', titles)
        dialog.deleteLater()

    def test_inactive_selector_first_click_activates_without_opening_menu(self) -> None:
        dialog = RunPipelineDialog(self.window)
        activator = dialog.module_activators[0]
        activator.button.setChecked(False)
        dialog.show()
        self.app.processEvents()
        QTest.mouseClick(activator.selector, Qt.MouseButton.LeftButton)
        self.assertTrue(activator.button.isChecked())
        self.assertFalse(activator.menu.isVisible())
        dialog.close()
        dialog.deleteLater()

    def test_rebuilding_releases_old_submenus_and_dialog_connections(self) -> None:
        dialog = RunPipelineDialog(self.window)
        self.window.llm_profile_selection_changed.connect(dialog.refreshLLMSelections)
        activator = next(a for a in dialog.module_activators if a.module_type == 'translator')
        activator.menu.rebuildMenu()
        old_menu = next(action.menu() for action in activator.menu.actions() if action.menu() is not None)
        old_ref = weakref.ref(old_menu)
        activator.menu.rebuildMenu()
        del old_menu
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        self.assertIsNone(old_ref())
        dialog_ref = weakref.ref(dialog)
        dialog.deleteLater()
        del activator, dialog
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        gc.collect()
        self.assertIsNone(dialog_ref())
        self.window.llm_profile_selection_changed.emit()


if __name__ == '__main__':
    unittest.main()
