import gc
import json
import os
import tempfile
import threading
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QObject, QEvent, QPoint, Qt
from qtpy.QtGui import QColor, QTextCursor, QTextDocument
from qtpy.QtTest import QSignalSpy, QTest
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QLabel,
    QMainWindow,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QWidget,
)

from ballontranslator.ui.run_pipeline_dialog import (
    PipelineModuleActivator,
    PipelineModuleButton,
    RunPipelineDialog,
)
from ballontranslator.ui.configpanel import ConfigPanel
from ballontranslator.ui.drawingpanel import DrawingPanel
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.llm_profile_widgets import ProfileDetailsWidget
from ballontranslator.ui.module_parse_widgets import ModuleParamDialog
from ballontranslator.ui.page_range_progress import PageRangeProgressWidget
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.mainwindowbars import TitleBar
from ballontranslator.ui.module_manager import ModuleManager
from ballontranslator.ui.text_engine.annotations import (
    apply_text_combine_upright,
    load_rich_text_html,
    text_combine_upright_ranges,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.pipeline_formatting import (
    AutoTateChuYokoThread,
    apply_auto_tate_chu_yoko,
)
from ballontranslator.utils.config import (
    AutoTateChuYokoConfig,
    LLMGlossaryMode,
    LLMTranslateContext,
    OCRTextPostprocess,
    ProgramConfig,
    RunStatus,
    json_dump_program_config,
    pcfg,
)
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.modules import GET_VALID_TEXTDETECTORS
from ballontranslator.modules.translators import base as translator_base
from ballontranslator.modules.translators.base import postprocess_translation_text
from ballontranslator.modules.translators.trans_llm import LLMTranslator


def get_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class RunPipelineDialogTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = get_app()

    def setUp(self):
        self._save_config_patcher = patch(
            'ballontranslator.ui.run_pipeline_dialog.save_config',
        )
        self.save_config_mock = self._save_config_patcher.start()
        self._module_settings_expanded = (
            RunPipelineDialog._module_settings_expanded
        )
        self._page_range = RunPipelineDialog._page_range
        RunPipelineDialog._module_settings_expanded = (
            False,
            False,
            False,
            False,
        )
        RunPipelineDialog._page_range = (1, None)
        self._stage_states = [pcfg.module.stage_enabled(idx) for idx in range(4)]
        self._pipeline_mode = pcfg.run_pipeline_mode
        self._render_without_text_style_update = (
            pcfg.render_without_text_style_update
        )
        self._pipeline_general_settings = (
            pcfg.module.keep_exist_textlines,
            pcfg.restore_ocr_empty,
            pcfg.module.ocr_font_detect,
            pcfg.module.ocr_text_postprocess,
            pcfg.module.check_need_inpaint,
            pcfg.module.filter_mask_by_bboxes,
            pcfg.module.translate_source,
            pcfg.module.translate_target,
            pcfg.module.translate_context,
            pcfg.module.llm_translate_context,
            pcfg.module.llm_prior_context_token_budget,
            pcfg.module.llm_glossary_path,
            pcfg.module.llm_glossary_mode,
        )
        self._visibility_states = (
            pcfg.show_textdetector_tool,
            pcfg.show_ocr_tool,
            pcfg.show_translator_tool,
            pcfg.show_inpainter_tool,
        )
        for idx in range(4):
            pcfg.module.set_stage_enabled(idx, True)
        pcfg.show_textdetector_tool = True
        pcfg.show_ocr_tool = True
        pcfg.show_translator_tool = True
        pcfg.show_inpainter_tool = True
        pcfg.run_pipeline_mode = 'pipeline'
        pcfg.render_without_text_style_update = False

    def tearDown(self):
        RunPipelineDialog._module_settings_expanded = (
            self._module_settings_expanded
        )
        RunPipelineDialog._page_range = self._page_range
        for idx, enabled in enumerate(self._stage_states):
            pcfg.module.set_stage_enabled(idx, enabled)
        (
            pcfg.show_textdetector_tool,
            pcfg.show_ocr_tool,
            pcfg.show_translator_tool,
            pcfg.show_inpainter_tool,
        ) = self._visibility_states
        pcfg.run_pipeline_mode = self._pipeline_mode
        pcfg.render_without_text_style_update = (
            self._render_without_text_style_update
        )
        (
            pcfg.module.keep_exist_textlines,
            pcfg.restore_ocr_empty,
            pcfg.module.ocr_font_detect,
            pcfg.module.ocr_text_postprocess,
            pcfg.module.check_need_inpaint,
            pcfg.module.filter_mask_by_bboxes,
            pcfg.module.translate_source,
            pcfg.module.translate_target,
            pcfg.module.translate_context,
            pcfg.module.llm_translate_context,
            pcfg.module.llm_prior_context_token_budget,
            pcfg.module.llm_glossary_path,
            pcfg.module.llm_glossary_mode,
        ) = self._pipeline_general_settings
        self._save_config_patcher.stop()

    def test_ocr_text_postprocess_radio_buttons_update_module_config(self):
        pcfg.module.ocr_text_postprocess = OCRTextPostprocess.CAPITALIZE
        dialog = RunPipelineDialog()

        buttons = dialog.findChildren(
            QRadioButton,
            'RunPipelineOCRTextPostprocessOption',
        )

        self.assertEqual(
            [button.text() for button in buttons],
            ['None', 'Capitalize', 'Uppercase'],
        )
        self.assertTrue(
            dialog.ocr_text_postprocess_buttons[
                OCRTextPostprocess.CAPITALIZE
            ].isChecked()
        )
        postprocess_label = next(
            label
            for label in dialog.findChildren(QLabel, 'RunPipelineSettingLabel')
            if label.text() == 'Letter Case'
        )
        self.assertTrue(postprocess_label.toolTip())
        self.assertTrue(all(button.toolTip() for button in buttons))
        dialog.ocr_text_postprocess_buttons[OCRTextPostprocess.UPPERCASE].click()
        self.assertEqual(
            pcfg.module.ocr_text_postprocess,
            OCRTextPostprocess.UPPERCASE,
        )
        dialog.close()

    def test_typesetting_letter_case_defaults_without_uppercase_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'config.json')
            with open(path, 'w', encoding='utf8') as config_file:
                json.dump({'let_uppercase_flag': True}, config_file)
            config = ProgramConfig.load(path)

        self.assertEqual(config.let_letter_case, OCRTextPostprocess.NONE)
        self.assertFalse(hasattr(config, 'let_uppercase_flag'))

    def test_typesetting_letter_case_buttons_update_config(self):
        original = pcfg.let_letter_case
        pcfg.let_letter_case = OCRTextPostprocess.CAPITALIZE
        panel = ConfigPanel()
        try:
            buttons = panel.let_letter_case_buttons
            self.assertEqual(
                [button.text() for button in buttons.values()],
                ['None', 'Capitalize', 'Uppercase'],
            )
            self.assertTrue(buttons[OCRTextPostprocess.CAPITALIZE].isChecked())
            self.assertTrue(all(button.toolTip() for button in buttons.values()))

            buttons[OCRTextPostprocess.UPPERCASE].click()

            self.assertEqual(
                pcfg.let_letter_case,
                OCRTextPostprocess.UPPERCASE,
            )
        finally:
            panel.close()
            pcfg.let_letter_case = original

    def test_quick_insert_characters_config_defaults_and_round_trip(self):
        default_characters = '『』「」♥♡★☆※♩♬'
        self.assertEqual(
            ProgramConfig().quick_insert_characters,
            default_characters,
        )

        config = ProgramConfig(quick_insert_characters='♥☆')
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertEqual(restored.quick_insert_characters, '♥☆')

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'config.json')
            with open(path, 'w', encoding='utf8') as config_file:
                json.dump(
                    {'darkmode': True, 'quick_insert_characters': []},
                    config_file,
                )
            invalid = ProgramConfig.load(path)
        self.assertTrue(invalid.darkmode)
        self.assertEqual(invalid.quick_insert_characters, default_characters)

    def test_quick_insert_characters_control_updates_config(self):
        original = pcfg.quick_insert_characters
        pcfg.quick_insert_characters = '♥☆'
        panel = ConfigPanel()
        try:
            self.assertEqual(panel.quick_insert_characters_edit.text(), '♥☆')

            panel.quick_insert_characters_edit.setText('!?')

            self.assertEqual(pcfg.quick_insert_characters, '!?')
        finally:
            panel.close()
            pcfg.quick_insert_characters = original

    def test_auto_tate_chu_yoko_config_defaults_and_round_trip(self):
        defaults = ProgramConfig().auto_tate_chu_yoko
        self.assertFalse(defaults.enabled)
        self.assertEqual(defaults.max_length, 4)
        self.assertTrue(defaults.include_numbers)
        self.assertFalse(defaults.include_letters)
        self.assertEqual(defaults.additional_chars, '')
        self.assertEqual(defaults.allowed_characters(), frozenset('0123456789'))

        categories = AutoTateChuYokoConfig(
            include_numbers=False,
            include_letters=True,
            additional_chars='!?A',
        )
        self.assertEqual(
            categories.allowed_characters(),
            frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!?'),
        )

        config = ProgramConfig(auto_tate_chu_yoko={
            'enabled': True,
            'max_length': 6,
            'include_numbers': False,
            'include_letters': True,
            'additional_chars': '!?',
        })
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertEqual(restored.auto_tate_chu_yoko, config.auto_tate_chu_yoko)

        invalid = ProgramConfig(auto_tate_chu_yoko={
            'enabled': 'yes',
            'max_length': 0,
            'include_numbers': 1,
            'include_letters': True,
            'additional_chars': [],
        })
        self.assertEqual(
            invalid.auto_tate_chu_yoko,
            AutoTateChuYokoConfig(include_letters=True),
        )

    def test_compact_vertical_punctuation_config_is_enabled_by_default(self):
        self.assertTrue(ProgramConfig().compact_vertical_punctuation_spacing)

        config = ProgramConfig(
            compact_vertical_punctuation_spacing=False,
        )
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertFalse(restored.compact_vertical_punctuation_spacing)

        invalid = ProgramConfig(
            compact_vertical_punctuation_spacing='yes',
        )
        self.assertTrue(invalid.compact_vertical_punctuation_spacing)

    def test_compact_vertical_punctuation_control_updates_config(self):
        original = pcfg.compact_vertical_punctuation_spacing
        pcfg.compact_vertical_punctuation_spacing = False
        panel = ConfigPanel()
        changes = []
        panel.compact_vertical_punctuation_changed.connect(changes.append)
        try:
            checker = panel.compact_vertical_punctuation_checker
            self.assertFalse(checker.isChecked())
            self.assertTrue(checker.toolTip())

            checker.click()

            self.assertTrue(pcfg.compact_vertical_punctuation_spacing)
            self.assertEqual(changes, [True])
        finally:
            panel.close()
            pcfg.compact_vertical_punctuation_spacing = original

    def test_auto_tate_chu_yoko_controls_update_config(self):
        original = pcfg.auto_tate_chu_yoko.copy()
        pcfg.auto_tate_chu_yoko = AutoTateChuYokoConfig()
        panel = ConfigPanel()
        apply_requested = Mock()
        panel.apply_auto_tate_chu_yoko_requested.connect(apply_requested)
        try:
            self.assertFalse(panel.auto_tate_chu_yoko_checker.isChecked())
            self.assertTrue(panel.auto_tate_chu_yoko_options.isHidden())
            self.assertTrue(panel.auto_tate_chu_yoko_apply_btn.isHidden())

            panel.auto_tate_chu_yoko_checker.click()
            panel.auto_tate_chu_yoko_max_length.setValue(6)
            panel.auto_tate_chu_yoko_numbers.click()
            panel.auto_tate_chu_yoko_letters.click()
            panel.auto_tate_chu_yoko_additional_chars.setText('!?')

            self.assertFalse(panel.auto_tate_chu_yoko_options.isHidden())
            self.assertFalse(panel.auto_tate_chu_yoko_apply_btn.isHidden())
            panel.auto_tate_chu_yoko_apply_btn.click()
            apply_requested.assert_called_once_with()
            self.assertEqual(
                pcfg.auto_tate_chu_yoko,
                AutoTateChuYokoConfig(
                    enabled=True,
                    max_length=6,
                    include_numbers=False,
                    include_letters=True,
                    additional_chars='!?',
                ),
            )
        finally:
            panel.close()
            pcfg.auto_tate_chu_yoko = original

    def test_dialog_initializes_pipeline_controls(self):
        project = SimpleNamespace(
            pages={'001.png': [], '002.png': [], '003.png': [], '004.png': []},
            _image_info={
                '001.png': {'finish_code': RunStatus.FIN_ALL},
                '002.png': {'finish_code': RunStatus.FIN_DET},
                '003.png': {
                    'finish_code': RunStatus.FIN_DET | RunStatus.FIN_OCR
                },
                '004.png': {
                    'finish_code': (
                        RunStatus.FIN_DET
                        | RunStatus.FIN_OCR
                        | RunStatus.FIN_INPAINT
                    )
                },
            },
        )
        project.get_page_progress = lambda page: (
            project._image_info[page]['finish_code'] & pcfg.module.finish_code
        ) == pcfg.module.finish_code
        dialog = RunPipelineDialog(project=project)
        window_type = getattr(Qt, 'WindowType', Qt)

        self.assertEqual(dialog.windowTitle(), 'Run')
        self.assertEqual(dialog.title_label.text(), 'Run')
        self.assertTrue(dialog.windowFlags() & window_type.Dialog)
        self.assertTrue(dialog.windowFlags() & window_type.FramelessWindowHint)
        selector = dialog.findChild(QComboBox, 'RunPipelineWorkflowSelector')
        self.assertEqual(selector.currentIndex(), 0)
        self.assertEqual(
            [selector.itemText(i) for i in range(selector.count())],
            ['Pipeline', 'Rendering'],
        )
        module_buttons = dialog.findChildren(PipelineModuleButton, 'RunPipelineModuleButton')
        self.assertEqual(len(module_buttons), 4)
        activators = dialog.findChildren(
            PipelineModuleActivator,
            'RunPipelineModuleActivator',
        )
        self.assertEqual(len(activators), 4)
        self.assertTrue(all(not activator.selector.isHidden() for activator in activators))
        self.assertTrue(all(button.isChecked() for button in module_buttons))
        module_buttons[0].click()
        self.assertFalse(module_buttons[0].isChecked())
        self.assertFalse(pcfg.module.enable_detect)
        self.assertFalse(activators[0].selector.isHidden())
        self.assertEqual(module_buttons[0].icon_label.size().width(), 20)
        icon_pixmap = module_buttons[0].icon_label.pixmap()
        self.assertEqual(
            icon_pixmap.width() / icon_pixmap.devicePixelRatio(),
            20,
        )
        self.assertTrue(dialog.settings_sections[0].isHidden())
        module_buttons[0].click()
        self.assertEqual(
            {
                label.text()
                for label in dialog.findChildren(QLabel, 'RunPipelineSectionTitle')
            },
            {'Activate Modules', 'Settings'},
        )
        self.assertIsNotNone(dialog.findChild(QDockWidget, 'RunPipelineContentDock'))
        stack = dialog.findChild(QStackedWidget, 'RunPipelineContentStack')
        self.assertEqual(stack.currentIndex(), 0)
        self.assertFalse(dialog.settings_body.isHidden())
        module_settings_headers = dialog.findChildren(
            QToolButton,
            'RunPipelineModuleSettingsHeader',
        )
        self.assertEqual(
            {header.text().strip() for header in module_settings_headers},
            {
                'Text Detection',
                'OCR',
                'Inpainting',
                'Translation',
            },
        )
        self.assertEqual(len(module_settings_headers), 4)
        self.assertTrue(
            all(not header.isChecked() for header in module_settings_headers)
        )
        self.assertTrue(
            all(
                not dialog.settings_body.isAncestorOf(header)
                for header in module_settings_headers
            )
        )
        self.assertFalse(
            dialog.findChildren(QFrame, 'RunPipelineSettingsSectionLine')
        )
        detector_body = dialog.module_settings_bodies[0]
        detector_header = dialog.module_settings_headers[0]
        self.assertTrue(detector_body.isHidden())
        collapsed_height = dialog.height()
        detector_header.click()
        self.assertFalse(detector_body.isHidden())
        self.assertFalse(detector_header.isHidden())
        self.assertGreater(dialog.height(), collapsed_height)
        expanded_height = dialog.height()
        remembered_dialog = RunPipelineDialog(project=project)
        self.assertTrue(remembered_dialog.module_settings_headers[0].isChecked())
        self.assertFalse(remembered_dialog.module_settings_bodies[0].isHidden())
        self.assertTrue(
            all(
                not remembered_dialog.module_settings_headers[index].isChecked()
                for index in range(1, 4)
            )
        )
        remembered_dialog.close()
        detector_header.click()
        self.assertTrue(detector_body.isHidden())
        self.assertLess(dialog.height(), expanded_height)
        self.assertTrue(
            all(
                not dialog.settings_sections[index].isHidden()
                for index in range(4)
            )
        )
        progress = dialog.progress_bar
        self.assertEqual(progress.finished_count, 1)
        self.assertEqual(progress.page_count, 4)
        module_buttons[3].click()
        self.assertEqual(progress.finished_count, 2)
        self.assertTrue(dialog.settings_sections[3].isHidden())
        module_buttons[3].click()
        self.assertFalse(dialog.settings_sections[3].isHidden())

        range_start = dialog.findChild(QSpinBox, 'RunPipelineRangeStart')
        range_end = dialog.findChild(QSpinBox, 'RunPipelineRangeEnd')
        range_start.setValue(2)
        range_end.setValue(3)
        self.assertEqual(dialog.selected_pages(), ['002.png', '003.png'])
        self.assertEqual(progress.start_index, 1)
        self.assertEqual(progress.end_index, 2)
        range_dialog = RunPipelineDialog(project=project)
        self.assertEqual(range_dialog.range_start.value(), 2)
        self.assertEqual(range_dialog.range_end.value(), 3)
        range_dialog.close()
        progress.set_range(1, 4)
        self.assertEqual(range_start.value(), 1)
        self.assertEqual(range_end.value(), 4)
        self.assertTrue(dialog.continue_button.isDefault())
        self.assertFalse(dialog.run_button.isHidden())
        self.assertFalse(dialog.continue_button.isHidden())
        self.assertTrue(dialog.render_button.isHidden())

        pipeline_height = dialog.height()
        selector.setCurrentIndex(1)
        self.assertEqual(stack.currentIndex(), 1)
        self.assertLess(dialog.height(), pipeline_height)
        self.assertTrue(dialog.run_button.isHidden())
        self.assertTrue(dialog.continue_button.isHidden())
        self.assertFalse(dialog.render_button.isHidden())
        self.assertTrue(dialog.render_button.isDefault())
        render_option = dialog.findChild(
            QCheckBox,
            'RunPipelineRenderWithoutTextStyleUpdate',
        )
        self.assertIsNotNone(render_option)
        self.assertFalse(render_option.isChecked())
        self.assertFalse(hasattr(dialog, 'cancel_button'))
        dialog.close()

    def test_action_buttons_return_the_pipeline_choice(self):
        dialog = RunPipelineDialog()
        dialog.run_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.RUN)
        dialog.close()

        dialog = RunPipelineDialog()
        dialog.continue_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.CONTINUE)

        dialog = RunPipelineDialog()
        dialog.workflow_selector.setCurrentIndex(1)
        dialog.render_button.click()
        self.assertEqual(dialog.result(), RunPipelineDialog.RENDER)

        dialog = RunPipelineDialog()
        dialog.close_button.click()
        self.assertEqual(dialog.result(), dialog.Rejected)

        dialog = RunPipelineDialog()
        dialog.show()
        dialog.close()
        self.assertEqual(self.save_config_mock.call_count, 5)

    def test_module_activator_selector_and_auxiliary_actions(self):
        dialog = RunPipelineDialog()
        activator = dialog.module_activators[0]
        selections = []
        config_requests = []
        dialog.module_selected.connect(
            lambda *args: selections.append(args)
        )
        dialog.module_config_requested.connect(
            lambda *args: config_requests.append(args)
        )

        if activator.selector.count() > 1:
            activator.selector.setCurrentIndex(
                (activator.selector.currentIndex() + 1)
                % activator.selector.count()
            )
            self.assertEqual(
                selections[-1],
                ('textdetector', activator.selector.currentText()),
            )
        activator.config_button.click()
        self.assertEqual(
            config_requests[-1],
            ('textdetector', activator.selector.currentText()),
        )
        activator.button.setChecked(True)
        activator.deactivate_button.click()
        self.assertFalse(activator.button.isChecked())
        activator.button.click()
        self.assertTrue(activator.button.isChecked())
        self.assertFalse(activator.selector.isHidden())
        dialog.close()

    def test_dialog_is_collectable_after_deferred_delete(self):
        dialog = RunPipelineDialog()
        dialog_ref = weakref.ref(dialog)

        dialog.deleteLater()
        del dialog
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()

        self.assertIsNone(dialog_ref())

    def test_pipeline_general_settings_update_config_and_emit_actions(self):
        dialog = RunPipelineDialog(translator_metadata={
            'supported_src_list': ['Japanese', 'English'],
            'supported_tgt_list': ['English', 'Chinese'],
        })
        source_changes = []
        dialog.translate_source_changed.connect(source_changes.append)

        dialog.keep_existing_lines.setChecked(
            not pcfg.module.keep_exist_textlines
        )
        self.assertEqual(
            pcfg.module.keep_exist_textlines,
            dialog.keep_existing_lines.isChecked(),
        )
        dialog.source_combobox.setCurrentText('English')
        self.assertEqual(pcfg.module.translate_source, 'English')
        self.assertEqual(source_changes, ['English'])
        context_index = dialog.context_combobox.findData('textblock')
        dialog.context_combobox.setCurrentIndex(context_index)
        self.assertEqual(pcfg.module.translate_context, 'textblock')
        self.assertFalse(dialog.context_row.isHidden())
        self.assertTrue(dialog.llm_context_row.isHidden())
        self.assertTrue(dialog.history_budget_row.isHidden())
        self.assertFalse(hasattr(dialog, 'show_MT_keyword_window'))
        dialog.close()

    def test_llm_context_and_glossary_controls_persist_disabled_values(self):
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_prior_context_token_budget = 8192
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching
        translate_context = pcfg.module.translate_context
        dialog = RunPipelineDialog(
            translator_metadata={'name': 'LLMTranslator'},
        )

        self.assertTrue(dialog.context_row.isHidden())
        self.assertFalse(dialog.llm_context_row.isHidden())
        self.assertTrue(dialog.history_budget_row.isHidden())
        self.assertEqual(
            [
                dialog.llm_context_combobox.itemText(index)
                for index in range(dialog.llm_context_combobox.count())
            ],
            ['page', '+history'],
        )
        history_index = dialog.llm_context_combobox.findData(
            LLMTranslateContext.HISTORY
        )
        dialog.llm_context_combobox.setCurrentIndex(history_index)
        self.assertFalse(dialog.history_budget_row.isHidden())
        dialog.prior_context_token_budget.setValue(16384)
        page_index = dialog.llm_context_combobox.findData(
            LLMTranslateContext.PAGE
        )
        dialog.llm_context_combobox.setCurrentIndex(page_index)
        dialog.glossary_path_edit.setText('/tmp/glossary.tsv')
        all_index = dialog.glossary_mode_combobox.findData(LLMGlossaryMode.All)
        dialog.glossary_mode_combobox.setCurrentIndex(all_index)
        dialog.glossary_path_edit.clear()

        self.assertEqual(
            (
                pcfg.module.llm_translate_context,
                pcfg.module.llm_prior_context_token_budget,
                pcfg.module.llm_glossary_path,
                pcfg.module.llm_glossary_mode,
            ),
            (
                LLMTranslateContext.PAGE,
                16384,
                '',
                LLMGlossaryMode.All,
            ),
        )
        self.assertEqual(pcfg.module.translate_context, translate_context)
        self.assertEqual(dialog.prior_context_token_budget.value(), 16384)
        self.assertTrue(dialog.history_budget_row.isHidden())
        self.assertTrue(dialog.glossary_mode_combobox.isEnabled())
        dialog.close()

    def test_copy_source_glossary_error_preserves_clipboard(self):
        clipboard = SimpleNamespace(setText=Mock())
        translator = LLMTranslator('日本語', '简体中文')
        owner = SimpleNamespace(
            canvas=SimpleNamespace(
                selected_text_items=lambda: [SimpleNamespace(idx=0)],
            ),
            module_manager=SimpleNamespace(translator=translator),
            st_manager=SimpleNamespace(
                pairwidget_list=[
                    SimpleNamespace(
                        e_source=SimpleNamespace(toPlainText=lambda: 'Hero'),
                    )
                ],
                app_clipborad=clipboard,
            ),
            tr=lambda text: text,
        )
        with tempfile.TemporaryDirectory() as directory:
            pcfg.module.llm_glossary_path = os.path.join(
                directory,
                'deleted-glossary.json',
            )
            with patch(
                'ballontranslator.ui.mainwindow.create_error_dialog',
            ) as show_error:
                MainWindow.on_copy_src(owner)

        clipboard.setText.assert_not_called()
        show_error.assert_called_once()
        error, message = show_error.call_args.args
        self.assertIn('Glossary file not found', str(error))
        self.assertEqual(message, 'Failed to copy source text')

    def test_translation_processing_preserves_selected_and_full_page_ordering(self):
        substitutions = ({
            'keyword': 'A',
            'sub': 'X',
            'use_reg': False,
            'case_sens': True,
        },)
        cases = (
            (False, 'Ａ', 'Ａ'),
            (False, 'A', 'X'),
            (True, 'Ａ', 'X'),
        )
        for full_page, source, expected in cases:
            with self.subTest(full_page=full_page, source=source):
                result = postprocess_translation_text(
                    source,
                    'English',
                    'English',
                    substitutions,
                    full_page=full_page,
                )
                self.assertEqual(result, expected)

    def test_translation_processing_converts_before_substitution(self):
        converter = SimpleNamespace(
            convert=lambda text: text.replace('后台', '後台')
        )
        substitutions = ({
            'keyword': '後台',
            'sub': '後臺',
            'use_reg': False,
            'case_sens': True,
        },)
        with patch.object(translator_base, '_CHS2CHT_CONVERTER', converter):
            result = postprocess_translation_text(
                '后台',
                '简体中文',
                '繁體中文',
                substitutions,
                convert_to_traditional=True,
                full_page=True,
            )
        self.assertEqual(result, '後臺')

    def test_translation_letter_case_runs_after_substitution(self):
        substitutions = ({
            'keyword': 'HERO',
            'sub': 'CHAMPION',
            'use_reg': False,
            'case_sens': True,
        },)

        result = postprocess_translation_text(
            'hELLO HERO. nEXT!',
            'English',
            'English',
            substitutions,
            letter_case=OCRTextPostprocess.CAPITALIZE,
            full_page=True,
        )

        self.assertEqual(result, 'Hello champion. Next!')

        leading_number = postprocess_translation_text(
            '123 hELLO. 45 nEXT!',
            'English',
            'English',
            (),
            letter_case=OCRTextPostprocess.CAPITALIZE,
            full_page=True,
        )
        self.assertEqual(leading_number, '123 Hello. 45 Next!')

    def test_pipeline_keeps_global_keyword_substitution_actions(self):
        panel = ConfigPanel()
        actions = []
        panel.show_OCR_keyword_window.connect(lambda: actions.append('ocr'))
        panel.show_pre_MT_keyword_window.connect(lambda: actions.append('pre_mt'))
        panel.show_MT_keyword_window.connect(lambda: actions.append('mt'))

        for button in (
            panel.replaceOCRkeywordBtn,
            panel.replacePreMTkeywordBtn,
            panel.replaceMTkeywordBtn,
        ):
            button.click()
        self.assertEqual(actions, ['ocr', 'pre_mt', 'mt'])
        self.assertFalse(hasattr(panel, 'pipeline_module_panels'))
        panel.close()

    def test_module_param_dialog_saves_on_close_and_is_deleted(self):
        params = {
            'delay': {
                'type': 'line_editor',
                'value': 1,
                'display_name': 'Delay',
            }
        }
        with patch(
            'ballontranslator.ui.module_parse_widgets.save_config'
        ) as save:
            dialog = ModuleParamDialog('ocr', 'demo', params, False)
            changes = []
            dialog.paramwidget_edited.connect(
                lambda *args: changes.append(args)
            )
            dialog_ref = weakref.ref(dialog)
            dialog.show()
            self.app.processEvents()
            dialog.param_widget.param_widgets['delay'].setText('2')
            self.assertEqual(changes[-1][:3], ('ocr', 'demo', 'delay'))
            dialog.close()
            save.assert_called_once_with()
            del dialog
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            gc.collect()
            self.assertIsNone(dialog_ref())

    def test_module_param_dialog_shows_empty_message(self):
        with patch('ballontranslator.ui.module_parse_widgets.save_config'):
            dialog = ModuleParamDialog('ocr', 'empty', None, False)
            empty_label = dialog.findChild(QLabel, 'ModuleParamEmptyLabel')
            self.assertEqual(empty_label.text(), 'No configurable param')
            dialog.close()

    def test_module_param_dialog_closes_and_deletes_on_outside_click(self):
        outside = QWidget()
        outside.show()
        with patch(
            'ballontranslator.ui.module_parse_widgets.save_config'
        ) as save:
            dialog = ModuleParamDialog('ocr', 'empty', None, False)
            dialog_ref = weakref.ref(dialog)
            dialog.show()
            self.app.processEvents()
            QTest.mouseClick(outside, Qt.MouseButton.LeftButton)
            self.app.processEvents()
            save.assert_called_once_with()
            del dialog
            QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            gc.collect()
            self.assertIsNone(dialog_ref())
        outside.close()

    def test_drawing_inpainter_selectors_are_independent_and_synchronized(self):
        panel = DrawingPanel(Canvas())
        panel.setInpainterOptions(['first', 'second'], 'first')
        brush_row, rect_row = panel._inpainter_selector_rows()
        self.assertIsNot(brush_row.selector, rect_row.selector)

        changes = []
        panel.inpainter_changed.connect(changes.append)
        brush_row.selector.setCurrentText('second')

        self.assertEqual(rect_row.selector.currentText(), 'second')
        self.assertEqual(changes, ['second'])
        panel.close()


    def test_dialog_uses_platform_move_resize_backend(self):
        dialog = RunPipelineDialog()
        dialog.show()
        self.app.processEvents()

        original_size = dialog.size()
        dialog.resize(original_size.width() + 80, original_size.height() + 60)
        self.assertEqual(dialog.width(), original_size.width() + 80)
        self.assertEqual(dialog.height(), original_size.height() + 60)

        with patch(
            'ballontranslator.ui.run_pipeline_dialog.'
            'FramelessMoveResize.startSystemMove'
        ) as start_move:
            QTest.mousePress(
                dialog.title_label,
                Qt.MouseButton.LeftButton,
                pos=dialog.title_label.rect().center(),
            )
            self.app.processEvents()
            QTest.mouseRelease(
                dialog.title_label,
                Qt.MouseButton.LeftButton,
                pos=dialog.title_label.rect().center(),
            )

            selector_center = dialog.workflow_selector.mapTo(
                dialog,
                dialog.workflow_selector.rect().center(),
            ).x()
            QTest.mouseClick(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(selector_center, 8),
            )
            QTest.mouseClick(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(dialog.width() - 10, 8),
            )
            self.app.processEvents()
            self.assertEqual(start_move.call_count, 3)

        with patch(
            'ballontranslator.ui.run_pipeline_dialog.'
            'FramelessMoveResize.starSystemResize'
        ) as start_resize:
            QTest.mousePress(
                dialog,
                Qt.MouseButton.LeftButton,
                pos=QPoint(1, dialog.height() // 2),
            )
            self.app.processEvents()
            start_resize.assert_called_once()

        dialog.close()
        self.assertFalse(dialog._app_event_filter_installed)

    def test_global_event_filter_checks_relevance_before_event_type(self):
        dialog = RunPipelineDialog()
        dialog.show()
        self.app.processEvents()
        irrelevant_event = SimpleNamespace(
            type=lambda: self.fail('event type should not be requested')
        )
        with patch('qtpy.QtWidgets.QDialog.eventFilter', return_value=False):
            self.assertFalse(dialog.eventFilter(QObject(), irrelevant_event))
        dialog.close()

    def test_mainwindow_dispatches_the_selected_pipeline_action(self):
        calls = []

        class FakeSignal:
            def connect(self, _slot):
                pass

        owner = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(is_all_pages_no_text=True),
            on_run_imgtrans=lambda **kwargs: calls.append(kwargs),
            module_manager=SimpleNamespace(
                translator_metadata=lambda: {},
                module_selection_changed=FakeSignal(),
            ),
            on_trans_src_changed=lambda _source: None,
            on_trans_tgt_changed=lambda _target: None,
            on_run_module_selected=lambda _module_type, _module_name: None,
            show_module_param_dialog=lambda _module_type, _module_name: None,
        )

        class FakeDialog:
            RUN = RunPipelineDialog.RUN
            CONTINUE = RunPipelineDialog.CONTINUE
            RENDER = RunPipelineDialog.RENDER
            result = 0
            preserve_style = False
            pages = ['page-2']
            deleted_count = 0

            def __init__(self, parent, project=None, translator_metadata=None):
                self.parent = parent
                self.project = project
                self.translator_metadata = translator_metadata
                self.translate_source_changed = FakeSignal()
                self.translate_target_changed = FakeSignal()
                self.module_selected = FakeSignal()
                self.module_config_requested = FakeSignal()
                self.render_without_text_style_update = SimpleNamespace(
                    isChecked=lambda: self.preserve_style
                )

            def exec_(self):
                return self.result

            def selected_pages(self):
                return self.pages

            def setModuleSelection(self, _module_type, _module_name):
                pass

            def deleteLater(self):
                type(self).deleted_count += 1

        with patch('ballontranslator.ui.mainwindow.RunPipelineDialog', FakeDialog):
            FakeDialog.result = FakeDialog.CONTINUE
            MainWindow.run_imgtrans(owner)
            self.assertEqual(
                calls,
                [{'continue_mode': True}],
            )

            calls.clear()
            FakeDialog.result = FakeDialog.RUN
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [{'pages_to_process': ['page-2']}])

            calls.clear()
            FakeDialog.result = FakeDialog.RENDER
            FakeDialog.preserve_style = True
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [{'render_only': True}])
            self.assertTrue(owner._run_imgtrans_wo_textstyle_update)

            calls.clear()
            FakeDialog.result = 0
            MainWindow.run_imgtrans(owner)
            self.assertEqual(calls, [])
            self.assertEqual(FakeDialog.deleted_count, 4)

    def test_view_actions_only_control_bottom_bar_visibility(self):
        window = QMainWindow()
        title_bar = TitleBar(window)
        visibility_texts = [action.text() for action in title_bar.moduleVisibilityActions]
        self.assertEqual(
            visibility_texts,
            [
                'Show Text Detection',
                'Show OCR',
                'Show Translation',
                'Show Inpainting',
            ],
        )
        self.assertTrue(all(action.isChecked() for action in title_bar.moduleVisibilityActions))
        view_actions = title_bar.viewMenu.actions()
        self.assertEqual(
            [action.text() for action in view_actions[:2]],
            ['Dark Mode', 'Display Language'],
        )
        self.assertFalse(any(action.isSeparator() for action in view_actions[:2]))
        self.assertFalse(hasattr(title_bar, 'runToolBtn'))
        self.assertEqual(title_bar.sponsorToolBtn.text(), 'Sponsor')
        self.assertFalse(title_bar.sponsorToolBtn.icon().isNull())
        self.assertEqual(
            [action.text() for action in title_bar.sponsorToolBtn.menu().actions()],
            ['Patreon', 'Afdian'],
        )
        self.assertEqual(
            [
                action.text()
                for action in title_bar.toolsToolBtn.menu().actions()
                if not action.isSeparator()
            ],
            ['区域合并工具', 'Font Exclusion'],
        )
        self.assertTrue(hasattr(title_bar, 'font_exclusion_trigger'))
        with patch(
            'ballontranslator.ui.mainwindowbars.QDesktopServices.openUrl'
        ) as open_url:
            title_bar.patreonSponsorAction.trigger()
            title_bar.afdianSponsorAction.trigger()
        self.assertEqual(
            [call.args[0].toString() for call in open_url.call_args_list],
            [
                'https://patreon.com/dreMaze',
                'https://afdian.com/a/dmMaze',
            ],
        )

        emitted = []
        title_bar.show_module.connect(lambda idx, checked: emitted.append((idx, checked)))
        title_bar.moduleVisibilityActions[0].trigger()
        self.assertEqual(emitted, [(0, False)])
        self.assertTrue(pcfg.module.enable_detect)

        widgets = [SimpleNamespace(visible=True) for _ in range(4)]
        for widget in widgets:
            widget.setVisible = lambda visible, target=widget: setattr(target, 'visible', visible)
        owner = SimpleNamespace(
            bottomBar=SimpleNamespace(
                textdet_selector=widgets[0],
                ocr_selector=widgets[1],
                trans_selector=widgets[2],
                inpaint_selector=widgets[3],
            )
        )
        owner._set_module_tool_visibility = lambda idx, visible: (
            MainWindow._set_module_tool_visibility(owner, idx, visible)
        )
        with patch('ballontranslator.ui.mainwindow.save_config') as save:
            MainWindow.on_show_module(owner, 1, False)
        self.assertFalse(pcfg.show_ocr_tool)
        self.assertFalse(widgets[1].visible)
        save.assert_called_once_with()

        MainWindow._set_module_tool_visibility(owner, 0, False)
        self.assertFalse(widgets[0].visible)
        self.assertTrue(widgets[2].visible)
        self.assertTrue(widgets[3].visible)
        title_bar.deleteLater()
        window.deleteLater()

    def test_tool_visibility_round_trips_through_program_config(self):
        config = ProgramConfig(show_ocr_tool=False)
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertFalse(restored.show_ocr_tool)

    def test_pipeline_selection_round_trips_through_program_config(self):
        pcfg.run_pipeline_mode = 'rendering'
        pcfg.render_without_text_style_update = True
        dialog = RunPipelineDialog()
        self.assertEqual(dialog.workflow_selector.currentIndex(), 1)
        self.assertFalse(dialog.render_button.isHidden())
        self.assertTrue(dialog.render_without_text_style_update.isChecked())

        dialog.workflow_selector.setCurrentIndex(0)
        self.assertEqual(pcfg.run_pipeline_mode, 'pipeline')
        dialog.render_without_text_style_update.setChecked(False)
        self.assertFalse(pcfg.render_without_text_style_update)
        dialog.close()

        config = ProgramConfig(
            run_pipeline_mode='rendering',
            render_without_text_style_update=True,
        )
        restored = ProgramConfig(**json.loads(json_dump_program_config(config)))
        self.assertEqual(restored.run_pipeline_mode, 'rendering')
        self.assertTrue(restored.render_without_text_style_update)

        pcfg.run_pipeline_mode = 'automation'
        dialog = RunPipelineDialog()
        self.assertEqual(dialog.workflow_selector.currentText(), 'Pipeline')
        self.assertEqual(pcfg.run_pipeline_mode, 'automation')
        dialog.close()

    def test_fresh_run_only_resets_and_dispatches_selected_pages(self):
        pcfg.module.set_stage_enabled(3, False)
        project = ProjImgTrans()
        project.pages = {
            '001.png': [TextBlock(text=['first'])],
            '002.png': [TextBlock(text=['second'])],
            '003.png': [TextBlock(text=['third'])],
        }
        project._image_info = {
            '001.png': {'finish_code': RunStatus.FIN_ALL},
            '002.png': {'finish_code': RunStatus.FIN_ALL},
            '003.png': {'finish_code': RunStatus.FIN_ALL},
        }
        calls = []
        owner = SimpleNamespace(
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            bottomBar=SimpleNamespace(
                textblockChecker=SimpleNamespace(isChecked=lambda: False)
            ),
            imgtrans_proj=project,
            st_manager=SimpleNamespace(updateTextBlkList=lambda: None),
            module_manager=SimpleNamespace(
                runImgtransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            ),
        )

        MainWindow.on_run_imgtrans(owner, pages_to_process=['002.png'])

        self.assertEqual(
            {page: info['finish_code'] for page, info in project._image_info.items()},
            {
                '001.png': RunStatus.FIN_ALL,
                '002.png': 0,
                '003.png': RunStatus.FIN_ALL,
            },
        )
        self.assertEqual(len(project.pages['001.png']), 1)
        self.assertEqual(project.pages['002.png'], [])
        self.assertEqual(len(project.pages['003.png']), 1)
        self.assertEqual(calls, [((['002.png'],), {'render_only': False})])

        pcfg.module.set_stage_enabled(0, False)
        project._image_info['002.png']['finish_code'] = RunStatus.FIN_ALL
        project.pages['002.png'] = [TextBlock(text=['second'])]
        owner.backup_blkstyles.clear()
        calls.clear()

        MainWindow.on_run_imgtrans(owner, pages_to_process=['002.png'])

        self.assertEqual(
            project._image_info['002.png']['finish_code'],
            RunStatus.FIN_DET | RunStatus.FIN_INPAINT,
        )
        self.assertEqual(
            calls,
            [
                (
                    (['002.png'],),
                    {'render_only': False},
                )
            ],
        )

    def test_continue_dispatches_all_unfinished_pages_ignoring_selected_range(self):
        pcfg.module.set_stage_enabled(0, False)
        project = SimpleNamespace(
            pages={
                '001.png': [TextBlock()],
                '002.png': [TextBlock()],
                '003.png': [TextBlock()],
            },
            get_page_progress=lambda page: page == '002.png',
        )
        calls = []
        owner = SimpleNamespace(
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            bottomBar=SimpleNamespace(
                textblockChecker=SimpleNamespace(isChecked=lambda: False)
            ),
            imgtrans_proj=project,
            st_manager=SimpleNamespace(updateTextBlkList=lambda: None),
            module_manager=SimpleNamespace(
                runImgtransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            ),
        )

        MainWindow.on_run_imgtrans(
            owner,
            continue_mode=True,
            pages_to_process=['002.png', '003.png'],
        )

        self.assertEqual(len(project.pages['002.png']), 1)
        self.assertEqual(len(project.pages['003.png']), 1)
        self.assertEqual(len(owner.backup_blkstyles), 2)
        self.assertEqual(
            calls,
            [
                (
                    (['001.png', '003.png'],),
                    {'render_only': False},
                )
            ],
        )

    def test_selected_run_commits_only_selected_source_edits(self):
        selected = TextBlock(xyxy=[2, 3, 8, 10], text=['selected old'])
        selected.lines = [
            [[0, 0], [10, 0], [10, 5], [0, 5]],
            [[0, 5], [10, 5], [10, 10], [0, 10]],
        ]
        original_lines = [list(map(list, line)) for line in selected.lines]
        unselected = TextBlock(text=['unselected old'])
        project = ProjImgTrans()
        project.pages = {'001.png': [selected, unselected]}
        project.current_img = '001.png'
        project.img_array = SimpleNamespace(shape=(12, 12, 3))
        calls = []
        owner = SimpleNamespace(
            imgtrans_proj=project,
            global_search_widget=SimpleNamespace(
                set_document_edited=lambda: None,
            ),
            st_manager=SimpleNamespace(
                pairwidget_list=[
                    SimpleNamespace(
                        e_source=SimpleNamespace(
                            toPlainText=lambda: 'selected new',
                        ),
                    ),
                    SimpleNamespace(
                        e_source=SimpleNamespace(
                            toPlainText=lambda: 'unselected unsaved',
                        ),
                    ),
                ],
            ),
            module_manager=SimpleNamespace(
                runBlktransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                ),
            ),
        )
        selected_item = SimpleNamespace(
            blk=selected,
            idx=0,
        )

        self.assertTrue(MainWindow.translateBlkitemList(owner, [selected_item], 0))

        self.assertEqual(selected.get_text(), 'selected new')
        self.assertEqual(selected.xyxy, [2, 3, 8, 10])
        self.assertEqual(selected.lines, original_lines)
        self.assertEqual(unselected.get_text(), 'unselected old')

    def test_render_only_snapshots_complete_global_format(self):
        global_format = FontFormat(
            font_family='Render Font',
            font_size=47,
            stroke_width=0.18,
            frgb=[1, 2, 3],
            srgb=[4, 5, 6],
            alignment=2,
            vertical=True,
            opacity=0.75,
        )
        calls = []
        owner = SimpleNamespace(
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=global_format)
            ),
            bottomBar=SimpleNamespace(
                textblockChecker=SimpleNamespace(isChecked=lambda: False)
            ),
            imgtrans_proj=SimpleNamespace(pages={}),
            st_manager=SimpleNamespace(updateTextBlkList=lambda: None),
            module_manager=SimpleNamespace(
                runImgtransPipeline=lambda *args, **kwargs: calls.append(
                    (args, kwargs)
                )
            ),
        )

        MainWindow.on_run_imgtrans(owner, render_only=True)

        snapshot = owner._render_global_format
        self.assertIsNot(snapshot, global_format)
        self.assertEqual(snapshot.font_family, 'Render Font')
        self.assertEqual(snapshot.font_size, 47)
        self.assertEqual(snapshot.stroke_width, 0.18)
        self.assertEqual(snapshot.frgb, [1, 2, 3])
        self.assertEqual(snapshot.srgb, [4, 5, 6])
        self.assertEqual(snapshot.alignment, 2)
        self.assertTrue(snapshot.vertical)
        self.assertEqual(snapshot.opacity, 0.75)
        global_format.font_size = 12
        self.assertEqual(snapshot.font_size, 47)
        self.assertEqual(calls, [((None,), {'render_only': True})])

    def test_render_only_applies_typesetting_flags_to_every_block(self):
        global_format = FontFormat(
            font_family='Render Font',
            font_size=47,
            stroke_width=0.18,
            frgb=[1, 2, 3],
            srgb=[4, 5, 6],
            alignment=2,
            vertical=True,
            opacity=0.75,
            shadow_radius=3,
            shadow_strength=0.4,
            shadow_color=[7, 8, 9],
            shadow_offset=[2, 1],
        )
        blocks = [TextBlock(), TextBlock()]
        project = SimpleNamespace(
            num_pages=1,
            get_blklist_byidx=lambda _: blocks,
            set_current_img_byidx=lambda _: None,
            save=lambda: None,
        )
        owner = SimpleNamespace(
            imgtrans_proj=project,
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=True,
            _render_global_format=global_format,
            postprocess_translations=lambda _: None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            st_manager=SimpleNamespace(
                auto_textlayout_flag=False,
                updateSceneTextitems=lambda: None,
                textblk_item_list=[],
            ),
            pageList=SimpleNamespace(
                currentIndex=lambda: SimpleNamespace(row=lambda: 0)
            ),
            canvas=SimpleNamespace(updateCanvas=lambda: None),
            saveCurrentPage=lambda *args: None,
        )
        flag_names = (
            'let_fntsize_flag',
            'let_fntstroke_flag',
            'let_fntcolor_flag',
            'let_fnt_scolor_flag',
            'let_alignment_flag',
            'let_fnteffect_flag',
            'let_writing_mode_flag',
            'let_family_flag',
        )
        old_flags = {name: getattr(pcfg, name) for name in flag_names}
        try:
            for name in flag_names:
                setattr(pcfg, name, 1)
            MainWindow.on_pagtrans_finished(owner, 0)
        finally:
            for name, value in old_flags.items():
                setattr(pcfg, name, value)

        for block in blocks:
            self.assertEqual(block.font_size, 47)
            self.assertEqual(block.stroke_width, 0.18)
            self.assertEqual(block.fontformat.frgb, [1, 2, 3])
            self.assertEqual(block.fontformat.srgb, [4, 5, 6])
            self.assertEqual(block.alignment, 2)
            self.assertTrue(block.vertical)
            self.assertEqual(block.font_family, 'Render Font')
            self.assertEqual(block.fontformat.opacity, 0.75)
            self.assertEqual(block.fontformat.shadow_radius, 3)
            self.assertEqual(block.fontformat.shadow_strength, 0.4)
            self.assertEqual(block.fontformat.shadow_color, [7, 8, 9])
            self.assertEqual(block.fontformat.shadow_offset, [2, 1])

    def test_pipeline_auto_tate_chu_yoko_preserves_plain_text_format(self):
        settings = AutoTateChuYokoConfig(
            enabled=True,
            max_length=3,
            include_numbers=True,
            include_letters=True,
            additional_chars='!?',
        )
        block = TextBlock(translation='12 ABC 1234 ?!')
        block.fontformat = FontFormat(
            font_family='DejaVu Sans',
            font_size=36,
            frgb=[10, 20, 30],
            italic=True,
            underline=True,
            vertical=True,
        )
        self.assertEqual(apply_auto_tate_chu_yoko([block], settings), 1)

        document = QTextDocument()
        load_rich_text_html(document, block.rich_text, vertical=True)
        ranges = text_combine_upright_ranges(document.firstBlock())
        self.assertEqual(
            [(start, length) for start, length, _group_id in ranges],
            [(0, 2), (3, 3), (12, 2)],
        )
        cursor = QTextCursor(document)
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        char_format = cursor.charFormat()
        self.assertEqual(char_format.font().family(), 'DejaVu Sans')
        self.assertAlmostEqual(
            char_format.fontPointSize(),
            block.fontformat.size_pt,
        )
        self.assertEqual(char_format.foreground().color(), QColor(10, 20, 30))
        self.assertTrue(char_format.fontItalic())
        self.assertTrue(char_format.fontUnderline())

        unchanged = TextBlock(translation='plain text')
        unchanged.vertical = True
        self.assertEqual(apply_auto_tate_chu_yoko([unchanged], settings), 0)
        self.assertEqual(unchanged.rich_text, '')

    def test_pipeline_auto_tate_chu_yoko_strips_horizontal_runs(self):
        document = QTextDocument('12')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        apply_text_combine_upright(cursor, True)

        block = TextBlock(translation='12')
        block.vertical = False
        block.rich_text = to_rich_text_html(document)
        settings = AutoTateChuYokoConfig(enabled=True, max_length=2)

        original_html = block.rich_text
        self.assertEqual(
            apply_auto_tate_chu_yoko(
                [block],
                AutoTateChuYokoConfig(enabled=False, max_length=2),
            ),
            0,
        )
        self.assertEqual(block.rich_text, original_html)

        self.assertEqual(apply_auto_tate_chu_yoko([block], settings), 1)
        restored = QTextDocument()
        load_rich_text_html(restored, block.rich_text, vertical=False)
        self.assertEqual(text_combine_upright_ranges(restored.firstBlock()), ())

    def test_auto_tate_chu_yoko_thread_processes_all_pages(self):
        number_block = TextBlock(translation='12')
        number_block.vertical = True
        letter_block = TextBlock(translation='AB')
        letter_block.vertical = True
        thread = AutoTateChuYokoThread()
        progress = QSignalSpy(thread.progress_changed)
        finished = QSignalSpy(thread.processing_finished)

        self.assertTrue(thread.start_processing(
            {
                '001.png': [number_block],
                '002.png': [letter_block],
            },
            AutoTateChuYokoConfig(
                enabled=True,
                max_length=2,
                include_letters=True,
            ),
        ))
        self.assertTrue(thread.wait(5000))
        self.app.processEvents()

        self.assertEqual(len(finished), 1)
        self.assertEqual(finished[0][0], 2)
        self.assertEqual(len(progress), 2)
        self.assertEqual(list(progress[-1]), [100, '002.png'])
        for block in (number_block, letter_block):
            document = QTextDocument()
            load_rich_text_html(document, block.rich_text, vertical=True)
            self.assertEqual(
                [(start, length) for start, length, _ in
                 text_combine_upright_ranges(document.firstBlock())],
                [(0, 2)],
            )

    def test_auto_tate_chu_yoko_thread_honors_stop_request(self):
        entered_formatter = threading.Event()
        release_formatter = threading.Event()

        def slow_formatter(*_args):
            entered_formatter.set()
            release_formatter.wait(5)
            return 0

        thread = AutoTateChuYokoThread()
        with patch(
            'ballontranslator.ui.text_engine.pipeline_formatting.apply_auto_tate_chu_yoko',
            side_effect=slow_formatter,
        ) as formatter:
            self.assertTrue(thread.start_processing(
                {
                    '001.png': [TextBlock(), TextBlock()],
                    '002.png': [TextBlock()],
                },
                AutoTateChuYokoConfig(enabled=True),
            ))
            self.assertTrue(entered_formatter.wait(5))
            thread.request_stop()
            release_formatter.set()
            self.assertTrue(thread.wait(5000))

        self.assertEqual(formatter.call_count, 1)

    def test_manual_auto_tate_chu_yoko_starts_document_worker_only(self):
        worker = Mock()
        worker.isRunning.return_value = False
        worker.start_processing.return_value = True
        progress = Mock()
        text_manager = Mock()
        pages = {'001.png': [TextBlock(translation='12')]}
        owner = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(is_empty=False, pages=pages),
            auto_tate_chu_yoko_thread=worker,
            auto_tate_chu_yoko_progress=progress,
            st_manager=text_manager,
        )

        MainWindow.apply_auto_tate_chu_yoko_to_project(owner)

        text_manager.updateTextBlkList.assert_called_once_with()
        worker.start_processing.assert_called_once_with(
            pages,
            pcfg.auto_tate_chu_yoko,
        )
        progress.zero_progress.assert_called_once_with()
        progress.show_fitted.assert_called_once_with()

    def test_manual_auto_tate_chu_yoko_syncs_only_changed_live_documents(self):
        changed = TextBlock(translation='12')
        changed.rich_text = '<html>changed</html>'
        unchanged = TextBlock(translation='plain')
        changed_item = SimpleNamespace(
            blk=changed,
            load_rich_text_html=Mock(),
        )
        unchanged_item = SimpleNamespace(
            blk=unchanged,
            load_rich_text_html=Mock(),
        )
        owner = SimpleNamespace(
            auto_tate_chu_yoko_progress=Mock(),
            st_manager=SimpleNamespace(
                textblk_item_list=[changed_item, unchanged_item]
            ),
            canvas=Mock(),
        )

        MainWindow.on_auto_tate_chu_yoko_processing_finished(
            owner,
            1,
            (changed,),
        )

        owner.auto_tate_chu_yoko_progress.hide.assert_called_once_with()
        changed_item.load_rich_text_html.assert_called_once_with(
            changed.rich_text
        )
        unchanged_item.load_rich_text_html.assert_not_called()
        owner.canvas.setProjSaveState.assert_called_once_with(True)

    def test_auto_tate_chu_yoko_runs_only_at_pipeline_format_boundaries(self):
        blocks = [TextBlock()]
        calls = []
        project = SimpleNamespace(
            num_pages=1,
            get_blklist_byidx=lambda _: blocks,
            set_current_img_byidx=lambda _: None,
            save=lambda: None,
        )
        owner = SimpleNamespace(
            imgtrans_proj=project,
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=FontFormat(),
            postprocess_translations=lambda _: None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=FontFormat())
            ),
            st_manager=SimpleNamespace(
                auto_textlayout_flag=False,
                updateSceneTextitems=lambda: None,
                textblk_item_list=[],
            ),
            pageList=SimpleNamespace(
                currentIndex=lambda: SimpleNamespace(row=lambda: 0)
            ),
            canvas=SimpleNamespace(updateCanvas=lambda: None),
            saveCurrentPage=lambda *args: None,
        )
        flag_names = (
            'let_fntsize_flag',
            'let_fntstroke_flag',
            'let_fntcolor_flag',
            'let_fnt_scolor_flag',
            'let_alignment_flag',
            'let_fnteffect_flag',
            'let_writing_mode_flag',
            'let_family_flag',
        )
        old_flags = {name: getattr(pcfg, name) for name in flag_names}
        original_settings = pcfg.auto_tate_chu_yoko.copy()
        old_stages = {
            name: getattr(pcfg.module, name)
            for name in (
                'enable_detect',
                'enable_ocr',
                'enable_translate',
                'enable_inpaint',
            )
        }
        formatter_patch = patch(
            'ballontranslator.ui.mainwindow.apply_auto_tate_chu_yoko',
        )
        formatter = formatter_patch.start()
        formatter.side_effect = lambda *_args: calls.append(True)
        try:
            for name in flag_names:
                setattr(pcfg, name, 0)
            pcfg.auto_tate_chu_yoko.enabled = True
            pcfg.module.enable_detect = False
            pcfg.module.enable_ocr = False
            pcfg.module.enable_inpaint = False

            pcfg.module.enable_translate = True
            MainWindow.on_pagtrans_finished(owner, 0)
            self.assertEqual(calls, [True])

            calls.clear()
            pcfg.module.enable_translate = False
            owner._render_only = True
            MainWindow.on_pagtrans_finished(owner, 0)
            self.assertEqual(calls, [True])

            calls.clear()
            owner._run_imgtrans_wo_textstyle_update = True
            MainWindow.on_pagtrans_finished(owner, 0)
            self.assertEqual(calls, [])

            owner._render_only = False
            owner._run_imgtrans_wo_textstyle_update = False
            pcfg.module.enable_ocr = True
            MainWindow.on_pagtrans_finished(owner, 0)
            self.assertEqual(calls, [])
        finally:
            formatter_patch.stop()
            for name, value in old_flags.items():
                setattr(pcfg, name, value)
            for name, value in old_stages.items():
                setattr(pcfg.module, name, value)
            pcfg.auto_tate_chu_yoko = original_settings

    def test_detected_vertical_alignment_defaults_to_center(self):
        global_format = FontFormat(alignment=0, vertical=True)
        blocks = []
        project = SimpleNamespace(
            num_pages=1,
            get_blklist_byidx=lambda _: blocks,
            set_current_img_byidx=lambda _: None,
            save=lambda: None,
        )
        owner = SimpleNamespace(
            imgtrans_proj=project,
            backup_blkstyles=[],
            _run_imgtrans_wo_textstyle_update=False,
            _render_only=False,
            _render_global_format=global_format,
            postprocess_translations=lambda _: None,
            textPanel=SimpleNamespace(
                formatpanel=SimpleNamespace(global_format=global_format)
            ),
            st_manager=SimpleNamespace(
                auto_textlayout_flag=False,
                updateSceneTextitems=lambda: None,
                textblk_item_list=[],
            ),
            pageList=SimpleNamespace(
                currentIndex=lambda: SimpleNamespace(row=lambda: 0)
            ),
            canvas=SimpleNamespace(updateCanvas=lambda: None),
            saveCurrentPage=lambda *args: None,
        )

        def run(block):
            blocks[:] = [block]
            MainWindow.on_pagtrans_finished(owner, 0)

        with patch.multiple(
            pcfg,
            let_alignment_flag=0,
            let_writing_mode_flag=0,
            let_fntsize_flag=0,
            let_fntstroke_flag=0,
            let_fntcolor_flag=0,
            let_fnt_scolor_flag=0,
            let_fnteffect_flag=0,
            let_family_flag=0,
        ), patch.multiple(
            pcfg.module,
            enable_detect=True,
            enable_ocr=False,
            enable_translate=False,
            enable_inpaint=False,
        ):
            detected_vertical = TextBlock(text_layout_version=1)
            detected_vertical.vertical = True
            detected_vertical.src_is_vertical = True
            detected_vertical.alignment = 2
            run(detected_vertical)
            self.assertEqual(detected_vertical.alignment, 1)

            pcfg.let_alignment_flag = 1
            global_alignment = TextBlock(text_layout_version=1)
            global_alignment.vertical = True
            global_alignment.src_is_vertical = True
            global_alignment.alignment = 2
            run(global_alignment)
            self.assertEqual(global_alignment.alignment, 0)

            pcfg.let_alignment_flag = 0
            pcfg.let_writing_mode_flag = 1
            forced_vertical = TextBlock(text_layout_version=1)
            forced_vertical.vertical = False
            forced_vertical.src_is_vertical = False
            forced_vertical.alignment = 2
            run(forced_vertical)
            self.assertTrue(forced_vertical.vertical)
            self.assertEqual(forced_vertical.alignment, 1)

    def test_render_only_skips_module_preparation_and_finishes_every_page(self):
        pages = []
        finished = []
        owner = SimpleNamespace(
            imgtrans_proj=SimpleNamespace(is_empty=False, num_pages=3),
            progress_msgbox=SimpleNamespace(hide=lambda: None),
            terminateRunningThread=lambda: None,
            page_trans_finished=SimpleNamespace(emit=pages.append),
            imgtrans_pipeline_finished=SimpleNamespace(
                emit=lambda: finished.append(True)
            ),
            _prepare_modules_then=lambda *args, **kwargs: self.fail(
                'Rendering must not prepare pipeline modules'
            ),
        )

        ModuleManager.runImgtransPipeline(owner, render_only=True)

        self.assertEqual(pages, [0, 1, 2])
        self.assertEqual(finished, [True])


if __name__ == '__main__':
    unittest.main()
