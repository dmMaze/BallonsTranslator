import copy
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import CodexSignInRequiredError, LLMUserActionRequiredError
from ballontranslator.modules.inpaint.inpaint_llm import LLMInpaint
from ballontranslator.modules.llm_image import LLMImageRequester, LLMImageRequestPolicy
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.ui import codex_account, module_manager
from ballontranslator.ui.mainwindow import MainWindow
from ballontranslator.ui.text_engine.effects.panel import TextEffectPanel
from ballontranslator.utils import global_callbacks, shared
from ballontranslator.utils.config import ModuleConfig, RunStatus, pcfg
from ballontranslator.utils.llm_profiles import default_profile, runtime_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


class CodexRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.saved_module = copy.deepcopy(pcfg.module.__dict__)
        self.profile = default_profile('Codex')
        self.profile.model = self.profile.vision_model = self.profile.image_model = ''
        pcfg.module.llm_profiles = [self.profile]
        pcfg.module.translator_llm_id = 'codex'
        pcfg.module.ocr_llm_id = 'codex'
        pcfg.module.inpaint_llm_id = 'codex'
        self.account = codex.CodexAccount()
        self.account._loaded = True
        self.account_patch = patch.object(codex, 'account', self.account)
        self.account_patch.start()
        self.addCleanup(self.account_patch.stop)
        self.callback_patch = patch.dict(global_callbacks._REGISTERED_CALLBACKS, clear=True)
        self.callback_patch.start()
        self.addCleanup(self.callback_patch.stop)
        self.bridge_patch = patch.object(codex_account, 'show_codex_sign_in_required')
        self.show_sign_in = self.bridge_patch.start()
        self.addCleanup(self.bridge_patch.stop)
        self.image = np.zeros((12, 12, 3), np.uint8)

    def tearDown(self) -> None:
        pcfg.module.__dict__.update(self.saved_module)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_missing_auth_precedes_blank_models_without_requests_or_retries(self) -> None:
        translator = LLMTranslator('English', '简体中文')
        ocr = LLMOCR()
        inpainter = LLMInpaint()
        requester = LLMImageRequester(image_request_policy=LLMImageRequestPolicy())
        pcfg.module.ocr_llm_page_level = True
        block = TextBlock(xyxy=[1, 1, 10, 10], text=['original'])
        operations = (
            lambda: translator.translate(['source']),
            lambda: ocr.run_ocr(self.image),
            lambda: ocr.run_ocr(self.image, [block], full_page=True),
            lambda: inpainter.inpaint(self.image, np.full((12, 12), 255, np.uint8)),
            lambda: requester.request_image_with_retries(self.profile, None, 'Draw paper.', ''),
        )
        with patch.object(codex, 'request_chat_completion', side_effect=AssertionError('Unexpected HTTP')), \
                patch.object(codex, 'request_image', side_effect=AssertionError('Unexpected HTTP')), \
                patch.object(requester, '_respect_delay') as delay:
            for operation in operations:
                with self.subTest(operation=operation), patch.object(
                    self.account, 'require_sign_in', wraps=self.account.require_sign_in,
                ) as require_sign_in:
                    with self.assertRaises(CodexSignInRequiredError) as caught:
                        operation()
                    self.assertFalse(caught.exception.invalid)
                    require_sign_in.assert_called_once()
            delay.assert_not_called()
        self.assertEqual(block.text, ['original'])

    def test_module_selection_construction_and_profile_reads_do_not_load_credentials(self) -> None:
        manager = module_manager.ModuleManager(ProjImgTrans())
        manager.translate_thread = SimpleNamespace(translator=None)
        self.addCleanup(manager.deleteLater)
        with patch.object(self.account, '_load', side_effect=AssertionError('Credential IO')), \
                patch.object(self.account, 'require_sign_in', side_effect=AssertionError('Auth preflight')):
            manager.selectTranslator('LLMTranslator')
            manager.selectOCR('LLMOCR')
            manager.selectInpainter('LLMInpaint')
            manager.translator_metadata()
            modules = (LLMTranslator('English', '简体中文'), LLMOCR(), LLMInpaint())
            for module in modules:
                try:
                    module.profile
                except LLMUserActionRequiredError:
                    pass
        self.assertEqual(pcfg.module.translator, 'LLMTranslator')
        self.assertEqual(pcfg.module.ocr, 'LLMOCR')
        self.assertEqual(pcfg.module.inpainter, 'LLMInpaint')

    def test_missing_saved_selection_resolves_to_codex_before_runtime_guard(self) -> None:
        with patch.object(self.account, '_load', side_effect=AssertionError('Credential IO')):
            module = ModuleConfig(
                llm_profiles=[self.profile], translator_llm_id='missing',
                ocr_llm_id='', inpaint_llm_id='removed',
            )
        for field in ('translator_llm_id', 'ocr_llm_id', 'inpaint_llm_id'):
            selected_id = getattr(module, field)
            self.assertEqual(selected_id, 'codex')
            self.assertEqual(runtime_profile(module.llm_profiles, selected_id).backend, 'codex')
        pcfg.module.__dict__.update(module.__dict__)
        with self.assertRaises(CodexSignInRequiredError):
            LLMTranslator('English', '简体中文').translate(['source'])

    def _project(self) -> ProjImgTrans:
        project = ProjImgTrans()
        project.pages = {
            key: [TextBlock(xyxy=[1, 1, 10, 10], text=['source'])]
            for key in ('one', 'two')
        }
        project._image_info = {key: {'finish_code': 0} for key in project.pages}
        project.current_img = 'one'
        project.img_array = self.image
        project.mask_array = np.full((12, 12), 255, np.uint8)
        project.inpainted_array = self.image.copy()
        project.read_img = Mock(return_value=self.image)
        return project

    def _translator_thread(self, *, low_vram: bool = False) -> module_manager.TranslateThread:
        thread = module_manager.TranslateThread()
        thread.module = thread.translator = SimpleNamespace(
            lang_source='English', lang_target='English', low_vram_mode=low_vram,
            is_computational_intensive=Mock(return_value=True),
            delay=Mock(return_value=0), set_stop_event=Mock(),
            translate_textblk_lst=Mock(side_effect=CodexSignInRequiredError(invalid=True)),
            on_page_translation_finished=Mock(),
        )
        self.addCleanup(thread.deleteLater)
        return thread

    def _pipeline(self, translator_thread: module_manager.TranslateThread) -> module_manager.ImgtransThread:
        thread = module_manager.ImgtransThread(
            SimpleNamespace(textdetector=None),
            SimpleNamespace(module=SimpleNamespace(run_ocr=Mock()), ocr=None),
            translator_thread,
            SimpleNamespace(inpainter=SimpleNamespace(inpaint=Mock())),
        )
        thread.imgtrans_proj = self._project()
        thread.process_idx_to_page_idx = {}
        self.addCleanup(thread.deleteLater)
        return thread

    def test_selected_blocks_stop_before_native_inpainting_after_auth_failure(self) -> None:
        translate = self._translator_thread()
        thread = self._pipeline(translate)
        finished = QSignalSpy(thread.finish_blktrans)
        thread._blktrans_pipeline(thread.imgtrans_proj.pages['one'], 2, [0], page_key='one')
        self.assertTrue(thread.isStopRequested())
        thread.inpainter.inpaint.assert_not_called()
        self.assertTrue(finished)
        self.show_sign_in.assert_called_once()
        self.assertTrue(self.show_sign_in.call_args.args[0].invalid)

    def test_serial_and_low_vram_runs_do_not_translate_next_page_or_count_failure(self) -> None:
        for index in range(4):
            pcfg.module.set_stage_enabled(index, index == 2)
        for low_vram in (False, True):
            with self.subTest(low_vram=low_vram):
                translate = self._translator_thread(low_vram=low_vram)
                thread = self._pipeline(translate)
                progress = QSignalSpy(thread.update_translate_progress)
                stopped = QSignalSpy(thread.pipeline_stopped)
                with patch.object(module_manager, 'unload_modules'):
                    thread._imgtrans_pipeline()
                self.assertTrue(thread.isStopRequested())
                self.assertEqual(thread.translate_counter, 0)
                self.assertEqual(len(progress), 0)
                self.assertEqual(len(stopped), 1)
                translate.translator.translate_textblk_lst.assert_called_once()
                self.assertEqual(translate.translator.translate_textblk_lst.call_args.kwargs['page_key'], 'one')
                for info in thread.imgtrans_proj._image_info.values():
                    self.assertFalse(info['finish_code'] & RunStatus.FIN_TRANSLATE)

    def test_parallel_translation_stops_without_counting_or_processing_queued_page(self) -> None:
        thread = self._translator_thread()
        thread.imgtrans_proj = self._project()
        thread.num_process_pages = 2
        thread.pipeline_stop_event = threading.Event()
        thread.pipeline_pagekey_queue = ['one', 'two']
        progress = QSignalSpy(thread.progress_changed)
        stopped = QSignalSpy(thread.module_thread_stopped)
        thread._run_translate_pipeline()
        self.assertTrue(thread.pipeline_stop_event.is_set())
        self.assertEqual(thread.finished_counter, 0)
        self.assertEqual(thread.pipeline_pagekey_queue, ['two'])
        self.assertEqual(len(progress), 0)
        self.assertEqual(len(stopped), 1)
        thread.translator.translate_textblk_lst.assert_called_once()

    def test_stopped_run_cannot_emit_success_before_workers_exit(self) -> None:
        manager = module_manager.ModuleManager(ProjImgTrans())
        self.addCleanup(manager.deleteLater)
        manager.imgtrans_thread = SimpleNamespace(isStopRequested=lambda: True)
        manager.progress_msgbox = SimpleNamespace(hide=Mock())
        finished = QSignalSpy(manager.imgtrans_pipeline_finished)
        with patch.object(manager, 'proj_finished', return_value=True):
            manager.finishImgtransPipeline()
        self.assertEqual(len(finished), 0)
        manager.on_imgtrans_thread_stopped()
        self.assertEqual(len(finished), 1)

    def test_headless_stop_finishes_pending_saves_and_exits_without_next_directory(self) -> None:
        window = SimpleNamespace(
            backup_blkstyles=[], module_manager=SimpleNamespace(
                imgtrans_thread=SimpleNamespace(isStopRequested=lambda: True),
            ),
            imsave_thread=SimpleNamespace(wait=Mock()),
            app=SimpleNamespace(quit=Mock()), run_next_dir=Mock(),
        )
        with patch.object(shared, 'HEADLESS', True), patch.object(shared, 'args', SimpleNamespace(
            export_translation_txt=False, export_source_txt=False,
        )):
            MainWindow.on_imgtrans_pipeline_finished(window)
        window.imsave_thread.wait.assert_called_once()
        window.app.quit.assert_called_once()
        window.run_next_dir.assert_not_called()

    def test_image_generation_error_routes_to_the_same_sign_in_bridge(self) -> None:
        panel = SimpleNamespace(_image_cards=lambda: [SimpleNamespace(index=3)])
        error = CodexSignInRequiredError(invalid=True)
        TextEffectPanel.show_image_generation_error(panel, 3, error)
        self.show_sign_in.assert_called_once_with(error)


if __name__ == '__main__':
    unittest.main()
