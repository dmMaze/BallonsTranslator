import unittest
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from qtpy.QtTest import QSignalSpy
from qtpy.QtWidgets import QApplication

from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.ocr.base import OCRBase, postprocess_ocr_text
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError
from ballontranslator.modules.textdetector.base import TextDetectorBase
from ballontranslator.ui import module_manager
from ballontranslator.utils.config import ModuleConfig, OCRTextPostprocess, pcfg
from ballontranslator.utils.registry import ModuleSpec
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.text_processing import capitalize_sentences
from ballontranslator.utils import shared
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.py_package_manager import MissingModuleRequirements
from ballontranslator.modules.translators.trans_google import TransGoogle


class SuccessfulOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['new text']


class FailingOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise ValueError('ocr backend failed')


class MissingKeyOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMModelRequiredError('profile-1', 'Profile 1', vision=True)


class FailingDetector(TextDetectorBase):
    def _detect(self, _img, _proj):
        raise ValueError('detector backend failed')

    def setup_detector(self):
        pass


class ReloadableOCR(OCRBase):
    _load_model_keys = {'model'}
    params = {}

    def __init__(self):
        super().__init__()
        self.name = 'reloadable_ocr'
        self.model = None
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        self.model = object()


class FakeOCRRegistry:
    def get_spec(self, module_name):
        return ModuleSpec(
            key=module_name,
            import_path='tests.fake_ocr',
            class_name='ReloadableOCR',
            dependencies=[],
        )

    def resolve_module(self, _module_name):
        return ReloadableOCR


class FakeManager:
    def __init__(self, thread):
        self.thread = thread

    def _thread_for_module_key(self, _module_key):
        return self.thread


class ModuleRunErrorTest(unittest.TestCase):

    def test_ocr_postprocesses_new_text_only_after_success(self):
        ocr = SuccessfulOCR()
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])
        substitutions = [{
            'keyword': 'new',
            'sub': 'final',
            'use_reg': False,
            'case_sens': True,
        }]

        with patch.object(pcfg, 'ocr_sublist', substitutions):
            with patch.object(
                pcfg.module,
                'ocr_text_postprocess',
                OCRTextPostprocess.UPPERCASE,
            ):
                ocr.run_ocr(
                    np.zeros((12, 12, 3), dtype=np.uint8),
                    [block],
                )

        self.assertEqual(block.text, 'FINAL TEXT')

    def test_ocr_restores_existing_text_after_failure(self):
        ocr = FailingOCR()
        ocr.name = 'failing_ocr'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(ModuleRunError) as caught:
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])
        self.assertEqual(caught.exception.module_key, 'ocr')
        self.assertEqual(caught.exception.module_name, 'failing_ocr')
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_ocr_preserves_llm_key_error_after_restoring_text(self):
        ocr = MissingKeyOCR()
        ocr.name = 'LLMOCR'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(LLMApiKeyRequiredError):
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])

    def test_ocr_preserves_llm_model_error_after_restoring_text(self):
        ocr = MissingModelOCR()
        ocr.name = 'LLMOCR'
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        with self.assertRaises(LLMModelRequiredError):
            ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), [block])

        self.assertEqual(block.text, ['old text'])

    def test_text_detector_wraps_runtime_failure(self):
        detector = FailingDetector()
        detector.name = 'failing_detector'

        with self.assertRaises(ModuleRunError) as caught:
            detector.detect(np.zeros((12, 12, 3), dtype=np.uint8))

        self.assertEqual(caught.exception.module_key, 'textdetector')
        self.assertEqual(caught.exception.module_name, 'failing_detector')
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_unloaded_same_module_is_not_ready(self):
        thread = module_manager.ModuleThread('ocr', FakeOCRRegistry())
        thread.module = ReloadableOCR()
        manager = FakeManager(thread)

        ready = module_manager.ModuleManager._module_ready(
            manager, 'ocr', 'reloadable_ocr'
        )

        self.assertFalse(ready)

    def test_same_module_is_reloaded_through_prepare_thread(self):
        thread = module_manager.ModuleThread('ocr', FakeOCRRegistry())
        module = ReloadableOCR()
        thread.module = module

        thread._set_module('reloadable_ocr')

        self.assertTrue(thread.last_set_success)
        self.assertIs(thread.module, module)
        self.assertTrue(module.all_model_loaded())
        self.assertEqual(module.load_count, 1)


class PipelineFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.project = ProjImgTrans()
        self.project.pages = {'page.png': [TextBlock(text=['source'], translation='old')]}
        self.project._image_info = {'page.png': {'finish_code': 0}}
        self.translate = module_manager.TranslateThread()
        self.thread = module_manager.ImgtransThread(None, None, self.translate, None)
        self.manager = module_manager.ModuleManager(self.project)
        self.manager.progress_msgbox = Mock()
        self.thread.pipeline_stopped.connect(self.manager.on_imgtrans_thread_stopped)
        self.stopped = QSignalSpy(self.thread.pipeline_stopped)
        self.finished = QSignalSpy(self.manager.imgtrans_pipeline_finished)

    def tearDown(self) -> None:
        self.thread.requestStop()
        if self.translate.pipeline_stop_event is not None:
            self.translate.pipeline_stop_event.set()
        self.assertTrue(self.thread.wait(3000))
        self.assertTrue(self.translate.wait(3000))
        self.app.processEvents()

    def _wait_for_shutdown(self) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.app.processEvents()
            if not self.thread.isRunning() and not self.translate.isRunning() and len(self.stopped):
                break
            time.sleep(0.005)
        self.assertTrue(self.thread.wait(100))
        self.assertTrue(self.translate.wait(100))
        self.app.processEvents()
        self.assertTrue(self.thread.isStopRequested())
        self.assertEqual(len(self.stopped), 1)
        self.assertEqual(len(self.finished), 1)
        self.manager.progress_msgbox.hide.assert_called_once()
        self.thread.on_module_thread_stopped()
        self.assertEqual(len(self.stopped), 1)

    def test_unexpected_exception_stops_run_and_next_run_can_start(self) -> None:
        self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
        with patch.object(module_manager, 'create_error_dialog') as dialog:
            self.thread.start()
            self._wait_for_shutdown()
        dialog.assert_called_once()
        self.assertIsNone(self.thread.job)

        self.manager.translate_thread = self.translate
        self.manager.terminateRunningThread = Mock()
        self.manager._thread_for_module_key = Mock(return_value=self.translate)
        self.manager._module_ready = Mock(return_value=True)
        self.manager._startImgtransPipeline = Mock(side_effect=lambda _pages: self.thread.runImgtransPipeline(self.project))
        with patch.object(pcfg.module, 'enable_detect', False), \
                patch.object(pcfg.module, 'enable_ocr', False), \
                patch.object(pcfg.module, 'enable_translate', True), \
                patch.object(pcfg.module, 'enable_inpaint', False), \
                patch.object(self.thread, '_imgtrans_pipeline') as next_job:
            self.manager.runImgtransPipeline()
            self.assertTrue(self.thread.wait(3000))
            self.app.processEvents()
        next_job.assert_called_once()
        self.assertFalse(self.thread.isStopRequested())
        self.assertIsNone(self.manager._pending_prepare_success)
        self.assertEqual(len(self.stopped), 1)

    def test_unexpected_exception_stops_waiting_parallel_translation(self) -> None:
        waiting = threading.Event()
        wait_for_stop = self.thread.stop_event.wait

        def wait(timeout: float) -> bool:
            waiting.set()
            return wait_for_stop(timeout)

        self.translate.translator = SimpleNamespace(delay=lambda: 0)
        self.translate.num_process_pages = 1
        with patch.object(self.thread.stop_event, 'wait', side_effect=wait), \
                patch.object(module_manager, 'create_error_dialog') as dialog:
            self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
            self.assertTrue(waiting.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            self.thread.start()
            self._wait_for_shutdown()
        dialog.assert_called_once()
        self.assertEqual(self.translate.finished_counter, 0)

    def test_stopped_emits_when_parallel_translation_finishes_last(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def delay() -> float:
            entered.set()
            release.wait(3)
            return 0

        self.translate.translator = SimpleNamespace(delay=delay)
        self.translate.num_process_pages = 1
        self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
        try:
            self.assertTrue(entered.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            with patch.object(module_manager, 'create_error_dialog') as dialog:
                self.thread.start()
                self.assertTrue(self.thread.wait(2000))
                self.assertTrue(self.translate.isRunning())
                release.set()
                self._wait_for_shutdown()
            dialog.assert_called_once()
            self.assertEqual(self.translate.finished_counter, 0)
        finally:
            release.set()

    def test_stopped_emits_if_translator_finishes_during_imgtrans_unwind(self) -> None:
        entered = threading.Event()
        release_translation = threading.Event()
        unwinding = threading.Event()
        release_imgtrans = threading.Event()

        def delay() -> float:
            entered.set()
            release_translation.wait(3)
            return 0

        emit_if_ready = self.thread._emit_pipeline_stopped_if_ready

        def hold_unwind(imgtrans_running: bool) -> None:
            emit_if_ready(imgtrans_running)
            if not imgtrans_running:
                release_translation.set()
                self.translate.wait(2000)
                unwinding.set()
                release_imgtrans.wait(3)

        self.translate.translator = SimpleNamespace(delay=delay)
        self.translate.num_process_pages = 1
        self.translate.runTranslatePipeline(self.project, self.thread.stop_event)
        try:
            self.assertTrue(entered.wait(2))
            self.thread.job = Mock(side_effect=ValueError('unexpected job failure'))
            with patch.object(module_manager, 'create_error_dialog'), \
                    patch.object(self.thread, '_emit_pipeline_stopped_if_ready', side_effect=hold_unwind):
                self.thread.start()
                self.assertTrue(unwinding.wait(2))
                self.app.processEvents()
                self.assertEqual(len(self.stopped), 0)
                release_imgtrans.set()
                self._wait_for_shutdown()
        finally:
            release_translation.set()
            release_imgtrans.set()

    def test_google_failure_does_not_mark_page_finished_and_shows_error(self) -> None:
        translator = TransGoogle('English', '简体中文')
        self.addCleanup(translator.internal_google_translator.requests_session.close)
        self.translate.translator = self.translate.module = translator
        block = self.project.pages['page.png'][0]
        with patch.object(translator.internal_google_translator.requests_session, 'request',
                          side_effect=ConnectionError('network unavailable')), \
                patch.object(module_manager, '_mark_translation_finished') as mark, \
                patch.object(module_manager, 'create_error_dialog') as dialog:
            self.assertFalse(self.thread._translate_full_page(self.project, 'page.png', [block]))
            self.assertFalse(self.translate._translate_page(self.project, 'page.png'))
        mark.assert_not_called()
        self.assertEqual(dialog.call_count, 2)
        self.assertEqual(block.translation, 'old')

    def test_declined_dependencies_notify_once_and_leave_no_pending_preparation(self) -> None:
        missing = [MissingModuleRequirements('translator', 'google', ['torch'])]
        self.manager.terminateRunningThread = Mock()
        self.manager._thread_for_module_key = Mock(return_value=self.translate)
        self.manager._module_ready = Mock(return_value=False)
        self.manager._missing_module_requirements_for_modules = Mock(return_value=missing)
        self.manager._startImgtransPipeline = Mock()
        self.manager.package_install_thread = Mock()
        self.manager.package_install_thread.isRunning.return_value = False

        for headless, install_accepted in ((False, False), (False, True), (True, False)):
            with self.subTest(headless=headless, install_accepted=install_accepted), \
                    patch.object(shared, 'HEADLESS', headless), \
                    patch.object(pcfg.package_manager, 'auto_install_missing_packages', False), \
                    patch.object(pcfg.module, 'enable_detect', False), \
                    patch.object(pcfg.module, 'enable_ocr', False), \
                    patch.object(pcfg.module, 'enable_translate', True), \
                    patch.object(pcfg.module, 'enable_inpaint', False), \
                    patch.object(self.manager, '_ask_install_missing_package_batch', return_value=install_accepted), \
                    patch.object(self.manager, '_confirm_torch_install_device', return_value=(False, None, None)), \
                    patch.object(module_manager, 'create_info_dialog') as info, \
                    patch.object(module_manager.LOGGER, 'error'):
                self.manager.runImgtransPipeline()
                if headless:
                    info.assert_not_called()
                else:
                    info.assert_called_once_with('Run cancelled: required packages were not installed.')
                self.assertEqual(self.manager._pending_prepare_queue, [])
                self.assertIsNone(self.manager._pending_prepare_success)
                self.assertIsNone(self.manager._pending_prepare_failure)
                self.assertEqual(self.manager._pending_batch_package_queue, [])
                self.assertIsNone(self.manager._pending_batch_package_success)
                self.assertIsNone(self.manager._pending_batch_package_failure)
        self.manager._startImgtransPipeline.assert_not_called()
        self.manager.package_install_thread.installPackages.assert_not_called()
        self.assertEqual(len(self.finished), 1)


if __name__ == '__main__':
    unittest.main()
