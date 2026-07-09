import threading
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

from ballontranslator.modules.ocr.base import OCRBase
from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
)
from ballontranslator.ui import module_manager
from ballontranslator.utils import shared
from ballontranslator.utils.textblock import TextBlock


class FakeModuleSignal:
    def connect(self, _slot):
        pass


class FakeTranslateThread:
    module_thread_stopped = FakeModuleSignal()
    finished = FakeModuleSignal()

    def __init__(self, translator):
        self.module = translator
        self.translator = translator

    def isRunning(self):
        return False


class MissingKeyTranslator:
    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(self, _blk_list):
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelTranslator:
    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(self, _blk_list):
        raise LLMModelRequiredError('profile-1', 'Profile 1')


class MissingKeyOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMModelRequiredError('profile-1', 'Profile 1', vision=True)


class MissingKeyInpainter:
    def inpaint(self, _img, _mask):
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelInpainter:
    def inpaint(self, _img, _mask):
        raise LLMModelRequiredError('profile-1', 'Profile 1', target='image_model')


class MissingBaseURLInpainter:
    def inpaint(self, _img, _mask):
        raise LLMBaseURLRequiredError('profile-1', 'Profile 1', target='image_base_url')


class FakeOCRThread:
    def __init__(self, ocr):
        self.module = ocr


class LLMKeyDialogDedupTest(unittest.TestCase):
    def setUp(self):
        self._old_headless = shared.HEADLESS
        self._old_emit = shared.show_llm_key_dialog_in_mainthread
        self._old_model_emit = shared.show_llm_model_dialog_in_mainthread
        self._old_base_url_emit = shared.show_llm_base_url_dialog_in_mainthread
        self.calls = []
        self.model_calls = []
        self.base_url_calls = []
        shared.HEADLESS = False
        shared.show_llm_key_dialog_in_mainthread = lambda profile_id, profile_name: self.calls.append(
            (profile_id, profile_name)
        )
        shared.show_llm_model_dialog_in_mainthread = (
            lambda profile_id, profile_name, target: self.model_calls.append(
                (profile_id, profile_name, target)
            )
        )
        shared.show_llm_base_url_dialog_in_mainthread = (
            lambda profile_id, profile_name, target: self.base_url_calls.append(
                (profile_id, profile_name, target)
            )
        )
        module_manager._reset_llm_key_required_dialogs()

    def tearDown(self):
        shared.HEADLESS = self._old_headless
        shared.show_llm_key_dialog_in_mainthread = self._old_emit
        shared.show_llm_model_dialog_in_mainthread = self._old_model_emit
        shared.show_llm_base_url_dialog_in_mainthread = self._old_base_url_emit
        module_manager._reset_llm_key_required_dialogs()

    def test_missing_llm_key_dialog_emits_once_until_reset(self):
        error = LLMApiKeyRequiredError('profile-1', 'Profile 1')

        module_manager._show_llm_key_required_dialog(error)
        module_manager._show_llm_key_required_dialog(error)

        self.assertEqual(self.calls, [('profile-1', 'Profile 1')])

        module_manager._reset_llm_key_required_dialogs()
        module_manager._show_llm_key_required_dialog(error)

        self.assertEqual(
            self.calls,
            [('profile-1', 'Profile 1'), ('profile-1', 'Profile 1')],
        )

    def test_missing_llm_model_dialog_emits_once_until_reset(self):
        error = LLMModelRequiredError('profile-1', 'Profile 1', vision=True)

        module_manager._show_llm_model_required_dialog(error)
        module_manager._show_llm_model_required_dialog(error)

        self.assertEqual(self.model_calls, [('profile-1', 'Profile 1', 'vision_model')])

        module_manager._reset_llm_key_required_dialogs()
        module_manager._show_llm_model_required_dialog(error)

        self.assertEqual(
            self.model_calls,
            [('profile-1', 'Profile 1', 'vision_model'), ('profile-1', 'Profile 1', 'vision_model')],
        )

    def test_missing_llm_base_url_dialog_emits_once_until_reset(self):
        error = LLMBaseURLRequiredError('profile-1', 'Profile 1', target='image_base_url')

        module_manager._show_llm_base_url_required_dialog(error)
        module_manager._show_llm_base_url_required_dialog(error)

        self.assertEqual(self.base_url_calls, [('profile-1', 'Profile 1', 'image_base_url')])

        module_manager._reset_llm_key_required_dialogs()
        module_manager._show_llm_base_url_required_dialog(error)

        self.assertEqual(
            self.base_url_calls,
            [
                ('profile-1', 'Profile 1', 'image_base_url'),
                ('profile-1', 'Profile 1', 'image_base_url'),
            ],
        )

    def test_missing_llm_key_stops_imgtrans_thread(self):
        translator = MissingKeyTranslator()
        thread = module_manager.ImgtransThread(
            None,
            None,
            FakeTranslateThread(translator),
            None,
        )

        self.assertFalse(thread.isStopRequested())

        result = thread._translate_textblocks([])

        self.assertFalse(result)
        self.assertTrue(thread.isStopRequested())
        self.assertIs(translator.stop_event, thread.stop_event)

    def test_missing_llm_model_stops_imgtrans_thread(self):
        translator = MissingModelTranslator()
        thread = module_manager.ImgtransThread(
            None,
            None,
            FakeTranslateThread(translator),
            None,
        )

        self.assertFalse(thread.isStopRequested())

        result = thread._translate_textblocks([])

        self.assertFalse(result)
        self.assertTrue(thread.isStopRequested())
        self.assertIs(translator.stop_event, thread.stop_event)
        self.assertEqual(self.model_calls, [('profile-1', 'Profile 1', 'model')])

    def test_module_thread_fatal_llm_key_error_sets_pipeline_stop_event(self):
        stop_event = threading.Event()
        thread = module_manager.ModuleThread('translator', None)
        thread.pipeline_stop_event = stop_event
        thread.job = lambda: (_ for _ in ()).throw(LLMApiKeyRequiredError('profile-1', 'Profile 1'))

        thread.run()

        self.assertTrue(stop_event.is_set())

    def test_module_thread_fatal_llm_model_error_sets_pipeline_stop_event(self):
        stop_event = threading.Event()
        thread = module_manager.ModuleThread('translator', None)
        thread.pipeline_stop_event = stop_event
        thread.job = lambda: (_ for _ in ()).throw(LLMModelRequiredError('profile-1', 'Profile 1'))

        thread.run()

        self.assertTrue(stop_event.is_set())
        self.assertEqual(self.model_calls, [('profile-1', 'Profile 1', 'model')])

    def test_module_thread_fatal_llm_base_url_error_sets_pipeline_stop_event(self):
        stop_event = threading.Event()
        thread = module_manager.ModuleThread('inpainter', None)
        thread.pipeline_stop_event = stop_event
        thread.job = lambda: (_ for _ in ()).throw(
            LLMBaseURLRequiredError('profile-1', 'Profile 1', target='image_base_url')
        )

        thread.run()

        self.assertTrue(stop_event.is_set())
        self.assertEqual(self.base_url_calls, [('profile-1', 'Profile 1', 'image_base_url')])

    def test_standalone_translate_page_clears_stale_pipeline_stop_event(self):
        thread = module_manager.TranslateThread()
        thread.pipeline_stop_event = threading.Event()

        with mock.patch.object(module_manager.TranslateThread, 'start', lambda self: None):
            thread.translatePage({'page-1': []}, 'page-1')

        self.assertIsNone(thread.pipeline_stop_event)

    def test_missing_llm_key_stops_ocr_block_pipeline(self):
        ocr = MissingKeyOCR()
        ocr.name = 'LLMOCR'
        thread = module_manager.ImgtransThread(
            None,
            FakeOCRThread(ocr),
            FakeTranslateThread(None),
            None,
        )
        thread.imgtrans_proj = SimpleNamespace(
            img_array=np.zeros((12, 12, 3), dtype=np.uint8),
            mask_array=None,
        )
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        thread._blktrans_pipeline([block], 0, [0])

        self.assertEqual(self.calls, [('profile-1', 'Profile 1')])
        self.assertTrue(thread.isStopRequested())
        self.assertEqual(block.text, ['old text'])

    def test_missing_llm_model_stops_ocr_block_pipeline(self):
        ocr = MissingModelOCR()
        ocr.name = 'LLMOCR'
        thread = module_manager.ImgtransThread(
            None,
            FakeOCRThread(ocr),
            FakeTranslateThread(None),
            None,
        )
        thread.imgtrans_proj = SimpleNamespace(
            img_array=np.zeros((12, 12, 3), dtype=np.uint8),
            mask_array=None,
        )
        block = TextBlock(xyxy=[0, 0, 10, 10], text=['old text'])

        thread._blktrans_pipeline([block], 0, [0])

        self.assertTrue(thread.isStopRequested())
        self.assertEqual(block.text, ['old text'])
        self.assertEqual(self.model_calls, [('profile-1', 'Profile 1', 'vision_model')])

    def test_missing_llm_key_stops_inpaint_thread(self):
        stop_event = threading.Event()
        thread = module_manager.InpaintThread()
        thread.module = MissingKeyInpainter()
        thread.pipeline_stop_event = stop_event

        thread._inpaint(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        )

        self.assertTrue(stop_event.is_set())
        self.assertEqual(self.calls, [('profile-1', 'Profile 1')])

    def test_missing_llm_model_stops_inpaint_thread(self):
        stop_event = threading.Event()
        thread = module_manager.InpaintThread()
        thread.module = MissingModelInpainter()
        thread.pipeline_stop_event = stop_event

        thread._inpaint(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        )

        self.assertTrue(stop_event.is_set())
        self.assertEqual(self.model_calls, [('profile-1', 'Profile 1', 'image_model')])

    def test_missing_llm_base_url_stops_inpaint_thread(self):
        stop_event = threading.Event()
        thread = module_manager.InpaintThread()
        thread.module = MissingBaseURLInpainter()
        thread.pipeline_stop_event = stop_event

        thread._inpaint(
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        )

        self.assertTrue(stop_event.is_set())
        self.assertEqual(self.base_url_calls, [('profile-1', 'Profile 1', 'image_base_url')])


if __name__ == '__main__':
    unittest.main()
