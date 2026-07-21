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
    ModuleRunError,
)
from ballontranslator.ui import module_manager
from ballontranslator.utils import shared
from ballontranslator.utils.config import RunStatus, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
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
        self.pipeline_calls = []
        self.queued_page_keys = []
        self.finished_counter = 0

    def isRunning(self):
        return False

    def runTranslatePipeline(self, project, stop_event):
        self.pipeline_calls.append((project, stop_event))

    def push_pagekey_queue(self, page_key):
        self.queued_page_keys.append(page_key)


class MissingKeyTranslator:
    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(
        self,
        _blk_list,
        *,
        project=None,
        page_key=None,
        full_page=False,
    ):
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelTranslator:
    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(
        self,
        _blk_list,
        *,
        project=None,
        page_key=None,
        full_page=False,
    ):
        raise LLMModelRequiredError('profile-1', 'Profile 1')


class SuccessfulTranslator:
    lang_source = 'English'
    lang_target = 'English'
    low_vram_mode = False

    def __init__(self):
        self.calls = []
        self.computational_intensive = False

    def is_computational_intensive(self):
        return self.computational_intensive

    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def translate_textblk_lst(
        self,
        blk_list,
        *,
        project=None,
        page_key=None,
        full_page=False,
    ):
        self.calls.append((blk_list, project, page_key, full_page))
        for block in blk_list:
            block.translation = 'ａ! hero'


class FailingTranslator(SuccessfulTranslator):
    def translate_textblk_lst(
        self,
        blk_list,
        *,
        project=None,
        page_key=None,
        full_page=False,
    ):
        self.calls.append((blk_list, project, page_key, full_page))
        raise RuntimeError('translation failed')


class FailingDetector:
    def detect(self, _img, _project):
        raise ModuleRunError('textdetector', 'failing', 'failed')


class FakeTextDetectThread:
    def __init__(self, detector):
        self.textdetector = detector


class MissingKeyOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMApiKeyRequiredError('profile-1', 'Profile 1')


class MissingModelOCR(OCRBase):
    def _ocr_blk_list(self, _img, blk_list, *args, **kwargs):
        blk_list[0].text = ['partial text']
        raise LLMModelRequiredError('profile-1', 'Profile 1', vision=True)


class FailingRuntimeOCR(OCRBase):
    def _ocr_blk_list(self, _img, _blk_list, *args, **kwargs):
        raise RuntimeError('ocr failed')


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
        self.ocr = ocr


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

    def test_page_failure_message_is_logged_and_shown(self):
        with mock.patch(
            'ballontranslator.utils.message.LOGGER.error',
        ) as log_error, mock.patch.object(
            shared,
            'create_errdialog_in_mainthread',
        ) as show_error:
            module_manager._create_page_error_dialog(
                RuntimeError('ocr failed'),
                'OCR Failed.',
                'PageFailureTest',
                'page-1',
            )

        logged = '\n'.join(
            str(call.args[0])
            for call in log_error.call_args_list
        )
        self.assertIn('Page: page-1', logged)
        self.assertIn('Page: page-1', show_error.call_args.args[0])

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

    def test_full_page_completion_reflects_translation_outcome(self):
        for translator_type, expected_success in (
            (SuccessfulTranslator, True),
            (FailingTranslator, False),
        ):
            with self.subTest(translator=translator_type.__name__):
                translator = translator_type()
                thread = module_manager.TranslateThread()
                thread.translator = translator
                page = [TextBlock(text=['source'], translation='old')]
                project = ProjImgTrans()
                project.pages = {'page-1': page}
                project._image_info = {'page-1': {'finish_code': 0}}
                project.mark_translation_finished('page-1', 'English')

                with mock.patch(
                    'ballontranslator.ui.module_manager.create_error_dialog',
                ) as show_error:
                    success = thread._translate_page(
                        project,
                        'page-1',
                    )

                self.assertIs(success, expected_success)
                self.assertEqual(
                    translator.calls,
                    [(page, project, 'page-1', True)],
                )
                info = project._image_info['page-1']
                self.assertIs(
                    bool(info['finish_code'] & RunStatus.FIN_TRANSLATE),
                    expected_success,
                )
                if expected_success:
                    self.assertEqual(info['translation_target'], 'English')
                else:
                    self.assertNotIn('translation_target', info)
                    self.assertIn('Page: page-1', show_error.call_args.args[1])

    def test_imgtrans_full_page_forwards_context_and_marks_only_success(self):
        for translator_type, expected_success in (
            (SuccessfulTranslator, True),
            (FailingTranslator, False),
        ):
            with self.subTest(translator=translator_type.__name__):
                translator = translator_type()
                thread = module_manager.ImgtransThread(
                    SimpleNamespace(textdetector=None),
                    FakeOCRThread(None),
                    FakeTranslateThread(translator),
                    SimpleNamespace(inpainter=None),
                )
                page = [TextBlock(text=['source'], translation='old')]
                project = ProjImgTrans()
                project.pages = {'page-1': page}
                project._image_info = {'page-1': {'finish_code': 0}}
                project.mark_translation_finished('page-1', 'English')

                with mock.patch(
                    'ballontranslator.ui.module_manager.create_error_dialog',
                ) as show_error:
                    success = thread._translate_full_page(
                        project,
                        'page-1',
                        page,
                    )

                self.assertIs(success, expected_success)
                self.assertEqual(
                    translator.calls,
                    [(page, project, 'page-1', True)],
                )
                info = project._image_info['page-1']
                self.assertIs(
                    bool(info['finish_code'] & RunStatus.FIN_TRANSLATE),
                    expected_success,
                )
                if expected_success:
                    self.assertEqual(info['translation_target'], 'English')
                else:
                    self.assertNotIn('translation_target', info)
                    self.assertIn('Page: page-1', show_error.call_args.args[1])

    def test_pipeline_translation_modes_share_the_expected_boundaries(self):
        """Headless runs this same router; there is no headless-only boundary."""
        old_stages = [pcfg.module.stage_enabled(index) for index in range(4)]
        try:
            for index in range(4):
                pcfg.module.set_stage_enabled(index, index == 2)

            for mode in ('parallel', 'direct', 'low-vram'):
                with self.subTest(mode=mode):
                    translator = SuccessfulTranslator()
                    translator.computational_intensive = mode == 'direct'
                    translator.low_vram_mode = mode == 'low-vram'
                    translate_thread = FakeTranslateThread(translator)
                    thread = module_manager.ImgtransThread(
                        SimpleNamespace(textdetector=None),
                        FakeOCRThread(None),
                        translate_thread,
                        SimpleNamespace(inpainter=None),
                    )
                    block = TextBlock(text=['source'], translation='old')
                    project = ProjImgTrans()
                    project.pages = {'page-1': [block]}
                    project._image_info = {'page-1': {'finish_code': 0}}
                    project.read_img = lambda _page: np.zeros(
                        (12, 12, 3),
                        dtype=np.uint8,
                    )
                    thread.imgtrans_proj = project
                    thread.pages_to_process = None
                    thread.process_idx_to_page_idx = {}

                    with mock.patch.object(
                        thread,
                        '_translate_full_page',
                        return_value=True,
                    ) as full_page, mock.patch(
                        'ballontranslator.ui.module_manager.unload_modules',
                    ):
                        thread._imgtrans_pipeline()

                    if mode == 'parallel':
                        self.assertEqual(
                            translate_thread.pipeline_calls,
                            [(project, thread.stop_event)],
                        )
                        self.assertEqual(
                            translate_thread.queued_page_keys,
                            ['page-1'],
                        )
                        full_page.assert_not_called()
                    else:
                        self.assertEqual(translate_thread.pipeline_calls, [])
                        self.assertEqual(translate_thread.queued_page_keys, [])
                        full_page.assert_called_once_with(
                            project,
                            'page-1',
                            [block],
                        )
        finally:
            for index, enabled in enumerate(old_stages):
                pcfg.module.set_stage_enabled(index, enabled)

    def test_selected_translation_refreshes_complete_page_target(self):
        for saved_target in ('English', '简体中文', None):
            with self.subTest(saved_target=saved_target):
                translator = SuccessfulTranslator()
                thread = module_manager.ImgtransThread(
                    None,
                    None,
                    FakeTranslateThread(translator),
                    None,
                )
                block = TextBlock(text=['source'], translation='old')
                project = ProjImgTrans()
                project.pages = {'page-1': [block]}
                project._image_info = {'page-1': {'finish_code': 0}}
                project.current_img = 'page-1'
                project.img_array = np.zeros((12, 12, 3), dtype=np.uint8)
                project.mask_array = None
                project.mark_translation_finished('page-1', 'English')
                if saved_target is None:
                    project._image_info['page-1'].pop('translation_target')
                else:
                    project._image_info['page-1'][
                        'translation_target'
                    ] = saved_target
                thread.imgtrans_proj = project

                thread._blktrans_pipeline(
                    [block],
                    -1,
                    [0],
                    page_key='page-1',
                )

                info = project._image_info['page-1']
                self.assertEqual(block.translation, 'ａ! hero')
                self.assertEqual(
                    translator.calls,
                    [([block], project, 'page-1', False)],
                )
                self.assertTrue(info['finish_code'] & RunStatus.FIN_TRANSLATE)
                self.assertEqual(info['translation_target'], 'English')

    def test_selected_translation_marks_only_complete_page(self):
        for remaining_translation, expected_complete in (
            ('', False),
            ('done', True),
        ):
            with self.subTest(remaining_translation=remaining_translation):
                translator = SuccessfulTranslator()
                thread = module_manager.ImgtransThread(
                    None,
                    None,
                    FakeTranslateThread(translator),
                    None,
                )
                selected = TextBlock(text=['selected'])
                remaining = TextBlock(
                    text=['remaining'],
                    translation=remaining_translation,
                )
                empty_source = TextBlock(text=[''], translation='')
                project = ProjImgTrans()
                project.pages = {
                    'page-1': [selected, remaining, empty_source],
                }
                project._image_info = {'page-1': {'finish_code': 0}}
                project.current_img = 'page-1'
                project.img_array = np.zeros((12, 12, 3), dtype=np.uint8)
                project.mask_array = None
                thread.imgtrans_proj = project

                thread._blktrans_pipeline(
                    [selected],
                    -1,
                    [0],
                    page_key='page-1',
                )

                info = project._image_info['page-1']
                self.assertIs(
                    bool(info['finish_code'] & RunStatus.FIN_TRANSLATE),
                    expected_complete,
                )
                if expected_complete:
                    self.assertEqual(info['translation_target'], 'English')
                else:
                    self.assertNotIn('translation_target', info)

    def test_detection_failure_still_invalidates_translation(self):
        block = TextBlock(text=['source'], translation='old')
        project = ProjImgTrans()
        project.pages = {'page-1': [block]}
        project._image_info = {'page-1': {'finish_code': 0}}
        project.read_img = lambda _page: np.zeros(
            (12, 12, 3),
            dtype=np.uint8,
        )
        project.mark_translation_finished('page-1', 'English')
        ocr_thread = SimpleNamespace(ocr=None)
        inpaint_thread = SimpleNamespace(inpainter=None)
        thread = module_manager.ImgtransThread(
            FakeTextDetectThread(FailingDetector()),
            ocr_thread,
            FakeTranslateThread(None),
            inpaint_thread,
        )
        thread.imgtrans_proj = project
        thread.process_idx_to_page_idx = {}
        old_stages = [pcfg.module.stage_enabled(index) for index in range(4)]
        try:
            for index in range(4):
                pcfg.module.set_stage_enabled(index, index == 0)
            with mock.patch(
                'ballontranslator.ui.module_manager.create_error_dialog',
            ) as show_error:
                thread._imgtrans_pipeline()
        finally:
            for index, enabled in enumerate(old_stages):
                pcfg.module.set_stage_enabled(index, enabled)

        info = project._image_info['page-1']
        self.assertFalse(info['finish_code'] & RunStatus.FIN_TRANSLATE)
        self.assertNotIn('translation_target', info)
        self.assertIn('Page: page-1', show_error.call_args.args[1])

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
