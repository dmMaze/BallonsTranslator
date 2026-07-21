import unittest
from unittest.mock import patch

import numpy as np

from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.ocr.base import OCRBase, postprocess_ocr_text
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError
from ballontranslator.modules.textdetector.base import TextDetectorBase
from ballontranslator.ui import module_manager
from ballontranslator.utils.config import ModuleConfig, OCRTextPostprocess, pcfg
from ballontranslator.utils.registry import ModuleSpec
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.text_processing import capitalize_sentences


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


if __name__ == '__main__':
    unittest.main()
