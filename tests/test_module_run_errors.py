import unittest
from unittest.mock import Mock, patch

import numpy as np

from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.ocr.base import OCRBase, postprocess_ocr_text
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError
from ballontranslator.modules.textdetector.base import TextDetectorBase
from ballontranslator.ui import module_manager
from ballontranslator.utils import global_callbacks
from ballontranslator.utils.config import ModuleConfig, OCRTextPostprocess, RunStatus, pcfg
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
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

    def test_empty_ocr_restoration_skips_unusable_regions_and_restores_masked_pixels(self) -> None:
        config = ModuleConfig()
        for index in range(4):
            config.set_stage_enabled(index, index == 1)
        config.ocr_font_detect = False
        config.ocr_text_postprocess = OCRTextPostprocess.NONE
        image = np.full((16, 16, 3), 23, dtype=np.uint8)
        original_mask = np.zeros((16, 16), dtype=np.uint8)
        original_mask[3:5, 3:5] = 255
        original_mask[10:12, 10:12] = 255

        for xyxy, angle in (([24, 24, 28, 28], 0), ([24, 24, 28, 28], 45), ([0, 0, 1, 4], 0)):
            with self.subTest(xyxy=xyxy, angle=angle), \
                    patch.object(module_manager, 'cfg_module', config), \
                    patch.object(pcfg, 'module', config), \
                    patch.object(pcfg, 'restore_ocr_empty', True), \
                    patch.dict(global_callbacks._REGISTERED_CALLBACKS, clear=True):
                project = ProjImgTrans()
                x1, y1, x2, y2 = xyxy
                project.pages = {
                    'outside.png': [TextBlock(
                        xyxy=xyxy, angle=angle, text=['old'],
                        lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                    )],
                    'inside.png': [TextBlock(
                        xyxy=[2, 2, 8, 8], text=['old'],
                        lines=[[[2, 2], [8, 2], [8, 8], [2, 8]]],
                    )],
                }
                project._image_info = {key: {'finish_code': 0} for key in project.pages}
                masks = {key: original_mask.copy() for key in project.pages}
                inpainted = {key: np.full_like(image, 99) for key in project.pages}
                project.read_img = Mock(return_value=image)
                project.load_mask_by_imgname = Mock(side_effect=masks.get)
                project.load_inpainted_by_imgname = Mock(side_effect=inpainted.get)
                project.save_mask = Mock()
                project.save_inpainted = Mock()
                workers = (
                    module_manager.TextDetectThread(), module_manager.OCRThread(),
                    module_manager.TranslateThread(), module_manager.InpaintThread(),
                )
                for worker in workers:
                    self.addCleanup(worker.deleteLater)
                ocr = workers[1].module = OCRBase()
                thread = module_manager.ImgtransThread(*workers)
                self.addCleanup(thread.deleteLater)
                with patch.object(ocr, 'ocr_img', return_value=''), \
                        patch.object(thread, 'start', side_effect=thread.run), \
                        patch.object(module_manager, 'create_error_dialog') as show_error:
                    thread.runImgtransPipeline(project)

                show_error.assert_not_called()
                self.assertEqual(thread.ocr_counter, 2)
                for key in project.pages:
                    self.assertEqual(project.pages[key], [])
                    self.assertTrue(project._image_info[key]['finish_code'] & RunStatus.FIN_OCR)
                np.testing.assert_array_equal(masks['outside.png'], original_mask)
                np.testing.assert_array_equal(inpainted['outside.png'], np.full_like(image, 99))
                expected_mask = original_mask.copy()
                expected_mask[2:8, 2:8] = 0
                expected_image = np.full_like(image, 99)
                expected_image[3:5, 3:5] = image[3:5, 3:5]
                np.testing.assert_array_equal(masks['inside.png'], expected_mask)
                np.testing.assert_array_equal(inpainted['inside.png'], expected_image)
                project.save_mask.assert_called_once()
                project.save_inpainted.assert_called_once()
                self.assertEqual(project.save_mask.call_args.args[0], 'inside.png')
                self.assertEqual(project.save_inpainted.call_args.args[0], 'inside.png')


if __name__ == '__main__':
    unittest.main()
