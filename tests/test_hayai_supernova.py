import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from ballontranslator.modules import OCR
from ballontranslator.modules.exceptions import ModuleRunError
from ballontranslator.modules.lazy_registry import _scan_file, validate_lazy_module_specs
from ballontranslator.modules.ocr import ocr_hayai_supernova
from ballontranslator.modules.ocr.ocr_hayai_supernova import HayaiOCRSupernova
from ballontranslator.utils.config import OCRTextPostprocess, pcfg
from ballontranslator.utils.textblock import TextBlock


class HayaiSupernovaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Mock(spec=["to", "eval", "ocr"])
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.model.ocr.return_value = "Hello 日本語"
        self.processor = object()
        self.config = SimpleNamespace()
        for name, target, value in (
            ("load_model", ocr_hayai_supernova.AutoModel, self.model),
            ("load_processor", ocr_hayai_supernova.AutoProcessor, self.processor),
            ("load_config", ocr_hayai_supernova.AutoConfig, self.config),
        ):
            patcher = patch.object(target, "from_pretrained", return_value=value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        for target, key, value in (
            (pcfg.module, "ocr_font_detect", False),
            (pcfg.module, "ocr_text_postprocess", OCRTextPostprocess.NONE),
            (pcfg, "ocr_sublist", []),
        ):
            patcher = patch.object(target, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ocr = HayaiOCRSupernova(device="cpu")
        self.addCleanup(self.ocr.unload_model)
        self.img = np.zeros((20, 30, 3), dtype=np.uint8)
        self.img[..., 0] = 255

    def test_registry_metadata_is_lazy_and_preserves_existing_backend(self) -> None:
        specs = _scan_file("ballontranslator/modules/ocr/ocr_hayai_supernova.py", "ocr")
        self.assertEqual(validate_lazy_module_specs(specs), [])
        self.assertEqual(specs[0].key, "hayai_ocr_v2_7_supernova")
        self.assertIsNotNone(OCR.get_spec(specs[0].key))
        self.assertIsNotNone(OCR.get_spec("hayai_ocr_v2"))
        self.assertEqual(specs[0].params["max_patches"]["value"], 256)
        self.assertEqual(specs[0].params["max_new_tokens"]["value"], 128)
        self.assertIs(specs[0].params["use_cuda_graph"]["value"], True)
        self.load_model.assert_not_called()
        self.load_processor.assert_not_called()
        self.load_config.assert_not_called()

    def test_loads_on_first_use_and_passes_live_settings(self) -> None:
        def recognize(image: np.ndarray, processor: object, **kwargs: object) -> str:
            self.assertTrue(torch.is_inference_mode_enabled())
            self.assertIs(image, self.img)
            self.assertIs(processor, self.processor)
            return "Hello 日本語"

        self.model.ocr.side_effect = recognize
        self.assertEqual(self.ocr.run_ocr(self.img), "Hello 日本語")
        self.load_model.assert_called_once_with(
            ocr_hayai_supernova.MODEL_PATH,
            config=self.config,
            trust_remote_code=True, use_safetensors=True, local_files_only=True,
        )
        self.load_processor.assert_called_once_with(
            ocr_hayai_supernova.MODEL_PATH,
            trust_remote_code=True, local_files_only=True,
        )
        self.load_config.assert_called_once_with(
            ocr_hayai_supernova.MODEL_PATH,
            trust_remote_code=True, local_files_only=True,
        )
        self.assertEqual(self.config.vision_config["hidden_size"], 768)
        self.assertEqual(self.config.vision_config["patch_size"], 16)
        self.model.eval.assert_called_once()
        self.assertEqual(self.model.ocr.call_args.kwargs, {
            "max_patches": 256, "max_new_tokens": 128, "use_cuda_graph": True,
        })
        for key, value in (("max_patches", "512"), ("max_new_tokens", "256"), ("use_cuda_graph", False)):
            self.ocr.updateParam(key, value)
        self.ocr.run_ocr(self.img)
        self.assertEqual(self.model.ocr.call_args.kwargs, {
            "max_patches": 512, "max_new_tokens": 256, "use_cuda_graph": False,
        })
        self.load_model.assert_called_once()
        self.model.to.assert_called_once_with("cpu")

    def test_block_crops_and_rgba_follow_the_shared_pipeline(self) -> None:
        blocks = [
            TextBlock(xyxy=[-2, 2, 12, 14], text=["old"]),
            TextBlock(xyxy=[30, 20, 40, 30], text=["outside"]),
        ]
        rgba = np.concatenate([self.img, np.full((20, 30, 1), 255, dtype=np.uint8)], axis=2)
        self.assertIs(self.ocr.run_ocr(rgba, blocks), blocks)
        self.assertEqual(blocks[0].text, "Hello 日本語")
        self.assertEqual(blocks[1].text, "")
        np.testing.assert_array_equal(self.model.ocr.call_args.args[0], self.img[2:14, :12])
        self.model.ocr.assert_called_once()

    def test_failure_preserves_all_existing_block_text(self) -> None:
        blocks = [
            TextBlock(xyxy=[0, 0, 10, 10], text=["first"]),
            TextBlock(xyxy=[10, 0, 20, 10], text=["second"]),
        ]
        self.model.ocr.side_effect = ["partial result", RuntimeError("inference failed")]
        with self.assertRaisesRegex(ModuleRunError, "inference failed"):
            self.ocr.run_ocr(self.img, blocks)
        self.assertEqual([block.text for block in blocks], [["first"], ["second"]])

    def test_failed_load_can_retry_and_unload_releases_both_objects(self) -> None:
        self.load_model.side_effect = RuntimeError("missing weights")
        with self.assertRaisesRegex(RuntimeError, "missing weights"):
            self.ocr.load_model()
        self.assertFalse(self.ocr.all_model_loaded())
        self.assertIsNone(self.ocr.processor)
        self.load_model.side_effect = None
        self.ocr.load_model()
        self.assertTrue(self.ocr.all_model_loaded())
        self.assertTrue(self.ocr.unload_model())
        self.assertIsNone(self.ocr.model)
        self.assertIsNone(self.ocr.processor)

    def test_device_transfer_waits_for_ocr_and_runs_once(self) -> None:
        self.ocr.load_model()
        self.model.to.reset_mock()
        self.ocr.updateParam("device", "cuda")
        self.model.to.assert_not_called()
        self.ocr.run_ocr(self.img)
        self.model.to.assert_called_once_with("cuda")
        self.ocr.run_ocr(self.img)
        self.model.to.assert_called_once_with("cuda")

    def test_saved_settings_discard_only_invalid_values(self) -> None:
        saved = {
            "max_patches": {"value": 7},
            "max_new_tokens": {"value": 64},
            "use_cuda_graph": {"value": "false"},
            "device": {"value": "cpu"},
            "obsolete": 1,
            "__param_patched": True,
        }
        original = copy.deepcopy(saved)
        with patch.object(HayaiOCRSupernova.logger, "warning") as warning:
            ocr = HayaiOCRSupernova(**saved)
        self.addCleanup(ocr.unload_model)
        self.assertEqual(ocr.get_param_value("max_patches"), 256)
        self.assertEqual(ocr.get_param_value("max_new_tokens"), 64)
        self.assertIs(ocr.get_param_value("use_cuda_graph"), True)
        self.assertEqual(warning.call_count, 3)
        self.assertEqual(saved, original)
        self.assertEqual(self.ocr.get_param_value("max_new_tokens"), 128)
        self.assertEqual(HayaiOCRSupernova.params["max_new_tokens"]["value"], 128)

    def test_invalid_live_settings_leave_previous_values_intact(self) -> None:
        for key, value in (
            ("max_patches", 1024), ("max_patches", True),
            ("max_new_tokens", 0), ("max_new_tokens", -1),
            ("max_new_tokens", 1.5), ("max_new_tokens", "invalid"),
            ("max_new_tokens", float("inf")), ("max_new_tokens", None),
            ("use_cuda_graph", "false"), ("device", "invalid_device"),
            ("unknown", 1),
        ):
            with self.subTest(key=key, value=value), patch.object(HayaiOCRSupernova.logger, "warning") as warning:
                before = copy.deepcopy(self.ocr.params)
                self.ocr.updateParam(key, value)
                self.assertEqual(self.ocr.params, before)
                warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
