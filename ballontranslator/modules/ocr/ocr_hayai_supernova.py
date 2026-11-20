from copy import deepcopy

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor, Siglip2VisionConfig

from .base import DEVICE_SELECTOR, OCRBase, register_OCR


MODEL_PATH = "data/models/hayai-ocr-v2.7-supernova"


@register_OCR("hayai_ocr_v2_7_supernova")
class HayaiOCRSupernova(OCRBase):
    """Crop recognition with Hayai OCR v2.7 Supernova's native processor.

    >>> HayaiOCRSupernova.params["max_patches"]["value"]
    256
    """

    dependencies = ["torch", "transformers>=5,<6"]
    params = {
        "max_patches": {
            "type": "selector",
            "options": [256, 384, 512],
            "value": 256,
            "display_name": "Max Patches",
            "description": (
                "256 is recommended. Use 384 or 512 for unusually dense "
                "multi-line blocks or tiny glyphs; larger budgets are slower."
            ),
        },
        "max_new_tokens": {
            "value": 128,
            "display_name": "Max New Tokens",
            "description": "Maximum output tokens per crop. Recommended: 128.",
        },
        "use_cuda_graph": {
            "type": "checkbox",
            "value": True,
            "display_name": "Use CUDA Graphs",
            "description": (
                "Enable CUDA graphs for faster decoding (recommended). "
                "Automatically ignored on non-CUDA devices."
            ),
        },
        "device": DEVICE_SELECTOR(),
        "description": (
            "Reads text in English, Korean, Japanese, and Chinese."
        ),
    }
    download_file_list = [{
        "url": (
            "https://huggingface.co/JustANormalTinkerer/"
            "hayai-ocr-v2.7-supernova/resolve/"
            "517bdeeb64b68c08939e12f7211e99943c1e1640/"
        ),
        "save_dir": MODEL_PATH,
        "files": [
            "config.json",
            "configuration_hayai.py",
            "model.safetensors",
            "modeling_hayai.py",
            "preprocessor_config.json",
            "processing_hayai.py",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "sha256_pre_calculated": [
            "e5cf0e59b888ef29155edb40b28f42184b1e22523fca3b38ba62261e743cb91b",
            "bbff66c6a1bb7ee2f8097359992546e1c440b65285c608c22c41b204d13927fe",
            "c7dd04f7138e74abb48ea0dcf6bad7f2bdedd704fe5f8ec6328962e799c65837",
            "cb78b50775a049e066ceb501667228c471f10365d66fb3f0de019b76ca3017a6",
            "bd64b7f81adb30a4d6ab4a24cda78ddad3eced0c1dfa6c58d87ed10dd069de95",
            "047db0d41055d44322a0e446d3fa8aa3cfb5525ef0d59351e8e8fa2e1443b484",
            "0da16f576f41bc6ab34f456d2cc2a9ef89b9076d68b9e86b612bfb94da5478cb",
            "f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13",
            "6fb6c69afaedf1275872d3e62e276fd4467bd00da7a84cbbb5566a2cd28f58f6",
        ],
        "concatenate_url_filename": 1,
    }]
    _load_model_keys = {"model", "processor"}

    def __init__(self, **params: object) -> None:
        self.params = deepcopy(self.params)
        self.model = None
        self.processor = None
        self.device = None
        clean = {}
        for key, content in params.items():
            if key.startswith("__") or key == "description":
                continue
            value = content.get("value", content) if isinstance(content, dict) else content
            try:
                clean[key] = self._validate_param(key, value)
            except (ValueError, TypeError, OverflowError, RuntimeError) as error:
                self.logger.warning("Discard invalid Supernova setting %r: %s", key, error)
        super().__init__(**clean)

    def _validate_param(self, key: str, value: object) -> object:
        if key not in self.params or key.startswith("__") or key == "description":
            raise ValueError(f"Unknown setting: {key}")
        if key == "device":
            if not isinstance(value, str):
                raise ValueError("device must be a PyTorch device name")
            torch.device(value)
            return value
        if key == "use_cuda_graph":
            if not isinstance(value, bool):
                raise ValueError("use_cuda_graph must be true or false")
            return value
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(f"{key} must be a positive integer")
        number = int(value)
        if number <= 0:
            raise ValueError(f"{key} must be a positive integer")
        if key == "max_patches" and number not in self.params[key]["options"]:
            raise ValueError("max_patches must be 256, 384 or 512")
        return number

    def _load_model(self) -> None:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True,
        )
        # The checkpoint omits the standard SigLIP2 base config; avoid a Hub lookup.
        config.vision_config = Siglip2VisionConfig().to_dict()
        device = self.get_param_value("device")
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            config=config,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
        ).to(device).eval()
        self.processor = processor
        self.model = model
        self.device = device

    def ocr_img(self, img: np.ndarray, **kwargs: object) -> str:
        device = self.get_param_value("device")
        if self.device != device:
            # Device transfers belong to the OCR worker, not the settings slot.
            self.model.to(device)
            self.device = device
        with torch.inference_mode():
            return self.model.ocr(
                img,
                self.processor,
                max_patches=self.get_param_value("max_patches"),
                max_new_tokens=self.get_param_value("max_new_tokens"),
                use_cuda_graph=self.get_param_value("use_cuda_graph"),
            )

    def updateParam(self, param_key: str, param_content: object) -> None:
        try:
            value = self._validate_param(param_key, param_content)
        except (ValueError, TypeError, OverflowError, RuntimeError) as error:
            # Reject invalid edits without raising through a Qt settings slot.
            self.logger.warning("Rejected Supernova setting %r: %s", param_key, error)
            return
        super().updateParam(param_key, value)
