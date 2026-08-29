from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    PreTrainedTokenizerFast,
    Siglip2ImageProcessor,
)

from .base import DEVICE_SELECTOR, OCRBase, register_OCR


MODEL_PATH = "data/models/hayai-ocr-v2"
VISION_MODEL_ID = "google/siglip2-base-patch16-naflex"


@register_OCR("hayai_ocr_v2")
class HayaiOCRV2(OCRBase):
    """Crop-level Hayai OCR v2 backend.

    >>> HayaiOCRV2.params["max_num_patches"]["value"]
    256
    """

    dependencies = ["torch", "transformers==4.57.6"]
    params = {
        "max_num_patches": {
            "type": "selector",
            "options": [256, 384, 512],
            "value": 256,
            "display_name": "Max Num Patches",
            "description": (
                "Maximum image patches. Use 256 for standard lines, or 384/512 "
                "for dense panels."
            ),
        },
        "device": DEVICE_SELECTOR(),
        "description": "Hayai OCR v2 crop recognition model.",
    }
    download_file_list = [{
        "url": (
            "https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2/resolve/"
            "3608bb2075b9b39cb9f63e57251bca665de248cd/"
        ),
        "save_dir": MODEL_PATH,
        "files": [
            "config.json",
            "configuration_hayai.py",
            "model.safetensors",
            "modeling_hayai.py",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "sha256_pre_calculated": [
            "581b762f1dfd55d0108f3f84e3f157bc762524af37fb0c19a7172a18b75582e2",
            "47abd38cf1bae7aef27d01f5b8b4aa0960a7bc625a8afad79c4762ff5e5ed970",
            "4c645b221db8428cda04991be234c18133bb8861142a3d87cba04c5099b02328",
            "3d78976206549964abd55f776ab059e002adc72d2167daf168e46a12a5f4ae62",
            "f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13",
            "6fb6c69afaedf1275872d3e62e276fd4467bd00da7a84cbbb5566a2cd28f58f6",
        ],
        "concatenate_url_filename": 1,
    }]
    _load_model_keys = {"model", "processor", "tokenizer"}

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model = None
        self.processor = None
        self.tokenizer = None

    @property
    def device(self) -> str:
        return self.get_param_value("device")

    @property
    def max_num_patches(self) -> int:
        value = self.get_param_value("max_num_patches")
        if value <= 0:
            raise ValueError("max_num_patches must be greater than zero")
        return value

    def _load_model(self) -> None:
        processor = Siglip2ImageProcessor.from_pretrained(VISION_MODEL_ID)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
        # Hayai defines custom block-causal attention and 2D mRoPE classes.
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_safetensors=True,
            local_files_only=True,
        ).to(self.device).eval()

        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model

    def ocr_img(self, img: np.ndarray, **kwargs: Any) -> str:
        image = Image.fromarray(img).convert("RGB")
        inputs = self.processor(
            images=[image],
            max_num_patches=self.max_num_patches,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            texts = self.model.generate(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
                tokenizer=self.tokenizer,
                max_new_tokens=128,
                repetition_penalty=1.0,
            )
        return texts[0]

    def updateParam(self, param_key: str, param_content: Any) -> None:
        if (
            param_key == "device"
            and self.model is not None
            and self.device != param_content
        ):
            self.model.to(param_content)
        super().updateParam(param_key, param_content)
