from transformers import AutoModelForCausalLM, AutoProcessor
import numpy as np
import torch
from typing import List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

MODEL_PATH = 'data/models/PaddleOCR-VL-For-Manga'


@register_OCR('PaddleOCRVLManga')
class PaddleOCRVLManga(OCRBase):
    params = {
        'device': DEVICE_SELECTOR(),
        "max_new_tokens": {
            "value": 512,
            "description": "Max generation tokens"
        }
    }
    device = DEFAULT_DEVICE

    download_file_list = [{
        'url': 'https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga/resolve/main/',
        'save_dir': 'data/models/PaddleOCR-VL-For-Manga',
        'files': [
            'generation_config.json',
            'modeling_paddleocr_vl.py',
            'model.safetensors',
            'preprocessor_config.json',
            'tokenizer_config.json',
            'added_tokens.json',
            'image_processing.py',
            'tokenizer.model',
            'special_tokens_map.json',
            'tokenizer.json',
            'processor_config.json',
            'chat_template.jinja',
            'configuration_paddleocr_vl.py',
            'config.json',
            'processing_paddleocr_vl.py',
            'README.md'
        ],
        'sha256_pre_calculated': [
            'cf202f984e003e92dceaa27e749b60b4e6e1b566a1df8486b5b41adf1d016cea',
            '351269f534882b2df200192adc3af6117910e9a7446caf0d6706366b1ed45d9f',
            '71fcee0e3618582d4c8acc705242aa79b471b6134e7023bf3820642ba638b602',
            'f417a7f977820dfe6828f3ec2e461c027fdb0662f25cae4e841ec1028e0b988a',
            '67c651ba09c22151a1fff31e8773a24f7607aef1541aa2f200b48552ed30e894',
            'f59f889088e0fe21c523e7cf121bb6dca3b0bb148cb7159fbb4572c74dfc5644',
            '242b36a14461d81fba2913a2e736d9f7a26422c6f2cdf484a5e5a17037f78147',
            '34ef7db83df785924fb83d7b887b6e822a031c56e15cff40aaf9b982988180df',
            '215bf3a1b155fafe3497f8790bedf280af92d29c2f0286c2f87a5c78baff8f7c',
            'f90f04fd8e5eb6dfa380f37d10c87392de8438dccb6768a2486b5a96ee76dba6',
            '1568858960a9760c54431dae693a6152e601ff55cdf6d2eab97a4a99958faea0',
            '344fea8b69546526a00996468f86f583fd65441582a36f2fa4abc794aa94094c',
            '753dd93654c3a9c8c85a3eaee1e3092dd12591b0f2dce0305e1abfb7a41ff160',
            '928aaf78567a273cb73ede3671253ec4e38eb60c27a30e945bcd13b4131a0147',
            'e29cb1e5f275f2bd3ce051bd5c9983a33894e693b2823a0e13d4c07c8c4f9e13',
            None
        ],
        'concatenate_url_filename': 1
    }]
    _load_model_keys = {'model', 'processor'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model = None
        self.processor = None

    def ocr_img(self, img: np.ndarray) -> str:
        # Prepare the prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "OCR:"},
                ],
            }
        ]

        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        # Generate text
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.get_param_value('max_new_tokens'),
                do_sample=False,
                use_cache=True
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]
        answer = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return answer.split('\n')

    def _load_model(self):
        if self.model is None:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device).eval()

            processor = AutoProcessor.from_pretrained(
                MODEL_PATH, trust_remote_code=True, use_fast=True
            )

            # Set pad_token_id to avoid warning during generation
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

            self.model = model
            self.processor = processor

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                # Extract region and convert RGBA to RGB if necessary for model input
                region = img[y1:y2, x1:x2]
                blk.text = self.ocr_img(region)
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)


