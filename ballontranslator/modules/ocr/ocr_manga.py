# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py
import re
import jaconv
from transformers import AutoFeatureExtractor, AutoTokenizer, LogitsProcessor, LogitsProcessorList, VisionEncoderDecoderModel
import numpy as np
import torch
from typing import List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

MANGA_OCR_PATH = r'data/models/manga-ocr-base'


class VectorizedNoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Tensor-only ``no_repeat_ngram_size`` blocking, backported from transformers main.

    The pinned transformers version builds Python dicts from ``.tolist()`` at
    every generation step, which forces a device sync per token. This bans the
    same tokens with tensor ops only: token t is banned for a row exactly when
    some earlier n-gram in that row starts with the row's last n-1 tokens and
    ends with t.

    >>> p = VectorizedNoRepeatNGramLogitsProcessor(2)
    >>> torch.isinf(p(torch.tensor([[1, 2, 1]]), torch.zeros(1, 4))).nonzero().tolist()
    [[0, 2]]
    """

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.ngram_size:
            return scores
        prefix = input_ids[:, cur_len + 1 - self.ngram_size:]
        windows = input_ids.unfold(dimension=1, size=self.ngram_size, step=1)
        matches = (windows[..., :-1] == prefix.unsqueeze(1)).all(dim=-1)
        vocab_size = scores.shape[-1]
        # Non-matching windows scatter into a spare column past the vocabulary.
        banned = scores.new_zeros((scores.shape[0], vocab_size + 1), dtype=torch.bool)
        banned.scatter_(1, torch.where(matches, windows[..., -1].clamp(max=vocab_size), vocab_size), True)
        return scores.masked_fill(banned[:, :vocab_size], -float('inf'))


class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_PATH, device='cpu'):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        # Swap the model's built-in n-gram blocker for the equivalent tensor-only one.
        ngram_size = self.model.generation_config.no_repeat_ngram_size or 0
        self.model.generation_config.no_repeat_ngram_size = 0
        self.logits_processor = LogitsProcessorList([VectorizedNoRepeatNGramLogitsProcessor(ngram_size)] if ngram_size > 0 else [])
        self.to(device)
        
    def to(self, device):
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, img: np.ndarray):
        x = self.feature_extractor(img, return_tensors="pt").pixel_values.squeeze()
        x = self.model.generate(x[None].to(self.model.device), logits_processor=self.logits_processor)[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text


@register_OCR('manga_ocr')
class MangaOCR(OCRBase):
    dependencies = ['torch', 'transformers==4.57.6', 'jaconv', 'fugashi', 'unidic-lite']

    params = {
        'device': DEVICE_SELECTOR()
    }
    device = DEFAULT_DEVICE

    download_file_list = [{
        'url': 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/',
        'files': ['pytorch_model.bin', 'config.json', 'preprocessor_config.json', 'README.md', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.txt'],
        'sha256_pre_calculated': ['c63e0bb5b3ff798c5991de18a8e0956c7ee6d1563aca6729029815eda6f5c2eb', None, None, None, None, None, None],
        'save_dir': 'data/models/manga-ocr-base',
        'concatenate_url_filename': 1,
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model: MangaOCR = None

    def _load_model(self):
        if self.model is None:
            self.model = MangaOcr(device=self.device)

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        return self.model(img)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)