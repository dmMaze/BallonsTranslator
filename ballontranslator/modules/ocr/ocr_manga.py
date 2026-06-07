# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py
import re
import jaconv
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import numpy as np
import torch
from typing import List

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

MANGA_OCR_PATH = r'data/models/manga-ocr-base'
class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_PATH, device='cpu'):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        self.to(device)
        
    def to(self, device):
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, img: np.ndarray):
        x = self.feature_extractor(img, return_tensors="pt").pixel_values.squeeze()
        x = self.model.generate(x[None].to(self.model.device))[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    # todo
    def ocr_batch(self, im_batch: torch.Tensor):
        raise NotImplementedError


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text


@register_OCR('manga_ocr')
class MangaOCR(OCRBase):
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

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model(img)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                # Extract region and convert RGBA to RGB if necessary for model input
                region = img[y1:y2, x1:x2]
                blk.text = self.model(region)
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device and self.model is not None:
            self.model.to(device)




if __name__ == '__main__':
    import cv2

    img_path = r'data/testpacks/textline/ballontranslator.png'
    manga_ocr = MangaOcr(pretrained_model_name_or_path=MANGA_OCR_PATH, device='cuda')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dummy = np.zeros((1024, 1024, 3), np.uint8)
    manga_ocr(dummy)
    # preprocessed = manga_ocr(img_path)

    # im_batch = 
    # img = (torch.from_numpy(img[np.newaxis, ...]).float() - 127.5) / 127.5
    # img = einops.rearrange(img, 'N H W C -> N C H W')
    import time
    
    for ii in range(10):
        t0 = time.time()
        out = manga_ocr(dummy)
        print(out, time.time() - t0)