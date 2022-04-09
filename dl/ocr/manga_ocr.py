# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py
import re
import jaconv
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import numpy as np
import torch

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