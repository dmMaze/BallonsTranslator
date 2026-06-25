import numpy as np

from .base import OCRBase, register_OCR

@register_OCR('none_ocr')
class OCRNone(OCRBase):
    def __init__(self, **params) -> None:
        super().__init__(**params)

    params = {
        'NOTICE': 'Not a OCR, just return original text.',
        'description': 'Not a OCR, just return original text.'
    }

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        return ''