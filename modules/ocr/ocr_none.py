import numpy as np

from .base import OCRBase, register_OCR, List, TextBlock

@register_OCR('none_ocr')
class OCRNone(OCRBase):
    def __init__(self, **params) -> None:
        super().__init__(**params)

    params = {
        'NOTICE': 'Not a OCR, just return original text.',
        'description': 'Not a OCR, just return original text.'
    }

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        pass

    def ocr_img(self, img: np.ndarray) -> str:
        return ''