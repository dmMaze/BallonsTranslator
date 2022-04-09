from typing import Tuple, List, Dict, Union
import numpy as np
import cv2

from ..textdetector.textblock import TextBlock

from utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ..moduleparamparser import ModuleParamParser, DEFAULT_DEVICE

class OCRBase(ModuleParamParser):

    def __init__(self, **setup_params) -> None:
        super().__init__(**setup_params)
        self.name = ''
        for key in OCR.module_dict:
            if OCR.module_dict[key] == self.__class__:
                self.name = key
                break
        self.setup_ocr()

    def setup_ocr(self):
        raise NotImplementedError

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None) -> Union[List[TextBlock], str]:
        if blk_list is None:
            return self.ocr_img(img)
        elif isinstance(blk_list, TextBlock):
            blk_list = [blk_list]
        return self.ocr_blk_list(blk_list)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        raise NotImplementedError

    def ocr_img(self, img: np.ndarray) -> str:
        raise NotImplementedError


from .model_32px import OCR32pxModel
OCR32PXMODEL: OCR32pxModel = None
OCR32PXMODEL_PATH = r'data/models/mit32px_ocr.ckpt'

def load_32px_model(model_path, device, chunk_size=16) -> OCR32pxModel:
    model = OCR32pxModel(model_path, device, max_chunk_size=chunk_size)
    return model

@register_OCR('mit32px')
class OCRMIT32px(OCRBase):
    setup_params = {
        'chunk_size': {
            'type': 'selector',
            'options': [
                8,
                16,
                24,
                32
            ],
            'select': 16
        },
        'device': {
            'type': 'selector',
            'options': [
                'cpu',
                'cuda'
            ],
            'select': DEFAULT_DEVICE
        },
        'description': 'OCRMIT32px'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    def setup_ocr(self):
        
        global OCR32PXMODEL
        self.device = self.setup_params['device']['select']
        self.chunk_size = int(self.setup_params['chunk_size']['select'])
        if OCR32PXMODEL is None:
            self.model = OCR32PXMODEL = \
                load_32px_model(OCR32PXMODEL_PATH, self.device, self.chunk_size)
        else:
            self.model = OCR32PXMODEL
            self.model.to(self.device)
            self.model.max_chunk_size = self.chunk_size

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model.ocr_img(img)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.setup_params['device']['select']
        chunk_size = int(self.setup_params['chunk_size']['select'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size



from .manga_ocr import MangaOcr
MANGA_OCR_MODEL: MangaOcr = None
def load_manga_ocr(device='cpu') -> MangaOcr:
    manga_ocr = MangaOcr(device=device)
    return manga_ocr

@register_OCR('manga_ocr')
class MangaOCR(OCRBase):
    setup_params = {
        'device': {
            'type': 'selector',
            'options': [
                'cpu',
                'cuda'
            ],
            'select': DEFAULT_DEVICE
        }
    }
    device = DEFAULT_DEVICE

    def setup_ocr(self):
        
        global MANGA_OCR_MODEL
        self.device = self.setup_params['device']['select']
        if MANGA_OCR_MODEL is None:
            self.model = MANGA_OCR_MODEL = load_manga_ocr(self.device)
        else:
            self.model = MANGA_OCR_MODEL
            self.model.to(self.device)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model(img)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            blk.text = self.model(img[y1:y2, x1:x2])

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.setup_params['device']['select']
        if self.device != device:
            self.model.to(device)
    


    

    

    
