from typing import Tuple, List, Dict, Union, Callable
from ordered_set import OrderedSet
import numpy as np
import logging

from ..textdetector.textblock import TextBlock

from utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ..base import BaseModule, DEFAULT_DEVICE

class OCRBase(BaseModule):

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in OCR.module_dict:
            if OCR.module_dict[key] == self.__class__:
                self.name = key
                break
        self.postprocess_hooks: OrderedSet[Callable] = OrderedSet()
        self.setup_ocr()

    def setup_ocr(self):
        raise NotImplementedError

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None) -> Union[List[TextBlock], str]:
        if blk_list is None:
            text = self.ocr_img(img)
            for callback in self.postprocess_hooks:
                text = callback(text)
            return text
        elif isinstance(blk_list, TextBlock):
            blk_list = [blk_list]

        for blk in blk_list:
            blk.text = []
        self.ocr_blk_list(img, blk_list)
        for blk in blk_list:
            if isinstance(blk.text, List):
                for ii, t in enumerate(blk.text):
                    for callback in self.postprocess_hooks:
                        blk.text[ii] = callback(t, blk=blk)
            else:
                for callback in self.postprocess_hooks:
                    blk.text = callback(blk.text, blk=blk)
        return blk_list

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]) -> None:
        raise NotImplementedError

    def ocr_img(self, img: np.ndarray) -> str:
        raise NotImplementedError

    def register_postprocess_hooks(self, callbacks: Union[List, Callable]):
        if callbacks is None:
            return
        if isinstance(callbacks, Callable):
            callbacks = [callbacks]
        for callback in callbacks:
            self.postprocess_hooks.add(callback)


from .model_32px import OCR32pxModel
OCR32PXMODEL: OCR32pxModel = None
OCR32PXMODEL_PATH = r'data/models/mit32px_ocr.ckpt'

def load_32px_model(model_path, device, chunk_size=16) -> OCR32pxModel:
    model = OCR32pxModel(model_path, device, max_chunk_size=chunk_size)
    return model

@register_OCR('mit32px')
class OCRMIT32px(OCRBase):
    params = {
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
                'cuda',
            ],
            'select': DEFAULT_DEVICE
        },
        'description': 'OCRMIT32px'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    def setup_ocr(self):
        
        global OCR32PXMODEL
        self.device = self.params['device']['select']
        self.chunk_size = int(self.params['chunk_size']['select'])
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
        device = self.params['device']['select']
        chunk_size = int(self.params['chunk_size']['select'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size




MANGA_OCR_MODEL = None
@register_OCR('manga_ocr')
class MangaOCR(OCRBase):
    params = {
        'device': {
            'type': 'selector',
            'options': [
                'cpu',
                'cuda',
            ],
            'select': DEFAULT_DEVICE
        }
    }
    device = DEFAULT_DEVICE

    def setup_ocr(self):

        from .manga_ocr import MangaOcr
        def load_manga_ocr(device='cpu') -> MangaOcr:
            manga_ocr = MangaOcr(device=device)
            return manga_ocr
        
        global MANGA_OCR_MODEL
        self.device = self.params['device']['select']
        if MANGA_OCR_MODEL is None:
            self.model = MANGA_OCR_MODEL = load_manga_ocr(self.device)
        else:
            self.model = MANGA_OCR_MODEL
            self.model.to(self.device)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model(img)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                blk.text = self.model(img[y1:y2, x1:x2])
            else:
                logging.warning('invalid textbbox to target img')
                blk.text = ['']

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['select']
        if self.device != device:
            self.model.to(device)




from .mit48px_ctc import OCR48pxCTC
OCR48PXMODEL: OCR48pxCTC = None
OCR48PXMODEL_PATH = r'data/models/mit48pxctc_ocr.ckpt'

def load_48px_model(model_path, device, chunk_size=16) -> OCR48pxCTC:
    model = OCR48pxCTC(model_path, device, max_chunk_size=chunk_size)
    return model

@register_OCR('mit48px_ctc')
class OCRMIT48pxCTC(OCRBase):
    params = {
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
                'cuda',
            ],
            'select': DEFAULT_DEVICE
        },
        'description': 'mit48px_ctc'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    def setup_ocr(self):
        
        global OCR48PXMODEL
        self.device = self.params['device']['select']
        self.chunk_size = int(self.params['chunk_size']['select'])
        if OCR48PXMODEL is None:
            self.model = OCR48PXMODEL = \
                load_48px_model(OCR48PXMODEL_PATH, self.device, self.chunk_size)
        else:
            self.model = OCR48PXMODEL
            self.model.to(self.device)
            self.model.max_chunk_size = self.chunk_size

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model.ocr_img(img)

    def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['select']
        chunk_size = int(self.params['chunk_size']['select'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size
    
import platform
if platform.mac_ver()[0] >= '10.15':
    from .macos_ocr import get_supported_languages
    APPLEVISIONFRAMEWORK = None
    @register_OCR('macos_ocr')
    class OCRApple(OCRBase):
        params = {
            'language': {
                'type':'selector',
                'options': list(get_supported_languages()[0]),
                'select': 'en-US',
            },
            # While this does appear 
            # it doesn't update the languages available
            # different recog level, different available langs
            # 'recognition_level': {
            #     'type': 'selector',
            #     'options': [
            #         'accurate',
            #         'fast',
            #     ],
            #     'select': 'accurate',
            # },
            'confidence_level': '0.1',
        }
        language = 'en-US'
        recognition = 'accurate'
        confidence = '0.1'

        def setup_ocr(self):
            global APPLEVISIONFRAMEWORK
            from .macos_ocr import AppleOCR
            if APPLEVISIONFRAMEWORK is None:
                self.model = APPLEVISIONFRAMEWORK = AppleOCR(lang=[self.language])
            else:
                self.model = APPLEVISIONFRAMEWORK

        def ocr_img(self, img: np.ndarray) -> str:
            return self.model(img)

        def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
            im_h, im_w = img.shape[:2]
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if y2 < im_h and x2 < im_w and \
                    x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                    blk.text = self.model(img[y1:y2, x1:x2])
                else:
                    logging.warning('invalid textbbox to target img')
                    blk.text = ['']

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            self.language = self.params['language']['select']
            self.model.lang = [self.language]

            # self.recognition = self.params['recognition_level']['select']
            # self.model.recog_level = self.recognition
            # self.params['language']['options'] = list(get_supported_languages(self.recognition)[0])

            self.confidence = self.params['confidence_level']
            self.model.min_confidence = self.confidence

if platform.system() == 'Windows' and platform.version() >= '10.0.10240.0':
    from .windows_ocr import get_supported_language_packs

    languages_display_name = [lang.display_name for lang in get_supported_language_packs()]
    languages_tag = [lang.language_tag for lang in get_supported_language_packs()]
    WINDOWSOCRENGINE = None
    @register_OCR('windows_ocr')
    class OCRWindows(OCRBase):
        params = {
            'language': {
                'type':'selector',
                'options': languages_display_name,
                'select': languages_display_name[0],
            }
        }
        language = languages_display_name[0]

        def setup_ocr(self):
            global WINDOWSOCRENGINE
            from .windows_ocr import WindowsOCR
            if WINDOWSOCRENGINE is None:
                self.engine = WINDOWSOCRENGINE = WindowsOCR()
            else:
                self.engine = WINDOWSOCRENGINE
            self.engine.lang = self.get_engine_lang()

        def get_engine_lang(self) -> str:
            language = self.params['language']['select'] 
            tag_name = languages_tag[languages_display_name.index(language)]
            return tag_name

        def ocr_img(self, img: np.ndarray) -> str:
            self.engine(img)

        def ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]) -> None:
            im_h, im_w = img.shape[:2]
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if y2 < im_h and x2 < im_w and \
                    x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2: 
                    blk.text = self.engine(img[y1:y2, x1:x2])
                else:
                    logging.warning('invalid textbbox to target img')
                    blk.text = ['']
        
        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            self.engine.lang = self.get_engine_lang()