from typing import Tuple, List, Dict, Union, Callable
from ordered_set import OrderedSet
import numpy as np
import logging
from collections import OrderedDict

from utils.textblock import TextBlock

from utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR, LOGGER

class OCRBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in OCR.module_dict:
            if OCR.module_dict[key] == self.__class__:
                self.name = key
                break

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None) -> Union[List[TextBlock], str]:

        if not self.all_model_loaded():
            self.load_model()

        if blk_list is None:
            text = self.ocr_img(img)
            return text
        elif isinstance(blk_list, TextBlock):
            blk_list = [blk_list]

        for blk in blk_list:
            if self.name != 'none_ocr':
                blk.text = []
                
        self._ocr_blk_list(img, blk_list)
        for callback_name, callback in self._postprocess_hooks.items():
            callback(textblocks=blk_list, img=img, ocr_module=self)
        # for blk in blk_list:
        #     if isinstance(blk.text, List):
        #         for ii, t in enumerate(blk.text):
        #             for callback in self.postprocess_hooks:
        #                 blk.text[ii] = callback(t, blk=blk)
        #     else:
        #         for callback in self.postprocess_hooks:
        #             blk.text = callback(blk.text, blk=blk)
        return blk_list

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]) -> None:
        raise NotImplementedError

    def ocr_img(self, img: np.ndarray) -> str:
        raise NotImplementedError


from .model_32px import OCR32pxModel
@register_OCR('mit32px')
class OCRMIT32px(OCRBase):
    params = {
        'chunk_size': {
            'type': 'selector',
            'options': [8, 16, 24, 32],
            'value': 16
        },
        'device': DEVICE_SELECTOR(),
        'description': 'OCRMIT32px'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr.zip',
        'files': ['ocr.ckpt'],
        'sha256_pre_calculated': ['d9f619a9dccce8ce88357d1b17d25f07806f225c033ea42c64e86c45446cfe71'],
        'save_files': ['data/models/mit32px_ocr.ckpt'],
        'archived_files': 'ocr.zip',
        'archive_sha256_pre_calculated': '47405638b96fa2540a5ee841a4cd792f25062c09d9458a973362d40785f95d7a',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.chunk_size = int(self.params['chunk_size']['value'])
        self.model: OCR32pxModel = None

    def _load_model(self):
        self.model = OCR32pxModel(r'data/models/mit32px_ocr.ckpt', self.device, self.chunk_size)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model.ocr_img(img)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        chunk_size = int(self.params['chunk_size']['value'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size



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
        from .manga_ocr import MangaOcr
        self.model = MangaOcr(device=self.device)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model(img)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
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
        device = self.params['device']['value']
        if self.device != device:
            self.model.to(device)



from .mit48px_ctc import OCR48pxCTC
@register_OCR('mit48px_ctc')
class OCRMIT48pxCTC(OCRBase):
    params = {
        'chunk_size': {
            'type': 'selector',
            'options': [8,16,24,32],
            'value': 16
        },
        'device': DEVICE_SELECTOR(),
        'description': 'mit48px_ctc'
    }
    device = DEFAULT_DEVICE
    chunk_size = 16

    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip',
        'files': ['ocr-ctc.ckpt', 'alphabet-all-v5.txt'],
        'sha256_pre_calculated': ['8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89', None],
        'save_files': ['data/models/mit48pxctc_ocr.ckpt', 'data/alphabet-all-v5.txt'],
        'archived_files': 'ocr-ctc.zip',
        'archive_sha256_pre_calculated': 'fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.chunk_size = int(self.params['chunk_size']['value'])
        self.model: OCR48pxCTC = None

    def _load_model(self):
        self.model = OCR48pxCTC(r'data/models/mit48pxctc_ocr.ckpt', self.device, self.chunk_size)

    def ocr_img(self, img: np.ndarray) -> str:
        return self.model.ocr_img(img)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        chunk_size = int(self.params['chunk_size']['value'])
        if self.device != device:
            self.model.to(device)
        self.chunk_size = chunk_size
        self.model.max_chunk_size = chunk_size



from .mit48px import Model48pxOCR
OCR48PXMODEL_PATH = r'data/models/ocr_ar_48px.ckpt'
@register_OCR('mit48px')
class OCRMIT48px(OCRBase):
    params = {
        'device': DEVICE_SELECTOR(),
        'description': 'mit48px'
    }
    device = DEFAULT_DEVICE

    download_file_list = [{
        'url': 'https://huggingface.co/zyddnys/manga-image-translator/resolve/main/',
        'files': [OCR48PXMODEL_PATH, 'data/alphabet-all-v7.txt'],
        'sha256_pre_calculated': ['29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d', None],
        'concatenate_url_filename': 2,
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.model: Model48pxOCR = None

    def _load_model(self):
        self.model = Model48pxOCR(OCR48PXMODEL_PATH, self.device)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['value']
        if self.device != device:
            self.model.to(device)

from .stariver_ocr import StariverOCR
@register_OCR('stariver_ocr')
class OCRStariver(OCRBase):
    params = {
        'token': 'Replace with your token',
        "refine":{
            'type': 'checkbox',
            'value': True
        },
        "filtrate":{
            'type': 'checkbox',
            'value': True
        },
        "disable_skip_area":{
            'type': 'checkbox',
            'value': True
        },
        "detect_scale": "3",
        "merge_threshold": "2",
        "force_expand":{
            'type': 'checkbox',
            'value': False,
            'description': '是否强制扩展图片像素，会导致识别速度下降'
        },
        'description': '星河云(团子翻译器) OCR API'
    }

    @property
    def token(self):
        return self.params['token']
    
    @property
    def expand_ratio(self):
        return float(self.params['expand_ratio'])
    
    @property
    def refine(self):
        return  self.params['refine']['value']
     
    @property
    def filtrate(self):
        self.params['filtrate']['value']
        
    @property
    def disable_skip_area(self):
        return self.params['disable_skip_area']['value']

    @property
    def detect_scale(self):
        return int(self.params['detect_scale'])
    
    @property
    def merge_threshold(self):
        return float(self.params['merge_threshold'])
    
    @property
    def force_expand(self):
        return self.params['force_expand']['value']

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.client = StariverOCR(self.token, refine=self.refine, filtrate=self.filtrate, disable_skip_area=self.disable_skip_area, detect_scale=self.detect_scale, merge_threshold=self.merge_threshold, force_expand=self.force_expand)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if y2 < im_h and x2 < im_w and \
                    x1 > 0 and y1 > 0 and x1 < x2 and y1 < y2:
                blk.text = self.client.ocr(img[y1:y2, x1:x2])
            else:
                logging.warning('invalid textbbox to target img')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.client.ocr(img)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.client.token = self.params['token']

@register_OCR('none_ocr')
class OCRNone(OCRBase):
    def __init__(self, **params) -> None:
        super().__init__(**params)

    params = {
        'NOTICE': 'Not a OCR, just return original text.',
        'description': 'Not a OCR, just return original text.'
    }

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        pass

    def ocr_img(self, img: np.ndarray) -> str:
        return ''
    
import platform
if platform.mac_ver()[0] >= '10.15':
    from .macos_ocr import get_supported_languages

    macos_ocr_supported_languages = get_supported_languages()

    if len(macos_ocr_supported_languages) > 0:
        @register_OCR('macos_ocr')
        class OCRApple(OCRBase):
            params = {
                'language': {
                    'type':'selector',
                    'options': list(get_supported_languages()[0]),
                    'value': 'en-US',
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
                #     'value': 'accurate',
                # },
                'confidence_level': '0.1',
            }
            language = 'en-US'
            recognition = 'accurate'
            confidence = '0.1'

            def __init__(self, **params) -> None:
                super().__init__(**params)
                from .macos_ocr import AppleOCR
                self.model = AppleOCR(lang=[self.language])

            def ocr_img(self, img: np.ndarray) -> str:
                return self.model(img)

            def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
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
                self.language = self.params['language']['value']
                self.model.lang = [self.language]

                # self.recognition = self.params['recognition_level']['value']
                # self.model.recog_level = self.recognition
                # self.params['language']['options'] = list(get_supported_languages(self.recognition)[0])

                self.confidence = self.params['confidence_level']
                self.model.min_confidence = self.confidence
    else:
        LOGGER.warning(f'No supported language packs found for MacOS, MacOS OCR will be unavailable.')
                

if platform.system() == 'Windows' and platform.version() >= '10.0.10240.0':
    from .windows_ocr import winocr_available_recognizer_languages

    if len(winocr_available_recognizer_languages) > 0:

        languages_display_name = [lang.display_name for lang in winocr_available_recognizer_languages]
        languages_tag = [lang.language_tag for lang in winocr_available_recognizer_languages]
        @register_OCR('windows_ocr')
        class OCRWindows(OCRBase):
            params = {
                'language': {
                    'type':'selector',
                    'options': languages_display_name,
                    'value': languages_display_name[0],
                }
            }
            language = languages_display_name[0]

            def __init__(self, **params) -> None:
                super().__init__(**params)
                from .windows_ocr import WindowsOCR
                self.engine = WindowsOCR()
                self.engine.lang = self.get_engine_lang()

            def get_engine_lang(self) -> str:
                language = self.params['language']['value'] 
                tag_name = languages_tag[languages_display_name.index(language)]
                return tag_name

            def ocr_img(self, img: np.ndarray) -> str:
                self.engine(img)

            def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]) -> None:
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

    else:
        LOGGER.warning(f'No supported language packs found for windows, Windows OCR will be unavailable.')