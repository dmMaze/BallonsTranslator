from typing import List
import numpy as np

from .base import OCRBase, register_OCR, DEVICE_SELECTOR, DEFAULT_DEVICE, TextBlock, OCR

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