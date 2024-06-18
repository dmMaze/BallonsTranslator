from typing import List
import numpy as np
from copy import deepcopy

from .base import DEVICE_SELECTOR, OCRBase, register_OCR, TextBlock

mit_params = {
    'chunk_size': {
        'type': 'selector',
        'options': [8, 16, 24, 32],
        'value': 16
    },
    'device': DEVICE_SELECTOR(),
    'description': 'OCRMIT32px'
}

class MITModels(OCRBase):

    _line_only = True
    _load_model_keys = {'model'}

    @property
    def chunk_size(self) -> int:
        return self.params['chunk_size']['value']

    @property
    def device(self) -> str:
        return self.params['device']['value']

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock]):
        return self.model(img, blk_list, chunk_size=self.chunk_size)

    def updateParam(self, param_key: str, param_content):
        if param_key == 'device' and self.device != param_content:
            self.model.to(param_content)
        super().updateParam(param_key, param_content)


from .model_32px import OCR32pxModel
@register_OCR('mit32px')
class OCRMIT32px(MITModels):

    params = deepcopy(mit_params)
    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr.zip',
        'files': ['ocr.ckpt'],
        'sha256_pre_calculated': ['d9f619a9dccce8ce88357d1b17d25f07806f225c033ea42c64e86c45446cfe71'],
        'save_files': ['data/models/mit32px_ocr.ckpt'],
        'archived_files': 'ocr.zip',
        'archive_sha256_pre_calculated': '47405638b96fa2540a5ee841a4cd792f25062c09d9458a973362d40785f95d7a',
    }]

    def _load_model(self):
        self.model = OCR32pxModel(r'data/models/mit32px_ocr.ckpt', self.device)


from .mit48px_ctc import OCR48pxCTC
@register_OCR('mit48px_ctc')
class OCRMIT48pxCTC(MITModels):

    params = deepcopy(mit_params)
    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip',
        'files': ['ocr-ctc.ckpt', 'alphabet-all-v5.txt'],
        'sha256_pre_calculated': ['8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89', None],
        'save_files': ['data/models/mit48pxctc_ocr.ckpt', 'data/alphabet-all-v5.txt'],
        'archived_files': 'ocr-ctc.zip',
        'archive_sha256_pre_calculated': 'fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101',
    }]

    def _load_model(self):
        self.model = OCR48pxCTC(r'data/models/mit48pxctc_ocr.ckpt', self.device)


from .mit48px import Model48pxOCR
OCR48PXMODEL_PATH = r'data/models/ocr_ar_48px.ckpt'
@register_OCR('mit48px')
class OCRMIT48px(MITModels):

    params = deepcopy(mit_params)
    download_file_list = [{
        'url': 'https://huggingface.co/zyddnys/manga-image-translator/resolve/main/',
        'files': [OCR48PXMODEL_PATH, 'data/alphabet-all-v7.txt'],
        'sha256_pre_calculated': ['29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d', None],
        'concatenate_url_filename': 2,
    }]

    def _load_model(self):
        self.model = Model48pxOCR(OCR48PXMODEL_PATH, self.device)