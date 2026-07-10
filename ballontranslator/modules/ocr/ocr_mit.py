from typing import List
import numpy as np
from copy import deepcopy

from .base import DEVICE_SELECTOR, OCRBase, register_OCR, TextBlock
from ballontranslator.utils.textblock import collect_textblock_regions

mit_params = {
    'chunk_size': {
        'type': 'selector',
        'options': [8, 16, 24, 32],
        'value': 16,
        'display_name': 'Chunk Size'
    },
    'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
    'description': 'OCRMIT48px'
}

MIT_DEPENDENCIES = ['torch', 'einops']

class MITModels(OCRBase):

    _line_only = True
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    @property
    def chunk_size(self) -> int:
        return self.params['chunk_size']['value']

    @property
    def device(self) -> str:
        return self.params['device']['value']

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], split_textblk=False, seg_func=None, *args, **kwargs):
        regions, textblk_lst_indices = collect_textblock_regions(img, blk_list, self.model.text_height, self.model.maxwidth, split_textblk, seg_func)
        return self.model(blk_list, regions, textblk_lst_indices, chunk_size=self.chunk_size)

    def updateParam(self, param_key: str, param_content):
        if param_key == 'device' and self.device != param_content and self.model is not None:
            self.model.to(param_content)
        super().updateParam(param_key, param_content)


from .mit48px_ctc import OCR48pxCTC
@register_OCR('mit48px_ctc')
class OCRMIT48pxCTC(MITModels):

    dependencies = MIT_DEPENDENCIES

    params = deepcopy(mit_params)
    download_file_list = [{
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ocr-ctc.zip',
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

    dependencies = MIT_DEPENDENCIES

    params = deepcopy(mit_params)
    download_file_list = [{
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ocr48px.zip',
        'save_files': [OCR48PXMODEL_PATH, 'data/alphabet-all-v7.txt'],
        'files': ['ocr48px/ocr_ar_48px.ckpt', 'ocr48px/alphabet-all-v7.txt'],
        'sha256_pre_calculated': ['29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d', None],
        'archived_files': 'ocr48px.zip',
        'archive_sha256_pre_calculated': 'c9e09b46510e78563ca0a645aba1bad5e81d90e4f2b9f94c23810c9cdce72aa8',
    }]

    def _load_model(self):
        self.model = Model48pxOCR(OCR48PXMODEL_PATH, self.device)
