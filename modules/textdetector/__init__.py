import numpy as np
import cv2
from typing import Union, List, Tuple
from collections import OrderedDict

from utils.textblock import TextBlock

from utils.registry import Registry
TEXTDETECTORS = Registry('textdetectors')
register_textdetectors = TEXTDETECTORS.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR

class TextDetectorBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/',
        'files': ['data/models/comictextdetector.pt', 'data/models/comictextdetector.pt.onnx'],
        'sha256_pre_calculated': ['1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb', '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'],
        'concatenate_url_filename': 2,
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TEXTDETECTORS.module_dict:
            if TEXTDETECTORS.module_dict[key] == self.__class__:
                self.name = key
                break

    def setup_detector(self):
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.all_model_loaded():
            self.load_model()
        return self._detect(img)

from .ctd import CTDModel
CTD_ONNX_PATH = 'data/models/comictextdetector.pt.onnx'
CTD_TORCH_PATH = 'data/models/comictextdetector.pt'

def load_ctd_model(model_path, device, detect_size=1024) -> CTDModel:
    model = CTDModel(model_path, detect_size=detect_size, device=device)
    
    return model

@register_textdetectors('ctd')
class ComicTextDetector(TextDetectorBase):

    params = {
        'detect_size': {
            'type': 'selector',
            'options': [896, 1024, 1152, 1280], 
            'select': 1024
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'select': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector'
    }
    _load_model_keys = {'model'}

    device = DEFAULT_DEVICE
    detect_size = 1024
    def __init__(self, **params) -> None:
        super().__init__(**params)

        self.device = self.params['device']['select']
        self.detect_size = int(self.params['detect_size']['select'])
        self.model: CTDModel = None

    def _load_model(self):
        if self.device != 'cpu':
            self.model = load_ctd_model(CTD_TORCH_PATH, self.device, self.detect_size)
        else:
            self.model = load_ctd_model(CTD_ONNX_PATH, self.device, self.detect_size)

    def _detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        _, mask, blk_list = self.model(img)
        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['select']
        detect_size = int(self.params['detect_size']['select'])
        if device != self.device:
            self.setup_detector()
        elif detect_size != self.detect_size:
            self.model.detect_size = detect_size