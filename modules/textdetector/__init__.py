import numpy as np
import cv2
from typing import Dict, List, Tuple
from collections import OrderedDict

from .textblock import TextBlock

from utils.registry import Registry
TEXTDETECTORS = Registry('textdetectors')
register_textdetectors = TEXTDETECTORS.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR

class TextDetectorBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TEXTDETECTORS.module_dict:
            if TEXTDETECTORS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.setup_detector()

    def setup_detector(self):
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        raise NotImplementedError

from .ctd import CTDModel
CTDMODEL_TORCH: CTDModel = None # if cuda is available
CTDMODEL_ONNX: CTDModel = None  # for cpu inference
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
            'options': [1024, 1152, 1280], 
            'select': 1280
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'select': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector'
    }

    device = DEFAULT_DEVICE
    detect_size = 1280
    def setup_detector(self):
        global CTDMODEL_TORCH, CTDMODEL_ONNX
        self.device = self.params['device']['select']
        self.detect_size = int(self.params['detect_size']['select'])
        if self.device != 'cpu':
            if CTDMODEL_TORCH is None:
                self.detector = CTDMODEL_TORCH = load_ctd_model(CTD_TORCH_PATH, self.device, self.detect_size)
            else:
                self.detector = CTDMODEL_TORCH
        else:
            if CTDMODEL_ONNX is None:
                self.detector = CTDMODEL_ONNX = load_ctd_model(CTD_ONNX_PATH, self.device, self.detect_size)
            else:
                self.detector = CTDMODEL_ONNX

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:

        _, mask, blk_list = self.detector(img)
        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params['device']['select']
        detect_size = int(self.params['detect_size']['select'])
        if device != self.device:
            self.setup_detector()
        elif detect_size != self.detect_size:
            self.detector.detect_size = detect_size