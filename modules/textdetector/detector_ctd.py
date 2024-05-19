import numpy as np
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR
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
            'value': 1024
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'value': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector'
    }
    _load_model_keys = {'model'}

    device = DEFAULT_DEVICE
    detect_size = 1024
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model: CTDModel = None

    @property
    def device(self):
        return self.params['device']['value']
    
    @property
    def detect_size(self):
        return int(self.params['detect_size']['value'])

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
        device = self.device
        if self.model is not None:
            if self.model.device != device:
                self.model.device = device
                if device != 'cpu':
                    self.model.load_model(CTD_TORCH_PATH)
                else:
                    self.model.load_model(CTD_ONNX_PATH)
            self.model.detect_size = self.detect_size