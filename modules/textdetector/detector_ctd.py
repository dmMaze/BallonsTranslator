import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from .ctd import CTDModel

CTD_ONNX_PATH = 'data/models/comictextdetector.pt.onnx'
CTD_TORCH_PATH = 'data/models/comictextdetector.pt'
CTD_DETECT_SIZE_OPTIONS = [896, 1024, 1152, 1280, 1400, 1536, 1600, 1792, 2048, 2400]

def load_ctd_model(
    model_path,
    device,
    detect_size=1024,
    half=False,
    nms_thresh=0.35,
    conf_thresh=0.4,
    det_rearrange_max_batches=4,
) -> CTDModel:
    model = CTDModel(
        model_path,
        detect_size=detect_size,
        device=device,
        half=half,
        nms_thresh=nms_thresh,
        conf_thresh=conf_thresh,
        det_rearrange_max_batches=det_rearrange_max_batches,
    )
    
    return model

@register_textdetectors('ctd')
class ComicTextDetector(TextDetectorBase):

    params = {
        'detect_size': {
            'type': 'selector',
            'options': CTD_DETECT_SIZE_OPTIONS,
            'value': 1280,
            'display_name': 'Detection Resolution'
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'value': 4,
            'display_name': 'Max Rearranged Detection Batches'
        },
        'confidence threshold': {
            'type': 'line_editor',
            'value': 0.4,
            'data_type': float,
            'display_name': 'Confidence Threshold'
        },
        'NMS threshold': {
            'type': 'line_editor',
            'value': 0.35,
            'data_type': float,
            'display_name': 'NMS Threshold'
        },
        'half precision': {
            'type': 'checkbox',
            'value': False,
            'display_name': 'Half Precision'
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector',
        'font size multiplier': {
            'type': 'line_editor',
            'value': 1.,
            'data_type': float,
            'display_name': 'Font Size Multiplier'
        },
        'font size max': {
            'type': 'line_editor',
            'value': -1,
            'data_type': int,
            'display_name': 'Maximum Font Size'
        },
        'font size min': {
            'type': 'line_editor',
            'value': -1,
            'data_type': int,
            'display_name': 'Minimum Font Size'
        },
        'mask dilate size': {
            'type': 'line_editor',
            'value': 2,
            'data_type': int,
            'display_name': 'Mask Dilation Size'
        }
    }
    _load_model_keys = {'model'}
    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/',
        'files': ['data/models/comictextdetector.pt', 'data/models/comictextdetector.pt.onnx'],
        'sha256_pre_calculated': ['1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb', '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'],
        'concatenate_url_filename': 2,
    }]

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

    @property
    def half_precision(self):
        return bool(self.params['half precision']['value'])

    @property
    def confidence_threshold(self):
        return float(self.params['confidence threshold']['value'])

    @property
    def nms_threshold(self):
        return float(self.params['NMS threshold']['value'])

    @property
    def det_rearrange_max_batches(self):
        return int(self.params['det_rearrange_max_batches']['value'])

    def _load_model(self):
        if self.device != 'cpu':
            self.model = load_ctd_model(
                CTD_TORCH_PATH,
                self.device,
                self.detect_size,
                self.half_precision,
                self.nms_threshold,
                self.confidence_threshold,
                self.det_rearrange_max_batches,
            )
        else:
            self.model = load_ctd_model(
                CTD_ONNX_PATH,
                self.device,
                self.detect_size,
                False,
                self.nms_threshold,
                self.confidence_threshold,
                self.det_rearrange_max_batches,
            )

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        _, mask, blk_list = self.model(img)
        
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.device
        if self.model is not None:
            if self.model.device != device or param_key == 'half precision':
                self.model.device = device
                self.model.half = self.half_precision if device != 'cpu' else False
                if device != 'cpu':
                    self.model.load_model(CTD_TORCH_PATH)
                else:
                    self.model.load_model(CTD_ONNX_PATH)
            self.model.detect_size = self.detect_size
            self.model.conf_thresh = self.confidence_threshold
            self.model.nms_thresh = self.nms_threshold
            self.model.det_rearrange_max_batches = self.det_rearrange_max_batches
