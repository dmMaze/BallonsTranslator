import os
import base64
import requests
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
        self.model: CTDModel = None

    @property
    def device(self):
        return self.params['device']['select']
    
    @property
    def detect_size(self):
        return int(self.params['detect_size']['select'])

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

@register_textdetectors('stariver_ocr')
class StariverDetector(TextDetectorBase):

    params = {
        'token': "Replace with your token",
        'expand_ratio': "0.01",
        'description': '星河云(团子翻译器) OCR 文字检测器'
    }

    @property
    def token(self):
        return self.params['token']
    
    @property
    def expand_ratio(self):
        return self.params['expand_ratio'].eval()
    
    def __init__(self, **params) -> None:
        self.url = 'https://dl.ap-sh.starivercs.cn/v2/manga_trans/advanced/manga_ocr'

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.token or self.token == 'Replace with your token':
            self.logger.error(f'token 没有设置。当前token：{self.token}')
            raise ValueError('token 没有设置。')
        img_encoded = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        payload = {
            "token": self.token,
            "mask": True,
            "refine": True,
            "filtrate": True,
            "disable_skip_area": True,
            "detect_scale": 3,
            "merge_threshold": 0.5,
            "low_accuracy_mode": False,
            "image": img_base64
        }
        response = requests.post(self.url, json=payload)
        response_data = response.json()['Data']

        blk_list = []
        for block in response_data.get('text_block', []):
            xyxy = [int(min(coord[0] for coord in block['block_coordinate'].values())),
                    int(min(coord[1] for coord in block['block_coordinate'].values())),
                    int(max(coord[0] for coord in block['block_coordinate'].values())),
                    int(max(coord[1] for coord in block['block_coordinate'].values()))]
            lines = [np.array([[coord[pos][0], coord[pos][1]] for pos in ['upper_left', 'upper_right', 'lower_right', 'lower_left']], dtype=np.float32) for coord in block['coordinate']]
            texts = block.get('texts', '')
            blk = TextBlock(
                xyxy=xyxy,
                lines=lines,
                language=block.get('language', 'unknown'),
                vertical=block.get('is_vertical', False),
                font_size=block.get('text_size', 0),
                distance=np.array([0, 0], dtype=np.float32),
                angle=0,
                vec=np.array([0, 0], dtype=np.float32),
                norm=0,
                merged=False,
                text=texts,
                fg_colors=np.array(block.get('foreground_color', [0, 0, 0]), dtype=np.float32),
                bg_colors=np.array(block.get('background_color', [0, 0, 0]), dtype=np.float32)
            )
            blk_list.append(blk)
        
        mask = self._decode_base64_mask(response_data['mask'])
        mask = self.expand_mask(mask)
        return mask, blk_list

    @staticmethod
    def _decode_base64_mask(base64_str: str) -> np.ndarray:
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return mask
    
    def expand_mask(self, mask: np.ndarray, expand_ratio: float = 0.01) -> np.ndarray:
        """
        在mask的原始部分上扩展mask，以便于提取更大的文字区域。
        :param mask: 输入的mask
        :param expand_ratio: 扩展比例，默认值为0.01
        :return: 扩展后的mask
        """
        # 确保mask是二值图像（只含0和255）
        mask = (mask > 0).astype(np.uint8) * 255

        # 获得图像的尺寸
        height, width = mask.shape
        
        # 计算kernel的大小（取图像尺寸的一部分，按比例expand_ratio）
        kernel_size = int(min(height, width) * expand_ratio)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保kernel尺寸是奇数

        # 创建一个正方形的kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 执行膨胀操作
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # 计算扩展后的mask
        dilated_mask = (dilated_mask > 0).astype(np.uint8) * 255
        
        return dilated_mask
    
    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)