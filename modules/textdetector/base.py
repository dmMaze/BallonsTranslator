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

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TEXTDETECTORS.module_dict:
            if TEXTDETECTORS.module_dict[key] == self.__class__:
                self.name = key
                break

    def _detect(self, *args, **kwargs) -> Tuple[np.ndarray, List[TextBlock]]:
        raise NotImplementedError

    def setup_detector(self):
        raise NotImplementedError

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.all_model_loaded():
            self.load_model()
        mask, blk_list = self._detect(img)
        for blk in blk_list:
            blk.det_model = self.name
        return mask, blk_list
