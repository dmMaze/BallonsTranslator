from typing import Tuple, List, Dict, Union, Callable
from copy import deepcopy
import numpy as np
import cv2
from collections import OrderedDict

from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR, LOGGER
from ..exceptions import ModuleRunError

class OCRBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()
    _line_only: bool = False

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in OCR.module_dict:
            if OCR.module_dict[key] == self.__class__:
                self.name = key
                break

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None, *args, **kwargs) -> Union[List[TextBlock], str]:
        if not self.all_model_loaded():
            self.load_model()

        original_text = None
        try:
            if img.ndim == 3 and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            if blk_list is None:
                text = self.ocr_img(img)
                return text
            elif isinstance(blk_list, TextBlock):
                blk_list = [blk_list]

            original_text = [(blk, deepcopy(blk.text)) for blk in blk_list]
            for blk in blk_list:
                if self.name != 'none_ocr':
                    blk.text = []

            self._ocr_blk_list(img, blk_list, *args, **kwargs)
            for callback_name, callback in self._postprocess_hooks.items():
                callback(textblocks=blk_list, img=img, ocr_module=self)

            return blk_list
        except (ModuleRunError, LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped):
            if original_text is not None:
                for blk, text in original_text:
                    blk.text = text
            raise
        except Exception as e:
            if original_text is not None:
                for blk, text in original_text:
                    blk.text = text
            raise ModuleRunError('ocr', self.name, str(e)) from e

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        """Processes a list of text blocks on the image."""
        im_h, im_w = img.shape[:2]

        for i, blk in enumerate(blk_list):
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode > 1:
                self.logger.debug(
                    f"Processing block {i+1}/{len(blk_list)}: ({x1, y1, x2, y2})"
                )

            y1c, y2c = max(0, y1), min(im_h, y2)
            x1c, x2c = max(0, x1), min(im_w, x2)

            if y1c < y2c and x1c < x2c:
                cropped_img = img[y1c:y2c, x1c:x2c]
                blk.text = self.ocr_img(cropped_img, **kwargs)
            else:
                blk.text = ""

    def ocr_img(self, img: np.ndarray, **kwargs) -> str:
        raise NotImplementedError
