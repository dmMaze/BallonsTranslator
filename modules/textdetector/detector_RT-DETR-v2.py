import os
import os.path as osp
from typing import Tuple, List

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR, DEFAULT_DEVICE
from utils.textblock import sort_regions
from utils.imgproc_utils import xywh2xyxypoly 

MODEL_DIR = 'data/models/ctbd'
DEFAULT_MODEL_PATH = osp.join(MODEL_DIR, 'pytorch_model.bin').replace('\\', '/')


# Clips the coordinates of a bounding box to the image boundaries
def _clip_coords_local(box: tuple, img_w: int, img_h: int) -> tuple:
    """
    Clips the coordinates of a bounding box (x1, y1, x2, y2) to the image boundaries.
    Local version for RTDetrV2Detector.
    """
    x1, y1, x2, y2 = box
    x1_c = np.clip(x1, 0, img_w - 1)
    y1_c = np.clip(y1, 0, img_h - 1)
    x2_c = np.clip(x2, 0, img_w - 1)
    y2_c = np.clip(y2, 0, img_h - 1)
    return int(round(x1_c)), int(round(y1_c)), int(round(x2_c)), int(round(y2_c))


@register_textdetectors('rtdetr_v2')
class RTDetrV2Detector(TextDetectorBase):

    params = {
        'model path': {
            'value': MODEL_DIR,
            'description': 'Path to the local directory containing the RT-DETR-V2 model files (e.g., pytorch_model.bin, config.json).',
            'path_selector': True,
            'path_type': 'dir',
            'size': 'large'
        },
        'confidence threshold': 0.3,
        'font size multiplier': 1.,
        'font size max': -1,
        'font size min': -1,
        'mask dilate size': 2,
        'device': DEVICE_SELECTOR(),
    }

    _load_model_keys = {'model', 'processor'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model: RTDetrForObjectDetection = None
        self.processor: RTDetrImageProcessor = None

    def _load_model(self):
        # Load the model from the specified path
        model_load_path = self.get_param_value('model path')
        device = self.get_param_value('device')
        
        if not osp.isdir(model_load_path):
            self.logger.error(f"Model directory not found: {model_load_path}")
            raise FileNotFoundError(f"RT-DETR-V2 model directory not found at {model_load_path}")

        try:
            self.logger.info(f"Loading RT-DETR-V2 model from local path: {model_load_path}")
            self.processor = RTDetrImageProcessor.from_pretrained(model_load_path)
            self.model = RTDetrForObjectDetection.from_pretrained(model_load_path)
            
            if device == 'auto':
                selected_device = DEFAULT_DEVICE
                self.logger.info(f"'auto' device selected, using default: {selected_device}")
            else:
                selected_device = device

            if selected_device == 'cuda' and not torch.cuda.is_available():
                self.logger.warning("CUDA device selected, but CUDA is not available. Falling back to CPU.")
                selected_device = 'cpu'
            
            self.model.to(selected_device)
            self.logger.info(f"RT-DETR-V2 model loaded successfully on device: {selected_device}")

        except Exception as e:
            self.logger.error(f"Failed to load RT-DETR-V2 model from {model_load_path}: {e}", exc_info=True)
            raise e

    def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
        # Perform text detection on the input image
        if self.model is None or self.processor is None:
            self.logger.error("RT-DETR-V2 model or processor not loaded. Cannot perform detection.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        im_h, im_w = img.shape[:2]
        device = self.model.device
        conf_threshold = self.get_param_value('confidence threshold')

        try:
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            self.logger.error(f"Failed to convert input image to PIL format: {e}", exc_info=True)
            return np.zeros_like(img[..., 0]), []

        try:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            self.logger.error(f"Failed during image preprocessing: {e}", exc_info=True)
            return np.zeros_like(img[..., 0]), []
            
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
             self.logger.error(f"Failed during model inference: {e}", exc_info=True)
             return np.zeros_like(img[..., 0]), []

        try:
            target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=conf_threshold
            )
        except Exception as e:
             self.logger.error(f"Failed during post-processing: {e}", exc_info=True)
             return np.zeros_like(img[..., 0]), []

        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        blk_list = []

        if results and len(results) > 0:
            result = results[0] 
            boxes = result['boxes']
            labels = result['labels']

            text_indices = torch.where((labels == 1) | (labels == 2))[0]

            if len(text_indices) > 0:
                text_boxes = boxes[text_indices].cpu().numpy()
                
                clipped_boxes = []
                valid_indices = []
                for i, box in enumerate(text_boxes):
                    x1, y1, x2, y2 = map(round, box)
                    # Use the local clipping function
                    x1, y1, x2, y2 = _clip_coords_local((x1, y1, x2, y2), im_w, im_h) 
                    if x1 < x2 and y1 < y2: 
                        clipped_boxes.append([x1, y1, x2, y2])
                        valid_indices.append(i)
                    else:
                         self.logger.debug(f"Skipping invalid box after clipping: original={box}, clipped=({x1},{y1},{x2},{y2})")
                
                text_boxes = np.array(clipped_boxes) if clipped_boxes else np.array([])

                if len(text_boxes) > 0:
                    for i, box in enumerate(text_boxes):
                        x1, y1, x2, y2 = box
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        
                        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int64)
                        blk = TextBlock(xyxy=box, lines=poly[np.newaxis, :, :]) 
                        
                        h = y2 - y1
                        blk.font_size = h 
                        blk._detected_font_size = h 
                        blk.vertical = False 

                        blk_list.append(blk)

        if not blk_list:
            self.logger.debug("No text boxes detected meeting the confidence threshold.")
            return mask, blk_list

        blk_list = sort_regions(blk_list)

        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            if hasattr(blk, '_detected_font_size'):
                sz = blk._detected_font_size * fnt_rsz
                if fnt_max > 0:
                    sz = min(fnt_max, sz)
                if fnt_min > 0:
                    sz = max(fnt_min, sz)
                blk.font_size = sz

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0 and len(blk_list) > 0 : 
            try:
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
                mask = cv2.dilate(mask, element)
            except Exception as e:
                self.logger.warning(f"Failed to dilate mask: {e}")

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        # Update parameters and reload the model if necessary
        old_value = self.params.get(param_key, {}).get('value')
        super().updateParam(param_key, param_content)
        new_value = self.get_param_value(param_key)

        if param_key in ['device', 'model path'] and old_value != new_value:
            self.logger.info(f"Parameter '{param_key}' changed. Reloading model.")
            self.model = None
            self.processor = None