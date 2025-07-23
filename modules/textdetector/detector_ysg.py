import os
import os.path as osp
from typing import Tuple, List

import torch
import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
from utils.textblock_mask import canny_flood
from utils.split_text_region import manga_split, split_textblock
from utils.imgproc_utils import xywh2xyxypoly
from utils.proj_imgtrans import ProjImgTrans

MODEL_DIR = 'data/models'
CKPT_LIST = []

def update_ckpt_list():
    if not osp.exists(MODEL_DIR):
        return
    global CKPT_LIST
    CKPT_LIST.clear()
    for p in os.listdir(MODEL_DIR):
        if p.startswith('ysgyolo') or p.startswith('ultralyticsyolo'):
            CKPT_LIST.append(osp.join(MODEL_DIR, p).replace('\\', '/'))


# https://github.com/dmMaze/BallonsTranslator/issues/811#issuecomment-2727375501
CLS_MAP = {
    'balloon': 'vertical_textline',
    'changfangtiao': 'horizontal_textline',
    'qipao': 'textblock',
    'fangkuai': 'angled_vertical_textline',
    'kuangwai': 'angled_horizontal_textline',
    'other': 'other'
}

update_ckpt_list()

@register_textdetectors('ysgyolo')
class YSGYoloDetector(TextDetectorBase):
    params = {
        'model path': {
            'type': 'selector',
            'options': CKPT_LIST,
            'value': 'data/models/ysgyolo_S150best.pt',
            'editable': True,
            'flush_btn': True,
            'path_selector': True,
            'path_filter': '*.pt *.ckpt *.pth *.safetensors',
            'size': 'median'
        },
        'merge text lines': True,
        'confidence threshold': 0.3,
        'IoU threshold': 0.5,
        'font size multiplier': 1.,
        'font size max': -1,
        'font size min': -1,
        'detect size': 1024,
        'device': DEVICE_SELECTOR(),
        # A better representation would be routing label to its model 
        # But a well trained model should be able to handle all these
        # So I think it's not worthwhile to implement it here
        'label': {
            'value': {  
                'vertical_textline': True, 
                'horizontal_textline': True, 
                'angled_vertical_textline': True, 
                'angled_horizontal_textline': True,
                'textblock': True
            }, 
            'type': 'check_group'
        },
        'source text is vertical': True,
        'mask dilate size': 2
    }

    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        update_ckpt_list()
    
    def _load_model(self):
        model_path = self.get_param_value('model path')
        if not osp.exists(model_path):
            global CKPT_LIST
            df_model_path = model_path
            for p in CKPT_LIST:
                if osp.exists(p):
                    df_model_path = p
                    break
            self.logger.warning(f'{model_path} does not exist, try fall back to default value {df_model_path}')
            model_path = df_model_path

        if 'rtdetr' in os.path.basename(model_path):
            from ultralytics import RTDETR as MODEL
        else:
            from ultralytics import YOLO as MODEL
        if not hasattr(self, 'model') or self.model is None:
            self.model = MODEL(model_path).to(device=self.get_param_value('device'))

    def get_valid_labels(self):
        valid_labels = [k for k, v in self.params['label']['value'].items() if v and k != 'textblock']
        return valid_labels

    @property
    def is_ysg(self):
        return osp.basename(self.get_param_value('model path').startswith('ysg'))

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:

        result = self.model.predict(
            source=img, save=False, show=False, verbose=False, 
            conf=self.get_param_value('confidence threshold'), iou=self.get_param_value('IoU threshold'),
            agnostic_nms=True
        )[0]
        valid_ids = []
        valid_labels = set(self.get_valid_labels())
        textblock_idx = -1
        for idx, name in result.names.items():
            if CLS_MAP[name] in valid_labels:
                valid_ids.append(idx)
            if name == 'qipao':
                textblock_idx = idx
        need_textblock = self.params['label']['value']['textblock'] == True

        mask = np.zeros_like(img[..., 0])
        if len(valid_ids) == 0 and not need_textblock:
            return [], mask

        im_h, im_w = img.shape[:2]
        pts_list = []

        blk_list = []

        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            device = dets.cls.device
            valid_mask = torch.zeros((dets.cls.shape[0]), device=device, dtype=torch.bool)
            for idx in valid_ids:
                valid_mask = torch.bitwise_or(valid_mask, dets.cls == idx)
            if torch.any(valid_mask):
                xyxy_list = dets.xyxy[valid_mask]
                xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                xyxy_list[:, [0, 2]] = torch.clip(xyxy_list[:, [0, 2]], 0, im_w - 1)
                xyxy_list[:, [1, 3]] = torch.clip(xyxy_list[:, [1, 3]], 0, im_h - 1)
                xyxy_list = xyxy_list.numpy()
                for xyxy in xyxy_list:
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                xyxy_list[:, [2, 3]] -= xyxy_list[:, [0, 1]]
                pts_list += xywh2xyxypoly(xyxy_list).reshape(-1, 4, 2).tolist()
            
            if need_textblock:
                valid_mask = dets.cls == textblock_idx
                is_vertical = self.get_param_value('source text is vertical')
                if torch.any(valid_mask):
                    xyxy_list = dets.xyxy[valid_mask]
                    xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                    xyxy_list[:, [0, 2]] = torch.clip(xyxy_list[:, [0, 2]], 0, im_w - 1)
                    xyxy_list[:, [1, 3]] = torch.clip(xyxy_list[:, [1, 3]], 0, im_h - 1)
                    xyxy_list = xyxy_list.numpy()
                    for xyxy in xyxy_list:
                        x1, y1, x2, y2 = xyxy
                        crop = img[y1: y2, x1: x2]
                        bmask  = canny_flood(crop)[0]
                        if is_vertical:
                            span_list = manga_split(bmask)
                            lines = [[line.left + x1, line.top + y1, line.width, line.height] for line in span_list]
                            lines = np.array(lines)[::-1]
                            font_sz = np.mean(lines[:, 2])
                        else:
                            span_list = split_textblock(bmask)[0]
                            lines = [[line.left + x1, line.top + y1, line.width, line.height] for line in span_list]
                            lines = np.array(lines)
                            font_sz = np.mean(lines[:, 3])
                        for line in lines:
                            x1, y1, x2, y2 = line
                            x2 += x1
                            y2 += y1
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        lines = xywh2xyxypoly(lines).reshape(-1, 4, 2).tolist()
                        blk = TextBlock(xyxy=xyxy, lines=np.array(lines), src_is_vertical=is_vertical, vertical=is_vertical)
                        blk.font_size = font_sz
                        blk._detected_font_size = font_sz
                        if is_vertical:
                            blk.alignment = 1
                        else:
                            blk.recalulate_alignment()

                        blk_list.append(blk)
                        
                        # cv2.imwrite('mask.jpg', mask)
                        # for ii in range(len(blk.lines)):
                        #     rst = blk.get_transformed_region(img, ii, 48)
                        #     cv2.imwrite('local_tst.jpg', rst)
                        #     pass

        # oriented objects
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            device = dets.cls.device
            valid_mask = torch.zeros((dets.cls.shape[0]), device=device, dtype=torch.bool)
            for idx in valid_ids:
                valid_mask = torch.bitwise_or(valid_mask, dets.cls == idx)
            if torch.any(valid_mask):
                xyxy_list = dets.xyxyxyxy[valid_mask]
                xyxy_list = xyxy_list.to(device='cpu', dtype=torch.float32).round().to(torch.int32)
                xyxy_list[..., 0] = torch.clip(xyxy_list[..., 0], 0, im_w - 1)
                xyxy_list[..., 1] = torch.clip(xyxy_list[..., 1], 0, im_h - 1)
                xyxy_list = xyxy_list.numpy()
                for pts in xyxy_list:
                    cv2.fillPoly(mask, [pts], 255)
                pts_list += xyxy_list.tolist()

        if self.get_param_value('merge text lines'):
            blk_list += mit_merge_textlines(pts_list, width=im_w, height=im_h)
        else:
            for pts in pts_list:
                pts_sorted, is_vertical = sort_pnts(pts)
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical)
                blk.adjust_bbox()
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)
        blk_list = sort_regions(blk_list)

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
        
        if param_key == 'model path':
            if hasattr(self, 'model'):
                del self.model

    def flush(self, param_key: str):
        if param_key == 'model path':
            update_ckpt_list()
            global CKPT_LIST
            return CKPT_LIST