import os
import os.path as osp
from typing import Tuple, List

import torch
import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
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


update_ckpt_list()

@register_textdetectors('ysgyolo')
class YSGYoloDetector(TextDetectorBase):
    params = {
        'model path': {
            'type': 'selector',
            'options': CKPT_LIST,
            'value': 'data/models/ysgyolo_1.2_OS1.0.pt',
            'editable': True,
            'flush_btn': True,
            'path_selector': True,
            'path_filter': '*.pt *.ckpt *.pth *.safetensors',
            'size': 'median',
            'display_name': '模型路径'
        },
        'merge text lines': {
            'display_name': '合并文本行', 'type': 'checkbox', 'value': True
        },
        'confidence threshold': {
            'display_name': '置信度阈值', 'type': 'line_editor', 'value': 0.3
        },
        'IoU threshold': {
            'display_name': 'IoU阈值', 'type': 'line_editor', 'value': 0.5
        },
        'font size multiplier': {
            'display_name': '字号乘数', 'type': 'line_editor', 'value': 1.
        },
        'font size max': {
            'display_name': '最大字号', 'type': 'line_editor', 'value': -1
        },
        'font size min': {
            'display_name': '最小字号', 'type': 'line_editor', 'value': -1
        },
        'detect size': {
            'display_name': '检测尺寸', 'type': 'line_editor', 'value': 1024
        },
        'device': {
            **DEVICE_SELECTOR(),
            'display_name': '设备'
        },
        'label': {
            'value': {
                'balloon': True,
                'qipao': True,
                'shuqing': True,
                'changfangtiao': True,
                'hengxie': True,
                'other': True
            },
            'type': 'check_group',
            'display_name': '标签'
        },
        'source text is vertical': {
            'display_name': '竖排文本', 'type': 'checkbox', 'value': True
        },
        'mask dilate size': {
            'display_name': '掩码扩张尺寸', 'type': 'line_editor', 'value': 2
        }
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
        return [k for k, v in self.params['label']['value'].items() if v]

    @property
    def is_ysg(self):
        return osp.basename(self.get_param_value('model path').startswith('ysg'))

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        result = self.model.predict(
            source=img, save=False, show=False, verbose=False,
            conf=self.get_param_value('confidence threshold'), iou=self.get_param_value('IoU threshold'),
            agnostic_nms=True
        )[0]

        valid_labels = set(self.get_valid_labels())
        valid_ids = [idx for idx, name in result.names.items() if name in valid_labels]

        mask = np.zeros_like(img[..., 0])
        if not valid_ids:
            return [], mask

        im_h, im_w = img.shape[:2]
        detected_items = []

        # Process standard boxes
        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    label_name = result.names[cls_idx]

                    xyxy = dets.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                    detected_items.append({'pts': pts, 'label': label_name})

        # Process oriented boxes
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                if cls_idx in valid_ids:
                    label_name = result.names[cls_idx]
                    pts = dets.xyxyxyxy[i].cpu().numpy().astype(int)
                    cv2.fillPoly(mask, [pts], 255)
                    detected_items.append({'pts': pts.tolist(), 'label': label_name})

        blk_list = []
        if self.get_param_value('merge text lines'):
            pts_only_list = [item['pts'] for item in detected_items]
            blk_list = mit_merge_textlines(pts_only_list, width=im_w, height=im_h)
        else:
            for item in detected_items:

                pts_sorted, is_vertical = sort_pnts(item['pts'])
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item['label'])
                blk.vertical = is_vertical
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
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
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