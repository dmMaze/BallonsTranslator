import os
import os.path as osp
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from ballontranslator.utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
from ballontranslator.utils.imgproc_utils import xywh2xyxypoly
from ballontranslator.utils.proj_imgtrans import ProjImgTrans

MODEL_DIR = 'data/models'
YSGYOLO_MODEL_PREFIXES = ('ysgyolo', 'ultralyticsyolo')

def find_model_paths(model_dir, prefixes):
    default_path_list = [
        'data/models/ysgyolo_yolo26_2.0.pt',
        'data/models/ysgyolo_yolo26OBB_2.0.pt'
    ]
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    if not osp.exists(model_dir):
        return []
    found_list = [
        osp.join(model_dir, p).replace('\\', '/')
        for p in sorted(os.listdir(model_dir))
        if p.startswith(tuple(prefixes))
    ]
    ordered_list = []
    for p in default_path_list:
        # Keep bundled download targets scoped to the requested model family,
        # just like the locally discovered checkpoint paths above.
        if not osp.basename(p).startswith(tuple(prefixes)):
            continue
        if p not in ordered_list:
            ordered_list.append(p)
    for p in found_list:
        if p not in ordered_list:
            ordered_list.append(p)

    return ordered_list


CKPT_LIST = find_model_paths(MODEL_DIR, YSGYOLO_MODEL_PREFIXES)


def update_ckpt_list():
    CKPT_LIST[:] = find_model_paths(MODEL_DIR, YSGYOLO_MODEL_PREFIXES)
    return CKPT_LIST


update_ckpt_list()


# Module-level metadata so detector variants can share one SafeEval-compatible
# params/download template instead of duplicating it.
_YSGYOLO_PARAMS = {
    'model path': {
        'type': 'selector',
        'options': CKPT_LIST,
        'value': 'data/models/ysgyolo_yolo26_2.0.pt',
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
            'other': False
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

_YSGYOLO_DOWNLOAD_FILE_LIST = [
    {
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ysgyolo_yolo26OBB_2.0.pt',
        'files': 'data/models/ysgyolo_yolo26OBB_2.0.pt',
        'sha256_pre_calculated': 'a9c03afb069285fc9e0d3fffc3c9ead440b687570952a945391ba22cd843dc3f'
    },
    {
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ysgyolo_yolo26_2.0.pt',
        'files': 'data/models/ysgyolo_yolo26_2.0.pt',
        'sha256_pre_calculated': '889347d65c8636dd188a8ed4f312b29658543faaa69016b5958ddf0559980e22'
    }
]

# Mirrors detector_ctd.download_file_list so the hybrid detector can prepare
# both models without depending on the CTD detector having been selected once.
_CTD_DOWNLOAD_FILE_LIST = [
    {
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt',
        'files': 'data/models/comictextdetector.pt',
        'sha256_pre_calculated': '1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb'
    },
    {
        'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt.onnx',
        'files': 'data/models/comictextdetector.pt.onnx',
        'sha256_pre_calculated': '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'
    }
]


def _dilate_mask(mask: np.ndarray, ksize: int) -> np.ndarray:
    if ksize > 0:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize))
        mask = cv2.dilate(mask, element)
    return mask


def _build_box_mask(blk_list: List[TextBlock], im_w: int, im_h: int) -> np.ndarray:
    """Draw the union of all detected text polygons without dilation."""

    mask = np.zeros((im_h, im_w), np.uint8)
    for blk in blk_list:
        for line in np.asarray(blk.lines, dtype=np.int32):
            cv2.fillPoly(mask, [line.reshape(-1, 2)], 255)
    return mask


def _bbox_overlap_ratio(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.
    area_a = max(ax2 - ax1, 0.) * max(ay2 - ay1, 0.)
    if area_a <= 0:
        return 0.
    return inter_w * inter_h / area_a


def _combine_refined_mask(
    box_mask: np.ndarray,
    fine_mask: np.ndarray,
    blk_list: List[TextBlock],
    ctd_blk_list: List[TextBlock],
    ctd_overlap_ratio: float = 0.25,
) -> np.ndarray:
    """Keep CTD's glyph mask inside YOLO boxes, with per-line coarse fallback.

    A YOLO text line whose region was not detected/refined by CTD keeps its
    coarse box mask so the replacement text is never left unmasked.

    Example:
        >>> box = np.zeros((6, 6), np.uint8)
        >>> box[1:5, 1:5] = 255
        >>> fine = np.zeros_like(box)
        >>> fine[2:3, 2:4] = 255
        >>> blk = TextBlock(lines=[[[1, 1], [5, 1], [5, 5], [1, 5]]])
        >>> blk.adjust_bbox()
        >>> ctd_blk = TextBlock(xyxy=[1, 1, 5, 5])
        >>> combined = _combine_refined_mask(box, fine, [blk], [ctd_blk])
        >>> int(combined[2, 2]), int(combined[1, 1])
        (255, 0)
    """

    combined = cv2.bitwise_and(fine_mask, box_mask)
    im_h, im_w = box_mask.shape[:2]
    for blk in blk_list:
        for line in np.asarray(blk.lines, dtype=np.int32):
            pts = line.reshape(-1, 2)
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max() + 1, pts[:, 1].max() + 1
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), im_w), min(int(y2), im_h)
            if x2 <= x1 or y2 <= y1:
                continue
            box_crop = box_mask[y1: y2, x1: x2]
            if not np.any(box_crop):
                continue
            line_bbox = [x1, y1, x2, y2]
            ctd_refined = any(
                _bbox_overlap_ratio(line_bbox, ctd_blk.xyxy) >= ctd_overlap_ratio
                for ctd_blk in ctd_blk_list
            )
            if ctd_refined:
                crop = combined[y1: y2, x1: x2]
                if np.any(crop):
                    continue
            local_pts = pts - np.array([x1, y1], dtype=np.int32)
            line_mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillPoly(line_mask, [local_pts], 255)
            line_region = cv2.bitwise_and(line_mask, box_crop)
            if np.any(line_region):
                combined[y1: y2, x1: x2] = cv2.bitwise_or(
                    combined[y1: y2, x1: x2], line_region)
    return combined


@register_textdetectors('ysgyolo')
class YSGYoloDetector(TextDetectorBase):
    dependencies = ['torch', 'ultralytics>=8.4.14']

    # Shared metadata template: the plain detector keeps the live list objects
    # (model list flush updates them) while variants deep-copy their own.
    params = _YSGYOLO_PARAMS

    _load_model_keys = {'model'}

    download_file_list = _YSGYOLO_DOWNLOAD_FILE_LIST

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
            self.logger.warning(
                f'{model_path} does not exist, try fall back to default value {df_model_path}. '
                f'Choose an available checkpoint in the detector\'s model path setting.'
            )
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

        mask = _dilate_mask(mask, self.get_param_value('mask dilate size'))

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


@register_textdetectors('ysg_ctd')
class YSGYoloCTDDetector(YSGYoloDetector):
    """YOLO decides which text to process, CTD refines its glyph mask.

    YSGYOLO reliably separates body text from stylized onomatopoeia but only
    produces rectangular masks. CTD produces glyph-tight masks but also fires
    on onomatopoeia. This detector keeps the YOLO boxes as the allowed region
    and intersects them with CTD's refined mask, so SFX outside the boxes are
    ignored while replacement/OCR still uses the YOLO text blocks.

    The YOLO stage must behave exactly like the ``ysgyolo`` detector, otherwise
    a text block the user trusts ysgyolo to find would be dropped before CTD
    ever contributes. Its shared params are therefore seeded from the saved
    ysgyolo config by ``TEXTDETECTOR_PARAM_INHERITANCE``; only the CTD-specific
    params below are hybrid-only.
    """

    dependencies = ['torch', 'ultralytics>=8.4.14', 'torchvision', 'einops']

    _load_model_keys = {'model', 'ctd_model'}

    params = {
        **deepcopy(_YSGYOLO_PARAMS),
        'ctd detect size': {
            'display_name': 'Detect Size', 'type': 'selector',
            'options': [896, 1024, 1152, 1280], 'value': 1280
        },
        'ctd det rearrange max batches': {
            'display_name': 'Max split batch size', 'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 'value': 4
        },
        'description': '先用 ysgyolo 检测文本框，再用 ComicTextDetector 在框内生成贴合文字轮廓的掩膜',
    }

    # Both model families are declared so selecting the hybrid prepares them.
    download_file_list = deepcopy(_YSGYOLO_DOWNLOAD_FILE_LIST) + _CTD_DOWNLOAD_FILE_LIST

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.ctd_model = None

    def _load_model(self):
        super()._load_model()
        self._load_ctd_model()

    def _load_ctd_model(self) -> None:
        from .detector_ctd import CTD_ONNX_PATH, CTD_TORCH_PATH
        from .ctd import CTDModel

        device = self.get_param_value('device')
        model_path = CTD_ONNX_PATH if device == 'cpu' else CTD_TORCH_PATH
        self.ctd_model = CTDModel(
            model_path,
            detect_size=int(self.get_param_value('ctd detect size')),
            device=device,
            det_rearrange_max_batches=self.get_param_value('ctd det rearrange max batches'),
        )

    def _detect_fine_text_mask(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], List[TextBlock]]:
        if self.ctd_model is None:
            return None, []
        try:
            _, fine_mask, ctd_blk_list = self.ctd_model(img)
        except Exception as error:
            # YOLO block detection is still useful without mask refinement.
            self.logger.warning(f'CTD mask refinement failed, falling back to ysgyolo masks: {error}')
            return None, []
        fine_mask = np.asarray(fine_mask).squeeze()
        if fine_mask.shape[:2] != img.shape[:2]:
            fine_mask = cv2.resize(fine_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        fine_mask = np.where(fine_mask > 127, 255, 0).astype(np.uint8)
        return fine_mask, ctd_blk_list

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        mask, blk_list = super()._detect(img, proj)
        if not blk_list:
            return mask, blk_list
        fine_mask, ctd_blk_list = self._detect_fine_text_mask(img)
        if fine_mask is None:
            return mask, blk_list
        im_h, im_w = img.shape[:2]
        box_mask = _build_box_mask(blk_list, im_w, im_h)
        mask = _combine_refined_mask(box_mask, fine_mask, blk_list, ctd_blk_list)
        mask = _dilate_mask(mask, self.get_param_value('mask dilate size'))
        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device':
            # Both backends must be rebuilt on the newly selected device.
            for key in ('model', 'ctd_model'):
                if hasattr(self, key):
                    setattr(self, key, None)
        elif param_key in {'ctd detect size', 'ctd det rearrange max batches'}:
            self.ctd_model = None
