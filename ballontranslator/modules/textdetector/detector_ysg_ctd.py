"""YSG text selection with CTD mask refinement, adapted from PR #1365."""

from copy import deepcopy
import os.path as osp
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .base import DEVICE_SELECTOR, TextBlock, TextDetectorBase, register_textdetectors
from .detector_ysg import find_model_paths
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import (
    examine_textblk, mit_merge_textlines, sort_pnts, sort_regions,
)


def _combine_refined_mask(
    fine_mask: np.ndarray,
    blocks: List[TextBlock],
    ctd_blocks: List[TextBlock],
) -> np.ndarray:
    """Clip CTD to YSG polygons, retaining each line CTD did not refine.

    >>> line = TextBlock(lines=[[[1, 1], [4, 1], [4, 4], [1, 4]]])
    >>> fine = np.zeros((6, 6), np.uint8)
    >>> fine[2, 2] = 255
    >>> ctd = TextBlock(xyxy=[1, 1, 5, 5])
    >>> int(np.count_nonzero(_combine_refined_mask(fine, [line], [ctd])))
    1
    >>> int(np.count_nonzero(_combine_refined_mask(fine, [line], [])))
    16
    """
    combined = np.zeros_like(fine_mask)
    height, width = fine_mask.shape
    ctd_boxes = np.asarray([block.xyxy for block in ctd_blocks]).reshape(-1, 4)
    for block in blocks:
        for line in block.lines:
            points = np.asarray(line, dtype=np.int32).reshape(-1, 2)
            x1, y1 = np.maximum(points.min(axis=0), 0)
            x2, y2 = np.minimum(points.max(axis=0) + 1, [width, height])
            if x2 <= x1 or y2 <= y1:
                continue
            line_mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
            cv2.fillPoly(line_mask, [points - [x1, y1]], 255)
            refined = cv2.bitwise_and(fine_mask[y1:y2, x1:x2], line_mask)
            intersection = np.maximum(
                np.minimum(ctd_boxes[:, 2:], [x2, y2])
                - np.maximum(ctd_boxes[:, :2], [x1, y1]), 0,
            )
            covered = np.any(
                intersection.prod(axis=1) >= 0.25 * (x2 - x1) * (y2 - y1)
            )
            # Inspect this polygon's original fine pixels, not a neighboring
            # polygon or a fallback already written into the combined mask.
            region = refined if covered and np.any(refined) else line_mask
            crop = combined[y1:y2, x1:x2]
            cv2.bitwise_or(crop, region, dst=crop)
    return combined


@register_textdetectors('ysg_ctd')
class YSGYoloCTDDetector(TextDetectorBase):
    """Keep YSG blocks and refine their masks using independently owned CTD.

    >>> detector = YSGYoloCTDDetector()
    >>> detector.name, detector.all_model_loaded()
    ('ysg_ctd', False)
    """

    dependencies = ['torch', 'ultralytics>=8.4.14', 'torchvision', 'einops']
    _load_model_keys = {'model', 'ctd_model'}

    # Literal metadata lets the existing AST scanner discover this detector
    # without importing either backend or changing the original detectors.
    params = {
        'model path': {
            'type': 'selector',
            'options': find_model_paths('data/models', ('ysgyolo', 'ultralyticsyolo')),
            'value': 'data/models/ysgyolo_yolo26_2.0.pt',
            'editable': True,
            'flush_btn': True,
            'path_selector': True,
            'path_filter': '*.pt *.ckpt *.pth *.safetensors',
            'size': 'median',
            'display_name': '模型路径',
        },
        'merge text lines': {
            'display_name': '合并文本行', 'type': 'checkbox', 'value': True,
        },
        'confidence threshold': {
            'display_name': '置信度阈值', 'type': 'line_editor', 'value': 0.3,
        },
        'IoU threshold': {
            'display_name': 'IoU阈值', 'type': 'line_editor', 'value': 0.5,
        },
        'font size multiplier': {
            'display_name': '字号乘数', 'type': 'line_editor', 'value': 1.,
        },
        'font size max': {
            'display_name': '最大字号', 'type': 'line_editor', 'value': -1,
        },
        'font size min': {
            'display_name': '最小字号', 'type': 'line_editor', 'value': -1,
        },
        'device': {**DEVICE_SELECTOR(), 'display_name': '设备'},
        'label': {
            'value': {
                'balloon': True, 'qipao': True, 'shuqing': True,
                'changfangtiao': True, 'hengxie': True, 'other': False,
            },
            'type': 'check_group',
            'display_name': '标签',
        },
        'mask dilate size': {
            'display_name': '掩码扩张尺寸', 'type': 'line_editor', 'value': 2,
        },
        'ctd detect size': {
            'display_name': 'CTD Detect Size', 'type': 'selector',
            'options': [896, 1024, 1152, 1280], 'value': 1280,
            'description': 'CTD input size on GPU; CPU uses the fixed 1024 ONNX model.',
        },
        'ctd det rearrange max batches': {
            'display_name': 'CTD Max split batch size', 'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 'value': 4,
        },
        'description': (
            'YSG text boxes with CTD glyph masks. Lines missed by CTD keep their '
            'box masks; CTD inference failure falls back to YSG masks. '
            'Settings are independent of ysgyolo and ctd.'
        ),
    }

    download_file_list = [
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ysgyolo_yolo26OBB_2.0.pt',
            'files': 'data/models/ysgyolo_yolo26OBB_2.0.pt',
            'sha256_pre_calculated': 'a9c03afb069285fc9e0d3fffc3c9ead440b687570952a945391ba22cd843dc3f',
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/ysgyolo_yolo26_2.0.pt',
            'files': 'data/models/ysgyolo_yolo26_2.0.pt',
            'sha256_pre_calculated': '889347d65c8636dd188a8ed4f312b29658543faaa69016b5958ddf0559980e22',
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt',
            'files': 'data/models/comictextdetector.pt',
            'sha256_pre_calculated': '1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb',
        },
        {
            'url': 'https://huggingface.co/dreMaz/mit_models/resolve/main/comictextdetector.pt.onnx',
            'files': 'data/models/comictextdetector.pt.onnx',
            'sha256_pre_calculated': '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f',
        },
    ]

    def __init__(self, **params) -> None:
        # BaseModule patches metadata in place; keep instance settings isolated.
        self.params = deepcopy(self.params)
        super().__init__(**params)
        self.ctd_model = None

    def _load_model(self) -> None:
        if getattr(self, 'model', None) is None:
            model_path = self.get_param_value('model path')
            if not osp.exists(model_path):
                # Prefer bundled standard weights before OBB/custom checkpoints.
                defaults = [
                    'data/models/ysgyolo_yolo26_2.0.pt',
                    'data/models/ysgyolo_yolo26OBB_2.0.pt',
                ]
                fallback = next((path for path in defaults if osp.exists(path)), None)
                if fallback is None:
                    fallback = next((path for path in self.flush('model path')
                                     if path not in defaults and osp.exists(path)), None)
                if fallback is None:
                    raise FileNotFoundError(f'YSG checkpoint not found: {model_path}. Choose an existing model path.')
                self.logger.warning(f'{model_path} does not exist; using {fallback}')
                model_path = fallback
            if 'rtdetr' in osp.basename(model_path):
                from ultralytics import RTDETR as Model
            else:
                from ultralytics import YOLO as Model
            self.model = Model(model_path).to(device=self.get_param_value('device'))
        if self.ctd_model is None:
            from .ctd import CTDModel
            from .detector_ctd import CTD_ONNX_PATH, CTD_TORCH_PATH

            device = self.get_param_value('device')
            self.ctd_model = CTDModel(
                CTD_ONNX_PATH if device == 'cpu' else CTD_TORCH_PATH,
                device=device,
                detect_size=self.get_param_value('ctd detect size'),
                det_rearrange_max_batches=self.get_param_value('ctd det rearrange max batches'),
            )

    def _detect(
        self, img: np.ndarray, proj: Optional[ProjImgTrans] = None,
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        blocks, polygons = self._detect_ysg_blocks(img)
        if not blocks:
            return np.zeros(img.shape[:2], np.uint8), blocks
        try:
            _, fine_mask, ctd_blocks = self.ctd_model(img)
        except Exception as error:
            self.logger.warning(f'CTD mask refinement failed; using YSG box masks: {error}')
            mask = np.zeros(img.shape[:2], np.uint8)
            # Keep the original detection polygons for the same fallback as YSG,
            # including any lines normalized by the block-merging utilities.
            for polygon in polygons:
                cv2.fillPoly(mask, [polygon], 255)
        else:
            mask = _combine_refined_mask(fine_mask, blocks, ctd_blocks)

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1), (ksize, ksize),
            )
            mask = cv2.dilate(mask, element)
        return mask, blocks

    def _detect_ysg_blocks(self, img: np.ndarray) -> Tuple[List[TextBlock], List[np.ndarray]]:
        """Preserve YSG block construction without building its discarded mask."""
        valid_labels = set(self.get_valid_labels())
        if not valid_labels:
            return [], []
        result = self.model.predict(
            source=img, save=False, show=False, verbose=False,
            conf=self.get_param_value('confidence threshold'),
            iou=self.get_param_value('IoU threshold'), agnostic_nms=True,
        )[0]
        valid_ids = {idx for idx, name in result.names.items() if name in valid_labels}
        if not valid_ids:
            return [], []

        items = []
        for detections, oriented in ((result.boxes, False), (result.obb, True)):
            if detections is None or len(detections.cls) == 0:
                continue
            # Transfer whole outputs once, rather than synchronizing the GPU
            # for every class id and every individual box.
            class_ids = detections.cls.cpu().numpy().astype(int)
            coordinates = detections.xyxyxyxy if oriented else detections.xyxy
            coordinates = coordinates.cpu().numpy().astype(int)
            for class_id, points in zip(class_ids, coordinates):
                if class_id not in valid_ids:
                    continue
                if not oriented:
                    x1, y1, x2, y2 = points
                    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                items.append((points, result.names[class_id]))

        if not items:
            return [], []
        height, width = img.shape[:2]
        polygons = [points for points, _ in items]
        if self.get_param_value('merge text lines'):
            blocks = mit_merge_textlines(polygons, width=width, height=height)
        else:
            blocks = []
            for points, label in items:
                points, vertical = sort_pnts(points)
                block = TextBlock(lines=[points], src_is_vertical=vertical, label=label)
                block.vertical = vertical
                block.adjust_bbox()
                examine_textblk(block, width, height)
                blocks.append(block)
        blocks = sort_regions(blocks)
        multiplier = self.get_param_value('font size multiplier')
        maximum = self.get_param_value('font size max')
        minimum = self.get_param_value('font size min')
        for block in blocks:
            size = block._detected_font_size * multiplier
            if maximum > 0:
                size = min(maximum, size)
            if minimum > 0:
                size = max(minimum, size)
            block.font_size = size
            block._detected_font_size = size
        return blocks, polygons

    def get_valid_labels(self) -> List[str]:
        return [label for label, enabled in self.get_param_value('label').items() if enabled]

    def flush(self, param_key: str) -> Optional[List[str]]:
        if param_key == 'model path':
            return find_model_paths('data/models', ('ysgyolo', 'ultralyticsyolo'))
        return None

    def updateParam(self, param_key: str, param_content: object) -> None:
        previous = self.get_param_value(param_key)
        super().updateParam(param_key, param_content)
        if self.get_param_value(param_key) == previous:
            return
        # Device/checkpoint edits defer loading to the existing module worker.
        if param_key == 'device':
            self.unload_model()
        elif param_key == 'model path':
            self.model = None
        elif self.ctd_model is not None:
            # CTD reads these attributes on every inference; its weights do not
            # depend on either value, so changing them needs no model reload.
            if param_key == 'ctd detect size':
                self.ctd_model.detect_size = self.get_param_value(param_key)
            elif param_key == 'ctd det rearrange max batches':
                self.ctd_model.det_rearrange_max_batches = self.get_param_value(param_key)
