"""Koharu Layout HighRes text regions and instance masks, using RF-DETR."""

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .base import DEVICE_SELECTOR, TextBlock, TextDetectorBase, register_textdetectors
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import examine_textblk, sort_regions
from ballontranslator.utils.bubble_shape import partition_bubble, trace_bubble

if TYPE_CHECKING:
    import torch


def _mask_angle(mask: np.ndarray, box: np.ndarray) -> Tuple[float, bool]:
    """Infer rotation and vertical writing from Koharu's mask projection profiles.

    >>> mask = np.zeros((90, 60), bool)
    >>> mask[10:14, 10:50] = mask[40:44, 10:50] = mask[70:74, 10:50] = True
    >>> _mask_angle(mask, np.array([0, 0, 60, 90]))
    (0.0, False)
    """
    # Match koharu-pipeline/stages/detection.rs: sample pixel centers within a 2px margin.
    left, top, right, bottom = box
    height, width = mask.shape
    x1, y1 = max(0, math.floor(left - 2)), max(0, math.floor(top - 2))
    x2, y2 = min(width, math.ceil(right + 2)), min(height, math.ceil(bottom + 2))
    y, x = np.nonzero(mask[y1:y2, x1:x2])
    fallback_vertical = bool(bottom - top > right - left)
    if not len(x):
        return 0.0, fallback_vertical
    x = x.astype(np.float64) + x1 + 0.5
    y = y.astype(np.float64) + y1 + 0.5
    min_x, max_x, min_y, max_y = x.min(), x.max(), y.min(), y.max()
    scores = [float('-inf'), float('-inf')]
    angles = [0.0, 0.0]
    for step in range(-90, 91):
        angle = step * 0.5
        radians = math.radians(angle)
        sin, cos = math.sin(radians), math.cos(radians)
        for direction, (axis_x, axis_y) in enumerate(((-sin, cos), (cos, sin))):
            minimum = (min_x if axis_x >= 0 else max_x) * axis_x + (min_y if axis_y >= 0 else max_y) * axis_y
            maximum = (max_x if axis_x >= 0 else min_x) * axis_x + (max_y if axis_y >= 0 else min_y) * axis_y
            origin = math.floor(minimum)
            length = math.ceil(maximum) - origin + 2
            projection = x * axis_x + y * axis_y - origin
            indices = projection.astype(np.intp)
            fraction = projection - indices
            histogram = np.bincount(indices, weights=1 - fraction, minlength=length)
            histogram += np.bincount(indices + 1, weights=fraction, minlength=length)
            score = float(histogram @ histogram) / len(x)
            if score > scores[direction]:
                scores[direction], angles[direction] = score, angle
    close = abs(scores[0] - scores[1]) <= max(scores) * 0.02
    vertical = fallback_vertical if close else scores[1] > scores[0]
    angle = angles[int(vertical)]
    return (0.0 if abs(angle) < 3 else angle), vertical


@register_textdetectors('koharu')
class KoharuDetector(TextDetectorBase):
    """Load local Koharu weights only when the module worker requests them.

    >>> detector = KoharuDetector()
    >>> detector.name, detector.all_model_loaded()
    ('koharu', False)
    >>> detector.get_param_value('label')
    {'text': True, 'onomatopoeia': True}
    """

    dependencies = [
        'torch', 'torchvision', 'rfdetr==1.7.0', 'transformers>=5.1,<6', 'safetensors',
    ]
    download_file_list = [{
        'url': 'https://huggingface.co/mayocream/koharu-layout-highres/resolve/68077eb09401a46e548190ff2786bd405113ea8b/',
        'files': ['model.safetensors', 'config.json'],
        'sha256_pre_calculated': [
            '03de2f77281974f5084cdd8652ace2b55dc15ac0378aed66654e62537cdda255',
            'ef4697d18fcdab4e9237e1e6890ea6e7efe4efa063f0088e3600e43fb72a7c29',
        ],
        'save_dir': 'data/models/koharu-layout-highres',
        'concatenate_url_filename': 1,
    }]
    _load_model_keys = {'model'}
    params = {
        'model path': {
            'type': 'selector',
            'options': ['data/models/koharu-layout-highres/model.safetensors'],
            'value': 'data/models/koharu-layout-highres/model.safetensors',
            'editable': True, 'path_selector': True,
            'path_filter': '*.safetensors', 'size': 'median',
            'display_name': 'Model path',
            'description': 'Default files download automatically. Custom weights need config.json in the same folder.',
        },
        'device': DEVICE_SELECTOR(not_supported=['xpu', 'privateuseone']),
        'detect size': {
            'type': 'selector', 'options': [768, 1152, 1536, 1920], 'value': 1152,
            'display_name': 'Detect size',
            'description': '1152 by default. Larger sizes may recover small text but use more memory.',
        },
        'text threshold': {
            'value': 0.25, 'display_name': 'Text threshold',
            'description': 'Minimum text confidence, from 0 to 1.',
        },
        'onomatopoeia threshold': {
            'value': 0.3, 'display_name': 'SFX threshold',
            'description': 'Minimum sound-effect confidence, from 0 to 1.',
        },
        'label': {
            'type': 'check_group', 'value': {'text': True, 'onomatopoeia': True},
            'display_name': 'Types',
            'description': (
                'text: dialogue/captions. onomatopoeia: sound effects. '
                'Unchecked types add no blocks or cleanup masks.'
            ),
        },
        'mask postprocessing': {
            'type': 'selector', 'options': ['koharu', 'dilate', 'ctd'], 'value': 'koharu',
            'display_name': 'Mask postprocessing',
            'description': (
                'koharu (default): page-scaled dilation then closing, using 6 pixels per 1024 pixels of the long side. '
                'These are fixed upstream settings; Mask dilation radius is ignored. '
                'dilate: fixed-radius expansion. ctd: image-color refinement then fixed-radius expansion, '
                'without loading a CTD model.'
            ),
        },
        'mask dilate size': {
            'value': 2, 'display_name': 'Mask dilation radius',
            'description': 'Extra expansion in page pixels for dilate/ctd; 0 disables this expansion. Ignored by koharu.',
        },
        'font size multiplier': {
            'value': 1.0, 'display_name': 'Font size multiplier',
            'description': 'Scales the fallback size. Auto layout instead fits translated text to the flow area.',
        },
        'description': (
            'Koharu Layout HighRes (RF-DETR). Requires Python 3.10+ and Transformers 5. '
            'Uses local weights, region boxes and segmentation masks; no CTD model is needed. '
            'Text direction and rotation are inferred from each instance mask. '
            'Some short dialogue may be classified as onomatopoeia.'
        ),
    }

    def __init__(self, **params: object) -> None:
        self.params = deepcopy(self.params)
        self.model = None
        self.num_select = 160
        # Sanitize saved values before BaseModule's permissive type coercion.
        clean = {}
        for key, content in params.items():
            if key.startswith('__') or key == 'description':
                continue
            value = content.get('value', content) if isinstance(content, dict) else content
            if key == 'label' and isinstance(value, dict):
                labels = self.params['label']['value'].copy()
                for label, enabled in value.items():
                    if label in labels and isinstance(enabled, bool):
                        labels[label] = enabled
                    else:
                        self.logger.warning('Discard invalid Koharu label %r: %r', label, enabled)
                value = labels
            try:
                clean[key] = self._validate_param(key, value)
            except (ValueError, TypeError, OverflowError) as error:
                self.logger.warning('Discard invalid Koharu setting %r: %s', key, error)
        super().__init__(**clean)

    def _validate_param(self, key: str, value: object) -> object:
        if key not in self.params or key.startswith('__') or key == 'description':
            raise ValueError(f'Unknown setting: {key}')
        if key == 'label':
            if (not isinstance(value, dict) or set(value) != {'text', 'onomatopoeia'}
                    or any(not isinstance(enabled, bool) for enabled in value.values())):
                raise ValueError('Labels must specify true/false for text and onomatopoeia')
            return value.copy()
        if key == 'mask postprocessing':
            if value not in self.params[key]['options']:
                raise ValueError('Mask postprocessing must be dilate, koharu or ctd')
            return value
        if key in {'model path', 'device'}:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a non-empty string')
            if key == 'device' and value not in {'cpu', 'cuda', 'mps'}:
                raise ValueError('Device must be cpu, cuda or mps')
            return value
        if isinstance(value, bool):
            raise ValueError(f'{key} must be numeric')
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f'{key} must be finite')
        if key.endswith('threshold') and not 0 <= number <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
        if key == 'font size multiplier' and number <= 0:
            raise ValueError('Font size multiplier must be positive')
        if key in {'detect size', 'mask dilate size'}:
            if not number.is_integer():
                raise ValueError(f'{key} must be an integer')
            if key == 'detect size' and int(number) not in self.params[key]['options']:
                raise ValueError('Detect size must be 768, 1152, 1536 or 1920')
            if key == 'mask dilate size' and not 0 <= number <= 100:
                raise ValueError('Mask dilation radius must be between 0 and 100')
            return int(number)
        return number

    def updateParam(self, param_key: str, param_content: object) -> None:
        try:
            value = self._validate_param(param_key, param_content)
        except (ValueError, TypeError, OverflowError) as error:
            # Settings edits arrive through Qt slots; reject without escaping into Qt.
            self.logger.warning('Rejected Koharu setting %r: %s', param_key, error)
            return
        if value == self.get_param_value(param_key):
            return
        self.set_param_value(param_key, value, convert_dtype=False)
        if param_key in {'model path', 'device'}:
            # The existing module worker owns the next load, never the settings UI.
            self.unload_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        path = Path(self.get_param_value('model path'))
        config_path = path.with_name('config.json')
        if not path.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                'Koharu needs model.safetensors and config.json together. '
                f'Check Model path (currently {path}), or select the default path '
                'and reload Koharu to download its files automatically.'
            )
        with config_path.open(encoding='utf-8') as handle:
            config = json.load(handle)
        if (not isinstance(config, dict) or config.get('architecture') != 'RFDETRSeg2XLarge'
                or config.get('classes') != {'0': 'text', '1': 'onomatopoeia', '2': 'bubble', '3': 'panel'}):
            raise ValueError('Expected a Koharu RFDETRSeg2XLarge config with text, onomatopoeia, bubble and panel classes')
        resolution, num_select = config.get('resolution'), config.get('num_select')
        if type(resolution) is not int or resolution <= 0 or resolution % 24:
            raise ValueError('Koharu config resolution must be a positive multiple of 24')
        if type(num_select) is not int or not 1 <= num_select <= 1500:
            raise ValueError('Koharu config num_select must be between 1 and 1500')
        try:
            from rfdetr import RFDETRSeg2XLarge
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                'Koharu requires Python 3.10+, rfdetr==1.7.0, transformers>=5.1,<6 '
                'and safetensors. Install its optional dependencies in the module settings.'
            ) from error
        # Keep checkpoint positional embeddings at their trained size even when
        # inference is smaller; pretrain_weights=None prevents unrelated downloads.
        model = RFDETRSeg2XLarge(
            pretrain_weights=None, resolution=resolution,
            positional_encoding_size=resolution // 12,
            num_classes=4, num_select=num_select, device='cpu',
        ).model.model
        model.load_state_dict(load_file(str(path)), strict=True)
        model = model.to(self.get_param_value('device')).eval()
        self.num_select = num_select
        self.model = model

    def _detect(
        self, img: np.ndarray, proj: Optional[ProjImgTrans] = None,
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        import torch
        import torch.nn.functional as F

        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8 or not all(img.shape[:2]):
            raise ValueError('Koharu expects a non-empty RGB uint8 image')
        if not any(self.get_param_value('label').values()):
            return np.zeros(img.shape[:2], np.uint8), []
        size = self.get_param_value('detect size')
        with torch.inference_mode():
            pixels = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)
            pixels = pixels.to(device=self.get_param_value('device'), dtype=torch.float32) / 255
            # Match Koharu's bilinear RGB resize and ImageNet normalization.
            pixels = F.interpolate(pixels, (size, size), mode='bilinear', align_corners=False)
            mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            output = self.model((pixels - mean) / std)
            mask, blocks = self._postprocess(output, img.shape[:2])
        return self._refine_mask(img, mask, blocks), blocks

    def _postprocess(
        self, output: Dict[str, 'torch.Tensor'], image_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        """Select query/class pairs before projecting masks, as in Koharu's processor.

        >>> import torch
        >>> output = {'pred_logits': torch.full((1, 300, 5), -20.),
        ...           'pred_boxes': torch.zeros(1, 300, 4),
        ...           'pred_masks': torch.zeros(1, 300, 1, 1)}
        >>> mask, blocks = KoharuDetector()._postprocess(output, (4, 6))
        >>> mask.shape, bool(mask.any()), blocks
        ((4, 6), False, [])
        """
        import torch
        import torch.nn.functional as F

        height, width = image_shape
        mask = np.zeros(image_shape, np.uint8)
        logits = output['pred_logits'][0]
        scores, indexes = logits.sigmoid().flatten().topk(self.num_select)
        labels = indexes % logits.shape[-1]
        queries = indexes // logits.shape[-1]
        enabled = self.get_param_value('label')
        thresholds = scores.new_tensor([
            self.get_param_value('text threshold') if enabled['text'] else float('inf'),
            self.get_param_value('onomatopoeia threshold') if enabled['onomatopoeia'] else float('inf'),
            .5, float('inf'), float('inf'),
        ])
        keep = scores > thresholds[labels]
        queries = queries[keep]
        labels = labels[keep].cpu().tolist()
        boxes = output['pred_boxes'][0, queries]
        boxes = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), dim=-1)
        boxes = (boxes * boxes.new_tensor([width, height, width, height])).cpu().numpy()
        # Transfer native masks once; projecting one at a time avoids an N-by-page allocation.
        masks = output['pred_masks'][0, queries].cpu()
        native_masks = masks.numpy() > 0
        bubbles = sorted((i for i, label in enumerate(labels) if label == 2),
                         key=lambda i: np.count_nonzero(native_masks[i]))
        associations = {i: [] for i in bubbles}
        blocks = []
        multiplier = self.get_param_value('font size multiplier')
        for index, (label, box, instance) in enumerate(zip(labels, boxes, masks)):
            if label == 2:
                continue
            if not np.isfinite(box).all():
                raise ValueError('Koharu returned non-finite box coordinates')
            x1, y1 = np.maximum(np.floor(box[:2]), 0).astype(int)
            x2, y2 = np.minimum(np.ceil(box[2:]), [width, height]).astype(int)
            if x2 <= x1 or y2 <= y1:
                continue
            projected = F.interpolate(instance[None, None], image_shape, mode='bilinear', align_corners=False)
            foreground = projected[0, 0].numpy() > 0
            mask[foreground] = 255
            angle, vertical = _mask_angle(foreground, box)
            block = TextBlock(
                xyxy=[x1, y1, x2, y2], src_is_vertical=vertical,
                label=('text', 'onomatopoeia')[label],
            )
            block.set_lines_by_xywh([x1, y1, x2 - x1, y2 - y1], angle=-angle)
            block.vertical = vertical
            examine_textblk(block, width, height)
            # Keep half-degree precision after the integer line-geometry estimates.
            block.angle = angle
            # The OCR crop is a region, not a measured source line. Keep a
            # numeric fallback; automatic typesetting measures its own range.
            block.font_size = max(1., min(x2 - x1, y2 - y1) * .25) * multiplier
            block._detected_font_size = -1
            blocks.append(block)
            foreground_y, foreground_x = np.nonzero(native_masks[index])
            foreground_count = len(foreground_x)
            if foreground_count:
                for bubble in bubbles:
                    if np.count_nonzero(native_masks[bubble, foreground_y, foreground_x]) >= .9 * foreground_count:
                        associations[bubble].append(block)
                        break
        for bubble, members in associations.items():
            if not members:
                continue
            bubble_mask = native_masks[bubble]
            if len(members) == 1:
                members[0].bubble_shape = trace_bubble(
                    bubble_mask, (width, height), members[0].angle, tuple(members[0].center()))
                members[0].bubble_layout = members[0].bubble_shape is not None
                continue
            anchors = np.array([(np.array(b.xyxy[:2]) + b.xyxy[2:]) / 2 for b in members])
            for block, cell in zip(members, partition_bubble(bubble_mask, (width, height), anchors)):
                block.bubble_shape = trace_bubble(cell, (width, height), block.angle, tuple(block.center()))
                block.bubble_layout = block.bubble_shape is not None
        return mask, sort_regions(blocks)

    def _refine_mask(self, img: np.ndarray, mask: np.ndarray, blocks: List[TextBlock]) -> np.ndarray:
        """Clean the union of enabled instances without changing detection geometry.

        >>> detector = KoharuDetector()
        >>> mask = np.zeros((16, 16), np.uint8)
        >>> mask[8, 8] = 255
        >>> cleaned = detector._refine_mask(np.zeros((16, 16, 3), np.uint8), mask, [])
        >>> int(np.count_nonzero(cleaned)), int(np.count_nonzero(mask))
        (5, 1)
        """
        if not mask.any():
            return mask
        mode = self.get_param_value('mask postprocessing')
        if mode == 'koharu':
            radius = min(255, max(1, math.floor(max(mask.shape) * 6 / 1024 + 0.5)))
            # Full-page L2 morphology preserves interactions and page-edge neighborhoods.
            for _ in range(2):
                distance = cv2.distanceTransform(255 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                mask = (distance <= radius).astype(np.uint8) * 255
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            return (distance > radius).astype(np.uint8) * 255
        if mode == 'ctd':
            from .ctd.textmask import refine_mask

            mask = refine_mask(img, mask, blocks)
        radius = self.get_param_value('mask dilate size')
        if radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            mask = cv2.dilate(mask, kernel)
        return mask
