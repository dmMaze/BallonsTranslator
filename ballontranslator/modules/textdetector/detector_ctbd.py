# detector_ctbd.py
import os
import math
import cv2
import numpy as np
from typing import Tuple, List, Optional, Callable

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR, ProjImgTrans
from ballontranslator.utils.logger import logger as LOGGER


def calculate_iou(rect1, rect2) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    >>> calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
    1.0
    >>> calculate_iou((0, 0, 10, 10), (5, 0, 15, 10))
    0.3333333333333333
    >>> calculate_iou((0, 0, 10, 10), (20, 20, 30, 30))
    0.0
    """
    x1, y1, x2, y2 = rect1
    px1, py1, px2, py2 = rect2
    xi1, yi1, xi2, yi2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    r1_area, r2_area = (x2 - x1) * (y2 - y1), (px2 - px1) * (py2 - py1)
    union_area = r1_area + r2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def do_rectangles_overlap(rect1, rect2, iou_threshold: float = 0.2) -> bool:
    """Check if two rectangles overlap with an IoU greater than or equal to a threshold.

    >>> do_rectangles_overlap((0, 0, 10, 10), (5, 0, 15, 10), 0.2)
    True
    >>> do_rectangles_overlap((0, 0, 10, 10), (20, 20, 30, 30), 0.2)
    False
    """
    return calculate_iou(rect1, rect2) >= iou_threshold


def does_rectangle_fit(bigger_rect, smaller_rect) -> bool:
    """Check if smaller_rect fits entirely inside bigger_rect.

    >>> does_rectangle_fit((0, 0, 10, 10), (2, 2, 8, 8))
    True
    >>> does_rectangle_fit((2, 2, 8, 8), (0, 0, 10, 10))
    False
    """
    x1, y1, x2, y2 = bigger_rect
    px1, py1, px2, py2 = smaller_rect
    return x1 <= px1 and y1 <= py1 and x2 >= px2 and y2 >= py2


def filter_bounding_boxes(bboxes, width_tolerance=5, height_tolerance=5):
    if bboxes is None or len(bboxes) == 0:
        return np.array([])
    return np.array(
        [
            b
            for b in bboxes
            if (b[2] - b[0]) > width_tolerance and (b[3] - b[1]) > height_tolerance
        ]
    )


def adjust_text_line_coordinates(line, x_off, y_off, image_shape):
    if line is None or len(line) != 4:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = line
    return (
        int(np.clip(x1 - x_off, 0, w)),
        int(np.clip(y1 - y_off, 0, h)),
        int(np.clip(x2 + x_off, 0, w)),
        int(np.clip(y2 + y_off, 0, h)),
    )


def detect_content_in_bbox(image_crop):
    if image_crop is None or image_crop.size == 0:
        return []
    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    bboxes = []
    h_crop, w_crop = image_crop.shape[:2]
    min_area = 10
    for thresh_type in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, 11, 2
        )
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if x > 0 and y > 0 and x + bw < w_crop and y + bh < h_crop:
                    bboxes.append((x, y, x + bw, y + bh))
    return bboxes


def get_inpaint_bboxes(text_bbox_abs, full_image, logger_instance=None):
    coords = adjust_text_line_coordinates(text_bbox_abs, 0, 10, full_image.shape)
    if coords is None:
        return []
    x1_crop, y1_crop, x2_crop, y2_crop = coords
    if y1_crop >= y2_crop or x1_crop >= x2_crop:
        return []
    crop_img = full_image[y1_crop:y2_crop, x1_crop:x2_crop]
    content_bboxes_relative = detect_content_in_bbox(crop_img)
    return [
        (x1_crop + lx1, y1_crop + ly1, x1_crop + lx2, y1_crop + ly2)
        for lx1, ly1, lx2, ly2 in content_bboxes_relative
    ]


def create_unified_mask(input_mask: np.ndarray, method: str = "hull") -> np.ndarray:
    if not isinstance(input_mask, np.ndarray) or input_mask.ndim != 2:
        return np.zeros_like(input_mask)
    contours, _ = cv2.findContours(
        input_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return np.zeros_like(input_mask)

    all_points = np.concatenate(contours, axis=0)
    unified_mask = np.zeros_like(input_mask)

    if method == "rectangle":
        x, y, w, h = cv2.boundingRect(all_points)
        cv2.rectangle(unified_mask, (x, y), (x + w, y + h), 255, -1)
    elif method == "hull":
        hull = cv2.convexHull(all_points)
        cv2.drawContours(unified_mask, [hull], 0, 255, -1)
    else:
        return input_mask
    return unified_mask


def merge_duplicate_boxes(boxes: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Merge boxes that overlap with an IoU greater than or equal to iou_thresh.

    >>> boxes = np.array([[10, 10, 50, 50], [12, 12, 48, 48], [100, 100, 150, 150]])
    >>> merge_duplicate_boxes(boxes, 0.7).tolist()
    [[10, 10, 50, 50], [100, 100, 150, 150]]
    """
    if boxes is None or len(boxes) < 2:
        return boxes if boxes is not None else np.array([])

    n = len(boxes)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if calculate_iou(boxes[i], boxes[j]) >= iou_thresh:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    components = []
    for i in range(n):
        if i not in visited:
            component = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                component.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)

    merged = []
    for comp in components:
        comp_boxes = boxes[comp]
        x1 = np.min(comp_boxes[:, 0])
        y1 = np.min(comp_boxes[:, 1])
        x2 = np.max(comp_boxes[:, 2])
        y2 = np.max(comp_boxes[:, 3])
        merged.append([x1, y1, x2, y2])

    return np.array(merged)


def remove_contained_boxes(boxes: np.ndarray, threshold: float) -> np.ndarray:
    """Remove boxes of the same category that are largely contained within another box.
    """
    if boxes is None or len(boxes) < 2:
        return boxes if boxes is not None else np.array([])

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # Sort by area descending (largest first)
    idxs = np.argsort(areas)[::-1]
    sorted_boxes = boxes[idxs]

    keep = []
    for box in sorted_boxes:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue

        is_contained = False
        for kept_box in keep:
            kx1, ky1, kx2, ky2 = kept_box
            # Calculate intersection
            ix1 = max(x1, kx1)
            iy1 = max(y1, ky1)
            ix2 = min(x2, kx2)
            iy2 = min(y2, ky2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection_area = iw * ih

            if intersection_area / area >= threshold:
                is_contained = True
                break

        if not is_contained:
            keep.append(box)

    return np.array(keep)


class ImageSlicer:
    def __init__(self, logger, **kwargs):
        self.logger = logger
        try:
            self.height_to_width_ratio_threshold = float(
                kwargs.get("slice_threshold_ratio", 3.5)
            )
            self.target_slice_ratio = float(kwargs.get("slice_target_ratio", 3.0))
            self.overlap_height_ratio = float(kwargs.get("slice_overlap_ratio", 0.2))
            self.min_slice_height_ratio = float(
                kwargs.get("slice_min_height_ratio", 0.7)
            )
            self.merge_iou_threshold = float(kwargs.get("slice_merge_iou", 0.2))
            self.duplicate_iou_threshold = float(kwargs.get("slice_duplicate_iou", 0.5))
            self.merge_y_distance_threshold_ratio = float(
                kwargs.get("slice_merge_y_dist", 0.1)
            )
            self.containment_threshold = float(
                kwargs.get("slice_containment_thresh", 0.85)
            )
        except ValueError as e:
            self.logger.error(f"Invalid parameter value for ImageSlicer: {e}")
            self._set_default_slicer_params()

    def _set_default_slicer_params(self):
        self.height_to_width_ratio_threshold, self.target_slice_ratio = 3.5, 3.0
        self.overlap_height_ratio, self.min_slice_height_ratio = 0.2, 0.7
        self.merge_iou_threshold, self.duplicate_iou_threshold = 0.2, 0.5
        self.merge_y_distance_threshold_ratio, self.containment_threshold = 0.1, 0.85
        self.logger.warning("ImageSlicer falling back to default parameters.")

    def should_slice(self, image: np.ndarray) -> bool:
        h, w = image.shape[:2]
        return w > 0 and (h / w) > self.height_to_width_ratio_threshold

    def calculate_slice_params(self, image_shape: tuple) -> tuple[int, int, int, int]:
        h, w = image_shape[:2]
        slice_w = w
        slice_h = max(1, int(slice_w * self.target_slice_ratio)) if w > 0 else h
        eff_h = max(1, int(slice_h * (1 - self.overlap_height_ratio)))
        num_s = math.ceil(h / eff_h) if eff_h > 0 else 1
        if (
            num_s > 1
            and slice_h > 0
            and (h - (num_s - 1) * eff_h) / slice_h < self.min_slice_height_ratio
        ):
            num_s -= 1
        return slice_w, slice_h, eff_h, max(1, num_s)

    def get_slice(
        self, image: np.ndarray, idx: int, eff_h: int, slice_h: int
    ) -> tuple[np.ndarray, int, int]:
        h, w = image.shape[:2]
        start_y = idx * eff_h
        _, _, _, num_s = self.calculate_slice_params(image.shape)
        end_y = h if idx == num_s - 1 else min(start_y + slice_h, h)
        start_y = min(start_y, h - 1)
        end_y = max(start_y + 1, end_y)
        if start_y >= end_y:
            return np.array([]), start_y, end_y
        return image[start_y:end_y, 0:w].copy(), start_y, end_y

    def adjust_box_coordinates(self, boxes: np.ndarray, start_y: int) -> np.ndarray:
        if boxes is None or boxes.size == 0:
            return np.array([])
        adj = boxes.copy()
        adj[:, [1, 3]] += start_y
        return adj

    def box_contained(self, b1, b2) -> tuple[bool, float, int]:
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        if a1 <= 0 or a2 <= 0:
            return False, 0, 0
        xi1, yi1, xi2, yi2 = (
            max(b1[0], b2[0]),
            max(b1[1], b2[1]),
            min(b1[2], b2[2]),
            min(b1[3], b2[3]),
        )
        ia = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        sa = min(a1, a2)
        ratio = ia / sa if sa > 0 else 0.0
        is_contained = ratio >= self.containment_threshold
        which = 1 if a1 >= a2 else 2 if is_contained else 0
        return is_contained, ratio, which

    def merge_overlapping_boxes(self, boxes: np.ndarray, img_h: int) -> np.ndarray:
        if boxes is None or boxes.size < 2:
            return boxes if boxes is not None else np.array([])
        bl = sorted(boxes.tolist(), key=lambda b: b[1])
        merge_y_dist_pixels = self.merge_y_distance_threshold_ratio * img_h
        i = 0
        while i < len(bl):
            j = i + 1
            merged_current = False
            while j < len(bl):
                b1, b2 = bl[i], bl[j]
                if b2[1] > b1[3] + merge_y_dist_pixels * 2:
                    break
                iou = calculate_iou(b1, b2)
                is_cont, _, which_contains = self.box_contained(b1, b2)
                if is_cont:
                    if which_contains == 1:
                        bl.pop(j)
                    else:
                        bl[i] = b2
                        bl.pop(j)
                        merged_current = True
                        break
                    continue
                if iou >= self.duplicate_iou_threshold:
                    a1, a2 = (b1[2] - b1[0]) * (b1[3] - b1[1]), (b2[2] - b2[0]) * (
                        b2[3] - b2[1]
                    )
                    if a2 > a1:
                        bl[i] = b2
                    bl.pop(j)
                    merged_current = True
                    break
                y_dist = min(abs(b1[1] - b2[3]), abs(b1[3] - b2[1]))
                x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                w1, w2 = (b1[2] - b1[0]), (b2[2] - b2[0])
                min_w = min(w1, w2) if min(w1, w2) > 0 else 1.0
                x_overlap_ratio = x_overlap / min_w
                a1, a2 = max(0, w1 * (b1[3] - b1[1])), max(0, w2 * (b2[3] - b2[1]))
                max_a = max(a1, a2)
                size_ratio = min(a1, a2) / max_a if max_a > 0 else 0.0
                if (
                    y_dist < merge_y_dist_pixels
                    and x_overlap_ratio > self.merge_iou_threshold
                    and size_ratio > 0.3
                ):
                    merged_box = [
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3]),
                    ]
                    if (
                        max_a == 0
                        or (merged_box[2] - merged_box[0])
                        * (merged_box[3] - merged_box[1])
                        <= 3 * max_a
                    ):
                        bl[i] = merged_box
                        bl.pop(j)
                        merged_current = True
                        break
                j += 1
            if not merged_current:
                i += 1
        return np.array(bl) if bl else np.array([])

    def process_slices_for_detection(
        self, image: np.ndarray, detect_func: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.should_slice(image):
            b, t = detect_func(image)
            return np.array(b if b is not None else []), np.array(
                t if t is not None else []
            )
        _, slice_h, eff_h, num_s = self.calculate_slice_params(image.shape)
        all_b, all_t = [], []
        for i in range(num_s):
            slice_img, start_y, _ = self.get_slice(image, i, eff_h, slice_h)
            if slice_img.size == 0:
                continue
            b_s, t_s = detect_func(slice_img)
            if b_s.size > 0:
                all_b.append(self.adjust_box_coordinates(b_s, start_y))
            if t_s.size > 0:
                all_t.append(self.adjust_box_coordinates(t_s, start_y))
        cb = np.vstack(all_b) if all_b else np.array([])
        ct = np.vstack(all_t) if all_t else np.array([])
        return self.merge_overlapping_boxes(
            cb, image.shape[0]
        ), self.merge_overlapping_boxes(ct, image.shape[0])


@register_textdetectors("ctbd")
class RTDetrV2TextDetector(TextDetectorBase):
    """RT-DETR-V2 text and speech bubble detector.

    >>> detector = RTDetrV2TextDetector()
    >>> detector.name
    'ctbd'
    """
    dependencies = ['onnxruntime']

    download_file_list = [
        {
            'url': 'https://huggingface.co/ogkalu/comic-text-and-bubble-detector/resolve/main/detector.onnx',
            'files': 'data/models/CTBD/detector.onnx',
            'sha256_pre_calculated': '065744e91c0594ad8663aa8b870ce3fb27222942eded5a3cc388ce23421bd195'
        }
    ]

    params = {
        'confidence_threshold': {
            'value': 0.3,
            'type': 'line_editor',
            'description': 'Minimum detection score (0.0-1.0).',
            'display_name': 'Confidence Threshold'
        },
        'inpaint_mask_dilate': {
            'value': 4,
            'type': 'line_editor',
            'description': 'Dilation kernel size (px) for the inpaint mask. Merges text fragments.',
            'display_name': 'Inpaint Mask Dilate'
        },
        'mask_unification_method': {
            'value': 'none',
            'type': 'selector',
            'options': ['none', 'rectangle', 'hull'],
            'description': "Method to unify fragments into one mask: 'none', 'bounding rectangle', or 'convex hull'.",
            'display_name': 'Mask Unification Method'
        },
        'device': DEVICE_SELECTOR(),
        'description': 'RT-DETR-V2 (ogkalu/comic-text-and-bubble-detector) for text/bubble detection.',
        'slice_threshold_ratio': {
            'value': 3.5,
            'type': 'line_editor',
            'description': 'H/W ratio to trigger image slicing.',
            'display_name': 'Slice Threshold Ratio'
        },
        'slice_target_ratio': {
            'value': 3.0,
            'type': 'line_editor',
            'description': 'Target H/W ratio for slices.',
            'display_name': 'Slice Target Ratio'
        },
        'slice_overlap_ratio': {
            'value': 0.2,
            'type': 'line_editor',
            'description': 'Slice overlap ratio (0.0-1.0).',
            'display_name': 'Slice Overlap Ratio'
        },
        'slice_min_height_ratio': {
            'value': 0.7,
            'type': 'line_editor',
            'description': 'Minimum height ratio for the last slice.',
            'display_name': 'Slice Min Height Ratio'
        },
        'slice_merge_iou': {
            'value': 0.2,
            'type': 'line_editor',
            'description': 'IoU threshold for merging text lines.',
            'display_name': 'Slice Merge IoU'
        },
        'slice_duplicate_iou': {
            'value': 0.5,
            'type': 'line_editor',
            'description': 'IoU threshold for removing duplicate boxes.',
            'display_name': 'Slice Duplicate IoU'
        },
        'slice_merge_y_dist': {
            'value': 0.1,
            'type': 'line_editor',
            'description': 'Max relative Y-distance for merging text lines.',
            'display_name': 'Slice Merge Y Distance'
        },
        'slice_containment_thresh': {
            'value': 0.85,
            'type': 'line_editor',
            'description': 'Containment threshold for merging boxes.',
            'display_name': 'Slice Containment Threshold'
        },
        'detect_bubbles': {
            'value': True,
            'type': 'checkbox',
            'description': 'Detect speech bubbles.',
            'display_name': 'Detect Bubbles'
        },
        'detect_text': {
            'value': True,
            'type': 'checkbox',
            'description': 'Detect text blocks.',
            'display_name': 'Detect Text'
        },
        # options UI shows raw values; no option i18n yet:
        # all=全部, text_bubble=框內字, text_free=框外字
        'text_region_filter': {
            'value': 'all',
            'type': 'selector',
            'options': ['all', 'text_bubble', 'text_free'],
            'description': 'Keep all text, only text inside speech bubbles, or only text outside speech bubbles.',
            'display_name': 'Text Region Filter'
        },
        'merge_duplicates': {
            'value': True,
            'type': 'checkbox',
            'description': 'Merge near-identical duplicate boxes of the same category.',
            'display_name': 'Merge Duplicates'
        },
        'merge_duplicates_iou': {
            'value': 0.7,
            'type': 'line_editor',
            'description': 'IoU threshold for merging duplicates within the same class.',
            'display_name': 'Merge Duplicates IoU'
        },
        'remove_contained_text': {
            'value': True,
            'type': 'checkbox',
            'description': 'Remove text boxes that are largely contained within other larger text boxes.',
            'display_name': 'Remove Contained Text'
        },
        'containment_threshold': {
            'value': 0.8,
            'type': 'line_editor',
            'description': 'Area ratio threshold to trigger removal of contained boxes (0.0-1.0).',
            'display_name': 'Containment Threshold'
        }
    }

    _load_model_keys = {'model', 'image_slicer'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None
        self.image_slicer = None

    @property
    def device(self):
        return self.params['device']['value']

    @property
    def confidence_threshold(self) -> float:
        return float(self.get_param_value('confidence_threshold'))

    @property
    def inpaint_mask_dilate(self) -> int:
        return int(self.get_param_value('inpaint_mask_dilate'))

    @property
    def detect_bubbles(self) -> bool:
        return bool(self.get_param_value('detect_bubbles'))

    @property
    def detect_text(self) -> bool:
        return bool(self.get_param_value('detect_text'))

    @property
    def text_region_filter(self) -> str:
        value = self.get_param_value('text_region_filter')
        return value if value in {'all', 'text_bubble', 'text_free'} else 'all'

    @property
    def merge_duplicates(self) -> bool:
        return bool(self.get_param_value('merge_duplicates'))

    @property
    def merge_duplicates_iou(self) -> float:
        return float(self.get_param_value('merge_duplicates_iou'))

    @property
    def remove_contained_text(self) -> bool:
        return bool(self.get_param_value('remove_contained_text'))

    @property
    def containment_threshold(self) -> float:
        return float(self.get_param_value('containment_threshold'))

    @property
    def mask_unification_method(self) -> str:
        return self.get_param_value('mask_unification_method')

    def _initialize_slicer(self):
        slicer_params = {
            k: self.get_param_value(k)
            for k in self.params.keys()
            if k.startswith('slice_')
        }
        self.image_slicer = ImageSlicer(self.logger, **slicer_params)

    def _load_model(self):
        use_cuda = self.device.startswith('cuda')

        # Preload torch/lib before importing onnxruntime to resolve CUDA/cuDNN DLL paths on Windows
        try:
            import torch
            if hasattr(os, 'add_dll_directory'):
                torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
                if os.path.exists(torch_lib_dir):
                    os.add_dll_directory(torch_lib_dir)
        except Exception:
            pass

        import onnxruntime as ort

        providers = ['CPUExecutionProvider']
        if use_cuda:
            available_providers = []
            try:
                available_providers = ort.get_available_providers()
            except Exception as e:
                self.logger.warning(f"Could not get available ONNX Runtime providers: {e}")

            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.logger.warning(
                    "CUDA device selected, but CUDAExecutionProvider is not available in onnxruntime.\n"
                    "Please install onnxruntime-gpu if you want GPU acceleration.\n"
                    "Falling back to CPU execution."
                )

        model_path = 'data/models/CTBD/detector.onnx'
        self.logger.info(f"Loading ONNX model from '{model_path}' with providers {providers}.")
        
        self.model = ort.InferenceSession(model_path, providers=providers)
        self._initialize_slicer()
        self.logger.info("RT-DETR-V2 ONNX model loaded successfully.")

    def _detect_single_slice(self, slice_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            return np.array([]), np.array([])
        try:
            h_orig, w_orig = slice_img.shape[:2]
            
            # Preprocessing: resize to 640x640, convert from BGR to RGB, normalize, transpose to CHW
            resized = cv2.resize(slice_img, (640, 640), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            chw = rgb.transpose(2, 0, 1)
            input_tensor = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0

            orig_target_sizes = np.array([[w_orig, h_orig]], dtype=np.int64)

            outputs = self.model.run(None, {
                "images": input_tensor,
                "orig_target_sizes": orig_target_sizes
            })

            labels, boxes, scores = outputs

            b_boxes, t_boxes = [], []
            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                if score < self.confidence_threshold:
                    continue
                # ONNX output format is [x_min, y_min, x_max, y_max]
                x1, y1, x2, y2 = map(int, box)
                coords = [x1, y1, x2, y2]
                needs_bubble_boxes = (
                    self.detect_bubbles or self.text_region_filter != 'all'
                )
                if label == 0 and needs_bubble_boxes:
                    b_boxes.append(coords)
                elif label in [1, 2] and self.detect_text:
                    t_boxes.append(coords)

            return np.array(b_boxes), np.array(t_boxes)
        except Exception as e:
            self.logger.error(f"Error in _detect_single_slice: {e}", exc_info=True)
            return np.array([]), np.array([])

    def _create_text_blocks_and_mask(
        self, img: np.ndarray, txt_boxes: np.ndarray, bub_boxes: np.ndarray
    ) -> Tuple[List[TextBlock], np.ndarray]:
        txt_boxes = filter_bounding_boxes(txt_boxes)
        bub_boxes = filter_bounding_boxes(bub_boxes)

        blocks = []
        final_inpaint_mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if txt_boxes.size == 0:
            return blocks, final_inpaint_mask

        bub_list = bub_boxes.tolist() if bub_boxes.size > 0 else []

        for tb_rect in txt_boxes:
            x1, y1, x2, y2 = tb_rect
            current_tb_rect_list = tb_rect.tolist()
            best_bubble = None
            for bb in bub_list:
                if does_rectangle_fit(bb, current_tb_rect_list):
                    best_bubble = bb
                    break
                elif do_rectangles_overlap(bb, current_tb_rect_list):
                    if best_bubble is None:
                        best_bubble = bb

            text_class = "text_bubble" if best_bubble else "text_free"
            if self.text_region_filter not in {'all', text_class}:
                continue

            # Build the mask only after filtering so excluded text is not erased.
            local_block_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            inpaint_regions = get_inpaint_bboxes(current_tb_rect_list, img, self.logger)
            for r_x1, r_y1, r_x2, r_y2 in inpaint_regions:
                cv2.rectangle(local_block_mask, (r_x1, r_y1), (r_x2, r_y2), 255, -1)

            dilate_kernel_size = self.inpaint_mask_dilate
            if dilate_kernel_size > 0:
                kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
                local_block_mask = cv2.dilate(local_block_mask, kernel, iterations=1)

            unification_method = self.mask_unification_method
            if unification_method != "none":
                local_block_mask = create_unified_mask(
                    local_block_mask, method=unification_method
                )

            final_inpaint_mask = cv2.bitwise_or(final_inpaint_mask, local_block_mask)

            block_to_add = TextBlock(
                xyxy=current_tb_rect_list,
                lines=[[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
                det_model=self.name,
                label=text_class,
            )
            setattr(block_to_add, 'bubble_xyxy', best_bubble)
            setattr(block_to_add, 'text_class', text_class)
            blocks.append(block_to_add)

        self.logger.debug(
            f"Processed {len(blocks)} text blocks, creating separate unified masks."
        )
        return blocks, final_inpaint_mask

    def _detect(
        self, img: np.ndarray, proj: Optional[ProjImgTrans] = None
    ) -> Tuple[np.ndarray, List[TextBlock]]:
        if not self.all_model_loaded():
            self.logger.warning("RT-DETR-V2 detector not initialized. Loading model...")
            self.load_model()
            if not self.all_model_loaded():
                self.logger.error("Failed to initialize RT-DETR-V2 detector model.")
                return np.zeros(img.shape[:2], dtype=np.uint8), []

        h, w = img.shape[:2]
        self.logger.info(f"Starting RT-DETR-V2 detection on {w}x{h} image...")
        bub_b, txt_b = self.image_slicer.process_slices_for_detection(
            img, self._detect_single_slice
        )

        # Merge near-identical duplicate boxes per category to clean up redundant predictions.
        if self.merge_duplicates:
            iou = self.merge_duplicates_iou
            if bub_b.size > 0:
                bub_b = merge_duplicate_boxes(bub_b, iou)
            if txt_b.size > 0:
                txt_b = merge_duplicate_boxes(txt_b, iou)
            self.logger.debug(
                f"After duplicate merging: {len(bub_b)} bubbles, {len(txt_b)} text boxes."
            )

        # Remove boxes that are nested inside other larger boxes of the same category.
        if self.remove_contained_text:
            thresh = self.containment_threshold
            if bub_b.size > 0:
                bub_b = remove_contained_boxes(bub_b, thresh)
            if txt_b.size > 0:
                txt_b = remove_contained_boxes(txt_b, thresh)
            self.logger.debug(
                f"After nesting cleanup: {len(bub_b)} bubbles, {len(txt_b)} text boxes."
            )

        blk_list, generated_mask = self._create_text_blocks_and_mask(img, txt_b, bub_b)
        self.logger.info(
            f"RT-DETR-V2 detection complete: Found {len(blk_list)} text blocks."
        )
        return generated_mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "device" and self.all_model_loaded():
            self.unload_model()
            self.load_model()
        elif param_key in [
            "confidence_threshold",
            "inpaint_mask_dilate",
            "mask_unification_method",
            "detect_bubbles",
            "detect_text",
            "text_region_filter",
            "merge_duplicates",
            "merge_duplicates_iou",
            "remove_contained_text",
            "containment_threshold",
        ]:
            self.logger.debug(f"RT-DETR-V2 parameter '{param_key}' updated.")
        elif param_key.startswith("slice_") and self.image_slicer:
            self.logger.debug("RT-DETR-V2 slicer parameters changed, re-initializing.")
            self._initialize_slicer()
