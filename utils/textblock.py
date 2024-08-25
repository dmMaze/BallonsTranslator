from typing import List, Tuple, Callable
import numpy as np
from shapely.geometry import Polygon
import math
import copy
import cv2
import re

from .imgproc_utils import union_area, xywh2xyxypoly, rotate_polygons, color_difference
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass
from .split_text_region import split_textblock as split_text_region

LANG_LIST = ['eng', 'ja', 'unknown']
LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

# https://ayaka.shn.hk/hanregex/
# https://medium.com/the-artificial-impostor/detecting-chinese-characters-in-unicode-strings-4ac839ba313a
CJKPATTERN = re.compile(r'[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9FFF]')


def sort_pnts(pts: np.ndarray):
    '''
    Direction must be provided for sorting.
    The longer structure vector (mean of long side vectors) of input points is used to determine the direction.
    It is reliable enough for text lines but not for blocks.
    '''

    if isinstance(pts, List):
        pts = np.array(pts)
    assert isinstance(pts, np.ndarray) and pts.shape == (4, 2)
    pairwise_vec = (pts[:, None] - pts[None]).reshape((16, -1))
    pairwise_vec_norm = np.linalg.norm(pairwise_vec, axis=1)
    long_side_ids = np.argsort(pairwise_vec_norm)[[8, 10]]
    long_side_vecs = pairwise_vec[long_side_ids]
    inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
    if inner_prod < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] <= struc_vec[1]

    if is_vertical:
        pts = pts[np.argsort(pts[:, 1])]
        pts = pts[[*np.argsort(pts[:2, 0]), *np.argsort(pts[2:, 0])[::-1] + 2]]
        return pts, is_vertical
    else:
        pts = pts[np.argsort(pts[:, 0])]
        pts_sorted = np.zeros_like(pts)
        pts_sorted[[0, 3]] = sorted(pts[[0, 1]], key=lambda x: x[1])
        pts_sorted[[1, 2]] = sorted(pts[[2, 3]], key=lambda x: x[1])
        return pts_sorted, is_vertical


@nested_dataclass
class TextBlock:
    xyxy: List = field(default_factory = lambda: [0, 0, 0, 0])
    lines: List = field(default_factory = lambda: [])
    language: str = 'unknown'
    vertical: bool = False
    font_size: float = -1.
    distance: np.ndarray = None
    angle: int = 0
    vec: List = None
    norm: float = -1
    merged: bool = False
    sort_weight: float = -1
    text: List = field(default_factory = lambda : [])
    translation: str = ""
    line_spacing: float = 1.
    letter_spacing: float = 1.
    font_family: str = ""
    bold: bool = False
    underline: bool = False
    italic: bool = False
    _alignment: int = -1
    rich_text: str = ""
    _bounding_rect: List = None
    default_stroke_width: float = 0.2
    stroke_decide_by_colordiff: bool = True
    font_weight: int = None
    opacity: float = 1.
    shadow_radius: float = 0.
    shadow_strength: float = 1.
    shadow_color: Tuple = (0, 0, 0)
    shadow_offset: List = field(default_factory = lambda : [0., 0.])
    src_is_vertical: bool = None
    _detected_font_size: float = -1

    region_mask: np.ndarray = None
    region_inpaint_dict: Dict = None

    fg_colors: np.ndarray = field(default_factory = lambda : np.array([0., 0., 0.], dtype=np.float32))
    bg_colors: np.ndarray = field(default_factory = lambda : np.array([0., 0., 0.], dtype=np.float32))

    deprecated_attributes: dict = field(default_factory = lambda: dict())

    def __post_init__(self):
        if self.xyxy is not None:
            self.xyxy = [int(num) for num in self.xyxy]
        if self.distance is not None:
            self.distance = np.array(self.distance, np.float32)
        if self.vec is not None:
            self.vec = np.array(self.vec, np.float32)
        if self.src_is_vertical is None:
            self.src_is_vertical = self.vertical

        da = self.deprecated_attributes
        if len(da) > 0:
            if 'accumulate_color' in da:
                self.fg_colors = np.array([da['fg_r'], da['fg_g'], da['fg_b']], dtype=np.float32)
                self.bg_colors = np.array([da['bg_r'], da['bg_g'], da['bg_b']], dtype=np.float32)
                nlines = len(self)
                if da['accumulate_color'] and len(self) > 0:
                    self.fg_colors /= nlines
                    self.bg_colors /= nlines
        del self.deprecated_attributes

    @property
    def detected_font_size(self):
        if self._detected_font_size > 0:
            return self._detected_font_size
        return self.font_size

    def adjust_bbox(self, with_bbox=False, x_range=None, y_range=None):
        lines = self.lines_array().astype(np.int32)
        if with_bbox:
            self.xyxy[0] = min(lines[..., 0].min(), self.xyxy[0])
            self.xyxy[1] = min(lines[..., 1].min(), self.xyxy[1])
            self.xyxy[2] = max(lines[..., 0].max(), self.xyxy[2])
            self.xyxy[3] = max(lines[..., 1].max(), self.xyxy[3])
        else:
            self.xyxy[0] = lines[..., 0].min()
            self.xyxy[1] = lines[..., 1].min()
            self.xyxy[2] = lines[..., 0].max()
            self.xyxy[3] = lines[..., 1].max()

        if x_range is not None:
            self.xyxy[0] = np.clip(self.xyxy[0], x_range[0], x_range[1])
            self.xyxy[2] = np.clip(self.xyxy[2], x_range[0], x_range[1])
        if y_range is not None:
            self.xyxy[1] = np.clip(self.xyxy[1], y_range[0], y_range[1])
            self.xyxy[3] = np.clip(self.xyxy[3], y_range[0], y_range[1])

    def sort_lines(self):
        if self.distance is not None:
            idx = np.argsort(self.distance)
            self.distance = self.distance[idx]
            lines = np.array(self.lines, dtype=np.int32)
            self.lines = lines[idx].tolist()

    def lines_array(self, dtype=np.float64):
        return np.array(self.lines, dtype=dtype)

    def set_lines_by_xywh(self, xywh: np.ndarray, angle=0, x_range=None, y_range=None, adjust_bbox=False):
        if isinstance(xywh, List):
            xywh = np.array(xywh)
        lines = xywh2xyxypoly(np.array([xywh]))
        if angle != 0:
            cx, cy = xywh[0], xywh[1]
            cx += xywh[2] / 2.
            cy += xywh[3] / 2.
            lines = rotate_polygons([cx, cy], lines, angle)

        lines = lines.reshape(-1, 4, 2)
        if x_range is not None:
            lines[..., 0] = np.clip(lines[..., 0], x_range[0], x_range[1])
        if y_range is not None:
            lines[..., 1] = np.clip(lines[..., 1], y_range[0], y_range[1])
        self.lines = lines.tolist()

        if adjust_bbox:
            self.adjust_bbox()

    def aspect_ratio(self) -> float:
        min_rect = self.min_rect()
        middle_pnts = (min_rect[:, [1, 2, 3, 0]] + min_rect) / 2
        norm_v = np.linalg.norm(middle_pnts[:, 2] - middle_pnts[:, 0])
        norm_h = np.linalg.norm(middle_pnts[:, 1] - middle_pnts[:, 3])
        return norm_v / norm_h

    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    def unrotated_polygons(self) -> np.ndarray:
        angled = self.angle != 0
        center = self.center()
        polygons = self.lines_array().reshape(-1, 8)
        if angled:
            polygons = rotate_polygons(center, polygons, self.angle)
        return angled, center, polygons
    
    def min_rect(self, rotate_back=True) -> List[int]:
        angled, center, polygons = self.unrotated_polygons()
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if angled and rotate_back:
            min_bbox = rotate_polygons(center, min_bbox, -self.angle)
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    def normalizd_width_list(self) -> List[float]:
        angled, center, polygons = self.unrotated_polygons()
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).sum())
        width_list = np.array(width_list)
        width_list = width_list / np.sum(width_list)
        return width_list.tolist()

    # equivalent to qt's boundingRect, ignore angle
    def bounding_rect(self) -> List[int]:
        if self._bounding_rect is None:
        # if True:
            min_bbox = self.min_rect(rotate_back=False)[0]
            x, y = min_bbox[0]
            w, h = min_bbox[2] - min_bbox[0]
            return [int(x), int(y), int(w), int(h)]
        return self._bounding_rect

    def __getattribute__(self, name: str):
        if name == 'pts':
            return self.lines_array()
        # else:
        return object.__getattribute__(self, name)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self):
        blk_dict = copy.deepcopy(vars(self))
        return blk_dict

    def get_transformed_region(self, img: np.ndarray, idx: int, textheight: int, maxwidth: int = None) -> np.ndarray :
        
        line = np.round(np.array(self.lines[idx])).astype(np.int64)
        x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].min(), line[:, 0].max(), line[:, 1].max()
        im_h, im_w = img.shape[:2]
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_croped = img[y1: y2, x1: x2]
        
        direction = 'v' if self.src_is_vertical else 'h'

        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        norm_v = np.linalg.norm(vec_v)
        norm_h = np.linalg.norm(vec_h)

        if textheight is None:
            if direction == 'h' :
                textheight = int(norm_v)
            else:
                textheight = int(norm_h)
        
        if norm_v <= 0 or norm_h <= 0:
            print('invalid textpolygon to target img')
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == 'h' :
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
        elif direction == 'v' :
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if maxwidth is not None:
            h, w = region.shape[: 2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))

        return region

    def get_text(self) -> str:
        if isinstance(self.text, str):
            return self.text
        text = ''
        for t in self.text:
            if text and t:
                if text[-1].isalpha() and t[0].isalpha() \
                    and CJKPATTERN.search(text[-1]) is None \
                    and CJKPATTERN.search(t[0]) is None:
                    text += ' '
            text += t

        return text.strip()

    def set_font_colors(self, fg_colors = None, bg_colors = None):
        if fg_colors is not None:
            self.fg_colors = np.array(fg_colors, dtype=np.float32)
        if bg_colors is not None:
            self.bg_colors = np.array(bg_colors, dtype=np.float32)

    def update_font_colors(self, fg_colors: np.ndarray, bg_colors: np.ndarray):
        nlines = len(self)
        if nlines > 0:
            if not isinstance(fg_colors, np.ndarray):
                fg_colors = np.array(fg_colors, dtype=np.float32)
            if not isinstance(bg_colors, np.ndarray):
                bg_colors = np.array(bg_colors, dtype=np.float32)
            self.fg_colors += fg_colors / nlines
            self.bg_colors += bg_colors / nlines

    def get_font_colors(self, bgr=False):

        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)

        if bgr:
            frgb = frgb[::-1]
            brgb = brgb[::-1]

        return frgb, brgb

    def xywh(self):
        x, y, w, h = self.xyxy
        return [x, y, w-x, h-y]

    # alignleft: 0, center: 1, right: 2 
    def alignment(self):
        if self._alignment >= 0:
            return self._alignment
        elif self.src_is_vertical:
            return 1
        lines = self.lines_array()
        if len(lines) == 1:
            return 1
        angled, center, polygons = self.unrotated_polygons()
        polygons = polygons.reshape(-1, 4, 2)
        
        left_std = np.std(polygons[:, 0, 0])
        # right_std = np.std(polygons[:, 1, 0])
        center_std = np.std((polygons[:, 0, 0] + polygons[:, 1, 0]) / 2)
        if left_std < center_std:
            return 0
        else:
            return 1

    def target_lang(self):
        return self.target_lang

    @property
    def stroke_width(self):
        if self.stroke_decide_by_colordiff:
            diff = color_difference(*self.get_font_colors())
            if diff < 15:
                return 0
        return self.default_stroke_width

    def adjust_pos(self, dx: int, dy: int):
        self.xyxy[0] += dx
        self.xyxy[1] += dy
        self.xyxy[2] += dx
        self.xyxy[3] += dy
        if self._bounding_rect is not None:
            self._bounding_rect[0] += dx
            self._bounding_rect[1] += dy

def sort_textblk_list(blk_list: List[TextBlock], im_w: int, im_h: int) -> List[TextBlock]:
    if len(blk_list) == 0:
        return blk_list
    num_ja = 0
    xyxy = []
    for blk in blk_list:
        if blk.language == 'ja':
            num_ja += 1
        xyxy.append(blk.xyxy)
    xyxy = np.array(xyxy)
    flip_lr = num_ja > len(blk_list) / 2
    im_oriw = im_w
    if im_w > im_h:
        im_w /= 2
    num_gridy, num_gridx = 4, 3
    img_area = im_h * im_w
    center_x = (xyxy[:, 0] + xyxy[:, 2]) / 2
    if flip_lr:
        if im_w != im_oriw:
            center_x = im_oriw - center_x
        else:
            center_x = im_w - center_x
    grid_x = (center_x / im_w * num_gridx).astype(np.int32)
    center_y = (xyxy[:, 1] + xyxy[:, 3]) / 2
    grid_y = (center_y / im_h * num_gridy).astype(np.int32)
    grid_indices = grid_y * num_gridx + grid_x
    grid_weights = grid_indices * img_area + 1.2 * (center_x - grid_x * im_w / num_gridx) + (center_y - grid_y * im_h / num_gridy)
    if im_w != im_oriw:
        grid_weights[np.where(grid_x >= num_gridx)] += img_area * num_gridy * num_gridx
    
    for blk, weight in zip(blk_list, grid_weights):
        blk.sort_weight = weight
    blk_list.sort(key=lambda blk: blk.sort_weight)
    return blk_list

def sort_regions(regions: List[TextBlock], right_to_left=None) -> List[TextBlock]:
    # from manga image translator
    # Sort regions from right to left, top to bottom
    
    nr = len(regions)
    if right_to_left is None and nr > 0:
        nv = 0
        for r in regions:
            if r.vertical:
                nv += 1
        right_to_left = nv / nr > 0
    
    sorted_regions = []
    for region in sorted(regions, key=lambda region: region.center()[1]):
        for i, sorted_region in enumerate(sorted_regions):
            if region.center()[1] > sorted_region.xyxy[3]:
                continue
            if region.center()[1] < sorted_region.xyxy[1]:
                sorted_regions.insert(i + 1, region)
                break

            # y center of region inside sorted_region so sort by x instead
            if right_to_left and region.center()[0] > sorted_region.center()[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center()[0] < sorted_region.center()[0]:
                sorted_regions.insert(i, region)
                break
        else:
            sorted_regions.append(region)
    return sorted_regions

def examine_textblk(blk: TextBlock, im_w: int, im_h: int, sort: bool = False) -> None:
    lines = blk.lines_array()
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors of textlines
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors of textlines
    # if sum of vertical vectors is longer, then text orientation is vertical, and vice versa.
    center_pnts = (lines[:, 0] + lines[:, 2]) / 2
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
    if blk.language == 'ja':
        vertical = norm_v > norm_h
    else:
        vertical = norm_v > norm_h * 2
    # calcuate distance between textlines and origin 
    if vertical:
        primary_vec, primary_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
        font_size = int(round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))
    
    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)     # rotation angle of textlines
    distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
    rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
    distance = np.abs(np.sin(rad_matrix) * distance)
    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle
    if vertical:
        blk.angle -= 90
    if abs(blk.angle) < 3:
        blk.angle = 0
    blk.font_size = font_size
    blk.vertical = blk.src_is_vertical = vertical
    blk.vec = primary_vec
    blk.norm = primary_norm
    if sort:
        blk.sort_lines()

def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.7, distance_tol=2) -> bool:
    if blk2.merged:
        return False
    fntsize_div = blk.font_size / blk2.font_size
    num_l1, num_l2 = len(blk), len(blk2)
    fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
    vec_prod = blk.vec @ blk2.vec
    vec_sum = blk.vec + blk2.vec
    cos_vec = vec_prod / blk.norm / blk2.norm
    distance = blk2.distance[-1] - blk.distance[-1]
    distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
    l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[-1])
    if not l1.intersects(l2):
        if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
            return False
        if abs(cos_vec) < 0.866:   # cos30
            return False
        if distance > distance_tol * fntsz_avg:
            return False
        if blk.vertical and blk2.vertical and distance_p1 > fntsz_avg * 2.5:
            return False
    # merge
    blk.lines.append(blk2.lines[0])
    blk.vec = vec_sum
    blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
    if blk.vertical:
        blk.angle -= 90
    blk.norm = np.linalg.norm(vec_sum)
    blk.distance = np.append(blk.distance, blk2.distance[-1])
    blk.font_size = fntsz_avg
    blk2.merged = True
    return True

def merge_textlines(blk_list: List[TextBlock]) -> List[TextBlock]:
    if len(blk_list) < 2:
        return blk_list
    blk_list.sort(key=lambda blk: blk.distance[0])
    merged_list = []
    for ii, current_blk in enumerate(blk_list):
        if current_blk.merged:
            continue
        for jj, blk in enumerate(blk_list[ii+1:]):
            try_merge_textline(current_blk, blk)
        merged_list.append(current_blk)
    for blk in merged_list:
        blk.adjust_bbox(with_bbox=False)
    return merged_list

def split_textblk(blk: TextBlock):
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines
    l0 = np.array(blk.lines[0])
    lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
    distance_tol = font_size * 2
    current_blk = copy.deepcopy(blk)
    current_blk.lines = [l0]
    sub_blk_list = [current_blk]
    textblock_splitted = False
    for jj, line in enumerate(lines[1:]):
        l1, l2 = Polygon(lines[jj]), Polygon(line)
        split = False
        if not l1.intersects(l2):
            line_disance = abs(distance[jj+1] - distance[jj])
            if line_disance > distance_tol:
                split = True
            elif blk.vertical and abs(blk.angle) < 15:
                if len(current_blk.lines) > 1 or line_disance > font_size:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = copy.deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)
    if len(sub_blk_list) > 1:
        textblock_splitted = True
        for current_blk in sub_blk_list:
            current_blk.adjust_bbox(with_bbox=False)
    return textblock_splitted, sub_blk_list

def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True) -> List[TextBlock]:
    blk_list: List[TextBlock] = []
    scattered_lines = {'ver': [], 'hor': []}
    for bbox, cls, conf in zip(*blks):
        # cls could give wrong result
        blk_list.append(TextBlock(bbox, language=LANG_LIST[cls]))

    # step1: filter & assign lines to textblocks
    bbox_score_thresh = 0.4
    mask_score_thresh = 0.1
    for ii, line in enumerate(lines):
        line, is_vertical = sort_pnts(line)
        bx1, bx2 = line[:, 0].min(), line[:, 0].max()
        by1, by2 = line[:, 1].min(), line[:, 1].max()
        bbox_score, bbox_idx = -1, -1
        line_area = (by2-by1) * (bx2-bx1)
        for jj, blk in enumerate(blk_list):
            score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
            if bbox_score < score:
                bbox_score = score
                bbox_idx = jj
        if bbox_score > bbox_score_thresh:
            blk_list[bbox_idx].lines.append(line)
        else:   # if no textblock was assigned, check whether there is "enough" textmask
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            blk = TextBlock([bx1, by1, bx2, by2], [line])
            examine_textblk(blk, im_w, im_h, sort=False)
            if blk.vertical:
                scattered_lines['ver'].append(blk)
            else:
                scattered_lines['hor'].append(blk)

    # step2: filter textblocks, sort & split textlines
    final_blk_list = []
    for blk in blk_list:
        # filter textblocks 
        if len(blk.lines) == 0:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        examine_textblk(blk, im_w, im_h, sort=True)
        
        # split manga text if there is a distance gap
        textblock_splitted = False
        if len(blk.lines) > 1:
            if blk.language == 'ja':
                textblock_splitted = True
            elif blk.vertical:
                textblock_splitted = True
        if textblock_splitted:
            textblock_splitted, sub_blk_list = split_textblk(blk)
        else:
            sub_blk_list = [blk]
        # modify textblock to fit its textlines
        if not textblock_splitted:
            for blk in sub_blk_list:
                blk.adjust_bbox(with_bbox=True)
        final_blk_list += sub_blk_list

    # step3: merge scattered lines, sort textblocks by "grid"
    final_blk_list += merge_textlines(scattered_lines['hor'])
    final_blk_list += merge_textlines(scattered_lines['ver'])
    if sort_blklist:
        # final_blk_list = sort_textblk_list(final_blk_list, im_w, im_h)
        final_blk_list = sort_regions(final_blk_list, )

    if len(final_blk_list) > 1:
        _final_blks = [final_blk_list[0]]
        for blk in final_blk_list[1:]:
            ax1, ay1, ax2, ay2 = blk.xyxy
            keep_blk = True
            aarea = (ax2 - ax1) * (ay2 - ay1) + 1e-6
            for eb in _final_blks:
                bx1, by1, bx2, by2 = eb.xyxy
                x1 = max(ax1, bx1)
                y1 = max(ay1, by1)
                x2 = min(ax2, bx2)
                y2 = min(ay2, by2)
                if y2 < y1 or x2 < x1:
                    continue
                inter_area = (y2 - y1) * (x2 - x1)
                if inter_area / aarea > 0.9:
                    keep_blk = False
                    break
            if keep_blk:
                _final_blks.append(blk)
        final_blk_list = _final_blks

    for blk in final_blk_list:
        if blk.language != 'ja' and not blk.vertical:
            num_lines = len(blk.lines)
            if num_lines == 0:
                continue
            # blk.line_spacing = blk.bounding_rect()[3] / num_lines / blk.font_size
            expand_size = max(int(blk.font_size * 0.1), 3)
            rad = np.deg2rad(blk.angle)
            shifted_vec = np.array([[[-1, -1],[1, -1],[1, 1],[-1, 1]]])
            shifted_vec = shifted_vec * np.array([[[np.sin(rad), np.cos(rad)]]]) * expand_size
            lines = blk.lines_array() + shifted_vec
            lines[..., 0] = np.clip(lines[..., 0], 0, im_w-1)
            lines[..., 1] = np.clip(lines[..., 1], 0, im_h-1)
            blk.lines = lines.astype(np.int64).tolist()
            blk.font_size += expand_size
        blk._detected_font_size = blk.font_size
            
    return final_blk_list

def visualize_textblocks(canvas, blk_list:  List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for ii, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        lines = blk.lines_array(dtype=np.int32)
        for jj, line in enumerate(lines):
            cv2.putText(canvas, str(jj), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        cv2.polylines(canvas, [blk.min_rect()], True, (127,127,0), 2)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, str(blk.angle), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, str(ii), (bx1, by1 + lw + 2), 0, lw / 3, (255,127,127), max(lw-1, 1), cv2.LINE_AA)
    return canvas

def collect_textblock_regions(img: np.ndarray, textblk_lst: List[TextBlock], text_height=48, maxwidth=8100, split_textblk = False, seg_func: Callable = None):
    regions = []
    textblk_lst_indices = []
    for blk_idx, textblk in enumerate(textblk_lst):
        for ii in range(len(textblk)):
            if split_textblk and len(textblk) == 1:
                assert seg_func is not None
                region = textblk.get_transformed_region(img, ii, None, maxwidth=None)
                mask  = seg_func(region)[0]
                split_lines = split_text_region(mask)[0]
                for jj, line in enumerate(split_lines):
                    bottom = line[3]
                    if len(split_lines) == 1:
                        bottom = region.shape[0]
                    r = region[line[1]: bottom]
                    h, w = r.shape[:2]
                    tgt_h, tgt_w = text_height, min(maxwidth, int(text_height / h * w))
                    if tgt_h != h or tgt_w != w:
                        r = cv2.resize(r, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
                    regions.append(r)
                    textblk_lst_indices.append(blk_idx)
                #     cv2.imwrite(f'local_region{jj}.jpg', r)
                # cv2.imwrite('local_mask.jpg', mask)
                # cv2.imwrite('local_region.jpg',region)
            else:
                textblk_lst_indices.append(blk_idx)
                region = textblk.get_transformed_region(img, ii, text_height, maxwidth=maxwidth)
                regions.append(region)

    return regions, textblk_lst_indices