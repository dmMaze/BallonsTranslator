import itertools
import functools
from typing import Tuple, List, ClassVar, Union, Any, Dict, Set
from collections import Counter
try:
    functools.cached_property
except AttributeError: # Supports Python versions below 3.8
    from backports.cached_property import cached_property
    functools.cached_property = cached_property

import numpy as np
from shapely.geometry import Polygon, MultiPoint
import cv2
import networkx as nx


class BBox(object):
    def __init__(self, x: int, y: int, w: int, h: int, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b

    def width(self):
        return self.w

    def height(self):
        return self.h

    def to_points(self):
        tl, tr, br, bl = np.array([self.x, self.y]), np.array([self.x + self.w, self.y]), np.array([self.x + self.w, self.y+ self.h]), np.array([self.x, self.y + self.h])
        return tl, tr, br, bl

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int32)
    

class Quadrilateral(object):
    """
    Helper for storing textlines that contains various helper functions.
    """
    def __init__(self, pts: np.ndarray, text: str, prob: float, fg_r: int = 0, fg_g: int = 0, fg_b: int = 0, bg_r: int = 0, bg_g: int = 0, bg_b: int = 0):    
        self.pts, is_vertical = sort_pnts(pts)
        if is_vertical:
            self.direction = 'v'
        else:
            self.direction = 'h'
        self.text = text
        self.prob = prob
        self.fg_r = fg_r
        self.fg_g = fg_g
        self.fg_b = fg_b
        self.bg_r = bg_r
        self.bg_g = bg_g
        self.bg_b = bg_b
        self.assigned_direction: str = None
        self.textlines: List[Quadrilateral] = []

    @functools.cached_property
    def structure(self) -> List[np.ndarray]:
        p1 = ((self.pts[0] + self.pts[1]) / 2).astype(int)
        p2 = ((self.pts[2] + self.pts[3]) / 2).astype(int)
        p3 = ((self.pts[1] + self.pts[2]) / 2).astype(int)
        p4 = ((self.pts[3] + self.pts[0]) / 2).astype(int)
        return [p1, p2, p3, p4]

    @functools.cached_property
    def valid(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product) * 180 / np.pi
        return abs(angle - 90) < 10

    @property
    def fg_colors(self):
        return np.array([self.fg_r, self.fg_g, self.fg_b])

    @property
    def bg_colors(self):
        return np.array([self.bg_r, self.bg_g, self.bg_b])

    @functools.cached_property
    def aspect_ratio(self) -> float:
        """hor/ver"""
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return np.linalg.norm(v2) / np.linalg.norm(v1)

    @functools.cached_property
    def font_size(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        return min(np.linalg.norm(v2), np.linalg.norm(v1))

    def width(self) -> int:
        return self.aabb.w

    def height(self) -> int:
        return self.aabb.h

    @functools.cached_property
    def xyxy(self):
        return self.aabb.x, self.aabb.y, self.aabb.x + self.aabb.w, self.aabb.y + self.aabb.h

    def clip(self, width, height):
        self.pts[:, 0] = np.clip(np.round(self.pts[:, 0]), 0, width)
        self.pts[:, 1] = np.clip(np.round(self.pts[:, 1]), 0, height)

    # @functools.cached_property
    # def points(self):
    #     ans = [a.astype(np.float32) for a in self.structure]
    #     return [Point(a[0], a[1]) for a in ans]

    @functools.cached_property
    def aabb(self) -> BBox:
        kq = self.pts
        max_coord = np.max(kq, axis = 0)
        min_coord = np.min(kq, axis = 0)
        return BBox(min_coord[0], min_coord[1], max_coord[0] - min_coord[0], max_coord[1] - min_coord[1], self.text, self.prob, self.fg_r, self.fg_g, self.fg_b, self.bg_r, self.bg_g, self.bg_b)

    def get_transformed_region(self, img, direction, textheight) -> np.ndarray:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v_vec = l1b - l1a
        h_vec = l2b - l2a
        ratio = np.linalg.norm(v_vec) / np.linalg.norm(h_vec)

        src_pts = self.pts.astype(np.int64).copy()
        im_h, im_w = img.shape[:2]

        x1, y1, x2, y2 = src_pts[:, 0].min(), src_pts[:, 1].min(), src_pts[:, 0].max(), src_pts[:, 1].max()
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        # cv2.warpPerspective could overflow if image size is too large, better crop it here
        img_croped = img[y1: y2, x1: x2]

        
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1

        self.assigned_direction = direction
        if direction == 'h':
            h = max(int(textheight), 2)
            w = max(int(round(textheight / ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_croped, M, (w, h))
            return region
        elif direction == 'v':
            w = max(int(textheight), 2)
            h = max(int(round(textheight * ratio)), 2)
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            region = cv2.warpPerspective(img_croped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return region

    @functools.cached_property
    def is_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        if abs(np.dot(unit_vector_1, e1)) < 1e-2 or abs(np.dot(unit_vector_1, e2)) < 1e-2:
            return True
        return False

    @functools.cached_property
    def is_approximate_axis_aligned(self) -> bool:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        v2 = l2b - l2a
        e1 = np.array([0, 1])
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        if abs(np.dot(unit_vector_1, e1)) < 0.05 or abs(np.dot(unit_vector_1, e2)) < 0.05 or abs(np.dot(unit_vector_2, e1)) < 0.05 or abs(np.dot(unit_vector_2, e2)) < 0.05:
            return True
        return False

    @functools.cached_property
    def cosangle(self) -> float:
        [l1a, l1b, l2a, l2b] = [a.astype(np.float32) for a in self.structure]
        v1 = l1b - l1a
        e2 = np.array([1, 0])
        unit_vector_1 = v1 / np.linalg.norm(v1)
        return np.dot(unit_vector_1, e2)

    @functools.cached_property
    def angle(self) -> float:
        return np.fmod(np.arccos(self.cosangle) + np.pi, np.pi)

    @functools.cached_property
    def centroid(self) -> np.ndarray:
        return np.average(self.pts, axis = 0)

    def distance_to_point(self, p: np.ndarray) -> float:
        d = 1.0e20
        for i in range(4):
            d = min(d, distance_point_point(p, self.pts[i]))
            d = min(d, distance_point_lineseg(p, self.pts[i], self.pts[(i + 1) % 4]))
        return d

    @functools.cached_property
    def polygon(self) -> Polygon:
        return MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(self.pts[2]), tuple(self.pts[3])]).convex_hull

    @functools.cached_property
    def area(self) -> float:
        return self.polygon.area

    def poly_distance(self, other) -> float:
        return self.polygon.distance(other.polygon)

    def distance(self, other, rho = 0.5) -> float:
        return self.distance_impl(other, rho)# + 1000 * abs(self.angle - other.angle)

    def distance_impl(self, other, rho = 0.5) -> float:
        # assert self.assigned_direction == other.assigned_direction
        #return gjk_distance(self.points, other.points)
        # b1 = self.aabb
        # b2 = b2.aabb
        # x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
        # x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
        # return rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
        pattern = ''
        if self.assigned_direction == 'h':
            pattern = 'h_left'
        else:
            pattern = 'v_top'
        fs = max(self.font_size, other.font_size)
        if self.assigned_direction == 'h':
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[3]), tuple(other.pts[0]), tuple(other.pts[3])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[1]), tuple(other.pts[2]), tuple(other.pts[1])]).convex_hull
            poly3 = MultiPoint([
                tuple(self.structure[0]),
                tuple(self.structure[1]),
                tuple(other.structure[0]),
                tuple(other.structure[1]),
            ]).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            dist3 = poly3.area / fs
            if dist1 < fs * rho:
                pattern = 'h_left'
            if dist2 < fs * rho and dist2 < dist1:
                pattern = 'h_right'
            if dist3 < fs * rho and dist3 < dist1 and dist3 < dist2:
                pattern = 'h_middle'
            if pattern == 'h_left':
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            elif pattern == 'h_right':
                return dist(self.pts[1][0], self.pts[1][1], other.pts[1][0], other.pts[1][1])
            else:
                return dist(self.structure[0][0], self.structure[0][1], other.structure[0][0], other.structure[0][1])
        else:
            poly1 = MultiPoint([tuple(self.pts[0]), tuple(self.pts[1]), tuple(other.pts[0]), tuple(other.pts[1])]).convex_hull
            poly2 = MultiPoint([tuple(self.pts[2]), tuple(self.pts[3]), tuple(other.pts[2]), tuple(other.pts[3])]).convex_hull
            dist1 = poly1.area / fs
            dist2 = poly2.area / fs
            if dist1 < fs * rho:
                pattern = 'v_top'
            if dist2 < fs * rho and dist2 < dist1:
                pattern = 'v_bottom'
            if pattern == 'v_top':
                return dist(self.pts[0][0], self.pts[0][1], other.pts[0][0], other.pts[0][1])
            else:
                return dist(self.pts[2][0], self.pts[2][1], other.pts[2][0], other.pts[2][1])

    def copy(self, new_pts: np.ndarray):
        return Quadrilateral(new_pts, self.text, self.prob, *self.fg_colors, *self.bg_colors)


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
    vec_norm_sort_ids = np.argsort(pairwise_vec_norm)
    long_side_ids = vec_norm_sort_ids[[8, 10]]
    pairwise_vec_norm_sorted = pairwise_vec_norm[vec_norm_sort_ids]
    long_side_vecs = pairwise_vec[long_side_ids]
    inner_prod = (long_side_vecs[0] * long_side_vecs[1]).sum()
    if inner_prod < 0:
        long_side_vecs[0] = -long_side_vecs[0]
    struc_vec = np.abs(long_side_vecs.mean(axis=0))
    is_vertical = struc_vec[0] * 1.2 <= struc_vec[1]
    if len(set(pairwise_vec_norm_sorted[4: 12])) == 1:  # is square
        is_vertical = False

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


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def distance_point_point(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


# from https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
def distance_point_lineseg(p: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    x = p[0]
    y = p[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = -1
    if len_sq != 0:
        param = dot / len_sq

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return np.sqrt(dx * dx + dy * dy)


def quadrilateral_can_merge_region(a: Quadrilateral, b: Quadrilateral, ratio = 1.9, discard_connection_gap = 2, char_gap_tolerance = 0.6, char_gap_tolerance2 = 1.5, font_size_ratio_tol = 1.5, aspect_ratio_tol = 2) -> bool:
    b1 = a.aabb
    b2 = b.aabb
    char_size = min(a.font_size, b.font_size)
    x1, y1, w1, h1 = b1.x, b1.y, b1.w, b1.h
    x2, y2, w2, h2 = b2.x, b2.y, b2.w, b2.h
    # dist = rect_distance(x1, y1, x1 + w1, y1 + h1, x2, y2, x2 + w2, y2 + h2)
    p1 = Polygon(a.pts)
    p2 = Polygon(b.pts)
    dist = p1.distance(p2)
    if dist > discard_connection_gap * char_size:
        return False
    if max(a.font_size, b.font_size) / char_size > font_size_ratio_tol:
        return False
    if a.aspect_ratio > aspect_ratio_tol and b.aspect_ratio < 1. / aspect_ratio_tol:
        return False
    if b.aspect_ratio > aspect_ratio_tol and a.aspect_ratio < 1. / aspect_ratio_tol:
        return False
    a_aa = a.is_approximate_axis_aligned
    b_aa = b.is_approximate_axis_aligned
    if a_aa and b_aa:
        if dist < char_size * char_gap_tolerance:
            if abs(x1 + w1 // 2 - (x2 + w2 // 2)) < char_gap_tolerance2:
                return True
            if w1 > h1 * ratio and h2 > w2 * ratio:
                return False
            if w2 > h2 * ratio and h1 > w1 * ratio:
                return False
            if w1 > h1 * ratio or w2 > h2 * ratio : # h
                return abs(x1 - x2) < char_size * char_gap_tolerance2 or abs(x1 + w1 - (x2 + w2)) < char_size * char_gap_tolerance2
            elif h1 > w1 * ratio or h2 > w2 * ratio : # v
                return abs(y1 - y2) < char_size * char_gap_tolerance2 or abs(y1 + h1 - (y2 + h2)) < char_size * char_gap_tolerance2
            return False
        else:
            return False
    if True:#not a_aa and not b_aa:
        if abs(a.angle - b.angle) < 15 * np.pi / 180:
            fs_a = a.font_size
            fs_b = b.font_size
            fs = min(fs_a, fs_b)
            if a.poly_distance(b) > fs * char_gap_tolerance2:
                return False
            if abs(fs_a - fs_b) / fs > 0.25:
                return False
            return True
    return False


def quadrilateral_can_merge_region_coarse(a: Quadrilateral, b: Quadrilateral, discard_connection_gap = 2, font_size_ratio_tol = 0.7) -> bool:
    if a.assigned_direction != b.assigned_direction:
        return False
    if abs(a.angle - b.angle) > 15 * np.pi / 180:
        return False
    fs_a = a.font_size
    fs_b = b.font_size
    fs = min(fs_a, fs_b)
    if abs(fs_a - fs_b) / fs > font_size_ratio_tol:
        return False
    fs = max(fs_a, fs_b)
    dist = a.poly_distance(b)
    if dist > discard_connection_gap * fs:
        return False
    return True


def split_text_region(
        bboxes: List[Quadrilateral],
        connected_region_indices: Set[int],
        width,
        height,
        gamma = 0.5,
        sigma = 2
    ) -> List[Set[int]]:

    connected_region_indices = list(connected_region_indices)

    # case 1
    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]

    # case 2
    if len(connected_region_indices) == 2:
        fs1 = bboxes[connected_region_indices[0]].font_size
        fs2 = bboxes[connected_region_indices[1]].font_size
        fs = max(fs1, fs2)

        # print(bboxes[connected_region_indices[0]].pts, bboxes[connected_region_indices[1]].pts)
        # print(fs, bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]), (1 + gamma) * fs)
        # print(bboxes[connected_region_indices[0]].angle, bboxes[connected_region_indices[1]].angle, 4 * np.pi / 180)

        if bboxes[connected_region_indices[0]].distance(bboxes[connected_region_indices[1]]) < (1 + gamma) * fs \
                and abs(bboxes[connected_region_indices[0]].angle - bboxes[connected_region_indices[1]].angle) < 0.2 * np.pi:
            return [set(connected_region_indices)]
        else:
            return [set([connected_region_indices[0]]), set([connected_region_indices[1]])]

    # case 3
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for (u, v) in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))
    # Get distances from neighbouring bboxes
    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm='kruskal', data=True)
    edges = sorted(edges, key=lambda a: a[2]['weight'], reverse=True)
    distances_sorted = [a[2]['weight'] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = Polygon(b1.pts).distance(Polygon(b2.pts))
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    # print(edges)
    # print(f'std: {distances_std} < thrshold: {std_threshold}, mean: {distances_mean}')
    # print(f'{distances_sorted[0]} <= {distances_mean + distances_std * sigma}' \
    #         f' or {distances_sorted[0]} <= {fontsize * (1 + gamma)}' \
    #         f' or {distances_sorted[0] - distances_sorted[1]} < {distances_std * sigma}')

    if (distances_sorted[0] <= distances_mean + distances_std * sigma \
            or distances_sorted[0] <= fontsize * (1 + gamma)) \
            and (distances_std < std_threshold \
            or max_poly_distance == 0 and max_centroid_alignment < 5):
        return [set(connected_region_indices)]
    else:
        # (split_u, split_v, _) = edges[0]
        # print(f'split between "{bboxes[split_u].pts}", "{bboxes[split_v].pts}"')
        G = nx.Graph()
        for idx in connected_region_indices:
            G.add_node(idx)
        # Split out the most deviating bbox
        for edge in edges[1:]:
            G.add_edge(edge[0], edge[1])
        ans = []
        for node_set in nx.algorithms.components.connected_components(G):
            ans.extend(split_text_region(bboxes, node_set, width, height))
        return ans



def merge_bboxes_text_region(bboxes: List[Quadrilateral], width, height):

    # step 1: divide into multiple text region candidates
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)

    for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
        # if quadrilateral_can_merge_region_coarse(ubox, vbox):
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3, font_size_ratio_tol=2,
                                          char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    # step 2: postprocess - further split each region
    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
         region_indices.extend(split_text_region(bboxes, node_set, width, height))

    # step 3: return regions
    for node_set in region_indices:
    # for node_set in nx.algorithms.components.connected_components(G):
        nodes = list(node_set)
        txtlns: List[Quadrilateral] = np.array(bboxes)[nodes]

        # calculate average fg and bg color
        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))

        # majority vote for direction
        dirs = [box.direction for box in txtlns]
        majority_dir_top_2 = Counter(dirs).most_common(2)
        if len(majority_dir_top_2) == 1 :
            majority_dir = majority_dir_top_2[0][0]
        elif majority_dir_top_2[0][1] == majority_dir_top_2[1][1] : # if top 2 have the same counts
            max_aspect_ratio = -100
            for box in txtlns :
                if box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_aspect_ratio :
                    max_aspect_ratio = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else :
            majority_dir = majority_dir_top_2[0][0]

        # sort textlines
        if majority_dir == 'h':
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == 'v':
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]

        # yield overall bbox and sorted indices
        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)


