"""Compact, editable closed cubic paths for detected text flow areas."""

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Mapping, Optional, Tuple

import cv2
import numpy as np

from .logger import logger as LOGGER


@dataclass(frozen=True)
class BubbleShape:
    """Normalized anchors with independent incoming/outgoing Bezier handles.

    Each node is (x, y, in_x, in_y, out_x, out_y); the last joins the first.
    Collapsed handles represent corners. ``frame`` is an unrotated page xywh;
    TextBlock.angle supplies rotation about its center. Layout scales nodes to
    the current logical rectangle, so moving/resizing never retraces the mask.

    >>> shape = BubbleShape((0, 0, 100, 80), ((0, 0, 0, 0, 0, 0),
    ...     (1, 0, 1, 0, 1, 0), (1, 1, 1, 1, 1, 1), (0, 1, 0, 1, 0, 1)))
    >>> shape.translated(10, 20).frame
    (10.0, 20.0, 100.0, 80.0)
    """

    frame: Tuple[float, float, float, float]
    nodes: Tuple[Tuple[float, ...], ...]

    def __post_init__(self) -> None:
        def numbers(values: object, length: int) -> tuple:
            if not isinstance(values, (list, tuple)) or len(values) != length:
                raise ValueError('invalid bubble path coordinates')
            if any(isinstance(v, bool) or not isinstance(v, Real)
                   or not math.isfinite(v) for v in values):
                raise ValueError('bubble path coordinates must be finite numbers')
            return tuple(float(v) for v in values)

        frame = numbers(self.frame, 4)
        if min(frame[2:]) <= 0 or max(abs(v) for v in frame) > 1e7:
            raise ValueError('invalid bubble frame')
        if not isinstance(self.nodes, (list, tuple)) or not 3 <= len(self.nodes) <= 64:
            raise ValueError('a bubble path needs 3 to 64 nodes')
        nodes = tuple(numbers(node, 6) for node in self.nodes)
        if any(abs(v) > 4 for node in nodes for v in node):
            raise ValueError('bubble handles exceed the normalized frame')
        object.__setattr__(self, 'frame', frame)
        object.__setattr__(self, 'nodes', nodes)

    def translated(self, dx: float, dy: float) -> 'BubbleShape':
        x, y, w, h = self.frame
        return replace(self, frame=(x + dx, y + dy, w, h))

    @classmethod
    def rectangle(cls, frame: Tuple[float, float, float, float]) -> 'BubbleShape':
        return cls(frame, tuple((x, y) * 3 for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))))

    def split_segment(self, index: int, t: float) -> 'BubbleShape':
        """Insert an anchor without changing the curve, including its closing segment.

        >>> shape = BubbleShape((0, 0, 10, 10),
        ...     ((0, 0, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0, 1, 0, 1)))
        >>> shape.split_segment(0, .5).nodes[1][:2]
        (0.5, 0.0)
        """
        if len(self.nodes) == 64 or not 0 <= index < len(self.nodes) or not 0 < t < 1:
            raise ValueError('invalid bubble segment split')
        nodes = [list(node) for node in self.nodes]
        following = (index + 1) % len(nodes)
        a, b = nodes[index], nodes[following]
        points = np.array((a[:2], a[4:], b[2:4], b[:2]))
        first = points[:-1] * (1 - t) + points[1:] * t
        second = first[:-1] * (1 - t) + first[1:] * t
        anchor = second[0] * (1 - t) + second[1] * t
        a[4:], b[2:4] = first[0], first[2]
        nodes.insert(index + 1, [*anchor, *second[0], *second[1]])
        return replace(self, nodes=tuple(tuple(node) for node in nodes))

    def to_serializable_dict(self) -> dict:
        return {'frame': self.frame, 'nodes': self.nodes}


def load_bubble_shape(value: object) -> Optional[BubbleShape]:
    if value is None or isinstance(value, BubbleShape):
        return value
    try:
        if not isinstance(value, Mapping):
            raise ValueError('expected a bubble path object')
        unknown = set(value) - {'frame', 'nodes'}
        if unknown:
            LOGGER.warning('Ignoring unknown bubble shape fields: %s', sorted(unknown))
        return BubbleShape(value['frame'], value['nodes'])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        LOGGER.warning('Ignoring invalid bubble shape: %s', error)
        return None


def _neck_cells(polygon: np.ndarray, anchors: np.ndarray, indices: list[int],
                tolerance: float) -> Optional[dict[int, np.ndarray]]:
    """Split at recessed corners only when every source anchor gets a lobe."""
    if len(indices) == 1:
        return {indices[0]: polygon}
    if not 4 <= len(polygon) <= 128:
        return None
    orientation = np.sign(cv2.contourArea(polygon, oriented=True))
    hull = cv2.convexHull(polygon)
    incoming = polygon - np.roll(polygon, 1, axis=0)
    outgoing = np.roll(polygon, -1, axis=0) - polygon
    turns = (incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]) * orientation
    reflex = np.flatnonzero(turns < -tolerance * tolerance * .05)
    support = {
        int(i): -turns[i] / max(np.linalg.norm(incoming[i]) * np.linalg.norm(outgoing[i]), 1e-6)
        * max(0., cv2.pointPolygonTest(hull, tuple(map(float, polygon[i])), True))
        for i in reflex
    }
    candidates = []
    for offset, first in enumerate(reflex):
        for second in reflex[offset + 1:]:
            length = np.linalg.norm(polygon[second] - polygon[first])
            if length > tolerance:
                candidates.append((min(support[first], support[second]) / length, length, first, second))
    # Rank before expensive validity checks; the first complete split is enough.
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    edge_indices = np.arange(len(polygon))
    next_indices = (edge_indices + 1) % len(polygon)
    for _, _, first, second in candidates:
        start, end = polygon[first], polygon[second]
        direction = end - start
        # A cut may touch its endpoints, but must not cross any other edge.
        keep = ((edge_indices != first) & (edge_indices != second)
                & (next_indices != first) & (next_indices != second))
        edge_start = polygon[keep]
        edge = outgoing[keep]
        delta = edge_start - start
        denominator = direction[0] * edge[:, 1] - direction[1] * edge[:, 0]
        nonparallel = np.abs(denominator) > 1e-6
        along_cut = (delta[:, 0] * edge[:, 1] - delta[:, 1] * edge[:, 0])[nonparallel] / denominator[nonparallel]
        along_edge = (delta[:, 0] * direction[1] - delta[:, 1] * direction[0])[nonparallel] / denominator[nonparallel]
        if np.any((along_cut >= 0) & (along_cut <= 1) & (along_edge >= 0) & (along_edge <= 1)):
            continue
        if any(cv2.pointPolygonTest(polygon, tuple(map(float, start + direction * t)), False) <= 0
               for t in (.2, .5, .8)):
            continue
        left = polygon[first:second + 1].copy()
        right = np.concatenate((polygon[second:], polygon[:first + 1]))
        left_indices, right_indices = [], []
        for index in indices:
            point = tuple(map(float, anchors[index]))
            in_left = cv2.pointPolygonTest(left, point, False) >= 0
            in_right = cv2.pointPolygonTest(right, point, False) >= 0
            if in_left == in_right:
                break
            (left_indices if in_left else right_indices).append(index)
        if not left_indices or not right_indices or len(left_indices) + len(right_indices) != len(indices):
            continue
        first_cells = _neck_cells(left, anchors, left_indices, tolerance)
        if first_cells is not None:
            second_cells = _neck_cells(right, anchors, right_indices, tolerance)
            if second_cells is not None:
                return first_cells | second_cells
    return None


def partition_bubble(mask: np.ndarray, page_size: Tuple[int, int], anchors: np.ndarray) -> list[np.ndarray]:
    """Prefer physical necks; use source bisectors when a lobe split is absent.

    All raster work stays at detector resolution. The page metric matters for
    non-square resize scales, and coincident anchors receive separate strips.

    >>> mask = np.ones((20, 40), np.uint8)
    >>> cells = partition_bubble(mask, (40, 20), np.array([[10, 10], [30, 10]]))
    >>> bool(np.any(cells[0] & cells[1]))
    False
    """
    if len(anchors) <= 1:
        return [mask.astype(bool)] if len(anchors) else []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [np.zeros_like(mask, dtype=bool) for _ in anchors]
    scale = np.array(page_size) / np.array(mask.shape[::-1])
    contour = ((max(contours, key=cv2.contourArea).reshape(-1, 2) + .5) * scale).astype(np.float32)
    tolerance = max(1., float(np.ptp(contour, axis=0).min())) * .0075
    polygon = cv2.approxPolyDP(contour, tolerance, True).reshape(-1, 2)
    cells = _neck_cells(polygon, anchors, list(range(len(anchors))), tolerance)
    owners = np.full(mask.shape, -1, np.int32)
    if cells is not None:
        for index, cell in cells.items():
            cv2.fillPoly(owners, [np.round(cell / scale - .5).astype(np.int32)], index)
    else:
        y, x = np.nonzero(mask)
        points = np.column_stack((x + .5, y + .5)) * scale
        groups: list[list[int]] = []
        sites = []
        threshold = max(1., float(np.ptp(contour, axis=0).min())) * .0025
        for index, anchor in enumerate(anchors):
            for group, site in zip(groups, sites):
                if np.linalg.norm(anchor - site) <= threshold:
                    group.append(index)
                    site[:] = anchors[group].mean(axis=0)
                    break
            else:
                groups.append([index])
                sites.append(np.array(anchor, dtype=float))
        nearest = ((points[:, None] - sites) ** 2).sum(axis=2).argmin(axis=1)
        for number, group in enumerate(groups):
            owned = nearest == number
            if not owned.any():
                continue
            if len(group) == 1:
                owners[y[owned], x[owned]] = group[0]
            else:
                span = points[owned]
                axis = int(np.argmax(np.ptp(span, axis=0)))
                low, high = span[:, axis].min(), span[:, axis].max()
                strips = np.minimum(((span[:, axis] - low) / max(high - low, 1.) * len(group)).astype(int), len(group) - 1)
                owners[y[owned], x[owned]] = np.array(group)[strips]
    owners[mask == 0] = -1
    return [owners == index for index in range(len(anchors))]


def trace_bubble(mask: np.ndarray, page_size: Tuple[int, int], angle: float = 0,
                 source_center: Optional[Tuple[float, float]] = None) -> Optional[BubbleShape]:
    """Trace at detector resolution, preferring four-node rectangles/ellipses.

    The fallback is a simplified closed polygon with collapsed cubic handles;
    it preserves concavities without curve-fitting overshoot into artwork.

    >>> mask = np.zeros((40, 60), np.uint8)
    >>> mask[5:35, 10:50] = 1
    >>> len(trace_bubble(mask, (600, 400)).nodes)
    4
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    if source_center is None or len(contours) == 1:
        contour = max(contours, key=cv2.contourArea)
    else:
        # Segmentation can have disconnected lobes; keep the one owning this text.
        point = np.asarray(source_center) * np.array(mask.shape[::-1]) / np.array(page_size) - .5
        contour = max(contours, key=lambda candidate: cv2.pointPolygonTest(candidate, tuple(point), True))
    area = cv2.contourArea(contour)
    if area < 4:
        return None
    rectangle = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rectangle)
    page_limit = np.array(mask.shape[::-1]) - .5
    if (area / max(np.prod(rectangle[1]), 1) >= .94
            and np.all(corners >= -.5) and np.all(corners <= page_limit)):
        anchors = corners
        nodes = np.tile(anchors, (1, 3))
    else:
        nodes = None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            # Compare only the native-resolution crop, never a page-sized mask.
            x, y, w, h = cv2.boundingRect(contour)
            candidate = np.zeros((h, w), np.uint8)
            center, axes, rotation = ellipse
            rad = math.radians(rotation)
            a, b = np.array(axes) / 2
            extent = np.array([math.hypot(a * math.cos(rad), b * math.sin(rad)),
                               math.hypot(a * math.sin(rad), b * math.cos(rad))])
            # A cropped ellipse must stay cropped; extrapolating its missing
            # side would let translated text run off the page.
            bounded = (np.all(np.array(center) - extent >= np.maximum([x - 1, y - 1], -.5))
                       and np.all(np.array(center) + extent <= np.minimum([x + w, y + h], page_limit)))
            cv2.ellipse(candidate, ((center[0] - x, center[1] - y), axes, rotation), 1, -1)
            original = mask[y:y+h, x:x+w] != 0
            overlap = np.count_nonzero(original & (candidate != 0))
            union = np.count_nonzero(original | (candidate != 0))
            if bounded and overlap / max(union, 1) >= .93:
                k = .5522847498307936
                nodes = np.array([[1, 0, 1, -k, 1, k], [0, 1, k, 1, -k, 1],
                                  [-1, 0, -1, k, -1, -k], [0, -1, -k, -1, k, -1]], float)
                points = nodes.reshape(-1, 2) * (np.array(axes) / 2)
                points = points @ np.array([[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]])
                nodes = (points + center).reshape(-1, 6)
        if nodes is None:
            epsilon = max(.75, cv2.arcLength(contour, True) * .005)
            anchors = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            while len(anchors) > 64:
                epsilon *= 1.5
                anchors = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(anchors) < 3:
                return None
            nodes = np.tile(anchors, (1, 3))

    points = nodes.reshape(-1, 2).astype(float)
    points = (points + .5) * (np.array(page_size) / np.array(mask.shape[::-1]))
    rad = math.radians(angle)
    rotation = np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]])
    points = points @ rotation  # deskew before choosing the text frame
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    if min(extent) <= 0:
        return None
    center = ((low + high) / 2) @ rotation.T
    frame = tuple(np.round(np.r_[center - extent / 2, extent], 3))
    nodes = np.round(((points - low) / extent).reshape(-1, 6), 5)
    return BubbleShape(frame, tuple(tuple(node) for node in nodes))
