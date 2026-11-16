"""Compact, editable closed cubic paths for detected text flow areas."""

from dataclasses import dataclass, replace
import math
from numbers import Real
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from shapely.geometry import LinearRing
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath, QPainterPathStroker, QPolygonF, QTransform

from .logger import logger as LOGGER


def bubble_path(nodes: Sequence[Sequence[float]], rect: Optional[QRectF] = None) -> QPainterPath:
    """Build the shared cubic path for tracing, wrapping and anchor editing."""
    path = QPainterPath()
    path.moveTo(*nodes[0][:2])
    for previous, node in zip(nodes, (*nodes[1:], nodes[0])):
        path.cubicTo(*previous[4:], *node[2:4], *node[:2])
    path.closeSubpath()
    if rect is not None:
        transform = QTransform()
        transform.translate(rect.left(), rect.top())
        transform.scale(rect.width(), rect.height())
        path = transform.map(path)
    return path


def _sample_bubble_nodes(nodes: np.ndarray) -> np.ndarray:
    """Sample each cubic at 32 intervals for coverage and curve validity checks."""
    t = np.linspace(0, 1, 33)[:, None]
    basis = np.concatenate(((1 - t) ** 3, 3 * (1 - t) ** 2 * t,
                            3 * (1 - t) * t * t, t ** 3), axis=1)
    controls = np.stack((nodes[:, :2], nodes[:, 4:],
                         np.roll(nodes[:, 2:4], -1, axis=0),
                         np.roll(nodes[:, :2], -1, axis=0)), axis=1)
    return basis @ controls


def _split_bubble_nodes(nodes: np.ndarray, index: int, t: float) -> np.ndarray:
    """Split a cubic in a writable node array, preserving its exact geometry."""
    following = (index + 1) % len(nodes)
    a, b = nodes[index], nodes[following]
    points = np.array((a[:2], a[4:], b[2:4], b[:2]))
    first = points[:-1] * (1 - t) + points[1:] * t
    second = first[:-1] * (1 - t) + first[1:] * t
    anchor = second[0] * (1 - t) + second[1] * t
    a[4:], b[2:4] = first[0], first[2]
    return np.insert(nodes, index + 1, np.r_[anchor, second[0], second[1]], axis=0)


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
        nodes = _split_bubble_nodes(np.array(self.nodes), index, t)
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


def _fit_bubble_curve(contour: np.ndarray, tolerance: float) -> Optional[np.ndarray]:
    """Fit and merge cubic arcs within a native-mask error budget.

    >>> contour = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32)
    >>> 3 <= len(_fit_bubble_curve(contour, 1.)) <= 12
    True
    """
    points = contour.reshape(-1, 2).astype(float)
    low, high = points.min(axis=0), points.max(axis=0)
    points = np.concatenate((points, points[:1]))
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    # Equal arc-length samples avoid weighting pixel staircases more heavily.
    distance = np.linspace(0, lengths[-1], min(256, max(24, len(points))), endpoint=False)
    points = np.column_stack([np.interp(distance, lengths, points[:, axis]) for axis in (0, 1)])
    points = np.concatenate((points, points[:1]))

    def fit(span: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, int]:
        t = np.linspace(0, 1, len(span))[:, None]
        start, end = span[0], span[-1]
        first, second = 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2
        residual = span - (1 - t) ** 3 * start - t ** 3 * end
        handles = np.linalg.lstsq(np.column_stack((first, second)), residual, rcond=None)[0]
        # Keep handles inside the body bounds, including clipped page edges.
        handles = np.clip(handles, low, high)
        curve = (1 - t) ** 3 * start + first * handles[0] + second * handles[1] + t ** 3 * end
        errors = np.linalg.norm(curve - span, axis=1)
        split = int(errors.argmax())
        return span, np.array([start, *handles, end]), float(errors[split]), split

    cuts = np.linspace(0, len(points) - 1, 5).astype(int)
    segments = [fit(points[start:end + 1]) for start, end in zip(cuts, cuts[1:])]
    # Leave room to split around the four initial boundaries before merging them.
    while len(segments) < 16:
        worst = max(range(len(segments)), key=lambda index: segments[index][2])
        span, _, error, split = segments[worst]
        if error <= tolerance:
            break
        split = min(max(split, 2), len(span) - 3)
        if split <= 0 or split >= len(span) - 1:
            break
        segments[worst:worst + 1] = [fit(span[:split + 1]), fit(span[split:])]
    if any(segment[2] > tolerance for segment in segments):
        return None
    # Initial quadrant boundaries are arbitrary; merge across the closing join too.
    while len(segments) > 3:
        for index, segment in enumerate(segments):
            following = (index + 1) % len(segments)
            merged = fit(np.concatenate((segment[0], segments[following][0][1:])))
            if merged[2] <= tolerance:
                if following == 0:
                    segments[0] = merged
                    segments.pop()
                else:
                    segments[index:following + 1] = [merged]
                break
        else:
            break
    if len(segments) > 12:
        return None
    nodes = np.array([np.r_[segment[1][0], segments[index - 1][1][2], segment[1][1]]
                      for index, segment in enumerate(segments)])
    outline = _sample_bubble_nodes(nodes)[:, :-1].reshape(-1, 2)
    return nodes if LinearRing(outline).is_simple else None


def _text_mask_path(mask: np.ndarray) -> QPainterPath:
    """Outline full native pixel cells, including thin strokes and isolated pixels.

    >>> path = _text_mask_path(np.array([[1, 0], [0, 1]], np.uint8))
    >>> path.contains(QPointF(.4, .4)), path.contains(QPointF(1., 0.))
    (True, False)
    """
    path = QPainterPath()
    x, y, width, height = cv2.boundingRect(mask)
    if not width or not height:
        return path
    # Contouring pixel centers would lose half a pixel and erase one-pixel strokes.
    cells = np.zeros((height * 2 + 1, width * 2 + 1), np.uint8)
    cells[1::2, 1::2] = mask[y:y + height, x:x + width]
    cells = cv2.dilate(cells, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(cells, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour.reshape(-1, 2) / 2 + [x - .5, y - .5]
        path.addPolygon(QPolygonF([QPointF(*point) for point in points]))
        path.closeSubpath()
    return path


def _expand_curve_sides(nodes: np.ndarray, source: QPainterPath, samples: np.ndarray) -> np.ndarray:
    """Move existing arcs outside the text contour, adding at most four supports.

    >>> nodes = np.tile([[0., 0.], [10., 0.], [10., 10.], [0., 10.]], (1, 3))
    >>> samples = np.stack([np.linspace(a[:2], b[:2], 33)
    ...     for a, b in zip(nodes, np.roll(nodes, -1, axis=0))])
    >>> source = QPainterPath(); source.addRect(QRectF(-2, 2, 14, 6))
    >>> len(_expand_curve_sides(nodes, source, samples))
    4
    """
    curve = samples[:, :-1].reshape(-1, 2)
    area = np.sum(curve[:, 0] * np.roll(curve[:, 1], -1)
                  - curve[:, 1] * np.roll(curve[:, 0], -1))
    orientation = 1 if area >= 0 else -1
    # Dense polygon fallbacks already have enough anchors; respect the saved limit.
    steps = 32 if len(nodes) <= 60 else 1
    points = curve if steps == 32 else nodes[:, :2]
    extent = np.maximum(np.ptp(points, axis=0), 1e-9)
    scores = (points / extent) @ np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]).T
    first = int(scores[:, 0].argmax())
    order = (first + orientation * np.arange(len(points))) % len(points)
    supports, cursor = [first], 1
    # Ordered supports also handle a sharp vertex shared by two corner directions.
    for corner in range(1, 4):
        stop = len(points) - (3 - corner)
        cursor += int(scores[order[cursor:stop], corner].argmax())
        supports.append(int(order[cursor]))
        cursor += 1
    parameters = np.array(supports) / steps
    nodes = nodes.copy()
    landmarks, ends = {}, {}
    for corner in np.argsort(parameters)[::-1]:
        index = int(parameters[corner])
        fraction = parameters[corner] - index
        if fraction:
            nodes = _split_bubble_nodes(nodes, index, fraction / ends.get(index, 1.))
            ends[index] = fraction
            landmarks = {key: value + int(value > index) for key, value in landmarks.items()}
            landmarks[int(corner)] = index + 1
        else:
            landmarks[int(corner)] = index

    normals = np.empty((len(nodes), 2))
    for side in range(4):
        start, end = landmarks[side], landmarks[(side + 1) % 4]
        if orientation < 0:
            start, end = end, start
        indices = np.arange(start, end + (len(nodes) if end < start else 0)) % len(nodes)
        fraction = (np.arange(len(indices)) + .5) / len(indices)
        if orientation < 0:
            fraction = 1 - fraction
        angle = (side - 1.5 + fraction) * math.pi / 2
        normals[indices] = np.column_stack((np.cos(angle), np.sin(angle)))
    # Supporting lines follow the mask contour, not the empty corners of its box.
    foreground = np.array([(point.x(), point.y()) for point in source.toFillPolygon()])
    limits = (foreground @ normals.T).max(axis=0) + .15
    # Independent projections can reverse nearby supports at a narrow notch.
    # Radial placement keeps the existing anchors in order around the text.
    center = foreground.mean(axis=0)
    for index in range(len(nodes)):
        # Ordered normals are nonparallel; avoid LAPACK's worker-stack overflow.
        (a, b), (c, d) = normals[[index - 1, index]]
        u, v = limits[[index - 1, index]]
        corner = np.array([u * d - b * v, a * v - u * c]) / (a * d - b * c)
        ray = corner - center
        distance = max(1., np.dot(nodes[index, :2] - center, ray) / np.dot(ray, ray))
        anchor = center + distance * ray
        nodes[index] += np.tile(anchor - nodes[index, :2], 3)
    # Reorient smooth handles after moving a support, avoiding hooked shoulders.
    anchors = nodes[:, :2]
    incoming = anchors - np.roll(anchors, 1, axis=0)
    outgoing = np.roll(anchors, -1, axis=0) - anchors
    tangent = (incoming / np.maximum(np.linalg.norm(incoming, axis=1)[:, None], 1e-12)
               + outgoing / np.maximum(np.linalg.norm(outgoing, axis=1)[:, None], 1e-12))
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-12)
    lengths = np.linalg.norm(nodes[:, 2:].reshape(-1, 2, 2) - anchors[:, None], axis=2)
    smooth = np.all(lengths > 1e-7, axis=1)
    nodes[smooth, 2:4] = anchors[smooth] - tangent[smooth] * lengths[smooth, :1]
    nodes[smooth, 4:] = anchors[smooth] + tangent[smooth] * lengths[smooth, 1:]
    # Keep arcs in separate angular sectors, advancing along their chords.
    # Their controls also stay outside the supporting lines to enclose the text.
    for index, normal in enumerate(normals):
        following = (index + 1) % len(nodes)
        first, last = nodes[[index, following], :2] - center
        sector = orientation * np.array([[-first[1], first[0]], [last[1], -last[0]]])
        for current, handle, other in ((index, slice(4, 6), following),
                                       (following, slice(2, 4), index)):
            anchor = nodes[current, :2]
            vector = nodes[current, handle] - anchor
            chord = nodes[other, :2] - anchor
            advance = np.dot(chord, vector)
            cap = min(1., max(0., advance / max(np.dot(vector, vector), 1e-12)))
            if advance > 0:
                cap = min(cap, .5 * np.dot(chord, chord) / advance)
            for inward in sector:
                direction = np.dot(vector, inward)
                if direction < 0:
                    cap = min(cap, np.dot(anchor - center, inward) / -direction)
            direction = np.dot(vector, normal)
            if direction < 0:
                cap = min(cap, (np.dot(anchor, normal) - limits[index]) / -direction)
            nodes[current, handle] = anchor + vector * max(0., cap)
    return nodes


def _cover_text_region(nodes: np.ndarray, required: QPainterPath) -> np.ndarray:
    """Expand one fitted path without refitting or changing its winding order.

    Adjust handle lengths along their tangents before moving neighboring anchors.
    Difficult concavities use the same arcs with at most four additional anchors.

    >>> nodes = np.tile([[0., 0.], [10., 0.], [10., 10.], [0., 10.]], (1, 3))
    >>> required = QPainterPath(); required.addRect(QRectF(2, 2, 10, 6))
    >>> len(_cover_text_region(nodes, required))
    4
    """
    nodes = np.asarray(nodes, dtype=float)
    path = bubble_path(nodes)
    if required.subtracted(path).isEmpty():
        return nodes
    stroker = QPainterPathStroker()
    stroker.setWidth(.1)
    padded = required.united(stroker.createStroke(required))

    def convexity(curve: np.ndarray) -> float:
        points = curve[:, :-1].reshape(-1, 2)
        if not LinearRing(points).is_simple:
            return -math.inf
        points = points.astype(np.float32)
        return cv2.contourArea(points) / max(cv2.contourArea(cv2.convexHull(points)), 1e-12)

    original = nodes
    original_samples = _sample_bubble_nodes(original)
    minimum_convexity = max(0., convexity(original_samples) - .01)
    bounds = path.boundingRect()
    low = np.array([bounds.left(), bounds.top()])
    high = np.array([bounds.right(), bounds.bottom()])
    rect = required.boundingRect()
    target_low = np.minimum(low, [rect.left() - .15, rect.top() - .15])
    target_high = np.maximum(high, [rect.right() + .15, rect.bottom() + .15])
    # First extend missing extents without folding a short arc across distant text.
    nodes = ((nodes.reshape(-1, 2) - low) * (target_high - target_low)
             / (high - low) + target_low).reshape(-1, 6)
    for _ in range(8):
        path = bubble_path(nodes)
        missing = padded.subtracted(path)
        if missing.isEmpty():
            break
        # Keep components separate: fill-polygon bridges run through empty space.
        polygons = [np.array([(point.x(), point.y()) for point in polygon])
                    for polygon in missing.toSubpathPolygons()]
        targets = np.concatenate([np.concatenate((p, (p + np.roll(p, 1, axis=0)) / 2))
                                  for p in polygons])
        targets = np.array([point for point in targets if not path.contains(QPointF(*point))])
        if not len(targets):
            break
        curve = _sample_bubble_nodes(nodes)
        starts = curve[:, :-1].reshape(-1, 2)
        edges = (curve[:, 1:] - curve[:, :-1]).reshape(-1, 2)
        projection = np.clip(np.sum((targets[:, None] - starts) * edges, axis=-1)
                             / np.maximum(np.sum(edges * edges, axis=-1), 1e-12), 0, 1)
        nearest = starts + projection[:, :, None] * edges
        distances = np.sum((targets[:, None] - nearest) ** 2, axis=-1)
        segments = distances.argmin(axis=1)
        worst = int(np.argmax(distances[np.arange(len(targets)), segments]))
        index = int(segments[worst])
        segment, fraction = divmod(index, 32)
        u = (fraction + projection[worst, index]) / 32
        delta = targets[worst] - nearest[worst, index]
        distance = np.linalg.norm(delta)
        if distance < 1e-7:
            break
        # The small native-pixel margin survives saved-frame and handle rounding.
        delta *= 1 + .1 / distance
        following = (segment + 1) % len(nodes)
        handles = np.array([nodes[segment, 4:] - nodes[segment, :2],
                            nodes[following, 2:4] - nodes[following, :2]])
        lengths = np.linalg.norm(handles, axis=1)
        directions = handles / np.maximum(lengths[:, None], 1e-12)
        coefficients = np.array([3 * (1 - u) ** 2 * u, 3 * (1 - u) * u * u])
        matrix = (directions * coefficients[:, None]).T
        extension = np.linalg.lstsq(matrix, delta, rcond=None)[0]
        chord = nodes[following, :2] - nodes[segment, :2]
        maximum = np.array([np.dot(chord, directions[0]), np.dot(-chord, directions[1])])
        extension = np.clip(extension, 0, np.maximum(0, maximum - lengths))
        nodes[segment, 4:] += directions[0] * extension[0]
        nodes[following, 2:4] += directions[1] * extension[1]
        delta -= matrix @ extension
        # Translate an anchor and both handles together to keep smooth joins smooth.
        weights = np.array([(1 - u) ** 2 * (1 + 2 * u), u * u * (3 - 2 * u)])
        offsets = weights[:, None] * delta / np.sum(weights * weights)
        nodes[segment] += np.tile(offsets[0], 3)
        nodes[following] += np.tile(offsets[1], 3)
    # Moving an angular fit's handles too far can create new inward dents.
    if (required.subtracted(bubble_path(nodes)).isEmpty()
            and convexity(_sample_bubble_nodes(nodes)) >= minimum_convexity):
        return nodes
    return _expand_curve_sides(original, required, original_samples)


def trace_bubble(mask: np.ndarray, page_size: Tuple[int, int], angle: float = 0,
                 text_mask: Optional[np.ndarray] = None) -> Optional[BubbleShape]:
    """Approximate the usable bubble body with a few editable cubic segments.

    Narrow tails/spikes do not guide typesetting. Prefer rectangles/ellipses,
    then fit short cubic chains for clipped, joined or otherwise irregular bodies.
    Then adjust that fit to cover the optional text mask on the same native grid.

    >>> mask = np.zeros((40, 60), np.uint8)
    >>> mask[5:35, 10:50] = 1
    >>> len(trace_bubble(mask, (600, 400)).nodes)
    4
    """
    scale = np.array(page_size) / np.array(mask.shape[::-1])
    source = None
    if text_mask is not None:
        source = np.asarray(text_mask, dtype=np.uint8)
        if source.shape != mask.shape:
            raise ValueError('text mask must match the bubble mask grid')
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    moments = cv2.moments(source, binaryImage=True) if source is not None and len(contours) > 1 else None
    if moments is None or not moments['m00']:
        contour = max(contours, key=cv2.contourArea)
    else:
        # Segmentation can have disconnected lobes; keep the one owning this text.
        point = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        contour = max(contours, key=lambda candidate: cv2.pointPolygonTest(candidate, point, True))
    contour_area = cv2.contourArea(contour)
    if contour_area < 4:
        return None

    x, y, width, height = cv2.boundingRect(contour)
    contour = contour - np.array([x, y])
    original = np.zeros((height, width), np.uint8)
    cv2.fillPoly(original, [contour], 1)
    radius = max(1, round(min(width, height) * .06))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    body = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Opening must not discard useful lobes connected by a thin neck.
    if len(contours) == 1:
        opened = contours[0]
        opened_area = cv2.contourArea(opened)
        if opened_area >= contour_area * .8:
            contour = opened
            contour_area = opened_area

    original_area = np.count_nonzero(original)
    candidate = np.zeros_like(original)

    def usable_primitive() -> bool:
        overlap = np.count_nonzero(candidate & original)
        # Small boundary errors are harmless inside the layout's ink-band inset.
        return (overlap >= np.count_nonzero(candidate) * .96
                and overlap >= original_area * .84)

    rectangle = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rectangle)
    limit = np.array([width, height]) - .5
    cv2.fillPoly(candidate, [np.rint(corners).astype(np.int32)], 1)
    if (contour_area / max(np.prod(rectangle[1]), 1) >= .9
            and np.all(corners >= -.5) and np.all(corners <= limit) and usable_primitive()):
        nodes = np.tile(corners, (1, 3))
    else:
        nodes = None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            center, axes, rotation = ellipse
            rad = math.radians(rotation)
            for shrink in (1., .97, .94):
                fitted_axes = tuple(np.array(axes) * shrink)
                a, b = np.array(fitted_axes) / 2
                extent = np.array([math.hypot(a * math.cos(rad), b * math.sin(rad)),
                                   math.hypot(a * math.sin(rad), b * math.cos(rad))])
                # A clipped ellipse must stay clipped rather than grow off the page.
                if np.any(np.array(center) - extent < -.5) or np.any(np.array(center) + extent > limit):
                    continue
                candidate.fill(0)
                cv2.ellipse(candidate, (center, fitted_axes, rotation), 1, -1)
                if not usable_primitive():
                    continue
                k = .5522847498307936
                nodes = np.array([[1, 0, 1, -k, 1, k], [0, 1, k, 1, -k, 1],
                                  [-1, 0, -1, k, -1, -k], [0, -1, -k, -1, k, -1]], float)
                points = nodes.reshape(-1, 2) * (np.array(fitted_axes) / 2)
                points = points @ np.array([[math.cos(rad), math.sin(rad)], [-math.sin(rad), math.cos(rad)]])
                nodes = (points + center).reshape(-1, 6)
                break
        if nodes is None:
            tolerance = max(.8, min(width, height) * .035)
            nodes = _fit_bubble_curve(contour, tolerance)
            if nodes is None:
                # Intricate concavities still need corners when bounded cubics cannot fit.
                epsilon = tolerance
                anchors = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                while len(anchors) > 64:
                    epsilon *= 1.5
                    anchors = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                if len(anchors) < 3:
                    return None
                nodes = np.tile(anchors, (1, 3))

    if source is not None:
        required = _text_mask_path(source)
        required.translate(-x, -y)
        nodes = _cover_text_region(nodes, required)
    points = nodes.reshape(-1, 2).astype(float)
    points = (points + [x + .5, y + .5]) * scale
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
