"""Glyph-local slant rendering for attached ``QTextLayout`` objects.

The renderer is deliberately read-only with respect to the document and its
layouts.  It consumes the glyph runs Qt has already shaped, then maps their ink
through the glyph-local shear and the placement/orientation supplied by
``text_engine.layout``.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QBrush,
    QColor,
    QGlyphRun,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRawFont,
    QTextBlock,
    QTextCharFormat,
    QTextFormat,
    QTextLayout,
    QTextLine,
    QTransform,
)

from ballontranslator.ui.misc import ndarray2pixmap, pixmap2ndarray
from ballontranslator.utils.fontformat import (
    TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    TEXT_TRANSFORM_GLYPH_SLANT_MIN,
    normalize_text_transform_value,
)


GLYPH_STROKE_FORMAT_PROPERTY = 0x100000 + 1239
FALLBACK_RASTER_MAX_SCALE = 8.0
FALLBACK_RASTER_MAX_PIXELS = 4_194_304
FALLBACK_RASTER_MAX_DIMENSION = 8192
FALLBACK_RASTER_MAX_BYTES = 32 * 1024 * 1024
_COLOR_FONT_TABLES = ('COLR', 'CBDT', 'sbix', 'SVG ')
_COLOR_FONT_CACHE = {}
_COLOR_FONT_CACHE_MAX_ENTRIES = 128
_GLYPH_PATH_CACHE = {}
_GLYPH_PATH_CACHE_MAX_ENTRIES = 4096
GLYPH_GEOMETRY_CACHE_MAX_ENTRIES = 16384
GLYPH_GEOMETRY_CACHE_MAX_BYTES = 64 * 1024 * 1024
GLYPH_PREVIEW_GEOMETRY_CACHE_MAX_ENTRIES = 4096
GLYPH_PREVIEW_GEOMETRY_CACHE_MAX_BYTES = 16 * 1024 * 1024


class GlyphRasterAllocationError(RuntimeError):
    """A pathless/color glyph raster could not be produced losslessly."""


class PaintSpan(NamedTuple):
    """One UTF-16 paint range with its fully merged character format."""

    start: int
    length: int
    char_format: QTextCharFormat


class FallbackGlyph(NamedTuple):
    run: QGlyphRun
    transform: QTransform
    bounds: QRectF
    raw_bounds: QRectF
    native_color: bool


class GlyphGeometry(NamedTuple):
    """Vector ink plus pathless glyph fallbacks for one paint span."""

    paths: Tuple[QPainterPath, ...]
    fallbacks: Tuple[FallbackGlyph, ...]
    bounds: QRectF

    @property
    def path(self) -> QPainterPath:
        """Return a compatibility union without changing paint fill rules."""
        combined = QPainterPath()
        for glyph_path in self.paths:
            combined = (
                QPainterPath(glyph_path)
                if combined.isEmpty()
                else combined.united(glyph_path)
            )
        return combined


def _glyph_geometry_cache_weight(geometry: GlyphGeometry) -> int:
    """Conservatively estimate Python and C++ path storage for eviction."""
    path_bytes = sum(
        256 + path.elementCount() * 96
        for path in geometry.paths
    )
    fallback_bytes = len(geometry.fallbacks) * 512
    return 512 + path_bytes + fallback_bytes


class WeightedGlyphGeometryCache:
    """Process-wide LRU bounded by both entries and estimated path bytes.

    >>> cache = WeightedGlyphGeometryCache(max_entries=2, max_weight=10)
    >>> cache.store('a', 1, weight=6)
    >>> cache.store('b', 2, weight=6)
    >>> cache.get('a') is None
    True
    >>> cache.get('b')
    2
    """

    def __init__(self, max_entries: int, max_weight: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_weight = max(1, int(max_weight))
        self._entries = OrderedDict()
        self._weights = {}
        self._key_namespaces = {}
        self._namespace_keys = {}
        self.total_weight = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key):
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def _remove(self, key) -> None:
        if key not in self._entries:
            return
        self._entries.pop(key)
        self.total_weight -= self._weights.pop(key)
        namespace = self._key_namespaces.pop(key, None)
        if namespace is not None:
            namespace_keys = self._namespace_keys[namespace]
            namespace_keys.discard(key)
            if not namespace_keys:
                self._namespace_keys.pop(namespace)

    def store(
        self,
        key,
        value,
        weight: Optional[int] = None,
        namespace=None,
    ) -> None:
        if weight is None:
            weight = _glyph_geometry_cache_weight(value)
        weight = max(1, int(weight))
        if weight > self.max_weight:
            return
        if key in self._entries:
            self._remove(key)
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.total_weight + weight > self.max_weight
        ):
            self._remove(next(iter(self._entries)))
        self._entries[key] = value
        self._weights[key] = weight
        if namespace is not None:
            self._key_namespaces[key] = namespace
            self._namespace_keys.setdefault(namespace, set()).add(key)
        self.total_weight += weight

    def discard_namespace(self, namespace) -> None:
        for key in tuple(self._namespace_keys.get(namespace, ())):
            self._remove(key)

    def clear(self) -> None:
        self._entries.clear()
        self._weights.clear()
        self._key_namespaces.clear()
        self._namespace_keys.clear()
        self.total_weight = 0


GLOBAL_GLYPH_GEOMETRY_CACHE = WeightedGlyphGeometryCache(
    GLYPH_GEOMETRY_CACHE_MAX_ENTRIES,
    GLYPH_GEOMETRY_CACHE_MAX_BYTES,
)
GLOBAL_GLYPH_PREVIEW_GEOMETRY_CACHE = WeightedGlyphGeometryCache(
    GLYPH_PREVIEW_GEOMETRY_CACHE_MAX_ENTRIES,
    GLYPH_PREVIEW_GEOMETRY_CACHE_MAX_BYTES,
)


def glyph_slant_transform(angle: float, baseline_y: float) -> QTransform:
    """Build ``x' = x - tan(angle) * (y - baseline_y)``."""
    angle = normalize_text_transform_value(
        angle,
        TEXT_TRANSFORM_GLYPH_SLANT_MIN,
        TEXT_TRANSFORM_GLYPH_SLANT_MAX,
    )
    tangent = math.tan(math.radians(angle))
    return QTransform(1.0, 0.0, -tangent, 1.0, tangent * baseline_y, 0.0)


def _mapped_rect(rect: QRectF, transform: QTransform) -> QRectF:
    if rect.isEmpty():
        return QRectF()
    polygon = QPolygonF(
        [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
    )
    return transform.map(polygon).boundingRect()


def _composed_transform(*transforms: QTransform) -> QTransform:
    """Compose maps in argument order without relying on binding ``*`` rules."""

    def mapped(point: QPointF) -> QPointF:
        for transform in transforms:
            point = transform.map(point)
        return point

    origin = mapped(QPointF(0.0, 0.0))
    x_axis = mapped(QPointF(1.0, 0.0))
    y_axis = mapped(QPointF(0.0, 1.0))
    return QTransform(
        x_axis.x() - origin.x(),
        x_axis.y() - origin.y(),
        y_axis.x() - origin.x(),
        y_axis.y() - origin.y(),
        origin.x(),
        origin.y(),
    )


def _format_at(block: QTextBlock, local_position: int) -> QTextCharFormat:
    absolute = block.position() + local_position
    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if (
            fragment.isValid()
            and fragment.position() <= absolute
            and absolute < fragment.position() + fragment.length()
        ):
            return QTextCharFormat(fragment.charFormat())
        iterator += 1
    return QTextCharFormat(block.charFormat())


def _selection_range(
    block: QTextBlock,
    selection: QAbstractTextDocumentLayout.Selection,
    line: Optional[QTextLine] = None,
) -> Optional[Tuple[int, int]]:
    start = selection.cursor.selectionStart() - block.position()
    end = selection.cursor.selectionEnd() - block.position()
    if end <= start:
        if (
            line is None
            or selection.cursor.hasSelection()
            or not selection.format.hasProperty(
                QTextFormat.FullWidthSelection
            )
            or not block.contains(selection.cursor.position())
        ):
            return None
        cursor_position = selection.cursor.position() - block.position()
        cursor_line = block.layout().lineForTextPosition(cursor_position)
        if (
            not cursor_line.isValid()
            or cursor_line.textStart() != line.textStart()
        ):
            return None
        return (
            line.textStart(),
            line.textStart() + line.textLength(),
        )
    block_length = max(0, block.length() - 1)
    start = max(0, min(start, block_length))
    end = max(start, min(end, block_length))
    return None if end <= start else (start, end)


def resolve_paint_spans(
    block: QTextBlock,
    line: QTextLine,
    additional_formats: Sequence[QTextLayout.FormatRange],
    selection: Optional[QAbstractTextDocumentLayout.Selection] = None,
) -> Tuple[PaintSpan, ...]:
    """Resolve fragment/additional/selection formats at UTF-16 boundaries.

    Additional formats are merged in their stored order.  A selection, when
    supplied, is merged last and clipped to its logical range by the caller.
    """
    line_start = line.textStart()
    line_end = line_start + line.textLength()
    if line_end <= line_start:
        return ()
    boundaries = {line_start, line_end}

    iterator = block.begin()
    while not iterator.atEnd():
        fragment = iterator.fragment()
        if fragment.isValid():
            start = fragment.position() - block.position()
            end = start + fragment.length()
            if start < line_end and end > line_start:
                boundaries.update((max(start, line_start), min(end, line_end)))
        iterator += 1

    for format_range in additional_formats:
        start = int(format_range.start)
        end = start + int(format_range.length)
        if start < line_end and end > line_start:
            boundaries.update((max(start, line_start), min(end, line_end)))

    selection_bounds = None
    if selection is not None:
        selection_bounds = _selection_range(block, selection, line)
        if selection_bounds is not None:
            start, end = selection_bounds
            if start < line_end and end > line_start:
                boundaries.update((max(start, line_start), min(end, line_end)))

    ordered = sorted(boundaries)
    spans: List[PaintSpan] = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        char_format = _format_at(block, start)
        for format_range in additional_formats:
            range_start = int(format_range.start)
            range_end = range_start + int(format_range.length)
            if range_start <= start < range_end:
                char_format.merge(format_range.format)
        if (
            selection is not None
            and selection_bounds is not None
            and selection_bounds[0] <= start < selection_bounds[1]
        ):
            char_format.merge(selection.format)
        spans.append(PaintSpan(start, end - start, char_format))
    return tuple(spans)


def logical_span_rect(
    line: QTextLine,
    start: int,
    length: int,
    offset: QPointF,
    orientation: QTransform,
) -> QRectF:
    """Return the unchanged logical cell/line rectangle for a UTF-16 span."""
    return _mapped_rect(
        _logical_span_base_rect(line, start, length, offset), orientation
    )


def _logical_span_base_rect(
    line: QTextLine,
    start: int,
    length: int,
    offset: QPointF,
) -> QRectF:
    """Return a span cell before vertical glyph orientation is applied."""
    relative_start = max(line.textStart(), start)
    relative_end = min(line.textStart() + line.textLength(), start + length)
    if relative_end <= relative_start:
        return QRectF()
    def cursor_x(position: int) -> float:
        value = line.cursorToX(position)
        # PyQt6 exposes the qreal* trailing overload as ``(x, trailing)``;
        # PyQt5/PySide6 return the qreal directly for the same call.
        if isinstance(value, (tuple, list)):
            value = value[0]
        return float(value)

    x1 = cursor_x(relative_start)
    x2 = cursor_x(relative_end)
    rect = QRectF(
        min(x1, x2) + offset.x(),
        line.y() + offset.y(),
        abs(x2 - x1),
        line.height(),
    )
    return rect


def _fallback_run(raw_font: QRawFont, glyph_index: int) -> QGlyphRun:
    run = QGlyphRun()
    run.setRawFont(raw_font)
    run.setGlyphIndexes([glyph_index])
    run.setPositions([QPointF(0.0, 0.0)])
    run.setUnderline(False)
    run.setOverline(False)
    run.setStrikeOut(False)
    return run


def _raw_font_has_color_glyphs(raw_font: QRawFont) -> bool:
    """Detect color-capable fonts whose native glyph can exceed its path."""
    try:
        is_valid = getattr(raw_font, 'isValid', None)
        if is_valid is not None and not is_valid():
            return False
        # Keep the QRawFont value itself alive as the cache key. On PyQt6,
        # familyName()/styleName() can dereference a null fallback-font face
        # even when isValid() reports true for a glyph run returned by Qt.
        key = (type(raw_font), raw_font)
        hash(key)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        key = None
    if key is not None and key in _COLOR_FONT_CACHE:
        return _COLOR_FONT_CACHE[key]
    has_color = False
    try:
        for table_name in _COLOR_FONT_TABLES:
            table = raw_font.fontTable(table_name)
            size = table.size() if hasattr(table, 'size') else len(table)
            if size > 0:
                has_color = True
                break
    except (AttributeError, RuntimeError, TypeError, ValueError):
        has_color = False
    if key is not None:
        while len(_COLOR_FONT_CACHE) >= _COLOR_FONT_CACHE_MAX_ENTRIES:
            _COLOR_FONT_CACHE.pop(next(iter(_COLOR_FONT_CACHE)))
        _COLOR_FONT_CACHE[key] = has_color
    return has_color


def _raw_glyph_path(raw_font: QRawFont, glyph_index: int) -> QPainterPath:
    """Return a bounded shared cache entry for immutable raw glyph outlines."""
    try:
        key = (type(raw_font), raw_font, int(glyph_index))
        hash(key)
    except (RuntimeError, TypeError, ValueError):
        return raw_font.pathForGlyph(glyph_index)
    cached = _GLYPH_PATH_CACHE.get(key)
    if cached is not None:
        return QPainterPath(cached)
    path = raw_font.pathForGlyph(glyph_index)
    while len(_GLYPH_PATH_CACHE) >= _GLYPH_PATH_CACHE_MAX_ENTRIES:
        _GLYPH_PATH_CACHE.pop(next(iter(_GLYPH_PATH_CACHE)))
    _GLYPH_PATH_CACHE[key] = QPainterPath(path)
    return path


def glyph_geometry(
    line: QTextLine,
    start: int,
    length: int,
    offset: QPointF,
    orientation: QTransform,
    angle: float,
) -> GlyphGeometry:
    """Build the exact vector/fallback geometry used by paint and bounds."""
    paths = []
    fallbacks = []
    bounds = QRectF()
    baseline = line.y() + line.ascent() + offset.y()
    shear = glyph_slant_transform(angle, baseline)

    for run in line.glyphRuns(start, length):
        raw_font = run.rawFont()
        native_color_glyphs = _raw_font_has_color_glyphs(raw_font)
        indexes = list(run.glyphIndexes())
        positions = list(run.positions())
        for glyph_index, position in zip(indexes, positions):
            translation = QTransform.fromTranslate(
                position.x() + offset.x(), position.y() + offset.y()
            )
            glyph_to_item = _composed_transform(translation, shear, orientation)
            glyph_path = _raw_glyph_path(raw_font, glyph_index)
            if not native_color_glyphs and not glyph_path.isEmpty():
                mapped_path = glyph_to_item.map(glyph_path)
                # Preserve the raw font's per-glyph fill rule. Drawing glyphs
                # separately prevents overlap cancellation without turning a
                # legitimate OddEven counter into solid ink.
                paths.append(mapped_path)
                glyph_bounds = mapped_path.boundingRect()
            else:
                raw_bounds = raw_font.boundingRect(glyph_index)
                glyph_bounds = _mapped_rect(raw_bounds, glyph_to_item)
                fallbacks.append(
                    FallbackGlyph(
                        _fallback_run(raw_font, glyph_index),
                        glyph_to_item,
                        glyph_bounds,
                        raw_bounds,
                        native_color_glyphs,
                    )
                )
            bounds = glyph_bounds if bounds.isNull() else bounds.united(glyph_bounds)
    return GlyphGeometry(tuple(paths), tuple(fallbacks), bounds)


def _foreground_brush(char_format: QTextCharFormat, painter: QPainter) -> QBrush:
    brush = char_format.foreground()
    if brush.style() == Qt.BrushStyle.NoBrush:
        return QBrush(painter.pen().brush())
    return brush


def _item_space_brush(brush: QBrush, glyph_transform: QTransform) -> QBrush:
    """Keep a brush field item-local while glyph ink uses a local transform."""
    compensated = QBrush(brush)
    inverse, invertible = glyph_transform.inverted()
    if invertible:
        compensated.setTransform(
            _composed_transform(compensated.transform(), inverse)
        )
    return compensated


def _fallback_device_scale(painter: QPainter) -> float:
    transform = painter.deviceTransform()
    a, b = transform.m11(), transform.m21()
    c, d = transform.m12(), transform.m22()
    trace = a * a + b * b + c * c + d * d
    determinant_squared = (a * d - b * c) ** 2
    discriminant = max(0.0, trace * trace - 4 * determinant_squared)
    scale = math.sqrt((trace + math.sqrt(discriminant)) / 2)
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0
    return min(max(1.0, scale), FALLBACK_RASTER_MAX_SCALE)


def _bounded_fallback_scale(rect: QRectF, requested_scale: float) -> float:
    width = max(rect.width(), 1.0)
    height = max(rect.height(), 1.0)
    scale = min(
        requested_scale,
        FALLBACK_RASTER_MAX_DIMENSION / width,
        FALLBACK_RASTER_MAX_DIMENSION / height,
        math.sqrt(FALLBACK_RASTER_MAX_PIXELS / (width * height)),
        math.sqrt((FALLBACK_RASTER_MAX_BYTES / 4) / (width * height)),
    )
    return max(scale, 1.0 / max(width, height))


def _aligned_fallback_rect(rect: QRectF, scale: float) -> QRectF:
    left = math.floor(rect.left() * scale) / scale
    top = math.floor(rect.top() * scale) / scale
    right = math.ceil(rect.right() * scale) / scale
    bottom = math.ceil(rect.bottom() * scale) / scale
    return QRectF(left, top, right - left, bottom - top)


def _fallback_raster_plan(
    painter: QPainter,
    bounds: QRectF,
    outline_radius: float,
) -> Tuple[QRectF, float, int, int]:
    requested_scale = _fallback_device_scale(painter)
    expanded = bounds.adjusted(
        -outline_radius,
        -outline_radius,
        outline_radius,
        outline_radius,
    )
    scale = _bounded_fallback_scale(expanded, requested_scale)
    # Two device pixels retain antialiasing and a transparent sampling border.
    guard = 2.0 / scale
    expanded = expanded.adjusted(-guard, -guard, guard, guard)
    scale = _bounded_fallback_scale(expanded, scale)
    for _ in range(4):
        raster_rect = _aligned_fallback_rect(expanded, scale)
        pixel_width = max(1, math.ceil(raster_rect.width() * scale))
        pixel_height = max(1, math.ceil(raster_rect.height() * scale))
        pixels = pixel_width * pixel_height
        if (
            pixel_width <= FALLBACK_RASTER_MAX_DIMENSION
            and pixel_height <= FALLBACK_RASTER_MAX_DIMENSION
            and pixels <= FALLBACK_RASTER_MAX_PIXELS
            and pixels * 4 <= FALLBACK_RASTER_MAX_BYTES
        ):
            return raster_rect, scale, pixel_width, pixel_height
        scale *= 0.98
    raise MemoryError('pathless glyph raster exceeds policy')


def _draw_direct_fallbacks(
    painter: QPainter,
    fallbacks: Sequence[FallbackGlyph],
    brush: QBrush,
) -> None:
    """Allocation-free last resort that always retains the native glyph fill."""
    for fallback in fallbacks:
        run = fallback.run
        transform = fallback.transform
        painter.save()
        try:
            painter.setTransform(transform, True)
            painter.setPen(
                QPen(_item_space_brush(brush, transform), 0.0)
            )
            painter.drawGlyphRun(QPointF(), run)
        finally:
            painter.restore()


def _dilate_fallback_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    dilated = alpha
    remaining = max(0, int(radius))
    # Repeated disk dilation is bounded while retaining a continuous thick
    # silhouette; it avoids allocating a quadratic huge-stroke kernel.
    while remaining:
        chunk = min(remaining, 64)
        diameter = chunk * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (diameter, diameter)
        )
        dilated = cv2.dilate(dilated, kernel)
        remaining -= chunk
    return dilated


def _native_color_glyph_image(
    fallback: FallbackGlyph,
    scale: float,
) -> Tuple[Optional[QImage], QRectF, bool]:
    """Rasterize without shear so Qt retains intrinsic color layers."""
    if fallback.raw_bounds.isEmpty():
        return None, QRectF(), False
    guard = 2.0 / scale
    raw_rect = _aligned_fallback_rect(
        fallback.raw_bounds.adjusted(-guard, -guard, guard, guard),
        scale,
    )
    pixel_width = max(1, math.ceil(raw_rect.width() * scale))
    pixel_height = max(1, math.ceil(raw_rect.height() * scale))
    pixels = pixel_width * pixel_height
    if (
        pixel_width > FALLBACK_RASTER_MAX_DIMENSION
        or pixel_height > FALLBACK_RASTER_MAX_DIMENSION
        or pixels > FALLBACK_RASTER_MAX_PIXELS
        or pixels * 4 > FALLBACK_RASTER_MAX_BYTES
    ):
        raise MemoryError('native color glyph raster exceeds policy')

    image = QImage(
        pixel_width, pixel_height, QImage.Format.Format_ARGB32
    )
    if image.isNull():
        raise MemoryError('unable to allocate native color glyph image')
    image.setDevicePixelRatio(scale)
    image.fill(Qt.GlobalColor.transparent)
    probe_color = QColor(3, 251, 5)
    image_painter = QPainter(image)
    if not image_painter.isActive():
        raise MemoryError('unable to begin native color glyph painter')
    try:
        image_painter.setRenderHint(
            QPainter.RenderHint.TextAntialiasing, True
        )
        image_painter.translate(-raw_rect.topLeft())
        image_painter.setPen(QPen(probe_color, 0.0))
        image_painter.drawGlyphRun(QPointF(), fallback.run)
    finally:
        image_painter.end()

    rgba = pixmap2ndarray(image, keep_alpha=True)
    if rgba is None:
        raise MemoryError('unable to access native color glyph pixels')
    alpha = rgba[..., 3]
    opaque = alpha > 250
    if not np.any(opaque) and int(alpha.max()) > 0:
        opaque = alpha >= max(1, int(alpha.max()) - 2)
    if not np.any(opaque):
        return image, raw_rect, False
    expected = np.array(
        [probe_color.red(), probe_color.green(), probe_color.blue()],
        dtype=np.int16,
    )
    actual = rgba[..., :3][opaque].astype(np.int16)
    intrinsic_color = bool(np.any(np.abs(actual - expected) > 8))
    return image, raw_rect, intrinsic_color


def _draw_fallbacks(
    painter: QPainter,
    fallbacks: Sequence[FallbackGlyph],
    brush: QBrush,
    outline: QPen,
    failure_handler=None,
    effect_pass: bool = False,
) -> None:
    if not fallbacks:
        return
    bounds = QRectF()
    for fallback in fallbacks:
        glyph_bounds = fallback.bounds
        bounds = (
            QRectF(glyph_bounds)
            if bounds.isNull()
            else bounds.united(glyph_bounds)
        )
    if bounds.isEmpty():
        _draw_direct_fallbacks(painter, fallbacks, brush)
        return

    outline_enabled = outline.style() != Qt.PenStyle.NoPen
    requested_scale = _fallback_device_scale(painter)
    outline_radius = (
        max(outline.widthF() / 2.0, 0.5 / requested_scale)
        if outline_enabled
        else 0.0
    )
    try:
        raster_rect, scale, pixel_width, pixel_height = (
            _fallback_raster_plan(painter, bounds, outline_radius)
        )
        fill_image = QImage(
            pixel_width, pixel_height, QImage.Format.Format_ARGB32
        )
        if fill_image.isNull():
            raise MemoryError('unable to allocate pathless glyph image')
        fill_image.setDevicePixelRatio(scale)
        fill_image.fill(Qt.GlobalColor.transparent)
        fill_painter = QPainter(fill_image)
        if not fill_painter.isActive():
            raise MemoryError('unable to begin pathless glyph painter')
        try:
            fill_painter.setRenderHint(
                QPainter.RenderHint.TextAntialiasing, True
            )
            fill_painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, True
            )
            fill_painter.translate(-raster_rect.topLeft())
            for fallback in fallbacks:
                run = fallback.run
                transform = fallback.transform
                native_image = None
                native_rect = QRectF()
                intrinsic_color = False
                if fallback.native_color:
                    native_image, native_rect, intrinsic_color = (
                        _native_color_glyph_image(fallback, scale)
                    )
                fill_painter.save()
                try:
                    fill_painter.setTransform(transform, True)
                    if intrinsic_color and native_image is not None:
                        fill_painter.drawImage(
                            native_rect.topLeft(), native_image
                        )
                    else:
                        fill_painter.setPen(
                            QPen(
                                _item_space_brush(brush, transform),
                                0.0,
                            )
                        )
                        fill_painter.drawGlyphRun(QPointF(), run)
                finally:
                    fill_painter.restore()
        finally:
            fill_painter.end()

        outline_pixmap = None
        if outline_enabled:
            fill_rgba = pixmap2ndarray(fill_image, keep_alpha=True)
            if fill_rgba is None:
                raise MemoryError('unable to access pathless glyph pixels')
            outline_alpha = _dilate_fallback_alpha(
                fill_rgba[..., 3], math.ceil(outline_radius * scale)
            )

            field_image = QImage(
                pixel_width, pixel_height, QImage.Format.Format_ARGB32
            )
            if field_image.isNull():
                raise MemoryError('unable to allocate pathless outline field')
            field_image.setDevicePixelRatio(scale)
            field_image.fill(Qt.GlobalColor.transparent)
            field_painter = QPainter(field_image)
            if not field_painter.isActive():
                raise MemoryError('unable to begin pathless outline painter')
            try:
                field_painter.translate(-raster_rect.topLeft())
                field_painter.fillRect(raster_rect, outline.brush())
            finally:
                field_painter.end()
            outline_rgba = pixmap2ndarray(field_image, keep_alpha=True)
            if outline_rgba is None:
                raise MemoryError('unable to access pathless outline pixels')
            outline_rgba[..., 3] = (
                outline_rgba[..., 3].astype(np.uint16)
                * outline_alpha.astype(np.uint16)
                // 255
            ).astype(np.uint8)
            outline_pixmap = ndarray2pixmap(outline_rgba)
            if outline_pixmap is None or outline_pixmap.isNull():
                raise MemoryError('unable to allocate pathless outline result')
            outline_pixmap.setDevicePixelRatio(scale)
        painter.save()
        try:
            painter.setRenderHint(
                QPainter.RenderHint.SmoothPixmapTransform, True
            )
            if outline_pixmap is not None:
                painter.drawPixmap(
                    raster_rect.topLeft(), outline_pixmap
                )
            painter.drawImage(raster_rect.topLeft(), fill_image)
        finally:
            painter.restore()
    except (
        MemoryError,
        OverflowError,
        RuntimeError,
        ValueError,
        TypeError,
        BufferError,
        cv2.error,
    ) as error:
        _draw_direct_fallbacks(painter, fallbacks, brush)
        if failure_handler is not None:
            failure_handler(
                GlyphRasterAllocationError(str(error)), effect_pass
            )


def draw_glyph_geometry(
    painter: QPainter,
    geometry: GlyphGeometry,
    char_format: QTextCharFormat,
    failure_handler=None,
) -> None:
    """Draw vector ink, document outline, and pathless glyph fallbacks."""
    brush = _foreground_brush(char_format, painter)
    outline = char_format.textOutline()
    if geometry.paths:
        painter.save()
        try:
            if outline.style() != Qt.PenStyle.NoPen:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(outline)
                for glyph_path in geometry.paths:
                    painter.drawPath(glyph_path)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(brush)
            for glyph_path in geometry.paths:
                painter.drawPath(glyph_path)
        finally:
            painter.restore()
    _draw_fallbacks(
        painter,
        geometry.fallbacks,
        brush,
        outline,
        failure_handler,
        bool(char_format.property(GLYPH_STROKE_FORMAT_PROPERTY)),
    )


def draw_uniform_glyph_geometries(
    painter: QPainter,
    geometries: Sequence[GlyphGeometry],
    char_format: QTextCharFormat,
    failure_handler=None,
) -> None:
    """Draw same-format geometry with one painter state transition.

    Paths remain separate draw operations so overlapping glyphs cannot cancel
    each other and each glyph retains its raw font fill rule.
    """
    brush = _foreground_brush(char_format, painter)
    outline = char_format.textOutline()
    if (
        outline.style() != Qt.PenStyle.NoPen
        or any(geometry.fallbacks for geometry in geometries)
    ):
        for geometry in geometries:
            draw_glyph_geometry(
                painter, geometry, char_format, failure_handler
            )
        return
    painter.save()
    try:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(brush)
        for geometry in geometries:
            for glyph_path in geometry.paths:
                painter.drawPath(glyph_path)
    finally:
        painter.restore()


def _draw_background(
    painter: QPainter,
    rect: QRectF,
    char_format: QTextCharFormat,
) -> None:
    brush = char_format.background()
    if not rect.isEmpty() and brush.style() != Qt.BrushStyle.NoBrush:
        painter.fillRect(rect, brush)


def _draw_decorations(
    painter: QPainter,
    rect: QRectF,
    char_format: QTextCharFormat,
    orientation: QTransform,
    baseline_y: float,
) -> None:
    if rect.isEmpty():
        return
    font = char_format.font()
    underline_style = char_format.underlineStyle()
    flags = (
        underline_style
        != QTextCharFormat.UnderlineStyle.NoUnderline,
        font.overline(),
        font.strikeOut(),
    )
    if not any(flags):
        return
    metrics = QFontMetricsF(font)
    foreground = _foreground_brush(char_format, painter)

    def underline_pen() -> QPen:
        color = char_format.underlineColor()
        brush = QBrush(color) if color.isValid() else foreground
        pen = QPen(brush, max(1.0, metrics.lineWidth()))
        style_map = {
            QTextCharFormat.UnderlineStyle.DashUnderline: Qt.PenStyle.DashLine,
            QTextCharFormat.UnderlineStyle.DotLine: Qt.PenStyle.DotLine,
            QTextCharFormat.UnderlineStyle.DashDotLine: Qt.PenStyle.DashDotLine,
            QTextCharFormat.UnderlineStyle.DashDotDotLine: Qt.PenStyle.DashDotDotLine,
        }
        pen.setStyle(
            style_map.get(underline_style, Qt.PenStyle.SolidLine)
        )
        return pen

    painter.save()
    try:
        painter.setTransform(orientation, True)
        if flags[0]:
            painter.setPen(underline_pen())
            underline_y = baseline_y + metrics.underlinePos()
            if (
                underline_style
                == QTextCharFormat.UnderlineStyle.WaveUnderline
            ):
                wave = QPainterPath(QPointF(rect.left(), underline_y))
                x = rect.left()
                phase = 1.0
                while x < rect.right():
                    x = min(rect.right(), x + 2.0)
                    wave.lineTo(x, underline_y + phase)
                    phase = -phase
                painter.drawPath(wave)
            else:
                painter.drawLine(
                    QPointF(rect.left(), underline_y),
                    QPointF(rect.right(), underline_y),
                )
        painter.setPen(QPen(foreground, max(1.0, metrics.lineWidth())))
        if flags[1]:
            overline_y = baseline_y - metrics.ascent()
            painter.drawLine(
                QPointF(rect.left(), overline_y),
                QPointF(rect.right(), overline_y),
            )
        if flags[2]:
            strike_y = baseline_y - metrics.strikeOutPos()
            painter.drawLine(
                QPointF(rect.left(), strike_y),
                QPointF(rect.right(), strike_y),
            )
    finally:
        painter.restore()


def _geometry_cache_key(
    namespace,
    start: int,
    length: int,
    offset: QPointF,
    orientation: QTransform,
    angle: float,
):
    return (
        namespace,
        start,
        length,
        offset.x(),
        offset.y(),
        orientation.m11(),
        orientation.m12(),
        orientation.m21(),
        orientation.m22(),
        orientation.dx(),
        orientation.dy(),
        angle,
    )


def _store_geometry(cache, key, geometry) -> None:
    store = getattr(cache, 'store', None)
    if store is not None:
        store(key, geometry)
        return
    while len(cache) >= GLYPH_GEOMETRY_CACHE_MAX_ENTRIES:
        cache.pop(next(iter(cache)))
    cache[key] = geometry


def draw_slanted_line(
    painter: QPainter,
    block: QTextBlock,
    line: QTextLine,
    offset: QPointF,
    orientation: QTransform,
    angle: float,
    context: QAbstractTextDocumentLayout.PaintContext,
    failure_handler=None,
    persistent_geometry_cache=None,
    cache_namespace=None,
) -> None:
    """Paint one already-laid-out line without changing logical geometry."""
    layout = block.layout()
    additional_formats = tuple(layout.formats())
    normal_spans = resolve_paint_spans(block, line, additional_formats)
    baseline_y = line.y() + line.ascent() + offset.y()
    geometry_cache = {}

    def span_geometry(span: PaintSpan) -> GlyphGeometry:
        key = (span.start, span.length)
        geometry = geometry_cache.get(key)
        if geometry is None:
            persistent_key = _geometry_cache_key(
                cache_namespace,
                span.start,
                span.length,
                offset,
                orientation,
                angle,
            )
            if persistent_geometry_cache is not None:
                geometry = persistent_geometry_cache.get(persistent_key)
            if geometry is None:
                geometry = glyph_geometry(
                    line,
                    span.start,
                    span.length,
                    offset,
                    orientation,
                    angle,
                )
                if persistent_geometry_cache is not None:
                    _store_geometry(
                        persistent_geometry_cache, persistent_key, geometry
                    )
            geometry_cache[key] = geometry
        return geometry

    effect_selections = tuple(
        selection
        for selection in context.selections
        if bool(
            selection.format.property(GLYPH_STROKE_FORMAT_PROPERTY)
        )
    )
    if effect_selections and len(effect_selections) == len(context.selections):
        # App stroke selections are paint instructions, not interactive
        # selection cells. Do not clip their outline overhang at artificial
        # rich-fragment boundaries and do not paint the normal fill beneath.
        for selection in effect_selections:
            selection_range = _selection_range(block, selection, line)
            if selection_range is None:
                continue
            for span in resolve_paint_spans(
                block, line, additional_formats, selection
            ):
                if not (
                    selection_range[0] <= span.start
                    and span.start < selection_range[1]
                ):
                    continue
                draw_glyph_geometry(
                    painter,
                    span_geometry(span),
                    span.char_format,
                    failure_handler,
                )
        return

    for span in normal_spans:
        rect = logical_span_rect(line, span.start, span.length, offset, orientation)
        _draw_background(painter, rect, span.char_format)

    selection_spans = []
    for selection in context.selections:
        spans = resolve_paint_spans(block, line, additional_formats, selection)
        selection_range = _selection_range(block, selection, line)
        if selection_range is None:
            continue
        full_width = (
            not selection.cursor.hasSelection()
            and selection.format.hasProperty(
                QTextFormat.FullWidthSelection
            )
        )
        if full_width:
            line_rect = QRectF(line.rect())
            line_rect.translate(offset)
            _draw_background(
                painter,
                _mapped_rect(line_rect, orientation),
                selection.format,
            )
        for span in spans:
            if not (
                selection_range[0] <= span.start
                and span.start < selection_range[1]
            ):
                continue
            rect = logical_span_rect(
                line, span.start, span.length, offset, orientation
            )
            if not full_width:
                _draw_background(painter, rect, span.char_format)
            selection_spans.append((span, rect))

    # Paint normal ink once. Selection foreground is a second, logically
    # clipped pass so ligature overhang outside the selection remains normal.
    for span in normal_spans:
        draw_glyph_geometry(
            painter,
            span_geometry(span),
            span.char_format,
            failure_handler,
        )
        _draw_decorations(
            painter,
            _logical_span_base_rect(
                line, span.start, span.length, offset
            ),
            span.char_format,
            orientation,
            baseline_y,
        )

    for span, rect in selection_spans:
        painter.save()
        try:
            painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
            draw_glyph_geometry(
                painter,
                span_geometry(span),
                span.char_format,
                failure_handler,
            )
            _draw_decorations(
                painter,
                _logical_span_base_rect(
                    line, span.start, span.length, offset
                ),
                span.char_format,
                orientation,
                baseline_y,
            )
        finally:
            painter.restore()


def draw_slanted_glyph_mask(
    painter: QPainter,
    line: QTextLine,
    start: int,
    length: int,
    offset: QPointF,
    orientation: QTransform,
    angle: float,
    failure_handler=None,
    persistent_geometry_cache=None,
    cache_namespace=None,
) -> None:
    """Draw only slanted glyph alpha for stroke/shadow mask construction."""
    char_format = QTextCharFormat()
    char_format.setForeground(QColor(Qt.GlobalColor.white))
    key = _geometry_cache_key(
        cache_namespace, start, length, offset, orientation, angle
    )
    geometry = (
        None
        if persistent_geometry_cache is None
        else persistent_geometry_cache.get(key)
    )
    if geometry is None:
        geometry = glyph_geometry(line, start, length, offset, orientation, angle)
        if persistent_geometry_cache is not None:
            _store_geometry(persistent_geometry_cache, key, geometry)
    draw_glyph_geometry(
        painter, geometry, char_format, failure_handler
    )


def slanted_line_geometry(
    line: QTextLine,
    offset: QPointF,
    orientation: QTransform,
    angle: float,
    persistent_geometry_cache=None,
    cache_namespace=None,
) -> GlyphGeometry:
    """Return geometry shared by exact ink measurement and painting."""
    start = line.textStart()
    length = line.textLength()
    key = _geometry_cache_key(
        cache_namespace, start, length, offset, orientation, angle
    )
    geometry = (
        None
        if persistent_geometry_cache is None
        else persistent_geometry_cache.get(key)
    )
    if geometry is None:
        geometry = glyph_geometry(
            line, start, length, offset, orientation, angle
        )
        if persistent_geometry_cache is not None:
            _store_geometry(persistent_geometry_cache, key, geometry)
    return geometry
