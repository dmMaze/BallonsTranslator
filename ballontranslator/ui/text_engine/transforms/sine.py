"""Invertible sine-wave mapping for completed text surfaces."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath

from ballontranslator.utils.fontformat import SineTextTransform


class SineMapper:
    """Apply the x-axis then y-axis sine wave in logical-box units.

    Each shear leaves its input axis unchanged, so reversing the two steps is
    exact even when high frequencies strongly distort the rendered surface.

    >>> mapper = SineMapper(
    ...     QRectF(0, 0, 100, 50), QRectF(0, 0, 100, 50),
    ...     SineTextTransform(),
    ... )
    >>> point = QPointF(25, 10)
    >>> restored = mapper.inverse_point(mapper.forward_point(point))
    >>> (round(restored.x(), 6), round(restored.y(), 6))
    (25.0, 10.0)
    """

    def __init__(
        self,
        logical_rect: QRectF,
        source_rect: QRectF,
        transform: SineTextTransform,
    ) -> None:
        if logical_rect.width() <= 0.0 or logical_rect.height() <= 0.0:
            raise ValueError('sine rectangle must have positive dimensions')
        self.logical_rect = QRectF(logical_rect)
        self.source_rect = QRectF(source_rect)
        self.transform = transform
        self.horizontal_amplitude = (
            self.transform.amplitude_y * self.logical_rect.width()
            if self.transform.frequency_y
            else 0.0
        )
        self.vertical_amplitude = (
            self.transform.amplitude_x * self.logical_rect.height()
            if self.transform.frequency_x
            else 0.0
        )

    @property
    def geometry_key(self) -> tuple:
        rect = self.logical_rect
        source = self.source_rect
        return (
            type(self),
            self.transform,
            rect.x(), rect.y(), rect.width(), rect.height(),
            source.x(), source.y(), source.width(), source.height(),
        )

    def _vertical_offset(
        self, source_x: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if self.transform.frequency_x == 0 or self.vertical_amplitude == 0.0:
            return (
                np.zeros_like(source_x)
                if isinstance(source_x, np.ndarray)
                else 0.0
            )
        position = (
            (source_x - self.logical_rect.left()) / self.logical_rect.width()
        )
        return self.vertical_amplitude * np.sin(
            math.pi * self.transform.frequency_x * position
            + math.tau * self.transform.phase_x
        )

    def _horizontal_offset(
        self, source_y: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        if self.transform.frequency_y == 0 or self.horizontal_amplitude == 0.0:
            return (
                np.zeros_like(source_y)
                if isinstance(source_y, np.ndarray)
                else 0.0
            )
        position = (
            (source_y - self.logical_rect.top()) / self.logical_rect.height()
        )
        return self.horizontal_amplitude * np.sin(
            math.pi * self.transform.frequency_y * position
            + math.tau * self.transform.phase_y
        )

    def forward_point(self, source: QPointF) -> QPointF:
        source = QPointF(source)
        mapped_y = source.y() + float(self._vertical_offset(source.x()))
        mapped_x = source.x() + float(self._horizontal_offset(mapped_y))
        return QPointF(mapped_x, mapped_y)

    def forward_arrays(
        self, source_x: np.ndarray, source_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        source_x = np.asarray(source_x, dtype=np.float64)
        source_y = np.asarray(source_y, dtype=np.float64)
        mapped_y = source_y + self._vertical_offset(source_x)
        mapped_x = source_x + self._horizontal_offset(mapped_y)
        return mapped_x, mapped_y

    def inverse_point(
        self,
        visual: QPointF,
        previous_source: Optional[QPointF] = None,
        *,
        extrapolate: bool = False,
    ) -> QPointF:
        visual = QPointF(visual)
        source_x = visual.x() - float(self._horizontal_offset(visual.y()))
        source_y = visual.y() - float(self._vertical_offset(source_x))
        return QPointF(source_x, source_y)

    def inverse_arrays(
        self,
        visual_x: np.ndarray,
        visual_y: np.ndarray,
        *,
        return_valid: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        visual_x = np.asarray(visual_x, dtype=np.float64)
        visual_y = np.asarray(visual_y, dtype=np.float64)
        source_x = visual_x - self._horizontal_offset(visual_y)
        source_y = visual_y - self._vertical_offset(source_x)
        if return_valid:
            valid = np.isfinite(source_x) & np.isfinite(source_y)
            return source_x, source_y, valid
        return source_x, source_y

    def visual_bounds(
        self, source_rect: Optional[QRectF] = None
    ) -> QRectF:
        rect = QRectF(self.source_rect if source_rect is None else source_rect)
        return rect.adjusted(
            -self.horizontal_amplitude,
            -self.vertical_amplitude,
            self.horizontal_amplitude,
            self.vertical_amplitude,
        )

    def map_rect_path(self, rect: QRectF) -> QPainterPath:
        """Map a rectangle without aliasing integer wave counts to a box."""
        rect = QRectF(rect)
        width_ratio = abs(rect.width()) / self.logical_rect.width()
        height_ratio = abs(rect.height()) / self.logical_rect.height()
        horizontal_span = self.transform.frequency_x * width_ratio
        vertical_variation = min(
            2.0 * self.transform.amplitude_x,
            self.transform.amplitude_x
            * math.pi
            * self.transform.frequency_x
            * width_ratio,
        )
        vertical_span = self.transform.frequency_y * (
            height_ratio + vertical_variation
        )
        segments = min(
            2048,
            max(
                8,
                math.ceil(8 * (horizontal_span + vertical_span)),
            ),
        )
        corners = (
            rect.topLeft(),
            rect.topRight(),
            rect.bottomRight(),
            rect.bottomLeft(),
        )
        path = QPainterPath()
        started = False
        for edge, start in enumerate(corners):
            end = corners[(edge + 1) % len(corners)]
            for index in range(segments + 1):
                if edge and index == 0:
                    continue
                ratio = index / segments
                point = self.forward_point(
                    start * (1.0 - ratio) + end * ratio
                )
                if not started:
                    path.moveTo(point)
                    started = True
                else:
                    path.lineTo(point)
        path.closeSubpath()
        return path
