"""Alpha-preserving circular dilation without an area-sized kernel."""

import math

import cv2
import numpy as np


def dilate_alpha_disc(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Apply OpenCV's discrete disc while preserving every alpha level.

    Large discs are unions of horizontal spans. A rectangular dilation is
    separable, so compute each distinct span once and shift its rows into the
    result instead of scanning a two-dimensional kernel at every pixel.

    >>> alpha = np.zeros((5, 5), dtype=np.uint8)
    >>> alpha[2, 2] = 100
    >>> int(dilate_alpha_disc(alpha, 1)[2, 1])
    100
    """
    if radius <= 0:
        return alpha
    if radius <= 16:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(alpha, kernel)
    height, width = alpha.shape
    result = np.zeros_like(alpha)
    x, y, ink_width, ink_height = cv2.boundingRect(alpha)
    if ink_width == 0 or ink_height == 0:
        return result
    left, top = max(0, x - radius), max(0, y - radius)
    right = min(width, x + ink_width + radius)
    bottom = min(height, y + ink_height + radius)
    # 패딩의 빈 영역은 제외하고, RGBA의 비연속 알파 뷰는 한 번만 복사한다.
    source = np.ascontiguousarray(alpha[top:bottom, left:right])
    grown = result[top:bottom, left:right]
    height, width = source.shape
    previous_span = -1
    # 중심에서 바깥으로 이동하면 같은 폭이 연속되므로 한 행 필터만 유지한다.
    for offset in range(min(radius, height - 1) + 1):
        half_span = min(
            width - 1,
            round(math.sqrt(max(0, radius * radius - offset * offset))),
        )
        if half_span != previous_span:
            row = cv2.dilate(source, np.ones((1, 2 * half_span + 1), np.uint8))
            previous_span = half_span
        if offset == 0:
            np.maximum(grown, row, out=grown)
        else:
            np.maximum(grown[offset:], row[:-offset], out=grown[offset:])
            np.maximum(grown[:-offset], row[offset:], out=grown[:-offset])
    return result
