"""Alpha-preserving circular dilation without an area-sized kernel."""

import math

import cv2
import numpy as np


def dilate_alpha_disc(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Apply OpenCV's discrete disc while preserving every alpha level.

    Large discs are unions of horizontal spans. Grow each span incrementally
    on ink-bearing rows and shift it into the result, preserving the discrete
    kernel and every alpha level without filtering transparent padding.

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
    left = max(0, x - radius)
    right = min(width, x + ink_width + radius)
    # 수평 필터는 잉크가 있는 행만 계산한다. 세로 확장은 아래 행 이동이 맡는다.
    source = np.ascontiguousarray(alpha[y:y + ink_height, left:right])
    width = source.shape[1]
    previous_span = 0
    row = source
    # 바깥에서 중심으로 폭을 늘리면 이전 최대 필터에 증가분만 적용해도 같다.
    for offset in range(min(radius, height - 1), -1, -1):
        half_span = min(
            width - 1,
            round(math.sqrt(max(0, radius * radius - offset * offset))),
        )
        if half_span != previous_span:
            increase = half_span - previous_span
            row = cv2.dilate(row, np.ones((1, 2 * increase + 1), np.uint8))
            previous_span = half_span
        for shift in ((0,) if offset == 0 else (-offset, offset)):
            top = max(0, y + shift)
            bottom = min(height, y + ink_height + shift)
            if top >= bottom:
                continue
            source_top = top - y - shift
            grown = result[top:bottom, left:right]
            np.maximum(
                grown, row[source_top:source_top + bottom - top], out=grown
            )
    return result
