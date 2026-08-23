from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (diameter, diameter)
    )
    return cv2.dilate(mask, kernel)


def _blur(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    ksize = radius * 2 + 1
    return cv2.GaussianBlur(
        mask,
        (ksize, ksize),
        ksize / 6,
        borderType=cv2.BORDER_CONSTANT,
    )


def _translate(mask: np.ndarray, offset: Tuple[float, float]) -> np.ndarray:
    xoffset, yoffset = offset
    return cv2.warpAffine(
        mask,
        np.float32(((1, 0, xoffset), (0, 1, yoffset))),
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _long_shadow_alpha(
    source_alpha: np.ndarray, offset: Tuple[int, int]
) -> np.ndarray:
    """Sweep a connected silhouette through every pixel to the terminal.

    >>> source = np.zeros((5, 5), dtype=np.uint8)
    >>> source[1, 1] = 255
    >>> np.argwhere(_long_shadow_alpha(source, (2, 2))).tolist()
    [[1, 1], [2, 2], [3, 3]]
    """
    xoffset, yoffset = offset
    if xoffset == 0 and yoffset == 0:
        return source_alpha.copy()
    width = abs(xoffset) + 1
    height = abs(yoffset) + 1
    start = (max(0, -xoffset), max(0, -yoffset))
    endpoint = (start[0] + xoffset, start[1] + yoffset)
    kernel = np.zeros((height, width), dtype=np.uint8)
    cv2.line(kernel, start, endpoint, 255, 1, cv2.LINE_8)
    return cv2.dilate(source_alpha, kernel, anchor=endpoint)


def render_shadow_rgba(
    source_alpha: np.ndarray,
    shadow_type: str,
    color: Tuple[int, int, int],
    opacity: float,
    offset: Tuple[float, float],
    blur_radius: int,
    spread_radius: int,
) -> np.ndarray:
    """Compile one typed Shadow into a transparent RGBA layer.

    The caller supplies pixel-space geometry, so this helper is independent of
    Qt device scale and can be tested without a painter.

    >>> alpha = np.zeros((3, 4), dtype=np.uint8)
    >>> alpha[1, 1] = 255
    >>> render_shadow_rgba(alpha, 'drop', (1, 2, 3), 1, (1, 0), 0, 0)[1, 2].tolist()
    [1, 2, 3, 255]
    """
    if shadow_type == 'long':
        mask = _long_shadow_alpha(
            source_alpha,
            (int(round(offset[0])), int(round(offset[1]))),
        )
    elif shadow_type == 'drop':
        mask = _translate(
            _blur(_dilate(source_alpha, spread_radius), blur_radius),
            offset,
        )
    elif shadow_type == 'inner':
        shifted = _translate(_blur(source_alpha, blur_radius), offset)
        mask = cv2.subtract(255, shifted)
        mask = _dilate(mask, spread_radius)
        product = mask.astype(np.uint16)
        np.multiply(product, source_alpha, out=product)
        product += 127
        product //= 255
        mask = product.astype(np.uint8)
    else:
        raise ValueError('unsupported shadow type')
    if opacity != 1.0:
        mask = np.clip(
            mask.astype(np.float32) * opacity, 0, 255
        ).astype(np.uint8)
    result = np.empty(source_alpha.shape + (4,), dtype=np.uint8)
    result[..., :3] = np.asarray(color, dtype=np.uint8)
    result[..., 3] = mask
    return result
