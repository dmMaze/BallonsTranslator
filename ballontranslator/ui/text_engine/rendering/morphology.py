"""Coverage-preserving dilation without an area-sized kernel."""

import math

import cv2
import numpy as np

from ballontranslator.utils.rgba import (
    premultiply_rgba_in_place,
    unpremultiply_rgba_in_place,
)


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
    return dilate_ellipse(alpha, radius, radius)


def dilate_ellipse(mask: np.ndarray, x_radius: int, y_radius: int) -> np.ndarray:
    """Max-filter a grayscale mask with OpenCV's discrete ellipse.

    Masks may carry ordered RGBA palette indices so expansion can retain one
    source color instead of mixing unrelated channels.

    >>> mask = np.zeros((3, 5), dtype=np.uint8)
    >>> mask[1, 2] = 80
    >>> dilate_ellipse(mask, 1, 0)[1].tolist()
    [0, 80, 80, 80, 0]
    """
    if x_radius <= 0 and y_radius <= 0:
        return mask
    if (x_radius <= 16 and y_radius <= 16) or min(x_radius, y_radius) == 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE if min(x_radius, y_radius) > 0 else cv2.MORPH_RECT,
            (2 * x_radius + 1, 2 * y_radius + 1),
        )
        return cv2.dilate(mask, kernel)
    height, width = mask.shape
    result = np.zeros_like(mask)
    support = mask if mask.dtype == np.uint8 else (mask != 0).astype(np.uint8)
    x, y, ink_width, ink_height = cv2.boundingRect(support)
    if ink_width == 0 or ink_height == 0:
        return result
    left = max(0, x - x_radius)
    right = min(width, x + ink_width + x_radius)
    source = np.ascontiguousarray(mask[y:y + ink_height, left:right])
    width = source.shape[1]
    previous_span = 0
    row = source
    # Successive 1D max filters compose by adding their radii.
    for offset in range(min(y_radius, height - 1), -1, -1):
        half_span = min(
            width - 1,
            round(
                math.sqrt(max(0, y_radius * y_radius - offset * offset))
                * (x_radius / y_radius)
            ),
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


def dilate_rgba(
    rgba: np.ndarray, x_radius: float, y_radius: float, *, ellipse: bool
) -> np.ndarray:
    """Expand coverage while retaining its source color and opacity.

    Max coverage selects a complete RGBA pixel; channel-wise RGB maxima would
    invent colors where glyphs meet. A fixed color order breaks coverage ties
    consistently across independently captured tiles.
    Fractional radii interpolate premultiplied pixels between adjacent kernels.

    >>> rgba = np.zeros((3, 5, 4), dtype=np.uint8)
    >>> rgba[1, 2] = (200, 100, 50, 80)
    >>> int(dilate_rgba(rgba, 1, 0, ellipse=False)[..., 3].max())
    80
    """
    if (x_radius <= 0 and y_radius <= 0) or not np.any(rgba[..., 3]):
        return rgba
    output = np.zeros_like(rgba)
    x, y, width, height = cv2.boundingRect(rgba[..., 3])
    rx, ry = math.ceil(x_radius), math.ceil(y_radius)
    left, top = max(0, x - rx), max(0, y - ry)
    right = min(rgba.shape[1], x + width + rx)
    bottom = min(rgba.shape[0], y + height + ry)
    rgba = rgba[top:bottom, left:right]
    alpha = rgba[..., 3]
    ink_y, ink_x = y - top, x - left
    ink = rgba[ink_y:ink_y + height, ink_x:ink_x + width]
    ink_alpha = ink[..., 3]
    x_floor, y_floor = math.floor(x_radius), math.floor(y_radius)
    x_fraction, y_fraction = x_radius - x_floor, y_radius - y_floor
    fractional = bool(x_fraction or y_fraction)
    color = ink[np.unravel_index(ink_alpha.argmax(), ink_alpha.shape)][:3]
    # Unpremultiplication perturbs straight RGB at low coverage; allow at most
    # one premultiplied channel unit when recognizing constant-color ink.
    monochrome = not np.any(ink[..., :3])
    if not monochrome:
        difference = np.abs(ink[..., :3].astype(np.int32) - color)
        monochrome = bool(np.all(
            difference * ink_alpha[..., None].astype(np.int32) <= 255
        ))
    if monochrome:
        scores = alpha
        result = np.zeros(alpha.shape, dtype=np.float32)
    else:
        # RGBA8 sorted as little-endian uint32 orders alpha first, then color.
        # Compact ranks keep OpenCV's optimized max filter available for rich
        # color. Even the largest allowed palette fits exactly in float32.
        palette, indices = np.unique(
            np.ascontiguousarray(ink).view('<u4'), return_inverse=True
        )
        score_dtype = (
            np.uint8 if palette.size < 256
            else np.uint16 if palette.size < 65536 else np.float32
        )
        ink_scores = (indices + 1).astype(score_dtype).reshape(ink_alpha.shape)
        ink_scores[ink_alpha == 0] = 0
        scores = np.zeros(alpha.shape, dtype=score_dtype)
        scores[ink_y:ink_y + height, ink_x:ink_x + width] = ink_scores
        pixels = np.zeros((palette.size + 1, 4), dtype=np.uint8)
        pixels[1:] = palette.view(np.uint8).reshape(-1, 4)
        if fractional:
            premultiply_rgba_in_place(pixels)
            result = np.zeros(rgba.shape, dtype=np.float32)
    for rx, wx in ((x_floor, 1 - x_fraction), (x_floor + 1, x_fraction)):
        for ry, wy in ((y_floor, 1 - y_fraction), (y_floor + 1, y_fraction)):
            weight = np.float32(wx * wy)
            if weight <= 0:
                continue
            if ellipse:
                grown = dilate_ellipse(scores, rx, ry)
            else:
                grown = cv2.dilate(
                    scores, np.ones((2 * ry + 1, 2 * rx + 1), np.uint8)
                )
            if monochrome:
                result += grown * weight
            else:
                indices = grown.astype(np.intp)
                if not fractional:
                    output[top:bottom, left:right] = pixels[indices]
                    return output
                result += pixels[indices] * weight
    if monochrome:
        expanded = np.empty(rgba.shape, dtype=np.uint8)
        expanded[..., :3] = color
        expanded[..., 3] = np.rint(result).astype(np.uint8)
        expanded[expanded[..., 3] == 0] = 0
    else:
        expanded = unpremultiply_rgba_in_place(np.rint(result).astype(np.uint8))
    output[top:bottom, left:right] = expanded
    return output
