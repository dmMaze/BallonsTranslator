from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from ..rendering.morphology import dilate_alpha_disc


# Blur is radius limited rather than area limited: OpenCV blurs separably, so
# its Gaussian costs O(radius) per pixel. Effect radii are a fraction of the
# font size, so a 200px font already reaches a 2000px radius at the slider
# limit and the exact kernel stalls the UI for minutes. Past the threshold
# here the blur carries the same geometry for a linear cost, and dilation does
# the same through the shared alpha morphology.
EXACT_BLUR_RADIUS = 24
# Deviation retained inside a reduced-resolution buffer, far above the point
# where the kernel is cheap so the buffer stays small without the resampling
# itself becoming the bottleneck.
COARSE_BLUR_SIGMA = 12.0


def _blur_sigma(radius: int) -> float:
    """Return the deviation :func:`_blur` applies at ``radius``.

    >>> _blur_sigma(0)
    0.0
    >>> _blur_sigma(4)
    1.5
    """
    return 0.0 if radius <= 0 else (radius * 2 + 1) / 6


def _coarse_blur(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur inside a buffer small enough for the kernel to be cheap.

    The result is band limited by ``sigma``, so it survives the round trip
    through a buffer subsampled well below it. ``resize`` rounds each axis
    independently, so the deviation is rescaled by the ratio it really used.
    """
    height, width = mask.shape
    factor = max(2, int(sigma // COARSE_BLUR_SIGMA))
    small_height = max(1, height // factor)
    small_width = max(1, width // factor)
    small = cv2.resize(
        mask,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(
        small,
        (0, 0),
        sigma * small_width / width,
        sigmaY=sigma * small_height / height,
        borderType=cv2.BORDER_CONSTANT,
    )
    return cv2.resize(
        small, (width, height), interpolation=cv2.INTER_LINEAR
    )


def _blur(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    if radius <= EXACT_BLUR_RADIUS:
        ksize = radius * 2 + 1
        return cv2.GaussianBlur(
            mask,
            (ksize, ksize),
            ksize / 6,
            borderType=cv2.BORDER_CONSTANT,
        )
    return _coarse_blur(mask, _blur_sigma(radius))


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


def _alpha_intersection(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    product = first.astype(np.uint16)
    np.multiply(product, second.astype(np.uint16), out=product)
    product += 127
    product //= 255
    return product.astype(np.uint8)


def render_glow_alpha(
    source_alpha: np.ndarray,
    glow_type: str,
    size_radius: int,
    spread_radius: int,
) -> np.ndarray:
    """Generate one Outer or Inner Glow coverage mask.

    >>> source = np.zeros((5, 5), dtype=np.uint8)
    >>> source[2, 2] = 255
    >>> int(render_glow_alpha(source, 'outer', 0, 1)[2, 1])
    255
    >>> int(render_glow_alpha(source, 'outer', 0, 1)[2, 2])
    0
    """
    if glow_type == 'outer':
        expanded = _blur(
            dilate_alpha_disc(source_alpha, spread_radius), size_radius
        )
        return _alpha_intersection(expanded, 255 - source_alpha)
    if glow_type == 'inner':
        edge = 255 - _blur(source_alpha, size_radius)
        return _alpha_intersection(
            dilate_alpha_disc(
                edge, spread_radius, _blur_sigma(size_radius)
            ),
            source_alpha,
        )
    raise ValueError('unsupported glow type')


def render_shadow_alpha(
    source_alpha: np.ndarray,
    shadow_type: str,
    opacity: float,
    offset: Tuple[float, float],
    blur_radius: int,
    spread_radius: int,
    canonical_alpha: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compile one typed Shadow into a paint-independent alpha layer.

    The caller supplies pixel-space geometry, so this helper is independent of
    Qt device scale and can be tested without a painter.

    >>> alpha = np.zeros((3, 4), dtype=np.uint8)
    >>> alpha[1, 1] = 255
    >>> int(render_shadow_alpha(
    ...     alpha, 'drop', 1, (1, 0), 0, 0, alpha
    ... )[1, 2])
    255
    >>> int(render_shadow_alpha(
    ...     alpha, 'drop', 1, (1, 0), 0, 0, alpha
    ... )[1, 1])
    0
    """
    if shadow_type == 'long':
        mask = _long_shadow_alpha(
            source_alpha,
            (int(round(offset[0])), int(round(offset[1]))),
        )
    elif shadow_type == 'drop':
        mask = _translate(
            _blur(dilate_alpha_disc(source_alpha, spread_radius), blur_radius),
            offset,
        )
    elif shadow_type == 'inner':
        shifted = _translate(_blur(source_alpha, blur_radius), offset)
        mask = cv2.subtract(255, shifted)
        mask = dilate_alpha_disc(
            mask, spread_radius, _blur_sigma(blur_radius)
        )
        mask = _alpha_intersection(mask, source_alpha)
    else:
        raise ValueError('unsupported shadow type')
    if shadow_type in {'drop', 'long'} and canonical_alpha is not None:
        # These are exterior layers. Keeping their source footprint made the
        # base-first compositor tint the canonical face that used to cover it.
        # Stroke is deliberately not clipped: global card order decides
        # whether a higher Shadow can cover a lower Stroke.
        mask = _alpha_intersection(mask, 255 - canonical_alpha)
    if opacity != 1.0:
        mask = np.clip(
            mask.astype(np.float32) * opacity, 0, 255
        ).astype(np.uint8)
    return mask
