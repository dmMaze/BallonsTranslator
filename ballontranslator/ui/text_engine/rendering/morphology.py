"""Radius independent alpha morphology shared by text rendering."""

from __future__ import annotations

import cv2
import numpy as np


# OpenCV can only decompose rectangular structuring elements, so dilating by
# an elliptical one costs a kernel area per pixel. Text radii are fractions of
# the font size, so a 300px font reaches a 600px radius inside the slider
# range and the exact kernel needs about a minute per surface. Past this
# threshold the growth is computed from geometry instead of from a kernel.
EXACT_DILATE_RADIUS = 16
# Radius retained inside a reduced-resolution buffer. Well past the point
# where a kernel is cheap, so the buffer stays small without the resampling
# itself becoming the bottleneck.
COARSE_DILATE_RADIUS = 16.0
# A source blurred by less than this behaves as a hard coverage edge, which is
# what the distance field reproduces.
HARD_EDGE_SIGMA = 2.0
# OpenCV's discrete ellipse covers marginally more than the disc of the same
# radius. Biasing the distance field by this much matches the two footprints.
DISC_EDGE_BIAS = 0.75


def disc_kernel(radius: int) -> np.ndarray:
    diameter = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (diameter, diameter)
    )


def _hard_dilate(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Grow a hard coverage mask through its exterior distance field.

    Dilating by a disc covers exactly the pixels within ``radius`` of the
    mask, which the distance field answers in one linear pass instead of one
    kernel area per pixel. Clipping the field keeps the boundary antialiased,
    and against a real distance it lands on a smoother edge than a kernel
    quantised to whole pixels can reach.
    """
    exterior = np.where(alpha >= 128, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(
        exterior, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    grown = np.clip(radius + DISC_EDGE_BIAS - distance, 0.0, 1.0)
    grown *= 255.0
    return np.maximum(alpha, grown.astype(np.uint8))


def _coarse_dilate(
    alpha: np.ndarray, radius: int, source_sigma: float
) -> np.ndarray:
    """Grow an already blurred mask inside a reduced-resolution buffer.

    A source blurred by ``source_sigma`` holds no detail finer than that, so
    subsampling below it is lossless. Max pooling is the downsample that
    matches a dilation: it never drops coverage the full kernel would find.
    """
    factor = int(min(radius // COARSE_DILATE_RADIUS, source_sigma))
    if factor < 2:
        return cv2.dilate(alpha, disc_kernel(radius))
    height, width = alpha.shape
    pooled = cv2.dilate(
        alpha, cv2.getStructuringElement(cv2.MORPH_RECT, (factor, factor))
    )
    small = cv2.resize(
        pooled,
        (max(1, width // factor), max(1, height // factor)),
        interpolation=cv2.INTER_NEAREST,
    )
    # Pooling already carried one cell of coverage, so the reduced kernel is
    # one step short of the full radius.
    small = cv2.dilate(
        small, disc_kernel(max(1, int(round(radius / factor)) - 1))
    )
    return np.maximum(
        alpha,
        cv2.resize(
            small, (width, height), interpolation=cv2.INTER_LINEAR
        ),
    )


def dilate_alpha_disc(
    alpha: np.ndarray, radius: int, source_sigma: float = 0.0
) -> np.ndarray:
    """Grow ``alpha`` by a disc of ``radius``.

    ``source_sigma`` is the deviation already blurred into ``alpha``. It only
    selects how the growth is computed; callers pass what their pipeline
    applied so a large radius never falls back to the quadratic kernel.

    >>> source = np.zeros((5, 5), dtype=np.uint8)
    >>> source[2, 2] = 255
    >>> int(dilate_alpha_disc(source, 1)[2, 1])
    255
    >>> int(dilate_alpha_disc(source, 0)[2, 1])
    0
    """
    if radius <= 0:
        return alpha
    if radius <= EXACT_DILATE_RADIUS:
        return cv2.dilate(alpha, disc_kernel(radius))
    if source_sigma <= HARD_EDGE_SIGMA:
        return _hard_dilate(alpha, radius)
    return _coarse_dilate(alpha, radius, source_sigma)
