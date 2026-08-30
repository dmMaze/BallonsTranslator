"""Small premultiplied-alpha helpers for finite Gaussian text filters."""

from __future__ import annotations

import cv2
import numpy as np


def premultiply_rgba_float32(rgba: np.ndarray) -> np.ndarray:
    """Convert straight RGBA8 to normalized premultiplied float32."""
    premultiplied = rgba.astype(np.float32) / np.float32(255.0)
    premultiplied[:, :, :3] *= premultiplied[:, :, 3:4]
    return premultiplied


def unpremultiply_rgba_float32(premultiplied: np.ndarray) -> np.ndarray:
    """Consume normalized premultiplied float32 into straight RGBA8."""
    np.clip(premultiplied, 0.0, 1.0, out=premultiplied)
    alpha = premultiplied[:, :, 3]
    np.divide(
        premultiplied[:, :, :3],
        alpha[:, :, np.newaxis],
        out=premultiplied[:, :, :3],
        where=alpha[:, :, np.newaxis] > 0.0,
    )
    premultiplied[:, :, :3][alpha == 0.0] = 0.0
    np.clip(
        premultiplied[:, :, :3], 0.0, 1.0,
        out=premultiplied[:, :, :3],
    )
    premultiplied *= np.float32(255.0)
    np.rint(premultiplied, out=premultiplied)
    rgba = np.ascontiguousarray(premultiplied.astype(np.uint8))
    rgba[:, :, :3][rgba[:, :, 3] == 0] = 0
    return rgba


def finite_gaussian(
    premultiplied: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Blur through an exact finite radius with transparent borders.

    >>> source = np.zeros((3, 3, 4), dtype=np.float32)
    >>> source[1, 1] = 1.0
    >>> blurred = finite_gaussian(source, 1)
    >>> bool(blurred.shape == source.shape and 0.0 < blurred[0, 0, 3] < 1.0)
    True
    """
    radius = max(0, int(radius))
    if radius == 0 or premultiplied.size == 0:
        return premultiplied
    diameter = radius * 2 + 1
    cv2.GaussianBlur(
        premultiplied,
        (diameter, diameter),
        sigmaX=max(0.5, radius / 2.0),
        sigmaY=max(0.5, radius / 2.0),
        borderType=cv2.BORDER_CONSTANT,
        dst=premultiplied,
    )
    return premultiplied
