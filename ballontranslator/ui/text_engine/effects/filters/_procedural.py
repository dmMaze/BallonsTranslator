"""Shared coordinate-stable primitives for built-in text filters."""

from __future__ import annotations

import cv2
import numpy as np


def coordinate_noise(
    height: int,
    width: int,
    origin_x: int,
    origin_y: int,
    seed: int,
    channel: int = 0,
) -> np.ndarray:
    """Return deterministic ``[-1, 1]`` noise at absolute pixel coordinates.

    >>> a = coordinate_noise(2, 3, 7, -2, 4)
    >>> np.array_equal(a[:, 1:], coordinate_noise(2, 2, 8, -2, 4))
    True
    """
    x = np.arange(origin_x, origin_x + width, dtype=np.int64).astype(np.uint64)
    y = np.arange(origin_y, origin_y + height, dtype=np.int64).astype(np.uint64)
    seed_term = np.uint64(
        ((seed & 0xFFFFFFFF) * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    )
    channel_term = np.uint64(
        ((channel & 0xFFFFFFFF) * 0x85EBCA77C2B2AE63)
        & 0xFFFFFFFFFFFFFFFF
    )
    value = (
        x[np.newaxis, :] * np.uint64(0x9E3779B185EBCA87)
        + y[:, np.newaxis] * np.uint64(0xC2B2AE3D27D4EB4F)
        + seed_term
        + channel_term
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    unit = (value >> np.uint64(40)).astype(np.float32) / np.float32(0xFFFFFF)
    return unit * np.float32(2.0) - np.float32(1.0)


def blurred_coordinate_noise(
    height: int,
    width: int,
    origin_x: int,
    origin_y: int,
    seed: int,
    radius: int,
    channel: int = 0,
) -> np.ndarray:
    """Blur absolute noise without depending on the current tile boundary."""
    radius = max(0, int(radius))
    if radius == 0:
        return coordinate_noise(
            height, width, origin_x, origin_y, seed, channel
        )
    expanded = coordinate_noise(
        height + radius * 2,
        width + radius * 2,
        origin_x - radius,
        origin_y - radius,
        seed,
        channel,
    )
    kernel = radius * 2 + 1
    blurred = cv2.GaussianBlur(
        expanded,
        (kernel, kernel),
        sigmaX=max(0.5, radius / 2.0),
        sigmaY=max(0.5, radius / 2.0),
        borderType=cv2.BORDER_CONSTANT,
    )
    return blurred[radius:radius + height, radius:radius + width]
