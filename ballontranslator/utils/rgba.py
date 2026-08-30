"""Small in-place helpers for alpha-correct RGBA interpolation."""

import numpy as np


def premultiply_rgba_in_place(rgba: np.ndarray) -> np.ndarray:
    """Premultiply an owned RGBA8 array for interpolation.

    >>> pixels = np.array([[[200, 100, 50, 128]]], dtype=np.uint8)
    >>> premultiply_rgba_in_place(pixels)[0, 0].tolist()
    [100, 50, 25, 128]
    """
    alpha = rgba[..., 3].astype(np.uint16)
    for channel_index in range(3):
        channel = rgba[..., channel_index].astype(np.uint16)
        channel *= alpha
        channel += 127
        channel //= 255
        rgba[..., channel_index] = channel.astype(np.uint8)
    return rgba


def unpremultiply_rgba_in_place(rgba: np.ndarray) -> np.ndarray:
    """Return premultiplied RGBA8 pixels to straight-alpha form in place.

    >>> pixels = np.array([[[100, 50, 25, 128], [0, 0, 0, 0]]], dtype=np.uint8)
    >>> unpremultiply_rgba_in_place(pixels)[0, 1].tolist()
    [0, 0, 0, 0]
    """
    alpha = rgba[..., 3]
    if np.all(alpha == 255):
        return rgba
    nonzero = alpha > 0
    rgba[..., :3][~nonzero] = 0
    if not np.any(nonzero):
        return rgba
    alpha_values = alpha[nonzero].astype(np.float32)
    for channel_index in range(3):
        channel = rgba[..., channel_index]
        values = channel[nonzero].astype(np.float32)
        values *= 255.0
        values /= alpha_values
        channel[nonzero] = np.clip(
            np.rint(values), 0, 255
        ).astype(np.uint8)
    return rgba
