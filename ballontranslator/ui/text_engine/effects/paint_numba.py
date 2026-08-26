"""Cached parallel raster kernel for linear-gradient effect paints."""

import os
import os.path as osp
import threading

from ballontranslator.utils import shared


NUMBA_CACHE_DIR = osp.join(shared.cache_dir, 'numba')
# Configure the documented cache boundary before Numba reads its environment.
os.environ['NUMBA_CACHE_DIR'] = NUMBA_CACHE_DIR

import numpy as np
from numba import njit, prange


_warmup_lock = threading.Lock()
_warmup_complete = False


@njit(cache=True, parallel=True)
def _colorize_linear_gradient_rgba(
    rgba,
    projected_x,
    projected_y,
    length,
    positions,
    colors,
    opacities,
    opaque_stops,
):
    """Mutate RGBA with the NumPy oracle's exact float32/integer rounding.

    >>> callable(_colorize_linear_gradient_rgba)
    True
    """
    height, width, _channels = rgba.shape
    for row in prange(height):
        for column in range(width):
            parameter = np.float32(0.5) + (
                projected_y[row] + projected_x[column]
            ) / length
            parameter = min(
                max(parameter, np.float32(0.0)), np.float32(1.0)
            )

            right = 1
            for index in range(1, positions.size - 1):
                if parameter >= positions[index]:
                    right = index + 1
                else:
                    break
            left = right - 1
            span = positions[right] - positions[left]
            if span <= np.float32(0.0):
                ratio = (
                    np.float32(1.0)
                    if parameter >= positions[right]
                    else np.float32(0.0)
                )
            else:
                ratio = (parameter - positions[left]) / span
                ratio = min(
                    max(ratio, np.float32(0.0)), np.float32(1.0)
                )

            if opaque_stops:
                paint_alpha = 255
            else:
                paint_alpha = int(np.rint(
                    opacities[left]
                    + (opacities[right] - opacities[left]) * ratio
                ))
            for channel in range(3):
                paint_value = int(np.rint(
                    colors[left, channel]
                    + (colors[right, channel] - colors[left, channel])
                    * ratio
                ))
                rgba[row, column, channel] = paint_value

            if not opaque_stops:
                rgba[row, column, 3] = (
                    int(rgba[row, column, 3]) * paint_alpha + 127
                ) // 255


def warm_effect_paint_numba_cache() -> None:
    """Load or compile the gradient raster signature once per process.

    >>> NUMBA_CACHE_DIR.endswith(osp.join('.btrans_cache', 'numba'))
    True
    """
    global _warmup_complete
    with _warmup_lock:
        if _warmup_complete:
            return
        rgba = np.asarray(
            (((200, 20, 40, 127),) * 3,), dtype=np.uint8
        )
        positions = np.asarray((0.5, 0.5), dtype=np.float32)
        colors = np.asarray(
            ((255, 0, 0), (0, 0, 255)), dtype=np.float32
        )
        opacities = np.full(2, 255.0, dtype=np.float32)
        _colorize_linear_gradient_rgba(
            rgba,
            np.asarray((-1.0, 0.0, 1.0), dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.float32(3.0),
            positions,
            colors,
            opacities,
            True,
        )
        expected = np.asarray((
            (255, 0, 0, 127),
            (0, 0, 255, 127),
            (0, 0, 255, 127),
        ), dtype=np.uint8)
        if not np.array_equal(rgba[0], expected):
            raise RuntimeError(
                'Numba effect-paint kernel warm-up failed validation'
            )
        _warmup_complete = True


def colorize_linear_gradient_rgba(
    rgba: np.ndarray,
    surface_left: float,
    surface_top: float,
    center_x: float,
    center_y: float,
    render_scale: float,
    direction_x: float,
    direction_y: float,
    length: float,
    positions: np.ndarray,
    colors: np.ndarray,
    opacities: np.ndarray,
) -> bool:
    """Colorize in place, or return ``False`` for the NumPy fallback.

    >>> isinstance(_warmup_complete, bool)
    True
    """
    if not _warmup_complete or not rgba.flags.c_contiguous:
        return False
    render_scale = np.float32(render_scale)
    projected_x = (
        np.float32(surface_left)
        + (np.arange(rgba.shape[1], dtype=np.float32) + np.float32(0.5))
        / render_scale
        - np.float32(center_x)
    )
    projected_x *= np.float32(direction_x)
    projected_y = (
        np.float32(surface_top)
        + (np.arange(rgba.shape[0], dtype=np.float32) + np.float32(0.5))
        / render_scale
        - np.float32(center_y)
    )
    projected_y *= np.float32(direction_y)
    _colorize_linear_gradient_rgba(
        rgba,
        projected_x,
        projected_y,
        np.float32(length),
        positions,
        colors,
        opacities,
        bool(np.all(opacities == np.float32(255.0))),
    )
    return True
