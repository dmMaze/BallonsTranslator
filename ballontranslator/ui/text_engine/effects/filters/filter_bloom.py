"""Thresholded additive bloom in premultiplied RGBA."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ballontranslator.ui.text_engine.effects.filters._gaussian import (
    finite_gaussian,
    premultiply_rgba_float32,
    unpremultiply_rgba_float32,
)
from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar


FILTER_META = {
    'filter_id': 'builtin:bloom',
    'name': 'Bloom',
    'schema_version': 1,
    'order': 50,
    'expands_alpha': True,
    'params': (
        {
            'key': 'threshold', 'label': 'Threshold', 'kind': 'float',
            'default': 0.6, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'radius', 'label': 'Radius', 'kind': 'float',
            'default': 6.0, 'minimum': 0.0, 'maximum': 32.0,
            'step': 0.1, 'decimals': 1, 'suffix': ' px',
        },
        {
            'key': 'intensity', 'label': 'Intensity', 'kind': 'float',
            'default': 0.8, 'minimum': 0.0, 'maximum': 3.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
    ),
}


def tile_halo(params: Mapping[str, FilterScalar], render_scale: float) -> int:
    if float(params['intensity']) <= 0.0:
        return 0
    return int(math.ceil(float(params['radius']) * render_scale))


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    """Add a finite bright-pass halo while retaining valid premultiplied color.

    >>> pixel = np.zeros((1, 1, 4), dtype=np.uint8)
    >>> params = {'threshold': 0.6, 'radius': 6.0, 'intensity': 0.0}
    >>> apply(pixel, params, FilterContext(1.0, 0, 0)) is pixel
    True
    """
    intensity = float(params['intensity'])
    threshold = float(params['threshold'])
    if intensity <= 0.0 or rgba.size == 0:
        return rgba

    source = premultiply_rgba_float32(rgba)
    peak = np.max(rgba[:, :, :3], axis=2).astype(np.float32)
    peak /= np.float32(255.0)
    if threshold >= 1.0:
        np.greater_equal(peak, np.float32(1.0), out=peak)
    else:
        np.subtract(peak, np.float32(threshold), out=peak)
        np.divide(peak, np.float32(1.0 - threshold), out=peak)
        np.clip(peak, 0.0, 1.0, out=peak)
    bright = source.copy()
    np.multiply(bright, peak[:, :, np.newaxis], out=bright)
    finite_gaussian(bright, tile_halo(params, context.render_scale))
    bright *= np.float32(intensity)

    np.clip(bright[:, :, 3], 0.0, 1.0, out=bright[:, :, 3])
    np.subtract(np.float32(1.0), source[:, :, 3], out=peak)
    np.multiply(bright[:, :, 3], peak, out=peak)
    np.add(source[:, :, 3], peak, out=source[:, :, 3])
    np.add(source[:, :, :3], bright[:, :, :3], out=source[:, :, :3])
    np.clip(source[:, :, :3], 0.0, 1.0, out=source[:, :, :3])
    np.minimum(
        source[:, :, :3], source[:, :, 3:4], out=source[:, :, :3]
    )
    del bright, peak
    return unpremultiply_rgba_float32(source)
