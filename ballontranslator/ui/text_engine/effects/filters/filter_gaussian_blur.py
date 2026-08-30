"""Finite premultiplied-alpha Gaussian blur."""

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
    'filter_id': 'builtin:gaussian_blur',
    'name': 'Gaussian Blur',
    'schema_version': 1,
    'order': 40,
    'expands_alpha': True,
    'params': (
        {
            'key': 'radius', 'label': 'Radius', 'kind': 'float',
            'default': 2.0, 'minimum': 0.0, 'maximum': 32.0,
            'step': 0.1, 'decimals': 1, 'suffix': ' px',
        },
    ),
}


def tile_halo(params: Mapping[str, FilterScalar], render_scale: float) -> int:
    return int(math.ceil(float(params['radius']) * render_scale))


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    radius = tile_halo(params, context.render_scale)
    if radius <= 0 or rgba.size == 0:
        return rgba
    return unpremultiply_rgba_float32(finite_gaussian(
        premultiply_rgba_float32(rgba), radius
    ))
