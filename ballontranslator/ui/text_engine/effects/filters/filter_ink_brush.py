"""Dry-brush ink erosion with a directional noise-density gradient.

A coordinate-stable fBm field is stretched along the user-chosen Streak
Direction and thresholds glyph coverage into a porous, bristle-like texture.
The erosion density interpolates linearly along Gradient Direction from the
start to the end density, and a distance bias frays the contour first.
"""

from __future__ import annotations

import math
from typing import Mapping

import cv2
import numpy as np

from ballontranslator.ui.text_engine.effects.filters.registry import FilterContext
from ballontranslator.utils.text_effects import FilterScalar

FILTER_META = {
    'filter_id': 'builtin:ink_brush',
    'name': 'Ink Brush',
    'schema_version': 2,
    'order': 70,
    'params': (
        {
            'key': 'start_density', 'label': 'Start Density', 'kind': 'float',
            'default': 0.05, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'end_density', 'label': 'End Density', 'kind': 'float',
            'default': 0.25, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'angle', 'label': 'Gradient Direction', 'kind': 'float',
            'default': 90.0, 'minimum': 0.0, 'maximum': 360.0,
            'step': 1.0, 'decimals': 0, 'suffix': '\u00b0',
        },
        {
            'key': 'size', 'label': 'Grain Size', 'kind': 'float',
            'default': 2.0, 'minimum': 0.5, 'maximum': 12.0,
            'step': 0.1, 'decimals': 1,
        },
        {
            'key': 'streak', 'label': 'Streak', 'kind': 'float',
            'default': 0.6, 'minimum': 0.0, 'maximum': 1.0,
            'step': 1.0, 'display_factor': 100.0, 'decimals': 1,
            'suffix': '%',
        },
        {
            'key': 'streak_angle', 'label': 'Streak Direction',
            'kind': 'float',
            'default': 0.0, 'minimum': 0.0, 'maximum': 360.0,
            'step': 1.0, 'decimals': 0, 'suffix': '\u00b0',
        },
        {
            'key': 'seed', 'label': 'Seed', 'kind': 'int',
            'default': 0, 'minimum': 0, 'maximum': 2147483647,
            'step': 1,
        },
    ),
}


def migrate_params(
    from_version: int, params: Mapping[str, FilterScalar]
) -> Mapping[str, FilterScalar]:
    """Map the removed density/variation pair onto start/end endpoints.

    The old pair described one strength at the gradient start and another at
    its end. The endpoint density is a hole fraction rather than a strength
    multiplier, so the mapping keeps the directional intent at a comparable
    visual weight instead of preserving pixels exactly.
    """
    migrated = {
        key: value for key, value in params.items()
        if key not in ('density', 'variation')
    }
    if from_version < 2:
        try:
            density = min(max(float(params.get('density', 0.5)), 0.0), 1.0)
            variation = min(max(float(params.get('variation', 0.7)), 0.0), 1.0)
        except (TypeError, ValueError):
            density, variation = 0.5, 0.7
        migrated['start_density'] = 0.6 * density * (1.0 - variation)
        migrated['end_density'] = 0.6 * density
    return migrated


# Quantiles of the three-octave value-noise marginal below. Mapping the raw
# field through them makes the density behave like a hole fraction instead of
# depending on the fBm distribution.
_NOISE_QUANTILES = np.array([
    0.02022, 0.22093, 0.26103, 0.28966, 0.31313, 0.33350, 0.35215, 0.36939,
    0.38567, 0.40127, 0.41636, 0.43098, 0.44527, 0.45914, 0.47295, 0.48653,
    0.50019, 0.51376, 0.52739, 0.54121, 0.55513, 0.56943, 0.58391, 0.59885,
    0.61442, 0.63076, 0.64810, 0.66669, 0.68702, 0.71040, 0.73918, 0.77881,
    0.96590,
], dtype=np.float32)
_UNIFORM_STOPS = np.linspace(0.0, 1.0, 33, dtype=np.float32)
_HASH_X = np.uint64(0x9E3779B185EBCA87)
_HASH_Y = np.uint64(0xC2B2AE3D27D4EB4F)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_EDGE_BIAS = np.float32(0.8)
# Rim bias falloff in logical pixels; it is independent of the grain size.
_EDGE_DECAY = 2.0
# Hard cutoff in decay lengths keeps the edge bias within the tile halo.
_EDGE_REACH = 6.0


def tile_halo(params: Mapping[str, FilterScalar], render_scale: float) -> int:
    """Cover the rim-distance reach used by the edge bias."""
    if (
        float(params['start_density']) <= 0.0
        and float(params['end_density']) <= 0.0
    ):
        return 0
    scale = max(float(render_scale), 1e-6)
    return int(math.ceil(_EDGE_REACH * _EDGE_DECAY * scale)) + 2


def _hash01(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic ``[0, 1)`` lattice hash at integer coordinates."""
    x = ix.astype(np.int64).astype(np.uint64)
    y = iy.astype(np.int64).astype(np.uint64)
    seed_term = (
        (int(seed) & 0xFFFFFFFF) * 0x165667B19E3779F9
    ) & 0xFFFFFFFFFFFFFFFF
    value = x * _HASH_X + y * _HASH_Y + np.uint64(seed_term)
    value ^= value >> np.uint64(30)
    value *= _MIX_A
    value ^= value >> np.uint64(27)
    value *= _MIX_B
    value ^= value >> np.uint64(31)
    return (value >> np.uint64(40)).astype(np.float32) / np.float32(0xFFFFFF)


def _value_noise(u: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    """Smooth coordinate-stable value noise in ``[0, 1]``."""
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = (u - x0.astype(np.float32)).astype(np.float32)
    fy = (v - y0.astype(np.float32)).astype(np.float32)
    fx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    fy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    c00 = _hash01(x0, y0, seed)
    c10 = _hash01(x0 + 1, y0, seed)
    c01 = _hash01(x0, y0 + 1, seed)
    c11 = _hash01(x0 + 1, y0 + 1, seed)
    top = c00 + (c10 - c00) * fx
    bottom = c01 + (c11 - c01) * fx
    return top + (bottom - top) * fy


def _fbm(u: np.ndarray, v: np.ndarray, seed: int) -> np.ndarray:
    value = _value_noise(u, v, seed) * np.float32(0.62)
    value += _value_noise(
        u * np.float32(2.03) + np.float32(17.3),
        v * np.float32(2.03) + np.float32(9.1),
        seed + 101,
    ) * np.float32(0.26)
    value += _value_noise(
        u * np.float32(4.11) + np.float32(3.7),
        v * np.float32(4.11) + np.float32(27.9),
        seed + 211,
    ) * np.float32(0.12)
    return value


def _uniformize(noise: np.ndarray) -> np.ndarray:
    """Map the fixed fBm marginal onto ``0..1`` for linear density control."""
    return np.interp(
        noise, _NOISE_QUANTILES, _UNIFORM_STOPS
    ).astype(np.float32)


def _smoothstep(
    edge0: np.ndarray, edge1: np.ndarray, value: np.ndarray
) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gradient_position(
    logical_x: np.ndarray,
    logical_y: np.ndarray,
    angle: float,
    width: float,
    height: float,
) -> np.ndarray:
    """Return ``0..1`` along ``angle`` across the item's logical box."""
    if width <= 0.0 or height <= 0.0:
        return np.full(logical_x.shape, 0.5, dtype=np.float32)
    radians = math.radians(angle)
    ux, uy = math.cos(radians), math.sin(radians)
    half_extent = abs(ux) * width * 0.5 + abs(uy) * height * 0.5
    offset = (
        (logical_x - width * 0.5) * ux
        + (logical_y - height * 0.5) * uy
    )
    return np.clip(
        0.5 + offset / max(2.0 * half_extent, 1e-6), 0.0, 1.0
    ).astype(np.float32)


def apply(
    rgba: np.ndarray,
    params: Mapping[str, FilterScalar],
    context: FilterContext,
) -> np.ndarray:
    """Erode alpha with anisotropic bristle noise.

    ``start_density`` and ``end_density`` are the erosion densities (hole
    fractions) at the two ends of the gradient; they interpolate linearly and
    100% erodes the ink completely. ``streak_angle`` rotates the bristle
    direction in the item's logical space.

    >>> rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    >>> apply(
    ...     rgba, {'start_density': 0.0, 'end_density': 0.0},
    ...     FilterContext(1.0, 0, 0),
    ... ) is rgba
    True
    """
    start_density = float(params['start_density'])
    end_density = float(params['end_density'])
    if (
        rgba.size == 0
        or (start_density <= 0.0 and end_density <= 0.0)
    ):
        return rgba
    height, width = rgba.shape[:2]
    scale = max(float(context.render_scale), 1e-6)
    size = float(params['size'])
    streak = float(params['streak'])
    seed = int(params['seed'])
    alpha = rgba[:, :, 3]
    if not np.any(alpha):
        return rgba
    coverage = np.ones((height, width), dtype=np.float32)
    rows, columns = np.nonzero(alpha > 0)
    local_alpha = alpha.astype(np.float32) / 255.0
    angle = np.float32(np.radians(float(params['streak_angle'])))
    cosine = np.float32(np.cos(angle))
    sine = np.float32(np.sin(angle))
    px = (
        np.arange(width, dtype=np.float32) + context.origin_x
    ) / scale
    py = (
        np.arange(height, dtype=np.float32) + context.origin_y
    ) / scale
    ink_x = px[columns]
    ink_y = py[rows]
    along = size * (1.0 + 4.0 * streak)
    u = (ink_x * cosine + ink_y * sine) / along
    v = (-ink_x * sine + ink_y * cosine) / size
    fine = _uniformize(_fbm(u, v, seed))
    position = _gradient_position(
        ink_x,
        ink_y,
        float(params['angle']),
        float(context.logical_width),
        float(context.logical_height),
    )
    density = start_density + (end_density - start_density) * position
    # The contour frays faster than the body: the erosion density rises near
    # the edge, so a low overall density still produces a dry-brush rim.
    ink = (alpha > 127).astype(np.uint8)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rim = np.exp(
        -distance[rows, columns] / max(1e-3, _EDGE_DECAY * scale)
    )
    rim[distance[rows, columns] >= _EDGE_REACH * _EDGE_DECAY * scale] = 0.0
    probability = np.clip(
        density * (1.0 + _EDGE_BIAS * rim), 0.0, 1.0
    )
    probability = np.where(density <= 0.0, 0.0, probability)
    probability = np.where(density >= 1.0, 1.0, probability)
    soft = np.float32(0.02)
    eroded = _smoothstep(probability - soft, probability + soft, fine)
    eroded = np.where(probability <= 0.0, 1.0, eroded)
    eroded = np.where(probability >= 1.0, 0.0, eroded)
    coverage[rows, columns] = eroded
    rgba[:, :, 3] = np.clip(
        np.rint(local_alpha * coverage * 255.0), 0, 255
    ).astype(np.uint8)
    return rgba
