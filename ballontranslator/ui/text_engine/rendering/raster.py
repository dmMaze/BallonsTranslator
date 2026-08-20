"""Bounded raster policy shared by transformed text effect rendering."""

import math
from typing import NamedTuple

import cv2

from .glyph import GlyphRasterAllocationError


EFFECT_ANTIALIAS_GUARD = 1.0
EFFECT_CLEAR_BORDER_GUARD = 1.0
EFFECT_RASTER_GUARD = EFFECT_ANTIALIAS_GUARD + EFFECT_CLEAR_BORDER_GUARD
EFFECT_CACHE_MAX_SCALE = 8.0
EFFECT_CACHE_MAX_PIXELS = 4_194_304
EFFECT_CACHE_MAX_DIMENSION = 8192
EFFECT_CACHE_MAX_BYTES = 32 * 1024 * 1024
EFFECT_TILE_MAX_EDGE = 2048


class EffectRasterPlan(NamedTuple):
    mode: str
    tier: float
    pixel_width: int
    pixel_height: int
    tile_edge: int


class EffectRasterAllocationError(RuntimeError):
    pass


EFFECT_RASTER_FAILURES = (
    EffectRasterAllocationError,
    GlyphRasterAllocationError,
    MemoryError,
    OverflowError,
    cv2.error,
)
RASTER_BRIDGE_FAILURES = (
    RuntimeError,
    ValueError,
    TypeError,
    BufferError,
)
RASTER_BOUNDARY_FAILURES = EFFECT_RASTER_FAILURES + RASTER_BRIDGE_FAILURES


def plan_effect_raster(
    width: float,
    height: float,
    requested_scale: float,
) -> EffectRasterPlan:
    """Choose a bounded full-surface tier or visible-tile plan.

    >>> plan_effect_raster(100, 80, 99).tier
    8.0
    >>> plan_effect_raster(100, 80, 0.5).tier
    0.5
    >>> plan_effect_raster(10000, 10000, 8).mode
    'tiles'
    """
    width = max(0.0, float(width))
    height = max(0.0, float(height))
    requested_scale = max(
        0.25, min(float(requested_scale), EFFECT_CACHE_MAX_SCALE)
    )
    tiers = (
        (8.0, 4.0, 2.0, 1.0)
        if requested_scale >= 1.0
        else (0.5, 0.25)
    )
    for tier in tiers:
        if tier > requested_scale:
            continue
        pixel_width = max(1, math.ceil(width * tier))
        pixel_height = max(1, math.ceil(height * tier))
        pixels = pixel_width * pixel_height
        if (
            pixel_width <= EFFECT_CACHE_MAX_DIMENSION
            and pixel_height <= EFFECT_CACHE_MAX_DIMENSION
            and pixels <= EFFECT_CACHE_MAX_PIXELS
            and pixels * 4 <= EFFECT_CACHE_MAX_BYTES
        ):
            return EffectRasterPlan(
                'full', tier, pixel_width, pixel_height, 0
            )
    tile_edge = min(
        EFFECT_TILE_MAX_EDGE,
        EFFECT_CACHE_MAX_DIMENSION,
        int(math.sqrt(EFFECT_CACHE_MAX_PIXELS)),
    )
    return EffectRasterPlan('tiles', 1.0, 0, 0, tile_edge)


def quality_raster_request(requested_scale: float) -> float:
    """Round settled rendering up to the shared quality tier.

    >>> quality_raster_request(3.9)
    4.0
    >>> quality_raster_request(1.0)
    1.0
    """
    requested_scale = max(1.0, float(requested_scale))
    return next(
        (
            tier
            for tier in (1.0, 2.0, 4.0, 8.0)
            if requested_scale <= tier * (1.0 + 1e-6)
        ),
        requested_scale,
    )
