"""Immutable TextBlock-owned alpha-mask values and permissive loading."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Mapping, Optional, Sequence, Tuple, Union

from .logger import logger as LOGGER


TEXT_ALPHA_MASK_VERSION = 1
ALPHA_BRUSH_MODES = ('erase', 'restore')
ALPHA_BRUSH_SIMPLIFY_TOLERANCE = 0.25


def _finite_number(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a number')
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted


def _point_tuple(value: Sequence[Real]) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError('alpha brush point must contain two numeric values')
    return (
        _finite_number('alpha brush point X', value[0]),
        _finite_number('alpha brush point Y', value[1]),
    )


@dataclass(frozen=True)
class AlphaBrushStroke:
    """One ordered hard round erase or restore brush stroke.

    Points are relative to the unpadded logical text rectangle origin.

    >>> AlphaBrushStroke('erase', 12, ((-2, 3),)).points
    ((-2.0, 3.0),)
    """

    mode: str
    diameter: float
    points: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.mode not in ALPHA_BRUSH_MODES:
            raise ValueError('alpha brush mode must be erase or restore')
        diameter = _finite_number('alpha brush diameter', self.diameter)
        if diameter <= 0.0:
            raise ValueError('alpha brush diameter must be greater than 0.0')
        points = tuple(_point_tuple(point) for point in self.points)
        if not points:
            raise ValueError('alpha brush stroke requires at least one point')
        object.__setattr__(self, 'diameter', diameter)
        object.__setattr__(self, 'points', points)

    def to_serializable_dict(self) -> dict:
        return {
            'mode': self.mode,
            'diameter': self.diameter,
            'points': [list(point) for point in self.points],
        }


@dataclass(frozen=True)
class TextAlphaMask:
    """Versioned immutable alpha-mask history owned by one TextBlock.

    >>> TextAlphaMask().is_neutral()
    True
    >>> TextAlphaMask(strokes=(AlphaBrushStroke('erase', 8, ((1, 2),)),)).is_neutral()
    False
    """

    version: int = TEXT_ALPHA_MASK_VERSION
    enabled: bool = True
    strokes: Tuple[AlphaBrushStroke, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or self.version != TEXT_ALPHA_MASK_VERSION
        ):
            raise ValueError('unsupported text alpha mask version')
        if not isinstance(self.enabled, bool):
            raise TypeError('text alpha mask enabled must be a bool')
        strokes = tuple(self.strokes)
        if any(not isinstance(stroke, AlphaBrushStroke) for stroke in strokes):
            raise TypeError('text alpha mask requires typed brush strokes')
        object.__setattr__(self, 'version', int(self.version))
        object.__setattr__(self, 'strokes', strokes)

    def is_neutral(self) -> bool:
        return not self.enabled or not self.strokes

    def to_serializable_dict(self) -> dict:
        return {
            'version': self.version,
            'enabled': self.enabled,
            'strokes': [
                stroke.to_serializable_dict() for stroke in self.strokes
            ],
        }


def simplify_alpha_brush_points(
    points: Sequence[Sequence[Real]],
    tolerance: Real = ALPHA_BRUSH_SIMPLIFY_TOLERANCE,
) -> Tuple[Tuple[float, float], ...]:
    """Deterministically simplify one sampled brush path.

    Endpoints are always retained and the default tolerance stays below one
    item-local pixel at normal scale.

    >>> simplify_alpha_brush_points(((0, 0), (1, 0.01), (2, 0)))
    ((0.0, 0.0), (2.0, 0.0))
    >>> simplify_alpha_brush_points(((-1, 2),))
    ((-1.0, 2.0),)
    """
    values = tuple(_point_tuple(point) for point in points)
    threshold = _finite_number('alpha brush simplify tolerance', tolerance)
    if threshold < 0.0:
        raise ValueError('alpha brush simplify tolerance must be non-negative')
    if len(values) < 3:
        return values

    keep = [False] * len(values)
    keep[0] = keep[-1] = True
    pending = [(0, len(values) - 1)]
    threshold_sq = threshold * threshold
    while pending:
        start_index, end_index = pending.pop()
        start_x, start_y = values[start_index]
        end_x, end_y = values[end_index]
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        segment_sq = delta_x * delta_x + delta_y * delta_y
        farthest_index = -1
        farthest_sq = threshold_sq
        for index in range(start_index + 1, end_index):
            point_x, point_y = values[index]
            if segment_sq == 0.0:
                distance_sq = (
                    (point_x - start_x) ** 2
                    + (point_y - start_y) ** 2
                )
            else:
                amount = (
                    (point_x - start_x) * delta_x
                    + (point_y - start_y) * delta_y
                ) / segment_sq
                amount = min(1.0, max(0.0, amount))
                nearest_x = start_x + amount * delta_x
                nearest_y = start_y + amount * delta_y
                distance_sq = (
                    (point_x - nearest_x) ** 2
                    + (point_y - nearest_y) ** 2
                )
            if distance_sq > farthest_sq:
                farthest_sq = distance_sq
                farthest_index = index
        if farthest_index >= 0:
            keep[farthest_index] = True
            pending.append((start_index, farthest_index))
            pending.append((farthest_index, end_index))
    return tuple(point for point, retained in zip(values, keep) if retained)


def _load_alpha_brush_stroke(
    payload: object,
    stroke_index: int,
) -> Optional[AlphaBrushStroke]:
    if not isinstance(payload, Mapping):
        LOGGER.warning(
            'Ignoring malformed text alpha mask stroke %s: %r.',
            stroke_index,
            payload,
        )
        return None

    unknown = set(payload) - {'mode', 'diameter', 'points'}
    if unknown:
        LOGGER.warning(
            'Ignoring unknown text alpha mask stroke fields at %s: %s.',
            stroke_index,
            sorted(unknown),
        )

    raw_points = payload.get('points')
    if not isinstance(raw_points, (list, tuple)):
        LOGGER.warning(
            'Ignoring text alpha mask stroke %s with malformed points.',
            stroke_index,
        )
        return None
    points = []
    for point_index, point in enumerate(raw_points):
        try:
            points.append(_point_tuple(point))
        except (TypeError, ValueError) as error:
            LOGGER.warning(
                'Ignoring malformed text alpha mask point %s in stroke %s: %s.',
                point_index,
                stroke_index,
                error,
            )
    if not points:
        LOGGER.warning(
            'Ignoring text alpha mask stroke %s without valid points.',
            stroke_index,
        )
        return None
    try:
        return AlphaBrushStroke(
            mode=payload.get('mode'),
            diameter=payload.get('diameter'),
            points=tuple(points),
        )
    except (TypeError, ValueError) as error:
        LOGGER.warning(
            'Ignoring malformed text alpha mask stroke %s: %s.',
            stroke_index,
            error,
        )
        return None


def load_text_alpha_mask(
    payload: Optional[Union[TextAlphaMask, Mapping[str, object]]],
) -> Optional[TextAlphaMask]:
    """Load optional mask data without rejecting the surrounding project.

    Bad strokes and points are isolated so valid siblings survive.

    >>> loaded = load_text_alpha_mask({'version': 1, 'enabled': True,
    ...     'strokes': [
    ...         {'mode': 'erase', 'diameter': 4, 'points': [[1, 2], [3, 4]]},
    ...     ]})
    >>> loaded.strokes[0].points
    ((1.0, 2.0), (3.0, 4.0))
    """
    if payload is None or isinstance(payload, TextAlphaMask):
        return payload
    if not isinstance(payload, Mapping):
        LOGGER.warning('Ignoring malformed text alpha mask payload %r.', payload)
        return None

    unknown = set(payload) - {'version', 'enabled', 'strokes'}
    if unknown:
        LOGGER.warning(
            'Ignoring unknown text alpha mask fields: %s.', sorted(unknown)
        )
    version = payload.get('version')
    if (
        isinstance(version, bool)
        or not isinstance(version, Integral)
        or version != TEXT_ALPHA_MASK_VERSION
    ):
        LOGGER.warning('Ignoring unsupported text alpha mask version %r.', version)
        return None
    enabled = payload.get('enabled', True)
    if not isinstance(enabled, bool):
        LOGGER.warning(
            'Ignoring invalid text alpha mask enabled value %r; using true.',
            enabled,
        )
        enabled = True
    raw_strokes = payload.get('strokes', ())
    if not isinstance(raw_strokes, (list, tuple)):
        LOGGER.warning('Ignoring text alpha mask with malformed strokes payload.')
        raw_strokes = ()
    strokes = tuple(
        stroke
        for index, raw_stroke in enumerate(raw_strokes)
        for stroke in (_load_alpha_brush_stroke(raw_stroke, index),)
        if stroke is not None
    )
    return TextAlphaMask(version=version, enabled=enabled, strokes=strokes)
