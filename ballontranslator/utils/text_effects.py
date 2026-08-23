"""Immutable text-effect values and pure primary-Stroke operations."""

from dataclasses import dataclass, field, replace
import math
from numbers import Integral, Real
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

from .logger import logger as LOGGER


def _float_in_range(
    name: str,
    value: Real,
    minimum: float,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a number')
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    if maximum is None:
        if converted < minimum:
            raise ValueError(f'{name} must be at least {minimum}')
    elif not minimum <= converted <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return converted


def _color_tuple(value: Sequence[int]) -> Tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError('solid paint color must contain three integer channels')
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, Integral)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(
            'solid paint color channels must be integers from 0 to 255'
        )
    return tuple(int(channel) for channel in value)


@dataclass(frozen=True)
class SolidPaint:
    """Immutable solid RGB paint persisted through a stable type name.

    >>> SolidPaint([12, 34, 56]).color
    (12, 34, 56)
    """

    color: Tuple[int, int, int] = (0, 0, 0)
    paint_type: str = field(init=False, default='solid')

    def __post_init__(self) -> None:
        object.__setattr__(self, 'color', _color_tuple(self.color))

    def to_serializable_dict(self) -> dict:
        return {
            'paint_type': self.paint_type,
            'color': list(self.color),
        }


@dataclass(frozen=True)
class StrokeEffect:
    """One immutable centered solid Stroke effect.

    Width remains relative to font size, matching the existing Stroke value.

    >>> StrokeEffect(width=0.2).effect_type
    'stroke'
    """

    enabled: bool = True
    opacity: float = 1.0
    blend_mode: str = 'normal'
    width: float = 0.1
    paint: SolidPaint = field(default_factory=SolidPaint)
    effect_type: str = field(init=False, default='stroke')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('stroke enabled must be a bool')
        object.__setattr__(
            self,
            'opacity',
            _float_in_range('stroke opacity', self.opacity, 0.0, 1.0),
        )
        if self.blend_mode != 'normal':
            raise ValueError('unsupported stroke blend mode')
        object.__setattr__(
            self,
            'width',
            _float_in_range('stroke width', self.width, 0.0),
        )
        if not isinstance(self.paint, SolidPaint):
            raise TypeError('stroke paint must be SolidPaint')

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'width': self.width,
            'paint': self.paint.to_serializable_dict(),
        }

    def is_neutral(self) -> bool:
        return not self.enabled or self.opacity == 0.0 or self.width == 0.0


@dataclass(frozen=True)
class TextEffectStack:
    """Complete immutable style-owned text-effect value.

    ``effects`` preserves semantic/visual order. Overall opacity applies to
    the completed item rather than an individual effect.

    >>> stack = with_primary_stroke(TextEffectStack(), width=0.25)
    >>> (len(stack), stack[0].width)
    (1, 0.25)
    """

    overall_opacity: float = 1.0
    effects: Tuple[StrokeEffect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'overall_opacity',
            _float_in_range(
                'overall opacity', self.overall_opacity, 0.0, 1.0
            ),
        )
        effects = tuple(self.effects)
        if any(not isinstance(effect, StrokeEffect) for effect in effects):
            raise TypeError('text effect stack requires typed effect values')
        object.__setattr__(self, 'effects', effects)

    def __iter__(self) -> Iterator[StrokeEffect]:
        return iter(self.effects)

    def __len__(self) -> int:
        return len(self.effects)

    def __getitem__(self, index: int) -> StrokeEffect:
        return self.effects[index]

    @property
    def has_active_effects(self) -> bool:
        return any(not effect.is_neutral() for effect in self.effects)

    def is_neutral(self) -> bool:
        return self.overall_opacity == 1.0 and not self.has_active_effects

    def to_serializable_dict(self) -> dict:
        return {
            'overall_opacity': self.overall_opacity,
            'effects': [
                effect.to_serializable_dict() for effect in self.effects
            ],
        }


def _unexpected_fields(
    payload: Mapping[str, object], allowed: Sequence[str], label: str
) -> None:
    unexpected = set(payload) - set(allowed)
    if unexpected:
        raise ValueError(f'unsupported {label} fields: {sorted(unexpected)}')


def _coerce_solid_paint(value: object) -> SolidPaint:
    if isinstance(value, SolidPaint):
        return value
    if not isinstance(value, dict):
        raise ValueError('stroke paint must be a value or typed payload')
    payload = dict(value)
    _unexpected_fields(payload, ('paint_type', 'color'), 'solid paint')
    if payload.pop('paint_type', None) != 'solid':
        raise ValueError('stroke paint payload requires paint_type solid')
    return SolidPaint(**payload)


def coerce_text_effect(value: Union[StrokeEffect, dict]) -> StrokeEffect:
    """Return a live Stroke or construct one strict typed payload.

    >>> coerce_text_effect({'effect_type': 'stroke', 'width': 0.2}).width
    0.2
    """
    if isinstance(value, StrokeEffect):
        return value
    if not isinstance(value, dict):
        raise ValueError('text effect must be a value or typed payload')
    payload = dict(value)
    _unexpected_fields(
        payload,
        (
            'effect_type',
            'enabled',
            'opacity',
            'blend_mode',
            'width',
            'paint',
        ),
        'text effect',
    )
    if payload.pop('effect_type', None) != 'stroke':
        raise ValueError('unsupported or missing text effect type')
    if 'paint' in payload:
        payload['paint'] = _coerce_solid_paint(payload['paint'])
    return StrokeEffect(**payload)


def coerce_text_effect_stack(
    value: Union[TextEffectStack, dict],
) -> TextEffectStack:
    """Load a stack payload while isolating malformed optional data.

    Invalid top-level fields fall back independently. Invalid effect entries
    are warned about and omitted without discarding valid siblings.

    >>> stack = coerce_text_effect_stack({'effects': [
    ...     {'effect_type': 'stroke', 'width': 0.3},
    ... ]})
    >>> len(stack)
    1
    """
    if isinstance(value, TextEffectStack):
        return value
    if not isinstance(value, dict):
        LOGGER.warning(
            'Ignoring invalid text effect stack (%r); using an empty stack.',
            value,
        )
        return TextEffectStack()

    payload = dict(value)
    unknown = set(payload) - {'overall_opacity', 'effects'}
    if unknown:
        LOGGER.warning(
            'Ignoring unsupported text effect stack fields: %s.',
            sorted(unknown),
        )

    overall_opacity = payload.get('overall_opacity', 1.0)
    try:
        overall_opacity = _float_in_range(
            'overall opacity', overall_opacity, 0.0, 1.0
        )
    except (TypeError, ValueError) as error:
        LOGGER.warning(
            'Ignoring invalid overall text opacity (%s); using 1.0.', error
        )
        overall_opacity = 1.0

    raw_effects = payload.get('effects', ())
    if not isinstance(raw_effects, (list, tuple)):
        LOGGER.warning(
            'Ignoring invalid text effect entries container (%r).',
            raw_effects,
        )
        raw_effects = ()
    effects = []
    for index, raw_effect in enumerate(raw_effects):
        try:
            effects.append(coerce_text_effect(raw_effect))
        except (TypeError, ValueError) as error:
            LOGGER.warning(
                'Ignoring invalid text effect at index %s (%s).',
                index,
                error,
            )
    return TextEffectStack(overall_opacity, tuple(effects))


def primary_stroke(stack: TextEffectStack) -> Optional[StrokeEffect]:
    """Return the first Stroke in semantic/visual order, if present."""
    if not isinstance(stack, TextEffectStack):
        raise TypeError('primary_stroke requires TextEffectStack')
    return next(
        (
            effect
            for effect in stack.effects
            if isinstance(effect, StrokeEffect)
        ),
        None,
    )


def ensure_primary_stroke(stack: TextEffectStack) -> TextEffectStack:
    """Return ``stack`` with a default primary Stroke when one is absent."""
    if not isinstance(stack, TextEffectStack):
        raise TypeError('ensure_primary_stroke requires TextEffectStack')
    if primary_stroke(stack) is not None:
        return stack
    return replace(stack, effects=stack.effects + (StrokeEffect(),))


def with_primary_stroke(
    stack: TextEffectStack, **parameters: object
) -> TextEffectStack:
    """Ensure then immutably update only the primary Stroke.

    Unspecified Stroke values and all later effect order are preserved.
    """
    ensured = ensure_primary_stroke(stack)
    stroke = primary_stroke(ensured)
    assert stroke is not None
    updated = replace(stroke, **parameters)
    if updated == stroke:
        return ensured
    effects = list(ensured.effects)
    effects[effects.index(stroke)] = updated
    return replace(ensured, effects=tuple(effects))


def with_non_stroke_effects(
    stack: TextEffectStack, source: TextEffectStack
) -> TextEffectStack:
    """Copy the currently supported non-Stroke style state from ``source``.

    Work Package 2 supports only overall opacity outside Stroke. Keeping this
    operation named prevents run callers from growing index assumptions when
    later typed effects extend the stack.

    >>> target = TextEffectStack(effects=(StrokeEffect(width=0.4),))
    >>> source = TextEffectStack(overall_opacity=0.6)
    >>> with_non_stroke_effects(target, source).effects == target.effects
    True
    """
    if not isinstance(stack, TextEffectStack) or not isinstance(
        source, TextEffectStack
    ):
        raise TypeError('with_non_stroke_effects requires TextEffectStack')
    if stack.overall_opacity == source.overall_opacity:
        return stack
    return replace(stack, overall_opacity=source.overall_opacity)
