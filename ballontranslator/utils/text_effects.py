"""Immutable typed text-effect values and stack editing helpers."""

from dataclasses import dataclass, field, replace
import math
from numbers import Integral, Real
from typing import Iterator, Mapping, Optional, Sequence, Tuple, Union

from .logger import logger as LOGGER
from .raster_assets import RasterAssetRef, coerce_raster_asset_ref


SHADOW_DISTANCE_LIMIT = 10.0
SHADOW_BLUR_LIMIT = 10.0
SHADOW_SPREAD_LIMIT = 10.0
TEXT_EFFECT_BLEND_MODES = (
    'normal',
    'darken', 'multiply', 'color_burn', 'linear_burn', 'darker_color',
    'lighten', 'screen', 'color_dodge', 'linear_dodge', 'lighter_color',
)
FilterScalar = Union[None, bool, int, float, str]
FilterParams = Tuple[Tuple[str, FilterScalar], ...]


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
        raise TypeError('paint color must contain three integer channels')
    if any(
        isinstance(channel, bool)
        or not isinstance(channel, Integral)
        or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(
            'paint color channels must be integers from 0 to 255'
        )
    return tuple(int(channel) for channel in value)


def _validated_blend_mode(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f'{name} blend mode must be a string')
    if value not in TEXT_EFFECT_BLEND_MODES:
        raise ValueError(f'unsupported {name} blend mode')
    return value


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
class GradientStop:
    """One immutable linear-gradient stop.

    >>> GradientStop(0.5, [12, 34, 56], 0.75).color
    (12, 34, 56)
    """

    position: float = 0.0
    color: Tuple[int, int, int] = (0, 0, 0)
    opacity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'position',
            _float_in_range(
                'gradient stop position', self.position, 0.0, 1.0
            ),
        )
        object.__setattr__(self, 'color', _color_tuple(self.color))
        object.__setattr__(
            self,
            'opacity',
            _float_in_range(
                'gradient stop opacity', self.opacity, 0.0, 1.0
            ),
        )

    def to_serializable_dict(self) -> dict:
        return {
            'position': self.position,
            'color': list(self.color),
            'opacity': self.opacity,
        }


def _default_gradient_stops() -> Tuple[GradientStop, GradientStop]:
    return (
        GradientStop(0.0, (0, 0, 0), 1.0),
        GradientStop(1.0, (255, 255, 255), 1.0),
    )


@dataclass(frozen=True)
class LinearGradientPaint:
    """Immutable logical-block linear-gradient paint.

    >>> LinearGradientPaint(angle=450).angle
    90.0
    """

    stops: Tuple[GradientStop, ...] = field(
        default_factory=_default_gradient_stops
    )
    angle: float = 0.0
    scale: float = 1.0
    paint_type: str = field(init=False, default='linear_gradient')

    def __post_init__(self) -> None:
        stops = tuple(self.stops)
        if not 2 <= len(stops) <= 32:
            raise ValueError('linear gradient requires 2 to 32 stops')
        if any(not isinstance(stop, GradientStop) for stop in stops):
            raise TypeError('linear gradient stops require GradientStop values')
        if any(
            current.position > following.position
            for current, following in zip(stops, stops[1:])
        ):
            raise ValueError('linear gradient stops must be ordered')
        object.__setattr__(self, 'stops', stops)
        angle = _float_in_range(
            'linear gradient angle', self.angle, -math.inf, math.inf
        )
        object.__setattr__(self, 'angle', angle % 360.0)
        object.__setattr__(
            self,
            'scale',
            _float_in_range(
                'linear gradient scale', self.scale, 0.1, 4.0
            ),
        )

    def to_serializable_dict(self) -> dict:
        return {
            'paint_type': self.paint_type,
            'stops': [stop.to_serializable_dict() for stop in self.stops],
            'angle': self.angle,
            'scale': self.scale,
        }


@dataclass(frozen=True)
class TexturePaint:
    """Map one managed raster across the unpadded logical text rectangle.

    Fill stretches to the logical rectangle, Fit contains the whole image,
    Crop covers and center-crops it, and Tile repeats from the logical origin.

    >>> TexturePaint(
    ...     RasterAssetRef('assets/' + 'a' * 64 + '.png'), mapping='tile'
    ... ).paint_type
    'texture'
    >>> TexturePaint().asset is None
    True
    """

    asset: Optional[RasterAssetRef] = None
    mapping: str = 'fill'
    scale: float = 1.0
    paint_type: str = field(init=False, default='texture')

    def __post_init__(self) -> None:
        if self.asset is not None and not isinstance(
            self.asset, RasterAssetRef
        ):
            raise TypeError(
                'texture paint asset must be RasterAssetRef or None'
            )
        if self.mapping not in {'fill', 'fit', 'crop', 'tile'}:
            raise ValueError('unsupported texture mapping')
        object.__setattr__(
            self,
            'scale',
            _float_in_range('texture scale', self.scale, 0.1, 4.0),
        )

    def to_serializable_dict(self) -> dict:
        return {
            'paint_type': self.paint_type,
            'asset': (
                None
                if self.asset is None
                else self.asset.to_serializable_dict()
            ),
            'mapping': self.mapping,
            'scale': self.scale,
        }


GeneratedEffectPaint = Union[SolidPaint, LinearGradientPaint]
EffectPaint = Union[GeneratedEffectPaint, TexturePaint]


def _effect_paint_is_transparent(paint: GeneratedEffectPaint) -> bool:
    return (
        isinstance(paint, LinearGradientPaint)
        and all(stop.opacity == 0.0 for stop in paint.stops)
    )


def effect_paint_fallback_color(
    paint: EffectPaint,
) -> Tuple[int, int, int]:
    """Return the stable RGB used by legacy solid-only boundaries.

    >>> effect_paint_fallback_color(LinearGradientPaint())
    (0, 0, 0)
    """
    if isinstance(paint, SolidPaint):
        return paint.color
    if isinstance(paint, LinearGradientPaint):
        return paint.stops[0].color
    if isinstance(paint, TexturePaint):
        return (0, 0, 0)
    raise TypeError('effect paint requires a typed paint value')


@dataclass(frozen=True)
class StrokeEffect:
    """One immutable positioned Stroke effect.

    Width is the full band relative to font size. Center splits it across the
    glyph edge; Inside and Outside place it wholly on the corresponding side.
    Newly created strokes default to Outside.

    >>> StrokeEffect(width=0.2).effect_type
    'stroke'
    """

    enabled: bool = True
    opacity: float = 1.0
    blend_mode: str = 'normal'
    width: float = 0.1
    paint: GeneratedEffectPaint = field(default_factory=SolidPaint)
    position: str = 'outside'
    effect_type: str = field(init=False, default='stroke')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('stroke enabled must be a bool')
        object.__setattr__(
            self,
            'opacity',
            _float_in_range('stroke opacity', self.opacity, 0.0, 1.0),
        )
        object.__setattr__(
            self,
            'blend_mode',
            _validated_blend_mode('stroke', self.blend_mode),
        )
        object.__setattr__(
            self,
            'width',
            _float_in_range('stroke width', self.width, 0.0),
        )
        if not isinstance(self.paint, (SolidPaint, LinearGradientPaint)):
            raise TypeError('stroke paint must be Solid or LinearGradient')
        if self.position not in {'inside', 'center', 'outside'}:
            raise ValueError('unsupported stroke position')

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'width': self.width,
            'position': self.position,
            'paint': self.paint.to_serializable_dict(),
        }

    def is_neutral(self) -> bool:
        return (
            not self.enabled
            or self.opacity == 0.0
            or self.width == 0.0
            or _effect_paint_is_transparent(self.paint)
        )


@dataclass(frozen=True)
class ShadowEffect:
    """One immutable Drop, Inner, or Long/Extrude shadow.

    Geometry values are relative to the text's maximum font size. Paint uses
    the same block-local Solid or Linear Gradient contract as Stroke and Glow.

    >>> ShadowEffect(shadow_type='long', angle=450).angle
    90.0
    """

    enabled: bool = True
    opacity: float = 1.0
    blend_mode: str = 'normal'
    shadow_type: str = 'drop'
    paint: GeneratedEffectPaint = field(default_factory=SolidPaint)
    angle: float = 0.0
    distance: float = 0.1
    blur: float = 0.15
    spread: float = 0.0
    effect_type: str = field(init=False, default='shadow')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('shadow enabled must be a bool')
        object.__setattr__(
            self,
            'opacity',
            _float_in_range('shadow opacity', self.opacity, 0.0, 1.0),
        )
        object.__setattr__(
            self,
            'blend_mode',
            _validated_blend_mode('shadow', self.blend_mode),
        )
        if self.shadow_type not in {'drop', 'inner', 'long'}:
            raise ValueError('unsupported shadow type')
        if not isinstance(self.paint, (SolidPaint, LinearGradientPaint)):
            raise TypeError('shadow paint must be Solid or LinearGradient')
        angle = _float_in_range(
            'shadow angle', self.angle, -math.inf, math.inf
        )
        object.__setattr__(self, 'angle', angle % 360.0)
        object.__setattr__(
            self,
            'distance',
            _float_in_range(
                'shadow distance',
                self.distance,
                0.0,
                SHADOW_DISTANCE_LIMIT,
            ),
        )
        object.__setattr__(
            self,
            'blur',
            _float_in_range(
                'shadow blur', self.blur, 0.0, SHADOW_BLUR_LIMIT
            ),
        )
        object.__setattr__(
            self,
            'spread',
            _float_in_range(
                'shadow spread', self.spread, 0.0, SHADOW_SPREAD_LIMIT
            ),
        )

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'shadow_type': self.shadow_type,
            'paint': self.paint.to_serializable_dict(),
            'angle': self.angle,
            'distance': self.distance,
            'blur': self.blur,
            'spread': self.spread,
        }

    def is_neutral(self) -> bool:
        return (
            not self.enabled
            or self.opacity == 0.0
            or _effect_paint_is_transparent(self.paint)
        )


@dataclass(frozen=True)
class GlowEffect:
    """One immutable Outer or Inner Glow.

    Geometry values are relative to the text's maximum font size.

    >>> GlowEffect(glow_type='inner', size=0.4).effect_type
    'glow'
    """

    enabled: bool = True
    opacity: float = 1.0
    blend_mode: str = 'normal'
    glow_type: str = 'outer'
    paint: GeneratedEffectPaint = field(
        default_factory=lambda: SolidPaint((255, 255, 255))
    )
    size: float = 0.2
    spread: float = 0.0
    effect_type: str = field(init=False, default='glow')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('glow enabled must be a bool')
        object.__setattr__(
            self,
            'opacity',
            _float_in_range('glow opacity', self.opacity, 0.0, 1.0),
        )
        object.__setattr__(
            self,
            'blend_mode',
            _validated_blend_mode('glow', self.blend_mode),
        )
        if self.glow_type not in {'outer', 'inner'}:
            raise ValueError('unsupported glow type')
        if not isinstance(self.paint, (SolidPaint, LinearGradientPaint)):
            raise TypeError('glow paint must be Solid or LinearGradient')
        object.__setattr__(
            self,
            'size',
            _float_in_range('glow size', self.size, 0.0, SHADOW_BLUR_LIMIT),
        )
        object.__setattr__(
            self,
            'spread',
            _float_in_range(
                'glow spread', self.spread, 0.0, SHADOW_SPREAD_LIMIT
            ),
        )

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'glow_type': self.glow_type,
            'paint': self.paint.to_serializable_dict(),
            'size': self.size,
            'spread': self.spread,
        }

    def is_neutral(self) -> bool:
        return (
            not self.enabled
            or self.opacity == 0.0
            or (self.size == 0.0 and self.spread == 0.0)
            or _effect_paint_is_transparent(self.paint)
        )


@dataclass(frozen=True)
class HollowEffect:
    """Suppress the foreground and interior effects while enabled.

    >>> HollowEffect(enabled=False).is_neutral()
    True
    """

    enabled: bool = True
    effect_type: str = field(init=False, default='hollow')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('hollow enabled must be a bool')

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
        }

    def is_neutral(self) -> bool:
        return not self.enabled


@dataclass(frozen=True)
class TextFillEffect:
    """One Gradient or Texture layer over the canonical text foreground.

    >>> TextFillEffect().paint.paint_type
    'linear_gradient'
    """

    enabled: bool = True
    opacity: float = 1.0
    blend_mode: str = 'normal'
    paint: Union[LinearGradientPaint, TexturePaint] = field(
        default_factory=LinearGradientPaint
    )
    effect_type: str = field(init=False, default='text_fill')

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError('text fill enabled must be a bool')
        object.__setattr__(
            self,
            'opacity',
            _float_in_range('text fill opacity', self.opacity, 0.0, 1.0),
        )
        object.__setattr__(
            self,
            'blend_mode',
            _validated_blend_mode('text fill', self.blend_mode),
        )
        if not isinstance(self.paint, (LinearGradientPaint, TexturePaint)):
            raise TypeError(
                'text fill paint must be LinearGradientPaint or TexturePaint'
            )

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'paint': self.paint.to_serializable_dict(),
        }

    def is_neutral(self) -> bool:
        return (
            not self.enabled
            or (
                isinstance(self.paint, TexturePaint)
                and self.paint.asset is None
            )
        )


@dataclass(frozen=True)
class ImageGenerationRecipe:
    """Saved editor inputs for regenerating one Image effect asset.

    The backend identity is intentionally opaque so projects keep the recipe
    when an optional backend is removed. Rendering never depends on this value.

    >>> ImageGenerationRecipe(context='none').backend
    'llm'
    """

    backend: str = 'llm'
    profile_id: str = ''
    model: str = ''
    context: str = 'source'
    prompt: str = ''

    def __post_init__(self) -> None:
        for name in ('backend', 'profile_id', 'model', 'prompt'):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f'image generation {name} must be a string')
        if not self.backend.strip():
            raise ValueError('image generation backend must not be empty')
        if self.context not in {'source', 'inpainted', 'lettered', 'none'}:
            raise ValueError('unsupported image generation context')

    def to_serializable_dict(self) -> dict:
        return {
            'backend': self.backend,
            'profile_id': self.profile_id,
            'model': self.model,
            'context': self.context,
            'prompt': self.prompt,
        }


@dataclass(frozen=True)
class ImageEffect:
    """One project-owned raster layer in the ordered effect stack.

    >>> ImageEffect().is_neutral()
    True
    >>> ImageEffect(mode='background').effect_type
    'image'
    """

    asset: Optional[RasterAssetRef] = None
    enabled: bool = True
    mode: str = 'foreground'
    generation: ImageGenerationRecipe = field(
        default_factory=ImageGenerationRecipe
    )
    effect_type: str = field(init=False, default='image')

    def __post_init__(self) -> None:
        if self.asset is not None and not isinstance(
            self.asset, RasterAssetRef
        ):
            raise TypeError('image effect asset must be RasterAssetRef or None')
        if not isinstance(self.enabled, bool):
            raise TypeError('image effect enabled must be a bool')
        if self.mode not in {'foreground', 'background'}:
            raise ValueError('unsupported image effect mode')
        if not isinstance(self.generation, ImageGenerationRecipe):
            raise TypeError(
                'image effect generation must be ImageGenerationRecipe'
            )

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'asset': (
                None
                if self.asset is None
                else self.asset.to_serializable_dict()
            ),
            'enabled': self.enabled,
            'mode': self.mode,
            'generation': self.generation.to_serializable_dict(),
        }

    def is_neutral(self) -> bool:
        return not self.enabled or self.asset is None


def _filter_params_tuple(value: object) -> FilterParams:
    """Return sorted, hashable flat JSON scalar filter parameters.

    >>> _filter_params_tuple({'z': 2, 'a': True})
    (('a', True), ('z', 2))
    """
    items = value.items() if isinstance(value, Mapping) else value
    if not isinstance(items, (Sequence, type({}.items()))):
        raise TypeError('filter params must be a flat mapping or item sequence')
    normalized = []
    keys = set()
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise TypeError('filter params must contain key/value pairs')
        key, parameter = item
        if not isinstance(key, str) or not key:
            raise ValueError('filter parameter names must be non-empty strings')
        if key in keys:
            raise ValueError(f'duplicate filter parameter: {key}')
        if not isinstance(parameter, (bool, int, float, str, type(None))):
            raise TypeError(f'filter parameter {key} must be a JSON scalar')
        if isinstance(parameter, float) and not math.isfinite(parameter):
            raise ValueError(f'filter parameter {key} must be finite')
        keys.add(key)
        normalized.append((key, parameter))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class FilterEffect:
    """One repeatable lazy filter request with hashable opaque parameters.

    Plug-in availability is deliberately not checked at this persistence
    boundary, so unknown and newer filter requests survive passive loading.

    >>> hash(FilterEffect('custom:demo', params={'amount': 0.5})) != 0
    True
    """

    filter_id: str
    schema_version: int = 1
    enabled: bool = True
    params: FilterParams = ()
    effect_type: str = field(init=False, default='filter')

    def __post_init__(self) -> None:
        if not isinstance(self.filter_id, str) or not self.filter_id.strip():
            raise ValueError('filter id must be a non-empty string')
        if any(character.isspace() for character in self.filter_id):
            raise ValueError('filter id must not contain whitespace')
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version <= 0
        ):
            raise ValueError('filter schema version must be a positive integer')
        if not isinstance(self.enabled, bool):
            raise TypeError('filter enabled must be a bool')
        object.__setattr__(self, 'schema_version', int(self.schema_version))
        object.__setattr__(self, 'params', _filter_params_tuple(self.params))

    def params_dict(self) -> dict:
        """Return the opaque parameters as a fresh deterministic mapping."""
        return dict(self.params)

    def to_serializable_dict(self) -> dict:
        return {
            'effect_type': self.effect_type,
            'enabled': self.enabled,
            'filter_id': self.filter_id,
            'schema_version': self.schema_version,
            'params': self.params_dict(),
        }

    def is_neutral(self) -> bool:
        return not self.enabled


TextEffect = Union[
    StrokeEffect,
    ShadowEffect,
    GlowEffect,
    HollowEffect,
    TextFillEffect,
    ImageEffect,
    FilterEffect,
]


def effect_phase(effect: TextEffect) -> str:
    """Return the canonical source/bounds phase for one typed effect.

    >>> effect_phase(ShadowEffect(shadow_type='inner'))
    'interior'
    """
    if isinstance(effect, ShadowEffect):
        return (
            'interior'
            if effect.shadow_type == 'inner'
            else 'exterior'
        )
    if isinstance(effect, GlowEffect):
        return 'interior' if effect.glow_type == 'inner' else 'exterior'
    if isinstance(effect, StrokeEffect):
        return 'stroke'
    if isinstance(effect, (HollowEffect, TextFillEffect)):
        return 'foreground'
    if isinstance(effect, ImageEffect):
        return 'image'
    if isinstance(effect, FilterEffect):
        return 'filter'
    raise TypeError('effect_phase requires a typed text effect')


def effect_structure_key(effect: TextEffect) -> object:
    """Return the mixed-selection identity for one stack position.

    >>> effect_structure_key(FilterEffect('builtin:noise', 2))
    ('filter', 'builtin:noise', 2)
    >>> effect_structure_key(TextFillEffect(paint=LinearGradientPaint()))
    ('text_fill', 'linear_gradient')
    """
    if isinstance(effect, FilterEffect):
        return effect.effect_type, effect.filter_id, effect.schema_version
    if isinstance(effect, TextFillEffect):
        return effect.effect_type, effect.paint.paint_type
    return effect.effect_type


@dataclass(frozen=True)
class TextEffectStack:
    """Complete immutable style-owned text-effect value.

    ``effects`` preserves the renderer's topmost-first layer order; the panel
    projects it in bottom-to-top application order. Overall opacity applies
    to the completed item rather than an individual effect.

    >>> stack = with_primary_stroke(TextEffectStack(), width=0.25)
    >>> (len(stack), stack[0].width)
    (1, 0.25)
    """

    overall_opacity: float = 1.0
    effects: Tuple[TextEffect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'overall_opacity',
            _float_in_range(
                'overall opacity', self.overall_opacity, 0.0, 1.0
            ),
        )
        effects = tuple(self.effects)
        if any(
            not isinstance(
                effect,
                (
                    StrokeEffect,
                    ShadowEffect,
                    GlowEffect,
                    HollowEffect,
                    TextFillEffect,
                    ImageEffect,
                    FilterEffect,
                ),
            )
            for effect in effects
        ):
            raise TypeError('text effect stack requires typed effect values')
        if sum(isinstance(effect, HollowEffect) for effect in effects) > 1:
            raise ValueError('text effect stack accepts at most one Hollow')
        object.__setattr__(self, 'effects', effects)

    def __iter__(self) -> Iterator[TextEffect]:
        return iter(self.effects)

    def __len__(self) -> int:
        return len(self.effects)

    def __getitem__(self, index: int) -> TextEffect:
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


def without_project_raster_effects(
    stack: TextEffectStack,
) -> TextEffectStack:
    """Remove project-only raster effects at application-style boundaries.

    Texture Fill and Image entries are removed; portable siblings remain.

    >>> asset = RasterAssetRef('assets/' + 'a' * 64 + '.png')
    >>> stack = TextEffectStack(effects=(
    ...     StrokeEffect(), TextFillEffect(paint=TexturePaint(asset)),
    ...     ImageEffect(asset),
    ... ))
    >>> filtered = without_project_raster_effects(stack)
    >>> tuple(type(effect).__name__ for effect in filtered)
    ('StrokeEffect',)
    """
    if not isinstance(stack, TextEffectStack):
        raise TypeError('project raster filtering requires TextEffectStack')
    effects = tuple(
        effect
        for effect in stack.effects
        if not (
            isinstance(effect, ImageEffect)
            or (
                isinstance(effect, TextFillEffect)
                and isinstance(effect.paint, TexturePaint)
            )
        )
    )
    return stack if effects == stack.effects else replace(stack, effects=effects)


def _unexpected_fields(
    payload: Mapping[str, object], allowed: Sequence[str], label: str
) -> None:
    unexpected = set(payload) - set(allowed)
    if unexpected:
        raise ValueError(f'unsupported {label} fields: {sorted(unexpected)}')


def _coerce_gradient_stop(value: object) -> GradientStop:
    if isinstance(value, GradientStop):
        return value
    if not isinstance(value, dict):
        raise ValueError('gradient stop must be a value or typed payload')
    payload = dict(value)
    _unexpected_fields(
        payload, ('position', 'color', 'opacity'), 'gradient stop'
    )
    return GradientStop(**payload)


def _coerce_effect_paint(value: object) -> EffectPaint:
    if isinstance(value, (SolidPaint, LinearGradientPaint, TexturePaint)):
        return value
    if not isinstance(value, dict):
        raise ValueError('effect paint must be a value or typed payload')
    payload = dict(value)
    paint_type = payload.pop('paint_type', None)
    if paint_type == 'solid':
        _unexpected_fields(payload, ('color',), 'solid paint')
        return SolidPaint(**payload)
    if paint_type == 'linear_gradient':
        _unexpected_fields(
            payload, ('stops', 'angle', 'scale'), 'linear gradient paint'
        )
        if 'stops' in payload:
            stops = payload['stops']
            if not isinstance(stops, (list, tuple)):
                raise ValueError('linear gradient stops must be a sequence')
            payload['stops'] = tuple(
                _coerce_gradient_stop(stop) for stop in stops
            )
        return LinearGradientPaint(**payload)
    if paint_type == 'texture':
        _unexpected_fields(
            payload, ('asset', 'mapping', 'scale'), 'texture paint'
        )
        asset = payload.get('asset')
        payload['asset'] = (
            None if asset is None else coerce_raster_asset_ref(asset)
        )
        return TexturePaint(**payload)
    raise ValueError('unsupported or missing effect paint type')


def _coerce_image_generation_recipe(
    value: object,
) -> ImageGenerationRecipe:
    if isinstance(value, ImageGenerationRecipe):
        return value
    if not isinstance(value, Mapping):
        raise ValueError('image generation recipe must be a typed payload')
    payload = dict(value)
    _unexpected_fields(
        payload,
        ('backend', 'profile_id', 'model', 'context', 'prompt'),
        'Image generation recipe',
    )
    return ImageGenerationRecipe(**payload)


def coerce_text_effect(value: Union[TextEffect, dict]) -> TextEffect:
    """Return a live effect or construct one strict typed payload.

    >>> coerce_text_effect({'effect_type': 'stroke', 'width': 0.2}).width
    0.2
    """
    if isinstance(
        value,
        (
            StrokeEffect,
            ShadowEffect,
            GlowEffect,
            HollowEffect,
            TextFillEffect,
            ImageEffect,
            FilterEffect,
        ),
    ):
        return value
    if not isinstance(value, dict):
        raise ValueError('text effect must be a value or typed payload')
    payload = dict(value)
    effect_type = payload.get('effect_type')
    if effect_type == 'stroke':
        _unexpected_fields(
            payload,
            (
                'effect_type', 'enabled', 'opacity', 'blend_mode', 'width',
                'paint', 'position',
            ),
            'Stroke effect',
        )
        payload.pop('effect_type')
        if 'paint' in payload:
            payload['paint'] = _coerce_effect_paint(payload['paint'])
        # Typed stacks saved before Stroke Position existed were centered.
        payload.setdefault('position', 'center')
        return StrokeEffect(**payload)
    if effect_type == 'shadow':
        _unexpected_fields(
            payload,
            (
                'effect_type', 'enabled', 'opacity', 'blend_mode',
                'shadow_type', 'paint', 'color', 'angle', 'distance',
                'blur', 'spread',
            ),
            'Shadow effect',
        )
        payload.pop('effect_type')
        has_legacy_color = 'color' in payload
        legacy_color = payload.pop('color', None)
        if 'paint' in payload:
            payload['paint'] = _coerce_effect_paint(payload['paint'])
        elif has_legacy_color:
            # Shadow payloads before gradient Fill stored a bare RGB value.
            payload['paint'] = SolidPaint(legacy_color)
        return ShadowEffect(**payload)
    if effect_type == 'glow':
        _unexpected_fields(
            payload,
            (
                'effect_type', 'enabled', 'opacity', 'blend_mode',
                'glow_type', 'paint', 'size', 'spread',
            ),
            'Glow effect',
        )
        payload.pop('effect_type')
        if 'paint' in payload:
            payload['paint'] = _coerce_effect_paint(payload['paint'])
        return GlowEffect(**payload)
    if effect_type == 'hollow':
        _unexpected_fields(
            payload, ('effect_type', 'enabled'), 'Hollow effect'
        )
        payload.pop('effect_type')
        return HollowEffect(**payload)
    if effect_type in {'gradient', 'gradient_overlay'}:
        _unexpected_fields(
            payload,
            ('effect_type', 'enabled', 'opacity', 'blend_mode', 'paint'),
            'Gradient effect',
        )
        payload.pop('effect_type')
        payload.pop('opacity', None)
        if 'paint' in payload:
            payload['paint'] = _coerce_effect_paint(payload['paint'])
        else:
            payload['paint'] = LinearGradientPaint()
        paint = payload.get('paint')
        if not isinstance(paint, LinearGradientPaint):
            raise ValueError(
                'legacy Gradient paint must be LinearGradientPaint'
            )
        return TextFillEffect(**payload)
    if effect_type == 'text_fill':
        _unexpected_fields(
            payload,
            ('effect_type', 'enabled', 'opacity', 'blend_mode', 'paint'),
            'Text Fill effect',
        )
        payload.pop('effect_type')
        if 'paint' not in payload:
            raise ValueError('text fill effect requires paint')
        payload['paint'] = _coerce_effect_paint(payload['paint'])
        return TextFillEffect(**payload)
    if effect_type == 'image':
        _unexpected_fields(
            payload,
            ('effect_type', 'asset', 'enabled', 'mode', 'generation'),
            'Image effect',
        )
        payload.pop('effect_type')
        asset = payload.get('asset')
        payload['asset'] = (
            None if asset is None else coerce_raster_asset_ref(asset)
        )
        if 'generation' in payload:
            payload['generation'] = _coerce_image_generation_recipe(
                payload['generation']
            )
        return ImageEffect(**payload)
    if effect_type == 'filter':
        _unexpected_fields(
            payload,
            (
                'effect_type', 'enabled', 'filter_id',
                'schema_version', 'params',
            ),
            'Filter effect',
        )
        payload.pop('effect_type')
        return FilterEffect(**payload)
    raise ValueError('unsupported or missing text effect type')


def _coerce_filter_effect_passive(payload: Mapping[str, object]) -> FilterEffect:
    """Load generic fields independently without resolving plug-in code."""
    raw = dict(payload)
    unknown = set(raw) - {
        'effect_type', 'enabled', 'filter_id', 'schema_version', 'params'
    }
    if unknown:
        LOGGER.warning(
            'Ignoring unsupported Filter effect fields: %s.', sorted(unknown)
        )
    filter_id = raw.get('filter_id')
    if not isinstance(filter_id, str) or not filter_id.strip():
        raise ValueError('filter id must be a non-empty string')

    enabled = raw.get('enabled', True)
    if not isinstance(enabled, bool):
        LOGGER.warning('Ignoring invalid Filter enabled value; using true.')
        enabled = True
    schema_version = raw.get('schema_version', 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, Integral)
        or schema_version <= 0
    ):
        LOGGER.warning(
            'Ignoring invalid Filter schema version; using version 1.'
        )
        schema_version = 1

    raw_params = raw.get('params', {})
    if not isinstance(raw_params, Mapping):
        LOGGER.warning('Ignoring invalid Filter params container.')
        raw_params = {}
    params = {}
    for key, value in raw_params.items():
        try:
            params.update(_filter_params_tuple(((key, value),)))
        except (TypeError, ValueError) as error:
            LOGGER.warning('Ignoring invalid Filter parameter (%s).', error)
    return FilterEffect(filter_id, schema_version, enabled, params)


def _coerce_image_effect_passive(
    payload: Mapping[str, object],
) -> ImageEffect:
    """Recover Image fields independently at the passive project boundary."""
    raw = dict(payload)
    unknown = set(raw) - {
        'effect_type', 'asset', 'enabled', 'mode', 'generation'
    }
    if unknown:
        LOGGER.warning(
            'Ignoring unsupported Image effect fields: %s.', sorted(unknown)
        )
    asset_value = raw.get('asset')
    try:
        asset = (
            None
            if asset_value is None
            else coerce_raster_asset_ref(asset_value)
        )
    except (TypeError, ValueError) as error:
        LOGGER.warning(
            'Ignoring invalid Image asset reference (%s).', error
        )
        asset = None
    enabled = raw.get('enabled', True)
    if not isinstance(enabled, bool):
        LOGGER.warning('Ignoring invalid Image enabled value; using true.')
        enabled = True
    mode = raw.get('mode', 'foreground')
    if mode not in {'foreground', 'background'}:
        LOGGER.warning('Ignoring invalid Image mode; using foreground.')
        mode = 'foreground'

    defaults = ImageGenerationRecipe()
    recipe_value = raw.get('generation')
    if recipe_value is None:
        recipe = defaults
    elif not isinstance(recipe_value, Mapping):
        LOGGER.warning(
            'Ignoring invalid Image generation recipe; using defaults.'
        )
        recipe = defaults
    else:
        recipe_raw = dict(recipe_value)
        recipe_unknown = set(recipe_raw) - {
            'backend', 'profile_id', 'model', 'context', 'prompt'
        }
        if recipe_unknown:
            LOGGER.warning(
                'Ignoring unsupported Image generation recipe fields: %s.',
                sorted(recipe_unknown),
            )
        recovered = {}
        for name in ('backend', 'profile_id', 'model', 'prompt'):
            value = recipe_raw.get(name, getattr(defaults, name))
            if not isinstance(value, str) or (
                name == 'backend' and not value.strip()
            ):
                LOGGER.warning(
                    'Ignoring invalid Image generation %s; using default.',
                    name,
                )
                value = getattr(defaults, name)
            recovered[name] = value
        context = recipe_raw.get('context', defaults.context)
        if context not in {'source', 'inpainted', 'lettered', 'none'}:
            LOGGER.warning(
                'Ignoring invalid Image generation context; using source.'
            )
            context = defaults.context
        recovered['context'] = context
        recipe = ImageGenerationRecipe(**recovered)
    return ImageEffect(asset, enabled, mode, recipe)


def _coerce_text_effect_passive(value: object) -> TextEffect:
    """Recover an invalid persisted blend mode without losing the effect."""
    if not isinstance(value, Mapping):
        return coerce_text_effect(value)
    payload = dict(value)
    effect_type = payload.get('effect_type')
    if (
        isinstance(effect_type, str)
        and effect_type in {
            'stroke', 'shadow', 'glow', 'text_fill',
            'gradient', 'gradient_overlay',
        }
        and 'blend_mode' in payload
        and payload['blend_mode'] not in TEXT_EFFECT_BLEND_MODES
    ):
        LOGGER.warning(
            'Ignoring invalid %s blend mode (%r); using normal.',
            effect_type,
            payload['blend_mode'],
        )
        payload['blend_mode'] = 'normal'
    return coerce_text_effect(payload)


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
    hollow_loaded = False
    for index, raw_effect in enumerate(raw_effects):
        try:
            effect = (
                _coerce_filter_effect_passive(raw_effect)
                if isinstance(raw_effect, Mapping)
                and raw_effect.get('effect_type') == 'filter'
                else _coerce_image_effect_passive(raw_effect)
                if isinstance(raw_effect, Mapping)
                and raw_effect.get('effect_type') == 'image'
                else _coerce_text_effect_passive(raw_effect)
            )
            if isinstance(effect, HollowEffect):
                if hollow_loaded:
                    raise ValueError('text effect stack accepts at most one Hollow')
                hollow_loaded = True
            effects.append(effect)
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


def generated_effect_insertion_index(
    effects: Sequence[TextEffect],
) -> int:
    """Insert after movable values and before only trailing structural state.

    Legacy stacks may retain Text Fill/Hollow in the middle, but their raw
    position must not change the projected movable order.

    >>> generated_effect_insertion_index((
    ...     FilterEffect('builtin:noise'), TextFillEffect(), GlowEffect()
    ... ))
    3
    >>> generated_effect_insertion_index((
    ...     FilterEffect('builtin:noise'), GlowEffect(), TextFillEffect()
    ... ))
    2
    """
    insertion = len(effects)
    while insertion > 0 and isinstance(
        effects[insertion - 1], (HollowEffect, TextFillEffect)
    ):
        insertion -= 1
    return insertion


def ensure_primary_stroke(stack: TextEffectStack) -> TextEffectStack:
    """Return ``stack`` with an applied-last primary Stroke when absent."""
    if not isinstance(stack, TextEffectStack):
        raise TypeError('ensure_primary_stroke requires TextEffectStack')
    if primary_stroke(stack) is not None:
        return stack
    effects = list(stack.effects)
    effects.insert(0, StrokeEffect())
    return replace(stack, effects=tuple(effects))


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
    """Copy non-Stroke style state while retaining every target Stroke.

    Source non-Stroke and target Stroke order are each preserved. Strokes land
    after existing movable values and before only trailing structural state,
    without trusting raw legacy Text Fill/Hollow positions.

    >>> target = TextEffectStack(effects=(StrokeEffect(width=0.4),))
    >>> source = TextEffectStack(overall_opacity=0.6)
    >>> with_non_stroke_effects(target, source).effects == target.effects
    True
    """
    if not isinstance(stack, TextEffectStack) or not isinstance(
        source, TextEffectStack
    ):
        raise TypeError('with_non_stroke_effects requires TextEffectStack')
    target_strokes = tuple(
        effect for effect in stack.effects if isinstance(effect, StrokeEffect)
    )
    source_non_strokes = [
        effect
        for effect in source.effects
        if not isinstance(effect, StrokeEffect)
    ]
    insertion = generated_effect_insertion_index(source_non_strokes)
    effects = list(source_non_strokes)
    effects[insertion:insertion] = target_strokes
    if (
        stack.overall_opacity == source.overall_opacity
        and stack.effects == tuple(effects)
    ):
        return stack
    return replace(
        stack,
        overall_opacity=source.overall_opacity,
        effects=tuple(effects),
    )


def hollow_effect(stack: TextEffectStack) -> Optional[HollowEffect]:
    """Return the stack's structural Hollow value, if present."""
    if not isinstance(stack, TextEffectStack):
        raise TypeError('hollow_effect requires TextEffectStack')
    return next(
        (
            effect
            for effect in stack.effects
            if isinstance(effect, HollowEffect)
        ),
        None,
    )
