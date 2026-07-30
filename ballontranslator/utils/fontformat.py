from dataclasses import (
    asdict,
    dataclass,
    field as dataclass_field,
    fields,
    replace,
)
import enum
import math
import re
import copy
from typing import ClassVar

import numpy as np

from . import shared
from .logger import logger as LOGGER
from .structures import Union, List, Config, field, nested_dataclass


TEXT_TRANSFORM_SCALE_MIN = 0.1
TEXT_TRANSFORM_SCALE_MAX = 4.0
TEXT_TRANSFORM_BOX_SLANT_MIN = -85.0
TEXT_TRANSFORM_BOX_SLANT_MAX = 85.0
TEXT_TRANSFORM_GLYPH_SLANT_MIN = -45.0
TEXT_TRANSFORM_GLYPH_SLANT_MAX = 45.0
TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MIN = 0.0
TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MAX = 0.8
TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MIN = -180.0
TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MAX = 180.0
TEXT_TRANSFORM_CURVATURE_MIN = -1.0
TEXT_TRANSFORM_CURVATURE_MAX = 1.0
TEXT_TRANSFORM_PRECISION = 6


def _transform_value_field_names(transform) -> tuple:
    """Return constructor fields, excluding derived fields such as the type."""
    return tuple(field.name for field in fields(transform) if field.init)


@dataclass(frozen=True)
class TextTransform:
    """Immutable base value for a persisted text-transform variant.

    Subclasses expose stable component names and normalization. Persistence
    stores ``transform_type`` with the variant-specific component payload.

    >>> SlantTextTransform().transform_type
    'slant'
    """

    transform_type: str = dataclass_field(init=False, default='base')
    # ``nonlinear`` means that QTransform cannot represent the operation and
    # the completed text surface must be inverse-warped instead.
    is_nonlinear: ClassVar[bool] = False

    def normalized(self) -> "TextTransform":
        raise NotImplementedError

    def with_value(self, name: str, value: float) -> "TextTransform":
        if name not in _transform_value_field_names(self):
            raise ValueError(
                f'unknown {self.transform_type} transform field {name}'
            )
        return replace(self, **{name: value}).normalized()

    def is_neutral(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class SlantTextTransform(TextTransform):
    """Affine box scale and shear applied in the ordered geometry stack."""

    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    slant_angle: float = 0.0
    transform_type: str = dataclass_field(init=False, default='slant')

    def normalized(self) -> "SlantTextTransform":
        return normalize_text_transform(
            self.horizontal_scale,
            self.vertical_scale,
            self.slant_angle,
        )

    def is_neutral(self) -> bool:
        return self == SlantTextTransform()


@dataclass(frozen=True)
class PerspectiveTextTransform(TextTransform):
    """Native projective depth transform around the text-box center."""

    strength: float = 0.0
    direction: float = 0.0
    transform_type: str = dataclass_field(init=False, default='perspective')

    def normalized(self) -> "PerspectiveTextTransform":
        return PerspectiveTextTransform(
            normalize_text_transform_value(
                self.strength,
                TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MIN,
                TEXT_TRANSFORM_PERSPECTIVE_STRENGTH_MAX,
            ),
            normalize_text_transform_value(
                self.direction,
                TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MIN,
                TEXT_TRANSFORM_PERSPECTIVE_DIRECTION_MAX,
            ),
        )

    def is_neutral(self) -> bool:
        return self.strength == 0.0


@dataclass(frozen=True)
class CurvatureTextTransform(TextTransform):
    """Signed circular bend applied to the completed text surface."""

    curvature: float = 0.0
    transform_type: str = dataclass_field(init=False, default='curvature')
    is_nonlinear: ClassVar[bool] = True

    def normalized(self) -> "CurvatureTextTransform":
        return CurvatureTextTransform(
            normalize_text_transform_value(
                self.curvature,
                TEXT_TRANSFORM_CURVATURE_MIN,
                TEXT_TRANSFORM_CURVATURE_MAX,
            )
        )

    def is_neutral(self) -> bool:
        return self.curvature == 0.0


@dataclass(frozen=True)
class TextTransformStack:
    """Immutable ordered text-geometry operations.

    Empty means no geometry transform. Neutral entries remain present for the
    editor but are skipped by the runtime compiler.

    >>> stack = TextTransformStack((CurvatureTextTransform(0.5),))
    >>> stack.has_nonlinear
    True
    """

    transforms: tuple[TextTransform, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'transforms',
            tuple(coerce_text_transform(value) for value in self.transforms),
        )

    def __iter__(self):
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, index):
        return self.transforms[index]

    def is_neutral(self) -> bool:
        return all(transform.is_neutral() for transform in self.transforms)

    @property
    def has_nonlinear(self) -> bool:
        return any(
            not transform.is_neutral() and transform.is_nonlinear
            for transform in self.transforms
        )


@dataclass(frozen=True)
class TextTransformState:
    """Complete immutable state edited by the transform undo command.

    Geometry operations stay ordered while Glyph Slant remains one layout
    effect applied before that geometry.

    >>> TextTransformState().glyph_slant_angle
    0.0
    """

    stack: TextTransformStack = dataclass_field(
        default_factory=TextTransformStack
    )
    glyph_slant_angle: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'stack', coerce_text_transform_stack(self.stack)
        )
        object.__setattr__(
            self,
            'glyph_slant_angle',
            normalize_text_transform_value(
                self.glyph_slant_angle,
                TEXT_TRANSFORM_GLYPH_SLANT_MIN,
                TEXT_TRANSFORM_GLYPH_SLANT_MAX,
            ),
        )


def normalize_text_transform_value(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite, clamped canonical text-transform component.

    Persistence validates stored values before normalization; this pure helper
    defines the canonical value shared by the model, UI, and undo commands.

    >>> normalize_text_transform_value(4.5, 0.1, 4.0)
    4.0
    >>> normalize_text_transform_value(-0.0, -45.0, 45.0)
    0.0
    >>> normalize_text_transform_value(float("nan"), 0.1, 4.0)
    Traceback (most recent call last):
    ...
    ValueError: text transform values must be finite numbers
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError("text transform values must be finite numbers")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("text transform values must be finite numbers")
    value = round(min(max(value, minimum), maximum), TEXT_TRANSFORM_PRECISION)
    return 0.0 if value == 0.0 else value


def normalize_text_transform(
    horizontal_scale: float,
    vertical_scale: float,
    slant_angle: float,
) -> SlantTextTransform:
    """Normalize the canonical affine Slant operation.

    >>> normalize_text_transform(1.23456789, 0.01, -90)
    SlantTextTransform(transform_type='slant', horizontal_scale=1.234568, vertical_scale=0.1, slant_angle=-85.0)
    """
    return SlantTextTransform(
        normalize_text_transform_value(
            horizontal_scale, TEXT_TRANSFORM_SCALE_MIN, TEXT_TRANSFORM_SCALE_MAX
        ),
        normalize_text_transform_value(
            vertical_scale, TEXT_TRANSFORM_SCALE_MIN, TEXT_TRANSFORM_SCALE_MAX
        ),
        normalize_text_transform_value(
            slant_angle,
            TEXT_TRANSFORM_BOX_SLANT_MIN,
            TEXT_TRANSFORM_BOX_SLANT_MAX,
        ),
    )


TEXT_TRANSFORM_TYPES = {
    'slant': SlantTextTransform,
    'perspective': PerspectiveTextTransform,
    'curvature': CurvatureTextTransform,
}


def create_text_transform(transform_type: str) -> TextTransform:
    """Create the neutral initial value for a registered transform type.

    UI-selectable variants must provide constructor defaults. Persisted
    payloads may still supply required variant fields through
    :func:`coerce_text_transform`.

    >>> create_text_transform('slant')
    SlantTextTransform(transform_type='slant', horizontal_scale=1.0, vertical_scale=1.0, slant_angle=0.0)
    """
    transform_class = TEXT_TRANSFORM_TYPES.get(transform_type)
    if transform_class is None:
        raise ValueError(f'unsupported text transform type {transform_type}')
    return transform_class().normalized()


def coerce_text_transform(value: Union[TextTransform, dict]) -> TextTransform:
    """Normalize a live value or construct a canonical persisted payload.

    >>> transform = coerce_text_transform(
    ...     {'transform_type': 'slant', 'slant_angle': 5}
    ... )
    >>> transform.slant_angle
    5.0
    >>> coerce_text_transform(
    ...     {'transform_type': 'slant', 'horizontal_scale': 5}
    ... )
    Traceback (most recent call last):
    ...
    ValueError: persisted slant transform values must be canonical
    """
    if isinstance(value, TextTransform):
        return value.normalized()
    if not isinstance(value, dict):
        raise ValueError('text transform must be a value or typed payload')
    payload = dict(value)
    if 'transform_type' not in payload:
        raise ValueError('text transform payload requires transform_type')
    transform_type = payload.pop('transform_type')
    transform_class = TEXT_TRANSFORM_TYPES.get(transform_type)
    if transform_class is None:
        raise ValueError(f'unsupported text transform type {transform_type}')
    value_fields = _transform_value_field_names(transform_class)
    unexpected = set(payload) - set(value_fields)
    if unexpected:
        raise ValueError(
            f'unsupported {transform_type} transform fields: {sorted(unexpected)}'
        )
    transform = transform_class(**payload)
    normalized = transform.normalized()
    if transform != normalized:
        raise ValueError(
            f'persisted {transform_type} transform values must be canonical'
        )
    return normalized


def coerce_text_transform_stack(value) -> TextTransformStack:
    """Return one canonical ordered stack and reject the old single payload.

    >>> coerce_text_transform_stack([
    ...     {'transform_type': 'curvature', 'curvature': 0.5},
    ... ])
    TextTransformStack(transforms=(CurvatureTextTransform(transform_type='curvature', curvature=0.5),))
    >>> coerce_text_transform_stack({'transform_type': 'curvature'})
    Traceback (most recent call last):
    ...
    ValueError: text transform stack must be an ordered list
    """
    if isinstance(value, TextTransformStack):
        return value
    if not isinstance(value, (list, tuple)):
        raise ValueError('text transform stack must be an ordered list')
    return TextTransformStack(tuple(value))


def pt2px(pt, to_int=False) -> float:
    if to_int:
        return int(round(pt * shared.LDPI / 72.))
    else:
        return pt * shared.LDPI / 72.

def px2pt(px) -> float:
    return px / shared.LDPI * 72.


class LineSpacingType(enum.IntEnum):
    Proportional = 0
    Distance = 1


class TextAlignment(enum.IntEnum):
    Left = 0
    Center = 1
    Right = 2


fontweight_qt5_to_qt6 = {0: 100, 12: 200, 25: 300, 50: 400, 57: 500, 63: 600, 75: 700, 81: 800, 87: 900}
fontweight_qt6_to_qt5 = {100: 0, 200: 12, 300: 25, 400: 50, 500: 57, 600: 63, 700: 75, 800: 81, 900: 87}

fontweight_pattern = re.compile(r'font-weight:(\d+)', re.DOTALL)

def fix_fontweight_qt(weight: Union[str, int]):

    def _fix_html_fntweight(matched):
        weight = int(matched.group(1))
        return f'font-weight:{fix_fontweight_qt(weight)}'

    if weight is None:
        return None
    if isinstance(weight, int):
        if shared.FLAG_QT6 and weight < 100:
            if weight in fontweight_qt5_to_qt6:
                weight = fontweight_qt5_to_qt6[weight]
        if not shared.FLAG_QT6 and weight >= 100:
            if weight in fontweight_qt6_to_qt5:
                weight = fontweight_qt6_to_qt5[weight]
    if isinstance(weight, str):
        weight = fontweight_pattern.sub(lambda matched: _fix_html_fntweight(matched), weight)
    return weight


@nested_dataclass
class FontFormat(Config):

    font_family: str = shared.DEFAULT_FONT_FAMILY # to always apply shared.DEFAULT_FONT_FAMILY
    font_size: float = 24
    stroke_width: float = 0.
    frgb: List = field(default_factory=lambda: [0, 0, 0])
    srgb: List = field(default_factory=lambda: [0, 0, 0])
    bold: bool = False
    underline: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    font_weight: int = None
    line_spacing: float = 1.2
    letter_spacing: float = 1.15
    opacity: float = 1.
    shadow_radius: float = 0.
    shadow_strength: float = 1.
    shadow_color: List = field(default_factory=lambda: [0, 0, 0])
    shadow_offset: List = field(default_factory=lambda: [0., 0.])
    gradient_enabled: bool = False
    gradient_start_color: List = field(default_factory=lambda: [0, 0, 0])
    gradient_end_color: List = field(default_factory=lambda: [255, 255, 255])
    gradient_angle: float = 0.
    gradient_size: float = 1.0
    _style_name: str = ''
    line_spacing_type: int = LineSpacingType.Proportional

    # Direct in-memory owner; persistence stores an ordered list of payloads.
    text_transform: Union[TextTransformStack, List] = field(
        default_factory=TextTransformStack
    )
    glyph_slant_angle: float = 0.0

    deprecated_attributes: dict = field(default_factory = lambda: dict())

    @property
    def size_pt(self):
        return px2pt(self.font_size)

    def __post_init__(self):
        da = self.deprecated_attributes
        if len(da) > 0:
            if 'size' in da:
                self.font_size = pt2px(da['size'])
            if 'weight' in da:
                self.font_weight = da['weight']
            if 'family' in da:
                self.font_family = da['family']

        self.font_weight = fix_fontweight_qt(self.font_weight)
        if not isinstance(self.text_transform, TextTransformStack):
            if isinstance(self.text_transform, (list, tuple)):
                transforms = []
                for index, value in enumerate(self.text_transform):
                    try:
                        transforms.append(coerce_text_transform(value))
                    except (TypeError, ValueError) as error:
                        LOGGER.warning(
                            'Ignoring invalid text transform config at index '
                            '%s (%s).',
                            index,
                            error,
                        )
                self.text_transform = TextTransformStack(tuple(transforms))
            else:
                LOGGER.warning(
                    'Ignoring invalid text transform stack (%r); '
                    'using an empty transform stack.',
                    self.text_transform,
                )
                self.text_transform = TextTransformStack()
        try:
            self.glyph_slant_angle = normalize_text_transform_value(
                self.glyph_slant_angle,
                TEXT_TRANSFORM_GLYPH_SLANT_MIN,
                TEXT_TRANSFORM_GLYPH_SLANT_MAX,
            )
        except ValueError as error:
            LOGGER.warning(
                'Ignoring invalid Glyph Slant config (%s); using 0.',
                error,
            )
            self.glyph_slant_angle = 0.0
        self.deprecated_attributes = {}

    def to_serializable_dict(self) -> dict:
        """Return config/project data with a typed transform payload."""
        serialized = vars(self).copy()
        serialized['text_transform'] = [
            asdict(transform) for transform in self.text_transform
        ]
        return serialized

    def deepcopy(self):
        fmt_copyed: FontFormat = None
        fmt_copyed = copy.deepcopy(self)
        return fmt_copyed

    def merge(self, target: Config, compare: bool = False):
        if id(self) == id(target):
            return set()
        tgt_keys = target.annotations_set()
        updated_keys = set()
        for key in tgt_keys:
            if not hasattr(self, key):
                continue
            if compare:
                if key != '_style_name':
                    if isinstance(target[key], np.ndarray):
                        is_diff = np.any(self[key] != target[key])
                    else:
                        is_diff = self[key] != target[key]
                    if is_diff:
                        self.update(key, copy.deepcopy(target[key]))
                        updated_keys.add(key)
            else:
                self.update(key, copy.deepcopy(target[key]))
        return updated_keys

    def foreground_color(self):
        return [int(round(x)) for x in self.frgb]
    
    def stroke_color(self):
        return [int(round(x)) for x in self.srgb]
