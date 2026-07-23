from dataclasses import (
    asdict,
    dataclass,
    field as dataclass_field,
    fields,
    replace,
)
from typing import Union
import enum
import math
import re
import copy

import numpy as np

from . import shared
from .structures import Tuple, Union, List, Dict, Config, field, nested_dataclass


TEXT_TRANSFORM_SCALE_MIN = 0.1
TEXT_TRANSFORM_SCALE_MAX = 4.0
TEXT_TRANSFORM_BOX_SLANT_MIN = -85.0
TEXT_TRANSFORM_BOX_SLANT_MAX = 85.0
TEXT_TRANSFORM_GLYPH_SLANT_MIN = -45.0
TEXT_TRANSFORM_GLYPH_SLANT_MAX = 45.0
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
    """Current affine box transform plus glyph-local slant."""

    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    slant_angle: float = 0.0
    glyph_slant_angle: float = 0.0
    transform_type: str = dataclass_field(init=False, default='slant')

    def normalized(self) -> "SlantTextTransform":
        return normalize_text_transform(
            self.horizontal_scale,
            self.vertical_scale,
            self.slant_angle,
            self.glyph_slant_angle,
        )

    def is_neutral(self) -> bool:
        return self == SlantTextTransform()


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
    glyph_slant_angle: float = 0.0,
) -> SlantTextTransform:
    """Normalize the canonical four-component text transform.

    Existing three-argument callers remain source-compatible and receive a
    neutral glyph slant.

    >>> normalize_text_transform(1.23456789, 0.01, -90)
    SlantTextTransform(transform_type='slant', horizontal_scale=1.234568, vertical_scale=0.1, slant_angle=-85.0, glyph_slant_angle=0.0)
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
        normalize_text_transform_value(
            glyph_slant_angle,
            TEXT_TRANSFORM_GLYPH_SLANT_MIN,
            TEXT_TRANSFORM_GLYPH_SLANT_MAX,
        ),
    )


TEXT_TRANSFORM_TYPES = {
    SlantTextTransform().transform_type: SlantTextTransform,
}


def coerce_text_transform(value=None, **flat_values) -> TextTransform:
    """Return a normalized transform from direct or persisted flat data.

    Old flat configs are consumed during ordinary ``FontFormat`` construction;
    no migration pass is required.

    >>> coerce_text_transform(horizontal_scale=2).horizontal_scale
    2.0
    >>> transform = coerce_text_transform(
    ...     {'transform_type': 'slant', 'slant_angle': 5}
    ... )
    >>> transform.slant_angle
    5.0
    """
    if isinstance(value, TextTransform):
        value_fields = _transform_value_field_names(value)
        if flat_values:
            updates = {
                name: component
                for name, component in flat_values.items()
                if name in value_fields
            }
            value = replace(value, **updates)
        return value.normalized()
    payload = {} if value is None else dict(value)
    transform_type = payload.pop('transform_type', 'slant')
    transform_class = TEXT_TRANSFORM_TYPES.get(transform_type)
    if transform_class is None:
        raise ValueError(f'unsupported text transform type {transform_type}')
    value_fields = _transform_value_field_names(transform_class)
    for name in value_fields:
        if name in flat_values:
            payload[name] = flat_values[name]
    unexpected = set(payload) - set(value_fields)
    if unexpected:
        raise ValueError(
            f'unsupported {transform_type} transform fields: {sorted(unexpected)}'
        )
    return transform_class(**payload).normalized()


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

    # Direct in-memory owner. Construction also accepts the previous flat
    # quartet through ``deprecated_attributes`` below.
    text_transform: Union[TextTransform, dict] = field(
        default_factory=SlantTextTransform
    )

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
        flat_transform = {
            name: da.pop(name)
            for name in _transform_value_field_names(SlantTextTransform)
            if name in da
        }
        self.text_transform = coerce_text_transform(
            self.text_transform,
            **flat_transform,
        )
        self.deprecated_attributes = {}

    def to_serializable_dict(self) -> dict:
        """Return config/project data with a typed transform payload."""
        serialized = vars(self).copy()
        serialized['text_transform'] = asdict(self.text_transform)
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
