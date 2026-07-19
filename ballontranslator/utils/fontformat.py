from typing import NamedTuple, Union
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


class TextTransform(NamedTuple):
    """Canonical post-layout box transform plus glyph-local slant."""

    horizontal_scale: float
    vertical_scale: float
    slant_angle: float
    glyph_slant_angle: float


def normalize_text_transform_value(value: float, minimum: float, maximum: float) -> float:
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
) -> TextTransform:
    """Normalize the canonical four-component text transform.

    Existing three-argument callers remain source-compatible and receive a
    neutral glyph slant.

    >>> tuple(normalize_text_transform(1.23456789, 0.01, -90))
    (1.234568, 0.1, -85.0, 0.0)
    """
    return TextTransform(
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

    # Post-layout visual geometry. These factors never alter document fonts or
    # wrapping; TextBlock.fontformat is their sole persistent owner.
    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    slant_angle: float = 0.0
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
        (
            self.horizontal_scale,
            self.vertical_scale,
            self.slant_angle,
            self.glyph_slant_angle,
        ) = normalize_text_transform(
            self.horizontal_scale,
            self.vertical_scale,
            self.slant_angle,
            self.glyph_slant_angle,
        )
        self.deprecated_attributes = {}

    @property
    def text_transform(self) -> TextTransform:
        return TextTransform(
            self.horizontal_scale,
            self.vertical_scale,
            self.slant_angle,
            self.glyph_slant_angle,
        )

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
