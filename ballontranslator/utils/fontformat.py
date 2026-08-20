from dataclasses import (
    asdict,
    dataclass,
    field as dataclass_field,
    fields,
    replace,
)
import enum
import re
import copy
from typing import ClassVar, Iterator, Sequence

import numpy as np

from . import shared
from .logger import logger as LOGGER
from .structures import Union, List, Config, field, nested_dataclass


TEXT_TRANSFORM_SCALE_MIN = 0.1
TEXT_TRANSFORM_SCALE_MAX = 4.0
TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN = -85.0
TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX = 85.0
TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN = -89.0
TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX = 89.0
TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN = -180.0
TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX = 180.0
TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MIN = 0.0
TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MAX = 0.8
TEXT_TRANSFORM_GLYPH_SLANT_MIN = -45.0
TEXT_TRANSFORM_GLYPH_SLANT_MAX = 45.0
TEXT_TRANSFORM_BEND_MIN = -1.0
TEXT_TRANSFORM_BEND_MAX = 1.0
TEXT_TRANSFORM_SINE_FREQUENCY_MIN = 0
TEXT_TRANSFORM_SINE_FREQUENCY_MAX = 64
TEXT_TRANSFORM_SINE_PHASE_MIN = 0.0
TEXT_TRANSFORM_SINE_PHASE_MAX = 1.0
TEXT_TRANSFORM_SINE_AMPLITUDE_MIN = 0.0
TEXT_TRANSFORM_SINE_AMPLITUDE_MAX = 1.0
TEXT_TRANSFORM_GRID_DIVISION_MIN = 1
TEXT_TRANSFORM_GRID_DIVISION_MAX = 32
TEXT_TRANSFORM_PRECISION = 6


def _transform_value_field_names(
    transform: Union["TextTransform", type["TextTransform"]],
) -> tuple[str, ...]:
    """Return constructor fields, excluding derived fields such as the type."""
    return tuple(field.name for field in fields(transform) if field.init)


@dataclass(frozen=True)
class TextTransform:
    """Immutable base value for a persisted text-transform variant.

    Subclasses expose stable component names. Persistence stores
    ``transform_type`` with the variant-specific component payload.

    >>> ProjectiveTextTransform().transform_type
    'projective'
    """

    transform_type: str = dataclass_field(init=False, default='base')
    # ``nonlinear`` means that QTransform cannot represent the operation and
    # the completed text surface must be inverse-warped instead.
    is_nonlinear: ClassVar[bool] = False

    def with_value(self, name: str, value: float) -> "TextTransform":
        if name not in _transform_value_field_names(self):
            raise ValueError(
                f'unknown {self.transform_type} transform field {name}'
            )
        return replace(self, **{name: value})

    def is_neutral(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ProjectiveTextTransform(TextTransform):
    """One native projective stage for affine and planar 3D controls.

    X and Y stop short of edge-on because a projected flat plane is singular
    at exactly 90 degrees.

    >>> ProjectiveTextTransform(rotation_x=45).rotation_x
    45
    """

    horizontal_scale: float = 1.0
    vertical_scale: float = 1.0
    horizontal_slant: float = 0.0
    vertical_slant: float = 0.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    perspective: float = 0.0
    transform_type: str = dataclass_field(init=False, default='projective')

    def is_neutral(self) -> bool:
        return (
            self.horizontal_scale == 1.0
            and self.vertical_scale == 1.0
            and self.horizontal_slant == 0.0
            and self.vertical_slant == 0.0
            and self.rotation_x == 0.0
            and self.rotation_y == 0.0
            and self.rotation_z == 0.0
        )


@dataclass(frozen=True)
class BendTextTransform(TextTransform):
    """Signed circular bend applied to the completed text surface."""

    bend: float = 0.0
    transform_type: str = dataclass_field(init=False, default='bend')
    is_nonlinear: ClassVar[bool] = True

    def is_neutral(self) -> bool:
        return self.bend == 0.0


@dataclass(frozen=True)
class SineTextTransform(TextTransform):
    """Two ordered sine shears over the completed text surface.

    Frequencies count half-waves. The x-axis wave is applied first so the
    paired mappings remain exactly invertible at every supported value.

    >>> SineTextTransform().is_neutral()
    False
    >>> SineTextTransform(frequency_x=0).is_neutral()
    True
    """

    frequency_x: int = 2
    frequency_y: int = 0
    phase_x: float = 0.0
    phase_y: float = 0.0
    amplitude_x: float = 0.1
    amplitude_y: float = 0.1
    transform_type: str = dataclass_field(init=False, default='sine')
    is_nonlinear: ClassVar[bool] = True

    def is_neutral(self) -> bool:
        return (
            self.frequency_x == 0
            or self.amplitude_x == 0.0
        ) and (
            self.frequency_y == 0
            or self.amplitude_y == 0.0
        )


def _default_grid_control_points(horizontal: int, vertical: int) -> tuple:
    return tuple(
        (
            round(column / horizontal, TEXT_TRANSFORM_PRECISION),
            round(row / vertical, TEXT_TRANSFORM_PRECISION),
        )
        for row in range(vertical + 1)
        for column in range(horizontal + 1)
    )


def _interpolate_grid_point_bilinear(
    points: tuple,
    horizontal: int,
    vertical: int,
    x: float,
    y: float,
) -> tuple:
    scaled_x = min(max(x, 0.0), 1.0) * horizontal
    scaled_y = min(max(y, 0.0), 1.0) * vertical
    column = min(int(scaled_x), horizontal - 1)
    row = min(int(scaled_y), vertical - 1)
    local_x = scaled_x - column
    local_y = scaled_y - row
    stride = horizontal + 1
    top_left = points[row * stride + column]
    top_right = points[row * stride + column + 1]
    bottom_left = points[(row + 1) * stride + column]
    bottom_right = points[(row + 1) * stride + column + 1]
    return tuple(
        (1.0 - local_y)
        * ((1.0 - local_x) * top_left[axis] + local_x * top_right[axis])
        + local_y
        * ((1.0 - local_x) * bottom_left[axis] + local_x * bottom_right[axis])
        for axis in range(2)
    )


def _resample_grid_control_points(
    points: tuple,
    old_horizontal: int,
    old_vertical: int,
    new_horizontal: int,
    new_vertical: int,
) -> tuple:
    return tuple(
        _interpolate_grid_point_bilinear(
            points,
            old_horizontal,
            old_vertical,
            column / new_horizontal,
            row / new_vertical,
        )
        for row in range(new_vertical + 1)
        for column in range(new_horizontal + 1)
    )


@dataclass(frozen=True)
class GridTextTransform(TextTransform):
    """Free-form grid deformation stored in normalized logical coordinates.

    Division counts describe cells, so the neutral 1 by 1 grid has four
    corner handles.

    >>> len(GridTextTransform().control_points)
    4
    >>> GridTextTransform(horizontal_divisions=2).is_neutral()
    True
    """

    horizontal_divisions: int = 1
    vertical_divisions: int = 1
    interpolation: str = 'bilinear'
    control_points: tuple = ()
    transform_type: str = dataclass_field(init=False, default='grid')
    is_nonlinear: ClassVar[bool] = True

    def __post_init__(self) -> None:
        points = self.control_points or _default_grid_control_points(
            self.horizontal_divisions,
            self.vertical_divisions,
        )
        object.__setattr__(
            self,
            'control_points',
            tuple(tuple(point) for point in points),
        )

    def with_value(
        self, name: str, value: Union[int, float, str]
    ) -> "GridTextTransform":
        if name in {'horizontal_divisions', 'vertical_divisions'}:
            horizontal = (
                value
                if name == 'horizontal_divisions'
                else self.horizontal_divisions
            )
            vertical = (
                value
                if name == 'vertical_divisions'
                else self.vertical_divisions
            )
            points = _resample_grid_control_points(
                self.control_points,
                self.horizontal_divisions,
                self.vertical_divisions,
                horizontal,
                vertical,
            )
            return GridTextTransform(
                horizontal,
                vertical,
                self.interpolation,
                points,
            )
        if name == 'interpolation':
            return replace(self, interpolation=value)
        return super().with_value(name, value)

    def with_control_points(
        self, points: Sequence[Sequence[float]]
    ) -> "GridTextTransform":
        return replace(self, control_points=tuple(points))

    def is_neutral(self) -> bool:
        return self.control_points == _default_grid_control_points(
            self.horizontal_divisions,
            self.vertical_divisions,
        )


@dataclass(frozen=True)
class TextTransformStack:
    """Complete immutable text-transform value.

    ``transforms`` contains the ordered global geometry operations. Glyph
    Slant remains a fixed layout effect applied before those operations.
    Neutral entries remain present for the editor but are skipped by the
    runtime compiler.

    >>> stack = TextTransformStack((BendTextTransform(0.5),))
    >>> stack.has_nonlinear
    True
    """

    transforms: tuple[TextTransform, ...] = ()
    glyph_slant_angle: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            'transforms',
            tuple(coerce_text_transform(value) for value in self.transforms),
        )

    def __iter__(self) -> Iterator[TextTransform]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, index: int) -> TextTransform:
        return self.transforms[index]

    def is_neutral(self) -> bool:
        return (
            not self.has_active_stages
            and self.glyph_slant_angle == 0.0
        )

    @property
    def has_active_stages(self) -> bool:
        return any(not transform.is_neutral() for transform in self.transforms)

    @property
    def has_nonlinear(self) -> bool:
        return any(
            not transform.is_neutral() and transform.is_nonlinear
            for transform in self.transforms
        )


TEXT_TRANSFORM_TYPES = {
    'projective': ProjectiveTextTransform,
    'bend': BendTextTransform,
    'sine': SineTextTransform,
    'grid': GridTextTransform,
}


def create_text_transform(transform_type: str) -> TextTransform:
    """Create the neutral initial value for a registered transform type.

    UI-selectable variants must provide constructor defaults. Persisted
    payloads may still supply required variant fields through
    :func:`coerce_text_transform`.

    >>> create_text_transform('projective')
    ProjectiveTextTransform(transform_type='projective', horizontal_scale=1.0, vertical_scale=1.0, horizontal_slant=0.0, vertical_slant=0.0, rotation_x=0.0, rotation_y=0.0, rotation_z=0.0, perspective=0.0)
    """
    transform_class = TEXT_TRANSFORM_TYPES.get(transform_type)
    if transform_class is None:
        raise ValueError(f'unsupported text transform type {transform_type}')
    return transform_class()


def coerce_text_transform(value: Union[TextTransform, dict]) -> TextTransform:
    """Return a live value or construct one typed persisted payload.

    >>> transform = coerce_text_transform(
    ...     {'transform_type': 'projective', 'rotation_z': 5}
    ... )
    >>> transform.rotation_z
    5
    """
    if isinstance(value, TextTransform):
        return value
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
    return transform_class(**payload)


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


class FontWeight(enum.IntEnum):
    Thin = 100
    ExtraLight = 200
    Light = 300
    Normal = 400
    Medium = 500
    DemiBold = 600
    Bold = 700
    ExtraBold = 800
    Black = 900


class TextAlignment(enum.IntEnum):
    Left = 0
    Center = 1
    Right = 2


_QT5_TO_FONT_WEIGHT = {
    0: FontWeight.Thin,
    12: FontWeight.ExtraLight,
    25: FontWeight.Light,
    50: FontWeight.Normal,
    57: FontWeight.Medium,
    63: FontWeight.DemiBold,
    75: FontWeight.Bold,
    81: FontWeight.ExtraBold,
    87: FontWeight.Black,
}
_FONT_WEIGHT_TO_QT5 = {
    int(canonical): native
    for native, canonical in _QT5_TO_FONT_WEIGHT.items()
}
_QT5_CSS_TO_FONT_WEIGHT = {
    native * 8: int(canonical)
    for native, canonical in _QT5_TO_FONT_WEIGHT.items()
}
_FONT_WEIGHT_TO_QT5_CSS = {
    canonical: native_css
    for native_css, canonical in _QT5_CSS_TO_FONT_WEIGHT.items()
}
_FONT_WEIGHT_CSS_PATTERN = re.compile(
    r'(font-weight\s*:\s*)(\d+)', re.IGNORECASE
)
_QT_RICH_TEXT_META = '<meta name="qrichtext" content="1" />'
_QT_RICH_TEXT_MARKER = 'name="qrichtext"'
_UTF8_META = '<meta charset="utf-8" />'
_CHARSET_META_MARKER = '<meta charset='


def _replace_css_font_weights(html: str, mapping: dict[int, int]) -> str:
    return _FONT_WEIGHT_CSS_PATTERN.sub(
        lambda match: (
            match.group(1)
            + str(mapping.get(int(match.group(2)), int(match.group(2))))
        ),
        html,
    )


def export_font_weight_html(html: str, *, qt6: bool) -> str:
    """Write canonical CSS weights from either Qt HTML serializer.

    >>> export_font_weight_html('font-weight:600', qt6=False)
    'font-weight:700'
    """
    if qt6:
        return html
    html = _replace_css_font_weights(html, _QT5_CSS_TO_FONT_WEIGHT)
    if (
        _QT_RICH_TEXT_META in html
        and _CHARSET_META_MARKER not in html.lower()
    ):
        html = html.replace(
            _QT_RICH_TEXT_META,
            _QT_RICH_TEXT_META + _UTF8_META,
            1,
        )
    return html


def import_font_weight_html(html: str, *, qt6: bool) -> str:
    """Adapt canonical or legacy Qt 5 CSS to the active Qt parser.

    >>> legacy = _QT_RICH_TEXT_META + '<b style="font-weight:600">x</b>'
    >>> 'font-weight:700' in import_font_weight_html(legacy, qt6=True)
    True
    >>> standard = _QT_RICH_TEXT_META + _UTF8_META
    >>> standard += '<b style="font-weight:700">x</b>'
    >>> 'font-weight:600' in import_font_weight_html(standard, qt6=False)
    True
    """
    lowered_html = html.lower()
    legacy_qt5 = (
        _QT_RICH_TEXT_MARKER in lowered_html
        and _CHARSET_META_MARKER not in lowered_html
    )
    if legacy_qt5:
        return (
            _replace_css_font_weights(html, _QT5_CSS_TO_FONT_WEIGHT)
            if qt6
            else html
        )
    return (
        html
        if qt6
        else _replace_css_font_weights(html, _FONT_WEIGHT_TO_QT5_CSS)
    )


def coerce_font_weight(weight: int) -> FontWeight:
    """Load one canonical weight.

    >>> coerce_font_weight(25) is FontWeight.Light
    True
    >>> coerce_font_weight(29) is FontWeight.Light
    True
    """
    if not isinstance(weight, bool) and isinstance(weight, int):
        if 0 <= weight < 100:
            native = min(
                _QT5_TO_FONT_WEIGHT,
                key=lambda candidate: abs(candidate - weight),
            )
            weight = _QT5_TO_FONT_WEIGHT[native]
        elif 100 <= weight <= 1000:
            weight = min(
                FontWeight,
                key=lambda candidate: abs(int(candidate) - weight),
            )
        try:
            return FontWeight(weight)
        except ValueError:
            pass
    if weight is not None:
        LOGGER.warning(
            'Ignoring invalid font weight %r; using Normal.',
            weight,
        )
    return FontWeight.Normal


def font_weight_to_qt(
    weight: FontWeight,
    *,
    qt6: bool = None,
) -> int:
    """Return the current Qt binding's native integer weight."""
    canonical = FontWeight(weight)
    if qt6 is None:
        qt6 = shared.FLAG_QT6
    if qt6:
        return int(canonical)
    return _FONT_WEIGHT_TO_QT5[int(canonical)]


def font_weight_from_qt(weight: int) -> FontWeight:
    """Return a canonical weight from either Qt 5 or Qt 6."""
    return coerce_font_weight(int(weight))


@nested_dataclass
class FontFormat(Config):

    font_family: str = shared.DEFAULT_FONT_FAMILY # to always apply shared.DEFAULT_FONT_FAMILY
    font_size: float = 24
    stroke_width: float = 0.
    frgb: List = field(default_factory=lambda: [0, 0, 0])
    srgb: List = field(default_factory=lambda: [0, 0, 0])
    underline: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    standard_vertical_roman_alignment: bool = True
    # None is constructor-only so legacy payloads can distinguish an omitted
    # weight from an explicitly saved Normal value. __post_init__ canonicalizes
    # every live instance to FontWeight.
    font_weight: FontWeight = None
    line_spacing: float = 1.2
    letter_spacing: float = 1.15
    ligature_common: str = 'default'
    ligature_discretionary: str = 'enabled'
    ligature_contextual: str = 'default'
    oldstyle_nums: str = 'default'
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

    # Runtime owns one value; persistence keeps its existing list and flat
    # Glyph Slant fields for project/config compatibility.
    text_transform: Union[TextTransformStack, List] = field(
        default_factory=TextTransformStack
    )
    deprecated_attributes: dict = field(default_factory = lambda: dict())

    @property
    def size_pt(self):
        return px2pt(self.font_size)

    @property
    def glyph_slant_angle(self) -> float:
        """Compatibility view of the stack-owned Glyph Slant value."""
        return self.text_transform.glyph_slant_angle

    @glyph_slant_angle.setter
    def glyph_slant_angle(self, value: float) -> None:
        self.text_transform = replace(
            self.text_transform,
            glyph_slant_angle=value,
        )

    def __post_init__(self) -> None:
        da = self.deprecated_attributes
        # nested_dataclass routes the compatibility-only JSON field here.
        glyph_slant_angle = da.get('glyph_slant_angle')
        if len(da) > 0:
            if 'size' in da:
                self.font_size = pt2px(da['size'])
            if self.font_weight is None and 'weight' in da:
                self.font_weight = da['weight']
            if 'family' in da:
                self.font_family = da['family']

        if not isinstance(self.standard_vertical_roman_alignment, bool):
            LOGGER.warning(
                'Ignoring invalid standard vertical Roman alignment value '
                '(%r); using the enabled default.',
                self.standard_vertical_roman_alignment,
            )
            self.standard_vertical_roman_alignment = True

        for name in (
            'ligature_common',
            'ligature_discretionary',
            'ligature_contextual',
            'oldstyle_nums',
        ):
            if getattr(self, name) not in {'default', 'enabled', 'disabled'}:
                LOGGER.warning(
                    'Ignoring invalid %s value (%r); using default.',
                    name,
                    getattr(self, name),
                )
                setattr(self, name, 'default')

        self.font_weight = coerce_font_weight(self.font_weight)
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
        if glyph_slant_angle is not None:
            self.text_transform = replace(
                self.text_transform,
                glyph_slant_angle=glyph_slant_angle,
            )
        self.deprecated_attributes = {}

    def to_serializable_dict(self) -> dict:
        """Return config/project data with a typed transform payload."""
        serialized = vars(self).copy()
        serialized['font_weight'] = int(FontWeight(self.font_weight))
        serialized['text_transform'] = [
            asdict(transform) for transform in self.text_transform
        ]
        serialized['glyph_slant_angle'] = (
            self.text_transform.glyph_slant_angle
        )
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
