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
TEXT_TRANSFORM_GRID_INTERPOLATION_TYPES = ('bilinear', 'catmull_rom')
TEXT_TRANSFORM_PRECISION = 6


def _transform_value_field_names(
    transform: Union["TextTransform", type["TextTransform"]],
) -> tuple[str, ...]:
    """Return constructor fields, excluding derived fields such as the type."""
    return tuple(field.name for field in fields(transform) if field.init)


@dataclass(frozen=True)
class TextTransform:
    """Immutable base value for a persisted text-transform variant.

    Subclasses expose stable component names and normalization. Persistence
    stores ``transform_type`` with the variant-specific component payload.

    >>> ProjectiveTextTransform().transform_type
    'projective'
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
class ProjectiveTextTransform(TextTransform):
    """One native projective stage for affine and planar 3D controls.

    X and Y stop short of edge-on because a projected flat plane is singular
    at exactly 90 degrees.

    >>> ProjectiveTextTransform(rotation_x=90).normalized().rotation_x
    89.0
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

    def normalized(self) -> "ProjectiveTextTransform":
        return ProjectiveTextTransform(
            normalize_text_transform_value(
                self.horizontal_scale,
                TEXT_TRANSFORM_SCALE_MIN,
                TEXT_TRANSFORM_SCALE_MAX,
            ),
            normalize_text_transform_value(
                self.vertical_scale,
                TEXT_TRANSFORM_SCALE_MIN,
                TEXT_TRANSFORM_SCALE_MAX,
            ),
            normalize_text_transform_value(
                self.horizontal_slant,
                TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN,
                TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX,
            ),
            normalize_text_transform_value(
                self.vertical_slant,
                TEXT_TRANSFORM_PROJECTIVE_SLANT_MIN,
                TEXT_TRANSFORM_PROJECTIVE_SLANT_MAX,
            ),
            normalize_text_transform_value(
                self.rotation_x,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
            ),
            normalize_text_transform_value(
                self.rotation_y,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MIN,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_XY_MAX,
            ),
            normalize_text_transform_value(
                self.rotation_z,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MIN,
                TEXT_TRANSFORM_PROJECTIVE_ROTATION_Z_MAX,
            ),
            normalize_text_transform_value(
                self.perspective,
                TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MIN,
                TEXT_TRANSFORM_PROJECTIVE_PERSPECTIVE_MAX,
            ),
        )

    def is_neutral(self) -> bool:
        normalized = self.normalized()
        return (
            normalized.horizontal_scale == 1.0
            and normalized.vertical_scale == 1.0
            and normalized.horizontal_slant == 0.0
            and normalized.vertical_slant == 0.0
            and normalized.rotation_x == 0.0
            and normalized.rotation_y == 0.0
            and normalized.rotation_z == 0.0
        )


@dataclass(frozen=True)
class BendTextTransform(TextTransform):
    """Signed circular bend applied to the completed text surface."""

    bend: float = 0.0
    transform_type: str = dataclass_field(init=False, default='bend')
    is_nonlinear: ClassVar[bool] = True

    def normalized(self) -> "BendTextTransform":
        return BendTextTransform(
            normalize_text_transform_value(
                self.bend,
                TEXT_TRANSFORM_BEND_MIN,
                TEXT_TRANSFORM_BEND_MAX,
            )
        )

    def is_neutral(self) -> bool:
        return self.bend == 0.0


def _normalize_sine_frequency(value: Union[int, float, np.number]) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.number)
    ):
        raise ValueError('sine frequencies must be integers from 0 to 64')
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError('sine frequencies must be integers from 0 to 64')
    frequency = int(numeric)
    if not (
        TEXT_TRANSFORM_SINE_FREQUENCY_MIN
        <= frequency
        <= TEXT_TRANSFORM_SINE_FREQUENCY_MAX
    ):
        raise ValueError('sine frequencies must be integers from 0 to 64')
    return frequency


@dataclass(frozen=True)
class SineTextTransform(TextTransform):
    """Two ordered sine shears over the completed text surface.

    Frequencies count half-waves. The x-axis wave is applied first so the
    paired mappings remain exactly invertible at every supported value.

    >>> SineTextTransform().normalized().is_neutral()
    False
    >>> SineTextTransform(frequency_x=0).normalized().is_neutral()
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

    def normalized(self) -> "SineTextTransform":
        return SineTextTransform(
            _normalize_sine_frequency(self.frequency_x),
            _normalize_sine_frequency(self.frequency_y),
            normalize_text_transform_value(
                self.phase_x,
                TEXT_TRANSFORM_SINE_PHASE_MIN,
                TEXT_TRANSFORM_SINE_PHASE_MAX,
            ),
            normalize_text_transform_value(
                self.phase_y,
                TEXT_TRANSFORM_SINE_PHASE_MIN,
                TEXT_TRANSFORM_SINE_PHASE_MAX,
            ),
            normalize_text_transform_value(
                self.amplitude_x,
                TEXT_TRANSFORM_SINE_AMPLITUDE_MIN,
                TEXT_TRANSFORM_SINE_AMPLITUDE_MAX,
            ),
            normalize_text_transform_value(
                self.amplitude_y,
                TEXT_TRANSFORM_SINE_AMPLITUDE_MIN,
                TEXT_TRANSFORM_SINE_AMPLITUDE_MAX,
            ),
        )

    def is_neutral(self) -> bool:
        normalized = self.normalized()
        return (
            normalized.frequency_x == 0
            or normalized.amplitude_x == 0.0
        ) and (
            normalized.frequency_y == 0
            or normalized.amplitude_y == 0.0
        )


def _normalize_grid_division(value: Union[int, float, np.number]) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError('grid divisions must be integers from 1 to 32')
    numeric = float(value)
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError('grid divisions must be integers from 1 to 32')
    division = int(numeric)
    if not (
        TEXT_TRANSFORM_GRID_DIVISION_MIN
        <= division
        <= TEXT_TRANSFORM_GRID_DIVISION_MAX
    ):
        raise ValueError('grid divisions must be integers from 1 to 32')
    return division


def _default_grid_control_points(horizontal: int, vertical: int) -> tuple:
    return tuple(
        (
            round(column / horizontal, TEXT_TRANSFORM_PRECISION),
            round(row / vertical, TEXT_TRANSFORM_PRECISION),
        )
        for row in range(vertical + 1)
        for column in range(horizontal + 1)
    )


def _normalize_grid_control_points(
    points: Sequence[Sequence[float]], horizontal: int, vertical: int
) -> tuple:
    expected = (horizontal + 1) * (vertical + 1)
    if not isinstance(points, (list, tuple)) or len(points) != expected:
        raise ValueError(f'grid transform requires {expected} control points')
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('grid control points must be finite coordinate pairs')
        coordinates = []
        for value in point:
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.number)
            ):
                raise ValueError(
                    'grid control points must be finite coordinate pairs'
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    'grid control points must be finite coordinate pairs'
                )
            coordinates.append(round(value, TEXT_TRANSFORM_PRECISION))
        normalized.append(tuple(coordinates))
    return tuple(normalized)


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

    >>> len(GridTextTransform().normalized().control_points)
    4
    >>> GridTextTransform(horizontal_divisions=2).normalized().is_neutral()
    True
    """

    horizontal_divisions: int = 1
    vertical_divisions: int = 1
    interpolation: str = 'bilinear'
    control_points: tuple = ()
    transform_type: str = dataclass_field(init=False, default='grid')
    is_nonlinear: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.control_points:
            try:
                points = tuple(tuple(point) for point in self.control_points)
            except TypeError as error:
                raise ValueError(
                    'grid control points must be coordinate pairs'
                ) from error
            object.__setattr__(self, 'control_points', points)

    def normalized(self) -> "GridTextTransform":
        horizontal = _normalize_grid_division(self.horizontal_divisions)
        vertical = _normalize_grid_division(self.vertical_divisions)
        if self.interpolation not in TEXT_TRANSFORM_GRID_INTERPOLATION_TYPES:
            raise ValueError(
                f'unsupported grid interpolation type {self.interpolation!r}'
            )
        points = (
            _normalize_grid_control_points(
                self.control_points, horizontal, vertical
            )
            if self.control_points
            else _default_grid_control_points(horizontal, vertical)
        )
        return GridTextTransform(
            horizontal,
            vertical,
            self.interpolation,
            points,
        )

    def with_value(
        self, name: str, value: Union[int, float, str]
    ) -> "GridTextTransform":
        current = self.normalized()
        if name in {'horizontal_divisions', 'vertical_divisions'}:
            horizontal = (
                _normalize_grid_division(value)
                if name == 'horizontal_divisions'
                else current.horizontal_divisions
            )
            vertical = (
                _normalize_grid_division(value)
                if name == 'vertical_divisions'
                else current.vertical_divisions
            )
            points = _resample_grid_control_points(
                current.control_points,
                current.horizontal_divisions,
                current.vertical_divisions,
                horizontal,
                vertical,
            )
            return GridTextTransform(
                horizontal,
                vertical,
                current.interpolation,
                points,
            ).normalized()
        if name == 'interpolation':
            return replace(current, interpolation=value).normalized()
        return super().with_value(name, value)

    def with_control_points(
        self, points: Sequence[Sequence[float]]
    ) -> "GridTextTransform":
        return replace(self, control_points=tuple(points)).normalized()

    def is_neutral(self) -> bool:
        normalized = self.normalized()
        return normalized.control_points == _default_grid_control_points(
            normalized.horizontal_divisions,
            normalized.vertical_divisions,
        )


@dataclass(frozen=True)
class TextTransformStack:
    """Immutable ordered text-geometry operations.

    Empty means no geometry transform. Neutral entries remain present for the
    editor but are skipped by the runtime compiler.

    >>> stack = TextTransformStack((BendTextTransform(0.5),))
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

    def __iter__(self) -> Iterator[TextTransform]:
        return iter(self.transforms)

    def __len__(self) -> int:
        return len(self.transforms)

    def __getitem__(self, index: int) -> TextTransform:
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
    return transform_class().normalized()


def coerce_text_transform(value: Union[TextTransform, dict]) -> TextTransform:
    """Normalize a live value or construct a canonical persisted payload.

    >>> transform = coerce_text_transform(
    ...     {'transform_type': 'projective', 'rotation_z': 5}
    ... )
    >>> transform.rotation_z
    5.0
    >>> coerce_text_transform(
    ...     {'transform_type': 'projective', 'horizontal_scale': 5}
    ... )
    Traceback (most recent call last):
    ...
    ValueError: persisted projective transform values must be canonical
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
    comparison = transform
    if isinstance(transform, GridTextTransform) and not transform.control_points:
        comparison = replace(
            transform,
            control_points=normalized.control_points,
        )
    if comparison != normalized:
        raise ValueError(
            f'persisted {transform_type} transform values must be canonical'
        )
    return normalized


def coerce_text_transform_stack(
    value: Union[
        TextTransformStack,
        Sequence[Union[TextTransform, dict]],
    ],
) -> TextTransformStack:
    """Return one canonical ordered stack and reject the old single payload.

    >>> coerce_text_transform_stack([
    ...     {'transform_type': 'bend', 'bend': 0.5},
    ... ])
    TextTransformStack(transforms=(BendTextTransform(transform_type='bend', bend=0.5),))
    >>> coerce_text_transform_stack({'transform_type': 'bend'})
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
