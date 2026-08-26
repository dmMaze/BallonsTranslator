"""Small lazy registry dedicated to deterministic text filters."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import importlib.util
import logging
import math
from pathlib import Path
import sys
from types import MappingProxyType, ModuleType
from typing import Callable, Mapping, Optional, Tuple

import numpy as np

from ballontranslator.utils.fontformat import TEXT_TRANSFORM_PRECISION
from ballontranslator.utils.text_effects import FilterEffect, FilterScalar


LOGGER = logging.getLogger(__name__)
_PARAM_KINDS = frozenset(('float', 'int', 'bool', 'choice'))
_SOURCE_SIGNATURE = Tuple[int, int, int, int]
_PARAM_WARNING_LIMIT = 64


class FilterMetadataError(ValueError):
    """Raised when a built-in filter's static contract is invalid."""


class FilterUnavailableError(RuntimeError):
    """Raised when an active filter cannot safely be resolved."""


@dataclass(frozen=True)
class FilterContext:
    """One filter invocation's stable render coordinates.

    ``origin_x`` and ``origin_y`` are physical pixels relative to the
    unpadded logical text origin, including negative effect overflow.

    >>> FilterContext(2.0, -4, 8, True).origin_x
    -4
    """

    render_scale: float
    origin_x: int
    origin_y: int
    strict_export: bool = False


@dataclass(frozen=True)
class FilterParamSpec:
    """Static UI/runtime metadata for one scalar filter parameter."""

    key: str
    label: str
    kind: str
    default: FilterScalar
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: float = 1.0
    display_factor: float = 1.0
    decimals: int = 0
    suffix: str = ''
    choices: Tuple[Tuple[str, FilterScalar], ...] = ()

    def normalize(self, value: object) -> FilterScalar:
        """Validate a live value without rewriting passive payloads."""
        if self.kind == 'bool':
            if not isinstance(value, bool):
                raise ValueError(f'{self.label} must be true or false')
            return value
        if self.kind == 'choice':
            if not any(value == candidate for _, candidate in self.choices):
                raise ValueError(f'{self.label} has an unsupported choice')
            return value  # type: ignore[return-value]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{self.label} must be numeric')
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f'{self.label} is outside the numeric range') from error
        if not math.isfinite(number):
            raise ValueError(f'{self.label} must be finite')
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f'{self.label} is below its minimum')
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f'{self.label} exceeds its maximum')
        if self.kind == 'int':
            if number != int(number):
                raise ValueError(f'{self.label} must be an integer')
            return int(number)
        return number


@dataclass(frozen=True)
class FilterSpec:
    """AST-readable filter identity and ordered parameter contract."""

    filter_id: str
    name: str
    schema_version: int
    params: Tuple[FilterParamSpec, ...]
    order: int
    source_path: Path
    resolved_source_path: Path
    builtin: bool
    source_signature: _SOURCE_SIGNATURE
    expands_alpha: bool = False

    def default_params(self) -> dict[str, FilterScalar]:
        return {parameter.key: parameter.default for parameter in self.params}

    def normalize_params(
        self, params: Mapping[str, FilterScalar]
    ) -> dict[str, FilterScalar]:
        """Return only declared params, isolating invalid known values."""
        active = {}
        for parameter in self.params:
            value = params.get(parameter.key, parameter.default)
            try:
                active[parameter.key] = parameter.normalize(value)
            except ValueError:
                active[parameter.key] = parameter.normalize(parameter.default)
        return active


@dataclass(frozen=True)
class FilterRuntime:
    """Resolved active implementation and validated parameters."""

    spec: FilterSpec
    params: Mapping[str, FilterScalar]
    apply: Callable[[np.ndarray, Mapping[str, FilterScalar], FilterContext], np.ndarray]
    tile_halo: Callable[[Mapping[str, FilterScalar], float], object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'params', MappingProxyType(dict(self.params))
        )


def _source_signature(path: Path) -> _SOURCE_SIGNATURE:
    stat = path.stat()
    return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _literal_filter_meta(path: Path) -> Mapping[str, object]:
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as error:
        raise FilterMetadataError(f'cannot parse {path.name}: {error}') from error
    values = [
        node.value
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            any(
                isinstance(target, ast.Name) and target.id == 'FILTER_META'
                for target in node.targets
            )
            if isinstance(node, ast.Assign)
            else isinstance(node.target, ast.Name)
            and node.target.id == 'FILTER_META'
        )
    ]
    if len(values) != 1:
        raise FilterMetadataError(
            f'{path.name} must define one literal FILTER_META'
        )
    try:
        meta = ast.literal_eval(values[0])
    except (TypeError, ValueError) as error:
        raise FilterMetadataError(
            f'{path.name} FILTER_META must be a literal mapping'
        ) from error
    if not isinstance(meta, Mapping):
        raise FilterMetadataError(f'{path.name} FILTER_META must be a mapping')
    return meta


def _scalar(value: object, label: str) -> FilterScalar:
    if not isinstance(value, (bool, int, float, str, type(None))):
        raise FilterMetadataError(f'{label} must be a JSON scalar')
    if isinstance(value, float) and not math.isfinite(value):
        raise FilterMetadataError(f'{label} must be finite')
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FilterMetadataError(f'{label} must be numeric')
    try:
        number = float(value)
    except OverflowError as error:
        raise FilterMetadataError(f'{label} is outside the numeric range') from error
    if not math.isfinite(number):
        raise FilterMetadataError(f'{label} must be finite')
    return number


def _param_spec(raw: object) -> FilterParamSpec:
    if not isinstance(raw, Mapping):
        raise FilterMetadataError('filter params must be literal mappings')
    key = raw.get('key')
    label = raw.get('label')
    kind = raw.get('kind')
    if not isinstance(key, str) or not key:
        raise FilterMetadataError('filter param key must be a non-empty string')
    if not isinstance(label, str) or not label:
        raise FilterMetadataError(f'filter param {key} needs a label')
    if kind not in _PARAM_KINDS:
        raise FilterMetadataError(f'filter param {key} has invalid kind')
    default = _scalar(raw.get('default'), f'filter param {key} default')
    choices: Tuple[Tuple[str, FilterScalar], ...] = ()
    if kind == 'choice':
        raw_choices = raw.get('choices')
        if not isinstance(raw_choices, (list, tuple)) or not raw_choices:
            raise FilterMetadataError(f'filter param {key} needs choices')
        parsed_choices = []
        for choice in raw_choices:
            if not isinstance(choice, (list, tuple)) or len(choice) != 2:
                raise FilterMetadataError(f'filter param {key} has invalid choice')
            choice_label, choice_value = choice
            if not isinstance(choice_label, str) or not choice_label:
                raise FilterMetadataError(
                    f'filter param {key} has invalid choice label'
                )
            parsed_choices.append(
                (choice_label, _scalar(choice_value, f'filter param {key} choice'))
            )
        choices = tuple(parsed_choices)
    decimals = raw.get('decimals', 0)
    if (
        isinstance(decimals, bool)
        or not isinstance(decimals, int)
        or not 0 <= decimals <= TEXT_TRANSFORM_PRECISION
    ):
        raise FilterMetadataError(
            f'filter param {key} decimals must be an integer from 0 to '
            f'{TEXT_TRANSFORM_PRECISION}'
        )
    suffix = raw.get('suffix', '')
    if not isinstance(suffix, str):
        raise FilterMetadataError(f'filter param {key} suffix must be a string')
    minimum = maximum = None
    step = display_factor = 1.0
    if kind in {'float', 'int'}:
        if raw.get('minimum') is None or raw.get('maximum') is None:
            raise FilterMetadataError(
                f'filter param {key} requires minimum and maximum'
            )
        minimum = _finite_number(
            raw['minimum'], f'filter param {key} minimum'
        )
        maximum = _finite_number(
            raw['maximum'], f'filter param {key} maximum'
        )
        if minimum > maximum:
            raise FilterMetadataError(
                f'filter param {key} minimum exceeds maximum'
            )
        step = _finite_number(raw.get('step', 1.0), f'filter param {key} step')
        display_factor = _finite_number(
            raw.get('display_factor', 1.0),
            f'filter param {key} display factor',
        )
        if step <= 0.0 or display_factor <= 0.0:
            raise FilterMetadataError(
                f'filter param {key} display values must be positive'
            )
    try:
        spec = FilterParamSpec(
            key=key,
            label=label,
            kind=kind,
            default=default,
            minimum=minimum,
            maximum=maximum,
            step=step,
            display_factor=display_factor,
            decimals=decimals,
            suffix=suffix,
            choices=choices,
        )
        spec.normalize(default)
    except (KeyError, OverflowError, TypeError, ValueError) as error:
        raise FilterMetadataError(f'invalid filter param {key}: {error}') from error
    return spec


def _filter_spec(path: Path, *, builtin: bool) -> FilterSpec:
    meta = _literal_filter_meta(path)
    filter_id = meta.get('filter_id')
    name = meta.get('name')
    version = meta.get('schema_version')
    suffix = path.stem.removeprefix('filter_')
    expected_id = ('builtin:' + suffix) if builtin else None
    if (
        not isinstance(filter_id, str)
        or ':' not in filter_id
        or filter_id.rsplit(':', 1)[-1] != suffix
        or (builtin and filter_id != expected_id)
    ):
        raise FilterMetadataError(
            f'{path.name} declares a filter_id that does not match its path'
        )
    if not isinstance(name, str) or not name:
        raise FilterMetadataError(f'{path.name} needs a non-empty name')
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise FilterMetadataError(f'{path.name} needs a positive schema_version')
    raw_params = meta.get('params', ())
    if not isinstance(raw_params, (list, tuple)):
        raise FilterMetadataError(f'{path.name} params must be a sequence')
    params = tuple(_param_spec(raw) for raw in raw_params)
    if len({parameter.key for parameter in params}) != len(params):
        raise FilterMetadataError(f'{path.name} has duplicate param keys')
    order = meta.get('order', 100)
    if isinstance(order, bool) or not isinstance(order, int):
        raise FilterMetadataError(f'{path.name} order must be an integer')
    expands_alpha = meta.get('expands_alpha', False)
    if not isinstance(expands_alpha, bool):
        raise FilterMetadataError(
            f'{path.name} expands_alpha must be true or false'
        )
    return FilterSpec(
        filter_id, name, version, params, order, path, path.resolve(), builtin,
        _source_signature(path), expands_alpha,
    )


class FilterRegistry:
    """Discover literal metadata once and lazily import active filters only.

    Custom filters are trusted local Python, not a sandbox. Discovery never
    executes them, and a registry snapshot intentionally requires restart to
    observe file additions or modifications.
    """

    def __init__(
        self,
        builtin_dir: Optional[Path] = None,
        custom_dir: Optional[Path] = None,
    ) -> None:
        filters_dir = Path(__file__).resolve().parent
        self._builtin_dir = filters_dir if builtin_dir is None else Path(builtin_dir)
        self._custom_dir = (
            filters_dir.parents[4] / 'custom_modules'
            if custom_dir is None
            else Path(custom_dir)
        )
        self._specs: Optional[Tuple[FilterSpec, ...]] = None
        self._spec_by_id: dict[str, FilterSpec] = {}
        self._modules: dict[str, ModuleType] = {}
        self._failures: dict[str, FilterUnavailableError] = {}
        self._param_warnings: set[tuple[str, int, str]] = set()

    def _scan_root(self, root: Path, *, builtin: bool) -> list[FilterSpec]:
        if not root.is_dir():
            return []
        resolved_root = root.resolve()
        specs = []
        for path in sorted(root.glob('filter_*.py')):
            try:
                if path.is_symlink() or not path.is_file():
                    raise FilterMetadataError(f'rejecting unsafe path {path}')
                if path.resolve().parent != resolved_root:
                    raise FilterMetadataError(f'rejecting path outside {root}')
                specs.append(_filter_spec(path, builtin=builtin))
            except (FilterMetadataError, OSError) as error:
                if builtin:
                    raise FilterMetadataError(str(error)) from error
                LOGGER.warning('Ignoring custom text filter %s (%s).', path, error)
        return specs

    @property
    def specs(self) -> Tuple[FilterSpec, ...]:
        if self._specs is None:
            builtins = self._scan_root(self._builtin_dir, builtin=True)
            by_id = {spec.filter_id: spec for spec in builtins}
            for spec in self._scan_root(self._custom_dir, builtin=False):
                if spec.filter_id in by_id:
                    LOGGER.warning(
                        'Ignoring duplicate custom text filter id %s.',
                        spec.filter_id,
                    )
                    continue
                by_id[spec.filter_id] = spec
            self._specs = tuple(
                sorted(by_id.values(), key=lambda spec: (spec.order, spec.filter_id))
            )
            self._spec_by_id = by_id
        return self._specs

    def get_spec(self, filter_id: str) -> Optional[FilterSpec]:
        self.specs
        return self._spec_by_id.get(filter_id)

    def get_runtime_failure(
        self, filter_id: str
    ) -> Optional[FilterUnavailableError]:
        """Return a previously observed lazy-load failure without importing."""
        return self._failures.get(filter_id)

    def _load_module(self, spec: FilterSpec) -> ModuleType:
        cached = self._modules.get(spec.filter_id)
        if cached is not None:
            return cached
        failure = self._failures.get(spec.filter_id)
        if failure is not None:
            raise failure
        module_name = None
        try:
            root = (
                self._builtin_dir if spec.builtin else self._custom_dir
            ).resolve()
            resolved = spec.source_path.resolve()
            if (
                spec.source_path.is_symlink()
                or not spec.source_path.is_file()
                or resolved != spec.resolved_source_path
                or resolved.parent != root
            ):
                raise FilterUnavailableError(
                    f'{spec.name} source path changed; restart is required'
                )
            if _source_signature(spec.source_path) != spec.source_signature:
                raise FilterUnavailableError(
                    f'{spec.name} changed on disk; restart is required'
                )
            module_name = (
                f'{__package__}.{spec.source_path.stem}'
                if spec.builtin
                else '_ballontranslator_custom_text_filter_'
                + spec.source_path.stem
            )
            module_spec = importlib.util.spec_from_file_location(
                module_name, spec.source_path
            )
            if module_spec is None or module_spec.loader is None:
                raise FilterUnavailableError(f'cannot import {spec.name}')
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            try:
                module_spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            runtime_meta = getattr(module, 'FILTER_META', None)
            if not isinstance(runtime_meta, Mapping):
                raise FilterUnavailableError(f'{spec.name} has no runtime metadata')
            if (
                runtime_meta.get('filter_id') != spec.filter_id
                or runtime_meta.get('schema_version') != spec.schema_version
            ):
                raise FilterUnavailableError(
                    f'{spec.name} runtime metadata does not match discovery'
                )
            if not callable(getattr(module, 'apply', None)):
                raise FilterUnavailableError(f'{spec.name} has no apply function')
            if not callable(getattr(module, 'tile_halo', None)):
                raise FilterUnavailableError(f'{spec.name} has no tile_halo function')
            self._modules[spec.filter_id] = module
            return module
        except FilterUnavailableError as error:
            if module_name is not None:
                sys.modules.pop(module_name, None)
            self._failures[spec.filter_id] = error
            raise
        except Exception as error:
            if module_name is not None:
                sys.modules.pop(module_name, None)
            unavailable = FilterUnavailableError(
                f'cannot load {spec.name}: {error}'
            )
            self._failures[spec.filter_id] = unavailable
            raise unavailable from error

    def resolve(self, effect: FilterEffect) -> FilterRuntime:
        spec = self.get_spec(effect.filter_id)
        if spec is None:
            raise FilterUnavailableError(f'missing text filter {effect.filter_id}')
        if effect.schema_version > spec.schema_version:
            raise FilterUnavailableError(
                f'{spec.name} schema {effect.schema_version} is incompatible'
            )
        module = self._load_module(spec)
        params = effect.params_dict()
        if effect.schema_version != spec.schema_version:
            migrate = getattr(module, 'migrate_params', None)
            if not callable(migrate):
                raise FilterUnavailableError(
                    f'{spec.name} schema {effect.schema_version} is incompatible'
                )
            try:
                migrated = migrate(effect.schema_version, dict(params))
            except Exception as error:
                raise FilterUnavailableError(
                    f'{spec.name} parameter migration failed: {error}'
                ) from error
            if not isinstance(migrated, Mapping):
                raise FilterUnavailableError(
                    f'{spec.name} parameter migration returned invalid data'
                )
            params = dict(migrated)
        try:
            active = spec.normalize_params(params)
        except (KeyError, ValueError) as error:
            raise FilterUnavailableError(f'{spec.name}: {error}') from error
        for parameter in spec.params:
            if parameter.key not in params:
                continue
            try:
                parameter.normalize(params[parameter.key])
            except ValueError as error:
                warning_key = (
                    effect.filter_id, effect.schema_version, parameter.key
                )
                if (
                    warning_key not in self._param_warnings
                    and len(self._param_warnings) < _PARAM_WARNING_LIMIT
                ):
                    self._param_warnings.add(warning_key)
                    LOGGER.warning(
                        'Ignoring invalid %s parameter %s (%s); using default.',
                        effect.filter_id,
                        parameter.key,
                        error,
                    )
        return FilterRuntime(
            spec, active, module.apply, module.tile_halo
        )


_FILTER_REGISTRY: Optional[FilterRegistry] = None


def get_filter_registry() -> FilterRegistry:
    """Return the process-local restart-scoped filter metadata snapshot."""
    global _FILTER_REGISTRY
    if _FILTER_REGISTRY is None:
        _FILTER_REGISTRY = FilterRegistry()
    return _FILTER_REGISTRY
