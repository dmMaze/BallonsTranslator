import json
import locale
import os
import time
from typing import Iterable, Optional, Set
from urllib.request import getproxies


HUGGINGFACE_ORIGIN = 'https://huggingface.co'
DEFAULT_HUGGINGFACE_MIRROR = 'https://hf-mirror.com'
DEFAULT_PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple'
HUGGINGFACE_MIRROR_OPTIONS = (None, DEFAULT_HUGGINGFACE_MIRROR)
PYPI_MIRROR_OPTIONS = (
    None,
    DEFAULT_PYPI_MIRROR,
)
MIRROR_FIELDS = ('huggingface', 'pypi')


def normalize_mirror_value(value: Optional[str]) -> Optional[str]:
    """Normalize a persisted or UI mirror value.

    >>> normalize_mirror_value('None') is None
    True
    >>> normalize_mirror_value(' https://hf-mirror.com ')
    'https://hf-mirror.com'
    """

    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() == 'none':
        return None
    return value.rstrip('/')


def mirror_to_display(value: Optional[str], none_label: str = 'None') -> str:
    """Return the combobox text for a mirror value.

    >>> mirror_to_display(None)
    'None'
    >>> mirror_to_display('https://hf-mirror.com')
    'https://hf-mirror.com'
    """

    return none_label if normalize_mirror_value(value) is None else normalize_mirror_value(value)


def mirror_from_display(value: str, none_label: str = 'None') -> Optional[str]:
    """Return the persisted value for combobox text.

    >>> mirror_from_display('None') is None
    True
    >>> mirror_from_display('https://mirrors.aliyun.com/pypi/simple')
    'https://mirrors.aliyun.com/pypi/simple'
    """

    if value == none_label:
        return None
    return normalize_mirror_value(value)


def display_options(values: Iterable[Optional[str]], none_label: str = 'None') -> list:
    return [mirror_to_display(value, none_label=none_label) for value in values]


def rewrite_huggingface_url(url: str, mirror: Optional[str]) -> str:
    """Rewrite explicit Hugging Face URLs when a mirror is configured.

    >>> rewrite_huggingface_url('https://huggingface.co/a/b', 'https://hf-mirror.com')
    'https://hf-mirror.com/a/b'
    >>> rewrite_huggingface_url('https://example.com/a/b', 'https://hf-mirror.com')
    'https://example.com/a/b'
    """

    mirror = normalize_mirror_value(mirror)
    if not mirror or not isinstance(url, str):
        return url
    if url == HUGGINGFACE_ORIGIN:
        return mirror
    if url.startswith(HUGGINGFACE_ORIGIN + '/'):
        return mirror + url[len(HUGGINGFACE_ORIGIN):]
    return url


def installer_env_with_pypi_mirror(env: Optional[dict] = None, mirror: Optional[str] = None) -> dict:
    """Return an installer environment with ``INDEX_URL`` when configured.

    >>> installer_env_with_pypi_mirror({'PATH': '/bin'}, None)
    {'PATH': '/bin'}
    >>> installer_env_with_pypi_mirror({}, 'https://example.invalid/simple')['INDEX_URL']
    'https://example.invalid/simple'
    """

    result = dict(env or os.environ.copy())
    mirror = normalize_mirror_value(mirror)
    if mirror:
        result['INDEX_URL'] = mirror
    return result


def read_saved_pypi_mirror(config_path: str) -> Optional[str]:
    """Read only the saved PyPI mirror from raw JSON.

    This is safe to call before full config loading and before core dependency
    repair.

    >>> read_saved_pypi_mirror('/path/that/does/not/exist') is None
    True
    """

    mirrors = _read_raw_mirrors(config_path)
    if not isinstance(mirrors, dict):
        return None
    return normalize_mirror_value(mirrors.get('pypi'))


def missing_mirror_fields(config_path: str) -> Set[str]:
    """Return mirror fields that were absent from a persisted config.

    Explicit JSON ``null`` values are present fields and should not be
    overwritten by automatic defaults.

    >>> sorted(_missing_mirror_fields_from_data({}))
    ['huggingface', 'pypi']
    >>> _missing_mirror_fields_from_data({'mirrors': {'pypi': None}})
    {'huggingface'}
    """

    data = _read_raw_config(config_path)
    if not isinstance(data, dict):
        return set(MIRROR_FIELDS)
    return _missing_mirror_fields_from_data(data)


def _missing_mirror_fields_from_data(data: dict) -> Set[str]:
    mirrors = data.get('mirrors')
    if not isinstance(mirrors, dict):
        return set(MIRROR_FIELDS)
    return {field for field in MIRROR_FIELDS if field not in mirrors}


def should_use_china_mirrors(
    locale_names: Iterable[str] = (),
    timezone_names: Iterable[str] = (),
) -> bool:
    """Return whether local OS hints clearly identify mainland China.

    >>> should_use_china_mirrors(locale_names=['zh_CN'])
    True
    >>> should_use_china_mirrors(locale_names=['zh'])
    False
    >>> should_use_china_mirrors(timezone_names=['Asia/Shanghai'])
    True
    >>> should_use_china_mirrors(timezone_names=['UTC+08:00'])
    False
    """

    return _has_mainland_china_locale(locale_names) or _has_mainland_china_timezone(timezone_names)


def collect_system_locale_names() -> list:
    candidates = []
    try:
        from qtpy.QtCore import QLocale
        candidates.append(QLocale.system().name())
    except Exception:
        pass
    candidates.extend([
        os.environ.get('LC_ALL', ''),
        os.environ.get('LC_MESSAGES', ''),
        os.environ.get('LANG', ''),
    ])
    try:
        candidates.append(locale.getlocale()[0] or '')
    except Exception:
        pass
    try:
        candidates.append(locale.getdefaultlocale()[0] or '')
    except Exception:
        pass
    return _unique_nonempty(candidates)


def collect_system_timezone_names() -> list:
    candidates = [os.environ.get('TZ', '')]
    candidates.extend(name for name in time.tzname if name)
    candidates.append(_read_etc_timezone())
    candidates.append(_localtime_zoneinfo_name())
    return _unique_nonempty(candidates)


def backfill_missing_mirror_defaults(
    mirrors_config,
    missing_fields: Iterable[str],
    locale_names: Iterable[str] = (),
    timezone_names: Iterable[str] = (),
) -> list:
    """Set mirror defaults for missing fields when local hints say China.

    >>> class Mirrors:
    ...     huggingface = None
    ...     pypi = None
    >>> mirrors = Mirrors()
    >>> backfill_missing_mirror_defaults(mirrors, {'pypi'}, locale_names=['zh_CN'])
    ['pypi']
    >>> mirrors.pypi
    'https://mirrors.aliyun.com/pypi/simple'
    """

    missing_fields = set(missing_fields)
    if not missing_fields or not should_use_china_mirrors(locale_names, timezone_names):
        return []

    updated = []
    if 'huggingface' in missing_fields and getattr(mirrors_config, 'huggingface', None) is None:
        mirrors_config.huggingface = DEFAULT_HUGGINGFACE_MIRROR
        updated.append('huggingface')
    if 'pypi' in missing_fields and getattr(mirrors_config, 'pypi', None) is None:
        mirrors_config.pypi = DEFAULT_PYPI_MIRROR
        updated.append('pypi')
    return updated


def _has_mainland_china_locale(locale_names: Iterable[str]) -> bool:
    for value in locale_names:
        if not value:
            continue
        normalized = str(value).strip().split('.', 1)[0].replace('-', '_')
        lower = normalized.lower()
        upper = normalized.upper()
        if upper == 'CN':
            return True
        if lower == 'zh_cn' or lower.endswith('_cn') or '_cn_' in lower:
            return True
        if 'Chinese (Simplified)_China'.lower() in lower:
            return True
    return False


def _has_mainland_china_timezone(timezone_names: Iterable[str]) -> bool:
    for value in timezone_names:
        if not value:
            continue
        normalized = str(value).strip().lower().replace('\\', '/')
        if normalized in {'asia/shanghai', 'prc'}:
            return True
        if normalized.endswith('/asia/shanghai') or 'zoneinfo/asia/shanghai' in normalized:
            return True
        if 'china standard time' in normalized or '中国标准时间' in normalized or '中国夏令时' in normalized:
            return True
    return False


def _read_raw_config(config_path: str):
    if not config_path or not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf8') as f:
            return json.load(f)
    except Exception:
        return None


def _read_raw_mirrors(config_path: str):
    data = _read_raw_config(config_path)
    if not isinstance(data, dict):
        return None
    return data.get('mirrors')


def _read_etc_timezone() -> str:
    try:
        with open('/etc/timezone', 'r', encoding='utf8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _localtime_zoneinfo_name() -> str:
    try:
        path = os.path.realpath('/etc/localtime')
    except Exception:
        return ''
    marker = 'zoneinfo/'
    if marker not in path:
        return path
    return path.split(marker, 1)[1]


def _unique_nonempty(values: Iterable[str]) -> list:
    unique = []
    for value in values:
        if not value:
            continue
        if value not in unique:
            unique.append(value)
    return unique


def has_effective_system_proxy() -> bool:
    proxies = getproxies()
    return any(key.lower() != 'no' and value for key, value in proxies.items())


def write_raw_mirror_config(config_path: str, huggingface: Optional[str], pypi: Optional[str]) -> bool:
    """Write a minimal first-run config containing network mirror choices.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = os.path.join(tmpdir, 'config.json')
    ...     _ = write_raw_mirror_config(path, None, 'https://example.invalid/simple')
    ...     read_saved_pypi_mirror(path)
    'https://example.invalid/simple'
    """

    if not config_path:
        return False

    try:
        config_dir = os.path.dirname(config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir)
        data = {
            'mirrors': {
                'huggingface': normalize_mirror_value(huggingface),
                'pypi': normalize_mirror_value(pypi),
            }
        }
        tmp_save_tgt = config_path + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp_save_tgt, config_path)
    except Exception:
        return False
    return True


def auto_fill_network_mirrors(config_path: str, logger=None) -> list:
    """Create first-run network mirror config when local hints need it.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = os.path.join(tmpdir, 'config.json')
    ...     _ = write_raw_mirror_config(path, None, None)
    ...     auto_fill_network_mirrors(path, logger=None)
    []
    """


    def log_info(message: str):
        if logger is not None:
            logger.info(message)

    if config_path and os.path.exists(config_path):
        log_info('Network mirror config already exists; skipping automatic mirror selection.')
        return []

    if has_effective_system_proxy():
        log_info('System proxy detected; skipping automatic mirror selection.')
        return []

    locale_names = collect_system_locale_names()
    timezone_names = collect_system_timezone_names()
    use_china_mirrors = should_use_china_mirrors(locale_names, timezone_names)
    log_info('Checking first-run network mirror defaults.')
    log_info(f'Network mirror heuristic locale hints: {locale_names}')
    log_info(f'Network mirror heuristic timezone hints: {timezone_names}')
    log_info(
        'Network mirror heuristic result: '
        f'{"mainland China detected" if use_china_mirrors else "mainland China not detected"}'
    )

    huggingface_mirror = DEFAULT_HUGGINGFACE_MIRROR if use_china_mirrors else None
    pypi_mirror = DEFAULT_PYPI_MIRROR if use_china_mirrors else None
    if not write_raw_mirror_config(config_path, huggingface_mirror, pypi_mirror):
        log_info('Failed to save first-run network mirror config.')
        return []

    if not use_china_mirrors:
        log_info('No network mirrors were selected automatically.')
        return []

    updated_mirrors = list(MIRROR_FIELDS)
    log_info(f'Automatically selected network mirrors for: {", ".join(updated_mirrors)}')
    return updated_mirrors
