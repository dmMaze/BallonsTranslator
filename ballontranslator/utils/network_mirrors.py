import json
import os
from typing import Iterable, Optional


HUGGINGFACE_ORIGIN = 'https://huggingface.co'
DEFAULT_HUGGINGFACE_MIRROR = 'https://hf-mirror.com'
DEFAULT_PYPI_MIRROR = 'https://mirrors.aliyun.com/pypi/simple'
HUGGINGFACE_MIRROR_OPTIONS = (None, DEFAULT_HUGGINGFACE_MIRROR)
PYPI_MIRROR_OPTIONS = (
    None,
    DEFAULT_PYPI_MIRROR,
)


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
