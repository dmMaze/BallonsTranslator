from pathlib import Path

from . import shared


def _read_pyproject_version(pyproject_path: Path) -> str:
    """Read the ``[project]`` version field from ``pyproject.toml``.

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = Path(tmp) / 'pyproject.toml'
    ...     _ = path.write_text('[project]\\nversion = "1.2.3"\\n', encoding='utf8')
    ...     _read_pyproject_version(path)
    '1.2.3'
    """

    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:
        with pyproject_path.open('rb') as f:
            data = tomllib.load(f)
        version = data.get('project', {}).get('version')
        if isinstance(version, str) and version:
            return version

    in_project_section = False
    for raw_line in pyproject_path.read_text(encoding='utf8').splitlines():
        line = raw_line.strip()
        if line.startswith('[') and line.endswith(']'):
            in_project_section = line == '[project]'
            continue
        if in_project_section and line.startswith('version'):
            key, sep, value = line.partition('=')
            if sep and key.strip() == 'version':
                return value.strip().strip('"\'')

    raise RuntimeError(f'Failed to read project version from {pyproject_path}')


def get_current_version(program_path: str = None) -> str:
    """Return the application version from the project metadata.

    >>> isinstance(get_current_version('/path/that/does/not/exist'), str)
    True
    """

    root = Path(program_path or shared.PROGRAM_PATH)
    pyproject_path = root / 'pyproject.toml'
    if not pyproject_path.exists():
        try:
            from importlib.metadata import version

            return version('ballontranslator')
        except Exception:
            pass
        return '0.0.0'
    return _read_pyproject_version(pyproject_path)


APP_VERSION = get_current_version()
