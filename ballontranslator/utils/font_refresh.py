"""Runtime font refresh boundaries; no system installation or private Qt API."""
from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .font_registry import (
    FontRegistry, RegisteredCustomFont, _candidate_from_parsed_face,
    _system_entry, build_custom_entries, merge_system_alias_entries,
    parse_font_name_data,
)

from ballontranslator.utils.logger import logger as LOGGER
FONT_EXTENSIONS = {'.ttf', '.otf', '.ttc', '.pfb'}


def log_font_refresh_debug(message: str, *args: object) -> None:
    """Emit optional diagnostics without changing shared logger configuration.

    >>> callable(log_font_refresh_debug)
    True
    """
    if os.environ.get('BT_FONT_REFRESH_DEBUG') == '1':
        LOGGER.debug(message, *args, stacklevel=2)


def runtime_font_refresh_supported() -> bool:
    """Use Qt 6.4 as a product support gate, not as a private-API requirement."""
    from qtpy import QT_VERSION
    from . import shared
    return not shared.HEADLESS and tuple(int(part) for part in QT_VERSION.split('.')[:2]) >= (6, 4)


@dataclass(frozen=True)
class FontconfigRefresh:
    status: str
    interval: Optional[int] = None
    detail: str = ''


def reinitialize_current_fontconfig() -> FontconfigRefresh:
    """Refresh the default FcConfig used by a normal Linux fontconfig backend.

    This assumes Qt and ctypes resolve the same shared library. Reinitializing
    replaces the default config, including programmatic changes to that config.

    >>> FontconfigRefresh('skipped').interval is None
    True
    """
    if not sys.platform.startswith('linux'):
        return FontconfigRefresh('skipped')
    errors = []
    for name in ('libfontconfig.so.1', 'libfontconfig.so'):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError as error:
            errors.append(str(error))
    else:
        return FontconfigRefresh('unavailable', detail='; '.join(errors))
    interval = None
    try:
        current = library.FcConfigGetCurrent
        current.argtypes = []
        current.restype = ctypes.c_void_p
        interval_fn = library.FcConfigGetRescanInterval
        interval_fn.argtypes = [ctypes.c_void_p]
        interval_fn.restype = ctypes.c_int
        interval = interval_fn(current())
        refresh = library.FcInitReinitialize
        refresh.argtypes = []
        refresh.restype = ctypes.c_int
    except AttributeError as error:
        return FontconfigRefresh('unavailable', interval, str(error))
    if not refresh():
        return FontconfigRefresh('failed', interval, 'FcInitReinitialize returned false')
    return FontconfigRefresh('refreshed', interval)


@dataclass(frozen=True)
class CustomFontFile:
    data: bytes
    fingerprint: tuple[int, int]
    names: List[Dict[str, Any]]


def scan_custom_fonts(
    directory: Path, fingerprints: Dict[str, tuple[int, int]],
) -> Dict[str, Optional[CustomFontFile]]:
    """Read changed files off the GUI thread; None retains an existing entry.

    Failure to read an existing file must not turn into an unintended removal.
    Directory enumeration errors abort the snapshot rather than publish a
    partially enumerated directory as a set of deletions.

    >>> scan_custom_fonts(Path('/nonexistent/font-refresh-example'), {})
    {}
    """
    log_font_refresh_debug('[font-refresh][scan] directory=%s previous=%d', directory, len(fingerprints))
    result: Dict[str, Optional[CustomFontFile]] = {}
    if not directory.exists():
        log_font_refresh_debug('[font-refresh][scan] directory absent; snapshot empty')
        return result
    def raise_error(error: OSError) -> None:
        raise error
    for folder, _dirs, names in os.walk(directory, onerror=raise_error):
        for name in sorted(names):
            path = Path(folder) / name
            if name.startswith('._') or path.suffix.casefold() not in FONT_EXTENSIONS:
                continue
            key = str(path.resolve())
            result[key] = None
            try:
                stat = path.stat()
                fingerprint = (stat.st_size, stat.st_mtime_ns)
                if fingerprints.get(key) == fingerprint:
                    continue
                data = path.read_bytes()
                after = path.stat()
                if (after.st_size, after.st_mtime_ns) != fingerprint:
                    raise OSError('Font changed while reading: ' + key)
                try:
                    parsed = parse_font_name_data(data)
                except Exception as error:
                    LOGGER.warning('Cannot parse font metadata %s: %s', key, error)
                    parsed = []
                result[key] = CustomFontFile(data, fingerprint, parsed)
            except OSError as error:
                LOGGER.warning('Keeping previous font registration for %s: %s', key, error)
    log_font_refresh_debug('[font-refresh][scan] found=%d changed=%d retained=%d removed=%d',
                len(result), sum(value is not None for value in result.values()),
                sum(value is None for value in result.values()), len(set(fingerprints) - set(result)))
    return result


def refresh_font_registry(
    database: Any, previous: FontRegistry, locale: str,
    custom_groups: Dict[str, Any], system_aliases: Dict[str, Any],
    files: Optional[Dict[str, Optional[CustomFontFile]]] = None,
) -> FontRegistry:
    """Synchronize Qt-visible system fonts and optionally changed custom files.

    Qt calls belong to the GUI thread. Registration ownership survives a later
    metadata failure; the caller only publishes the new picker index on success.

    >>> callable(refresh_font_registry)
    True
    """
    registrations = previous.registrations
    if files is not None:
        for path, snapshot in files.items():
            if snapshot is None:
                continue
            log_font_refresh_debug('[font-refresh][custom] registering %s (%s)', path, 'replace' if path in registrations else 'add')
            font_id = database.addApplicationFontFromData(snapshot.data)
            if font_id < 0:
                LOGGER.warning('Unable to register custom font %s; retaining previous face', path)
                continue
            try:
                families = list(database.applicationFontFamilies(font_id))
                if not families:
                    raise ValueError('No font families registered')
                faces = []
                for names in snapshot.names or [{'face_index': 0, 'names': []}]:
                    face = _candidate_from_parsed_face(Path(path), names, families, database, locale)
                    if face is not None:
                        faces.append(face)
                old = registrations.get(path)
                if old is not None and not database.removeApplicationFont(old.font_id):
                    raise RuntimeError('Unable to remove previous custom font registration')
                registrations[path] = RegisteredCustomFont(font_id, snapshot.fingerprint, faces)
            except Exception:
                database.removeApplicationFont(font_id)
                raise
        for path in list(registrations):
            if path not in files:
                if not database.removeApplicationFont(registrations[path].font_id):
                    raise RuntimeError('Unable to remove custom font: ' + path)
                log_font_refresh_debug('[font-refresh][custom] removed %s', path)
                del registrations[path]
    custom_entries = previous.custom_entries if files is None else build_custom_entries(
        [face for registration in registrations.values() for face in registration.faces], custom_groups,
    )
    system_entries = merge_system_alias_entries(
        [_system_entry(database, family) for family in sorted(database.families(), key=str.casefold)],
        system_aliases,
    )
    registry = FontRegistry(custom_entries=custom_entries, system_entries=system_entries,
                            registrations=dict(registrations))
    registry._font_database = database
    registry._display_locale = locale
    return registry


def invalidate_qt_fonts(database: Any, seed: bytes) -> None:
    """Exploit application-font removal to invalidate Qt, without clearing peers.

    This is a tested implementation side effect, not a public refresh contract.

    >>> callable(invalidate_qt_fonts)
    True
    """
    from time import perf_counter
    started = perf_counter()
    log_font_refresh_debug('[font-refresh][qt] adding temporary seed; bytes=%d', len(seed))
    font_id = database.addApplicationFontFromData(seed)
    if font_id < 0:
        raise RuntimeError('Qt rejected the font refresh resource')
    if not database.removeApplicationFont(font_id):
        # Retry cleanup of only our own ID; never remove all application fonts.
        database.removeApplicationFont(font_id)
        raise RuntimeError('Qt could not release the font refresh resource')

    log_font_refresh_debug('[font-refresh][qt] removed temporary ID=%d; invalidated in %.1f ms', font_id, (perf_counter() - started) * 1000)
