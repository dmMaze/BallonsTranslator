"""Qt-free glossary loading, selection, and deterministic rendering."""

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import stat
from typing import Iterable, List, Optional, Sequence, Tuple, Union


GLOSSARY_MODE_MATCHING = "matching"
GLOSSARY_MODE_ALL = "all"
GLOSSARY_MODES = (GLOSSARY_MODE_MATCHING, GLOSSARY_MODE_ALL)

_COMMENT_PREFIXES = ("#", "//", "\\\\")
_TEXT_SUFFIXES = (".txt", ".tsv")

PathValue = Union[str, os.PathLike]


class GlossaryError(ValueError):
    """A configured glossary cannot be read or parsed."""


@dataclass(frozen=True)
class GlossaryEntry:
    """One immutable source-to-translation glossary mapping.

    >>> entry = GlossaryEntry("勇者", "Hero")
    >>> (entry.source, entry.translation, entry.note)
    ('勇者', 'Hero', '')
    """

    source: str
    translation: str
    note: str = ""


def normalize_glossary_path(path: Optional[PathValue]) -> str:
    """Expand and normalize a configured glossary path.

    An empty setting stays empty so callers can treat it as disabled.

    >>> normalize_glossary_path("")
    ''
    """

    if path is None:
        return ""
    raw_path = os.fspath(path)
    if isinstance(raw_path, bytes):
        raw_path = os.fsdecode(raw_path)
    raw_path = raw_path.strip()
    if not raw_path:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    return os.path.realpath(os.path.abspath(os.path.normpath(expanded)))


def load_glossary(path: Optional[PathValue]) -> Tuple[GlossaryEntry, ...]:
    """Load and validate a glossary, returning entries in file order.

    Parsed data is cached by normalized path, nanosecond modification time,
    and file size. An empty path disables the glossary.

    >>> load_glossary("")
    ()
    """

    normalized_path = normalize_glossary_path(path)
    if not normalized_path:
        return ()

    try:
        file_stat = os.stat(normalized_path)
    except FileNotFoundError:
        raise GlossaryError(
            'Glossary file not found: "{}".'.format(normalized_path)
        ) from None
    except OSError as exc:
        raise GlossaryError(
            'Could not access glossary "{}": {}.'.format(
                normalized_path, _os_error_message(exc)
            )
        ) from None

    if not stat.S_ISREG(file_stat.st_mode):
        raise GlossaryError(
            'Glossary path is not a file: "{}".'.format(normalized_path)
        )

    return _load_glossary_cached(
        normalized_path,
        file_stat.st_mtime_ns,
        file_stat.st_size,
    )


def select_glossary(
    entries: Iterable[GlossaryEntry],
    sources: Iterable[str],
    mode: str,
) -> Tuple[GlossaryEntry, ...]:
    """Select entries using deterministic literal matching.

    Matching is case-insensitive, considers the joined page sources, and
    returns each loaded entry at most once in file order.

    >>> entries = (GlossaryEntry("Hero", "勇者"), GlossaryEntry("Mage", "魔法使"))
    >>> select_glossary(entries, ["A HERO appears."], "matching")
    (GlossaryEntry(source='Hero', translation='勇者', note=''),)
    """

    ordered_entries = tuple(entries)
    if mode == GLOSSARY_MODE_ALL:
        return ordered_entries
    if mode != GLOSSARY_MODE_MATCHING:
        raise GlossaryError(
            'Invalid glossary mode "{}"; expected "matching" or "all".'.format(
                mode
            )
        )

    if isinstance(sources, str):
        joined_sources = sources
    else:
        joined_sources = "\n".join(
            str(source) for source in sources if source is not None
        )
    folded_sources = joined_sources.casefold()
    return tuple(
        entry
        for entry in ordered_entries
        if entry.source.casefold() in folded_sources
    )


def render_glossary(entries: Sequence[GlossaryEntry]) -> str:
    """Render selected entries as stable compact JSON.

    >>> render_glossary([GlossaryEntry("勇者", "Hero", "title")])
    '{"glossary":[{"source":"勇者","translation":"Hero","note":"title"}]}'
    """

    if not entries:
        return ""
    payload = {
        "glossary": [
            {
                "source": entry.source,
                "translation": entry.translation,
                "note": entry.note,
            }
            for entry in entries
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=16)
def _load_glossary_cached(
    normalized_path: str,
    _mtime_ns: int,
    _size: int,
) -> Tuple[GlossaryEntry, ...]:
    try:
        text = Path(normalized_path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        line_number = exc.object[:exc.start].count(b"\n") + 1
        raise _line_error(
            normalized_path,
            line_number,
            "file is not valid UTF-8",
        ) from None
    except OSError as exc:
        raise GlossaryError(
            'Could not read glossary "{}": {}.'.format(
                normalized_path, _os_error_message(exc)
            )
        ) from None

    suffix = Path(normalized_path).suffix.casefold()
    if suffix == ".json":
        rows = _parse_json_rows(text, normalized_path)
    elif suffix in _TEXT_SUFFIXES:
        rows = _parse_text_rows(text, normalized_path, suffix)
    else:
        raise GlossaryError(
            'Unsupported glossary format for "{}"; expected .json, .txt, or .tsv.'.format(
                normalized_path
            )
        )
    return _deduplicate_rows(rows, normalized_path)


def _parse_json_rows(
    text: str,
    path: str,
) -> List[Tuple[GlossaryEntry, str]]:
    try:
        decoded_rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _line_error(
            path,
            exc.lineno,
            "invalid JSON: {}".format(exc.msg),
        ) from None
    if not isinstance(decoded_rows, list):
        raise _line_error(path, 1, "expected a JSON array")

    rows: List[Tuple[GlossaryEntry, str]] = []
    for item_number, item in enumerate(decoded_rows, start=1):
        location = "entry {}".format(item_number)
        if not isinstance(item, dict):
            raise _location_error(path, location, "expected an object")
        if "src" not in item:
            raise _location_error(path, location, 'missing "src"')
        if "dst" not in item:
            raise _location_error(path, location, 'missing "dst"')
        rows.append(
            (
                _make_entry(
                    item["src"],
                    item["dst"],
                    item.get("info", ""),
                    path,
                    location,
                ),
                location,
            )
        )
    return rows


def _parse_text_rows(
    text: str,
    path: str,
    suffix: str,
) -> List[Tuple[GlossaryEntry, str]]:
    rows: List[Tuple[GlossaryEntry, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(_COMMENT_PREFIXES):
            continue

        if suffix == ".tsv" or "\t" in line:
            fields = line.split("\t")
            if len(fields) not in (2, 3):
                raise _line_error(
                    path,
                    line_number,
                    "expected two or three tab-separated fields",
                )
            source, translation = fields[:2]
            note = fields[2] if len(fields) == 3 else ""
        else:
            if "->" not in line:
                raise _line_error(
                    path,
                    line_number,
                    'expected "source->translation #optional note" or tab-separated fields',
                )
            source, translation_and_note = line.split("->", 1)
            if "#" in translation_and_note:
                translation, note = translation_and_note.split("#", 1)
            else:
                translation, note = translation_and_note, ""

        rows.append(
            (
                _make_entry(
                    source,
                    translation,
                    note,
                    path,
                    "line {}".format(line_number),
                ),
                "line {}".format(line_number),
            )
        )
    return rows


def _make_entry(
    source: object,
    translation: object,
    note: object,
    path: str,
    location: str,
) -> GlossaryEntry:
    if not isinstance(source, str):
        raise _location_error(path, location, 'field "src" must be a string')
    if not isinstance(translation, str):
        raise _location_error(path, location, 'field "dst" must be a string')
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise _location_error(path, location, 'field "info" must be a string')

    source = source.strip()
    translation = translation.strip()
    note = note.strip()
    if not source:
        raise _location_error(path, location, "source must not be empty")
    if not translation:
        raise _location_error(path, location, "translation must not be empty")
    return GlossaryEntry(source, translation, note)


def _deduplicate_rows(
    rows: Iterable[Tuple[GlossaryEntry, str]],
    path: str,
) -> Tuple[GlossaryEntry, ...]:
    entries: List[GlossaryEntry] = []
    seen_rows = set()
    first_target_by_source = {}
    for entry, location in rows:
        source_key = entry.source.casefold()
        first_target = first_target_by_source.get(source_key)
        if first_target is not None and first_target[0] != entry.translation:
            previous_translation, previous_location = first_target
            raise _location_error(
                path,
                location,
                'source {!r} conflicts with {}: {!r} versus {!r}'.format(
                    entry.source,
                    previous_location,
                    previous_translation,
                    entry.translation,
                ),
            )
        if first_target is None:
            first_target_by_source[source_key] = (entry.translation, location)

        if entry in seen_rows:
            continue
        seen_rows.add(entry)
        entries.append(entry)
    return tuple(entries)


def _line_error(path: str, line_number: int, message: str) -> GlossaryError:
    return _location_error(path, "line {}".format(max(1, line_number)), message)


def _location_error(path: str, location: str, message: str) -> GlossaryError:
    return GlossaryError(
        'Invalid glossary "{}" at {}: {}.'.format(
            path,
            location,
            message.rstrip("."),
        )
    )


def _os_error_message(exc: OSError) -> str:
    return str(exc.strerror or exc).rstrip(".")
