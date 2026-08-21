from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path
import struct
from typing import Any, Dict, Iterable, List, Optional, Set


LOGGER = logging.getLogger(__name__)


FONT_EXTS = {'.ttf', '.otf', '.ttc', '.pfb'}
NAME_IDS = {
    1: 'family',
    2: 'subfamily',
    4: 'full_name',
    6: 'postscript_name',
    16: 'typographic_family',
    17: 'typographic_subfamily',
}
WINDOWS_LANGS = {
    0x0404: 'zh-TW',
    0x0409: 'en-US',
    0x0411: 'ja-JP',
    0x0412: 'ko-KR',
    0x0804: 'zh-CN',
}
MAC_LANGS = {
    23: 'ko-KR',
}
MAC_ENCODINGS = {
    3: ['x-mac-korean', 'cp949', 'euc_kr'],
}
WEIGHT_BY_STYLE = {
    'thin': 100,
    'extralight': 200,
    'extra light': 200,
    'ultralight': 200,
    'ultra light': 200,
    'light': 300,
    'regular': 400,
    'normal': 400,
    'book': 400,
    'medium': 500,
    'demibold': 600,
    'demi bold': 600,
    'semibold': 600,
    'semi bold': 600,
    'bold': 700,
    'extrabold': 800,
    'extra bold': 800,
    'ultrabold': 800,
    'ultra bold': 800,
    'black': 900,
    'heavy': 900,
}
EXACT_WEIGHT_BY_STYLE = {
    'b': 700,
    'l': 300,
    'm': 500,
}
WINDOWS_LEGACY_RASTER_FAMILIES = {
    'Fixedsys',
    'MS Sans Serif',
    'MS Serif',
    'Small Fonts',
    'System',
    'Terminal',
}


@dataclass
class FontFace:
    """A concrete renderable font face.

    >>> face = FontFace('Noto Sans KR', 'Noto Sans KR', 'Noto Sans KR', 'Regular', 400)
    >>> face.storage_family
    'Noto Sans KR'
    """

    canonical_family: str
    display_family: str
    qt_family: str
    style_name: str = 'Regular'
    weight: Optional[int] = None
    file_path: Optional[str] = None
    face_index: int = 0
    full_name: Optional[str] = None
    postscript_name: Optional[str] = None
    original_family: Optional[str] = None
    aliases: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)

    @property
    def storage_family(self) -> str:
        return self.canonical_family


@dataclass
class FontEntry:
    """A picker entry that may contain one or more concrete faces.

    Optional custom groups use a pseudo entry for display only. Saving must use
    the selected face canonical instead of blindly saving the entry canonical.

    >>> face = FontFace('Korail Round Gothic Bold', 'Korail', 'Korail Round Gothic Bold', 'B', 700)
    >>> entry = FontEntry('Korail Round Gothic', 'Korail', 'Korail Round Gothic Bold', 'custom', faces=[face], is_pseudo_group=True)
    >>> entry.storage_family_for_weight(700)
    'Korail Round Gothic Bold'
    """

    canonical_family: str
    display_family: str
    qt_family: str
    source: str
    file_paths: List[str] = field(default_factory=list)
    weights: List[int] = field(default_factory=list)
    styles: List[str] = field(default_factory=list)
    faces: List[FontFace] = field(default_factory=list)
    is_scalable: bool = True
    aliases: Set[str] = field(default_factory=set)
    alias_source: str = 'none'
    warnings: List[str] = field(default_factory=list)
    is_pseudo_group: bool = False

    def storage_family_for_weight(self, weight: Optional[int] = None) -> str:
        if not self.is_pseudo_group:
            return self.canonical_family
        face = self.face_for_weight(weight)
        return face.storage_family if face is not None else self.canonical_family

    def face_for_weight(self, weight: Optional[int] = None) -> Optional[FontFace]:
        if not self.faces:
            return None
        if weight is None:
            return self.faces[0]
        weighted = [face for face in self.faces if face.weight is not None]
        if not weighted:
            return self.faces[0]
        return min(weighted, key=lambda face: (abs(face.weight - weight), -face.weight))


@dataclass
class ResolvedFont:
    """Runtime resolution result for a saved or selected family name.

    >>> result = ResolvedFont('Noto Sans KR', 'Noto Sans KR', 'Noto Sans KR')
    >>> result.qt_family
    'Noto Sans KR'
    """

    requested_family: str
    canonical_family: str
    qt_family: str
    entry: Optional[FontEntry] = None
    face: Optional[FontFace] = None


@dataclass
class FontRegistry:
    """Runtime-only font registry.

    >>> entry = FontEntry('A', 'Display A', 'A', 'custom')
    >>> reg = FontRegistry(custom_entries=[entry], system_entries=[])
    >>> reg.resolve_family('Display A').qt_family
    'A'
    """

    custom_entries: List[FontEntry] = field(default_factory=list)
    system_entries: List[FontEntry] = field(default_factory=list)
    entries_by_key: Dict[str, FontEntry] = field(default_factory=dict)
    faces_by_key: Dict[str, tuple] = field(default_factory=dict)

    def __post_init__(self):
        self.rebuild_index()

    def rebuild_index(self):
        self.entries_by_key = {}
        self.faces_by_key = {}
        for entry in [*self.system_entries, *self.custom_entries]:
            keys = {entry.canonical_family, entry.display_family, entry.qt_family, *entry.aliases}
            entry_keys = {normalize_key(key) for key in {entry.canonical_family, entry.display_family, entry.qt_family} if key}
            face_key_counts = defaultdict(int)
            face_keys_by_face = []
            for face in entry.faces:
                face_keys = {face.canonical_family, face.display_family, face.qt_family, *face.aliases}
                normalized_face_keys = {normalize_key(key) for key in face_keys if key}
                face_keys_by_face.append((face, normalized_face_keys))
                for normalized in normalized_face_keys:
                    face_key_counts[normalized] += 1
                keys.update(face_keys)
            for face, normalized_face_keys in face_keys_by_face:
                for normalized in normalized_face_keys:
                    if normalized not in entry_keys and face_key_counts[normalized] == 1:
                        self.faces_by_key[normalized] = (entry, face)
            for key in keys:
                if key:
                    self.entries_by_key[normalize_key(key)] = entry

    def entries(self, only_custom: bool = False) -> List[FontEntry]:
        return self.grouped_entries(only_custom)

    def grouped_entries(self, only_custom: bool = False) -> List[FontEntry]:
        if only_custom:
            return sorted(self.custom_entries, key=lambda entry: entry.display_family.casefold())

        custom_keys = {normalize_key(entry.canonical_family) for entry in self.custom_entries}
        system = [entry for entry in self.system_entries if normalize_key(entry.canonical_family) not in custom_keys]
        return sorted([*system, *self.custom_entries], key=lambda entry: entry.display_family.casefold())

    def separate_face_entries(self, only_custom: bool = False) -> List[FontEntry]:
        entries = []
        custom_keys = {normalize_key(entry.canonical_family) for entry in self.custom_entries}
        system_entries = [entry for entry in self.system_entries if normalize_key(entry.canonical_family) not in custom_keys]
        source_entries = self.custom_entries if only_custom else [*system_entries, *self.custom_entries]
        for entry in source_entries:
            if not entry.faces:
                entries.append(entry)
                continue
            for face in entry.faces:
                display_family = face.display_family
                if normalize_key(display_family) == normalize_key(entry.display_family) and face.style_name:
                    display_family = f'{display_family} {face.style_name}'
                entries.append(
                    FontEntry(
                        canonical_family=face.storage_family,
                        display_family=display_family,
                        qt_family=face.qt_family,
                        source=entry.source,
                        file_paths=list(entry.file_paths),
                        weights=[face.weight] if face.weight is not None else [],
                        styles=[face.style_name] if face.style_name else [],
                        faces=[face],
                        is_scalable=entry.is_scalable,
                        aliases={alias for alias in {face.canonical_family, face.display_family, face.qt_family, *face.aliases} if alias},
                        alias_source=entry.alias_source,
                        warnings=list(entry.warnings),
                    )
                )
        return sorted(entries, key=lambda item: item.display_family.casefold())

    def legacy_family_list(self, only_custom: bool = False) -> List[str]:
        """Return renderable family strings for the old QFontComboBox path."""
        families = []
        seen = set()
        for entry in self.entries(only_custom):
            family = entry.qt_family or entry.canonical_family
            key = normalize_key(family)
            if family and key not in seen:
                seen.add(key)
                families.append(family)
        return families

    def resolve_family(self, family: str, weight: Optional[int] = None) -> ResolvedFont:
        key = normalize_key(family)
        entry = self.entries_by_key.get(key)
        if entry is None:
            return ResolvedFont(family, family, family)
        face_match = self.faces_by_key.get(key)
        face = face_match[1] if face_match is not None and face_match[0] is entry else entry.face_for_weight(weight)
        qt_family = face.qt_family if face is not None else entry.qt_family
        canonical = face.storage_family if entry.is_pseudo_group and face is not None else entry.canonical_family
        return ResolvedFont(family, canonical, qt_family, entry=entry, face=face)


def normalize_key(value: str) -> str:
    return ' '.join(value.casefold().split())


def _name_id_label(name_id: int) -> str:
    return NAME_IDS.get(name_id, f'name_{name_id}')


def _decode_name(raw: bytes, platform_id: int, encoding_id: int) -> str:
    """Decode a TrueType/OpenType name table string.

    >>> _decode_name(bytes.fromhex('b3 aa b4 ae'), 1, 3)
    '나눔'
    """
    encodings = []
    if platform_id in (0, 3):
        encodings.extend(['utf-16-be', 'utf-8'])
    elif platform_id == 1:
        encodings.extend(MAC_ENCODINGS.get(encoding_id, []))
        encodings.extend(['mac_roman', 'latin-1'])
    else:
        encodings.extend(['utf-8', 'latin-1'])

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        text = text.replace('\x00', '').strip()
        if text:
            return text
    return raw.decode('latin-1', errors='replace').replace('\x00', '').strip()


def _language_label(platform_id: int, language_id: int) -> str:
    if platform_id == 1:
        return MAC_LANGS.get(language_id, f'0x{language_id:04x}')
    return WINDOWS_LANGS.get(language_id, f'0x{language_id:04x}')


def _read_u16(data: bytes, offset: int) -> int:
    return struct.unpack_from('>H', data, offset)[0]


def _read_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from('>L', data, offset)[0]


def _sfnt_offsets(data: bytes) -> List[int]:
    if len(data) < 12:
        return []

    tag = data[:4]
    if tag == b'ttcf':
        count = _read_u32(data, 8)
        offsets = []
        for index in range(count):
            pos = 12 + index * 4
            if pos + 4 <= len(data):
                offsets.append(_read_u32(data, pos))
        return offsets

    if tag in (b'\x00\x01\x00\x00', b'OTTO', b'true'):
        return [0]

    return []


def _table_offset(data: bytes, sfnt_offset: int, table_tag: bytes) -> Optional[tuple]:
    if sfnt_offset + 12 > len(data):
        return None
    num_tables = _read_u16(data, sfnt_offset + 4)
    table_dir = sfnt_offset + 12
    for index in range(num_tables):
        pos = table_dir + index * 16
        if pos + 16 > len(data):
            return None
        if data[pos:pos + 4] == table_tag:
            return _read_u32(data, pos + 8), _read_u32(data, pos + 12)
    return None


def _parse_os2_weight(data: bytes, sfnt_offset: int) -> Optional[int]:
    table = _table_offset(data, sfnt_offset, b'OS/2')
    if table is None:
        return None
    offset, length = table
    if length < 6 or offset + 6 > len(data):
        return None
    weight = _read_u16(data, offset + 4)
    # Some legacy fonts store 1-9 instead of 100-900.
    if 1 <= weight <= 9:
        weight *= 100
    if 1 <= weight <= 1000:
        return weight
    return None


def parse_font_names(path: Path) -> List[Dict[str, Any]]:
    data = path.read_bytes()
    faces = []
    for face_index, sfnt_offset in enumerate(_sfnt_offsets(data)):
        os2_weight = _parse_os2_weight(data, sfnt_offset)
        table = _table_offset(data, sfnt_offset, b'name')
        if table is None:
            faces.append({'face_index': face_index, 'error': 'name table not found', 'names': [], 'os2_weight': os2_weight})
            continue

        name_offset, name_length = table
        if name_offset + min(name_length, 6) > len(data):
            faces.append({'face_index': face_index, 'error': 'invalid name table', 'names': [], 'os2_weight': os2_weight})
            continue

        count = _read_u16(data, name_offset + 2)
        string_offset = name_offset + _read_u16(data, name_offset + 4)
        names = []
        seen = set()
        for record_index in range(count):
            pos = name_offset + 6 + record_index * 12
            if pos + 12 > len(data):
                break
            platform_id = _read_u16(data, pos)
            encoding_id = _read_u16(data, pos + 2)
            language_id = _read_u16(data, pos + 4)
            name_id = _read_u16(data, pos + 6)
            length = _read_u16(data, pos + 8)
            offset = _read_u16(data, pos + 10)
            if name_id not in NAME_IDS:
                continue

            raw_start = string_offset + offset
            raw_end = raw_start + length
            if raw_end > len(data):
                continue
            value = _decode_name(data[raw_start:raw_end], platform_id, encoding_id)
            if not value:
                continue
            dedupe_key = (platform_id, encoding_id, language_id, name_id, value)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            names.append(
                {
                    'label': _name_id_label(name_id),
                    'name_id': name_id,
                    'value': value,
                    'platform_id': platform_id,
                    'encoding_id': encoding_id,
                    'language_id': language_id,
                    'language': _language_label(platform_id, language_id),
                }
            )
        faces.append({'face_index': face_index, 'names': names, 'os2_weight': os2_weight})
    return faces


def records_by_label(face: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    return [record for record in face.get('names', []) if record.get('label') == label]


def choose_english(records: Iterable[Dict[str, Any]]) -> Optional[str]:
    records = list(records)
    for record in records:
        if record.get('language') == 'en-US' and record.get('value'):
            return record['value']
    for record in records:
        value = record.get('value')
        if value and value.isascii():
            return value
    return None


def choose_localized(records: Iterable[Dict[str, Any]], locale: str) -> Optional[str]:
    records = list(records)
    locale = locale.replace('_', '-')
    records_by_platform = sorted(records, key=_name_record_platform_priority)
    for record in records_by_platform:
        if record.get('language') == locale and record.get('value'):
            return record['value']
    language_prefix = locale.split('-', 1)[0]
    for record in records_by_platform:
        language = str(record.get('language', ''))
        if language.startswith(language_prefix) and record.get('value'):
            return record['value']
    return choose_english(records) or choose_first(records)


def _name_record_platform_priority(record: Dict[str, Any]) -> int:
    platform_id = record.get('platform_id')
    return {3: 0, 0: 1, 1: 2}.get(platform_id, 3)


def choose_localized_pair(
    primary_records: Iterable[Dict[str, Any]],
    fallback_records: Iterable[Dict[str, Any]],
    locale: str,
) -> Optional[str]:
    """Prefer exact-locale records before falling back to broader name IDs.

    >>> choose_localized_pair([{'language': '0x0017', 'value': 'A'}], [{'language': 'ko-KR', 'value': '가'}], 'ko-KR')
    '가'
    """
    primary_records = list(primary_records)
    fallback_records = list(fallback_records)
    locale = locale.replace('_', '-')
    language_prefix = locale.split('-', 1)[0]

    for records in (primary_records, fallback_records):
        for record in sorted(records, key=_name_record_platform_priority):
            if record.get('language') == locale and record.get('value'):
                return record['value']
    for records in (primary_records, fallback_records):
        for record in sorted(records, key=_name_record_platform_priority):
            language = str(record.get('language', ''))
            if language.startswith(language_prefix) and record.get('value'):
                return record['value']
    return choose_english(primary_records) or choose_english(fallback_records) or choose_first(primary_records) or choose_first(fallback_records)


def choose_first(records: Iterable[Dict[str, Any]]) -> Optional[str]:
    for record in records:
        if record.get('value'):
            return record['value']
    return None


def simplify_style(value: Optional[str]) -> str:
    if not value:
        return 'Regular'
    return value.strip() or 'Regular'


def infer_weight(style: str, qt_weights: Iterable[int]) -> Optional[int]:
    """Guess a weight from a style name; OS/2 usWeightClass should take priority over this.

    >>> infer_weight('8 ExtraBold', [])
    800
    """
    normalized_style = normalize_key(style)
    if normalized_style in EXACT_WEIGHT_BY_STYLE:
        return EXACT_WEIGHT_BY_STYLE[normalized_style]
    if normalized_style in WEIGHT_BY_STYLE:
        return WEIGHT_BY_STYLE[normalized_style]
    # Longest token first so 'extrabold' wins over its substring 'bold'.
    for token in sorted(WEIGHT_BY_STYLE, key=len, reverse=True):
        if token in normalized_style:
            return WEIGHT_BY_STYLE[token]
    qt_weights = [weight for weight in qt_weights if weight is not None]
    if qt_weights:
        return sorted(qt_weights, key=lambda value: (abs(value - 400), value))[0]
    return None


def qt_family_weights(qfont_db: Any, family: str) -> List[int]:
    weights = []
    for style in qfont_db.styles(family):
        try:
            weights.append(int(qfont_db.weight(family, style)))
        except Exception:
            continue
    return sorted(set(weights))


def _looks_corrupt_qt_family(family: str) -> bool:
    return bool(family) and all(char in {'?', ' '} for char in family)


def _choose_qt_family(canonical_family: Optional[str], qt_families: List[str]) -> Optional[str]:
    if canonical_family:
        for family in qt_families:
            if normalize_key(family) == normalize_key(canonical_family):
                return family
    for family in qt_families:
        if not _looks_corrupt_qt_family(family):
            return family
    return canonical_family


def _candidate_from_parsed_face(
    font_path: Path,
    parsed_face: Dict[str, Any],
    qt_families: List[str],
    qfont_db: Any,
    locale: str,
) -> Optional[FontFace]:
    if parsed_face.get('error'):
        family = qt_families[0] if qt_families else font_path.stem
        return FontFace(
            canonical_family=family,
            display_family=family,
            qt_family=family,
            weight=parsed_face.get('os2_weight'),
            file_path=str(font_path),
            face_index=int(parsed_face.get('face_index', 0)),
            warnings=[f"parse_error: {parsed_face['error']}"],
        )

    typo_family = records_by_label(parsed_face, 'typographic_family')
    family = records_by_label(parsed_face, 'family')
    typo_subfamily = records_by_label(parsed_face, 'typographic_subfamily')
    subfamily = records_by_label(parsed_face, 'subfamily')
    full_name = records_by_label(parsed_face, 'full_name')
    postscript_name = records_by_label(parsed_face, 'postscript_name')

    english_family = choose_english(typo_family) or choose_english(family)
    localized_family = choose_localized_pair(typo_family, family, locale)
    canonical_family = english_family or choose_first(typo_family) or choose_first(family)
    qt_family = _choose_qt_family(canonical_family, qt_families)
    if not canonical_family:
        canonical_family = qt_family
    if not canonical_family or not qt_family:
        return None

    style_name = simplify_style(
        choose_english(typo_subfamily)
        or choose_english(subfamily)
        or choose_first(typo_subfamily)
        or choose_first(subfamily)
    )
    display_family = localized_family or english_family or canonical_family
    display_face = choose_localized(full_name, locale)
    weight = parsed_face.get('os2_weight')
    if weight is None:
        qt_weights = []
        for family_name in [qt_family, *qt_families]:
            qt_weights.extend(qt_family_weights(qfont_db, family_name))
        weight = infer_weight(style_name, qt_weights)

    aliases = {canonical_family, display_family, qt_family, *[family for family in qt_families if not _looks_corrupt_qt_family(family)]}
    for records in (typo_family, family, full_name, postscript_name):
        aliases.update(record['value'] for record in records if record.get('value'))

    warnings = []
    if localized_family and english_family and normalize_key(localized_family) != normalize_key(english_family):
        warnings.append('localized_display_differs')
    if qt_family and canonical_family and normalize_key(qt_family) != normalize_key(canonical_family):
        warnings.append('qt_family_differs_from_canonical')

    return FontFace(
        canonical_family=canonical_family,
        display_family=display_family,
        qt_family=qt_family,
        style_name=style_name,
        weight=weight,
        file_path=str(font_path),
        face_index=int(parsed_face.get('face_index', 0)),
        full_name=display_face,
        postscript_name=choose_english(postscript_name) or choose_first(postscript_name),
        original_family=choose_first(family),
        aliases={alias for alias in aliases if alias},
        warnings=warnings,
    )


def _json_groups(raw: Any, section: str) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        groups = raw.get(section, raw.get('groups', []))
    elif isinstance(raw, list):
        groups = raw
    else:
        return []
    return groups if isinstance(groups, list) else []


def _load_json_groups(
    path: Optional[str], section: str
) -> List[Dict[str, Any]]:
    if not path:
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        LOGGER.warning('Ignore invalid font registry %s: %s', path, exc)
        return []
    return _json_groups(raw, section)


def _localized_config_display(
    value: Any, canonical: str, locale: str
) -> str:
    if not isinstance(value, dict):
        return value if isinstance(value, str) and value else canonical
    normalized_locale = locale.replace('_', '-').casefold()
    language = normalized_locale.split('-', 1)[0]
    by_locale = {
        key.replace('_', '-').casefold(): text
        for key, text in value.items()
        if isinstance(key, str) and isinstance(text, str) and text
    }
    return (
        by_locale.get(normalized_locale)
        or by_locale.get(language)
        or by_locale.get('default')
        or canonical
    )


def load_custom_group_table(
    path: Optional[str], locale: str = 'en-US'
) -> Dict[str, Dict[str, Any]]:
    table = {}
    for group in _load_json_groups(path, 'custom_groups'):
        if not isinstance(group, dict) or not group.get('canonical'):
            LOGGER.warning('Ignore invalid custom font group: %r', group)
            continue
        canonical = group['canonical']
        display = _localized_config_display(
            group.get('display'), canonical, locale
        )
        members = group.get('members', [])
        if not isinstance(members, list):
            LOGGER.warning('Ignore invalid members for font group %s', canonical)
            continue
        normalized_group = {
            'canonical': canonical,
            'display': display,
            'members': members,
            'note': group.get('note', ''),
        }
        for member in members:
            if not isinstance(member, dict) or not member.get('canonical'):
                LOGGER.warning(
                    'Ignore invalid member in font group %s: %r',
                    canonical,
                    member,
                )
                continue
            member = dict(member)
            member['display'] = _localized_config_display(
                member.get('display'), member['canonical'], locale
            )
            member_names = [member['canonical'], *member.get('aliases', [])]
            for name in member_names:
                table[normalize_key(name)] = {**normalized_group, 'member': member}
    return table


def load_system_alias_table(
    path: Optional[str], locale: str = 'en-US'
) -> Dict[str, Dict[str, Any]]:
    table = {}
    for group in _load_json_groups(path, 'system_aliases'):
        if not isinstance(group, dict) or not group.get('canonical'):
            LOGGER.warning('Ignore invalid system font alias: %r', group)
            continue
        canonical = group['canonical']
        aliases = [canonical, *group.get('aliases', [])]
        normalized_group = {
            'canonical': canonical,
            'display': _localized_config_display(
                group.get('display'), canonical, locale
            ),
            'aliases': aliases,
            'note': group.get('note', ''),
        }
        for alias in aliases:
            table[normalize_key(alias)] = normalized_group
    return table


def collect_custom_faces(font_paths: Iterable[str], qfont_db: Any, locale: str) -> List[FontFace]:
    faces = []
    for font_path_str in font_paths:
        font_path = Path(font_path_str).resolve()
        font_id = qfont_db.addApplicationFont(str(font_path))
        qt_families = [
            family.strip()
            for family in (qfont_db.applicationFontFamilies(font_id) if font_id >= 0 else [])
            if family and family.strip()
        ]
        try:
            parsed_faces = parse_font_names(font_path)
        except Exception as exc:
            parsed_faces = [{'face_index': 0, 'error': repr(exc), 'names': []}]
        for parsed_face in parsed_faces:
            candidate = _candidate_from_parsed_face(font_path, parsed_face, qt_families, qfont_db, locale)
            if candidate is not None:
                faces.append(candidate)
    return faces


def _custom_group_member_face(face: FontFace, custom_group_table: Dict[str, Dict[str, Any]]) -> FontFace:
    group_member = custom_group_table.get(normalize_key(face.canonical_family), {}).get('member', {})
    weight = group_member.get('weight', face.weight)
    style_name = group_member.get('style', face.style_name)
    aliases = set(face.aliases)
    aliases.update(group_member.get('aliases', []))
    display_family = group_member.get('display', face.display_family)
    return replace(
        face,
        display_family=display_family,
        style_name=style_name,
        weight=int(weight) if weight is not None else None,
        aliases={alias for alias in aliases if alias},
    )


def _disambiguate_duplicate_weights(faces: List[FontFace]) -> List[FontFace]:
    """Split faces that declare the same OS/2 weight using their style names.

    Vendors sometimes pin several light faces to 250 to dodge an old GDI
    rendering bug, which would make all but one of them unreachable by weight.

    >>> thin = FontFace('P', 'P', 'P', '1 Thin', 250)
    >>> extralight = FontFace('P', 'P', 'P', '2 ExtraLight', 250)
    >>> [face.weight for face in _disambiguate_duplicate_weights([thin, extralight])]
    [100, 200]
    """
    by_weight: Dict[Optional[int], List[FontFace]] = defaultdict(list)
    for face in faces:
        by_weight[face.weight].append(face)

    result = []
    for weight, group in by_weight.items():
        if weight is None or len(group) == 1:
            result.extend(group)
            continue
        inferred = [infer_weight(face.style_name, []) for face in group]
        if all(value is not None for value in inferred) and len(set(inferred)) == len(group):
            result.extend(
                replace(face, weight=value, warnings=[*face.warnings, 'weight_disambiguated_by_style'])
                for face, value in zip(group, inferred)
            )
        else:
            result.extend(group)
    return result


def build_custom_entries(faces: List[FontFace], custom_group_table: Optional[Dict[str, Dict[str, Any]]] = None) -> List[FontEntry]:
    custom_group_table = custom_group_table or {}
    groups: Dict[str, List[FontFace]] = defaultdict(list)
    for face in faces:
        group = custom_group_table.get(normalize_key(face.canonical_family))
        group_key = normalize_key(group['canonical']) if group else normalize_key(face.canonical_family)
        groups[group_key].append(face)

    entries = []
    for _, group_faces in groups.items():
        first = group_faces[0]
        custom_group = custom_group_table.get(normalize_key(first.canonical_family))
        display_family = custom_group['display'] if custom_group else first.display_family
        canonical_family = custom_group['canonical'] if custom_group else first.canonical_family
        is_pseudo_group = custom_group is not None
        if is_pseudo_group:
            group_faces = [_custom_group_member_face(face, custom_group_table) for face in group_faces]
        else:
            group_faces = _disambiguate_duplicate_weights(group_faces)
        first = group_faces[0]

        weights = []
        styles = []
        aliases = {canonical_family, display_family}
        file_paths = []
        warnings = []
        for face in group_faces:
            if face.weight is not None:
                weights.append(int(face.weight))
            if face.style_name:
                styles.append(face.style_name)
            if face.file_path:
                file_paths.append(face.file_path)
            aliases.update(face.aliases)
            warnings.extend(face.warnings)

        if is_pseudo_group and len({face.canonical_family for face in group_faces}) > 1:
            warnings.append('grouped_by_optional_custom_table')
        qt_family = first.qt_family
        entries.append(
            FontEntry(
                canonical_family=canonical_family,
                display_family=display_family,
                qt_family=qt_family,
                source='custom',
                file_paths=sorted(set(file_paths)),
                weights=sorted(set(weights)),
                styles=sorted(set(styles), key=str.casefold),
                faces=sorted(group_faces, key=lambda face: (face.weight or 400, face.canonical_family.casefold())),
                is_scalable=True,
                aliases={alias for alias in aliases if alias},
                alias_source='optional-table' if is_pseudo_group else 'name-table',
                warnings=sorted(set(warnings)),
                is_pseudo_group=is_pseudo_group,
            )
        )
    return sorted(entries, key=lambda entry: entry.display_family.casefold())


def _system_entry(qfont_db: Any, family: str) -> FontEntry:
    styles = sorted(qfont_db.styles(family), key=str.casefold)
    weights = sorted({int(qfont_db.weight(family, style)) for style in styles})
    scalable = any(qfont_db.isScalable(family, style) for style in styles) if styles else qfont_db.isScalable(family)
    warnings = []
    if family in WINDOWS_LEGACY_RASTER_FAMILIES:
        warnings.append('windows_legacy_raster_candidate')
    if not scalable:
        warnings.append('not_scalable')
    faces = [
        FontFace(
            canonical_family=family,
            display_family=family,
            qt_family=family,
            style_name=style,
            weight=int(qfont_db.weight(family, style)),
            aliases={family},
        )
        for style in styles
    ]
    return FontEntry(
        canonical_family=family,
        display_family=family,
        qt_family=family,
        source='system',
        weights=weights,
        styles=styles,
        faces=faces,
        is_scalable=scalable,
        aliases={family},
        warnings=warnings,
    )


def merge_system_alias_entries(entries: List[FontEntry], alias_table: Dict[str, Dict[str, Any]]) -> List[FontEntry]:
    if not alias_table:
        return entries

    grouped: Dict[str, List[FontEntry]] = defaultdict(list)
    passthrough = []
    for entry in entries:
        alias_group = alias_table.get(normalize_key(entry.canonical_family))
        if alias_group is None:
            passthrough.append(entry)
        else:
            grouped[alias_group['canonical']].append(entry)

    merged_entries = []
    for canonical, group_entries in grouped.items():
        alias_group = alias_table[normalize_key(canonical)]
        display = alias_group.get('display', canonical)
        aliases = {canonical, display, *alias_group.get('aliases', [])}
        primary = next((entry for entry in group_entries if normalize_key(entry.canonical_family) == normalize_key(canonical)), group_entries[0])
        file_paths = sorted({path for entry in group_entries for path in entry.file_paths})
        weights = sorted({weight for weight in primary.weights if weight is not None})
        styles = sorted({style for style in primary.styles if style}, key=str.casefold)
        faces = list(primary.faces)
        warnings = sorted({warning for entry in group_entries for warning in entry.warnings})
        if len(group_entries) > 1:
            warnings.append('merged_by_optional_alias_table')

        merged_aliases = {alias for entry in group_entries for alias in entry.aliases}
        merged_aliases.update(alias for alias in aliases if alias)
        merged_entries.append(
            FontEntry(
                canonical_family=canonical,
                display_family=display,
                qt_family=primary.qt_family,
                source='system',
                file_paths=file_paths,
                weights=weights,
                styles=styles,
                faces=faces,
                is_scalable=any(entry.is_scalable for entry in group_entries),
                aliases=merged_aliases,
                alias_source='optional-table',
                warnings=warnings,
            )
        )

    return sorted([*passthrough, *merged_entries], key=lambda entry: entry.display_family.casefold())


def build_font_registry(
    qfont_db: Any,
    font_paths: Iterable[str],
    system_families: Iterable[str],
    locale: str = 'en-US',
    font_registry_config_path: Optional[str] = None,
    custom_group_table_path: Optional[str] = None,
    system_alias_table_path: Optional[str] = None,
) -> FontRegistry:
    custom_group_table = load_custom_group_table(
        font_registry_config_path, locale
    )
    custom_group_table.update(
        load_custom_group_table(custom_group_table_path, locale)
    )
    system_alias_table = load_system_alias_table(
        font_registry_config_path, locale
    )
    system_alias_table.update(
        load_system_alias_table(system_alias_table_path, locale)
    )
    custom_faces = collect_custom_faces(font_paths, qfont_db, locale)
    custom_entries = build_custom_entries(custom_faces, custom_group_table)
    system_entries = [_system_entry(qfont_db, family) for family in sorted(system_families, key=str.casefold)]
    system_entries = merge_system_alias_entries(system_entries, system_alias_table)
    return FontRegistry(custom_entries=custom_entries, system_entries=system_entries)
