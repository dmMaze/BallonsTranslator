from pathlib import Path
import struct
from types import SimpleNamespace

from ballontranslator.utils.font_registry import (
    FontEntry,
    FontFace,
    FontRegistry,
    _candidate_from_parsed_face,
    _disambiguate_duplicate_weights,
    _sfnt_offsets,
    _system_entry,
    _parse_name_table,
    _system_display_family,
    build_font_registry,
    collect_custom_faces,
    choose_localized_pair,
    ensure_font_registry_overrides,
    load_custom_group_table,
    load_system_alias_table,
    parse_font_names,
    qt_family_weights,
)


class _EmptyFontDatabase:
    def styles(self, _family: str) -> list[str]:
        return []


class _Qt5FontDatabase:
    def styles(self, _family: str) -> list[str]:
        return ['Extra Light', 'Regular', 'Bold']

    def weight(self, _family: str, style: str) -> int:
        return {
            'Extra Light': 12,
            'Regular': 50,
            'Bold': 75,
        }[style]

    def isScalable(self, _family: str, _style: str = '') -> bool:
        return True


class _ApplicationFontDatabase(_EmptyFontDatabase):
    def addApplicationFont(self, _path: str) -> int:
        return 7

    def applicationFontFamilies(self, font_id: int) -> list[str]:
        assert font_id == 7
        return ['Example Type 1']


class _RejectedFontDatabase(_EmptyFontDatabase):
    def __init__(self) -> None:
        self.paths: list[str] = []

    def addApplicationFont(self, _path: str) -> int:
        self.paths.append(_path)
        return -1

    def applicationFontFamilies(self, _font_id: int) -> list[str]:
        raise AssertionError('families must not be queried for a rejected font')


def _name(label: str, language: str, value: str) -> dict:
    return {
        'label': label,
        'language': language,
        'value': value,
        'platform_id': 3,
    }


def test_localized_family_is_display_only() -> None:
    parsed = {
        'face_index': 0,
        'os2_weight': 400,
        'names': [
            _name('typographic_family', 'en-US', 'Example Sans'),
            _name('typographic_family', 'ko-KR', '예제 산스'),
            _name('typographic_subfamily', 'en-US', 'Regular'),
        ],
    }

    face = _candidate_from_parsed_face(
        Path('example.ttf'),
        parsed,
        ['Example Sans'],
        _EmptyFontDatabase(),
        'ko-KR',
    )

    assert face is not None
    assert face.display_family == '예제 산스'
    assert face.canonical_family == 'Example Sans'
    assert face.storage_family == 'Example Sans'


def _name_table(records: list[tuple[int, int, str]]) -> bytes:
    strings = bytearray()
    headers = bytearray()
    for name_id, language_id, text in records:
        encoded = text.encode('utf-16-be')
        headers.extend(struct.pack('>6H', 3, 1, language_id, name_id, len(encoded), len(strings)))
        strings.extend(encoded)
    return struct.pack('>3H', 0, len(records), 6 + len(headers)) + headers + strings


def test_file_name_table_parser_preserves_localized_records(tmp_path: Path) -> None:
    table = _name_table([(1, 0x0409, 'Legacy Sans'), (16, 0x0804, '示例黑体')])
    font_path = tmp_path / 'example.otf'
    font_path.write_bytes(
        struct.pack('>4s4H', b'OTTO', 1, 0, 0, 0)
        + struct.pack('>4s3I', b'name', 0, 28, len(table)) + table
    )
    faces = parse_font_names(font_path)
    assert len(faces) == 1
    assert [(name['label'], name['language'], name['value']) for name in faces[0]['names']] == [
        ('family', 'en-US', 'Legacy Sans'),
        ('typographic_family', 'zh-CN', '示例黑体'),
    ]


def test_system_metadata_localizes_only_display_names(monkeypatch) -> None:
    from qtpy.QtGui import QRawFont

    table = _name_table([
        (1, 0x0409, 'Legacy Sans'),
        (16, 0x0409, 'Example Sans'),
        (16, 0x0804, '示例黑体'),
        (6, 0x0409, 'ExampleSans-Regular'),
    ])
    raw = SimpleNamespace(isValid=lambda: True, fontTable=lambda _: table)
    monkeypatch.setattr(QRawFont, 'fromFont', lambda _: raw)
    monkeypatch.setattr(_Qt5FontDatabase, 'font', lambda *_: object(), raising=False)

    for locale, display in [('zh-CN', '示例黑体'), ('en-US', 'Example Sans')]:
        registry = build_font_registry(_Qt5FontDatabase(), [], ['Legacy Sans'], locale)
        entry = registry.entries()[0]
        assert registry.display_family(entry) == display
        assert entry.storage_family_for_weight(400) == 'Legacy Sans'
        assert registry.resolve_family('Legacy Sans', 400).qt_family == 'Legacy Sans'
        assert registry.resolve_family(display, 400).canonical_family == 'Legacy Sans'
        assert registry.family_for_export('Legacy Sans') == 'Legacy Sans'


def test_system_display_falls_back_for_missing_or_truncated_metadata(monkeypatch) -> None:
    from qtpy.QtGui import QRawFont

    database = SimpleNamespace(font=lambda *_: object())
    for table in (b'', b'\x00', _name_table([(16, 0x0804, '示例')])[:-1]):
        assert _parse_name_table(table) == []
        raw = SimpleNamespace(isValid=lambda: True, fontTable=lambda _: table)
        monkeypatch.setattr(QRawFont, 'fromFont', lambda _: raw)
        assert _system_display_family(database, 'Legacy Sans', 'zh-CN') == 'Legacy Sans'
    table = _name_table([(1, 0x0409, 'Unrelated Fallback Font')])
    assert _system_display_family(database, 'Legacy Sans', 'zh-CN') == 'Legacy Sans'


def test_localized_system_labels_do_not_shadow_saved_names(monkeypatch) -> None:
    monkeypatch.setattr(
        'ballontranslator.utils.font_registry._system_display_family',
        lambda _db, _family, _locale: 'Existing Family',
    )
    registry = build_font_registry(
        _Qt5FontDatabase(), [], ['Existing Family', 'Other Family'], 'en-US',
    )
    assert registry.resolve_family('Existing Family').canonical_family == 'Existing Family'
    assert registry.resolve_family('Other Family').canonical_family == 'Other Family'
    other = registry.resolve_family('Other Family').entry
    display = registry.display_family(other)
    assert display == 'Existing Family (Other Family)'
    assert registry.resolve_family(display).canonical_family == 'Other Family'


def test_system_labels_are_resolved_on_demand_and_cached(monkeypatch) -> None:
    calls = []
    def read_label(_db, family, _locale):
        calls.append(family)
        return 'Display ' + family
    monkeypatch.setattr(
        'ballontranslator.utils.font_registry._system_display_family', read_label,
    )
    registry = build_font_registry(_Qt5FontDatabase(), [], ['One', 'Two'])
    one = registry.resolve_family('One').entry
    assert calls == []
    assert registry.display_family(one, resolve=False) == 'One'
    assert calls == []
    assert registry.display_family(one) == 'Display One'
    assert registry.display_family(one) == 'Display One'
    assert calls == ['One']


def test_family_fallback_prefers_unicode_records_over_legacy_mac_names() -> None:
    records = [
        {**_name('family', '0x0000', 'EPSON ä€ÉSÉVÉbÉNëÃÇl'), 'platform_id': 1},
        _name('family', 'ja-JP', 'EPSON 丸ゴシック体Ｍ'),
    ]
    assert choose_localized_pair([], records, 'zh-CN') == 'EPSON 丸ゴシック体Ｍ'


def test_duplicate_vendor_weights_split_only_with_distinct_styles() -> None:
    faces = [
        FontFace('Example', 'Example', 'Example', 'Thin', 250),
        FontFace('Example', 'Example', 'Example', 'Medium', 250),
        FontFace('Example', 'Example', 'Example', 'Bold', 250),
    ]

    corrected = _disambiguate_duplicate_weights(faces)

    assert [face.weight for face in corrected] == [100, 500, 700]
    assert all('weight_disambiguated_by_style' in face.warnings for face in corrected)


def test_qt5_database_weights_are_canonicalized_at_discovery() -> None:
    database = _Qt5FontDatabase()

    entry = _system_entry(database, 'Example Sans')

    assert qt_family_weights(database, 'Example Sans') == [200, 400, 700]
    assert entry.weights == [200, 400, 700]
    assert [face.weight for face in entry.faces] == [700, 200, 400]


def test_ambiguous_duplicate_vendor_weights_are_preserved() -> None:
    faces = [
        FontFace('Example', 'Example', 'Example', 'Book', 250),
        FontFace('Example', 'Example', 'Example', 'Text', 250),
    ]

    corrected = _disambiguate_duplicate_weights(faces)

    assert [face.weight for face in corrected] == [250, 250]
    assert all(not face.warnings for face in corrected)


def test_grouping_does_not_change_storage_for_shared_family_faces() -> None:
    faces = [
        FontFace('Example Sans', '예제 산스', 'Example Sans', 'Light', 300),
        FontFace('Example Sans', '예제 산스', 'Example Sans', 'Bold', 700),
    ]
    grouped = FontEntry(
        'Example Sans',
        '예제 산스',
        'Example Sans',
        'custom',
        weights=[300, 700],
        faces=faces,
    )

    assert grouped.storage_family_for_weight(300) == faces[0].storage_family
    assert grouped.storage_family_for_weight(700) == faces[1].storage_family


def test_pseudo_group_saves_the_selected_weight_specific_family() -> None:
    faces = [
        FontFace(
            'Example Sans Light', '예제 산스 Light',
            'Example Sans Light', 'Light', 300,
        ),
        FontFace(
            'Example Sans Bold', '예제 산스 Bold',
            'Example Sans Bold', 'Bold', 700,
        ),
    ]
    grouped = FontEntry(
        'Example Sans',
        '예제 산스',
        'Example Sans Light',
        'custom',
        weights=[300, 700],
        faces=faces,
        is_pseudo_group=True,
    )

    assert grouped.storage_family_for_weight(300) == 'Example Sans Light'
    assert grouped.storage_family_for_weight(700) == 'Example Sans Bold'


def test_export_index_omits_ambiguous_shared_qt_family() -> None:
    faces = [
        FontFace(
            'Example Light', 'Example Light', 'Qt Example', 'Light', 300,
        ),
        FontFace(
            'Example Bold', 'Example Bold', 'Qt Example', 'Bold', 700,
        ),
    ]
    registry = FontRegistry(custom_entries=[FontEntry(
        'Example', 'Example', 'Qt Example', 'custom',
        faces=faces, weights=[300, 700], is_pseudo_group=True,
    )])

    assert registry.family_for_export('Qt Example') == 'Qt Example'
    assert registry.family_for_export('Example Bold') == 'Example Bold'


def test_invalid_optional_registry_is_ignored(tmp_path: Path) -> None:
    registry_path = tmp_path / 'font_registry.json'
    registry_path.write_text('{invalid', encoding='utf-8')

    assert load_custom_group_table(str(registry_path)) == {}
    assert load_system_alias_table(str(registry_path)) == {}


def test_font_registry_overrides_are_created_once_from_defaults(
    tmp_path: Path,
) -> None:
    default_path = tmp_path / 'resources' / 'font_registry_overrides.json'
    default_path.parent.mkdir()
    default_path.write_text('{"system_aliases": []}', encoding='utf8')

    override_path = ensure_font_registry_overrides(str(tmp_path))

    assert override_path == tmp_path / 'config' / 'font_registry_overrides.json'
    assert override_path.read_text(encoding='utf8') == '{"system_aliases": []}'

    override_path.write_text('{"custom_groups": []}', encoding='utf8')
    assert ensure_font_registry_overrides(str(tmp_path)) == override_path
    assert override_path.read_text(encoding='utf8') == '{"custom_groups": []}'


def test_invalid_registry_fields_discard_only_the_bad_portion(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / 'font_registry.json'
    registry_path.write_text(
        '{"custom_groups":['
        '{"canonical":7,"members":[]},'
        '{"canonical":"Example","members":['
        '{"canonical":[],"aliases":[]},'
        '{"canonical":"Example Bold","display":4,'
        '"aliases":"bad","weight":"bold","style":[]}'
        ']}],"system_aliases":['
        '{"canonical":9,"aliases":[]},'
        '{"canonical":"Batang","display":4,"aliases":"bad"}'
        ']}',
        encoding='utf-8',
    )

    custom = load_custom_group_table(str(registry_path))
    system = load_system_alias_table(str(registry_path))

    member = custom['example bold']['member']
    assert member['display'] == 'Example Bold'
    assert member['aliases'] == []
    assert 'weight' not in member
    assert 'style' not in member
    assert system['batang']['display'] == 'Batang'
    assert system['batang']['aliases'] == ['Batang']


def test_ttc_face_count_is_limited_by_available_offsets() -> None:
    data = b'ttcf\x00\x01\x00\x00\xff\xff\xff\xff'

    assert _sfnt_offsets(data) == []


def test_non_sfnt_font_registered_by_qt_remains_available(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / 'example.pfb'
    font_path.write_bytes(b'not an sfnt font')

    faces = collect_custom_faces(
        [str(font_path)], _ApplicationFontDatabase(), 'en-US'
    )

    assert len(faces) == 1
    assert faces[0].canonical_family == 'Example Type 1'
    assert faces[0].qt_family == 'Example Type 1'


def test_font_rejected_by_qt_is_not_added_to_registry(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / 'broken.ttf'
    font_path.write_bytes(b'not a font')

    database = _RejectedFontDatabase()
    faces = collect_custom_faces([str(font_path)], database, 'en-US')

    assert faces == []
    assert database.paths == [str(font_path.resolve())]


def test_macos_appledouble_sidecar_is_not_registered(
    tmp_path: Path,
) -> None:
    font_path = tmp_path / '._example.ttf'
    font_path.write_bytes(b'finder metadata')

    database = _RejectedFontDatabase()
    faces = collect_custom_faces([str(font_path)], database, 'en-US')

    assert faces == []
    assert database.paths == []


def test_optional_display_alias_follows_ui_locale(tmp_path: Path) -> None:
    registry_path = tmp_path / 'font_registry.json'
    registry_path.write_text(
        '{"system_aliases":[{"canonical":"Batang",'
        '"display":{"ko-KR":"바탕"},"aliases":["바탕"]}]}',
        encoding='utf-8',
    )

    korean = load_system_alias_table(str(registry_path), 'ko-KR')
    english = load_system_alias_table(str(registry_path), 'en-US')

    assert korean['batang']['display'] == '바탕'
    assert english['batang']['display'] == 'Batang'


def test_registry_build_uses_override_file(tmp_path: Path) -> None:
    registry_path = tmp_path / 'font_registry_overrides.json'
    registry_path.write_text(
        '{"system_aliases":[{"canonical":"Batang",'
        '"display":"User"}]}',
        encoding='utf-8',
    )

    registry = build_font_registry(
        _Qt5FontDatabase(),
        [],
        ['Batang'],
        font_registry_path=str(registry_path),
    )

    assert registry.entries()[0].display_family == 'User'


def test_entry_lookup_keys_include_localized_face_aliases() -> None:
    face = FontFace(
        'Batang', '바탕', 'Batang', aliases={'바탕체 별칭'}
    )
    entry = FontEntry(
        'Batang', '바탕', 'Batang', 'system', faces=[face]
    )

    assert '바탕체 별칭' in entry.lookup_keys()


def test_redundant_weight_family_is_collapsed_into_picker_base() -> None:
    base = FontEntry(
        'Example', 'Example', 'Example', 'system', weights=[300, 500]
    )
    light = FontEntry(
        'Example Light', 'Example Light', 'Example Light', 'system',
        weights=[300],
    )
    registry = FontRegistry(system_entries=[base, light])

    assert registry.entries() == [base]
    assert registry.picker_entry_for_family('Example Light') is base


def test_weight_family_remains_when_picker_base_is_excluded() -> None:
    base = FontEntry(
        'Example', 'Example', 'Example', 'system', weights=[300, 500]
    )
    light = FontEntry(
        'Example Light', 'Example Light', 'Example Light', 'system',
        weights=[300],
    )
    other = FontEntry(
        'Other', 'Other', 'Other', 'system', weights=[300, 400]
    )
    multiweight_light = FontEntry(
        'Other Light', 'Other Light', 'Other Light', 'system',
        weights=[300, 400],
    )
    registry = FontRegistry(
        system_entries=[base, light, other, multiweight_light]
    )

    assert registry.entries(excluded=['Example']) == [
        light, other, multiweight_light,
    ]
    assert registry.entries(excluded=['Example Light']) == [
        base, other, multiweight_light,
    ]


def test_system_weight_alias_rule_does_not_hide_custom_entries() -> None:
    system_base = FontEntry(
        'Example', 'Example', 'Example', 'system', weights=[300, 500]
    )
    system_light = FontEntry(
        'Example Light', 'Example Light', 'Example Light', 'system',
        weights=[300],
    )
    custom_base = FontEntry(
        'Example', 'Custom Example', 'Custom Example', 'custom',
        weights=[500],
    )
    registry = FontRegistry(
        system_entries=[system_base, system_light],
        custom_entries=[custom_base],
    )

    assert registry.entries() == [custom_base, system_light]
    assert registry.picker_entry_for_family('Example Light') is system_light
