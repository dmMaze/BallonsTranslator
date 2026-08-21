from pathlib import Path

from ballontranslator.utils.font_registry import (
    FontEntry,
    FontFace,
    _candidate_from_parsed_face,
    _disambiguate_duplicate_weights,
    load_custom_group_table,
    load_system_alias_table,
)


class _EmptyFontDatabase:
    def styles(self, _family: str) -> list[str]:
        return []


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


def test_duplicate_vendor_weights_split_only_with_distinct_styles() -> None:
    faces = [
        FontFace('Example', 'Example', 'Example', 'Thin', 250),
        FontFace('Example', 'Example', 'Example', 'Medium', 250),
        FontFace('Example', 'Example', 'Example', 'Bold', 250),
    ]

    corrected = _disambiguate_duplicate_weights(faces)

    assert [face.weight for face in corrected] == [100, 500, 700]
    assert all('weight_disambiguated_by_style' in face.warnings for face in corrected)


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


def test_invalid_optional_registry_is_ignored(tmp_path: Path) -> None:
    registry_path = tmp_path / 'font_registry.json'
    registry_path.write_text('{invalid', encoding='utf-8')

    assert load_custom_group_table(str(registry_path)) == {}
    assert load_system_alias_table(str(registry_path)) == {}


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
