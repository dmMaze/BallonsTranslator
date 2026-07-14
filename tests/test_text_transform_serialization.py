import copy
import json
import math
import os
import sys
import unittest
import warnings

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import (
    QColor,
    QFont,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from qtpy.QtWidgets import QApplication


def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


_APP = qapp()

EMPTY_PROJECT_DIR = os.path.join(
    os.path.dirname(__file__), 'fixtures', 'text_transform', 'empty-project'
)
FIXTURE_DIR = os.path.dirname(EMPTY_PROJECT_DIR)


def load_fixture(name):
    with open(os.path.join(FIXTURE_DIR, name), 'r', encoding='utf-8') as fixture:
        return json.load(fixture)

from ballontranslator.utils.proj_imgtrans import (
    AmbiguousLegacyTextTransformError,
    InvalidTextTransformPayloadError,
    ProjectDirNotExistException,
    ProjectLoadFailureException,
    ProjImgTrans,
    TextBlkEncoder,
    UnsupportedTextTransformVersionError,
    migrate_text_transform_payload,
)
from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.textblock import TextBlock


def project_payload(block, root_version=None):
    payload = {
        'directory': 'fixture-project',
        'pages': {'page.png': [block]},
    }
    if root_version is not None:
        payload['text_transform_schema_version'] = root_version
    return payload


def canonical_transform(block):
    fontformat = block['fontformat']
    return (
        fontformat['horizontal_scale'],
        fontformat['vertical_scale'],
        fontformat['slant_angle'],
    )


def resolved_font(char_format, default_font):
    return char_format.font().resolve(default_font)


def make_effective_v0_html():
    """Build the observable v0 signature directly with public Qt formats."""
    effective_default = QFont('Arial')
    effective_default.setPointSizeF(36.0)
    effective_default.setStretch(150)

    document = QTextDocument()
    document.setDefaultFont(effective_default)
    cursor = QTextCursor(document)

    rich = QTextCharFormat()
    rich.setFontPointSize(28.0)
    rich.setFontStretch(150)
    rich.setFontItalic(True)
    rich.setFontUnderline(True)
    bold = QFont()
    bold.setBold(True)
    rich.setFontWeight(bold.weight())
    rich.setForeground(QColor(10, 20, 30))
    cursor.insertText('Rich', rich)

    empty_block = QTextBlockFormat()
    empty_block.setLeftMargin(4.0)
    formatted_empty = QTextCharFormat()
    formatted_empty.setFontPointSize(42.0)
    formatted_empty.setFontStretch(150)
    formatted_empty.setFontWeight(bold.weight())
    formatted_empty.setFontUnderline(True)
    cursor.insertBlock(empty_block, formatted_empty)

    html = document.toHtml()
    # The failed v0 writer recorded native stretch ratios because Qt's HTML
    # serializer omits QFont.stretch(). The empty paragraph has a zero-length
    # run at its QTextBlock position.
    stretch_runs = [
        {'start': 0, 'length': 4, 'stretch': 150, 'ratio': [1, 1]},
        {
            'start': 5,
            'length': 0,
            'stretch': 150,
            'ratio': [1, 1],
            'empty_block': True,
        },
    ]
    metadata = json.dumps(stretch_runs, separators=(',', ':'))
    return html.replace(
        '</body>',
        f'<!--ballontranslator-logical-stretch-v1:{metadata}--></body>',
    )


def make_unstretched_qt_html():
    logical_default = QFont('Arial')
    logical_default.setPointSizeF(18.0)
    logical_default.setStretch(100)
    document = QTextDocument()
    document.setDefaultFont(logical_default)
    QTextCursor(document).insertText('No persisted stretch evidence')
    return document.toHtml()


def replace_stretch_metadata(html, runs):
    marker = '<!--ballontranslator-logical-stretch-v1:'
    start = html.index(marker)
    end = html.index('-->', start) + 3
    metadata = json.dumps(runs, separators=(',', ':'))
    return html[:start] + marker + metadata + '-->' + html[end:]


class TextTransformSerializationTest(unittest.TestCase):

    def test_named_migration_fixture_matrix(self):
        upstream, upstream_warnings = migrate_text_transform_payload(
            load_fixture('upstream_legacy.json')
        )
        self.assertEqual(upstream_warnings, [])
        self.assertEqual(
            canonical_transform(upstream['pages']['page.png'][0]),
            (1.0, 1.0, 0.0),
        )

        logical, logical_warnings = migrate_text_transform_payload(
            load_fixture('pr1238_logical_v1.json')
        )
        self.assertEqual(logical_warnings, [])
        self.assertEqual(
            canonical_transform(logical['pages']['page.png'][0]),
            (1.25, 0.8, 12.5),
        )

        for exact_name in (
            'pr1238_effective_v0_exact.json',
            'formatted_empty_v0.json',
        ):
            with self.subTest(exact_name=exact_name):
                exact, exact_warnings = migrate_text_transform_payload(
                    load_fixture(exact_name)
                )
                self.assertTrue(exact_warnings)
                self.assertEqual(exact['text_transform_schema_version'], 1)

        with self.assertRaises(AmbiguousLegacyTextTransformError):
            migrate_text_transform_payload(
                load_fixture('pr1238_effective_v0_ambiguous_stretch.json')
            )
        for future_name in ('future_root_v2.json', 'future_block_marker_v2.json'):
            with self.subTest(future_name=future_name):
                with self.assertRaises(UnsupportedTextTransformVersionError):
                    migrate_text_transform_payload(load_fixture(future_name))

    def test_textblock_deepcopy_keeps_one_independent_canonical_owner(self):
        original = TextBlock(
            xyxy=[1, 2, 3, 4],
            translation='copy fixture',
            fontformat=FontFormat(
                horizontal_scale=1.25,
                vertical_scale=0.75,
                slant_angle=9.0,
            ),
        )
        duplicate = copy.deepcopy(original)

        self.assertIsNot(duplicate, original)
        self.assertIsNot(duplicate.fontformat, original.fontformat)
        self.assertEqual(duplicate.fontformat.text_transform, (1.25, 0.75, 9.0))
        duplicate.fontformat.horizontal_scale = 2.0
        self.assertEqual(original.fontformat.horizontal_scale, 1.25)


    def test_old_upstream_payload_gets_neutral_defaults_without_html_changes(self):
        html = '<p><span style="font-weight:600">unchanged</span></p>'
        source = project_payload(
            {
                'text': 'unchanged',
                'rich_text': html,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'italic': True,
                },
            }
        )
        before = copy.deepcopy(source)

        migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(source, before)
        self.assertEqual(migration_warnings, [])
        self.assertEqual(migrated['text_transform_schema_version'], 1)
        self.assertEqual(canonical_transform(block), (1.0, 1.0, 0.0))
        self.assertEqual(block['rich_text'], html)
        self.assertEqual(block['fontformat']['font_family'], 'Arial')
        self.assertTrue(block['fontformat']['italic'])

    def test_canonical_v1_is_normalized_without_rewriting_html(self):
        html = '<p>canonical rich text</p>'
        source = project_payload(
            {
                'rich_text': html,
                'fontformat': {
                    'horizontal_scale': 1.23456789,
                    'vertical_scale': 0.75,
                    'slant_angle': -0.0,
                },
            },
            root_version=1,
        )
        before = copy.deepcopy(source)

        migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(source, before)
        self.assertEqual(migration_warnings, [])
        self.assertEqual(canonical_transform(block), (1.234568, 0.75, 0.0))
        self.assertEqual(block['rich_text'], html)

    def test_failed_logical_marker_one_canonicalizes_aliases_only(self):
        html = '<p><i>already logical</i></p>'
        source = project_payload(
            {
                'rich_text': html,
                'rich_text_transform_version': 1,
                'fontformat': {
                    'horizontal_scale': 1.25,
                    'vertical_scale': 0.8,
                    'italic_angle': 12.5,
                    'underline': True,
                },
            }
        )

        migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(migration_warnings, [])
        self.assertEqual(canonical_transform(block), (1.25, 0.8, 12.5))
        self.assertEqual(block['rich_text'], html)
        self.assertTrue(block['fontformat']['underline'])
        self.assertNotIn('italic_angle', block['fontformat'])
        self.assertNotIn('rich_text_transform_version', block)

    def test_block_level_aliases_move_to_canonical_fontformat(self):
        source = project_payload(
            {
                'horizontal_scale': 1.6,
                'vertical_scale': 0.7,
                'italic_angle': -8.0,
                'rich_text_transform_version': 1,
                'translation': 'preserved',
                'fontformat': {'bold': True},
            }
        )

        migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(migration_warnings, [])
        self.assertEqual(canonical_transform(block), (1.6, 0.7, -8.0))
        self.assertEqual(block['translation'], 'preserved')
        self.assertTrue(block['fontformat']['bold'])
        self.assertNotIn('horizontal_scale', block)
        self.assertNotIn('vertical_scale', block)
        self.assertNotIn('italic_angle', block)

    def test_conflicting_known_aliases_are_invalid(self):
        source = project_payload(
            {
                'horizontal_scale': 1.5,
                'rich_text_transform_version': 1,
                'fontformat': {'horizontal_scale': 1.25},
            }
        )

        with self.assertRaisesRegex(
            InvalidTextTransformPayloadError, 'conflicting aliases'
        ):
            migrate_text_transform_payload(source)

    def test_nonnumeric_and_nonfinite_transform_values_are_rejected(self):
        invalid_values = ('1.2', None, True, math.nan, math.inf, -math.inf)
        for field in ('horizontal_scale', 'vertical_scale', 'slant_angle'):
            for invalid in invalid_values:
                fontformat = {
                    'horizontal_scale': 1.0,
                    'vertical_scale': 1.0,
                    'slant_angle': 0.0,
                }
                fontformat[field] = invalid
                source = project_payload(
                    {'rich_text': '', 'fontformat': fontformat}, root_version=1
                )
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaises(InvalidTextTransformPayloadError):
                        migrate_text_transform_payload(source)

    def test_nonnumeric_and_nonfinite_versions_are_rejected(self):
        for invalid in ('1', None, True, math.nan, math.inf, -1):
            source = project_payload({'rich_text': ''}, root_version=invalid)
            if invalid is None:
                source['text_transform_schema_version'] = None
            with self.subTest(root_version=invalid):
                    with self.assertRaises(InvalidTextTransformPayloadError):
                        migrate_text_transform_payload(source)

    def test_non_neutral_empty_v0_requires_exact_effective_signature(self):
        cases = (
            (None, False),
            (None, True),
            (1, True),
        )
        for root_version, explicit_marker in cases:
            with self.subTest(
                root_version=root_version,
                explicit_marker=explicit_marker,
            ):
                block = {
                    'rich_text': '',
                    'fontformat': {
                        'font_family': 'Arial',
                        'font_size': 36,
                        'horizontal_scale': 2.0,
                        'vertical_scale': 1.5,
                        'slant_angle': 0.0,
                    },
                }
                if explicit_marker:
                    block['rich_text_transform_version'] = 0
                source = project_payload(block, root_version=root_version)
                before = copy.deepcopy(source)

                with self.assertRaisesRegex(
                    AmbiguousLegacyTextTransformError,
                    'rich_text is empty',
                ):
                    migrate_text_transform_payload(source)

                self.assertEqual(source, before)

        for invalid in ('1', True, math.nan, math.inf, -1):
            source = project_payload(
                {'rich_text': '', 'rich_text_transform_version': invalid}
            )
            with self.subTest(block_version=invalid):
                with self.assertRaises(InvalidTextTransformPayloadError):
                    migrate_text_transform_payload(source)

    def test_finite_out_of_range_values_clamp_with_migration_warnings(self):
        source = project_payload(
            {
                'rich_text_transform_version': 1,
                'fontformat': {
                    'horizontal_scale': 9.0,
                    'vertical_scale': -2.0,
                    'slant_angle': 90.0,
                },
            }
        )

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(canonical_transform(block), (4.0, 0.1, 45.0))
        self.assertEqual(len(migration_warnings), 3)
        self.assertEqual(emitted, [])
        self.assertTrue(any('horizontal_scale' in item for item in migration_warnings))
        self.assertTrue(any('vertical_scale' in item for item in migration_warnings))
        self.assertTrue(any('slant_angle' in item for item in migration_warnings))

    def test_future_root_version_rejects_without_input_mutation(self):
        source = project_payload(
            {
                'horizontal_scale': 1.5,
                'fontformat': {'bold': True},
            },
            root_version=2,
        )
        before = copy.deepcopy(source)

        with warnings.catch_warnings(record=True) as emitted:
            with self.assertRaises(UnsupportedTextTransformVersionError):
                migrate_text_transform_payload(source)

        self.assertEqual(source, before)
        self.assertEqual(emitted, [])

    def test_future_rejection_does_not_mutate_existing_project_state(self):
        project = ProjImgTrans()
        old_pages = {'existing.png': [TextBlock(translation='keep me')]}
        old_info = {'existing.png': {'finish_code': 7}}
        project.pages = old_pages
        project._image_info = old_info
        project.current_img = 'existing.png'
        project.img_array = object()
        old_image = project.img_array

        source = project_payload({'rich_text': ''}, root_version=2)
        with self.assertRaises(UnsupportedTextTransformVersionError):
            project.load_from_dict(source)

        self.assertIs(project.pages, old_pages)
        self.assertIs(project._image_info, old_info)
        self.assertEqual(project.current_img, 'existing.png')
        self.assertIs(project.img_array, old_image)

    def test_json_boundary_preserves_state_and_original_future_version_cause(self):
        project = ProjImgTrans()
        project.directory = 'original-directory'
        project.proj_path = 'original-project.json'
        old_pages = {'existing.png': [TextBlock(translation='keep me')]}
        old_info = {'existing.png': {'finish_code': 7}}
        project.pages = old_pages
        project._image_info = old_info
        project.current_img = 'existing.png'
        future_path = os.path.join(FIXTURE_DIR, 'future_root_v2.json')

        with self.assertRaises(ProjectLoadFailureException) as raised:
            project.load_from_json(future_path)

        self.assertIsInstance(
            raised.exception.__cause__, UnsupportedTextTransformVersionError
        )
        self.assertIn('unsupported text transform schema version 2', str(raised.exception))
        self.assertEqual(project.directory, 'original-directory')
        self.assertEqual(project.proj_path, 'original-project.json')
        self.assertIs(project.pages, old_pages)
        self.assertIs(project._image_info, old_info)
        self.assertEqual(project.current_img, 'existing.png')

    def test_missing_new_project_directory_does_not_mutate_existing_state(self):
        project = ProjImgTrans()
        project.directory = 'original-directory'
        project.proj_path = 'original-project.json'
        old_pages = {'existing.png': [TextBlock(translation='keep me')]}
        old_info = {'existing.png': {'finish_code': 7}}
        old_warnings = ['keep prior warning until a load commits']
        project.pages = old_pages
        project._image_info = old_info
        project.current_img = 'existing.png'
        project.text_transform_migration_warnings = old_warnings
        missing_directory = os.path.join(FIXTURE_DIR, 'does-not-exist')

        with self.assertRaises(ProjectDirNotExistException):
            project.load(missing_directory)

        self.assertEqual(project.directory, 'original-directory')
        self.assertEqual(project.proj_path, 'original-project.json')
        self.assertIs(project.pages, old_pages)
        self.assertIs(project._image_info, old_info)
        self.assertEqual(project.current_img, 'existing.png')
        self.assertIs(project.text_transform_migration_warnings, old_warnings)

    def test_late_validation_failure_does_not_mutate_existing_project_state(self):
        project = ProjImgTrans()
        project.directory = EMPTY_PROJECT_DIR
        old_pages = {'existing.png': [TextBlock(translation='keep me')]}
        old_info = {'existing.png': {'finish_code': 7}}
        project.pages = old_pages
        project._image_info = old_info
        project.current_img = 'existing.png'

        source = {
            'text_transform_schema_version': 1,
            'pages': {'missing.png': [{'translation': 'candidate'}]},
            'image_info': [],
        }
        with self.assertRaisesRegex(Exception, 'image_info must be an object'):
            project.load_from_dict(source)

        self.assertIs(project.pages, old_pages)
        self.assertIs(project._image_info, old_info)
        self.assertEqual(project.current_img, 'existing.png')

    def test_future_block_marker_preflights_all_blocks_without_mutation(self):
        source = {
            'pages': {
                'page.png': [
                    {
                        'rich_text_transform_version': 1,
                        'fontformat': {'horizontal_scale': 9.0},
                    },
                    {
                        'rich_text_transform_version': 2,
                        'fontformat': {'vertical_scale': 0.5},
                    },
                ]
            }
        }
        before = copy.deepcopy(source)

        with warnings.catch_warnings(record=True) as emitted:
            with self.assertRaises(UnsupportedTextTransformVersionError):
                migrate_text_transform_payload(source)

        self.assertEqual(source, before)
        self.assertEqual(emitted, [])

    def test_canonical_output_removes_all_aliases_and_legacy_markers(self):
        source = project_payload(
            {
                'horizontal_scale': 1.4,
                'vertical_scale': 0.6,
                'italic_angle': 11.0,
                'rich_text_transform_version': 1,
                'fontformat': {
                    'horizontal_scale': 1.4,
                    'vertical_scale': 0.6,
                    'italic_angle': 11.0,
                },
            }
        )

        migrated, _ = migrate_text_transform_payload(source)
        block = migrated['pages']['page.png'][0]
        serialized = json.dumps(migrated, sort_keys=True)

        self.assertEqual(canonical_transform(block), (1.4, 0.6, 11.0))
        self.assertNotIn('horizontal_scale', block)
        self.assertNotIn('vertical_scale', block)
        self.assertNotIn('italic_angle', block)
        self.assertNotIn('italic_angle', block['fontformat'])
        self.assertNotIn('"rich_text_transform_version"', serialized)
        self.assertNotIn('"italic_angle"', serialized)

    def test_exact_effective_v0_qt_html_restores_formatted_empty_paragraph(self):
        effective_html = make_effective_v0_html()
        self.assertIn('font-size:36pt', effective_html)
        self.assertIn('-qt-paragraph-type:empty', effective_html)
        self.assertIn('font-size:42pt', effective_html)
        self.assertIn('ballontranslator-logical-stretch-v1:', effective_html)
        source = project_payload(
            {
                'rich_text': effective_html,
                'rich_text_transform_version': 0,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'horizontal_scale': 3.0,
                    'vertical_scale': 2.0,
                    'slant_angle': 7.0,
                },
            }
        )

        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            migrated, migration_warnings = migrate_text_transform_payload(source)

        block = migrated['pages']['page.png'][0]
        self.assertEqual(canonical_transform(block), (3.0, 2.0, 7.0))
        self.assertEqual(len(migration_warnings), 1)
        self.assertIn('rich_text was restored', migration_warnings[0])
        self.assertEqual(emitted, [])

        logical_default = QFont('Arial')
        logical_default.setPointSizeF(18.0)
        logical_default.setStretch(100)
        logical_document = QTextDocument()
        logical_document.setDefaultFont(logical_default)
        logical_document.setHtml(block['rich_text'])

        self.assertEqual(logical_document.toPlainText(), 'Rich\n')
        first_block = logical_document.firstBlock()
        rich_fragment = first_block.begin().fragment()
        rich_font = resolved_font(rich_fragment.charFormat(), logical_default)
        self.assertAlmostEqual(rich_font.pointSizeF(), 14.0)
        self.assertEqual(rich_font.stretch(), 100)
        self.assertTrue(rich_font.italic())
        self.assertTrue(rich_font.bold())
        self.assertTrue(rich_font.underline())
        self.assertEqual(
            rich_fragment.charFormat().foreground().color(), QColor(10, 20, 30)
        )

        empty_block = first_block.next()
        self.assertTrue(empty_block.isValid())
        self.assertEqual(empty_block.text(), '')
        self.assertAlmostEqual(empty_block.blockFormat().leftMargin(), 4.0)
        empty_font = resolved_font(empty_block.charFormat(), logical_default)
        self.assertAlmostEqual(empty_font.pointSizeF(), 21.0)
        self.assertEqual(empty_font.stretch(), 100)
        self.assertTrue(empty_font.bold())
        self.assertTrue(empty_font.underline())
        self.assertFalse(empty_block.next().isValid())

    def test_ambiguous_horizontal_only_v0_stretch_is_rejected(self):
        html = make_unstretched_qt_html()
        self.assertNotIn('font-stretch', html)
        source = project_payload(
            {
                'rich_text': html,
                'rich_text_transform_version': 0,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'horizontal_scale': 1.5,
                    'vertical_scale': 1.0,
                    'slant_angle': 0.0,
                },
            }
        )
        before = copy.deepcopy(source)

        with self.assertRaises(AmbiguousLegacyTextTransformError):
            migrate_text_transform_payload(source)

        self.assertEqual(source, before)

    def test_stale_overlapping_and_invalid_empty_v0_metadata_are_rejected(self):
        invalid_run_sets = (
            [
                {
                    'start': 999,
                    'length': 1,
                    'stretch': 150,
                    'ratio': [1, 1],
                }
            ],
            [
                {
                    'start': 0,
                    'length': 4,
                    'stretch': 150,
                    'ratio': [1, 1],
                },
                {
                    'start': 2,
                    'length': 2,
                    'stretch': 150,
                    'ratio': [1, 1],
                },
            ],
            [
                {
                    'start': 0,
                    'length': 0,
                    'stretch': 150,
                    'ratio': [1, 1],
                }
            ],
            [
                {
                    'start': 0,
                    'length': 1,
                    'stretch': 150,
                    'ratio': [1, 1],
                }
            ],
            [
                {
                    'start': 1,
                    'length': 2,
                    'stretch': 150,
                    'ratio': [1, 1],
                }
            ],
            [
                {
                    'start': 0,
                    'length': 4,
                    'stretch': 150.0,
                    'ratio': [1, 1],
                }
            ],
        )
        for runs in invalid_run_sets:
            source = project_payload(
                {
                    'rich_text': replace_stretch_metadata(
                        make_effective_v0_html(), runs
                    ),
                    'rich_text_transform_version': 0,
                    'fontformat': {
                        'font_family': 'Arial',
                        'font_size': 24,
                        'horizontal_scale': 3.0,
                        'vertical_scale': 2.0,
                    },
                }
            )
            with self.subTest(runs=runs):
                with self.assertRaises(AmbiguousLegacyTextTransformError):
                    migrate_text_transform_payload(source)

    def test_duplicate_metadata_and_noninteger_ratios_are_rejected(self):
        exact_html = make_effective_v0_html()
        marker = '<!--ballontranslator-logical-stretch-v1:'
        duplicate_html = exact_html.replace(
            '</body>',
            f'{marker}[]--></body>',
        )
        invalid_ratios = ([True, True], [1.0, 1.0])
        candidates = [duplicate_html]
        for ratio in invalid_ratios:
            runs = [
                {'start': 0, 'length': 4, 'stretch': 150, 'ratio': ratio},
                {
                    'start': 5,
                    'length': 0,
                    'stretch': 150,
                    'ratio': [1, 1],
                    'empty_block': True,
                },
            ]
            candidates.append(replace_stretch_metadata(exact_html, runs))

        for html in candidates:
            source = project_payload(
                {
                    'rich_text': html,
                    'rich_text_transform_version': 0,
                    'fontformat': {
                        'font_family': 'Arial',
                        'font_size': 24,
                        'horizontal_scale': 3.0,
                        'vertical_scale': 2.0,
                    },
                }
            )
            with self.subTest(html=html[-160:]):
                with self.assertRaises(AmbiguousLegacyTextTransformError):
                    migrate_text_transform_payload(source)

    def test_explicit_v0_marker_is_recovered_even_under_v1_root(self):
        source = project_payload(
            {
                'rich_text': make_effective_v0_html(),
                'rich_text_transform_version': 0,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'horizontal_scale': 3.0,
                    'vertical_scale': 2.0,
                },
            },
            root_version=1,
        )

        migrated, migration_warnings = migrate_text_transform_payload(source)

        html = migrated['pages']['page.png'][0]['rich_text']
        self.assertNotIn('font-size:36pt', html)
        self.assertNotIn('ballontranslator-logical-stretch-v1:', html)
        self.assertTrue(any('rich_text was restored' in item for item in migration_warnings))

    def test_explicit_v0_marker_is_resolved_per_block_in_either_order(self):
        def effective_block():
            return {
                'rich_text': make_effective_v0_html(),
                'rich_text_transform_version': 0,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'horizontal_scale': 3.0,
                    'vertical_scale': 2.0,
                },
            }

        def canonical_block():
            return {
                'rich_text': '<p>already canonical</p>',
                'fontformat': {
                    'horizontal_scale': 1.25,
                    'vertical_scale': 0.8,
                    'slant_angle': 4.0,
                },
            }

        for blocks in (
            [effective_block(), canonical_block()],
            [canonical_block(), effective_block()],
        ):
            source = {
                'text_transform_schema_version': 1,
                'pages': {'page.png': blocks},
            }
            migrated, migration_warnings = migrate_text_transform_payload(source)
            migrated_blocks = migrated['pages']['page.png']

            effective_index = 0 if 'Rich' in blocks[0]['rich_text'] else 1
            canonical_index = 1 - effective_index
            self.assertNotIn(
                'ballontranslator-logical-stretch-v1:',
                migrated_blocks[effective_index]['rich_text'],
            )
            self.assertEqual(
                migrated_blocks[canonical_index]['rich_text'],
                '<p>already canonical</p>',
            )
            self.assertEqual(len(migration_warnings), 1)

    def test_canonicalized_effective_migration_is_idempotent(self):
        source = project_payload(
            {
                'rich_text': make_effective_v0_html(),
                'rich_text_transform_version': 0,
                'fontformat': {
                    'font_family': 'Arial',
                    'font_size': 24,
                    'horizontal_scale': 3.0,
                    'vertical_scale': 2.0,
                    'italic_angle': 7.0,
                },
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            first, first_warnings = migrate_text_transform_payload(source)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always')
            second, second_warnings = migrate_text_transform_payload(first)

        self.assertTrue(first_warnings)
        self.assertEqual(second_warnings, [])
        self.assertEqual(emitted, [])
        self.assertEqual(second, first)

    def test_project_encoder_load_and_repeated_save_shape_are_canonical(self):
        block = TextBlock(
            xyxy=[3, 4, 103, 54],
            _bounding_rect=[3, 4, 100, 50],
            translation='round trip',
            rich_text='<p><b>logical rich text</b></p>',
            fontformat=FontFormat(
                horizontal_scale=1.234568,
                vertical_scale=0.625,
                slant_angle=-14.5,
            ),
        )
        source_project = ProjImgTrans()
        source_project.directory = EMPTY_PROJECT_DIR
        source_project.pages = {'missing.png': [block]}
        source_project.not_found_pages = {}
        source_project._image_info = {'missing.png': {'finish_code': 0}}
        source_project.current_img = None

        encoded = json.loads(
            json.dumps(source_project.to_dict(), cls=TextBlkEncoder)
        )
        restored = ProjImgTrans()
        restored.directory = EMPTY_PROJECT_DIR
        restored.proj_path = os.path.join(EMPTY_PROJECT_DIR, 'fixture.json')
        restored.load_from_dict(encoded)

        restored_block = restored.not_found_pages['missing.png'][0]
        self.assertEqual(
            restored_block.fontformat.text_transform,
            (1.234568, 0.625, -14.5),
        )
        self.assertEqual(restored_block.rich_text, block.rich_text)

        saved_again = json.loads(
            json.dumps(restored.to_dict(), cls=TextBlkEncoder)
        )
        remigrated, remigration_warnings = migrate_text_transform_payload(
            saved_again
        )
        self.assertEqual(remigration_warnings, [])
        self.assertEqual(remigrated, saved_again)
        serialized = json.dumps(saved_again, sort_keys=True)
        self.assertNotIn('italic_angle', serialized)
        self.assertNotIn('rich_text_transform_version', serialized)


if __name__ == '__main__':
    unittest.main()
