import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QAbstractTextDocumentLayout,
    QBrush,
    QColor,
    QFont,
    QFontInfo,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
    QTextFormat,
    QTransform,
)
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QGraphicsScene,
    QHBoxLayout,
    QWidget,
)
try:
    from qtpy.QtGui import QUndoStack
except ImportError:
    from qtpy.QtWidgets import QUndoStack

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.canvas import Canvas
from ballontranslator.ui.misc import doc_replace, pixmap2ndarray
from ballontranslator.ui.text_engine.annotations import (
    AnnotationProperty,
    FONT_FEATURES_AVAILABLE,
    FONT_VARIANT_LIGATURES_NONE,
    FONT_VARIANT_LIGATURES_NORMAL,
    LETTER_SPACING_ATTRIBUTE,
    LINE_DISTANCE_ATTRIBUTE,
    LIGATURE_COMMON,
    LIGATURE_CONTEXTUAL,
    LIGATURE_DEFAULT,
    LIGATURE_DISABLED,
    LIGATURE_DISCRETIONARY,
    LIGATURE_ENABLED,
    LIGATURE_HISTORICAL,
    OLDSTYLE_NUMS,
    TEXT_COMBINE_ID_ATTRIBUTE,
    apply_auto_text_combine_upright,
    apply_emphasis,
    apply_ligature_axis,
    apply_oldstyle_nums,
    apply_letter_spacing,
    apply_line_spacing,
    apply_text_combine_upright,
    apply_ruby,
    canonical_font_variant_ligatures,
    create_rich_text_mime,
    emphasis_values,
    font_variant_ligatures_value,
    insert_rich_text_mime,
    letter_spacing_value,
    ligature_axis_value,
    line_spacing_values,
    load_rich_text_html,
    oldstyle_nums_value,
    ruby_containers,
    text_combine_upright_ranges,
    text_combine_upright_values,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.formatting.panel import (
    EmphasisToolButton,
    FontFormatPanel,
)
from ballontranslator.ui.text_engine.editing.manager import SceneTextManager
from ballontranslator.ui.text_engine.editing.commands import TextItemEditCommand
from ballontranslator.ui.text_engine.editing.commands import propagate_user_edit
from ballontranslator.ui.text_engine import effect_renderer as effect_rendering
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.rendering import emphasis as emphasis_rendering
from ballontranslator.ui.text_engine.rendering import glyph as glyph_rendering
from ballontranslator.ui.text_engine.rendering.emphasis import (
    draw_emphasis_marks,
    emphasis_ink_bounds,
)
from ballontranslator.ui.text_engine.rendering.indexing import _grapheme_ranges
from ballontranslator.ui.text_engine.rendering.native_document import (
    NATIVE_DOCUMENT_CACHE,
    NATIVE_DOCUMENT_CACHE_MAX_ENTRIES,
)
from ballontranslator.ui.text_engine.rendering.tate_chu_yoko import (
    tate_chu_yoko_ink_bounds,
    tate_chu_yoko_natural_bounds,
)
from ballontranslator.utils import config as C, shared
from ballontranslator.utils.fontformat import (
    FontFormat,
    LineSpacingType,
    TextAlignment,
    TextTransformStack,
    pt2px,
)
from ballontranslator.utils.textblock import TextBlock


def _format_at(document: QTextDocument, start: int, length: int = 1):
    cursor = QTextCursor(document)
    cursor.setPosition(start)
    cursor.setPosition(start + length, QTextCursor.MoveMode.KeepAnchor)
    return cursor.charFormat()


def _glyph_indexes(document: QTextDocument) -> tuple[int, ...]:
    document.setTextWidth(1000)
    document.documentLayout().documentSize()
    layout = document.firstBlock().layout()
    return tuple(
        glyph
        for line_index in range(layout.lineCount())
        for run in layout.lineAt(line_index).glyphRuns()
        for glyph in run.glyphIndexes()
    )


def _glyph_count(document: QTextDocument) -> int:
    return len(_glyph_indexes(document))


def _line_spacing_at(
    document: QTextDocument,
    block_number: int,
    fallback: float = 1.2,
    fallback_type: int = LineSpacingType.Proportional,
) -> tuple[float, LineSpacingType]:
    return line_spacing_values(
        document.findBlockByNumber(block_number).blockFormat(),
        fallback,
        fallback_type,
    )


class RichTextAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _make_item(
        vertical: bool = False,
        *,
        bounds=(0, 0, 600, 300),
        text: str = '強調 test',
    ) -> TextBlkItem:
        block = TextBlock(list(bounds))
        block._bounding_rect = list(bounds)
        block.vertical = vertical
        block.translation = text
        return TextBlkItem(block, 0)

    def test_old_qt_html_loads_without_extensions_or_format_loss(self):
        source = QTextDocument()
        source.setPlainText('old rich text')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        cursor.mergeCharFormat(char_format)
        old_html = source.toHtml()

        restored = QTextDocument()
        load_rich_text_html(restored, old_html)

        self.assertEqual(restored.toPlainText(), 'old rich text')
        self.assertTrue(_format_at(restored, 0, 3).font().bold())
        self.assertNotIn(
            'text-emphasis-style',
            to_rich_text_html(restored),
        )

    def test_old_qt_html_skips_extension_parser_and_keeps_spacing_fallback(self):
        source = QTextDocument()
        source.setPlainText('old rich text')
        restored = QTextDocument()

        with patch(
            'ballontranslator.ui.text_engine.annotations.'
            '_rich_text_extensions_from_html'
        ) as parse_extensions:
            load_rich_text_html(
                restored,
                source.toHtml(),
                letter_spacing_fallback=1.25,
            )

        parse_extensions.assert_not_called()
        self.assertEqual(restored.toPlainText(), 'old rich text')
        self.assertEqual(letter_spacing_value(_format_at(restored, 0)), 1.25)

    def test_emphasis_inline_round_trip_keeps_fragment_style(self):
        source = QTextDocument()
        source.setPlainText('A𠮷B')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#c02040'))
        cursor.mergeCharFormat(char_format)
        apply_emphasis(cursor, 'filled sesame', 'under left')

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        restored_format = _format_at(restored, 1, 2)
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertTrue(html.startswith('<!DOCTYPE html>'))
        self.assertIn('text-emphasis-style: filled sesame', html)
        self.assertIn('text-emphasis-position: under left', html)
        self.assertNotIn('ballontranslator-rich-text', html)
        self.assertEqual(restored.toPlainText(), 'A𠮷B')
        self.assertEqual(legacy_reader.toPlainText(), 'A𠮷B')
        self.assertTrue(_format_at(legacy_reader, 1, 2).font().bold())
        self.assertEqual(
            emphasis_values(_format_at(legacy_reader, 1, 2))[0],
            'none',
        )
        self.assertEqual(
            emphasis_values(restored_format),
            ('filled sesame', 'under left'),
        )
        self.assertTrue(restored_format.font().bold())
        self.assertTrue(restored_format.font().italic())
        self.assertEqual(restored_format.foreground().color(), QColor('#c02040'))

    def test_invalid_inline_extension_drops_only_the_annotation(self):
        source = QTextDocument()
        source.setPlainText('safe text')
        html = source.toHtml().replace(
            'safe text',
            '<span style="text-emphasis-style: sparks; '
            'text-emphasis-position: over right;">safe</span> text',
        )
        restored = QTextDocument()

        load_rich_text_html(restored, html)

        self.assertEqual(restored.toPlainText(), 'safe text')
        self.assertEqual(
            emphasis_values(_format_at(restored, 0)),
            ('none', 'over right'),
        )

    def test_invalid_letter_spacing_keeps_legacy_fallback(self):
        source = QTextDocument()
        source.setPlainText('safe text')
        html = source.toHtml().replace(
            'safe text',
            f'<span {LETTER_SPACING_ATTRIBUTE}="wide">safe</span> text',
        )
        restored = QTextDocument()

        load_rich_text_html(
            restored,
            html,
            letter_spacing_fallback=1.25,
        )

        self.assertEqual(restored.toPlainText(), 'safe text')
        self.assertEqual(
            letter_spacing_value(_format_at(restored, 0)),
            1.25,
        )

    def test_selection_and_insertion_format_are_independent(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'filled dot', 'over right')

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_emphasis(cursor, 'open circle', 'under left')
        cursor.insertText('D')

        self.assertEqual(
            emphasis_values(_format_at(document, 0))[0], 'none'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 1))[0], 'filled dot'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 2))[0], 'none'
        )
        self.assertEqual(
            emphasis_values(_format_at(document, 3)),
            ('open circle', 'under left'),
        )

    def test_letter_spacing_selection_and_insertion_are_independent(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_letter_spacing(cursor, 1.4, vertical=False)

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_letter_spacing(cursor, 0.8, vertical=False)
        cursor.insertText('D')

        self.assertEqual(letter_spacing_value(_format_at(document, 0)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(document, 1)), 1.4)
        self.assertEqual(letter_spacing_value(_format_at(document, 2)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(document, 3)), 0.8)

    def test_font_variant_css_round_trip_and_normal_reset(self):
        source = QTextDocument('fiX')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_ligature_axis(
            cursor, LIGATURE_COMMON, LIGATURE_ENABLED, vertical=False
        )
        apply_ligature_axis(
            cursor,
            LIGATURE_DISCRETIONARY,
            LIGATURE_ENABLED,
            vertical=False,
        )
        apply_ligature_axis(
            cursor,
            LIGATURE_CONTEXTUAL,
            LIGATURE_DISABLED,
            vertical=False,
        )
        apply_oldstyle_nums(cursor, LIGATURE_ENABLED)

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)

        self.assertIn(
            'font-variant-ligatures: common-ligatures '
            'discretionary-ligatures no-contextual',
            html,
        )
        self.assertIn('font-variant-numeric: oldstyle-nums', html)
        restored_format = _format_at(restored, 0, 2)
        self.assertEqual(
            ligature_axis_value(restored_format, LIGATURE_COMMON),
            LIGATURE_ENABLED,
        )
        self.assertEqual(
            ligature_axis_value(restored_format, LIGATURE_DISCRETIONARY),
            LIGATURE_ENABLED,
        )
        self.assertEqual(
            ligature_axis_value(restored_format, LIGATURE_CONTEXTUAL),
            LIGATURE_DISABLED,
        )
        self.assertEqual(
            oldstyle_nums_value(restored_format),
            LIGATURE_ENABLED,
        )
        if FONT_FEATURES_AVAILABLE:
            font = restored_format.font()
            for name, value in (('onum', 1), ('lnum', 0)):
                tag = QFont.Tag.fromString(name)
                self.assertTrue(font.isFeatureSet(tag))
                self.assertEqual(font.featureValue(tag), value)
        self.assertEqual(
            font_variant_ligatures_value(_format_at(restored, 2)),
            FONT_VARIANT_LIGATURES_NORMAL,
        )

        apply_oldstyle_nums(cursor, LIGATURE_DISABLED)
        self.assertIn(
            'font-variant-numeric: lining-nums',
            to_rich_text_html(source),
        )
        if FONT_FEATURES_AVAILABLE:
            font = _format_at(source, 0, 2).font()
            for name, value in (('onum', 0), ('lnum', 1)):
                tag = QFont.Tag.fromString(name)
                self.assertTrue(font.isFeatureSet(tag))
                self.assertEqual(font.featureValue(tag), value)

        for axis in (
            LIGATURE_COMMON,
            LIGATURE_DISCRETIONARY,
            LIGATURE_CONTEXTUAL,
        ):
            apply_ligature_axis(
                cursor, axis, LIGATURE_DEFAULT, vertical=False
            )
        apply_oldstyle_nums(cursor, LIGATURE_DEFAULT)
        self.assertNotIn(
            'font-variant-ligatures', to_rich_text_html(source)
        )
        self.assertNotIn(
            'font-variant-numeric', to_rich_text_html(source)
        )

    def test_ligature_css_none_and_qt5_safe_values_round_trip(self):
        self.assertEqual(
            canonical_font_variant_ligatures(
                'contextual discretionary-ligatures'
            ),
            'discretionary-ligatures contextual',
        )
        self.assertIsNone(canonical_font_variant_ligatures(
            'contextual no-contextual'
        ))
        document = QTextDocument()
        load_rich_text_html(
            document,
            '<p><span style="font-variant-ligatures: none;">fi</span>'
            '<span style="font-variant-ligatures: '
            'discretionary-ligatures no-contextual;">st</span></p>',
        )

        none_format = _format_at(document, 0, 2)
        self.assertEqual(
            font_variant_ligatures_value(none_format),
            FONT_VARIANT_LIGATURES_NONE,
        )
        for axis in (
            LIGATURE_COMMON,
            LIGATURE_DISCRETIONARY,
            LIGATURE_HISTORICAL,
            LIGATURE_CONTEXTUAL,
        ):
            self.assertEqual(
                ligature_axis_value(none_format, axis),
                LIGATURE_DISABLED,
            )
        if FONT_FEATURES_AVAILABLE:
            font = none_format.font()
            for name in ('liga', 'clig', 'dlig', 'hlig', 'calt'):
                tag = QFont.Tag.fromString(name)
                self.assertTrue(font.isFeatureSet(tag))
                self.assertEqual(font.featureValue(tag), 0)
        trailing_format = _format_at(document, 2, 2)
        self.assertEqual(
            ligature_axis_value(
                trailing_format, LIGATURE_DISCRETIONARY
            ),
            LIGATURE_ENABLED,
        )
        self.assertEqual(
            ligature_axis_value(trailing_format, LIGATURE_CONTEXTUAL),
            LIGATURE_DISABLED,
        )
        exported = to_rich_text_html(document)
        self.assertIn('font-variant-ligatures: none', exported)
        self.assertIn(
            'font-variant-ligatures: '
            'discretionary-ligatures no-contextual',
            exported,
        )

    def test_explicit_ligatures_override_tracking_on_qt6(self):
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        def shaped_document(state: str, spacing: float) -> QTextDocument:
            document = QTextDocument('fi')
            document.setDefaultFont(QFont('DejaVu Serif', 24))
            cursor = QTextCursor(document)
            cursor.select(QTextCursor.SelectionType.Document)
            apply_letter_spacing(cursor, spacing, vertical=False)
            apply_ligature_axis(
                cursor, LIGATURE_COMMON, state, vertical=False
            )
            return document

        self.assertEqual(
            _glyph_count(shaped_document(LIGATURE_DEFAULT, 1.0)),
            1,
        )
        self.assertEqual(
            _glyph_count(shaped_document(LIGATURE_ENABLED, 1.0)),
            1,
        )
        self.assertEqual(
            _glyph_count(shaped_document(LIGATURE_DISABLED, 1.0)),
            2,
        )
        self.assertEqual(
            _glyph_count(shaped_document(LIGATURE_DEFAULT, 1.15)),
            2,
        )
        tracked = shaped_document(LIGATURE_ENABLED, 1.15)
        self.assertEqual(
            _glyph_count(tracked),
            1 if FONT_FEATURES_AVAILABLE else 2,
        )
        if FONT_FEATURES_AVAILABLE:
            font = _format_at(tracked, 0, 2).font()
            for name in ('liga', 'clig'):
                tag = QFont.Tag.fromString(name)
                self.assertTrue(font.isFeatureSet(tag))
                self.assertEqual(font.featureValue(tag), 1)

    def test_ligature_import_preserves_standard_css_letter_spacing(self):
        document = QTextDocument()
        load_rich_text_html(
            document,
            '<span style="letter-spacing: 0.2em; '
            'font-variant-ligatures: no-common-ligatures;">fi</span>',
        )

        char_format = _format_at(document, 0)
        self.assertEqual(
            char_format.fontLetterSpacingType(),
            QFont.SpacingType.PercentageSpacing,
        )
        self.assertAlmostEqual(char_format.fontLetterSpacing(), 120.0)

    def test_reapplying_ligature_axis_is_a_document_noop(self):
        document = QTextDocument('fi')
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        apply_ligature_axis(
            cursor, LIGATURE_COMMON, LIGATURE_DISABLED, vertical=False
        )
        document.clearUndoRedoStacks()
        changes = []
        document.contentsChanged.connect(lambda: changes.append(True))

        apply_ligature_axis(
            cursor, LIGATURE_COMMON, LIGATURE_DISABLED, vertical=False
        )

        self.assertEqual(document.availableUndoSteps(), 0)
        self.assertEqual(changes, [])

    def test_whole_item_ligature_format_reaches_empty_paragraphs(self):
        item = self._make_item(False, text='\nA\n\nB')
        item.setLigatureAxis(LIGATURE_COMMON, LIGATURE_DISABLED)
        for block_number in (0, 2):
            block = item.document().findBlockByNumber(block_number)
            cursor = QTextCursor(block)
            self.assertEqual(
                ligature_axis_value(cursor.charFormat(), LIGATURE_COMMON),
                LIGATURE_DISABLED,
            )
            cursor.insertText('fi')
            self.assertEqual(
                ligature_axis_value(cursor.charFormat(), LIGATURE_COMMON),
                LIGATURE_DISABLED,
            )

    def test_common_ligature_insertion_survives_writing_mode_round_trip(self):
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        item = self._make_item(False, text='X')
        item.setFontFamily('DejaVu Serif')
        item.setLetterSpacing(1.0)
        item.startEdit()
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setLigatureAxis(LIGATURE_COMMON, LIGATURE_ENABLED)

        item.setVertical(True)
        item.setVertical(False)
        cursor = item.textCursor()
        self.assertFalse(cursor.charFormat().hasProperty(
            QTextFormat.Property.FontLetterSpacing
        ))
        cursor.insertText('fi')
        item.setTextCursor(cursor)

        self.assertEqual(_glyph_count(item.document()), 2)

    def test_qt6_discretionary_shaping_overrides_tracking(self):
        if not FONT_FEATURES_AVAILABLE:
            self.skipTest('Qt 6.11 font features are unavailable')
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        document = QTextDocument('st')
        document.setDefaultFont(QFont('DejaVu Serif', 24))
        cursor = QTextCursor(document)
        cursor.select(QTextCursor.SelectionType.Document)
        self.assertEqual(_glyph_count(document), 2)
        apply_ligature_axis(
            cursor,
            LIGATURE_DISCRETIONARY,
            LIGATURE_ENABLED,
            vertical=False,
        )
        self.assertEqual(_glyph_count(document), 1)
        apply_letter_spacing(cursor, 1.15, vertical=False)
        self.assertEqual(_glyph_count(document), 1)
        dlig = QFont.Tag.fromString('dlig')
        font = _format_at(document, 0, 2).font()
        self.assertTrue(font.isFeatureSet(dlig))
        self.assertEqual(font.featureValue(dlig), 1)
        apply_letter_spacing(cursor, 1.0, vertical=False)
        self.assertEqual(_glyph_count(document), 1)

    def test_qt6_contextual_alternates_shape(self):
        if not FONT_FEATURES_AVAILABLE:
            self.skipTest('Qt 6.11 font features are unavailable')
        if QFontInfo(QFont('Fira Code')).family() != 'Fira Code':
            self.skipTest('Fira Code is unavailable')
        contextual = QTextDocument('->x')
        contextual.setDefaultFont(QFont('Fira Code', 24))
        contextual_cursor = QTextCursor(contextual)
        contextual_cursor.select(QTextCursor.SelectionType.Document)
        default_glyphs = _glyph_indexes(contextual)
        apply_ligature_axis(
            contextual_cursor,
            LIGATURE_CONTEXTUAL,
            LIGATURE_DISABLED,
            vertical=False,
        )
        self.assertNotEqual(_glyph_indexes(contextual), default_glyphs)
        apply_ligature_axis(
            contextual_cursor,
            LIGATURE_CONTEXTUAL,
            LIGATURE_ENABLED,
            vertical=False,
        )
        self.assertEqual(_glyph_indexes(contextual), default_glyphs)

        vertical = self._make_item(True, text='->x')
        vertical.setFontFamily('Fira Code')
        vertical.setLigatureAxis(
            LIGATURE_CONTEXTUAL, LIGATURE_DISABLED
        )
        vertical.layout.reLayout()
        layout = vertical.document().firstBlock().layout()
        disabled_glyphs = tuple(
            glyph
            for line_index in range(layout.lineCount())
            for run in layout.lineAt(line_index).glyphRuns()
            for glyph in run.glyphIndexes()
        )
        vertical.setLigatureAxis(
            LIGATURE_CONTEXTUAL, LIGATURE_ENABLED
        )
        vertical.layout.reLayout()
        layout = vertical.document().firstBlock().layout()
        enabled_glyphs = tuple(
            glyph
            for line_index in range(layout.lineCount())
            for run in layout.lineAt(line_index).glyphRuns()
            for glyph in run.glyphIndexes()
        )
        self.assertNotEqual(enabled_glyphs, disabled_glyphs)
        self.assertEqual(
            [
                layout.lineAt(index).textLength()
                for index in range(layout.lineCount())
            ],
            [1, 1, 1],
        )

    def test_vertical_cells_do_not_ligate_but_tate_chu_yoko_can(self):
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        def vertical_item() -> TextBlkItem:
            item = self._make_item(True, text='fi')
            item.setFontFamily('DejaVu Serif')
            item.setLetterSpacing(1.0)
            item.setLigatureAxis(LIGATURE_COMMON, LIGATURE_ENABLED)
            item.layout.reLayout()
            return item

        ordinary = vertical_item()
        ordinary_layout = ordinary.document().firstBlock().layout()
        self.assertEqual(
            [
                ordinary_layout.lineAt(index).textLength()
                for index in range(ordinary_layout.lineCount())
            ],
            [1, 1],
        )
        ordinary.setVertical(False)
        self.assertEqual(_glyph_count(ordinary.document()), 1)
        ordinary.setVertical(True)
        ordinary_layout = ordinary.document().firstBlock().layout()
        self.assertEqual(
            [
                ordinary_layout.lineAt(index).textLength()
                for index in range(ordinary_layout.lineCount())
            ],
            [1, 1],
        )

        combined = vertical_item()
        cursor = QTextCursor(combined.document())
        cursor.select(QTextCursor.SelectionType.Document)
        combined.setTextCursor(cursor)
        combined.setTateChuYoko(True)
        combined.layout.reLayout()
        combined_line = combined.document().firstBlock().layout().lineAt(0)
        self.assertEqual(combined_line.textLength(), 2)
        self.assertEqual(
            sum(
                len(run.glyphIndexes())
                for run in combined_line.glyphRuns()
            ),
            1,
        )

    def test_qt6_vertical_discretionary_ligature_keeps_logical_cells(self):
        if not FONT_FEATURES_AVAILABLE:
            self.skipTest('Qt 6.11 font features are unavailable')
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        item = self._make_item(True, text='st')
        item.setFontFamily('DejaVu Serif')
        tracking = 1.15
        item.setLetterSpacing(tracking)
        item.setLigatureAxis(
            LIGATURE_DISCRETIONARY, LIGATURE_ENABLED
        )
        item.layout.reLayout()
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        cells = item.layout._vertical_line_cells(block, 0)

        self.assertEqual(block.layout().lineCount(), 1)
        self.assertEqual(line.textLength(), 2)
        self.assertEqual(
            sum(len(run.glyphIndexes()) for run in line.glyphRuns()),
            1,
        )
        self.assertEqual(
            [(start, end) for start, end, *_rest in cells],
            [(0, 1), (1, 2)],
        )
        cell_height = item.layout.get_char_fontfmt(0, 0).tbr.height()
        tracked_extent = cell_height * tracking
        self.assertAlmostEqual(
            cells[-1][3] - cells[0][2], tracked_extent
        )
        for _start, _end, top, bottom, _is_space in cells:
            self.assertAlmostEqual(bottom - top, tracked_extent / 2)
        self.assertEqual(
            [
                item.layout.source_cursor_rect(position).top()
                for position in range(3)
            ],
            [cells[0][2], cells[0][3], cells[1][3]],
        )
        x = line.x() + 1.0
        for start, end, top, bottom, _is_space in cells:
            height = bottom - top
            self.assertEqual(
                item.layout.hitTest(
                    QPointF(x, top + height / 4),
                    Qt.HitTestAccuracy.FuzzyHit,
                ),
                start,
            )
            self.assertEqual(
                item.layout.hitTest(
                    QPointF(x, bottom - height / 4),
                    Qt.HitTestAccuracy.FuzzyHit,
                ),
                end,
            )

        wrapped = self._make_item(
            True, bounds=(0, 0, 200, 45), text='Ast'
        )
        wrapped.setFontFamily('DejaVu Serif')
        wrapped.setLetterSpacing(1.0)
        wrapped.setLigatureAxis(
            LIGATURE_DISCRETIONARY, LIGATURE_ENABLED
        )
        wrapped.layout.reLayout()
        wrapped_layout = wrapped.document().firstBlock().layout()
        leading = wrapped_layout.lineAt(0)
        cluster = wrapped_layout.lineAt(1)
        self.assertEqual(cluster.textLength(), 2)
        self.assertNotAlmostEqual(cluster.x(), leading.x())
        self.assertAlmostEqual(cluster.y(), 0.0)

    def test_qt6_vertical_discretionary_punctuation_uses_shaped_advance(self):
        if not FONT_FEATURES_AVAILABLE:
            self.skipTest('Qt 6.11 font features are unavailable')
        if QFontInfo(QFont('YW HeiTi')).family() != 'YW HeiTi':
            self.skipTest('YW HeiTi is unavailable')

        item = self._make_item(
            True, bounds=(0, 0, 320, 1050), text='哈!!!~~哈'
        )
        item.setStandardVerticalRomanAlignment(False)
        item.setFontFamily('YW HeiTi')
        item.setFontSize(120)
        item.setLetterSpacing(1.0)
        item.setLigatureAxis(
            LIGATURE_DISCRETIONARY, LIGATURE_ENABLED
        )
        item.layout.reLayout()
        block = item.document().firstBlock()
        layout = block.layout()

        self.assertEqual(
            [
                layout.lineAt(index).textLength()
                for index in range(layout.lineCount())
            ],
            [1, 3, 2, 1],
        )
        first_x = layout.lineAt(0).x()
        for index in range(layout.lineCount()):
            self.assertAlmostEqual(layout.lineAt(index).x(), first_x)
        for index in (1, 2):
            line = layout.lineAt(index)
            cells = item.layout._vertical_line_cells(block, index)
            self.assertEqual(len(cells), line.textLength())
            self.assertAlmostEqual(
                cells[-1][3] - cells[0][2],
                line.naturalTextWidth(),
            )

    def test_qt6_vertical_substitution_uses_result_glyph_orientation(self):
        if not FONT_FEATURES_AVAILABLE:
            self.skipTest('Qt 6.11 font features are unavailable')
        if QFontInfo(QFont('YW HeiTi')).family() != 'YW HeiTi':
            self.skipTest('YW HeiTi is unavailable')

        item = self._make_item(
            True, bounds=(0, 0, 320, 500), text='@01'
        )
        item.setFontFamily('YW HeiTi')
        item.setFontSize(120)
        item.setLetterSpacing(1.0)
        item.setLigatureAxis(
            LIGATURE_DISCRETIONARY, LIGATURE_ENABLED
        )
        item.layout.reLayout()
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        runs = line.glyphRuns()

        self.assertEqual((line.textStart(), line.textLength()), (0, 3))
        self.assertEqual(len(runs), 1)
        self.assertEqual(
            list(runs[0].glyphIndexes()),
            list(runs[0].rawFont().glyphIndexesForString('「')),
        )
        self.assertFalse(item.layout.needs_vertical_rotation('@'))
        self.assertFalse(
            item.layout.vertical_line_placement(block, 0)[2].isIdentity()
        )

    def test_vertical_single_punctuation_keeps_native_cell_advance(self):
        if QFontInfo(QFont('DejaVu Serif')).family() != 'DejaVu Serif':
            self.skipTest('DejaVu Serif is unavailable')

        for state, spacing in (
            (LIGATURE_DISABLED, 1.0),
            (LIGATURE_ENABLED, 1.15),
        ):
            with self.subTest(state=state, spacing=spacing):
                item = self._make_item(True, text='~')
                item.setFontFamily('DejaVu Serif')
                item.setFontSize(120)
                item.setLetterSpacing(spacing)
                item.setLigatureAxis(LIGATURE_COMMON, state)
                item.layout.reLayout()
                block = item.document().firstBlock()
                line = block.layout().lineAt(0)
                cells = item.layout._vertical_line_cells(block, 0)
                char_format = item.layout.get_char_fontfmt(0, 0)
                expected_advance = (
                    line.naturalTextWidth()
                    + char_format.tbr.height() * (spacing - 1)
                )

                self.assertEqual(line.textLength(), 1)
                self.assertAlmostEqual(
                    cells[0][3] - cells[0][2], expected_advance
                )

    def test_raw_utf16_propagation_handles_supplementary_replacement(self):
        target = self._make_item(False, text='aX')

        propagate_user_edit(target, 1, 1, '\U0001f600')

        self.assertEqual(target.toPlainText(), 'a\U0001f600')

        clustered = self._make_item(False, text='stX')
        clustered.setFontFamily('DejaVu Serif')
        clustered.setLigatureAxis(
            LIGATURE_DISCRETIONARY, LIGATURE_ENABLED
        )
        propagate_user_edit(clustered, 1, 1, 'a')
        self.assertEqual(clustered.toPlainText(), 'saX')

    def test_line_spacing_selection_is_block_scoped_and_end_exclusive(self):
        document = QTextDocument()
        document.setPlainText('AA\nBB\nCC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)

        apply_line_spacing(
            cursor, 1.5, LineSpacingType.Proportional
        )

        self.assertEqual(
            _line_spacing_at(document, 0),
            (1.5, LineSpacingType.Proportional),
        )
        self.assertEqual(
            _line_spacing_at(document, 1),
            (1.2, LineSpacingType.Proportional),
        )

        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_line_spacing(cursor, 0.8, LineSpacingType.Distance)

        self.assertEqual(
            _line_spacing_at(document, 0),
            (0.8, LineSpacingType.Distance),
        )
        self.assertEqual(
            _line_spacing_at(document, 1),
            (0.8, LineSpacingType.Distance),
        )
        self.assertEqual(
            _line_spacing_at(document, 2),
            (1.2, LineSpacingType.Proportional),
        )

    def test_line_spacing_caret_formats_current_block_and_enter_inherits(self):
        document = QTextDocument()
        document.setPlainText('A\nB')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        apply_line_spacing(cursor, 0.7, LineSpacingType.Distance)

        cursor.insertText('\nX')

        self.assertEqual(document.toPlainText(), 'A\nX\nB')
        self.assertEqual(
            _line_spacing_at(document, 0),
            (0.7, LineSpacingType.Distance),
        )
        self.assertEqual(
            _line_spacing_at(document, 1),
            (0.7, LineSpacingType.Distance),
        )
        self.assertEqual(
            _line_spacing_at(document, 2),
            (1.2, LineSpacingType.Proportional),
        )

    def test_line_spacing_html_uses_css_and_exact_app_metadata(self):
        source = QTextDocument()
        source.setPlainText('AA\nBB')
        first = QTextCursor(source)
        first.setPosition(0)
        apply_line_spacing(first, 1.25, LineSpacingType.Proportional)
        second = QTextCursor(source)
        second.movePosition(QTextCursor.MoveOperation.End)
        apply_line_spacing(second, 0.75, LineSpacingType.Distance)

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        css_reader = QTextDocument()
        css_reader.setHtml(html)

        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('line-height: 1.25;', html)
        self.assertIn('line-height: calc(1em + 7.5px);', html)
        self.assertIn(f'{LINE_DISTANCE_ATTRIBUTE}="0.75"', html)
        self.assertEqual(
            _line_spacing_at(restored, 0),
            (1.25, LineSpacingType.Proportional),
        )
        self.assertEqual(
            _line_spacing_at(restored, 1),
            (0.75, LineSpacingType.Distance),
        )
        self.assertEqual(
            _line_spacing_at(css_reader, 0),
            (1.25, LineSpacingType.Proportional),
        )

        apply_line_spacing(second, 1.1, LineSpacingType.Proportional)
        self.assertNotIn(
            LINE_DISTANCE_ATTRIBUTE,
            to_rich_text_html(source, html),
        )

    def test_standard_css_line_height_is_imported(self):
        document = QTextDocument()
        load_rich_text_html(
            document,
            '<p style="line-height: 1.35">A</p><p>B</p>',
        )

        self.assertEqual(
            _line_spacing_at(document, 0),
            (1.35, LineSpacingType.Proportional),
        )
        self.assertEqual(
            _line_spacing_at(
                document,
                1,
                0.6,
                LineSpacingType.Distance,
            ),
            (0.6, LineSpacingType.Distance),
        )

    def test_invalid_exact_line_spacing_keeps_valid_css(self):
        document = QTextDocument()
        load_rich_text_html(
            document,
            '<p style="line-height: 1.4" '
            f'{LINE_DISTANCE_ATTRIBUTE}="bad">A</p>',
        )

        self.assertEqual(
            _line_spacing_at(document, 0),
            (1.4, LineSpacingType.Proportional),
        )

    def test_letter_spacing_inline_html_round_trip(self):
        source = QTextDocument()
        source.setPlainText('A𠮷&B\nCD')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_letter_spacing(cursor, 1.35, vertical=False)

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        restored_with_conflicting_fallback = QTextDocument()
        load_rich_text_html(
            restored_with_conflicting_fallback,
            html,
            letter_spacing_fallback=2.0,
        )
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertIn('style="letter-spacing: 0.35em;"', html)
        self.assertIn(
            f'{LETTER_SPACING_ATTRIBUTE}="1.35"',
            html,
        )
        self.assertEqual(restored.toPlainText(), 'A𠮷&B\nCD')
        self.assertEqual(letter_spacing_value(_format_at(restored, 0)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(restored, 1, 2)), 1.35)
        self.assertEqual(letter_spacing_value(_format_at(restored, 3)), 1.35)
        self.assertEqual(letter_spacing_value(_format_at(restored, 4)), 1.0)
        self.assertEqual(letter_spacing_value(_format_at(restored, 6)), 1.0)
        self.assertEqual(
            letter_spacing_value(
                _format_at(restored_with_conflicting_fallback, 0)
            ),
            1.0,
        )
        self.assertEqual(
            letter_spacing_value(
                _format_at(restored_with_conflicting_fallback, 1)
            ),
            1.35,
        )
        self.assertEqual(legacy_reader.toPlainText(), 'A𠮷&B\nCD')
        self.assertEqual(
            _format_at(legacy_reader, 1, 2).font().letterSpacing(),
            135.0,
        )

    def test_old_item_spacing_migrates_to_inline_html(self):
        source = QTextDocument()
        source.setPlainText('legacy')
        old_html = source.toHtml()

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                block = TextBlock([0, 0, 300, 300])
                block._bounding_rect = [0, 0, 300, 300]
                block.translation = 'legacy'
                block.rich_text = old_html
                block.fontformat.vertical = vertical
                block.fontformat.letter_spacing = 1.35
                item = TextBlkItem(block, 0)

                for position in range(len('legacy')):
                    self.assertEqual(
                        letter_spacing_value(
                            _format_at(item.document(), position)
                        ),
                        1.35,
                    )
                migrated_html = item.toHtml()
                self.assertIn('style="letter-spacing: 0.35em;"', migrated_html)
                self.assertIn(
                    f'{LETTER_SPACING_ATTRIBUTE}="1.35"',
                    migrated_html,
                )
                restored = QTextDocument()
                load_rich_text_html(restored, migrated_html)
                self.assertEqual(
                    letter_spacing_value(_format_at(restored, 0)),
                    1.35,
                )

    def test_item_letter_spacing_uses_selection_then_insertion_format(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.5)

        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 0)),
            1.15,
        )
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.5,
        )
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 2)),
            1.15,
        )
        self.assertEqual(item.fontformat.letter_spacing, 1.15)

        cursor = item.textCursor()
        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setLetterSpacing(0.8)
        cursor = item.textCursor()
        cursor.insertText('D')
        item.setTextCursor(cursor)

        self.assertEqual(item.toPlainText(), 'ABCD')
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 3)),
            0.8,
        )
        self.assertEqual(item.fontformat.letter_spacing, 1.15)

    def test_nonediting_letter_spacing_updates_the_item_default(self):
        item = self._make_item(False, text='ABC')

        item.setLetterSpacing(1.6)

        self.assertEqual(item.fontformat.letter_spacing, 1.6)
        for position in range(3):
            self.assertEqual(
                letter_spacing_value(_format_at(item.document(), position)),
                1.6,
            )

    def test_item_line_spacing_uses_selection_without_changing_default(self):
        item = self._make_item(False, text='A\nB')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        item.setLineSpacingType(LineSpacingType.Distance)
        item.setLineSpacing(0.7)

        self.assertEqual(
            _line_spacing_at(item.document(), 0),
            (1.2, LineSpacingType.Proportional),
        )
        self.assertEqual(
            _line_spacing_at(item.document(), 1),
            (0.7, LineSpacingType.Distance),
        )
        self.assertEqual(item.fontformat.line_spacing, 1.2)
        self.assertEqual(
            item.fontformat.line_spacing_type,
            LineSpacingType.Proportional,
        )

    def test_nonediting_line_spacing_updates_default_and_every_block(self):
        item = self._make_item(False, text='A\nB')

        item.setLineSpacingType(LineSpacingType.Distance)
        item.setLineSpacing(0.65)

        self.assertEqual(item.fontformat.line_spacing, 0.65)
        self.assertEqual(
            item.fontformat.line_spacing_type,
            LineSpacingType.Distance,
        )
        for block_number in range(2):
            self.assertEqual(
                _line_spacing_at(item.document(), block_number),
                (0.65, LineSpacingType.Distance),
            )

    def test_whole_format_undo_path_preserves_saved_paragraph_spacing(self):
        item = self._make_item(False, text='A\nB')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        item.setTextCursor(cursor)
        item._set_line_spacing_pair(0.7, LineSpacingType.Distance)
        item.endEdit()
        old_html = item.toHtml()
        old_format = item.get_fontformat()
        replacement = old_format.deepcopy()
        replacement.line_spacing = 2.0

        item.set_fontformat(replacement, set_char_format=True)
        item.load_rich_text_html(old_html)
        item.set_fontformat(old_format)

        self.assertEqual(
            _line_spacing_at(item.document(), 0),
            (1.2, LineSpacingType.Proportional),
        )
        self.assertEqual(
            _line_spacing_at(item.document(), 1),
            (0.7, LineSpacingType.Distance),
        )

    def test_set_fontformat_relayouts_legacy_line_spacing_fallback(self):
        item = self._make_item(False, text='A\nB')

        def gap() -> float:
            first = item.document().findBlockByNumber(0).layout().lineAt(0)
            second = item.document().findBlockByNumber(1).layout().lineAt(0)
            return second.position().y() - first.position().y()

        original = item.fontformat.deepcopy()
        original_gap = gap()
        expanded = original.deepcopy()
        expanded.line_spacing = 2.0

        item.set_fontformat(expanded)
        expanded_gap = gap()
        item.set_fontformat(original)
        restored_gap = gap()

        self.assertGreater(expanded_gap, original_gap)
        self.assertAlmostEqual(restored_gap, original_gap, places=5)

    def test_line_spacing_uses_destination_block_in_both_writing_modes(self):
        def gap(item: TextBlkItem) -> float:
            first = item.document().findBlockByNumber(0).layout().lineAt(0)
            second = item.document().findBlockByNumber(1).layout().lineAt(0)
            if item.fontformat.vertical:
                return abs(second.position().x() - first.position().x())
            return abs(second.position().y() - first.position().y())

        def set_block_spacing(
            item: TextBlkItem,
            block_number: int,
            value: float,
        ) -> None:
            item.startEdit()
            block = item.document().findBlockByNumber(block_number)
            cursor = item.textCursor()
            cursor.setPosition(block.position())
            item.setTextCursor(cursor)
            item.setLineSpacing(value)

        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                baseline_item = self._make_item(vertical, text='A\nB')
                baseline = gap(baseline_item)
                first_item = self._make_item(vertical, text='A\nB')
                set_block_spacing(first_item, 0, 2.0)
                second_item = self._make_item(vertical, text='A\nB')
                set_block_spacing(second_item, 1, 2.0)

                self.assertAlmostEqual(gap(first_item), baseline, places=5)
                self.assertGreater(gap(second_item), baseline)

    def test_vertical_letter_spacing_is_per_character_and_survives_switch(self):
        item = self._make_item(False, text='甲乙丙')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(2.0)

        item.setVertical(True)
        heights = [bottom - top for top, bottom in item.layout.y_offset_lst[0]]
        self.assertGreater(heights[1], heights[0])
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            2.0,
        )
        self.assertEqual(
            _format_at(item.document(), 1).font().letterSpacing(),
            100.0,
        )

        item.setVertical(False)
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            2.0,
        )
        self.assertEqual(
            _format_at(item.document(), 1).font().letterSpacing(),
            200.0,
        )

    def test_spacing_insertion_format_survives_writing_mode_switch(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setLetterSpacing(0.75)

        item.setVertical(True)
        cursor = item.textCursor()
        cursor.insertText('D')
        item.setTextCursor(cursor)

        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 3)),
            0.75,
        )
        self.assertEqual(
            _format_at(item.document(), 3).font().letterSpacing(),
            100.0,
        )

    def test_nonediting_item_applies_to_document_and_restores_cursor(self):
        item = self._make_item()
        cursor = item.textCursor()
        cursor.setPosition(2)
        item.setTextCursor(cursor)

        item.setEmphasis('filled circle', 'over right')

        self.assertEqual(item.textCursor().position(), 2)
        self.assertFalse(item.textCursor().hasSelection())
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 0, 2)),
            ('filled circle', 'over right'),
        )
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 2, 1)),
            ('filled circle', 'over right'),
        )

    def test_custom_clipboard_round_trip_preserves_annotations(self):
        source = QTextDocument()
        source.setPlainText('copy me')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'open sesame', 'over right')
        apply_text_combine_upright(cursor, True)
        apply_letter_spacing(cursor, 1.4, vertical=False)
        apply_ligature_axis(
            cursor,
            LIGATURE_DISCRETIONARY,
            LIGATURE_ENABLED,
            vertical=False,
        )
        apply_line_spacing(cursor, 0.8, LineSpacingType.Distance)

        mime = create_rich_text_mime(cursor)
        target = QTextDocument()
        inserted = insert_rich_text_mime(QTextCursor(target), mime)

        self.assertTrue(inserted)
        self.assertIn(LETTER_SPACING_ATTRIBUTE, mime.html())
        self.assertIn(LINE_DISTANCE_ATTRIBUTE, mime.html())
        self.assertIn('text-emphasis-style: open sesame', mime.html())
        self.assertIn('text-combine-upright: all', mime.html())
        self.assertIn(
            'font-variant-ligatures: discretionary-ligatures',
            mime.html(),
        )
        self.assertEqual(target.toPlainText(), 'copy')
        self.assertEqual(
            emphasis_values(_format_at(target, 0, 4)),
            ('open sesame', 'over right'),
        )
        self.assertEqual(
            letter_spacing_value(_format_at(target, 0, 4)),
            1.4,
        )
        self.assertEqual(
            text_combine_upright_values(_format_at(target, 0, 4))[0],
            'all',
        )
        self.assertEqual(
            ligature_axis_value(
                _format_at(target, 0, 4),
                LIGATURE_DISCRETIONARY,
            ),
            LIGATURE_ENABLED,
        )
        self.assertEqual(
            _line_spacing_at(target, 0),
            (0.8, LineSpacingType.Distance),
        )

    def test_text_combine_inline_round_trip_keeps_qt_html_readable(self):
        source = QTextDocument()
        source.setPlainText('A12B')
        cursor = QTextCursor(source)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#2070c0'))
        cursor.mergeCharFormat(char_format)
        apply_text_combine_upright(cursor, True)
        _value, source_group_id = text_combine_upright_values(
            _format_at(source, 1, 2)
        )

        html = to_rich_text_html(source)
        restored = QTextDocument()
        load_rich_text_html(restored, html)
        legacy_reader = QTextDocument()
        legacy_reader.setHtml(html)

        self.assertIn('text-combine-upright: all', html)
        self.assertIn(TEXT_COMBINE_ID_ATTRIBUTE, html)
        self.assertEqual(restored.toPlainText(), 'A12B')
        self.assertEqual(legacy_reader.toPlainText(), 'A12B')
        self.assertEqual(
            text_combine_upright_values(_format_at(legacy_reader, 1, 2))[0],
            'none',
        )
        value, group_id = text_combine_upright_values(
            _format_at(restored, 1, 2)
        )
        self.assertEqual(value, 'all')
        self.assertEqual(group_id, source_group_id)
        restored_format = _format_at(restored, 1, 2)
        self.assertTrue(restored_format.font().bold())
        self.assertTrue(restored_format.font().italic())
        self.assertEqual(
            restored_format.foreground().color(), QColor('#2070c0')
        )

    def test_text_combine_selection_and_insertion_format_group_runs(self):
        document = QTextDocument()
        document.setPlainText('ABC')
        cursor = QTextCursor(document)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(cursor, True)

        cursor.clearSelection()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        apply_text_combine_upright(cursor, True)
        cursor.insertText('1')
        cursor.insertText('2')
        apply_text_combine_upright(cursor, False)
        cursor.insertText('3')

        ranges = text_combine_upright_ranges(document.firstBlock())
        self.assertEqual([(start, length) for start, length, _id in ranges], [
            (1, 1),
            (3, 2),
        ])
        self.assertNotEqual(ranges[0][2], ranges[1][2])
        self.assertEqual(
            text_combine_upright_values(_format_at(document, 5))[0],
            'none',
        )

    def test_auto_text_combine_replaces_runs_and_skips_long_or_ruby_text(self):
        document = QTextDocument()
        document.setPlainText('12-ABC-12345-東京-🅰🅱')

        old_run = QTextCursor(document)
        old_run.setPosition(7)
        old_run.setPosition(12, QTextCursor.MoveMode.KeepAnchor)
        apply_text_combine_upright(old_run, True)

        ruby = QTextCursor(document)
        ruby.setPosition(13)
        ruby.setPosition(15, QTextCursor.MoveMode.KeepAnchor)
        apply_ruby(ruby, 'group', 'とうきょう')

        changed = apply_auto_text_combine_upright(
            document,
            frozenset('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ東京🅰🅱'),
            4,
        )
        self.assertTrue(changed)

        ranges = text_combine_upright_ranges(document.firstBlock())
        self.assertEqual(
            [(start, length) for start, length, _group_id in ranges],
            [(0, 2), (3, 3), (16, 4)],
        )
        self.assertEqual(len({group_id for *_range, group_id in ranges}), 3)
        self.assertEqual(len(ruby_containers(document)), 1)

        self.assertTrue(
            apply_auto_text_combine_upright(document, frozenset(), 4)
        )
        self.assertEqual(text_combine_upright_ranges(document.firstBlock()), ())
        self.assertEqual(len(ruby_containers(document)), 1)
        self.assertFalse(
            apply_auto_text_combine_upright(document, frozenset(), 4)
        )

    def test_adjacent_text_combine_runs_and_pastes_keep_boundaries(self):
        source = QTextDocument()
        source.setPlainText('1234')
        cursor = QTextCursor(source)
        for start, end in ((0, 2), (2, 4)):
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            apply_text_combine_upright(cursor, True)

        source_ranges = text_combine_upright_ranges(source.firstBlock())
        self.assertEqual([length for _start, length, _id in source_ranges], [2, 2])
        self.assertNotEqual(source_ranges[0][2], source_ranges[1][2])

        item = self._make_item(True, text='1234')
        item.startEdit()
        item_cursor = item.textCursor()
        for start, end in ((0, 2), (2, 4)):
            item_cursor.setPosition(start)
            item_cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            item.setTextCursor(item_cursor)
            item.setTateChuYoko(True)
        item_layout = item.document().firstBlock().layout()
        self.assertEqual(item_layout.lineCount(), 2)
        self.assertEqual(
            [item_layout.lineAt(index).textLength() for index in range(2)],
            [2, 2],
        )

        cursor.select(QTextCursor.SelectionType.Document)
        mime = create_rich_text_mime(cursor)
        target = QTextDocument()
        target_cursor = QTextCursor(target)
        self.assertTrue(insert_rich_text_mime(target_cursor, mime))
        target_cursor.movePosition(QTextCursor.MoveOperation.End)
        self.assertTrue(insert_rich_text_mime(target_cursor, mime))

        pasted_ranges = text_combine_upright_ranges(target.firstBlock())
        self.assertEqual(
            [length for _start, length, _id in pasted_ranges],
            [2, 2, 2, 2],
        )
        self.assertEqual(len({group_id for *_range, group_id in pasted_ranges}), 4)

    def test_textblock_rich_text_round_trip_uses_production_item_boundary(self):
        source = self._make_item()
        source.startEdit()
        cursor = source.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        source.setTextCursor(cursor)
        source.setEmphasis('open triangle', 'under left')
        cursor.setPosition(3)
        cursor.setPosition(7, QTextCursor.MoveMode.KeepAnchor)
        source.setTextCursor(cursor)
        source.setTateChuYoko(True)

        block = TextBlock([0, 0, 600, 300])
        block._bounding_rect = [0, 0, 600, 300]
        block.translation = source.toPlainText()
        block.rich_text = source.toHtml()
        restored = TextBlkItem(block, 1)

        self.assertEqual(restored.toPlainText(), source.toPlainText())
        self.assertEqual(
            emphasis_values(_format_at(restored.document(), 0, 2)),
            ('open triangle', 'under left'),
        )
        self.assertEqual(
            text_combine_upright_values(
                _format_at(restored.document(), 3, 4)
            )[0],
            'all',
        )

    def test_text_combine_uses_one_natural_width_vertical_cell(self):
        item = self._make_item(True, text='年12月')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')

        block = item.document().firstBlock()
        text_layout = block.layout()
        self.assertEqual(text_layout.lineCount(), 3)
        line = text_layout.lineAt(1)
        self.assertEqual((line.textStart(), line.textLength()), (1, 2))
        cell = item.layout.tate_chu_yoko_cell_rect(block, 1)
        self.assertIsNotNone(cell)
        natural_bounds = tate_chu_yoko_natural_bounds(line)
        self.assertGreaterEqual(cell.width(), natural_bounds.width())
        ink = tate_chu_yoko_ink_bounds(line, cell)
        self.assertTrue(cell.adjusted(-0.01, -0.01, 0.01, 0.01).contains(ink))
        _line, offset, orientation = item.layout.vertical_line_placement(
            block, 1
        )
        self.assertAlmostEqual(orientation.m11(), 1.0)
        self.assertAlmostEqual(orientation.m22(), 1.0)
        self.assertAlmostEqual(orientation.m12(), 0.0)
        self.assertAlmostEqual(orientation.m21(), 0.0)
        mark_ink = emphasis_ink_bounds(
            block,
            line,
            vertical=True,
            offset=offset,
            orientation=orientation,
        )
        self.assertTrue(
            item.boundingRect().adjusted(
                -0.01, -0.01, 0.01, 0.01
            ).contains(mark_ink)
        )

        caret = item.layout.source_cursor_rect(2)
        self.assertAlmostEqual(caret.width(), 2.0)
        self.assertGreater(caret.height(), caret.width())
        positions = [
            item.layout.hitTest(
                QPointF(x, cell.center().y()),
                Qt.HitTestAccuracy.FuzzyHit,
            )
            for x in (
                cell.left() + 0.25,
                cell.center().x(),
                cell.right() - 0.25,
            )
        ]
        self.assertEqual(positions, [1, 2, 3])

    def test_text_combine_cursor_blink_invalidates_mapped_rect(self):
        item = self._make_item(True, text='年12月')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)

        updates = []
        item.layout.update.connect(
            lambda *args: updates.append(
                QRectF(args[0]) if args else QRectF(item.boundingRect())
            )
        )
        image = QImage(
            item.boundingRect().size().toSize(),
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        context = QAbstractTextDocumentLayout.PaintContext()
        try:
            context.cursorPosition = 2
            item.layout.draw(painter, context)
            caret = item.layout.source_cursor_rect(2)
            updates.clear()

            context.cursorPosition = -1
            item.layout.draw(painter, context)
        finally:
            painter.end()

        self.assertTrue(any(rect.contains(caret) for rect in updates))

        supplementary = self._make_item(True, text='A𠮷1B')
        supplementary.startEdit()
        cursor = supplementary.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        supplementary.setTextCursor(cursor)
        supplementary.setTateChuYoko(True)
        supplementary_layout = supplementary.document().firstBlock().layout()
        self.assertEqual(supplementary_layout.lineCount(), 3)
        self.assertEqual(supplementary_layout.lineAt(1).textLength(), 3)

        spaced = self._make_item(True, text='年1 2月')
        spaced.startEdit()
        cursor = spaced.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        spaced.setTextCursor(cursor)
        spaced.setTateChuYoko(True)
        spaced_layout = spaced.document().firstBlock().layout()
        self.assertEqual(spaced_layout.lineCount(), 3)
        self.assertEqual(spaced_layout.lineAt(1).textLength(), 3)

        mixed_size = self._make_item(True, text='A12B')
        mixed_size.startEdit()
        cursor = mixed_size.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        large = QTextCharFormat()
        large.setFontPointSize(72.0)
        cursor.mergeCharFormat(large)
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        mixed_size.setTextCursor(cursor)
        mixed_size.setTateChuYoko(True)
        mixed_block = mixed_size.document().firstBlock()
        mixed_line = mixed_block.layout().lineAt(1)
        mixed_cell = mixed_size.layout.tate_chu_yoko_cell_rect(
            mixed_block, 1
        )
        mixed_ink = tate_chu_yoko_ink_bounds(mixed_line, mixed_cell)
        self.assertTrue(
            mixed_cell.adjusted(-0.01, -0.01, 0.01, 0.01).contains(
                mixed_ink
            )
        )

        partial_mark = self._make_item(True, text='12')
        partial_mark.startEdit()
        cursor = partial_mark.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        partial_mark.setTextCursor(cursor)
        partial_mark.setTateChuYoko(True)
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        partial_mark.setTextCursor(cursor)
        partial_mark.setEmphasis('filled dot', 'over right')
        partial_block = partial_mark.document().firstBlock()
        partial_line, offset, orientation = (
            partial_mark.layout.vertical_line_placement(partial_block, 0)
        )
        self.assertFalse(
            emphasis_ink_bounds(
                partial_block,
                partial_line,
                vertical=True,
                offset=offset,
                orientation=orientation,
            ).isEmpty()
        )

    def test_text_combine_overhang_does_not_move_columns_or_border(self):
        item = self._make_item(
            True,
            bounds=(100, 20, 100, 90),
            text='甲12乙丙丁戊',
        )
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        self.app.processEvents()

        def line_x_positions() -> list[float]:
            text_layout = item.document().firstBlock().layout()
            return [
                text_layout.lineAt(index).x()
                for index in range(text_layout.lineCount())
            ]

        logical_rect = item.rect()
        layout_width = item.layout.max_width
        column_positions = line_x_positions()
        old_paint_width = item.boundingRect().width()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.insertText('3456')
        item.setTextCursor(cursor)
        self.app.processEvents()

        block = item.document().firstBlock()
        line = block.layout().lineForTextPosition(1)
        cell = item.layout.tate_chu_yoko_cell_rect(
            block, line.lineNumber()
        )
        ink = tate_chu_yoko_ink_bounds(line, cell)
        self.assertEqual(item.rect(), logical_rect)
        self.assertAlmostEqual(item.layout.max_width, layout_width)
        self.assertEqual(line_x_positions(), column_positions)
        self.assertGreater(item.boundingRect().width(), old_paint_width)
        self.assertTrue(item.boundingRect().contains(ink))
        left_hit = QPointF(cell.left() + 0.01, cell.center().y())
        right_hit = QPointF(cell.right() - 0.01, cell.center().y())
        self.assertTrue(item.shape().contains(left_hit))
        self.assertTrue(item.shape().contains(right_hit))
        self.assertEqual(
            item.layout.hitTest(
                left_hit,
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            line.textStart(),
        )
        self.assertEqual(
            item.layout.hitTest(
                right_hit,
                Qt.HitTestAccuracy.FuzzyHit,
            ),
            line.textStart() + line.textLength(),
        )

    def test_text_combine_is_persistent_but_visually_inert_horizontally(self):
        plain = self._make_item(False, text='A12B')
        combined = self._make_item(False, text='A12B')
        combined.startEdit()
        cursor = combined.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        combined.setTextCursor(cursor)
        combined.setTateChuYoko(True)

        plain_line = plain.document().firstBlock().layout().lineAt(0)
        combined_line = combined.document().firstBlock().layout().lineAt(0)
        self.assertEqual(combined.toPlainText(), plain.toPlainText())
        self.assertEqual(combined_line.textLength(), plain_line.textLength())
        self.assertAlmostEqual(
            combined_line.naturalTextWidth(), plain_line.naturalTextWidth()
        )
        self.assertEqual(
            text_combine_upright_values(
                _format_at(combined.document(), 1, 2)
            )[0],
            'all',
        )
        combined.setVertical(True)
        vertical_layout = combined.document().firstBlock().layout()
        self.assertEqual(vertical_layout.lineCount(), 3)
        self.assertEqual(vertical_layout.lineAt(1).textLength(), 2)

    def test_wrapped_text_combine_reserves_its_own_visible_column_width(self):
        item = self._make_item(
            True,
            bounds=(0, 0, 180, 55),
            text='年年12',
        )
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        cursor.setPosition(4, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontPointSize(24.0)
        cursor.mergeCharFormat(char_format)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')

        block = item.document().firstBlock()
        line = block.layout().lineAt(2)
        record = item.layout.per_char_records[0][2]
        cell = item.layout.tate_chu_yoko_cell_rect(block, 2)
        self.assertLess(line.x(), block.layout().lineAt(0).x())
        self.assertGreater(record['line_width'], record['text_combine_width'])
        fixed_point_tolerance = 1.0 / 64.0 + 0.001
        self.assertGreaterEqual(
            cell.left(),
            item.layout.layout_left - fixed_point_tolerance,
        )
        self.assertLessEqual(
            cell.right(), line.x() + record['line_width'] + 0.01
        )

    def test_text_combine_renders_with_styles_effects_and_glyph_slant(self):
        block = TextBlock([0, 0, 180, 160])
        block._bounding_rect = [0, 0, 180, 160]
        block.vertical = True
        block.translation = '年12月'
        block.fontformat.glyph_slant_angle = 12.0
        block.fontformat.stroke_width = 0.08
        block.fontformat.shadow_radius = 0.06
        block.fontformat.shadow_strength = 0.7
        block.fontformat.shadow_offset = [0.05, 0.04]
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)

        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontWeight(QFont.Weight.Bold)
        char_format.setFontItalic(True)
        char_format.setForeground(QColor('#df4050'))
        cursor.mergeCharFormat(char_format)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.setEmphasis('filled sesame', 'over right')
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        renderer = item.geometry_controller.layout_renderer
        self.assertIsNotNone(renderer)
        ink_bounds = renderer.ink_bounds()
        self.assertFalse(ink_bounds.isEmpty())
        self.assertTrue(
            item.boundingRect().adjusted(-0.01, -0.01, 0.01, 0.01).contains(
                ink_bounds
            )
        )
        self.assertGreater(item.effect_renderer.padding(), 0.0)

        image = QImage(
            260,
            220,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(QColor(0, 0, 0, 0))
        painter = QPainter(image)
        try:
            scene.render(
                painter,
                QRectF(0, 0, 260, 220),
                scene.itemsBoundingRect(),
            )
        finally:
            painter.end()
        byte_count = (
            image.sizeInBytes()
            if hasattr(image, 'sizeInBytes')
            else image.byteCount()
        )
        pixels = bytes(image.bits().asstring(byte_count))
        self.assertTrue(any(pixels[3::4]))

    def test_annotation_effect_cache_tracks_render_scale(self):
        for effect in ('stroke', 'shadow'):
            with self.subTest(effect=effect):
                block = TextBlock([0, 0, 140, 140])
                block._bounding_rect = [0, 0, 140, 140]
                block.vertical = True
                block.alignment = TextAlignment.Right
                block.translation = '天天'
                block.fontformat.font_size = 48
                if effect == 'stroke':
                    block.fontformat.stroke_width = 0.2
                else:
                    block.fontformat.shadow_radius = 0.04
                    block.fontformat.shadow_strength = 0.8
                    block.fontformat.shadow_offset = [0.04, 0.04]
                item = TextBlkItem(block, 0)
                scene = QGraphicsScene()
                scene.addItem(item)

                item.startEdit()
                cursor = item.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                item.setTextCursor(cursor)
                item.setTateChuYoko(True)
                item.setEmphasis('filled dot', 'over right')
                item.endEdit(keep_focus=False)
                self.app.processEvents()

                source = scene.itemsBoundingRect()
                image = QImage(
                    max(1, round(source.width() * 4)),
                    max(1, round(source.height() * 4)),
                    QImage.Format.Format_ARGB32_Premultiplied,
                )
                image.fill(Qt.GlobalColor.transparent)
                painter = QPainter(image)
                try:
                    scene.render(painter, QRectF(image.rect()), source)
                finally:
                    painter.end()

                renderer = item.effect_renderer
                self.assertEqual(renderer.background_pixmap_scale, 4.0)
                self.assertEqual(
                    renderer.background_pixmap.devicePixelRatioF(), 4.0
                )
                alpha = pixmap2ndarray(
                    renderer.background_pixmap, keep_alpha=True
                )[..., 3]
                self.assertTrue(((alpha > 0) & (alpha < 255)).any())
                scene.removeItem(item)

    def test_native_stroke_alignment_is_transient_and_reversible(self):
        block = TextBlock([0, 0, 90, 180])
        block._bounding_rect = [0, 0, 90, 180]
        block.translation = '哈尔滨\n佛学院'
        block.vertical = True
        block.fontformat.font_family = 'Source Han Sans'
        block.fontformat.font_size = 7.0
        block.fontformat.stroke_width = 0.2
        item = TextBlkItem(block, 0)
        document = item.document()
        revision = document.revision()
        html = to_rich_text_html(document)
        undo_steps = document.availableUndoSteps()

        item.repaint_background(6.0)
        alignment_ranges = []
        document_block = document.firstBlock()
        while document_block.isValid():
            alignment_ranges.extend(
                entry
                for entry in document_block.layout().formats()
                if bool(entry.format.property(
                    effect_rendering.STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY
                ))
            )
            document_block = document_block.next()
        self.assertEqual(len(alignment_ranges), document.blockCount())
        outline = alignment_ranges[0].format.textOutline()
        self.assertEqual(outline.style(), Qt.PenStyle.SolidLine)
        self.assertEqual(outline.color().alpha(), 0)
        self.assertEqual(outline.widthF(), 0.0)
        self.assertEqual(document.revision(), revision)
        self.assertEqual(to_rich_text_html(document), html)
        self.assertEqual(document.availableUndoSteps(), undo_steps)
        with patch.object(
            item.layout, 'reLayout', wraps=item.layout.reLayout
        ) as relayout:
            item.repaint_background(6.0)
        relayout.assert_not_called()

        item.setStrokeWidth(0.0)
        document_block = document.firstBlock()
        while document_block.isValid():
            self.assertFalse(any(
                bool(entry.format.property(
                    effect_rendering.STROKE_ALIGNMENT_LAYOUT_FORMAT_PROPERTY
                ))
                for entry in document_block.layout().formats()
            ))
            document_block = document_block.next()
        self.assertEqual(to_rich_text_html(document), html)

    def test_native_stroke_alignment_blocks_reentrant_repaint(self):
        block = TextBlock([0, 0, 90, 180])
        block._bounding_rect = [0, 0, 90, 180]
        block.translation = '哈尔滨\n佛学院'
        block.vertical = True
        item = TextBlkItem(block, 0)
        renderer = item.effect_renderer
        original_relayout = item.layout.reLayout
        block.fontformat.stroke_width = 0.2
        guard_states = []

        def reentrant_relayout() -> None:
            guard_states.append(item.repainting)
            item.repaint_background(6.0)
            original_relayout()

        with patch.object(
            item.layout, 'reLayout', side_effect=reentrant_relayout
        ) as relayout, patch.object(
            renderer,
            '_render_effect_surface',
            wraps=renderer._render_effect_surface,
        ) as render_surface:
            item.repaint_background(6.0)

        self.assertGreaterEqual(relayout.call_count, 1)
        self.assertTrue(all(guard_states))
        render_surface.assert_called_once()
        self.assertFalse(item.repainting)

    def test_small_native_stroke_is_centered_on_production_fill(self):
        if QFontInfo(QFont('Source Han Sans')).family() != 'Source Han Sans':
            self.skipTest('Source Han Sans is unavailable')
        block = TextBlock([0, 0, 30, 30])
        block._bounding_rect = [0, 0, 30, 30]
        block.translation = '佛'
        block.vertical = True
        block.fontformat.font_family = 'Source Han Sans'
        block.fontformat.font_size = 7.0
        block.fontformat.stroke_width = 0.2
        block.fontformat.frgb = [255, 0, 0]
        block.fontformat.srgb = [0, 0, 255]
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.addCleanup(scene.removeItem, item)

        source = item.boundingRect()
        scale = 6
        pixmap = QPixmap(
            round(source.width() * scale),
            round(source.height() * scale),
        )
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            scene.render(painter, QRectF(pixmap.rect()), source)
        finally:
            painter.end()
        pixels = pixmap2ndarray(pixmap, keep_alpha=True)
        outer_y, outer_x = (pixels[..., 3] > 8).nonzero()
        fill_y, fill_x = (
            (pixels[..., 0] > pixels[..., 2] * 1.5)
            & (pixels[..., 0] > 32)
        ).nonzero()
        self.assertTrue(outer_x.size)
        self.assertTrue(fill_x.size)
        margins = (
            int(fill_x.min() - outer_x.min()),
            int(fill_y.min() - outer_y.min()),
            int(outer_x.max() - fill_x.max()),
            int(outer_y.max() - fill_y.max()),
        )
        self.assertLessEqual(max(margins) - min(margins), 1)

    def test_font_size_stepper_keeps_stroke_and_fill_aligned(self):
        if QFontInfo(QFont('Source Han Sans')).family() != 'Source Han Sans':
            self.skipTest('Source Han Sans is unavailable')
        previous_canvas = getattr(SW, 'canvas', None)
        previous_active_format = C.active_format
        canvas = Canvas()
        SW.canvas = canvas
        self.addCleanup(setattr, SW, 'canvas', previous_canvas)
        self.addCleanup(
            setattr, C, 'active_format', previous_active_format
        )
        self.addCleanup(canvas.gv.deleteLater)

        host = QWidget()
        host_layout = QHBoxLayout(host)
        host_layout.addWidget(canvas.gv)
        self.addCleanup(host.deleteLater)

        block = TextBlock([0, 0, 30, 30])
        block._bounding_rect = [0, 0, 30, 30]
        block.translation = '佛'
        block.fontformat.vertical = True
        block.fontformat.font_family = 'Source Han Sans'
        block.fontformat.font_size = 7.0
        block.fontformat.stroke_width = 0.2
        block.fontformat.frgb = [255, 0, 0]
        block.fontformat.srgb = [0, 0, 255]
        item = TextBlkItem(block, 0)
        item.setParentItem(canvas.textLayer)
        item.setSelected(True)
        self.addCleanup(item.setParentItem, None)

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        panel.global_format = FontFormat()
        host_layout.addWidget(panel)
        host.resize(900, 500)
        host.show()
        panel.set_textblk_item(item)
        item.startEdit()
        self.app.processEvents()

        changes = []
        panel.fontsizebox.param_changed.connect(
            lambda name, value: changes.append((name, value))
        )
        QTest.mouseClick(
            panel.fontsizebox.upBtn,
            Qt.MouseButton.LeftButton,
        )
        self.app.processEvents()

        self.assertEqual(changes, [('font_size', 9.0)])
        self.assertEqual(item.get_fontformat().font_size, 9.0)
        self.assertTrue(panel.fontsizebox.fcombobox.hasFocus())
        item.set_ui_guide_suppressed(True)
        source = item.sceneBoundingRect()
        scale = 6
        pixmap = QPixmap(
            round(source.width() * scale),
            round(source.height() * scale),
        )
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        try:
            canvas.render(painter, QRectF(pixmap.rect()), source)
        finally:
            painter.end()
        pixels = pixmap2ndarray(pixmap, keep_alpha=True)
        outer_y, outer_x = (
            (pixels[..., 2] > pixels[..., 0] * 1.5)
            & (pixels[..., 2] > 32)
        ).nonzero()
        fill_y, fill_x = (
            (pixels[..., 0] > pixels[..., 2] * 1.5)
            & (pixels[..., 0] > 32)
        ).nonzero()
        margins = (
            int(fill_x.min() - outer_x.min()),
            int(fill_y.min() - outer_y.min()),
            int(outer_x.max() - fill_x.max()),
            int(outer_y.max() - fill_y.max()),
        )
        self.assertLessEqual(max(margins) - min(margins), 1)

    def test_nonediting_effect_change_invalidates_scene_cache(self):
        for effect in ('stroke width', 'shadow color'):
            with self.subTest(effect=effect):
                block = TextBlock([0, 0, 140, 140])
                block._bounding_rect = [0, 0, 140, 140]
                block.translation = 'Effect'
                if effect == 'stroke width':
                    block.fontformat.stroke_width = 0.05
                else:
                    block.fontformat.shadow_radius = 0.1
                    block.fontformat.shadow_strength = 0.8
                item = TextBlkItem(block, 0)
                item.setSelected(True)
                scene = QGraphicsScene()
                scene.addItem(item)
                self.app.processEvents()

                changed_regions = []
                scene.changed.connect(changed_regions.extend)
                old_cache_key = item.effect_renderer.background_pixmap.cacheKey()
                if effect == 'stroke width':
                    item.setStrokeWidth(0.2)
                else:
                    shadow = item.fontformat.deepcopy()
                    shadow.shadow_color = [255, 0, 0]
                    item.setShadow(shadow)
                self.app.processEvents()

                self.assertFalse(item.isEditing())
                self.assertTrue(item.isSelected())
                self.assertNotEqual(
                    item.effect_renderer.background_pixmap.cacheKey(),
                    old_cache_key,
                )
                self.assertTrue(changed_regions)
                scene.removeItem(item)

    def test_tate_chu_yoko_stroke_has_no_small_glyph_cavity(self):
        block = TextBlock([0, 0, 140, 140])
        block._bounding_rect = [0, 0, 140, 140]
        block.vertical = True
        block.translation = '!'
        block.fontformat.font_size = 48
        block.fontformat.stroke_width = 0.4
        item = TextBlkItem(block, 0)

        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        item.repaint_background()
        alpha = pixmap2ndarray(
            item.effect_renderer.background_pixmap, keep_alpha=True
        )[..., 3]
        occupied_y, occupied_x = alpha.nonzero()
        center_x = int(round((occupied_x.min() + occupied_x.max()) / 2))
        center_column = alpha[:, center_x]
        occupied_column = center_column.nonzero()[0]

        self.assertGreater(occupied_column.size, 0)
        self.assertTrue(
            (
                center_column[
                    occupied_column[0]:occupied_column[-1] + 1
                ]
                > 0
            ).all()
        )

    def test_document_replace_keeps_annotation_attached_to_replacement(self):
        source = QTextDocument()
        source.setPlainText('foo bar')
        cursor = QTextCursor(source)
        cursor.setPosition(0)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        apply_emphasis(cursor, 'filled dot', 'over right')
        apply_text_combine_upright(cursor, True)

        edited = QTextDocument()
        load_rich_text_html(edited, to_rich_text_html(source))
        doc_replace(edited, [[0, 3]], 'longer')
        restored = QTextDocument()
        load_rich_text_html(restored, to_rich_text_html(edited))

        self.assertEqual(restored.toPlainText(), 'longer bar')
        self.assertEqual(
            emphasis_values(_format_at(restored, 0, 6)),
            ('filled dot', 'over right'),
        )
        self.assertEqual(
            emphasis_values(_format_at(restored, 6)),
            ('none', 'over right'),
        )
        self.assertEqual(
            text_combine_upright_values(_format_at(restored, 0, 6))[0],
            'all',
        )

    def test_grapheme_ranges_match_qt_utf16_positions(self):
        self.assertEqual(
            _grapheme_ranges('A𠮷e\u0301👩\u200d👩\u200d👧\u200d👦'),
            ((0, 1), (1, 3), (3, 5), (5, 16)),
        )

    def test_vertical_emphasis_keeps_supplementary_layout_records_aligned(self):
        item = self._make_item(True, text='A𠮷B')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setEmphasis('filled dot', 'over right')

        text_layout = item.document().firstBlock().layout()
        self.assertEqual(
            [
                text_layout.lineAt(index).textLength()
                for index in range(text_layout.lineCount())
            ],
            [1, 2, 1],
        )
        self.assertEqual(
            len(item.layout.line_spaces_lst[0]),
            text_layout.lineCount(),
        )
        self.assertEqual(
            len(item.layout.y_offset_lst[0]),
            text_layout.lineCount(),
        )

    def test_emphasis_uses_cached_native_document_without_mutating_live_document(self):
        item = self._make_item(False, text='AA')
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setEmphasis('filled circle', 'over right')

        cursor = QTextCursor(item.document())
        cursor.select(QTextCursor.SelectionType.Document)
        outline = QTextCharFormat()
        outline.setTextOutline(QPen(QColor('#203060'), 8.0))
        outline.setProperty(
            glyph_rendering.GLYPH_DILATED_STROKE_FORMAT_PROPERTY,
            True,
        )
        cursor.mergeCharFormat(outline)
        item.layout.reLayout()

        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(
            NATIVE_DOCUMENT_CACHE.clear
        )
        live_document = item.document()
        live_revision = live_document.revision()
        live_text = live_document.toPlainText()
        live_html = to_rich_text_html(live_document)
        block = live_document.firstBlock()
        line = block.layout().lineAt(0)
        context = QAbstractTextDocumentLayout.PaintContext()
        image = QImage(
            700,
            300,
            QImage.Format.Format_ARGB32_Premultiplied,
        )
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        calls = []
        native_draw = QTextDocument.drawContents

        def record_draw(document, *args):
            calls.append(document)
            return native_draw(document, *args)

        try:
            with patch.object(
                QTextDocument,
                'drawContents',
                new=record_draw,
            ), patch.object(
                glyph_rendering,
                '_draw_dilated_path_stroke',
            ) as dilated_stroke:
                draw_emphasis_marks(
                    painter,
                    block,
                    line,
                    context,
                    vertical=False,
                )
        finally:
            painter.end()

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0], calls[1])
        self.assertIsNot(calls[0], live_document)
        self.assertEqual(len(NATIVE_DOCUMENT_CACHE), 1)
        cached_document = next(
            iter(NATIVE_DOCUMENT_CACHE.values())
        ).document
        self.assertIsNot(cached_document, live_document)
        self.assertEqual(cached_document.toPlainText(), '●')
        self.assertEqual(cached_document.documentMargin(), 0.0)
        self.assertEqual(
            cached_document.firstBlock().layout().lineAt(0).position(),
            QPointF(),
        )
        dilated_stroke.assert_not_called()
        self.assertEqual(live_document.revision(), live_revision)
        self.assertEqual(live_document.toPlainText(), live_text)
        self.assertEqual(to_rich_text_html(live_document), live_html)

    def test_emphasis_document_cache_derives_only_native_mark_paint(self):
        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(
            NATIVE_DOCUMENT_CACHE.clear
        )
        font = QFont('DejaVu Sans')
        font.setPointSizeF(40.0)
        font.setBold(True)
        font.setItalic(True)
        font.setUnderline(True)
        font.setOverline(True)
        font.setStrikeOut(True)
        gradient = QLinearGradient(0.0, 0.0, 100.0, 0.0)
        gradient.setColorAt(0.0, QColor('#e03020'))
        gradient.setColorAt(1.0, QColor('#2040e0'))
        source = QTextCharFormat()
        source.setFont(font)
        source.setForeground(QBrush(gradient))
        source.setBackground(QColor('#40ff80'))
        source.setTextOutline(QPen(QColor('#102030'), 12.0))
        source.setProperty(
            AnnotationProperty.EMPHASIS_STYLE,
            'filled circle',
        )
        source.setProperty(
            AnnotationProperty.EMPHASIS_POSITION,
            'over right',
        )
        source.setProperty(
            glyph_rendering.GLYPH_DILATED_STROKE_FORMAT_PROPERTY,
            True,
        )

        first = emphasis_rendering._mark_document(
            'filled circle', source
        )
        identical = emphasis_rendering._mark_document(
            'filled circle', source
        )
        cursor = QTextCursor(first.document)
        cursor.setPosition(0)
        cursor.movePosition(
            QTextCursor.MoveOperation.NextCharacter,
            QTextCursor.MoveMode.KeepAnchor,
        )
        derived = cursor.charFormat()

        self.assertIs(first, identical)
        self.assertAlmostEqual(derived.fontPointSize(), 20.0)
        self.assertTrue(derived.font().bold())
        self.assertTrue(derived.font().italic())
        self.assertFalse(derived.font().underline())
        self.assertFalse(derived.font().overline())
        self.assertFalse(derived.font().strikeOut())
        self.assertEqual(derived.foreground(), source.foreground())
        self.assertEqual(
            derived.background().style(),
            Qt.BrushStyle.NoBrush,
        )
        self.assertEqual(derived.textOutline().color(), QColor('#102030'))
        self.assertAlmostEqual(derived.textOutline().widthF(), 6.0)
        for annotation_property in AnnotationProperty:
            self.assertFalse(
                derived.hasProperty(int(annotation_property)),
            )
        self.assertFalse(
            derived.hasProperty(
                glyph_rendering.GLYPH_DILATED_STROKE_FORMAT_PROPERTY
            )
        )
        tag_type = getattr(QFont, 'Tag', None)
        if tag_type is not None and hasattr(derived.font(), 'featureValue'):
            ruby_tag = tag_type.fromString('ruby')
            self.assertEqual(derived.font().featureValue(ruby_tag), 1)
        self.assertFalse(first.glyph_bounds.isEmpty())
        self.assertTrue(first.ink_bounds.contains(first.glyph_bounds))
        self.assertGreater(first.ink_bounds.width(), first.glyph_bounds.width())

        ignored = QTextCharFormat(source)
        ignored.setBackground(QColor('#ff00ff'))
        ignored.setFontUnderline(False)
        self.assertIs(
            emphasis_rendering._mark_document('filled circle', ignored),
            first,
        )
        recolored = QTextCharFormat(source)
        recolored.setForeground(QColor('#abcdef'))
        self.assertIsNot(
            emphasis_rendering._mark_document('filled circle', recolored),
            first,
        )
        resized_outline = QTextCharFormat(source)
        resized_outline.setTextOutline(QPen(QColor('#102030'), 14.0))
        self.assertIsNot(
            emphasis_rendering._mark_document(
                'filled circle', resized_outline
            ),
            first,
        )
        self.assertIsNot(
            emphasis_rendering._mark_document('open circle', source),
            first,
        )

        NATIVE_DOCUMENT_CACHE.clear()
        oldest = None
        for index in range(
            NATIVE_DOCUMENT_CACHE_MAX_ENTRIES + 1
        ):
            varied = QTextCharFormat(source)
            varied.setFontPointSize(10.0 + index)
            entry = emphasis_rendering._mark_document(
                'filled circle', varied
            )
            if index == 0:
                oldest = entry
        self.assertEqual(
            len(NATIVE_DOCUMENT_CACHE),
            NATIVE_DOCUMENT_CACHE_MAX_ENTRIES,
        )
        self.assertNotIn(
            oldest,
            NATIVE_DOCUMENT_CACHE.values(),
        )

    def test_emphasis_native_document_preserves_gradient_and_opacity(self):
        item = self._make_item(False, text='A     A')
        item.fontformat.gradient_start_color = [230, 30, 20]
        item.fontformat.gradient_end_color = [20, 40, 230]
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setEmphasis('filled circle', 'over right')
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setGradientEnabled(True)

        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(
            NATIVE_DOCUMENT_CACHE.clear
        )
        block = item.document().firstBlock()
        line = block.layout().lineAt(0)
        context = QAbstractTextDocumentLayout.PaintContext()
        marks = tuple(
            emphasis_rendering._iter_emphasis_marks(
                block,
                line,
                vertical=False,
                context=context,
            )
        )
        self.assertEqual(len(marks), 2)
        cached_foreground = _format_at(
            marks[0].source.document, 0
        ).foreground()

        def render(opacity: float) -> QImage:
            image = QImage(
                800,
                300,
                QImage.Format.Format_ARGB32_Premultiplied,
            )
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            try:
                painter.translate(80.0, 120.0)
                painter.setOpacity(opacity)
                draw_emphasis_marks(
                    painter,
                    block,
                    line,
                    context,
                    vertical=False,
                )
            finally:
                painter.end()
            return image

        opaque = render(1.0)
        translucent = render(0.4)
        self.assertEqual(
            _format_at(marks[0].source.document, 0).foreground(),
            cached_foreground,
        )

        def channel_means(image: QImage, mark) -> tuple[float, float]:
            bounds = mark.ink_bounds.translated(80.0, 120.0).toAlignedRect()
            colors = []
            for y in range(
                max(0, bounds.top()),
                min(image.height(), bounds.bottom() + 1),
            ):
                for x in range(
                    max(0, bounds.left()),
                    min(image.width(), bounds.right() + 1),
                ):
                    color = image.pixelColor(x, y)
                    if color.alpha() > 64:
                        colors.append(color)
            self.assertTrue(colors)
            return (
                sum(color.red() for color in colors) / len(colors),
                sum(color.blue() for color in colors) / len(colors),
            )

        left_red, left_blue = channel_means(opaque, marks[0])
        right_red, right_blue = channel_means(opaque, marks[1])
        self.assertGreater(left_red, left_blue)
        self.assertGreater(left_red, right_red)
        self.assertGreater(right_blue, left_blue)
        self.assertLess(
            max(
                translucent.pixelColor(x, y).alpha()
                for y in range(translucent.height())
                for x in range(translucent.width())
            ),
            max(
                opaque.pixelColor(x, y).alpha()
                for y in range(opaque.height())
                for x in range(opaque.width())
            ),
        )

    def test_native_emphasis_ink_uses_shared_horizontal_and_vertical_placement(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._make_item(vertical, text='A')
                item.startEdit()
                cursor = item.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                item.setTextCursor(cursor)
                item.setEmphasis('open circle', 'under left')
                block = item.document().firstBlock()
                if vertical:
                    line, offset, orientation = (
                        item.layout.vertical_line_placement(block, 0)
                    )
                else:
                    line = block.layout().lineAt(0)
                    offset = QPointF()
                    orientation = None
                kwargs = {
                    'vertical': vertical,
                    'offset': offset,
                }
                if orientation is not None:
                    kwargs['orientation'] = orientation
                marks = tuple(
                    emphasis_rendering._iter_emphasis_marks(
                        block,
                        line,
                        **kwargs,
                    )
                )
                self.assertEqual(len(marks), 1)
                cell = glyph_rendering.logical_span_rect(
                    line,
                    0,
                    1,
                    offset,
                    orientation or QTransform(),
                )
                if vertical:
                    self.assertLess(marks[0].ink_bounds.right(), cell.left())
                else:
                    self.assertGreater(marks[0].ink_bounds.top(), cell.bottom())
                self.assertEqual(
                    emphasis_ink_bounds(block, line, **kwargs),
                    marks[0].ink_bounds,
                )
                self.assertTrue(
                    item.boundingRect().adjusted(
                        -0.02, -0.02, 0.02, 0.02
                    ).contains(marks[0].ink_bounds)
                )

    def test_emphasis_adds_css_like_line_and_column_leading(self):
        horizontal_plain = self._make_item(False)
        horizontal_marked = self._make_item(False)
        vertical_plain = self._make_item(True)
        vertical_marked = self._make_item(True)

        for item in (horizontal_marked, vertical_marked):
            cursor = item.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            item.setTextCursor(cursor)
            item.startEdit()
            item.setEmphasis('filled sesame', 'over right')

        plain_y = horizontal_plain.document().firstBlock().layout().lineAt(0).y()
        marked_y = horizontal_marked.document().firstBlock().layout().lineAt(0).y()
        self.assertGreater(marked_y, plain_y)
        self.assertGreater(
            vertical_marked.layout.shrink_width,
            vertical_plain.layout.shrink_width,
        )

    def test_effect_outline_does_not_reflow_emphasis_layout(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                item = self._make_item(vertical, text='A強調B')
                item.startEdit()
                cursor = item.textCursor()
                cursor.setPosition(1)
                cursor.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
                item.setTextCursor(cursor)
                item.setEmphasis('filled dot', 'over right')

                text_layout = item.document().firstBlock().layout()
                before = tuple(
                    text_layout.lineAt(index).position()
                    for index in range(text_layout.lineCount())
                )

                # The neutral effect path adds this outline to its temporary
                # clone. It must affect ink only, never line placement.
                cursor = QTextCursor(item.document())
                cursor.select(QTextCursor.SelectionType.Document)
                outline = QTextCharFormat()
                outline.setTextOutline(QPen(QColor('black'), 12.0))
                cursor.mergeCharFormat(outline)

                text_layout = item.document().firstBlock().layout()
                after = tuple(
                    text_layout.lineAt(index).position()
                    for index in range(text_layout.lineCount())
                )
                self.assertEqual(len(after), len(before))
                for actual, expected in zip(after, before):
                    self.assertAlmostEqual(actual.x(), expected.x())
                    self.assertAlmostEqual(actual.y(), expected.y())

    def test_effect_stroke_scales_native_emphasis_outline_with_mark_font(self):
        block = TextBlock([0, 0, 240, 160])
        block._bounding_rect = [0, 0, 240, 160]
        block.translation = 'A'
        block.fontformat.font_size = 40.0
        block.fontformat.stroke_width = 0.25
        item = TextBlkItem(block, 0)
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setEmphasis('open circle', 'over right')
        item.endEdit(keep_focus=False)

        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(
            NATIVE_DOCUMENT_CACHE.clear
        )
        item.repaint_background()

        outlined = [
            _format_at(entry.document, 0)
            for entry in NATIVE_DOCUMENT_CACHE.values()
            if _format_at(entry.document, 0).textOutline().style()
            != Qt.PenStyle.NoPen
            and _format_at(entry.document, 0).textOutline().color().alpha()
            > 0
        ]
        self.assertTrue(outlined)
        expected_width = (
            pt2px(_format_at(item.document(), 0).fontPointSize())
            * item.fontformat.stroke_width
            * emphasis_rendering.EMPHASIS_FONT_SCALE
        )
        for mark_format in outlined:
            self.assertAlmostEqual(
                mark_format.textOutline().widthF(),
                expected_width,
            )
            self.assertFalse(
                mark_format.hasProperty(
                    glyph_rendering.GLYPH_DILATED_STROKE_FORMAT_PROPERTY
                )
            )

    def test_vertical_glyph_slant_stroke_keeps_emphasis_out_of_dilation(self):
        block = TextBlock([0, 0, 180, 220])
        block._bounding_rect = [0, 0, 180, 220]
        block.translation = 'A'
        block.vertical = True
        block.fontformat.font_size = 48.0
        block.fontformat.stroke_width = 0.2
        block.fontformat.text_transform = TextTransformStack((), 11.0)
        item = TextBlkItem(block, 0)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.addCleanup(scene.removeItem, item)
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setEmphasis('open circle', 'over right')
        item.endEdit(keep_focus=False)
        self.app.processEvents()

        document_block = item.document().firstBlock()
        line, offset, orientation = item.layout.vertical_line_placement(
            document_block, 0
        )
        normal_mark = next(
            emphasis_rendering._iter_emphasis_marks(
                document_block,
                line,
                vertical=True,
                offset=offset,
                orientation=orientation,
            )
        )
        normal_bounds = QRectF(normal_mark.ink_bounds)
        surface_rect = QRectF(item.boundingRect())
        NATIVE_DOCUMENT_CACHE.clear()
        self.addCleanup(
            NATIVE_DOCUMENT_CACHE.clear
        )

        mask_flags = []
        dilation_inputs = []
        native_documents = []
        draw_mask = item.geometry_controller.draw_layout_selection_mask
        dilate = effect_rendering.cv2.dilate
        native_draw = QTextDocument.drawContents

        def record_mask(painter, context, *, include_annotations=True):
            mask_flags.append(include_annotations)
            return draw_mask(
                painter,
                context,
                include_annotations=include_annotations,
            )

        def record_dilate(source, kernel, *args, **kwargs):
            dilation_inputs.append(source.copy())
            return dilate(source, kernel, *args, **kwargs)

        def record_document(document, *args):
            native_documents.append(document)
            return native_draw(document, *args)

        with patch.object(
            item.geometry_controller,
            'draw_layout_selection_mask',
            new=record_mask,
        ), patch.object(
            effect_rendering.cv2,
            'dilate',
            new=record_dilate,
        ), patch.object(
            QTextDocument,
            'drawContents',
            new=record_document,
        ):
            item.repaint_background()

        def alpha_region(alpha, bounds: QRectF, padding: float = 0.0):
            local = QRectF(bounds).translated(-surface_rect.topLeft())
            local.adjust(-padding, -padding, padding, padding)
            pixels = local.toAlignedRect()
            left = max(0, pixels.left())
            top = max(0, pixels.top())
            right = min(alpha.shape[1], pixels.right() + 1)
            bottom = min(alpha.shape[0], pixels.bottom() + 1)
            return alpha[top:bottom, left:right]

        self.assertTrue(mask_flags)
        self.assertFalse(any(mask_flags))
        self.assertTrue(dilation_inputs)
        self.assertTrue(all(alpha.any() for alpha in dilation_inputs))
        self.assertTrue(
            all(
                not alpha_region(alpha, normal_bounds).any()
                for alpha in dilation_inputs
            )
        )

        expected_width = (
            pt2px(_format_at(item.document(), 0).fontPointSize())
            * item.fontformat.stroke_width
            * emphasis_rendering.EMPHASIS_FONT_SCALE
        )
        outlined_documents = []
        for entry in NATIVE_DOCUMENT_CACHE.values():
            mark_format = _format_at(entry.document, 0)
            if (
                mark_format.textOutline().style() == Qt.PenStyle.NoPen
                or mark_format.textOutline().color().alpha() == 0
            ):
                continue
            outlined_documents.append(entry.document)
            self.assertAlmostEqual(
                mark_format.textOutline().widthF(), expected_width
            )
        self.assertTrue(outlined_documents)
        self.assertTrue(
            any(
                painted is outlined
                for painted in native_documents
                for outlined in outlined_documents
            )
        )

        stroke_context = item.effect_renderer._stroke_paint_context()
        outlined_mark = next(
            emphasis_rendering._iter_emphasis_marks(
                document_block,
                line,
                vertical=True,
                context=stroke_context,
                offset=offset,
                orientation=orientation,
            )
        )
        self.assertAlmostEqual(
            outlined_mark.ink_bounds.width() - normal_bounds.width(),
            expected_width,
        )
        self.assertTrue(
            item.boundingRect().adjusted(
                -0.02, -0.02, 0.02, 0.02
            ).contains(outlined_mark.ink_bounds)
        )
        final_alpha = pixmap2ndarray(
            item.effect_renderer.background_pixmap,
            keep_alpha=True,
        )[..., 3]
        mark_region = alpha_region(
            final_alpha, outlined_mark.ink_bounds, padding=4.0
        )
        self.assertTrue(mark_region.any())
        local_ink = QRectF(outlined_mark.ink_bounds).translated(
            -surface_rect.topLeft()
        )
        self.assertGreater(local_ink.left(), 0.0)
        self.assertGreater(local_ink.top(), 0.0)
        self.assertLess(local_ink.right(), final_alpha.shape[1])
        self.assertLess(local_ink.bottom(), final_alpha.shape[0])

    def test_wrapped_vertical_columns_keep_mark_space_and_render(self):
        text = '強調文字列強調文字列'
        plain = self._make_item(True, bounds=(0, 0, 180, 90), text=text)
        marked = self._make_item(True, bounds=(0, 0, 180, 90), text=text)
        marked.startEdit()
        cursor = marked.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        marked.setTextCursor(cursor)
        marked.setEmphasis('filled sesame', 'over right')

        def column_positions(item: TextBlkItem):
            layout = item.document().firstBlock().layout()
            return sorted(
                {
                    round(layout.lineAt(index).x(), 3)
                    for index in range(layout.lineCount())
                }
            )

        plain_columns = column_positions(plain)
        marked_columns = column_positions(marked)
        self.assertGreaterEqual(len(marked_columns), 2)
        self.assertGreater(
            marked_columns[1] - marked_columns[0],
            plain_columns[1] - plain_columns[0],
        )

        image = QImage(400, 200, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        try:
            marked.document().drawContents(painter)
        finally:
            painter.end()
        self.assertFalse(image.isNull())

    def test_document_undo_redo_restores_emphasis(self):
        item = self._make_item(False)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setEmphasis('filled dot', 'over right')

        item.document().undo()
        self.assertEqual(emphasis_values(_format_at(item.document(), 0))[0], 'none')
        item.document().redo()
        self.assertEqual(
            emphasis_values(_format_at(item.document(), 0))[0],
            'filled dot',
        )

    def test_document_undo_redo_restores_text_combine(self):
        item = self._make_item(True, text='12')
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)
        item.setTateChuYoko(True)

        item.document().undo()
        self.assertEqual(
            text_combine_upright_values(_format_at(item.document(), 0))[0],
            'none',
        )
        item.document().redo()
        self.assertEqual(
            text_combine_upright_values(_format_at(item.document(), 0))[0],
            'all',
        )

    def test_document_undo_redo_restores_letter_spacing(self):
        item = self._make_item(False, text='ABC')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(1)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLetterSpacing(1.8)

        item.document().undo()
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.15,
        )
        item.document().redo()
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1)),
            1.8,
        )

    def test_document_undo_redo_restores_ligature_axis(self):
        item = self._make_item(False, text='fiX')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)
        item.setLigatureAxis(LIGATURE_COMMON, LIGATURE_DISABLED)

        item.document().undo()
        self.assertEqual(
            ligature_axis_value(
                _format_at(item.document(), 0), LIGATURE_COMMON
            ),
            LIGATURE_DEFAULT,
        )
        item.document().redo()
        self.assertEqual(
            ligature_axis_value(
                _format_at(item.document(), 0), LIGATURE_COMMON
            ),
            LIGATURE_DISABLED,
        )

    def test_document_undo_redo_restores_line_spacing_pair(self):
        item = self._make_item(False, text='A\nB')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(2)
        item.setTextCursor(cursor)
        item._set_line_spacing_pair(0.8, LineSpacingType.Distance)

        item.document().undo()
        self.assertEqual(
            _line_spacing_at(item.document(), 1),
            (1.2, LineSpacingType.Proportional),
        )
        item.document().redo()
        self.assertEqual(
            _line_spacing_at(item.document(), 1),
            (0.8, LineSpacingType.Distance),
        )

    def test_backward_selection_uses_selected_format_and_keeps_direction(self):
        item = self._make_item(False, text='A——B')
        item.startEdit()

        selected = item.textCursor()
        selected.setPosition(1)
        selected.setPosition(3, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(selected)
        item.setLetterSpacing(1.4)

        backward = item.textCursor()
        backward.setPosition(3)
        backward.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(backward)

        self.assertEqual(item.letter_spacing_value(), 1.4)
        item.setLetterSpacing(1.8)
        item.setFontItalic(
            True,
            set_selected=True,
            restore_cursor=True,
        )

        restored = item.textCursor()
        self.assertEqual((restored.position(), restored.anchor()), (1, 3))
        self.assertEqual(item.letter_spacing_value(), 1.8)
        self.assertTrue(item.get_fontformat().italic)
        self.assertEqual(
            letter_spacing_value(_format_at(item.document(), 1, 2)),
            1.8,
        )

    def test_emphasis_button_toggles_and_applies_selected_mark(self):
        button = EmphasisToolButton()
        edits = []
        button.emphasis_changed.connect(
            lambda style, position: edits.append((style, position))
        )
        button._position_actions['under left'].trigger()
        self.assertEqual(edits, [])
        button.set_values('open circle', 'under left')

        button.click()
        button.click()
        button._style_actions['filled sesame'].trigger()

        self.assertEqual(
            edits,
            [
                ('none', 'under left'),
                ('open circle', 'under left'),
                ('filled sesame', 'under left'),
            ],
        )

    def test_emphasis_button_undo_redo_uses_one_command_per_edit(self):
        item = self._make_item(False, text='AB')
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        panel.global_format = FontFormat()
        panel.textblk_item = item
        panel.set_active_format(item.get_fontformat())

        stack = QUndoStack()
        pushed_steps = []

        def push_history(num_steps: int, is_formatting: bool) -> None:
            pushed_steps.append((num_steps, is_formatting))
            stack.push(TextItemEditCommand(item, None, num_steps, panel))

        item.push_undo_stack.connect(push_history)
        self.addCleanup(item.push_undo_stack.disconnect, push_history)
        button = panel.formatBtnGroup.emphasisBtn
        with patch(
            'ballontranslator.ui.text_engine.formatting.panel.'
            'restore_canvas_view_focus'
        ):
            button._style_actions['filled circle'].trigger()
            button._position_actions['under left'].trigger()
            button.click()

        self.assertEqual(len(pushed_steps), 3)
        self.assertTrue(
            all(
                num_steps > 0 and is_formatting
                for num_steps, is_formatting in pushed_steps
            )
        )
        self.assertEqual(stack.count(), 3)
        self.assertEqual(item.emphasis_values(), ('none', 'under left'))

        stack.undo()
        self.assertEqual(
            item.emphasis_values(), ('filled circle', 'under left')
        )
        self.assertTrue(button.isChecked())
        stack.undo()
        self.assertEqual(
            item.emphasis_values(), ('filled circle', 'over right')
        )
        stack.undo()
        self.assertEqual(item.emphasis_values()[0], 'none')
        self.assertFalse(button.isChecked())
        self.assertEqual(stack.count(), 3)

        stack.redo()
        stack.redo()
        stack.redo()
        self.assertEqual(item.emphasis_values(), ('none', 'under left'))
        self.assertFalse(button.isChecked())
        self.assertEqual(len(pushed_steps), 3)

    def test_font_panel_tate_chu_yoko_switch_edits_selection(self):
        item = self._make_item(True, text='12')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        panel.global_format = FontFormat()
        panel.textblk_item = item
        panel.set_active_format(item.fontformat)

        with patch(
            'ballontranslator.ui.text_engine.formatting.panel.'
            'restore_canvas_view_focus'
        ):
            panel.tateChuYokoChecker.click()

        self.assertTrue(panel.tateChuYokoChecker.isChecked())
        self.assertTrue(item.tate_chu_yoko_enabled())

        panel.set_tate_chu_yoko_enabled(False)
        manager = SimpleNamespace(
            formatpanel=panel,
            sender=lambda: item,
        )
        SceneTextManager.on_inline_format_changed(manager)

        self.assertTrue(panel.tateChuYokoChecker.isChecked())

    def test_advanced_font_feature_edits_only_inline_format(self):
        item = self._make_item(False, text='stX')
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        panel.global_format = FontFormat()
        panel.textblk_item = item
        panel.set_active_format(item.fontformat)
        propagated = []
        item.propagate_user_edited.connect(
            lambda *args: propagated.append(args)
        )
        stack = QUndoStack()
        pushed_steps = []

        def push_history(num_steps: int, is_formatting: bool) -> None:
            pushed_steps.append((num_steps, is_formatting))
            stack.push(TextItemEditCommand(item, None, num_steps, panel))

        item.push_undo_stack.connect(push_history)
        self.addCleanup(item.push_undo_stack.disconnect, push_history)

        expected_axes = {LIGATURE_COMMON}
        if FONT_FEATURES_AVAILABLE:
            expected_axes.update((
                LIGATURE_DISCRETIONARY,
                OLDSTYLE_NUMS,
                LIGATURE_CONTEXTUAL,
            ))
        self.assertEqual(
            set(panel.textadvancedfmt_panel.ligature_comboboxes),
            expected_axes,
        )
        axis = (
            OLDSTYLE_NUMS
            if FONT_FEATURES_AVAILABLE
            else LIGATURE_COMMON
        )
        state = (
            LIGATURE_ENABLED
            if FONT_FEATURES_AVAILABLE
            else LIGATURE_DISABLED
        )
        combo = panel.textadvancedfmt_panel.ligature_comboboxes[axis]
        index = combo.findData(state)
        with patch(
            'ballontranslator.ui.text_engine.formatting.panel.'
            'restore_canvas_view_focus'
        ):
            combo.setCurrentIndex(index)
            combo.activated.emit(index)

        value_at = (
            oldstyle_nums_value
            if axis == OLDSTYLE_NUMS
            else lambda char_format: ligature_axis_value(
                char_format, axis
            )
        )

        self.assertEqual(
            value_at(_format_at(item.document(), 0, 2)),
            state,
        )
        self.assertEqual(
            value_at(_format_at(item.document(), 2)),
            LIGATURE_DEFAULT,
        )
        self.assertEqual(len(pushed_steps), 1)
        self.assertGreater(pushed_steps[0][0], 0)
        self.assertTrue(pushed_steps[0][1])
        stack.undo()
        self.assertEqual(
            value_at(_format_at(item.document(), 0)),
            LIGATURE_DEFAULT,
        )
        stack.redo()
        self.assertEqual(
            value_at(_format_at(item.document(), 0)),
            state,
        )
        self.assertEqual(propagated, [])

    def test_font_panel_tracks_the_active_inline_format(self):
        previous_active_format = C.active_format
        self.addCleanup(
            setattr, C, 'active_format', previous_active_format
        )
        item = self._make_item(False, text='AB')
        item.startEdit()
        modified = QTextCursor(item.document())
        modified.setPosition(0)
        modified.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        char_format = QTextCharFormat()
        char_format.setFontFamily('Courier New')
        char_format.setFontPointSize(18)
        char_format.setFontWeight(QFont.Weight.Black)
        char_format.setFontItalic(True)
        char_format.setFontUnderline(True)
        char_format.setForeground(QColor(12, 34, 56))
        modified.mergeCharFormat(char_format)
        apply_emphasis(modified, 'open circle', 'under left')
        apply_letter_spacing(modified, 1.4, vertical=False)
        apply_ligature_axis(
            modified,
            LIGATURE_COMMON,
            LIGATURE_ENABLED,
            vertical=False,
        )
        apply_ligature_axis(
            modified,
            LIGATURE_DISCRETIONARY,
            LIGATURE_ENABLED,
            vertical=False,
        )
        apply_ligature_axis(
            modified,
            LIGATURE_CONTEXTUAL,
            LIGATURE_DISABLED,
            vertical=False,
        )
        apply_oldstyle_nums(modified, LIGATURE_ENABLED)

        caret = QTextCursor(item.document())
        caret.setPosition(1)
        item.setTextCursor(caret)
        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        panel.global_format = FontFormat()
        panel.textblk_item = item
        panel.set_active_format(item.fontformat)
        manager = SimpleNamespace(
            formatpanel=panel,
            sender=lambda: item,
        )
        feedback = []
        for signal in (
            panel.familybox.param_changed,
            panel.fontWeightBox.param_changed,
            panel.fontsizebox.param_changed,
            panel.lineSpacingBox.param_changed,
            panel.letterSpacingBox.param_changed,
            panel.formatBtnGroup.param_changed,
            panel.formatBtnGroup.emphasisBtn.emphasis_changed,
            panel.textadvancedfmt_panel.ligature_axis_changed,
        ):
            signal.connect(lambda *_args: feedback.append(True))
        revision = item.document().revision()

        SceneTextManager.on_inline_format_changed(manager)

        self.assertEqual(feedback, [])
        self.assertEqual(item.document().revision(), revision)
        active = item.get_fontformat()
        self.assertEqual(panel.familybox.currentText(), active.font_family)
        self.assertEqual(panel.fontWeightBox.weight(), active.font_weight)
        self.assertAlmostEqual(
            float(panel.fontsizebox.getFontSize()), active.font_size
        )
        self.assertEqual(panel.colorPicker.rgb(), (12, 34, 56))
        self.assertTrue(panel.formatBtnGroup.italicBtn.isChecked())
        self.assertTrue(panel.formatBtnGroup.underlineBtn.isChecked())
        self.assertEqual(
            panel.formatBtnGroup.emphasisBtn.values(),
            ('open circle', 'under left'),
        )
        self.assertEqual(panel.letterSpacingBox.value(), 1.4)
        self.assertEqual(
            panel.textadvancedfmt_panel.ligature_comboboxes[
                LIGATURE_COMMON
            ].currentData(),
            LIGATURE_ENABLED,
        )
        if FONT_FEATURES_AVAILABLE:
            self.assertEqual(
                panel.textadvancedfmt_panel.ligature_comboboxes[
                    LIGATURE_DISCRETIONARY
                ].currentData(),
                LIGATURE_ENABLED,
            )
            self.assertEqual(
                panel.textadvancedfmt_panel.ligature_comboboxes[
                    OLDSTYLE_NUMS
                ].currentData(),
                LIGATURE_ENABLED,
            )
            self.assertEqual(
                panel.textadvancedfmt_panel.ligature_comboboxes[
                    LIGATURE_CONTEXTUAL
                ].currentData(),
                LIGATURE_DISABLED,
            )
        self.assertEqual(C.active_format.font_weight, active.font_weight)

        caret.setPosition(2)
        item.setTextCursor(caret)
        SceneTextManager.on_inline_format_changed(manager)

        normal = item.get_fontformat()
        self.assertEqual(panel.fontWeightBox.weight(), normal.font_weight)
        self.assertEqual(panel.colorPicker.rgb(), tuple(normal.frgb))
        self.assertFalse(panel.formatBtnGroup.italicBtn.isChecked())
        self.assertFalse(panel.formatBtnGroup.underlineBtn.isChecked())
        self.assertFalse(panel.formatBtnGroup.emphasisBtn.isChecked())
        self.assertTrue(all(
            combo.currentData() == LIGATURE_DEFAULT
            for combo in panel.textadvancedfmt_panel.ligature_comboboxes.values()
        ))


if __name__ == '__main__':
    unittest.main()
