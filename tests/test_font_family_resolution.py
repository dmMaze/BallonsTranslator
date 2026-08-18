import os
import unittest


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import QT6
from qtpy.QtGui import QFont, QFontDatabase, QRawFont, QTextDocument
from qtpy.QtWidgets import QApplication

from ballontranslator.ui.text_engine.font_family import (
    font_family_for_project,
    font_family_for_qt,
    html_uses_project_font_family,
    qfont_with_family,
    register_qt_font_family_aliases,
    restore_project_font_families_in_html,
)
from ballontranslator.ui.text_engine.annotations import (
    load_rich_text_html,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.ui.text_engine.pipeline_formatting import (
    _load_text_block_document,
)
from ballontranslator.utils.textblock import TextBlock


class FontFamilyResolutionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_internal_alias_round_trips_without_leaking_into_text(self):
        family = '[test-vendor]Synthetic Font'
        aliases = register_qt_font_family_aliases(
            [family],
            lambda _family: [],
        )
        alias = aliases[family]

        self.assertNotIn('[', alias)
        self.assertEqual(font_family_for_qt(family), alias)
        self.assertEqual(font_family_for_project(alias), family)

        html = (
            f'<span style="font-family:\'{alias}\';">'
            f'{alias}</span>'
        )
        restored = restore_project_font_families_in_html(html)
        self.assertIn(f"font-family:'{family}'", restored)
        self.assertIn(f'>{alias}</span>', restored)
        self.assertTrue(html_uses_project_font_family(restored))
        self.assertFalse(
            html_uses_project_font_family(
                f"<span style=\"font-family:'{alias}'\">text</span>"
            )
        )
        self.assertFalse(
            html_uses_project_font_family('<p>ordinary text</p>')
        )

    def test_valid_foundry_name_is_not_aliased(self):
        family = 'Synthetic Family [Foundry]'

        aliases = register_qt_font_family_aliases(
            [family],
            lambda _family: ['Regular'],
        )

        self.assertEqual(aliases, {})
        self.assertEqual(font_family_for_qt(family), family)

    def test_comma_family_remains_one_qt_family(self):
        family = 'Synthetic, Comma Family'

        font = qfont_with_family(QFont('Sans Serif', 18), family)

        if QT6:
            self.assertEqual(font.families(), [family])
        else:
            # Qt 5 exposes only the single-family accessor reliably.
            self.assertEqual(font.family(), family)

    def test_replacing_html_font_clears_the_old_qt_family_list(self):
        document = QTextDocument()
        document.setHtml(
            "<span style=\"font-family:'Inter'; font-size:22pt; "
            'font-style:italic;\">text</span>'
        )
        source = document.firstBlock().begin().fragment().charFormat().font()

        font = qfont_with_family(source, 'DejaVu Sans')

        self.assertEqual(font.family(), 'DejaVu Sans')
        self.assertEqual(font.families(), ['DejaVu Sans'])
        self.assertEqual(font.pointSizeF(), 22)
        self.assertTrue(font.italic())

    def test_buding_uses_real_face_through_horizontal_vertical_switch(self):
        family = '[toolbox]BuDing-JF'
        database = QFontDatabase if QT6 else QFontDatabase()
        if family not in database.families():
            self.skipTest('BuDing-JF is not installed')
        if database.styles(family):
            self.skipTest('this Qt backend selects bracketed families directly')

        alias = register_qt_font_family_aliases(
            database.families(),
            database.styles,
        )[family]
        bad_raw_font = QRawFont.fromFont(QFont(family, 32))
        expected_raw_font = QRawFont.fromFont(
            qfont_with_family(QFont('Sans Serif', 32), family)
        )
        bad_signature = (
            bad_raw_font.styleName(),
            bad_raw_font.unitsPerEm(),
            tuple(bad_raw_font.glyphIndexesForString('木A')),
        )
        expected_signature = (
            expected_raw_font.styleName(),
            expected_raw_font.unitsPerEm(),
            tuple(expected_raw_font.glyphIndexesForString('木A')),
        )
        self.assertNotEqual(bad_signature, expected_signature)
        self.assertEqual(expected_raw_font.unitsPerEm(), 1000)

        bounds = [0, 0, 220, 300]
        block = TextBlock(bounds)
        block._bounding_rect = list(bounds)
        block.translation = '测试木A，横排转竖排。'
        block.fontformat.font_family = family
        block.fontformat.font_size = 32
        item = TextBlkItem(block, 0)
        item.startEdit()
        item.setVertical(True)

        char_font = item.document().firstBlock().begin().fragment().charFormat().font()
        actual_raw_font = QRawFont.fromFont(char_font)
        actual_signature = (
            actual_raw_font.styleName(),
            actual_raw_font.unitsPerEm(),
            tuple(actual_raw_font.glyphIndexesForString('木A')),
        )
        self.assertEqual(char_font.family(), alias)
        self.assertEqual(actual_signature, expected_signature)
        self.assertEqual(item.get_fontformat().font_family, family)

        pipeline_document = _load_text_block_document(block)
        pipeline_font = (
            pipeline_document.firstBlock().begin().fragment().charFormat().font()
        )
        self.assertEqual(pipeline_font.family(), alias)
        self.assertEqual(
            QRawFont.fromFont(pipeline_font).unitsPerEm(),
            expected_raw_font.unitsPerEm(),
        )

        document = QTextDocument()
        load_rich_text_html(
            document,
            f"<span style=\"font-family:'{family}';\">木A</span>",
        )
        rich_font = document.firstBlock().begin().fragment().charFormat().font()
        self.assertEqual(rich_font.family(), alias)
        exported = to_rich_text_html(document)
        self.assertIn(family, exported)
        self.assertNotIn(alias, exported)


if __name__ == '__main__':
    unittest.main()
