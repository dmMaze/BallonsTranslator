import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import QT6
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QFont, QFontDatabase, QPalette, QRawFont, QTextDocument
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui.text_engine.font_family import (
    font_family_for_project,
    font_family_for_qt,
    html_uses_project_font_family,
    normalize_document_font_families,
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
from ballontranslator.utils import shared
from ballontranslator.utils.font_registry import FontEntry, FontFace, FontRegistry
from ballontranslator.utils.fontformat import FontWeight, font_weight_to_qt


class FontFamilyResolutionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_picker_does_not_open_all_fonts_on_show_or_autocomplete(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [FontEntry(f'Family {i}', f'Family {i}', f'Family {i}', 'system')
                   for i in range(200)]
        registry = FontRegistry(system_entries=entries)
        registry._font_database = object()
        with patch.object(shared, 'FONT_REGISTRY', registry), patch(
            'ballontranslator.utils.font_registry._system_display_family',
            side_effect=lambda _db, family, _locale: 'Display ' + family,
        ) as lookup:
            combo = FontFamilyComboBox()
            try:
                combo.update_font_entries(entries)
                combo.show()
                self.app.processEvents()
                combo.showPopup()
                self.app.processEvents()
                # Rendering and completion must not eagerly open the collection.
                self.assertLess(lookup.call_count, len(entries) // 2)
                combo.hidePopup()
                combo.setFocus()
                combo.lineEdit().selectAll()
                QTest.keyClicks(combo.lineEdit(), 'Family')
                self.app.processEvents()
                self.assertTrue(combo.completer().popup().isVisible())
                self.assertLess(lookup.call_count, len(entries) // 2)
                combo.completer().popup().hide()
                combo.set_current_family('Family 199')
                self.assertEqual(combo.currentText(), 'Display Family 199')
                self.assertEqual(combo.current_storage_family(), 'Family 199')
            finally:
                combo.close()
                combo.deleteLater()

    def test_picker_filters_font_list_by_substring(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [
            FontEntry('Stored Sans', 'Alpha Sans', 'DejaVu Sans', 'custom'),
            FontEntry('Stored Serif', 'Beta Serif', 'DejaVu Serif', 'custom'),
            FontEntry('Stored Mono', 'Symbols [Mono]', 'DejaVu Sans Mono', 'custom'),
        ]
        registry = FontRegistry(custom_entries=entries)
        with patch.object(shared, 'FONT_REGISTRY', registry):
            combo = FontFamilyComboBox()
            try:
                combo.update_font_entries(entries)
                combo.show()
                combo.setFocus()
                self.app.processEvents()
                editor = combo.lineEdit()
                completer = combo.completer()
                changes = Mock()
                combo.param_changed.connect(changes)
                for query, expected in (
                    ('sAnS', ['Alpha Sans']),
                    ('[M', ['Symbols [Mono]']),
                    ('eri', ['Beta Serif']),
                ):
                    with self.subTest(query=query):
                        editor.selectAll()
                        QTest.keyClicks(editor, query)
                        self.app.processEvents()
                        matches = combo.view().model()
                        self.assertEqual([
                            matches.index(row, 0).data()
                            for row in range(matches.rowCount())
                        ], expected)
                        self.assertEqual(editor.text(), query)
                        changes.assert_not_called()
                        popup = completer.popup()
                        self.assertEqual(
                            popup is not None and popup.isVisible(), bool(expected),
                        )

                # The combo's own dropdown must show the filtered list too,
                # and accept the canonical family even at the same proxy row.
                completer.popup().hide()
                combo.showPopup()
                self.app.processEvents()
                self.assertEqual(combo.count(), 1)
                self.assertEqual(combo.itemText(0), 'Beta Serif')
                QTest.keyClick(combo.view(), Qt.Key.Key_Return)
                self.app.processEvents()
                self.assertEqual(combo.currentText(), 'Beta Serif')
                self.assertEqual(combo.current_storage_family(), 'Stored Serif')
                self.assertEqual(combo.count(), len(entries))
                self.assertFalse(combo.view().isVisible())
                changes.assert_called_once_with('font_family', 'Stored Serif')

                editor.selectAll()
                QTest.keyClicks(editor, 'Sans')
                self.assertEqual(combo.count(), 1)
                combo.update_font_entries(entries)
                self.assertEqual(combo.current_storage_family(), 'Stored Serif')
                self.assertEqual(combo.count(), len(entries))
            finally:
                combo.close()
                combo.deleteLater()

    def test_picker_completion_applies_family_only_once(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [
            FontEntry('Stored Sans', 'Alpha Sans', 'DejaVu Sans', 'custom'),
            FontEntry('Stored Serif', 'Beta Serif', 'DejaVu Serif', 'custom'),
        ]
        with patch.object(shared, 'FONT_REGISTRY', FontRegistry(custom_entries=entries)):
            combo = FontFamilyComboBox()
            try:
                combo.update_font_entries(entries)
                combo.set_current_family('Stored Sans')
                combo.show()
                combo.setFocus()
                self.app.processEvents()
                changes = Mock()
                combo.param_changed.connect(changes)
                editor = combo.lineEdit()
                editor.selectAll()
                QTest.keyClicks(editor, 'eri')
                popup = combo.completer().popup()
                QTest.keyClick(popup, Qt.Key.Key_Down)
                QTest.keyClick(popup, Qt.Key.Key_Return)
                self.app.processEvents()
                self.assertEqual(combo.currentText(), 'Beta Serif')
                changes.assert_called_once_with('font_family', 'Stored Serif')

                combo.clearFocus()
                self.app.processEvents()
                changes.assert_called_once_with('font_family', 'Stored Serif')

                # Choosing the displayed family again is still meaningful
                # when applying it to a mixed selection of text items.
                combo.showPopup()
                QTest.keyClick(combo.view(), Qt.Key.Key_Return)
                self.app.processEvents()
                self.assertEqual(changes.call_count, 2)
            finally:
                combo.close()
                combo.deleteLater()

    def test_picker_refresh_preserves_committed_family_during_search(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [
            FontEntry('Stored Sans', 'Alpha Sans', 'DejaVu Sans', 'custom'),
            FontEntry('Stored Serif', 'Beta Serif', 'DejaVu Serif', 'custom'),
        ]
        for committed, available in (
            ('Stored Sans', entries), ('Stored Sans', entries[1:]), ('', entries),
        ):
            with self.subTest(committed=committed, selected_font_present=len(available) == 2), patch.object(
                shared, 'FONT_REGISTRY', FontRegistry(custom_entries=entries),
            ):
                combo = FontFamilyComboBox()
                try:
                    combo.update_font_entries(entries)
                    combo.set_current_family(committed)
                    combo.show()
                    combo.setFocus()
                    self.app.processEvents()
                    changes = Mock()
                    combo.param_changed.connect(changes)
                    combo.lineEdit().selectAll()
                    QTest.keyClicks(combo.lineEdit(), 'Beta Serif')
                    self.assertTrue(combo.completer().popup().isVisible())
                    changes.assert_not_called()

                    shared.FONT_REGISTRY = FontRegistry(custom_entries=available)
                    combo.update_font_entries(available)
                    self.app.processEvents()
                    self.assertEqual(combo.current_storage_family(), committed)
                    self.assertFalse(combo.completer().popup().isVisible())
                    changes.assert_not_called()
                finally:
                    combo.close()
                    combo.deleteLater()

    def test_picker_completion_inherits_window_theme(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [FontEntry(name, name, name, 'custom') for name in (
            'DejaVu Sans', 'Liberation Sans',
        )]
        original_palette = self.app.palette()
        original_stylesheet = self.app.styleSheet()
        window = QWidget()
        with patch.object(shared, 'FONT_REGISTRY', FontRegistry(custom_entries=entries)):
            try:
                # Production themes MainWindow, not QApplication. A dark OS
                # palette must not leak into a light-themed completion popup.
                self.app.setStyleSheet('')
                palette = QPalette(original_palette)
                palette.setColor(QPalette.ColorRole.Base, QColor('black'))
                palette.setColor(QPalette.ColorRole.Window, QColor('black'))
                self.app.setPalette(palette)
                window.setStyleSheet('QWidget { background-color: #eceef5; color: #333333; }')
                combo = FontFamilyComboBox(window)
                combo.update_font_entries(entries)
                window.resize(400, 300)
                combo.resize(260, 30)
                window.show()
                self.app.processEvents()

                # Reuse the same popup through theme changes as well.
                for theme_index, background in enumerate(('#eceef5', '#262a30', '#eceef5')):
                    with self.subTest(background=background):
                        if theme_index:
                            window.setStyleSheet(
                                f'QWidget {{ background-color: {background}; color: #808080; }}'
                            )
                        combo.setFocus()
                        combo.lineEdit().selectAll()
                        QTest.keyClicks(combo.lineEdit(), 'Sans')
                        self.app.processEvents()
                        popup = combo.completer().popup()
                        self.assertTrue(popup.isVisible())
                        search_image = popup.viewport().grab().toImage()
                        popup.hide()
                        combo.showPopup()
                        self.app.processEvents()
                        dropdown_image = combo.view().viewport().grab().toImage()
                        # Sample empty row space, away from glyphs/selection.
                        for image in (search_image, dropdown_image):
                            self.assertEqual(
                                image.pixelColor(image.width() - 3, image.height() - 3),
                                QColor(background),
                            )
                        combo.hidePopup()
            finally:
                window.close()
                window.deleteLater()
                self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                self.app.setPalette(original_palette)
                self.app.setStyleSheet(original_stylesheet)

    def test_picker_completion_shows_full_rows_and_releases_delegate(self) -> None:
        from ballontranslator.ui.text_engine.formatting.panel import FontFamilyComboBox

        entries = [FontEntry(name, name, name, 'custom') for name in (
            'DejaVu Sans', 'Liberation Sans', 'Noto Sans',
        )]
        with patch.object(shared, 'FONT_REGISTRY', FontRegistry(custom_entries=entries)):
            combo = FontFamilyComboBox()
            combo.update_font_entries(entries)
            combo.move(100, 100)
            combo.show()
            combo.setFocus()
            self.app.processEvents()
            combo.lineEdit().selectAll()
            QTest.keyClicks(combo.lineEdit(), 'Sans')
            self.app.processEvents()
            popup = combo.completer().popup()
            last = popup.model().index(popup.model().rowCount() - 1, 0)
            self.assertLessEqual(
                popup.visualRect(last).bottom(), popup.viewport().rect().bottom(),
            )
            self.assertFalse(popup.verticalScrollBar().isVisible())
            combo_ref = weakref.ref(combo)
            delegate_ref = weakref.ref(popup.itemDelegate())
            combo.close()
            combo.deleteLater()
            del combo, popup
            self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            gc.collect()
            self.assertIsNone(combo_ref())
            self.assertIsNone(delegate_ref())

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

    def test_html_family_precheck_uses_indexed_css_names(self):
        class MembershipOnlyDict(dict):
            def __iter__(self):
                raise AssertionError('registry keys must not be scanned')

        registry = SimpleNamespace(
            entries_by_key=MembershipOnlyDict({
                'a & b, display': object(),
            })
        )
        html = (
            "<span style=\"font-family:'A &amp; B, Display', serif\">"
            'text</span>'
        )

        with patch.object(shared, 'FONT_REGISTRY', registry):
            self.assertTrue(html_uses_project_font_family(html))
            self.assertFalse(html_uses_project_font_family(
                "<span style=\"font-family:'Unknown'\">text</span>"
            ))

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

    def test_alias_normalization_handles_multiple_rich_text_fragments(self):
        family = '[test-normalize]Synthetic Font'
        alias = register_qt_font_family_aliases(
            [family], lambda _family: []
        )[family]
        document = QTextDocument()
        document.setHtml(
            f"<span style=\"font-family:'{family}'; color:#ff0000\">A</span>"
            f"<span style=\"font-family:'{family}'; color:#00ff00\">B</span>"
            f"<span style=\"font-family:'{family}'; color:#0000ff\">C</span>"
        )

        replacements = normalize_document_font_families(document)

        families = []
        iterator = document.firstBlock().begin()
        while not iterator.atEnd():
            families.append(iterator.fragment().charFormat().font().family())
            iterator += 1
        self.assertEqual(replacements, 3)
        self.assertEqual(families, [alias, alias, alias])

    def test_registry_resolution_does_not_pin_weight_or_italic_face(self):
        database = QFontDatabase if QT6 else QFontDatabase()
        family = 'DejaVu Sans'
        if family not in database.families():
            self.skipTest(f'{family} is not installed')
        entry = FontEntry(
            family,
            family,
            family,
            'system',
            weights=[400, 700],
            faces=[
                FontFace(family, family, family, 'Book', 400),
                FontFace(family, family, family, 'Bold', 700),
            ],
        )
        font = QFont(family, 18)
        font.setWeight(
            QFont.Weight(font_weight_to_qt(FontWeight.Bold, qt6=QT6))
        )
        font.setItalic(True)

        with patch.object(
            shared,
            'FONT_REGISTRY',
            FontRegistry(system_entries=[entry]),
        ):
            resolved = qfont_with_family(font, family)

        style_name = QRawFont.fromFont(resolved).styleName().casefold()
        self.assertIn('bold', style_name)
        self.assertTrue(
            'italic' in style_name or 'oblique' in style_name,
            style_name,
        )

    def test_html_exports_real_face_names_instead_of_picker_groups(self):
        faces = [
            FontFace(
                'Example Light', 'Example Light', 'Qt Example Light',
                'Light', 300,
            ),
            FontFace(
                'Example Bold', 'Example Bold', 'Qt Example Bold',
                'Bold', 700,
            ),
        ]
        entry = FontEntry(
            'Example',
            'Example',
            'Qt Example Light',
            'custom',
            faces=faces,
            weights=[300, 700],
            is_pseudo_group=True,
        )
        html = "<span style=\"font-family:'Qt Example Bold'\">x</span>"

        with patch.object(
            shared,
            'FONT_REGISTRY',
            FontRegistry(custom_entries=[entry]),
        ):
            restored = restore_project_font_families_in_html(html)

        self.assertIn("font-family:'Example Bold'", restored)
        self.assertNotIn("font-family:'Example'", restored)

    def test_html_exports_canonical_system_alias_name(self):
        face = FontFace(
            '바탕', '바탕', '바탕', 'Regular', 400, aliases={'Batang'}
        )
        entry = FontEntry(
            'Batang', '바탕', '바탕', 'system',
            faces=[face], weights=[400], aliases={'Batang', '바탕'},
            alias_source='optional-table',
        )
        html = "<span style=\"font-family:'바탕'\">x</span>"

        with patch.object(
            shared,
            'FONT_REGISTRY',
            FontRegistry(system_entries=[entry]),
        ):
            restored = restore_project_font_families_in_html(html)

        self.assertIn("font-family:'Batang'", restored)
        self.assertNotIn("font-family:'바탕'", restored)

    def test_html_export_uses_index_and_preserves_entity_escaping(self):
        face = FontFace(
            'Canonical & Name', 'Canonical & Name', 'A & B, Display',
            'Regular', 400,
        )
        registry = FontRegistry(custom_entries=[FontEntry(
            'Canonical & Name', 'Canonical & Name', 'A & B, Display',
            'custom', faces=[face], weights=[400],
        )])
        registry.entries = lambda *_args: (_ for _ in ()).throw(
            AssertionError('registry entries must not be scanned')
        )
        html = (
            "<span style=\"font-family:'A &amp; B, Display', serif\">"
            'x</span>'
        )

        with patch.object(shared, 'FONT_REGISTRY', registry):
            restored = restore_project_font_families_in_html(html)

        self.assertIn(
            "font-family:'Canonical &amp; Name', serif", restored
        )

    def test_internal_qt_alias_exports_registry_canonical_name(self):
        qt_family = '[localized-vendor]Synthetic Font'
        internal = register_qt_font_family_aliases(
            [qt_family], lambda _family: []
        )[qt_family]
        face = FontFace(
            'Canonical Synthetic Font',
            qt_family,
            qt_family,
            'Regular',
            400,
        )
        entry = FontEntry(
            'Canonical Synthetic Font',
            qt_family,
            qt_family,
            'custom',
            faces=[face],
            weights=[400],
        )
        html = f"<span style=\"font-family:'{internal}'\">x</span>"

        with patch.object(
            shared,
            'FONT_REGISTRY',
            FontRegistry(custom_entries=[entry]),
        ):
            restored = restore_project_font_families_in_html(html)

        self.assertIn(
            "font-family:'Canonical Synthetic Font'", restored
        )
        self.assertNotIn(internal, restored)

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
