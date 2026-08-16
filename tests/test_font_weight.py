import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy import QT6
from qtpy.QtGui import QFont, QTextCharFormat, QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication

from ballontranslator.ui import shared_widget as SW
from ballontranslator.ui.text_engine.annotations import (
    load_rich_text_html,
    to_rich_text_html,
)
from ballontranslator.ui.text_engine.formatting.panel import (
    FontFamilyComboBox,
    FontFormatPanel,
    FontWeightComboBox,
    _split_weight_family_name,
    _weight_family_aliases,
)
from ballontranslator.ui.text_engine.item import TextBlkItem
from ballontranslator.utils import config as C
from ballontranslator.utils import shared
from ballontranslator.utils.fontformat import (
    FontFormat,
    FontWeight,
    font_weight_from_qt,
    font_weight_to_qt,
)
from ballontranslator.utils.textblock import TextBlock


def get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class FontWeightPersistenceTest(unittest.TestCase):

    def test_qt5_project_weights_normalize_to_css_scale(self):
        self.assertIs(FontFormat(font_weight=25).font_weight, FontWeight.Light)
        self.assertIs(FontFormat(font_weight=75).font_weight, FontWeight.Bold)

        block = TextBlock(font_weight=25)

        self.assertIs(block.font_weight, FontWeight.Light)

    def test_every_weight_round_trips_across_qt_boundaries(self):
        for weight in FontWeight:
            with self.subTest(weight=weight):
                self.assertIs(
                    font_weight_from_qt(font_weight_to_qt(weight, qt6=True)),
                    weight,
                )
                self.assertIs(
                    font_weight_from_qt(font_weight_to_qt(weight, qt6=False)),
                    weight,
                )

    def test_rich_text_writes_standard_css_and_round_trips(self):
        for weight in FontWeight:
            with self.subTest(weight=weight):
                document = QTextDocument()
                document.setPlainText('x')
                cursor = QTextCursor(document)
                cursor.select(QTextCursor.SelectionType.Document)
                char_format = QTextCharFormat()
                char_format.setFontWeight(
                    QFont.Weight(font_weight_to_qt(weight, qt6=QT6))
                )
                cursor.mergeCharFormat(char_format)

                html = to_rich_text_html(document)

                self.assertIn(
                    f'font-weight:{int(weight)}', html.replace(' ', '')
                )
                restored = QTextDocument()
                load_rich_text_html(restored, html)
                restored_cursor = QTextCursor(restored)
                restored_cursor.select(
                    QTextCursor.SelectionType.Document
                )
                self.assertIs(
                    font_weight_from_qt(
                        restored_cursor.charFormat().fontWeight()
                    ),
                    weight,
                )

    def test_legacy_qt5_rich_text_keeps_bold_weight(self):
        html = (
            '<html><head><meta name="qrichtext" content="1" />'
            '</head><body><span style="font-weight:600">x</span>'
            '</body></html>'
        )
        restored = QTextDocument()

        load_rich_text_html(restored, html)

        cursor = QTextCursor(restored)
        cursor.select(QTextCursor.SelectionType.Document)
        self.assertIs(
            font_weight_from_qt(cursor.charFormat().fontWeight()),
            FontWeight.Bold,
        )


class _FontDatabaseStub:
    _weights = {
        ('Example', 'Regular'): 400,
        ('Example', 'Light'): 300,
        ('Example Light', 'Light'): 300,
        ('Example Medium', 'Regular'): 400,
    }

    @classmethod
    def styles(cls, family: str) -> list[str]:
        return [
            style
            for candidate, style in cls._weights
            if candidate == family
        ]

    @classmethod
    def weight(cls, family: str, style: str) -> int:
        return cls._weights[(family, style)]


class FontWeightUiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        self.old_active_format = C.active_format
        self.old_canvas = getattr(SW, 'canvas', None)
        self.old_font_families = shared.FONT_FAMILIES
        SW.canvas = SimpleNamespace(selected_text_items=lambda: [])

    def tearDown(self) -> None:
        C.active_format = self.old_active_format
        SW.canvas = self.old_canvas
        shared.FONT_FAMILIES = self.old_font_families

    def _make_panel(self) -> FontFormatPanel:
        with patch.object(
            shared,
            'register_view_widget',
            lambda *_args: None,
            create=True,
        ):
            panel = FontFormatPanel(self.app)
        self.addCleanup(panel.deleteLater)
        return panel

    def test_selector_exposes_all_supported_weights(self):
        selector = FontWeightComboBox()
        self.addCleanup(selector.deleteLater)

        self.assertEqual(
            [selector.itemData(index) for index in range(selector.count())],
            [int(weight) for weight in FontWeight],
        )
        self.assertIs(selector.weight(), FontWeight.Normal)

    def test_bold_shortcut_action_keeps_its_normal_bold_toggle(self):
        panel = self._make_panel()
        active = FontFormat(font_weight=FontWeight.Light)
        panel.global_format = active
        panel.set_active_format(active)

        panel.toggle_bold()

        self.assertIs(active.font_weight, FontWeight.Bold)
        self.assertIs(panel.fontWeightBox.weight(), FontWeight.Bold)

        panel.toggle_bold()

        self.assertIs(active.font_weight, FontWeight.Normal)
        self.assertIs(panel.fontWeightBox.weight(), FontWeight.Normal)

    def test_explicit_weight_change_canonicalizes_a_weight_alias(self):
        shared.FONT_FAMILIES = {'Example', 'Example Light'}
        panel = self._make_panel()
        active = FontFormat(
            font_family='Example Light',
            font_weight=FontWeight.Light,
        )
        panel.global_format = active
        panel.familybox.canonical_weight_aliases = {
            'Example Light': ('Example', FontWeight.Light),
        }
        panel.set_active_format(active)

        panel.on_font_weight_changed('font_weight', FontWeight.Medium)

        self.assertEqual(active.font_family, 'Example')
        self.assertIs(active.font_weight, FontWeight.Medium)

    def test_selected_text_receives_only_the_new_weight(self):
        block = TextBlock([0, 0, 300, 100])
        block._bounding_rect = [0, 0, 300, 100]
        block.translation = 'AB'
        item = TextBlkItem(block)
        self.addCleanup(item.deleteLater)
        item.startEdit()
        cursor = item.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        item.setTextCursor(cursor)

        item.setFontWeight(
            FontWeight.Black,
            set_selected=True,
            restore_cursor=True,
        )

        first = QTextCursor(item.document())
        first.setPosition(0)
        first.setPosition(1, QTextCursor.MoveMode.KeepAnchor)
        second = QTextCursor(item.document())
        second.setPosition(1)
        second.setPosition(2, QTextCursor.MoveMode.KeepAnchor)
        self.assertIs(
            font_weight_from_qt(first.charFormat().fontWeight()),
            FontWeight.Black,
        )
        self.assertIs(
            font_weight_from_qt(second.charFormat().fontWeight()),
            FontWeight.Normal,
        )

    def test_family_alias_filter_is_conservative(self):
        families = {
            'Example',
            'Example Light',
            'Example Medium',
            'Missing Base Bold',
        }
        with patch(
            'ballontranslator.ui.text_engine.formatting.panel._font_database',
            return_value=_FontDatabaseStub,
        ):
            aliases = _weight_family_aliases(families)

        self.assertEqual(
            aliases,
            {'Example Light': ('Example', FontWeight.Light)},
        )
        self.assertEqual(
            _split_weight_family_name('Example SemiBold'),
            ('Example', FontWeight.DemiBold),
        )

    def test_alias_stays_available_when_its_base_is_filtered_out(self):
        shared.FONT_FAMILIES = {'Example', 'Example Light'}
        aliases = {'Example Light': ('Example', FontWeight.Light)}
        combo = FontFamilyComboBox()
        self.addCleanup(combo.deleteLater)
        with patch(
            'ballontranslator.ui.text_engine.formatting.panel.'
            '_weight_family_aliases',
            return_value=aliases,
        ):
            combo.update_font_list(['Example Light'])

        self.assertEqual(
            [combo.itemText(index) for index in range(combo.count())],
            ['Example Light'],
        )
        self.assertEqual(
            combo.canonical_family('Example Light'),
            ('Example Light', None),
        )

        combo.set_displayed_font('Example Light')
        combo.update_font_list(['Example', 'Example Light'])

        self.assertEqual(
            [combo.itemText(index) for index in range(combo.count())],
            ['Example'],
        )
        self.assertEqual(combo.currentText(), 'Example Light')
        self.assertEqual(
            combo.canonical_family('Example Light'),
            ('Example', FontWeight.Light),
        )


if __name__ == '__main__':
    unittest.main()
