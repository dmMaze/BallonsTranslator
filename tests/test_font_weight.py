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
from ballontranslator.utils.font_registry import FontEntry, FontFace, FontRegistry
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

    def test_legacy_bold_migrates_without_overriding_explicit_weight(self):
        self.assertIs(
            FontFormat(bold=True).font_weight,
            FontWeight.Bold,
        )
        self.assertIs(
            FontFormat(bold=True, font_weight=FontWeight.Light).font_weight,
            FontWeight.Light,
        )
        self.assertIs(
            TextBlock(bold=True).font_weight,
            FontWeight.Bold,
        )
        self.assertIs(
            TextBlock(bold=True, font_weight=25).font_weight,
            FontWeight.Light,
        )

    def test_serialized_format_drops_legacy_migration_fields(self):
        serialized = FontFormat(bold=True).to_serializable_dict()

        self.assertEqual(serialized['font_weight'], int(FontWeight.Bold))
        self.assertNotIn('bold', serialized)
        self.assertNotIn('deprecated_attributes', serialized)

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


class FontWeightUiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = get_app()

    def setUp(self) -> None:
        self.old_active_format = C.active_format
        self.old_canvas = getattr(SW, 'canvas', None)
        self.old_font_registry = shared.FONT_REGISTRY
        SW.canvas = SimpleNamespace(selected_text_items=lambda: [])

    def tearDown(self) -> None:
        C.active_format = self.old_active_format
        SW.canvas = self.old_canvas
        shared.FONT_REGISTRY = self.old_font_registry

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

    def test_selector_shows_detected_weights_bold_and_current_weight(self):
        selector = FontWeightComboBox()
        self.addCleanup(selector.deleteLater)

        selector.update_weights([300, 500], FontWeight.Black)

        self.assertEqual(
            [selector.itemData(index) for index in range(selector.count())],
            [300, 500, 700, 900],
        )
        self.assertIs(selector.weight(), FontWeight.Black)

    def test_family_selection_preserves_requested_weight(self):
        panel = self._make_panel()
        active = FontFormat(font_family='Example', font_weight=FontWeight.Light)
        panel.global_format = active
        panel.set_active_format(active)
        entry = FontEntry(
            'Example Book',
            'Example Book',
            'Example Book',
            'custom',
            weights=[250],
        )
        panel.familybox.update_font_entries([entry])

        panel.familybox.setCurrentIndex(0)

        self.assertEqual(active.font_family, 'Example Book')
        self.assertIs(active.font_weight, FontWeight.Light)
        self.assertEqual(
            [
                panel.fontWeightBox.itemData(index)
                for index in range(panel.fontWeightBox.count())
            ],
            [200, 300, 700],
        )

    def test_explicit_weight_change_removes_old_unsupported_weight(self):
        panel = self._make_panel()
        entry = FontEntry(
            'Example', 'Example', 'Example', 'custom',
            weights=[300, 500],
        )
        shared.FONT_REGISTRY = FontRegistry(custom_entries=[entry])
        active = FontFormat(
            font_family='Example',
            font_weight=FontWeight.Black,
        )
        panel.global_format = active
        panel.set_active_format(active)
        panel.update_font_entries([entry])

        self.assertGreaterEqual(
            panel.fontWeightBox.findData(int(FontWeight.Black)),
            0,
        )

        panel.on_font_weight_changed('font_weight', FontWeight.Medium)

        self.assertIs(active.font_weight, FontWeight.Medium)
        self.assertEqual(
            [
                panel.fontWeightBox.itemData(index)
                for index in range(panel.fontWeightBox.count())
            ],
            [300, 500, 700],
        )

    def test_picker_group_changes_storage_face_with_weight(self):
        panel = self._make_panel()
        faces = [
            FontFace(
                'Example Light', 'Example Light', 'Example Light',
                'Light', 300,
            ),
            FontFace(
                'Example Bold', 'Example Bold', 'Example Bold',
                'Bold', 700,
            ),
        ]
        entry = FontEntry(
            'Example', 'Example', 'Example Light', 'custom',
            weights=[300, 700], faces=faces, is_pseudo_group=True,
        )
        shared.FONT_REGISTRY = FontRegistry(custom_entries=[entry])
        active = FontFormat(
            font_family='Example Light', font_weight=FontWeight.Light,
        )
        panel.global_format = active
        panel.familybox.update_font_entries([entry])
        panel.set_active_format(active)

        panel.on_font_weight_changed('font_weight', FontWeight.Bold)

        self.assertEqual(active.font_family, 'Example Bold')
        self.assertIs(active.font_weight, FontWeight.Bold)

    def test_picker_group_face_and_weight_use_one_document_edit(self):
        faces = [
            FontFace(
                'DejaVu Sans', 'Example Light', 'DejaVu Sans',
                'Light', 300,
            ),
            FontFace(
                'DejaVu Serif', 'Example Bold', 'DejaVu Serif',
                'Bold', 700,
            ),
        ]
        entry = FontEntry(
            'Example', 'Example', 'DejaVu Sans', 'custom',
            weights=[300, 700], faces=faces, is_pseudo_group=True,
        )
        shared.FONT_REGISTRY = FontRegistry(custom_entries=[entry])
        block = TextBlock([0, 0, 300, 100])
        block._bounding_rect = [0, 0, 300, 100]
        block.translation = 'AB'
        block.font_family = 'DejaVu Sans'
        block.font_weight = FontWeight.Light
        item = TextBlkItem(block)
        self.addCleanup(item.deleteLater)
        item.startEdit()
        cursor = item.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        item.setTextCursor(cursor)

        panel = self._make_panel()
        panel.global_format = FontFormat()
        panel.textblk_item = item
        panel.familybox.update_font_entries([entry])
        active = item.get_fontformat()
        panel.set_active_format(active)
        pushed_steps = []
        item.push_undo_stack.connect(
            lambda count, formatting: pushed_steps.append(
                (count, formatting)
            )
        )

        with patch(
            'ballontranslator.ui.text_engine.formatting.commands.'
            'restore_canvas_view_focus'
        ):
            panel.on_font_weight_changed(
                'font_weight', FontWeight.Bold
            )

        self.assertEqual(len(pushed_steps), 1)
        self.assertGreater(pushed_steps[0][0], 0)
        self.assertTrue(pushed_steps[0][1])
        self.assertEqual(active.font_family, 'DejaVu Serif')
        self.assertIs(active.font_weight, FontWeight.Bold)
        formatted = QTextCursor(item.document())
        formatted.select(QTextCursor.SelectionType.Document)
        self.assertEqual(formatted.charFormat().font().family(), 'DejaVu Serif')
        self.assertIs(
            font_weight_from_qt(formatted.charFormat().fontWeight()),
            FontWeight.Bold,
        )

        item.document().undo()
        restored = QTextCursor(item.document())
        restored.select(QTextCursor.SelectionType.Document)
        self.assertEqual(restored.charFormat().font().family(), 'DejaVu Sans')
        self.assertIs(
            font_weight_from_qt(restored.charFormat().fontWeight()),
            FontWeight.Light,
        )

    def test_editable_family_resolves_display_name_and_rejects_unknowns(self):
        combo = FontFamilyComboBox()
        self.addCleanup(combo.deleteLater)
        entry = FontEntry(
            'Example Sans', 'Example Display', 'Example Sans', 'custom',
            aliases={'예제 산스'},
        )
        shared.FONT_REGISTRY = FontRegistry(custom_entries=[entry])
        combo.update_font_entries([entry])
        changes = []
        combo.param_changed.connect(lambda _name, value: changes.append(value))

        combo.setEditText('예제 산스')
        combo.apply_fontfamily()

        self.assertEqual(changes, ['Example Sans'])

        combo.set_current_family('Missing Legacy Font')
        combo.apply_fontfamily()
        combo.setEditText('Made Up Font')
        combo.apply_fontfamily()

        self.assertEqual(changes, ['Example Sans', 'Missing Legacy Font'])
        self.assertEqual(combo.currentText(), 'Missing Legacy Font')

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

    def test_explicit_weight_change_resolves_a_hidden_group_face(self):
        faces = [
            FontFace(
                'Example Light', 'Example Light', 'Example Light',
                'Light', 300,
            ),
            FontFace(
                'Example Bold', 'Example Bold', 'Example Bold',
                'Bold', 700,
            ),
        ]
        entry = FontEntry(
            'Example', 'Example', 'Example Light', 'custom',
            weights=[300, 700], faces=faces, is_pseudo_group=True,
        )
        shared.FONT_REGISTRY = FontRegistry(custom_entries=[entry])
        panel = self._make_panel()
        active = FontFormat(
            font_family='Example Light',
            font_weight=FontWeight.Light,
        )
        panel.global_format = active
        panel.set_active_format(active)

        panel.on_font_weight_changed('font_weight', FontWeight.Bold)

        self.assertEqual(active.font_family, 'Example Bold')
        self.assertIs(active.font_weight, FontWeight.Bold)

    def test_explicit_weight_change_canonicalizes_hidden_weight_family(self):
        base = FontEntry(
            'Example', 'Example', 'Example', 'system', weights=[300, 500]
        )
        light = FontEntry(
            'Example Light', 'Example Light', 'Example Light', 'system',
            weights=[300],
        )
        registry = FontRegistry(system_entries=[base, light])
        shared.FONT_REGISTRY = registry
        panel = self._make_panel()
        active = FontFormat(
            font_family='Example Light',
            font_weight=FontWeight.Light,
        )
        panel.global_format = active
        panel.familybox.update_font_entries(registry.entries())
        panel.set_active_format(active)

        self.assertEqual(
            [
                panel.fontWeightBox.itemData(index)
                for index in range(panel.fontWeightBox.count())
            ],
            [300, 500, 700],
        )

        panel.on_font_weight_changed('font_weight', FontWeight.Medium)

        self.assertEqual(active.font_family, 'Example')
        self.assertIs(active.font_weight, FontWeight.Medium)

    def test_weight_family_is_preserved_when_base_is_filtered_out(self):
        base = FontEntry(
            'Example', 'Example', 'Example', 'system', weights=[300, 500]
        )
        light = FontEntry(
            'Example Light', 'Example Light', 'Example Light', 'system',
            weights=[300],
        )
        registry = FontRegistry(system_entries=[base, light])
        shared.FONT_REGISTRY = registry
        panel = self._make_panel()
        active = FontFormat(
            font_family='Example Light',
            font_weight=FontWeight.Light,
        )
        panel.global_format = active
        panel.familybox.update_font_entries(
            registry.entries(excluded=['Example'])
        )
        panel.set_active_format(active)

        self.assertEqual(
            [
                panel.fontWeightBox.itemData(index)
                for index in range(panel.fontWeightBox.count())
            ],
            [300, 700],
        )

        panel.on_font_weight_changed('font_weight', FontWeight.Medium)

        self.assertEqual(active.font_family, 'Example Light')
        self.assertIs(active.font_weight, FontWeight.Medium)
        self.assertEqual(
            [
                panel.fontWeightBox.itemData(index)
                for index in range(panel.fontWeightBox.count())
            ],
            [300, 500, 700],
        )

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

if __name__ == '__main__':
    unittest.main()
