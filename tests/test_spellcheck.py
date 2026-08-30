import os
import unittest
from unittest.mock import patch


os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtGui import QTextCursor, QTextDocument
from qtpy.QtWidgets import QApplication, QWidget

from ballontranslator.ui import spellcheck as spellcheck_module
from ballontranslator.ui.spellcheck import (
    SpellCheckHighlighter,
    SpellCheckManager,
    iter_spellcheck_words,
)
from ballontranslator.ui.text_engine.editing import widgets as widgets_module
from ballontranslator.ui.text_engine.editing.widgets import (
    SourceTextEdit,
    TransTextEdit,
)
from ballontranslator.utils.config import pcfg


class _FakeSpellCheckManager:
    def __init__(self, *, correct: bool = False) -> None:
        self.correct = correct
        self.available_calls = 0
        self.checked = []

    def register_highlighter(self, _highlighter) -> None:
        pass

    def is_available(self) -> bool:
        self.available_calls += 1
        return True

    def is_correct(self, word: str) -> bool:
        self.checked.append(word)
        return self.correct

    def get_suggestions(self, _word: str):
        return ['hello']


class _FakePopup:
    def __init__(self) -> None:
        self.hidden = False
        self.shown = False

    def hide(self) -> None:
        self.hidden = True

    def show(self) -> None:
        self.shown = True


class SpellCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_word_spans_cover_cjk_and_preserve_utf16_offsets(self) -> None:
        self.assertEqual(
            list(iter_spellcheck_words('한helo世界wrld')),
            [(3, 4, 'helo'), (9, 4, 'wrld')],
        )
        self.assertEqual(
            list(iter_spellcheck_words('𠀀helo')),
            [(2, 4, 'helo')],
        )
        self.assertEqual(list(iter_spellcheck_words('ㇰ々한글漢字')), [])

    def test_highlighter_checks_only_non_cjk_runs(self) -> None:
        manager = _FakeSpellCheckManager()
        document = QTextDocument('한helo世界wrld𠀀')
        with patch.object(
            SpellCheckManager, 'get_instance', return_value=manager
        ), patch.object(pcfg, 'spellcheck_enabled', True):
            highlighter = SpellCheckHighlighter(document)
            highlighter.rehighlight()

        self.assertEqual(manager.checked, ['helo', 'wrld'])
        ranges = document.firstBlock().layout().formats()
        self.assertEqual(
            [(format_range.start, format_range.length) for format_range in ranges],
            [(3, 4), (9, 4)],
        )

    def test_disabled_highlighter_does_no_package_or_text_work(self) -> None:
        manager = _FakeSpellCheckManager()
        with patch.object(SpellCheckManager, 'get_instance', return_value=manager):
            highlighter = SpellCheckHighlighter(None)

        with patch.object(pcfg, 'spellcheck_enabled', False), patch.object(
            spellcheck_module,
            'iter_spellcheck_words',
            side_effect=AssertionError('disabled spellcheck scanned text'),
        ):
            highlighter.highlightBlock('helo世界')

        self.assertEqual(manager.available_calls, 0)
        self.assertEqual(manager.checked, [])

    def test_source_guard_precedes_matching_and_keeps_translation_enabled(self) -> None:
        manager = _FakeSpellCheckManager(correct=True)
        parent = QWidget()
        with patch.object(
            SpellCheckManager, 'get_instance', return_value=manager
        ), patch.object(pcfg, 'spellcheck_enabled', True), patch.object(
            pcfg, 'spellcheck_on_source_enabled', False
        ):
            source = SourceTextEdit(0, parent)
            translation = TransTextEdit(1, parent)
            self._select_word(source, 'helo')
            self._select_word(translation, 'helo')

            manager.available_calls = 0
            manager.checked.clear()
            with patch.object(
                widgets_module,
                'iter_spellcheck_words',
                side_effect=AssertionError('disabled source spellcheck scanned text'),
            ):
                source.on_selection_changed()

            self.assertEqual(manager.available_calls, 0)
            self.assertEqual(manager.checked, [])

            with patch.object(pcfg, 'spellcheck_enabled', False), patch.object(
                widgets_module,
                'iter_spellcheck_words',
                side_effect=AssertionError('disabled spellcheck scanned selection'),
            ):
                translation.on_selection_changed()

            self.assertEqual(manager.available_calls, 0)
            self.assertEqual(manager.checked, [])

            translation.on_selection_changed()
            self.assertEqual(manager.checked, ['helo'])

            with patch.object(pcfg, 'spellcheck_on_source_enabled', True):
                source.on_selection_changed()
            self.assertEqual(manager.checked, ['helo', 'helo'])

        source.deleteLater()
        translation.deleteLater()
        parent.deleteLater()

    def test_pending_source_suggestion_rechecks_config(self) -> None:
        manager = _FakeSpellCheckManager()
        parent = QWidget()
        with patch.object(SpellCheckManager, 'get_instance', return_value=manager):
            source = SourceTextEdit(0, parent)

        popup = _FakePopup()
        source.suggestion_popup = popup
        source.current_suggestion_word = 'helo'
        with patch.object(pcfg, 'spellcheck_enabled', True), patch.object(
            pcfg, 'spellcheck_on_source_enabled', False
        ):
            source.show_suggestions_popup(QTextCursor(), 'helo', ['hello'])

        self.assertTrue(popup.hidden)
        self.assertFalse(popup.shown)
        self.assertIsNone(source.current_suggestion_word)
        source.deleteLater()
        parent.deleteLater()

    @staticmethod
    def _select_word(editor: SourceTextEdit, text: str) -> None:
        editor.blockSignals(True)
        try:
            editor.setPlainText(text)
            cursor = editor.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            editor.setTextCursor(cursor)
        finally:
            editor.blockSignals(False)


if __name__ == '__main__':
    unittest.main()
