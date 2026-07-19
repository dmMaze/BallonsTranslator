import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


# The glossary module is intentionally stdlib-only. Load it without importing
# the eager module package so this test does not require unrelated OCR models.
GLOSSARY_PATH = (
    Path(__file__).resolve().parents[1]
    / "ballontranslator"
    / "modules"
    / "context"
    / "glossary.py"
)
GLOSSARY_SPEC = importlib.util.spec_from_file_location(
    "_translator_glossary_for_tests",
    GLOSSARY_PATH,
)
glossary = importlib.util.module_from_spec(GLOSSARY_SPEC)
sys.modules[GLOSSARY_SPEC.name] = glossary
GLOSSARY_SPEC.loader.exec_module(glossary)

GLOSSARY_MODE_ALL = glossary.GLOSSARY_MODE_ALL
GLOSSARY_MODE_MATCHING = glossary.GLOSSARY_MODE_MATCHING
GlossaryEntry = glossary.GlossaryEntry
GlossaryError = glossary.GlossaryError
load_glossary = glossary.load_glossary
normalize_glossary_path = glossary.normalize_glossary_path
render_glossary = glossary.render_glossary
select_glossary = glossary.select_glossary


class TranslatorGlossaryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_supported_glossary_formats_load_in_file_order(self):
        two_entries = (
            GlossaryEntry("勇者", "Hero", "title"),
            GlossaryEntry("魔王", "Demon King", ""),
        )
        cases = (
            (
                "terms.json",
                json.dumps(
                    [
                        {"src": " 勇者 ", "dst": " Hero ", "info": " title "},
                        {"src": "魔王", "dst": "Demon King"},
                        {"src": "勇者", "dst": "Hero", "info": "title"},
                    ],
                    ensure_ascii=False,
                ),
                two_entries,
            ),
            (
                "terms.txt",
                "# heading\n\n// disabled\n勇者->Hero # title\n魔王->Demon King\n",
                two_entries,
            ),
            (
                "terms.tsv",
                "# heading\n勇者\tHero\ttitle\n魔王\tDemon King\n",
                two_entries,
            ),
            (
                "galtransl.txt",
                "勇者\tHero\ttitle\n",
                (GlossaryEntry("勇者", "Hero", "title"),),
            ),
        )
        for name, content, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(load_glossary(self._write(name, content)), expected)

    def test_path_expansion_and_normalization_share_cached_result(self):
        path = self._write("terms.json", '[{"src":"a","dst":"A"}]')

        with mock.patch.dict(
            os.environ,
            {"HOME": str(self.root), "TEST_GLOSSARY_FILE": str(path)},
        ):
            direct = load_glossary(path)
            from_home = load_glossary("~/terms.json")
            from_variable = load_glossary("$TEST_GLOSSARY_FILE")

        self.assertIs(direct, from_home)
        self.assertIs(direct, from_variable)
        self.assertEqual(normalize_glossary_path(path), str(path.resolve()))

    def test_cache_reloads_after_file_signature_changes(self):
        path = self._write("terms.json", '[{"src":"a","dst":"A"}]')
        first = load_glossary(path)

        self.assertIs(first, load_glossary(path))

        path.write_text('[{"src":"longer","dst":"B"}]', encoding="utf-8")
        second = load_glossary(path)

        self.assertIsNot(first, second)
        self.assertEqual(second, (GlossaryEntry("longer", "B"),))

    def test_matching_is_casefolded_literal_and_preserves_file_order(self):
        entries = (
            GlossaryEntry("Mage", "魔法使"),
            GlossaryEntry("Straße", "street"),
            GlossaryEntry("C++", "language"),
            GlossaryEntry("[boss]", "leader"),
        )

        selected = select_glossary(
            entries,
            ["STRASSE appears twice: strasse.", "C++ is literal; C++ again."],
            GLOSSARY_MODE_MATCHING,
        )

        self.assertEqual(selected, (entries[1], entries[2]))
        self.assertEqual(
            select_glossary(entries, ["nothing"], GLOSSARY_MODE_ALL),
            entries,
        )

    def test_invalid_mode_fails_concisely(self):
        with self.assertRaisesRegex(GlossaryError, "Invalid glossary mode"):
            select_glossary((GlossaryEntry("a", "A"),), ["a"], "unknown")

    def test_rendering_is_compact_unicode_json_in_file_order(self):
        rendered = render_glossary(
            (
                GlossaryEntry("勇者", "Hero", "title"),
                GlossaryEntry("魔王", "Demon King"),
            )
        )

        self.assertEqual(
            rendered,
            '{"glossary":[{"source":"勇者","translation":"Hero","note":"title"},'
            '{"source":"魔王","translation":"Demon King","note":""}]}',
        )
        self.assertEqual(render_glossary(()), "")

    def test_conflicting_and_malformed_glossaries_report_locations(self):
        cases = (
            (
                "conflict.txt",
                "# heading\nHero->勇者\n\nhero->英雄\n",
                r"at line 4.*conflicts with line 2",
            ),
            (
                "conflict.json",
                '[{"src":"Hero","dst":"勇者"},{"src":"Hero","dst":"英雄"}]',
                r"at entry 2.*conflicts with entry 1",
            ),
            (
                "bad.txt",
                "# heading\n\nnot a glossary row\n",
                "at line 3",
            ),
            (
                "bad.json",
                '[\n  {"src":"Hero","dst":"勇者"},\n  nope\n]\n',
                "at line 3",
            ),
            ("missing-src.json", '[{"dst":"A"}]', 'missing "src"'),
            ("missing-dst.json", '[{"src":"a"}]', 'missing "dst"'),
            (
                "invalid-src.json",
                '[{"src":1,"dst":"A"}]',
                'field "src" must be a string',
            ),
            (
                "empty-dst.json",
                '[{"src":"a","dst":""}]',
                "translation must not be empty",
            ),
            (
                "invalid-info.json",
                '[{"src":"a","dst":"A","info":1}]',
                'field "info" must be a string',
            ),
        )
        for name, content, error in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(GlossaryError, error):
                    load_glossary(self._write(name, content))

    def test_missing_unreadable_and_unsupported_files_fail_concisely(self):
        missing = self.root / "missing.json"
        with self.assertRaisesRegex(GlossaryError, "Glossary file not found"):
            load_glossary(missing)

        unreadable = self._write("unreadable.json", "[]")
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=PermissionError(13, "Permission denied"),
        ):
            with self.assertRaisesRegex(GlossaryError, "Could not read glossary"):
                load_glossary(unreadable)

        unsupported = self._write("terms.csv", "a,b")
        with self.assertRaisesRegex(GlossaryError, "Unsupported glossary format"):
            load_glossary(unsupported)

if __name__ == "__main__":
    unittest.main()
