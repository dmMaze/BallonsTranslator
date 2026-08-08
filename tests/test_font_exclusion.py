import unittest

from ballontranslator.utils.shared import LEGACY_FONTS, get_filtered_font_list


class TestGetFilteredFontList(unittest.TestCase):
    def test_filters_excluded_names(self):
        result = get_filtered_font_list(['Arial', 'Times', 'Courier'], ['Times'])
        self.assertEqual(result, ['Arial', 'Courier'])

    def test_no_excluded_returns_all(self):
        self.assertEqual(get_filtered_font_list(['Arial', 'Times']), ['Arial', 'Times'])

    def test_empty_font_list(self):
        self.assertEqual(get_filtered_font_list([], ['Arial']), [])

    def test_exclude_all_fonts(self):
        self.assertEqual(get_filtered_font_list(['Arial'], ['Arial']), [])

    def test_accepts_set_input(self):
        # FONT_FAMILIES is a set at runtime, CUSTOM_FONTS is a list
        result = get_filtered_font_list({'Arial', 'Times'}, ['Times'])
        self.assertEqual(result, ['Arial'])

    def test_missing_excluded_name_is_ignored(self):
        result = get_filtered_font_list(['Arial'], ['NotInstalled'])
        self.assertEqual(result, ['Arial'])

    def test_legacy_fonts_contains_windows_legacy_families(self):
        self.assertTrue(LEGACY_FONTS)
        self.assertIn('MS Sans Serif', LEGACY_FONTS)


if __name__ == '__main__':
    unittest.main()
