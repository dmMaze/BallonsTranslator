import re
from html import unescape

class OcrFixEngine:
    def __init__(self) -> None:
    # def __init__(self, local_no: int, start: int, end: int) -> None:
        # self.local_no = local_no
        # self.start = start
        # self.end = end

        import pathlib
        path = pathlib.Path(__file__).parent  / 'en_US'
        from spylls.hunspell.dictionary import Dictionary
        DICT_PATH = 'j:\Comic translate\spylls\examples\en_US'
        # dictionary = Dictionary.from_files(str(path))
        self.dictionary = Dictionary.from_files(DICT_PATH)

    def DoSuggest(self, word: str):
        return self.dictionary.suggest(word)

    def GetUnknownWordsViaDictionary(self, text: str) -> list:
        split_chars = set(' -.?,!;:\"“”()[]{}|<>/+¿¡…—–♪♫„«»‹›؛،؟\u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u200E\u200F\u2028\u2029\u202A\u202B\u202C\u202D\u202E\u202F\u3000\uFEFF')
        unknown_words = []
        for word in text.split():
            word = word.strip(''.join(split_chars))
            if not self.is_number(word):
                # if len(word) == 1 or not self.dictionary.lookup(word):
                if not self.dictionary.lookup(word):
                    unknown_words.append(word)
        return unknown_words

    def CountUnknownWordsViaDictionary(self, text: str) -> int:
        split_chars = set(' -.?,!;:\"“”()[]{}|<>/+¿¡…—–♪♫„«»‹›؛،؟\u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u200E\u200F\u2028\u2029\u202A\u202B\u202C\u202D\u202E\u202F\u3000\uFEFF')
        for word in text.split():
            word = word.strip(''.join(split_chars))
            if not self.is_word_known_or_number2(word):
                # dictionary.lookup(word)
                # print(dictionary.lookup('spylls'))
                # False
                # for suggestion in dictionary.suggest('spylls'):
                    # print(suggestion)
                correct = len(word) > 1 and dictionary.lookup(word)
                # if not correct:
                #     correct = len(word) > 2 and hunspell.spell(word.strip('\''))

                # if not correct and len(word) == 1 and three_letter_iso_language_name == 'eng' and word in ['I', 'A', 'a']:
                #     correct = True

                if correct:
                    number_of_correct_words += 1
                else:
                    words_not_found += 1
            elif len(word) > 3:
                number_of_correct_words += 1

    # def CountUnknownWordsViaDictionary(pattern: re.Pattern, text: str) -> Tuple[int, Dict]:
    def count_unknown_words_via_dictionary(line, hunspell, three_letter_iso_language_name, word_skip_list, name_list, name_list_uppercase, name_list_obj, spell_check_word_lists):
        """
        Count the number of unknown words in a line using a dictionary.

        Args:
            line (str): The line to check.
            hunspell: A hunspell object.
            three_letter_iso_language_name (str): The three letter ISO language name.
            word_skip_list (set): A set of words to skip.
            name_list (set): A set of names.
            name_list_uppercase (set): A set of uppercase names.
            name_list_obj: An object with an is_in_names_multi_word_list method.
            spell_check_word_lists: An object with a has_user_word method.

        Returns:
            int: The number of unknown words.
        """
        number_of_correct_words = 0
        if not hunspell:
            return 0

        min_length = 2
        if True:  # Replace with your configuration
            min_length = 1

        words_not_found = 0
        split_chars = set(' -.?,!;:\"“”()[]{}|<>/+¿¡…—–♪♫„«»‹›؛،؟\u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u200B\u200E\u200F\u2028\u2029\u202A\u202B\u202C\u202D\u202E\u202F\u3000\uFEFF')
        words = remove_open_close_tags(line, 'i').split()
        for word in words:
            word = word.strip(''.join(split_chars))
            if len(word) >= min_length:
                if not is_word_known_or_number(word, line, word_skip_list, name_list, name_list_uppercase, name_list_obj, spell_check_word_lists):
                    correct = len(word) > 1 and hunspell.spell(word)
                    if not correct:
                        correct = len(word) > 2 and hunspell.spell(word.strip('\''))

                    if not correct and len(word) == 1 and three_letter_iso_language_name == 'eng' and word in ['I', 'A', 'a']:
                        correct = True

                    if correct:
                        number_of_correct_words += 1
                    else:
                        words_not_found += 1
                elif len(word) > 3:
                    number_of_correct_words += 1

        return words_not_found, number_of_correct_words

    def is_number(self, word):
        """
        Check if a word is known or a number.

        Args:
            word (str): The word to check.
            line (str): The line the word is from.
            word_skip_list (set): A set of words to skip.
            name_list (set): A set of names.
            name_list_uppercase (set): A set of uppercase names.
            name_list_obj: An object with an is_in_names_multi_word_list method.
            spell_check_word_lists: An object with a has_user_word method.

        Returns:
            bool: True if the word is known or a number, False otherwise.
        """
        if word.strip('\'').replace('$', '').replace('£', '').replace('¥', '').replace('¢', '').replace('.', '', 1).isdigit():
            return True

        return False

    def is_word_known_or_number(word, line, word_skip_list, name_list, name_list_uppercase, name_list_obj, spell_check_word_lists):
        """
        Check if a word is known or a number.

        Args:
            word (str): The word to check.
            line (str): The line the word is from.
            word_skip_list (set): A set of words to skip.
            name_list (set): A set of names.
            name_list_uppercase (set): A set of uppercase names.
            name_list_obj: An object with an is_in_names_multi_word_list method.
            spell_check_word_lists: An object with a has_user_word method.

        Returns:
            bool: True if the word is known or a number, False otherwise.
        """
        if word.strip('\'').replace('$', '').replace('£', '').replace('¥', '').replace('¢', '').replace('.', '', 1).isdigit():
            return True

        if word in word_skip_list:
            return True

        if word.strip('\'') in name_list:
            return True

        if word.strip('\'') in name_list_uppercase:
            return True

        if spell_check_word_lists and spell_check_word_lists.has_user_word(word.lower()):
            return True

        if spell_check_word_lists and spell_check_word_lists.has_user_word(word.strip('\'').lower()):
            return True

        if len(word) > 2 and word in name_list_uppercase:
            return True

        if len(word) > 2 and word in name_list_obj.name_list_with_apostrophe:
            return True

        if name_list_obj and name_list_obj.is_in_names_multi_word_list(line, word):
            return True

        return False


    def remove_open_close_tags(source, *tags):
        """
        Remove all of the specified opening and closing tags from the source HTML string.

        Args:
            source (str): The source string to search for specified HTML tags.
            tags (str): The HTML tags to remove.

        Returns:
            str: A new string without the specified opening and closing tags.
        """
        if not source or '<' not in source:
            return source

        pattern = r'<\s*\/?(\w+)[^>]*>'
        return re.sub(pattern, lambda m: '' if m.group(1).lower() in [tag.lower() for tag in tags] else m.group(0), source)

    REGEX_ALONE_IAS_L = re.compile(r"\bl\b", re.IGNORECASE)
    REGEX_LOWERCASE_L = re.compile(r"[A-ZÆØÅÄÖÉÈÀÙÂÊÎÔÛËÏ]l[A-ZÆØÅÄÖÉÈÀÙÂÊÎÔÛËÏ]", re.IGNORECASE)
    REGEX_UPPERCASE_I = re.compile(r"[a-zæøåöääöéèàùâêîôûëï]I\.", re.IGNORECASE)
    REGEX_NUMBER1 = re.compile(r"(?<=\d) 1(?!/\d)", re.IGNORECASE)

    def count_unknown_words_via_dictionary(line, out_numberOfCorrectWords):
        out_numberOfCorrectWords = 0
        if _hunspell is None:
            return 0

        minLength = 2
        if Configuration.Settings.Tools.CheckOneLetterWords:
            minLength = 1

        wordsNotFound = 0
        words = HtmlUtil.remove_open_close_tags(line, HtmlUtil.TagItalic).split(" \r\n\t")
        for i in range(len(words)):
            word = words[i].strip(SpellCheckWordLists.SplitChars)
            if len(word) >= minLength:
                if not is_word_known_or_number(word, line):
                    correct = len(word) > 1 and _hunspell.spell(word)
                    if not correct:
                        correct = len(word) > 2 and _hunspell.spell(word.replace("'", ""))
                    if not correct and len(word) == 1 and _threeLetterIsoLanguageName == "eng" and (word == "I" or word == "A" or word == "a"):
                        correct = True
                    if correct:
                        out_numberOfCorrectWords += 1
                    else:
                        wordsNotFound += 1
                elif len(word) > 3:
                    out_numberOfCorrectWords += 1

        return wordsNotFound

    def remove_open_close_tags(source, tags):
        if not source or '<' not in source:
            return source

        # This pattern matches these tag formats:
        # <tag*>
        # < tag*>
        # </tag*>
        # < /tag*>
        # </ tag*>
        # < / tag*>
        return re.sub(r'<(\w+)>.*?</\1>', '', source, flags=re.IGNORECASE)

    # ... (rest of the code)

    def is_word_known_or_number(word, line):
        if re.match(r'^\d+(?:\.\d+)?$', word):
            return True

        if word in _wordSkipList:
            return True

        if word.strip("'") in _nameList:
            return True

        if word.strip("'").upper() in _nameListUppercase:
            return True

        if _spellCheckWordLists is not None and word.lower() in _spellCheckWordLists.user_words:
            return True

        if _spellCheckWordLists is not None and word.strip("'").lower() in _spellCheckWordLists.user_words:
            return True

        if len(word) > 2 and word.upper() in _nameListUppercase:
            return True

        if len(word) > 2 and word in _nameListWithApostrophe:
            return True

        if _nameListObj is not None and _nameListObj.is_in_names_multi_word_list(line, word):
            return True

        return False
