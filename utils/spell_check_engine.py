from spylls.hunspell.dictionary import Dictionary
from utils.logger import logger as LOGGER
from typing import List
import os
import json
from utils.download_util import download_and_check_files

en_flist = [
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/en/index.aff',
        'sha256_pre_calculated': ['8ae1f19d4840d957728ad90555d5a8dff6cc5c046279c95ff0c00fc0a0136c7b'],
        'files': 'data/spellcheck/en/index.aff'
    },
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/en/index.dic',
        'sha256_pre_calculated': ['f0b1a234bd178bdd01875b2a392a9647f888b8fe879f79c52aae62c2759b3647'],
        'files': 'data/spellcheck/en/index.dic'
    },
]
fr_flist = [
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/fr/index.aff',
        'sha256_pre_calculated': ['05a735d34c912e4e381ff08ee7c747923ccf5cf9dca81d8467982fa1ca51c2b7'],
        'files': 'data/spellcheck/fr/index.aff'
    },
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/fr/index.dic',
        'sha256_pre_calculated': ['984e933237bc1224a48f42828233be9b03228260ef67aa8e2bdddcd03a26230d'],
        'files': 'data/spellcheck/fr/index.dic'
    },
]
it_flist = [
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/it/index.aff',
        'sha256_pre_calculated': ['5770cd3e16d494c045b4a9a4a9fcd7962577e642d0384a7129c020a12cdd2c79'],
        'files': 'data/spellcheck/it/index.aff'
    },
    {
        'url': 'https://github.com/wooorm/dictionaries/raw/refs/heads/main/dictionaries/it/index.dic',
        'sha256_pre_calculated': ['b1348fbdb6f441ea9dd7e33b2cfcb96ead39ccd5e48bf894972774cd5aa86abb'],
        'files': 'data/spellcheck/it/index.dic'
    },
]

class SpellCheckEngine:
    def __init__(self, lang = 'en') -> None:
        self.logger = LOGGER

        if lang == 'en':
            for files_download_kwargs in en_flist:
                download_and_check_files(**files_download_kwargs)
            self.dictionary = Dictionary.from_files('data/spellcheck/en/index')
        elif lang == 'fr':
            for files_download_kwargs in fr_flist:
                download_and_check_files(**files_download_kwargs)
            self.dictionary = Dictionary.from_files('data/spellcheck/fr/index')
        elif lang == 'it':
            for files_download_kwargs in it_flist:
                download_and_check_files(**files_download_kwargs)
            self.dictionary = Dictionary.from_files('data/spellcheck/it/index')

        # Define the path for saving/loading data
        self.data_file = f"SpellCheckEngine_{lang}.json"
        self._load_data()

    def _load_data(self):
        """Loads skipped_words and replace_words from the JSON file."""
        default_skipped = []
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)                    
                    self.skipped_words = data.get('skipped_words', default_skipped)
                self.logger.info(f"Successfully loaded data from {self.data_file}")
                
            except json.JSONDecodeError:
                self.logger.error(f"Error decoding JSON from {self.data_file}. Using default values.")
                self.skipped_words = default_skipped
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while loading {self.data_file}: {e}. Using defaults.")
                self.skipped_words = default_skipped
        else:
            self.logger.info(f"Data file {self.data_file} not found. Initializing with defaults.")
            self.skipped_words = default_skipped

    def _save_data(self):
        """Saves the current skipped_words and replace_words to the JSON file."""
        data_to_save = {
            'skipped_words': self.skipped_words,
        }        
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Successfully saved data to {self.data_file}")
        except IOError as e:
            self.logger.error(f"Failed to save data to {self.data_file}: {e}")

    def DoSuggest(self, word: str):
        return self.dictionary.suggest(word)

    def GetUnknownWordsViaDictionaryFromList(self, words_with_objects: List) -> list:
        split_chars = set('!?,:.\"\'();')
        unknown_words = []
        for item_tuple in words_with_objects:
            text_content, textblock_obj = item_tuple # Unpack the tuple

            for word in text_content.split():
                word = word.strip(''.join(split_chars))

                if (word in self.skipped_words):
                    self.logger.debug(f'word {word} skipped')
                    continue
                if self.is_number(word):
                    self.logger.debug(f'number {word} skipped')
                    continue
                if not self.dictionary.lookup(word):
                    unknown_words.append((word, textblock_obj))
        return unknown_words  

    def onWordDeleted(self, word: str):
        self.skipped_words.append(word)
        self._save_data()

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