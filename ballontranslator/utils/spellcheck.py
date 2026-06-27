import os
import re
import weakref
from typing import List, Set
from qtpy.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from qtpy.QtCore import Qt

from . import shared

class SpellCheckManager:
    """SpellCheckManager manages spell checking, custom dictionaries, and suggestion retrieval.

    >>> manager = SpellCheckManager.get_instance()
    >>> isinstance(manager, SpellCheckManager)
    True
    """
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.spell = None
        self.custom_words: Set[str] = set()
        self.external_words: Set[str] = set()
        self.highlighters = weakref.WeakSet()
        self.dict_path = os.path.join(shared.PROGRAM_PATH, 'config', 'custom_dictionary.txt')
        self._load_custom_dictionary()
        self.load_external_dictionary()

    def _load_custom_dictionary(self):
        """Loads custom user-added words from local storage.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager._load_custom_dictionary()
        """
        if os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            self.custom_words.add(word)
            except Exception as e:
                print(f"Error loading custom dictionary: {e}")

    def _save_custom_dictionary(self):
        """Saves custom user-added words to local storage.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager._save_custom_dictionary()
        """
        try:
            os.makedirs(os.path.dirname(self.dict_path), exist_ok=True)
            with open(self.dict_path, 'w', encoding='utf-8') as f:
                for word in sorted(self.custom_words):
                    f.write(word + '\n')
        except Exception as e:
            print(f"Error saving custom dictionary: {e}")

    def _parse_and_load_file(self, path):
        try:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(path, 'r', encoding='cp1251', errors='ignore') as f:
                    lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '/' in line:
                    word = line.split('/', 1)[0].strip().lower()
                else:
                    word = line.lower()
                if word.isdigit():
                    continue
                if word:
                    self.external_words.add(word)
        except Exception as e:
            from ballontranslator.utils.logger import logger as LOGGER
            LOGGER.error(f"Error reading dictionary file {path}: {e}")

    def load_external_dictionary(self):
        """Loads external and repository dictionary files specified in program config.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.load_external_dictionary()
        """
        self.external_words.clear()
        try:
            from ballontranslator.utils.config import pcfg
            
            # Load enabled repository dictionaries
            repo_dicts = getattr(pcfg, 'spellcheck_repo_dicts', '').split(',')
            loaded_repos = 0
            for dict_name in repo_dicts:
                dict_name = dict_name.strip()
                if dict_name:
                    dict_path = os.path.join(shared.PROGRAM_PATH, 'data', 'dictionaries', dict_name)
                    if os.path.exists(dict_path):
                        self._parse_and_load_file(dict_path)
                        loaded_repos += 1
            
            # Load multiple external dictionaries
            ext_paths = getattr(pcfg, 'spellcheck_external_dict_path', '').split(';')
            loaded_externals = 0
            for path in ext_paths:
                path = path.strip()
                if path and os.path.exists(path):
                    self._parse_and_load_file(path)
                    loaded_externals += 1
                    
            from ballontranslator.utils.logger import logger as LOGGER
            LOGGER.info(
                f"SpellCheckManager: loaded {len(self.external_words)} words "
                f"(from {loaded_repos} repo dictionaries, {loaded_externals} external dictionaries)"
            )
        except Exception as e:
            from ballontranslator.utils.logger import logger as LOGGER
            LOGGER.error(f"Error loading dictionaries: {e}")
            
        for hl in list(self.highlighters):
            hl.clear_cache()

    def is_available(self) -> bool:
        """Checks if the required pyspellchecker package is installed.

        >>> manager = SpellCheckManager.get_instance()
        >>> isinstance(manager.is_available(), bool)
        True
        """
        try:
            import spellchecker
            return True
        except ImportError:
            return False

    def load_spellchecker(self):
        """Initializes the underlying SpellChecker instance with default dictionaries.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.load_spellchecker()
        """
        if self.spell is not None:
            return
        
        try:
            from spellchecker import SpellChecker
            # Load English and Russian dictionaries with fast distance=1
            self.spell = SpellChecker(language=['en', 'ru'], distance=1)
            from ballontranslator.utils.logger import logger as LOGGER
            LOGGER.info("SpellChecker initialized with languages: en, ru (distance=1)")
        except Exception as e:
            from ballontranslator.utils.logger import logger as LOGGER
            LOGGER.error(f"Failed to load SpellChecker: {e}")
            self.spell = None

    def is_correct(self, word: str) -> bool:
        """Checks if a given word is correctly spelled.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.is_correct("hello") in (True, False)
        True
        """
        word_lower = word.lower()
        if word_lower in self.custom_words or word_lower in self.external_words:
            return True
        
        # If it contains digits or special chars, ignore it
        if not word_lower.isalpha():
            return True

        if self.spell is None:
            self.load_spellchecker()
            if self.spell is None:
                return True # If spellchecker cannot be loaded, assume correct

        # pyspellchecker: known returns words that are in the dictionary
        try:
            return bool(self.spell.known([word_lower]))
        except Exception:
            return True

    def get_suggestions(self, word: str) -> List[str]:
        """Gets spelling suggestions for a misspelled word.

        >>> manager = SpellCheckManager.get_instance()
        >>> isinstance(manager.get_suggestions("helo"), list)
        True
        """
        if len(word) > 15:
            return []

        if self.spell is None:
            self.load_spellchecker()
            if self.spell is None:
                return []
        
        word_lower = word.lower()
        try:
            candidates = self.spell.candidates(word_lower)
        except Exception:
            return []
        if not candidates:
            return []
        
        # Sort or filter if needed, limit to 5
        suggestions = list(candidates)[:5]
        
        # Match casing if possible (e.g. capitalized)
        if word.istitle():
            suggestions = [s.capitalize() for s in suggestions]
        elif word.isupper():
            suggestions = [s.upper() for s in suggestions]
            
        return suggestions

    def add_to_dictionary(self, word: str):
        """Adds a word to the custom dictionary, and triggers re-highlighting.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.add_to_dictionary("customword")
        """
        word_lower = word.lower()
        if word_lower and word_lower not in self.custom_words:
            self.custom_words.add(word_lower)
            self._save_custom_dictionary()
            # Clear cache of all highlighters and trigger rehighlight
            for hl in list(self.highlighters):
                hl.clear_cache()

    def register_highlighter(self, highlighter):
        """Registers a SpellCheckHighlighter to track active highlighters.

        >>> manager = SpellCheckManager.get_instance()
        >>> hl = SpellCheckHighlighter(None)
        >>> manager.register_highlighter(hl)
        """
        self.highlighters.add(highlighter)

    def notify_config_changed(self):
        """Notifies all registered highlighters that config changed, clearing cache."""
        self.load_external_dictionary()


class SpellCheckHighlighter(QSyntaxHighlighter):
    """SpellCheckHighlighter highlights misspelled words using a red wavy underline.

    >>> hl = SpellCheckHighlighter(None)
    >>> isinstance(hl, SpellCheckHighlighter)
    True
    """
    def __init__(self, parent_document):
        super().__init__(parent_document)
        self.manager = SpellCheckManager.get_instance()
        self.manager.register_highlighter(self)
        self._cache = {}
        
        self.misspelled_format = QTextCharFormat()
        self.misspelled_format.setUnderlineColor(QColor(235, 75, 75)) # Sleek red
        self.misspelled_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

    def clear_cache(self):
        self._cache.clear()
        if self.document():
            self.rehighlight()

    def highlightBlock(self, text):
        from ballontranslator.utils.config import pcfg
        # Check if spellcheck is enabled in settings and if spellchecker package is installed
        if not getattr(pcfg, 'spellcheck_enabled', True):
            return
        if not self.manager.is_available():
            return

        # Pattern matches words in English and Russian
        for match in re.finditer(r'\b[a-zA-Zа-яА-ЯёЁ]+\b', text):
            word = match.group(0)
            if len(word) <= 1:
                continue
            
            word_lower = word.lower()
            if word_lower not in self._cache:
                self._cache[word_lower] = self.manager.is_correct(word_lower)
                
            if not self._cache[word_lower]:
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, self.misspelled_format)
