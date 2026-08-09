import os
import re
import weakref
import threading
from typing import List, Set

from qtpy.QtCore import Qt, QTimer, QThread, Signal, QSize, QObject
from qtpy.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
from qtpy.QtWidgets import (
    QListWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLineEdit, QDialog, QLabel, QWidget, QListWidgetItem, QTextEdit
)

from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.download_util import download_url_to_file
from ballontranslator.utils.logger import logger as LOGGER

# Predefined dictionary repository URLs mapping language names to LibreOffice raw URLs.
DICTIONARY_URLS = {
    "Arabic (ar)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ar/ar.dic",
    "Belarusian (be_BY)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/be_BY/be-official.dic",
    "Bulgarian (bg_BG)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/bg_BG/bg_BG.dic",
    "Bosnian (bs_BA)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/bs_BA/bs_BA.dic",
    "Catalan (ca)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ca/dictionaries/ca.dic",
    "Czech (cs_CZ)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/cs_CZ/cs_CZ.dic",
    "Danish (da_DK)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/da_DK/da_DK.dic",
    "German (de_DE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/de/de_DE_frami.dic",
    "Greek (el_GR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/el_GR/el_GR.dic",
    "English (en_US)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_US.dic",
    "English (en_GB)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/en/en_GB.dic",
    "Spanish (es_ES)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/es/es_ES.dic",
    "Estonian (et_EE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/et_EE/et_EE.dic",
    "Persian (fa_IR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fa_IR/fa-IR.dic",
    "French (fr_FR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/fr_FR/fr.dic",
    "Croatian (hr_HR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/hr_HR/hr_HR.dic",
    "Hungarian (hu_HU)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/hu_HU/hu_HU.dic",
    "Indonesian (id)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/id/id_ID.dic",
    "Icelandic (is)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/is/is.dic",
    "Italian (it_IT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/it_IT/it_IT.dic",
    "Korean (ko_KR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ko_KR/ko_KR.dic",
    "Lithuanian (lt_LT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/lt_LT/lt.dic",
    "Latvian (lv_LV)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/lv_LV/lv_LV.dic",
    "Dutch (nl_NL)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/nl_NL/nl_NL.dic",
    "Polish (pl_PL)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pl_PL/pl_PL.dic",
    "Portuguese (pt_BR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_BR/pt_BR.dic",
    "Portuguese (pt_PT)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/pt_PT/pt_PT.dic",
    "Romanian (ro_RO)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ro/ro_RO.dic",
    "Russian (ru_RU)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/ru_RU/ru_RU.dic",
    "Russian (Large - all inflections) (ru_RU)": "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt",
    "Slovak (sk_SK)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sk_SK/sk_SK.dic",
    "Slovenian (sl_SI)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sl_SI/sl_SI.dic",
    "Swedish (sv_SE)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/sv_SE/dictionaries/sv_SE.dic",
    "Turkish (tr_TR)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/tr_TR/tr_TR.dic",
    "Ukrainian (uk_UA)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/uk_UA/uk_UA.dic",
    "Vietnamese (vi)": "https://raw.githubusercontent.com/LibreOffice/dictionaries/master/vi/vi_VN.dic"
}


class SpellCheckManager(QObject):
    """SpellCheckManager manages spell checking, custom dictionaries, and suggestion retrieval.

    >>> manager = SpellCheckManager.get_instance()
    >>> isinstance(manager, SpellCheckManager)
    True
    """
    dicts_loaded = Signal()
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        super().__init__()
        self.spell = None
        self._is_available = None
        self.custom_words: Set[str] = set()
        self.external_words: Set[str] = set()
        self.highlighters = weakref.WeakSet()
        self.dict_path = os.path.join(shared.PROGRAM_PATH, 'config', 'custom_dictionary.txt')
        self._reload_queued = False
        self._load_custom_dictionary()
        self.dicts_loaded.connect(self.on_dicts_loaded)
        self.load_spellchecker_async()

    def on_dicts_loaded(self):
        for hl in list(self.highlighters):
            hl.clear_cache()
            hl.rehighlight()

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

    def _parse_and_load_file_to_set(self, path, target_set):
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
                    target_set.add(word)
        except Exception as e:
            LOGGER.error(f"Error reading dictionary file {path}: {e}")

    def load_external_dictionary(self):
        """Loads external and repository dictionary files specified in program config.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.load_external_dictionary()
        """
        temp_words = set()
        try:
            # Load enabled repository dictionaries
            repo_dicts = getattr(pcfg, 'spellcheck_repo_dicts', '').split(',')
            loaded_repos = 0
            for dict_name in repo_dicts:
                dict_name = dict_name.strip()
                if dict_name:
                    dict_path = os.path.join(shared.PROGRAM_PATH, 'data', 'dictionaries', dict_name)
                    if os.path.exists(dict_path):
                        self._parse_and_load_file_to_set(dict_path, temp_words)
                        loaded_repos += 1
            
            # Load multiple external dictionaries
            ext_paths = getattr(pcfg, 'spellcheck_external_dict_path', '').split(';')
            loaded_externals = 0
            for path in ext_paths:
                path = path.strip()
                if path and os.path.exists(path):
                    self._parse_and_load_file_to_set(path, temp_words)
                    loaded_externals += 1
                    
            LOGGER.info(
                f"SpellCheckManager: loaded {len(temp_words)} words "
                f"(from {loaded_repos} repo dictionaries, {loaded_externals} external dictionaries)"
            )
        except Exception as e:
            LOGGER.error(f"Error loading dictionaries: {e}")
            
        self.external_words = temp_words
        for hl in list(self.highlighters):
            hl.clear_cache()

    def load_spellchecker_async(self):
        if not getattr(pcfg, 'spellcheck_enabled', True) or not self.is_available():
            if self.spell is not None or self.external_words:
                self.spell = None
                self.external_words = set()
                LOGGER.info("Spell Checker is disabled or not available. Unloaded dictionaries.")
                for hl in list(self.highlighters):
                    hl.clear_cache()
                    hl.rehighlight()
            return

        if hasattr(self, '_loading_thread') and self._loading_thread and self._loading_thread.is_alive():
            self._reload_queued = True
            return
        
        self._reload_queued = False

        def bg_load():
            while True:
                self._reload_queued = False
                distance = getattr(pcfg, 'spellcheck_distance', 1)

                if not getattr(pcfg, 'spellcheck_enabled', True):
                    self.spell = None
                    self.external_words = set()
                    return

                if self.spell is None or getattr(self.spell, '_distance', 1) != distance:
                    try:
                        from spellchecker import SpellChecker
                        self.spell = SpellChecker(language=['en', 'ru'], distance=distance)
                        LOGGER.info(f"SpellChecker initialized in background (distance={distance})")
                    except Exception as e:
                        LOGGER.error(f"Failed to load SpellChecker in background: {e}")

                temp_words = set()
                try:
                    repo_dicts = getattr(pcfg, 'spellcheck_repo_dicts', '').split(',')
                    loaded_repos = 0
                    for dict_name in repo_dicts:
                        dict_name = dict_name.strip()
                        if dict_name:
                            dict_path = os.path.join(shared.PROGRAM_PATH, 'data', 'dictionaries', dict_name)
                            if os.path.exists(dict_path):
                                self._parse_and_load_file_to_set(dict_path, temp_words)
                                loaded_repos += 1

                    ext_paths = getattr(pcfg, 'spellcheck_external_dict_path', '').split(';')
                    loaded_externals = 0
                    for path in ext_paths:
                        path = path.strip()
                        if path and os.path.exists(path):
                            self._parse_and_load_file_to_set(path, temp_words)
                            loaded_externals += 1

                    if not getattr(pcfg, 'spellcheck_enabled', True):
                        self.spell = None
                        self.external_words = set()
                        LOGGER.info("Spell Checker is disabled. Cancelled dictionary loading.")
                        return

                    LOGGER.info(
                        f"SpellCheckManager: loaded {len(temp_words)} words in background "
                        f"(from {loaded_repos} repo dictionaries, {loaded_externals} external dictionaries)"
                    )
                except Exception as e:
                    LOGGER.error(f"Error loading dictionaries in background: {e}")

                self.external_words = temp_words

                self.dicts_loaded.emit()

                if not self._reload_queued:
                    break

        self._loading_thread = threading.Thread(target=bg_load, daemon=True)
        self._loading_thread.start()

    def is_available(self) -> bool:
        """Checks if the required pyspellchecker package is installed.

        >>> manager = SpellCheckManager.get_instance()
        >>> isinstance(manager.is_available(), bool)
        True
        """
        if self._is_available is not None:
            return self._is_available
        try:
            import spellchecker
            self._is_available = True
            return True
        except ImportError:
            self._is_available = False
            return False

    def load_spellchecker(self):
        """Initializes the underlying SpellChecker instance with default dictionaries.

        >>> manager = SpellCheckManager.get_instance()
        >>> manager.load_spellchecker()
        """
        distance = getattr(pcfg, 'spellcheck_distance', 1)
        if self.spell is not None:
            if getattr(self.spell, '_distance', 1) == distance:
                return
            else:
                self.spell = None
        
        try:
            from spellchecker import SpellChecker
            # Load English and Russian dictionaries with configured distance
            self.spell = SpellChecker(language=['en', 'ru'], distance=distance)
            LOGGER.info(f"SpellChecker initialized with languages: en, ru (distance={distance})")
        except Exception as e:
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
            self.load_spellchecker_async()
            return True # If spellchecker cannot be loaded or is loading, assume correct

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
            self.load_spellchecker_async()
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
                hl.rehighlight()

    def register_highlighter(self, highlighter):
        """Registers a SpellCheckHighlighter to track active highlighters.

        >>> manager = SpellCheckManager.get_instance()
        >>> hl = SpellCheckHighlighter(None)
        >>> manager.register_highlighter(hl)
        """
        self.highlighters.add(highlighter)

    def notify_config_changed(self):
        """Notifies all registered highlighters that config changed, clearing cache."""
        for hl in list(self.highlighters):
            hl.clear_cache()
        self.load_spellchecker_async()

    def install_pyspellchecker(self, parent) -> bool:
        """Prompts the user to install pyspellchecker and handles the installation process.

        Returns True if the package is installed successfully, False otherwise.
        """
        from qtpy.QtWidgets import QMessageBox, QProgressDialog
        reply = QMessageBox.question(
            parent,
            parent.tr("Install Dependency"),
            parent.tr("The required package 'pyspellchecker' is not installed. Would you like to install it now?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return False

        from ballontranslator.ui.packageinstall_thread import PackageInstallThread

        progress = QProgressDialog(parent.tr("Installing pyspellchecker..."), None, 0, 0, parent)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setCancelButton(None)

        thread = PackageInstallThread(parent)
        success = False

        def on_finished():
            nonlocal success
            progress.close()
            if thread.last_success:
                import importlib
                importlib.invalidate_caches()
                self._is_available = True
                QMessageBox.information(
                    parent,
                    parent.tr("Installation Complete"),
                    parent.tr("Package 'pyspellchecker' installed successfully!")
                )
                success = True
            else:
                err_msg = str(thread.last_error) if thread.last_error else parent.tr("Unknown error")
                QMessageBox.warning(
                    parent,
                    parent.tr("Installation Failed"),
                    parent.tr("Failed to install 'pyspellchecker':\n") + err_msg
                )

        thread.finish_install.connect(on_finished)
        thread.installPackages(['pyspellchecker'])
        progress.exec_()
        thread.wait()

        return success


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
        try:
            if self.document():
                self.rehighlight()
        except RuntimeError:
            pass

    def rehighlight(self):
        try:
            super().rehighlight()
        except RuntimeError:
            pass

    def highlightBlock(self, text):
        # Check if spellcheck is enabled in settings and if spellchecker package is installed
        if not getattr(pcfg, 'spellcheck_enabled', True):
            return
        if not self.manager.is_available():
            return

        # Check if editor is a Source block editor and if spell checking on source is enabled
        editor = self.parent()
        if editor and not isinstance(editor, QTextEdit):
            # Fallback if document was passed
            doc_parent = editor.parent()
            if doc_parent and isinstance(doc_parent, QTextEdit):
                editor = doc_parent
        
        if editor and isinstance(editor, QTextEdit):
            editor_class = editor.__class__.__name__
            if editor_class == 'SourceTextEdit':
                if not getattr(pcfg, 'spellcheck_on_source_enabled', False):
                    return

        # Pattern matches words in any language using Unicode character classes
        pattern = r'\b[^\W\d_]+\b'
        for match in re.finditer(pattern, text):
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


class DictDownloadThread(QThread):
    """QThread to handle dictionary downloads in the background.

    >>> thread = DictDownloadThread("http://example.com/dict.dic", "save/path.dic")
    >>> isinstance(thread, DictDownloadThread)
    True
    """
    progress = Signal(int, int)
    finished = Signal(bool, str)

    def __init__(self, url, save_path):
        super().__init__()
        self.url = url
        self.save_path = save_path
        self.cancel_event = threading.Event()

    def run(self):
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            download_url_to_file(
                self.url,
                self.save_path,
                progress_callback=self._progress_callback,
                cancel_event=self.cancel_event
            )
            self.finished.emit(True, "")
        except Exception as e:
            self.finished.emit(False, str(e))

    def _progress_callback(self, payload):
        if payload.get('event') == 'file_progress':
            self.progress.emit(payload.get('downloaded', 0), payload.get('total', 0))

    def cancel(self):
        self.cancel_event.set()


class WordListItemWidget(QWidget):
    """Custom QWidget representing a single custom word in the manager dialog.

    >>> widget = WordListItemWidget("hello")
    >>> isinstance(widget, WordListItemWidget)
    True
    """
    delete_requested = Signal(str)

    def __init__(self, word: str, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.word = word
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(12)

        self.label = QLabel(word, self)
        self.label.setStyleSheet("font-family: 'Segoe UI', Arial; font-size: 13px; font-weight: 500; background-color: transparent;")

        self.delete_btn = QPushButton("×", self)
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setToolTip(self.tr("Delete word"))
        self.delete_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #ff4d4d;
                border: none;
                font-size: 18px;
                font-weight: bold;
                padding: 0px;
                min-width: 24px;
                max-width: 24px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                color: #ff1a1a;
                background-color: rgba(255, 77, 77, 12%);
                border-radius: 4px;
            }
        """)
        self.delete_btn.clicked.connect(self._request_delete)
        self.delete_btn.setVisible(False)
        self.delete_btn.setAutoDefault(False)
        self.delete_btn.setDefault(False)

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.delete_btn)

    def _request_delete(self, _checked: bool = False) -> None:
        self.delete_requested.emit(self.word)

    def enterEvent(self, event):
        self.delete_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.delete_btn.setVisible(False)
        super().leaveEvent(event)


class _AddWordLineEdit(QLineEdit):
    add_requested = Signal()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.add_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class AddWordItemWidget(QWidget):
    """Custom QWidget containing input line and add button for entering custom words.

    >>> widget = AddWordItemWidget()
    >>> isinstance(widget, AddWordItemWidget)
    True
    """
    word_added = Signal(str)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(12)

        self.input_field = _AddWordLineEdit(self)
        self.input_field.setPlaceholderText(self.tr("Add new word..."))
        self.input_field.setFixedHeight(26)
        self.input_field.setStyleSheet("font-family: 'Segoe UI', Arial; font-size: 13px;")
        
        self.input_field.add_requested.connect(self.trigger_add)

        self.add_btn = QPushButton("+", self)
        self.add_btn.setFixedSize(26, 26)
        self.add_btn.setToolTip(self.tr("Add word"))
        self.add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_btn.setAutoDefault(False)
        self.add_btn.setDefault(False)
        self.add_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b3d40;
                color: #ffffff;
                border: 1px solid #45474a;
                border-radius: 4px;
                padding: 0px;
                min-width: 26px;
                max-width: 26px;
                min-height: 26px;
                max-height: 26px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1e93e5;
                border-color: #1e93e5;
            }
        """)
        self.add_btn.clicked.connect(self.trigger_add)

        layout.addWidget(self.input_field)
        layout.addWidget(self.add_btn)

    def trigger_add(self, _checked: bool = False) -> None:
        word = self.input_field.text().strip().lower()
        if word:
            self.word_added.emit(word)
            self.input_field.clear()
            self.input_field.setFocus()


class DictionaryManagerDialog(QDialog):
    """Dialog allowing users to manage their custom dictionary list.

    >>> dialog = DictionaryManagerDialog(None)
    >>> isinstance(dialog, DictionaryManagerDialog)
    True
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Custom Dictionary Manager"))
        self.resize(450, 520)

        self.manager = SpellCheckManager.get_instance()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.list_widget.setStyleSheet("""
            QListWidget {
                outline: 0;
            }
            QListWidget::item {
                background-color: transparent;
            }
            QListWidget::item:selected {
                background-color: transparent;
                color: inherit;
            }
            QListWidget::item:hover {
                background-color: rgba(255, 255, 255, 6%);
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.list_widget)

        self.close_btn = QPushButton(self.tr("Close"), self)
        self.close_btn.setFixedHeight(32)
        self.close_btn.setAutoDefault(False)
        self.close_btn.setDefault(False)
        self.close_btn.clicked.connect(self.accept)
        layout.addWidget(self.close_btn)

        self.populate_list()

    def populate_list(self):
        self.list_widget.clear()

        for word in sorted(self.manager.custom_words):
            item = QListWidgetItem(self.list_widget)
            widget = WordListItemWidget(word, self)
            widget.delete_requested.connect(self.delete_word)
            item.setSizeHint(QSize(0, 36))
            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, widget)

        input_item = QListWidgetItem(self.list_widget)
        input_item.setFlags(input_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        input_widget = AddWordItemWidget(self)
        input_widget.word_added.connect(self.add_word)
        input_item.setSizeHint(QSize(0, 36))
        self.list_widget.addItem(input_item)
        self.list_widget.setItemWidget(input_item, input_widget)

    def add_word(self, word):
        word_lower = word.lower()
        if word_lower and word_lower not in self.manager.custom_words:
            self.manager.custom_words.add(word_lower)
            self.populate_list()

    def delete_word(self, word):
        word_lower = word.lower()
        self.manager.custom_words.discard(word_lower)
        self.populate_list()

    def done(self, r):
        super().done(r)
        self.manager._save_custom_dictionary()
        self.manager.notify_config_changed()
        for hl in list(self.manager.highlighters):
            hl.clear_cache()
            hl.rehighlight()
