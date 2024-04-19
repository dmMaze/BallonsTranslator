from .base import *


"""
google translator API copied from https://pypi.org/project/deep-translator/
"""



from .constants import BASE_URLS, GOOGLE_LANGUAGES_TO_CODES, GOOGLE_LANGUAGES_SECONDARY_NAMES
from .exceptions import TooManyRequests, LanguageNotSupportedException, TranslationNotFound, NotValidPayload, RequestError, InvalidSourceOrTargetLanguage, NotValidLength
from bs4 import BeautifulSoup
import requests
from time import sleep
import warnings
import logging
from abc import ABC, abstractmethod
import string

class GoogleTransBase(ABC):
    """
    Abstract class that serve as a parent translator for other different translators
    """
    def __init__(self,
                 base_url=None,
                 source="auto",
                 target="en",
                 payload_key=None,
                 element_tag=None,
                 element_query=None,
                 **url_params):
        """
        @param source: source language to translate from
        @param target: target language to translate to
        """
        if source == target:
            raise InvalidSourceOrTargetLanguage(source)

        self.__base_url = base_url
        self._source = source
        self._target = target
        self._url_params = url_params
        self._element_tag = element_tag
        self._element_query = element_query
        self.payload_key = payload_key
        self.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) '
                                      'AppleWebit/535.19'
                                      '(KHTML, like Gecko) Chrome/18.0.1025.168 Safari/535.19'}
        super(GoogleTransBase, self).__init__()

    @staticmethod
    def _validate_payload(payload, min_chars=1, max_chars=5000):
        """
        validate the target text to translate
        @param payload: text to translate
        @return: bool
        """

        if not payload or not isinstance(payload, str) or not payload.strip() or payload.isdigit():
            raise NotValidPayload(payload)

        # check if payload contains only symbols
        if all(i in string.punctuation for i in payload):
            raise NotValidPayload(payload)

        if not GoogleTransBase.__check_length(payload, min_chars, max_chars):
            raise NotValidLength(payload, min_chars, max_chars)
        return True

    @staticmethod
    def __check_length(payload, min_chars, max_chars):
        """
        check length of the provided target text to translate
        @param payload: text to translate
        @param min_chars: minimum characters allowed
        @param max_chars: maximum characters allowed
        @return: bool
        """
        return True if min_chars <= len(payload) < max_chars else False

    @abstractmethod
    def translate(self, text, **kwargs):
        """
        translate a text using a translator under the hood and return the translated text
        @param text: text to translate
        @param kwargs: additional arguments
        @return: str
        """
        return NotImplemented('You need to implement the translate method!')


class GoogleTranslator(GoogleTransBase):
    """
    class that wraps functions, which use google translate under the hood to translate text(s)
    """
    _languages = GOOGLE_LANGUAGES_TO_CODES
    supported_languages = list(_languages.keys())

    def __init__(self, source="auto", target="en", proxies=None, **kwargs):
        """
        @param source: source language to translate from
        @param target: target language to translate to
        """
        self.__base_url = BASE_URLS.get("GOOGLE_TRANSLATE")
        self.proxies = proxies

        # code snipppet that converts the language into lower-case and skip lower-case conversion for abbreviations
        # since abbreviations like zh-CN if converted to lower-case will result into error
        #######################################
        source_lower = source
        target_lower = target
        if not source in self._languages.values():
            source_lower=source.lower()
        if not target in self._languages.values():
            target_lower=target.lower()
        #######################################

        # if self.is_language_supported(source_lower, target_lower):
        self._source, self._target = self._map_language_to_code(source_lower, target_lower)
        super(GoogleTranslator, self).__init__(base_url=self.__base_url,
                                               source=self._source,
                                               target=self._target,
                                               element_tag='div',
                                               element_query={"class": "t0"},
                                               payload_key='q',  # key of text in the url
                                               tl=self._target,
                                               sl=self._source,
                                               **kwargs)

        self._alt_element_query = {"class": "result-container"}

    @staticmethod
    def get_supported_languages(as_dict=False, **kwargs):
        """
        return the supported languages by the google translator
        @param as_dict: if True, the languages will be returned as a dictionary mapping languages to their abbreviations
        @return: list or dict
        """
        return GoogleTranslator.supported_languages if not as_dict else GoogleTranslator._languages

    def is_secondary(self, lang):
        """
        Function to check if lang is a secondary name of any primary language
        @param lang: language name
        @return: primary name of a language if found otherwise False
        """
        for primary_name, secondary_names in GOOGLE_LANGUAGES_SECONDARY_NAMES.items():
            if lang in secondary_names:
                return primary_name
        return False

    def _map_language_to_code(self, *languages):
        """
        map language to its corresponding code (abbreviation) if the language was passed by its full name by the user
        @param languages: list of languages
        @return: mapped value of the language or raise an exception if the language is not supported
        """
        for language in languages:
            if language in self._languages.values() or language == 'auto':
                yield language
            elif language in self._languages.keys():
                yield self._languages[language]
            else:
                yield self._languages[self.is_secondary(language)]

    def is_language_supported(self, *languages):
        """
        check if the language is supported by the translator
        @param languages: list of languages
        @return: bool or raise an Exception
        """
        for lang in languages:
            if lang != 'auto' and lang not in self._languages.keys():
                if lang != 'auto' and lang not in self._languages.values():
                    if not self.is_secondary(lang):
                        raise LanguageNotSupportedException(lang)
        return True

    def translate(self, text, **kwargs):
        """
        function that uses google translate to translate a text
        @param text: desired text to translate
        @return: str: translated text
        """

        if self._validate_payload(text):
            text = text.strip()

            if self.payload_key:
                self._url_params[self.payload_key] = text
            response = requests.get(self.__base_url,
                                    params=self._url_params,
                                    proxies=self.proxies)
            if response.status_code == 429:
                raise TooManyRequests()

            if response.status_code != 200:
                raise RequestError()

            soup = BeautifulSoup(response.text, 'html.parser')

            element = soup.find(self._element_tag, self._element_query)

            if not element:
                element = soup.find(self._element_tag, self._alt_element_query)
                if not element:
                    raise TranslationNotFound(text)
            if element.get_text(strip=True) == text.strip():
                to_translate_alpha = ''.join(ch for ch in text.strip() if ch.isalnum())
                translated_alpha = ''.join(ch for ch in element.get_text(strip=True) if ch.isalnum())
                if to_translate_alpha and translated_alpha and to_translate_alpha == translated_alpha:
                    self._url_params["tl"] = self._target
                    if "hl" not in self._url_params:
                        return text.strip()
                    del self._url_params["hl"]
                    return self.translate(text)

            else:
                return element.get_text(strip=True)

    def translate_file(self, path, **kwargs):
        """
        translate directly from file
        @param path: path to the target file
        @type path: str
        @param kwargs: additional args
        @return: str
        """
        try:
            with open(path) as f:
                text = f.read().strip()
            return self.translate(text)
        except Exception as e:
            raise e

    def translate_sentences(self, sentences=None, **kwargs):
        """
        translate many sentences together. This makes sense if you have sentences with different languages
        and you want to translate all to unified language. This is handy because it detects
        automatically the language of each sentence and then translate it.

        @param sentences: list of sentences to translate
        @return: list of all translated sentences
        """
        warnings.warn("deprecated. Use the translate_batch function instead", DeprecationWarning, stacklevel=2)
        logging.warning("deprecated. Use the translate_batch function instead")
        if not sentences:
            raise NotValidPayload(sentences)

        translated_sentences = []
        try:
            for sentence in sentences:
                translated = self.translate(text=sentence)
                translated_sentences.append(translated)

            return translated_sentences

        except Exception as e:
            raise e

    def translate_batch(self, batch=None, **kwargs):
        """
        translate a list of texts
        @param batch: list of texts you want to translate
        @return: list of translations
        """
        if not batch:
            raise Exception("Enter your text list that you want to translate")
        arr = []
        for i, text in enumerate(batch):

            translated = self.translate(text, **kwargs)
            arr.append(translated)
        return arr


@register_translator('google')
class TransGoogle(BaseTranslator):

    concate_text = True
    params: Dict = {
        'delay': 0.0,
    }
    
    def _setup_translator(self):
        self.lang_map['简体中文'] = 'zh-CN'
        self.lang_map['繁體中文'] = 'zh-TW'
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        self.lang_map['한국어'] = 'ko'
        self.lang_map['Tiếng Việt'] = 'vi'
        self.lang_map['čeština'] = 'cs'
        self.lang_map['Nederlands'] = 'nl'
        self.lang_map['Français'] = 'fr'
        self.lang_map['Deutsch'] = 'de'
        self.lang_map['magyar nyelv'] = 'hu'
        self.lang_map['Italiano'] = 'it'
        self.lang_map['Polski'] = 'pl'
        self.lang_map['Português'] = 'pt'
        self.lang_map['Limba română'] = 'ro'
        self.lang_map['русский язык'] = 'ru'
        self.lang_map['Español'] = 'es'
        self.lang_map['Türk dili'] = 'tr'
        self.lang_map['Indonesia'] = 'id'
        self.lang_map['Thai'] = 'th'
        self.lang_map['Arabic'] = 'ar'
        self.lang_map['Malayalam'] = 'ml'
        self.lang_map['Tamil'] = 'ta'
        self.lang_map['Hindi'] = 'hi'

        self.googletrans = GoogleTranslator()
        
    def _translate(self, src_list: List[str]) -> List[str]:
        
        self.googletrans._source = self.lang_map[self.lang_source]
        self.googletrans._url_params['sl'] = self.lang_map[self.lang_source]
        self.googletrans._target = self.lang_map[self.lang_target]
        self.googletrans._url_params['tl'] = self.lang_map[self.lang_target]
        self.googletrans.__base_url = "https://translate.google.com/m"
        translations = [self.googletrans.translate(t) for t in src_list]

        return translations
