# This module is a Python port of the TypeScript library: https://github.com/FOSWLY/translate
# It integrates multiple translation services into a single module for BallonTranslator.

import time
import json
import requests
import re
from typing import Dict, List, Optional

from .base import BaseTranslator, register_translator
from utils.logger import logger as LOGGER


# --- Custom Exceptions ---
class ProviderError(Exception):
    """Base exception for provider-related errors."""

    pass


class TranslateError(ProviderError):
    """Exception for translation failures."""

    pass


# --- Internal Provider Classes (Ported from TypeScript library) ---


class FOSWLYProviderBase:
    """A base class for internal providers to share common logic like requests session."""

    def __init__(self, session_opts: Dict = None):
        self.session = requests.Session()
        if session_opts:
            self.session.headers.update(session_opts.get("headers", {}))

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        raise NotImplementedError

    def _request(self, url, method="POST", **kwargs):
        try:
            response = self.session.request(method, url, timeout=15, **kwargs)

            if response.status_code != 200:
                LOGGER.error(
                    f"[{self.__class__.__name__}] HTTP {response.status_code}: {response.reason}. Response: {response.text[:200]}"
                )
                raise ProviderError(f"HTTP {response.status_code} {response.reason}")

            content_type = response.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                LOGGER.error(
                    f"[{self.__class__.__name__}] Unexpected Content-Type: {content_type}. Raw response: {response.text[:500]}"
                )
                raise ProviderError(
                    f"Unexpected server response format. Expected JSON, got {content_type}."
                )

            return response.json()

        except requests.exceptions.RequestException as e:
            LOGGER.error(f"[{self.__class__.__name__}] Request failed: {e}")
            raise ProviderError(f"Request failed: {e}")
        except json.JSONDecodeError:
            raw_text = (
                response.text[:200] if hasattr(response, "text") else "NoResponseObject"
            )
            LOGGER.error(
                f"[{self.__class__.__name__}] Failed to decode JSON. Raw response: {raw_text}"
            )
            raise ProviderError("Failed to decode JSON response from API.")


class YandexBrowserProvider(FOSWLYProviderBase):
    """Ported logic from src/providers/yandexbrowser.ts"""

    def __init__(self, **kwargs):
        super().__init__(
            {
                "headers": {
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 YaBrowser/24.7.0.0 Safari/537.36",
                }
            }
        )
        self.api_url = "https://browser.translate.yandex.net/api/v1/tr.json"
        self.srv = "browser_video_translation"

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        lang = f"{from_lang}-{to_lang}" if from_lang != "auto" else to_lang
        params = {"srv": self.srv, "lang": lang, "text": text, "format": "plain"}
        response = self._request(
            f"{self.api_url}/translate", method="GET", params=params
        )
        if response.get("code") == 200:
            return response.get("text", [""])[0]
        raise TranslateError(response.get("message", "Yandex.Browser API error"))


class YandexCloudProvider(FOSWLYProviderBase):
    """Ported logic from src/providers/yandexcloud.ts"""

    def __init__(self, **kwargs):
        super().__init__({"headers": {"Content-Type": "application/json"}})
        self.api_url = "https://cloud.yandex.ru/api/translate"
        self.api_key = kwargs.get("api_key")

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        body = {
            "sourceLanguageCode": from_lang,
            "targetLanguageCode": to_lang,
            "texts": [text],
        }
        if self.api_key:
            headers = {"Authorization": f"Api-Key {self.api_key}"}
            api_url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
            response = self._request(api_url, json=body, headers=headers)
        else:
            response = self._request(f"{self.api_url}/translate", json=body)

        if "translations" in response and response["translations"]:
            return response["translations"][0].get("text", "")
        raise TranslateError(response.get("message", "Yandex.Cloud API error"))


class MSEdgeTranslateProvider(FOSWLYProviderBase):
    """Ported logic from src/providers/msedge.ts"""

    def __init__(self, **kwargs):
        super().__init__({"headers": {"Content-Type": "application/json"}})
        self.api_url = "https://api-edge.cognitive.microsofttranslator.com"
        self.session_url = "https://edge.microsoft.com"
        self.token, self.token_timestamp = None, 0

    def _get_token(self):
        if (
            self.token and (time.time() - self.token_timestamp) < 580
        ):  # 10 min expiry with 20s buffer
            return self.token
        try:
            res = self.session.get(f"{self.session_url}/translate/auth", timeout=10)
            res.raise_for_status()
            self.token, self.token_timestamp = res.text, time.time()
            return self.token
        except Exception as e:
            raise ProviderError(f"Failed to get MSEdge token: {e}")

    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        params = {"to": to_lang, "api-version": "3.0"}
        if from_lang != "auto":
            params["from"] = from_lang
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        body = [{"Text": text}]
        response = self._request(
            f"{self.api_url}/translate", params=params, headers=headers, json=body
        )
        if response and response[0].get("translations"):
            return response[0]["translations"][0].get("text", "")
        raise TranslateError(
            response.get("error", {}).get("message", "MSEdge API error")
        )


# === Main Translator Class for Ballon Translator ===


@register_translator("Yandex-FOSWLY")
class YandexFOSWLYTranslator(BaseTranslator):
    """
    Integrates multiple translation services from the FOSWLY/translate library.
    Original TypeScript library: https://github.com/FOSWLY/translate
    """

    concate_text = True
    params: Dict = {
        "service": {
            "type": "selector",
            "options": ["YandexBrowser", "YandexCloud", "MSEdge"],
            "value": "YandexBrowser",
            "description": "Select the translation service from the FOSWLY library.",
        },
        "yandex_cloud_api_key": {
            "value": "",
            "description": "API Key for Yandex.Cloud. If empty, a keyless method will be attempted.",
        },
        "delay": 0.1,
    }

    def _setup_translator(self):
        # A comprehensive language map based on Google's list
        self.lang_map = {
            "Auto": "auto",
            "Afrikaans": "af",
            "Albanian": "sq",
            "Amharic": "am",
            "Arabic": "ar",
            "Armenian": "hy",
            "Assamese": "as",
            "Azerbaijani": "az",
            "Bangla": "bn",
            "Basque": "eu",
            "Belarusian": "be",
            "Bengali": "bn",
            "Bosnian": "bs",
            "Breton": "br",
            "Bulgarian": "bg",
            "Burmese": "my",
            "Catalan": "ca",
            "Cebuano": "ceb",
            "Cherokee": "chr",
            "简体中文": "zh",
            "繁體中文": "zh-TW",
            "Corsican": "co",
            "Croatian": "hr",
            "čeština": "cs",
            "Danish": "da",
            "Nederlands": "nl",
            "English": "en",
            "Esperanto": "eo",
            "Estonian": "et",
            "Faroese": "fo",
            "Filipino": "fil",
            "Finnish": "fi",
            "Français": "fr",
            "Frisian": "fy",
            "Galician": "gl",
            "Georgian": "ka",
            "Deutsch": "de",
            "Greek": "el",
            "Gujarati": "gu",
            "Haitian Creole": "ht",
            "Hausa": "ha",
            "Hawaiian": "haw",
            "Hebrew": "he",
            "Hindi": "hi",
            "Hmong": "hmn",
            "magyar nyelv": "hu",
            "Icelandic": "is",
            "Igbo": "ig",
            "Indonesian": "id",
            "Interlingua": "ia",
            "Irish": "ga",
            "Italiano": "it",
            "日本語": "ja",
            "Javanese": "jv",
            "Kannada": "kn",
            "Kazakh": "kk",
            "Khmer": "km",
            "한국어": "ko",
            "Kurdish": "ku",
            "Kyrgyz": "ky",
            "Lao": "lo",
            "Latin": "la",
            "Latvian": "lv",
            "Lithuanian": "lt",
            "Luxembourgish": "lb",
            "Macedonian": "mk",
            "Malagasy": "mg",
            "Malay": "ms",
            "Malayalam": "ml",
            "Maltese": "mt",
            "Maori": "mi",
            "Marathi": "mr",
            "Mongolian": "mn",
            "Nepali": "ne",
            "Norwegian": "no",
            "Occitan": "oc",
            "Oriya": "or",
            "Pashto": "ps",
            "Persian": "fa",
            "Polski": "pl",
            "Português": "pt",
            "Punjabi": "pa",
            "Quechua": "qu",
            "limba română": "ro",
            "русский язык": "ru",
            "Samoan": "sm",
            "Scots Gaelic": "gd",
            "Serbian (Cyrillic)": "sr-Cyrl",
            "Serbian (Latin)": "sr-Latn",
            "Shona": "sn",
            "Sindhi": "sd",
            "Sinhala": "si",
            "Slovak": "sk",
            "Slovenian": "sl",
            "Somali": "so",
            "Español": "es",
            "Sundanese": "su",
            "Swahili": "sw",
            "Swedish": "sv",
            "Tagalog": "tl",
            "Tajik": "tg",
            "Tamil": "ta",
            "Tatar": "tt",
            "Telugu": "te",
            "Thai": "th",
            "Tibetan": "bo",
            "Tigrinya": "ti",
            "Tongan": "to",
            "Türk dili": "tr",
            "Ukrainian": "uk",
            "Urdu": "ur",
            "Uyghur": "ug",
            "Uzbek": "uz",
            "Tiếng Việt": "vi",
            "Welsh": "cy",
            "Xhosa": "xh",
            "Yiddish": "yi",
            "Yoruba": "yo",
            "Zulu": "zu",
        }

        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        # Initialize providers on setup or when params change.
        self.providers.clear()
        self.providers = {
            "YandexBrowser": YandexBrowserProvider(),
            "YandexCloud": YandexCloudProvider(
                api_key=self.params.get("yandex_cloud_api_key", {}).get("value")
            ),
            "MSEdge": MSEdgeTranslateProvider(),
        }

    def _get_provider(self, service_name: str):
        provider = self.providers.get(service_name)
        if not provider:
            self._initialize_providers()
            provider = self.providers.get(service_name)
            if not provider:
                raise ProviderError(
                    f"Selected service '{service_name}' is not available or failed to initialize."
                )
        return provider

    def _translate(self, src_list: List[str]) -> List[str]:
        selected_service = self.params["service"]["value"]
        provider = self._get_provider(selected_service)

        source_lang = self.lang_map.get(self.lang_source, "auto")
        target_lang = self.lang_map.get(self.lang_target, "en")

        translated_list = []
        for text in src_list:
            if not text.strip():
                translated_list.append(text)
                continue

            try:
                time.sleep(self.delay())
                translated_text = provider.translate(text, source_lang, target_lang)
                translated_list.append(translated_text)
            except Exception as e:
                LOGGER.error(f"Translation error with {selected_service}: {e}")
                translated_list.append(f"[ERROR: {e}]")

        return translated_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        # Re-initialize providers if a key parameter changes to apply new settings.
        if "api_key" in param_key or param_key == "service":
            LOGGER.info(
                f"Parameter '{param_key}' changed, re-initializing providers..."
            )
            self._initialize_providers()
