from .base import *
import requests
import json
import html # For html.unescape


# --- exceptions ---
class ProviderError(Exception):
    pass


class TranslateError(ProviderError):
    pass


# --- Constants for Google Translate ---
USER_AGENT_BROWSER = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
# Use the API key from your example as a constant
GOOGLE_API_KEY = "AIzaSyATBXajvzQLTDHEQbcpq0Ihe0vWDHmO520"
GOOGLE_API_URL_BASE = "https://translate-pa.googleapis.com/v1"  # Base API URL


class GoogleTranslateProviderPython:
    """
    Провайдер для взаимодействия с неофициальным Google Translate API (translateHtml).
    Использует предопределенный API ключ.
    """

    api_url_path_segment = "/translateHtml"  # Path to the translation endpoint

    def __init__(self, timeout: int = 10):
        self.base_headers = {
            "X-Goog-API-Key": GOOGLE_API_KEY,  # Use the constant
            "Content-Type": "application/json+protobuf",
            "User-Agent": USER_AGENT_BROWSER,
        }
        self.fetch_opts = {"timeout": timeout}
        self.requests_session = requests.Session()
        self.requests_session.headers.update(self.base_headers)

    def _request(self, method: str = "POST", json_payload: Dict = None):
        actual_url = f"{GOOGLE_API_URL_BASE}{self.api_url_path_segment}"

        try:
            response = self.requests_session.request(
                method, actual_url, json=json_payload, **self.fetch_opts
            )

            if response.status_code >= 400:
                message = response.reason
                try:
                    error_data = response.json()
                    if "error" in error_data and isinstance(error_data["error"], dict):
                        message = error_data["error"].get("message", response.reason)
                except json.JSONDecodeError:
                    pass  # Using response.reason
                raise ProviderError(f"HTTP {response.status_code}: {message}")

            response_data = response.json()

            if isinstance(response_data, dict) and "error" in response_data:
                error_details = response_data.get("error")
                msg = "API error"
                if isinstance(error_details, dict) and "message" in error_details:
                    msg = error_details["message"]
                raise ProviderError(msg)
            return response_data
        except requests.exceptions.RequestException as e:
            raise ProviderError(f"Request failed: {e}")
        except json.JSONDecodeError:
            raw_text = (
                response.text[:200]
                if response and hasattr(response, "text")
                else "NoResponseObject"
            )
            raise ProviderError(f"Failed to decode JSON. Raw: {raw_text}")

    def translate(
        self, text_list: List[str], target_language: str, source_language: str = "auto"
    ) -> Dict[str, any]:
        """
        Переводит список текстов.
        source_language: 'auto' или код языка (например, 'en')
        target_language: код языка (например, 'ru')
        """
        if not text_list:
            return {"lang": target_language, "translations": []}

        translations_result = []
        for text_item in text_list:
            if not text_item or not text_item.strip():
                translations_result.append("")
                continue

            payload = [[[text_item], source_language, target_language], "wt_lib"]

            try:
                response_data = self._request(method="POST", json_payload=payload)

                extracted_text = None
                if (
                    response_data
                    and isinstance(response_data, list)
                    and len(response_data) > 0
                ):
                    if isinstance(response_data[0], list) and len(response_data[0]) > 0:
                        first_inner_item = response_data[0][0]
                        if isinstance(first_inner_item, str):
                            extracted_text = first_inner_item
                        elif (
                            isinstance(first_inner_item, list)
                            and len(first_inner_item) > 0
                            and isinstance(first_inner_item[0], str)
                        ):
                            extracted_text = first_inner_item[0]

                if extracted_text:
                    translations_result.append(html.unescape(extracted_text))
                else:
                    translations_result.append("")
            except ProviderError:
                translations_result.append("")

        return {"lang": target_language, "translations": translations_result}


@register_translator("google")
class TransGoogle(BaseTranslator):

    concate_text = False
    params: Dict = {
        "delay": 0.0,
    }

    def _setup_translator(self):
        self.internal_google_translator = GoogleTranslateProviderPython()

        self.lang_map["Auto"] = "auto"
        self.lang_map["简体中文"] = "zh-CN"
        self.lang_map["繁體中文"] = "zh-TW"
        self.lang_map["日本語"] = "ja"
        self.lang_map["English"] = "en"
        self.lang_map["한국어"] = "ko"
        self.lang_map["Tiếng Việt"] = "vi"
        self.lang_map["čeština"] = "cs"
        self.lang_map["Nederlands"] = "nl"
        self.lang_map["Français"] = "fr"
        self.lang_map["Deutsch"] = "de"
        self.lang_map["magyar nyelv"] = "hu"
        self.lang_map["Italiano"] = "it"
        self.lang_map["Polski"] = "pl"
        self.lang_map["Português"] = "pt"
        self.lang_map["limba română"] = "ro"
        self.lang_map["русский язык"] = "ru"
        self.lang_map["Español"] = "es"
        self.lang_map["Türk dili"] = "tr"
        self.lang_map["украї́нська мо́ва"] = "uk"
        self.lang_map["Thai"] = "th"
        self.lang_map["Arabic"] = "ar"
        self.lang_map["Hindi"] = "hi"
        self.lang_map["Malayalam"] = "ml"
        self.lang_map["Tamil"] = "ta"

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        try:
            source_lang_code = self.lang_map.get(self.lang_source, "auto")
            target_lang_code = self.lang_map.get(self.lang_target, "en")

            response_data = self.internal_google_translator.translate(
                src_list,
                target_language=target_lang_code,
                source_language=source_lang_code,
            )

            if response_data and isinstance(response_data.get("translations"), list):
                translated_texts = response_data["translations"]
                if len(translated_texts) == len(src_list):
                    return translated_texts

            # In case of mismatch or error, we return empty strings
            return [""] * len(src_list)

        except ProviderError as e:
            LOGGER.error(f"Google Translate provider error: {e}")
            return [""] * len(src_list)
        except Exception as e:
            LOGGER.error(f"An unexpected error occurred in Google Translate: {e}")
            return [""] * len(src_list)
