import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List, Optional

import openai
import httpx

from .base import register_OCR, OCRBase, TextBlock


@register_OCR("llm_ocr")
class LLM_OCR(OCRBase):
    lang_map = {
        "Auto Detect": None,
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
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Corsican": "co",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Faroese": "fo",
        "Filipino": "fil",
        "Finnish": "fi",
        "French": "fr",
        "Frisian": "fy",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Indonesian": "id",
        "Interlingua": "ia",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Khmer": "km",
        "Korean": "ko",
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
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russian": "ru",
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
        "Spanish": "es",
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
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu",
    }

    popular_models = [
        "OAI: gpt-4o-mini",
        "OAI: gpt-4-vision-preview",
        "OAI: gpt-4o",
        "OAI: gpt-4",
        "GGL: gemini-1.5-pro-latest",
        "GGL: gemini-1.5-flash-latest",
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "OpenRouter"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "api_key": {
            "value": "",
            "description": "API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "model": {
            "type": "selector",
            "options": popular_models,
            "value": "OAI: gpt-4o-mini",
            "description": "Select the model to use.",
        },
        "override_model": {
            "value": "",
            "description": "Specify a custom model name to override the selected one.",
        },
        "language": {
            "type": "selector",
            "options": list(lang_map.keys()),
            "value": "Japanese",
            "description": "Language for OCR.",
        },
        "detail_level": {
            "type": "selector",
            "options": ["auto", "low", "high"],
            "value": "auto",
            "description": "Controls image detail level for vision models.",
        },
        "prompt": {
            "type": "editor",
            "value": "Perform OCR on the provided manga image snippet. The language is **{language}**.\nRecognize all text, including handwritten sound effects (SFX).\n**CRITICAL INSTRUCTION:** If you see jumbled characters, it is likely vertical text that was read horizontally. First, mentally reconstruct the correct vertical text.\n**OUTPUT FORMATTING:** All recognized text from the image must be consolidated into a **single, continuous horizontal line**. Do not use newlines.\nYour final output must be ONLY the recognized text. No explanations.",
            "description": "The main prompt for the OCR task. Use {language} placeholder.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are a specialized OCR engine for manga and comics. Your primary function is to accurately extract and consolidate all recognized text from an image into a **single, continuous horizontal line**. You must return only the raw, recognized text. You do not interpret, translate, or explain the content. You are designed to intelligently handle common OCR errors, such as reconstructing jumbled characters that result from misreading vertical text.",
            "description": "Optional system prompt to guide the model's behavior.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port)",
        },
        "delay": {"value": 1.0, "description": "Delay in seconds between requests."},
        "requests_per_minute": {
            "value": 15,
            "description": "Maximum number of requests per minute per key.",
        },
        "max_response_tokens": {
            "value": 4096,
            "description": "Maximum number of tokens in the LLM's response.",
        },
        "description": "OCR using various vision-capable LLMs.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.current_key_index = 0

    def _initialize_client(self, api_key_to_use: str):
        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"

        http_client = None
        if self.proxy:
            try:
                proxy_mounts = {"all://": httpx.HTTPTransport(proxy=self.proxy)}
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{self.proxy}': {e}.")

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        self.client = openai.OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=http_client
        )

    # --- Property Getters (similar to translator) ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def api_key(self) -> str:
        return self.get_param_value("api_key")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str):
            return []
        return [
            key.strip()
            for key in keys_str.strip().replace("\n", ";").split(";")
            if key.strip()
        ]

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override_model") or None

    @property
    def language(self) -> str:
        return self.get_param_value("language")

    @property
    def detail_level(self) -> str:
        return self.get_param_value("detail_level")

    @property
    def prompt(self) -> str:
        return self.get_param_value("prompt")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self) -> int:
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self) -> int:
        return int(self.get_param_value("max_response_tokens"))

    @property
    def request_delay(self) -> float:
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 1.0

    def _respect_delay(self):
        # This logic is identical to the one in LLM_API_Translator
        current_time = time.time()
        rpm = self.requests_per_minute
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f}s."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f}s.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        # This logic is identical to the one in LLM_API_Translator
        rpm = self.requests_per_minute
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) for key {key[:6]}... reached. Waiting {wait_time:.2f}s."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        # This logic is identical to the one in LLM_API_Translator
        api_keys = self.multiple_keys_list
        single_key = self.api_key
        if not api_keys and not single_key:
            self.logger.error("No API keys provided.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All API keys are rate-limited.")
        return None

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        api_key_to_use = self._select_api_key()
        if not api_key_to_use:
            return "[ERROR: No available API key]"

        # Re-initialize client if key is different from the last one used
        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)

        self._respect_delay()
        try:
            lang_name = self.language
            prompt_text = (prompt_override or self.prompt).format(language=lang_name)

            image_content_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }

            if self.provider in ["OpenAI", "Google", "OpenRouter"]:
                detail_setting = self.detail_level
                if detail_setting in ["low", "high"]:
                    image_content_part["image_url"]["detail"] = detail_setting

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        image_content_part,
                    ],
                }
            ]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            model_name = self.override_model or self.model
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]

            self.logger.debug(f"OCR request with model: {model_name}")

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=self.max_response_tokens,
            )

            if response.choices and response.choices[0].message.content:
                full_text = (
                    response.choices[0].message.content.replace("\n", " ").strip()
                )
                self.logger.debug(f"OCR result: {full_text}")
                return full_text
            else:
                self.logger.warning("No text found in OCR response.")
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return f"[ERROR: {type(e).__name__}]"

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", cropped_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                blk.text = self.ocr(img_base64, prompt_override=kwargs.get("prompt"))
            else:
                blk.text = ""

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["api_key", "multiple_keys", "endpoint", "proxy", "provider"]:
            self.client = None  # Force re-initialization on next call
        if param_key in ["requests_per_minute", "delay"]:
            self.request_count_minute = 0
            self.minute_start_time = time.time()
            self.last_request_time = 0
