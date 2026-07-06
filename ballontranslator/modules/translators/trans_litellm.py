import re
import time
import json
import traceback
from typing import List, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from .base import BaseTranslator, register_translator

_TRANSIENT_ERROR_QUALNAMES = {
    "litellm.exceptions.RateLimitError",
    "litellm.exceptions.APIConnectionError",
    "litellm.exceptions.Timeout",
    "litellm.exceptions.InternalServerError",
    "litellm.exceptions.ServiceUnavailableError",
}


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return True
    qualname = f"{type(exc).__module__}.{type(exc).__name__}"
    return qualname in _TRANSIENT_ERROR_QUALNAMES


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="A list of all translated elements."
    )


@register_translator("LiteLLM")
class LiteLLMTranslator(BaseTranslator):
    dependencies = ['litellm']

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "model": {
            "value": "openai/gpt-4o",
            "description": "LiteLLM model string (e.g. openai/gpt-4o, anthropic/claude-sonnet-4-20250514, gemini/gemini-2.5-flash). See https://docs.litellm.ai/docs/providers",
        },
        "api_key": {
            "value": "",
            "description": "API key for the provider. Leave empty to use provider env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).",
        },
        "api_base": {
            "value": "",
            "description": "Custom API base URL (e.g. http://localhost:4000 for LiteLLM proxy). Leave empty for provider defaults.",
        },
        "system_prompt": {
            "type": "editor",
            "value": 'You are an expert translator. Your task is to accurately translate the given text snippets. You MUST provide the output strictly in the specified JSON format, without any additional explanations or markdown formatting. The JSON object must have a single key \'translations\', which is a list of objects, each with an \'id\' (integer) and a \'translation\' (string).\n\nExample Output Schema:\n{"translations": [{"id": 1, "translation": "Translated text here."}]}',
            "description": "System message to instruct the LLM on its role and required output format.",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "max_tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response.",
        },
        "retry_attempts": {
            "value": 3,
            "description": "Number of retry attempts on API failures.",
        },
        "retry_timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "invalid_repeat_count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count.",
        },
        "delay": {
            "value": 0.3,
            "description": "Delay in seconds between requests.",
        },
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "čeština": "Czech",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.last_request_time = 0

    def _assemble_prompt(self, queries: List[str], to_lang: str) -> str:
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        input_elements = [
            {"id": i + 1, "source": query} for i, query in enumerate(queries)
        ]
        input_json_str = json.dumps(input_elements, ensure_ascii=False, indent=2)
        return (
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"INPUT:\n{input_json_str}"
        )

    def _respect_delay(self):
        delay = float(self.get_param_value("delay"))
        elapsed = time.time() - self.last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self.last_request_time = time.time()

    def _request_translation(self, prompt: str) -> Optional[TranslationResponse]:
        import litellm

        model = self.get_param_value("model")
        api_key = self.get_param_value("api_key") or None
        api_base = self.get_param_value("api_base") or None
        temperature = float(self.get_param_value("temperature"))
        max_tokens = int(self.get_param_value("max_tokens"))
        system_prompt = self.get_param_value("system_prompt")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "drop_params": True,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base

        kwargs["response_format"] = {"type": "json_object"}

        self._respect_delay()

        completion = litellm.completion(**kwargs)

        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            raw_content = completion.choices[0].message.content
            json_to_parse = raw_content.strip()

            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL
            )
            if match:
                json_to_parse = match.group(1)
            else:
                start = json_to_parse.find("{")
                end = json_to_parse.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_to_parse = json_to_parse[start : end + 1]

            try:
                data = json.loads(json_to_parse)
                validated = TranslationResponse.model_validate(data)
            except (ValidationError, json.JSONDecodeError):
                try:
                    simple_data = json.loads(json_to_parse)
                    fixed = []
                    if isinstance(simple_data, dict) and all(
                        k.isdigit() for k in simple_data.keys()
                    ):
                        fixed = [
                            {"id": int(k), "translation": v}
                            for k, v in simple_data.items()
                        ]
                    elif isinstance(simple_data, list):
                        fixed = simple_data
                    if fixed:
                        validated = TranslationResponse.model_validate(
                            {"translations": fixed}
                        )
                    else:
                        raise
                except (ValidationError, json.JSONDecodeError, Exception):
                    self.logger.error(
                        f"Failed to parse LLM response. Raw: {raw_content[:200]}"
                    )
                    raise
        else:
            self.logger.warning("No valid message content in API response.")
            return None

        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        return validated

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        retry_attempts = int(self.get_param_value("retry_attempts"))
        retry_timeout = int(self.get_param_value("retry_timeout"))
        invalid_repeat_count = int(self.get_param_value("invalid_repeat_count"))

        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        prompt = self._assemble_prompt(src_list, to_lang)

        api_retry = 0
        mismatch_retry = 0

        while True:
            try:
                parsed = self._request_translation(prompt)
                if parsed is None:
                    raise ValueError("Empty response from API.")

                result_map = {
                    elem.id: elem.translation for elem in parsed.translations
                }
                translations = []
                for i in range(1, len(src_list) + 1):
                    if i in result_map:
                        translations.append(result_map[i])
                    else:
                        self.logger.warning(
                            f"Missing translation for id {i}. Using source text."
                        )
                        translations.append(src_list[i - 1])

                if len(parsed.translations) != len(src_list):
                    mismatch_retry += 1
                    if mismatch_retry <= invalid_repeat_count:
                        self.logger.warning(
                            f"Translation count mismatch ({len(parsed.translations)} vs {len(src_list)}). "
                            f"Retry {mismatch_retry}/{invalid_repeat_count}."
                        )
                        continue

                return translations

            except Exception as e:
                if _is_transient_error(e):
                    api_retry += 1
                    if api_retry > retry_attempts:
                        self.logger.error(
                            f"All {retry_attempts} retries exhausted. Error: {e}"
                        )
                        self.logger.debug(traceback.format_exc())
                        return src_list

                    self.logger.warning(
                        f"Attempt {api_retry}/{retry_attempts} failed (transient): {e}. "
                        f"Retrying in {retry_timeout}s."
                    )
                    time.sleep(retry_timeout)
                else:
                    self.logger.error(f"Non-retryable error: {e}")
                    self.logger.debug(traceback.format_exc())
                    raise
