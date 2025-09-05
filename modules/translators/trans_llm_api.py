import re
import time
import json
import traceback
from typing import List, Dict, Optional, Type

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""
    pass

class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(..., description="The translated text corresponding to the id.")

class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(..., description="A list of all translated elements.")


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "LLM Studio"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate through these keys.",
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.5-flash",
                "GGL: gemini-2.5-flash-lite",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
                "LLMS: (override model field)",
            ],
            "value": "OAI: gpt-4o",
            "description": "Select a model that supports JSON Mode for structured output.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are an expert translator. Your task is to accurately translate the given text snippets. You must provide the output strictly in the specified JSON format, without any additional explanations or markdown formatting. The JSON must conform to this schema: {\"translations\": [{\"id\": integer, \"translation\": string}]}. If a text snippet is untranslatable or does not require translation, return the original text in the 'translation' field.",
            "description": "System message to instruct the LLM on its role and required output format."
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count.",
        },
        "max requests per minute": {
            "value": 20,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response.",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "top p": {
            "value": 1.0,
            "description": "Top P for sampling.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API connection or parsing failures.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {"value": 0.0, "description": "Frequency penalty (OpenAI)."},
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese", "繁體中文": "Traditional Chinese", "日本語": "Japanese",
            "English": "English", "한국어": "Korean", "Tiếng Việt": "Vietnamese",
            "čeština": "Czech", "Français": "French", "Deutsch": "German",
            "magyar nyelv": "Hungarian", "Italiano": "Italian", "Polski": "Polish",
            "Português": "Portuguese", "limba română": "Romanian", "русский язык": "Russian",
            "Español": "Spanish", "Türk dili": "Turkish", "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai", "Arabic": "Arabic", "Malayalam": "Malayalam",
            "Tamil": "Tamil", "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.client = None

    def _initialize_client(self, api_key_to_use: str) -> bool:
        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            else: # Grok
                endpoint = "https://api.x.ai/v1"

        proxy = self.proxy
        http_client = None
        if proxy:
            try:
                proxy_mounts = {
                    "http://": httpx.HTTPTransport(proxy=proxy),
                    "https://": httpx.HTTPTransport(proxy=proxy),
                }
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy.")
                http_client = httpx.Client()
        else:
            http_client = httpx.Client()
        
        masked_key = api_key_to_use[:4] + "..." + api_key_to_use[-4:] if len(api_key_to_use) > 8 else api_key_to_use
        self.logger.debug(f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}")

        try:
            self.client = OpenAI(api_key=api_key_to_use, base_url=endpoint, http_client=http_client)
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            return False

    # --- Property getters ---
    @property
    def provider(self) -> str: return self.get_param_value("provider")
    @property
    def apikey(self) -> str: return self.get_param_value("apikey")
    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str): return []
        return [key.strip() for key in keys_str.strip().replace('\n', ';').split(";") if key.strip()]
    @property
    def model(self) -> str: return self.get_param_value("model")
    @property
    def override_model(self) -> Optional[str]: return self.get_param_value("override model") or None
    @property
    def endpoint(self) -> Optional[str]: return self.get_param_value("endpoint") or None
    @property
    def temperature(self) -> float: return float(self.get_param_value("temperature"))
    @property
    def top_p(self) -> float: return float(self.get_param_value("top p"))
    @property
    def max_tokens(self) -> int: return int(self.get_param_value("max tokens"))
    @property
    def retry_attempts(self) -> int: return int(self.get_param_value("retry attempts"))
    @property
    def retry_timeout(self) -> int: return int(self.get_param_value("retry timeout"))
    @property
    def proxy(self) -> str: return self.get_param_value("proxy")
    @property
    def chat_system_template(self) -> str: return self.get_param_value("chat system template")
    @property
    def invalid_repeat_count(self) -> int: return int(self.get_param_value("invalid repeat count"))
    @property
    def frequency_penalty(self) -> float: return float(self.get_param_value("frequency penalty"))
    @property
    def presence_penalty(self) -> float: return float(self.get_param_value("presence penalty"))
    @property
    def max_rpm(self) -> int: return int(self.get_param_value("max requests per minute"))
    @property
    def global_delay(self) -> float: return float(self.get_param_value("delay"))

    def _assemble_prompts(self, queries: List[str], to_lang: str, max_len_approx=8000):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        prompt_instructions = (
            f"Please translate the following {from_lang} text snippets to {to_lang}. "
            f"For each snippet, provide its translation corresponding to its original ID.\n\n"
        )
        current_prompt_content = ""
        num_src = 0
        i_offset = 0

        for i, query in enumerate(queries):
            element = f"id={i + 1 - i_offset}: {query}\n"
            if len(prompt_instructions) + len(current_prompt_content) + len(element) > max_len_approx and num_src > 0:
                yield prompt_instructions + current_prompt_content, num_src
                current_prompt_content = element
                num_src = 1
                i_offset = i
            else:
                current_prompt_content += element
                num_src += 1

        if num_src > 0:
            yield prompt_instructions + current_prompt_content, num_src

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0; self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds.")
                    time.sleep(wait_time)
                self.request_count_minute = 0; self.minute_start_time = time.time()
        
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if hasattr(self, 'debug_mode') and self.debug_mode: self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time(); self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0: return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.apikey
        if not api_keys and not single_key:
            self.logger.error("No API keys provided in parameters.")
            return None
        
        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time(); count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60: count = 0; start_time = now
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time(); count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All available API keys are currently rate-limited.")
        return None
        
    def _request_translation(self, prompt: str) -> Optional[TranslationResponse]:
        current_api_key = "lm-studio"
        if self.provider != "LLM Studio":
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError("Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1).")

        if not self._initialize_client(current_api_key):
             raise ConnectionError("Failed to initialize API client.")
        
        self._respect_delay()
        
        model_name = self.override_model or self.model
        if ": " in model_name: model_name = model_name.split(": ", 1)[1]

        messages = [
            {"role": "system", "content": self.chat_system_template},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.provider == "LLM Studio":
            self.logger.debug("Using 'json_schema' mode for LLM Studio.")
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": TranslationResponse.model_json_schema()
                },
            }
        elif self.provider in ["OpenAI", "Grok"]:
            self.logger.debug(f"Using 'json_object' mode for {self.provider}.")
            api_args["response_format"] = {"type": "json_object"}
        
        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty
        
        try:
            completion = self.client.chat.completions.create(**api_args)
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            raw_content = completion.choices[0].message.content
            json_to_parse = raw_content.strip()

            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL)
            if match:
                self.logger.debug("Markdown code block detected. Extracting JSON content.")
                json_to_parse = match.group(1)
            else:
                start = json_to_parse.find('{')
                end = json_to_parse.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_to_parse = json_to_parse[start:end+1]

            try:
                data_to_validate = json.loads(json_to_parse)
                
                if self.provider == "Google" and isinstance(data_to_validate, list):
                    self.logger.debug("Google API returned a list, wrapping it in {'translations': ...} for validation.")
                    data_to_validate = {"translations": data_to_validate}

                validated_response = TranslationResponse.model_validate(data_to_validate)
            except (ValidationError, json.JSONDecodeError) as e:
                self.logger.error(f"Pydantic validation or JSON parsing failed: {e}")
                self.logger.debug(f"Raw JSON content from API: {raw_content}")
                raise
        else:
            self.logger.warning("No valid message content in API response.")
            return None

        if hasattr(completion, 'usage') and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0
            
        return validated_response

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list: return []
        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        
        for prompt, num_src in self._assemble_prompts(src_list, to_lang=to_lang):
            api_retry_attempt = 0
            mismatch_retry_attempt = 0
            
            while True:
                try:
                    parsed_response = self._request_translation(prompt)
                    
                    if not parsed_response or not parsed_response.translations:
                        raise ValueError("Received empty or invalid parsed response from API.")

                    if len(parsed_response.translations) != num_src:
                        raise InvalidNumTranslations(f"Expected {num_src}, got {len(parsed_response.translations)}")
                    
                    translations_dict = {item.id: item.translation for item in parsed_response.translations}
                    ordered_translations = [translations_dict.get(i, "") for i in range(1, num_src + 1)]
                    
                    translations.extend(ordered_translations)
                    self.logger.info(f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}")
                    break

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    self.logger.warning(f"Translation structure mismatch: {e}. Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}.")
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error("Failed to get correct translation structure after retries.")
                        translations.extend(["[ERROR: Structure Mismatch]"] * num_src)
                        break
                    time.sleep(self.retry_timeout / 2)
                
                except Exception as e:
                    api_retry_attempt += 1
                    self.logger.warning(f"API request/parsing failed: {e}. Attempt {api_retry_attempt}/{self.retry_attempts}.")
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.error(f"Failed to translate batch after {self.retry_attempts} attempts: {traceback.format_exc()}")
                        translations.extend([f"[ERROR: API Failed]"] * num_src)
                        break
                    time.sleep(self.retry_timeout)
                    
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        # self.logger.debug(f"Parameter '{param_key}' updated.")
        
        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
        #   self.logger.info(f"Client will be re-initialized on next request due to change in '{param_key}'.")
            self.client = None