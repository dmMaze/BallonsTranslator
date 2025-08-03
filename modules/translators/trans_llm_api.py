import re
import time
import yaml
import traceback
from typing import List, Dict, Optional

import httpx
from openai import OpenAI

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    pass


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {  # Один API-ключ, если не заданы несколько
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). One key per line for readability.",
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.0-flash-exp",
                "GGL: gemini-2.0-flash",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
            ],
            "value": "",
            "description": "Select the model. Provider prefix indicates the provider. Leave empty for provider default.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty to use provider default.",
        },
        "prompt template": {
            "type": "editor",
            "value": "Please help me to translate the following text from a manga to {to_lang} (if it's already in {to_lang} or looks like gibberish you have to output it as it is instead):\n",
        },
        "chat system template": {
            "type": "editor",
            "value": "You are a professional translation engine, please translate the text into a colloquial, elegant and fluent content, without referencing machine translations. You must only translate the text content, never interpret it. If there's any issue in the text, output the text as is.\nTranslate to {to_lang}.",
        },
        "chat sample": {
            "type": "editor",
            "value": """日本語-简体中文:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
        - アイス あげた
        - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
        - テレビで見た！
    target:
        - 小孤独目击了两人的接吻
        - 二里酱
        - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
        - 我给了她冰激凌
        - 喜多酱 и你是怎么样的关系啊...
        - 我在电视上看到的！""",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of invalid repeat counts before considering translation failed.",
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
            "value": 0.5,
            "description": "Temperature for sampling (OpenAI). Google models may ignore this.",
        },
        "top p": {
            "value": 1.,
            "description": "Top P for sampling. Forced to 1 for Google models.",
        },
        "retry attempts": {
            "value": 5,
            "description": "Number of retry attempts on failure.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
        "low vram mode": {
            "value": False,
            "description": "Check if running locally and facing VRAM issues.",
            "type": "checkbox",
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
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        # Для контроля лимита по каждому ключу:
        self.key_usage = {}  # { api_key: (count, minute_start_time) }
        self._initialize_client()

    def _initialize_client(self):
        # Настраиваем httpx клиент с поддержкой proxy
        if self.proxy:
            proxy_mounts = {
                "http://": httpx.HTTPTransport(proxy=self.proxy),
                "https://": httpx.HTTPTransport(proxy=self.proxy),
            }
            transport = httpx.Client(mounts=proxy_mounts)
        else:
            transport = httpx.Client()

        # Определяем API-ключ: если заданы несколько, берём первый
        api_keys = self.multiple_keys_list
        if api_keys:
            api_key_to_use = api_keys[0]
        else:
            api_key_to_use = self.apikey

        if not api_key_to_use:
            self.logger.warning(
                "No API key provided. Please set either 'apikey' or 'multiple_keys'."
            )
            self.client = None
            return

        # Определяем endpoint: если не задан, выбираем по провайдеру
        endpoint = self.endpoint
        if not endpoint:
            if self.provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif self.provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            else:
                endpoint = "https://api.x.ai/v1"
                api_key_to_use = api_key_to_use.strip()  # only for XAI

        # Маскируем API ключ в логах – показываем только первые 6 символов
        masked_key = api_key_to_use[:6] + "*" * (len(api_key_to_use) - 6)
        self.logger.debug(
            f"Initializing OpenAI client with API key: {masked_key} and endpoint: {endpoint}"
        )
        self.client = OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=transport
        )

    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys").strip()
        return [key.strip() for key in keys_str.split(";") if key.strip()]

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        return float(self.get_param_value("top p"))

    @property
    def max_tokens(self) -> int:
        return int(self.get_param_value("max tokens"))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value("retry attempts"))

    @property
    def retry_timeout(self) -> int:
        return int(self.get_param_value("retry timeout"))

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def chat_system_template(self) -> str:
        to_lang = self.lang_map[self.lang_target]
        return self.params["chat system template"]["value"].format(to_lang=to_lang)

    @property
    def chat_sample(self):
        model_name = self.model
        if model_name == "gpt3":
            return None
        samples = self.params["chat sample"]["value"]
        try:
            samples = yaml.load(
                self.params["chat sample"]["value"], Loader=yaml.FullLoader
            )
        except Exception as e:
            self.logger.error(f"Failed to parse sample: {samples} - {e}")
            return None
        src_tgt = self.lang_source + "-" + self.lang_target
        if src_tgt in samples:
            sample_data = samples[src_tgt]
            src_queries = "\n".join(
                [f"<|{i+1}|>{s}" for i, s in enumerate(sample_data["source"])]
            )
            tgt_queries = "\n".join(
                [f"<|{i+1}|>{t}" for i, t in enumerate(sample_data["target"])]
            )
            return [src_queries, tgt_queries]
        return None

    def _assemble_prompts(
        self, queries: List[str], to_lang: str = None, max_tokens=None
    ):
        if to_lang is None:
            to_lang = self.lang_map[self.lang_target]
        prompt_template = (
            self.params["prompt template"]["value"].format(to_lang=to_lang).rstrip()
        )
        prompt = prompt_template
        num_src = 0
        i_offset = 0

        if max_tokens is None:
            max_tokens = self.max_tokens

        for i, query in enumerate(queries):
            prompt += f"\n<|{i+1-i_offset}|>{query}"
            num_src += 1
            if max_tokens * 2 and len("".join(queries[i + 1 :])) > max_tokens:
                yield prompt.lstrip(), num_src
                prompt = prompt_template
                i_offset = i + 1
                num_src = 0
        yield prompt.lstrip(), num_src

    def _format_prompt_log(self, prompt: str) -> str:
        chat_sample = self.chat_sample
        if self.model != "gpt3" and chat_sample:
            return "\n".join(
                [
                    "System:",
                    self.chat_system_template,
                    "User Sample:",
                    chat_sample[0],
                    "Assistant Sample:",
                    chat_sample[1],
                    "User Prompt:",
                    prompt,
                ]
            )
        return "\n".join(["System:", self.chat_system_template, "User Prompt:", prompt])

    def _respect_delay(self):
        current_time = time.time()

        # Глобальный лимит запросов (если указан)
        if int(self.params["max requests per minute"]["value"]) > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time

            if self.request_count_minute >= int(
                self.params["max requests per minute"]["value"]
            ):
                wait_time = 62 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Reached global RPM limit. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.debug(
                f"Time since last request: {time_since_last_request} seconds"
            )

        delay = float(self.params["delay"]["value"])
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if self.debug_mode:
                self.logger.debug(f"Waiting {sleep_time} seconds before next request")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str):
        rpm = int(self.get_param_value("max requests per minute"))
        if rpm <= 0:
            return
        count, start_time = self.key_usage.get(key, (0, time.time()))
        now = time.time()
        if now - start_time >= 60:
            self.key_usage[key] = (0, now)
            return
        if count >= rpm:
            wait_time = 60 - (now - start_time)
            self.logger.warning(
                f"Key {key[:6]}... reached RPM limit. Waiting {wait_time:.2f} seconds."
            )
            time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())

    def _select_api_key(self) -> str:
        api_keys = self.multiple_keys_list
        if api_keys:
            # Ротация ключей с учетом лимита по каждому
            for _ in range(len(api_keys)):
                index = self.current_key_index % len(api_keys)
                key = api_keys[index]
                self._respect_key_limit(key)
                count, start_time = self.key_usage.get(key, (0, time.time()))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (self.current_key_index + 1) % len(api_keys)
                return key
        else:
            return self.apikey

    def _request_translation_gpt3(self, prompt: str) -> str:
        response = self.client.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=self.max_tokens // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=float(self.params["frequency penalty"]["value"]),
            presence_penalty=float(self.params["presence penalty"]["value"]),
        )

        if response.choices:  # Проверка на наличие choices
            if response.choices[0].text:  # Проверка на наличие text в первом choice
                text_content = response.choices[0].text
                if text_content is None:  # Проверка на None text_content
                    if self.debug_mode:
                        self.logger.warning("Completion text content is None.")
                    return ""  # Возвращаем пустую строку, если text_content None
            else:
                if self.debug_mode:
                    self.logger.warning("No text found in completion choice.")
                return ""  # Возвращаем пустую строку, если нет text
        else:
            if self.debug_mode:
                self.logger.warning("No choices found in completion response.")
            return ""  # Возвращаем пустую строку, если нет choices

        if response.usage:  # Проверка на наличие usage
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens
        else:
            if self.debug_mode:
                self.logger.warning("Usage data not found in completion response.")
                self.token_count_last = (
                    0  # Устанавливаем token_count_last в 0, если usage нет
                )

        return text_content  # Возвращаем text_content после всех проверок

    def _request_translation(self, prompt: str, chat_sample: List[str]) -> str:
        self._respect_delay()

        current_api_key = self._select_api_key()
        if not current_api_key:
            return (
                "Error: No API key provided in 'apikey' or 'multiple_keys' parameter."
            )

        provider = self.provider
        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        # Обновляем клиента, чтобы использовать выбранный API ключ/endpoint
        self._initialize_client()

        self.logger.debug(f"Current Provider: {provider}")
        self.logger.debug(f"Using model name for API call: {model_name}")

        if model_name == "gpt3":
            return self._request_translation_gpt3(prompt)
        elif model_name in ["gpt35-turbo", "gpt4"]:
            model_name = model_name.replace("gpt35-turbo", "gpt-3.5-turbo").replace(
                "gpt4", "gpt-4"
            )

        if self.debug_mode:
            self.logger.info(f"Using model: {model_name}, Provider: {provider}")

        if provider == "Google":
            result = self._request_translation_with_chat_sample_google(
                prompt, model_name, chat_sample
            )
        else:
            result = self._request_translation_with_chat_sample_openai(
                prompt, model_name, chat_sample
            )

        if not isinstance(result, str):
            result = str(result)
        return result

    def _request_translation_with_chat_sample_openai(
        self, prompt: str, model: str, chat_sample: List[str]
    ) -> str:
        messages = [
            {"role": "system", "content": self.chat_system_template},
            {"role": "user", "content": prompt},
        ]
        if chat_sample:
            messages.insert(1, {"role": "user", "content": chat_sample[0]})
            messages.insert(2, {"role": "assistant", "content": chat_sample[1]})

        func_args = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens // 2,
        }
        # 如果是 grok/x.ai，不传 penalty 参数
        if not (self.provider == "Grok" or (self.endpoint and "x.ai" in self.endpoint)):
            func_args["frequency_penalty"] = float(self.params["frequency penalty"]["value"])
            func_args["presence_penalty"] = float(self.params["presence penalty"]["value"])

        response = self.client.chat.completions.create(**func_args)

        if response.choices:  # Проверка на наличие choices
            if response.choices[
                0
            ].message:  # Проверка на наличие message в первом choice
                content = response.choices[0].message.content
                if content is None:  # Проверка на None content
                    if self.debug_mode:
                        self.logger.warning("Chat completion content is None.")
                    return ""  # Возвращаем пустую строку, если content None
            else:
                if self.debug_mode:
                    self.logger.warning("No message found in chat completion choice.")
                return ""  # Возвращаем пустую строку, если нет message
        else:
            if self.debug_mode:
                self.logger.warning("No choices found in chat completion response.")
            return ""  # Возвращаем пустую строку, если нет choices

        if response.usage:  # Проверка на наличие usage
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens
        else:
            if self.debug_mode:
                self.logger.warning("Usage data not found in chat completion response.")
                self.token_count_last = (
                    0  # Устанавливаем token_count_last в 0, если usage нет
                )
        return content  # Возвращаем content после всех проверок

    def _request_translation_with_chat_sample_google(
        self, prompt: str, model: str, chat_sample: List[str]
    ) -> str:
        messages = [
            {"role": "system", "content": self.chat_system_template},
            {"role": "user", "content": prompt},
        ]
        if chat_sample:
            messages.insert(1, {"role": "user", "content": chat_sample[0]})
            messages.insert(2, {"role": "assistant", "content": chat_sample[1]})

        func_args = {
            "model": model,
            "messages": messages,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens // 2,
        }

        response = self.client.chat.completions.create(**func_args)
        if response.choices:
            if response.choices[0].message:
                return response.choices[0].message.content
        return ""  # Возвращаем пустую строку, если нет content

    def _translate(self, src_list: List[str]) -> List[str]:
        translations = []
        to_lang = self.lang_map[self.lang_target]
        queries = src_list
        chat_sample = self.chat_sample

        for prompt, num_src in self._assemble_prompts(queries, to_lang=to_lang):
            retry_attempt = 0
            while True:
                try:
                    response = self._request_translation(prompt, chat_sample)
                    if not isinstance(response, str):
                        response = str(response)
                    new_translations = re.split(r"<\|\d+\|>", response)[-num_src:]
                    if len(new_translations) != num_src:
                        _tr2 = re.sub(r"<\|\d+\|>", "", response).split("\n")
                        if len(_tr2) == num_src:
                            new_translations = _tr2
                        else:
                            raise InvalidNumTranslations
                    break
                except InvalidNumTranslations:
                    retry_attempt += 1
                    message = f"Translation count mismatch:\nprompt:\n{prompt}\ntranslations:\n{new_translations}\nresponse:\n{response}"
                    if retry_attempt >= self.retry_attempts:
                        self.logger.error(message)
                        new_translations = [""] * num_src
                        break
                    self.logger.warning(
                        message + f"\nRetrying. Attempt: {retry_attempt}"
                    )
                except Exception as e:
                    retry_attempt += 1
                    if retry_attempt >= self.retry_attempts:
                        new_translations = [""] * num_src
                        break
                    self.logger.warning(
                        f"Translation failed: {e}. Attempt: {retry_attempt}, sleep {self.retry_timeout}s..."
                    )
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    time.sleep(self.retry_timeout)
            translations.extend([t.strip() for t in new_translations])

        if self.token_count_last:
            self.logger.info(
                f"Used {self.token_count_last} tokens (Total: {self.token_count})"
            )
        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        self.logger.debug(
            f"updateParam called for key: {param_key}, content: {param_content}"
        )
        if param_key in [
            "proxy",
            "multiple_keys",
            "apikey",
            "provider",
            "endpoint",
            "model",
            "override_model",
        ]:
            self._initialize_client()
