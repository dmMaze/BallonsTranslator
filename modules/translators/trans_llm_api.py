import re
import time
import json
import traceback
from typing import List, Dict, Optional, Type

import httpx
import openai
from pydantic import BaseModel, Field, ValidationError

from .base import BaseTranslator, register_translator


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""

    pass


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="A list of all translated elements."
    )


class GlossaryEntry(BaseModel):
    source: str = Field(..., description="Source term, name, place, or recurring phrase.")
    target: str = Field(..., description="Preferred translated form.")
    category: str = Field(
        default="term",
        description="Entry type such as character, place, organization, title, or term.",
    )
    note: str = Field(default="", description="Short optional usage note.")


class GlossaryResponse(BaseModel):
    entries: List[GlossaryEntry] = Field(
        default_factory=list,
        description="Reusable glossary entries extracted from translated text.",
    )


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio", "Ollama"],
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
                "OAI: gpt-4.1",
                "OAI: gpt-4.1-mini",
                "OAI: o4-mini",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.5-flash",
                "GGL: gemini-2.5-flash-lite",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
                "OR: qwen/qwen3-235b-a22b",
                "OR: qwen/qwen3-32b",
                "OR: google/gemma-3-27b-it",
                "OR: (override model field)",
                "LLMS: (override model field)",
                "OLLAMA: qwen3",
                "OLLAMA: gemma3",
                "OLLAMA: (override model field)",
            ],
            "value": "OAI: gpt-4o",
            "description": "Select a model that supports structured JSON output, or use the override field for newer model IDs.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "system_prompt": {
            "type": "editor",
            "value": 'You are an expert translator. Your task is to accurately translate the given text snippets. You MUST provide the output strictly in the specified JSON format, without any additional explanations or markdown formatting. The JSON object must have a single key \'translations\', which is a list of objects, each with an \'id\' (integer) and a \'translation\' (string).\n\nExample Output Schema:\n{"translations": [{"id": 1, "translation": "Translated text here."}]}',
            "description": "System message to instruct the LLM on its role and required output format.",
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
        "reasoning": {
            "type": "checkbox",
            "value": False,
            "description": "Enable provider-specific reasoning controls for models that support thinking or reasoning.",
        },
        "reasoning level": {
            "type": "selector",
            "options": ["low", "medium", "high"],
            "value": "medium",
            "description": "Reasoning effort used when reasoning is enabled.",
        },
        "reflection": {
            "type": "checkbox",
            "value": False,
            "description": "Run a second API call after translation so the model can review, score, and revise the result.",
        },
        "reflection prompt": {
            "type": "editor",
            "value": "Review the draft translation against the original source text. Check meaning, terminology, tone, fluency, punctuation, and whether the number of translated items matches the input. Revise only where the translation can be improved. Return only the final improved JSON object in the required schema.",
            "description": "Instructions used for the optional reflection/revision API call.",
        },
        "use glossary": {
            "type": "checkbox",
            "value": True,
            "description": "Include the glossary in translation prompts so character names, places, organizations, titles, and recurring terms stay consistent.",
        },
        "auto build glossary": {
            "type": "checkbox",
            "value": True,
            "description": "After each LLM translation batch, ask the model to extract reusable glossary entries from the source/translation pairs. This improves consistency but adds extra API calls.",
        },
        "glossary refinement pass": {
            "type": "checkbox",
            "value": True,
            "description": "Run a second LLM pass after translation to align the translated batch with the current glossary. This can improve names and terminology but adds latency and token cost.",
        },
        "glossary max entries": {
            "value": 200,
            "description": "Maximum number of glossary entries kept in the translator settings. Higher values preserve more terms but increase prompt size and cost.",
        },
        "glossary": {
            "type": "editor",
            "value": "",
            "description": "Persistent glossary used by the translator. Format: source => target [category] # optional note. You can edit it manually; auto build glossary appends or updates entries.",
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
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
        "low vram mode": {
            'value': False,
            'description': 'check it if you\'re running it locally on a single device and encountered a crash due to vram OOM',
            'type': 'checkbox',
        }
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
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Grok":
                endpoint = "https://api.x.ai/v1"
            elif provider == "Ollama":
                endpoint = "http://localhost:11434/v1"

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
                self.logger.error(
                    f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
                )
                http_client = httpx.Client()
        else:
            http_client = httpx.Client()

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        try:
            self.client = openai.OpenAI(
                api_key=api_key_to_use, base_url=endpoint, http_client=http_client
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            return False

    # --- Property getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

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
    def reasoning_enabled(self) -> bool:
        return bool(self.get_param_value("reasoning"))

    @property
    def reasoning_level(self) -> str:
        level = str(self.get_param_value("reasoning level") or "medium").lower()
        return level if level in {"low", "medium", "high"} else "medium"

    @property
    def reflection_enabled(self) -> bool:
        return bool(self.get_param_value("reflection"))

    @property
    def reflection_prompt(self) -> str:
        return self.get_param_value("reflection prompt")

    @property
    def use_glossary_enabled(self) -> bool:
        return bool(self.get_param_value("use glossary"))

    @property
    def auto_build_glossary_enabled(self) -> bool:
        return bool(self.get_param_value("auto build glossary"))

    @property
    def glossary_refinement_enabled(self) -> bool:
        return bool(self.get_param_value("glossary refinement pass"))

    @property
    def glossary_max_entries(self) -> int:
        return max(int(self.get_param_value("glossary max entries")), 0)

    @property
    def glossary_text(self) -> str:
        return self.get_param_value("glossary") or ""

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
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def invalid_repeat_count(self) -> int:
        return int(self.get_param_value("invalid repeat count"))

    @property
    def frequency_penalty(self) -> float:
        return float(self.get_param_value("frequency penalty"))

    @property
    def presence_penalty(self) -> float:
        return float(self.get_param_value("presence penalty"))

    @property
    def max_rpm(self) -> int:
        return int(self.get_param_value("max requests per minute"))

    @property
    def global_delay(self) -> float:
        return float(self.get_param_value("delay"))

    def _assemble_prompts(self, queries: List[str], to_lang: str):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        input_elements = [
            {"id": i + 1, "source": query} for i, query in enumerate(queries)
        ]
        input_json_str = json.dumps(input_elements, ensure_ascii=False, indent=2)
        glossary_section = self._glossary_prompt_section()

        prompt = (
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"{glossary_section}"
            f"INPUT:\n{input_json_str}"
        )

        yield prompt, len(queries)

    def _glossary_prompt_section(self) -> str:
        if not self.use_glossary_enabled:
            return ""
        glossary = self.glossary_text.strip()
        if not glossary:
            return ""
        return (
            "GLOSSARY:\n"
            "Use these preferred translations consistently for names, places, "
            "characters, organizations, titles, and recurring terms. Keep the "
            "natural grammar of the target language, but do not rename these "
            "entries unless the source clearly requires it.\n"
            f"{glossary}\n\n"
        )

    def _system_prompt_with_reasoning_policy(self) -> str:
        prompt = self.system_prompt
        if self.reasoning_enabled:
            policy = (
                f"Use {self.reasoning_level} reasoning effort internally if the "
                "selected model supports it. Do not include reasoning, analysis, "
                "chain-of-thought, or <think> blocks in the final response; output "
                "only the requested JSON object."
            )
        else:
            policy = (
                "Do not include reasoning, analysis, chain-of-thought, or <think> "
                "blocks in the response; output only the requested JSON object."
            )
        return f"{prompt}\n\n{policy}"

    def _build_reasoning_extra_body(self) -> Dict:
        if not self.reasoning_enabled:
            return {}

        level = self.reasoning_level
        provider = self.provider
        if provider == "OpenRouter":
            return {"reasoning": {"effort": level}}
        if provider in ["LLM Studio", "Ollama"]:
            return {"reasoning": {"effort": level}, "think": True}
        if provider in ["Google", "Grok"]:
            return {"reasoning_effort": level}
        return {}

    def _strip_reasoning_markup(self, content: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            lambda match: match.group(1),
            cleaned,
            flags=re.DOTALL,
        )
        return cleaned.strip()

    def _build_reflection_prompt(
        self, original_prompt: str, draft_response: TranslationResponse
    ) -> str:
        draft_json = draft_response.model_dump_json(indent=2)
        return (
            f"{self.reflection_prompt}\n\n"
            "ORIGINAL TRANSLATION TASK:\n"
            f"{original_prompt}\n\n"
            "DRAFT TRANSLATION JSON:\n"
            f"{draft_json}\n\n"
            "Return the reviewed and improved translation as JSON with the same "
            "'translations' list and the same numeric ids."
        )

    def _format_glossary_entry(self, entry: GlossaryEntry) -> str:
        category = (entry.category or "term").strip()
        note = (entry.note or "").strip()
        line = f"{entry.source.strip()} => {entry.target.strip()} [{category}]"
        if note:
            line = f"{line} # {note}"
        return line

    def _parse_glossary_lines(self) -> Dict[str, str]:
        entries = {}
        for line in self.glossary_text.splitlines():
            clean = line.strip()
            if not clean or clean.startswith("#") or "=>" not in clean:
                continue
            source = clean.split("=>", 1)[0].strip()
            if source:
                entries[source] = clean
        return entries

    def _save_glossary_entries(self, entries: List[GlossaryEntry]):
        if not entries or self.glossary_max_entries == 0:
            return

        glossary_lines = self._parse_glossary_lines()
        for entry in entries:
            source = entry.source.strip()
            target = entry.target.strip()
            if not source or not target:
                continue
            glossary_lines[source] = self._format_glossary_entry(entry)

        limited_lines = list(glossary_lines.values())[-self.glossary_max_entries :]
        self.set_param_value("glossary", "\n".join(limited_lines), convert_dtype=False)

    def _build_glossary_extraction_prompt(
        self, src_list: List[str], translations: List[str], to_lang: str
    ) -> str:
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        pairs = [
            {"id": i + 1, "source": source, "translation": translation}
            for i, (source, translation) in enumerate(zip(src_list, translations))
        ]
        existing_glossary = self.glossary_text.strip() or "(empty)"
        return (
            f"Extract a reusable translation glossary from {from_lang} to {to_lang}.\n"
            "Focus only on stable entries that should stay consistent across pages: "
            "character names, place names, organizations, titles, named items, "
            "recurring special terms, honorifics, and catchphrases. Do not add "
            "generic words or full sentences unless they are fixed terms.\n\n"
            "Return JSON with key 'entries'. Each entry must contain source, target, "
            "category, and optional note.\n\n"
            f"EXISTING GLOSSARY:\n{existing_glossary}\n\n"
            f"TRANSLATION PAIRS:\n{json.dumps(pairs, ensure_ascii=False, indent=2)}"
        )

    def _update_glossary_from_batch(
        self, src_list: List[str], translations: List[str], to_lang: str
    ):
        if not self.auto_build_glossary_enabled or not src_list:
            return

        system_prompt = (
            "You extract concise translation glossaries. Return only valid JSON "
            "matching this schema: {'entries': [{'source': str, 'target': str, "
            "'category': str, 'note': str}]}."
        )
        prompt = self._build_glossary_extraction_prompt(src_list, translations, to_lang)
        try:
            response = self._request_model_object(prompt, GlossaryResponse, system_prompt)
            if isinstance(response, GlossaryResponse):
                self._save_glossary_entries(response.entries)
                if response.entries:
                    self.logger.info(
                        f"Glossary updated with {len(response.entries)} extracted entries."
                    )
        except Exception as e:
            self.logger.warning(
                f"Glossary extraction failed; continuing without glossary update. {type(e).__name__}: {e}"
            )

    def _build_glossary_refinement_prompt(
        self, src_list: List[str], translations: List[str], to_lang: str
    ) -> str:
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        items = [
            {"id": i + 1, "source": source, "translation": translation}
            for i, (source, translation) in enumerate(zip(src_list, translations))
        ]
        return (
            f"Revise the translations from {from_lang} to {to_lang} using the glossary.\n"
            "Only change text where the glossary improves consistency for names, "
            "places, characters, organizations, titles, or recurring terms. Preserve "
            "meaning, tone, line count, ids, and natural target-language grammar. "
            "Return only JSON in the required translation schema.\n\n"
            f"{self._glossary_prompt_section()}"
            f"TRANSLATIONS TO REVIEW:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    def _refine_translations_with_glossary(
        self, src_list: List[str], translations: List[str], to_lang: str
    ) -> List[str]:
        if (
            not self.glossary_refinement_enabled
            or not self.use_glossary_enabled
            or not self.glossary_text.strip()
            or not src_list
        ):
            return translations

        prompt = self._build_glossary_refinement_prompt(src_list, translations, to_lang)
        try:
            response = self._request_translation(prompt, is_reflection=True)
            if response and len(response.translations) == len(src_list):
                translations_by_id = {
                    item.id: item.translation for item in response.translations
                }
                self.logger.info("Glossary refinement pass completed.")
                return [
                    translations_by_id.get(i, translations[i - 1])
                    for i in range(1, len(src_list) + 1)
                ]
            self.logger.warning("Glossary refinement returned an invalid translation count.")
        except Exception as e:
            self.logger.warning(
                f"Glossary refinement failed; using previous translation. {type(e).__name__}: {e}"
            )
        return translations

    def _record_usage(self, completion):
        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

    def _request_model_object(
        self, prompt: str, response_model: Type[BaseModel], system_prompt: str
    ) -> Optional[BaseModel]:
        current_api_key = self._select_api_key()

        if not current_api_key:
            if self.provider in ["LLM Studio", "Ollama"]:
                current_api_key = "dummy-key"
            else:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1)."
            )

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        self._respect_delay()

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        api_args = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        if self.provider == "LLM Studio":
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": response_model.model_json_schema()},
            }
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter", "Ollama"]:
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty
            if self.reasoning_enabled:
                api_args["reasoning_effort"] = self.reasoning_level

        extra_body = self._build_reasoning_extra_body()
        if extra_body:
            api_args["extra_body"] = extra_body

        completion = self._create_completion(api_args)
        self._record_usage(completion)

        if not (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            return None

        raw_content = completion.choices[0].message.content
        json_to_parse = self._strip_reasoning_markup(raw_content)
        start = json_to_parse.find("{")
        end = json_to_parse.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_to_parse = json_to_parse[start : end + 1]

        return response_model.model_validate(json.loads(json_to_parse))

    def _create_completion(self, api_args: Dict):
        try:
            return self.client.chat.completions.create(**api_args)
        except openai.BadRequestError as e:
            retry_args = dict(api_args)
            changed = False

            if "max_tokens" in retry_args:
                retry_args["max_completion_tokens"] = retry_args.pop("max_tokens")
                changed = True

            for key in [
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ]:
                if key in retry_args:
                    retry_args.pop(key)
                    changed = True

            if not changed:
                raise

            self.logger.warning(
                "Request was rejected by the provider. Retrying with reasoning-model compatible arguments."
            )
            try:
                return self.client.chat.completions.create(**retry_args)
            except Exception:
                raise e

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if hasattr(self, "debug_mode") and self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds."
                )
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
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60:
                    count = 0
                    start_time = now
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
        self.logger.error("All available API keys are currently rate-limited.")
        return None

    def _request_translation(
        self, prompt: str, is_reflection: bool = False
    ) -> Optional[TranslationResponse]:
        current_api_key = self._select_api_key()

        if not current_api_key:
            if self.provider in ["LLM Studio", "Ollama"]:
                current_api_key = "dummy-key"
            else:
                raise ConnectionError("No available API key found.")

        if self.provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1)."
            )

        if not self._initialize_client(current_api_key):
            raise ConnectionError("Failed to initialize API client.")

        self._respect_delay()

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        messages = [
            {"role": "system", "content": self._system_prompt_with_reasoning_policy()},
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
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif self.provider in ["OpenAI", "Grok", "Google", "OpenRouter", "Ollama"]:
            self.logger.debug(f"Using 'json_object' mode for {self.provider}.")
            api_args["response_format"] = {"type": "json_object"}

        if self.provider == "OpenAI":
            api_args["frequency_penalty"] = self.frequency_penalty
            api_args["presence_penalty"] = self.presence_penalty
            if self.reasoning_enabled:
                api_args["reasoning_effort"] = self.reasoning_level

        extra_body = self._build_reasoning_extra_body()
        if extra_body:
            api_args["extra_body"] = extra_body

        try:
            completion = self._create_completion(api_args)
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            raise

        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            raw_content = completion.choices[0].message.content
            json_to_parse = self._strip_reasoning_markup(raw_content)

            match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL
            )
            if match:
                self.logger.debug(
                    "Markdown code block detected. Extracting JSON content."
                )
                json_to_parse = match.group(1)
            else:
                start = json_to_parse.find("{")
                end = json_to_parse.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_to_parse = json_to_parse[start : end + 1]
            try:
                data_to_validate = json.loads(json_to_parse)
                validated_response = TranslationResponse.model_validate(
                    data_to_validate
                )
            except (ValidationError, json.JSONDecodeError) as e:
                self.logger.warning(
                    f"Initial Pydantic validation failed: {e}. Attempting to fix simple dictionary or list format."
                )
                try:
                    simple_data = json.loads(json_to_parse)
                    fixed_translations = []

                    if isinstance(simple_data, dict) and all(
                        k.isdigit() for k in simple_data.keys()
                    ):
                        fixed_translations = [
                            {"id": int(k), "translation": v}
                            for k, v in simple_data.items()
                        ]
                    elif isinstance(simple_data, list):
                        fixed_translations = simple_data

                    if fixed_translations:
                        fixed_data = {"translations": fixed_translations}
                        self.logger.debug(
                            f"Transformed simple response to: {fixed_data}"
                        )
                        validated_response = TranslationResponse.model_validate(
                            fixed_data
                        )
                        self.logger.info(
                            "Successfully parsed response after fixing simple format."
                        )
                    else:
                        raise e
                except (ValidationError, json.JSONDecodeError, Exception) as final_e:
                    self.logger.error(
                        f"Pydantic validation or JSON parsing failed even after attempting fix: {final_e}"
                    )
                    self.logger.debug(f"Raw JSON content from API: {raw_content}")
                    raise
        else:
            self.logger.warning("No valid message content in API response.")
            return None

        self._record_usage(completion)

        if self.reflection_enabled and not is_reflection:
            reflection_prompt = self._build_reflection_prompt(prompt, validated_response)
            try:
                reflected_response = self._request_translation(
                    reflection_prompt, is_reflection=True
                )
                if reflected_response and reflected_response.translations:
                    self.logger.info(
                        "Reflection pass completed and returned revised translations."
                    )
                    return reflected_response
            except Exception as e:
                self.logger.warning(
                    f"Reflection pass failed; using initial translation. {type(e).__name__}: {e}"
                )

        return validated_response

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIStatusError,
            httpx.RequestError,
        )

        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)

        for prompt, num_src in self._assemble_prompts(src_list, to_lang=to_lang):
            api_retry_attempt = 0
            mismatch_retry_attempt = 0

            while True:
                try:
                    parsed_response = self._request_translation(prompt)

                    if not parsed_response or not parsed_response.translations:
                        raise ValueError(
                            "Received empty or invalid parsed response from API."
                        )

                    if len(parsed_response.translations) != num_src:
                        raise InvalidNumTranslations(
                            f"Expected {num_src}, got {len(parsed_response.translations)}"
                        )

                    translations_dict = {
                        item.id: item.translation
                        for item in parsed_response.translations
                    }
                    ordered_translations = [
                        translations_dict.get(i, "") for i in range(1, num_src + 1)
                    ]
                    batch_sources = src_list[
                        len(translations) : len(translations) + num_src
                    ]
                    self._update_glossary_from_batch(
                        batch_sources, ordered_translations, to_lang
                    )
                    ordered_translations = self._refine_translations_with_glossary(
                        batch_sources, ordered_translations, to_lang
                    )

                    translations.extend(ordered_translations)
                    self.logger.info(
                        f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}"
                    )
                    break

                except InvalidNumTranslations as e:
                    mismatch_retry_attempt += 1
                    self.logger.warning(
                        f"Translation structure mismatch: {e}. Attempt {mismatch_retry_attempt}/{self.invalid_repeat_count}."
                    )
                    if mismatch_retry_attempt >= self.invalid_repeat_count:
                        self.logger.error(
                            "Fatal Error: Failed to get correct translation structure after retries."
                        )
                        translations.extend(["[ERROR: Structure Mismatch]"] * num_src)
                        break
                    time.sleep(self.retry_timeout / 2)

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    self.logger.warning(
                        f"API Error (retryable): {type(e).__name__} - {e}. Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.error(
                            f"Fatal Error: Failed to connect to API after {self.retry_attempts} attempts."
                        )
                        translations.extend([f"[ERROR: API Failed]"] * num_src)
                        break
                    time.sleep(self.retry_timeout)

                except (
                    ValidationError,
                    json.JSONDecodeError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                    ValueError,
                ) as e:
                    self.logger.error(
                        f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                    )
                    self.logger.debug(traceback.format_exc())
                    translations.extend([f"[ERROR: {type(e).__name__}]"] * num_src)
                    break

        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.client = None

