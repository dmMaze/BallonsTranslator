import json
import threading
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import requests

from .base import register_translator
from .trans_google import GoogleTranslateProviderPython, ProviderError
from .trans_llm_api import LLM_API_Translator


DEEPL_FREE_API_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_API_URL = "https://api.deepl.com/v2/translate"


@register_translator("Two-Step Translator")
class TwoStepTranslator(LLM_API_Translator):
    concate_text = False
    cht_require_convert = True

    params: Dict = {
        "first step translator": {
            "type": "selector",
            "options": ["google", "DeepL Free", "DeepL"],
            "value": "google",
            "description": "Machine translator used for the first draft before LLM refinement.",
        },
        "deepl api key": {
            "value": "",
            "description": "DeepL or DeepL Free API key used when the first step translator is DeepL.",
        },
        "fallback to first step": {
            "type": "checkbox",
            "value": True,
            "description": "Return the first-step machine translation if the LLM refinement fails.",
        },
        "first step delay": {
            "value": 0.5,
            "description": "Seconds to wait after each Google/DeepL first-step request. Increase this to reduce request bursts and lower the risk of temporary provider blocking; set to 0 for maximum speed.",
        },
        "parallel first step during pipeline": {
            "type": "checkbox",
            "value": False,
            "description": "During full RUN, start the Google/DeepL draft translation in the background as soon as OCR finishes for a page. The LLM refinement still waits until detection, OCR, and inpainting are done.",
        },
        "unload vision models before llm": {
            "type": "checkbox",
            "value": True,
            "description": "When parallel first-step translation is enabled, unload text detection, OCR, and inpainting models before the final Ollama/LLM refinement step to free RAM/VRAM.",
        },
        **deepcopy(LLM_API_Translator.params),
    }
    params["provider"]["value"] = "Ollama"
    params["model"]["value"] = "OLLAMA: qwen3"
    params["endpoint"]["value"] = "http://localhost:11434/v1"
    params["system_prompt"][
        "value"
    ] = (
        "You are a translation editor. Improve draft machine translations by "
        "checking meaning, terminology, tone, fluency, punctuation, and line "
        "count. Return strictly valid JSON with one key 'translations', a list "
        "of objects with 'id' and 'translation'. Do not include explanations."
    )

    def _setup_translator(self):
        super()._setup_translator()
        self.google_translator = GoogleTranslateProviderPython()
        self._draft_cache: Dict[Tuple[str, str, str, Tuple[str, ...]], List[str]] = {}
        self._draft_cache_lock = threading.RLock()

    @property
    def first_step_translator(self) -> str:
        return self.get_param_value("first step translator")

    @property
    def deepl_api_key(self) -> str:
        return self.get_param_value("deepl api key")

    @property
    def fallback_to_first_step(self) -> bool:
        return bool(self.get_param_value("fallback to first step"))

    @property
    def first_step_delay(self) -> float:
        try:
            return max(float(self.get_param_value("first step delay")), 0.0)
        except Exception:
            return 0.0

    @property
    def parallel_first_step_during_pipeline(self) -> bool:
        return bool(self.get_param_value("parallel first step during pipeline"))

    @property
    def unload_vision_models_before_llm(self) -> bool:
        return bool(self.get_param_value("unload vision models before llm"))

    def pipeline_pretranslation_enabled(self) -> bool:
        return self.parallel_first_step_during_pipeline

    def should_unload_before_llm_refinement(self) -> bool:
        return self.unload_vision_models_before_llm

    def _draft_cache_key(self, src_list: List[str]) -> Tuple[str, str, str, Tuple[str, ...]]:
        return (
            self.lang_source,
            self.lang_target,
            self.first_step_translator,
            tuple(src_list),
        )

    def _google_lang_map(self) -> Dict[str, str]:
        return {
            "Auto": "auto",
            "\u7b80\u4f53\u4e2d\u6587": "zh-CN",
            "\u7e41\u9ad4\u4e2d\u6587": "zh-TW",
            "\u65e5\u672c\u8a9e": "ja",
            "English": "en",
            "\ud55c\uad6d\uc5b4": "ko",
            "Ti\u1ebfng Vi\u1ec7t": "vi",
            "\u010de\u0161tina": "cs",
            "Nederlands": "nl",
            "Fran\u00e7ais": "fr",
            "Deutsch": "de",
            "magyar nyelv": "hu",
            "Italiano": "it",
            "Polski": "pl",
            "Portugu\u00eas": "pt",
            "limba rom\u00e2n\u0103": "ro",
            "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u044f\u0437\u044b\u043a": "ru",
            "Espa\u00f1ol": "es",
            "T\u00fcrk dili": "tr",
            "\u0443\u043a\u0440\u0430\u0457\u0301\u043d\u0441\u044c\u043a\u0430 \u043c\u043e\u0301\u0432\u0430": "uk",
            "Thai": "th",
            "Arabic": "ar",
            "Hindi": "hi",
            "Malayalam": "ml",
            "Tamil": "ta",
        }

    def _deepl_lang_map(self) -> Dict[str, str]:
        return {
            "Auto": "",
            "\u7b80\u4f53\u4e2d\u6587": "ZH",
            "\u65e5\u672c\u8a9e": "JA",
            "English": "EN-US",
            "Fran\u00e7ais": "FR",
            "Deutsch": "DE",
            "Italiano": "IT",
            "Portugu\u00eas": "PT-PT",
            "Brazilian Portuguese": "PT-BR",
            "\u0440\u0443\u0441\u0441\u043a\u0438\u0439 \u044f\u0437\u044b\u043a": "RU",
            "Espa\u00f1ol": "ES",
            "Nederlands": "NL",
            "Polski": "PL",
            "\u010de\u0161tina": "CS",
            "\ud55c\uad6d\uc5b4": "KO",
            "Arabic": "AR",
        }

    def _translate_with_google(self, src_list: List[str]) -> List[str]:
        lang_map = self._google_lang_map()
        response = self.google_translator.translate(
            src_list,
            source_language=lang_map.get(self.lang_source, "auto"),
            target_language=lang_map.get(self.lang_target, "en"),
        )
        translations = response.get("translations", []) if response else []
        if len(translations) == len(src_list):
            return translations
        return [""] * len(src_list)

    def _translate_with_deepl(self, src_list: List[str], free: bool) -> List[str]:
        if not self.deepl_api_key:
            self.logger.error("DeepL first-step translation requires a DeepL API key.")
            return [""] * len(src_list)

        lang_map = self._deepl_lang_map()
        target_lang = lang_map.get(self.lang_target, "EN-US")
        source_lang = lang_map.get(self.lang_source, "")
        data = [("auth_key", self.deepl_api_key), ("target_lang", target_lang)]
        if source_lang:
            data.append(("source_lang", source_lang))
        for text in src_list:
            data.append(("text", text))

        url = DEEPL_FREE_API_URL if free else DEEPL_API_URL
        try:
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            payload = response.json()
            translations = [
                item.get("text", "") for item in payload.get("translations", [])
            ]
            if len(translations) == len(src_list):
                return translations
        except Exception as e:
            self.logger.error(f"DeepL first-step translation failed: {e}")
        return [""] * len(src_list)

    def _first_step_translate(self, src_list: List[str]) -> List[str]:
        cache_key = self._draft_cache_key(src_list)
        with self._draft_cache_lock:
            cached = self._draft_cache.get(cache_key)
            if cached is not None:
                return list(cached)

        provider = self.first_step_translator
        draft_list = None
        try:
            if provider == "google":
                draft_list = self._translate_with_google(src_list)
            elif provider == "DeepL Free":
                draft_list = self._translate_with_deepl(src_list, free=True)
            elif provider == "DeepL":
                draft_list = self._translate_with_deepl(src_list, free=False)
            if draft_list is not None:
                with self._draft_cache_lock:
                    self._draft_cache[cache_key] = list(draft_list)
                delay = self.first_step_delay
                if delay > 0:
                    time.sleep(delay)
                return draft_list
        except ProviderError as e:
            self.logger.error(f"First-step translator provider error: {e}")
        except Exception as e:
            self.logger.error(f"First-step translation failed: {e}")
        return [""] * len(src_list)

    def _collect_translation_inputs(self, textblk_lst) -> Tuple[List[int], List[str], List[str]]:
        non_empty_ids = []
        text_list = []
        translations = []
        for ii, blk in enumerate(textblk_lst):
            text = blk.get_text()
            if text.strip() != "":
                non_empty_ids.append(ii)
                text_list.append(text)
            translations.append(text)

        for callback in self._preprocess_hooks.values():
            callback(
                translations=translations,
                textblocks=textblk_lst,
                translator=self,
                source_text=text_list,
            )
        return non_empty_ids, text_list, translations

    def pretranslate_textblk_lst(self, textblk_lst) -> None:
        non_empty_ids, src_list, _ = self._collect_translation_inputs(textblk_lst)
        if not src_list:
            return
        draft_list = self._first_step_translate(src_list)
        for ii, idx in enumerate(non_empty_ids):
            textblk_lst[idx].translation_draft = draft_list[ii]
        self.logger.info(
            f"Prepared first-step {self.first_step_translator} draft for {len(src_list)} text blocks."
        )

    def translate_textblk_lst(self, textblk_lst: List) -> None:
        non_empty_ids, text_list, translations = self._collect_translation_inputs(textblk_lst)
        if text_list:
            draft_list = self._first_step_translate(text_list)
            for ii, idx in enumerate(non_empty_ids):
                textblk_lst[idx].translation_draft = draft_list[ii]

            refined = self.translate(text_list)
            for ii, idx in enumerate(non_empty_ids):
                translations[idx] = refined[ii]

        for callback in self._postprocess_hooks.values():
            callback(
                translations=translations,
                textblocks=textblk_lst,
                translator=self,
            )

        for tr, blk in zip(translations, textblk_lst):
            blk.translation = tr

    def _assemble_refinement_prompt(
        self, src_list: List[str], draft_list: List[str], to_lang: str
    ) -> str:
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)
        items = [
            {"id": i + 1, "source": source, "draft_translation": draft}
            for i, (source, draft) in enumerate(zip(src_list, draft_list))
        ]
        return (
            f"Improve the draft translations from {from_lang} to {to_lang}. "
            "Use the source text as the authority and preserve each id exactly. "
            "Return only JSON in the required schema.\n\n"
            f"{self._glossary_prompt_section()}"
            f"INPUT:\n{json.dumps(items, ensure_ascii=False, indent=2)}"
        )

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        draft_list = self._first_step_translate(src_list)
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)
        prompt = self._assemble_refinement_prompt(src_list, draft_list, to_lang)

        try:
            parsed_response = self._request_translation(prompt, is_reflection=True)
            if parsed_response and len(parsed_response.translations) == len(src_list):
                translations_by_id = {
                    item.id: item.translation for item in parsed_response.translations
                }
                translations = [
                    translations_by_id.get(i, draft_list[i - 1])
                    for i in range(1, len(src_list) + 1)
                ]
                self._update_glossary_from_batch(src_list, translations, to_lang)
                return self._refine_translations_with_glossary(
                    src_list, translations, to_lang
                )
            self.logger.error("LLM refinement returned an invalid translation count.")
        except Exception as e:
            self.logger.error(f"LLM refinement failed: {type(e).__name__}: {e}")

        if self.fallback_to_first_step:
            return draft_list
        return [""] * len(src_list)
