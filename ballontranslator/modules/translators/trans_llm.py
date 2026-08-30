from dataclasses import replace
import hashlib
import json
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..context.errors import (
    ContextLengthError,
    is_context_length_error,
)
from ..context.glossary import (
    load_glossary,
    select_glossary,
)
from ..context.history import (
    ContextAction,
    ContextDiagnostic,
    ContextReason,
    HistoryPage,
    HistoryWindow,
    HistoryWindowKey,
    eligible_history_for_request,
    window_rebuild_reason,
)
from ..context.token_usage import (
    format_completion_token_usage,
    messages_token_count,
)
from ..context.translation_context import (
    MemoryCheckpoint,
    PageSummary,
    RequestContext,
    fit_page_summaries,
    memory_compaction_messages,
    memory_coverage_line,
    memory_message_content,
    memory_window_signature,
    page_summary_context_token_count,
    parse_memory_response,
    project_memory_checkpoint,
    project_memory_signatures,
    recover_context_length,
    saved_page_summary_text,
    snapshot_omitted_page_summaries,
    snapshot_page_summaries,
)
from ..llm_chat import (
    LLMChatRequester,
    LLMChatRequestError,
    openai_chat_completion_args,
    openai_json_response_format,
)
from ..llm_vision import EncodedChatImage, encode_chat_image
from .base import BaseTranslator, register_translator
from .llm_translation_contract import (
    InvalidNumTranslations,
    TranslationPromptSpec,
    assemble_translation_request,
    parse_translation_response,
    render_history_page,
    render_user_prompt,
    translation_json_schema,
    translation_system_prompt,
)
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ballontranslator.utils.config import (
    LLMTranslateContext,
    RunStatus,
    pcfg,
)
from ballontranslator.utils.io_utils import text_is_empty
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    runtime_profile,
)
from ballontranslator.utils.proj_imgtrans import (
    LLM_COMPACT_MEMORY_VERSION,
    LLM_VISUAL_SUMMARY_VERSION,
    ProjImgTrans,
)


MAX_PAGE_LONG_SIDE = 1536
PAGE_IMAGE_JPEG_QUALITY = 85


@register_translator("LLMTranslator")
class LLMTranslator(LLMChatRequester, BaseTranslator):
    """Profile-backed OpenAI-compatible translator.

    Example:
        >>> parse_translation_response(
        ...     '{"translations":[{"id":1,"translation":"心"}]}', 1
        ... ).translations
        ('心',)
    """

    dependencies = ['openai>=2.8.1', 'httpx[socks,brotli]', 'tiktoken>=0.7.0']

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "description": "Translate using the selected text-capable LLM profile.",
        "max requests per minute": {
            "value": 20,
            "display_name": "Max Requests Per Minute",
            "description": "Global request limit for LLM translation.",
        },
        "delay": {
            "value": 0.3,
            "display_name": "Delay",
            "description": "Delay between LLM requests in seconds.",
        },
        "retry attempts": {
            "value": 5,
            "display_name": "Retry Attempts",
            "description": "Retries for API or parsing failures.",
        },
        "retry timeout": {
            "value": 7.0,
            "display_name": "Retry Timeout",
            "description": "Delay between retries in seconds.",
        },
        "proxy": {
            "value": "",
            "display_name": "Proxy",
            "description": "Proxy address used for the OpenAI-compatible client.",
        },
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['繁體中文'] = 'Traditional Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.lang_map['English'] = 'English'
        self.lang_map['한국어'] = 'Korean'
        self.lang_map['Tiếng Việt'] = 'Vietnamese'
        self.lang_map['čeština'] = 'Czech'
        self.lang_map['Français'] = 'French'
        self.lang_map['Deutsch'] = 'German'
        self.lang_map['magyar nyelv'] = 'Hungarian'
        self.lang_map['Italiano'] = 'Italian'
        self.lang_map['Polski'] = 'Polish'
        self.lang_map['Português'] = 'Portuguese'
        self.lang_map['limba română'] = 'Romanian'
        self.lang_map['русский язык'] = 'Russian'
        self.lang_map['Español'] = 'Spanish'
        self.lang_map['Türk dili'] = 'Turkish'
        self.lang_map['украї́нська мо́ва'] = 'Ukrainian'
        self.lang_map['Thai'] = 'Thai'
        self.lang_map['Arabic'] = 'Arabic'
        self.lang_map['Malayalam'] = 'Malayalam'
        self.lang_map['Tamil'] = 'Tamil'
        self.lang_map['Hindi'] = 'Hindi'

        self._history_window: Optional[HistoryWindow] = None
        self._pending_visual_summaries: Dict[str, Dict] = {}
        self._pending_memory_checkpoints: Dict[
            str,
            Tuple[str, MemoryCheckpoint],
        ] = {}

    @property
    def profile(self) -> LLMProfile:
        profile = runtime_profile(
            pcfg.module.llm_profiles,
            pcfg.module.translator_llm_id,
        )
        if not profile.support_text:
            raise RuntimeError(f'LLM profile "{profile.name}" does not have text translation enabled.')
        self._text_model(profile)
        return profile

    @staticmethod
    def _text_model(profile: LLMProfile) -> str:
        model = str(profile.model or '').strip()
        model_options = [str(option).strip() for option in profile.model_options if str(option).strip()]
        if not model or not model_options:
            raise LLMModelRequiredError(profile.id, profile.name)
        return model

    @staticmethod
    def _vision_model(profile: LLMProfile) -> str:
        if not profile.support_vision:
            raise RuntimeError(
                f'LLM profile "{profile.name}" does not have vision enabled.'
            )
        model = str(profile.vision_model or '').strip()
        model_options = [
            str(option).strip()
            for option in profile.vision_model_options
            if str(option).strip()
        ]
        if not model or not model_options:
            raise LLMModelRequiredError(
                profile.id,
                profile.name,
                target='vision_model',
            )
        return model

    def unload_model(self, empty_cache=False):
        self._history_window = None
        getattr(self, '_pending_visual_summaries', {}).clear()
        getattr(self, '_pending_memory_checkpoints', {}).clear()
        return super().unload_model(empty_cache=empty_cache)

    @staticmethod
    def _source_signature(sources: Tuple[str, ...]) -> str:
        payload = json.dumps(sources, ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _summary_fingerprint(
        self,
        *,
        image_sha256: str,
        source_signature: str,
        source_language: str,
        target_language: str,
        profile: LLMProfile,
        model: Optional[str] = None,
    ) -> str:
        """Fingerprint the inputs that produced one optional summary.

        >>> translator = LLMTranslator.__new__(LLMTranslator)
        >>> len(translator._summary_fingerprint(
        ...     image_sha256='a' * 64,
        ...     source_signature='b' * 64,
        ...     source_language='Japanese',
        ...     target_language='English',
        ...     profile=LLMProfile(id='p', base_url='https://example.test',
        ...                        vision_model='vision'),
        ... ))
        64
        """
        payload = {
            'version': LLM_VISUAL_SUMMARY_VERSION,
            'image_sha256': image_sha256,
            'source_signature': source_signature,
            'source_language': source_language,
            'target_language': target_language,
            'profile_id': str(profile.id),
            'provider': str(profile.base_url or '').strip(),
            'model': str(model or profile.vision_model or '').strip(),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _scaled_page_image(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        long_side = max(height, width)
        if long_side <= MAX_PAGE_LONG_SIDE:
            return image
        scale = MAX_PAGE_LONG_SIDE / long_side
        size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _vision_request_context(
        self,
        project: ProjImgTrans,
        page_key: str,
        profile: LLMProfile,
    ) -> EncodedChatImage:
        """Read, normalize, and freeze one page image before retries."""
        self._vision_model(profile)
        image = project.read_img(page_key)
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise RuntimeError(f'Unable to read page image: {page_key}')
        return encode_chat_image(
            self._scaled_page_image(image),
            detail=str(profile.vision_detail_level or 'None'),
            jpeg_quality=PAGE_IMAGE_JPEG_QUALITY,
            failure_message=f'Failed to encode page image: {page_key}',
        )

    def translate(
        self,
        text,
        *,
        project: Optional[ProjImgTrans] = None,
        page_key: Optional[str] = None,
        commit_history_window: bool = False,
    ):
        """Translate one request with an immutable project-context snapshot.

        The override mirrors the relevant ``BaseTranslator`` behavior while
        keeping the rendered messages fixed across provider retries. The caller
        decides whether this page-level request may advance the reusable window.

        >>> LLMTranslator('日本語', '简体中文').translate([])
        []
        """
        if text_is_empty(text):
            return text
        if not self.all_model_loaded():
            self.load_model()

        is_list = isinstance(text, List)
        queries = tuple(text) if is_list else (text,)
        profile = self.profile
        source_language = str(self.lang_source)
        target_language = str(self.lang_target)
        history_enabled = (
            pcfg.module.llm_translate_context == LLMTranslateContext.HISTORY
        )
        history_budget = max(
            0,
            int(pcfg.module.llm_prior_context_token_budget),
        )
        glossary_path = str(pcfg.module.llm_glossary_path or '')
        glossary_mode = pcfg.module.llm_glossary_mode
        vision_enabled = bool(pcfg.module.llm_translate_vision)
        summary_enabled = bool(pcfg.module.llm_translate_summary)
        memory_enabled = bool(pcfg.module.llm_translate_memory)
        target_language_name = self._translated_lang(target_language)
        prompt_spec = TranslationPromptSpec(
            source_language=self._translated_lang(source_language),
            target_language=target_language_name,
            system_prompt=translation_system_prompt(
                profile.prompt,
                target_language_name,
                history_enabled=history_enabled,
                summary_enabled=summary_enabled,
            ),
            summary_enabled=summary_enabled,
            history_enabled=history_enabled,
        )
        summary_slot_empty = False
        if page_key is not None:
            self._pending_visual_summaries.pop(str(page_key), None)
            self._pending_memory_checkpoints.pop(str(page_key), None)
            image_info = getattr(project, '_image_info', None)
            if (
                summary_enabled
                and isinstance(image_info, dict)
                and str(page_key) in image_info
            ):
                summary_slot_empty = (
                    project.get_llm_visual_summary(str(page_key)) is None
                )
        memory_record_signature, saved_memory_window_signature = (
            project_memory_signatures(project)
            if project is not None and memory_enabled
            else ('', '')
        )
        vision_request = None
        if (
            vision_enabled
            and project is not None
            and page_key is not None
        ):
            vision_request = self._vision_request_context(
                project,
                str(page_key),
                profile,
            )
        model = (
            self._vision_model(profile)
            if vision_request is not None
            else self._text_model(profile)
        )
        request_context = self._snapshot_request_context(
            project,
            page_key,
            profile,
            model=model,
            prompt_spec=prompt_spec,
            source_language=source_language,
            target_language=target_language,
            history_budget=history_budget,
            glossary_path=glossary_path,
            glossary_mode=glossary_mode,
            memory_enabled=memory_enabled,
        )
        text_trans = self._translate(
            queries,
            profile=profile,
            prompt_spec=prompt_spec,
            source_language=source_language,
            target_language=target_language,
            request_context=request_context,
            page_key=page_key,
            commit_history_window=commit_history_window,
            vision_request=vision_request,
            summary_slot_empty=summary_slot_empty,
        )
        if (
            commit_history_window
            and project is not None
            and page_key is not None
            and request_context is not None
            and request_context.memory is not None
        ):
            generated_signature = memory_window_signature(
                request_context.memory
            )
            if generated_signature != saved_memory_window_signature:
                self._pending_memory_checkpoints[str(page_key)] = (
                    memory_record_signature,
                    request_context.memory,
                )

        if text_trans is None:
            text_trans = [''] * len(text) if is_list else ''
        elif not is_list:
            text_trans = text_trans[0]

        if is_list:
            try:
                assert len(text_trans) == len(text)
            except Exception:
                LOGGER.error(
                    'This translator seems to messed up the translation which resulted in inconsistent translated line count.\n '
                    'Set concate_text to False or change textblk_break in the source code may solve the problem.'
                )
                raise
        return text_trans

    def on_page_translation_finished(
        self,
        project: ProjImgTrans,
        page_key: str,
    ) -> None:
        """Commit generated context only after a finalized full page."""
        pending_memory = self._pending_memory_checkpoints.pop(
            str(page_key),
            None,
        )
        record = self._pending_visual_summaries.pop(str(page_key), None)
        pages = getattr(project, 'pages', None)
        if not isinstance(pages, dict) or page_key not in pages:
            return
        if pending_memory is not None:
            previous_signature, memory = pending_memory
            current_signature, _ = project_memory_signatures(project)
            if current_signature != previous_signature:
                self.logger.warning(
                    'Compacted LLM memory was not saved because the project '
                    'memory was edited during translation.'
                )
            else:
                try:
                    self._persist_memory_checkpoint(project, memory)
                except Exception as error:
                    self.logger.warning(
                        'Unable to save compact LLM memory: %s',
                        error,
                    )
        if record is None:
            return
        _, sources, _ = BaseTranslator._prepare_textblock_sources(
            self,
            pages[page_key],
        )
        if self._source_signature(tuple(sources)) != record['source_signature']:
            self.logger.warning(
                'LLM page summary was not saved because page sources changed: %s',
                page_key,
            )
            return
        try:
            # Once present, generated or edited page memory belongs to the user.
            # A later translation may fill an empty slot but never replaces it.
            if project.get_llm_visual_summary(page_key) is None:
                project.set_llm_visual_summary(page_key, record)
        except Exception as error:
            # Translation is already final; optional context persistence must not
            # turn a successful page into a failed pipeline stage.
            self.logger.warning(
                'Unable to save LLM page summary for %s: %s',
                page_key,
                error,
            )

    def delay(self) -> float:
        return self.get_param_value('delay')

    def _translated_lang(self, lang: str) -> str:
        return self.lang_map.get(lang, lang)

    def _persist_memory_checkpoint(
        self,
        project: ProjImgTrans,
        memory: MemoryCheckpoint,
    ) -> None:
        """Copy a successful compaction result into project-owned state."""
        project.set_llm_compact_memory({
            'version': LLM_COMPACT_MEMORY_VERSION,
            'text': memory.text,
            'covered_pages': list(memory.covered_page_keys),
        })

    def _snapshot_request_context(
        self,
        project: Optional[ProjImgTrans],
        page_key: Optional[str],
        profile: LLMProfile,
        *,
        prompt_spec: TranslationPromptSpec,
        source_language: str,
        target_language: str,
        history_budget: int,
        glossary_path: str,
        glossary_mode: str,
        memory_enabled: bool,
        model: Optional[str] = None,
    ) -> Optional[RequestContext]:
        """Freeze user-owned context and eligible page history for one request.

        The returned messages remain immutable across ordinary provider retries;
        the runtime window is only a cache optimization over authoritative project
        state.

        >>> translator = LLMTranslator.__new__(LLMTranslator)
        >>> translator._history_window = None
        >>> translator._history_window is None
        True
        """
        use_history = prompt_spec.history_enabled
        summary_enabled = prompt_spec.summary_enabled
        model = model or self._text_model(profile)
        memory = (
            project_memory_checkpoint(project, model)
            if memory_enabled
            else None
        )
        saved_summaries = (
            snapshot_page_summaries(project, str(page_key))
            if summary_enabled and page_key is not None
            else ()
        )
        current_summaries = tuple(
            summary
            for summary in saved_summaries
            if summary.page_key == str(page_key)
        )
        current_summary_tokens = page_summary_context_token_count(
            current_summaries,
            model,
        )
        if (
            not use_history
            and not glossary_path
            and memory is None
            and not saved_summaries
        ):
            # Preserve the legacy prompt shape when both optional features are off.
            self._history_window = None
            disabled_diagnostic = ContextDiagnostic(
                page_key=str(page_key or ''),
                action=ContextAction.DISABLED,
                page_count=0,
                token_count=0,
                token_budget=history_budget,
            )
            self.logger.debug(str(disabled_diagnostic))
            return None

        glossary = load_glossary(glossary_path)
        if not use_history:
            # A glossary can operate alone, but must not retain a stale history window.
            self._history_window = None
        history = ()
        window_key = None
        diagnostic = ContextDiagnostic(
            page_key=str(page_key or ''),
            action=(
                ContextAction.DISABLED
                if not use_history
                else ContextAction.EMPTY
            ),
            page_count=0,
            token_count=(
                (memory.token_count if memory is not None else 0)
                + current_summary_tokens
            ),
            token_budget=history_budget,
            rebuild_reason=(
                ContextReason.HISTORY_DISABLED
                if not use_history
                else ContextReason.MISSING_PROJECT_PAGE
            ),
        )
        if use_history and project is not None and page_key is not None:
            # A reload gets a new identity even at the same path; the remaining
            # fields define how the reusable history window is rendered or sized.
            window_key = HistoryWindowKey(
                load_identity=getattr(project, 'load_identity', None),
                settings=(
                    ('source_language', source_language),
                    ('model', str(model)),
                    ('system_prompt', prompt_spec.system_prompt),
                    ('token_budget', int(history_budget)),
                    ('memory_enabled', memory_enabled),
                    (
                        'memory_signature',
                        memory_window_signature(memory),
                    ),
                ),
            )
            rebuild_reason = window_rebuild_reason(
                self._history_window,
                project,
                str(page_key),
                window_key,
            )
            previous_page = None
            if rebuild_reason is None:
                # Re-snapshot retained pages so edits cannot leak through cached messages.
                fresh_retained = tuple(
                    self._snapshot_history_page(
                        project,
                        page.page_key,
                        target_language,
                        summary_enabled=summary_enabled,
                    )
                    for page in self._history_window.history
                )
                if any(
                    fresh != rendered.snapshot
                    for fresh, rendered in zip(
                        fresh_retained,
                        self._history_window.history,
                    )
                ):
                    rebuild_reason = ContextReason.SNAPSHOT_CHANGED
                else:
                    # Only an adjacent page that finished successfully may extend the window.
                    previous_page = self._snapshot_history_page(
                        project,
                        self._history_window.request_page_key,
                        target_language,
                        summary_enabled=summary_enabled,
                    )
                    if previous_page is None:
                        rebuild_reason = ContextReason.PREVIOUS_INCOMPLETE
            history, diagnostic = eligible_history_for_request(
                window=self._history_window,
                project=project,
                page_key=str(page_key),
                previous_page=previous_page,
                token_budget=history_budget,
                rebuild_reason=rebuild_reason,
                snapshot_page=lambda candidate_key: self._snapshot_history_page(
                    project,
                    candidate_key,
                    target_language,
                    summary_enabled=summary_enabled,
                ),
                render_page=lambda page: render_history_page(
                    page,
                    model,
                    prompt_spec,
                ),
                reserved_tokens=(
                    memory.token_count
                    if memory is not None
                    else 0
                ) + current_summary_tokens,
            )
            if memory_enabled:
                omitted_summaries = snapshot_omitted_page_summaries(
                    project,
                    str(page_key),
                    tuple(page.page_key for page in history),
                )
                previous_memory = memory
                memory_token_limit = max(
                    0,
                    history_budget
                    - current_summary_tokens
                    - sum(page.token_count for page in history),
                )
                candidate_memory = self._compact_omitted_summaries(
                    previous=memory,
                    summaries=omitted_summaries,
                    profile=profile,
                    model=model,
                    history_budget=history_budget,
                    memory_token_limit=memory_token_limit,
                )
                if (
                    candidate_memory is not None
                    and candidate_memory != previous_memory
                    and candidate_memory.token_count > memory_token_limit
                ):
                    # The compactor is bounded before its request; keep this
                    # check at the caller so an invalid result can never retire
                    # exact history that was absent from the compaction input.
                    self.logger.warning(
                        'LLM memory compaction exceeded the remaining context '
                        'capacity (%s/%s tokens); keeping the previous checkpoint.',
                        candidate_memory.token_count,
                        memory_token_limit,
                    )
                    memory = previous_memory
                else:
                    memory = candidate_memory
                if memory != previous_memory and memory is not None:
                    window_key = replace(
                        window_key,
                        settings=tuple(
                            (
                                name,
                                memory_window_signature(memory),
                            )
                            if name == 'memory_signature'
                            else (name, value)
                            for name, value in window_key.settings
                        ),
                    )
                    diagnostic = replace(
                        diagnostic,
                        token_count=(
                            sum(page.token_count for page in history)
                            + (memory.token_count if memory is not None else 0)
                        ),
                    )

        # Saved summaries absent from selected bilingual history stay in the
        # volatile suffix, so completion state cannot hide user-owned context.
        represented_summary_pages = {
            page.page_key
            for page in history
            if page.snapshot.summary
        }
        summary_candidates = tuple(
            summary
            for summary in saved_summaries
            if summary.page_key not in represented_summary_pages
        )
        summary_budget = max(
            0,
            history_budget
            - sum(page.token_count for page in history)
            - (memory.token_count if memory is not None else 0),
        )
        page_summaries = fit_page_summaries(
            summary_candidates,
            model,
            summary_budget,
            required_page_key=str(page_key) if page_key is not None else None,
        )
        summary_token_count = page_summary_context_token_count(
            page_summaries,
            model,
        )
        diagnostic = replace(
            diagnostic,
            token_count=(
                sum(page.token_count for page in history)
                + (memory.token_count if memory is not None else 0)
                + summary_token_count
            ),
        )

        self.logger.debug(str(diagnostic))
        return RequestContext(
            history=history,
            glossary=glossary,
            glossary_mode=glossary_mode,
            history_budget=int(history_budget),
            window_key=window_key,
            request_page_key=str(page_key) if page_key is not None else None,
            diagnostic=diagnostic,
            memory=memory,
            page_summaries=page_summaries,
            summary_token_count=summary_token_count,
            current_summary_token_count=current_summary_tokens,
        )

    def _snapshot_history_page(
        self,
        project: Optional[ProjImgTrans],
        page_key: str,
        target_language: str,
        *,
        summary_enabled: bool = False,
    ) -> Optional[HistoryPage]:
        """Copy one eligible page without retaining its mutable text blocks.

        >>> LLMTranslator.__new__(LLMTranslator)._snapshot_history_page(
        ...     None, '001.png', 'English') is None
        True
        """
        pages = getattr(project, 'pages', None)
        image_info = getattr(project, '_image_info', None)
        if not isinstance(pages, dict) or page_key not in pages:
            return None
        if not isinstance(image_info, dict):
            return None
        info = image_info.get(page_key, {})
        if not isinstance(info, dict) or not (
            int(info.get('finish_code', 0)) & RunStatus.FIN_TRANSLATE
        ):
            return None
        # Missing target metadata is intentionally compatible with old projects.
        if (
            'translation_target' in info
            and info['translation_target'] != target_language
        ):
            return None

        blocks = pages[page_key]
        translations = []
        for block in blocks:
            source = block.get_text()
            if not source or not source.strip():
                continue
            translation = getattr(block, 'translation', '')
            if not translation or not str(translation).strip():
                # Page chunks are indivisible; never seed a partially translated page.
                return None
            translations.append(str(translation))
        if not translations:
            return None
        _, sources, _ = BaseTranslator._prepare_textblock_sources(
            self,
            blocks,
        )
        summary = ''
        if summary_enabled:
            summary = saved_page_summary_text(project, page_key)
        return HistoryPage(
            page_key=str(page_key),
            sources=tuple(sources),
            translations=tuple(translations),
            summary=summary,
        )

    def _compact_memory(
        self,
        *,
        previous: Optional[MemoryCheckpoint],
        summaries: Tuple[PageSummary, ...],
        profile: LLMProfile,
        model: str,
        target_tokens: int,
    ) -> Optional[MemoryCheckpoint]:
        messages = memory_compaction_messages(
            previous,
            summaries,
            target_tokens,
        )
        # Compaction is always a text request, independently of Vision.
        api_args = self._api_args(profile, messages)
        for limit_key in ('max_completion_tokens', 'max_tokens'):
            if limit_key in api_args:
                api_args[limit_key] = min(
                    int(api_args[limit_key]),
                    target_tokens,
                )
        api_args['response_format'] = openai_json_response_format(
            profile,
            'translation_memory',
            {
                'type': 'object',
                'properties': {'memory': {'type': 'string'}},
                'required': ['memory'],
                'additionalProperties': False,
            },
        )
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()
        try:
            result = self.request_chat_completion(profile, api_args)
            self._log_token_usage(result, page_key='memory-compaction')
            memory_text = parse_memory_response(result.content)
        except LLMRequestStopped:
            raise
        except Exception as error:
            self.logger.warning('LLM memory compaction skipped: %s', error)
            return None
        covered = list(previous.covered_page_keys if previous else ())
        covered_set = set(covered)
        for page in summaries:
            if page.page_key not in covered_set:
                covered.append(page.page_key)
                covered_set.add(page.page_key)
        covered_page_keys = tuple(covered)
        memory_text = (
            f'{memory_coverage_line(covered_page_keys)}\n\n'
            f'{memory_text}'
        )
        memory_message = [{
            'role': 'system',
            'content': memory_message_content(memory_text),
        }]
        token_count = messages_token_count(memory_message, model)
        if token_count > target_tokens:
            self.logger.warning(
                'LLM memory compaction exceeded its target (%s/%s tokens); '
                'keeping the previous checkpoint.',
                token_count,
                target_tokens,
            )
            return None
        return MemoryCheckpoint(
            text=memory_text,
            covered_page_keys=covered_page_keys,
            token_count=token_count,
        )

    def _compact_omitted_summaries(
        self,
        *,
        previous: Optional[MemoryCheckpoint],
        summaries: Tuple[PageSummary, ...],
        profile: LLMProfile,
        model: str,
        history_budget: int,
        memory_token_limit: int,
    ) -> Optional[MemoryCheckpoint]:
        covered = set(previous.covered_page_keys if previous else ())
        # Summary and memory text are independently user-owned. Coverage avoids
        # repeating automatic compaction; edits never invalidate either record.
        candidates = tuple(
            summary
            for summary in summaries
            if summary.text and summary.page_key not in covered
        )
        if not candidates:
            return previous

        target_tokens = min(
            max(64, int(history_budget * 0.20)),
            max(0, memory_token_limit),
        )
        if target_tokens <= 0:
            return previous
        input_limit = max(128, history_budget)
        compaction_model = self._text_model(profile)
        selected = ()
        # A project can expose more saved summaries than one bounded request
        # should carry. Prefer the most recent omitted summaries.
        for summary in reversed(candidates):
            proposed = (summary,) + selected
            messages = memory_compaction_messages(
                previous,
                proposed,
                target_tokens,
            )
            if messages_token_count(messages, compaction_model) <= input_limit:
                selected = proposed
        if not selected:
            return previous
        checkpoint = self._compact_memory(
            previous=previous,
            summaries=selected,
            profile=profile,
            model=compaction_model,
            target_tokens=target_tokens,
        )
        if checkpoint is None:
            return previous
        if model == compaction_model:
            return checkpoint
        memory_message = [{
            'role': 'system',
            'content': memory_message_content(checkpoint.text),
        }]
        translated_token_count = messages_token_count(memory_message, model)
        if translated_token_count > target_tokens:
            self.logger.warning(
                'LLM memory compaction exceeded its target for the translation '
                'model (%s/%s tokens); keeping the previous checkpoint.',
                translated_token_count,
                target_tokens,
            )
            return previous
        return replace(checkpoint, token_count=translated_token_count)

    def build_copy_prompt(self, src_list: List[str]) -> str:
        glossary_path = str(pcfg.module.llm_glossary_path or '')
        glossary_mode = pcfg.module.llm_glossary_mode
        glossary = load_glossary(glossary_path)
        selected_glossary = select_glossary(
            glossary,
            src_list,
            glossary_mode,
        ) if glossary else ()
        return render_user_prompt(
            tuple(src_list),
            self._translated_lang(self.lang_source),
            self._translated_lang(self.lang_target),
            selected_glossary,
        ).strip()

    def _api_args(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        expected_translations: int = 1,
        *,
        vision_enabled: bool = False,
        summary_enabled: bool = False,
    ) -> Dict:
        model = (
            self._vision_model(profile)
            if vision_enabled
            else self._text_model(profile)
        )
        api_args = {
            "model": model,
            "messages": messages,
        }
        api_args.update(openai_chat_completion_args(profile, model))
        api_args["response_format"] = openai_json_response_format(
            profile,
            'translation_response',
            (
                translation_json_schema(
                    expected_translations,
                    summary_enabled=summary_enabled,
                )
                if profile.json_schema_response_format
                else {}
            ),
        )

        for penalty, api_key in (
            (profile.frequency_penalty, 'frequency_penalty'),
            (profile.presence_penalty, 'presence_penalty'),
        ):
            penalty = float(penalty or 0.0)
            if penalty > 0:
                api_args[api_key] = penalty

        thinking_level = str(profile.thinking_level or 'None')
        if thinking_level.lower() != 'none':
            api_args["reasoning_effort"] = thinking_level
        return api_args

    def _log_token_usage(
        self,
        completion,
        *,
        page_key=None,
        attempt: Optional[int] = None,
    ):
        summary = format_completion_token_usage(completion)
        if summary:
            details = []
            if page_key is not None:
                safe_page_key = str(page_key).replace('\r', ' ').replace('\n', ' ')
                details.append(f'page={safe_page_key or "-"}')
            if attempt is not None:
                details.append(f'attempt={attempt}')
            details.append(summary)
            self.logger.debug(f'LLM token usage: {", ".join(details)}')

    def _request_translation(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        *,
        expected_translations: int = 1,
        usage_page_key=None,
        usage_attempt: Optional[int] = None,
        vision_enabled: bool = False,
        summary_enabled: bool = False,
    ) -> str:
        try:
            result = self.request_chat_completion(
                profile,
                self._api_args(
                    profile,
                    messages,
                    expected_translations,
                    vision_enabled=vision_enabled,
                    summary_enabled=summary_enabled,
                ),
            )
        except LLMChatRequestError as error:
            if is_context_length_error(error.provider_error):
                raise ContextLengthError(str(error)) from error.provider_error
            raise

        self._log_token_usage(
            result,
            page_key=usage_page_key,
            attempt=usage_attempt,
        )
        return result.content

    def _translate(
        self,
        src_list: Sequence[str],
        *,
        prompt_spec: TranslationPromptSpec,
        source_language: str,
        target_language: str,
        profile: Optional[LLMProfile] = None,
        request_context: Optional[RequestContext] = None,
        page_key: Optional[str] = None,
        commit_history_window: bool = True,
        vision_request: Optional[EncodedChatImage] = None,
        summary_slot_empty: bool = True,
    ) -> List[str]:
        """Translate with ordinary retries and optional-context recovery.

        Context recovery never truncates the current input or glossary, and a
        requested window commit occurs only after the response parses successfully.

        >>> spec = TranslationPromptSpec('Japanese', 'English', 'system', False)
        >>> LLMTranslator.__new__(LLMTranslator)._translate(
        ...     [], prompt_spec=spec,
        ...     source_language='Japanese', target_language='English')
        []
        """
        queries = tuple(src_list)
        if not queries:
            return []
        if profile is None:
            profile = self.profile
        summary_enabled = prompt_spec.summary_enabled
        usage_page_key = (
            request_context.request_page_key
            if request_context is not None
            and request_context.request_page_key is not None
            else page_key
        )
        messages, prompt = assemble_translation_request(
            queries,
            prompt_spec=prompt_spec,
            request_context=request_context,
            image_part=(
                vision_request.image_part()
                if vision_request is not None
                else None
            ),
        )
        retry_attempt = 0
        provider_attempt = 0
        active_context = request_context
        has_optional_summaries = bool(
            active_context is not None
            and any(
                summary.page_key != active_context.request_page_key
                for summary in active_context.page_summaries
            )
        )
        recovery_limit = (
            len(active_context.history)
            + int(has_optional_summaries)
            if active_context is not None
            else 0
        )
        recovery_attempts = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                provider_attempt += 1
                request_kwargs = {
                    'expected_translations': len(queries),
                    'usage_page_key': usage_page_key,
                    'usage_attempt': provider_attempt,
                }
                if vision_request is not None:
                    request_kwargs['vision_enabled'] = True
                if summary_enabled:
                    request_kwargs['summary_enabled'] = summary_enabled
                raw_response = self._request_translation(
                    profile,
                    messages,
                    **request_kwargs,
                )
                parsed = parse_translation_response(
                    raw_response,
                    len(queries),
                )
                translations = list(parsed.translations)
                successful_context = active_context
                break
            except ContextLengthError:
                # Provider tokenization can exceed our estimate; remove optional
                # summaries, then whole history pages, without consuming retries.
                if recovery_attempts >= recovery_limit:
                    raise
                recovered_context = recover_context_length(active_context)
                if recovered_context is None:
                    raise
                self.logger.debug(str(recovered_context.diagnostic))
                recovery_attempts += 1
                active_context = recovered_context
                messages, prompt = assemble_translation_request(
                    queries,
                    prompt_spec=prompt_spec,
                    request_context=active_context,
                    image_part=(
                        vision_request.image_part()
                        if vision_request is not None
                        else None
                    ),
                )
                continue
            except LLMApiKeyRequiredError:
                raise
            except LLMModelRequiredError:
                raise
            except LLMRequestStopped:
                raise
            except Exception as e:
                if isinstance(e, InvalidNumTranslations):
                    self.logger.error(f"Failed to parse matching translation count for prompt:\n{prompt}\n{e}")
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    self.logger.error(f"LLM translation failed: {e}")
                    self.logger.debug(traceback.format_exc())
                    raise
                self.logger.warning(f"LLM translation failed due to {e}. Attempt: {retry_attempt}")
                self._wait(self.get_param_value('retry timeout'))

        if (
            commit_history_window
            and page_key is not None
            and summary_enabled
            and summary_slot_empty
            and parsed.page_summary
        ):
            source_signature = self._source_signature(queries)
            image_sha256 = (
                vision_request.image_sha256
                if vision_request is not None
                else ''
            )
            selected_model = (
                self._vision_model(profile)
                if vision_request is not None
                else self._text_model(profile)
            )
            self._pending_visual_summaries[str(page_key)] = {
                'version': LLM_VISUAL_SUMMARY_VERSION,
                'text': parsed.page_summary,
                'fingerprint': self._summary_fingerprint(
                    image_sha256=image_sha256,
                    source_signature=source_signature,
                    source_language=source_language,
                    target_language=target_language,
                    profile=profile,
                    model=selected_model,
                ),
                'image_sha256': image_sha256,
                'source_signature': source_signature,
                'source_language': source_language,
                'target_language': target_language,
                'profile_id': str(profile.id),
                'provider': str(profile.base_url or '').strip(),
                'model': selected_model,
                'vision_model': (
                    selected_model if vision_request is not None else ''
                ),
            }
        elif (
            commit_history_window
            and page_key is not None
            and summary_enabled
            and not parsed.page_summary
        ):
            self.logger.warning(
                'LLM translation returned no usable page summary for %s; '
                'the saved summary was left unchanged.',
                page_key,
            )

        # Keep eviction/growth speculative until every response parsed successfully.
        if (
            commit_history_window
            and successful_context is not None
            and successful_context.window_key is not None
            and successful_context.request_page_key is not None
        ):
            self._history_window = HistoryWindow(
                key=successful_context.window_key,
                request_page_key=successful_context.request_page_key,
                history=successful_context.history,
                token_count=sum(
                    page.token_count for page in successful_context.history
                ),
            )
        return translations
