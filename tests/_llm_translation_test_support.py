from ballontranslator.modules.translators.llm_translation_contract import (
    TranslationPromptSpec,
    assemble_translation_request,
    translation_system_prompt,
)
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import (
    LLMGlossaryMode,
    LLMTranslateContext,
    pcfg,
)
from ballontranslator.utils.llm_profiles import default_profile
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock


def _block(source: str, translation: str = '') -> TextBlock:
    return TextBlock(text=[source], translation=translation)


class LLMTranslationTestMixin:
    def setUp(self) -> None:
        super().setUp()
        self.translator = LLMTranslator('日本語', '简体中文')
        self.profile = default_profile('OpenAI')
        self.profile.api_key = 'sk-test'
        self._settings = {
            'llm_translate_context': pcfg.module.llm_translate_context,
            'llm_prior_context_token_budget': pcfg.module.llm_prior_context_token_budget,
            'llm_glossary_path': pcfg.module.llm_glossary_path,
            'llm_glossary_mode': pcfg.module.llm_glossary_mode,
            'llm_translate_vision': pcfg.module.llm_translate_vision,
            'llm_translate_summary_memory': (
                pcfg.module.llm_translate_summary_memory
            ),
            'llm_translate_overwrite_summary': (
                pcfg.module.llm_translate_overwrite_summary
            ),
        }
        # Keep the split suites independent from ambient config and test order.
        pcfg.module.llm_translate_context = LLMTranslateContext.PAGE
        pcfg.module.llm_prior_context_token_budget = 4096
        pcfg.module.llm_glossary_path = ''
        pcfg.module.llm_glossary_mode = LLMGlossaryMode.Matching
        pcfg.module.llm_translate_vision = False
        pcfg.module.llm_translate_summary_memory = False
        pcfg.module.llm_translate_overwrite_summary = False
        self._retry_settings = {
            key: self.translator.get_param_value(key)
            for key in ('retry attempts', 'retry timeout')
        }

    def tearDown(self) -> None:
        for name, value in self._settings.items():
            setattr(pcfg.module, name, value)
        for name, value in self._retry_settings.items():
            self.translator.set_param_value(name, value)
        super().tearDown()

    def _prompt_spec(
        self,
        profile=None,
        *,
        summary_enabled: bool = False,
        history_enabled=None,
    ) -> TranslationPromptSpec:
        profile = profile or self.profile
        history_enabled = (
            pcfg.module.llm_translate_context == LLMTranslateContext.HISTORY
            if history_enabled is None
            else bool(history_enabled)
        )
        source_language = self.translator._translated_lang(
            self.translator.lang_source
        )
        target_language = self.translator._translated_lang(
            self.translator.lang_target
        )
        return TranslationPromptSpec(
            source_language=source_language,
            target_language=target_language,
            system_prompt=translation_system_prompt(
                profile.prompt,
                target_language,
                history_enabled=history_enabled,
                summary_enabled=summary_enabled,
            ),
            summary_enabled=summary_enabled,
            history_enabled=history_enabled,
        )

    def _snapshot_request_context(
        self,
        project,
        page_key,
        profile,
        *,
        model=None,
        summary_enabled: bool = False,
    ):
        history_enabled = (
            pcfg.module.llm_translate_context == LLMTranslateContext.HISTORY
        )
        return self.translator._snapshot_request_context(
            project,
            page_key,
            profile,
            prompt_spec=self._prompt_spec(
                profile,
                summary_enabled=summary_enabled,
                history_enabled=history_enabled,
            ),
            source_language=str(self.translator.lang_source),
            target_language=str(self.translator.lang_target),
            history_budget=max(
                0,
                int(pcfg.module.llm_prior_context_token_budget),
            ),
            glossary_path=str(pcfg.module.llm_glossary_path or ''),
            glossary_mode=pcfg.module.llm_glossary_mode,
            memory_enabled=bool(pcfg.module.llm_translate_summary_memory),
            model=model or self.translator._text_model(profile),
        )

    def _assemble_request(
        self,
        queries,
        profile,
        request_context=None,
        *,
        vision_request=None,
        summary_enabled: bool = False,
    ):
        return assemble_translation_request(
            tuple(queries),
            prompt_spec=self._prompt_spec(
                profile,
                summary_enabled=summary_enabled,
            ),
            request_context=request_context,
            image_part=(
                vision_request.image_part()
                if vision_request is not None
                else None
            ),
        )

    def _translate(self, src_list, **kwargs):
        profile = kwargs.get('profile') or self.profile
        summary_enabled = bool(kwargs.pop('summary_enabled', False))
        return self.translator._translate(
            src_list,
            prompt_spec=self._prompt_spec(
                profile,
                summary_enabled=summary_enabled,
            ),
            **kwargs,
        )

    @staticmethod
    def _project(page_count: int) -> ProjImgTrans:
        project = ProjImgTrans()
        project.pages = {
            f'{index:03}.png': [
                _block(f'source-{index}', f'target-{index}')
            ]
            for index in range(1, page_count + 1)
        }
        project._image_info = {
            page_key: {'finish_code': 0}
            for page_key in project.pages
        }
        return project

    @staticmethod
    def _complete(
        project: ProjImgTrans,
        page_key: str,
        target: str = '简体中文',
    ) -> None:
        project.mark_translation_finished(page_key, target)
