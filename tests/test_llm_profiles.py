import json
import os
import tempfile
import unittest

from ballontranslator.utils.config import ModuleConfig, ProgramConfig, json_dump_program_config
from ballontranslator.utils.llm_profiles import (
    DEFAULT_TRANSLATION_PROMPT,
    LLMProfile,
    copy_profile,
    default_profile,
    profile_by_id,
    restore_builtin_profiles,
)
from ballontranslator.utils.secret_store import SecretStore, is_portable_secret


class LLMProfileMigrationTest(unittest.TestCase):
    def test_backup_config_loads_selected_old_llm_translator_as_deepseek_profile(self):
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config-backup.json')
        if not os.path.exists(cfg_path):
            self.skipTest('config/config-backup.json is not available in this checkout')

        cfg = ProgramConfig.load(cfg_path)

        selected = profile_by_id(cfg.module.llm_profiles, cfg.module.llm_profile)
        self.assertEqual(cfg.module.translator, 'LLMTranslator')
        self.assertNotIn('ChatGPT', cfg.module.translator_params)
        self.assertNotIn('ChatGPT_exp', cfg.module.translator_params)
        self.assertNotIn('LLM_API_Translator', cfg.module.translator_params)
        self.assertIsInstance(selected, LLMProfile)
        self.assertEqual(selected.id, 'deepseek')
        self.assertEqual(selected.model, 'deepseek-v4-flash')
        self.assertEqual(selected.name, 'DeepSeek')
        self.assertEqual(selected.max_tokens, 4096)
        deepseek_profiles = [p for p in cfg.module.llm_profiles if p.id == 'deepseek']
        self.assertEqual(len(deepseek_profiles), 1)

    def test_restore_builtins_keeps_user_profiles(self):
        openai = default_profile('OpenAI')
        custom = copy_profile(default_profile('OpenAI'))
        custom.id = 'custom'
        profiles = restore_builtin_profiles([custom, openai])

        self.assertEqual(profiles[0].id, 'custom')
        self.assertTrue(any(p.id == 'openai' and p.built_in for p in profiles))

    def test_restore_builtins_keeps_filled_builtin_api_key(self):
        openai = default_profile('OpenAI')
        openai.api_key = SecretStore().store('openai', 'sk-demo')
        openai.model = 'temporary-model'

        profiles = restore_builtin_profiles([openai])
        restored = profile_by_id(profiles, 'openai')

        self.assertEqual(restored.model, 'gpt-5.5')
        self.assertEqual(SecretStore().resolve(restored.api_key).value, 'sk-demo')

    def test_openai_builtin_includes_current_model_seed(self):
        profile = default_profile('OpenAI')

        self.assertEqual(profile.model, 'gpt-5.5')
        self.assertIn('gpt-5.5', profile.model_options)
        self.assertIn('None', profile.thinking_level_options)
        self.assertNotIn('none', profile.thinking_level_options)
        self.assertEqual(profile.prompt, DEFAULT_TRANSLATION_PROMPT)
        self.assertEqual(profile.max_tokens, 8192)
        self.assertFalse(profile.json_schema_response_format)

    def test_lm_studio_builtin_uses_json_schema_response_format(self):
        profile = default_profile('LM Studio')

        self.assertTrue(profile.json_schema_response_format)


class SecretStoreTest(unittest.TestCase):
    def test_secret_store_obfuscates_plaintext_portably(self):
        store = SecretStore()
        saved = store.store('profile', 'sk-demo')

        self.assertTrue(is_portable_secret(saved))
        self.assertNotIn('sk-demo', json.dumps(saved))
        self.assertEqual(store.resolve(saved).value, 'sk-demo')

    def test_prepare_for_save_preserves_existing_obfuscated_value(self):
        store = SecretStore()
        saved = store.store('profile', 'sk-demo')

        prepared = store.prepare_for_save('profile', saved)

        self.assertEqual(prepared, saved)

    def test_saved_config_obfuscates_plaintext_secret(self):
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='openai')

        saved = json_dump_program_config(cfg)
        saved_dict = json.loads(saved)
        api_key = saved_dict['llm_profiles'][0]['api_key']

        self.assertNotIn('sk-demo', saved)
        self.assertTrue(is_portable_secret(api_key))
        self.assertEqual(SecretStore().resolve(api_key).value, 'sk-demo')

    def test_saved_config_keeps_obfuscated_secret_not_resolved_secret(self):
        profile = default_profile('OpenAI')
        profile.api_key = SecretStore().store('openai', 'sk-demo')
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='openai')

        saved = json_dump_program_config(cfg)

        self.assertNotIn('sk-demo', saved)

    def test_saved_config_serializes_profile_dataclass(self):
        profile = default_profile('DeepSeek')
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='deepseek')

        saved = json_dump_program_config(cfg)
        saved_dict = json.loads(saved)

        saved_profile = saved_dict['llm_profiles'][0]
        self.assertEqual(saved_profile['id'], 'deepseek')
        self.assertEqual(saved_profile['base_url'], 'https://api.deepseek.com')
        self.assertEqual(saved_profile['max_tokens'], 8192)

    def test_saved_profiles_roundtrip_without_provider(self):
        profile = default_profile('OpenAI')
        profile.api_key = 'sk-demo'
        cfg = ProgramConfig(module=ModuleConfig(llm_profiles=[profile], llm_profile='openai'))
        saved = json_dump_program_config(cfg)

        with tempfile.NamedTemporaryFile('w+', encoding='utf8') as temp:
            temp.write(saved)
            temp.flush()
            loaded = ProgramConfig.load(temp.name)

        selected = profile_by_id(loaded.module.llm_profiles, loaded.module.llm_profile)
        self.assertIsNotNone(selected)
        self.assertIsInstance(selected, LLMProfile)
        self.assertEqual(selected.id, 'openai')
        self.assertEqual(SecretStore().resolve(selected.api_key).value, 'sk-demo')


if __name__ == '__main__':
    unittest.main()
