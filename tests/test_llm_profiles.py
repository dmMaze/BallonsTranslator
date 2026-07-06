import copy
import json
import unittest

from ballontranslator.utils.config import ModuleConfig, json_dump_program_config
from ballontranslator.utils.llm_profiles import (
    copy_profile,
    dedupe_profiles,
    default_profile,
    migrate_module_llm_profiles,
    profile_by_id,
    profile_from_old_settings,
    restore_builtin_profiles,
)
from ballontranslator.utils.secret_store import SecretStore, is_portable_secret


class LLMProfileMigrationTest(unittest.TestCase):
    def test_backup_config_migrates_old_llm_translators_to_profiles(self):
        module = {
            'translator': 'LLM_API_Translator',
            'translator_params': {
                'ChatGPT': {
                    'api key': 'sk-demo',
                    'model': 'gpt-4o',
                    'override model': 'deepseek-v4-flash',
                    '3rd party api url': 'https://api.deepseek.com',
                    'prompt template': 'Translate to {to_lang}:',
                    'chat system template': 'Translate to {to_lang}.',
                    'chat sample': '',
                    'max requests per minute': 17,
                    'delay': 0.7,
                    'retry attempts': 4,
                    'retry timeout': 12,
                    'proxy': 'socks5://127.0.0.1:1080',
                },
                'ChatGPT_exp': {
                    'api key': 'sk-demo',
                    'model': 'gpt-4o',
                    'override model': 'deepseek-v4-flash',
                    '3rd party api url': 'https://api.deepseek.com',
                },
                'LLM_API_Translator': {
                    'provider': 'OpenAI',
                    'apikey': 'sk-demo',
                    'multiple_keys': 'ignored',
                    'model': 'LLMS: (override model field)',
                    'override model': 'deepseek-v4-flash',
                    'endpoint': 'https://api.deepseek.com',
                    'max requests per minute': 21,
                    'delay': 0.2,
                    'retry attempts': 3,
                    'retry timeout': 15,
                    'proxy': '',
                },
            },
        }

        migrate_module_llm_profiles(module, SecretStore(enable_keyring=False))

        self.assertEqual(module['translator'], 'LLMTranslator')
        self.assertFalse({'ChatGPT', 'ChatGPT_exp', 'LLM_API_Translator'} & set(module['translator_params']))
        selected = profile_by_id(module['llm_profiles'], module['llm_profile'])
        self.assertIsNotNone(selected)
        self.assertEqual(selected['provider'], 'DeepSeek')
        self.assertEqual(selected['model'], 'deepseek-v4-flash')
        self.assertEqual(selected['name'], 'DeepSeek')
        self.assertNotIn('prompt mode', selected)
        deepseek_profiles = [p for p in module['llm_profiles'] if p['provider'] == 'DeepSeek']
        self.assertEqual(len(deepseek_profiles), 1)
        self.assertNotIn('multiple_keys', selected)
        self.assertNotIn('prompt template', selected)
        self.assertNotIn('chat system template', selected)
        self.assertNotIn('max requests per minute', selected)
        self.assertNotIn('delay', selected)
        self.assertNotIn('retry attempts', selected)
        self.assertNotIn('retry timeout', selected)
        self.assertNotIn('proxy', selected)
        llm_params = module['translator_params']['LLMTranslator']
        self.assertEqual(llm_params['max requests per minute'], 21)
        self.assertEqual(llm_params['delay'], 0.2)
        self.assertEqual(llm_params['retry attempts'], 3)
        self.assertEqual(llm_params['retry timeout'], 15)
        self.assertEqual(llm_params['proxy'], '')

    def test_dedupe_provider_profiles_prefers_selected_profile(self):
        json_profile = default_profile('DeepSeek')
        json_profile['id'] = 'deepseek-json'
        json_profile['name'] = 'DeepSeek JSON'
        xml_profile = default_profile('DeepSeek')
        xml_profile['id'] = 'deepseek-xml'
        xml_profile['name'] = 'DeepSeek XML'

        profiles = dedupe_profiles([json_profile, xml_profile], selected_profile_id='deepseek-xml')

        deepseek_profiles = [p for p in profiles if p['provider'] == 'DeepSeek']
        self.assertEqual(len(deepseek_profiles), 1)
        self.assertEqual(deepseek_profiles[0]['id'], 'deepseek')
        self.assertEqual(deepseek_profiles[0]['name'], 'DeepSeek')
        self.assertNotIn('prompt mode', deepseek_profiles[0])

    def test_copied_builtin_profile_is_not_deduped_back_into_builtin(self):
        builtin = default_profile('OpenAI')
        copied = copy_profile(builtin)

        profiles = dedupe_profiles([builtin, copied], selected_profile_id=copied['id'])

        self.assertEqual(len(profiles), 2)
        self.assertTrue(profile_by_id(profiles, copied['id']))

    def test_custom_profiles_with_same_settings_keep_separate_ids(self):
        first = copy_profile(default_profile('OpenAI'))
        first['id'] = 'custom-one'
        second = copy.deepcopy(first)
        second['id'] = 'custom-two'

        profiles = dedupe_profiles([first, second])

        self.assertTrue(profile_by_id(profiles, 'custom-one'))
        self.assertTrue(profile_by_id(profiles, 'custom-two'))

    def test_deprecated_deepseek_models_fall_back_deterministically(self):
        reasoner = profile_from_old_settings(
            'LLM_API_Translator',
            {'endpoint': 'https://api.deepseek.com', 'apikey': 'k', 'model': 'deepseek-reasoner'},
            secret_store=SecretStore(False),
        )
        chat = profile_from_old_settings(
            'LLM_API_Translator',
            {'endpoint': 'https://api.deepseek.com', 'apikey': 'k', 'model': 'deepseek-chat'},
            secret_store=SecretStore(False),
        )

        self.assertEqual(reasoner['model'], 'deepseek-v4-flash')
        self.assertEqual(reasoner['thinking level'], 'high')
        self.assertEqual(chat['model'], 'deepseek-v4-flash')
        self.assertEqual(chat['thinking level'], 'None')

    def test_deepseek_ignores_openai_preset_model_without_override(self):
        profile = profile_from_old_settings(
            'ChatGPT',
            {'3rd party api url': 'https://api.deepseek.com', 'api key': 'k', 'model': 'gpt-4o'},
            secret_store=SecretStore(False),
        )

        self.assertEqual(profile['provider'], 'DeepSeek')
        self.assertEqual(profile['model'], 'deepseek-v4-flash')

    def test_restore_builtins_keeps_user_profiles(self):
        custom = copy_profile(default_profile('OpenAI'))
        custom['id'] = 'custom'
        profiles = restore_builtin_profiles([custom, {'id': 'openai', 'provider': 'OpenAI', 'built_in': True}])

        self.assertEqual(profiles[0]['id'], 'custom')
        self.assertTrue(any(p['id'] == 'openai' and p.get('built_in') for p in profiles))

    def test_restore_builtins_keeps_filled_builtin_api_key(self):
        openai = default_profile('OpenAI')
        openai['api key'] = SecretStore().store('openai', 'sk-demo')
        openai['model'] = 'temporary-model'

        profiles = restore_builtin_profiles([openai])
        restored = profile_by_id(profiles, 'openai')

        self.assertEqual(restored['model'], 'gpt-5.5')
        self.assertEqual(SecretStore().resolve(restored['api key']).value, 'sk-demo')

    def test_openai_builtin_includes_current_model_seed(self):
        profile = default_profile('OpenAI')

        self.assertEqual(profile['model'], 'gpt-5.5')
        self.assertIn('gpt-5.5', profile['model options'])
        self.assertIn('None', profile['thinking level options'])
        self.assertNotIn('none', profile['thinking level options'])
        self.assertIn('system prompt', profile)
        self.assertNotIn('prompt template', profile)
        self.assertNotIn('chat system template', profile)
        self.assertNotIn('max requests per minute', profile)
        self.assertNotIn('delay', profile)
        self.assertNotIn('retry attempts', profile)
        self.assertNotIn('retry timeout', profile)
        self.assertNotIn('proxy', profile)

    def test_interim_profile_runtime_params_move_to_llm_translator(self):
        profile = default_profile('DeepSeek')
        profile.update({
            'max requests per minute': 11,
            'delay': 0.4,
            'retry attempts': 6,
            'retry timeout': 9,
            'proxy': 'http://127.0.0.1:7890',
        })
        module = {
            'translator': 'LLMTranslator',
            'translator_params': {},
            'llm_profiles': [profile],
            'llm_profile': 'deepseek',
        }

        migrate_module_llm_profiles(module, SecretStore(enable_keyring=False))

        selected = profile_by_id(module['llm_profiles'], module['llm_profile'])
        self.assertNotIn('max requests per minute', selected)
        llm_params = module['translator_params']['LLMTranslator']
        self.assertEqual(llm_params['max requests per minute'], 11)
        self.assertEqual(llm_params['delay'], 0.4)
        self.assertEqual(llm_params['retry attempts'], 6)
        self.assertEqual(llm_params['retry timeout'], 9)
        self.assertEqual(llm_params['proxy'], 'http://127.0.0.1:7890')


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
        profile['api key'] = 'sk-demo'
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='openai')

        saved = json_dump_program_config(cfg)
        saved_dict = json.loads(saved)
        api_key = saved_dict['llm_profiles'][0]['api key']

        self.assertNotIn('sk-demo', saved)
        self.assertTrue(is_portable_secret(api_key))
        self.assertEqual(SecretStore().resolve(api_key).value, 'sk-demo')

    def test_saved_config_keeps_obfuscated_secret_not_resolved_secret(self):
        profile = default_profile('OpenAI')
        profile['api key'] = SecretStore().store('openai', 'sk-demo')
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='openai')

        saved = json_dump_program_config(cfg)

        self.assertNotIn('sk-demo', saved)

    def test_saved_config_omits_internal_provider_field(self):
        profile = default_profile('DeepSeek')
        profile['prompt template'] = 'old prompt'
        profile['chat system template'] = 'old system'
        profile['retry attempts'] = 7
        profile['proxy'] = 'http://127.0.0.1:7890'
        cfg = ModuleConfig(llm_profiles=[profile], llm_profile='deepseek')

        saved = json_dump_program_config(cfg)

        self.assertNotIn('"provider"', saved)
        self.assertNotIn('"prompt mode"', saved)
        self.assertNotIn('"prompt mode options"', saved)
        self.assertNotIn('"prompt template"', saved)
        self.assertNotIn('"chat system template"', saved)
        self.assertNotIn('"retry attempts"', saved)
        self.assertNotIn('"proxy"', saved)


if __name__ == '__main__':
    unittest.main()
