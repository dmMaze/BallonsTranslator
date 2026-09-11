import copy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_chat import LLMChatRequestError
from ballontranslator.modules.llm_codex import CodexRequestError, request_codex_completion
from ballontranslator.modules.context.token_usage import format_run_token_usage
from ballontranslator.modules.ocr.ocr_llm import LLMOCR
from ballontranslator.modules.translators.trans_llm import LLMTranslator
from ballontranslator.utils.config import ModuleConfig, ProgramConfig, pcfg, json_dump_program_config
from ballontranslator.utils.config import LLMTranslateContext
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.textblock import TextBlock
from ballontranslator.utils.llm_profiles import (
    default_profile, default_profiles, load_profiles, profile_from_config,
)


FAKE_SERVER = r'''
import json, subprocess, sys, time
scenario, log = sys.argv[1:]
def emit(value):
    print(json.dumps(value), flush=True)
def event(method, **params):
    emit({'method': method, 'params': {'threadId': 'thread-1', **params}})
for line in sys.stdin:
    request = json.loads(line)
    with open(log, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(request) + '\n')
    method = request['method']
    if method == 'initialized':
        continue
    result = {}
    if method == 'initialize' and scenario == 'hung_child':
        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        time.sleep(60)
    if method == 'initialize' and scenario == 'exit':
        print('fixture startup failure', file=sys.stderr, flush=True)
        sys.exit(1)
    if method == 'initialize' and scenario == 'malformed':
        print('not JSON', flush=True)
        continue
    if method == 'initialize' and scenario == 'invalid_result':
        emit({'id': request['id']})
        continue
    if method == 'account/read':
        result = {'account': None if scenario == 'logged_out' else {
            'type': 'apiKey' if scenario == 'api_key' else 'chatgpt'}}
    elif method == 'thread/start':
        result = {'thread': {'id': 'thread-1'}}
    elif method == 'turn/start':
        if scenario == 'rpc_error':
            emit({'id': request['id'], 'error': {'code': -1, 'message': 'unsupported model'}})
            continue
        event('turn/started', turn={'id': 'turn-1'})
        if scenario == 'invalid_event':
            emit({'method': 'turn/completed', 'params': []})
            continue
        if scenario != 'early':
            emit({'id': request['id'], 'result': {'turn': {'id': 'turn-1'}}})
        if scenario in ('timeout', 'cancel'):
            continue
        if scenario == 'tool':
            emit({'id': 'approval-1', 'method': 'item/commandExecution/requestApproval', 'params': {}})
            continue
        if scenario == 'disconnect':
            sys.exit(1)
        event('item/completed', item={'type': 'agentMessage', 'id': 'comment',
              'text': 'Thinking...', 'phase': 'commentary'})
        event('item/agentMessage/delta', itemId='answer', delta='ignored partial')
        event('item/completed', item={'type': 'agentMessage', 'id': 'answer',
              'text': '{"1":"譯文一","2":"譯文二"}', 'phase': 'final_answer'})
        tokens = {
            'inputTokens': 100, 'outputTokens': 20, 'totalTokens': 120,
            'cachedInputTokens': 50, 'reasoningOutputTokens': 5}
        if scenario != 'no_usage':
            event('thread/tokenUsage/updated', tokenUsage={'last': tokens, 'total': tokens})
        if scenario == 'repeated_usage':
            total = {'inputTokens': 200, 'outputTokens': 50, 'totalTokens': 250,
                     'cachedInputTokens': 80, 'reasoningOutputTokens': 15, 'cacheWriteInputTokens': 10}
            for _ in range(2):
                event('thread/tokenUsage/updated', tokenUsage={'last': tokens, 'total': total})
        status = 'interrupted' if scenario == 'interrupted' else 'failed' if scenario in ('quota', 'context') else 'completed'
        event('turn/completed', turn={'id': 'turn-1', 'status': status,
            'error': {'message': 'usage limit reached', 'codexErrorInfo':
                      'contextWindowExceeded' if scenario == 'context' else 'usageLimitExceeded'}})
        if scenario == 'early':
            emit({'id': request['id'], 'result': {'turn': {'id': 'turn-1'}}})
        continue
    emit({'id': request['id'], 'result': result})
'''


class CodexTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.script = Path(self.temp.name, 'server.py')
        self.script.write_text(FAKE_SERVER, encoding='utf-8')
        self.log = Path(self.temp.name, 'requests.jsonl')
        self.scenario = 'success'
        self.profile = default_profile('Codex')
        self.profile.thinking_level = 'low'
        self.processes = []
        popen = subprocess.Popen

        def launch(command, **kwargs):
            if command[0] == 'taskkill':
                return popen(command, **kwargs)
            self.command = command
            self.child_env = kwargs['env']
            process = popen([sys.executable, str(self.script), self.scenario, str(self.log)], **kwargs)
            self.processes.append(process)
            return process

        self.addCleanup(mock.patch.stopall)
        mock.patch('ballontranslator.modules.llm_codex.shutil.which', return_value=sys.executable).start()
        mock.patch('ballontranslator.modules.llm_codex.subprocess.Popen', side_effect=launch).start()
        self.args = {
            'model': 'gpt-5.6-sol',
            'messages': [
                {'role': 'system', 'content': 'Keep IDs and translate.'},
                {'role': 'user', 'content': '["previous"]'},
                {'role': 'assistant', 'content': '{"1":"previous translation"}'},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': '["first", "second"]'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}},
                ]},
            ],
            'response_format': {'json_schema': {'schema': {'type': 'object'}}},
        }

    def tearDown(self) -> None:
        for process in self.processes:
            self.assertIsNotNone(process.poll(), 'Codex process leaked')
            self.assertTrue(process.stdin.closed)
            self.assertTrue(process.stdout.closed)

    def requests(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_messages_schema_usage_and_ephemeral_isolation(self) -> None:
        before = copy.deepcopy(self.args)
        with mock.patch.dict(os.environ, {'OPENAI_API_KEY': 'unused-key'}):
            result = request_codex_completion(self.profile, self.args)
        self.assertEqual(json.loads(result.content), {'1': '譯文一', '2': '譯文二'})
        self.assertEqual(result.usage.total_tokens, 120)
        self.assertEqual(result.usage.prompt_tokens_details['cached_tokens'], 50)
        self.assertEqual(self.args, before)
        self.assertNotIn('OPENAI_API_KEY', self.child_env)
        requests = {r['method']: r['params'] for r in self.requests()}
        thread = requests['thread/start']
        self.assertTrue(thread['ephemeral'])
        self.assertEqual(thread['modelProvider'], 'openai')
        self.assertEqual(thread['sandbox'], 'read-only')
        self.assertEqual(thread['environments'], [])
        self.assertFalse(Path(thread['cwd']).exists())
        history = requests['thread/inject_items']['items']
        self.assertEqual([m['role'] for m in history], ['system', 'user', 'assistant'])
        self.assertEqual(history[-1]['content'][0]['type'], 'output_text')
        turn = requests['turn/start']
        self.assertEqual(turn['input'][1], {'type': 'image', 'url': 'data:image/png;base64,AAAA'})
        self.assertEqual(turn['outputSchema'], {'type': 'object'})
        self.assertEqual(turn['effort'], 'low')

    def test_completion_before_start_response_is_not_lost(self) -> None:
        self.scenario = 'early'
        self.assertIn('譯文一', request_codex_completion(self.profile, self.args).content)

    def test_session_persistence_is_opt_in_and_can_be_disabled(self) -> None:
        for enabled in (False, True, False):
            self.profile.codex_save_sessions = enabled
            self.assertEqual(request_codex_completion(self.profile, self.args).usage.total_tokens, 120)
        requests = self.requests()
        self.assertEqual([r['params']['ephemeral'] for r in requests if r['method'] == 'thread/start'],
                         [True, False, True])
        self.assertNotIn('thread/resume', [r['method'] for r in requests])
        count = len(self.processes)
        self.profile.codex_save_sessions = 'false'
        with self.assertRaisesRegex(CodexRequestError, 'Save Codex Sessions'):
            request_codex_completion(self.profile, self.args)
        self.assertEqual(len(self.processes), count)

    def test_failed_turns_and_protocol_errors_stop_without_resubmission(self) -> None:
        for scenario in ('logged_out', 'api_key', 'exit', 'malformed', 'invalid_result',
                         'invalid_event', 'rpc_error', 'quota', 'tool', 'disconnect'):
            with self.subTest(scenario=scenario):
                self.scenario = scenario
                with self.assertRaises(CodexRequestError):
                    request_codex_completion(self.profile, self.args)

    def test_context_error_reaches_existing_context_recovery(self) -> None:
        self.scenario = 'context'
        with self.assertRaises(LLMChatRequestError):
            request_codex_completion(self.profile, self.args)

    def test_timeout_interrupts_turn_and_cleans_up(self) -> None:
        self.scenario = 'timeout'
        self.profile.codex_timeout = 1
        started = time.monotonic()
        with self.assertRaisesRegex(CodexRequestError, 'timed out'):
            request_codex_completion(self.profile, self.args)
        self.assertLess(time.monotonic() - started, 4)
        self.assertIn('turn/interrupt', [r['method'] for r in self.requests()])

    def test_cancellation_during_request_and_before_launch(self) -> None:
        self.scenario = 'cancel'
        stopped = threading.Event()
        timer = threading.Timer(0.3, stopped.set)
        timer.start()
        self.addCleanup(timer.cancel)
        with self.assertRaises(LLMRequestStopped):
            request_codex_completion(self.profile, self.args, stopped)
        self.assertIn('turn/interrupt', [r['method'] for r in self.requests()])
        count = len(self.processes)
        with self.assertRaises(LLMRequestStopped):
            request_codex_completion(self.profile, self.args, stopped)
        self.assertEqual(len(self.processes), count)

    def test_stalled_launcher_and_child_are_terminated(self) -> None:
        self.scenario = 'hung_child'
        self.profile.codex_timeout = 1
        started = time.monotonic()
        with self.assertRaisesRegex(CodexRequestError, 'timed out'):
            request_codex_completion(self.profile, self.args)
        self.assertLess(time.monotonic() - started, 6)

    def test_missing_cli_and_invalid_timeout_do_not_launch(self) -> None:
        with mock.patch('ballontranslator.modules.llm_codex.shutil.which', return_value=None):
            with self.assertRaisesRegex(CodexRequestError, 'Install the official'):
                request_codex_completion(self.profile, self.args)
        self.profile.codex_timeout = 0
        with self.assertRaisesRegex(CodexRequestError, 'Timeout'):
            request_codex_completion(self.profile, self.args)
        self.assertFalse(self.processes)

    def test_unsupported_thinking_is_rejected_before_launch(self) -> None:
        for model, effort in (('gpt-6-astra', 'Disabled'), ('gpt-5.6-sol', 'minimal'),
                              ('gpt-5.6-luna', 'ultra'), ('gpt-5.5', 'max'),
                              ('custom-model', 'low'), ('gpt-6-astra', 'Auto'),
                              ('gpt-5.5', ''), ('custom-model', 'Auto'), ('gpt-6-astra', 'none')):
            with self.subTest(model=model, effort=effort):
                self.args['model'] = model
                self.profile.thinking_level = effort
                with self.assertRaisesRegex(CodexRequestError, 'does not support thinking level'):
                    request_codex_completion(self.profile, self.args)
        self.assertFalse(self.processes)

    def test_supported_thinking_values_reach_turn_start(self) -> None:
        for model, effort in (('gpt-6-astra', 'ultra'), ('gpt-5.6-sol', 'max'),
                              ('gpt-5.6-luna', 'max'), ('gpt-5.5', 'xhigh'),
                              ('gpt-5.5', 'medium'), ('gpt-5.6-sol', 'none'),
                              ('gpt-5.6-terra', 'none'), ('gpt-5.6-luna', 'none'), ('gpt-5.5', 'none')):
            with self.subTest(model=model, effort=effort):
                self.args['model'] = model
                self.profile.thinking_level = effort
                request_codex_completion(self.profile, self.args)
                turn = [r['params'] for r in self.requests() if r['method'] == 'turn/start'][-1]
                self.assertEqual(turn['effort'], effort)

    def test_translation_and_ocr_usage_reach_info_log_once_per_response(self) -> None:
        import numpy as np

        logger = logging.getLogger('BallonTranslator')
        for scenario in ('repeated_usage', 'no_usage'):
            with self.subTest(scenario=scenario), mock.patch.object(pcfg, 'module', ModuleConfig(
                llm_profiles=[self.profile], translator_llm_id='codex', ocr_llm_id='codex',
            )):
                self.scenario = scenario
                usage_path = Path(self.temp.name, f'{scenario}.log')
                handler = logging.FileHandler(usage_path, encoding='utf-8')
                handler.setLevel(logging.INFO)
                previous_level = logger.level
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)
                try:
                    translator = LLMTranslator('English', '繁體中文', **{'delay': 0})
                    translator.translate(['first', 'second'])
                    ocr = LLMOCR(**{'delay': 0})
                    ocr.ocr_img(np.zeros((12, 12, 3), dtype=np.uint8))
                finally:
                    logger.removeHandler(handler)
                    handler.close()
                    logger.setLevel(previous_level)
                lines = [line for line in usage_path.read_text(encoding='utf-8').splitlines()
                         if 'token usage:' in line]
                self.assertEqual(len(lines), 2)
                self.assertIn('LLM token usage:', lines[0])
                self.assertIn('LLM OCR token usage:', lines[1])
                run_usage = format_run_token_usage([translator.usage_totals, ocr.usage_totals])
                self.assertIn('requests=2,', run_usage)
                if scenario == 'no_usage':
                    self.assertTrue(all('usage=unavailable' in line for line in lines))
                    self.assertEqual(ocr.token_count, 0)
                    self.assertIn('missing_usage_requests=2', run_usage)
                    self.assertIn('estimated_cost_usd=unavailable', run_usage)
                else:
                    for field in ('prompt=200', 'completion=50', 'reasoning=15',
                                  'total=250', 'cache_hit=80', 'cache_write=10'):
                        self.assertTrue(all(field in line for line in lines), field)
                    self.assertEqual(ocr.token_count, 250)
                    self.assertIn('cumulative_total=250', lines[1])
                    self.assertIn('total_tokens=500,', run_usage)
                    self.assertIn('estimated_cost_usd=0.003044,', run_usage)

    def test_real_translator_and_ocr_route_without_openai_client(self) -> None:
        import numpy as np

        self.profile.thinking_level = 'high'
        with mock.patch.object(pcfg, 'module', ModuleConfig(
            llm_profiles=[self.profile], translator_llm_id='codex', ocr_llm_id='codex',
            ocr_llm_page_level=True,
        )), mock.patch('ballontranslator.modules.llm_chat.LLMChatRequester._openai_module',
                       side_effect=AssertionError('HTTP client must not be used')):
            translator = LLMTranslator('English', '繁體中文', **{'delay': 0})
            self.assertEqual(translator.translate(['first', 'second']), ['譯文一', '譯文二'])
            ocr = LLMOCR(**{'delay': 0})
            self.assertIn('譯文一', ocr.ocr_img(np.zeros((12, 12, 3), dtype=np.uint8)))
            self.assertEqual(ocr.token_count, 120)
            turns = [r['params'] for r in self.requests() if r['method'] == 'turn/start']
            self.assertEqual([turn['effort'] for turn in turns], ['high', 'none'])
            self.assertEqual(len({request['params']['model'] for request in self.requests()
                                  if request['method'] == 'thread/start'}), 1)
            self.scenario = 'quota'
            count = len(self.processes)
            with self.assertRaises(CodexRequestError):
                translator.translate(['first', 'second'])
            self.assertEqual(len(self.processes), count + 1)
            blocks = [TextBlock(xyxy=[0, 0, 12, 12], text=['original OCR'], translation='saved translation')]
            with self.assertRaises(CodexRequestError):
                translator.translate_textblk_lst(blocks)
            self.assertEqual(blocks[0].translation, 'saved translation')
            with self.assertRaises(CodexRequestError):
                ocr.run_ocr(np.zeros((12, 12, 3), dtype=np.uint8), blocks, full_page=True)
            self.assertEqual(blocks[0].get_text(), 'original OCR')


class CodexProfileTest(unittest.TestCase):
    def test_old_profiles_and_bad_optional_settings_load(self) -> None:
        self.assertEqual(profile_from_config({'id': 'old'}).transport, 'OpenAI-compatible')
        self.assertFalse(profile_from_config({'id': 'old'}).codex_save_sessions)
        profile = load_profiles([{'id': 'bad', 'transport': [], 'codex_timeout': 'no',
                                 'codex_executable': None, 'codex_save_sessions': 'false',
                                 'prompt': 'keep this'}])[0]
        self.assertEqual(profile.transport, 'OpenAI-compatible')
        self.assertEqual(profile.codex_timeout, 180)
        self.assertEqual(profile.codex_executable, 'codex')
        self.assertFalse(profile.codex_save_sessions)
        self.assertEqual(profile.prompt, 'keep this')
        self.assertEqual(default_profiles()[0].id, 'openai')

    def test_codex_profile_survives_config_save_and_load(self) -> None:
        profile = default_profile('Codex')
        profile.codex_executable = 'C:/tools/codex.exe'
        profile.codex_timeout = 300
        profile.codex_save_sessions = True
        cfg = ProgramConfig(module=ModuleConfig(llm_profiles=[profile], translator_llm_id='codex'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'config.json')
            path.write_text(json_dump_program_config(cfg), encoding='utf-8')
            loaded = ProgramConfig.load(str(path)).module.llm_profiles[0]
        self.assertEqual(loaded.transport, 'Codex App Server')
        self.assertEqual(loaded.codex_executable, profile.codex_executable)
        self.assertEqual(loaded.codex_timeout, 300)
        self.assertTrue(loaded.codex_save_sessions)


@unittest.skipUnless(os.environ.get('BALLONTRANSLATOR_CODEX_LIVE') == '1',
                     'Opt-in: requires official Codex CLI ChatGPT login and uses subscription quota.')
class CodexLiveTest(unittest.TestCase):
    def test_ocr_vision_history_and_project_reload(self) -> None:
        import cv2
        import numpy as np

        profile = default_profile('Codex')
        profile.codex_executable = os.environ.get('BALLONTRANSLATOR_CODEX_EXECUTABLE', 'codex')
        profile.thinking_level = 'low'
        config = ModuleConfig(
            llm_profiles=[profile], translator_llm_id='codex', ocr_llm_id='codex',
            ocr_llm_page_level=True, ocr_llm_sort_reading_order=True,
            llm_translate_vision=True, llm_translate_summary_memory=True,
            llm_translate_context=LLMTranslateContext.HISTORY,
        )
        image = np.full((280, 700, 3), 255, dtype=np.uint8)
        cv2.putText(image, 'HELLO', (80, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        cv2.putText(image, 'THANK YOU', (80, 215), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        with mock.patch.object(pcfg, 'module', config), tempfile.TemporaryDirectory() as directory:
            for page in ('001.png', '002.png'):
                self.assertTrue(cv2.imwrite(str(Path(directory, page)), image))
            project = ProjImgTrans(directory)
            ocr = LLMOCR(**{'delay': 0, 'retry attempts': 1})
            crop_text = ocr.run_ocr(image[:120])
            self.assertEqual(crop_text.strip().upper(), 'HELLO')
            project.pages['001.png'] = [
                TextBlock(xyxy=[65, 30, 320, 105]),
                TextBlock(xyxy=[65, 160, 400, 235]),
            ]
            result = ocr.run_ocr(image, project.pages['001.png'], full_page=True)
            self.assertEqual([block.get_text().upper() for block in result], ['HELLO', 'THANK YOU'])
            project.pages['001.png'] = result
            project.pages['002.png'] = [TextBlock(text=['Good morning!']), TextBlock(text=['See you tomorrow!'])]
            translator = LLMTranslator('English', '繁體中文', **{'delay': 0, 'retry attempts': 1})
            for page in ('001.png', '002.png'):
                translator.translate_textblk_lst(project.pages[page], project=project, page_key=page, full_page=True)
                project.mark_translation_finished(page, '繁體中文')
                translator.on_page_translation_finished(project, page)
                for block in project.pages[page]:
                    self.assertTrue(any('\u4e00' <= char <= '\u9fff' for char in block.translation))
                project.save()
            loaded = ProjImgTrans(directory)
            self.assertEqual(
                [b.translation for b in loaded.pages['002.png']],
                [b.translation for b in project.pages['002.png']],
            )
            print('Codex live OCR:', [b.get_text() for b in result])
            print('Codex live translations:', [b.translation for blocks in loaded.pages.values() for b in blocks])


if __name__ == '__main__':
    unittest.main()
