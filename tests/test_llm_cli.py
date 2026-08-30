import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_cli import (
    CLIInvocation,
    LLMCLIError,
    LLMCLINonRetryableError,
    _run_invocation,
    _cli_environment,
    build_cli_invocation,
    parse_cli_output,
    render_cli_prompt,
    resolve_cli_executable,
)
from ballontranslator.utils.llm_profiles import new_cli_profile


class LLMCLITest(unittest.TestCase):
    def test_cli_profiles_are_text_only(self):
        for backend in ('codex', 'claude', 'antigravity', 'grok'):
            with self.subTest(backend=backend):
                profile = new_cli_profile(backend)
                self.assertEqual(profile.transport, 'cli')
                self.assertEqual(profile.cli_backend, backend)
                self.assertTrue(profile.support_text)
                self.assertFalse(profile.support_vision)
                self.assertFalse(profile.support_image)
                self.assertFalse(profile.require_api_key)

    def test_prompt_data_stays_off_argv(self):
        source = 'private source text'
        prompt = render_cli_prompt([
            {'role': 'system', 'content': 'Return JSON.'},
            {'role': 'user', 'content': source},
        ])
        schema = {
            'type': 'object',
            'properties': {'1': {'type': 'string'}},
            'required': ['1'],
            'additionalProperties': False,
        }
        with tempfile.TemporaryDirectory() as workdir:
            for backend in ('codex', 'claude', 'antigravity', 'grok'):
                with self.subTest(backend=backend):
                    profile = new_cli_profile(backend)
                    profile.cli_executable = sys.executable
                    invocation = build_cli_invocation(
                        profile, prompt, schema, workdir
                    )
                    self.assertNotIn(source, ' '.join(invocation.argv))
                    if backend == 'grok':
                        prompt_path = invocation.argv[
                            invocation.argv.index('--prompt-file') + 1
                        ]
                        self.assertIn(source, Path(prompt_path).read_text())
                        self.assertEqual(invocation.stdin, '')
                    else:
                        self.assertIn(source, invocation.stdin)

    def test_provider_outputs_are_normalized(self):
        expected = '{"1":"번역"}'
        cases = {
            'codex': expected,
            'claude': json.dumps({
                'type': 'result',
                'is_error': False,
                'structured_output': {'1': '번역'},
            }),
            'antigravity': '\n'.join((
                json.dumps({'event': 'init'}),
                json.dumps({
                    'event': 'result',
                    'result': {
                        'status': 'SUCCESS',
                        'response': expected,
                    },
                }),
            )),
            'grok': json.dumps({
                'structuredOutput': {'1': '번역'},
                'stopReason': 'end_turn',
            }),
        }
        for backend, stdout in cases.items():
            with self.subTest(backend=backend):
                self.assertEqual(
                    json.loads(parse_cli_output(backend, stdout)),
                    {'1': '번역'},
                )

    def test_grok_authentication_error_is_non_retryable(self):
        with self.assertRaises(LLMCLINonRetryableError):
            parse_cli_output('grok', json.dumps({
                'type': 'error',
                'message': 'Not signed in. Run `grok login`.',
            }))

    def test_grok_invocation_is_isolated_and_toolless(self):
        profile = new_cli_profile('grok')
        profile.cli_executable = sys.executable
        with tempfile.TemporaryDirectory() as workdir:
            invocation = build_cli_invocation(
                profile, 'translate', {'type': 'object'}, workdir
            )

        self.assertIn('--verbatim', invocation.argv)
        self.assertEqual(
            invocation.argv[invocation.argv.index('--sandbox') + 1],
            'off',
        )
        self.assertEqual(
            invocation.argv[invocation.argv.index('--tools') + 1],
            '',
        )
        self.assertEqual(
            invocation.argv[invocation.argv.index('--deny') + 1],
            'MCPTool(*)',
        )
        self.assertIn('--no-memory', invocation.argv)

    def test_subprocess_uses_stdin_and_can_be_stopped_before_start(self):
        invocation = CLIInvocation(
            (
                sys.executable,
                '-c',
                'import sys; print(sys.stdin.read(), end="")',
            ),
            'hello',
            'test',
        )
        with tempfile.TemporaryDirectory() as workdir:
            self.assertEqual(
                _run_invocation(invocation, workdir=workdir, timeout=2),
                'hello',
            )
            stopped = threading.Event()
            stopped.set()
            with self.assertRaises(LLMRequestStopped):
                _run_invocation(
                    invocation,
                    workdir=workdir,
                    stop_event=stopped,
                    timeout=2,
                )

    def test_cli_errors_do_not_silently_return_empty_text(self):
        with self.assertRaises(LLMCLIError):
            parse_cli_output('antigravity', json.dumps({'event': 'init'}))

    def test_subprocess_environment_does_not_forward_app_secrets(self):
        old_value = os.environ.get('BALLONTRANS_TEST_SECRET')
        os.environ['BALLONTRANS_TEST_SECRET'] = 'secret'
        try:
            environment = _cli_environment()
        finally:
            if old_value is None:
                os.environ.pop('BALLONTRANS_TEST_SECRET', None)
            else:
                os.environ['BALLONTRANS_TEST_SECRET'] = old_value

        self.assertNotIn('BALLONTRANS_TEST_SECRET', environment)
        self.assertEqual(environment['NO_COLOR'], '1')

    def test_missing_explicit_executable_fails_without_retry_hint(self):
        profile = new_cli_profile('codex')
        profile.cli_executable = '/missing/codex'

        with self.assertRaises(LLMCLINonRetryableError):
            resolve_cli_executable(profile)

    def test_authentication_process_failure_is_non_retryable(self):
        invocation = CLIInvocation(
            (
                sys.executable,
                '-c',
                'import sys; print("Error authenticating", file=sys.stderr); sys.exit(1)',
            ),
            '',
            'test',
        )
        with tempfile.TemporaryDirectory() as workdir:
            with self.assertRaises(LLMCLINonRetryableError):
                _run_invocation(invocation, workdir=workdir, timeout=2)

    def test_inflight_process_stops_cooperatively(self):
        invocation = CLIInvocation(
            (sys.executable, '-c', 'import time; time.sleep(10)'),
            '',
            'test',
        )
        stopped = threading.Event()
        timer = threading.Timer(0.2, stopped.set)
        started = time.monotonic()
        timer.start()
        try:
            with tempfile.TemporaryDirectory() as workdir:
                with self.assertRaises(LLMRequestStopped):
                    _run_invocation(
                        invocation,
                        workdir=workdir,
                        stop_event=stopped,
                        timeout=5,
                    )
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 2)


if __name__ == '__main__':
    unittest.main()
