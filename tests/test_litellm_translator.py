import json
import importlib
import os
import sys
import types
import unittest
from unittest import mock
from types import SimpleNamespace


def _make_litellm_stub():
    """Install a fake litellm module so tests run without the real package."""
    fake = types.ModuleType("litellm")
    fake.completion = mock.MagicMock(name="litellm.completion")

    exc_mod = types.ModuleType("litellm.exceptions")
    exc_mod.RateLimitError = type("RateLimitError", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.APIConnectionError = type("APIConnectionError", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.Timeout = type("Timeout", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.InternalServerError = type("InternalServerError", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.AuthenticationError = type("AuthenticationError", (Exception,), {"__module__": "litellm.exceptions"})
    exc_mod.NotFoundError = type("NotFoundError", (Exception,), {"__module__": "litellm.exceptions"})

    fake.exceptions = exc_mod
    sys.modules["litellm"] = fake
    sys.modules["litellm.exceptions"] = exc_mod
    return fake


_litellm_stub = _make_litellm_stub()


def _load_trans_litellm():
    """Load trans_litellm.py directly, bypassing the heavy ballontranslator package imports."""
    base_mod = types.ModuleType("ballontranslator.modules.translators.base")
    base_mod.BaseTranslator = type("BaseTranslator", (), {
        "get_param_value": lambda self, k: "",
        "logger": mock.MagicMock(),
    })
    base_mod.register_translator = lambda name: lambda cls: cls

    parent = types.ModuleType("ballontranslator.modules.translators")
    parent.base = base_mod
    sys.modules.setdefault("ballontranslator", types.ModuleType("ballontranslator"))
    sys.modules.setdefault("ballontranslator.modules", types.ModuleType("ballontranslator.modules"))
    sys.modules["ballontranslator.modules.translators"] = parent
    sys.modules["ballontranslator.modules.translators.base"] = base_mod

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..", "ballontranslator", "modules", "translators", "trans_litellm.py",
    )
    spec = importlib.util.spec_from_file_location(
        "ballontranslator.modules.translators.trans_litellm",
        file_path,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "ballontranslator.modules.translators"
    spec.loader.exec_module(mod)
    return mod


_mod = _load_trans_litellm()
LiteLLMTranslator = _mod.LiteLLMTranslator
_is_transient_error = _mod._is_transient_error


def _make_completion_response(translations_json):
    """Build a fake litellm.completion() response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(translations_json))
            )
        ],
        usage=SimpleNamespace(total_tokens=42),
    )


def _make_translator(**overrides):
    """Instantiate LiteLLMTranslator with sensible test defaults."""
    t = LiteLLMTranslator.__new__(LiteLLMTranslator)
    t.logger = mock.MagicMock()
    t.lang_source = "English"
    t.lang_target = "Japanese"
    t.token_count = 0
    t.token_count_last = 0
    t.last_request_time = 0

    defaults = {
        "model": "openai/gpt-4o-mini",
        "api_key": "sk-test-key",
        "api_base": "",
        "system_prompt": "You are a translator.",
        "temperature": 0.1,
        "max_tokens": 4096,
        "retry_attempts": 2,
        "retry_timeout": 0,
        "invalid_repeat_count": 1,
        "delay": 0,
    }
    defaults.update(overrides)
    t.get_param_value = mock.MagicMock(side_effect=lambda k: defaults.get(k, ""))
    t.lang_map = {
        "English": "English",
        "Japanese": "Japanese",
    }

    return t


class TestLiteLLMTranslator(unittest.TestCase):

    def test_successful_translation(self):
        """Happy path: correct JSON response with matching IDs."""
        t = _make_translator()
        response = _make_completion_response({
            "translations": [
                {"id": 1, "translation": "hello in Japanese"},
                {"id": 2, "translation": "world in Japanese"},
            ]
        })
        _litellm_stub.completion.return_value = response

        result = t._translate(["hello", "world"])

        self.assertEqual(result, ["hello in Japanese", "world in Japanese"])
        _litellm_stub.completion.assert_called_once()
        call_kwargs = _litellm_stub.completion.call_args[1]
        self.assertTrue(call_kwargs["drop_params"])
        self.assertEqual(call_kwargs["model"], "openai/gpt-4o-mini")
        self.assertEqual(call_kwargs["api_key"], "sk-test-key")

    def test_drop_params_always_set(self):
        """drop_params=True must always be in the completion call."""
        t = _make_translator()
        response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.return_value = response

        t._translate(["test"])

        call_kwargs = _litellm_stub.completion.call_args[1]
        self.assertIn("drop_params", call_kwargs)
        self.assertTrue(call_kwargs["drop_params"])

    def test_api_key_omitted_when_empty(self):
        """When api_key is empty, it should NOT be passed to litellm (fallback to env vars)."""
        t = _make_translator(api_key="")
        response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.return_value = response

        t._translate(["test"])

        call_kwargs = _litellm_stub.completion.call_args[1]
        self.assertNotIn("api_key", call_kwargs)

    def test_api_base_omitted_when_empty(self):
        """When api_base is empty, it should NOT be passed to litellm."""
        t = _make_translator(api_base="")
        response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.return_value = response

        t._translate(["test"])

        call_kwargs = _litellm_stub.completion.call_args[1]
        self.assertNotIn("api_base", call_kwargs)

    def test_api_base_forwarded_when_set(self):
        """When api_base is set, it must be forwarded to litellm."""
        t = _make_translator(api_base="http://localhost:4000/v1")
        response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.return_value = response

        t._translate(["test"])

        call_kwargs = _litellm_stub.completion.call_args[1]
        self.assertEqual(call_kwargs["api_base"], "http://localhost:4000/v1")

    def test_empty_input_returns_empty(self):
        """Empty source list should return empty list without calling API."""
        t = _make_translator()
        _litellm_stub.completion.reset_mock()

        result = t._translate([])

        self.assertEqual(result, [])
        _litellm_stub.completion.assert_not_called()

    def test_missing_ids_fallback_to_source(self):
        """If response has fewer translations than sources, missing ones use source text."""
        t = _make_translator(invalid_repeat_count=0)
        response = _make_completion_response({
            "translations": [
                {"id": 1, "translation": "first translated"},
            ]
        })
        _litellm_stub.completion.return_value = response

        result = t._translate(["first", "second"])

        self.assertEqual(result[0], "first translated")
        self.assertEqual(result[1], "second")

    def test_null_response_triggers_retry(self):
        """None response should trigger retry, then return source on exhaustion."""
        t = _make_translator(retry_attempts=1, retry_timeout=0)
        empty_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=None,
        )
        _litellm_stub.completion.return_value = empty_response

        result = t._translate(["hello"])

        self.assertEqual(result, ["hello"])
        self.assertEqual(_litellm_stub.completion.call_count, 2)

    def test_malformed_json_triggers_retry(self):
        """Malformed JSON response should trigger retry."""
        t = _make_translator(retry_attempts=1, retry_timeout=0)
        bad_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json at all"))],
            usage=None,
        )
        _litellm_stub.completion.return_value = bad_response

        result = t._translate(["hello"])

        self.assertEqual(result, ["hello"])

    def test_transient_error_retries(self):
        """RateLimitError (transient) should trigger retry."""
        t = _make_translator(retry_attempts=2, retry_timeout=0)
        rate_limit_err = _litellm_stub.exceptions.RateLimitError("rate limited")
        good_response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.side_effect = [rate_limit_err, good_response]

        result = t._translate(["hello"])

        self.assertEqual(result, ["ok"])
        self.assertEqual(_litellm_stub.completion.call_count, 2)

    def test_auth_error_does_not_retry(self):
        """AuthenticationError (non-transient) should raise immediately, not retry."""
        t = _make_translator(retry_attempts=3, retry_timeout=0)
        auth_err = _litellm_stub.exceptions.AuthenticationError("bad key")
        _litellm_stub.completion.side_effect = auth_err

        with self.assertRaises(type(auth_err)):
            t._translate(["hello"])

        self.assertEqual(_litellm_stub.completion.call_count, 1)

    def test_markdown_code_block_extraction(self):
        """Response wrapped in ```json ... ``` should be extracted correctly."""
        t = _make_translator()
        wrapped = '```json\n{"translations": [{"id": 1, "translation": "extracted"}]}\n```'
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=wrapped))],
            usage=SimpleNamespace(total_tokens=10),
        )
        _litellm_stub.completion.return_value = response

        result = t._translate(["test"])

        self.assertEqual(result, ["extracted"])

    def test_simple_dict_format_fallback(self):
        """Response as {"1": "text"} should be parsed via fallback."""
        t = _make_translator()
        simple = json.dumps({"1": "fallback parsed"})
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=simple))],
            usage=SimpleNamespace(total_tokens=10),
        )
        _litellm_stub.completion.return_value = response

        result = t._translate(["test"])

        self.assertEqual(result, ["fallback parsed"])

    def test_token_counting(self):
        """Token count should accumulate from usage."""
        t = _make_translator()
        response = _make_completion_response({
            "translations": [{"id": 1, "translation": "ok"}]
        })
        _litellm_stub.completion.return_value = response

        t._translate(["hello"])

        self.assertEqual(t.token_count, 42)
        self.assertEqual(t.token_count_last, 42)

    def setUp(self):
        _litellm_stub.completion.reset_mock()
        _litellm_stub.completion.side_effect = None


class TestIsTransientError(unittest.TestCase):

    def test_rate_limit_is_transient(self):
        err = _litellm_stub.exceptions.RateLimitError("429")
        self.assertTrue(_is_transient_error(err))

    def test_timeout_is_transient(self):
        err = _litellm_stub.exceptions.Timeout("timed out")
        self.assertTrue(_is_transient_error(err))

    def test_connection_error_is_transient(self):
        err = _litellm_stub.exceptions.APIConnectionError("connection refused")
        self.assertTrue(_is_transient_error(err))

    def test_auth_error_is_not_transient(self):
        err = _litellm_stub.exceptions.AuthenticationError("bad key")
        self.assertFalse(_is_transient_error(err))

    def test_not_found_is_not_transient(self):
        err = _litellm_stub.exceptions.NotFoundError("model not found")
        self.assertFalse(_is_transient_error(err))

    def test_value_error_is_transient(self):
        self.assertTrue(_is_transient_error(ValueError("empty response")))

    def test_json_decode_error_is_transient(self):
        self.assertTrue(_is_transient_error(json.JSONDecodeError("bad", "", 0)))


if __name__ == "__main__":
    unittest.main()
