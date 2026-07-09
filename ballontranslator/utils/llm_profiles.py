from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from ballontranslator.utils.secret_store import SecretStore
from ballontranslator.utils.structures import Config, field, nested_dataclass


LLM_TRANSLATOR_KEY = "LLMTranslator"
LLM_OCR_KEY = "LLMOCR"
LLM_INPAINT_KEY = "LLMInpaint"
OLD_LLM_TRANSLATORS = ("ChatGPT", "ChatGPT_exp", "LLM_API_Translator")

THINKING_LEVEL_OPTIONS = ["None", "minimal", "low", "medium", "high", "xhigh"]
VISION_DETAIL_LEVEL_OPTIONS = ["None", "auto", "low", "high"]
PROVIDER_ALIASES = {
    "Google": "Gemini",
}

PROVIDER_DEFAULTS = {
    "OpenAI": {
        "id": "openai",
        "base_url": "https://api.openai.com/v1",
        "require_api_key": True,
        "model": "gpt-5.5",
        "support_vision": True,
        "vision_model": "gpt-5.5",
        "vision_detail_level": "auto",
        "model_options": [
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        "vision_model_options": [
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        "support_image": True,
        "image_model_options": ["gpt-image-2"]
    },
    "DeepSeek": {
        "id": "deepseek",
        "base_url": "https://api.deepseek.com",
        "require_api_key": True,
        "model": "deepseek-v4-flash",
        "model_options": ["deepseek-v4-flash", "deepseek-v4-pro"],
    },
    "Gemini": {
        "id": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "require_api_key": True,
        "model": "gemini-3.5-flash",
        "support_vision": True,
        "vision_model": "gemini-flash-latest",
        "vision_detail_level": "auto",
        "model_options": ["gemini-flash-latest", "gemini-pro-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
        "vision_model_options": ["gemini-flash-latest", "gemini-pro-latest"],
        "support_image": True,
        "image_model_options": ["gemini-3.1-flash-lite-image"],
        "image_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"
    },
    "OpenRouter": {
        "id": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "require_api_key": True,
        "model": "openai/gpt-5.5",
        "support_vision": True,
        "vision_model": "openai/gpt-5.5",
        "vision_detail_level": "auto",
        "model_options": ["openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-4o",
        "anthropic/claude-sonnet-4", "~anthropic/claude-sonnet-latest", "~x-ai/grok-latest",
        "~google/gemini-flash-latest", "~google/gemini-pro-latest",
        "qwen/qwen3.7-plus", "qwen/qwen3.7-max", "qwen/qwen3.6-plus"
        ],
        "vision_model_options": ["openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-4o",
        "anthropic/claude-sonnet-4", "~anthropic/claude-sonnet-latest", "~x-ai/grok-latest",
        "~google/gemini-flash-latest", "~google/gemini-pro-latest",
        "qwen/qwen3.7-plus", "qwen/qwen3.7-max", "qwen/qwen3.6-plus"
        ],
        "support_image": True,
        "image_base_url": "https://openrouter.ai/api/v1",
        "image_model": "black-forest-labs/flux.2-klein-4b",
        "image_model_options": ["black-forest-labs/flux.2-klein-4b"]
    },
    "LM Studio": {
        "id": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "require_api_key": False,
        "json_schema_response_format": True,
        "model": "local-model",
        "model_options": ["local-model"],
    },
    "Ollama": {
        "id": "ollama",
        "base_url": "http://localhost:11434/v1/",
        "require_api_key": False,
        "model": "llama3.1",
        "support_vision": True,
        "vision_model": "llama3.1",
        "vision_detail_level": "auto",
        "model_options": ["llama3.1", "qwen2.5", "mistral"],
        "vision_model_options": ["llama3.1", "qwen2.5", "mistral"],
    },
}

DEFAULT_TRANSLATION_PROMPT = (
    "Translate faithfully and fluently. Preserve the original meaning, tone, speaker intent, "
    "and formatting as much as possible. Keep names, honorifics, and terminology consistent."
)

def _normal_url(url: str) -> str:
    url = (url or "").strip()
    if url == "http://localhost:11434/v1/":
        return url
    return url.rstrip("/")


def _strip_model_prefix(model: str) -> str:
    model = (model or "").strip()
    if ": " in model:
        model = model.split(": ", 1)[1]
    if model == "(override model field)":
        model = ""
    return model


@nested_dataclass
class LLMProfile(Config):
    """Typed persistent LLM profile config.

    Example:
        >>> LLMProfile.from_provider('DeepSeek').base_url
        'https://api.deepseek.com'
    """

    id: str = ""
    name: str = ""
    built_in: bool = False
    base_url: str = ""
    api_key: Any = ""
    require_api_key: bool = True
    model: str = ""
    model_options: List[str] = field(default_factory=list)
    support_text: bool = True
    support_vision: bool = False
    vision_model: str = ""
    vision_model_options: List[str] = field(default_factory=list)
    vision_detail_level: str = "None"
    vision_detail_level_options: List[str] = field(default_factory=lambda: list(VISION_DETAIL_LEVEL_OPTIONS))
    support_image: bool = False
    image_base_url: str = ""
    image_model: str = ""
    image_model_options: List[str] = field(default_factory=list)
    thinking_level: str = "None"
    thinking_level_options: List[str] = field(default_factory=lambda: list(THINKING_LEVEL_OPTIONS))
    prompt: str = DEFAULT_TRANSLATION_PROMPT
    invalid_repeat_count: int = 2
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_schema_response_format: bool = False
    low_vram_mode: bool = False

    @classmethod
    def from_provider(cls, provider: str) -> "LLMProfile":
        provider = canonical_provider(provider)
        info = copy.deepcopy(PROVIDER_DEFAULTS[provider])
        return cls(**info, name=provider, built_in=True)

    def to_dict(self) -> Dict:
        return copy.deepcopy(self.__dict__)


def profile_from_config(profile: Any) -> LLMProfile:
    if isinstance(profile, LLMProfile):
        return copy.deepcopy(profile)
    if isinstance(profile, Mapping):
        return LLMProfile(**copy.deepcopy(dict(profile)))
    raise TypeError(f"Unsupported LLM profile config: {type(profile)!r}")


def profile_to_dict(profile: Any) -> Dict:
    return profile_from_config(profile).to_dict()


def default_profile(provider: str) -> LLMProfile:
    """Create a built-in profile for a provider.

    Example:
        >>> default_profile('Ollama').require_api_key
        False
    """

    return LLMProfile.from_provider(provider)


def default_profiles() -> List[LLMProfile]:
    return [default_profile(provider) for provider in PROVIDER_DEFAULTS]


def canonical_provider(provider: str) -> str:
    return PROVIDER_ALIASES.get(provider, provider)


def _provider_from_profile_id(profile_id: str) -> str:
    for provider, defaults in PROVIDER_DEFAULTS.items():
        if profile_id == defaults["id"]:
            return provider
    return ""


def _profile_value(profile: Any, attr: str, default=None) -> Any:
    if isinstance(profile, LLMProfile):
        return getattr(profile, attr)
    if isinstance(profile, Mapping):
        if attr in profile:
            return profile[attr]
    return default


def _builtin_profile_id(profile: Any) -> str:
    profile_id = str(_profile_value(profile, "id", "") or "")
    if _profile_value(profile, "built_in") and _provider_from_profile_id(profile_id):
        return profile_id
    return ""


def profile_by_id(profiles: List[Any], profile_id: str) -> Optional[Any]:
    for profile in profiles:
        if _profile_value(profile, "id") == profile_id:
            return profile
    return None


def load_profiles(profiles: List[Any]) -> List[LLMProfile]:
    loaded = []
    for profile in profiles or []:
        if not isinstance(profile, (Mapping, LLMProfile)):
            continue
        loaded.append(profile_from_config(profile))
    return loaded


def _dedupe_profile_entries(entries: List[Tuple[Any, bool]], selected_profile_id: str = "") -> List[LLMProfile]:
    deduped = []
    by_key = {}
    for raw_profile, selected_old_translator in entries:
        profile = profile_from_config(raw_profile)
        selected_profile = profile.id == selected_profile_id
        builtin_id = _builtin_profile_id(profile)
        key = ("builtin", builtin_id) if builtin_id else ("custom", profile.id)
        provider = _provider_from_profile_id(builtin_id)
        has_key = bool(profile.api_key)
        builtin_model = bool(provider and profile.model in PROVIDER_DEFAULTS[provider]["model_options"][:2])
        score = (bool(selected_old_translator), bool(selected_profile), has_key, builtin_model)
        existing_idx = by_key.get(key)
        if existing_idx is None:
            by_key[key] = len(deduped)
            deduped.append((profile, score))
        elif score > deduped[existing_idx][1]:
            deduped[existing_idx] = (profile, score)

    profiles = []
    for profile, _score in deduped:
        builtin_id = _builtin_profile_id(profile)
        provider = _provider_from_profile_id(builtin_id)
        if provider:
            defaults = PROVIDER_DEFAULTS[provider]
            profile.id = defaults["id"]
            profile.name = provider
            profile.built_in = True
        profiles.append(profile)
    return profiles


def restore_builtin_profiles(existing_profiles: List[Any]) -> List[LLMProfile]:
    """Replace all built-in profiles while keeping user profiles.

    Example:
        >>> custom = copy_profile(default_profile('OpenAI'))
        >>> custom.id = 'custom'
        >>> restore_builtin_profiles([default_profile('OpenAI'), custom])[0].id
        'custom'
    """

    existing = load_profiles(existing_profiles)
    user_profiles = [p for p in existing if not p.built_in]
    preserved_keys = {}
    for profile in existing:
        if not profile.built_in or not profile.api_key:
            continue
        builtin_id = _builtin_profile_id(profile)
        if builtin_id:
            preserved_keys[builtin_id] = copy.deepcopy(profile.api_key)

    builtins = default_profiles()
    for profile in builtins:
        api_key = preserved_keys.get(profile.id)
        if api_key:
            profile.api_key = api_key
    return user_profiles + builtins


def copy_profile(profile: Any) -> LLMProfile:
    copied = profile_from_config(profile)
    copied.id = _stable_profile_id(copied.id, copied.base_url, copied.model, copied.api_key, suffix="copy")
    copied.name = copied.name + " Copy"
    copied.built_in = False
    return copied


def resolve_api_key(profile: Any, secret_store: SecretStore = None) -> str:
    secret_store = secret_store or SecretStore()
    return secret_store.resolve(_profile_value(profile, "api_key", "")).value


def store_api_key(profile: LLMProfile, api_key: str, secret_store: SecretStore = None) -> None:
    secret_store = secret_store or SecretStore()
    profile.api_key = secret_store.store(profile.id, api_key or "")


def _stable_profile_id(provider: str, base_url: str, model: str, api_key: Any, suffix: str = "") -> str:
    seed = "|".join([provider or "", _normal_url(base_url or ""), model or "", str(api_key or ""), suffix])
    digest = hashlib.sha1(seed.encode("utf8")).hexdigest()[:10]
    provider_slug = (provider or "llm").lower().replace(" ", "-")
    return f"{provider_slug}-{digest}"


def _infer_provider(provider: str, base_url: str, model: str) -> str:
    provider = canonical_provider(provider)
    url = _normal_url(base_url).lower()
    model = (model or "").lower()
    if "api.deepseek.com" in url or model.startswith("deepseek"):
        return "DeepSeek"
    if "generativelanguage.googleapis.com" in url or model.startswith("gemini"):
        return "Gemini"
    if "api.x.ai" in url or model.startswith("grok"):
        return "Grok"
    if "openrouter.ai" in url:
        return "OpenRouter"
    if "localhost:1234" in url:
        return "LM Studio"
    if "localhost:11434" in url or "127.0.0.1:11434" in url:
        return "Ollama"
    if provider in PROVIDER_DEFAULTS:
        return provider
    return "OpenAI"


def normalize_model(provider: str, model: str) -> Tuple[str, str]:
    provider = canonical_provider(provider)
    model = _strip_model_prefix(model)
    thinking = "None"
    if provider == "OpenAI":
        aliases = {
            "gpt3": "gpt-5.5",
            "text-davinci-003": "gpt-5.5",
            "gpt35-turbo": "gpt-5.5",
            "gpt-3.5-turbo": "gpt-5.5",
            "gpt4": "gpt-5.5",
            "gpt-4": "gpt-5.5",
        }
        model = aliases.get(model, model)
    elif provider == "DeepSeek":
        if model == "deepseek-reasoner":
            model = "deepseek-v4-flash"
            thinking = "high"
        elif model == "deepseek-chat":
            model = "deepseek-v4-flash"
            thinking = "None"
    if not model:
        model = PROVIDER_DEFAULTS[provider]["model"]
    return model, thinking


def _old_base_url(old_key: str, params: Dict, provider: str) -> str:
    if old_key == "LLM_API_Translator":
        base_url = params.get("endpoint") or ""
    else:
        base_url = params.get("3rd party api url") or ""
    if base_url:
        return str(base_url).strip()
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["OpenAI"])["base_url"]


def _old_api_key(old_key: str, params: Dict) -> str:
    if old_key == "LLM_API_Translator":
        return str(params.get("apikey") or "").strip()
    return str(params.get("api key") or "").strip()


def profile_from_old_settings(old_key: str, params: Dict, secret_store: SecretStore = None) -> Optional[LLMProfile]:
    """Convert one old LLM translator config to a new profile.

    Example:
        >>> profile_from_old_settings('LLM_API_Translator', {}, secret_store=SecretStore()) is None
        True
    """

    params = params or {}
    override_model = _strip_model_prefix(str(params.get("override model") or ""))
    raw_model = _strip_model_prefix(str(override_model or params.get("model") or ""))
    using_override = bool(override_model)
    raw_provider = str(params.get("provider") or "")
    provider = _infer_provider(raw_provider, _old_base_url(old_key, params, "OpenAI"), raw_model)
    base_url = _old_base_url(old_key, params, provider)
    model, thinking = normalize_model(provider, raw_model)
    if provider in {"OpenAI", "DeepSeek"} and not using_override:
        if model not in PROVIDER_DEFAULTS[provider]["model_options"]:
            model = PROVIDER_DEFAULTS[provider]["model"]
            thinking = "None"
    api_key = _old_api_key(old_key, params)
    require_key = PROVIDER_DEFAULTS[provider]["require_api_key"]
    if require_key and not api_key:
        return None

    profile = default_profile(provider)
    matched_builtin = (
        _normal_url(base_url) == _normal_url(PROVIDER_DEFAULTS[provider]["base_url"])
        and model in PROVIDER_DEFAULTS[provider]["model_options"]
    )
    profile.id = PROVIDER_DEFAULTS[provider]["id"] if matched_builtin else _stable_profile_id(provider, base_url, model, api_key)
    profile.name = provider if matched_builtin else f"{provider} {model}"
    profile.built_in = matched_builtin
    profile.base_url = base_url
    profile.require_api_key = require_key
    profile.model = model
    profile.thinking_level = thinking
    if "invalid repeat count" in params:
        profile.invalid_repeat_count = params["invalid repeat count"]
    if "max tokens" in params:
        profile.max_tokens = params["max tokens"]
    if "temperature" in params:
        profile.temperature = params["temperature"]
    if "top p" in params:
        profile.top_p = params["top p"]
    if "frequency penalty" in params:
        profile.frequency_penalty = params["frequency penalty"]
    if "presence penalty" in params:
        profile.presence_penalty = params["presence penalty"]
    if "low vram mode" in params:
        profile.low_vram_mode = params["low vram mode"]
    if api_key:
        store_api_key(profile, api_key, secret_store=secret_store)
    if profile.model not in profile.model_options:
        profile.model_options.insert(0, profile.model)
    return profile


def migrate_module_llm_profiles(module_cfg: Dict, secret_store: SecretStore = None) -> Dict:
    """Migrate old LLM translator settings in a raw module config dict.

    Example:
        >>> migrate_module_llm_profiles({}, secret_store=SecretStore()) == {}
        True
    """

    if not isinstance(module_cfg, dict):
        return module_cfg
    secret_store = secret_store or SecretStore()
    raw_profiles = module_cfg.get("llm_profiles") or []
    profiles_were_missing = "llm_profiles" not in module_cfg
    if isinstance(raw_profiles, list):
        profiles = load_profiles(raw_profiles) if raw_profiles else default_profiles()
    else:
        profiles = default_profiles()

    trans_params = module_cfg.get("translator_params")
    if not isinstance(trans_params, dict):
        trans_params = {}
        module_cfg["translator_params"] = trans_params
    current_translator = module_cfg.get("translator")
    migrated_entries = []
    selected_profile = None
    for old_key in OLD_LLM_TRANSLATORS:
        if old_key not in trans_params:
            continue
        old_params = trans_params.get(old_key) or {}
        profile = profile_from_old_settings(
            old_key,
            old_params,
            secret_store=secret_store,
        )
        trans_params.pop(old_key, None)
        if profile is None:
            continue
        migrated_entries.append((profile, bool(current_translator == old_key)))
        if current_translator == old_key:
            selected_profile = profile.id

    if migrated_entries:
        profiles = _dedupe_profile_entries(
            migrated_entries + [(profile, False) for profile in profiles],
            selected_profile or module_cfg.get("translator_llm_id", ""),
        )
        if selected_profile:
            module_cfg["translator"] = LLM_TRANSLATOR_KEY
            module_cfg["translator_llm_id"] = selected_profile
    elif profiles_were_missing:
        profiles = default_profiles()

    module_cfg["llm_profiles"] = profiles
    return module_cfg
