from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from ballontranslator.utils.secret_store import SecretStore


LLM_TRANSLATOR_KEY = "LLMTranslator"
OLD_LLM_TRANSLATORS = ("ChatGPT", "ChatGPT_exp", "LLM_API_Translator")
LLM_TRANSLATOR_RUNTIME_PARAM_DEFAULTS = {
    "max requests per minute": 20,
    "delay": 0.3,
    "retry attempts": 5,
    "retry timeout": 15,
    "proxy": "",
}
LLM_TRANSLATOR_RUNTIME_PARAM_KEYS = tuple(LLM_TRANSLATOR_RUNTIME_PARAM_DEFAULTS)

THINKING_LEVEL_OPTIONS = ["None", "minimal", "low", "medium", "high", "xhigh"]

PROVIDER_DEFAULTS = {
    "OpenAI": {
        "id": "openai",
        "base url": "https://api.openai.com/v1",
        "require api key": True,
        "model": "gpt-5.5",
        "model options": ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4", "gpt-5.4-mini", "gpt-4o", "gpt-4o-mini"],
    },
    "DeepSeek": {
        "id": "deepseek",
        "base url": "https://api.deepseek.com",
        "require api key": True,
        "model": "deepseek-v4-flash",
        "model options": ["deepseek-v4-flash", "deepseek-v4-pro"],
    },
    "Google": {
        "id": "google",
        "base url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "require api key": True,
        "model": "gemini-3.5-flash",
        "model options": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
    },
    "Grok": {
        "id": "grok",
        "base url": "https://api.x.ai/v1",
        "require api key": True,
        "model": "grok-4",
        "model options": ["grok-4", "grok-3", "grok-3-mini"],
    },
    "OpenRouter": {
        "id": "openrouter",
        "base url": "https://openrouter.ai/api/v1",
        "require api key": True,
        "model": "openai/gpt-5.5",
        "model options": ["openai/gpt-5.5", "openai/gpt-5.4", "openai/gpt-4o", "anthropic/claude-sonnet-4"],
    },
    "LM Studio": {
        "id": "lmstudio",
        "base url": "http://localhost:1234/v1",
        "require api key": False,
        "model": "local-model",
        "model options": ["local-model"],
    },
    "Ollama": {
        "id": "ollama",
        "base url": "http://localhost:11434/v1/",
        "require api key": False,
        "model": "llama3.1",
        "model options": ["llama3.1", "qwen2.5", "mistral"],
    },
}

DEFAULT_JSON_SYSTEM_PROMPT = (
    "You are an expert translator. Your task is to accurately translate the given text snippets. "
    "You MUST provide the output strictly in the specified JSON format, without any additional "
    "explanations or markdown formatting. The JSON object must have a single key 'translations', "
    "which is a list of objects, each with an 'id' (integer) and a 'translation' (string).\n\n"
    "Example Output Schema:\n"
    '{"translations": [{"id": 1, "translation": "Translated text here."}]}'
)
DEFAULT_LEGACY_PROMPT_TEMPLATE = (
    "Please help me to translate the following text from a manga to {to_lang} "
    "(if it's already in {to_lang} or looks like gibberish you have to output it as it is instead):\n"
)
DEFAULT_CHAT_SAMPLE = """日本語-简体中文:
    source:
        - 二人のちゅーを 目撃した ぼっちちゃん
        - ふたりさん
        - 大好きなお友達には あいさつ代わりに ちゅーするんだって
        - アイス あげた
        - 喜多ちゃんとは どどど どういった ご関係なのでしようか...
        - テレビで見た！
    target:
        - 小孤独目击了两人的接吻
        - 二里酱
        - 我听说人们会把亲吻作为与喜爱的朋友打招呼的方式
        - 我给了她冰激凌
        - 喜多酱和你是怎么样的关系啊...
        - 我在电视上看到的！"""


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


def default_profile(provider: str) -> Dict:
    """Create a built-in profile for a provider.

    Example:
        >>> default_profile('Ollama')['require api key']
        False
    """

    info = PROVIDER_DEFAULTS[provider]
    return {
        "id": info["id"],
        "name": provider,
        "provider": provider,
        "built_in": True,
        "base url": info["base url"],
        "api key": "",
        "require api key": info["require api key"],
        "model": info["model"],
        "model options": list(info["model options"]),
        "thinking level": "None",
        "thinking level options": list(THINKING_LEVEL_OPTIONS),
        "system prompt": DEFAULT_JSON_SYSTEM_PROMPT,
        "chat sample": DEFAULT_CHAT_SAMPLE,
        "invalid repeat count": 2,
        "max tokens": 4096,
        "temperature": 0.1,
        "top p": 1.0,
        "frequency penalty": 0.0,
        "presence penalty": 0.0,
        "low vram mode": False,
    }


def default_profiles() -> List[Dict]:
    return [default_profile(provider) for provider in PROVIDER_DEFAULTS]


def _builtin_provider_for_profile(profile: Dict) -> Optional[str]:
    if not profile.get("built_in"):
        return None
    provider = profile.get("provider")
    if provider not in PROVIDER_DEFAULTS:
        return None
    defaults = PROVIDER_DEFAULTS[provider]
    if _normal_url(profile.get("base url", "")) != _normal_url(defaults["base url"]):
        return None
    if _strip_model_prefix(str(profile.get("model") or "")) not in defaults["model options"]:
        return None
    return provider


def _profile_dedupe_key(profile: Dict) -> tuple:
    provider = _builtin_provider_for_profile(profile)
    if provider is not None:
        return ("builtin", provider)
    return ("custom", profile.get("id"))


def profile_by_id(profiles: List[Dict], profile_id: str) -> Optional[Dict]:
    for profile in profiles:
        if profile.get("id") == profile_id:
            return profile
    return None


def ensure_profile_defaults(profile: Dict) -> Dict:
    raw_model = _strip_model_prefix(str((profile or {}).get("model") or ""))
    raw_base_url = str((profile or {}).get("base url") or "")
    declared_provider = (profile or {}).get("provider") if (profile or {}).get("provider") in PROVIDER_DEFAULTS else ""
    provider = infer_provider(str(declared_provider or ""), raw_base_url, raw_model)
    base = default_profile(provider)
    merged = copy.deepcopy(base)
    merged.update(profile or {})
    provider = infer_provider("", merged.get("base url") or base["base url"], merged.get("model") or base["model"])
    if provider != merged.get("provider"):
        base = default_profile(provider)
        updated = copy.deepcopy(base)
        updated.update(merged)
        merged = updated
    merged["provider"] = provider
    merged["id"] = str(merged.get("id") or _stable_profile_id(provider, merged.get("base url"), merged.get("model"), merged.get("api key")))
    merged["name"] = str(merged.get("name") or provider)
    merged["base url"] = str(merged.get("base url") or base["base url"])
    merged["api key"] = copy.deepcopy(merged.get("api key") or "")
    merged["model"] = _strip_model_prefix(str(merged.get("model") or base["model"])) or base["model"]
    merged["model options"] = [str(item) for item in merged.get("model options", []) if str(item)]
    if merged["model"] not in merged["model options"]:
        merged["model options"].insert(0, merged["model"])
    merged["thinking level"] = str(merged.get("thinking level") or "None")
    if merged["thinking level"].lower() == "none":
        merged["thinking level"] = "None"
    if merged["thinking level"] not in THINKING_LEVEL_OPTIONS:
        merged["thinking level"] = "None"
    merged["thinking level options"] = list(THINKING_LEVEL_OPTIONS)
    old_prompt_mode = merged.get("prompt mode")
    if old_prompt_mode in {"Legacy delimiter", "XML"}:
        merged["system prompt"] = DEFAULT_JSON_SYSTEM_PROMPT
    merged.pop("prompt mode", None)
    merged.pop("prompt mode options", None)
    merged.pop("prompt template", None)
    merged.pop("chat system template", None)
    legacy_names = {f"{provider} {prompt_mode}" for prompt_mode in ("JSON", "Legacy delimiter", "XML")}
    if _builtin_provider_for_profile(merged) is not None and merged.get("name") in legacy_names:
        merged["name"] = provider
    for key in LLM_TRANSLATOR_RUNTIME_PARAM_KEYS:
        merged.pop(key, None)
    for key in ("invalid repeat count", "max tokens"):
        try:
            merged[key] = int(merged[key])
        except Exception:
            merged[key] = int(base[key])
    for key in ("temperature", "top p", "frequency penalty", "presence penalty"):
        try:
            merged[key] = float(merged[key])
        except Exception:
            merged[key] = float(base[key])
    merged["require api key"] = bool(merged.get("require api key"))
    merged["low vram mode"] = bool(merged.get("low vram mode"))
    return merged


def normalize_profiles(profiles: List[Dict]) -> List[Dict]:
    seen = set()
    normalized = []
    for profile in profiles or []:
        if not isinstance(profile, dict):
            continue
        item = ensure_profile_defaults(profile)
        profile_id = item["id"]
        if profile_id in seen:
            item["id"] = _stable_profile_id(item["provider"], item["base url"], item["model"], item["api key"])
        seen.add(item["id"])
        normalized.append(item)
    return normalized


def dedupe_profiles(profiles: List[Dict], selected_profile_id: str = "") -> List[Dict]:
    """Collapse duplicate provider profiles while preserving the best candidate.

    Example:
        >>> profiles = [default_profile('DeepSeek'), default_profile('DeepSeek')]
        >>> len(dedupe_profiles(profiles))
        1
    """

    deduped = []
    by_key = {}
    for profile in normalize_profiles(profiles):
        profile["__selected_profile"] = profile.get("id") == selected_profile_id
        key = _profile_dedupe_key(profile)
        existing_idx = by_key.get(key)
        if existing_idx is None:
            by_key[key] = len(deduped)
            deduped.append(profile)
        elif _profile_score(profile) > _profile_score(deduped[existing_idx]):
            deduped[existing_idx] = profile

    for profile in deduped:
        provider = _builtin_provider_for_profile(profile)
        if provider is not None:
            defaults = PROVIDER_DEFAULTS[provider]
            profile["id"] = defaults["id"]
            profile["name"] = provider
            profile["built_in"] = True
        profile.pop("__selected_profile", None)
    return normalize_profiles(deduped)


def restore_builtin_profiles(existing_profiles: List[Dict]) -> List[Dict]:
    """Replace all built-in profiles while keeping user profiles.

    Example:
        >>> restore_builtin_profiles([{'id': 'openai', 'built_in': True}, {'id': 'custom', 'provider': 'OpenAI', 'built_in': False, 'base url': 'https://example.test/v1'}])[0]['id']
        'custom'
    """

    existing = normalize_profiles(existing_profiles)
    user_profiles = [p for p in existing if not p.get("built_in")]
    preserved_keys = {}
    for profile in existing:
        if not profile.get("built_in") or not profile.get("api key"):
            continue
        provider = _builtin_provider_for_profile(profile) or profile.get("provider")
        if provider in PROVIDER_DEFAULTS:
            preserved_keys[provider] = copy.deepcopy(profile.get("api key"))

    builtins = default_profiles()
    for profile in builtins:
        api_key = preserved_keys.get(profile.get("provider"))
        if api_key:
            profile["api key"] = api_key
    return user_profiles + builtins


def copy_profile(profile: Dict) -> Dict:
    copied = ensure_profile_defaults(profile)
    copied["id"] = _stable_profile_id(copied["provider"], copied["base url"], copied["model"], copied["api key"], suffix="copy")
    copied["name"] = copied["name"] + " Copy"
    copied["built_in"] = False
    return copied


def resolve_api_key(profile: Dict, secret_store: SecretStore = None) -> str:
    secret_store = secret_store or SecretStore()
    return secret_store.resolve((profile or {}).get("api key", "")).value


def store_api_key(profile: Dict, api_key: str, secret_store: SecretStore = None) -> None:
    secret_store = secret_store or SecretStore()
    profile["api key"] = secret_store.store(profile.get("id", ""), api_key or "")


def _plain_param_value(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _extract_llm_runtime_params(params: Dict) -> Dict:
    if not isinstance(params, dict):
        return {}
    return {
        key: _plain_param_value(params[key])
        for key in LLM_TRANSLATOR_RUNTIME_PARAM_KEYS
        if key in params
    }


def _merge_llm_runtime_params(trans_params: Dict, values: Dict, overwrite: bool = False) -> None:
    if not values:
        return
    llm_params = trans_params.setdefault(LLM_TRANSLATOR_KEY, {})
    if not isinstance(llm_params, dict):
        llm_params = {}
        trans_params[LLM_TRANSLATOR_KEY] = llm_params
    for key, value in values.items():
        if overwrite or key not in llm_params:
            llm_params[key] = value


def _stable_profile_id(provider: str, base_url: str, model: str, api_key: Any, suffix: str = "") -> str:
    seed = "|".join([provider or "", _normal_url(base_url or ""), model or "", str(api_key or ""), suffix])
    digest = hashlib.sha1(seed.encode("utf8")).hexdigest()[:10]
    provider_slug = (provider or "llm").lower().replace(" ", "-")
    return f"{provider_slug}-{digest}"


def infer_provider(provider: str, base_url: str, model: str) -> str:
    url = _normal_url(base_url).lower()
    model = (model or "").lower()
    if "api.deepseek.com" in url or model.startswith("deepseek"):
        return "DeepSeek"
    if "generativelanguage.googleapis.com" in url or model.startswith("gemini"):
        return "Google"
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
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["OpenAI"])["base url"]


def _old_api_key(old_key: str, params: Dict) -> str:
    if old_key == "LLM_API_Translator":
        return str(params.get("apikey") or "").strip()
    return str(params.get("api key") or "").strip()


def profile_from_old_settings(old_key: str, params: Dict, selected: bool = False, secret_store: SecretStore = None) -> Optional[Dict]:
    """Convert one old LLM translator config to a new profile.

    Example:
        >>> p = profile_from_old_settings('LLM_API_Translator', {'endpoint': 'https://api.deepseek.com', 'apikey': 'k', 'override model': 'deepseek-v4-flash'}, secret_store=SecretStore(False))
        >>> p['provider'], p['model'], p['system prompt'] == DEFAULT_JSON_SYSTEM_PROMPT
        ('DeepSeek', 'deepseek-v4-flash', True)
    """

    params = params or {}
    declared_provider = params.get("provider") if old_key == "LLM_API_Translator" else "OpenAI"
    override_model = _strip_model_prefix(str(params.get("override model") or ""))
    raw_model = _strip_model_prefix(str(override_model or params.get("model") or ""))
    using_override = bool(override_model)
    provider = infer_provider(str(declared_provider or ""), _old_base_url(old_key, params, str(declared_provider or "OpenAI")), raw_model)
    base_url = _old_base_url(old_key, params, provider)
    model, thinking = normalize_model(provider, raw_model)
    if provider in {"OpenAI", "DeepSeek"} and not using_override:
        if model not in PROVIDER_DEFAULTS[provider]["model options"]:
            model = PROVIDER_DEFAULTS[provider]["model"]
            thinking = "None"
    api_key = _old_api_key(old_key, params)
    require_key = PROVIDER_DEFAULTS[provider]["require api key"]
    if require_key and not api_key:
        return None

    system_prompt = str(params.get("system_prompt") or DEFAULT_JSON_SYSTEM_PROMPT)

    profile = default_profile(provider)
    matched_builtin = (
        _normal_url(base_url) == _normal_url(PROVIDER_DEFAULTS[provider]["base url"])
        and model in PROVIDER_DEFAULTS[provider]["model options"]
    )
    profile.update({
        "id": PROVIDER_DEFAULTS[provider]["id"] if matched_builtin else _stable_profile_id(provider, base_url, model, api_key),
        "name": provider if matched_builtin else f"{provider} {model}",
        "built_in": matched_builtin,
        "base url": base_url,
        "require api key": require_key,
        "model": model,
        "system prompt": system_prompt,
        "chat sample": str(params.get("chat sample") or DEFAULT_CHAT_SAMPLE),
        "thinking level": thinking,
        "__migrated_from": old_key,
        "__selected_old_translator": bool(selected),
    })
    for key in ("invalid repeat count", "max tokens", "temperature", "top p", "frequency penalty", "presence penalty", "low vram mode"):
        if key in params:
            profile[key] = params[key]
    if api_key:
        store_api_key(profile, api_key, secret_store=secret_store)
    return ensure_profile_defaults(profile)


def _profile_score(profile: Dict) -> tuple:
    has_key = bool(profile.get("api key"))
    builtin_model = profile.get("model") in PROVIDER_DEFAULTS[profile["provider"]]["model options"][:2]
    return (bool(profile.get("__selected_old_translator") or profile.get("__selected_profile")), has_key, builtin_model)


def _dedupe_profiles(profiles: List[Dict]) -> List[Dict]:
    return dedupe_profiles(profiles)


def migrate_module_llm_profiles(module_cfg: Dict, secret_store: SecretStore = None) -> Dict:
    """Migrate old LLM translator settings in a raw module config dict.

    Example:
        >>> cfg = {'translator': 'LLM_API_Translator', 'translator_params': {'LLM_API_Translator': {'endpoint': 'https://api.deepseek.com', 'apikey': 'k', 'override model': 'deepseek-v4-flash'}}}
        >>> migrate_module_llm_profiles(cfg, secret_store=SecretStore(False))['translator']
        'LLMTranslator'
    """

    if not isinstance(module_cfg, dict):
        return module_cfg
    secret_store = secret_store or SecretStore()
    raw_profiles = module_cfg.get("llm_profiles") or []
    profiles_were_missing = "llm_profiles" not in module_cfg
    selected_runtime_params = {}
    if isinstance(raw_profiles, list):
        selected_profile_id = module_cfg.get("llm_profile", "")
        for raw_profile in raw_profiles:
            if isinstance(raw_profile, dict) and raw_profile.get("id") == selected_profile_id:
                selected_runtime_params = _extract_llm_runtime_params(raw_profile)
                break
        if not selected_runtime_params:
            for raw_profile in raw_profiles:
                selected_runtime_params = _extract_llm_runtime_params(raw_profile)
                if selected_runtime_params:
                    break
    old_profiles = normalize_profiles(raw_profiles)
    profiles = old_profiles if old_profiles else default_profiles()

    trans_params = module_cfg.get("translator_params")
    if not isinstance(trans_params, dict):
        trans_params = {}
        module_cfg["translator_params"] = trans_params
    current_translator = module_cfg.get("translator")
    migrated = []
    selected_profile = None
    for old_key in OLD_LLM_TRANSLATORS:
        if old_key not in trans_params:
            continue
        old_params = trans_params.get(old_key) or {}
        runtime_params = _extract_llm_runtime_params(old_params)
        if current_translator == old_key and runtime_params:
            selected_runtime_params = runtime_params
        elif not selected_runtime_params and runtime_params:
            selected_runtime_params = runtime_params
        profile = profile_from_old_settings(
            old_key,
            old_params,
            selected=current_translator == old_key,
            secret_store=secret_store,
        )
        trans_params.pop(old_key, None)
        if profile is None:
            continue
        migrated.append(profile)
        if current_translator == old_key:
            selected_profile = profile["id"]

    if migrated:
        profiles = _dedupe_profiles(migrated + profiles)
        if selected_profile:
            module_cfg["translator"] = LLM_TRANSLATOR_KEY
            module_cfg["llm_profile"] = selected_profile
    elif profiles_were_missing:
        profiles = default_profiles()

    profiles = dedupe_profiles(profiles, module_cfg.get("llm_profile", ""))
    if not profile_by_id(profiles, module_cfg.get("llm_profile", "")):
        module_cfg["llm_profile"] = profiles[0]["id"] if profiles else ""
    _merge_llm_runtime_params(trans_params, selected_runtime_params)
    module_cfg["llm_profiles"] = profiles
    return module_cfg
