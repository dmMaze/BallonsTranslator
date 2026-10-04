from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
    get_type_hints,
)
from urllib.parse import urlsplit

from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.secret_store import SecretStore
from ballontranslator.utils.structures import Config, field, nested_dataclass


LLM_TRANSLATOR_KEY = "LLMTranslator"
LLM_OCR_KEY = "LLMOCR"
LLM_INPAINT_KEY = "LLMInpaint"
CODEX_MODEL_OPTIONS = (
    "gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
)

THINKING_AUTO = "Auto"
THINKING_DISABLED = "Disabled"
THINKING_LEVEL_OPTIONS = [
    THINKING_AUTO,
    THINKING_DISABLED,
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]
VISION_DETAIL_LEVEL_OPTIONS = ["None", "auto", "low", "high"]
PROVIDER_DEFAULTS = {
    "DeepSeek": {
        "id": "deepseek",
        "base_url": "https://api.deepseek.com",
        "require_api_key": True,
        "model": "deepseek-flash",
        "model_options": ["deepseek-flash", "deepseek-v4-pro"],
        "support_vision": True,
        "vision_model": "deepseek-flash",
        "vision_model_options": [
            "deepseek-flash",
        ],
        "vision_detail_level": "auto",
    },
    "OpenAI": {
        "id": "openai",
        "base_url": "https://api.openai.com/v1",
        "require_api_key": True,
        "model": "gpt-5.6-luna",
        "support_vision": True,
        "vision_model": "gpt-5.6-luna",
        "vision_detail_level": "auto",
        "model_options": [
            "gpt-6-astra", "gpt-6.1-sol",
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        "vision_model_options": [
            "gpt-6-astra", "gpt-6.1-sol",
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
            "gpt-5.5", "gpt-5.4", "gpt-5.4-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini",
        ],
        "support_image": True,
        "image_base_url": "https://api.openai.com/v1/images/edits",
        "image_model_options": ["gpt-image-2"],
        "image_model": "gpt-image-2",
    },
    "Gemini": {
        "id": "google",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "require_api_key": True,
        "model": "gemini-3.1-flash-lite",
        "thinking_level": "medium",
        "support_vision": True,
        "vision_model": "gemini-3.1-flash-lite",
        "vision_detail_level": "auto",
        "model_options": ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-pro-latest", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
        "vision_model_options": ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-pro-latest"],
        "support_image": True,
        "image_model_options": ["gemini-3.1-flash-lite-image"],
        "image_base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "json_schema_response_format": True,
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
        "image_base_url": "https://openrouter.ai/api/v1/images",
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

DEFAULT_OCR_PROMPT = (
    "Extract every visible text string from this image. "
    "The text may be vertical manga/comic text or left-to-right text; infer the intended reading order from the image. If "
    "characters look jumbled because vertical text was read horizontally, reconstruct the intended "
    "vertical order. Return only the recognized text, using spaces instead of line breaks when possible. "
    "If no text is visible, return an empty response."
)

DEFAULT_INPAINT_PROMPT = (
    "Clean up this comic or manga image for further scanlation. Remove all visible text elements, "
    "including speech bubble lettering, captions, sound effects, signs, labels, and text-like "
    "watermarks. Keep all non-text artwork intact: characters, faces, line art, screentones, "
    "backgrounds, speech bubbles, panel borders, lighting, colors, texture, and composition. "
    "Do not translate, redraw with new text, add captions, or explain the edit. Return only the "
    "cleaned image."
)

@nested_dataclass
class LLMProfile(Config):
    """Typed persistent LLM profile config.

    Example:
        >>> LLMProfile.from_provider('DeepSeek').base_url
        'https://api.deepseek.com'
    """

    id: str = ""
    profile_type = "llm"
    name: str = ""
    backend: str = "openai"
    title_url: str = ""
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
    thinking_level: str = THINKING_AUTO
    thinking_level_options: List[str] = field(default_factory=lambda: list(THINKING_LEVEL_OPTIONS))
    prompt: str = DEFAULT_TRANSLATION_PROMPT
    vision_prompt: str = DEFAULT_OCR_PROMPT
    image_prompt: str = DEFAULT_INPAINT_PROMPT
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_schema_response_format: bool = False
    low_vram_mode: bool = False

    def __post_init__(self) -> None:
        if self.backend not in ('openai', 'codex', 'unavailable'):
            LOGGER.warning('Discard invalid LLM profile backend for %s.', self.id or self.name)
            # An unknown backend must never fall through to a paid API request.
            self.backend = 'unavailable'
        for key in ('vision_model_options', 'image_model_options'):
            raw_options = getattr(self, key)
            options = list(dict.fromkeys(
                option.strip() for option in raw_options
                if isinstance(option, str) and option.strip() and '→' not in option and '->' not in option
            )) if isinstance(raw_options, list) else []
            if options != raw_options:
                LOGGER.warning('Discard invalid %s entries for %s.', key, self.id or self.name)
            setattr(self, key, options)
        try:
            reasoning_model, image_model = split_image_model_selection(self.image_model)
            self.image_model = f'{reasoning_model} → {image_model}' if reasoning_model else image_model
        except ValueError:
            LOGGER.warning('Discard invalid image model selection for %s.', self.id or self.name)
            self.image_model = image_model = ''
        if self.backend == 'codex':
            self.api_key = ''
            self.require_api_key = False
            self.support_text = self.support_vision = self.support_image = True
            self.image_model_options = _merge_profile_options(None, self.image_model_options, image_model)
            self.json_schema_response_format = True
        if not isinstance(self.title_url, str) or (
            self.title_url and not is_profile_title_url(self.title_url)
        ):
            LOGGER.warning('Discard invalid LLM profile title_url for %s.', self.id or self.name)
            self.title_url = ''

    @classmethod
    def from_provider(cls, provider: str) -> "LLMProfile":
        info = copy.deepcopy(PROVIDER_DEFAULTS[provider])
        return cls(**info, name=provider, built_in=True)

    def to_dict(self) -> Dict:
        data = copy.deepcopy(self.__dict__)
        if self.backend == 'codex':
            data['api_key'] = ''
            # Text/vision options come from the catalog; image choices are user-owned.
            for key in ('model_options', 'vision_model_options',
                        'thinking_level_options', 'support_text', 'support_vision', 'support_image'):
                data.pop(key, None)
        return data


def split_image_model_selection(value: str) -> Tuple[str, str]:
    """Resolve a direct image ID or an explicit GPT image-tool selection.

    >>> split_image_model_selection('gpt-6-sol → gpt-image-2')
    ('gpt-6-sol', 'gpt-image-2')
    >>> split_image_model_selection('image-model')
    ('', 'image-model')
    """
    if not isinstance(value, str):
        raise ValueError('The image model selection must be text.')
    reasoning, separator, image = value.replace('->', '→').partition('→')
    if not separator:
        return '', reasoning.strip()
    reasoning, image = reasoning.strip(), image.strip()
    if (not reasoning.startswith('gpt-') or reasoning.startswith('gpt-image-')
            or not image.startswith('gpt-image-') or '→' in image
            or any(character.isspace() for character in reasoning + image)):
        raise ValueError('Image assistance requires a GPT model and a GPT Image model.')
    return reasoning, image


def image_responses_url(profile: LLMProfile) -> str:
    """Resolve Responses from a base or endpoint on the configured image service.

    >>> image_responses_url(default_profile('OpenAI'))
    'https://api.openai.com/v1/responses'
    """
    if profile.backend != 'openai':
        return ''
    try:
        url = urlsplit(profile.image_base_url.strip())
        host = url.hostname or ''
    except (AttributeError, ValueError):
        return ''
    if (url.scheme not in ('http', 'https') or not host
            or host == 'generativelanguage.googleapis.com'
            or host == 'openrouter.ai' or host.endswith('.openrouter.ai')):
        return ''
    path = url.path.rstrip('/')
    for suffix in ('/images/edits', '/images/generations', '/responses'):
        if path.endswith(suffix):
            return url._replace(path=path[:-len(suffix)] + '/responses', fragment='').geturl()
    if not path or path.endswith('/v1'):
        return url._replace(path=path + '/responses', fragment='').geturl()
    return ''


def image_model_choices(profile: LLMProfile) -> List[str]:
    """Derive image choices on demand; saved options contain only image IDs.

    >>> profile = LLMProfile(backend='codex', image_model_options=['gpt-image-2'],
    ...                      vision_model_options=['gpt-6-sol', 'other-model'])
    >>> image_model_choices(profile)
    ['gpt-image-2', 'gpt-6-sol → gpt-image-2']
    """
    choices = list(profile.image_model_options)
    if profile.backend != 'codex' and not image_responses_url(profile):
        return choices
    reasoning_models = [model for model in profile.vision_model_options
                        if model.startswith('gpt-') and not model.startswith('gpt-image-')
                        and not any(character.isspace() for character in model)
                        and '→' not in model and '->' not in model]
    image_models = [model for model in choices if model.startswith('gpt-image-')
                    and not any(character.isspace() for character in model)
                    and '→' not in model and '->' not in model]
    return choices + [f'{reasoning} → {image}' for reasoning in reasoning_models for image in image_models]


def is_profile_title_url(value: str) -> bool:
    try:
        url = urlsplit(value)
        return (
            url.scheme in ('http', 'https') and bool(url.hostname)
            and not any(character.isspace() for character in value)
        )
    except ValueError:
        return False


def parse_profile_title(text: str) -> Tuple[str, str]:
    """Accept a plain title or one Markdown-style web link.

    >>> parse_profile_title('[Example](https://example.com)')
    ('Example', 'https://example.com')
    >>> parse_profile_title('[Example](file:///tmp/example)')
    ('[Example](file:///tmp/example)', '')
    """
    text = text.strip()
    match = re.fullmatch(r'\[(.+)\]\((\S+)\)', text)
    if match and match[1].strip() and is_profile_title_url(match[2]):
        return match[1].strip(), match[2]
    return text, ''


def normalize_thinking_level(value: Any) -> str:
    """Return the canonical persisted reasoning-control value.

    >>> normalize_thinking_level('None')
    'Auto'
    >>> normalize_thinking_level('Disabled')
    'Disabled'
    """
    if not isinstance(value, str):
        return THINKING_AUTO
    level = value.strip()
    if level.lower() in {'', 'none', 'auto'}:
        return THINKING_AUTO
    if level.lower() == 'disabled':
        return THINKING_DISABLED
    return level


def _normalize_profile_thinking(profile: LLMProfile) -> LLMProfile:
    """Migrate legacy reasoning settings without dropping custom options.

    >>> profile = LLMProfile(
    ...     thinking_level='None',
    ...     thinking_level_options=['None', 'high'],
    ... )
    >>> migrated = _normalize_profile_thinking(profile)
    >>> migrated.thinking_level, migrated.thinking_level_options
    ('Auto', ['Auto', 'Disabled', 'high'])
    """
    raw_level = profile.thinking_level
    if not isinstance(raw_level, str):
        LOGGER.warning(
            'Discard invalid LLM profile thinking_level for %s.',
            profile.id or profile.name or '<unnamed>',
        )
    profile.thinking_level = normalize_thinking_level(raw_level)

    raw_options = profile.thinking_level_options
    if not isinstance(raw_options, list):
        LOGGER.warning(
            'Discard invalid LLM profile thinking_level_options for %s.',
            profile.id or profile.name or '<unnamed>',
        )
        raw_options = THINKING_LEVEL_OPTIONS
    elif any(not isinstance(option, str) for option in raw_options):
        LOGGER.warning(
            'Discard invalid entries from LLM profile '
            'thinking_level_options for %s.',
            profile.id or profile.name or '<unnamed>',
        )

    normalized_options = []
    for option in [THINKING_AUTO, THINKING_DISABLED, *raw_options]:
        if not isinstance(option, str):
            continue
        option = normalize_thinking_level(option)
        if option not in normalized_options:
            normalized_options.append(option)
    if profile.thinking_level not in normalized_options:
        normalized_options.append(profile.thinking_level)
    profile.thinking_level_options = normalized_options
    return profile


def profile_from_config(profile: Any) -> LLMProfile:
    if isinstance(profile, LLMProfile):
        loaded = copy.deepcopy(profile)
    elif isinstance(profile, Mapping):
        data = copy.deepcopy(dict(profile))
        # Only missing fields inherit a new built-in link; an empty URL is a
        # deliberate user choice and must survive subsequent loads.
        if 'title_url' not in data:
            provider = _provider_from_profile_id(_builtin_profile_id(profile))
            data['title_url'] = PROVIDER_DEFAULTS.get(provider, {}).get('title_url', '')
        loaded = LLMProfile(**data)
    else:
        raise TypeError(f"Unsupported LLM profile config: {type(profile)!r}")
    return _normalize_profile_thinking(loaded)


def profile_to_dict(profile: Any) -> Dict:
    return profile_from_config(profile).to_dict()


def profile_to_export_dict(profile: Any) -> Dict:
    """Return the JSON mapping used by profile clipboard export.

    Clipboard export resolves the API key to plaintext; normal config saving
    continues to apply portable obfuscation independently.

    Example:
        >>> profile_to_export_dict(LLMProfile())['profile_type']
        'llm'
    """

    exported = profile_to_dict(profile)
    if exported.get('backend') == 'codex':
        raise ValueError('Codex settings cannot be exported as an API profile.')
    exported['profile_type'] = LLMProfile.profile_type
    exported['api_key'] = resolve_api_key(profile)
    return exported


def _normalize_profile_data(data: Mapping) -> Dict[str, Any]:
    """Keep valid supplied values while omitted or invalid fields use defaults."""

    type_hints = get_type_hints(LLMProfile)
    normalized = {}
    for key, expected in type_hints.items():
        if key not in data:
            continue
        value = data[key]
        origin = get_origin(expected)
        args = get_args(expected)
        if origin is list:
            item_type = args[0] if args else Any
            if isinstance(value, list):
                valid = [item for item in value if item_type is Any or isinstance(item, item_type)]
                if valid or not value:
                    normalized[key] = valid
            continue
        if key == 'api_key':
            if isinstance(value, str):
                normalized[key] = value
            continue
        if expected is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                normalized[key] = float(value)
        elif expected is int:
            if isinstance(value, int) and not isinstance(value, bool):
                normalized[key] = value
        elif key == 'backend' or isinstance(value, expected):
            # Keep invalid backend values for __post_init__ to disable. Dropping
            # one here would silently select the default API transport instead.
            normalized[key] = value
    return normalized


def profiles_from_json(value: str) -> List[LLMProfile]:
    """Parse one or more valid LLM profiles from clipboard JSON.

    Example:
        >>> len(profiles_from_json(json.dumps(profile_to_export_dict(LLMProfile()))))
        1
    """

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    candidates = [decoded] if isinstance(decoded, Mapping) else decoded if isinstance(decoded, list) else []
    profiles = []
    for candidate in candidates:
        if (not isinstance(candidate, Mapping)
                or candidate.get('profile_type') != LLMProfile.profile_type
                or candidate.get('backend') == 'codex'):
            continue
        data = copy.deepcopy(dict(candidate))
        data.pop('profile_type', None)
        data = _normalize_profile_data(data)
        try:
            profiles.append(profile_from_config(data))
        except (TypeError, ValueError):
            continue
    return profiles


def default_profile(provider: str) -> LLMProfile:
    """Create a built-in profile for a provider.

    Example:
        >>> default_profile('Ollama').require_api_key
        False
    """

    return LLMProfile.from_provider(provider)


def default_profiles() -> List[LLMProfile]:
    return [default_profile(provider) for provider in PROVIDER_DEFAULTS]


def default_codex_profile() -> LLMProfile:
    """Create the dedicated Codex settings entry in the shared runtime format.

    >>> default_codex_profile().id
    'codex'
    """
    return LLMProfile(
        id='codex', name='Codex', backend='codex', built_in=True,
        model_options=list(CODEX_MODEL_OPTIONS),
        vision_model_options=list(CODEX_MODEL_OPTIONS),
        image_model='gpt-image-2', image_model_options=['gpt-image-2'],
        vision_detail_level='auto',
    )


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


def profile_by_id(
    profiles: Sequence[Any],
    profile_id: str,
) -> Optional[Any]:
    for profile in profiles:
        if _profile_value(profile, "id") == profile_id:
            return profile
    return None


def normalize_codex_models(value: Any) -> Dict[str, Dict[str, List[str]]]:
    """Keep only public, usable catalog metadata; never persist account data.

    >>> normalize_codex_models({'m': {'modalities': ['text'], 'efforts': ['high']}})
    {'m': {'modalities': ['text'], 'efforts': ['high']}}
    """
    if not isinstance(value, dict):
        LOGGER.warning('Discard invalid Codex model catalog.')
        return {}
    models = {}
    for model, entry in value.items():
        if not isinstance(model, str) or not model.strip() or not isinstance(entry, dict):
            LOGGER.warning('Discard invalid Codex model catalog entry.')
            continue
        if set(entry) - {'modalities', 'efforts'}:
            LOGGER.warning('Discard unknown fields from Codex model catalog entry.')
        clean = {}
        # Reasoning levels belong to the model catalog and can grow independently
        # of this app; supported input modalities are an application constraint.
        for key, allowed in (('modalities', ('text', 'image')), ('efforts', None)):
            items = entry.get(key, [])
            if not isinstance(items, list):
                LOGGER.warning('Discard invalid Codex model %s for %s.', key, model)
                items = []
            clean[key] = list(dict.fromkeys(
                item for item in items
                if isinstance(item, str) and item.strip() and (allowed is None or item in allowed)
            ))
            if clean[key] != items:
                LOGGER.warning('Discard invalid Codex model %s entries for %s.', key, model)
        if clean['modalities']:
            models[model] = clean
    return models


def codex_thinking_options(profile: LLMProfile, models: Dict) -> List[str]:
    return [THINKING_AUTO] + [
        THINKING_DISABLED if effort == 'none' else effort
        for effort in models.get(profile.model, {}).get('efforts', [])
    ]


def sync_codex_profile(profile: LLMProfile, models: Dict) -> None:
    """Refresh text/vision options without changing selections or saved image choices.

    >>> profile = default_codex_profile()
    >>> sync_codex_profile(profile, {'m': {'modalities': ['text'], 'efforts': ['high']}})
    >>> profile.model, profile.model_options
    ('', ['m'])
    """
    profile.api_key = ''
    profile.require_api_key = False
    if models:
        profile.model_options = [model for model, entry in models.items() if 'text' in entry['modalities']]
        profile.vision_model_options = [model for model, entry in models.items() if 'image' in entry['modalities']]
    else:
        defaults = list(CODEX_MODEL_OPTIONS)
        profile.model_options = _merge_profile_options(defaults, None, profile.model)
        profile.vision_model_options = _merge_profile_options(defaults, None, profile.vision_model)
    profile.thinking_level_options = codex_thinking_options(profile, models)


def runtime_profile(
    profiles: Sequence[Any],
    selected_profile_id: str,
) -> LLMProfile:
    """Copy the selected runtime profile, falling back to the first entry.

    >>> runtime_profile([LLMProfile(id='first')], 'missing').id
    'first'
    """
    selected = profile_by_id(profiles, selected_profile_id)
    if selected is None and profiles:
        selected = profiles[0]
    if selected is None:
        raise RuntimeError('No LLM profile is configured.')
    return profile_from_config(selected)


def _merge_profile_options(default_options: Any, saved_options: Any, selected: Any) -> List[str]:
    merged = []
    for option in (default_options if isinstance(default_options, list) else []):
        if isinstance(option, str) and option not in merged:
            merged.append(option)
    for option in (saved_options if isinstance(saved_options, list) else []):
        if isinstance(option, str) and option not in merged:
            merged.append(option)
    if isinstance(selected, str) and selected and selected not in merged:
        merged.append(selected)
    return merged


def _merge_builtin_profile_options(profile: LLMProfile) -> LLMProfile:
    provider = _provider_from_profile_id(_builtin_profile_id(profile))
    if not provider:
        if profile.built_in:
            # Removed presets remain user-owned profiles, including their keys.
            LOGGER.warning('Load unrecognized built-in LLM profile as a custom profile.')
            profile.built_in = False
        return profile
    defaults = PROVIDER_DEFAULTS[provider]
    profile.model_options = _merge_profile_options(
        defaults.get('model_options'), profile.model_options, profile.model,
    )
    profile.vision_model_options = _merge_profile_options(
        defaults.get('vision_model_options'), profile.vision_model_options, profile.vision_model,
    )
    profile.image_model_options = _merge_profile_options(
        defaults.get('image_model_options'), profile.image_model_options,
        split_image_model_selection(profile.image_model)[1],
    )
    if provider == 'OpenAI' and not profile.image_base_url:
        profile.image_base_url = defaults['image_base_url']
    return profile


def load_profiles(profiles: List[Any]) -> List[LLMProfile]:
    """Load API profiles and one canonical Codex settings entry.

    >>> [profile.id for profile in load_profiles([])]
    ['codex']
    """
    loaded = []
    codex = None
    for profile in profiles or []:
        if not isinstance(profile, (Mapping, LLMProfile)):
            LOGGER.warning('Discard invalid LLM profile config entry.')
            continue
        profile_id = _profile_value(profile, 'id')
        is_codex = _profile_value(profile, 'backend') == 'codex'
        if is_codex != (profile_id == 'codex'):
            LOGGER.warning('Discard LLM profile with an invalid Codex identity.')
            continue
        if is_codex:
            if codex is not None:
                LOGGER.warning('Discard duplicate Codex settings entry.')
                continue
            data = profile.__dict__ if isinstance(profile, LLMProfile) else dict(profile)
            normalized = _normalize_profile_data(data)
            if any(key not in normalized or normalized[key] != value for key, value in data.items()):
                LOGGER.warning('Discard invalid or unknown Codex settings fields.')
            normalized.update(id='codex', backend='codex', name='Codex', built_in=True, title_url='')
            codex = profile_from_config({**default_codex_profile().__dict__, **normalized})
            loaded.append(codex)
            continue
        loaded.append(_merge_builtin_profile_options(profile_from_config(profile)))
    if codex is None:
        loaded.append(default_codex_profile())
    return loaded


def restore_builtin_profiles(existing_profiles: List[LLMProfile]) -> List[LLMProfile]:
    """Restore API defaults in normalized config, keeping user and Codex settings.

    Example:
        >>> custom = copy_profile(default_profile('OpenAI'))
        >>> custom.id = 'custom'
        >>> restore_builtin_profiles([default_profile('OpenAI'), custom])[0].id
        'custom'
    """

    codex_profiles = [profile for profile in existing_profiles if profile.backend == 'codex']
    user_profiles = [p for p in existing_profiles if not p.built_in]
    preserved_keys = {}
    for profile in existing_profiles:
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
    return user_profiles + builtins + codex_profiles


def copy_profile(profile: Any) -> LLMProfile:
    copied = profile_from_config(profile)
    if copied.backend == 'codex':
        raise ValueError('Codex settings cannot be copied as an API profile.')
    # The profile panel assigns a unique ID before adding the copy to config.
    copied.id = ''
    copied.name = copied.name + " Copy"
    copied.built_in = False
    return copied


def resolve_api_key(profile: Any, secret_store: SecretStore = None) -> str:
    secret_store = secret_store or SecretStore()
    return secret_store.resolve(_profile_value(profile, "api_key", "")).value


def store_api_key(profile: LLMProfile, api_key: str, secret_store: SecretStore = None) -> None:
    secret_store = secret_store or SecretStore()
    profile.api_key = secret_store.store(profile.id, api_key or "")
