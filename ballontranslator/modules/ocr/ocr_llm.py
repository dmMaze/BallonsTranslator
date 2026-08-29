import base64
from typing import Dict, List

import cv2
import numpy as np

from ..llm_chat import LLMChatRequester, openai_chat_completion_args
from .base import OCRBase, register_OCR
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    profile_by_id,
    profile_from_config,
)

DEFAULT_OCR_SYSTEM_PROMPT = (
    "You are an OCR engine for comic and manga image crops. Your job is to recognize visible text only. "
    "Return raw recognized text and nothing else."
)


@register_OCR("LLMOCR")
class LLMOCR(LLMChatRequester, OCRBase):
    """Profile-backed OCR using OpenAI-compatible vision chat models.

    Example:
        >>> LLMOCR._normalized_text('a\\n b ')
        'a b'
    """

    dependencies = ['openai>=2.8.1', 'httpx[socks,brotli]']

    params: Dict = {
        "max requests per minute": {
            "value": 20,
            "display_name": "Max Requests Per Minute",
            "description": "Global request limit for LLM OCR.",
        },
        "delay": {
            "value": 0.3,
            "display_name": "Delay",
            "description": "Delay between LLM OCR requests in seconds.",
        },
        "retry attempts": {
            "value": 3,
            "display_name": "Retry Attempts",
            "description": "Retries for API failures.",
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
        "description": "OCR using the selected vision-capable LLM profile.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.token_count = 0
        self.token_count_last = 0

    @property
    def profile(self) -> LLMProfile:
        profile = profile_by_id(pcfg.module.llm_profiles, pcfg.module.ocr_llm_id)
        if profile is None and pcfg.module.llm_profiles:
            profile = pcfg.module.llm_profiles[0]
        if profile is None:
            raise RuntimeError('No LLM profile is configured.')
        profile = profile_from_config(profile)
        if not profile.support_vision:
            raise RuntimeError(f'LLM profile "{profile.name}" does not have vision enabled.')
        self._vision_model(profile)
        return profile

    @staticmethod
    def _vision_model(profile: LLMProfile) -> str:
        model = str(profile.vision_model or '').strip()
        model_options = [str(option).strip() for option in profile.vision_model_options if str(option).strip()]
        if not model or not model_options:
            raise LLMModelRequiredError(profile.id, profile.name, target='vision_model')
        return model

    @staticmethod
    def _normalized_text(text: str) -> str:
        return ' '.join(str(text or '').replace('\r', '\n').split()).strip()

    def _image_content_part(self, img: np.ndarray, profile: LLMProfile) -> Dict:
        success, buffer = cv2.imencode(".jpg", img)
        if not success:
            raise RuntimeError('Failed to encode OCR image.')
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        image_content_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
        }
        detail_level = str(profile.vision_detail_level or 'None')
        if detail_level.lower() != 'none':
            image_content_part["image_url"]["detail"] = detail_level
        return image_content_part

    def _messages(self, img: np.ndarray, profile: LLMProfile, prompt: str = None) -> List[Dict]:
        return [
            {"role": "system", "content": DEFAULT_OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt if prompt is not None else profile.vision_prompt},
                    self._image_content_part(img, profile),
                ],
            },
        ]

    def _api_args(self, profile: LLMProfile, messages: List[Dict]):
        model = self._vision_model(profile)
        api_args = {
            "model": model,
            "messages": messages,
        }
        api_args.update(openai_chat_completion_args(profile, model))
        return api_args

    def _request_ocr(self, profile: LLMProfile, messages: List[Dict]) -> str:
        result = self.request_chat_completion(
            profile,
            self._api_args(profile, messages),
        )
        if result.usage is not None:
            self.token_count += result.usage.total_tokens
            self.token_count_last = result.usage.total_tokens
        else:
            self.token_count_last = 0
        return result.content

    def ocr_img(self, img: np.ndarray, *, prompt: str = None, **kwargs) -> str:
        profile = self.profile
        messages = self._messages(img, profile, prompt=prompt)
        retry_attempt = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._normalized_text(self._request_ocr(profile, messages))
                if self.token_count_last:
                    self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
                return result
            except LLMApiKeyRequiredError:
                raise
            except LLMModelRequiredError:
                raise
            except LLMRequestStopped:
                raise
            except Exception as e:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(f'LLM OCR failed: {e}') from e
                self.logger.warning(f"LLM OCR failed due to {e}. Attempt: {retry_attempt}")
                self._wait(self.get_param_value('retry timeout'))
