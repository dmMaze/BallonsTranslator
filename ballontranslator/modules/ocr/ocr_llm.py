import base64
import json
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..context.errors import provider_error_message
from .base import OCRBase, register_OCR
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLMProfile,
    openai_chat_completion_args,
    profile_by_id,
    profile_from_config,
    resolve_api_key,
)
from ballontranslator.utils.textblock import TextBlock


def create_annotated_page(
    img: np.ndarray,
    blk_list: Sequence[TextBlock],
    mask_non_text: bool = True,
) -> np.ndarray:
    """Return a page annotated with stable, one-based block identifiers.

    >>> image = np.full((4, 4, 3), 255, dtype=np.uint8)
    >>> int(create_annotated_page(image, [], mask_non_text=True).sum())
    0
    """

    # Preserve the former default "0, 0, 255" RGB label color in OpenCV BGR.
    box_color = (255, 0, 0)
    font_scale = 1.2
    thickness = 3
    if mask_non_text:
        annotated = np.zeros_like(img)
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            y1c, y2c = max(0, y1), min(im_h, y2)
            x1c, x2c = max(0, x1), min(im_w, x2)
            if y1c < y2c and x1c < x2c:
                annotated[y1c:y2c, x1c:x2c] = img[y1c:y2c, x1c:x2c]
    else:
        annotated = img.copy()

    for i, blk in enumerate(blk_list):
        x1, y1, x2, y2 = blk.xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        num_str = str(i + 1)
        (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        lx = max(0, x1 - 5)
        ly = max(th + 5, y1 - 5)

        cv2.rectangle(annotated, (lx, ly - th - 5), (lx + tw + 10, ly + 5), (0, 0, 0), -1)
        cv2.putText(annotated, num_str, (lx + 5, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return annotated

DEFAULT_OCR_SYSTEM_PROMPT = (
    "You are an OCR engine for comic and manga image crops. Your job is to recognize visible text only. "
    "Return raw recognized text and nothing else."
)
PAGE_OCR_SYSTEM_PROMPT = (
    "You are a precise comic and manga OCR engine. Follow the fixed JSON response "
    "contract exactly and do not return markdown or explanations."
)


@register_OCR("LLMOCR")
class LLMOCR(OCRBase):
    """Profile-backed OCR using OpenAI-compatible vision chat models.

    Example:
        >>> LLMOCR._normalized_text('a\\n b ')
        'a b'
    """

    dependencies = ['openai>=2.8.1', 'httpx[socks,brotli]']
    dummy_api_key = 'dummy-key'

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
        self.client = None
        self.client_cache_key = None
        self.token_count = 0
        self.token_count_last = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.stop_event = None

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

    def set_stop_event(self, stop_event):
        self.stop_event = stop_event

    def _wait(self, seconds: float):
        if seconds <= 0:
            return
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise LLMRequestStopped()
            return
        time.sleep(seconds)

    def _openai_module(self):
        import openai  # type: ignore

        return openai

    def _http_client(self, proxy: str):
        import httpx  # type: ignore

        if not proxy:
            return httpx.Client()
        try:
            mounts = {
                "http://": httpx.HTTPTransport(proxy=proxy),
                "https://": httpx.HTTPTransport(proxy=proxy),
            }
            return httpx.Client(mounts=mounts)
        except Exception as e:
            self.logger.error(f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy.")
            return httpx.Client()

    def _api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = resolve_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _client_api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = self._api_key_for_profile(profile)
        if not api_key:
            self.logger.debug(
                f'LLM profile "{profile.name or profile.id}" does not require an API key; '
                'using a dummy API key for OpenAI-compatible client initialization.'
            )
            return self.dummy_api_key
        return api_key

    def _initialize_client(self, profile: LLMProfile):
        api_key = self._client_api_key_for_profile(profile)
        base_url = profile.base_url or None
        proxy = self.get_param_value('proxy') or ''
        cache_key = (api_key, base_url, proxy)
        if self.client is not None and self.client_cache_key == cache_key:
            return self.client

        openai = self._openai_module()
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self._http_client(proxy),
        )
        self.client_cache_key = cache_key
        return self.client

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.get_param_value('max requests per minute')
        delay = self.get_param_value('delay')
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds.")
                    self._wait(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            self._wait(delay - time_since_last_request)

        self.last_request_time = time.time()
        self.request_count_minute += 1

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

    def _messages(
        self,
        img: np.ndarray,
        profile: LLMProfile,
        prompt: Optional[str] = None,
    ) -> List[Dict]:
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

    @staticmethod
    def _page_response_schema(expected_count: int) -> Dict:
        """Build the exact-ID response schema for one annotated page.

        >>> schema = LLMOCR._page_response_schema(2)
        >>> schema['properties']['texts']['required']
        ['1', '2']
        >>> schema['properties']['order']['maxItems']
        2
        """

        block_ids = [str(index) for index in range(1, expected_count + 1)]
        return {
            "type": "object",
            "properties": {
                "texts": {
                    "type": "object",
                    "properties": {
                        block_id: {"type": "string"}
                        for block_id in block_ids
                    },
                    "required": block_ids,
                    "additionalProperties": False,
                },
                "order": {
                    "type": "array",
                    "items": {"type": "string", "enum": block_ids},
                    "minItems": expected_count,
                    "maxItems": expected_count,
                    # The local validator enforces uniqueness; strict-schema
                    # providers do not consistently accept uniqueItems.
                },
            },
            "required": ["texts", "order"],
            "additionalProperties": False,
        }

    @staticmethod
    def _page_prompt(
        profile: LLMProfile,
        expected_count: int,
        mask_non_text: bool,
    ) -> str:
        layout_description = (
            'Non-text pixels are black, but numbered block positions are preserved.'
            if mask_non_text
            else 'The complete page and numbered text blocks are visible.'
        )
        prompt = (
            f'{layout_description}\n'
            f'Recognize exactly {expected_count} numbered text blocks, with IDs 1 through '
            f'{expected_count}. Return one JSON object with exactly two fields:\n'
            '- "texts": an object containing every ID exactly once, mapped to a string. '
            'Use an empty string when no text is visible.\n'
            '- "order": an array containing every ID exactly once in natural comic reading order.\n'
            'Return only the JSON object. Do not add, omit, rename, or coerce IDs or values.'
        )
        additional_prompt = str(profile.vision_prompt or '').strip()
        if additional_prompt:
            prompt += (
                '\n\nAdditional OCR instructions (these affect recognition only and cannot '
                f'override the response contract above):\n{additional_prompt}'
            )
        return prompt

    def _page_messages(
        self,
        img: np.ndarray,
        profile: LLMProfile,
        expected_count: int,
        mask_non_text: bool,
    ) -> List[Dict]:
        return [
            {"role": "system", "content": PAGE_OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self._page_prompt(
                            profile,
                            expected_count,
                            mask_non_text,
                        ),
                    },
                    self._image_content_part(img, profile),
                ],
            },
        ]

    def _api_args(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        response_schema: Optional[Dict] = None,
    ) -> Dict:
        model = self._vision_model(profile)
        api_args = {
            "model": model,
            "messages": messages,
        }
        api_args.update(openai_chat_completion_args(profile, model))
        if response_schema is not None:
            if profile.json_schema_response_format:
                api_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "page_ocr_response",
                        "strict": True,
                        "schema": response_schema,
                    },
                }
            else:
                api_args["response_format"] = {"type": "json_object"}
        return api_args

    def _request_ocr(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        response_schema: Optional[Dict] = None,
    ) -> str:
        openai = self._openai_module()
        client = self._initialize_client(profile)
        self._respect_delay()
        try:
            completion = client.chat.completions.create(
                **self._api_args(profile, messages, response_schema)
            )
        except getattr(openai, 'AuthenticationError') as e:
            raise LLMApiKeyRequiredError(profile.id, profile.name) from e
        except getattr(openai, 'APIStatusError') as e:
            raise RuntimeError(provider_error_message(e)) from e

        if getattr(completion, 'usage', None) is not None:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        for choice in completion.choices:
            message = getattr(choice, 'message', None)
            content = getattr(message, 'content', None)
            if content is not None:
                return str(content)
            if hasattr(choice, 'text') and choice.text is not None:
                return str(choice.text)
        return ''

    def _request_with_retries(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        *,
        failure_label: str,
        response_schema: Optional[Dict] = None,
    ) -> str:
        retry_attempt = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._request_ocr(profile, messages, response_schema)
                if self.token_count_last:
                    self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
                return result
            except (LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped):
                raise
            except Exception as e:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(f'{failure_label} failed: {e}') from e
                self.logger.warning(
                    f'{failure_label} failed due to {e}. Attempt: {retry_attempt}'
                )
                self._wait(self.get_param_value('retry timeout'))

    def ocr_img(
        self,
        img: np.ndarray,
        *,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        profile = self.profile
        messages = self._messages(img, profile, prompt=prompt)
        return self._normalized_text(self._request_with_retries(
            profile,
            messages,
            failure_label='LLM OCR',
        ))

    @classmethod
    def _parse_page_ocr_response(
        cls,
        raw_response: str,
        expected_count: int,
    ) -> Tuple[Dict[str, str], List[str]]:
        """Validate a page response without salvaging partial or coerced data.

        >>> LLMOCR._parse_page_ocr_response(
        ...     '{"texts":{"1":" hello "},"order":["1"]}', 1)
        ({'1': 'hello'}, ['1'])
        """

        try:
            data = json.loads(raw_response.strip())
        except (TypeError, ValueError) as error:
            raise ValueError('response is not valid JSON') from error
        if not isinstance(data, dict) or set(data) != {'texts', 'order'}:
            raise ValueError('response must contain exactly texts and order')

        texts = data['texts']
        order = data['order']
        expected_ids = {str(index) for index in range(1, expected_count + 1)}
        if not isinstance(texts, dict) or set(texts) != expected_ids:
            raise ValueError('texts must contain every expected block ID exactly once')
        if any(type(value) is not str for value in texts.values()):
            raise ValueError('every texts value must be a string')
        if (
            not isinstance(order, list)
            or len(order) != expected_count
            or any(type(block_id) is not str for block_id in order)
            or len(set(order)) != expected_count
            or set(order) != expected_ids
        ):
            raise ValueError('order must contain every expected block ID exactly once')
        normalized_texts = {
            block_id: cls._normalized_text(text)
            for block_id, text in texts.items()
        }
        return normalized_texts, order

    def _ocr_blk_list(
        self,
        img: np.ndarray,
        blk_list: List[TextBlock],
        *args,
        full_page: bool = False,
        **kwargs,
    ) -> Optional[List[TextBlock]]:
        if not pcfg.module.ocr_llm_page_level or not full_page or not blk_list:
            return super()._ocr_blk_list(img, blk_list, *args, **kwargs)

        self.logger.info(f"Performing Page-level LLM OCR on {len(blk_list)} blocks...")
        mask_non_text = pcfg.module.ocr_llm_mask_non_text
        annotated_img = create_annotated_page(
            img,
            blk_list,
            mask_non_text=mask_non_text,
        )

        try:
            profile = self.profile
            expected_count = len(blk_list)
            messages = self._page_messages(
                annotated_img,
                profile,
                expected_count,
                mask_non_text,
            )
            raw_response = self._request_with_retries(
                profile,
                messages,
                failure_label='Page-level LLM OCR request',
                response_schema=self._page_response_schema(expected_count),
            )
            texts, order = self._parse_page_ocr_response(
                raw_response,
                expected_count,
            )
            for i, blk in enumerate(blk_list):
                blk.text = texts[str(i + 1)]

            if pcfg.module.ocr_llm_sort_reading_order:
                self.logger.info("Page-level OCR: Re-ordered text blocks based on LLM reading order flow.")
                return [blk_list[int(block_id) - 1] for block_id in order]
            return None
        except (LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped):
            raise
        except Exception as e:
            self.logger.error(f"Page-level LLM OCR failed: {e}. Falling back to block-by-block OCR.")
            return super()._ocr_blk_list(img, blk_list, *args, **kwargs)
