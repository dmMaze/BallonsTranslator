import json
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..llm_chat import (
    LLMChatRequester,
    openai_chat_completion_args,
    openai_json_response_format,
)
from ..llm_vision import encode_chat_image
from .base import OCRBase, register_OCR
from ballontranslator.modules.exceptions import LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    DEFAULT_OCR_PROMPT,
    LLMProfile,
    runtime_profile,
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
        profile = runtime_profile(
            pcfg.module.llm_profiles,
            pcfg.module.ocr_llm_id,
        )
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
        return encode_chat_image(
            img,
            detail=str(profile.vision_detail_level or 'None'),
            failure_message='Failed to encode OCR image.',
        ).image_part()

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
    def _page_response_schema(
        expected_count: int,
        sort_reading_order: bool,
    ) -> Dict:
        """Build the exact-ID response schema for one annotated page.

        >>> schema = LLMOCR._page_response_schema(2, True)
        >>> schema['properties']['texts']['required']
        ['1', '2']
        >>> schema['properties']['order']['maxItems']
        2
        """

        block_ids = [str(index) for index in range(1, expected_count + 1)]
        schema = {
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
            },
            "required": ["texts"],
            "additionalProperties": False,
        }
        if sort_reading_order:
            schema["properties"]["order"] = {
                "type": "array",
                "items": {"type": "string", "enum": block_ids},
                "minItems": expected_count,
                "maxItems": expected_count,
                # The local validator enforces uniqueness; strict-schema
                # providers do not consistently accept uniqueItems.
            }
            schema["required"].append("order")
        return schema

    @staticmethod
    def _page_prompt(
        profile: LLMProfile,
        expected_count: int,
        mask_non_text: bool,
        sort_reading_order: bool,
    ) -> str:
        layout_description = (
            'Non-text pixels are black, but numbered block positions are preserved.'
            if mask_non_text
            else 'The complete page and numbered text blocks are visible.'
        )
        response_fields = (
            '- "texts": an object containing every ID exactly once, mapped to a string. '
            'Use an empty string when no text is visible.'
        )
        field_count = 'one field'
        if sort_reading_order:
            response_fields += (
                '\n- "order": an array containing every ID exactly once in natural comic '
                'reading order.'
            )
            field_count = 'two fields'
        prompt = (
            f'{layout_description}\n'
            f'Recognize exactly {expected_count} numbered text blocks, with IDs 1 through '
            f'{expected_count}. Read vertical text in its intended character order. '
            f'Return one JSON object with exactly {field_count}:\n{response_fields}\n'
            'Return only the JSON object. Do not add, omit, rename, or coerce IDs or values.'
        )
        additional_prompt = str(profile.vision_prompt or '').strip()
        if additional_prompt and additional_prompt != DEFAULT_OCR_PROMPT:
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
        sort_reading_order: bool,
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
                            sort_reading_order,
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
            api_args["response_format"] = openai_json_response_format(
                profile,
                'page_ocr_response',
                response_schema,
            )
        return api_args

    def _request_ocr(
        self,
        profile: LLMProfile,
        messages: List[Dict],
        response_schema: Optional[Dict] = None,
    ) -> str:
        result = self.request_chat_completion(
            profile,
            self._api_args(profile, messages, response_schema),
        )
        if result.usage is not None:
            self.token_count += result.usage.total_tokens
            self.token_count_last = result.usage.total_tokens
        else:
            self.token_count_last = 0
        return result.content

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
        sort_reading_order: bool,
    ) -> Tuple[Dict[str, str], Optional[List[str]]]:
        """Validate a page response without salvaging partial or coerced data.

        >>> LLMOCR._parse_page_ocr_response(
        ...     '{"texts":{"1":" hello "},"order":["1"]}', 1, True)
        ({'1': 'hello'}, ['1'])
        """

        try:
            data = json.loads(raw_response.strip())
        except (TypeError, ValueError) as error:
            raise ValueError('response is not valid JSON') from error
        expected_fields = {'texts', 'order'} if sort_reading_order else {'texts'}
        if not isinstance(data, dict) or set(data) != expected_fields:
            fields_label = 'texts and order' if sort_reading_order else 'texts'
            raise ValueError(f'response must contain exactly {fields_label}')

        texts = data['texts']
        order = data.get('order')
        expected_ids = {str(index) for index in range(1, expected_count + 1)}
        if not isinstance(texts, dict) or set(texts) != expected_ids:
            raise ValueError('texts must contain every expected block ID exactly once')
        if any(type(value) is not str for value in texts.values()):
            raise ValueError('every texts value must be a string')
        if sort_reading_order:
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
        sort_reading_order = pcfg.module.ocr_llm_sort_reading_order
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
                sort_reading_order,
            )
            raw_response = self._request_with_retries(
                profile,
                messages,
                failure_label='Page-level LLM OCR request',
                response_schema=self._page_response_schema(
                    expected_count,
                    sort_reading_order,
                ),
            )
            texts, order = self._parse_page_ocr_response(
                raw_response,
                expected_count,
                sort_reading_order,
            )
            for i, blk in enumerate(blk_list):
                blk.text = texts[str(i + 1)]

            if sort_reading_order and order is not None:
                self.logger.info("Page-level OCR: Re-ordered text blocks based on LLM reading order flow.")
                return [blk_list[int(block_id) - 1] for block_id in order]
            return None
        except (LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped):
            raise
        except Exception as e:
            self.logger.error(f"Page-level LLM OCR failed: {e}. Falling back to block-by-block OCR.")
            return super()._ocr_blk_list(img, blk_list, *args, **kwargs)
