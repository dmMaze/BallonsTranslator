import base64
import json
import re
import time
from typing import Dict, List

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
    blk_list: List[TextBlock],
    box_color: tuple = (0, 0, 255),
    font_scale: float = 1.2,
    thickness: int = 3,
    censored: bool = True
) -> np.ndarray:
    if censored:
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
        "page_level_ocr": {
            "value": True,
            "type": "checkbox",
            "display_name": "Page-Level OCR",
            "description": "Process the entire page in a single request with numbered boxes instead of cropped slices.",
        },
        "censorship": {
            "value": True,
            "type": "checkbox",
            "display_name": "Censorship (Blackout Image)",
            "description": "Black out all non-text areas of the page image before sending it to the Vision LLM.",
        },
        "sort_by_llm": {
            "value": True,
            "type": "checkbox",
            "display_name": "Sort Reading Order",
            "description": "Re-order text blocks according to the reading flow determined by the Vision LLM.",
        },
        "font_scale": {
            "value": 1.2,
            "type": "line_editor",
            "display_name": "Label Font Scale",
            "description": "Font size scale for block number labels on the page image.",
        },
        "box_color": {
            "value": "0, 0, 255",
            "type": "line_editor",
            "display_name": "Box Color (RGB)",
            "description": "Box border color in RGB (e.g. '0, 0, 255').",
        },
        "custom_prompt": {
            "value": "",
            "type": "line_editor",
            "display_name": "Custom OCR Prompt",
            "description": "Additional custom instructions appended to the OCR prompt.",
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
        openai = self._openai_module()
        client = self._initialize_client(profile)
        self._respect_delay()
        try:
            completion = client.chat.completions.create(**self._api_args(profile, messages))
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

    @property
    def page_level_ocr(self) -> bool:
        return bool(self.get_param_value('page_level_ocr'))

    @property
    def censorship(self) -> bool:
        return bool(self.get_param_value('censorship'))

    @property
    def sort_by_llm(self) -> bool:
        return bool(self.get_param_value('sort_by_llm'))

    @property
    def font_scale(self) -> float:
        try:
            return float(self.get_param_value('font_scale'))
        except (ValueError, TypeError):
            return 1.2

    @property
    def box_color_bgr(self) -> tuple:
        raw = str(self.get_param_value('box_color') or '0, 0, 255')
        try:
            parts = [int(p.strip()) for p in raw.split(',')]
            if len(parts) == 3:
                r, g, b = parts
                return (b, g, r)
        except Exception:
            pass
        return (255, 0, 0)

    @property
    def custom_prompt_override(self) -> str:
        return str(self.get_param_value('custom_prompt') or '').strip()

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

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if not self.page_level_ocr or not blk_list:
            return super()._ocr_blk_list(img, blk_list, *args, **kwargs)

        self.logger.info(f"Performing Page-level LLM OCR on {len(blk_list)} blocks...")
        annotated_img = create_annotated_page(
            img,
            blk_list,
            box_color=self.box_color_bgr,
            font_scale=self.font_scale,
            thickness=3,
            censored=self.censorship
        )

        custom_p = self.custom_prompt_override
        img_layout_desc = "all non-text areas are blacked out for safety" if self.censorship else "the full page layout is visible"
        prompt = (
            f"The input image is a page from a comic/manga where {img_layout_desc}. "
            f"There are {len(blk_list)} text blocks labeled with colored boxes and numbers from 1 to {len(blk_list)}.\n\n"
            "Your task is to perform OCR on each block individually and return the exact text for each block number.\n"
            "CRITICAL: Analyze the visual panel layout and flow of speech bubbles on the page to determine the correct reading order (typically right-to-left, top-to-bottom for Japanese manga). "
            "Sort the keys in the returned JSON object in this correct reading order so they follow the natural flow of the story.\n\n"
        )
        if custom_p:
            prompt += f"Apply these additional OCR instructions: {custom_p}\n\n"
        prompt += (
            "Return ONLY a valid JSON object mapping block numbers to their text. "
            "For example:\n"
            "{\n"
            '  "1": "First block text",\n'
            '  "2": "Second block text"\n'
            "}\n\n"
            "Do not include any explanation, code blocks, or markdown formatting in your response. "
            "If a block is completely empty or contains no text, map it to an empty string."
        )

        try:
            raw_response = self._request_page_ocr(annotated_img, prompt)
            parsed_results = self._parse_page_ocr_response(raw_response, len(blk_list))

            # Set text for all blocks and fall back for missing ones
            for i, blk in enumerate(blk_list):
                blk_num_str = str(i + 1)
                if blk_num_str in parsed_results:
                    blk.text = parsed_results[blk_num_str]
                else:
                    self.logger.warning(f"Block #{blk_num_str} text was missing in response. Falling back to crop OCR.")
                    self._ocr_single_block_fallback(img, blk)

            # Re-order the blocks list in-place based on the reading order determined by the LLM
            if self.sort_by_llm:
                ordered_blks = []
                seen_indices = set()
                for blk_num_str in parsed_results.keys():
                    try:
                        idx = int(blk_num_str) - 1
                        if 0 <= idx < len(blk_list) and idx not in seen_indices:
                            ordered_blks.append(blk_list[idx])
                            seen_indices.add(idx)
                    except ValueError:
                        continue
                # Append any blocks that were not returned by the LLM
                for idx, blk in enumerate(blk_list):
                    if idx not in seen_indices:
                        ordered_blks.append(blk)

                blk_list[:] = ordered_blks
                self.logger.info("Page-level OCR: Re-ordered text blocks based on LLM reading order flow.")

        except Exception as e:
            self.logger.error(f"Page-level LLM OCR failed: {e}. Falling back to block-by-block OCR.")
            return super()._ocr_blk_list(img, blk_list, *args, **kwargs)

    def _request_page_ocr(self, img: np.ndarray, prompt: str) -> str:
        profile = self.profile
        messages = [
            {"role": "system", "content": "You are a precise comic/manga OCR assistant. You output raw text in a JSON mapping format."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    self._image_content_part(img, profile),
                ],
            },
        ]

        retry_attempt = 0
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._request_ocr(profile, messages)
                return result
            except (LLMApiKeyRequiredError, LLMModelRequiredError, LLMRequestStopped):
                raise
            except Exception as e:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(f'Page-level LLM OCR request failed: {e}') from e
                self.logger.warning(f"Page-level LLM OCR request failed: {e}. Attempt: {retry_attempt}")
                self._wait(self.get_param_value('retry timeout'))

    def _parse_page_ocr_response(self, raw_response: str, expected_count: int) -> Dict[str, str]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    result[str(k).strip()] = self._normalized_text(str(v))
                return result
        except Exception as e:
            self.logger.error(f"Failed to parse page-level OCR JSON response: {e}. Raw response: {raw_response}")

        # Fallback to regex pattern matching
        result = {}
        pattern = re.compile(r'"(\d+)"\s*:\s*"([^"]*)"')
        for match in pattern.finditer(cleaned):
            result[match.group(1)] = self._normalized_text(match.group(2))
        return result

    def _ocr_single_block_fallback(self, img: np.ndarray, blk: TextBlock):
        im_h, im_w = img.shape[:2]
        x1, y1, x2, y2 = blk.xyxy
        y1c, y2c = max(0, y1), min(im_h, y2)
        x1c, x2c = max(0, x1), min(im_w, x2)
        if y1c < y2c and x1c < x2c:
            cropped_img = img[y1c:y2c, x1c:x2c]
            blk.text = self.ocr_img(cropped_img)
        else:
            blk.text = ""
