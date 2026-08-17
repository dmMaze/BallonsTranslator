import base64
import io
import time
from typing import Dict, List
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
from PIL import Image

from .base import InpainterBase, register_inpainter
from ..textdetector import TextBlock
from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
    LLMRequestStopped,
)
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, profile_by_id, profile_from_config, resolve_api_key


@register_inpainter("LLMInpaint")
class LLMInpaint(InpainterBase):
    """Profile-backed image cleanup using image-capable LLM APIs.

    Example:
        >>> LLMInpaint._image_model_required('demo', ['demo'])
        'demo'
    """

    dependencies = ['httpx[socks,brotli]']

    params: Dict = {
        "max requests per minute": {
            "value": 5,
            "display_name": "Max Requests Per Minute",
            "description": "Global request limit for LLM image cleanup.",
        },
        "delay": {
            "value": 0.5,
            "display_name": "Delay",
            "description": "Delay between LLM image cleanup requests in seconds.",
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
        "request timeout": {
            "value": 180.0,
            "display_name": "Request Timeout",
            "description": "HTTP timeout for image cleanup requests in seconds. Set to 0 to disable.",
        },
        "max resolution": {
            "type": "selector",
            "options": [0, 256, 768, 1280],
            "value": 1280,
            "display_name": "Max Resolution",
            "description": "Scale images down before sending them to the LLM. Set to 0 to keep the original size.",
        },
        "inpaint by block": {
            "type": "checkbox",
            "value": True,
            "display_name": "Inpaint By Block",
            "description": "Send each text block crop separately instead of sending the whole image.",
        },
        "proxy": {
            "value": "",
            "display_name": "Proxy",
            "description": "Proxy address used for the OpenAI-compatible client.",
        },
        "description": "Inpaint using the selected image-capable LLM profile.",
    }

    inpaint_by_block = True

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._sync_inpaint_by_block()
        self.client = None
        self.client_cache_key = None
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.stop_event = None

    @property
    def profile(self) -> LLMProfile:
        profile = profile_by_id(pcfg.module.llm_profiles, pcfg.module.inpaint_llm_id)
        if profile is None and pcfg.module.llm_profiles:
            profile = pcfg.module.llm_profiles[0]
        if profile is None:
            raise RuntimeError('No LLM profile is configured.')
        profile = profile_from_config(profile)
        if not profile.support_image:
            raise RuntimeError(f'LLM profile "{profile.name}" does not have image cleanup enabled.')
        self._image_model(profile)
        self._image_base_url(profile)
        return profile

    @staticmethod
    def _image_model_required(model: str, model_options: List[str]) -> str:
        model = str(model or '').strip()
        options = [str(option).strip() for option in model_options if str(option).strip()]
        if not model or not options:
            return ''
        return model

    @classmethod
    def _image_model(cls, profile: LLMProfile) -> str:
        model = cls._image_model_required(profile.image_model, profile.image_model_options)
        if not model:
            raise LLMModelRequiredError(profile.id, profile.name, target='image_model')
        return model

    @staticmethod
    def _image_base_url(profile: LLMProfile) -> str:
        base_url = str(profile.image_base_url or '').strip()
        if not base_url:
            raise LLMBaseURLRequiredError(profile.id, profile.name, target='image_base_url')
        return base_url

    def _sync_inpaint_by_block(self):
        value = self.get_param_value('inpaint by block')
        if isinstance(value, str):
            value = value.lower().strip() == 'true'
        self.inpaint_by_block = bool(value)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'inpaint by block':
            self._sync_inpaint_by_block()

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

    def _request_timeout(self):
        try:
            timeout = float(self.get_param_value('request timeout') or 0)
        except (TypeError, ValueError):
            timeout = 180.0
        return None if timeout <= 0 else timeout

    def _max_resolution(self) -> int:
        try:
            return int(self.get_param_value('max resolution') or 0)
        except (TypeError, ValueError):
            return 1280

    def _scale_image_for_request(self, img: np.ndarray) -> np.ndarray:
        max_resolution = self._max_resolution()
        if max_resolution <= 0:
            return img
        height, width = img.shape[:2]
        long_side = max(height, width)
        if long_side <= max_resolution:
            return img
        scale = max_resolution / long_side
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def _http_client(self, proxy: str):
        import httpx  # type: ignore

        client_kwargs = {'timeout': self._request_timeout()}
        if not proxy:
            return httpx.Client(**client_kwargs)
        try:
            mounts = {
                "http://": httpx.HTTPTransport(proxy=proxy),
                "https://": httpx.HTTPTransport(proxy=proxy),
            }
            return httpx.Client(mounts=mounts, **client_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy.")
            return httpx.Client(**client_kwargs)

    def _api_key_for_profile(self, profile: LLMProfile) -> str:
        api_key = resolve_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _initialize_client(self, profile: LLMProfile):
        api_key = self._api_key_for_profile(profile)
        base_url = self._image_base_url(profile)
        proxy = self.get_param_value('proxy') or ''
        request_timeout = self._request_timeout()
        cache_key = (api_key, base_url, proxy, request_timeout)
        if self.client is not None and self.client_cache_key == cache_key:
            return self.client

        self.client = self._http_client(proxy)
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
    def _response_error_message(response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get('error')
                if isinstance(err, dict) and err.get('message'):
                    return str(err['message'])
                if data.get('message'):
                    return str(data['message'])
                if data.get('detail'):
                    return str(data['detail'])
        except Exception:
            pass
        text = getattr(response, 'text', '')
        if text:
            return str(text)
        status_code = getattr(response, 'status_code', '')
        reason = getattr(response, 'reason_phrase', '')
        return f'HTTP {status_code} {reason}'.strip()

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        base = base_url.rstrip('/')
        endpoint = '/' + path.strip('/')
        if urlparse(base).path.rstrip('/').endswith(endpoint):
            return base
        return f"{base}{endpoint}"

    @staticmethod
    def _is_openrouter_url(base_url: str) -> bool:
        host = urlparse(base_url).netloc.lower()
        return host == 'openrouter.ai' or host.endswith('.openrouter.ai')

    @staticmethod
    def _is_gemini_url(base_url: str) -> bool:
        return urlparse(base_url).netloc.lower() == 'generativelanguage.googleapis.com'

    @classmethod
    def _gemini_generate_content_url(cls, base_url: str, model: str) -> str:
        base = base_url.rstrip('/')
        parsed = urlparse(base)
        path = parsed.path.rstrip('/')
        if path.endswith(':generateContent'):
            return base
        if path.endswith('/openai'):
            path = path[:-len('/openai')]
            base = urlunparse(parsed._replace(path=path, params='', query='', fragment='')).rstrip('/')
        model_path = model if model.startswith('models/') else f'models/{model}'
        return cls._join_url(base, f'/{model_path}:generateContent')

    @staticmethod
    def _png_image_file(img: np.ndarray) -> io.BytesIO:
        if img.ndim != 3 or img.shape[2] < 3:
            raise RuntimeError('LLM image cleanup requires an RGB image.')
        rgb = img[:, :, :3]
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format='PNG')
        buffer.seek(0)
        buffer.name = 'image.png'
        return buffer

    def _api_args(self, profile: LLMProfile, image_file, prompt: str = None) -> Dict:
        return {
            "model": self._image_model(profile),
            "image": image_file,
            "prompt": prompt if prompt is not None else profile.image_prompt,
        }

    def _openrouter_api_args(self, profile: LLMProfile, image_file, prompt: str = None) -> Dict:
        encoded_image = base64.b64encode(image_file.getvalue()).decode('ascii')
        return {
            "model": self._image_model(profile),
            "prompt": prompt if prompt is not None else profile.image_prompt,
            "input_references": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}",
                    },
                }
            ],
            "output_format": "png",
            "n": 1,
        }

    def _gemini_api_args(self, profile: LLMProfile, image_file, prompt: str = None) -> Dict:
        encoded_image = base64.b64encode(image_file.getvalue()).decode('ascii')
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt if prompt is not None else profile.image_prompt,
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": encoded_image,
                            },
                        },
                    ],
                },
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
            },
        }

    @staticmethod
    def _response_field(item, field_name: str):
        if isinstance(item, dict):
            return item.get(field_name)
        return getattr(item, field_name, None)

    def _decode_image_bytes(self, raw: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        return np.array(image)

    def _download_image(self, url: str) -> np.ndarray:
        proxy = self.get_param_value('proxy') or ''
        client = self._http_client(proxy)
        try:
            response = client.get(url)
            response.raise_for_status()
            return self._decode_image_bytes(response.content)
        finally:
            client.close()

    def _decode_response_image(self, response) -> np.ndarray:
        data = self._response_field(response, 'data')
        if not data:
            raise RuntimeError('LLM image cleanup returned no image data.')

        item = data[0]
        b64_json = self._response_field(item, 'b64_json')
        if b64_json:
            return self._decode_image_bytes(base64.b64decode(b64_json))

        url = self._response_field(item, 'url')
        if url:
            return self._download_image(str(url))

        raise RuntimeError('LLM image cleanup returned no decodable image.')

    def _decode_gemini_response_image(self, response) -> np.ndarray:
        candidates = self._response_field(response, 'candidates') or []
        for candidate in candidates:
            content = self._response_field(candidate, 'content') or {}
            for part in self._response_field(content, 'parts') or []:
                inline_data = (
                    self._response_field(part, 'inline_data')
                    or self._response_field(part, 'inlineData')
                )
                data = self._response_field(inline_data, 'data') if inline_data else None
                if data:
                    return self._decode_image_bytes(base64.b64decode(str(data)))

        output_image = (
            self._response_field(response, 'output_image')
            or self._response_field(response, 'outputImage')
        )
        data = self._response_field(output_image, 'data') if output_image else None
        if data:
            return self._decode_image_bytes(base64.b64decode(str(data)))

        steps = self._response_field(response, 'steps') or []
        for step in steps:
            if self._response_field(step, 'type') != 'model_output':
                continue
            for content_block in self._response_field(step, 'content') or []:
                if self._response_field(content_block, 'type') != 'image':
                    continue
                data = self._response_field(content_block, 'data')
                if data:
                    return self._decode_image_bytes(base64.b64decode(str(data)))

        raise RuntimeError('Gemini image cleanup returned no decodable image.')

    def _headers(self, api_key: str, json_request: bool = False) -> Dict:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        if json_request:
            headers['Content-Type'] = 'application/json'
        return headers

    @staticmethod
    def _gemini_headers(api_key: str) -> Dict:
        return {
            'x-goog-api-key': api_key,
            'Content-Type': 'application/json',
        }

    def _raise_for_response(self, profile: LLMProfile, response):
        status_code = getattr(response, 'status_code', 200)
        if status_code < 400:
            return
        if status_code in (401, 403):
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        raise RuntimeError(self._response_error_message(response))

    def _request_openrouter_inpaint(self, client, profile: LLMProfile, image_file, prompt: str = None) -> np.ndarray:
        base_url = self._image_base_url(profile)
        api_key = self._api_key_for_profile(profile)
        response = client.post(
            self._join_url(base_url, '/images'),
            headers=self._headers(api_key, json_request=True),
            json=self._openrouter_api_args(profile, image_file, prompt=prompt),
        )
        self._raise_for_response(profile, response)
        return self._decode_response_image(response.json())

    def _request_gemini_inpaint(self, client, profile: LLMProfile, image_file, prompt: str = None) -> np.ndarray:
        base_url = self._image_base_url(profile)
        api_key = self._api_key_for_profile(profile)
        model = self._image_model(profile)
        response = client.post(
            self._gemini_generate_content_url(base_url, model),
            headers=self._gemini_headers(api_key),
            json=self._gemini_api_args(profile, image_file, prompt=prompt),
        )
        self._raise_for_response(profile, response)
        return self._decode_gemini_response_image(response.json())

    def _request_openai_compatible_inpaint(self, client, profile: LLMProfile, image_file, prompt: str = None) -> np.ndarray:
        base_url = self._image_base_url(profile)
        api_key = self._api_key_for_profile(profile)
        args = self._api_args(profile, image_file, prompt=prompt)
        response = client.post(
            base_url,
            headers=self._headers(api_key),
            data={
                'model': args['model'],
                'prompt': args['prompt'],
            },
            files={
                'image': ('image.png', image_file.getvalue(), 'image/png'),
            },
        )
        self._raise_for_response(profile, response)
        return self._decode_response_image(response.json())

    def _request_inpaint(self, profile: LLMProfile, img: np.ndarray, prompt: str = None) -> np.ndarray:
        client = self._initialize_client(profile)
        request_img = self._scale_image_for_request(img)
        image_file = self._png_image_file(request_img)
        self._respect_delay()
        try:
            base_url = self._image_base_url(profile)
            if self._is_gemini_url(base_url):
                result = self._request_gemini_inpaint(client, profile, image_file, prompt=prompt)
            elif self._is_openrouter_url(base_url):
                result = self._request_openrouter_inpaint(client, profile, image_file, prompt=prompt)
            else:
                result = self._request_openai_compatible_inpaint(client, profile, image_file, prompt=prompt)
            if result.shape[:2] != img.shape[:2]:
                result = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            return result
        finally:
            image_file.close()

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        profile = self.profile
        retry_attempt = 0
        mask_original = (mask > 127)[..., None].astype(np.uint8)
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                result = self._request_inpaint(profile, img)
                if result.shape[:2] != img.shape[:2]:
                    result = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                result = result.astype(np.uint8, copy=False)
                img_inpainted = result * mask_original + img * (1 - mask_original)
                return img_inpainted
            except LLMApiKeyRequiredError:
                raise
            except LLMModelRequiredError:
                raise
            except LLMBaseURLRequiredError:
                raise
            except LLMRequestStopped:
                raise
            except Exception as e:
                retry_attempt += 1
                if retry_attempt >= self.get_param_value('retry attempts'):
                    raise RuntimeError(f'LLM image cleanup failed: {e}') from e
                self.logger.warning(f"LLM image cleanup failed due to {e}. Attempt: {retry_attempt}")
                self._wait(self.get_param_value('retry timeout'))
