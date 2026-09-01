"""Reusable profile-backed LLM image request transport."""

import base64
import io
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
from PIL import Image

from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
    LLMRequestStopped,
    LLMUserActionRequiredError,
)
from ballontranslator.utils.llm_profiles import LLMProfile, resolve_api_key
from ballontranslator.utils.logger import logger as LOGGER


@dataclass(frozen=True)
class LLMImageRequestPolicy:
    """Network policy shared by inpainting and Image-card generation.

    >>> LLMImageRequestPolicy.from_module_params({
    ...     'request timeout': {'value': 12},
    ... }).request_timeout
    12.0
    """

    max_requests_per_minute: int = 5
    delay: float = 0.5
    retry_attempts: int = 3
    retry_timeout: float = 7.0
    request_timeout: float = 180.0
    max_resolution: int = 1280
    proxy: str = ''

    @classmethod
    def from_module_params(
        cls, params: Optional[Mapping[str, object]]
    ) -> "LLMImageRequestPolicy":
        values = {} if params is None else dict(params)

        def read(name: str, default: object) -> object:
            value = values.get(name, default)
            return value.get('value', default) if isinstance(value, dict) else value

        def integer(name: str, default: int) -> int:
            try:
                return int(read(name, default))
            except (TypeError, ValueError):
                return default

        def number(name: str, default: float) -> float:
            try:
                return float(read(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            max_requests_per_minute=max(
                0, integer('max requests per minute', 5)
            ),
            delay=max(0.0, number('delay', 0.5)),
            retry_attempts=max(1, integer('retry attempts', 3)),
            retry_timeout=max(0.0, number('retry timeout', 7.0)),
            request_timeout=number('request timeout', 180.0),
            max_resolution=max(0, integer('max resolution', 1280)),
            proxy=str(read('proxy', '') or ''),
        )


class _SharedLLMImageThrottle:
    """Reserve request starts across all short-lived image requesters.

    The condition is held only while inspecting or updating timestamps; HTTP
    work never runs under this lock.

    >>> len(_SharedLLMImageThrottle()._request_times)
    0
    """

    WINDOW_SECONDS = 60.1
    WAIT_SLICE_SECONDS = 0.05

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._request_times: deque[float] = deque()
        self._last_request_time: Optional[float] = None
        self._next_allowed_time = 0.0

    def reserve(
        self,
        *,
        delay: float,
        max_requests_per_minute: int,
        stop_event: Optional[threading.Event],
    ) -> None:
        """Wait cooperatively, then reserve one global request-start slot."""
        delay = max(0.0, float(delay))
        rpm = max(0, int(max_requests_per_minute))
        while True:
            if stop_event is not None and stop_event.is_set():
                raise LLMRequestStopped()
            with self._condition:
                now = time.monotonic()
                cutoff = now - self.WINDOW_SECONDS
                while (
                    self._request_times
                    and self._request_times[0] <= cutoff
                ):
                    self._request_times.popleft()
                wait_until = self._next_allowed_time
                if self._last_request_time is not None:
                    wait_until = max(
                        wait_until,
                        self._last_request_time + delay,
                    )
                if rpm > 0 and len(self._request_times) >= rpm:
                    wait_until = max(
                        wait_until,
                        self._request_times[-rpm] + self.WINDOW_SECONDS,
                    )
                wait_time = wait_until - now
                if wait_time <= 0:
                    self._request_times.append(now)
                    self._last_request_time = now
                    self._next_allowed_time = now + delay
                    self._condition.notify_all()
                    return
                self._condition.wait(
                    min(wait_time, self.WAIT_SLICE_SECONDS)
                )


_LLM_IMAGE_THROTTLE = _SharedLLMImageThrottle()


class LLMImageRequester:
    """Issue optional-context image requests through one LLM profile.

    ``LLMInpaint`` inherits this transport and supplies parameters through its
    normal module API. Other callers pass an immutable policy snapshot.

    >>> LLMImageRequester._generation_url(
    ...     'https://api.example/v1/images/edits'
    ... )
    'https://api.example/v1/images/generations'
    """

    def __init__(
        self,
        *args,
        image_request_policy: Optional[LLMImageRequestPolicy] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._image_request_policy = image_request_policy
        self.client = None
        self.client_cache_key = None
        self.stop_event = None

    def _request_param(self, name: str):
        policy = self._image_request_policy
        if policy is None:
            return self.get_param_value(name)
        return getattr(policy, name.replace(' ', '_'))

    @staticmethod
    def _image_model_required(
        model: str, model_options: Sequence[str]
    ) -> str:
        model = str(model or '').strip()
        options = [
            str(option).strip()
            for option in model_options
            if str(option).strip()
        ]
        if not model or not options:
            return ''
        return model

    @classmethod
    def _image_model(
        cls, profile: LLMProfile, model: Optional[str] = None
    ) -> str:
        if model is None:
            resolved = cls._image_model_required(
                profile.image_model, profile.image_model_options
            )
        else:
            resolved = str(model or '').strip()
        if not resolved:
            raise LLMModelRequiredError(
                profile.id, profile.name, target='image_model'
            )
        return resolved

    @staticmethod
    def _image_base_url(profile: LLMProfile) -> str:
        base_url = str(profile.image_base_url or '').strip()
        if not base_url:
            raise LLMBaseURLRequiredError(
                profile.id, profile.name, target='image_base_url'
            )
        return base_url

    def set_stop_event(self, stop_event) -> None:
        self.stop_event = stop_event

    def _wait(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.stop_event is not None:
            if self.stop_event.wait(seconds):
                raise LLMRequestStopped()
            return
        time.sleep(seconds)

    def _request_timeout(self):
        try:
            timeout = float(self._request_param('request timeout') or 0)
        except (TypeError, ValueError):
            timeout = 180.0
        return None if timeout <= 0 else timeout

    def _max_resolution(self) -> int:
        try:
            return int(self._request_param('max resolution') or 0)
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
        size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def _http_client(self, proxy: str):
        import httpx  # type: ignore

        client_kwargs = {'timeout': self._request_timeout()}
        if not proxy:
            return httpx.Client(**client_kwargs)
        try:
            mounts = {
                'http://': httpx.HTTPTransport(proxy=proxy),
                'https://': httpx.HTTPTransport(proxy=proxy),
            }
            return httpx.Client(mounts=mounts, **client_kwargs)
        except Exception as error:
            LOGGER.error(
                "Failed to initialize proxy '%s': %s. Proceeding without proxy.",
                proxy,
                error,
            )
            return httpx.Client(**client_kwargs)

    @staticmethod
    def _api_key_for_profile(profile: LLMProfile) -> str:
        api_key = resolve_api_key(profile).strip()
        if profile.require_api_key and not api_key:
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        return api_key

    def _initialize_client(self, profile: LLMProfile):
        api_key = self._api_key_for_profile(profile)
        base_url = self._image_base_url(profile)
        proxy = self._request_param('proxy') or ''
        cache_key = (api_key, base_url, proxy, self._request_timeout())
        if self.client is not None and self.client_cache_key == cache_key:
            return self.client
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = self._http_client(proxy)
        self.client_cache_key = cache_key
        return self.client

    def close(self) -> None:
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
                self.client_cache_key = None

    def _respect_delay(self) -> None:
        _LLM_IMAGE_THROTTLE.reserve(
            delay=float(self._request_param('delay') or 0),
            max_requests_per_minute=int(
                self._request_param('max requests per minute') or 0
            ),
            stop_event=self.stop_event,
        )

    @staticmethod
    def _response_error_message(response) -> str:
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get('error')
                if isinstance(error, dict) and error.get('message'):
                    return str(error['message'])
                for key in ('message', 'detail'):
                    if data.get(key):
                        return str(data[key])
        except Exception:
            pass
        text = getattr(response, 'text', '')
        if text:
            return str(text)
        status = getattr(response, 'status_code', '')
        reason = getattr(response, 'reason_phrase', '')
        return f'HTTP {status} {reason}'.strip()

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        base = base_url.rstrip('/')
        endpoint = '/' + path.strip('/')
        if urlparse(base).path.rstrip('/').endswith(endpoint):
            return base
        return f'{base}{endpoint}'

    @staticmethod
    def _is_openrouter_url(base_url: str) -> bool:
        host = urlparse(base_url).netloc.lower()
        return host == 'openrouter.ai' or host.endswith('.openrouter.ai')

    @staticmethod
    def _is_gemini_url(base_url: str) -> bool:
        return (
            urlparse(base_url).netloc.lower()
            == 'generativelanguage.googleapis.com'
        )

    @classmethod
    def _gemini_generate_content_url(
        cls, base_url: str, model: str
    ) -> str:
        base = base_url.rstrip('/')
        parsed = urlparse(base)
        path = parsed.path.rstrip('/')
        if path.endswith(':generateContent'):
            return base
        if path.endswith('/openai'):
            path = path[:-len('/openai')]
            base = urlunparse(
                parsed._replace(path=path, params='', query='', fragment='')
            ).rstrip('/')
        model_path = model if model.startswith('models/') else f'models/{model}'
        return cls._join_url(base, f'/{model_path}:generateContent')

    @staticmethod
    def _generation_url(base_url: str) -> str:
        parsed = urlparse(base_url)
        path = parsed.path.rstrip('/')
        if path.endswith('/images/edits'):
            path = path[:-len('/images/edits')] + '/images/generations'
            return urlunparse(parsed._replace(path=path))
        return base_url

    @staticmethod
    def _png_image_file(img: np.ndarray) -> io.BytesIO:
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise RuntimeError('LLM image requests require an RGB(A) image.')
        buffer = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(img)).save(buffer, format='PNG')
        buffer.seek(0)
        buffer.name = 'image.png'
        return buffer

    def _api_args(
        self,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        result = {
            'model': self._image_model(profile, model),
            'prompt': prompt if prompt is not None else profile.image_prompt,
        }
        if image_file is not None:
            result['image'] = image_file
        return result

    def _openrouter_api_args(
        self,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        result = {
            'model': self._image_model(profile, model),
            'prompt': prompt if prompt is not None else profile.image_prompt,
            'output_format': 'png',
            'n': 1,
        }
        if image_file is not None:
            encoded = base64.b64encode(image_file.getvalue()).decode('ascii')
            result['input_references'] = [{
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/png;base64,{encoded}',
                },
            }]
        return result

    def _gemini_api_args(
        self,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Dict:
        del model
        parts = [{
            'text': prompt if prompt is not None else profile.image_prompt,
        }]
        if image_file is not None:
            encoded = base64.b64encode(image_file.getvalue()).decode('ascii')
            parts.append({
                'inline_data': {
                    'mime_type': 'image/png',
                    'data': encoded,
                },
            })
        return {
            'contents': [{'parts': parts}],
            'generationConfig': {'responseModalities': ['IMAGE']},
        }

    @staticmethod
    def _response_field(item, field_name: str):
        if isinstance(item, dict):
            return item.get(field_name)
        return getattr(item, field_name, None)

    @staticmethod
    def _decode_image_bytes(raw: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(raw)) as image:
            has_alpha = 'A' in image.getbands() or 'transparency' in image.info
            return np.array(image.convert('RGBA' if has_alpha else 'RGB'))

    def _download_image(self, url: str) -> np.ndarray:
        client = self._http_client(self._request_param('proxy') or '')
        try:
            response = client.get(url)
            response.raise_for_status()
            return self._decode_image_bytes(response.content)
        finally:
            client.close()

    def _decode_response_image(self, response) -> np.ndarray:
        data = self._response_field(response, 'data')
        if not data:
            raise RuntimeError('LLM image request returned no image data.')
        item = data[0]
        encoded = self._response_field(item, 'b64_json')
        if encoded:
            return self._decode_image_bytes(base64.b64decode(encoded))
        url = self._response_field(item, 'url')
        if url:
            return self._download_image(str(url))
        raise RuntimeError('LLM image request returned no decodable image.')

    def _decode_gemini_response_image(self, response) -> np.ndarray:
        candidates = self._response_field(response, 'candidates') or []
        for candidate in candidates:
            content = self._response_field(candidate, 'content') or {}
            for part in self._response_field(content, 'parts') or []:
                inline = (
                    self._response_field(part, 'inline_data')
                    or self._response_field(part, 'inlineData')
                )
                data = self._response_field(inline, 'data') if inline else None
                if data:
                    return self._decode_image_bytes(base64.b64decode(str(data)))
        output = (
            self._response_field(response, 'output_image')
            or self._response_field(response, 'outputImage')
        )
        data = self._response_field(output, 'data') if output else None
        if data:
            return self._decode_image_bytes(base64.b64decode(str(data)))
        for step in self._response_field(response, 'steps') or []:
            if self._response_field(step, 'type') != 'model_output':
                continue
            for block in self._response_field(step, 'content') or []:
                if self._response_field(block, 'type') == 'image':
                    data = self._response_field(block, 'data')
                    if data:
                        return self._decode_image_bytes(
                            base64.b64decode(str(data))
                        )
        raise RuntimeError('Gemini image request returned no decodable image.')

    @staticmethod
    def _headers(api_key: str, json_request: bool = False) -> Dict:
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

    def _raise_for_response(self, profile: LLMProfile, response) -> None:
        status_code = getattr(response, 'status_code', 200)
        if status_code < 400:
            return
        if status_code in (401, 403):
            raise LLMApiKeyRequiredError(profile.id, profile.name)
        raise RuntimeError(self._response_error_message(response))

    def _request_openrouter_image(
        self,
        client,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        response = client.post(
            self._join_url(self._image_base_url(profile), '/images'),
            headers=self._headers(
                self._api_key_for_profile(profile), json_request=True
            ),
            json=self._openrouter_api_args(
                profile, image_file, prompt=prompt, model=model
            ),
        )
        self._raise_for_response(profile, response)
        return self._decode_response_image(response.json())

    def _request_gemini_image(
        self,
        client,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        selected_model = self._image_model(profile, model)
        response = client.post(
            self._gemini_generate_content_url(
                self._image_base_url(profile), selected_model
            ),
            headers=self._gemini_headers(
                self._api_key_for_profile(profile)
            ),
            json=self._gemini_api_args(
                profile, image_file, prompt=prompt, model=selected_model
            ),
        )
        self._raise_for_response(profile, response)
        return self._decode_gemini_response_image(response.json())

    def _request_openai_compatible_image(
        self,
        client,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        args = self._api_args(
            profile, image_file, prompt=prompt, model=model
        )
        base_url = self._image_base_url(profile)
        headers = self._headers(self._api_key_for_profile(profile))
        if image_file is None:
            response = client.post(
                self._generation_url(base_url),
                headers=self._headers(
                    self._api_key_for_profile(profile), json_request=True
                ),
                json={'model': args['model'], 'prompt': args['prompt']},
            )
        else:
            response = client.post(
                base_url,
                headers=headers,
                data={'model': args['model'], 'prompt': args['prompt']},
                files={
                    'image': (
                        'image.png', image_file.getvalue(), 'image/png'
                    ),
                },
            )
        self._raise_for_response(profile, response)
        return self._decode_response_image(response.json())

    def request_image(
        self,
        profile: LLMProfile,
        image: Optional[np.ndarray],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        *,
        resize_to_input: bool = False,
    ) -> np.ndarray:
        """Return one generated RGB(A) image for optional input context."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()
        client = self._initialize_client(profile)
        original_shape = None if image is None else image.shape[:2]
        request_image = (
            None if image is None else self._scale_image_for_request(image)
        )
        image_file = (
            None
            if request_image is None
            else self._png_image_file(request_image)
        )
        try:
            self._respect_delay()
            if self.stop_event is not None and self.stop_event.is_set():
                # A reserved slot remains counted, but Stop must still win
                # before the synchronous provider call begins.
                raise LLMRequestStopped()
            base_url = self._image_base_url(profile)
            if self._is_gemini_url(base_url):
                result = self._request_gemini_image(
                    client, profile, image_file, prompt=prompt, model=model
                )
            elif self._is_openrouter_url(base_url):
                result = self._request_openrouter_image(
                    client, profile, image_file, prompt=prompt, model=model
                )
            else:
                result = self._request_openai_compatible_image(
                    client, profile, image_file, prompt=prompt, model=model
                )
            if (
                resize_to_input
                and original_shape is not None
                and result.shape[:2] != original_shape
            ):
                result = cv2.resize(
                    result,
                    (original_shape[1], original_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            return np.ascontiguousarray(result.astype(np.uint8, copy=False))
        finally:
            if image_file is not None:
                image_file.close()

    def _request_inpaint(
        self,
        profile: LLMProfile,
        img: np.ndarray,
        prompt: Optional[str] = None,
    ) -> np.ndarray:
        result = self.request_image(
            profile, img, prompt=prompt, resize_to_input=True
        )
        channels = img.shape[2]
        if result.shape[2] != channels:
            if channels == 3 and result.shape[2] == 4:
                result = result[:, :, :3]
            elif channels == 4 and result.shape[2] == 3:
                alpha = np.full(result.shape[:2] + (1,), 255, dtype=np.uint8)
                result = np.concatenate((result, alpha), axis=2)
            else:
                raise RuntimeError('LLM image response channel count changed.')
        return np.ascontiguousarray(result)

    def request_image_with_retries(
        self,
        profile: LLMProfile,
        image: Optional[np.ndarray],
        prompt: str,
        model: str,
    ) -> np.ndarray:
        attempts = max(1, int(self._request_param('retry attempts') or 1))
        for attempt in range(attempts):
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            try:
                return self.request_image(
                    profile,
                    image,
                    prompt=prompt,
                    model=model,
                    resize_to_input=image is not None,
                )
            except (LLMUserActionRequiredError, LLMRequestStopped):
                raise
            except Exception as error:
                if attempt + 1 >= attempts:
                    raise RuntimeError(
                        f'LLM image generation failed: {error}'
                    ) from error
                LOGGER.warning(
                    'LLM image generation failed due to %s. Attempt: %s',
                    error,
                    attempt + 1,
                )
                self._wait(float(self._request_param('retry timeout') or 0))
        raise AssertionError('unreachable image request retry state')
