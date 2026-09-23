"""Reusable profile-backed LLM image request transport."""

import base64
import io
import json
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import islice
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
from ballontranslator.utils.llm_profiles import (
    LLMProfile, image_responses_url, resolve_api_key, split_image_model_selection,
)
from ballontranslator.utils.logger import logger as LOGGER
from . import image_generation


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
        self._image_cache_keys: Dict[tuple, str] = {}

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

    def set_stop_event(self, stop_event: Optional[threading.Event]) -> None:
        if stop_event is not self.stop_event:
            self._image_cache_keys.clear()
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
        if profile.backend != 'openai':
            raise LLMUserActionRequiredError('Image editing is unavailable for this LLM profile backend.')
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
    def _diagnostic_url(value: str) -> str:
        """Keep endpoint identity without credentials or signed query values.

        >>> LLMImageRequester._diagnostic_url('https://user:secret@example/v1?token=secret')
        'https://example/v1'
        """
        try:
            parsed = urlparse(value)
            return urlunparse(parsed._replace(netloc=parsed.netloc.rsplit('@', 1)[-1],
                                             params='', query='', fragment=''))
        except ValueError:
            return '<invalid URL>'

    @classmethod
    def _diagnostic_text(cls, value: str, secrets: Sequence[Optional[str]]) -> str:
        for secret in secrets:
            if isinstance(secret, str) and secret:
                # Error-body excerpts may contain a JSON-escaped copy of input.
                for encoded in (secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]):
                    value = value.replace(encoded, '<redacted>')
        value = re.sub(r'https?://[^\s<>"\']+', lambda match: cls._diagnostic_url(match[0]), value)
        value = re.sub(r'data:[^,\s]*;base64,[A-Za-z0-9+/=]+|[A-Za-z0-9+/=]{80,}', '<image data omitted>', value)
        return ' '.join(value.split())[:2048]

    def _log_image_response_failure(
        self, profile: LLMProfile, response, endpoint: str,
        secrets: Sequence[Optional[str]], data: object = None,
        response_body: Optional[str] = None,
    ) -> None:
        def preview(value: object, depth: int = 0) -> object:
            if depth > 4:
                return '<omitted>'
            if isinstance(value, dict):
                return {str(key): ('<omitted>' if str(key).lower() in (
                    'b64_json', 'image', 'images', 'mask', 'image_url', 'input',
                    'input_references', 'prompt', 'revised_prompt', 'headers',
                    'authorization', 'api_key', 'access_token', 'refresh_token',
                    'result',
                ) or (key == 'data' and isinstance(item, str)) else preview(item, depth + 1))
                        for key, item in islice(value.items(), 12)}
            if isinstance(value, list):
                return [preview(item, depth + 1) for item in value[:3]]
            if isinstance(value, str):
                return self._diagnostic_text(value, secrets)[:512]
            return value

        headers = getattr(response, 'headers', {})
        content_type = str(headers.get('content-type', ''))
        body = (json.dumps(preview(data), ensure_ascii=False) if data is not None
                else '<image body omitted>' if content_type.lower().startswith('image/')
                else response_body if response_body is not None
                else str(getattr(response, 'text', '')))
        try:
            endpoint = str(response.url)
        except (AttributeError, RuntimeError):
            pass
        LOGGER.warning(
            'LLM image response failed: profile_id=%r, profile_name=%r, endpoint=%r, '
            'status=%s, content_type=%r, location=%r, response=%s',
            self._diagnostic_text(str(profile.id), secrets),
            self._diagnostic_text(str(profile.name), secrets),
            self._diagnostic_text(self._diagnostic_url(endpoint), secrets),
            getattr(response, 'status_code', ''), self._diagnostic_text(content_type, secrets)[:128],
            self._diagnostic_text(self._diagnostic_url(str(headers.get('location', ''))), secrets),
            self._diagnostic_text(body, secrets) or '<empty>',
        )

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

    def _decode_api_image_response(
        self, profile: LLMProfile, response, endpoint: str,
        prompt: Optional[str], *, gemini: bool = False, assisted: bool = False,
    ) -> np.ndarray:
        """Report failed provider responses without logging successful image bodies.

        >>> image = requester._decode_api_image_response(profile, response, endpoint, None)  # doctest: +SKIP
        """
        data = None
        body = bytearray() if assisted else None
        incomplete_body = False
        try:
            api_key = resolve_api_key(profile)
        except Exception:
            api_key = profile.api_key
        secrets = (api_key, profile.image_prompt, prompt)
        try:
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            status = response.status_code
            if assisted:
                limit = 16 * 1024 if status >= 300 else image_generation.MAX_IMAGE_RESPONSE_BYTES
                try:
                    for chunk in response.iter_bytes():
                        if self.stop_event is not None and self.stop_event.is_set():
                            raise LLMRequestStopped()
                        received = len(body) + len(chunk)
                        if received > limit or (status >= 300 and received == limit):
                            incomplete_body = True
                            if status >= 300:
                                body.extend(chunk[:limit - len(body)])
                                break
                            raise LLMUserActionRequiredError('The image provider returned an oversized response.')
                        body.extend(chunk)
                except LLMRequestStopped:
                    raise
                except Exception:
                    # Error diagnostics are best effort: an interrupted error
                    # body must not turn known auth/permission failures into retries.
                    incomplete_body = True
                    if status < 300:
                        raise
                if self.stop_event is not None and self.stop_event.is_set():
                    raise LLMRequestStopped()
            if 300 <= status < 400:
                location = self._diagnostic_text(self._diagnostic_url(
                    str(getattr(response, 'headers', {}).get('location', ''))), secrets)
                raise LLMUserActionRequiredError(
                    f'Image endpoint returned HTTP {status} redirect'
                    + (f' to {location}' if location else '')
                    + '. Set the image Base URL to the provider’s image endpoint.'
                )
            try:
                data = json.loads(body) if assisted else response.json()
            except ValueError:
                if status < 400:
                    raise LLMUserActionRequiredError(
                        ('The image provider returned an invalid Responses result. ' if assisted
                         else f'Image endpoint returned HTTP {status} with an empty or non-JSON response. ')
                        + 'Check the image Base URL; it must be the provider’s image endpoint.'
                    ) from None
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            error_data = data
            if assisted and isinstance(data, dict) and data.get('type') == 'response.failed':
                error_data = data.get('response', data)
            error = error_data.get('error', error_data) if isinstance(error_data, dict) else error_data
            auth_error = isinstance(error, dict) and (
                str(error.get('code') or error.get('type') or '').lower() in (
                    'invalid_api_key', 'incorrect_api_key', 'missing_api_key',
                    'expired_api_key', 'invalid_token', 'authentication_error',
                )
            )
            if status == 401 or auth_error:
                raise LLMApiKeyRequiredError(profile.id, profile.name)
            if status >= 400 or (isinstance(error_data, dict) and error_data.get('error')):
                message = next((error[key] for key in ('message', 'detail', 'code')
                                if isinstance(error.get(key), str) and error[key]), '') if isinstance(error, dict) else error
                if not isinstance(message, str) or not message:
                    # Structured validation errors may echo inputs. Their safe
                    # preview is logged separately, never stringify them for retries.
                    message = (body.decode('utf-8', errors='replace') if assisted
                               else str(getattr(response, 'text', ''))) if data is None and not incomplete_body else ''
                message = message or f'Image service returned an error (HTTP {status}).'
                message = self._diagnostic_text(message, secrets)
                if status == 403 or (assisted and status < 500 and status not in (408, 429)):
                    raise LLMUserActionRequiredError(
                        f'Image service rejected this request (HTTP {status}): {message} '
                        'Check the selected model, endpoint, and account access.'
                    )
                raise RuntimeError(f'Image service returned HTTP {status}: {message}')
            if assisted:
                raw = image_generation.decode_responses_image(data)
                if self.stop_event is not None and self.stop_event.is_set():
                    raise LLMRequestStopped()
                try:
                    return self._decode_image_bytes(raw)
                except (OSError, ValueError) as error:
                    raise LLMUserActionRequiredError('The image provider returned an invalid image.') from error
            return (self._decode_gemini_response_image(data) if gemini
                    else self._decode_response_image(data))
        except LLMRequestStopped:
            raise
        except Exception as error:
            try:
                self._log_image_response_failure(
                    profile, response, endpoint, secrets, data,
                    '<incomplete response body omitted>' if incomplete_body else
                    body.decode('utf-8', errors='replace') if body is not None and data is None else None,
                )
            except Exception:
                # Diagnostics must not replace the response or authentication error.
                pass
            # Download failures can include signed URLs in their exception text;
            # outer retry owners must not re-log their query credentials.
            if not isinstance(error, LLMUserActionRequiredError):
                message = self._diagnostic_text(str(error), secrets)
                if message != str(error):
                    raise RuntimeError(message) from None
            raise

    def _request_openrouter_image(
        self,
        client,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        endpoint = self._join_url(self._image_base_url(profile), '/images')
        response = client.post(
            endpoint,
            headers=self._headers(
                self._api_key_for_profile(profile), json_request=True
            ),
            json=self._openrouter_api_args(
                profile, image_file, prompt=prompt, model=model
            ),
        )
        return self._decode_api_image_response(profile, response, endpoint, prompt)

    def _request_gemini_image(
        self,
        client,
        profile: LLMProfile,
        image_file: Optional[io.BytesIO],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> np.ndarray:
        selected_model = self._image_model(profile, model)
        endpoint = self._gemini_generate_content_url(self._image_base_url(profile), selected_model)
        response = client.post(
            endpoint,
            headers=self._gemini_headers(
                self._api_key_for_profile(profile)
            ),
            json=self._gemini_api_args(
                profile, image_file, prompt=prompt, model=selected_model
            ),
        )
        return self._decode_api_image_response(profile, response, endpoint, prompt, gemini=True)

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
        endpoint = self._generation_url(base_url) if image_file is None else base_url
        if image_file is None:
            response = client.post(
                endpoint,
                headers=self._headers(
                    self._api_key_for_profile(profile), json_request=True
                ),
                json={'model': args['model'], 'prompt': args['prompt']},
            )
        else:
            response = client.post(
                endpoint,
                headers=headers,
                data={'model': args['model'], 'prompt': args['prompt']},
                files={
                    'image': (
                        'image.png', image_file.getvalue(), 'image/png'
                    ),
                },
            )
        return self._decode_api_image_response(profile, response, endpoint, prompt)

    def _image_cache_key(self, identity: tuple) -> str:
        # Reuse the job's identity across crops and retries, without chat history.
        if identity not in self._image_cache_keys:
            self._image_cache_keys[identity] = str(uuid.uuid4())
        return self._image_cache_keys[identity]

    def _image_reference_bytes(
        self, image: Optional[np.ndarray], mask: Optional[np.ndarray], prompt: str,
    ) -> tuple[Optional[bytes], Optional[bytes], str]:
        image_bytes = mask_bytes = None
        if image is not None:
            with self._png_image_file(image) as image_file:
                image_bytes = image_file.getvalue()
        if mask is not None:
            # References describe the editable area; local compositing enforces it.
            mask_rgb = np.repeat((mask > 127)[..., None], 3, axis=2).astype(np.uint8) * 255
            with self._png_image_file(mask_rgb) as mask_file:
                mask_bytes = mask_file.getvalue()
            prompt += (
                '\nImage 1 is the source image. Image 2 is a mask: white marks '
                'the editable region and black marks pixels to preserve. '
                'Apply the requested cleanup only inside the white region. '
                'Return only the edited image 1, preserving its framing, '
                'aspect ratio, artwork, and positions. Do not include the mask.'
            )
        return image_bytes, mask_bytes, prompt

    def _request_assisted_image(
        self, client, profile: LLMProfile, endpoint: str,
        image: Optional[np.ndarray], mask: Optional[np.ndarray],
        prompt: Optional[str], model: str, reasoning_model: str,
    ) -> np.ndarray:
        """Read one bounded Responses result using the profile's API client.

        >>> pixels = requester._request_assisted_image(client, profile, endpoint, None, None, 'Draw.', 'gpt-image-2', 'gpt-6-sol')  # doctest: +SKIP
        """
        image_bytes, mask_bytes, instructions = self._image_reference_bytes(
            image, mask, profile.image_prompt if prompt is None else prompt,
        )
        payload = image_generation.responses_image_payload(
            reasoning_model, model, instructions,
            image_generation.image_references(image_bytes, mask_bytes),
            self._image_cache_key((profile.id, reasoning_model, self.client_cache_key)),
            stream=False,
        )
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()
        with client.stream('POST', endpoint, headers=self._headers(
            self._api_key_for_profile(profile), json_request=True,
        ), json=payload) as response:
            return self._decode_api_image_response(profile, response, endpoint, prompt, assisted=True)

    def _request_codex_image(
        self,
        profile: LLMProfile,
        image: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        prompt: Optional[str],
        model: str,
        reasoning_model: str,
    ) -> np.ndarray:
        """Send source and mask references through subscription authentication.

        >>> pixels = requester._request_codex_image(profile, crop, mask, None, 'gpt-image-2', '')  # doctest: +SKIP
        """
        from .codex import account, request_image

        image_bytes, mask_bytes, instructions = self._image_reference_bytes(
            image, mask, profile.image_prompt if prompt is None else prompt,
        )
        cache_key = (self._image_cache_key((profile.id, reasoning_model, account.generation))
                     if reasoning_model else '')
        raw = request_image(
            model, instructions, image_bytes, mask_bytes,
            self.stop_event, proxy=self._request_param('proxy') or '',
            timeout=self._request_timeout(),
            reasoning_model=reasoning_model,
            cache_key=cache_key,
        )
        return self._decode_image_bytes(raw)

    def request_image(
        self,
        profile: LLMProfile,
        image: Optional[np.ndarray],
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        *,
        resize_to_input: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return one generated RGB(A) image for optional input context."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise LLMRequestStopped()
        if mask is not None and (image is None or mask.shape != image.shape[:2]):
            raise ValueError('The inpaint mask must match the input image dimensions.')
        if profile.backend == 'codex':
            from .codex import account
            account.require_sign_in(self.stop_event)
        try:
            reasoning_model, selected_model = split_image_model_selection(self._image_model(profile, model))
        except ValueError as error:
            raise LLMUserActionRequiredError(str(error)) from error
        instructions = profile.image_prompt if prompt is None else prompt
        if (reasoning_model or profile.backend == 'codex') and (
            not isinstance(instructions, str) or not instructions.strip()
        ):
            raise LLMUserActionRequiredError('Enter an image prompt before requesting an image.')
        responses_url = image_responses_url(profile) if reasoning_model else ''
        if reasoning_model and profile.backend != 'codex' and not responses_url:
            raise LLMUserActionRequiredError(
                'Assisted image editing needs an API base URL (root or /v1), '
                'or an endpoint ending in /images/edits, /images/generations, '
                'or /responses on a service '
                'that supports the Responses image tool. Update the endpoint '
                'or select a direct image model.'
            )
        client = None if profile.backend == 'codex' else self._initialize_client(profile)
        original_shape = None if image is None else image.shape[:2]
        request_image = (
            None if image is None else self._scale_image_for_request(image)
        )
        if mask is not None and (profile.backend == 'codex' or reasoning_model) and mask.shape != request_image.shape[:2]:
            # Area coverage keeps thin marks that nearest sampling loses.
            mask = cv2.resize((mask > 127).astype(np.float32),
                              (request_image.shape[1], request_image.shape[0]),
                              interpolation=cv2.INTER_AREA)
            mask = (mask > 0).astype(np.uint8) * 255
        padding = (0, 0, 0, 0)
        if request_image is not None and (profile.backend == 'codex' or reasoning_model):
            height, width = request_image.shape[:2]
            # Keep extreme crops within 3:1 for these image routes. Padding
            # preserves source geometry; strip it before final resizing.
            extra_h = max(0, (width + 2) // 3 - height)
            extra_w = max(0, (height + 2) // 3 - width)
            padding = (extra_h // 2, extra_h - extra_h // 2,
                       extra_w // 2, extra_w - extra_w // 2)
            if any(padding):
                request_image = cv2.copyMakeBorder(request_image, *padding, cv2.BORDER_REPLICATE)
                if mask is not None:
                    mask = cv2.copyMakeBorder(mask, *padding, cv2.BORDER_CONSTANT, value=0)
        image_file = (
            None
            if request_image is None or profile.backend == 'codex' or reasoning_model
            else self._png_image_file(request_image)
        )
        try:
            self._respect_delay()
            if self.stop_event is not None and self.stop_event.is_set():
                # A reserved slot remains counted, but Stop must still win
                # before the synchronous provider call begins.
                raise LLMRequestStopped()
            base_url = '' if profile.backend == 'codex' else self._image_base_url(profile)
            if profile.backend == 'codex':
                result = self._request_codex_image(
                    profile, request_image, mask, prompt, selected_model, reasoning_model
                )
            elif reasoning_model:
                result = self._request_assisted_image(
                    client, profile, responses_url, request_image, mask, prompt,
                    selected_model, reasoning_model,
                )
            elif self._is_gemini_url(base_url):
                result = self._request_gemini_image(
                    client, profile, image_file, prompt=prompt, model=selected_model
                )
            elif self._is_openrouter_url(base_url):
                result = self._request_openrouter_image(
                    client, profile, image_file, prompt=prompt, model=selected_model
                )
            else:
                result = self._request_openai_compatible_image(
                    client, profile, image_file, prompt=prompt, model=selected_model
                )
            if self.stop_event is not None and self.stop_event.is_set():
                raise LLMRequestStopped()
            if any(padding):
                result = cv2.resize(result, (request_image.shape[1], request_image.shape[0]), interpolation=cv2.INTER_LINEAR)
                top, _, left, _ = padding
                result = result[top:top + height, left:left + width]
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
        *,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        result = self.request_image(
            profile, img, prompt=prompt, resize_to_input=True, mask=mask
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
