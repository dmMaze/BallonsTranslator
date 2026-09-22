"""Stateless ChatGPT/Codex HTTP transport and app-owned OAuth credentials."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlsplit

from .context.errors import ContextLengthError, is_context_length_error
from .exceptions import CodexSignInRequiredError, LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.utils.llm_profiles import CODEX_IMAGE_MODEL, LLMProfile, THINKING_AUTO, THINKING_DISABLED, normalize_codex_models
from ballontranslator.utils.logger import logger as LOGGER

if TYPE_CHECKING:
    import httpx
    from keyring.backend import KeyringBackend
    from .llm_chat import LLMChatResult


_KEYRING_SERVICE = 'BallonsTranslator Codex'
# Backend catalog compatibility, independent of any installed CLI or SDK.
CATALOG_VERSION = '0.155.1'
API_URL = 'https://chatgpt.com/backend-api/codex'
AUTH_URL = 'https://auth.openai.com'
CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
# Match the native Codex image tool's output bound; allow JSON envelope overhead.
_MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_IMAGE_BASE64_BYTES = ((_MAX_IMAGE_BYTES + 2) // 3) * 4
_MAX_IMAGE_RESPONSE_BYTES = _MAX_IMAGE_BASE64_BYTES + 1024 * 1024


def _system_keyring() -> KeyringBackend:
    """Select a native vault without letting a chainer fall through to files.

    >>> vault = _system_keyring()  # doctest: +SKIP
    """
    import keyring
    from keyring.backends.chainer import ChainerBackend

    selected = keyring.get_keyring()
    candidates = selected.backends if type(selected) is ChainerBackend else (selected,)
    for backend in candidates:
        if (type(backend).__module__, type(backend).__name__) in {
            ('keyring.backends.Windows', 'WinVaultKeyring'),
            ('keyring.backends.macOS', 'Keyring'),
            ('keyring.backends.SecretService', 'Keyring'),
        }:
            return backend
    raise RuntimeError('No supported system credential store is available.')


def _http_client(proxy: str = '') -> httpx.AsyncClient:
    import httpx

    kwargs = {'timeout': httpx.Timeout(300.0, connect=10.0), 'follow_redirects': False}
    if proxy:
        # Environment proxy mounts otherwise outrank all://. The explicit
        # transport still honors environment certificate configuration.
        kwargs['trust_env'] = False
        kwargs['mounts'] = {'all://': httpx.AsyncHTTPTransport(proxy=proxy)}
    return httpx.AsyncClient(**kwargs)


def _run(operation: Coroutine, stop_event: Optional[threading.Event]) -> Any:
    """Run HTTP work in the calling worker, cancelling socket waits promptly.

    >>> _run(asyncio.sleep(0, result='completed'), None)
    'completed'
    """
    generation = account.generation

    async def run() -> Any:
        task = asyncio.create_task(operation)
        try:
            while True:
                if (stop_event is not None and stop_event.is_set()) or generation != account.generation:
                    raise LLMRequestStopped()
                done, _ = await asyncio.wait((task,), timeout=0.05)
                if done:
                    if (stop_event is not None and stop_event.is_set()) or generation != account.generation:
                        raise LLMRequestStopped()
                    return task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        return asyncio.run(run())
    except CodexSignInRequiredError as error:
        # Do not change generation here: the originating auth error must reach
        # the UI. A late failure must not invalidate a newly committed account.
        with account._state_lock:
            if error.invalid and generation == account.generation and not account.changing:
                account.auth_invalid = True
        raise
    except (LLMRequestStopped, LLMUserActionRequiredError, ContextLengthError):
        raise
    except Exception:
        # OAuth codes, tokens and request URLs must not reach provider tracebacks.
        raise RuntimeError('Codex could not complete the request. Check the connection and retry.') from None


def _raise_service_error(payload: Dict, status: Optional[int] = None) -> None:
    error = payload.get('error') or payload
    message = str(error.get('message', '')) if isinstance(error, dict) else str(payload.get('error_description', ''))
    if status is None:
        status = payload.get('status_code') or (error.get('status_code') if isinstance(error, dict) else None)
    if status == 401:
        raise CodexSignInRequiredError(invalid=True) from None
    provider_error = RuntimeError(message)
    provider_error.body = payload
    provider_error.status_code = status
    if is_context_length_error(provider_error):
        raise ContextLengthError('Codex input exceeds the model context window.') from None
    code = str(error.get('code', error.get('type', ''))) if isinstance(error, dict) else str(error)
    detail = (code + ' ' + message).lower()
    auth_codes = {
        'authentication_error', 'authentication_required', 'invalid_authentication',
        'unauthorized', 'invalid_token', 'token_expired', 'token_revoked',
        'expired_token', 'invalid_access_token', 'access_token_expired', 'invalid_api_key',
        'invalid_grant', 'invalid_refresh_token', 'refresh_token_invalid',
        'refresh_token_expired', 'refresh_token_reused', 'refresh_token_revoked', 'refresh_token_invalidated',
        'session_expired',
    }
    codes = {str(error.get(key, '')).lower() for key in ('code', 'type')} if isinstance(error, dict) else {code.lower()}
    refresh_message = message.lower()
    invalid_refresh = ('refresh token' in refresh_message or 'refresh_token' in refresh_message) and any(
        word in refresh_message for word in ('expired', 'invalid', 'revoked', 'reused', 'already been used')
    )
    auth_message = any(term in detail for term in (
        'authentication failed', 'authentication required', 'invalid authentication token',
        'invalid access token', 'invalid bearer token', 'expired access token',
        'token has expired', 'token is expired', 'token expired',
    ))
    if codes & auth_codes or invalid_refresh or auth_message:
        raise CodexSignInRequiredError(invalid=True) from None
    if any(word in detail for word in ('quota', 'usage limit', 'usage_limit', 'insufficient credit', 'billing')):
        raise LLMUserActionRequiredError('The ChatGPT account has reached its Codex usage limit. Check the account limits before retrying.') from None
    if 'model' in detail and any(word in detail for word in ('not found', 'not supported', 'unavailable', 'does not exist', 'model_not_found')):
        raise LLMUserActionRequiredError('This Codex model is unavailable. Refresh models and select an available model.') from None
    if status == 403:
        raise LLMUserActionRequiredError('The ChatGPT account does not have permission for this Codex request.') from None
    raise RuntimeError('Codex could not complete the request. Check the connection and retry.') from None


async def _check_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    await response.aread()
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    _raise_service_error(payload if isinstance(payload, dict) else {}, response.status_code)


def _jwt_claims(token: str) -> Dict:
    # Claims are metadata from the TLS-authenticated token endpoint, never a
    # substitute for the backend's authentication of the access token.
    try:
        payload = token.split('.')[1]
        decoded = json.loads(base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4)))
        return decoded if isinstance(decoded, dict) else {}
    except (ValueError, IndexError, UnicodeDecodeError):
        return {}


class CodexAccount:
    """Own only this integration's credentials and serialize token rotation.

    >>> CodexAccount().generation
    0
    """

    def __init__(self) -> None:
        self.generation = 0
        self.changing = False
        self._lock = threading.Lock()
        # Auth errors must publish without waiting for token-renewal HTTP under _lock.
        self._state_lock = threading.Lock()
        self._loaded = False
        self._credentials = None
        self._needs_save = False
        self._storage_mode = ''
        self.auth_invalid = False

    @staticmethod
    def _path() -> Path:
        from ballontranslator.utils.shared import CONFIG_PATH
        # Never import, overwrite or log out the SDK/CLI's auth.json account.
        return Path(CONFIG_PATH).resolve().parent / 'codex' / 'http-auth.json'

    def invalidate(self) -> None:
        with self._state_lock:
            self.generation += 1

    def _set_storage_mode(self, mode: str) -> None:
        if self._storage_mode != mode:
            self._storage_mode = mode
            if mode == 'obfuscated':
                LOGGER.warning('Codex credentials are stored with reversible obfuscation, not encryption.')

    def _key_id(self) -> str:
        # Separate config directories must not replace or delete each other's key.
        return hashlib.sha256(os.fsencode(self._path().resolve())).hexdigest()

    @property
    def cached_account_label(self) -> Optional[str]:
        """Return known local sign-in state without loading credentials or doing IO.

        >>> CodexAccount().cached_account_label is None
        True
        """
        if self.auth_invalid:
            return ''
        if not self._loaded:
            return None
        credentials = self._credentials
        return '' if credentials is None else credentials.get('email') or credentials['account_id']

    def _load(self) -> None:
        if self._loaded:
            return
        storage_mode = ''
        try:
            data = json.loads(self._path().read_text(encoding='utf-8'))
        except FileNotFoundError:
            data = None
        except (OSError, ValueError):
            raise LLMUserActionRequiredError('Codex credentials could not be read. Sign in again from the Codex settings panel.') from None
        if data is not None:
            from ballontranslator.utils.secret_store import SecretStore, is_portable_secret

            if isinstance(data, dict) and data.get('storage') == 'system' and data.get('version') == 1:
                storage_mode = 'system'
                try:
                    from cryptography.fernet import Fernet

                    key = _system_keyring().get_password(_KEYRING_SERVICE, self._key_id())
                    # A passive read must never replace a missing or locked key.
                    data = json.loads(Fernet(key).decrypt(data['value'].encode('ascii')))
                except Exception:
                    raise LLMUserActionRequiredError('Codex credentials could not be unlocked. Unlock the system credential store and retry, or sign in again.') from None
            elif is_portable_secret(data):
                storage_mode = 'obfuscated'
                try:
                    data = json.loads(SecretStore().resolve(data).value)
                except ValueError:
                    raise LLMUserActionRequiredError('Codex credentials are invalid. Sign in again from the Codex settings panel.') from None
            elif isinstance(data, dict) and 'storage' not in data:
                # Passive reads ignore plaintext without rewriting it or raising
                # a startup error. Only an explicit sign-in may replace the file.
                data = None
            else:
                raise LLMUserActionRequiredError('Codex credentials use an unsupported storage format. Sign in again from the Codex settings panel.')
        if data is not None and not self._valid_credentials(data):
            raise LLMUserActionRequiredError('Codex credentials are invalid. Sign in again from the Codex settings panel.')
        self._credentials = data
        self._loaded = True
        self._set_storage_mode(storage_mode)

    def _require_credentials(self) -> None:
        if self.changing:
            raise LLMUserActionRequiredError('Finish the Codex account change before running a task.')
        if self.auth_invalid:
            raise CodexSignInRequiredError(invalid=True)
        self._load()
        if self._credentials is None:
            raise CodexSignInRequiredError()

    def require_sign_in(self, stop_event: Optional[threading.Event] = None) -> None:
        """Check app-owned credentials before runtime validation, without HTTP.

        >>> account = CodexAccount()
        >>> account._loaded = True
        >>> account.require_sign_in()
        Traceback (most recent call last):
            ...
        ballontranslator.modules.exceptions.CodexSignInRequiredError: Sign in with ChatGPT to use Codex.
        """
        generation = self.generation
        while True:
            if (stop_event is not None and stop_event.is_set()) or generation != self.generation:
                raise LLMRequestStopped()
            if self._lock.acquire(timeout=0.05):
                break
        try:
            if (stop_event is not None and stop_event.is_set()) or generation != self.generation:
                raise LLMRequestStopped()
            self._require_credentials()
            if (stop_event is not None and stop_event.is_set()) or generation != self.generation:
                raise LLMRequestStopped()
        finally:
            self._lock.release()

    @staticmethod
    def _valid_credentials(data: Any) -> bool:
        return (
            isinstance(data, dict)
            and all(isinstance(data.get(key), str) and data[key] for key in ('access_token', 'refresh_token', 'account_id'))
            and isinstance(data.get('expires_at'), (int, float))
            and not isinstance(data.get('expires_at'), bool)
            and math.isfinite(data['expires_at'])
            and isinstance(data.get('email', ''), str)
        )

    def _save(self) -> None:
        if not self._valid_credentials(self._credentials):
            raise LLMUserActionRequiredError('Codex credentials are invalid. Sign in again from the Codex settings panel.')
        plaintext = json.dumps(self._credentials)
        try:
            from cryptography.fernet import Fernet

            vault = _system_keyring()
            key_id = self._key_id()
            key = vault.get_password(_KEYRING_SERVICE, key_id)
            if key is None:
                key = Fernet.generate_key().decode('ascii')
                vault.set_password(_KEYRING_SERVICE, key_id, key)
                # Some native backends can fail a write without raising.
                if vault.get_password(_KEYRING_SERVICE, key_id) != key:
                    raise RuntimeError('The system credential store did not retain the key.')
            data = {'storage': 'system', 'version': 1,
                    'value': Fernet(key).encrypt(plaintext.encode('utf-8')).decode('ascii')}
            storage_mode = 'system'
        except Exception:
            from ballontranslator.utils.secret_store import SecretStore

            data = SecretStore().store('codex', plaintext)
            storage_mode = 'obfuscated'
        path = self._path()
        temp_path = None
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(prefix='.http-auth-', dir=str(path.parent))
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
            self._needs_save = False
            self._set_storage_mode(storage_mode)
        except OSError:
            raise LLMUserActionRequiredError('Codex credentials could not be saved. Check permissions for config/codex, then retry.') from None
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    pass

    def _update_tokens(self, payload: Dict, previous: Optional[Dict] = None) -> None:
        candidate = dict(previous or {})
        for key in ('access_token', 'refresh_token', 'id_token'):
            if key in payload:
                if not isinstance(payload[key], str) or not payload[key]:
                    raise LLMUserActionRequiredError('Codex returned incomplete credentials. Sign in again.')
                candidate[key] = payload[key]
        claims = _jwt_claims(candidate.get('id_token', ''))
        access_claims = _jwt_claims(candidate.get('access_token', ''))
        auth = claims.get('https://api.openai.com/auth', {})
        access_auth = access_claims.get('https://api.openai.com/auth', {})
        candidate['account_id'] = auth.get('chatgpt_account_id') or access_auth.get('chatgpt_account_id') or candidate.get('account_id', '')
        candidate['email'] = claims.get('email') or candidate.get('email', '')
        expires = access_claims.get('exp')
        if isinstance(expires, (int, float)):
            candidate['expires_at'] = expires
        elif isinstance(payload.get('expires_in'), (int, float)):
            candidate['expires_at'] = time.time() + payload['expires_in']
        elif 'access_token' in payload:
            candidate['expires_at'] = time.time() + 3600
        if not self._valid_credentials(candidate):
            raise LLMUserActionRequiredError('Codex returned incomplete credentials. Sign in again.')
        # Rotation may invalidate the old refresh token immediately. Retain the
        # new token in memory even when storage fails; retry saving before use.
        self._credentials = candidate
        self._loaded = True
        with self._state_lock:
            self.auth_invalid = False
        self._needs_save = True
        self._save()

    async def tokens(self, client: httpx.AsyncClient, rejected_token: str = '') -> Dict:
        generation = self.generation
        while not self._lock.acquire(blocking=False):
            await asyncio.sleep(0.05)
        try:
            self._require_credentials()
            if self._needs_save:
                self._save()
            current = self._credentials
            if current['expires_at'] <= time.time() + 60 or (rejected_token and current['access_token'] == rejected_token):
                response = await client.post(AUTH_URL + '/oauth/token', data={
                    'grant_type': 'refresh_token', 'refresh_token': current['refresh_token'], 'client_id': CLIENT_ID,
                })
                await _check_response(response)
                self._update_tokens(response.json(), current)
            return dict(self._credentials)
        except CodexSignInRequiredError as error:
            with self._state_lock:
                if error.invalid and generation == self.generation:
                    self.auth_invalid = True
            raise
        finally:
            self._lock.release()

    def login(self, stop_event: threading.Event, show_url: Callable[[str], None]) -> None:
        self.changing = True
        self.invalidate()
        try:
            _run(self._login(stop_event, show_url), stop_event)
        finally:
            self.changing = False

    async def _login(self, stop_event: threading.Event, show_url: Callable[[str], None]) -> None:
        generation = self.generation
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode('ascii')).digest()).decode('ascii').rstrip('=')
        callback = asyncio.get_running_loop().create_future()

        async def receive(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            status, body = '400 Bad Request', 'This sign-in callback is invalid.'
            try:
                line = await asyncio.wait_for(reader.readline(), 2)
                method, target, _ = line.decode('ascii').strip().split(' ', 2)
                parsed = urlsplit(target)
                query = parse_qs(parsed.query)
                if (method == 'GET' and parsed.path == '/auth/callback'
                        and hmac.compare_digest(query.get('state', [''])[0], state)):
                    code = query.get('code', [''])[0]
                    if code and not callback.done():
                        callback.set_result(code)
                        status, body = '200 OK', 'Return to BallonsTranslator to complete sign-in.'
                    elif query.get('error') and not callback.done():
                        callback.set_exception(LLMUserActionRequiredError('ChatGPT sign-in was declined. Try signing in again.'))
            except (ValueError, UnicodeDecodeError, asyncio.TimeoutError):
                pass
            finally:
                writer.write(('HTTP/1.1 ' + status + '\r\nContent-Type: text/plain; charset=utf-8\r\nConnection: close\r\n\r\n' + body).encode('utf-8'))
                try:
                    await writer.drain()
                finally:
                    writer.close()

        server = None
        for port in (1455, 1457):
            try:
                server = await asyncio.start_server(receive, '127.0.0.1', port, limit=8192)
                break
            except OSError:
                continue
        if server is None:
            raise LLMUserActionRequiredError('Codex sign-in needs localhost port 1455 or 1457. Close another sign-in window and retry.')
        redirect_uri = f'http://localhost:{port}/auth/callback'
        async with server:
            show_url(AUTH_URL + '/oauth/authorize?' + urlencode({
                'response_type': 'code', 'client_id': CLIENT_ID, 'redirect_uri': redirect_uri,
                'scope': 'openid profile email offline_access', 'code_challenge': challenge,
                'code_challenge_method': 'S256', 'id_token_add_organizations': 'true',
                'codex_cli_simplified_flow': 'true', 'state': state, 'originator': 'ballontranslator',
            }))
            try:
                code = await asyncio.wait_for(callback, 180)
            except asyncio.TimeoutError:
                raise LLMUserActionRequiredError('ChatGPT sign-in timed out. Try signing in again.') from None
        async with _http_client() as client:
            response = await client.post(AUTH_URL + '/oauth/token', data={
                'grant_type': 'authorization_code', 'client_id': CLIENT_ID,
                'redirect_uri': redirect_uri, 'code': code, 'code_verifier': verifier,
            })
            await _check_response(response)
            while not self._lock.acquire(blocking=False):
                await asyncio.sleep(0.05)
            try:
                if stop_event.is_set() or generation != self.generation:
                    raise LLMRequestStopped()
                self._update_tokens(response.json())
            finally:
                self._lock.release()

    def logout(self, stop_event: threading.Event) -> None:
        self.changing = True
        self.invalidate()

        async def clear() -> None:
            while not self._lock.acquire(blocking=False):
                await asyncio.sleep(0.05)
            try:
                try:
                    self._path().unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    raise LLMUserActionRequiredError('Codex credentials could not be removed. Check permissions for config/codex and retry.') from None
                self._credentials = None
                self._loaded = True
                with self._state_lock:
                    self.auth_invalid = False
                self._needs_save = False
                storage_mode = self._storage_mode
                self._set_storage_mode('')
                # Remove tokens first: vault cleanup may be unavailable or locked.
                try:
                    vault = _system_keyring()
                    key_id = self._key_id()
                    if vault.get_password(_KEYRING_SERVICE, key_id) is not None:
                        vault.delete_password(_KEYRING_SERVICE, key_id)
                except Exception:
                    if storage_mode == 'system':
                        LOGGER.warning('Codex credentials were removed, but the system storage key could not be removed.')
            finally:
                self._lock.release()

        try:
            _run(clear(), stop_event)
        finally:
            self.changing = False

    def catalog(self, stop_event: threading.Event) -> Dict:
        self.require_sign_in(stop_event)

        async def read() -> Dict:
            async with _http_client() as client:
                rejected = ''
                for attempt in range(2):
                    tokens = await self.tokens(client, rejected)
                    response = await client.get(API_URL + '/models', params={'client_version': CATALOG_VERSION}, headers=_headers(tokens))
                    if response.status_code == 401 and attempt == 0:
                        rejected = tokens['access_token']
                        continue
                    await _check_response(response)
                    payload = response.json()
                    if isinstance(payload, dict) and payload.get('error'):
                        _raise_service_error(payload)
                    if not isinstance(payload, dict) or not isinstance(payload.get('models'), list):
                        raise RuntimeError('Codex returned an invalid model catalog.')
                    models = {}
                    for model in payload['models']:
                        if not isinstance(model, dict) or not isinstance(model.get('slug'), str) or not model['slug'].strip():
                            LOGGER.warning('Discard invalid Codex model catalog entry.')
                            continue
                        if model.get('visibility', 'list') != 'list':
                            continue
                        raw_efforts = model.get('supported_reasoning_levels', [])
                        if not isinstance(raw_efforts, list):
                            LOGGER.warning('Discard invalid Codex reasoning levels.')
                            raw_efforts = []
                        efforts = [entry['effort'] for entry in raw_efforts
                                   if isinstance(entry, dict) and isinstance(entry.get('effort'), str)]
                        if len(efforts) != len(raw_efforts):
                            LOGGER.warning('Discard invalid Codex reasoning level entries.')
                        modalities = model.get('input_modalities', ['text'])
                        if not isinstance(modalities, list):
                            LOGGER.warning('Discard invalid Codex input modalities.')
                            modalities = ['text']
                        models[model['slug']] = {
                            'modalities': modalities, 'efforts': efforts,
                        }
                    return normalize_codex_models(models)

        return _run(read(), stop_event)


account = CodexAccount()


def _headers(tokens: Dict, cache_key: str = '') -> Dict[str, str]:
    headers = {'Authorization': 'Bearer ' + tokens['access_token'], 'ChatGPT-Account-Id': tokens['account_id'],
               'originator': 'ballontranslator', 'User-Agent': 'BallonsTranslator', 'Accept': 'application/json'}
    if cache_key:
        headers['session_id'] = cache_key
        headers['Accept'] = 'text/event-stream'
    return headers


def request_image(
    model: str,
    prompt: str,
    image: Optional[bytes],
    mask: Optional[bytes],
    stop_event: Optional[threading.Event],
    *,
    proxy: str = '',
    timeout: Optional[float] = 180.0,
) -> bytes:
    """Return one completed Codex image; the caller decodes and composites it.

    The native image API uses JSON references, with no dedicated mask field.
    The feature owner describes the optional second reference in its prompt.

    >>> result = request_image('gpt-image-2', 'Remove masked text.',
    ...                        page_png, mask_png, stop)  # doctest: +SKIP
    """
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    account.require_sign_in(stop_event)
    generation = account.generation
    if model != CODEX_IMAGE_MODEL:
        raise LLMUserActionRequiredError('Select gpt-image-2 for Codex image generation and editing.')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Codex image requests require a prompt.')
    if mask is not None and image is None:
        raise ValueError('A Codex image mask requires an input image.')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Codex image request timeout must be positive or None.')
    references = []
    for raw in (image, mask):
        if raw is None:
            continue
        if not isinstance(raw, bytes) or not raw.startswith(b'\x89PNG\r\n\x1a\n') or len(raw) > _MAX_IMAGE_BYTES:
            raise ValueError('Codex images must be nonempty PNG bytes within the image size limit.')
        references.append({'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')})
    payload = {'model': model, 'prompt': prompt, 'n': 1, 'background': 'auto', 'quality': 'auto', 'size': 'auto'}
    if references:
        payload['images'] = references
    endpoint = '/images/edits' if references else '/images/generations'

    async def request() -> Dict:
        async with _http_client(proxy) as client:
            rejected = ''
            for attempt in range(2):
                tokens = await account.tokens(client, rejected)
                async with client.stream('POST', API_URL + endpoint, headers=_headers(tokens),
                                         json=payload, timeout=timeout) as response:
                    if response.status_code == 401 and attempt == 0:
                        rejected = tokens['access_token']
                        continue
                    await _check_response(response)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_IMAGE_RESPONSE_BYTES:
                            raise LLMUserActionRequiredError('Codex returned an oversized image response. Reduce the request size.')
                    result = json.loads(body)
                    if isinstance(result, dict) and result.get('error'):
                        _raise_service_error(result)
                    return result

    response = _run(request(), stop_event)
    data = response.get('data') if isinstance(response, dict) else None
    encoded = data[0].get('b64_json') if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) else None
    if not isinstance(encoded, str) or not encoded or len(encoded) > _MAX_IMAGE_BASE64_BYTES:
        raise RuntimeError('Codex returned no valid inline image data.')
    try:
        result = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise RuntimeError('Codex returned invalid image data.') from None
    if not result or len(result) > _MAX_IMAGE_BYTES:
        raise RuntimeError('Codex returned empty or oversized image data.')
    if (stop_event is not None and stop_event.is_set()) or generation != account.generation:
        raise LLMRequestStopped()
    return result


def _request_messages(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """Map only supplied messages to Responses roles; retain image order.

    >>> _request_messages([{'role': 'system', 'content': 'contract'},
    ...                    {'role': 'user', 'content': 'page'}])[0]
    'contract'
    """
    instructions, inputs = [], []
    for message in messages:
        role, content = message['role'], message.get('content', '')
        if role in ('system', 'developer'):
            if not isinstance(content, str):
                raise ValueError('Codex instructions must be text.')
            instructions.append(content)
            continue
        if role not in ('user', 'assistant'):
            raise ValueError('Unsupported Codex message role.')
        parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        response_parts = []
        for part in parts:
            if part['type'] == 'text':
                response_parts.append({'type': 'output_text' if role == 'assistant' else 'input_text', 'text': part['text']})
            elif part['type'] == 'image_url' and role == 'user':
                image = part['image_url']
                image_part = {'type': 'input_image', 'image_url': image['url']}
                if image.get('detail') in ('auto', 'low', 'high'):
                    image_part['detail'] = image['detail']
                response_parts.append(image_part)
            else:
                raise ValueError('Unsupported Codex message content.')
        inputs.append({'type': 'message', 'role': role, 'content': response_parts})
    return '\n\n'.join(instructions), inputs


async def _read_completion(response: httpx.Response) -> Dict:
    completed_items, data = [], []
    async for line in response.aiter_lines():
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif not line and data:
            raw, data = '\n'.join(data), []
            if raw == '[DONE]':
                break
            event = json.loads(raw)
            kind = event.get('type')
            if kind == 'response.output_item.done':
                completed_items.append(event['item'])
            elif kind == 'response.completed':
                result = event['response']
                if result.get('status') != 'completed':
                    raise RuntimeError('Codex response did not complete.')
                if not result.get('output'):
                    result['output'] = completed_items
                return result
            elif kind == 'response.incomplete':
                raise LLMUserActionRequiredError('Codex output was truncated. Reduce the current input or thinking level and retry.')
            elif kind in ('response.failed', 'error'):
                _raise_service_error(event.get('response', event))
    raise RuntimeError('Codex response stream ended before completion.')


def request_chat_completion(profile: LLMProfile, api_args: Dict, stop_event: Optional[threading.Event],
                            cache_key: str, proxy: str = '') -> LLMChatResult:
    """Send stateless input with a cache identity owned by the current job.

    >>> _request_messages([{'role': 'user', 'content': 'page'}])[1][0]['role']
    'user'
    """
    from .llm_chat import LLMChatResult
    from ballontranslator.utils.config import pcfg
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    account.require_sign_in(stop_event)
    model = api_args['model']
    entry = pcfg.module.codex_models.get(model)
    if not entry:
        raise LLMUserActionRequiredError('Refresh the Codex models and select an available model.')
    effort = profile.thinking_level
    effort = None if effort == THINKING_AUTO else 'none' if effort == THINKING_DISABLED else effort
    if effort is not None and effort not in entry['efforts']:
        raise LLMUserActionRequiredError('The selected Codex model does not support this thinking level. Choose Auto or a supported level.')
    instructions, inputs = _request_messages(api_args['messages'])
    if any(part['type'] == 'input_image' for item in inputs for part in item['content']) and 'image' not in entry['modalities']:
        raise LLMUserActionRequiredError('The selected Codex model does not support image input.')
    payload = {'model': model, 'instructions': instructions, 'input': inputs, 'tools': [],
               'tool_choice': 'none', 'store': False, 'stream': True, 'prompt_cache_key': cache_key}
    if effort is not None:
        payload['reasoning'] = {'effort': effort}
    schema = api_args.get('response_format', {}).get('json_schema')
    if schema:
        payload['text'] = {'format': {'type': 'json_schema', **schema}}

    async def request() -> LLMChatResult:
        async with _http_client(proxy) as client:
            rejected = ''
            for attempt in range(2):
                tokens = await account.tokens(client, rejected)
                async with client.stream('POST', API_URL + '/responses', headers=_headers(tokens, cache_key), json=payload) as response:
                    if response.status_code == 401 and attempt == 0:
                        rejected = tokens['access_token']
                        continue
                    await _check_response(response)
                    result = await _read_completion(response)
                messages = [item for item in result.get('output', [])
                            if item.get('type') == 'message' and item.get('role') == 'assistant'
                            and item.get('phase') in (None, 'final_answer')]
                final = [item for item in messages if item.get('phase') == 'final_answer'] or messages
                if not final:
                    raise RuntimeError('Codex response contained no final message.')
                if any(part.get('type') == 'refusal' for item in final for part in item.get('content', [])):
                    raise LLMUserActionRequiredError('Codex declined the request. Review the current input before retrying.')
                content = ''.join(part['text'] for item in final for part in item.get('content', []) if part.get('type') == 'output_text')
                raw_usage, usage = result.get('usage'), None
                if isinstance(raw_usage, dict) and all(
                    type(raw_usage.get(key)) is int and raw_usage[key] >= 0
                    for key in ('input_tokens', 'output_tokens', 'total_tokens')
                ):
                    usage = SimpleNamespace(
                        prompt_tokens=raw_usage['input_tokens'], completion_tokens=raw_usage['output_tokens'],
                        total_tokens=raw_usage['total_tokens'],
                        prompt_tokens_details=raw_usage.get('input_tokens_details'),
                        completion_tokens_details=raw_usage.get('output_tokens_details'),
                    )
                return LLMChatResult(content=content, finish_reason='stop', usage=usage)

    return _run(request(), stop_event)
