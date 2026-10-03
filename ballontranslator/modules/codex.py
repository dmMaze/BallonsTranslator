"""Stateless ChatGPT/Codex HTTP transport and app-owned OAuth credentials."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path
import secrets
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Mapping, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import parse_qs, urlencode, urlsplit

from .responses_ws import ResponsesWebSocket, run_async
from .context.errors import ContextLengthError, is_context_length_error
from .exceptions import CodexSignInRequiredError, LLMRequestStopped, LLMUserActionRequiredError
from . import image_generation
from ballontranslator.utils.llm_profiles import LLMProfile, THINKING_AUTO, THINKING_DISABLED, normalize_codex_models
from ballontranslator.utils.logger import logger as LOGGER

if TYPE_CHECKING:
    import httpx
    from keyring.backend import KeyringBackend
    from websockets.datastructures import Headers as WebSocketHeaders
    from .llm_chat import LLMChatResult


_KEYRING_SERVICE = 'BallonsTranslator Codex'
# False keeps plain-text history, without provider output replay or WS continuation.
CODEX_REPLAY_ENABLED = False
# Shared catalog/generation identity; retain this compatible floor when offline.
_client_version = '0.159.0'
_client_version_checked_at = float('-inf')
_client_version_lock = threading.Lock()
API_URL = 'https://chatgpt.com/backend-api/codex'
AUTH_URL = 'https://auth.openai.com'
CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'


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


async def _latest_client_version(proxy: str = '') -> str:
    """Resolve stable release metadata at most hourly across worker event loops.

    >>> asyncio.run(_latest_client_version())  # doctest: +SKIP
    '0.159.0'
    """
    global _client_version, _client_version_checked_at
    import httpx

    while not _client_version_lock.acquire(blocking=False):
        await asyncio.sleep(0.05)
    try:
        if time.monotonic() - _client_version_checked_at < 3600:
            return _client_version
        try:
            # Use a separate unauthenticated client: never forward backend
            # credentials, cookies or routing headers to release discovery.
            async with _http_client(proxy) as client:
                response = await asyncio.wait_for(client.get(
                    'https://api.github.com/repos/openai/codex/releases/latest',
                    headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'BallonsTranslator'},
                    timeout=5.0,
                ), timeout=5.0)
                response.raise_for_status()
                release = response.json()
            tag = release.get('tag_name') if isinstance(release, dict) else None
            if (not isinstance(tag, str) or not re.fullmatch(r'rust-v[0-9]+\.[0-9]+\.[0-9]+', tag)
                    or release.get('prerelease') or release.get('draft')):
                raise ValueError('Invalid stable Codex release.')
            version = tag[len('rust-v'):]
            if tuple(map(int, version.split('.'))) >= tuple(map(int, _client_version.split('.'))):
                _client_version = version
        except (httpx.HTTPError, asyncio.TimeoutError, ValueError):
            LOGGER.warning('Could not refresh the Codex client version; using %s.', _client_version)
        _client_version_checked_at = time.monotonic()
        return _client_version
    finally:
        _client_version_lock.release()


def _run(operation: Coroutine, stop_event: Optional[threading.Event], *, generation: Optional[int] = None,
         session: Optional[CodexChatSession] = None) -> Any:
    """Run HTTP work in the calling worker, cancelling socket waits promptly.

    >>> _run(asyncio.sleep(0, result='completed'), None)
    'completed'
    """
    # Inference callers capture ownership before preparing the payload, which
    # can take time when hashing images or retained encrypted reasoning.
    if generation is None:
        generation = account.generation

    try:
        if session is not None:
            return session.run(operation, stop_event)
        return run_async(operation, stop_event, is_current=lambda: generation == account.generation)
    except (LLMRequestStopped, LLMUserActionRequiredError, ContextLengthError):
        raise
    except Exception as error:
        # OAuth codes, tokens and request URLs must not reach provider tracebacks.
        LOGGER.debug('Codex request failure: exception=%s', type(error).__name__)
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
    unavailable_model = bool(codes & {
        'model_not_found', 'invalid_model', 'unsupported_model', 'model_not_supported', 'model_not_available',
    })
    if not unavailable_model and (codes & {'unsupported_tool', 'unsupported_tool_type'} or (
        ('image_generation' in detail or 'image generation' in detail)
        and any(word in detail for word in ('unsupported', 'not supported', 'not available', 'not enabled', 'not allowed', 'invalid value', 'unknown tool'))
    )):
        raise LLMUserActionRequiredError('Codex assisted editing is unavailable for this request. Select another model pair or a direct image model.') from None
    parameter = error.get('param', '') if isinstance(error, dict) else ''
    if not unavailable_model and parameter != 'model' and (
        codes & {'invalid_parameter', 'unsupported_parameter', 'unknown_parameter'}
        or (parameter and 'invalid_request_error' in codes)
    ):
        # Parameter rejection may say "not supported on this model" without
        # rejecting the model. Expose only a bounded field path, never raw input.
        field = (f' {parameter!r}' if isinstance(parameter, str)
                 and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.\[\]]{0,127}', parameter) else '')
        http_status = f' (HTTP {status})' if isinstance(status, int) else ''
        raise LLMUserActionRequiredError(
            f'Codex rejected request parameter{field}{http_status}. '
            'This option is unsupported or invalid for this request.'
        ) from None
    model_error = detail.replace('_', ' ')
    if unavailable_model or ('model' in model_error and any(word in model_error for word in (
        'not found', 'not supported', 'unsupported', 'unavailable', 'not available',
        'does not exist', 'invalid model',
    ))):
        raise LLMUserActionRequiredError('This Codex model is unavailable. Select another model or check the model ID and account access.') from None
    if status == 403:
        raise LLMUserActionRequiredError('The ChatGPT account does not have permission for this Codex request.') from None
    raise RuntimeError('Codex could not complete the request. Check the connection and retry.') from None


async def _check_response(response: httpx.Response) -> None:
    if response.is_success:
        return
    LOGGER.debug('Codex HTTP failure: status=%d', response.status_code)
    if response.status_code == 401:
        # The status establishes rejection even if its body is interrupted.
        raise CodexSignInRequiredError(invalid=True)
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

    def reject_access_token(self, token: str, generation: int) -> None:
        """Publish an exhausted auth failure only for the token still in use."""
        with self._state_lock:
            if generation != self.generation or self.changing:
                raise LLMRequestStopped()
            if self._credentials is None or self._credentials['access_token'] != token:
                # Another request renewed the account while this response was
                # in flight. Let the owner retry without invalidating its token.
                raise RuntimeError('Codex authentication changed during the request. Retry the request.')
            self.auth_invalid = True

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
        generation = self.generation

        async def read() -> Dict:
            async with _http_client() as client:
                rejected = ''
                for attempt in range(2):
                    tokens = await self.tokens(client, rejected)
                    try:
                        headers = await _headers(tokens)
                        response = await client.get(API_URL + '/models', params={'client_version': headers['version']}, headers=headers)
                        await _check_response(response)
                        payload = response.json()
                        if isinstance(payload, dict) and payload.get('error'):
                            _raise_service_error(payload)
                    except CodexSignInRequiredError:
                        if attempt == 0:
                            rejected = tokens['access_token']
                            continue
                        self.reject_access_token(tokens['access_token'], generation)
                        raise
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


class CodexTurnState:
    """Keep the first routing token only for one feature owner's retry loop.

    >>> turn = CodexTurnState()
    >>> turn.capture({'X-Codex-Turn-State': 'opaque'})
    >>> turn.capture({'x-codex-turn-state': 'replacement'})
    >>> turn.value
    'opaque'
    """

    def __init__(self) -> None:
        self.value = ''
        self._identity: Optional[Tuple[str, int]] = None

    def bind(self, cache_key: str, generation: int) -> None:
        identity = cache_key, generation
        if self._identity != identity:
            self._identity = identity
            self.value = ''

    def capture(self, headers: object) -> None:
        if self.value or not isinstance(headers, Mapping):
            return
        # WebSocket Headers.items() raises for unrelated repeated headers such
        # as Set-Cookie. Inspect names first and read only the routing header.
        for name in headers:
            if isinstance(name, str) and name.lower() == 'x-codex-turn-state':
                value = headers.get_all(name) if hasattr(headers, 'get_all') else headers[name]
                if isinstance(value, list) and value:
                    value = value[0]
                if isinstance(value, str) and value:
                    self.value = value
                    LOGGER.debug('Codex turn-state captured: %s', _request_fingerprint(value))
                return


async def _headers(tokens: Dict, cache_key: str = '', *, turn: Optional[CodexTurnState] = None,
                   proxy: str = '', cookies: Optional[httpx.Cookies] = None, model: str = '') -> Dict[str, str]:
    version = await _latest_client_version(proxy)
    headers = {'Authorization': 'Bearer ' + tokens['access_token'], 'ChatGPT-Account-Id': tokens['account_id'],
               'originator': 'codex_cli_rs', 'version': version,
               'User-Agent': f'codex_cli_rs/{version} (BallonsTranslator)', 'Accept': 'application/json'}
    if cache_key:
        # Match Codex's Responses client: our job is one session/thread, and
        # x-client-request-id uses that identity too, unlike the public API.
        headers['session-id'] = cache_key
        headers['thread-id'] = cache_key
        headers['x-client-request-id'] = cache_key
        headers['Accept'] = 'text/event-stream'
        headers['Content-Type'] = 'application/json'
        headers['OpenAI-Beta'] = 'responses=experimental'
    if turn is not None and turn.value:
        headers['x-codex-turn-state'] = turn.value
    # Official Codex provides the target model to the subscription router.
    # Invalid header characters must not turn an optional hint into a failure.
    if model and re.fullmatch(r'[\x20-\x7e]+', model):
        headers['x-codex-routing-hint'] = 'model=' + model
    if cookies is not None:
        import httpx

        request = httpx.Request('GET', API_URL + '/responses')
        cookies.set_cookie_header(request)
        if 'Cookie' in request.headers:
            headers['Cookie'] = request.headers['Cookie']
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
    reasoning_model: str = '',
    cache_key: str = '',
) -> bytes:
    """Return one completed Codex image; the caller decodes and composites it.

    An optional reasoning model uses the Responses image-generation tool.
    Both routes describe the optional mask reference through the caller's prompt.
    Assisted requests reuse the calling job's cache key without retaining history.

    >>> result = request_image('gpt-image-2', 'Remove masked text.',
    ...                        page_png, mask_png, stop)  # doctest: +SKIP
    """
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    account.require_sign_in(stop_event)
    generation = account.generation
    if not isinstance(model, str) or not model.strip():
        raise LLMUserActionRequiredError('Select an image model for Codex image generation and editing.')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Codex image requests require a prompt.')
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Codex image request timeout must be positive or None.')
    if not isinstance(reasoning_model, str):
        raise ValueError('The Codex image reasoning model must be text.')
    reasoning_model = reasoning_model.strip()
    if reasoning_model:
        from ballontranslator.utils.config import pcfg
        entry = pcfg.module.codex_models.get(reasoning_model)
        if not entry or 'image' not in entry['modalities']:
            raise LLMUserActionRequiredError('Refresh Codex models and select an available vision model in the image model pair, or select a direct image model.')
    references = image_generation.image_references(image, mask)
    if reasoning_model:
        endpoint = '/responses'
        session_key = cache_key or secrets.token_hex(16)
        payload = image_generation.responses_image_payload(
            reasoning_model, model, prompt, references, session_key, stream=True,
        )
    else:
        endpoint = '/images/edits' if references else '/images/generations'
        session_key = ''
        payload = {'model': model, 'prompt': prompt, 'n': 1, 'background': 'auto', 'quality': 'auto', 'size': 'auto'}
        if references:
            payload['images'] = references

    async def request() -> Dict:
        async with _http_client(proxy) as client:
            rejected = ''
            for attempt in range(2):
                tokens = await account.tokens(client, rejected)
                try:
                    async with client.stream('POST', API_URL + endpoint, headers=await _headers(tokens, session_key, proxy=proxy, model=reasoning_model),
                                             json=payload, timeout=timeout) as response:
                        await _check_response(response)
                        if reasoning_model:
                            # Responses may repeat the full image in output_item.done
                            # and response.completed; both copies count toward the bound.
                            return await _read_completion(response, max_response_bytes=2 * image_generation.MAX_IMAGE_RESPONSE_BYTES)
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > image_generation.MAX_IMAGE_RESPONSE_BYTES:
                                raise LLMUserActionRequiredError('Codex returned an oversized image response. Reduce the request size.')
                        result = json.loads(body)
                        if isinstance(result, dict) and result.get('error'):
                            _raise_service_error(result)
                        return result
                except CodexSignInRequiredError:
                    if attempt == 0:
                        rejected = tokens['access_token']
                        continue
                    account.reject_access_token(tokens['access_token'], generation)
                    raise

    response = _run(request(), stop_event, generation=generation)
    if reasoning_model:
        result = image_generation.decode_responses_image(response)
    else:
        data = response.get('data') if isinstance(response, dict) else None
        encoded = data[0].get('b64_json') if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) else None
        result = image_generation.decode_inline_image(encoded, error_type=RuntimeError)
    if (stop_event is not None and stop_event.is_set()) or generation != account.generation:
        raise LLMRequestStopped()
    return result


def _request_messages(messages: List[Dict], cache_key: str = '', generation: Optional[int] = None) -> Tuple[str, List[Dict]]:
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
        replay = message.get('codex_response')
        if (CODEX_REPLAY_ENABLED and role == 'assistant' and replay is not None and replay.codex_response_items
                and replay.codex_cache_key == cache_key and replay.codex_account_generation == generation):
            # Replay provider output verbatim, including encrypted reasoning and
            # message IDs/phase. Validate ownership here: the account/job can
            # change after the history snapshot, while request throttling waits.
            inputs.extend(replay.codex_response_items)
            continue
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


async def _bounded_response_lines(response: httpx.Response, limit: int) -> AsyncIterator[str]:
    """Bound bytes before a line decoder can buffer a large inline image.

    >>> lines = _bounded_response_lines(response, 1024)  # doctest: +SKIP
    """
    received, scanned = 0, 0
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > limit:
            raise LLMUserActionRequiredError('Codex returned an oversized image response. Reduce the request size.')
        buffer.extend(chunk)
        while True:
            end = buffer.find(b'\n', scanned)
            if end < 0:
                scanned = len(buffer)
                break
            yield buffer[:end].rstrip(b'\r').decode('utf-8')
            del buffer[:end + 1]
            scanned = 0
    if buffer:
        yield buffer.rstrip(b'\r').decode('utf-8')


def _completed_response(event: Dict, completed_items: List[Dict],
                        turn: Optional[CodexTurnState] = None) -> Optional[Dict]:
    kind = event.get('type')
    if kind == 'response.metadata' and turn is not None:
        turn.capture(event.get('headers'))
    elif kind == 'response.output_item.done':
        completed_items.append(event['item'])
    elif kind in ('response.completed', 'response.done'):
        result = event['response']
        if result.get('status') != 'completed':
            raise RuntimeError('Codex response did not complete.')
        if not result.get('output'):
            result['output'] = completed_items
        return result
    elif kind == 'response.incomplete':
        raise LLMUserActionRequiredError('Codex output was truncated. Reduce the current input or thinking level and retry.')
    elif kind in ('response.failed', 'error'):
        _raise_service_error(event.get('response', event), status=event.get('status', event.get('status_code')))
    return None


async def _read_completion(response: httpx.Response, *, max_response_bytes: Optional[int] = None,
                           turn: Optional[CodexTurnState] = None) -> Dict:
    if turn is not None:
        turn.capture(response.headers)
    completed_items, data = [], []
    lines = response.aiter_lines() if max_response_bytes is None else _bounded_response_lines(response, max_response_bytes)
    async for line in lines:
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif not line and data:
            raw, data = '\n'.join(data), []
            if raw == '[DONE]':
                break
            result = _completed_response(json.loads(raw), completed_items, turn)
            if result is not None:
                return result
    raise RuntimeError('Codex response stream ended before completion.')


class CodexChatSession(ResponsesWebSocket):
    """Keep subscription authentication and routing state out of the shared transport."""

    def __init__(self, cache_key: str, proxy: str = '', *, account_generation: int,
                 websocket: bool = True) -> None:
        import httpx

        super().__init__(API_URL + '/responses', cache_key, proxy, websocket=websocket)
        self.account_generation = account_generation
        self._cookies = httpx.Cookies()

    def _is_current(self) -> bool:
        return not self.closed and self.account_generation == account.generation

    def _dispose(self) -> None:
        if self.closed:
            self._cookies.clear()
        super()._dispose()

    def capture_cookies(self, headers: Union[httpx.Headers, WebSocketHeaders]) -> None:
        """Retain only infrastructure cookies, with standard scope/expiry rules."""
        import httpx

        if self.closed or self.account_generation != account.generation:
            return
        # HTTPX and websockets expose repeated Set-Cookie headers differently.
        values = (headers.get_list('set-cookie') if isinstance(headers, httpx.Headers)
                  else headers.get_all('set-cookie'))
        allowed = []
        for value in values:
            name = value.partition('=')[0].strip()
            # Match official Codex's infrastructure allowlist. Account/session
            # cookies must never survive in this job's routing state.
            if name in {'__cf_bm', '__cflb', '__cfruid', '__cfseq', '__cfwaitingroom',
                        '__oailb', '_cfuvid', 'cf_clearance', 'cf_ob_info', 'cf_use_ob'} or name.startswith('cf_chl_'):
                allowed.append(('set-cookie', value))
        if allowed:
            response = httpx.Response(200, headers=allowed,
                                      request=httpx.Request('GET', API_URL + '/responses'))
            self._cookies.extract_cookies(response)

    async def request(self, payload: Dict, tokens: Dict, turn: CodexTurnState) -> Optional[Dict]:
        if not CODEX_REPLAY_ENABLED:
            self._previous = None
        metadata = {}

        async def headers() -> Dict[str, str]:
            result = await _headers(tokens, self.cache_key, turn=turn, proxy=self.proxy,
                                    cookies=self._cookies, model=payload['model'])
            for name in ('Accept', 'Content-Type'):
                result.pop(name, None)
            result['OpenAI-Beta'] = 'responses_websockets=2026-02-06'
            return result

        def capture(response_headers: object) -> None:
            self.capture_cookies(response_headers)
            turn.capture(response_headers)
            if turn.value:
                metadata['x-codex-turn-state'] = turn.value

        if turn.value:
            metadata['x-codex-turn-state'] = turn.value
        try:
            return await super().request(
                payload, headers, lambda event, items: _completed_response(event, items, turn),
                on_headers=capture, client_metadata=metadata,
            )
        except Exception as error:
            # Import only after the shared transport has checked availability.
            from websockets.exceptions import InvalidStatus
            if isinstance(error, InvalidStatus):
                _raise_service_error({}, status=error.response.status_code)
            raise


def _request_fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()[:16]


def request_chat_completion(profile: LLMProfile, api_args: Dict, stop_event: Optional[threading.Event],
                            cache_key: str, proxy: str = '', *, session: Optional[CodexChatSession] = None,
                            turn: Optional[CodexTurnState] = None) -> LLMChatResult:
    """Send stateless input with a cache identity owned by the current job.

    >>> _request_messages([{'role': 'user', 'content': 'page'}])[1][0]['role']
    'user'
    """
    from .llm_chat import LLMChatResult
    from ballontranslator.utils.config import pcfg
    if stop_event is not None and stop_event.is_set():
        raise LLMRequestStopped()
    account.require_sign_in(stop_event)
    generation = account.generation
    if session is not None and session.account_generation != generation:
        session.close()
        raise LLMRequestStopped()
    if turn is None:
        turn = CodexTurnState()
    turn.bind(cache_key, generation)
    model = api_args['model']
    entry = pcfg.module.codex_models.get(model)
    if not entry:
        raise LLMUserActionRequiredError('Refresh the Codex models and select an available model.')
    effort = profile.thinking_level
    effort = None if effort == THINKING_AUTO else 'none' if effort == THINKING_DISABLED else effort
    if effort is not None and effort not in entry['efforts']:
        raise LLMUserActionRequiredError('The selected Codex model does not support this thinking level. Choose Auto or a supported level.')
    instructions, inputs = _request_messages(api_args['messages'], cache_key, generation)
    if any(part['type'] == 'input_image' for item in inputs for part in (item.get('content') or [])) and 'image' not in entry['modalities']:
        raise LLMUserActionRequiredError('The selected Codex model does not support image input.')
    payload = {'model': model, 'instructions': instructions, 'input': inputs, 'tools': [],
               'tool_choice': 'none', 'store': False, 'stream': True, 'prompt_cache_key': cache_key,
               'include': ['reasoning.encrypted_content']}
    if effort is not None:
        payload['reasoning'] = {'effort': effort}
    schema = api_args.get('response_format', {}).get('json_schema')
    if schema:
        payload['text'] = {'format': {'type': 'json_schema', **schema}}
    # Fingerprint the assembled request without logging project text, images or
    # session identities. Per-item hashes expose changes to retained history.
    LOGGER.debug(
        'Codex request fingerprints: session=%s, settings=%s, instructions=%s, input=%s',
        _request_fingerprint(cache_key),
        _request_fingerprint({key: value for key, value in payload.items()
                              if key not in ('prompt_cache_key', 'instructions', 'input')}),
        _request_fingerprint(instructions),
        [(item.get('role', item['type']), _request_fingerprint(item)) for item in inputs],
    )

    async def request() -> LLMChatResult:
        async with _http_client(proxy) as client:
            rejected = ''
            for attempt in range(2):
                tokens = await account.tokens(client, rejected)
                try:
                    result = await session.request(payload, tokens, turn) if session is not None else None
                    if result is None:
                        headers = await _headers(tokens, cache_key, turn=turn, proxy=proxy,
                                                 cookies=session._cookies if session is not None else None, model=model)
                        LOGGER.debug('Codex SSE request: turn_state=%s, routing_cookies=%s',
                                     _request_fingerprint(turn.value) if turn.value else 'absent', 'Cookie' in headers)
                        async with client.stream('POST', API_URL + '/responses', headers=headers, json=payload) as response:
                            if session is not None:
                                session.capture_cookies(response.headers)
                            await _check_response(response)
                            result = await _read_completion(response, turn=turn)
                except CodexSignInRequiredError:
                    if attempt == 0:
                        rejected = tokens['access_token']
                        continue
                    account.reject_access_token(tokens['access_token'], generation)
                    raise
                reasoning = result.get('reasoning')
                LOGGER.debug(
                    'Codex response context: model=%s, reasoning_context=%s',
                    result.get('model', 'not_reported'),
                    reasoning.get('context', 'not_reported') if isinstance(reasoning, dict) else 'not_reported',
                )
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
                return LLMChatResult(
                    content=content, finish_reason='stop', usage=usage,
                    prompt_cache_diagnostics=result.get('prompt_cache_diagnostics'),
                    codex_response_items=tuple(
                        item for item in result.get('output', [])
                        if (item.get('type') == 'message' and item.get('role') == 'assistant')
                        or (item.get('type') == 'reasoning'
                            and isinstance(item.get('encrypted_content'), str) and item['encrypted_content'])
                    ),
                    codex_cache_key=cache_key,
                    codex_account_generation=generation,
                )

    return (_run(request(), stop_event, generation=generation) if session is None or not session.websocket
            else _run(request(), stop_event, generation=generation, session=session))
