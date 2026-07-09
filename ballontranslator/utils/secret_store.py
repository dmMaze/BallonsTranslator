from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict


PORTABLE_SECRET_STORAGE = "portable_obfuscated"
PORTABLE_SECRET_VERSION = 1
KEYRING_PREFIX = "keyring://BallonsTranslator/llm/"

# This is intentionally an app-bundled obfuscation key, not a user secret.
# It hides keys from plain-text config scanners while keeping profiles portable.
_APP_SECRET = b"BallonsTranslator portable LLM profile secret v1"


def is_keyring_reference(value: Any) -> bool:
    """Return whether a saved secret value points to the removed OS keychain path.

    Example:
        >>> is_keyring_reference('keyring://BallonsTranslator/llm/demo/api-key')
        True
        >>> is_keyring_reference('sk-demo')
        False
    """

    return isinstance(value, str) and value.startswith(KEYRING_PREFIX)


def is_portable_secret(value: Any) -> bool:
    """Return whether a config value is a portable obfuscated secret object.

    Example:
        >>> is_portable_secret({'storage': 'portable_obfuscated', 'version': 1, 'value': 'x'})
        True
        >>> is_portable_secret('sk-demo')
        False
    """

    if not isinstance(value, dict) or value.get("storage") != PORTABLE_SECRET_STORAGE:
        return False
    try:
        version = int(value.get("version") or 0)
    except Exception:
        return False
    return version == PORTABLE_SECRET_VERSION and isinstance(value.get("value"), str)


@dataclass
class SecretResolution:
    value: str
    from_obfuscated: bool = False
    legacy_keyring: bool = False
    error: str = ""


class SecretStore:
    """Portable reversible obfuscation for API keys stored in config.

    This is not a secure password vault. The key is bundled with the app so
    profiles can move across systems; the purpose is only to avoid plain-text
    scanner hits in ``config.json``.

    Example:
        >>> store = SecretStore()
        >>> saved = store.store('profile', 'sk-demo')
        >>> is_portable_secret(saved)
        True
        >>> store.resolve(saved).value
        'sk-demo'
    """

    def __init__(self):
        self.last_error = ""

    @staticmethod
    def reference(profile_id: str) -> str:
        safe_id = str(profile_id).strip() or "default"
        return f"{KEYRING_PREFIX}{safe_id}/api-key"

    @staticmethod
    def _keystream(nonce: bytes, length: int) -> bytes:
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(_APP_SECRET + nonce + counter.to_bytes(4, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])

    @classmethod
    def _xor(cls, data: bytes, nonce: bytes) -> bytes:
        stream = cls._keystream(nonce, len(data))
        return bytes(byte ^ stream_byte for byte, stream_byte in zip(data, stream))

    def store(self, profile_id: str, secret: str) -> Dict[str, Any] | str:
        secret = secret or ""
        if secret == "":
            return ""
        nonce = os.urandom(12)
        plaintext = secret.encode("utf8")
        payload = nonce + self._xor(plaintext, nonce)
        return {
            "storage": PORTABLE_SECRET_STORAGE,
            "version": PORTABLE_SECRET_VERSION,
            "value": base64.urlsafe_b64encode(payload).decode("ascii"),
        }

    def resolve(self, saved_value: Any) -> SecretResolution:
        if saved_value in (None, ""):
            return SecretResolution("")
        if is_portable_secret(saved_value):
            try:
                payload = base64.urlsafe_b64decode(saved_value["value"].encode("ascii"))
                nonce, ciphertext = payload[:12], payload[12:]
                value = self._xor(ciphertext, nonce).decode("utf8")
                return SecretResolution(value, from_obfuscated=True)
            except Exception as e:
                self.last_error = str(e)
                return SecretResolution("", from_obfuscated=True, error=str(e))
        if is_keyring_reference(saved_value):
            error = "Legacy OS keychain references are no longer portable and cannot be resolved from config."
            self.last_error = error
            return SecretResolution("", legacy_keyring=True, error=error)
        if isinstance(saved_value, str):
            return SecretResolution(saved_value)
        return SecretResolution(str(saved_value))

    def prepare_for_save(self, profile_id: str, saved_value: Any) -> Any:
        if saved_value in (None, ""):
            return ""
        if is_portable_secret(saved_value) or is_keyring_reference(saved_value):
            return saved_value
        return self.store(profile_id, str(saved_value))

    def delete(self, saved_value: Any) -> None:
        return None
