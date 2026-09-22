from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from cryptography.fernet import Fernet

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import LLMUserActionRequiredError
from ballontranslator.utils.secret_store import SecretStore


def credentials(account_id: str = 'private-account-id') -> dict:
    claims = {'email': 'private@example.com', 'https://api.openai.com/auth': {'chatgpt_account_id': account_id}}
    identity = 'header.' + base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=') + '.signature'
    return {'access_token': 'private-access-token', 'refresh_token': 'private-refresh-token',
            'id_token': identity, 'account_id': account_id,
            'email': 'private@example.com', 'expires_at': time.time() + 3600}


class MemoryVault:
    def __init__(self) -> None:
        self.entries = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.entries.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.entries[service, username] = password

    def delete_password(self, service: str, username: str) -> None:
        del self.entries[service, username]


class CodexCredentialStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'codex' / 'http-auth.json'
        self.vault = MemoryVault()
        self.account = self.new_account()
        self.account._credentials = credentials()
        self.account._loaded = True
        for patcher in (patch.object(codex, '_system_keyring', return_value=self.vault),
                        patch.object(codex, 'account', self.account)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def new_account(self, path: Path | None = None) -> codex.CodexAccount:
        account = codex.CodexAccount()
        account._path = lambda: path or self.path
        return account

    def assert_round_trip(self, mode: str) -> None:
        restarted = self.new_account()
        restarted._load()
        self.assertEqual(restarted._credentials, self.account._credentials)
        self.assertEqual(restarted._storage_mode, mode)
        self.assertEqual(self.account._storage_mode, mode)
        saved = self.path.read_text()
        for key in ('access_token', 'refresh_token', 'id_token', 'account_id', 'email'):
            self.assertNotIn(self.account._credentials[key], saved)

    def test_secure_restart_keeps_large_token_bundle_out_of_small_vault(self) -> None:
        self.account._credentials['access_token'] *= 2000
        self.account._save()
        self.assert_round_trip('system')
        key, = self.vault.entries.values()
        self.assertEqual(len(key), 44)
        envelope = json.loads(self.path.read_text())
        decoded = json.loads(Fernet(key).decrypt(envelope['value'].encode()))
        self.assertEqual(decoded, self.account._credentials)
        if os.name != 'nt':
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_tampered_ciphertext_is_rejected_without_rewriting_credentials(self) -> None:
        self.account._save()
        envelope = json.loads(self.path.read_text())
        ciphertext = envelope['value']
        envelope['value'] = ciphertext[:50] + ('A' if ciphertext[50] != 'A' else 'B') + ciphertext[51:]
        self.path.write_text(json.dumps(envelope))
        unchanged = self.path.read_bytes()
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'unlocked'):
            self.new_account()._load()
        self.assertEqual(self.path.read_bytes(), unchanged)

    def test_missing_locked_and_invalid_keys_never_regenerate_or_replace_file(self) -> None:
        self.account._save()
        unchanged = self.path.read_bytes()
        for key in (None, 'invalid-key', Fernet.generate_key().decode()):
            with self.subTest(key=key), patch.object(self.vault, 'get_password', return_value=key), \
                    patch.object(self.vault, 'set_password') as write:
                restarted = self.new_account()
                with self.assertRaisesRegex(LLMUserActionRequiredError, 'unlocked'):
                    restarted._load()
                self.assertFalse(restarted._loaded)
                self.assertIsNone(restarted._credentials)
                self.assertEqual(self.path.read_bytes(), unchanged)
                write.assert_not_called()
        with patch.object(self.vault, 'get_password', side_effect=RuntimeError('secret backend detail')), \
                patch.object(self.vault, 'set_password') as write:
            with self.assertRaises(LLMUserActionRequiredError) as caught:
                self.new_account()._load()
            self.assertNotIn('secret backend detail', str(caught.exception))
            self.assertEqual(self.path.read_bytes(), unchanged)
            write.assert_not_called()

    def test_unavailable_or_broken_secure_storage_uses_obfuscation(self) -> None:
        with patch.object(codex, '_system_keyring', side_effect=ImportError):
            self.account._save()
            self.assert_round_trip('obfuscated')
        with patch.dict(sys.modules, {'cryptography.fernet': None}):
            self.account._save()
            self.assert_round_trip('obfuscated')
        with patch.object(self.vault, 'set_password', side_effect=RuntimeError('private vault failure')):
            self.account._save()
            self.assert_round_trip('obfuscated')
        with patch.object(self.vault, 'set_password'):
            self.account._save()
            self.assert_round_trip('obfuscated')

    def test_secure_file_with_unavailable_vault_stays_intact_until_explicit_login(self) -> None:
        self.account._save()
        unchanged = self.path.read_bytes()
        with patch.object(codex, '_system_keyring', side_effect=RuntimeError('Vault unavailable')):
            restarted = self.new_account()
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'could not be unlocked'):
                restarted._load()
            self.assertEqual(self.path.read_bytes(), unchanged)
            restarted._update_tokens(credentials())
            self.assertEqual(restarted._storage_mode, 'obfuscated')
            loaded = self.new_account()
            loaded._load()
            self.assertEqual(loaded._credentials, restarted._credentials)

    def test_existing_plaintext_is_ignored_and_explicit_login_can_replace_it(self) -> None:
        self.path.parent.mkdir()
        self.path.write_text(json.dumps(credentials()))
        unchanged = self.path.read_bytes()
        restarted = self.new_account()
        with patch.object(codex, '_system_keyring') as vault:
            restarted._load()
            vault.assert_not_called()
        self.assertEqual(self.path.read_bytes(), unchanged)
        self.assertTrue(restarted._loaded)
        self.assertIsNone(restarted._credentials)
        self.assertEqual(restarted.cached_account_label, '')
        self.assertEqual(restarted._storage_mode, '')
        restarted._update_tokens(credentials())
        loaded = self.new_account()
        loaded._load()
        self.assertEqual(loaded._credentials, restarted._credentials)
        self.assertEqual(loaded._storage_mode, 'system')

    def test_unknown_protected_format_is_preserved_and_reported(self) -> None:
        self.path.parent.mkdir()
        self.path.write_text(json.dumps({'storage': 'system', 'version': 99, 'value': 'unknown'}))
        unchanged = self.path.read_bytes()
        with self.assertRaisesRegex(LLMUserActionRequiredError, 'unsupported storage format'):
            self.new_account()._load()
        self.assertEqual(self.path.read_bytes(), unchanged)

    def test_authenticated_and_obfuscated_invalid_credentials_still_fail_validation(self) -> None:
        self.account._save()
        key, = self.vault.entries.values()
        for data in ({}, dict(credentials(), expires_at=True), dict(credentials(), refresh_token='')):
            plaintext = json.dumps(data)
            envelopes = (
                {'storage': 'system', 'version': 1, 'value': Fernet(key).encrypt(plaintext.encode()).decode()},
                SecretStore().store('codex', plaintext),
            )
            for envelope in envelopes:
                with self.subTest(data=data, storage=envelope['storage']):
                    self.path.write_text(json.dumps(envelope))
                    with self.assertRaisesRegex(LLMUserActionRequiredError, 'invalid'):
                        self.new_account()._load()

    def test_refresh_after_vault_locks_saves_rotated_tokens_with_obfuscation(self) -> None:
        self.account._credentials['expires_at'] = 0
        self.account._save()
        restarted = self.new_account()
        restarted._load()
        payload = {'access_token': 'new-private-access', 'refresh_token': 'new-private-refresh', 'expires_in': 3600}

        async def refresh() -> dict:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))) as client:
                return await restarted.tokens(client)

        with patch.object(self.vault, 'get_password', side_effect=RuntimeError('locked')):
            updated = asyncio.run(refresh())
            loaded = self.new_account()
            loaded._load()
        self.assertEqual(updated['refresh_token'], payload['refresh_token'])
        self.assertEqual(loaded._credentials, updated)
        self.assertEqual(restarted._storage_mode, 'obfuscated')

    def test_failed_atomic_write_retains_rotated_credentials_and_previous_readable_file(self) -> None:
        self.account._save()
        unchanged = self.path.read_bytes()
        with patch.object(codex.os, 'replace', side_effect=PermissionError):
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'saved'):
                self.account._update_tokens({'access_token': 'new-access', 'refresh_token': 'new-refresh'}, self.account._credentials)
        self.assertEqual(self.path.read_bytes(), unchanged)
        self.assertEqual(self.account._credentials['refresh_token'], 'new-refresh')
        self.assertTrue(self.account._needs_save)
        self.assertEqual(list(self.path.parent.glob('.http-auth-*')), [])
        loaded = self.new_account()
        loaded._load()
        self.assertEqual(loaded._credentials['refresh_token'], 'private-refresh-token')
        with patch.object(self.vault, 'get_password', side_effect=RuntimeError('locked')):
            self.account._save()
            self.assert_round_trip('obfuscated')
        self.assertFalse(self.account._needs_save)

    def test_account_switch_reuses_key_and_logout_only_removes_this_config_key(self) -> None:
        self.account._save()
        other = self.new_account(self.path.parent.parent / 'other' / 'http-auth.json')
        other._credentials = credentials()
        other_credentials = dict(other._credentials)
        other._save()
        self.assertEqual(len(self.vault.entries), 2)
        own_key = self.vault.get_password(codex._KEYRING_SERVICE, self.account._key_id())
        self.account._update_tokens(credentials('another-account'))
        self.assertEqual(self.vault.get_password(codex._KEYRING_SERVICE, self.account._key_id()), own_key)
        legacy = self.path.with_name('auth.json')
        legacy.write_text('other-auth-data')
        self.account.logout(threading.Event())
        self.assertFalse(self.path.exists())
        self.assertEqual(self.account.cached_account_label, '')
        self.assertEqual(self.account._storage_mode, '')
        self.assertEqual(legacy.read_text(), 'other-auth-data')
        self.assertEqual(len(self.vault.entries), 1)
        other._load()
        self.assertEqual(other._credentials, other_credentials)

    def test_failed_logout_keeps_file_key_and_cached_account(self) -> None:
        self.account._save()
        unchanged = self.path.read_bytes()
        key = dict(self.vault.entries)
        with patch.object(Path, 'unlink', side_effect=PermissionError):
            with self.assertRaisesRegex(LLMUserActionRequiredError, 'removed'):
                self.account.logout(threading.Event())
        self.assertEqual(self.path.read_bytes(), unchanged)
        self.assertEqual(self.vault.entries, key)
        self.assertEqual(self.account._storage_mode, 'system')
        self.assertEqual(self.account.cached_account_label, 'private@example.com')

    def test_saved_mode_changes_only_after_successful_write(self) -> None:
        self.account._save()
        self.assertEqual(self.account._storage_mode, 'system')
        self.assertFalse(self.account._needs_save)
        with patch.object(codex, '_system_keyring', side_effect=ImportError):
            with patch.object(codex.os, 'replace', side_effect=PermissionError):
                with self.assertRaises(LLMUserActionRequiredError):
                    self.account._update_tokens({'refresh_token': 'rotated'}, self.account._credentials)
            self.assertEqual(self.account._storage_mode, 'system')
            self.assertTrue(self.account._needs_save)
            self.account._save()
        self.assertEqual(self.account._storage_mode, 'obfuscated')
        self.assertFalse(self.account._needs_save)
        self.account.logout(threading.Event())
        self.assertEqual(self.account._storage_mode, '')

    def test_headless_fallback_warning_is_generic_and_emitted_on_first_observation(self) -> None:
        with patch.object(codex, '_system_keyring', side_effect=RuntimeError('private backend details')), \
                patch.object(codex.LOGGER, 'warning') as warning:
            self.account._save()
            self.account._save()
            restarted = self.new_account()
            restarted._load()
            restarted._load()
        self.assertEqual([call.args[0] for call in warning.call_args_list], [
            'Codex credentials are stored with reversible obfuscation, not encryption.',
            'Codex credentials are stored with reversible obfuscation, not encryption.',
        ])

    def test_logout_succeeds_if_key_deletion_fails_and_logs_no_backend_secrets(self) -> None:
        self.account._save()
        with patch.object(self.vault, 'delete_password', side_effect=RuntimeError('private failure')), \
                patch.object(codex.LOGGER, 'warning') as warning:
            self.account.logout(threading.Event())
        self.assertFalse(self.path.exists())
        self.assertEqual(self.account.cached_account_label, '')
        self.assertNotIn('private failure', str(warning.call_args))
        self.assertTrue(warning.called)

    def test_cached_status_and_construction_do_not_access_storage(self) -> None:
        with patch.object(codex, '_system_keyring', side_effect=AssertionError('vault accessed')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('file accessed')):
            account = codex.CodexAccount()
            self.assertEqual(account._storage_mode, '')
            self.assertIsNone(account.cached_account_label)


class CodexNativeVaultSelectionTest(unittest.TestCase):
    def test_only_native_backends_can_receive_keys_even_inside_chainer(self) -> None:
        keyring = ModuleType('keyring')
        chainer = ModuleType('keyring.backends.chainer')
        chainer.ChainerBackend = type('ChainerBackend', (), {})
        fake_modules = {'keyring': keyring, 'keyring.backends.chainer': chainer}
        for module, name in (
            ('keyring.backends.Windows', 'WinVaultKeyring'),
            ('keyring.backends.macOS', 'Keyring'),
            ('keyring.backends.SecretService', 'Keyring'),
        ):
            backend = type(name, (MemoryVault,), {'__module__': module})()
            keyring.get_keyring = lambda: backend
            with self.subTest(module=module, name=name), patch.dict(sys.modules, fake_modules):
                self.assertIs(codex._system_keyring(), backend)
        plaintext = SimpleNamespace(priority=100, get_password=Mock(), set_password=Mock())
        unsupported = [type(name, (MemoryVault,), {'__module__': module})() for module, name in (
            ('keyrings.alt.file', 'PlaintextKeyring'),
            ('keyring.backends.null', 'Keyring'),
            ('keyring.backends.kwallet', 'DBusKeyring'),
        )]
        for selected in (plaintext, chainer.ChainerBackend(), *unsupported):
            selected.backends = (plaintext,)
            keyring.get_keyring = lambda: selected
            with patch.dict(sys.modules, fake_modules), self.assertRaisesRegex(RuntimeError, 'No supported'):
                codex._system_keyring()
        selected = chainer.ChainerBackend()
        selected.backends = (plaintext, backend)
        keyring.get_keyring = lambda: selected
        with patch.dict(sys.modules, fake_modules):
            self.assertIs(codex._system_keyring(), backend)
        plaintext.get_password.assert_not_called()
        plaintext.set_password.assert_not_called()


if __name__ == '__main__':
    unittest.main()
