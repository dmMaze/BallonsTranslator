import gc
import os
import threading
import unittest
import weakref
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.modules import codex
from ballontranslator.modules.exceptions import CodexSignInRequiredError, LLMRequestStopped
from ballontranslator.ui import codex_account
from ballontranslator.ui.codex_settings import CodexSettingsPanel
from ballontranslator.utils import shared
from ballontranslator.utils.config import ModuleConfig, pcfg


class CodexSignInDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.account = codex.CodexAccount()
        self.account._loaded = True
        for patcher in (
            patch.object(codex, 'account', self.account),
            patch.object(codex_account, 'account', self.account),
            patch.object(pcfg, 'module', ModuleConfig()),
            patch.object(shared, 'HEADLESS', False),
            patch.object(self.app, '_codex_account_controller', None, create=True),
            patch.object(codex, '_http_client', side_effect=AssertionError('Unexpected HTTP')),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.controller = codex_account.CodexAccountController.instance()
        warning = patch('ballontranslator.ui.codex_account.QMessageBox.warning')
        self.warning = warning.start()
        self.addCleanup(warning.stop)

    def tearDown(self) -> None:
        if self.controller._sign_in_dialog is not None:
            self.controller._sign_in_dialog.reject()
        if self.controller.worker is not None:
            self.controller.cancel()
            self.finish_worker()
        self.controller.deleteLater()
        self.drain_deletes()
        self.doCleanups()
        super().tearDown()

    def drain_deletes(self) -> None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        gc.collect()

    def finish_worker(self) -> None:
        worker = self.controller.worker
        self.assertIsNotNone(worker)
        self.assertTrue(worker.wait(2000))
        self.app.processEvents()
        self.assertIsNone(self.controller.worker)

    def commit_login(self, stop_event: threading.Event, show_url) -> None:
        self.account._credentials = {'account_id': 'signed-in-account'}
        self.account.auth_invalid = False

    def test_unsigned_refresh_opens_dialog_immediately_and_startup_stays_quiet(self) -> None:
        self.controller.restoreSession()
        self.finish_worker()
        self.assertIsNone(self.controller._sign_in_dialog)
        with patch.object(self.account, 'catalog', side_effect=AssertionError('Unexpected catalog request')):
            self.controller.start('refresh')
        dialog = self.controller._sign_in_dialog
        self.assertIsNotNone(dialog)
        self.assertIsNone(self.controller.worker)
        self.assertIn('need to sign in', dialog.message.text())
        self.assertFalse(dialog.sign_in_button.isHidden())
        self.assertTrue(dialog.continue_button.isHidden())
        self.warning.assert_not_called()

    def test_worker_failure_dispatches_one_invalid_dialog_on_gui_thread(self) -> None:
        self.account.auth_invalid = True
        thread = threading.Thread(target=codex_account.show_codex_sign_in_required,
                                  args=(CodexSignInRequiredError(invalid=True),))
        thread.start()
        thread.join(2)
        self.assertIsNone(self.controller._sign_in_dialog)
        self.app.processEvents()
        dialog = self.controller._sign_in_dialog
        self.assertIs(dialog.thread(), self.app.thread())
        self.assertIn('no longer valid', dialog.message.text())
        codex_account.show_codex_sign_in_required(CodexSignInRequiredError(invalid=True))
        self.assertIs(self.controller._sign_in_dialog, dialog)
        self.assertEqual(self.controller.account, '')
        self.warning.assert_not_called()

    def test_rejected_model_refresh_opens_invalid_signin_dialog(self) -> None:
        self.commit_login(threading.Event(), None)
        self.controller.account = self.account.cached_account_label

        def reject_catalog(stop_event: threading.Event) -> None:
            self.account.auth_invalid = True
            raise CodexSignInRequiredError(invalid=True)

        with patch.object(self.account, 'catalog', side_effect=reject_catalog) as refresh:
            self.controller.start('refresh')
            self.finish_worker()
        refresh.assert_called_once()
        dialog = self.controller._sign_in_dialog
        self.assertIn('no longer valid', dialog.message.text())
        self.assertFalse(dialog.sign_in_button.isHidden())
        self.assertTrue(dialog.continue_button.isHidden())
        self.assertEqual(self.controller.account, '')
        self.warning.assert_not_called()

    def test_signin_updates_same_dialog_to_continue_and_releases_it(self) -> None:
        panel = CodexSettingsPanel()
        self.addCleanup(panel.deleteLater)
        self.controller.start('refresh')
        dialog = self.controller._sign_in_dialog
        catalog = {'new-model': {'modalities': ['text', 'image'], 'efforts': ['high']}}
        with patch.object(self.account, 'login', side_effect=self.commit_login) as login, \
                patch.object(self.account, 'catalog', return_value=catalog) as refresh:
            dialog.sign_in_button.click()
            self.assertFalse(dialog.sign_in_button.isEnabled())
            self.finish_worker()
        login.assert_called_once()
        refresh.assert_called_once()
        self.assertIs(self.controller._sign_in_dialog, dialog)
        self.assertIn('You are signed in', dialog.message.text())
        self.assertTrue(dialog.sign_in_button.isHidden())
        self.assertTrue(dialog.cancel_button.isHidden())
        self.assertFalse(dialog.continue_button.isHidden())
        self.assertFalse(panel.account_status.isHidden())
        self.assertEqual(pcfg.module.codex_models, catalog)
        references = [weakref.ref(dialog), weakref.ref(dialog.sign_in_button)]
        with patch.object(self.controller, 'start') as restart:
            dialog.continue_button.click()
        restart.assert_not_called()
        self.assertIsNone(self.controller._sign_in_dialog)
        del dialog
        self.drain_deletes()
        self.assertTrue(all(ref() is None for ref in references))

    def test_login_failure_stays_in_dialog_and_can_be_retried(self) -> None:
        self.controller.start('refresh')
        dialog = self.controller._sign_in_dialog
        with patch.object(self.account, 'login', side_effect=RuntimeError('private provider data')):
            dialog.sign_in_button.click()
            self.finish_worker()
        self.assertTrue(dialog.sign_in_button.isEnabled())
        self.assertFalse(dialog.error.isHidden())
        self.assertNotIn('private provider data', dialog.error.text())
        self.assertTrue(dialog.continue_button.isHidden())
        self.warning.assert_not_called()

    def test_cancel_closes_dialog_and_stops_ongoing_signin_without_reopening(self) -> None:
        started = threading.Event()

        def login(stop_event: threading.Event, show_url) -> None:
            started.set()
            stop_event.wait(2)
            raise LLMRequestStopped()

        self.controller.start('refresh')
        dialog = self.controller._sign_in_dialog
        with patch.object(self.account, 'login', side_effect=login):
            dialog.sign_in_button.click()
            self.assertTrue(started.wait(2))
            worker = self.controller.worker
            dialog.cancel_button.click()
            self.assertTrue(worker.stop_event.is_set())
            self.finish_worker()
        self.assertIsNone(self.controller._sign_in_dialog)
        self.warning.assert_not_called()

    def test_committed_signin_succeeds_even_if_catalog_refresh_fails(self) -> None:
        self.controller.start('refresh')
        dialog = self.controller._sign_in_dialog
        with patch.object(self.account, 'login', side_effect=self.commit_login), \
                patch.object(self.account, 'catalog', side_effect=RuntimeError('catalog failure')):
            dialog.sign_in_button.click()
            self.finish_worker()
        self.assertFalse(dialog.continue_button.isHidden())
        self.assertTrue(dialog.cancel_button.isHidden())
        self.assertFalse(dialog.error.isHidden())
        self.warning.assert_not_called()

    def test_headless_auth_failure_never_creates_a_dialog(self) -> None:
        with patch.object(shared, 'HEADLESS', True):
            codex_account.show_codex_sign_in_required(CodexSignInRequiredError())
        self.assertIsNone(self.controller._sign_in_dialog)


if __name__ == '__main__':
    unittest.main()
