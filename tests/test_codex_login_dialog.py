import time
import unittest
import weakref
from unittest import mock

from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtWidgets import QApplication

from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.ui.codex_login_dialog import CodexLoginDialog
from ballontranslator.utils.llm_profiles import default_profile


class CodexLoginDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.dialog = CodexLoginDialog(default_profile('Codex'))
        self.dialog.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        if self.dialog.worker is not None:
            self.dialog.worker.cancel()
            self.dialog.worker.wait(5000)
            self.app.processEvents()
        self.dialog.reject()
        self.dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def wait_until(self, predicate) -> None:
        end = time.monotonic() + 5
        while not predicate() and time.monotonic() < end:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertTrue(predicate())

    def test_reuse_is_lazy_and_api_key_is_cleared(self) -> None:
        with mock.patch('ballontranslator.ui.codex_login_dialog.authenticate_codex',
                        return_value={'type': 'apiKey'}) as authenticate:
            authenticate.assert_not_called()
            self.dialog.method.setCurrentIndex(3)
            self.dialog.api_key.setText('test-secret')
            self.dialog._start()
            self.assertEqual(self.dialog.api_key.text(), '')
            self.wait_until(lambda: self.dialog.worker is None)
            self.assertEqual(authenticate.call_args.args[1:3], ('apiKey', 'test-secret'))
            self.assertIn('API Key', self.dialog.status.text())
            self.assertTrue(self.dialog.continue_button.isEnabled())

    def test_device_code_stays_visible_until_cancel_and_close_waits(self) -> None:
        def login(profile, method, key, challenge, stop):
            challenge({'url': 'https://example.com/device', 'code': 'ABCD-1234'})
            stop.wait(5)
            raise LLMRequestStopped()

        with mock.patch('ballontranslator.ui.codex_login_dialog.authenticate_codex', side_effect=login), \
                mock.patch('ballontranslator.ui.codex_login_dialog.QDesktopServices.openUrl') as browser:
            self.dialog.method.setCurrentIndex(2)
            self.dialog._start()
            worker = self.dialog.worker
            worker_ref = weakref.ref(worker)
            self.wait_until(lambda: self.dialog.code.text() == 'ABCD-1234')
            self.assertTrue(self.dialog.code.isVisible())
            browser.assert_not_called()
            self.dialog.reject()
            self.assertIs(self.dialog.worker, worker)
            self.wait_until(lambda: self.dialog.worker is None)
            self.assertFalse(self.dialog.isVisible())
            self.assertEqual(self.dialog.code.text(), '')
            del worker
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()
            self.assertIsNone(worker_ref())

    def test_browser_login_opens_url_and_displays_result(self) -> None:
        def login(profile, method, key, challenge, stop):
            challenge({'url': 'https://example.com/login', 'code': ''})
            return {'type': 'chatgpt', 'email': 'test@example.com'}

        with mock.patch('ballontranslator.ui.codex_login_dialog.authenticate_codex', side_effect=login), \
                mock.patch('ballontranslator.ui.codex_login_dialog.QDesktopServices.openUrl', return_value=True) as browser:
            self.dialog.method.setCurrentIndex(1)
            self.dialog._start()
            self.wait_until(lambda: self.dialog.worker is None)
            browser.assert_called_once()
            self.assertIn('test@example.com', self.dialog.status.text())

    def test_worker_error_never_displays_api_key(self) -> None:
        with mock.patch('ballontranslator.ui.codex_login_dialog.authenticate_codex',
                        side_effect=RuntimeError('Rejected test-secret')):
            self.dialog.method.setCurrentIndex(3)
            self.dialog.api_key.setText('test-secret')
            self.dialog._start()
            self.wait_until(lambda: self.dialog.worker is None)
            self.assertNotIn('test-secret', self.dialog.status.text())
            self.assertIn('[redacted]', self.dialog.status.text())


if __name__ == '__main__':
    unittest.main()
