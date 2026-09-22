"""Shared, asynchronous account controls for Codex settings."""

from __future__ import annotations

import threading

from typing import Dict, Optional, TYPE_CHECKING

from qtpy.QtCore import QObject, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QApplication, QMessageBox

from ballontranslator.modules.codex import account
from ballontranslator.modules.exceptions import CodexSignInRequiredError, LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import profile_by_id, sync_codex_profile
from ballontranslator.utils.logger import logger as LOGGER

if TYPE_CHECKING:
    from .codex_sign_in import CodexSignInDialog


def show_codex_sign_in_required(error: CodexSignInRequiredError) -> None:
    """Forward worker failures to the existing GUI account owner without doing IO."""
    LOGGER.error(str(error))
    if shared.HEADLESS:
        return
    controller = getattr(QApplication.instance(), '_codex_account_controller', None)
    if controller is not None:
        controller.sign_in_required.emit(error.invalid)


class CodexAccountWorker(QThread):
    """One explicit account operation, retained by the application until finished.

    >>> CodexAccountWorker.__name__
    'CodexAccountWorker'
    """

    login_url = Signal(str)

    def __init__(self, action: str, parent: QObject) -> None:
        super().__init__(parent)
        self.action = action
        self.stop_event = threading.Event()
        self.account: Optional[str] = None
        self.models: Optional[Dict] = None
        self.error: Optional[Exception] = None

    def run(self) -> None:
        try:
            if self.action == 'logout':
                account.logout(self.stop_event)
                return
            if self.action == 'login':
                account.login(self.stop_event, self.login_url.emit)
            self.models = account.catalog(self.stop_event)
        except LLMRequestStopped:
            pass
        except CodexSignInRequiredError as error:
            # Startup discovers local state silently; explicit refresh requires sign-in.
            if self.action != 'restore' or error.invalid:
                self.error = error.with_traceback(None)
        except LLMUserActionRequiredError as error:
            self.error = error.with_traceback(None)
        except Exception:
            # HTTP diagnostics can contain OAuth URLs; do not surface them.
            self.error = RuntimeError(self.tr('Codex could not connect. Check the network, then refresh or sign in again.'))
        finally:
            # Login/logout can commit before a later refresh failure or cancel.
            self.account = account.cached_account_label


class CodexAccountController(QObject):
    """Own the single GUI account operation and publish catalog updates.

    >>> CodexAccountController.__name__
    'CodexAccountController'
    """

    changed = Signal()
    catalog_changed = Signal()
    sign_in_required = Signal(bool)

    def __init__(self, parent: QApplication) -> None:
        super().__init__(parent)
        self.worker = None
        self.account = account.cached_account_label or ''
        self._sign_in_dialog: Optional[CodexSignInDialog] = None
        self.sign_in_required.connect(self.showSignInRequired)
        parent.aboutToQuit.connect(self.shutdown)

    @classmethod
    def instance(cls) -> 'CodexAccountController':
        app = QApplication.instance()
        controller = getattr(app, '_codex_account_controller', None)
        if controller is None:
            controller = cls(app)
            app._codex_account_controller = controller
        return controller

    def restoreSession(self) -> None:
        """Restore saved sign-in through the normal background catalog refresh."""
        self.start('restore')

    def start(self, action: str) -> None:
        if self.worker is not None:
            return
        if action == 'refresh' and account.cached_account_label == '':
            self.sign_in_required.emit(account.auth_invalid)
            return
        self.worker = CodexAccountWorker(action, self)
        self.worker.login_url.connect(self.openLoginUrl)
        self.worker.finished.connect(self.finish)
        self.changed.emit()
        self.worker.start()

    def openLoginUrl(self, url: str) -> None:
        if self.worker is None or self.worker.stop_event.is_set():
            return
        if not QDesktopServices.openUrl(QUrl(url)):
            self.cancel()
            self.showAccountError(self.tr('Could not open the browser. Check your default browser settings and try signing in again.'))

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.stop_event.set()
            self.changed.emit()

    def finish(self) -> None:
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        if worker.account is not None:
            self.account = worker.account
        # Public model metadata survives logout; only an authenticated catalog
        # result replaces it, including a successful empty response.
        models = worker.models if worker.account else None
        if models is not None and (
            pcfg.module.codex_models != models
            or (worker.action == 'login' and not worker.error and not worker.stop_event.is_set())
        ):
            pcfg.module.codex_models = models
            sync_codex_profile(profile_by_id(pcfg.module.llm_profiles, 'codex'), models)
            self.catalog_changed.emit()
        error = None if worker.stop_event.is_set() else worker.error
        worker.deleteLater()
        self.changed.emit()
        if isinstance(error, CodexSignInRequiredError):
            self.sign_in_required.emit(error.invalid)
        elif error:
            self.showAccountError(str(error))

    def showSignInRequired(self, invalid: bool) -> None:
        from .codex_sign_in import CodexSignInDialog

        self.account = account.cached_account_label or ''
        self.changed.emit()
        if self._sign_in_dialog is None:
            # Controller ownership keeps recovery alive after a worker or progress dialog closes.
            self._sign_in_dialog = CodexSignInDialog(self, invalid)
            self._sign_in_dialog.finished.connect(self._clearSignInDialog)
            self._sign_in_dialog.show()
        else:
            self._sign_in_dialog.setSignInRequired(invalid)
        self._sign_in_dialog.raise_()
        self._sign_in_dialog.activateWindow()

    def _clearSignInDialog(self) -> None:
        dialog = self._sign_in_dialog
        self._sign_in_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def showAccountError(self, message: str) -> None:
        if self._sign_in_dialog is not None:
            self._sign_in_dialog.showError(message)
        else:
            QMessageBox.warning(QApplication.activeWindow(), self.tr('Codex'), message)

    def shutdown(self) -> None:
        if self._sign_in_dialog is not None:
            self._sign_in_dialog.reject()
        account.invalidate()
        if self.worker is not None:
            self.worker.stop_event.set()
            # Network awaits are cancelled by the account generation change.
            self.worker.wait()
