"""Shared, asynchronous account controls for Codex profile cards."""

from __future__ import annotations

import importlib
import threading

from qtpy.QtCore import QObject, QThread, QUrl, Qt, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ballontranslator.modules.codex import (
    HTTP_REQUIREMENT, account, http_client_available,
)
from ballontranslator.modules.exceptions import LLMRequestStopped, LLMUserActionRequiredError
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import sync_codex_profile
from .package_manager import create_package_manager


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
        self.account = ''
        self.models = {}
        self.error = ''

    def run(self) -> None:
        try:
            if self.action == 'install':
                result = create_package_manager().install([HTTP_REQUIREMENT])
                if not result.ok:
                    self.error = self.tr('HTTP client installation failed. Check the package manager settings and try again.')
                importlib.invalidate_caches()
                return
            if self.action == 'logout':
                account.logout(self.stop_event)
                return
            if self.action == 'login':
                account.login(self.stop_event, self.login_url.emit)
            self.account, self.models = account.catalog(self.stop_event)
        except LLMRequestStopped:
            pass
        except LLMUserActionRequiredError as error:
            self.error = str(error)
        except Exception:
            # HTTP diagnostics can contain OAuth URLs; do not surface them.
            self.error = self.tr('Codex could not connect. Check the network, then refresh or sign in again.')


class CodexAccountController(QObject):
    """Own the single GUI account operation and publish catalog updates.

    >>> CodexAccountController.__name__
    'CodexAccountController'
    """

    changed = Signal()
    catalog_changed = Signal()

    def __init__(self, parent: QApplication) -> None:
        super().__init__(parent)
        self.worker = None
        self.account = ''
        self.status = self.unavailableMessage() or self.tr('Refresh to check the ChatGPT connection.')
        parent.aboutToQuit.connect(self.shutdown)

    @classmethod
    def instance(cls) -> 'CodexAccountController':
        app = QApplication.instance()
        controller = getattr(app, '_codex_account_controller', None)
        if controller is None:
            controller = cls(app)
            app._codex_account_controller = controller
        return controller

    def start(self, action: str) -> None:
        if self.worker is not None:
            return
        if action in ('login', 'refresh') and not http_client_available():
            self.status = self.unavailableMessage()
            self.changed.emit()
            return
        self.worker = CodexAccountWorker(action, self)
        self.worker.login_url.connect(self.openLoginUrl)
        self.worker.finished.connect(self.finish)
        self.status = self.tr('Installing HTTP client...') if action == 'install' else self.tr('Connecting to Codex...')
        self.changed.emit()
        self.worker.start()

    def unavailableMessage(self) -> str:
        if http_client_available():
            return ''
        return self.tr('Install the HTTP client from this card to connect to Codex.')

    def openLoginUrl(self, url: str) -> None:
        if self.worker is None or self.worker.stop_event.is_set():
            return
        self.status = self.tr('Complete ChatGPT sign-in in your browser.')
        if not QDesktopServices.openUrl(QUrl(url)):
            self.status = self.tr('Could not open the browser. Cancel and try signing in again.')
        self.changed.emit()

    def cancel(self) -> None:
        if self.worker is not None and self.worker.action != 'install':
            self.worker.stop_event.set()
            self.status = self.tr('Cancelling...')
            self.changed.emit()

    def finish(self) -> None:
        worker = self.worker
        if worker is None:
            return
        self.worker = None
        if worker.action == 'install':
            self.status = worker.error or self.tr('HTTP client installed. Sign in with ChatGPT to continue.')
        elif worker.stop_event.is_set():
            self.status = self.tr('Cancelled. Refresh to check the connection.')
        elif worker.error:
            self.status = worker.error
        else:
            self.account = worker.account
            self.status = self.tr('Connected: ') + self.account if self.account else self.tr('Not signed in.')
            if pcfg.module.codex_models != worker.models:
                pcfg.module.codex_models = worker.models
                for profile in pcfg.module.llm_profiles:
                    sync_codex_profile(profile, pcfg.module.codex_models)
                self.catalog_changed.emit()
        worker.deleteLater()
        self.changed.emit()

    def shutdown(self) -> None:
        account.invalidate()
        if self.worker is not None:
            self.worker.stop_event.set()
            # Network awaits are cancelled by the account generation change.
            # Package installation must finish before Qt exits.
            self.worker.wait()


class CodexAccountWidget(QWidget):
    """Display the shared account without network or credential IO during construction.

    >>> CodexAccountWidget.__name__
    'CodexAccountWidget'
    """

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.controller = CodexAccountController.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel(self)
        self.status_label.setObjectName('LLMProfileFieldLabel')
        self.status_label.setTextFormat(Qt.TextFormat.PlainText)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        row = QHBoxLayout()
        self.buttons = {}
        for action, text in (
            ('install', self.tr('Install HTTP client')),
            ('login', self.tr('Sign in with ChatGPT')),
            ('refresh', self.tr('Refresh models')),
            ('logout', self.tr('Sign out')),
            ('cancel', self.tr('Cancel')),
        ):
            button = QPushButton(text, self)
            button.setProperty('codexAction', action)
            button.clicked.connect(self.performAction)
            row.addWidget(button)
            self.buttons[action] = button
        row.addStretch(1)
        layout.addLayout(row)
        self.controller.changed.connect(self.refresh)
        self.refresh()

    def performAction(self) -> None:
        action = self.sender().property('codexAction')
        if action == 'cancel':
            self.controller.cancel()
        else:
            self.controller.start(action)

    def refresh(self) -> None:
        unavailable = not http_client_available()
        worker = self.controller.worker
        self.status_label.setText(self.controller.status)
        for action, button in self.buttons.items():
            if action == 'install':
                button.setVisible(bool(unavailable))
                button.setEnabled(worker is None)
            elif action == 'cancel':
                button.setVisible(worker is not None and worker.action != 'install')
                button.setEnabled(worker is not None and not worker.stop_event.is_set())
            else:
                button.setEnabled(worker is None and (action == 'logout' or not unavailable))
