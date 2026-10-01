"""Codex-owned authentication UI; requests run outside the Qt thread."""

from dataclasses import replace
import threading
from typing import Optional

from qtpy.QtCore import Qt, QThread, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QCloseEvent
from qtpy.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)

from .framelesswindow import DialogCloseButton, OutsideClickFramelessMixin
from ballontranslator.modules.exceptions import LLMRequestStopped
from ballontranslator.modules.llm_codex import authenticate_codex
from ballontranslator.utils.llm_profiles import LLMProfile


class CodexLoginWorker(QThread):
    """Own one cancellable SDK login operation.

    >>> issubclass(CodexLoginWorker, QThread)
    True
    """

    challenge = Signal(dict)
    account_ready = Signal(dict)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, profile: LLMProfile, method: str, api_key: str,
                 parent: QWidget) -> None:
        super().__init__(parent)
        self.profile = replace(profile, codex_timeout=(
            max(300, profile.codex_timeout) if method in ('chatgpt', 'chatgptDeviceCode')
            else profile.codex_timeout
        ))
        self.method = method
        self.api_key = api_key
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            account = authenticate_codex(
                self.profile, self.method, self.api_key,
                self.challenge.emit, self.stop_event,
            )
            self.account_ready.emit(account)
        except LLMRequestStopped:
            self.cancelled.emit()
        except Exception as error:
            detail = str(error)
            if self.api_key.strip():
                detail = detail.replace(self.api_key.strip(), '[redacted]')
            self.failed.emit(detail)
        finally:
            self.api_key = ''

    def cancel(self) -> None:
        self.stop_event.set()


class CodexLoginDialog(OutsideClickFramelessMixin, QDialog):
    """Manage the shared Codex login without persisting profile credentials.

    >>> issubclass(CodexLoginDialog, QDialog)
    True
    """

    def __init__(self, profile: LLMProfile, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.profile = replace(profile)
        self.worker: Optional[CodexLoginWorker] = None
        self._closing = False
        self.setWindowTitle(self.tr('Codex Login'))
        self.setObjectName('ModuleParamDialog')
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumWidth(480)
        root = QVBoxLayout(self)
        surface = QFrame(self)
        surface.setObjectName('ModuleParamSurface')
        root.addWidget(surface)
        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        self.title_bar = QWidget(surface)
        title = QHBoxLayout(self.title_bar)
        title.addWidget(QLabel(self.tr('Codex Login'), self.title_bar))
        title.addStretch()
        self.close_button = DialogCloseButton(self.title_bar)
        self.close_button.clicked.connect(self.reject)
        title.addWidget(self.close_button)
        layout.addWidget(self.title_bar)
        note = QLabel(self.tr(
            'Uses the Codex login shared by applications with the same CODEX_HOME. '
            'Signing in replaces that login. API Key uses separate API billing.'
        ), surface)
        note.setWordWrap(True)
        layout.addWidget(note)
        self.method = QComboBox(surface)
        for label, value in (
            (self.tr('Reuse existing login'), 'reuse'),
            (self.tr('Browser login'), 'chatgpt'),
            (self.tr('Device code'), 'chatgptDeviceCode'),
            (self.tr('API Key'), 'apiKey'),
        ):
            self.method.addItem(label, value)
        layout.addWidget(self.method)
        self.api_key = QLineEdit(surface)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText(self.tr('API Key (saved by Codex, not in this profile)'))
        self.api_key.setAccessibleName(self.tr('API Key'))
        self.api_key.hide()
        layout.addWidget(self.api_key)
        self.status = QLabel(self.tr('Select a login method and continue.'), surface)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.url = QLineEdit(surface)
        self.url.setReadOnly(True)
        self.url.setAccessibleName(self.tr('Login URL'))
        self.url.hide()
        layout.addWidget(self.url)
        self.code = QLineEdit(surface)
        self.code.setReadOnly(True)
        self.code.setAccessibleName(self.tr('Device code'))
        self.code.hide()
        layout.addWidget(self.code)
        self.open_browser = QPushButton(self.tr('Open browser'), surface)
        self.open_browser.clicked.connect(self._open_browser)
        self.open_browser.hide()
        layout.addWidget(self.open_browser)
        buttons = QHBoxLayout()
        self.continue_button = QPushButton(self.tr('Continue'), surface)
        self.continue_button.clicked.connect(self._start)
        buttons.addWidget(self.continue_button)
        self.cancel_button = QPushButton(self.tr('Cancel login'), surface)
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.method.currentIndexChanged.connect(self._method_changed)

    def _method_changed(self) -> None:
        self.api_key.clear()
        self.api_key.setVisible(self.method.currentData() == 'apiKey')

    def _start(self) -> None:
        if self.worker is not None:
            return
        method = self.method.currentData()
        key = self.api_key.text().strip()
        if method == 'apiKey' and not key:
            self.status.setText(self.tr('Enter an API key.'))
            return
        self.url.clear()
        self.code.clear()
        self.url.hide()
        self.code.hide()
        self.open_browser.hide()
        self.worker = CodexLoginWorker(self.profile, method, key, self)
        self.api_key.clear()
        self.worker.challenge.connect(self._challenge)
        self.worker.account_ready.connect(self._account_ready)
        self.worker.failed.connect(self.status.setText)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.finished.connect(self._finished)
        self.method.setEnabled(False)
        self.api_key.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText(self.tr('Checking login…') if method == 'reuse' else self.tr('Signing in…'))
        self.worker.start()

    def _challenge(self, challenge: dict) -> None:
        if self._closing or self.worker is None or self.worker.stop_event.is_set():
            return
        self.url.setText(challenge['url'])
        self.url.show()
        self.code.setText(challenge['code'])
        self.code.setVisible(bool(challenge['code']))
        self.open_browser.show()
        self.status.setText(self.tr('Complete login in your browser. Enter the device code if shown.'))
        if self.worker.method == 'chatgpt':
            self._open_browser()

    def _open_browser(self) -> None:
        url = QUrl(self.url.text())
        if url.scheme() in ('http', 'https') and not QDesktopServices.openUrl(url):
            self.status.setText(self.tr('Could not open the browser. Copy the login URL instead.'))

    def _account_ready(self, account: dict) -> None:
        kind = account.get('type')
        if kind == 'chatgpt':
            self.status.setText(self.tr('Signed in with ChatGPT: {account}').format(
                account=account.get('email') or account.get('planType') or 'ChatGPT',
            ))
        elif kind == 'apiKey':
            self.status.setText(self.tr('Signed in with API Key (API billing).'))
        else:
            self.status.setText(self.tr('No existing login. Choose a sign-in method.'))

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status.setText(self.tr('Cancelling login…'))
            self.cancel_button.setEnabled(False)

    def _cancelled(self) -> None:
        self.status.setText(self.tr('Login cancelled.'))

    def _finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
        self.method.setEnabled(True)
        self.api_key.setEnabled(True)
        self.continue_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.url.clear()
        self.code.clear()
        self.url.hide()
        self.code.hide()
        self.open_browser.hide()
        if self._closing:
            super().reject()

    def reject(self) -> None:
        # Do not destroy a QThread while SDK cleanup is still running.
        if self.worker is not None:
            self._closing = True
            self._cancel()
            return
        self.api_key.clear()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            event.ignore()
            self.reject()
        else:
            super().closeEvent(event)

    def _dismiss_transient_window(self) -> None:
        self.reject()

    def _preserve_on_outside_click(self) -> bool:
        return True
