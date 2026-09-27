"""Shared Codex sign-in controls and recovery dialog."""

from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .codex_account import CodexAccountController
from .misc import themed_icon_path


class CodexSignInButton(QPushButton):
    """Use the same browser sign-in affordance in settings and recovery."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setText(self.tr('Sign in'))
        self.setObjectName('CodexSignInButton')
        self.setToolTip(self.tr('Sign in with ChatGPT'))
        self.setIcon(QIcon(themed_icon_path('arrow-up-right.svg')))
        self.setIconSize(QSize(14, 14))
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)


class CodexSignInDialog(QDialog):
    """Keep authentication recovery open until cancelled or acknowledged.

    >>> CodexSignInDialog.__name__
    'CodexSignInDialog'
    """

    def __init__(self, controller: CodexAccountController, invalid: bool) -> None:
        super().__init__()
        self.controller = controller
        self.invalid = invalid
        self.setWindowTitle(self.tr('Codex sign-in'))
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        self.message = QLabel(self)
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.message)
        self.error = QLabel(self)
        self.error.setWordWrap(True)
        self.error.setTextFormat(Qt.TextFormat.PlainText)
        self.error.hide()
        layout.addWidget(self.error)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.sign_in_button = CodexSignInButton(self)
        self.sign_in_button.setObjectName('RunPipelinePrimaryButton')
        self.sign_in_button.clicked.connect(self.startSignIn)
        buttons.addWidget(self.sign_in_button)
        self.cancel_button = QPushButton(self.tr('Cancel'), self)
        self.cancel_button.setObjectName('RunPipelineSecondaryButton')
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.cancel_button)
        self.continue_button = QPushButton(self.tr('Continue'), self)
        self.continue_button.setObjectName('RunPipelinePrimaryButton')
        self.continue_button.clicked.connect(self.accept)
        buttons.addWidget(self.continue_button)
        layout.addLayout(buttons)
        controller.changed.connect(self.refreshAccount)
        self.refreshAccount()

    def setSignInRequired(self, invalid: bool) -> None:
        self.invalid = self.invalid or invalid
        self.refreshAccount()

    def startSignIn(self) -> None:
        self.error.clear()
        self.error.hide()
        self.controller.start('login')

    def refreshAccount(self) -> None:
        signed_in = bool(self.controller.account)
        if signed_in:
            self.message.setText(self.tr('You are signed in to Codex.'))
        elif self.invalid:
            self.message.setText(self.tr('Your Codex sign-in is no longer valid. Please sign in again.'))
        else:
            self.message.setText(self.tr('You need to sign in to use Codex.'))
        self.sign_in_button.setVisible(not signed_in)
        self.sign_in_button.setEnabled(self.controller.worker is None)
        self.cancel_button.setVisible(not signed_in)
        self.continue_button.setVisible(signed_in)

    def showError(self, message: str) -> None:
        self.error.setText(message)
        self.error.show()

    def reject(self) -> None:
        worker = self.controller.worker
        if worker is not None and worker.action == 'login':
            self.controller.cancel()
        super().reject()
