import sys
from typing import Iterable, Optional, Tuple

from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .package_manager import create_package_manager
from ballontranslator.modules.lazy_registry import probe_torch_package
from ballontranslator.utils.torch_install_helper import TORCH_CUDA_VERSION_OPTIONS, TORCH_INSTALL_DEVICE_OPTIONS


CUDA_VERSION_LABELS = {
    'cu128': '12.8',
    'cu118': '11.8',
}

NOTE_STYLE = '''
QFrame#TorchInstallNote {
    background-color: rgba(30, 147, 229, 24);
    border-left: 2px solid rgb(30, 147, 229);
    border-radius: 2px;
}
QLabel#TorchInstallNoteTitle {
    font-weight: 600;
    color: rgb(30, 147, 229);
}
'''


class TorchInstallHelperDialog(QDialog):
    """Confirm the torch wheel target before installing torch-family packages.

    >>> dialog = TorchInstallHelperDialog(['torch'], 'cpu')  # doctest: +SKIP
    >>> dialog.selected_device()  # doctest: +SKIP
    'cpu'
    """

    def __init__(
        self,
        requirements: Iterable[str],
        initial_device: str,
        initial_cuda_version: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.requirements = list(dict.fromkeys(requirements))
        self.package_manager = create_package_manager()
        self.torch_is_installed = probe_torch_package()[0] is not None

        self.setWindowTitle(self.tr('Torch Install Helper'))
        layout = QVBoxLayout(self)

        target_row = QHBoxLayout()
        device_label = QLabel(self.tr('Device'), self)
        self.device_combo = QComboBox(self)
        for device in TORCH_INSTALL_DEVICE_OPTIONS:
            self.device_combo.addItem(device, device)
        initial_index = self.device_combo.findData(initial_device)
        if initial_index < 0:
            initial_index = self.device_combo.findData('cpu')
        self.device_combo.setCurrentIndex(max(initial_index, 0))
        target_row.addWidget(device_label)
        target_row.addWidget(self.device_combo, 1)

        cuda_label = QLabel(self.tr('CUDA'), self)
        self.cuda_combo = QComboBox(self)
        for cuda_version in TORCH_CUDA_VERSION_OPTIONS:
            label = CUDA_VERSION_LABELS.get(cuda_version, cuda_version)
            self.cuda_combo.addItem(label, cuda_version)
        initial_cuda_index = self.cuda_combo.findData(initial_cuda_version or 'cu128')
        if initial_cuda_index < 0:
            initial_cuda_index = self.cuda_combo.findData('cu128')
        self.cuda_combo.setCurrentIndex(max(initial_cuda_index, 0))
        target_row.addWidget(cuda_label)
        target_row.addWidget(self.cuda_combo, 1)
        layout.addLayout(target_row)

        self.note_text = QLabel(self)
        self.note_text.setWordWrap(True)
        layout.addWidget(self._note_widget(self.note_text))

        command_label = QLabel(self.tr('Install command'), self)
        layout.addWidget(command_label)
        self.command_preview = QPlainTextEdit(self)
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumSize(620, 120)
        layout.addWidget(self.command_preview)

        button_row = QHBoxLayout()
        button_row.addStretch()
        confirm_btn = QPushButton(self.tr('Confirm'), self)
        cancel_btn = QPushButton(self.tr('Cancel'), self)
        confirm_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(confirm_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

        self.device_combo.currentIndexChanged.connect(self.update_cuda_controls)
        self.cuda_combo.currentIndexChanged.connect(self.update_command_preview)
        self.update_cuda_controls()
        self.update_command_preview()

    def selected_device(self) -> str:
        return self.device_combo.currentData() or 'cpu'

    def selected_cuda_version(self) -> Optional[str]:
        if self.selected_device() != 'cuda':
            return None
        return self.cuda_combo.currentData() or 'cu128'

    def update_cuda_controls(self):
        self.cuda_combo.setEnabled(self.selected_device() == 'cuda')
        self.update_note_text()
        self.update_command_preview()

    def update_note_text(self):
        notes = [self.tr(
            'NVIDIA: choose CUDA. Intel Arc/Core Ultra: choose XPU. Not sure: choose CPU.'
        )]
        if self.selected_device() == 'cuda':
            notes.append(self.tr(
                'CUDA 12.8 is for RTX 20 / GTX 16 or newer. CUDA 11.8 is for GTX 10 or older.'
            ))
        if self.torch_is_installed:
            notes.append(self.tr(
                'Existing Torch packages will be removed before installation. If Torch is already loaded, '
                'restart the app before installing; otherwise removal may fail. '
                'Restart after a successful installation to use the new Torch version.'
            ))
        self.note_text.setText('\n\n'.join(notes))

    def update_command_preview(self):
        try:
            command = self.package_manager.preview_command(
                self.requirements,
                torch_device=self.selected_device(),
                torch_cuda_version=self.selected_cuda_version(),
            )
        except Exception as e:
            command = str(e)
        self.command_preview.setPlainText(command)

    def _note_widget(self, content_label: QLabel) -> QFrame:
        frame = QFrame(self)
        frame.setObjectName('TorchInstallNote')
        frame.setStyleSheet(NOTE_STYLE)
        note_layout = QVBoxLayout(frame)
        note_layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel(self.tr('NOTE'), frame)
        title.setObjectName('TorchInstallNoteTitle')
        note_layout.addWidget(title)
        note_layout.addWidget(content_label)
        return frame


def confirm_torch_install_device(
    requirements: Iterable[str],
    parent: Optional[QWidget] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return the user-confirmed torch install target, or cancel.

    >>> confirm_torch_install_device(['einops'])  # doctest: +SKIP
    (True, None, None)
    """

    package_manager = create_package_manager()
    if not _supports_torch_install_dialog() or not package_manager.needs_torch_install_choice(requirements):
        return True, None, None
    dialog = TorchInstallHelperDialog(
        requirements,
        package_manager.torch_install_device(requirements),
        package_manager.torch_install_cuda_version(requirements),
        parent=parent,
    )
    accepted = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Accepted')
    if dialog.exec() != accepted:
        return False, None, None
    return True, dialog.selected_device(), dialog.selected_cuda_version()


def _supports_torch_install_dialog() -> bool:
    """Return whether the torch install helper should be shown on this platform.

    >>> isinstance(_supports_torch_install_dialog(), bool)
    True
    """

    return sys.platform in {'win32', 'linux'}
