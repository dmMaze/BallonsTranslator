from typing import List, Optional

from qtpy.QtCore import QThread, Signal

from .package_manager import create_package_manager

class PackageInstallThread(QThread):

    finish_install = Signal()
    package_prepare_progress = Signal(dict)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requirements = []
        self.torch_device = None
        self.torch_cuda_version = None
        self.last_success = False
        self.last_error = None

    def installPackages(
        self,
        requirements: List[str],
        torch_device: Optional[str] = None,
        torch_cuda_version: Optional[str] = None,
    ) -> bool:
        if self.isRunning():
            return False
        self.requirements = list(dict.fromkeys(requirements))
        self.torch_device = torch_device
        self.torch_cuda_version = torch_cuda_version
        self.last_success = False
        self.last_error = None
        self.start()
        return True

    def _emit_prepare_progress(self, payload: dict):
        self.package_prepare_progress.emit(dict(payload))

    def run(self):
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(
            self.requirements,
            progress_callback=self._emit_prepare_progress,
            torch_device=self.torch_device,
            torch_cuda_version=self.torch_cuda_version,
        )
        self.last_success = result.ok
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(self.requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
        self.finish_install.emit()
