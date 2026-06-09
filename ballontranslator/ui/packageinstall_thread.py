from typing import List

from qtpy.QtCore import QThread, Signal

from .package_manager import create_package_manager

class PackageInstallThread(QThread):

    finish_install = Signal()
    package_prepare_progress = Signal(dict)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requirements = []
        self.last_success = False
        self.last_error = None

    def installPackages(self, requirements: List[str]) -> bool:
        if self.isRunning():
            return False
        self.requirements = list(dict.fromkeys(requirements))
        self.last_success = False
        self.last_error = None
        self.start()
        return True

    def _emit_prepare_progress(self, payload: dict):
        self.package_prepare_progress.emit(dict(payload))

    def run(self):
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(self.requirements, progress_callback=self._emit_prepare_progress)
        self.last_success = result.ok
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(self.requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
        self.finish_install.emit()
