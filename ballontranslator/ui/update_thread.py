import traceback

from qtpy.QtCore import QThread, Signal

from ballontranslator.utils.updater import BallonsTranslatorUpdater


class UpdateCheckThread(QThread):
    progress_changed = Signal(dict)
    update_finished = Signal(object)
    update_failed = Signal(str, str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.release_info = None
        self.current_version = None
        self._show_release_info = False
        self._busy = False
        self.finished.connect(self._clearBusy)

    def isBusy(self) -> bool:
        return self._busy or self.isRunning()

    def checkLatest(self, show_release_info: bool = False) -> bool:
        if self.isBusy():
            return False
        self._busy = True
        self.release_info = None
        self.current_version = None
        self._show_release_info = show_release_info
        self.start()
        return True

    def applyUpdate(self, release_info, current_version: str) -> bool:
        if self.isBusy():
            return False
        self._busy = True
        self.release_info = release_info
        self.current_version = current_version
        self._show_release_info = False
        self.start()
        return True

    def _clearBusy(self) -> None:
        self._busy = False

    def run(self) -> None:
        try:
            updater = BallonsTranslatorUpdater(progress_callback=self.progress_changed.emit)
            if self.release_info is None:
                if self._show_release_info:
                    result = updater.preview_cached_release()
                else:
                    result = updater.check_latest_release()
            else:
                result = updater.apply_update(self.release_info, self.current_version)
            self.update_finished.emit(result)
        except Exception as e:
            self.update_failed.emit(str(e), traceback.format_exc())
