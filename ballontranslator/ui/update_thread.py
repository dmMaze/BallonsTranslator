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

    def checkLatest(self) -> None:
        self.release_info = None
        self.current_version = None
        self.start()

    def applyUpdate(self, release_info, current_version: str) -> None:
        self.release_info = release_info
        self.current_version = current_version
        self.start()

    def run(self) -> None:
        try:
            updater = BallonsTranslatorUpdater(progress_callback=self.progress_changed.emit)
            if self.release_info is None:
                result = updater.check_latest_release()
            else:
                result = updater.apply_update(self.release_info, self.current_version)
            self.update_finished.emit(result)
        except Exception as e:
            self.update_failed.emit(str(e), traceback.format_exc())
