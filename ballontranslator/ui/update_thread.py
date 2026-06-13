import traceback

from qtpy.QtCore import QThread, Signal

from ballontranslator.utils.updater import BallonsTranslatorUpdater


class UpdateCheckThread(QThread):
    progress_changed = Signal(dict)
    update_finished = Signal(object)
    update_failed = Signal(str, str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        try:
            updater = BallonsTranslatorUpdater(progress_callback=self.progress_changed.emit)
            self.update_finished.emit(updater.check_and_update())
        except Exception as e:
            self.update_failed.emit(str(e), traceback.format_exc())
