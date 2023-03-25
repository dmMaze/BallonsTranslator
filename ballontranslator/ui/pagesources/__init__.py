from gallery_dl.job import DownloadJob
from gallery_dl import config
from qtpy.QtCore import Signal, QThread
from utils.logger import logger as LOGGER
from ui.misc import ProgramConfig
from ui.imgtrans_proj import ProjImgTrans
from ui.constants import DOWNLOAD_PATH
import os

class SourceDownload(QThread):
    open_downloaded_proj = Signal(str)
    update_progress_bar = Signal(int)
    finished_downloading = Signal()

    def __init__(self, config: ProgramConfig, imgtrans_proj: ProjImgTrans, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = None
        self.config_pnl = config
        self.imgtrans_proj = imgtrans_proj
        self.path = f'{DOWNLOAD_PATH}/'
        self.url = ''

    def ValidateUrl(self):
        if 'https://' not in self.url:
            self.url = 'https://' + self.url

    def PassUrlToImgTransProj(self):
        self.imgtrans_proj.src_download_link = self.url

    def FetchImages(self):
        config.load()
        config.set((), "skip", False)
        job = DownloadJob(self.url)
        job.run()

    def FindNewestFolderAndSetPath(self):
        source_dirs = self.SubDirList(self.path)
        manga_dirs, chapter_dirs = [], []

        for source_dir in source_dirs:
            manga_dirs += self.SubDirList(source_dir)

        for manga_dir in manga_dirs:
            chapter_dirs += self.SubDirList(manga_dir)

        subdirs = source_dirs + manga_dirs + chapter_dirs

        latest_subdir = max(subdirs, key=os.path.getmtime)

        self.path = latest_subdir

    @staticmethod
    def SubDirList(path):
        path_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                path_list.append(file_path)
        return path_list

    def openDownloadedProj(self, proj_path):
        self.open_downloaded_proj.emit(proj_path)

    def _SyncSourceDownload(self):
        #  TODO keep track of downloaded page
        import time
        # for i in range(100):
        #     self.update_progress_bar.emit(i)
        #     LOGGER.info(i)
        #     time.sleep(0.05)
        self.url = self.config_pnl.src_link_flag
        if self.url:
            LOGGER.info(f'Url set to {self.url}')

            self.ValidateUrl()
            self.FetchImages()
            self.FindNewestFolderAndSetPath()
            self.PassUrlToImgTransProj()

            LOGGER.info(f'Project path set to {self.path}')

            if self.path:
                self.openDownloadedProj(self.path)

            self.finished_downloading.emit()

    def run(self):
        self._SyncSourceDownload()
