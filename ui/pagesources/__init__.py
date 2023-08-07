from gallery_dl.job import DownloadJob
from gallery_dl import config, util
from qtpy.QtCore import Signal, QThread
from utils.logger import logger as LOGGER
from ui.config import ProgramConfig
from ui.imgtrans_proj import ProjImgTrans
from ui.constants import DOWNLOAD_PATH
import os


class SourceDownload(QThread):
    update_progress_bar = Signal(int)
    open_downloaded_proj = Signal(str)
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
        job = SourceJob(self.url)
        job.update_progress_bar_job.connect(self.update_progress_bar.emit)
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


class SourceJob(DownloadJob, QThread):
    update_progress_bar_job = Signal(int)

    def __init__(self, url):
        QThread.__init__(self)
        DownloadJob.__init__(self, url)
        self.progress = 0

    def handle_url(self, url, kwdict):
        """Download the resource specified in 'url'"""
        hooks = self.hooks
        pathfmt = self.pathfmt
        archive = self.archive

        progress_chunk = round(100 / kwdict['count'])

        # prepare download
        pathfmt.set_filename(kwdict)

        if "prepare" in hooks:
            for callback in hooks["prepare"]:
                callback(pathfmt)

        if archive and archive.check(kwdict):
            pathfmt.fix_extension()
            self.handle_skip()
            return

        if pathfmt.extension and not self.metadata_http:
            pathfmt.build_path()

            if pathfmt.exists():
                if archive:
                    archive.add(kwdict)
                self.handle_skip()
                return

        if self.sleep:
            self.extractor.sleep(self.sleep(), "download")

        # download from URL
        if not self.download(url):

            # use fallback URLs if available/enabled
            fallback = kwdict.get("_fallback", ()) if self.fallback else ()
            for num, url in enumerate(fallback, 1):
                util.remove_file(pathfmt.temppath)
                self.log.info("Trying fallback URL #%d", num)
                if self.download(url):
                    break
            else:
                # download failed
                self.status |= 4
                self.log.error("Failed to download %s",
                               pathfmt.filename or url)
                return

        if not pathfmt.temppath:
            if archive:
                archive.add(kwdict)
            self.handle_skip()
            return

        # run post processors
        if "file" in hooks:
            for callback in hooks["file"]:
                callback(pathfmt)

        # download succeeded
        self.progress += progress_chunk
        self.update_progress_bar_job.emit(self.progress)
        pathfmt.finalize()
        self.out.success(pathfmt.path)
        self._skipcnt = 0
        if archive:
            archive.add(kwdict)
        if "after" in hooks:
            for callback in hooks["after"]:
                callback(pathfmt)

