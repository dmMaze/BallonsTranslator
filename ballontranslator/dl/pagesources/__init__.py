from gallery_dl.job import DownloadJob
from gallery_dl import config
from ui.mainwindow import MainWindow
from utils.logger import logger as LOGGER
from ui.misc import ProgramConfig
from ui.imgtrans_proj import ProjImgTrans

class SourceDownload:
    def __init__(self, config: ProgramConfig, imgtrans_proj: ProjImgTrans, menu: MainWindow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_pnl = config
        self.imgtrans_proj = imgtrans_proj
        self.menu = menu
        self.path = ''
        self.url = ''

    def ReturnFullPathToProject(self) -> str:
        return self.path

    def ValidateUrl(self):
        if 'https://' not in self.url:
            self.url = 'https://' + self.url

    def FetchImageUrls(self):
        config.load()
        job = DownloadJob(self.url)
        job.run()
        self.path = job.pathfmt.directory

    def download_source(self):
        self.url = self.config_pnl.src_link_flag
        if self.url:
            LOGGER.info(f'Url set to {self.url}')

            self.ValidateUrl()
            self.FetchImageUrls()

            proj_path = self.ReturnFullPathToProject()
            LOGGER.info(f'Project path set to {proj_path}')

            if proj_path:
                self.menu.openDir(proj_path)
                LOGGER.info('Download complete')

