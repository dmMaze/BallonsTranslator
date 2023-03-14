from .mainwindow import MainWindow
from dl.pagesources import SourceBase
from utils.logger import logger as LOGGER
from .misc import ProgramConfig
from .imgtrans_proj import ProjImgTrans


class SourceManager(SourceBase):

    def __init__(self, config: ProgramConfig, imgtrans_proj: ProjImgTrans, menu: MainWindow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_pnl = config
        self.imgtrans_proj = imgtrans_proj
        self.menu = menu

    def download_source(self):
        title = self.config_pnl.src_title_flag
        url = self.config_pnl.src_link_flag
        force_redownload = self.config_pnl.src_force_download_flag
        if url:
            LOGGER.info(f'Force download set to {force_redownload}')
            LOGGER.info(f'Url set to {url}')
            LOGGER.info(f'Project title set to {title}')
            self.run(url=url, force_redownload=force_redownload, title=title)
            proj_path = self.ReturnFullPathToProject()
            LOGGER.info(f'Project path set to {proj_path}')
            if proj_path:
                self.menu.openDir(proj_path)
                LOGGER.info('Download complete')
