import numpy as np
import os.path as osp
import traceback

from qtpy.QtCore import Qt, Signal, QUrl, QThread
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QDialog, QMessageBox, QFileDialog

from utils.logger import logger as LOGGER
from utils.io_utils import imread, imwrite
from .imgtrans_proj import ProjImgTrans
from .stylewidgets import ProgressMessageBox


class ThreadBase(QThread):
    exception_occured = Signal(str, str, str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = None
        self.exception_occured.connect(self.handleRunTimeException)
    
    def run(self):
        if self.job is not None:
            try:
                self.job()
            except Exception as e:
                self.exception_occured.emit(self.tr('Execution error'), str(e), traceback.format_exc())
        self.job = None

    def handleRunTimeException(self, msg: str, detail: str = None, verbose: str = ''):
        if detail is not None:
            msg += ': ' + detail
        LOGGER.error(msg + '\n' + verbose)
        err = QMessageBox()
        err.setText(msg)
        err.setDetailedText(verbose)
        err.exec()

class ImgSaveThread(ThreadBase):
    img_writed = Signal(str)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_save_list = []

    def saveImg(self, save_path: str, img: QImage, pagename_in_proj: str = ''):
        self.im_save_list.append((save_path, img, pagename_in_proj))
        if self.job is None:
            self.job = self._save_img
            self.start()

    def _save_img(self):
        while True:
            if len(self.im_save_list) == 0:
                break
            save_path, img, pagename_in_proj = self.im_save_list.pop(0)
            if isinstance(img, QImage) or isinstance(img, QPixmap):
                img.save(save_path)
            elif isinstance(img, np.ndarray):
                imwrite(save_path, img)
            self.img_writed.emit(pagename_in_proj)


class ImgTransProjFileIOThread(ThreadBase):

    fin_page = Signal()
    fin_io = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj: ProjImgTrans = None
        self.fin_counter = 0
        self.num_pages = 0
        self.fin_page.connect(self.on_fin_page)
        self.progress_bar = ProgressMessageBox('task')

    def on_fin_page(self):
        self.fin_counter += 1
        progress = int(self.fin_counter / self.num_pages * 100)
        self.progress_bar.updateTaskProgress(progress)
        if self.fin_counter == self.num_pages:
            self.progress_bar.hide()

    def handleRunTimeException(self, msg: str, detail: str = None, verbose: str = ''):
        super().handleRunTimeException(msg, detail, verbose)
        self.progress_bar.hide()


class ExportDocThread(ImgTransProjFileIOThread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar.setTaskName(self.tr('Export as doc...'))

    def exportAsDoc(self, proj: ProjImgTrans):
        doc_path = proj.doc_path()
        if osp.exists(doc_path):
            msg = QMessageBox()
            msg.setText(self.tr('Overwrite ') + doc_path + '?')
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            ret = msg.exec_()
            if ret == QMessageBox.StandardButton.No:
                return
        if self.job is None:
            self.proj = proj
            self.job = self._export_as_doc
            self.start()
            self.progress_bar.updateTaskProgress(0)
            self.progress_bar.show()

    def _export_as_doc(self):
        if self.proj is None:
            return
        self.fin_counter = 0
        self.num_pages = self.proj.num_pages
        if self.num_pages > 0:
            self.proj.dump_doc(fin_page_signal=self.fin_page)
        self.proj = None
        self.progress_bar.hide()
        self.fin_io.emit()


class ImportDocThread(ImgTransProjFileIOThread):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.progress_bar.setTaskName(self.tr('Import doc...'))
        self.doc_path = None
    
    def importDoc(self, proj: ProjImgTrans):
        dialog = QFileDialog()
        dialog.setDefaultSuffix('.docx')
        url = QUrl(proj.directory)
        doc_path = dialog.getOpenFileUrl(self.parent(), self.tr('Import *.docx'), directory=url, filter="Microsoft Word Documents (*.doc *.docx)")[0].toLocalFile()
        if osp.exists(doc_path) and self.job is None:
            self.proj = proj
            self.job = self._import_doc
            self.doc_path = doc_path
            self.start()
            self.progress_bar.updateTaskProgress(0)
            self.progress_bar.show()

    def _import_doc(self):
        if self.proj is None:
            return
        self.fin_counter = 0
        self.num_pages = self.proj.num_pages
        self.proj.load_doc(self.doc_path, fin_page_signal=self.fin_page)
        self.proj = None
        self.progress_bar.hide()
        self.fin_io.emit()


class ExportKraThread(ImgTransProjFileIOThread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_bar.setTaskName(self.tr('Export as krita document...'))

    def exportAsKra(self, proj: ProjImgTrans):
        if self.job is None:
            self.proj = proj
            self.job = self._export_as_kra
            self.start()
            self.progress_bar.updateTaskProgress(0)
            self.progress_bar.show()

    def _export_as_kra(self):
        if self.proj is None:
            return
        self.fin_counter = 0
        self.num_pages = self.proj.num_pages
        if self.num_pages > 0:
            self.proj.dump_kra(fin_page_signal=self.fin_page)
        self.proj = None
        self.progress_bar.hide()
        self.fin_io.emit()
