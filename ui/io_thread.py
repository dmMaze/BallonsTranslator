import numpy as np
import os.path as osp
import traceback

from qtpy.QtCore import Qt, Signal, QUrl, QThread
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtWidgets import QDialog, QMessageBox, QFileDialog

from utils.logger import logger as LOGGER
from utils.io_utils import imread, imwrite
from utils.message import create_error_dialog
from utils.proj_imgtrans import ProjImgTrans
from .custom_widget import ProgressMessageBox
from .misc import pixmap2ndarray


class ThreadBase(QThread):

    _thread_exception_type = None
    _thread_error_msg = 'Thread job failed.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = None

    def on_exec_failed(self):
        return
    
    def run(self):
        if self.job is not None:
            try:
                self.job()
            except Exception as e:
                self.on_exec_failed()
                create_error_dialog(e, self._thread_error_msg, self._thread_exception_type)
        self.job = None

class ImgSaveThread(ThreadBase):

    img_writed = Signal(str)
    _thread_exception_type = 'ImgSaveThread'
    _thread_error_msg = 'Failed to save image.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.im_save_list = []

    def saveImg(self, save_path: str, img: QImage, pagename_in_proj: str = '', save_params: dict = None, keep_alpha=False):
        self.im_save_list.append((save_path, img, pagename_in_proj, save_params, keep_alpha))
        if self.job is None:
            self.job = self._save_img
            self.start()

    def _save_img(self):
        while True:
            if len(self.im_save_list) == 0:
                break
            save_path, img, pagename_in_proj, save_params, keep_alpha = self.im_save_list[0]
            if save_params is None:
                save_params = {}
            if isinstance(img, QImage) or isinstance(img, QPixmap):
                img = pixmap2ndarray(img, keep_alpha=keep_alpha)
            imwrite(save_path, img, **save_params)
            self.img_writed.emit(pagename_in_proj)
            self.im_save_list.pop(0)

    def on_exec_failed(self):
        if len(self.im_save_list) > 0:
            self.im_save_list.pop(0)
            if len(self.im_save_list) == 0:
                self.job = None
            else:
                try:
                    self.job()
                except Exception as e:
                    self.on_exec_failed()
                    create_error_dialog(e, self._thread_error_msg, self._thread_exception_type)




class ImgTransProjFileIOThread(ThreadBase):

    fin_page = Signal()
    fin_io = Signal()

    _thread_exception_type = 'ImgTransProjFileIOThread'

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

    def on_exec_failed(self):
        self.progress_bar.hide()


class ExportDocThread(ImgTransProjFileIOThread):

    _thread_error_msg = 'Failed to export Doc'

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

    _thread_error_msg = 'Failed to import Doc'

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

    

class MergeThread(ThreadBase):
    progress_changed = Signal(int, int)
    merge_finished = Signal(int, int)
    
    _thread_exception_type = 'MergeThread'
    _thread_error_msg = 'Region merge failed.'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.json_path = None
        self.img_list = []
        self.config = None
        self.stop_requested = False
        self.progress_bar = ProgressMessageBox(self.tr('Region Merge: '))
    
    def runMerge(self, json_path: str, img_list: list, config: dict):
        if self.isRunning():
            return False
        self.json_path = json_path
        self.img_list = img_list
        self.config = config
        self.stop_requested = False
        self.job = self._run_merge
        self.start()
        return True
    
    def requestStop(self):
        self.stop_requested = True
    
    def _run_merge(self):
        from utils import merger
        import json
        import copy
        
        success_count = 0
        fail_count = 0
        total = len(self.img_list)
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            LOGGER.error(f'Failed to read JSON file: {e}')
            self.merge_finished.emit(0, total)
            return
        
        if 'pages' not in data:
            LOGGER.error('JSON file is not in BallonsTranslator format')
            self.merge_finished.emit(0, total)
            return
        
        mode = self.config.get("MERGE_MODE", "NONE")
        modified = False
        
        failed_images = []
        
        for i, img_name in enumerate(self.img_list):
            if self.stop_requested:
                LOGGER.info('Region merge stopped by user')
                break
            
            if img_name not in data['pages']:
                fail_count += 1
                failed_images.append((img_name, "Image not found in JSON"))
                self.progress_changed.emit(i + 1, total)
                continue
            
            initial_shapes = data['pages'][img_name]
            if not initial_shapes:
                fail_count += 1
                failed_images.append((img_name, "No text boxes"))
                self.progress_changed.emit(i + 1, total)
                continue
            
            try:
                initial_count = len(initial_shapes)
                total_merged = 0
                
                if mode == "VERTICAL":
                    final_shapes, count = merger.perform_merge(initial_shapes, "VERTICAL", self.config)
                    total_merged += count
                elif mode == "HORIZONTAL":
                    final_shapes, count = merger.perform_merge(initial_shapes, "HORIZONTAL", self.config)
                    total_merged += count
                elif mode == "VERTICAL_THEN_HORIZONTAL":
                    temp, count1 = merger.perform_merge(initial_shapes, "VERTICAL", self.config)
                    final_shapes, count2 = merger.perform_merge(temp, "HORIZONTAL", self.config)
                    total_merged += (count1 + count2)
                elif mode == "HORIZONTAL_THEN_VERTICAL":
                    temp, count1 = merger.perform_merge(initial_shapes, "HORIZONTAL", self.config)
                    final_shapes, count2 = merger.perform_merge(temp, "VERTICAL", self.config)
                    total_merged += (count1 + count2)
                else:
                    final_shapes = initial_shapes
                
                if total_merged > 0:
                    data['pages'][img_name] = final_shapes
                    modified = True
                    success_count += 1
                else:
                    fail_count += 1
                    labels = set(s.get('label', '') for s in initial_shapes)
                    reason = f"No mergeable boxes ({initial_count} boxes, labels: {', '.join(labels) or 'none'})"
                    failed_images.append((img_name, reason))
                    
            except Exception as e:
                LOGGER.error(f'Failed to merge {img_name}: {e}')
                fail_count += 1
                failed_images.append((img_name, f"Exception: {e}"))
            
            self.progress_changed.emit(i + 1, total)
        
        if failed_images:
            print(f"\n{'='*60}")
            print(f"Region merge failure list ({len(failed_images)} total):")
            print(f"{'='*60}")
            for img_name, reason in failed_images:
                print(f"  {img_name}: {reason}")
            print(f"{'='*60}\n")
        
        if modified and not self.stop_requested:
            try:
                with open(self.json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                LOGGER.error(f'Failed to write JSON file: {e}')
        
        self.merge_finished.emit(success_count, fail_count)
