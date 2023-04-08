import time
from typing import Union, List, Dict, Callable
import numpy as np
import traceback
import os.path as osp

from qtpy.QtCore import QThread, Signal, QObject, QLocale
from qtpy.QtWidgets import QMessageBox

from utils.logger import logger as LOGGER
from utils.registry import Registry
from utils.imgproc_utils import enlarge_window
from dl.translators import MissingTranslatorParams
from dl import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, \
    VALID_TRANSLATORS, VALID_TEXTDETECTORS, VALID_INPAINTERS, VALID_OCR, \
    TranslatorBase, InpainterBase, TextDetectorBase, OCRBase
import dl
dl.translators.SYSTEM_LANG = QLocale.system().name()

from .stylewidgets import ImgtransProgressMessageBox
from .configpanel import ConfigPanel
from .misc import DLModuleConfig, ProgramConfig
from .imgtrans_proj import ProjImgTrans

from dl.textdetector import TextBlock


class ModuleThread(QThread):

    exception_occurred = Signal(str, str, str)
    finish_set_module = Signal()

    def __init__(self, dl_config: DLModuleConfig, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dl_config = dl_config
        self.job = None
        self.module: Union[TextDetectorBase, TranslatorBase, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.imgtrans_proj: ProjImgTrans = None

    def _set_module(self, module_name: str):
        old_module = self.module
        try:
            module: Union[TextDetectorBase, TranslatorBase, InpainterBase, OCRBase] \
                = self.module_register.module_dict[module_name]
            setup_params = self.dl_config.get_setup_params(self.module_key)[module_name]
            if setup_params is not None:
                self.module = module(**setup_params)
            else:
                self.module = module()
        except Exception as e:
            self.module = old_module
            msg = self.tr('Failed to set ') + module_name
            
            self.exception_occurred.emit(msg, str(e), traceback.format_exc())
        self.finish_set_module.emit()

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter == len(self.imgtrans_proj.pages):
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    def __init__(self, dl_config: DLModuleConfig, *args, **kwargs) -> None:
        super().__init__(dl_config, 'inpainter', INPAINTERS, *args, **kwargs)

    @property
    def inpainter(self) -> InpainterBase:
        return self.module

    def setInpainter(self, inpainter: str):
        self.job = lambda : self._set_module(inpainter)
        self.start()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        self.job = lambda : self._inpaint(img, mask, img_key, inpaint_rect)
        self.start()
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect=None):
        inpaint_dict = {}
        self.inpainting = True
        try:
            inpainted = self.inpainter.inpaint(img, mask)
            inpaint_dict = {
                'inpainted': inpainted,
                'img': img,
                'mask': mask,
                'img_key': img_key,
                'inpaint_rect': inpaint_rect
            }
            self.finish_inpaint.emit(inpaint_dict)
        except Exception as e:
            self.exception_occurred.emit(self.tr('Inpainting Failed.'), str(e), traceback.format_exc())
            self.inpainting = False
        self.inpainting = False

class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, dl_config: DLModuleConfig, *args, **kwargs) -> None:
        super().__init__(dl_config, 'textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str):
        self.job = lambda : self._set_module(textdetector)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, dl_config: DLModuleConfig, *args, **kwargs) -> None:
        super().__init__(dl_config, 'ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str):
        self.job = lambda : self._set_module(ocr)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class TranslateThread(ModuleThread):
    finish_translate_page = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, dl_config: DLModuleConfig, *args, **kwargs) -> None:
        super().__init__(dl_config, 'translator', TRANSLATORS, *args, **kwargs)
        self.translator: TranslatorBase = self.module

    def _set_translator(self, translator: str):
        
        old_translator = self.translator
        source, target = self.dl_config.translate_source, self.dl_config.translate_target
        if self.translator is not None:
            if self.translator.name == translator:
                return
        
        try:
            setup_params = self.dl_config.translator_setup_params[translator]
            translator_module: TranslatorBase = TRANSLATORS.module_dict[translator]
            if setup_params is not None:
                self.translator = translator_module(source, target, raise_unsupported_lang=False, **setup_params)
            else:
                self.translator = translator_module(source, target, raise_unsupported_lang=False)
            self.dl_config.translate_source = self.translator.lang_source
            self.dl_config.translate_target = self.translator.lang_target
            self.dl_config.translator = self.translator.name
        except Exception as e:
            if old_translator is None:
                old_translator = TRANSLATORS.module_dict['google']('简体中文', 'English', raise_unsupported_lang=False)
            self.translator = old_translator
            msg = self.tr('Failed to set translator ') + translator
            self.exception_occurred.emit(msg, repr(e), traceback.format_exc())
        self.module = self.translator
        self.finish_set_module.emit()

    def setTranslator(self, translator: str):
        if translator in ['Sugoi']:
            self._set_translator(translator)
        else:
            self.job = lambda : self._set_translator(translator)
            self.start()

    def _translate_page(self, page_dict, page_key: str, raise_exception=False, emit_finished=True):
        page = page_dict[page_key]
        try:
            self.translator.translate_textblk_lst(page)
        except MissingTranslatorParams as e:
            if raise_exception:
                raise e
            else:
                self.exception_occurred.emit(e + self.tr(' is required for '), '', traceback.format_exc())
        except Exception as e:
            if raise_exception:
                raise e
            else:
                self.exception_occurred.emit(self.tr('Translation Failed.'), repr(e), traceback.format_exc())
        if emit_finished:
            self.finish_translate_page.emit(page_key)

    def translatePage(self, page_dict, page_key: str):
        self.job = lambda: self._translate_page(page_dict, page_key)
        self.start()

    def push_pagekey_queue(self, page_key: str):
        self.pipeline_pagekey_queue.append(page_key)

    def runTranslatePipeline(self, imgtrans_proj: ProjImgTrans):
        self.initImgtransPipeline(imgtrans_proj)
        self.job = self._run_translate_pipeline
        self.start()

    def _run_translate_pipeline(self):
        num_pages = len(self.imgtrans_proj.pages)
        delay = self.translator.delay()

        while not self.pipeline_finished():
            if len(self.pipeline_pagekey_queue) == 0:
                time.sleep(0.1)
                continue
            
            page_key = self.pipeline_pagekey_queue.pop(0)
            self.blockSignals(True)
            try:
                self._translate_page(self.imgtrans_proj.pages, page_key, raise_exception=True, emit_finished=False)
            except Exception as e:
                
                # TODO: allowing retry/skip/terminate

                msg = self.tr('Translation Failed.')
                if isinstance(e, MissingTranslatorParams):
                    msg = msg + '\n' + str(e) + self.tr(' is required for ' + self.translator.name)
                    
                self.blockSignals(False)
                self.exception_occurred.emit(msg, repr(e), traceback.format_exc())
                self.imgtrans_proj = None
                self.finished_counter = 0
                self.pipeline_pagekey_queue = []
                return
            self.blockSignals(False)
            self.finished_counter += 1
            self.progress_changed.emit(self.finished_counter)

            if not self.pipeline_finished() and delay > 0:
                time.sleep(delay)


class ImgtransThread(QThread):

    finished = Signal(object)
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_translate_progress = Signal(int)
    update_inpaint_progress = Signal(int)
    exception_occurred = Signal(str, str)

    finish_blktrans_stage = Signal(str, int)

    def __init__(self, 
                 dl_config: DLModuleConfig, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 translate_thread: TranslateThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dl_config = dl_config
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.translate_thread = translate_thread
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.mask_postprocess = None

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr
    
    @property
    def translator(self) -> TranslatorBase:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans):
        self.imgtrans_proj = imgtrans_proj
        self.job = self._imgtrans_pipeline
        self.start()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int):
        self.job = lambda : self._blktrans_pipeline(blk_list, tgt_img, mode)
        self.start()

    def _blktrans_pipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int):
        if mode >= 0:
            self.ocr_thread.module.run_ocr(tgt_img, blk_list)
            self.finish_blktrans_stage.emit('ocr', 100)
        if mode != 0:
            self.translate_thread.module.translate_textblk_lst(blk_list)
            self.finish_blktrans_stage.emit('translate', 100)
        if mode > 1:
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy = enlarge_window(blk.xyxy, im_w, im_h)
                xyxy = np.array(xyxy)
                x1, y1, x2, y2 = xyxy.astype(np.int64)
                blk.region_inpaint_dict = None
                if y2 - y1 > 2 and x2 - x1 > 2:
                    im = np.copy(tgt_img[y1: y2, x1: x2])
                    maskseg_method = self.get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im)
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        blk.region_inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2], 'inpainted': inpainted}
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans_stage.emit(str(mode), 0)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0
        self.num_pages = num_pages = len(self.imgtrans_proj.pages)

        if self.translator is not None:
            self.parallel_trans = not self.translator.is_computational_intensive()
        else:
            self.parallel_trans = False
        if self.dl_config.enable_translate and self.parallel_trans:
            self.translate_thread.runTranslatePipeline(self.imgtrans_proj)

        for imgname in self.imgtrans_proj.pages:
            img = self.imgtrans_proj.read_img(imgname)

            mask, blk_list = self.textdetector.detect(img)
            if self.mask_postprocess is not None:
                mask = self.mask_postprocess(mask)
                
            self.imgtrans_proj.save_mask(imgname, mask)

            self.detect_counter += 1
            self.update_detect_progress.emit(self.detect_counter)
            self.imgtrans_proj.pages[imgname] = blk_list

            if self.dl_config.enable_ocr:
                self.ocr.run_ocr(img, blk_list)
                self.ocr_counter += 1
                self.update_ocr_progress.emit(self.ocr_counter)

                if self.dl_config.enable_translate:
                    try:
                        if self.parallel_trans:
                            self.translate_thread.push_pagekey_queue(imgname)
                        else:
                            self.translator.translate_textblk_lst(blk_list)
                            self.translate_counter += 1
                            self.update_translate_progress.emit(self.translate_counter)
                    except Exception as e:
                        self.dl_config.enable_translate = False
                        self.update_translate_progress.emit(num_pages)
                        self.exception_occurred.emit(self.tr('Translation Failed.'), repr(e))
                        
            if self.dl_config.enable_inpaint:
                inpainted = self.inpainter.inpaint(img, mask, blk_list)
                self.imgtrans_proj.save_inpainted(imgname, inpainted)
                self.inpaint_counter += 1
                self.update_inpaint_progress.emit(self.inpaint_counter)
        
    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.detect_counter == self.num_pages

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None or not self.dl_config.enable_ocr:
            return True
        return self.ocr_counter == self.num_pages

    def translate_finished(self) -> bool:
        if self.imgtrans_proj is None \
            or not self.dl_config.enable_ocr \
            or not self.dl_config.enable_translate:
            return True
        if self.parallel_trans:
            return self.translate_thread.pipeline_finished()
        return self.translate_counter == self.num_pages

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or not self.dl_config.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None

    def recent_finished_index(self, ref_counter=None) -> int:
        counter = self.detect_counter
        if self.dl_config.enable_ocr:
            counter = min(counter, self.ocr_counter)
            if self.dl_config.enable_translate:
                if self.parallel_trans:
                    counter = min(counter, self.translate_thread.finished_counter)
                else:
                    counter = min(counter, self.translate_counter)
                    
        if self.dl_config.enable_inpaint:
            counter = min(counter, self.inpaint_counter)
        
        if ref_counter is not None:
            return min(counter, ref_counter) - 1
        return counter - 1

def merge_config_module_params(config_params: Dict, module_keys: List, get_module: Callable) -> Dict:
    for module_key in module_keys:
        module_params = get_module(module_key).setup_params
        if module_key not in config_params or config_params[module_key] is None:
            config_params[module_key] = module_params
        else:
            cfg_param = config_params[module_key]
            cfg_key_set = set(cfg_param.keys())
            module_key_set = set(module_params.keys())
            for ck in cfg_key_set:
                if ck not in module_key_set:
                    LOGGER.warning(f'Found invalid {module_key} config: {ck}')
                    cfg_param.pop(ck)
            for mk in module_key_set:
                if mk not in cfg_key_set:
                    LOGGER.info(f'Found new {module_key} config: {mk}')
                    cfg_param[mk] = module_params[mk]
    return config_params

class DLManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    update_translator_status = Signal(str, str, str)
    update_source_download_status = Signal(str)
    update_inpainter_status = Signal(str)
    finish_translate_page = Signal(str)
    canvas_inpaint_finished = Signal(dict)

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int)
    page_trans_finished = Signal(int)

    run_canvas_inpaint = False
    def __init__(self, 
                 config: ProgramConfig, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.dl_config = config.dl
        self.imgtrans_proj = imgtrans_proj

    def setupThread(self, config_panel: ConfigPanel, imgtrans_progress_msgbox: ImgtransProgressMessageBox, ocr_postprocess: Callable = None, translate_postprocess: Callable = None):
        dl_config = self.dl_config
        self.textdetect_thread = TextDetectThread(dl_config)
        self.textdetect_thread.finish_set_module.connect(self.on_finish_setdetector)
        self.textdetect_thread.exception_occurred.connect(self.handleRunTimeException)

        self.ocr_thread = OCRThread(dl_config)
        self.ocr_thread.finish_set_module.connect(self.on_finish_setocr)
        self.ocr_thread.exception_occurred.connect(self.handleRunTimeException)

        self.translate_thread = TranslateThread(dl_config)
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)
        self.translate_thread.finish_set_module.connect(self.on_finish_settranslator)
        self.translate_thread.finish_translate_page.connect(self.on_finish_translate_page)
        self.translate_thread.exception_occurred.connect(self.handleRunTimeException)        

        self.inpaint_thread = InpaintThread(dl_config)
        self.inpaint_thread.finish_set_module.connect(self.on_finish_setinpainter)
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)
        self.inpaint_thread.exception_occurred.connect(self.handleRunTimeException)        

        self.progress_msgbox = imgtrans_progress_msgbox

        self.imgtrans_thread = ImgtransThread(dl_config, self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.exception_occurred.connect(self.handleRunTimeException)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)

        self.translator_panel = translator_panel = config_panel.trans_config_panel        
        translator_setup_params = merge_config_module_params(dl_config.translator_setup_params, VALID_TRANSLATORS, TRANSLATORS.get)
        translator_panel.setupModulesParamWidgets(translator_setup_params)
        translator_panel.translator_changed.connect(self.setTranslator)
        translator_panel.source_combobox.currentTextChanged.connect(self.on_translatorsource_changed)
        translator_panel.target_combobox.currentTextChanged.connect(self.on_translatortarget_changed)
        translator_panel.paramwidget_edited.connect(self.on_translatorparam_edited)
        self.translate_postprocess = translate_postprocess

        self.inpaint_panel = inpainter_panel = config_panel.inpaint_config_panel
        inpainter_setup_params = merge_config_module_params(dl_config.inpainter_setup_params, VALID_INPAINTERS, INPAINTERS.get)
        inpainter_panel.setupModulesParamWidgets(inpainter_setup_params)
        inpainter_panel.paramwidget_edited.connect(self.on_inpainterparam_edited)
        inpainter_panel.inpainter_changed.connect(self.setInpainter)
        inpainter_panel.needInpaintChecker.checker_changed.connect(self.on_inpainter_checker_changed)
        inpainter_panel.needInpaintChecker.checker.setChecked(dl_config.check_need_inpaint)

        self.textdetect_panel = textdetector_panel = config_panel.detect_config_panel
        textdetector_setup_params = merge_config_module_params(dl_config.textdetector_setup_params, VALID_TEXTDETECTORS, TEXTDETECTORS.get)
        textdetector_panel.setupModulesParamWidgets(textdetector_setup_params)
        textdetector_panel.paramwidget_edited.connect(self.on_textdetectorparam_edited)
        textdetector_panel.detector_changed.connect(self.setTextDetector)

        self.ocr_panel = ocr_panel = config_panel.ocr_config_panel
        ocr_setup_params = merge_config_module_params(dl_config.ocr_setup_params, VALID_OCR, OCR.get)
        ocr_panel.setupModulesParamWidgets(ocr_setup_params)
        ocr_panel.paramwidget_edited.connect(self.on_ocrparam_edited)
        ocr_panel.ocr_changed.connect(self.setOCR)
        self.ocr_postprocess = ocr_postprocess

        self.on_finish_setsourcedownload()

        self.setTextDetector()
        self.setOCR()
        if self.dl_config.enable_translate:
            self.setTranslator()
        self.setInpainter()

    @property
    def translator(self) -> TranslatorBase:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr

    def translatePage(self, run_target: bool, page_key: str):
        if not run_target:
            if self.translate_thread.isRunning():
                LOGGER.warning('Terminating a running translation thread.')
                self.translate_thread.terminate()
            return
        self.translate_thread.translatePage(self.imgtrans_proj.pages, page_key)

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.terminate()
        if self.ocr_thread.isRunning():
            self.ocr_thread.terminate()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.terminate()
        if self.translate_thread.isRunning():
            self.translate_thread.terminate()

    def runImgtransPipeline(self):
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        self.progress_msgbox.show_all_bars()
        if not self.dl_config.enable_ocr:
            self.progress_msgbox.ocr_bar.hide()
            self.progress_msgbox.translate_bar.hide()
        else:
            self.progress_msgbox.ocr_bar.show()
            if not self.dl_config.enable_translate:
                self.progress_msgbox.translate_bar.hide()
            else:
                self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj)

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int):
        self.terminateRunningThread()
        self.progress_msgbox.hide_all_bars()
        if mode >= 0:
            self.progress_msgbox.ocr_bar.show()
        if mode == 2:
            self.progress_msgbox.inpaint_bar.show()
        if mode != 0:
            self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show()
        self.imgtrans_thread.runBlktransPipeline(blk_list, tgt_img, mode)

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'translate':
            self.progress_msgbox.updateTranslateProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        elif stage in {'-1', '0', '1', '2'}:
            self.blktrans_pipeline_finished.emit(int(stage))
            self.progress_msgbox.hide()
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')

    def on_update_detect_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateDetectProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_ocr_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateOCRProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_translate_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateTranslateProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_inpaint_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateInpaintProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def finishImgtransPipeline(self):
        if self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
                and self.imgtrans_thread.translate_finished() \
                    and self.imgtrans_thread.inpaint_finished():
            self.imgtrans_proj.save()
            self.progress_msgbox.hide()
            self.imgtrans_proj.save()
            self.imgtrans_pipeline_finished.emit()

    def setTranslator(self, translator: str = None):
        if translator is None:
            translator = self.dl_config.translator
        if self.translate_thread.isRunning():
            LOGGER.warning('Terminating a running translation thread.')
            self.translate_thread.terminate()
        self.update_translator_status.emit('...', self.dl_config.translate_source, self.dl_config.translate_target)
        self.translate_thread.setTranslator(translator)

    def setInpainter(self, inpainter: str = None):
        if inpainter is None:
            inpainter =self.dl_config.inpainter
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Terminating a running inpaint thread.')
            self.inpaint_thread.terminate()
        self.inpaint_thread.setInpainter(inpainter)

    def setTextDetector(self, textdetector: str = None):
        if textdetector is None:
            textdetector = self.dl_config.textdetector
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Terminating a running text detection thread.')
            self.textdetect_thread.terminate()
        self.textdetect_thread.setTextDetector(textdetector)

    def setOCR(self, ocr: str = None):
        if ocr is None:
            ocr = self.dl_config.ocr
        if self.ocr_thread.isRunning():
            LOGGER.warning('Terminating a running OCR thread.')
            self.ocr_thread.terminate()
        self.ocr_thread.setOCR(ocr)

    def on_finish_setdetector(self):
        if self.textdetector is not None:
            self.dl_config.textdetector = self.textdetector.name
            LOGGER.info('Text detector set to {}'.format(self.textdetector.name))

    def on_finish_setocr(self):
        if self.ocr is not None:
            self.dl_config.ocr = self.ocr.name
            self.ocr_panel.setOCR(self.ocr.name)
            self.ocr_thread.module.register_postprocess_hooks(self.ocr_postprocess)
            LOGGER.info('OCR set to {}'.format(self.ocr.name))

    def on_finish_setinpainter(self):
        if self.inpainter is not None:
            self.dl_config.inpainter = self.inpainter.name
            self.inpaint_panel.setInpainter(self.inpainter.name)
            self.update_inpainter_status.emit(self.dl_config.inpainter)
            LOGGER.info('Inpainter set to {}'.format(self.inpainter.name))

    def on_finish_settranslator(self):
        translator = self.translator
        if translator is not None:
            self.dl_config.translator = translator.name
            self.update_translator_status.emit(self.dl_config.translator, self.dl_config.translate_source, self.dl_config.translate_target)
            self.translator_panel.finishSetTranslator(translator)
            self.translate_thread.module.register_postprocess_hooks(self.translate_postprocess)
            LOGGER.info('Translator set to {}'.format(self.translator.name))
        else:
            LOGGER.error('invalid translator')
            self.update_translator_status.emit(self.tr('Invalid'), '', '')

    def on_finish_setsourcedownload(self):
        if self.config.src_link_flag:
            self.update_source_download_status.emit(self.config.src_link_flag)
        
    def on_finish_translate_page(self, page_key: str):
        self.finish_translate_page.emit(page_key)
    
    def on_finish_inpaint(self, inpaint_dict: dict):
        if self.run_canvas_inpaint:
            self.canvas_inpaint_finished.emit(inpaint_dict)
            self.run_canvas_inpaint = False

    def canvas_inpaint(self, inpaint_dict):
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)

    def on_settranslator_failed(self, translator: str, msg: str):
        self.handleRunTimeException(f'Failed to set translator {translator}', msg)

    def on_setinpainter_failed(self, inpainter: str, msg: str):
        self.handleRunTimeException(f'Failed to set inpainter {inpainter}', msg)

    def on_translatorsource_changed(self):
        text = self.translator_panel.source_combobox.currentText()
        if self.translator is not None:
            self.translator.set_source(text)
        self.dl_config.translate_source = text
        self.update_translator_status.emit(self.dl_config.translator, self.dl_config.translate_source, self.dl_config.translate_target)

    def on_translatortarget_changed(self):
        text = self.translator_panel.target_combobox.currentText()
        if self.translator is not None:
            self.translator.set_target(text)
        self.dl_config.translate_target = text
        self.update_translator_status.emit(self.dl_config.translator, self.dl_config.translate_source, self.dl_config.translate_target)

    def setOCRMode(self, enable: bool):
        self.dl_config.enable_ocr = enable
        if not enable:
            self.dl_config.enable_translate = False

    def setTransMode(self, enable: bool):
        self.dl_config.enable_translate = enable
        if enable:
            if self.translator is None:
                self.setTranslator()
            self.update_translator_status.emit(self.dl_config.translator, self.dl_config.translate_source, self.dl_config.translate_target)
        else:
            self.update_translator_status.emit('', '', '')
    
    def on_translatorparam_edited(self, param_key: str, param_content: str):
        if self.translator is not None:
            self.updateModuleSetupParam(self.translator, param_key, param_content)
            self.dl_config.translator_setup_params[self.translator.name] = self.translator.setup_params

    def on_inpainterparam_edited(self, param_key: str, param_content: str):
        if self.inpainter is not None:
            self.updateModuleSetupParam(self.inpainter, param_key, param_content)
            self.dl_config.inpainter_setup_params[self.inpainter.name] = self.inpainter.setup_params

    def on_textdetectorparam_edited(self, param_key: str, param_content: str):
        if self.textdetector is not None:
            self.updateModuleSetupParam(self.textdetector, param_key, param_content)
            self.dl_config.textdetector_setup_params[self.textdetector.name] = self.textdetector.setup_params

    def on_ocrparam_edited(self, param_key: str, param_content: str):
        if self.ocr is not None:
            self.updateModuleSetupParam(self.ocr, param_key, param_content)
            self.dl_config.ocr_setup_params[self.ocr.name] = self.ocr.setup_params

    def updateModuleSetupParam(self, 
                               module: Union[InpainterBase, TranslatorBase],
                               param_key: str, param_content: str):
            module.updateParam(param_key, param_content)
        
    def handleRunTimeException(self, msg: str, detail: str = None, verbose: str = ''):
        if detail is not None:
            msg += ': ' + detail
        LOGGER.error(msg + '\n' + verbose)
        err = QMessageBox()
        err.setText(msg)
        err.setDetailedText(verbose)
        err.exec()

    def handle_page_changed(self):
        if not self.imgtrans_thread.isRunning():
            if self.inpaint_thread.inpainting:
                self.run_canvas_inpaint = False
                self.inpaint_thread.terminate()

    def on_inpainter_checker_changed(self, is_checked: bool):
        self.dl_config.check_need_inpaint = is_checked
        InpainterBase.check_need_inpaint = is_checked