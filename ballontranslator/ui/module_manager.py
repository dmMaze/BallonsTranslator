import threading
from typing import Union, List, Callable
import os.path as osp

import numpy as np
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer
from qtpy.QtWidgets import QFileDialog, QMessageBox, QCheckBox

from .funcmaps import get_maskseg_method
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.registry import LazyModuleError, Registry
from ballontranslator.utils.imgproc_utils import enlarge_window, get_block_mask
from ballontranslator.utils.io_utils import text_is_empty
from ballontranslator.modules.translators import MissingTranslatorParams
from ballontranslator.modules.base import BaseModule, soft_empty_cache
from ballontranslator.modules import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, \
    GET_VALID_TRANSLATORS, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, \
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
import ballontranslator.modules as modules
modules.translators.SYSTEM_LANG = QLocale.system().name()
from ballontranslator.utils.textblock import TextBlock, sort_regions
from ballontranslator.utils import shared
from ballontranslator.utils.message import create_error_dialog, create_info_dialog
from ballontranslator.utils.download_util import DownloadCancelled
from ballontranslator.modules.package_import_names import PACKAGE_IMPORT_NAMES
from ballontranslator.modules.prepare_local_files import MissingDependency, ensure_module_files
from ballontranslator.utils.py_package_manager import (
    MissingModuleRequirements,
    PyPackageManager,
    collect_missing_module_requirements,
)
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox, ProgressMessageBox
from .configpanel import ConfigPanel
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils import config as program_config
from ballontranslator.utils.config import pcfg, RunStatus
cfg_module = pcfg.module


class ModuleThread(QThread):

    finish_set_module = Signal()
    module_prepare_progress = Signal(dict)
    _failed_set_module_msg = 'Failed to set module.'
    module_thread_stopped = Signal()

    def __init__(self, module_key: str, MODULE_REGISTER: Registry, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.job = None
        self.module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = None
        self.module_register = MODULE_REGISTER
        self.module_key = module_key

        self.pipeline_pagekey_queue = []
        self.finished_counter = 0
        self.num_process_pages = 0
        self.imgtrans_proj: ProjImgTrans = None
        self.pipeline_stop_event = None
        # Shared with download helpers so Stop can cancel preparation cooperatively.
        self.cancel_event = threading.Event()
        self.last_set_success = False
        self.last_set_module_name = ''
        self.last_error = None
        self.last_missing_requirements = []

    def _emit_prepare_progress(self, payload: dict):
        payload = dict(payload)
        payload.setdefault('module_key', self.module_key)
        payload.setdefault('module', self.last_set_module_name)
        self.module_prepare_progress.emit(payload)

    def _prepare_module_class(self, module_name: str):
        # Prepare dependency/files first; resolving the class is the first real import.
        spec = self.module_register.get_spec(module_name)
        if spec is None:
            raise KeyError(f'Unknown {self.module_key} module: {module_name}')
        self._emit_prepare_progress({'event': 'checking_dependencies', 'message': self.tr('Checking dependencies')})
        if not ensure_module_files(spec, progress_callback=self._emit_prepare_progress, cancel_event=self.cancel_event):
            raise RuntimeError(f'Failed to prepare files for {self.module_key} module: {module_name}')
        if self.cancel_event.is_set():
            raise DownloadCancelled('Module preparation cancelled by user.')
        self._emit_prepare_progress({'event': 'importing', 'message': self.tr('Importing module')})
        module_class = self.module_register.resolve_module(module_name)
        if module_class is None:
            raise KeyError(f'Failed to resolve {self.module_key} module: {module_name}')
        return module_class

    def _set_module(self, module_name: str):
        old_module = self.module
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        try:
            if old_module is not None and old_module.name == module_name:
                self.last_set_success = True
                return
            module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = self._prepare_module_class(module_name)
            params = cfg_module.get_params(self.module_key).get(module_name)
            self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
            if params is not None:
                self.module = module(**params)
            else:
                self.module = module()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            if not pcfg.module.load_model_on_demand:
                self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
                self.module.load_model()
                if self.cancel_event.is_set():
                    raise DownloadCancelled('Module preparation cancelled by user.')
            if old_module is not None:
                old_module.unload_model(empty_cache=True)
                del old_module
            self.last_set_success = True
        except DownloadCancelled as e:
            # Cancellation/failure keeps the previously active module instance.
            self.module = old_module
            self.last_error = e
            LOGGER.info(f'Cancelled preparing {self.module_key} module {module_name}.')
        except MissingDependency as e:
            self.module = old_module
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.module = old_module
            missing = self._missing_requirements_from_exception(module_name, e)
            if missing:
                self.last_error = e
                self.last_missing_requirements = missing
            else:
                self.last_error = e
        finally:
            self.finish_set_module.emit()

    def _package_manager(self) -> PyPackageManager:
        pmcfg = pcfg.package_manager
        return PyPackageManager(
            backend=pmcfg.installer_backend,
            extra_args=pmcfg.extra_install_args,
            package_import_names=PACKAGE_IMPORT_NAMES,
        )

    def _missing_requirements_from_exception(self, module_name: str, exception: Exception) -> List[str]:
        spec = self.module_register.get_spec(module_name)
        dependencies = spec.dependencies if spec is not None and spec.dependencies else []
        if not dependencies:
            return []

        import_name = None
        probe = exception
        while probe is not None:
            if isinstance(probe, ModuleNotFoundError):
                import_name = probe.name or import_name
                if import_name:
                    break
            if isinstance(probe, LazyModuleError) and isinstance(probe.__cause__, ModuleNotFoundError):
                import_name = probe.__cause__.name or import_name
                if import_name:
                    break
            probe = probe.__cause__ or probe.__context__

        if not import_name:
            return []
        requirement = self._package_manager().requirement_for_import_name(import_name, dependencies)
        return [requirement] if requirement is not None else []

    def installMissingPackagesAndSetModule(self, module_name: str, requirements: List[str]):
        self.job = lambda: self._install_missing_then_set(module_name, requirements)
        self.start()

    def _install_missing_then_set(self, module_name: str, requirements: List[str]):
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = self._package_manager().install(requirements, progress_callback=self._emit_prepare_progress)
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
            self.finish_set_module.emit()
            return
        self._set_module(module_name)

    def pipeline_finished(self):
        if self.imgtrans_proj is None:
            return True
        elif self.finished_counter >= self.num_process_pages:
            return True
        return False

    def initImgtransPipeline(self, proj: ProjImgTrans, stop_event: threading.Event = None):
        if self.isRunning():
            self.terminate()
        self.imgtrans_proj = proj
        self.finished_counter = 0
        self.pipeline_pagekey_queue.clear()
        self.pipeline_stop_event = stop_event

    def requestCancelModuleInit(self):
        self.cancel_event.set()

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None


class PackageInstallThread(QThread):

    finish_install = Signal()
    package_prepare_progress = Signal(dict)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requirements = []
        self.last_success = False
        self.last_error = None

    def installPackages(self, requirements: List[str]) -> bool:
        if self.isRunning():
            return False
        self.requirements = list(dict.fromkeys(requirements))
        self.last_success = False
        self.last_error = None
        self.start()
        return True

    def _package_manager(self) -> PyPackageManager:
        pmcfg = pcfg.package_manager
        return PyPackageManager(
            backend=pmcfg.installer_backend,
            extra_args=pmcfg.extra_install_args,
            package_import_names=PACKAGE_IMPORT_NAMES,
        )

    def _emit_prepare_progress(self, payload: dict):
        self.package_prepare_progress.emit(dict(payload))

    def run(self):
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = self._package_manager().install(self.requirements, progress_callback=self._emit_prepare_progress)
        self.last_success = result.ok
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(self.requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
        self.finish_install.emit()


class InpaintThread(ModuleThread):

    finish_inpaint = Signal(dict)
    inpainting = False    
    inpaint_failed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('inpainter', INPAINTERS, *args, **kwargs)

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
            create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
            self.inpainting = False
            self.inpaint_failed.emit()
        self.inpainting = False


class TextDetectThread(ModuleThread):
    
    finish_detect_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('textdetector', TEXTDETECTORS, *args, **kwargs)

    def setTextDetector(self, textdetector: str):
        self.job = lambda : self._set_module(textdetector)
        self.start()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.module


class OCRThread(ModuleThread):

    finish_ocr_page = Signal(str)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__('ocr', OCR, *args, **kwargs)

    def setOCR(self, ocr: str):
        self.job = lambda : self._set_module(ocr)
        self.start()
    
    @property
    def ocr(self) -> OCRBase:
        return self.module


class TranslateThread(ModuleThread):

    finish_translate_page = Signal(str)
    progress_changed = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('translator', TRANSLATORS, *args, **kwargs)
        self.translator: BaseTranslator = self.module

    def _set_translator(self, translator: str):
        
        old_translator = self.translator
        source, target = cfg_module.translate_source, cfg_module.translate_target
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        if self.translator is not None:
            if self.translator.name == translator:
                self.last_set_success = True
                self.finish_set_module.emit()
                return
        
        try:
            params = cfg_module.translator_params.get(translator)
            translator_module: BaseTranslator = self._prepare_module_class(translator)
            self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
            if params is not None:
                self.translator = translator_module(source, target, raise_unsupported_lang=False, **params)
            else:
                self.translator = translator_module(source, target, raise_unsupported_lang=False)
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            if not pcfg.module.load_model_on_demand:
                self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
                self.translator.load_model()
                if self.cancel_event.is_set():
                    raise DownloadCancelled('Module preparation cancelled by user.')
            cfg_module.translate_source = self.translator.lang_source
            cfg_module.translate_target = self.translator.lang_target
            cfg_module.translator = self.translator.name
            if old_translator is not None:
                old_translator.unload_model(empty_cache=True)
                del old_translator
            self.last_set_success = True
        except DownloadCancelled as e:
            self.translator = old_translator
            self.last_error = e
            LOGGER.info(f'Cancelled preparing translator {translator}.')
        except MissingDependency as e:
            self.translator = old_translator
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.translator = old_translator
            missing = self._missing_requirements_from_exception(translator, e)
            if missing:
                self.last_error = e
                self.last_missing_requirements = missing
            else:
                self.last_error = e
        finally:
            self.module = self.translator
            self.finish_set_module.emit()

    def setTranslator(self, translator: str):
        self.job = lambda : self._set_translator(translator)
        self.start()

    def installMissingPackagesAndSetModule(self, translator: str, requirements: List[str]):
        self.job = lambda: self._install_missing_then_set_translator(translator, requirements)
        self.start()

    def _install_missing_then_set_translator(self, translator: str, requirements: List[str]):
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = self._package_manager().install(requirements, progress_callback=self._emit_prepare_progress)
        if not result.ok:
            self.last_error = RuntimeError(
                f'Failed to install package(s): {", ".join(requirements)}\n'
                f'Command: {result.command_text}\n'
                f'Exit code: {result.returncode}\n'
                f'{result.stderr or result.stdout or result.error}'
            )
            self.module = self.translator
            self.finish_set_module.emit()
            return
        self._set_translator(translator)

    def _translate_page(self, page_dict, page_key: str, emit_finished=True):
        page = page_dict[page_key]
        try:
            self.translator.translate_textblk_lst(page)
        except Exception as e:
            create_error_dialog(e, self.tr('Translation Failed.'), 'TranslationFailed')
        if emit_finished:
            self.finish_translate_page.emit(page_key)

    def translatePage(self, page_dict, page_key: str):
        self.job = lambda: self._translate_page(page_dict, page_key)
        self.start()

    def push_pagekey_queue(self, page_key: str):
        self.pipeline_pagekey_queue.append(page_key)

    def runTranslatePipeline(self, imgtrans_proj: ProjImgTrans, stop_event: threading.Event):
        self.initImgtransPipeline(imgtrans_proj, stop_event)
        self.job = self._run_translate_pipeline
        self.start()


    def _run_translate_pipeline(self):
        delay = self.translator.delay()
        stop_event = self.pipeline_stop_event or threading.Event()

        while not self.pipeline_finished():
            if stop_event.is_set():
                self.module_thread_stopped.emit()
                break

            if len(self.pipeline_pagekey_queue) == 0:
                stop_event.wait(0.1)
                continue
            
            page_key = self.pipeline_pagekey_queue.pop(0)
            self.blockSignals(True)
            trans_success = True
            try:
                self._translate_page(self.imgtrans_proj.pages, page_key, emit_finished=False)
            except Exception as e:
                # TODO: allowing retry/skip/terminate
                trans_success = False
                msg = self.tr('Translation Failed.')
                if isinstance(e, MissingTranslatorParams):
                    msg = msg + '\n' + str(e) + self.tr(' is required for ' + self.translator.name)
                    
                self.blockSignals(False)
                create_error_dialog(e, msg, 'TranslationFailed')
                # self.imgtrans_proj = None
                # self.finished_counter = 0
                # self.pipeline_pagekey_queue = []
                # return
            self.blockSignals(False)
            self.finished_counter += 1
            if trans_success:
                self.imgtrans_proj.update_page_progress(page_key, RunStatus.FIN_TRANSLATE)
            self.progress_changed.emit(self.finished_counter)

            if not self.pipeline_finished() and delay > 0:
                stop_event.wait(delay)


class ImgtransThread(QThread):

    pipeline_stopped = Signal()
    update_detect_progress = Signal(int)
    update_ocr_progress = Signal(int)
    update_translate_progress = Signal(int)
    update_inpaint_progress = Signal(int)

    finish_blktrans_stage = Signal(str, int)
    finish_blktrans = Signal(int, list)
    unload_modules = Signal(list)

    detect_counter = 0
    ocr_counter = 0
    translate_counter = 0
    inpaint_counter = 0

    def __init__(self, 
                 textdetect_thread: TextDetectThread,
                 ocr_thread: OCRThread,
                 translate_thread: TranslateThread,
                 inpaint_thread: InpaintThread,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.textdetect_thread = textdetect_thread
        self.ocr_thread = ocr_thread
        self.translate_thread = translate_thread
        self.translate_thread.module_thread_stopped.connect(self.on_module_thread_stopped)
        self.translate_thread.finished.connect(self.on_module_thread_stopped)
        self.inpaint_thread = inpaint_thread
        self.job = None
        self.imgtrans_proj: ProjImgTrans = None
        self.stop_event = threading.Event()
        self._pipeline_stop_emitted = False
        self.pages_to_process = None  # 需要处理的页面列表（用于继续运行模式）

    def on_module_thread_stopped(self):
        self._emit_pipeline_stopped_if_ready(imgtrans_running=self.isRunning())

    def isStopRequested(self):
        return self.stop_event.is_set()

    def clearStopRequest(self):
        self.stop_event.clear()
        self._pipeline_stop_emitted = False

    def _emit_pipeline_stopped_if_ready(self, imgtrans_running: bool):
        if not self.isStopRequested() or self._pipeline_stop_emitted:
            return
        if imgtrans_running or self.translate_thread.isRunning():
            return
        self._pipeline_stop_emitted = True
        self.pipeline_stopped.emit()

    @property
    def textdetector(self) -> TextDetectorBase:
        return self.textdetect_thread.textdetector

    @property
    def ocr(self) -> OCRBase:
        return self.ocr_thread.ocr
    
    @property
    def translator(self) -> BaseTranslator:
        return self.translate_thread.translator

    @property
    def inpainter(self) -> InpainterBase:
        return self.inpaint_thread.inpainter

    def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans, pages_to_process=None):
        self.imgtrans_proj = imgtrans_proj
        self.pages_to_process = pages_to_process  # 保存需要处理的页面列表
        self.num_pages = len(self.imgtrans_proj.pages)
        self.clearStopRequest()
        # 创建处理索引到实际页面索引的映射
        self.process_idx_to_page_idx = {}
        self.job = self._imgtrans_pipeline
        self.start()
    
    def requestStop(self):
        """请求停止当前任务"""
        self.stop_event.set()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.clearStopRequest()
        self.job = lambda : self._blktrans_pipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)
        self.start()

    def _blktrans_pipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if mode >= 0 and mode < 3:
            try:
                self.ocr_thread.module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except Exception as e:
                create_error_dialog(e, self.tr('OCR Failed.'), 'OCRFailed')
            self.finish_blktrans.emit(mode, blk_ids)

        if mode != 0 and mode < 3:
            self.translate_thread.module.translate_textblk_lst(blk_list)
            self.finish_blktrans.emit(mode, blk_ids)
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
                    maskseg_method = get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=tgt_mask[y1: y2, x1: x2])
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        blk.region_inpaint_dict = {'img': im, 'mask': mask, 'inpaint_rect': [x1, y1, x2, y2], 'inpainted': inpainted}
                    self.finish_blktrans_stage.emit('inpaint', int((ii+1) * progress_prod))
        self.finish_blktrans.emit(mode, blk_ids)

    def _imgtrans_pipeline(self):
        self.detect_counter = 0
        self.ocr_counter = 0
        self.translate_counter = 0
        self.inpaint_counter = 0
        
        # 如果指定了pages_to_process，只处理这些页面
        all_pages = list(self.imgtrans_proj.pages.keys())
        if self.pages_to_process is not None and len(self.pages_to_process) > 0:
            pages_to_iterate = self.pages_to_process
            self.num_pages = num_pages = len(self.pages_to_process)
            # 建立处理索引到实际页面索引的映射
            for process_idx, page_name in enumerate(pages_to_iterate):
                if page_name in all_pages:
                    self.process_idx_to_page_idx[process_idx] = all_pages.index(page_name)
            LOGGER.info(f'Processing specific pages: {len(pages_to_iterate)} pages')
        else:
            pages_to_iterate = all_pages
            self.num_pages = num_pages = len(self.imgtrans_proj.pages)
            # 处理索引等于实际页面索引
            for i in range(num_pages):
                self.process_idx_to_page_idx[i] = i
            LOGGER.info(f'Processing all {num_pages} pages')
        self.textdetect_thread.num_process_pages = self.num_pages
        self.ocr_thread.num_process_pages = self.num_pages
        self.inpaint_thread.num_process_pages = self.num_pages
        self.translate_thread.num_process_pages = self.num_pages

        low_vram_trans = False
        if self.translator is not None:
            low_vram_trans = self.translator.low_vram_mode
            self.parallel_trans = not self.translator.is_computational_intensive() and not low_vram_trans
        else:
            self.parallel_trans = False
        if self.parallel_trans and cfg_module.enable_translate:
            self.translate_thread.runTranslatePipeline(self.imgtrans_proj, self.stop_event)

        for imgname in pages_to_iterate:
            
            # 检查是否请求停止
            if self.isStopRequested():
                LOGGER.info('Image translation pipeline stopped by user')
                break
                
            img = self.imgtrans_proj.read_img(imgname)
            mask = blk_list = None
            need_save_mask = False
            blk_removed: List[TextBlock] = []
            if cfg_module.enable_detect:
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                except Exception as e:
                    create_error_dialog(e, self.tr('Text Detection Failed.'), 'TextDetectFailed')
                    blk_list = []
                self.detect_counter += 1
                if pcfg.module.keep_exist_textlines:
                    blk_list = self.imgtrans_proj.pages[imgname] + blk_list
                    blk_list = sort_regions(blk_list)
                    existed_mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    if existed_mask is not None:
                        mask = np.bitwise_or(mask, existed_mask)
                self.imgtrans_proj.pages[imgname] = blk_list

                if mask is not None and not cfg_module.enable_ocr:
                    self.imgtrans_proj.save_mask(imgname, mask)
                    need_save_mask = False
                    
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_DET)
                self.update_detect_progress.emit(self.detect_counter)

            if blk_list is None:
                blk_list = self.imgtrans_proj.pages[imgname] if imgname in self.imgtrans_proj.pages else []

            if cfg_module.enable_ocr:
                try:
                    self.ocr.run_ocr(img, blk_list)
                except Exception as e:
                    create_error_dialog(e, self.tr('OCR Failed.'), 'OCRFailed')
                self.ocr_counter += 1

                if pcfg.restore_ocr_empty:
                    blk_list_updated = []
                    for blk in blk_list:
                        text = blk.get_text()
                        if text_is_empty(text):
                            blk_removed.append(blk)
                        else:
                            blk_list_updated.append(blk)

                    if len(blk_removed) > 0:
                        blk_list.clear()
                        blk_list += blk_list_updated
                        
                        if mask is None:
                            mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                        if mask is not None:
                            inpainted = None
                            if not cfg_module.enable_inpaint:
                                inpainted = self.imgtrans_proj.load_inpainted_by_imgname(imgname)
                            for blk in blk_removed:
                                xywh = blk.bounding_rect()
                                blk_mask, xyxy = get_block_mask(xywh, mask, blk.angle)
                                x1, y1, x2, y2 = xyxy
                                if blk_mask is not None:
                                    mask[y1: y2, x1: x2] = 0
                                    if inpainted is not None:
                                        mskpnt = np.where(blk_mask)
                                        inpainted[y1: y2, x1: x2][mskpnt] = img[y1: y2, x1: x2][mskpnt]
                                    need_save_mask = True
                            if inpainted is not None and need_save_mask:
                                self.imgtrans_proj.save_inpainted(imgname, inpainted)
                            if need_save_mask:
                                self.imgtrans_proj.save_mask(imgname, mask)
                                need_save_mask = False

                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)

            if need_save_mask and mask is not None:
                self.imgtrans_proj.save_mask(imgname, mask)
                need_save_mask = False

            if cfg_module.enable_translate:
                if self.parallel_trans:
                    self.translate_thread.push_pagekey_queue(imgname)
                elif not low_vram_trans:
                    self.translator.translate_textblk_lst(blk_list)
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)
                        
            if cfg_module.enable_inpaint:
                if mask is None:
                    mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    
                if mask is not None:
                    try:
                        inpainted = self.inpainter.inpaint(img, mask, blk_list)
                        self.imgtrans_proj.save_inpainted(imgname, inpainted)
                    except Exception as e:
                        create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
                    
                self.inpaint_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                self.update_inpaint_progress.emit(self.inpaint_counter)
            else:
                if len(blk_removed) > 0:
                    self.imgtrans_proj.load_mask_by_imgname
        
        if cfg_module.enable_translate and low_vram_trans:
            unload_modules(self, ['textdetector', 'inpainter', 'ocr'])
            for imgname in pages_to_iterate:
                # 检查是否请求停止
                if self.isStopRequested():
                    LOGGER.info('Translation stopped by user')
                    break
                    
                blk_list = self.imgtrans_proj.pages[imgname]
                self.translator.translate_textblk_lst(blk_list)
                self.translate_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
                self.update_translate_progress.emit(self.translate_counter)

        self._emit_pipeline_stopped_if_ready(imgtrans_running=False)

    def detect_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.detect_counter == self.num_pages or not cfg_module.enable_detect

    def ocr_finished(self) -> bool:
        if self.imgtrans_proj is None:
            return True
        return self.ocr_counter == self.num_pages or not cfg_module.enable_ocr

    def translate_finished(self) -> bool:
        if self.imgtrans_proj is None \
            or not cfg_module.enable_ocr \
            or not cfg_module.enable_translate:
            return True
        if self.parallel_trans:
            # 检查翻译计数器是否达到需要处理的页面数
            return self.translate_thread.finished_counter >= self.num_pages
        return self.translate_counter == self.num_pages or not cfg_module.enable_translate

    def inpaint_finished(self) -> bool:
        if self.imgtrans_proj is None or not cfg_module.enable_inpaint:
            return True
        return self.inpaint_counter == self.num_pages or not cfg_module.enable_inpaint

    def run(self):
        if self.job is not None:
            self.job()
        self.job = None

    def recent_finished_index(self, ref_counter: int) -> int:
        if cfg_module.enable_detect:
            ref_counter = min(ref_counter, self.detect_counter)
        if cfg_module.enable_ocr:
            ref_counter = min(ref_counter, self.ocr_counter)
        if cfg_module.enable_inpaint:
            ref_counter = min(ref_counter, self.inpaint_counter)
        if cfg_module.enable_translate:
            if self.parallel_trans:
                ref_counter = min(ref_counter, self.translate_thread.finished_counter)
            else:
                ref_counter = min(ref_counter, self.translate_counter)

        process_idx = ref_counter - 1
        # 将处理索引转换为实际页面索引
        if hasattr(self, 'process_idx_to_page_idx') and process_idx in self.process_idx_to_page_idx:
            return self.process_idx_to_page_idx[process_idx]
        return process_idx


def unload_modules(self, module_names):
    model_deleted = False
    for module in module_names:
        module: BaseModule = getattr(self, module)
        if module is not None:
            module_deleted = module.unload_model()
            model_deleted = model_deleted or module_deleted
    if model_deleted:
        soft_empty_cache()


class ModuleManager(QObject):
    imgtrans_proj: ProjImgTrans = None

    finish_translate_page = Signal(str)
    canvas_inpaint_finished = Signal(dict)
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)

    run_canvas_inpaint = False
    is_waiting_th = False
    block_set_inpainter = False

    def __init__(self, 
                 imgtrans_proj: ProjImgTrans,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.imgtrans_proj = imgtrans_proj
        self.check_inpaint_fin_timer = QTimer(self)
        self.check_inpaint_fin_timer.timeout.connect(self.check_inpaint_th_finished)
        self.prepare_msgbox: ProgressMessageBox = None
        self._preparing_thread: ModuleThread = None
        self._pending_prepare_queue = []
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._pending_batch_package_queue = []
        self._pending_batch_package_success = None
        self._pending_batch_package_failure = None
        self._package_install_retried = set()
        self.package_install_thread: PackageInstallThread = None

    def setupThread(self, config_panel: ConfigPanel, imgtrans_progress_msgbox: ImgtransProgressMessageBox, ocr_postprocess: Callable = None, translate_preprocess: Callable = None, translate_postprocess: Callable = None, parent_widget=None):
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()
        
        self.translate_thread = TranslateThread()
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)
        self.translate_thread.finish_translate_page.connect(self.on_finish_translate_page)  

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)

        self.prepare_msgbox = ProgressMessageBox(self.tr('Preparing module: '), True, parent_widget)
        self.prepare_msgbox.setStyleSheet(imgtrans_progress_msgbox.styleSheet())
        self.prepare_msgbox.stop_clicked.connect(self.cancelModulePreparation)

        for module_thread in [self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread]:
            module_thread.module_prepare_progress.connect(self.on_module_prepare_progress)
            module_thread.finish_set_module.connect(lambda th=module_thread: self.on_module_prepare_finished(th))

        self.package_install_thread = PackageInstallThread()
        self.package_install_thread.package_prepare_progress.connect(self.on_module_prepare_progress)
        self.package_install_thread.finish_install.connect(self.on_batch_package_install_finished)

        self.progress_msgbox = imgtrans_progress_msgbox
        self.progress_msgbox.stop_clicked.connect(self.stopImgtransPipeline)

        self.imgtrans_thread = ImgtransThread(self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread)
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)

        self.translator_panel = translator_panel = config_panel.trans_config_panel        
        resolve_device_defaults = program_config.config_created_on_load

        translator_params = merge_config_module_params(
            cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get,
            resolve_runtime_device_defaults=resolve_device_defaults)
        translator_panel.addModulesParamWidgets(translator_params)
        translator_panel.translator_changed.connect(self.setTranslator)
        translator_panel.paramwidget_edited.connect(self.on_translatorparam_edited)
        translator_panel.translateByTextblockBox.checker_changed.connect(self.on_translatebyblock_checker_changed)
        translator_panel.translateByTextblockBox.checker.setChecked(cfg_module.translate_by_textblock)

        from ballontranslator.modules.translators.hooks import chs2cht
        BaseTranslator.register_preprocess_hooks({'keyword_sub': translate_preprocess})
        BaseTranslator.register_postprocess_hooks({'chs2cht': chs2cht, 'keyword_sub': translate_postprocess})

        self.inpaint_panel = inpainter_panel = config_panel.inpaint_config_panel
        inpainter_params = merge_config_module_params(
            cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get,
            resolve_runtime_device_defaults=resolve_device_defaults)
        inpainter_panel.addModulesParamWidgets(inpainter_params)
        inpainter_panel.paramwidget_edited.connect(self.on_inpainterparam_edited)
        inpainter_panel.inpainter_changed.connect(self.setInpainter)
        inpainter_panel.needInpaintChecker.checker_changed.connect(self.on_inpainter_checker_changed)
        inpainter_panel.needInpaintChecker.checker.setChecked(cfg_module.check_need_inpaint)

        self.textdetect_panel = textdetector_panel = config_panel.detect_config_panel
        textdetector_params = merge_config_module_params(
            cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get,
            resolve_runtime_device_defaults=resolve_device_defaults)
        textdetector_panel.addModulesParamWidgets(textdetector_params)
        textdetector_panel.paramwidget_edited.connect(self.on_textdetectorparam_edited)
        textdetector_panel.detector_changed.connect(self.setTextDetector)

        self.ocr_panel = ocr_panel = config_panel.ocr_config_panel
        ocr_params = merge_config_module_params(
            cfg_module.ocr_params, GET_VALID_OCR(), OCR.get,
            resolve_runtime_device_defaults=resolve_device_defaults)
        ocr_panel.addModulesParamWidgets(ocr_params)
        ocr_panel.paramwidget_edited.connect(self.on_ocrparam_edited)
        ocr_panel.ocr_changed.connect(self.setOCR)
        OCRBase.register_postprocess_hooks(ocr_postprocess)

        config_panel.unload_models.connect(self.unload_all_models)
        config_panel.prepare_selected_modules.connect(self.prepareSelectedModules)


    def _thread_for_module_key(self, module_key: str) -> ModuleThread:
        return {
            'textdetector': self.textdetect_thread,
            'ocr': self.ocr_thread,
            'translator': self.translate_thread,
            'inpainter': self.inpaint_thread,
        }[module_key]

    def _setter_for_module_key(self, module_key: str):
        return {
            'textdetector': self.setTextDetector,
            'ocr': self.setOCR,
            'translator': self.setTranslator,
            'inpainter': self.setInpainter,
        }[module_key]

    def _registry_for_module_key(self, module_key: str) -> Registry:
        return {
            'textdetector': TEXTDETECTORS,
            'ocr': OCR,
            'translator': TRANSLATORS,
            'inpainter': INPAINTERS,
        }[module_key]

    def _module_ready(self, module_key: str, module_name: str) -> bool:
        thread = self._thread_for_module_key(module_key)
        module = thread.module
        return module is not None and module.name == module_name

    def translator_metadata(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        if self.translator is not None and self.translator.name == translator:
            src_list = self.translator.supported_src_list
            tgt_list = self.translator.supported_tgt_list
            source = self.translator.lang_source
            target = self.translator.lang_target
        else:
            spec = TRANSLATORS.get_spec(translator)
            default_langs = list(modules.translators.LANGMAP_GLOBAL.keys())
            src_list = spec.supported_src_list if spec is not None and spec.supported_src_list else default_langs
            tgt_list = spec.supported_tgt_list if spec is not None and spec.supported_tgt_list else default_langs
            if not src_list:
                src_list = default_langs
            if not tgt_list:
                tgt_list = default_langs
            source = cfg_module.translate_source if cfg_module.translate_source in src_list else src_list[0]
            target = cfg_module.translate_target if cfg_module.translate_target in tgt_list else tgt_list[0]
        return {
            'name': translator,
            'supported_src_list': src_list,
            'supported_tgt_list': tgt_list,
            'lang_source': source,
            'lang_target': target,
        }

    def _prepare_modules_then(self, required_modules: List[tuple], on_success: Callable, on_failure: Callable = None):
        # RUN may need several stages; prepare them serially to keep one modal dialog.
        queue = [(module_key, module_name) for module_key, module_name in required_modules if not self._module_ready(module_key, module_name)]
        if not queue:
            on_success()
            return
        missing_modules = self._missing_module_requirements_for_modules(queue)
        if missing_modules:
            self._handle_batch_missing_packages(queue, missing_modules, on_success, on_failure)
            return
        self._begin_prepare_queue(queue, on_success, on_failure)

    def _begin_prepare_queue(self, queue: List[tuple], on_success: Callable, on_failure: Callable = None):
        self._package_install_retried.clear()
        self._pending_prepare_queue = queue
        self._pending_prepare_success = on_success
        self._pending_prepare_failure = on_failure
        self._continue_pending_prepare()

    def _missing_module_requirements_for_modules(self, required_modules: List[tuple]) -> List[MissingModuleRequirements]:
        module_specs = []
        for module_key, module_name in required_modules:
            spec = self._registry_for_module_key(module_key).get_spec(module_name)
            if spec is None:
                continue
            module_specs.append((module_key, module_name, spec))
        return collect_missing_module_requirements(module_specs, self._package_manager())

    def _continue_pending_prepare(self):
        while self._pending_prepare_queue:
            module_key, module_name = self._pending_prepare_queue.pop(0)
            if self._module_ready(module_key, module_name):
                continue
            self._setter_for_module_key(module_key)(module_name)
            return

        on_success = self._pending_prepare_success
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._package_install_retried.clear()
        if on_success is not None:
            on_success()

    def _package_manager(self) -> PyPackageManager:
        pmcfg = pcfg.package_manager
        return PyPackageManager(
            backend=pmcfg.installer_backend,
            extra_args=pmcfg.extra_install_args,
            package_import_names=PACKAGE_IMPORT_NAMES,
        )

    @staticmethod
    def _combined_missing_requirements(missing_modules: List[MissingModuleRequirements]) -> List[str]:
        """Return a de-duplicated install list for a batch of modules.

        >>> items = [
        ...     MissingModuleRequirements('ocr', 'mit48px', ['torch', 'einops']),
        ...     MissingModuleRequirements('textdetector', 'ctd', ['torch', 'torchvision']),
        ... ]
        >>> ModuleManager._combined_missing_requirements(items)
        ['torch', 'einops', 'torchvision']
        """

        return list(dict.fromkeys(
            requirement
            for missing_module in missing_modules
            for requirement in missing_module.requirements
        ))

    @staticmethod
    def _missing_modules_text(missing_modules: List[MissingModuleRequirements]) -> str:
        """Format missing packages for the batch install dialog.

        >>> item = MissingModuleRequirements('ocr', 'mit48px', ['torch', 'einops'])
        >>> ModuleManager._missing_modules_text([item])
        'ocr/mit48px: torch, einops'
        """

        return '\n'.join(
            f'{item.module_key}/{item.module_name}: {", ".join(item.requirements)}'
            for item in missing_modules
        )

    def _handle_batch_missing_packages(
        self,
        queue: List[tuple],
        missing_modules: List[MissingModuleRequirements],
        on_success: Callable,
        on_failure: Callable = None,
    ):
        requirements = self._combined_missing_requirements(missing_modules)
        missing_text = self._missing_modules_text(missing_modules)

        if pcfg.package_manager.auto_install_missing_packages:
            self._install_batch_missing_packages(queue, requirements, on_success, on_failure)
            return

        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            LOGGER.error(f'Selected modules require package(s):\n{missing_text}')
            if on_failure is not None:
                on_failure()
            return

        if not pcfg.package_manager.show_missing_package_prompts:
            create_error_dialog(
                RuntimeError(self.tr('Selected modules require missing package(s):\n{modules}').format(modules=missing_text)),
                self.tr('Missing packages'),
                'MissingPackages:batch',
            )
            if on_failure is not None:
                on_failure()
            return

        if self._ask_install_missing_package_batch(missing_modules, requirements):
            self._install_batch_missing_packages(queue, requirements, on_success, on_failure)
            return

        if on_failure is not None:
            on_failure()

    def _ask_install_missing_package_batch(
        self,
        missing_modules: List[MissingModuleRequirements],
        requirements: List[str],
    ) -> bool:
        command_preview = self._package_manager().preview_command(requirements)
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Question)
        msgbox.setWindowTitle(self.tr('Missing packages'))
        msgbox.setText(
            self.tr('Selected modules require missing package(s):\n{modules}').format(
                modules=self._missing_modules_text(missing_modules),
            )
        )
        msgbox.setInformativeText(self.tr('Install the missing package(s) now?'))
        msgbox.setDetailedText(command_preview)
        install_btn = msgbox.addButton(self.tr('Install'), QMessageBox.AcceptRole)
        msgbox.addButton(self.tr('Not Now'), QMessageBox.RejectRole)
        remember_checker = QCheckBox(self.tr("Remember and don't show again"))
        msgbox.setCheckBox(remember_checker)
        msgbox.exec()
        if remember_checker.isChecked():
            pcfg.package_manager.show_missing_package_prompts = False
            program_config.save_config()
        return msgbox.clickedButton() is install_btn

    def _install_batch_missing_packages(
        self,
        queue: List[tuple],
        requirements: List[str],
        on_success: Callable,
        on_failure: Callable = None,
    ):
        if self.package_install_thread is None:
            raise RuntimeError('Package install thread is not initialized.')
        if self.package_install_thread.isRunning():
            LOGGER.warning('Package installation is already running.')
            if on_failure is not None:
                on_failure()
            return
        self._pending_batch_package_queue = queue
        self._pending_batch_package_success = on_success
        self._pending_batch_package_failure = on_failure
        self._show_package_install_dialog(requirements)
        self.package_install_thread.installPackages(requirements)

    def _show_package_install_dialog(self, requirements: List[str]):
        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            LOGGER.info(f'Installing package(s): {", ".join(requirements)}')
            return
        self.prepare_msgbox.zero_progress()
        self.prepare_msgbox.setTaskName(self.tr('Installing packages: '))
        self.prepare_msgbox.updateTaskProgress(0, ', '.join(requirements))
        if self.prepare_msgbox.stop_button is not None:
            self.prepare_msgbox.stop_button.setEnabled(False)
            self.prepare_msgbox.stop_button.setText(self.tr('Installing...'))
        self.prepare_msgbox.setFixedWidth(self.progress_msgbox.sizeHint().width())
        self.prepare_msgbox.show_fitted()

    def on_batch_package_install_finished(self):
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.prepare_msgbox.done(0)

        queue = self._pending_batch_package_queue
        on_success = self._pending_batch_package_success
        on_failure = self._pending_batch_package_failure
        self._pending_batch_package_queue = []
        self._pending_batch_package_success = None
        self._pending_batch_package_failure = None

        if self.package_install_thread.last_success:
            if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
                self._begin_prepare_queue(queue, on_success, on_failure)
            else:
                QTimer.singleShot(0, lambda: self._begin_prepare_queue(queue, on_success, on_failure))
            return

        if self.package_install_thread.last_error is not None:
            create_error_dialog(
                self.package_install_thread.last_error,
                self.tr('Failed to install packages'),
                'InstallPackages:batch',
            )
        if on_failure is not None:
            on_failure()

    def _show_prepare_dialog(self, thread: ModuleThread, module_name: str):
        self._preparing_thread = thread
        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            LOGGER.info(f'Preparing {thread.module_key} module: {module_name}')
            return
        # Match the RUN progress widget geometry/style for first-run preparation.
        self.prepare_msgbox.zero_progress()
        self.prepare_msgbox.setTaskName(self.tr('Preparing module: '))
        self.prepare_msgbox.updateTaskProgress(0, module_name)
        self.prepare_msgbox.setFixedWidth(self.progress_msgbox.sizeHint().width())
        self.prepare_msgbox.show_fitted()

    def cancelModulePreparation(self):
        if self._preparing_thread is not None:
            self._preparing_thread.requestCancelModuleInit()

    def on_module_prepare_progress(self, payload: dict):
        # Download callbacks report bytes; imports/model-loads report coarse stages.
        event = payload.get('event', '')
        module_name = payload.get('module', '')
        message = payload.get('message', '')
        path = payload.get('path') or payload.get('file') or ''
        if path:
            path = osp.basename(path)

        progress = {
            'checking_dependencies': 5,
            'module_file_group': 10,
            'file_check': 15,
            'file_start': 20,
            'archive_extract': 75,
            'archive_move': 85,
            'installing_packages': 25,
            'package_output': 25,
            'importing': 88,
            'instantiating': 92,
            'loading_model': 96,
            'file_done': 100,
        }.get(event, 0)
        if event == 'file_progress':
            total = payload.get('total')
            downloaded = payload.get('downloaded') or 0
            if total:
                progress = max(20, min(74, int(downloaded / total * 54) + 20))
            else:
                progress = 35
        if not message:
            message = path or module_name

        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            if event in {'checking_dependencies', 'file_start', 'archive_extract', 'installing_packages', 'package_output', 'importing', 'loading_model'}:
                LOGGER.info(f'{message}: {event}')
            return
        if self.prepare_msgbox is not None:
            self.prepare_msgbox.updateTaskProgress(progress, message)

    def on_module_prepare_finished(self, thread: ModuleThread):
        if self._preparing_thread is thread:
            if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
                self.prepare_msgbox.done(0)
            self._preparing_thread = None

        if not thread.last_set_success and thread.last_missing_requirements:
            if self._handle_missing_packages(thread):
                return

        if self._pending_prepare_success is None and self._pending_prepare_failure is None:
            if not thread.last_set_success and thread.last_error is not None:
                self._show_module_prepare_failure(thread)
            return

        if thread.last_set_success:
            if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
                self._continue_pending_prepare()
            else:
                # Let Qt finish closing the prepare dialog before showing RUN progress.
                QTimer.singleShot(0, self._continue_pending_prepare)
        else:
            self._show_module_prepare_failure(thread)
            self._finish_pending_prepare_failure()

    def _finish_pending_prepare_failure(self):
        on_failure = self._pending_prepare_failure
        self._pending_prepare_queue = []
        self._pending_prepare_success = None
        self._pending_prepare_failure = None
        self._package_install_retried.clear()
        if on_failure is not None:
            on_failure()

    def _clear_install_retry_if_standalone_prepare(self):
        if self._pending_prepare_success is None and self._pending_prepare_failure is None:
            self._package_install_retried.clear()

    def _handle_missing_packages(self, thread: ModuleThread) -> bool:
        requirements = list(dict.fromkeys(thread.last_missing_requirements))
        retry_key = (thread.module_key, thread.last_set_module_name, tuple(requirements))
        if retry_key in self._package_install_retried:
            self._show_module_prepare_failure(thread)
            if self._pending_prepare_success is not None or self._pending_prepare_failure is not None:
                self._finish_pending_prepare_failure()
            return True

        if pcfg.package_manager.auto_install_missing_packages:
            self._install_missing_packages_and_retry(thread, requirements, retry_key)
            return True

        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            LOGGER.error(
                f'Module {thread.last_set_module_name} requires package(s): '
                f'{", ".join(requirements)}'
            )
            return False

        if not pcfg.package_manager.show_missing_package_prompts:
            return False

        should_install = self._ask_install_missing_packages(thread, requirements)
        if should_install:
            self._install_missing_packages_and_retry(thread, requirements, retry_key)
            return True
        return False

    def _ask_install_missing_packages(self, thread: ModuleThread, requirements: List[str]) -> bool:
        command_preview = self._package_manager().preview_command(requirements)
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Question)
        msgbox.setWindowTitle(self.tr('Missing packages'))
        msgbox.setText(
            self.tr('Module "{module}" requires missing package(s):\n{packages}').format(
                module=thread.last_set_module_name,
                packages=', '.join(requirements),
            )
        )
        msgbox.setInformativeText(self.tr('Install the missing package(s) now?'))
        msgbox.setDetailedText(command_preview)
        install_btn = msgbox.addButton(self.tr('Install'), QMessageBox.AcceptRole)
        msgbox.addButton(self.tr('Not Now'), QMessageBox.RejectRole)
        remember_checker = QCheckBox(self.tr("Remember and don't show again"))
        msgbox.setCheckBox(remember_checker)
        msgbox.exec()
        if remember_checker.isChecked():
            pcfg.package_manager.show_missing_package_prompts = False
            program_config.save_config()
        return msgbox.clickedButton() is install_btn

    def _install_missing_packages_and_retry(self, thread: ModuleThread, requirements: List[str], retry_key: tuple):
        self._package_install_retried.add(retry_key)
        self._show_prepare_dialog(thread, thread.last_set_module_name)
        thread.installMissingPackagesAndSetModule(thread.last_set_module_name, requirements)

    def _show_module_prepare_failure(self, thread: ModuleThread):
        if thread.last_error is None:
            return
        msg = self.tr('Failed to set module ') + thread.last_set_module_name
        create_error_dialog(thread.last_error, msg, f'FailedSetModule:{thread.module_key}:{thread.last_set_module_name}')

    def unload_all_models(self):
        unload_modules(self, ('textdetector', 'inpainter', 'ocr', 'translator'))

    @property
    def translator(self) -> BaseTranslator:
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
        self._prepare_modules_then(
            [('translator', cfg_module.translator)],
            lambda: self.translate_thread.translatePage(self.imgtrans_proj.pages, page_key),
        )

    def inpainterBusy(self):
        return self.inpaint_thread.isRunning()

    def inpaint(self, img: np.ndarray, mask: np.ndarray, img_key: str = None, inpaint_rect = None, **kwargs):
        if self.inpaint_thread.isRunning():
            LOGGER.warning('Waiting for inpainting to finish')
            return
        self.inpaint_thread.inpaint(img, mask, img_key, inpaint_rect)

    def terminateRunningThread(self):
        if self.textdetect_thread.isRunning():
            self.textdetect_thread.quit()
        if self.ocr_thread.isRunning():
            self.ocr_thread.quit()
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.quit()
        if self.translate_thread.isRunning():
            self.translate_thread.quit()

    def _module_preparation_running(self) -> bool:
        module_threads = [self.textdetect_thread, self.ocr_thread, self.translate_thread, self.inpaint_thread]
        package_running = self.package_install_thread is not None and self.package_install_thread.isRunning()
        return package_running or any(thread.isRunning() for thread in module_threads)

    def _selected_module_name(self, module_key: str) -> str:
        panel = {
            'textdetector': self.textdetect_panel,
            'ocr': self.ocr_panel,
            'translator': self.translator_panel,
            'inpainter': self.inpaint_panel,
        }[module_key]
        selected = panel.module_combobox.currentText()
        if selected:
            return selected
        return getattr(cfg_module, module_key)

    def _selected_modules(self) -> List[tuple]:
        return [
            ('textdetector', self._selected_module_name('textdetector')),
            ('ocr', self._selected_module_name('ocr')),
            ('translator', self._selected_module_name('translator')),
            ('inpainter', self._selected_module_name('inpainter')),
        ]

    def prepareSelectedModules(self):
        if self.imgtrans_thread.isRunning() or self._module_preparation_running():
            create_info_dialog(self.tr('Module preparation is already running.'))
            return
        self._prepare_modules_then(
            self._selected_modules(),
            lambda: create_info_dialog(self.tr('Selected modules are ready.')),
        )

    def check_inpaint_th_finished(self):
        if self.inpaint_thread.isRunning():
            return
        self.block_set_inpainter = False
        self.check_inpaint_fin_timer.stop()
        self.inpaint_th_finished.emit()

    def runImgtransPipeline(self, pages_to_process=None):
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if cfg_module.all_stages_disabled() and self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            for ii in range(self.imgtrans_proj.num_pages):
                self.page_trans_finished.emit(ii)
            self.imgtrans_pipeline_finished.emit()
            return

        required_modules = []
        if cfg_module.enable_detect:
            required_modules.append(('textdetector', cfg_module.textdetector))
        if cfg_module.enable_ocr:
            required_modules.append(('ocr', cfg_module.ocr))
        if cfg_module.enable_translate:
            required_modules.append(('translator', cfg_module.translator))
        if cfg_module.enable_inpaint:
            required_modules.append(('inpainter', cfg_module.inpainter))
        self._prepare_modules_then(
            required_modules,
            lambda: self._startImgtransPipeline(pages_to_process),
            on_failure=lambda: self.imgtrans_pipeline_finished.emit() if (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) else None,
        )

    def _startImgtransPipeline(self, pages_to_process=None):
        if self.prepare_msgbox is not None and self.prepare_msgbox.isVisible():
            self.prepare_msgbox.done(0)
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.translate_bar.setVisible(cfg_module.enable_translate)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show_fitted()
        self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj, pages_to_process)
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        LOGGER.info('Stopping image translation pipeline...')
        self.imgtrans_thread.requestStop()

    def runBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        self.terminateRunningThread()
        required_modules = []
        if mode >= 0 and mode < 3:
            required_modules.append(('ocr', cfg_module.ocr))
        if mode != 0 and mode < 3:
            required_modules.append(('translator', cfg_module.translator))
        if mode >= 2:
            required_modules.append(('inpainter', cfg_module.inpainter))
        self._prepare_modules_then(
            required_modules,
            lambda: self._startBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask),
        )

    def _startBlktransPipeline(self, blk_list: List[TextBlock], tgt_img: np.ndarray, mode: int, blk_ids: List[int], tgt_mask):
        if self.prepare_msgbox is not None and self.prepare_msgbox.isVisible():
            self.prepare_msgbox.done(0)
        self.progress_msgbox.hide_all_bars()
        if mode >= 0 and mode < 3:
            self.progress_msgbox.ocr_bar.show()
        if mode >= 2:
            self.progress_msgbox.inpaint_bar.show()
        if mode != 0 and mode < 3:
            self.progress_msgbox.translate_bar.show()
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show_fitted()
        self.imgtrans_thread.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask)

    def on_finish_blktrans_stage(self, stage: str, progress: int):
        if stage == 'ocr':
            self.progress_msgbox.updateOCRProgress(progress)
        elif stage == 'translate':
            self.progress_msgbox.updateTranslateProgress(progress)
        elif stage == 'inpaint':
            self.progress_msgbox.updateInpaintProgress(progress)
        else:
            raise NotImplementedError(f'Unknown stage: {stage}')
        
    def on_finish_blktrans(self, mode: int, blk_ids: List):
        self.blktrans_pipeline_finished.emit(mode, blk_ids)
        self.progress_msgbox.hide()

    def on_update_detect_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'detect' in shared.pbar:
            shared.pbar['detect'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateDetectProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_ocr_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'ocr' in shared.pbar:
            shared.pbar['ocr'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateOCRProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_translate_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'translate' in shared.pbar:
            shared.pbar['translate'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateTranslateProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_inpaint_progress(self, progress: int):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if 'inpaint' in shared.pbar:
            shared.pbar['inpaint'].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        self.progress_msgbox.updateInpaintProgress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def progress(self):
        progress = {}
        num_pages = self.imgtrans_thread.num_pages
        if cfg_module.enable_detect:
            progress['detect'] = self.imgtrans_thread.detect_counter / num_pages
        if cfg_module.enable_ocr:
            progress['ocr'] = self.imgtrans_thread.ocr_counter / num_pages
        if cfg_module.enable_inpaint:
            progress['inpaint'] = self.imgtrans_thread.inpaint_counter / num_pages
        if cfg_module.enable_translate:
            progress['translate'] = self.imgtrans_thread.translate_counter / num_pages
        return progress

    def proj_finished(self):
        if self.imgtrans_thread.detect_finished() \
            and self.imgtrans_thread.ocr_finished() \
                and self.imgtrans_thread.translate_finished() \
                    and self.imgtrans_thread.inpaint_finished():
            return True
        return False

    def finishImgtransPipeline(self):
        if self.proj_finished():
            self.progress_msgbox.hide()
            self.imgtrans_pipeline_finished.emit()
    
    def on_imgtrans_thread_stopped(self):
        """线程完成时确保关闭进度对话框"""
        # 线程完成了，直接关闭窗口
        self.progress_msgbox.hide()
        self.imgtrans_pipeline_finished.emit()

    def setTranslator(self, translator: str = None):
        if translator is None:
            translator = cfg_module.translator
        if self.translate_thread.isRunning():
            LOGGER.warning('Cancelling a running translation/module preparation thread.')
            self.translate_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.translate_thread, translator)
        self.translate_thread.setTranslator(translator)

    def setInpainter(self, inpainter: str = None):
        
        if self.block_set_inpainter:
            return
        
        if inpainter is None:
            inpainter =cfg_module.inpainter
        
        if self.inpaint_thread.isRunning():
            self.inpaint_thread.requestCancelModuleInit()
            return

        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.inpaint_thread, inpainter)
        self.inpaint_thread.setInpainter(inpainter)

    def setTextDetector(self, textdetector: str = None):
        if textdetector is None:
            textdetector = cfg_module.textdetector
        if self.textdetect_thread.isRunning():
            LOGGER.warning('Cancelling a running text detection/module preparation thread.')
            self.textdetect_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.textdetect_thread, textdetector)
        self.textdetect_thread.setTextDetector(textdetector)

    def setOCR(self, ocr: str = None):
        if ocr is None:
            ocr = cfg_module.ocr
        if self.ocr_thread.isRunning():
            LOGGER.warning('Cancelling a running OCR/module preparation thread.')
            self.ocr_thread.requestCancelModuleInit()
            return
        self._clear_install_retry_if_standalone_prepare()
        self._show_prepare_dialog(self.ocr_thread, ocr)
        self.ocr_thread.setOCR(ocr)

    def on_finish_translate_page(self, page_key: str):
        self.finish_translate_page.emit(page_key)
    
    def on_finish_inpaint(self, inpaint_dict: dict):
        if self.run_canvas_inpaint:
            self.canvas_inpaint_finished.emit(inpaint_dict)
            self.run_canvas_inpaint = False

    def canvas_inpaint(self, inpaint_dict):
        self._prepare_modules_then(
            [('inpainter', cfg_module.inpainter)],
            lambda: self._start_canvas_inpaint(inpaint_dict),
        )

    def _start_canvas_inpaint(self, inpaint_dict):
        self.run_canvas_inpaint = True
        self.inpaint(**inpaint_dict)
    
    def on_translatorparam_edited(self, param_key: str, param_content: dict):
        self.updatePanelParam('translator', self.translator_panel, self.translator, param_key, param_content)

    def on_inpainterparam_edited(self, param_key: str, param_content: dict):
        self.updatePanelParam('inpainter', self.inpaint_panel, self.inpainter, param_key, param_content)

    def on_textdetectorparam_edited(self, param_key: str, param_content: dict):
        self.updatePanelParam('textdetector', self.textdetect_panel, self.textdetector, param_key, param_content)

    def on_ocrparam_edited(self, param_key: str, param_content: dict):
        self.updatePanelParam('ocr', self.ocr_panel, self.ocr, param_key, param_content)

    def updatePanelParam(self, module_key: str, panel, active_module, param_key: str, param_content: dict):
        module_name = panel.module_combobox.currentText()
        if active_module is not None and active_module.name == module_name:
            self.updateModuleSetupParam(active_module, param_key, param_content)
            cfg_module.get_params(module_key)[module_name] = active_module.params
            return
        self.updateStoredModuleSetupParam(module_key, module_name, param_key, param_content)

    def updateStoredModuleSetupParam(self, module_key: str, module_name: str, param_key: str, param_content: dict):
        if param_content.get('flush', False):
            return

        module_params = cfg_module.get_params(module_key).get(module_name)
        if module_params is None or param_key not in module_params:
            return

        param_value = param_content['content']
        if param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module_params[param_key].get('path_filter', None)
            param_value = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if not osp.exists(param_value):
                return
            param_widget: ParamComboBox = param_content['widget']
            param_widget.setCurrentText(param_value)

        self.updateParamDictValue(module_params, param_key, param_value)

    def updateParamDictValue(self, params: dict, param_key: str, param_value, convert_dtype=True):
        p = params[param_key]
        if isinstance(p, dict):
            if convert_dtype:
                try:
                    val_type = p.get('data_type', type(p['value']))
                    param_value = val_type(param_value)
                except ValueError:
                    dtype = type(p['value'])
                    LOGGER.warning(f'Invalid param value {param_value} for defined dtype: {dtype}')
                    return
            p['value'] = param_value
        else:
            if convert_dtype:
                try:
                    param_value = type(p)(param_value)
                except ValueError:
                    LOGGER.warning(f'Invalid param value {param_value} for defined dtype: {type(p)}, revert to original value {p}')
                    return
            params[param_key] = param_value

    def updateModuleSetupParam(self, 
                               module: Union[InpainterBase, BaseTranslator],
                               param_key: str, param_content: dict):
            
        if param_content.get('flush', False):
            param_widget: ParamComboBox = param_content['widget']
            param_widget.blockSignals(True)
            current_item = param_widget.currentText()
            param_widget.clear()
            param_widget.addItems(module.flush(param_key))
            param_widget.setCurrentText(current_item)
            param_widget.blockSignals(False)
        elif param_content.get('select_path', False):
            dialog = QFileDialog()
            f = module.params[param_key].get('path_filter', None)
            p = dialog.getOpenFileUrl(self.parent(), filter=f)[0].toLocalFile()
            if osp.exists(p):
                param_widget: ParamComboBox = param_content['widget']
                param_widget.setCurrentText(p)
        else:
            module.updateParam(param_key, param_content['content'])

    def handle_page_changed(self):
        if not self.imgtrans_thread.isRunning():
            if self.inpaint_thread.inpainting:
                self.run_canvas_inpaint = False
                self.inpaint_thread.terminate()

    def on_inpainter_checker_changed(self, is_checked: bool):
        cfg_module.check_need_inpaint = is_checked
        InpainterBase.check_need_inpaint = is_checked

    def on_translatebyblock_checker_changed(self, is_checked: bool):
        cfg_module.translate_by_textblock = is_checked
        BaseTranslator.translate_by_textblock = is_checked
