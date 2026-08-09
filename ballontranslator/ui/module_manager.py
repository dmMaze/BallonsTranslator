import threading
from typing import Union, List, Callable
import os.path as osp

import numpy as np
from packaging.requirements import Requirement
from qtpy.QtCore import QThread, Signal, QObject, QLocale, QTimer
from qtpy.QtWidgets import QFileDialog, QMessageBox, QCheckBox

from .funcmaps import get_maskseg_method
from .packageinstall_thread import PackageInstallThread
from .package_manager import create_package_manager
from .torch_install_dialog import confirm_torch_install_device
from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.registry import LazyModuleError, Registry
from ballontranslator.utils.imgproc_utils import enlarge_window, get_block_mask
from ballontranslator.utils.io_utils import text_is_empty
from ballontranslator.modules.translators import MissingTranslatorParams
from ballontranslator.modules.exceptions import (
    LLMApiKeyRequiredError,
    LLMBaseURLRequiredError,
    LLMModelRequiredError,
    LLMRequestStopped,
    ModuleRunError,
)
from ballontranslator.modules.base import BaseModule, soft_empty_cache
from ballontranslator.modules import INPAINTERS, TRANSLATORS, TEXTDETECTORS, OCR, \
    GET_VALID_TRANSLATORS, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_OCR, \
    BaseTranslator, InpainterBase, TextDetectorBase, OCRBase, merge_config_module_params
import ballontranslator.modules as modules
modules.translators.SYSTEM_LANG = QLocale.system().name()
from ballontranslator.utils.textblock import TextBlock, sort_regions
from ballontranslator.utils import shared
from ballontranslator.utils.message import create_error_dialog, create_info_dialog
from ballontranslator.utils.download_util import DownloadCancelled, sizeof_fmt
from ballontranslator.modules.prepare_local_files import MissingDependency, ensure_module_files
from ballontranslator.utils.py_package_manager import (
    MissingModuleRequirements,
    collect_missing_module_requirements,
)
from .custom_widget import ImgtransProgressMessageBox, ParamComboBox, ProgressMessageBox
from .configpanel import ConfigPanel
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from ballontranslator.utils.config import pcfg, RunStatus, save_config
from ballontranslator.utils.llm_profiles import LLM_INPAINT_KEY
from ballontranslator.utils.global_callbacks import register_global_callback
cfg_module = pcfg.module

# RUN can report the same missing LLM key from multiple page workers.
_llm_key_dialog_lock = threading.Lock()
_shown_llm_key_dialog_profiles = set()
_llm_model_dialog_lock = threading.Lock()
_shown_llm_model_dialog_profiles = set()
_llm_base_url_dialog_lock = threading.Lock()
_shown_llm_base_url_dialog_profiles = set()


def _create_page_error_dialog(
    exception: Exception,
    error_msg: str,
    exception_type: str,
    page_key: str = None,
    page_label: str = 'Page',
):

    if page_key:
        error_msg = f'{error_msg}\n{page_label}: {page_key}'

    create_error_dialog(
        exception,
        error_msg,
        exception_type,
    )


def _show_llm_key_required_dialog(error: LLMApiKeyRequiredError):
    profile_key = error.profile_id or error.profile_name
    with _llm_key_dialog_lock:
        if profile_key in _shown_llm_key_dialog_profiles:
            return
        _shown_llm_key_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_key_dialog_in_mainthread(error.profile_id, error.profile_name)
    else:
        LOGGER.error(str(error))


def _show_llm_model_required_dialog(error: LLMModelRequiredError):
    profile_key = (error.profile_id or error.profile_name, error.target)
    with _llm_model_dialog_lock:
        if profile_key in _shown_llm_model_dialog_profiles:
            return
        _shown_llm_model_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_model_dialog_in_mainthread(error.profile_id, error.profile_name, error.target)
    else:
        LOGGER.error(str(error))


def _show_llm_base_url_required_dialog(error: LLMBaseURLRequiredError):
    profile_key = (error.profile_id or error.profile_name, error.target)
    with _llm_base_url_dialog_lock:
        if profile_key in _shown_llm_base_url_dialog_profiles:
            return
        _shown_llm_base_url_dialog_profiles.add(profile_key)
    if not shared.HEADLESS:
        shared.show_llm_base_url_dialog_in_mainthread(error.profile_id, error.profile_name, error.target)
    else:
        LOGGER.error(str(error))


def _reset_llm_key_required_dialogs():
    with _llm_key_dialog_lock:
        _shown_llm_key_dialog_profiles.clear()
    with _llm_model_dialog_lock:
        _shown_llm_model_dialog_profiles.clear()
    with _llm_base_url_dialog_lock:
        _shown_llm_base_url_dialog_profiles.clear()


class ModuleThread(QThread):
    """Worker thread that prepares and swaps one lazily selected module.

    Dependency checks, file preparation, imports, and model loading happen here
    so the active module remains usable if preparation is cancelled.

    Example:
        >>> thread = ModuleThread('ocr', registry)  # doctest: +SKIP
        >>> thread.requestCancelModuleInit()  # doctest: +SKIP
        >>> thread.cancel_event.is_set()  # doctest: +SKIP
        True
    """

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

    def _discard_module_instance(self, module):
        if module is None:
            return
        try:
            module.unload_model(empty_cache=True)
        except Exception as e:
            LOGGER.warning(f'Failed to unload {self.module_key} module after failed preparation: {e}')

    def _set_module(self, module_name: str):
        old_module = self.module
        new_module = None
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        try:
            same_module = old_module is not None and old_module.name == module_name
            if same_module and old_module.all_model_loaded():
                self.last_set_success = True
                return
            module: Union[TextDetectorBase, BaseTranslator, InpainterBase, OCRBase] = self._prepare_module_class(module_name)
            if same_module:
                new_module = old_module
                old_module = None
            else:
                params = cfg_module.get_params(self.module_key).get(module_name)
                self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
                if params is not None:
                    new_module = module(**params)
                else:
                    new_module = module()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
            new_module.load_model()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            if old_module is not None:
                old_module.unload_model(empty_cache=True)
                old_module = None
            self.module = new_module
            new_module = None
            self.last_set_success = True
        except DownloadCancelled as e:
            # Failed preparation should leave the user-selected module unresolved, not fall back.
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            self.last_error = e
            LOGGER.info(f'Cancelled preparing {self.module_key} module {module_name}.')
        except MissingDependency as e:
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.module = None
            self._discard_module_instance(new_module)
            self._discard_module_instance(old_module)
            missing = self._missing_requirements_from_exception(module_name, e)
            if missing:
                self.last_error = e
                self.last_missing_requirements = missing
            else:
                self.last_error = e
        finally:
            self.finish_set_module.emit()

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
        requirement = create_package_manager().requirement_for_import_name(import_name, dependencies)
        return [requirement] if requirement is not None else []

    def installMissingPackagesAndSetModule(
        self,
        module_name: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.job = lambda: self._install_missing_then_set(
            module_name,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        self.start()

    def _install_missing_then_set(
        self,
        module_name: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.last_set_module_name = module_name
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(
            requirements,
            progress_callback=self._emit_prepare_progress,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
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
        try:
            if self.job is not None:
                self.job()
        except LLMApiKeyRequiredError as e:
            _show_llm_key_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMModelRequiredError as e:
            _show_llm_model_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMBaseURLRequiredError as e:
            _show_llm_base_url_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMRequestStopped:
            LOGGER.info(f'{self.module_key} task stopped by user.')
        except Exception as e:
            create_error_dialog(e, self.tr('Module task failed.'), f'ModuleThreadFailed:{self.module_key}')
        finally:
            self.job = None


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
        except LLMApiKeyRequiredError as e:
            _show_llm_key_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
            self.inpainting = False
            self.inpaint_failed.emit()
        except LLMModelRequiredError as e:
            _show_llm_model_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
            self.inpainting = False
            self.inpaint_failed.emit()
        except LLMBaseURLRequiredError as e:
            _show_llm_base_url_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
            self.inpainting = False
            self.inpaint_failed.emit()
        except LLMRequestStopped:
            LOGGER.info('Inpainting stopped by user.')
            self.inpainting = False
            self.inpaint_failed.emit()
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

    progress_changed = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__('translator', TRANSLATORS, *args, **kwargs)
        self.translator: BaseTranslator = self.module

    def _set_translator(self, translator: str):
        
        old_translator = self.translator
        new_translator = None
        source, target = cfg_module.translate_source, cfg_module.translate_target
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        try:
            same_translator = old_translator is not None and old_translator.name == translator
            if same_translator and old_translator.all_model_loaded():
                self.last_set_success = True
                self.finish_set_module.emit()
                return
            params = cfg_module.translator_params.get(translator)
            translator_module: BaseTranslator = self._prepare_module_class(translator)
            if same_translator:
                new_translator = old_translator
                old_translator = None
            else:
                self._emit_prepare_progress({'event': 'instantiating', 'message': self.tr('Creating module')})
                if params is not None:
                    new_translator = translator_module(source, target, raise_unsupported_lang=False, **params)
                else:
                    new_translator = translator_module(source, target, raise_unsupported_lang=False)
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            self._emit_prepare_progress({'event': 'loading_model', 'message': self.tr('Loading model')})
            new_translator.load_model()
            if self.cancel_event.is_set():
                raise DownloadCancelled('Module preparation cancelled by user.')
            cfg_module.translate_source = new_translator.lang_source
            cfg_module.translate_target = new_translator.lang_target
            cfg_module.translator = new_translator.name
            if old_translator is not None:
                old_translator.unload_model(empty_cache=True)
                old_translator = None
            self.translator = new_translator
            new_translator = None
            self.last_set_success = True
        except DownloadCancelled as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
            self.last_error = e
            LOGGER.info(f'Cancelled preparing translator {translator}.')
        except MissingDependency as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
            self.last_error = e
            self.last_missing_requirements = e.requirements
        except Exception as e:
            self.translator = None
            self._discard_module_instance(new_translator)
            self._discard_module_instance(old_translator)
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

    def installMissingPackagesAndSetModule(
        self,
        translator: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.job = lambda: self._install_missing_then_set_translator(
            translator,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
        self.start()

    def _install_missing_then_set_translator(
        self,
        translator: str,
        requirements: List[str],
        torch_device: str = None,
        torch_cuda_version: str = None,
    ):
        self.last_set_module_name = translator
        self.last_set_success = False
        self.last_error = None
        self.last_missing_requirements = []
        self.cancel_event.clear()
        self._emit_prepare_progress({'event': 'installing_packages', 'message': self.tr('Installing packages')})
        result = create_package_manager().install(
            requirements,
            progress_callback=self._emit_prepare_progress,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )
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

    def _translate_page(
        self,
        project: ProjImgTrans,
        page_key: str,
    ):
        page = project.pages[page_key]
        # A failed or partially completed full-page request must never leave the
        # old page eligible as history.
        project.begin_full_page_translation(page_key)
        success = True
        if hasattr(self.translator, 'set_stop_event'):
            self.translator.set_stop_event(self.pipeline_stop_event)
        try:
            self.translator.translate_textblk_lst(
                page,
                project=project,
                page_key=page_key,
                full_page=True,
            )
        except LLMApiKeyRequiredError as e:
            success = False
            _show_llm_key_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMModelRequiredError as e:
            success = False
            _show_llm_model_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMBaseURLRequiredError as e:
            success = False
            _show_llm_base_url_required_dialog(e)
            if self.pipeline_stop_event is not None:
                self.pipeline_stop_event.set()
        except LLMRequestStopped:
            success = False
            LOGGER.info('Translation stopped by user.')
        except Exception as e:
            success = False
            _create_page_error_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
        if success:
            project.mark_translation_finished(page_key, self.translator.lang_target)
        return success

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
            try:
                self._translate_page(self.imgtrans_proj, page_key)
            except Exception as e:
                # TODO: allowing retry/skip/terminate
                msg = _failure_message_for_page(
                    self.tr('Translation Failed.'),
                    page_key,
                    self.tr('Page'),
                )
                if isinstance(e, MissingTranslatorParams):
                    msg = msg + '\n' + self.tr('{param} is required for {translator}').format(
                        param=str(e),
                        translator=self.translator.name,
                    )
                    
                self.blockSignals(False)
                create_error_dialog(e, msg, 'TranslationFailed')
                # self.imgtrans_proj = None
                # self.finished_counter = 0
                # self.pipeline_pagekey_queue = []
                # return
            self.blockSignals(False)
            self.finished_counter += 1
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
        self.pages_to_process = None
        register_global_callback('user_request_stop', self.isStopRequested)

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

    def runImgtransPipeline(
        self,
        imgtrans_proj: ProjImgTrans,
        pages_to_process=None,
    ):
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

    def _stop_on_stage_failure(
        self,
        exception: ModuleRunError,
        error_msg: str,
        exception_type: str,
        page_key: str = None,
    ):
        _create_page_error_dialog(
            exception,
            error_msg,
            exception_type,
            page_key,
            self.tr('Page'),
        )
        self.requestStop()

    def runBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        self.clearStopRequest()
        self.job = lambda : self._blktrans_pipeline(
            blk_list,
            mode,
            blk_ids,
            page_key=page_key,
        )
        self.start()

    def _translate_textblocks(
        self,
        blk_list: List[TextBlock],
        *,
        project: ProjImgTrans = None,
        page_key: str = None,
        full_page: bool = False,
    ) -> bool:
        translator = self.translate_thread.module or self.translate_thread.translator
        if translator is None:
            _create_page_error_dialog(
                RuntimeError('Translator is not loaded.'),
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
            return False
        if hasattr(translator, 'set_stop_event'):
            translator.set_stop_event(self.stop_event)
        try:
            translator.translate_textblk_lst(
                blk_list,
                project=project,
                page_key=page_key,
                full_page=full_page,
            )
            return True
        except LLMApiKeyRequiredError as e:
            _show_llm_key_required_dialog(e)
            self.requestStop()
        except LLMModelRequiredError as e:
            _show_llm_model_required_dialog(e)
            self.requestStop()
        except LLMBaseURLRequiredError as e:
            _show_llm_base_url_required_dialog(e)
            self.requestStop()
        except LLMRequestStopped:
            LOGGER.info('Translation stopped by user.')
        except Exception as e:
            _create_page_error_dialog(
                e,
                self.tr('Translation Failed.'),
                'TranslationFailed',
                page_key,
                self.tr('Page'),
            )
        return False

    def _translate_full_page(
        self,
        project: ProjImgTrans,
        page_key: str,
        blk_list: List[TextBlock],
    ) -> bool:
        project.begin_full_page_translation(page_key)
        success = self._translate_textblocks(
            blk_list,
            project=project,
            page_key=page_key,
            full_page=True,
        )
        if success:
            project.mark_translation_finished(
                page_key,
                self.translator.lang_target,
            )
        return success

    def _blktrans_pipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        tgt_img = self.imgtrans_proj.img_array if self.imgtrans_proj is not None else None
        tgt_mask = self.imgtrans_proj.mask_array if self.imgtrans_proj is not None else None
        if tgt_img is None:
            self.finish_blktrans.emit(mode, blk_ids)
            return
        if page_key is None:
            page_key = getattr(self.imgtrans_proj, 'current_img', None)
        if mode >= 0 and mode < 3:
            ocr_module = self.ocr_thread.module
            if hasattr(ocr_module, 'set_stop_event'):
                ocr_module.set_stop_event(self.stop_event)
            try:
                ocr_module.run_ocr(tgt_img, blk_list, split_textblk=True)
            except LLMApiKeyRequiredError as e:
                _show_llm_key_required_dialog(e)
                self.requestStop()
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except LLMModelRequiredError as e:
                _show_llm_model_required_dialog(e)
                self.requestStop()
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except LLMBaseURLRequiredError as e:
                _show_llm_base_url_required_dialog(e)
                self.requestStop()
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except LLMRequestStopped:
                LOGGER.info('OCR stopped by user.')
                self.finish_blktrans.emit(mode, blk_ids)
                return
            except ModuleRunError as e:
                _create_page_error_dialog(
                    e,
                    self.tr('OCR Failed.'),
                    'OCRFailed',
                    page_key,
                    self.tr('Page'),
                )
                self.finish_blktrans.emit(mode, blk_ids)
                return
            self.finish_blktrans.emit(mode, blk_ids)

        if mode != 0 and mode < 3:
            success = self._translate_textblocks(
                blk_list,
                project=self.imgtrans_proj,
                page_key=page_key,
            )
            # A selected run completes the page only when no source-bearing block
            # is still waiting for a translation.
            if success and page_key is not None:
                page = self.imgtrans_proj.pages.get(page_key)
                if page is not None and all(
                    not block.get_text().strip()
                    or bool(str(getattr(block, 'translation', '') or '').strip())
                    for block in page
                ):
                    self.imgtrans_proj.mark_translation_finished(
                        page_key,
                        self.translator.lang_target,
                    )
            self.finish_blktrans.emit(mode, blk_ids)
        if mode > 1:
            if hasattr(self.inpaint_thread.inpainter, 'set_stop_event'):
                self.inpaint_thread.inpainter.set_stop_event(self.stop_event)
            to_inpaint = self.imgtrans_proj.inpainted_array
            im_h, im_w = tgt_img.shape[:2]
            progress_prod = 100. / len(blk_list) if len(blk_list) > 0 else 0
            for ii, blk in enumerate(blk_list):
                xyxy_ori = np.array(blk.xyxy, dtype=np.int64)
                xyxy_ori[::2] = np.clip(xyxy_ori[::2], 0, im_w)
                xyxy_ori[1::2] = np.clip(xyxy_ori[1::2], 0, im_h)
                ox1, oy1, ox2, oy2 = xyxy_ori
                blk.region_inpaint_dict = None
                if oy2 - oy1 > 2 and ox2 - ox1 > 2:
                    xyxy = enlarge_window(xyxy_ori.tolist(), im_w, im_h)
                    xyxy = np.array(xyxy, dtype=np.int64)
                    x1, y1, x2, y2 = xyxy.astype(np.int64)
                    im = np.copy(to_inpaint[y1: y2, x1: x2])
                    maskseg_method = get_maskseg_method()
                    inpaint_mask_array, ballon_mask, bub_dict = maskseg_method(im, mask=tgt_mask[y1: y2, x1: x2])
                    mask = self.post_process_mask(inpaint_mask_array)
                    if mask.sum() > 0:
                        try:
                            inpainted = self.inpaint_thread.inpainter.inpaint(im, mask)
                        except LLMApiKeyRequiredError as e:
                            _show_llm_key_required_dialog(e)
                            self.requestStop()
                            self.finish_blktrans.emit(mode, blk_ids)
                            return
                        except LLMModelRequiredError as e:
                            _show_llm_model_required_dialog(e)
                            self.requestStop()
                            self.finish_blktrans.emit(mode, blk_ids)
                            return
                        except LLMBaseURLRequiredError as e:
                            _show_llm_base_url_required_dialog(e)
                            self.requestStop()
                            self.finish_blktrans.emit(mode, blk_ids)
                            return
                        except LLMRequestStopped:
                            LOGGER.info('Inpainting stopped by user.')
                            self.requestStop()
                            self.finish_blktrans.emit(mode, blk_ids)
                            return
                        rx1, ry1 = max(0, ox1 - x1), max(0, oy1 - y1)
                        rx2, ry2 = min(x2 - x1, ox2 - x1), min(y2 - y1, oy2 - y1)
                        if ry2 > ry1 and rx2 > rx1:
                            blk.region_inpaint_dict = {
                                'img': im[ry1: ry2, rx1: rx2],
                                'mask': mask[ry1: ry2, rx1: rx2],
                                'inpaint_rect': [int(ox1), int(oy1), int(ox2), int(oy2)],
                                'inpainted': inpainted[ry1: ry2, rx1: rx2]
                            }
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
                # Detection can replace or reorder blocks, so old translations
                # are never compatible even if detection later fails.
                self.imgtrans_proj.begin_detection(imgname)
                try:
                    mask, blk_list = self.textdetector.detect(img, self.imgtrans_proj)
                    need_save_mask = True
                except ModuleRunError as e:
                    self._stop_on_stage_failure(
                        e,
                        self.tr('Text Detection Failed.'),
                        'TextDetectFailed',
                        imgname,
                    )
                    break
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
                if hasattr(self.ocr, 'set_stop_event'):
                    self.ocr.set_stop_event(self.stop_event)
                try:
                    self.ocr.run_ocr(img, blk_list)
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
                except LLMApiKeyRequiredError as e:
                    _show_llm_key_required_dialog(e)
                    self.requestStop()
                    break
                except LLMModelRequiredError as e:
                    _show_llm_model_required_dialog(e)
                    self.requestStop()
                    break
                except LLMBaseURLRequiredError as e:
                    _show_llm_base_url_required_dialog(e)
                    self.requestStop()
                    break
                except LLMRequestStopped:
                    LOGGER.info('OCR stopped by user.')
                    self.requestStop()
                    break
                except ModuleRunError as e:
                    self._stop_on_stage_failure(
                        e,
                        self.tr('OCR Failed.'),
                        'OCRFailed',
                        imgname,
                    )
                    break
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                self.update_ocr_progress.emit(self.ocr_counter)

            if need_save_mask and mask is not None:
                self.imgtrans_proj.save_mask(imgname, mask)
                need_save_mask = False

            # Headless mode uses this same router; it has no translation-only branch.
            if cfg_module.enable_translate:
                if self.parallel_trans:
                    self.translate_thread.push_pagekey_queue(imgname)
                elif not low_vram_trans:
                    self._translate_full_page(
                        self.imgtrans_proj,
                        imgname,
                        blk_list,
                    )
                    self.translate_counter += 1
                    self.update_translate_progress.emit(self.translate_counter)

            if self.isStopRequested():
                LOGGER.info('Image translation pipeline stopped.')
                break
                        
            if cfg_module.enable_inpaint:
                if hasattr(self.inpainter, 'set_stop_event'):
                    self.inpainter.set_stop_event(self.stop_event)
                if mask is None:
                    mask = self.imgtrans_proj.load_mask_by_imgname(imgname)
                    
                if mask is not None or cfg_module.inpainter == LLM_INPAINT_KEY:
                    try:
                        inpainted = self.inpainter.inpaint(img, mask, blk_list)
                        self.imgtrans_proj.save_inpainted(imgname, inpainted)
                    except LLMApiKeyRequiredError as e:
                        _show_llm_key_required_dialog(e)
                        self.requestStop()
                        break
                    except LLMModelRequiredError as e:
                        _show_llm_model_required_dialog(e)
                        self.requestStop()
                        break
                    except LLMBaseURLRequiredError as e:
                        _show_llm_base_url_required_dialog(e)
                        self.requestStop()
                        break
                    except LLMRequestStopped:
                        LOGGER.info('Inpainting stopped by user.')
                        self.requestStop()
                        break
                    except Exception as e:
                        create_error_dialog(e, self.tr('Inpainting Failed.'), 'InpaintFailed')
                    
                self.inpaint_counter += 1
                self.imgtrans_proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
                self.update_inpaint_progress.emit(self.inpaint_counter)
            else:
                if len(blk_removed) > 0:
                    self.imgtrans_proj.load_mask_by_imgname
        
        if cfg_module.enable_translate and low_vram_trans and not self.isStopRequested():
            unload_modules(self, ['textdetector', 'inpainter', 'ocr'])
            for imgname in pages_to_iterate:
                # 检查是否请求停止
                if self.isStopRequested():
                    LOGGER.info('Translation stopped by user')
                    break
                    
                blk_list = self.imgtrans_proj.pages[imgname]
                self._translate_full_page(
                    self.imgtrans_proj,
                    imgname,
                    blk_list,
                )
                self.translate_counter += 1
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
        try:
            if self.job is not None:
                self.job()
        except LLMApiKeyRequiredError as e:
            _show_llm_key_required_dialog(e)
        except LLMModelRequiredError as e:
            _show_llm_model_required_dialog(e)
        except LLMBaseURLRequiredError as e:
            _show_llm_base_url_required_dialog(e)
        except LLMRequestStopped:
            LOGGER.info('Image translation task stopped by user.')
            self._emit_pipeline_stopped_if_ready(imgtrans_running=False)
        except Exception as e:
            create_error_dialog(e, self.tr('Image translation failed.'), 'ImageTranslationFailed')
        finally:
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

    canvas_inpaint_finished = Signal(dict)
    inpaint_th_finished = Signal()

    imgtrans_pipeline_finished = Signal()
    blktrans_pipeline_finished = Signal(int, list)
    page_trans_finished = Signal(int)
    module_selection_changed = Signal(str, str)

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
        self.config_panel: ConfigPanel = None
        self.parent_widget = None

    def setupThread(
        self,
        config_panel: ConfigPanel,
        imgtrans_progress_msgbox: ImgtransProgressMessageBox,
        parent_widget=None,
    ):
        self.config_panel = config_panel
        self.parent_widget = parent_widget
        self.textdetect_thread = TextDetectThread()

        self.ocr_thread = OCRThread()
        
        self.translate_thread = TranslateThread()
        self.translate_thread.progress_changed.connect(self.on_update_translate_progress)

        self.inpaint_thread = InpaintThread()
        self.inpaint_thread.finish_inpaint.connect(self.on_finish_inpaint)

        self.prepare_msgbox = ProgressMessageBox(self.tr('Preparing module: '), True, parent_widget)
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
        self.imgtrans_thread.imgtrans_proj = self.imgtrans_proj
        self.imgtrans_thread.update_detect_progress.connect(self.on_update_detect_progress)
        self.imgtrans_thread.update_ocr_progress.connect(self.on_update_ocr_progress)
        self.imgtrans_thread.update_translate_progress.connect(self.on_update_translate_progress)
        self.imgtrans_thread.update_inpaint_progress.connect(self.on_update_inpaint_progress)
        self.imgtrans_thread.finish_blktrans_stage.connect(self.on_finish_blktrans_stage)
        self.imgtrans_thread.finish_blktrans.connect(self.on_finish_blktrans)
        self.imgtrans_thread.pipeline_stopped.connect(self.on_imgtrans_thread_stopped)

        merge_config_module_params(
            cfg_module.translator_params, GET_VALID_TRANSLATORS(), TRANSLATORS.get)
        merge_config_module_params(
            cfg_module.inpainter_params, GET_VALID_INPAINTERS(), INPAINTERS.get)
        merge_config_module_params(
            cfg_module.textdetector_params, GET_VALID_TEXTDETECTORS(), TEXTDETECTORS.get)
        merge_config_module_params(
            cfg_module.ocr_params, GET_VALID_OCR(), OCR.get)

        config_panel.unload_models.connect(self.unload_all_models)
        config_panel.prepare_selected_modules.connect(self.prepareSelectedModules)
        config_panel.reinstall_torch.connect(self.reinstallTorch)


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
        return module is not None and module.name == module_name and module.all_model_loaded()

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
        return collect_missing_module_requirements(module_specs, create_package_manager())

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

    def _install_and_auto_install_checked(self, msgbox: QMessageBox, install_btn, auto_install_checker: QCheckBox) -> bool:
        should_install = msgbox.clickedButton() is install_btn
        if should_install and auto_install_checker.isChecked():
            pcfg.package_manager.auto_install_missing_packages = True
            if self.config_panel is not None and not self.config_panel.package_auto_install_checker.isChecked():
                self.config_panel.package_auto_install_checker.setChecked(True)
            save_config()
        return should_install

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

        if shared.HEADLESS:
            LOGGER.error(f'Selected modules require package(s):\n{missing_text}')
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
        command_preview = create_package_manager().preview_command(requirements)
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
        auto_install_checker = QCheckBox(self.tr("Install all missing packages and don't show again"))
        msgbox.setCheckBox(auto_install_checker)
        msgbox.exec()
        return self._install_and_auto_install_checked(msgbox, install_btn, auto_install_checker)

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
        accepted, torch_device, torch_cuda_version = self._confirm_torch_install_device(requirements)
        if not accepted:
            if on_failure is not None:
                on_failure()
            return
        self._pending_batch_package_queue = queue
        self._pending_batch_package_success = on_success
        self._pending_batch_package_failure = on_failure
        self._show_package_install_dialog(requirements)
        self.package_install_thread.installPackages(
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )

    def _confirm_torch_install_device(self, requirements: List[str]):
        if shared.HEADLESS:
            return True, None, None
        return confirm_torch_install_device(requirements, parent=self.parent_widget)

    def reinstallTorch(self) -> None:
        if self.package_install_thread is None or self.package_install_thread.isRunning():
            LOGGER.warning('Package installation is already running.')
            return
        accepted, torch_device, torch_cuda_version = confirm_torch_install_device(
            ['torch'], parent=self.config_panel,
        )
        if not accepted:
            return
        self.config_panel.setTorchInstalling()
        self._show_package_install_dialog(['torch'])
        if not self.package_install_thread.installPackages(
            ['torch'],
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        ):
            self.config_panel.refreshTorchStatus()

    def _show_package_install_dialog(self, requirements: List[str]):
        if shared.HEADLESS:
            LOGGER.info(f'Installing package(s): {", ".join(requirements)}')
            return
        self.prepare_msgbox.zero_progress()
        self.prepare_msgbox.setTaskName(self.tr('Installing packages: '))
        self.prepare_msgbox.updateTaskProgress(0, self._installing_packages_summary(requirements))
        if self.prepare_msgbox.stop_button is not None:
            self.prepare_msgbox.stop_button.setEnabled(False)
            self.prepare_msgbox.stop_button.setText(self.tr('Installing...'))
        self.prepare_msgbox.setFixedWidth(self.progress_msgbox.sizeHint().width())
        self.prepare_msgbox.show_fitted()

    def on_batch_package_install_finished(self):
        if not shared.HEADLESS:
            self.prepare_msgbox.done(0)
        if self.config_panel is not None:
            self.config_panel.refreshTorchStatus()

        queue = self._pending_batch_package_queue
        on_success = self._pending_batch_package_success
        on_failure = self._pending_batch_package_failure
        self._pending_batch_package_queue = []
        self._pending_batch_package_success = None
        self._pending_batch_package_failure = None

        if self.package_install_thread.last_success:
            if shared.HEADLESS:
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
        if shared.HEADLESS:
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

    @staticmethod
    def _installing_packages_summary(requirements: List[str]) -> str:
        """Return a compact install summary for the modal progress panel.

        >>> ModuleManager._installing_packages_summary(['torch', 'torchvision'])
        'torch...'
        >>> ModuleManager._installing_packages_summary(['einops'])
        'einops'
        """

        reqs = list(dict.fromkeys(requirements))
        if not reqs:
            return 'packages'
        try:
            first = Requirement(reqs[0]).name
        except Exception:
            first = str(reqs[0])
        return first + ('...' if len(reqs) > 1 else '')

    def _package_download_progress_message(self, payload: dict):
        """Format installer download progress for the preparation dialog.

        >>> manager = ModuleManager.__new__(ModuleManager)
        >>> manager._package_download_progress_message({
        ...     'message': 'Downloading torch.whl',
        ...     'downloaded': 50,
        ...     'total': 100,
        ...     'speed': 25,
        ...     'eta': 2,
        ... })
        ('Downloading torch.whl', '50.0% | 25.0 B/s | ETA 0:00:02')
        """

        message = payload.get('message') or self.tr('Downloading package')
        downloaded = payload.get('downloaded') or 0
        total = payload.get('total')
        status = []
        if total:
            percent = downloaded / total * 100
            status.append(f'{percent:.1f}%')

        speed = payload.get('speed')
        if speed:
            status.append(self.tr('{speed}/s').format(speed=sizeof_fmt(speed)))
        eta = payload.get('eta')
        if eta is not None:
            status.append(self.tr('ETA {eta}').format(eta=self._format_duration(eta)))
        return message, ' | '.join(status)

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(int(seconds), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours}:{minutes:02d}:{seconds:02d}'

    def on_module_prepare_progress(self, payload: dict):
        # Download callbacks report bytes; imports/model-loads report coarse stages.
        event = payload.get('event', '')
        module_name = payload.get('module', '')
        message = payload.get('message', '')
        verbose_message = None
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
            'package_download_progress': 30,
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
        elif event == 'package_download_progress':
            message, verbose_message = self._package_download_progress_message(payload)
            total = payload.get('total')
            downloaded = payload.get('downloaded') or 0
            if total:
                progress = max(0, min(100, int(downloaded / total * 100)))
        elif event in {'installing_packages', 'package_output'}:
            verbose_message = ''
        if not message:
            message = path or module_name

        if shared.HEADLESS:
            if event in {'checking_dependencies', 'file_start', 'archive_extract', 'installing_packages', 'package_output', 'importing', 'loading_model'}:
                LOGGER.info(f'{message}: {event}')
            return
        if self.prepare_msgbox is not None:
            self.prepare_msgbox.updateTaskProgress(progress, message, verbose_message)

    def on_module_prepare_finished(self, thread: ModuleThread):
        if self._preparing_thread is thread:
            if not shared.HEADLESS:
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
            if shared.HEADLESS:
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

        if shared.HEADLESS:
            LOGGER.error(
                f'Module {thread.last_set_module_name} requires package(s): '
                f'{", ".join(requirements)}'
            )
            return False

        should_install = self._ask_install_missing_packages(thread, requirements)
        if should_install:
            self._install_missing_packages_and_retry(thread, requirements, retry_key)
            return True
        return False

    def _ask_install_missing_packages(self, thread: ModuleThread, requirements: List[str]) -> bool:
        command_preview = create_package_manager().preview_command(requirements)
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
        auto_install_checker = QCheckBox(self.tr("Install all missing packages and don't show again"))
        msgbox.setCheckBox(auto_install_checker)
        msgbox.exec()
        return self._install_and_auto_install_checked(msgbox, install_btn, auto_install_checker)

    def _install_missing_packages_and_retry(self, thread: ModuleThread, requirements: List[str], retry_key: tuple):
        accepted, torch_device, torch_cuda_version = self._confirm_torch_install_device(requirements)
        if not accepted:
            return
        self._package_install_retried.add(retry_key)
        self._show_prepare_dialog(thread, thread.last_set_module_name)
        thread.installMissingPackagesAndSetModule(
            thread.last_set_module_name,
            requirements,
            torch_device=torch_device,
            torch_cuda_version=torch_cuda_version,
        )

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

    def _selected_modules(self) -> List[tuple]:
        return [
            ('textdetector', cfg_module.textdetector),
            ('ocr', cfg_module.ocr),
            ('translator', cfg_module.translator),
            ('inpainter', cfg_module.inpainter),
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

    def runImgtransPipeline(
        self,
        pages_to_process=None,
        render_only=False,
    ):
        _reset_llm_key_required_dialogs()
        if self.imgtrans_proj.is_empty:
            LOGGER.info('proj file is empty, nothing to do')
            self.progress_msgbox.hide()
            return
        self.last_finished_index = -1
        self.terminateRunningThread()
        
        if (render_only or cfg_module.all_stages_disabled()) and \
                self.imgtrans_proj is not None and self.imgtrans_proj.num_pages > 0:
            # Rendering reuses the normal page-finished path without preparing
            # or running any pipeline module.
            page_indexes = range(self.imgtrans_proj.num_pages)
            if not render_only and pages_to_process is not None:
                page_index_by_name = {
                    page_name: index
                    for index, page_name in enumerate(self.imgtrans_proj.pages)
                }
                page_indexes = (
                    page_index_by_name[page_name]
                    for page_name in pages_to_process
                    if page_name in page_index_by_name
                )
            for ii in page_indexes:
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
            lambda: self._startImgtransPipeline(
                pages_to_process,
            ),
            on_failure=lambda: self.imgtrans_pipeline_finished.emit() if shared.HEADLESS else None,
        )

    def _startImgtransPipeline(
        self,
        pages_to_process=None,
    ):
        if self.prepare_msgbox is not None and self.prepare_msgbox.isVisible():
            self.prepare_msgbox.done(0)
        self.progress_msgbox.detect_bar.setVisible(cfg_module.enable_detect)
        self.progress_msgbox.ocr_bar.setVisible(cfg_module.enable_ocr)
        self.progress_msgbox.translate_bar.setVisible(cfg_module.enable_translate)
        self.progress_msgbox.inpaint_bar.setVisible(cfg_module.enable_inpaint)
        self.progress_msgbox.zero_progress()
        self.progress_msgbox.show_fitted()
        self.imgtrans_thread.runImgtransPipeline(
            self.imgtrans_proj,
            pages_to_process,
        )
    
    def stopImgtransPipeline(self):
        """停止图像翻译流程"""
        LOGGER.info('Stopping image translation pipeline...')
        self.imgtrans_thread.requestStop()

    def runBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
        _reset_llm_key_required_dialogs()
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
            lambda: self._startBlktransPipeline(
                blk_list,
                mode,
                blk_ids,
                page_key=page_key,
            ),
        )

    def _startBlktransPipeline(
        self,
        blk_list: List[TextBlock],
        mode: int,
        blk_ids: List[int],
        *,
        page_key: str = None,
    ):
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
        self.imgtrans_thread.runBlktransPipeline(
            blk_list,
            mode,
            blk_ids,
            page_key=page_key,
        )

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

    def _on_update_stage_progress(self, stage: str, progress: int, update_progress: Callable[[int], None]):
        ri = self.imgtrans_thread.recent_finished_index(progress)
        if stage in shared.pbar:
            shared.pbar[stage].update(1)
        progress = int(progress / self.imgtrans_thread.num_pages * 100)
        update_progress(progress)
        if ri != self.last_finished_index:
            self.last_finished_index = ri
            self.page_trans_finished.emit(ri)
        if progress == 100:
            self.finishImgtransPipeline()

    def on_update_detect_progress(self, progress: int):
        self._on_update_stage_progress('detect', progress, self.progress_msgbox.updateDetectProgress)

    def on_update_ocr_progress(self, progress: int):
        self._on_update_stage_progress('ocr', progress, self.progress_msgbox.updateOCRProgress)

    def on_update_translate_progress(self, progress: int):
        self._on_update_stage_progress('translate', progress, self.progress_msgbox.updateTranslateProgress)

    def on_update_inpaint_progress(self, progress: int):
        self._on_update_stage_progress('inpaint', progress, self.progress_msgbox.updateInpaintProgress)

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

    def _select_module(self, module_key: str, module_name: str, valid_modules: List[str]):
        if module_name is None:
            module_name = getattr(cfg_module, module_key)
        if module_name not in valid_modules:
            LOGGER.warning(f'Ignoring invalid {module_key} module selection: {module_name}')
            return

        setattr(cfg_module, module_key, module_name)
        self.module_selection_changed.emit(module_key, module_name)

    def selectTranslator(self, translator: str = None):
        self._select_module('translator', translator, GET_VALID_TRANSLATORS())

    def selectInpainter(self, inpainter: str = None):
        self._select_module('inpainter', inpainter, GET_VALID_INPAINTERS())

    def selectTextDetector(self, textdetector: str = None):
        self._select_module('textdetector', textdetector, GET_VALID_TEXTDETECTORS())

    def selectOCR(self, ocr: str = None):
        self._select_module('ocr', ocr, GET_VALID_OCR())

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
    
    def moduleParams(self, module_key: str, module_name: str) -> dict:
        return cfg_module.get_params(module_key).get(module_name)

    def moduleRuntimeActionsEnabled(
        self,
        module_key: str,
        module_name: str,
    ) -> bool:
        active_module = self._active_module(module_key)
        return active_module is not None and active_module.name == module_name

    def _active_module(self, module_key: str):
        return {
            'translator': self.translator,
            'inpainter': self.inpainter,
            'textdetector': self.textdetector,
            'ocr': self.ocr,
        }[module_key]

    def onModuleParamEdited(
        self,
        module_key: str,
        module_name: str,
        param_key: str,
        param_content: dict,
    ) -> None:
        active_module = self._active_module(module_key)
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
