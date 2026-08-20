import os.path as osp
import os, re, traceback, sys
from typing import List, Optional, Tuple, Union
from pathlib import Path
import subprocess
from functools import partial
import time

from tqdm import tqdm
from qtpy.QtWidgets import QAction, QFileDialog, QMenu, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit, QDialog
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal, QTimer
from qtpy.QtGui import QContextMenuEvent, QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QPainter, QClipboard

from ballontranslator.utils.logger import logger as LOGGER
from ballontranslator.utils.text_processing import is_cjk
from ballontranslator.utils.textblock import TextBlock, TextAlignment
from ballontranslator.utils import shared
from ballontranslator.utils.message import create_error_dialog, create_info_dialog
from ballontranslator.modules import GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from .misc import parse_stylesheet, set_html_family, QKEY
from ballontranslator.utils.config import (
    FontFormat,
    ProgramConfig,
    RunStatus,
    load_textstyle_from,
    pcfg,
    save_config,
    save_text_styles,
    text_styles,
)
from ballontranslator.utils.proj_imgtrans import ProjImgTrans
from .canvas import Canvas
from .configpanel import ConfigPanel
from .module_manager import ModuleManager
from .text_engine.editing.widgets import SourceTextEdit, TransTextEdit
from .drawingpanel import DrawingPanel
from .text_engine.editing.manager import (
    PasteSrcItemsCommand,
    SceneTextManager,
    SceneTextReplacementReason,
    TextPanel,
)
from .mainwindowbars import TitleBar, LeftBar, BottomBar
from .menu_style import install_app_style_filters
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread
from .update_thread import UpdateCheckThread
from .update_dialog import UpdateReleaseDialog
from .run_pipeline_dialog import RunPipelineDialog
from .custom_widget import Widget, ViewWidget
from .global_search_widget import GlobalSearchWidget
from .text_engine.editing.commands import GlobalRepalceAllCommand
from .text_engine.transforms.grid import start_grid_numba_warmup
from .text_engine.pipeline_formatting import (
    AutoTateChuYokoThread,
    apply_auto_tate_chu_yoko,
)
from .framelesswindow import FramelessWindow, FramelessMoveResize
from .drawing_commands import RunBlkTransCommand
from .keywordsubwidget import KeywordSubWidget
from .module_parse_widgets import ModuleParamDialog
from . import shared_widget as SW
from .custom_widget import MessageBox, FrameLessMessageBox, ImgtransProgressMessageBox, ProgressMessageBox

class PageListView(QListWidget):

    reveal_file = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()

        return super().contextMenuEvent(e)

mainwindow_cls = Widget if shared.HEADLESS else FramelessWindow


class MainWindow(mainwindow_cls):

    imgtrans_proj: ProjImgTrans = ProjImgTrans()
    save_on_page_changed = True
    opening_dir = False
    page_changing = False

    translator = None

    restart_signal = Signal()
    create_errdialog = Signal(str, str, str)
    create_infodialog = Signal(dict)
    show_llm_key_dialog = Signal(str, str)
    show_llm_model_dialog = Signal(str, str, str)
    show_llm_base_url_dialog = Signal(str, str, str)
    
    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', **exec_args) -> None:
        super().__init__()

        self.app = app
        install_app_style_filters(self.app)
        self.resetStyleSheet()

        shared.create_errdialog_in_mainthread = self.create_errdialog.emit
        self.create_errdialog.connect(self.on_create_errdialog)
        shared.create_infodialog_in_mainthread = self.create_infodialog.emit
        self.create_infodialog.connect(self.on_create_infodialog)
        shared.show_llm_key_dialog_in_mainthread = self.show_llm_key_dialog.emit
        self.show_llm_key_dialog.connect(self.on_show_llm_key_dialog)
        shared.show_llm_model_dialog_in_mainthread = self.show_llm_model_dialog.emit
        self.show_llm_model_dialog.connect(self.on_show_llm_model_dialog)
        shared.show_llm_base_url_dialog_in_mainthread = self.show_llm_base_url_dialog.emit
        self.show_llm_base_url_dialog.connect(self.on_show_llm_base_url_dialog)
        shared.register_view_widget = self.register_view_widget

        self.backup_blkstyles = []
        self.module_param_dialog: Optional[ModuleParamDialog] = None
        self._run_imgtrans_wo_textstyle_update = False
        self._render_only = False
        self._render_global_format = None

        self.setupThread()
        self.setupUi()
        self.validateModuleSelections()
        self.setupConfig()
        self.setupShortcuts()
        self.setupRegisterWidget()
        if not shared.ON_WINDOWS:
            FramelessMoveResize.toggleMaxState(self)
        self.setAcceptDrops(True)

        if open_dir != '' and osp.exists(open_dir):
            self.OpenProj(open_dir)
        elif pcfg.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.OpenProj(proj_dir)

        if shared.HEADLESS:
            self.run_batch(**exec_args)

        if shared.ON_MACOS:
            # https://bugreports.qt.io/browse/QTBUG-133215
            self.hideSystemTitleBar()
            self.showMaximized()

        show_release_info = exec_args.get('show_release_info', False)
        if not shared.HEADLESS and (show_release_info or pcfg.check_update_on_startup):
            # Defer startup update checks until the event loop can paint progress.
            QTimer.singleShot(
                500,
                lambda: self.check_for_updates(
                    manual=False,
                    show_release_info=show_release_info,
                ),
            )
        # The callback cannot run until construction returns to the event loop.
        QTimer.singleShot(0, start_grid_numba_warmup)

    def setupThread(self):
        self.imsave_thread = ImgSaveThread()
        self.export_doc_thread = ExportDocThread()
        self.export_doc_thread.fin_io.connect(self.on_fin_export_doc)
        self.import_doc_thread = ImportDocThread(self)
        self.import_doc_thread.fin_io.connect(self.on_fin_import_doc)
        self.update_thread = UpdateCheckThread()
        self.update_thread.progress_changed.connect(self.on_update_progress_changed)
        self.update_thread.update_finished.connect(self.on_update_finished)
        self.update_thread.update_failed.connect(self.on_update_failed)
        self.update_progress_msgbox = ProgressMessageBox(self.tr('Updating: '), False, self)
        self._update_progress_visible = False
        self.auto_tate_chu_yoko_thread = AutoTateChuYokoThread(self)
        self.auto_tate_chu_yoko_progress = ProgressMessageBox(
            '',
            True,
            self,
        )
        self.auto_tate_chu_yoko_thread.progress_changed.connect(
            self.auto_tate_chu_yoko_progress.updateTaskProgress
        )
        self.auto_tate_chu_yoko_thread.processing_finished.connect(
            self.on_auto_tate_chu_yoko_processing_finished
        )
        self.auto_tate_chu_yoko_progress.stop_clicked.connect(
            self.auto_tate_chu_yoko_thread.request_stop
        )
        self.auto_tate_chu_yoko_progress.showed.connect(
            self.on_imgtrans_progressbox_showed
        )

    def resetStyleSheet(self):
        theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
        application_stylesheet = parse_stylesheet(theme)
        if self.styleSheet() != application_stylesheet:
            self.setStyleSheet(application_stylesheet)

    def setupUi(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        self.configPanel = ConfigPanel(self)
        self.configPanel.show_pre_MT_keyword_window.connect(
            self.show_pre_MT_keyword_window
        )
        self.configPanel.show_MT_keyword_window.connect(
            self.show_MT_keyword_window
        )
        self.configPanel.show_OCR_keyword_window.connect(
            self.show_OCR_keyword_window
        )

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.OpenProj)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.manual_save)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.import_doc.connect(self.on_import_doc)
        self.leftBar.export_src_txt.connect(lambda : self.on_export_txt(dump_target='source'))
        self.leftBar.export_trans_txt.connect(lambda : self.on_export_txt(dump_target='translation'))
        self.leftBar.export_src_md.connect(lambda : self.on_export_txt(dump_target='source', suffix='.md'))
        self.leftBar.export_trans_md.connect(lambda : self.on_export_txt(dump_target='translation', suffix='.md'))
        self.leftBar.import_trans_txt.connect(self.on_import_trans_txt)

        self.pageList = PageListView()
        self.pageList.setObjectName('PageListArea')
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)

        self.leftStackWidget = QStackedWidget(self)
        self.leftStackWidget.addWidget(self.pageList)

        self.global_search_widget = GlobalSearchWidget(self.leftStackWidget)
        self.global_search_widget.req_update_pagetext.connect(self.on_req_update_pagetext)
        self.global_search_widget.req_move_page.connect(self.on_req_move_page)
        self.imsave_thread.img_writed.connect(self.global_search_widget.on_img_writed)
        self.global_search_widget.search_tree.result_item_clicked.connect(self.on_search_result_item_clicked)
        self.leftStackWidget.addWidget(self.global_search_widget)
        
        self.centralStackWidget = QStackedWidget(self)
        
        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # set up canvas
        SW.canvas = self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)
        self.canvas.originallayer_trans_slider = self.bottomBar.originalSlider
        self.canvas.textlayer_trans_slider = self.bottomBar.textlayerSlider
        self.canvas.copy_src_signal.connect(self.on_copy_src)
        self.canvas.paste_src_signal.connect(self.on_paste_src)

        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.bottomBar.textlayerSlider.valueChanged.connect(self.canvas.setTextLayerTransparencyBySlider)
        
        self.drawingPanel = DrawingPanel(self.canvas)
        self.textPanel = TextPanel(self.app)
        self.textPanel.formatpanel.foldTextBtn.checkStateChanged.connect(self.fold_textarea)
        self.textPanel.formatpanel.sourceBtn.checkStateChanged.connect(self.show_source_text)
        self.textPanel.formatpanel.transBtn.checkStateChanged.connect(self.show_trans_text)
        self.textPanel.formatpanel.textstyle_panel.export_style.connect(self.export_tstyles)
        self.textPanel.formatpanel.textstyle_panel.import_style.connect(self.import_tstyles)

        self.ocrSubWidget = KeywordSubWidget(self.tr("Keyword substitution for source text"))
        self.ocrSubWidget.setParent(self)
        self.ocrSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.ocrSubWidget.hide()
        self.mtPreSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation source text"))
        self.mtPreSubWidget.setParent(self)
        self.mtPreSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtPreSubWidget.hide()
        self.mtSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation"))
        self.mtSubWidget.setParent(self)
        self.mtSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtSubWidget.hide()

        SW.st_manager = self.st_manager = SceneTextManager(self.app, self, self.canvas, self.textPanel)
        self.st_manager.new_textblk.connect(self.canvas.search_widget.on_new_textblk)
        self.canvas.search_widget.pairwidget_list = self.st_manager.pairwidget_list
        self.canvas.search_widget.textblk_item_list = self.st_manager.textblk_item_list
        self.canvas.search_widget.replace_one.connect(self.st_manager.on_page_replace_one)
        self.canvas.search_widget.replace_all.connect(self.st_manager.on_page_replace_all)

        # comic trans pannel
        self.rightComicTransStackPanel = QStackedWidget(self)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)
        self.rightComicTransStackPanel.currentChanged.connect(self.on_transpanel_changed)

        self.comicTransSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)

        self.centralStackWidget.addWidget(self.comicTransSplitter)

        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        margin = mainVBoxLayout.contentsMargins()
        self.main_margin = margin
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.comicTransSplitter.setStretchFactor(0, 1)
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.comicTransSplitter.setStretchFactor(2, 1)
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox(self)
    def on_finish_settranslator(self):
        module_manager = self.module_manager
        translator = module_manager.translator
        if translator is not None:
            name = translator.name
            pcfg.module.translator = name
            self.setTranslatorSelectionFromMetadata(name)
            LOGGER.info('Translator set to {}'.format(name))
        
    def on_show_module(self, idx, checked):
        visibility_attrs = (
            'show_textdetector_tool',
            'show_ocr_tool',
            'show_translator_tool',
            'show_inpainter_tool',
        )
        if not 0 <= idx < len(visibility_attrs):
            return
        setattr(pcfg, visibility_attrs[idx], checked)
        self._set_module_tool_visibility(idx, checked)
        save_config()

    def _set_module_tool_visibility(self, idx, visible):
        module_widgets = (
            self.bottomBar.textdet_selector,
            self.bottomBar.ocr_selector,
            self.bottomBar.trans_selector,
            self.bottomBar.inpaint_selector,
        )
        if 0 <= idx < len(module_widgets):
            module_widgets[idx].setVisible(visible)

    def setTranslatorSelectionFromMetadata(self, translator: str = None):
        metadata = self.module_manager.translator_metadata(translator)
        pcfg.module.translate_source = metadata['lang_source']
        pcfg.module.translate_target = metadata['lang_target']
        self.bottomBar.trans_selector.setTranslatorMetadata(
            metadata['name'],
            metadata['supported_src_list'],
            metadata['supported_tgt_list'],
            metadata['lang_source'],
            metadata['lang_target'],
        )
    def on_module_selection_changed(self, module_key: str, module_name: str):
        profile_id = ''
        if module_key == 'translator':
            self.setTranslatorSelectionFromMetadata(module_name)
            profile_id = pcfg.module.translator_llm_id
        elif module_key == 'textdetector':
            self.bottomBar.textdet_selector.setSelectedValue(module_name)
        elif module_key == 'ocr':
            self.bottomBar.ocr_selector.setSelectedValue(module_name)
            profile_id = pcfg.module.ocr_llm_id
        elif module_key == 'inpainter':
            self.bottomBar.inpaint_selector.setSelectedValue(module_name)
            self.drawingPanel.setInpainter(module_name)
            profile_id = pcfg.module.inpaint_llm_id
        if profile_id:
            self.configPanel.llm_profiles_panel.refreshSelectionBorders(profile_id)

    def validateModuleSelections(self):
        def valid_or_first(value, valid_values):
            if not valid_values:
                return value
            return value if value in valid_values else valid_values[0]

        pcfg.module.textdetector = valid_or_first(pcfg.module.textdetector, GET_VALID_TEXTDETECTORS())
        pcfg.module.ocr = valid_or_first(pcfg.module.ocr, GET_VALID_OCR())
        pcfg.module.inpainter = valid_or_first(pcfg.module.inpainter, GET_VALID_INPAINTERS())
        pcfg.module.translator = valid_or_first(pcfg.module.translator, GET_VALID_TRANSLATORS())

    def setupConfig(self):

        self.bottomBar.originalSlider.setValue(int(pcfg.original_transparency * 100))
        self.bottomBar.trans_selector.selector.addItems(GET_VALID_TRANSLATORS())
        self.bottomBar.ocr_selector.selector.addItems(GET_VALID_OCR())
        self.bottomBar.textdet_selector.selector.addItems(GET_VALID_TEXTDETECTORS())
        self.bottomBar.inpaint_selector.selector.addItems(GET_VALID_INPAINTERS())

        self.bottomBar.textdet_selector.setSelectedValue(pcfg.module.textdetector)
        self.bottomBar.ocr_selector.setSelectedValue(pcfg.module.ocr)
        self.bottomBar.inpaint_selector.setSelectedValue(pcfg.module.inpainter)
        self.drawingPanel.setInpainterOptions(
            GET_VALID_INPAINTERS(),
            pcfg.module.inpainter,
        )

        self.module_manager = module_manager = ModuleManager(self.imgtrans_proj)
        module_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        module_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        module_manager.setupThread(
            self.configPanel,
            self.imgtrans_progress_msgbox,
            parent_widget=self,
        )
        module_manager.module_selection_changed.connect(self.on_module_selection_changed)
        module_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        # Preparation and RUN dialogs share placement so the first RUN is stable.
        module_manager.prepare_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        module_manager.blktrans_pipeline_finished.connect(self.on_blktrans_finished)
        module_manager.imgtrans_thread.post_process_mask = self.drawingPanel.rectPanel.post_process_mask
        module_manager.translate_thread.finish_set_module.connect(self.on_finish_settranslator)
        self.setTranslatorSelectionFromMetadata(pcfg.module.translator)

        self.bottomBar.textdet_selector.selector.currentTextChanged.connect(self.on_textdet_changed)
        self.bottomBar.inpaint_selector.selector.currentTextChanged.connect(self.on_inpaint_changed)
        self.bottomBar.trans_selector.cfg_clicked.connect(self.to_trans_config)
        self.bottomBar.trans_selector.edit_clicked.connect(self.focus_llm_profile)
        self.bottomBar.trans_selector.selector.currentTextChanged.connect(self.on_trans_changed)
        self.bottomBar.trans_selector.llm_profile_changed.connect(self.on_llm_profile_changed)
        self.bottomBar.trans_selector.tgt_selector.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.bottomBar.trans_selector.src_selector.currentTextChanged.connect(self.on_trans_src_changed)
        self.bottomBar.textdet_selector.cfg_clicked.connect(self.to_detect_config)
        self.bottomBar.inpaint_selector.cfg_clicked.connect(self.to_inpaint_config)
        self.bottomBar.inpaint_selector.edit_clicked.connect(self.focus_llm_profile)
        self.bottomBar.inpaint_selector.llm_profile_changed.connect(self.on_inpaint_llm_profile_changed)
        self.bottomBar.ocr_selector.cfg_clicked.connect(self.to_ocr_config)
        self.bottomBar.ocr_selector.edit_clicked.connect(self.focus_llm_profile)
        self.bottomBar.ocr_selector.selector.currentTextChanged.connect(self.on_ocr_changed)
        self.bottomBar.ocr_selector.llm_profile_changed.connect(self.on_ocr_llm_profile_changed)
        self.drawingPanel.inpainter_changed.connect(module_manager.selectInpainter)
        self.drawingPanel.inpainter_config_requested.connect(self.to_inpaint_config)
        for idx, action in enumerate(self.titleBar.moduleVisibilityActions):
            self._set_module_tool_visibility(idx, action.isChecked())

        self.configPanel.llm_profiles_panel.profile_ui_updated.connect(self.on_llm_profile_ui_updated)
        self.configPanel.llm_profiles_panel.profile_summary_changed.connect(self.on_llm_profile_summary_changed)
        self.configPanel.llm_profiles_panel.set_translator_requested.connect(
            self.bottomBar.trans_selector.selectLLMProfile
        )
        self.configPanel.llm_profiles_panel.set_ocr_requested.connect(
            self.bottomBar.ocr_selector.selectLLMProfile
        )
        self.configPanel.llm_profiles_panel.set_inpainter_requested.connect(
            self.bottomBar.inpaint_selector.selectLLMProfile
        )

        self.drawingPanel.maskTransperancySlider.setValue(int(pcfg.mask_transparency * 100))
        self.leftBar.initRecentProjMenu(pcfg.recent_proj_list)
        self.leftBar.showPageListLabel.setChecked(pcfg.show_page_list)
        self.updatePageList()
        self.leftBar.save_config.connect(self.save_config)
        self.leftBar.imgTransChecker.setChecked(True)
        self.st_manager.formatpanel.global_format = pcfg.global_fontformat
        self.st_manager.formatpanel.set_active_format(pcfg.global_fontformat)
        
        self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)
        self.st_manager.formatpanel.foldTextBtn.setChecked(pcfg.fold_textarea)
        self.st_manager.formatpanel.transBtn.setCheckState(pcfg.show_trans_text)
        self.st_manager.formatpanel.sourceBtn.setCheckState(pcfg.show_source_text)
        self.fold_textarea(pcfg.fold_textarea)
        self.show_trans_text(pcfg.show_trans_text)
        self.show_source_text(pcfg.show_source_text)

        self.leftBar.run_imgtrans_clicked.connect(self.run_imgtrans)

        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)

        self.drawingPanel.set_config(pcfg.drawpanel)
        self.drawingPanel.initDLModule(module_manager)

        self.global_search_widget.imgtrans_proj = self.imgtrans_proj
        self.global_search_widget.setupReplaceThread(self.st_manager.pairwidget_list, self.st_manager.textblk_item_list)
        self.global_search_widget.replace_thread.finished.connect(self.on_global_replace_finished)

        self.configPanel.setupConfig()
        self.configPanel.save_config.connect(self.save_config)
        self.configPanel.check_update.connect(self.check_for_updates)
        self.configPanel.reload_textstyle.connect(self.load_textstyle_from_proj_dir)
        self.configPanel.font_list_changed.connect(self.on_show_only_custom_font)
        self.configPanel.compact_vertical_punctuation_changed.connect(
            self.st_manager.refresh_vertical_layouts
        )
        self.configPanel.apply_auto_tate_chu_yoko_requested.connect(
            self.apply_auto_tate_chu_yoko_to_project
        )
        if pcfg.let_show_only_custom_fonts_flag or pcfg.excluded_fonts:
            self.on_show_only_custom_font(pcfg.let_show_only_custom_fonts_flag)

        textblock_mode = pcfg.imgtrans_textblock
        if pcfg.imgtrans_textedit:
            if textblock_mode:
                self.bottomBar.textblockChecker.setChecked(True)
            self.bottomBar.texteditChecker.click()
        elif pcfg.imgtrans_paintmode:
            self.bottomBar.paintChecker.click()

        self.textPanel.formatpanel.textstyle_panel.initStyles(text_styles)

        self.canvas.search_widget.whole_word_toggle.setChecked(pcfg.fsearch_whole_word)
        self.canvas.search_widget.case_sensitive_toggle.setChecked(pcfg.fsearch_case)
        self.canvas.search_widget.regex_toggle.setChecked(pcfg.fsearch_regex)
        self.canvas.search_widget.range_combobox.setCurrentIndex(pcfg.fsearch_range)
        self.global_search_widget.whole_word_toggle.setChecked(pcfg.gsearch_whole_word)
        self.global_search_widget.case_sensitive_toggle.setChecked(pcfg.gsearch_case)
        self.global_search_widget.regex_toggle.setChecked(pcfg.gsearch_regex)
        self.global_search_widget.range_combobox.setCurrentIndex(pcfg.gsearch_range)

        if self.rightComicTransStackPanel.isHidden():
            self.setPaintMode()

        try:
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.ocr_sublist = []
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)

        try:
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.pre_mt_sublist = []
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)

        try:
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.mt_sublist = []
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)

    def setupImgTransUI(self):
        self.centralStackWidget.setCurrentIndex(0)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        self.centralStackWidget.setCurrentIndex(0)
        self.configPanel.showConfigDialog()

    def check_for_updates(self, manual: bool = True, show_release_info: bool = False):
        if self.update_thread.isBusy():
            LOGGER.info('Ignored update check request because an update check or update is already running.')
            return
        self._manual_update_check = manual
        self.configPanel.setUpdateChecking(True)
        self.configPanel.setLatestVersion(self.tr('Checking...'))
        self.update_thread.checkLatest(show_release_info=show_release_info)

    def apply_confirmed_update(self, release_info, current_version: str):
        if self.update_thread.isBusy():
            LOGGER.info('Ignored update apply request because an update check or update is already running.')
            return
        self.configPanel.setUpdateChecking(True)
        self.update_progress_msgbox.zero_progress()
        self.update_progress_msgbox.setTaskName(self.tr('Downloading update: '))
        self.update_progress_msgbox.updateTaskProgress(0, release_info.version)
        self.update_progress_msgbox.show_fitted()
        self._update_progress_visible = True
        self.update_thread.applyUpdate(release_info, current_version)

    def on_update_progress_changed(self, payload: dict):
        if not self._update_progress_visible:
            return
        progress = payload.get('progress', 0)
        message = payload.get('message', '')
        event = payload.get('event', '')
        task_names = {
            'backup_source': self.tr('Backing up current version: '),
            'download_start': self.tr('Downloading update: '),
            'download_progress': self.tr('Downloading update: '),
            'download_done': self.tr('Downloading update: '),
            'git_safety': self.tr('Saving local changes: '),
            'extract_source': self.tr('Installing update: '),
            'replace_source': self.tr('Installing update: '),
            'done': self.tr('Installing update: '),
        }
        self.update_progress_msgbox.setTaskName(task_names.get(event, self.tr('Updating: ')))
        self.update_progress_msgbox.updateTaskProgress(progress, message)

    def on_update_finished(self, result):
        if self._update_progress_visible:
            self.update_progress_msgbox.done(0)
            self._update_progress_visible = False
        self.configPanel.setUpdateChecking(False)
        if result.latest_version:
            self.configPanel.setLatestVersion(result.latest_version)

        if result.status in {'available', 'preview'}:
            allow_update = result.status == 'available'
            if self.confirm_update_release(result, allow_update=allow_update) and allow_update:
                QTimer.singleShot(
                    0,
                    lambda info=result.release_info, current=result.current_version: self.apply_confirmed_update(info, current),
                )
            return

        if result.status == 'up_to_date':
            if self._manual_update_check:
                create_info_dialog(
                    self.tr('Already up-to-date.') + f'\n{result.current_version}'
                )
            else:
                LOGGER.info(f'BallonsTranslator is already up-to-date: {result.current_version}')
            return

        if result.status == 'updated':
            # launch.restart() closes this window synchronously. closeEvent()
            # conditionally saves the project and waits for image IO before exec.
            self.update_thread.wait()
            self.restart_signal.emit()
            return

        LOGGER.warning(f'Ignored unexpected updater result status: {result.status}')

    def on_update_failed(self, error_msg: str, detail_traceback: str):
        if self._update_progress_visible:
            self.update_progress_msgbox.done(0)
            self._update_progress_visible = False
        self.configPanel.setUpdateChecking(False)
        self.on_create_errdialog(
            error_msg + '\n' + self.tr('Failed to check for updates.'),
            detail_traceback,
            '',
        )

    def confirm_update_release(self, result, allow_update: bool = True) -> bool:
        dialog = UpdateReleaseDialog(
            result,
            self,
            display_language=pcfg.display_lang,
            allow_update=allow_update,
        )
        accepted = getattr(getattr(QDialog, 'DialogCode', QDialog), 'Accepted')
        return dialog.exec() == accepted

    def set_display_lang(self, lang: str):
        self.retranslateUI()

    def OpenProj(self, proj_path: str):
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)
        
        if pcfg.let_textstyle_indep_flag and not shared.HEADLESS:
            self.load_textstyle_from_proj_dir(from_proj=True)

    def load_textstyle_from_proj_dir(self, from_proj=False):
        if from_proj:
            text_style_path = osp.join(self.imgtrans_proj.directory, 'textstyles.json')
        else:
            text_style_path = 'config/textstyles/default.json'
        if osp.exists(text_style_path):
            load_textstyle_from(text_style_path)
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
        else:
            pcfg.text_styles_path = text_style_path
            save_text_styles()

    def on_show_only_custom_font(self, only_custom: bool) -> None:
        if only_custom:
            font_list = shared.CUSTOM_FONTS
        else:
            font_list = shared.FONT_FAMILIES
        font_list = shared.get_filtered_font_list(font_list, pcfg.excluded_fonts)
        self.textPanel.formatpanel.familybox.update_font_list(font_list)

    def openDir(self, directory: str):
        try:
            self.opening_dir = True
            # 在加载项目前检查并生成TIF文件的预览图
            self.generate_tif_thumbnails(directory)
            # 重新加载项目，此时应该只加载预览图
            self.imgtrans_proj.load(directory)
            self.st_manager.clearSceneTextitems(
                SceneTextReplacementReason.PROJECT_RELOAD
            )
            self.canvas.clear_undostack(update_saved_step=True)
            self.titleBar.setTitleContent(osp.basename(directory))
            self.updatePageList()
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project ') + directory)
            return

    def generate_tif_thumbnails(self, directory: str):
        """
        为目录中的TIF文件生成预览图，并确保只加载预览图
        """
        try:
            from ballontranslator.utils.io_utils import create_thumbnail, find_tif_files
            # 查找目录中的所有TIF文件
            tif_files = find_tif_files(directory)
            
            # 为每个TIF文件生成预览图
            for tif_file in tif_files:
                tif_path = osp.join(directory, tif_file)
                # 检查是否已经存在对应的预览图
                base_path = Path(tif_path)
                thumb_path = base_path.parent / f"{base_path.stem}_thumb.jpg"
                
                # 如果预览图不存在，则生成预览图
                if not osp.exists(thumb_path):
                    create_thumbnail(tif_path, max_width=1000)
                    
        except Exception as e:
            LOGGER.error(f"Failed to generate TIF thumbnails: {e}")
        
    def dropOpenDir(self, directory: str):
        if isinstance(directory, str) and osp.exists(directory):
            self.leftBar.updateRecentProjList(directory)
            self.OpenProj(directory)

    def openJsonProj(self, json_path: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems(
                SceneTextReplacementReason.PROJECT_RELOAD
            )
            self.canvas.clear_undostack(update_saved_step=True)
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project from') + json_path)
        
    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.imgtrans_proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.imgtrans_proj.directory, imgname)), imgname)
        for imgname in self.imgtrans_proj.pages:
            lstitem =  item_func(imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.imgtrans_proj.current_img:
                self.pageList.setCurrentItem(lstitem)

    def pageLabelStateChanged(self):
        setup = self.leftBar.showPageListLabel.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            if self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.pageList)
        else:
            self.leftStackWidget.hide()
        pcfg.show_page_list = setup
        save_config()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Pending numeric edits are not dirty until they commit. Resolve them
        # before the close-time dirty check and final config snapshot.
        self.st_manager.formatpanel.resolve_text_transform_edits_for_save()
        if self.auto_tate_chu_yoko_thread.isRunning():
            self.auto_tate_chu_yoko_thread.request_stop()
            self.auto_tate_chu_yoko_thread.wait()
        if not self.imgtrans_proj.is_empty:
            self.conditional_save(keep_exist_as_backup=True)
        while True:
            if not self.imsave_thread.isRunning():
                break
            time.sleep(0.1)
        self.st_manager.hovering_transwidget = None
        self.st_manager.blockSignals(True)
        self.canvas.prepareClose()
        self.save_config()
        return super().closeEvent(event)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                if not shared.ON_MACOS:
                    self.titleBar.maxBtn.setChecked(True)
        elif event.type() == QEvent.Type.ActivationChange:
            self.canvas.on_activation_changed()

        super().changeEvent(event)
    
    def retranslateUI(self):
        # according to https://stackoverflow.com/questions/27635068/how-to-retranslate-dynamically-created-widgets
        # we got to do it manually ... I'd rather restart the program
        msg = QMessageBox()
        msg.setText(self.tr('Restart to apply changes? \n'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.restart_signal.emit()

    def save_config(self):
        save_config()

    def onHideCanvas(self):
        self.canvas.clearToolStates()

    def conditional_save(self, keep_exist_as_backup=False):
        if self.canvas.projstate_unsaved and not self.opening_dir:
            update_scene_text = save_proj = self.canvas.text_change_unsaved()
            save_rst_only = not self.canvas.draw_change_unsaved()
            if not save_rst_only:
                save_proj = True
            
            self.saveCurrentPage(update_scene_text, save_proj, restore_interface=True, save_rst_only=save_rst_only, keep_exist_as_backup=keep_exist_as_backup)

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if not self.opening_dir:
                # Typed transform edits belong to the old page and must commit
                # before its dirty check. Live drags are previews and cancel.
                self.st_manager.formatpanel.resolve_text_transform_edits_for_page_change()
            if self.save_on_page_changed:
                self.conditional_save()
            self.st_manager.clearSceneTextitems(
                SceneTextReplacementReason.PAGE_CHANGE
            )
            self.imgtrans_proj.set_current_img(item.text())
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.updateCanvas()
            self.st_manager.populateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
            self.module_manager.handle_page_changed()
            self.drawingPanel.handle_page_changed()
            
        self.page_changing = False

    def setupShortcuts(self):
        self.titleBar.nextpage_trigger.connect(self.shortcutNext) 
        self.titleBar.prevpage_trigger.connect(self.shortcutBefore)
        self.titleBar.textedit_trigger.connect(self.shortcutTextedit)
        self.titleBar.drawboard_trigger.connect(self.shortcutDrawboard)
        self.titleBar.redo_trigger.connect(self.on_redo)
        self.titleBar.undo_trigger.connect(self.on_undo)
        self.titleBar.page_search_trigger.connect(self.on_page_search)
        self.titleBar.global_search_trigger.connect(self.on_global_search)
        self.titleBar.replacePreMTkeyword_trigger.connect(self.show_pre_MT_keyword_window)
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.show_module.connect(self.on_show_module)
        self.titleBar.importtstyle_trigger.connect(self.import_tstyles)
        self.titleBar.exporttstyle_trigger.connect(self.export_tstyles)
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)
        self.titleBar.merge_tool_trigger.connect(self.on_open_merge_tool)
        self.titleBar.path_reorder_trigger.connect(self.on_path_reorder)
        self.canvas.path_reorder_mode_changed.connect(
            self.titleBar.path_reorder_action.setChecked
        )
        self.titleBar.font_exclusion_trigger.connect(
            self.configPanel.show_font_exclusion_dialog
        )

        shortcutA = QShortcut(QKeySequence("A"), self)
        shortcutA.activated.connect(self.shortcutBefore)
        shortcutPageUp = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToPreviousPage), self)
        shortcutPageUp.activated.connect(self.shortcutBefore)

        shortcutD = QShortcut(QKeySequence("D"), self)
        shortcutD.activated.connect(self.shortcutNext)
        shortcutPageDown = QShortcut(QKeySequence(QKeySequence.StandardKey.MoveToNextPage), self)
        shortcutPageDown.activated.connect(self.shortcutNext)

        shortcutTextblock = QShortcut(QKeySequence("W"), self)
        shortcutTextblock.activated.connect(self.shortcutTextblock)
        shortcutZoomIn = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        shortcutZoomIn.activated.connect(self.canvas.gv.scale_up_signal)
        shortcutZoomOut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        shortcutZoomOut.activated.connect(self.canvas.gv.scale_down_signal)
        shortcutCtrlD = QShortcut(QKeySequence("Ctrl+D"), self)
        shortcutCtrlD.activated.connect(self.shortcutCtrlD)
        shortcutSpace = QShortcut(QKeySequence("Space"), self)
        shortcutSpace.activated.connect(self.shortcutSpace)
        shortcutSelectAll = QShortcut(QKeySequence.StandardKey.SelectAll, self)
        shortcutSelectAll.activated.connect(self.shortcutSelectAll)

        shortcutEscape = QShortcut(QKeySequence("Escape"), self)
        shortcutEscape.activated.connect(self.shortcutEscape)

        shortcutBold = QShortcut(QKeySequence.StandardKey.Bold, self)
        shortcutBold.activated.connect(self.shortcutBold)
        shortcutItalic = QShortcut(QKeySequence.StandardKey.Italic, self)
        shortcutItalic.activated.connect(self.shortcutItalic)
        shortcutUnderline = QShortcut(QKeySequence.StandardKey.Underline, self)
        shortcutUnderline.activated.connect(self.shortcutUnderline)
        shortcutCapitalize = QShortcut(QKeySequence("Ctrl+Q"), self)
        shortcutCapitalize.activated.connect(
            self.st_manager.capitalize_selected_textitems
        )

        shortcutDelete = QShortcut(QKeySequence.StandardKey.Delete, self)
        shortcutDelete.activated.connect(self.shortcutDelete)

        drawpanel_shortcuts = {'hand': 'H', 'rect': 'R', 'inpaint': 'J', 'pen': 'B'}
        for tool_name, shortcut_key in drawpanel_shortcuts.items():
            shortcut = QShortcut(QKeySequence(shortcut_key), self)
            key = getattr(QKEY, f'Key_{shortcut_key}')
            shortcut.activated.connect(partial(
                self.drawingPanel.shortcutSetCurrentToolByName,
                tool_name,
                key,
            ))
            self.drawingPanel.setShortcutTip(tool_name, shortcut_key)

    def shortcutNext(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_D:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row + 1) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_A:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(-1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(-1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row - 1 + page_count) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutTextedit(self):
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.texteditChecker.click()

    def shortcutTextblock(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.bottomBar.texteditChecker.isChecked():
                self.bottomBar.textblockChecker.click()

    def shortcutDrawboard(self):
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.paintChecker.click()

    def shortcutCtrlD(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.delete_btn.click()
            elif self.canvas.textEditMode():
                self.canvas.delete_textblks.emit(0)

    def shortcutSelectAll(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.textPanel.isVisible():
                self.st_manager.set_blkitems_selection(True)

    def shortcutSpace(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.inpaint_btn.click()

    def shortcutBold(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.toggle_bold()

    def shortcutDelete(self):
        if self.canvas.gv.isVisible():
            self.canvas.delete_textblks.emit(1)

    def shortcutItalic(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.italicBtn.click()

    def shortcutUnderline(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.underlineBtn.click()

    def on_redo(self):
        self.st_manager.formatpanel.resolve_text_transform_edits_for_history_change()
        self.canvas.redo()

    def on_undo(self):
        self.st_manager.formatpanel.resolve_text_transform_edits_for_history_change()
        self.canvas.undo()

    def on_page_search(self) -> None:
        if self.canvas.gv.isVisible():
            self.canvas.cancel_path_reorder()
            fo = self.app.focusObject()
            sel_text = ''
            tgt_edit = None
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
                tgt_edit = self.st_manager.pairwidget_list[blkitem.idx].e_trans
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
                if isinstance(fo, SourceTextEdit):
                    tgt_edit = fo
            se = self.canvas.search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)

            if self.canvas.search_widget.isHidden():
                self.canvas.search_widget.show()
            self.canvas.search_widget.setCurrentEditor(tgt_edit)

    def on_global_search(self):
        if self.canvas.gv.isVisible():
            if not self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.click()
            fo = self.app.focusObject()
            sel_text = ''
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
            se = self.global_search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)
                
                self.global_search_widget.commit_search()

    def show_pre_MT_keyword_window(self):
        self.mtPreSubWidget.show()

    def show_MT_keyword_window(self):
        self.mtSubWidget.show()


    def show_OCR_keyword_window(self):
        self.ocrSubWidget.show()

    def on_open_merge_tool(self):
        """打开区域合并工具对话框"""
        if not hasattr(self, 'merge_dialog') or self.merge_dialog is None:
            from .merge_dialog import MergeDialog
            from qtpy.QtCore import QThread
            from qtpy.QtWidgets import QProgressDialog
            from ballontranslator.utils import merger
            
            self.merge_dialog = MergeDialog(self)
            self.merge_dialog.run_current_clicked.connect(lambda: self.run_merge_task(on_current=True))
            self.merge_dialog.run_all_clicked.connect(lambda: self.run_merge_task(on_current=False))
        
        if self.merge_dialog.isVisible():
            self.merge_dialog.raise_()
            self.merge_dialog.activateWindow()
        else:
            self.merge_dialog.show()

    def on_path_reorder(self, checked: bool) -> None:
        if not checked:
            self.canvas.cancel_path_reorder()
            return
        if (
            self.centralStackWidget.currentIndex() != 0
            or len(self.st_manager.textblk_item_list) < 2
        ):
            self.titleBar.path_reorder_action.setChecked(False)
            return

        if not self.bottomBar.texteditChecker.isChecked():
            self.bottomBar.texteditChecker.click()
        editing_item = self.canvas.editing_textblkitem
        if editing_item is not None and editing_item.isEditing():
            editing_item.endEdit(keep_focus=False)
        if not self.canvas.start_path_reorder():
            self.titleBar.path_reorder_action.setChecked(False)

    def run_merge_task(self, on_current=False):
        """执行区域合并任务"""
        from ballontranslator.utils import merger
        from qtpy.QtWidgets import QMessageBox
        
        if self.imgtrans_proj.is_empty:
            QMessageBox.warning(self, "警告", "请先打开一个项目")
            return
        
        config = self.merge_dialog.get_config()
        
        if on_current:
            # 对当前文件运行 - 直接在内存中操作，不读写文件
            from ballontranslator.utils.textblock import TextBlock
            
            current_img = self.imgtrans_proj.current_img
            if not current_img:
                QMessageBox.warning(self, "警告", "没有当前文件")
                return
            
            # 直接从内存获取当前页面的文本框
            if current_img not in self.imgtrans_proj.pages:
                QMessageBox.warning(self, "警告", "当前页面数据不存在")
                return
            
            textblocks = self.imgtrans_proj.pages[current_img]
            if not textblocks:
                QMessageBox.warning(self, "提示", "当前页面没有文本框")
                return
            
            # 将 TextBlock 对象转换为字典格式（merger 需要字典）
            initial_shapes = [blk.to_dict() for blk in textblocks]
            
            initial_count = len(initial_shapes)
            mode = config.get("MERGE_MODE", "NONE")
            total_merged = 0
            
            # 在内存中执行合并
            if mode == "VERTICAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "VERTICAL", config)
                total_merged += count
            elif mode == "HORIZONTAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                total_merged += count
            elif mode == "VERTICAL_THEN_HORIZONTAL":
                temp, count1 = merger.perform_merge(initial_shapes, "VERTICAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "HORIZONTAL", config)
                total_merged += (count1 + count2)
            elif mode == "HORIZONTAL_THEN_VERTICAL":
                temp, count1 = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "VERTICAL", config)
                total_merged += (count1 + count2)
            else:
                final_shapes = initial_shapes
            
            if total_merged > 0:
                # 将字典转回 TextBlock 对象并更新内存
                self.imgtrans_proj.pages[current_img] = [TextBlock(**blk_dict) for blk_dict in final_shapes]
                # 刷新画布
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
                final_count = len(final_shapes)
                QMessageBox.information(self, "成功", f"合并完成: 框数 {initial_count} -> {final_count} (减少了 {initial_count - final_count} 个)")
            else:
                # 提供更详细的提示
                labels = set(s.get('label', '') for s in initial_shapes)
                detail_msg = f"未发生任何合并。\n共有 {initial_count} 个文本框。\n标签类型: {', '.join(labels) or '无'}\n\n"
                detail_msg += "建议：\n"
                detail_msg += "1. 尝试增大最大间隙值（如 100-200）\n"
                detail_msg += "2. 降低最小重叠比例（如 50-70%）\n"
                detail_msg += "3. 取消勾选'启用排除合并的标签'\n"
                detail_msg += "4. 检查标签是否在黑名单中"
                QMessageBox.warning(self, "提示", detail_msg)
        else:
            # 对所有文件运行
            img_list = list(self.imgtrans_proj.pages.keys())
            if not img_list:
                QMessageBox.warning(self, "警告", "项目中没有图片")
                return
            
            # 使用项目的 JSON 文件路径
            json_path = self.imgtrans_proj.proj_path
            if not json_path or not osp.exists(json_path):
                QMessageBox.warning(self, "警告", f"找不到项目 JSON 文件: {json_path}")
                return
            
            # 使用后台线程执行合并
            self.run_merge_all_async(json_path, img_list, config)
    
    def run_merge_all_async(self, json_path, img_list, config):
        """异步执行所有文件的合并"""
        from .io_thread import MergeThread
        
        # 创建合并线程（如果不存在）
        if not hasattr(self, 'merge_thread'):
            self.merge_thread = MergeThread()
            self.merge_thread.progress_changed.connect(self.on_merge_progress)
            self.merge_thread.merge_finished.connect(self.on_merge_finished)
            self.merge_thread.progress_bar.stop_clicked.connect(self.on_merge_stop)
        
        # 启动合并
        if self.merge_thread.runMerge(json_path, img_list, config):
            # 显示进度对话框
            self.merge_thread.progress_bar.zero_progress()
            self.merge_thread.progress_bar.show()
    
    def on_merge_progress(self, current, total):
        """合并进度更新"""
        progress = int(current / total * 100)
        self.merge_thread.progress_bar.updateTaskProgress(progress, f' {current}/{total}')
    
    def on_merge_stop(self):
        """停止合并"""
        if hasattr(self, 'merge_thread'):
            self.merge_thread.requestStop()
            self.merge_thread.progress_bar.hide()
    
    def on_merge_finished(self, success_count, fail_count):
        """合并完成"""
        self.merge_thread.progress_bar.hide()
        
        # 重新加载整个项目
        try:
            json_path = self.imgtrans_proj.proj_path
            current_img = self.imgtrans_proj.current_img
            self.imgtrans_proj.load_from_json(json_path)
            if current_img and current_img in self.imgtrans_proj.pages:
                self.imgtrans_proj.set_current_img(current_img)
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
        except:
            pass
        
        # 显示结果
        total = success_count + fail_count
        QMessageBox.information(self, "完成", f"区域合并完成\n成功: {success_count}/{total}\n失败: {fail_count}/{total}")

    def on_req_update_pagetext(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()

    # 20260418 全部替换并重新渲染 会导致图片切换不保存图片的bug
    def on_req_move_page(self, page_name: str, force_save=False):
        ori_save = self.save_on_page_changed
        self.save_on_page_changed = False
        current_img = self.imgtrans_proj.current_img

        if current_img == page_name and not force_save:
            # 修复 Bug：提前返回时必须恢复自动保存的开关状态
            self.save_on_page_changed = ori_save 
            return

        # This path disables the normal page-change save callback. Commit the
        # old page's pending transform before its own dirty check, while
        # suppressing the search-result invalidation normally caused by a new
        # text undo command during an in-progress replace/rerender operation.
        page_changing = self.page_changing
        self.page_changing = True
        try:
            self.st_manager.formatpanel.resolve_text_transform_edits_for_save()
        finally:
            self.page_changing = page_changing

        if current_img not in self.global_search_widget.page_set:
            if self.canvas.projstate_unsaved: 
                self.saveCurrentPage()
        else:
            self.saveCurrentPage(save_rst_only=True)

        self.pageList.setCurrentRow(self.imgtrans_proj.pagename2idx(page_name))
        self.save_on_page_changed = ori_save
    # 20260418 全部替换并重新渲染 会导致图片切换不保存图片的bug end

    def on_search_result_item_clicked(self, pagename: str, blk_idx: int, is_src: bool, start: int, end: int):
        idx = self.imgtrans_proj.pagename2idx(pagename)
        self.pageList.setCurrentRow(idx)
        pw = self.st_manager.pairwidget_list[blk_idx]
        edit = pw.e_source if is_src else pw.e_trans
        edit.setFocus()
        edit.ensure_scene_visible.emit()
        cursor = QTextCursor(edit.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)

    def shortcutEscape(self):
        if self.canvas.path_reorder_active:
            self.canvas.cancel_path_reorder()
            return
        if self.canvas.handle_transform_modal_shortcut(QKEY.Key_Escape):
            return
        if self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        elif self.canvas.editing_textblkitem is not None and self.canvas.editing_textblkitem.isEditing():
            self.canvas.editing_textblkitem.endEdit()

    def setPaintMode(self):
        if self.bottomBar.paintChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.canvas.setPaintMode(True)
            self.bottomBar.originalSlider.show()
            self.bottomBar.textlayerSlider.show()
            self.bottomBar.textblockChecker.hide()
        else:
            self.canvas.setPaintMode(False)
            self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)

    def setTextEditMode(self):
        if self.bottomBar.texteditChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.bottomBar.textblockChecker.show()
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.st_manager.setTextEditMode(True)
            self.setTextBlockMode()
        else:
            self.bottomBar.textblockChecker.hide()
            self.rightComicTransStackPanel.setHidden(True)
            self.st_manager.setTextEditMode(False)
        self.canvas.setPaintMode(False)

    def setTextBlockMode(self):
        mode = self.bottomBar.textblockChecker.isChecked()
        self.canvas.setTextBlockMode(mode)
        pcfg.imgtrans_textblock = mode
        self.st_manager.showTextblkItemRect(mode)

    def manual_save(self):
        if self.leftBar.imgTransChecker.isChecked()\
            and self.imgtrans_proj.directory is not None:
            LOGGER.debug('Manually saving...')
            self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)

    def saveCurrentPage(self, update_scene_text=True, save_proj=True, restore_interface=False, save_rst_only=False, keep_exist_as_backup=False):
        
        if not self.imgtrans_proj.img_valid:
            return

        if update_scene_text or save_proj:
            # Resolve text-transform editor state before both the canonical
            # project snapshot and the result-image render consume it. The
            # render-only translation completion path saves its project first.
            self.st_manager.formatpanel.resolve_text_transform_edits_for_save()

        if restore_interface:
            set_canvas_focus = self.canvas.hasFocus()
            sel_textitem = self.canvas.selected_text_items()
            n_sel_textitems = len(sel_textitem)
            editing_textitem = None
            if n_sel_textitems == 1 and sel_textitem[0].isEditing():
                editing_textitem = sel_textitem[0]
        
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        
        if self.rightComicTransStackPanel.isHidden():
            self.bottomBar.texteditChecker.click()

        restore_textblock_mode = False
        if pcfg.imgtrans_textblock:
            restore_textblock_mode = True
            self.bottomBar.textblockChecker.click()

        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        if save_proj:
            try:
                self.imgtrans_proj.save(keep_exist_as_backup=keep_exist_as_backup)
                if not save_rst_only:
                    mask_path = self.imgtrans_proj.get_mask_path()
                    mask_array = self.imgtrans_proj.mask_array
                    if mask_array is not None:
                        self.imsave_thread.saveImg(mask_path, mask_array, save_params={'ext': pcfg.intermediate_imgsave_ext})
                    inpainted_path = self.imgtrans_proj.get_inpainted_path()
                    if self.canvas.drawingLayer.drawed():
                        inpainted = self.canvas.base_pixmap.copy()
                        painter = QPainter(inpainted)
                        painter.drawPixmap(0, 0, self.canvas.drawingLayer.get_drawed_pixmap())
                        painter.end()
                    else:
                        inpainted = self.imgtrans_proj.inpainted_array
                    if inpainted is not None:
                        self.imsave_thread.saveImg(inpainted_path, inpainted, save_params={'ext': pcfg.intermediate_imgsave_ext}, keep_alpha=self.imgtrans_proj.current_has_alpha())
            except Exception as e:
                LOGGER.error(f"Failed to save project files: {e}")

        # Render the final result image properly
        try:
            img = self.canvas.render_result_img()
            imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
            self.imsave_thread.saveImg(imsave_path, img, self.imgtrans_proj.current_img, save_params={'ext': pcfg.imgsave_ext, 'quality': pcfg.imgsave_quality}, keep_alpha=self.imgtrans_proj.current_has_alpha())
            self.canvas.setProjSaveState(False)
            self.canvas.update_saved_undostep()
        
        except Exception as e:
            LOGGER.error(f"Failed to render and save result image: {e}")

        if restore_interface:
            if restore_textblock_mode:
                self.bottomBar.textblockChecker.click()
            if hide_tsc:
                self.st_manager.txtblkShapeControl.show()
            if set_canvas_focus:
                self.canvas.setFocus()
            if n_sel_textitems > 0:
                self.canvas.block_selection_signal = True
                for blk in sel_textitem:
                    blk.setSelected(True)
                self.st_manager.on_incanvas_selection_changed()
                self.canvas.block_selection_signal = False
            if editing_textitem is not None:
                editing_textitem.startEdit()
        
    def to_trans_config(self):
        self.show_module_param_dialog('translator', pcfg.module.translator)

    def focus_llm_profile(self, profile_id: str = None, expand_details: bool = True, target: str = 'api_key'):
        self.configPanel.focusOnLLMProfile(
            profile_id or pcfg.module.translator_llm_id,
            expand_details=expand_details,
            target=target,
        )

    def to_inpaint_config(self):
        self.show_module_param_dialog('inpainter', pcfg.module.inpainter)

    def to_ocr_config(self):
        self.show_module_param_dialog('ocr', pcfg.module.ocr)

    def to_detect_config(self):
        self.show_module_param_dialog('textdetector', pcfg.module.textdetector)

    def show_module_param_dialog(
        self,
        module_type: str,
        module_name: str,
    ) -> None:
        current = self.module_param_dialog
        if current is not None and current.isVisible():
            if (
                current.module_type == module_type
                and current.module_key == module_name
            ):
                current.raise_()
                current.activateWindow()
                return
            current.close()

        parent = self.sender()
        if not isinstance(parent, RunPipelineDialog):
            parent = self
        dialog = ModuleParamDialog(
            module_type,
            module_name,
            self.module_manager.moduleParams(module_type, module_name),
            self.module_manager.moduleRuntimeActionsEnabled(
                module_type,
                module_name,
            ),
            parent,
        )
        dialog.paramwidget_edited.connect(
            self.module_manager.onModuleParamEdited
        )
        dialog.finished.connect(self._clear_module_param_dialog)
        dialog.destroyed.connect(
            self._clear_destroyed_module_param_dialog
        )
        if isinstance(parent, RunPipelineDialog):
            parent.finished.connect(dialog.close)
        self.module_param_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _clear_module_param_dialog(self, _result=None) -> None:
        if self.sender() is self.module_param_dialog:
            self.module_param_dialog = None

    def _clear_destroyed_module_param_dialog(self, _dialog=None) -> None:
        current = self.module_param_dialog
        if current is None:
            return
        try:
            current.objectName()
        except RuntimeError:
            self.module_param_dialog = None

    def on_run_module_selected(
        self,
        module_type: str,
        module_name: str,
    ) -> None:
        setter = {
            'textdetector': self.module_manager.selectTextDetector,
            'ocr': self.module_manager.selectOCR,
            'translator': self.module_manager.selectTranslator,
            'inpainter': self.module_manager.selectInpainter,
        }[module_type]
        setter(module_name)
        dialog = self.sender()
        if module_type == 'translator' and isinstance(
            dialog,
            RunPipelineDialog,
        ):
            dialog.setTranslatorMetadata(
                self.module_manager.translator_metadata(module_name)
            )

    def on_textdet_changed(self):
        module = self.bottomBar.textdet_selector.selector.currentText()
        self.module_manager.selectTextDetector(module)

    def on_ocr_changed(self):
        module = self.bottomBar.ocr_selector.selector.currentText()
        self.module_manager.selectOCR(module)
        self.bottomBar.ocr_selector.updateButtonText()

    def on_trans_changed(self):
        module = self.bottomBar.trans_selector.selector.currentText()
        self.module_manager.selectTranslator(module)
        self.bottomBar.trans_selector.updateButtonText()

    def on_llm_profile_changed(self, profile_id: str):
        if profile_id:
            pcfg.module.translator_llm_id = profile_id
            self.configPanel.llm_profiles_panel.syncProfile(profile_id)
            self.configPanel.llm_profiles_panel.setSelectedProfile('translator', profile_id)
        self.bottomBar.trans_selector.updateButtonText()

    def on_ocr_llm_profile_changed(self, profile_id: str):
        if profile_id:
            pcfg.module.ocr_llm_id = profile_id
            self.configPanel.llm_profiles_panel.syncProfile(profile_id)
            self.configPanel.llm_profiles_panel.setSelectedProfile('ocr', profile_id)
        self.bottomBar.ocr_selector.updateButtonText()

    def on_inpaint_llm_profile_changed(self, profile_id: str):
        if profile_id:
            pcfg.module.inpaint_llm_id = profile_id
            self.configPanel.llm_profiles_panel.syncProfile(profile_id)
            self.configPanel.llm_profiles_panel.setSelectedProfile('inpainter', profile_id)
        self.bottomBar.inpaint_selector.updateButtonText()

    def on_llm_profile_ui_updated(self):
        self.bottomBar.trans_selector.updateButtonText()
        self.bottomBar.ocr_selector.updateButtonText()
        self.bottomBar.inpaint_selector.updateButtonText()

    def on_llm_profile_summary_changed(self):
        self.bottomBar.trans_selector.updateButtonText()
        self.bottomBar.ocr_selector.updateButtonText()
        self.bottomBar.inpaint_selector.updateButtonText()

    def on_trans_src_changed(self, text: str = None):
        sender = self.sender()
        if text is None:
            text = sender.currentText()
        translator = self.module_manager.translator
        if translator is not None and translator.name == pcfg.module.translator:
            translator.set_source(text)
        pcfg.module.translate_source = text
        combobox = self.bottomBar.trans_selector.src_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)

    def on_trans_tgt_changed(self, text: str = None):
        sender = self.sender()
        if text is None:
            text = sender.currentText()
        translator = self.module_manager.translator
        if translator is not None and translator.name == pcfg.module.translator:
            translator.set_target(text)
        pcfg.module.translate_target = text
        combobox = self.bottomBar.trans_selector.tgt_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)

    def on_inpaint_changed(self):
        module = self.bottomBar.inpaint_selector.selector.currentText()
        self.module_manager.selectInpainter(module)
        self.bottomBar.inpaint_selector.updateButtonText()

    def translateBlkitemList(self, blkitem_list: List, mode: int) -> bool:

        tgt_img = self.imgtrans_proj.img_array
        if tgt_img is None:
            return False
        
        if len(blkitem_list) < 1:
            return False
        
        self.global_search_widget.set_document_edited()
        
        blk_list, blk_ids = [], []
        for blkitem in blkitem_list:
            blk: TextBlock = blkitem.blk
            blk.text = self.st_manager.pairwidget_list[
                blkitem.idx
            ].e_source.toPlainText()
            blk_ids.append(blkitem.idx)
            blk_list.append(blk)

        page_key = self.imgtrans_proj.current_img
        self.module_manager.runBlktransPipeline(
            blk_list,
            mode,
            blk_ids,
            page_key=page_key,
        )
        return True

    def on_imgtrans_pipeline_finished(self):
        self.backup_blkstyles.clear()
        self._run_imgtrans_wo_textstyle_update = False
        self._render_only = False
        self._render_global_format = None
        if pcfg.module.empty_runcache and not shared.HEADLESS:
            self.module_manager.unload_all_models()
        if shared.args.export_translation_txt:
            self.on_export_txt('translation')
        if shared.args.export_source_txt:
            self.on_export_txt('source')
        if shared.HEADLESS:
            self.run_next_dir()

    def postprocess_translations(self, blk_list: List[TextBlock]) -> None:
        if not is_cjk(pcfg.module.translate_target):
            for blk in blk_list:
                if blk.vertical:
                    blk.alignment = TextAlignment.Center
                blk.vertical = False

    def on_pagtrans_finished(self, page_index: int):
        blk_list = self.imgtrans_proj.get_blklist_byidx(page_index)
        ffmt_list = None
        if len(self.backup_blkstyles) == self.imgtrans_proj.num_pages and len(self.backup_blkstyles[page_index]) == len(blk_list):
            ffmt_list: List[FontFormat] = self.backup_blkstyles[page_index]

        self.postprocess_translations(blk_list)
                
        # override font format if necessary
        override_fnt_size = pcfg.let_fntsize_flag == 1
        override_fnt_stroke = pcfg.let_fntstroke_flag == 1
        override_fnt_color = pcfg.let_fntcolor_flag == 1
        override_fnt_scolor = pcfg.let_fnt_scolor_flag == 1
        override_alignment = pcfg.let_alignment_flag == 1
        override_effect = pcfg.let_fnteffect_flag == 1
        override_writing_mode = pcfg.let_writing_mode_flag == 1
        override_font_family = pcfg.let_family_flag == 1
        gf = self._render_global_format
        if gf is None:
            gf = self.textPanel.formatpanel.global_format

        enable_detect = pcfg.module.enable_detect and not self._render_only
        enable_ocr = pcfg.module.enable_ocr and not self._render_only
        enable_translate = pcfg.module.enable_translate and not self._render_only
        enable_inpaint = pcfg.module.enable_inpaint and not self._render_only
        inpaint_only = enable_inpaint and not (
            enable_detect or enable_ocr or enable_translate
        )
        
        if not inpaint_only:
            for ii, blk in enumerate(blk_list):
                if self._run_imgtrans_wo_textstyle_update and ffmt_list is not None:
                    blk.fontformat.merge(ffmt_list[ii])
                else:
                    if override_fnt_size or \
                        blk.font_size < 0:  # fall back to global font size if font size is not valid, it will be set to -1 for detected blocks
                        blk.font_size = gf.font_size
                    elif blk._detected_font_size > 0 and not enable_detect:
                        blk.font_size = blk._detected_font_size
                    if override_fnt_stroke:
                        blk.stroke_width = gf.stroke_width
                    elif enable_ocr:
                        blk.recalulate_stroke_width()
                    if override_fnt_color:
                        blk.set_font_colors(fg_colors=gf.frgb)
                    if override_fnt_scolor:
                        blk.set_font_colors(bg_colors=gf.srgb)
                    if override_writing_mode:
                        blk.vertical = gf.vertical
                    if override_alignment:
                        blk.alignment = gf.alignment
                    elif enable_detect:
                        if blk.vertical:
                            blk.alignment = TextAlignment.Center
                        elif not blk.src_is_vertical:
                            blk.recalulate_alignment()
                    if override_effect:
                        blk.opacity = gf.opacity
                        blk.shadow_color = gf.shadow_color
                        blk.shadow_radius = gf.shadow_radius
                        blk.shadow_strength = gf.shadow_strength
                        blk.shadow_offset = gf.shadow_offset
                    if override_font_family or blk.font_family is None:
                        blk.font_family = gf.font_family
                        if blk.rich_text:
                            blk.rich_text = set_html_family(blk.rich_text, gf.font_family)
                    
                    blk.line_spacing = gf.line_spacing
                    blk.letter_spacing = gf.letter_spacing
                    blk.italic = gf.italic
                    blk.font_weight = gf.font_weight
                    blk.underline = gf.underline
                    blk.fontformat.standard_vertical_roman_alignment = (
                        gf.standard_vertical_roman_alignment
                    )
                    sw = blk.stroke_width
                    if sw > 0 and enable_ocr and enable_detect and not override_fnt_size:
                        blk.font_size = blk.font_size / (1 + sw)

                    # Apply the complete global text-transform stack.
                    blk.fontformat.text_transform = gf.text_transform

            if pcfg.auto_tate_chu_yoko.enabled and (
                enable_translate
                or (
                    self._render_only
                    and not self._run_imgtrans_wo_textstyle_update
                )
            ):
                apply_auto_tate_chu_yoko(
                    blk_list,
                    pcfg.auto_tate_chu_yoko,
                )

            self.st_manager.auto_textlayout_flag = pcfg.let_autolayout_flag and \
                (enable_detect or enable_translate)
        
        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()

        if not enable_detect and enable_translate:
            for blkitem in self.st_manager.textblk_item_list:
                blkitem.squeezeBoundingRect()

        if page_index + 1 == self.imgtrans_proj.num_pages:
            self.st_manager.auto_textlayout_flag = False

        # save proj file on page trans finished
        self.imgtrans_proj.save()

        self.saveCurrentPage(False, False)

    def on_savestate_changed(self, unsaved: bool):
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def on_textstack_changed(self):
        if not self.page_changing:
            self.global_search_widget.set_document_edited()

    def on_run_blktrans(self, mode: int):
        blkitem_list = self.canvas.selected_text_items()
        self.translateBlkitemList(blkitem_list, mode)

    def on_blktrans_finished(self, mode: int, blk_ids: List[int]):

        if len(blk_ids) < 1:
            return
        
        blkitem_list = [self.st_manager.textblk_item_list[idx] for idx in blk_ids]

        pairw_list = []
        for blk in blkitem_list:
            pairw_list.append(self.st_manager.pairwidget_list[blk.idx])
        self.canvas.push_undo_command(RunBlkTransCommand(self.canvas, blkitem_list, pairw_list, mode))

    def apply_auto_tate_chu_yoko_to_project(self) -> None:
        if (
            self.imgtrans_proj.is_empty
            or self.auto_tate_chu_yoko_thread.isRunning()
        ):
            return

        # Capture live edits before the worker mutates the project documents.
        self.st_manager.updateTextBlkList()
        self.auto_tate_chu_yoko_progress.zero_progress()
        if self.auto_tate_chu_yoko_thread.start_processing(
            self.imgtrans_proj.pages,
            pcfg.auto_tate_chu_yoko,
        ):
            self.auto_tate_chu_yoko_progress.show_fitted()

    def on_auto_tate_chu_yoko_processing_finished(
        self,
        changed_count: int,
        changed_blocks: Tuple[TextBlock, ...],
    ) -> None:
        self.auto_tate_chu_yoko_progress.hide()
        if not changed_count:
            return

        changed_ids = {id(block) for block in changed_blocks}
        for block_item in self.st_manager.textblk_item_list:
            if id(block_item.blk) in changed_ids:
                block_item.load_rich_text_html(block_item.blk.rich_text)
        self.canvas.setProjSaveState(True)

    def on_imgtrans_progressbox_showed(self):
        # Handles both the preparation dialog and the RUN progress dialog.
        msgbox = self.sender()
        if msgbox is None or not hasattr(msgbox, 'size'):
            msgbox = self.module_manager.progress_msgbox
        if hasattr(msgbox, 'fit_to_content'):
            msgbox.fit_to_content()
        msg_size = msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        msgbox.move(p)

    def on_closebtn_clicked(self):
        if self.imsave_thread.isRunning():
            self.imsave_thread.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Saving image...'))
            self.imsave_thread.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

    def on_display_lang_changed(self, lang: str):
        if lang != pcfg.display_lang:
            pcfg.display_lang = lang
            self.set_display_lang(lang)
    
    def run_imgtrans(self):
        dialog = RunPipelineDialog(
            self,
            project=self.imgtrans_proj,
            translator_metadata=self.module_manager.translator_metadata(),
        )
        dialog.translate_source_changed.connect(self.on_trans_src_changed)
        dialog.translate_target_changed.connect(self.on_trans_tgt_changed)
        dialog.module_selected.connect(self.on_run_module_selected)
        dialog.module_config_requested.connect(self.show_module_param_dialog)
        self.module_manager.module_selection_changed.connect(
            dialog.setModuleSelection
        )
        try:
            result = dialog.exec_()
            if result == RunPipelineDialog.CONTINUE:
                self._run_imgtrans_wo_textstyle_update = False
                self.on_run_imgtrans(continue_mode=True)
                return
            if result == RunPipelineDialog.RENDER:
                self._run_imgtrans_wo_textstyle_update = (
                    dialog.render_without_text_style_update.isChecked()
                )
                self.on_run_imgtrans(render_only=True)
                return
            if result != RunPipelineDialog.RUN:
                return
            self._run_imgtrans_wo_textstyle_update = False
            self.on_run_imgtrans(pages_to_process=dialog.selected_pages())
        finally:
            dialog.deleteLater()

    def on_run_imgtrans(
        self,
        continue_mode=False,
        render_only=False,
        pages_to_process=None,
    ):
        self.backup_blkstyles.clear()
        self._render_only = render_only
        self._render_global_format = (
            self.textPanel.formatpanel.global_format.deepcopy()
            if render_only
            else None
        )

        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        enable_detect = pcfg.module.enable_detect and not render_only
        enable_ocr = pcfg.module.enable_ocr and not render_only
        enable_translate = pcfg.module.enable_translate and not render_only
        enable_inpaint = pcfg.module.enable_inpaint and not render_only
        all_disabled = not (
            enable_detect or enable_ocr or enable_translate or enable_inpaint
        )
        
        all_page_names = list(self.imgtrans_proj.pages)
        # Continue always scans the whole project; the dialog range applies only
        # to a fresh run.
        has_explicit_range = pages_to_process is not None and not continue_mode
        requested_pages = (
            all_page_names
            if not has_explicit_range
            else [
                page
                for page in pages_to_process
                if page in self.imgtrans_proj.pages
            ]
        )
        if has_explicit_range and not requested_pages and not render_only:
            return

        pipeline_pages = None
        if continue_mode:
            pipeline_pages = [
                page_name
                for page_name in requested_pages
                if not self.imgtrans_proj.get_page_progress(page_name)
            ]
            if not pipeline_pages:
                return
            requested_pages = pipeline_pages
        elif not render_only:
            progress_mask = (
                RunStatus.FIN_ALL
                if enable_detect
                else pcfg.module.finish_code
            )
            for page_name in requested_pages:
                self.imgtrans_proj.clear_page_progress(
                    page_name,
                    progress_mask,
                )
            if has_explicit_range:
                pipeline_pages = requested_pages

        if enable_detect:
            for page in requested_pages:
                if not pcfg.module.keep_exist_textlines and not continue_mode:
                    self.imgtrans_proj.pages[page].clear()
        else:
            self.st_manager.updateTextBlkList()
            for page_name in requested_pages:
                blklist = self.imgtrans_proj.pages[page_name]
                ffmt_list = []
                self.backup_blkstyles.append(ffmt_list)
                for textblk in blklist:
                    ffmt_list.append(textblk.fontformat.deepcopy())
                    if enable_ocr:
                        textblk.text = []
                        textblk.set_font_colors((0, 0, 0), (0, 0, 0))
                    if enable_translate or (all_disabled and not self._run_imgtrans_wo_textstyle_update) or enable_ocr:
                        textblk.rich_text = ''
                    textblk.vertical = textblk.src_is_vertical

        self.module_manager.runImgtransPipeline(
            pipeline_pages,
            render_only=render_only,
        )

    def on_transpanel_changed(self):
        self.canvas.editor_index = self.rightComicTransStackPanel.currentIndex()
        if not self.canvas.textEditMode() and self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        self.canvas.updateLayers()

    def import_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        p = QFileDialog.getOpenFileName(self, self.tr("Import Text Styles"), ddir, None, "(.json)")
        if not isinstance(p, str):
            p = p[0]
        if p == '':
            return
        try:
            load_textstyle_from(p, raise_exception=True)
            save_config()
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed to load from {p}'))

    def export_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        savep = QFileDialog.getSaveFileName(self, self.tr("Save Text Styles"), ddir, None, "(.json)")
        if not isinstance(savep, str):
            savep = savep[0]
        if savep == '':
            return
        suffix = Path(savep).suffix
        if suffix != '.json':
            if suffix == '':
                savep = savep + '.json'
            else:
                savep = savep.replace(suffix, '.json')
        oldp = pcfg.text_styles_path
        try:
            pcfg.text_styles_path = savep
            save_text_styles(raise_exception=True)
            save_config()
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed save to {savep}'))
            pcfg.text_styles_path = oldp

    def fold_textarea(self, fold: bool):
        pcfg.fold_textarea = fold
        self.textPanel.textEditList.setFoldTextarea(fold)

    def show_source_text(self, show: bool):
        pcfg.show_source_text = show
        self.textPanel.textEditList.setSourceVisible(show)

    def show_trans_text(self, show: bool):
        pcfg.show_trans_text = show
        self.textPanel.textEditList.setTransVisible(show)

    def on_export_doc(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()
        self.export_doc_thread.exportAsDoc(self.imgtrans_proj)

    def on_import_doc(self):
        self.import_doc_thread.importDoc(self.imgtrans_proj)

    def on_export_txt(self, dump_target, suffix='.txt'):
        try:
            self.imgtrans_proj.dump_txt(dump_target=dump_target, suffix=suffix)
            create_info_dialog(self.tr('Text file exported to ') + self.imgtrans_proj.dump_txt_path(dump_target, suffix))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export as TEXT file'))

    def on_import_trans_txt(self):
        try:
            selected_file = ''
            dialog = QFileDialog()
            selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.md/*.txt'), filter="*.txt *.md *.TXT *.MD")[0].toLocalFile())
            if not osp.exists(selected_file):
                return

            all_matched, match_rst = self.imgtrans_proj.load_translation_from_txt(
                selected_file,
                target_language=pcfg.module.translate_target,
            )
            matched_pages = match_rst['matched_pages']

            if self.imgtrans_proj.current_img in matched_pages:
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()

            if all_matched:
                msg = self.tr('Translation imported and matched successfully.')
            else:
                msg = self.tr('Imported txt file not fully matched with current project, please make sure source txt file structured like results from \"export TXT/markdown\"')
                if len(match_rst['missing_pages']) > 0:
                    msg += '\n' + self.tr('Missing pages: ') + '\n'
                    msg += '\n'.join(match_rst['missing_pages'])
                if len(match_rst['unexpected_pages']) > 0:
                    msg += '\n' + self.tr('Unexpected pages: ') + '\n'
                    msg += '\n'.join(match_rst['unexpected_pages'])
                if len(match_rst['unmatched_pages']) > 0:
                    msg += '\n' + self.tr('Unmatched pages: ') + '\n'
                    msg += '\n'.join(match_rst['unmatched_pages'])
                msg = msg.strip()

            for pagename in matched_pages:
                for blk in self.imgtrans_proj.pages[pagename]:
                    blk.translation = self.mtSubWidget.sub_text(blk.translation)
            
            create_info_dialog(msg)

        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation from ') + selected_file)

    def on_reveal_file(self):
        current_img_path = self.imgtrans_proj.current_img_path()
        if sys.platform == 'win32':
            # qprocess seems to fuck up with "\""
            p = "\""+str(Path(current_img_path))+"\""
            subprocess.Popen("explorer.exe /select,"+p, shell=True)
        elif sys.platform == 'darwin':
            p = "\""+current_img_path+"\""
            subprocess.Popen("open -R "+p, shell=True)

    def on_set_gsearch_widget(self):
        setup = self.leftBar.globalSearchChecker.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            self.leftBar.showPageListLabel.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    def on_fin_export_doc(self):
        msg = QMessageBox()
        msg.setText(self.tr('Export to ') + self.imgtrans_proj.doc_path())
        msg.exec_()

    def on_fin_import_doc(self):
        self.st_manager.updateSceneTextitems()

    def on_global_replace_finished(self):
        rt = self.global_search_widget.replace_thread
        self.canvas.push_text_command(
            GlobalRepalceAllCommand(rt.sceneitem_list, rt.background_list, rt.target_text, self.imgtrans_proj)
        )
        rt.sceneitem_list = None
        rt.background_list = None

    def on_darkmode_triggered(self):
        pcfg.darkmode = self.titleBar.darkModeAction.isChecked()
        self.resetStyleSheet()
        self.save_config()

    def on_copy_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return

        try:
            if self.module_manager.translator is not None and hasattr(self.module_manager.translator, 'build_copy_prompt'):
                src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText() for blk in blks]
                src_txt = self.module_manager.translator.build_copy_prompt(src_list)
            else:
                src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText().strip().replace('\n', ' ') for blk in blks]
                src_txt = '\n'.join(src_list)
        except Exception as e:
            create_error_dialog(
                e,
                self.tr('Failed to copy source text'),
            )
            return

        self.st_manager.app_clipborad.setText(src_txt, QClipboard.Mode.Clipboard)

    def on_paste_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return

        src_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_source for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        
        n_paragraph = min(len(src_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        
        src_widget_list = src_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]

        self.canvas.push_undo_command(PasteSrcItemsCommand(src_widget_list, text_list))
    
    def run_batch(self, exec_dirs: Union[List, str], **kwargs):
        if not isinstance(exec_dirs, List):
            exec_dirs = exec_dirs.split(',')
        valid_dirs = []
        for d in exec_dirs:
            if osp.exists(d):
                valid_dirs.append(d)
            else:
                LOGGER.warning(f'target directory {d} does not exist.')
        self.exec_dirs = valid_dirs
        self.run_next_dir()

    def run_next_dir(self):
        if len(self.exec_dirs) == 0:
            while self.imsave_thread.isRunning():
                time.sleep(0.1)
            LOGGER.info(f'finished translating all dirs, please enter next dirs to translate (separated by comma). enter "exit" to quit app.')
            new_exec_dirs = input()
            if new_exec_dirs.strip().lower() == 'exit':
                LOGGER.info(f'exiting app...')
                self.app.quit()
                return
            self.run_batch(new_exec_dirs)
            return
        d = self.exec_dirs.pop(0)
        
        LOGGER.info(f'translating {d} ...')
        self.openDir(d)
        shared.pbar = {}
        npages = len(self.imgtrans_proj.pages)
        if npages > 0:
            if pcfg.module.enable_detect:
                shared.pbar['detect'] = tqdm(range(npages), desc="Text Detection")
            if pcfg.module.enable_ocr:
                shared.pbar['ocr'] = tqdm(range(npages), desc="OCR")
            if pcfg.module.enable_translate:
                shared.pbar['translate'] = tqdm(range(npages), desc="Translation")
            if pcfg.module.enable_inpaint:
                shared.pbar['inpaint'] = tqdm(range(npages), desc="Inpaint")
        self.on_run_imgtrans()

    def on_create_errdialog(self, error_msg: str, detail_traceback: str = '', exception_type: str = ''):
        try:
            if exception_type != '':
                shared.showed_exception.add(exception_type)
            err = QMessageBox()
            err.setText(error_msg)
            err.setDetailedText(detail_traceback)
            err.exec()
            if exception_type != '':
                shared.showed_exception.remove(exception_type)
        except:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict):
        QMessageBox.StandardButton.NoButton
        dialog = MessageBox(**info_dict)
        dialog.show()   # exec_ will block main thread

    def on_show_llm_key_dialog(self, profile_id: str, profile_name: str):
        dialog_key = profile_id or profile_name
        # QMessageBox.exec() runs a nested event loop, so queued RUN signals can re-enter here.
        exception_type = f'LLMApiKeyRequired:{dialog_key}'
        if exception_type in shared.showed_exception:
            return
        shared.showed_exception.add(exception_type)

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(self.tr('API key required'))
        msg.setText(self.tr('The selected LLM profile requires an API key.'))
        msg.setInformativeText(
            self.tr('Fill the API key before running this LLM task for: {profile_name}').format(profile_name=profile_name)
        )
        fill_btn = msg.addButton(self.tr('Fill API Key'), QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        try:
            msg.exec()
            if msg.clickedButton() == fill_btn:
                self.focus_llm_profile(profile_id, expand_details=False)
        finally:
            shared.showed_exception.discard(exception_type)

    def on_show_llm_model_dialog(self, profile_id: str, profile_name: str, target: str):
        if isinstance(target, bool):
            target = 'vision_model' if target else 'model'
        target = target if target in {'model', 'vision_model', 'image_model'} else 'model'
        dialog_key = profile_id or profile_name
        # QMessageBox.exec() runs a nested event loop, so queued RUN signals can re-enter here.
        exception_type = f'LLMModelRequired:{dialog_key}:{target}'
        if exception_type in shared.showed_exception:
            return
        shared.showed_exception.add(exception_type)

        title_by_target = {
            'model': self.tr('Model required'),
            'vision_model': self.tr('Vision model required'),
            'image_model': self.tr('Image model required'),
        }
        field_by_target = {
            'model': self.tr('model'),
            'vision_model': self.tr('vision model'),
            'image_model': self.tr('image model'),
        }
        title = title_by_target[target]
        field_name = field_by_target[target]
        display_profile_name = profile_name or profile_id or self.tr('LLM Profile')
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(
            self.tr('The selected LLM profile requires a {field_name}.').format(field_name=field_name)
        )
        msg.setInformativeText(
            self.tr('Fill the {field_name} before running this LLM task for: {profile_name}').format(
                field_name=field_name,
                profile_name=display_profile_name,
            )
        )
        fill_btn = msg.addButton(self.tr('Fill Model'), QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        try:
            msg.exec()
            if msg.clickedButton() == fill_btn:
                if target == 'vision_model':
                    target_profile_id = profile_id or pcfg.module.ocr_llm_id
                elif target == 'image_model':
                    target_profile_id = profile_id or pcfg.module.inpaint_llm_id
                else:
                    target_profile_id = profile_id or pcfg.module.translator_llm_id
                self.focus_llm_profile(
                    target_profile_id,
                    expand_details=False,
                    target=target,
                )
        finally:
            shared.showed_exception.discard(exception_type)

    def on_show_llm_base_url_dialog(self, profile_id: str, profile_name: str, target: str):
        target = target if target in {'base_url', 'image_base_url'} else 'base_url'
        dialog_key = profile_id or profile_name
        exception_type = f'LLMBaseURLRequired:{dialog_key}:{target}'
        if exception_type in shared.showed_exception:
            return
        shared.showed_exception.add(exception_type)

        title_by_target = {
            'base_url': self.tr('Base URL required'),
            'image_base_url': self.tr('Image base URL required'),
        }
        field_by_target = {
            'base_url': self.tr('base URL'),
            'image_base_url': self.tr('image base URL'),
        }
        title = title_by_target[target]
        field_name = field_by_target[target]
        display_profile_name = profile_name or profile_id or self.tr('LLM Profile')
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(
            self.tr('The selected LLM profile requires this field: {field_name}.').format(field_name=field_name)
        )
        msg.setInformativeText(
            self.tr('Fill the {field_name} before running this LLM task for: {profile_name}').format(
                field_name=field_name,
                profile_name=display_profile_name,
            )
        )
        fill_btn = msg.addButton(self.tr('Fill URL'), QMessageBox.AcceptRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        try:
            msg.exec()
            if msg.clickedButton() == fill_btn:
                target_profile_id = profile_id or pcfg.module.inpaint_llm_id
                self.focus_llm_profile(
                    target_profile_id,
                    expand_details=True,
                    target=target,
                )
        finally:
            shared.showed_exception.discard(exception_type)

    def setupRegisterWidget(self):
        self.titleBar.viewMenu.addSeparator()
        for cfg_name in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget[cfg_name]
            widget: ViewWidget = d['widget']
            action = QAction(widget.action_name, self.titleBar)
            action.setCheckable(True)
            visible = getattr(pcfg, cfg_name)
            action.setChecked(visible)
            action.triggered.connect(self.action_set_view_visible)
            self.titleBar.viewMenu.addAction(action)
            d['action'] = action
            shared.action_to_view_config_name[action] = cfg_name
            widget.set_expend_area(expend=getattr(pcfg, widget.config_expand_name), set_config=False)
            widget.view_hide_btn_clicked.connect(self.on_hide_view_widget)
            widget.setVisible(visible)

    def register_view_widget(self, widget: ViewWidget):
        assert widget.config_name not in shared.config_name_to_view_widget
        d = {'widget': widget}
        shared.config_name_to_view_widget[widget.config_name] = d

    def action_set_view_visible(self):
        action: QAction = self.sender()
        show = action.isChecked()
        cfg_name = shared.action_to_view_config_name[action]
        widget: ViewWidget = shared.config_name_to_view_widget[cfg_name]['widget']
        widget.setVisible(show)
        setattr(pcfg, cfg_name, show)

    def on_hide_view_widget(self, cfg_name: str):
        d = shared.config_name_to_view_widget[cfg_name]
        widget: ViewWidget = d['widget']
        widget.setVisible(False)
        action: QAction = d['action']
        action.setChecked(False)
        setattr(pcfg, cfg_name, False)
