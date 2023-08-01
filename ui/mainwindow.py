import os.path as osp
import os, re, traceback
from typing import List

from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal
from qtpy.QtGui import QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QImage, QPainter

from utils.logger import logger as LOGGER
from utils.io_utils import json_dump_nested_obj
from utils.text_processing import is_cjk, full_len, half_len
from modules.textdetector.textblock import TextBlock

from .misc import pt2px, parse_stylesheet, ProgramConfig
from .imgtrans_proj import ProjImgTrans
from .canvas import Canvas
from .configpanel import ConfigPanel
from .module_manager import ModuleManager
from .pagesources import SourceDownload
from .textedit_area import TextPanel, SourceTextEdit, SelectTextMiniMenu
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager
from .mainwindowbars import TitleBar, LeftBar, BottomBar
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread
from .stylewidgets import FrameLessMessageBox, ImgtransProgressMessageBox, SourceDownloadProgressMessageBox
from .preset_widget import PresetPanel
from .constants import CONFIG_PATH
from .global_search_widget import GlobalSearchWidget
from . import constants as C
from .textedit_commands import GlobalRepalceAllCommand
from .framelesswindow import FramelessWindow
from .drawing_commands import RunBlkTransCommand
from .keywordsubwidget import KeywordSubWidget

class PageListView(QListWidget):    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMaximumWidth(512)
        self.setIconSize(QSize(C.PAGELIST_THUMBNAIL_SIZE, C.PAGELIST_THUMBNAIL_SIZE))


class MainWindow(FramelessWindow):

    imgtrans_proj: ProjImgTrans = ProjImgTrans()
    save_on_page_changed = True
    opening_dir = False
    page_changing = False
    postprocess_mt_toggle = True

    translator = None

    restart_signal = Signal()
    
    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.config = config
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.app = app
        self.setupThread()
        self.setupUi()
        self.setupConfig()
        self.setupShortcuts()
        self.showMaximized()

        self.setAcceptDrops(True)

        if open_dir != '' and osp.exists(open_dir):
            self.OpenProj(open_dir)
        elif self.config.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.OpenProj(proj_dir)

    def setStyleSheet(self, styleSheet: str) -> None:
        self.imgtrans_progress_msgbox.setStyleSheet(styleSheet)
        self.source_download_msgbox.setStyleSheet(styleSheet)
        self.export_doc_thread.progress_bar.setStyleSheet(styleSheet)
        self.import_doc_thread.progress_bar.setStyleSheet(styleSheet)
        # sel_menu_size = self.selectext_minimenu.sizeHint()
        # self.selectext_minimenu.setFixedWidth(sel_menu_size.width())
        return super().setStyleSheet(styleSheet)

    def setupThread(self):
        self.imsave_thread = ImgSaveThread()
        self.export_doc_thread = ExportDocThread()
        self.export_doc_thread.fin_io.connect(self.on_fin_export_doc)
        self.import_doc_thread = ImportDocThread(self)
        self.import_doc_thread.fin_io.connect(self.on_fin_import_doc)

    def resetStyleSheet(self, reverse_icon: bool = False):
        theme = 'eva-dark' if self.config.darkmode else 'eva-light'
        self.setStyleSheet(parse_stylesheet(theme, reverse_icon))

    def setupUi(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        self.configPanel = ConfigPanel(self.config, self)
        self.configPanel.trans_config_panel.show_MT_keyword_window.connect(self.show_MT_keyword_window)
        self.configPanel.ocr_config_panel.show_OCR_keyword_window.connect(self.show_OCR_keyword_window)
        self.config = self.configPanel.config

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.openDir)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.save_proj)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.import_doc.connect(self.on_import_doc)

        self.pageList = PageListView()
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

        self.configPanel.src_link_textbox.setText(self.config.src_link_flag)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # set up comic canvas
        self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)

        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.bottomBar.textlayerSlider.valueChanged.connect(self.canvas.setTextLayerTransparencyBySlider)
        
        self.drawingPanel = DrawingPanel(self.canvas, self.configPanel.inpaint_config_panel)
        self.textPanel = TextPanel(self.app)
        self.textPanel.formatpanel.effect_panel.setParent(self)
        self.textPanel.formatpanel.effect_panel.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.CustomizeWindowHint)
        self.textPanel.formatpanel.fontfmtLabel.clicked.connect(self.show_fontstyle_presets)
        
        self.presetPanel = PresetPanel(self)
        self.presetPanel.setParent(self)
        self.presetPanel.setWindowFlags(Qt.WindowType.Window)
        self.presetPanel.global_fmt_str = self.textPanel.formatpanel.global_fontfmt_str
        self.presetPanel.hide()
        self.presetPanel.hide_signal.connect(self.save_config)
        self.presetPanel.load_preset.connect(self.textPanel.formatpanel.on_load_preset)

        self.ocrSubWidget = KeywordSubWidget(self.tr("Keyword substitution for OCR"))
        self.ocrSubWidget.setParent(self)
        self.ocrSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.ocrSubWidget.hide()
        self.mtSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation"))
        self.mtSubWidget.setParent(self)
        self.mtSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtSubWidget.hide()

        self.st_manager = SceneTextManager(self.app, self, self.canvas, self.textPanel)
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
        # self.comicTransSplitter.addWidget(self.pageList)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)

        self.centralStackWidget.addWidget(self.comicTransSplitter)
        self.centralStackWidget.addWidget(self.configPanel)

        self.selectext_minimenu = self.st_manager.selectext_minimenu = SelectTextMiniMenu(self.app, self.configPanel.config, self)
        self.selectext_minimenu.block_current_editor.connect(self.st_manager.on_block_current_editor)
        self.selectext_minimenu.hide()

        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        margin = mainVBoxLayout.contentsMargins()
        self.main_margin = margin
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox()
        self.source_download_msgbox = SourceDownloadProgressMessageBox()
        self.resetStyleSheet()

    def setupConfig(self):

        config = self.st_manager.config = self.config

        self.bottomBar.originalSlider.setValue(int(config.original_transparency * 100))
        self.drawingPanel.maskTransperancySlider.setValue(int(config.mask_transparency * 100))
        self.leftBar.initRecentProjMenu(config.recent_proj_list)
        self.leftBar.save_config.connect(self.save_config)
        self.leftBar.imgTransChecker.setChecked(True)
        self.st_manager.formatpanel.global_format = config.global_fontformat
        self.st_manager.formatpanel.set_active_format(config.global_fontformat)
        
        self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)

        self.bottomBar.ocrChecker.setCheckState(config.module.enable_ocr)
        self.bottomBar.transChecker.setChecked(config.module.enable_translate)

        self.module_manager = module_manager = ModuleManager(config, self.imgtrans_proj)
        module_manager.update_translator_status.connect(self.updateTranslatorStatus)
        module_manager.update_source_download_status.connect(self.updateSourceDownloadStatus)
        self.configPanel.update_source_download_status.connect(self.updateSourceDownloadStatus)
        module_manager.update_inpainter_status.connect(self.updateInpainterStatus)
        module_manager.finish_translate_page.connect(self.finishTranslatePage)
        module_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        module_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        module_manager.setupThread(self.configPanel, self.imgtrans_progress_msgbox, self.ocr_postprocess, self.translate_postprocess)
        module_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        module_manager.imgtrans_thread.mask_postprocess = self.drawingPanel.rectPanel.post_process_mask
        module_manager.blktrans_pipeline_finished.connect(self.on_blktrans_finished)
        module_manager.imgtrans_thread.get_maskseg_method = self.drawingPanel.rectPanel.get_maskseg_method
        module_manager.imgtrans_thread.post_process_mask = self.drawingPanel.rectPanel.post_process_mask

        self.leftBar.run_imgtrans.connect(self.on_run_imgtrans)
        self.leftBar.run_sync_source.connect(self.on_run_sync_source)
        self.bottomBar.ocrcheck_statechanged.connect(module_manager.setOCRMode)
        self.bottomBar.transcheck_statechanged.connect(module_manager.setTransMode)
        self.bottomBar.inpaint_btn_clicked.connect(self.inpaintBtnClicked)
        self.bottomBar.source_download_btn_clicked.connect(self.SourceDownloadBtnClicked)
        self.bottomBar.translatorStatusbtn.clicked.connect(self.translatorStatusBtnPressed)
        self.bottomBar.transTranspageBtn.run_target.connect(self.on_transpagebtn_pressed)

        self.titleBar.darkModeAction.setChecked(config.darkmode)

        self.drawingPanel.set_config(config.drawpanel)
        self.drawingPanel.initDLModule(module_manager)

        self.global_search_widget.imgtrans_proj = self.imgtrans_proj
        self.global_search_widget.setupReplaceThread(self.st_manager.pairwidget_list, self.st_manager.textblk_item_list)
        self.global_search_widget.replace_thread.finished.connect(self.on_global_replace_finished)

        self.configPanel.setupConfig()
        self.configPanel.save_config.connect(self.save_config)

        self.source_download = SourceDownload(config, self.imgtrans_proj, self.source_download_msgbox)
        self.source_download.open_downloaded_proj.connect(self.openDir)
        self.source_download.update_progress_bar.connect(self.source_download_msgbox.updateDownloadBar)
        self.source_download.finished_downloading.connect(self.on_finished_sync_source)

        textblock_mode = config.imgtrans_textblock
        if config.imgtrans_textedit:
            if textblock_mode:
                self.bottomBar.textblockChecker.setChecked(True)
            self.bottomBar.texteditChecker.click()
        elif config.imgtrans_paintmode:
            self.bottomBar.paintChecker.click()

        self.presetPanel.initPresets(config.font_presets)

        self.canvas.search_widget.set_config(config)
        self.global_search_widget.set_config(config)

        if self.rightComicTransStackPanel.isHidden():
            self.setPaintMode()

        try:
            self.ocrSubWidget.loadCfgSublist(config.ocr_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            config.ocr_sublist = []
            self.ocrSubWidget.loadCfgSublist(config.ocr_sublist)

        try:
            self.mtSubWidget.loadCfgSublist(config.mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            config.mt_sublist = []
            self.mtSubWidget.loadCfgSublist(config.mt_sublist)

    def setupImgTransUI(self):
        self.centralStackWidget.setCurrentIndex(0)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        self.centralStackWidget.setCurrentIndex(1)

    def set_display_lang(self, lang: str):
        self.retranslateUI()

    def OpenProj(self, proj_path: str):
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)

    def openDir(self, directory: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load(directory)
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(osp.basename(directory))
            self.updatePageList()
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            LOGGER.exception(e)
            LOGGER.warning("Failed to load project from " + directory)
            LOGGER.warning("If you were trying to download images check IMPLEMENTED_SOURCES.md for more information")
            self.module_manager.handleRunTimeException(self.tr('Failed to load project ') + directory, '')
            return
        
    def dropOpenDir(self, directory: str):
        if isinstance(directory, str) and osp.exists(directory):
            self.leftBar.updateRecentProjList(directory)
            self.openDir(directory)

    def openJsonProj(self, json_path: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems()
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self.opening_dir = False
        except Exception as e:
            self.opening_dir = False
            LOGGER.exception(e)
            LOGGER.warning("Failed to load project from " + json_path)
            self.module_manager.handleRunTimeException(self.tr('Failed to load project ') + json_path, '')
        
    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.imgtrans_proj.pages) >= C.PAGELIST_THUMBNAIL_MAXNUM:
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self.st_manager.hovering_transwidget = None
        self.canvas.prepareClose()
        self.save_config()
        if not self.imgtrans_proj.is_empty:
            self.imgtrans_proj.save()
        return super().closeEvent(event)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
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
        self.config.imgtrans_paintmode = self.bottomBar.paintChecker.isChecked()
        self.config.imgtrans_textedit = self.bottomBar.texteditChecker.isChecked()
        self.config.mask_transparency = self.canvas.mask_transparency
        self.config.original_transparency = self.canvas.original_transparency
        self.config.drawpanel = self.drawingPanel.get_config()
        with open(CONFIG_PATH, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(self.config))

    def onHideCanvas(self):
        self.canvas.alt_pressed = False
        self.canvas.scale_tool_mode = False

    def conditional_manual_save(self):
        if self.canvas.projstate_unsaved and not self.opening_dir:
            update_scene_text = save_proj = self.canvas.text_change_unsaved()
            save_rst_only = not self.canvas.draw_change_unsaved()
            if not save_rst_only:
                save_proj = True
            
            self.saveCurrentPage(update_scene_text, save_proj, restore_interface=True, save_rst_only=save_rst_only)

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if self.save_on_page_changed:
                self.conditional_manual_save()
            self.imgtrans_proj.set_current_img(item.text())
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.updateCanvas()
            self.st_manager.hovering_transwidget = None
            self.st_manager.updateSceneTextitems()
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
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.run_trigger.connect(self.leftBar.runImgtransBtn.click)
        self.titleBar.translate_page_trigger.connect(self.bottomBar.transTranspageBtn.click)
        self.titleBar.fontstyle_trigger.connect(self.show_fontstyle_presets)
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)

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

        shortcutDelete = QShortcut(QKeySequence.StandardKey.Delete, self)
        shortcutDelete.activated.connect(self.shortcutDelete)

    def shortcutNext(self):
        if self.centralStackWidget.currentIndex() == 0:
            index = self.pageList.currentIndex()
            page_count = self.pageList.count()
            if index.isValid():
                row = index.row()
                row = (row + 1) % page_count
                self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        if self.centralStackWidget.currentIndex() == 0:
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
            self.textPanel.formatpanel.formatBtnGroup.boldBtn.click()

    def shortcutDelete(self):
        if self.canvas.gv.isVisible():
            self.canvas.delete_textblks.emit(0)

    def shortcutItalic(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.italicBtn.click()

    def shortcutUnderline(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.underlineBtn.click()

    def on_redo(self):
        self.canvas.redo()

    def on_undo(self):
        self.canvas.undo()

    def on_page_search(self):
        if self.canvas.gv.isVisible():
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

    def show_MT_keyword_window(self):
        self.mtSubWidget.show()


    def show_OCR_keyword_window(self):
        self.ocrSubWidget.show()

    def on_req_update_pagetext(self):
        self.global_search_widget.searched_textstack_step = self.canvas.text_undo_stack.index()
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()

    def on_req_move_page(self, page_name: str, force_save=False):
        ori_save = self.save_on_page_changed
        self.save_on_page_changed = False
        current_img = self.imgtrans_proj.current_img
        if current_img == page_name and not force_save:
            return
        if current_img not in self.global_search_widget.page_set:
            if self.canvas.projstate_unsaved: 
                self.saveCurrentPage()
        else:
            self.saveCurrentPage(save_rst_only=True)
        self.pageList.setCurrentRow(self.imgtrans_proj.pagename2idx(page_name))
        self.save_on_page_changed = ori_save

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
        if self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()

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
        self.config.imgtrans_textblock = mode
        self.st_manager.showTextblkItemRect(mode)

    def save_proj(self):
        if self.leftBar.imgTransChecker.isChecked()\
            and self.imgtrans_proj.directory is not None:
            
            # self.saveCurrentPage(update_scene_text=True, restore_interface=True)
            self.conditional_manual_save()

    def saveCurrentPage(self, update_scene_text=True, save_proj=True, restore_interface=False, save_rst_only=False):
        
        if not self.imgtrans_proj.img_valid:
            return
        
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        
        if self.rightComicTransStackPanel.isHidden():
            self.bottomBar.texteditChecker.click()

        trans_idx = self.rightComicTransStackPanel.currentIndex()
        if trans_idx != 1:
            self.bottomBar.texteditChecker.click()

        restore_original_transparency = None
        if self.bottomBar.originalSlider.value() != 0:
            restore_original_transparency = self.bottomBar.originalSlider.value()
            self.bottomBar.originalSlider.setValue(0)

        restore_textlayer_transparency = None
        if self.bottomBar.textlayerSlider.value() != 100:
            restore_textlayer_transparency = self.bottomBar.textlayerSlider.value()
            self.bottomBar.textlayerSlider.setValue(100)

        restore_textblock_mode = False
        if self.config.imgtrans_textblock:
            restore_textblock_mode = True
            self.bottomBar.textblockChecker.click()

        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        if save_proj:
            self.imgtrans_proj.save()
            if not save_rst_only:
                mask_path = self.imgtrans_proj.get_mask_path()
                mask_array = self.imgtrans_proj.mask_array
                self.imsave_thread.saveImg(mask_path, mask_array)
                inpainted_path = self.imgtrans_proj.get_inpainted_path()
                if self.canvas.drawingLayer.drawed():
                    inpainted = self.canvas.inpaintLayer.pixmap()
                    painter = QPainter(inpainted)
                    painter.drawPixmap(0, 0, self.canvas.drawingLayer.get_drawed_pixmap())
                    painter.end()
                else:
                    inpainted = self.imgtrans_proj.inpainted_array
                self.imsave_thread.saveImg(inpainted_path, inpainted)

        img = QImage(self.canvas.imgLayer.pixmap().size(), QImage.Format.Format_ARGB32)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.canvas.clearSelection()
        self.canvas.render(painter)
        painter.end()
        imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
        self.imsave_thread.saveImg(imsave_path, img, self.imgtrans_proj.current_img)
            
        self.canvas.setProjSaveState(False)
        self.canvas.update_saved_undostep()

        if restore_interface:
            if restore_original_transparency is not None:
                self.bottomBar.originalSlider.setValue(restore_original_transparency)
            if restore_textlayer_transparency is not None:
                self.bottomBar.textlayerSlider.setValue(restore_textlayer_transparency)
            if trans_idx != 1:
                self.bottomBar.paintChecker.click()
            if restore_textblock_mode:
                self.bottomBar.textblockChecker.click()
            if hide_tsc:
                self.st_manager.txtblkShapeControl.show()
        
    def translatorStatusBtnPressed(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnTranslator()

    def inpaintBtnClicked(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnInpaint()

    def SourceDownloadBtnClicked(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnSourceDownload()

    def updateTranslatorStatus(self, translator: str, source: str, target: str):
        if translator == '':
            self.bottomBar.translatorStatusbtn.hide()
            self.bottomBar.translatorStatusbtn.hide()
        else:
            self.bottomBar.translatorStatusbtn.updateStatus(translator, source, target)
            self.bottomBar.translatorStatusbtn.show()
            self.bottomBar.transTranspageBtn.show()

    def updateSourceDownloadStatus(self, url: str):
        if url == '':
            self.bottomBar.sourceStatusBtn.hide()
            self.bottomBar.sourceStatusBtn.hide()
        else:
            self.bottomBar.sourceStatusBtn.updateStatus(url)
            self.bottomBar.sourceStatusBtn.show()
            self.bottomBar.sourceStatusBtn.show()

    def updateInpainterStatus(self, inpainter: str):
        self.bottomBar.inpainterStatBtn.updateStatus(inpainter)

    def on_transpagebtn_pressed(self, run_target: bool):
        page_key = self.imgtrans_proj.current_img
        if page_key is None:
            self.bottomBar.transTranspageBtn.setRunText()
            return
        # if run_target and self.canvas.text_change_unsaved():
        #     self.st_manager.updateTextBlkList()
        
        # self.module_manager.translatePage(run_target, page_key)

        blkitem_list = self.st_manager.textblk_item_list

        if len(blkitem_list) < 1:
            if self.bottomBar.transTranspageBtn.running:
                self.bottomBar.transTranspageBtn.setRunText()
            return
        
        self.translateBlkitemList(blkitem_list, -1)


    def translateBlkitemList(self, blkitem_list: List, mode: int) -> bool:

        tgt_img = self.imgtrans_proj.img_array
        if tgt_img is None:
            return False
        
        if len(blkitem_list) < 1:
            return False
        
        self.global_search_widget.set_document_edited()
        
        im_h, im_w = tgt_img.shape[:2]

        blk_list, blk_ids = [], []
        for blkitem in blkitem_list:
            blk = blkitem.blk
            blk._bounding_rect = blkitem.absBoundingRect()
            blk.vertical = blkitem.is_vertical
            blk.text = self.st_manager.pairwidget_list[blkitem.idx].e_source.toPlainText()
            blk_ids.append(blkitem.idx)
            blk.set_lines_by_xywh(blk._bounding_rect, angle=-blk.angle, x_range=[0, im_w-1], y_range=[0, im_h-1], adjust_bbox=True)
            blk_list.append(blk)

        self.module_manager.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids)
        return True


    def finishTranslatePage(self, page_key):
        self.bottomBar.transTranspageBtn.setRunText()
        if page_key == self.imgtrans_proj.current_img:
            self.st_manager.updateTranslation()

    def on_imgtrans_pipeline_finished(self):
        self.postprocess_mt_toggle = True

    def postprocess_translations(self, blk_list: List[TextBlock]) -> None:
        src_is_cjk = is_cjk(self.config.module.translate_source)
        tgt_is_cjk = is_cjk(self.config.module.translate_target)
        if tgt_is_cjk:
            for blk in blk_list:
                if src_is_cjk:
                    blk.translation = full_len(blk.translation)
                else:
                    blk.translation = half_len(blk.translation)
                    blk.translation = re.sub(r'([?.!"])\s+', r'\1', blk.translation)    # remove spaces following punctuations
        elif src_is_cjk:
            for blk in blk_list:
                if blk.vertical:
                    blk._alignment = 1
                blk.translation = half_len(blk.translation)
                blk.vertical = False

        for blk in blk_list:
            blk.translation = self.mtSubWidget.sub_text(blk.translation)

    def on_pagtrans_finished(self, page_index: int):
        blk_list = self.imgtrans_proj.get_blklist_byidx(page_index)
        self.postprocess_translations(blk_list)
                
        # override font format if necessary
        override_fnt_size = self.config.let_fntsize_flag == 1
        override_fnt_stroke = self.config.let_fntstroke_flag == 1
        override_fnt_color = self.config.let_fntcolor_flag == 1
        override_fnt_scolor = self.config.let_fnt_scolor_flag == 1
        override_alignment = self.config.let_alignment_flag == 1
        override_effect = self.config.let_fnteffect_flag == 1
        gf = self.textPanel.formatpanel.global_format
        
        for blk in blk_list:
            if override_fnt_size:
                blk.font_size = pt2px(gf.size)
            if override_fnt_stroke:
                blk.default_stroke_width = gf.stroke_width
                blk.stroke_decide_by_colordiff = False
            if override_fnt_color:
                blk.set_font_colors(frgb=gf.frgb, accumulate=False)
            if override_fnt_scolor:
                blk.set_font_colors(srgb=gf.srgb, accumulate=False)
            if override_alignment:
                blk._alignment = gf.alignment
            if override_effect:
                blk.opacity = gf.opacity
                blk.shadow_color = gf.shadow_color
                blk.shadow_radius = gf.shadow_radius
                blk.shadow_strength = gf.shadow_strength
                blk.shadow_offset = gf.shadow_offset
            
            blk.line_spacing = gf.line_spacing
            blk.letter_spacing = gf.letter_spacing
            sw = blk.stroke_width
            if sw > 0:
                blk.font_size = int(blk.font_size / (1 + sw))

        self.st_manager.auto_textlayout_flag = self.config.let_autolayout_flag
        
        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
        
        self.saveCurrentPage(False, False)
        if page_index + 1 == self.imgtrans_proj.num_pages:
            self.st_manager.auto_textlayout_flag = False

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

        if self.bottomBar.transTranspageBtn.running:
            self.bottomBar.transTranspageBtn.setRunText()

        if len(blk_ids) < 1:
            return
        
        blkitem_list = [self.st_manager.textblk_item_list[idx] for idx in blk_ids]

        pairw_list = []
        for blk in blkitem_list:
            pairw_list.append(self.st_manager.pairwidget_list[blk.idx])
        self.canvas.push_undo_command(RunBlkTransCommand(self.canvas, blkitem_list, pairw_list, mode))

    def on_imgtrans_progressbox_showed(self):
        msg_size = self.module_manager.progress_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.module_manager.progress_msgbox.move(p)

    def on_source_download_progressbox_showed(self):
        msg_size = self.source_download_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.source_download_msgbox.move(p)

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
        if lang != self.config.display_lang:
            self.config.display_lang = lang
            self.set_display_lang(lang)

    def on_run_imgtrans(self):
        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        self.postprocess_mt_toggle = False
        self.module_manager.runImgtransPipeline()

    def on_run_sync_source(self):
        self.source_download_msgbox.show_all_bars()
        self.source_download_msgbox.zero_progress()
        self.source_download_msgbox.show()
        self.on_source_download_progressbox_showed()
        self.source_download.start()

    def on_finished_sync_source(self):
        self.source_download_msgbox.hide_all_bars()
        self.source_download_msgbox.hide()


    def on_transpanel_changed(self):
        self.canvas.editor_index = self.rightComicTransStackPanel.currentIndex()
        if not self.canvas.textEditMode() and self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()

    def show_fontstyle_presets(self):
        fmt = self.textPanel.formatpanel.active_format
        fmt_name = self.textPanel.formatpanel.fontfmtLabel.text()
        self.presetPanel.updateCurrentFontFormat(fmt, fmt_name)
        self.presetPanel.show()

    def on_export_doc(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()
        self.export_doc_thread.exportAsDoc(self.imgtrans_proj)

    def on_import_doc(self):
        self.import_doc_thread.importDoc(self.imgtrans_proj)

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
        self.canvas.text_undo_stack.push(
            GlobalRepalceAllCommand(rt.sceneitem_list, rt.background_list, rt.target_text, self.imgtrans_proj)
        )
        rt.sceneitem_list = None
        rt.background_list = None

    def on_darkmode_triggered(self):
        self.config.darkmode = self.titleBar.darkModeAction.isChecked()
        self.resetStyleSheet(reverse_icon=True)
        self.save_config()

    def ocr_postprocess(self, text: str, blk: TextBlock = None) -> str:
        text = self.ocrSubWidget.sub_text(text)
        return text

    def translate_postprocess(self, text: str, blk: TextBlock = None) -> str:
        if self.postprocess_mt_toggle:
            text = self.mtSubWidget.sub_text(text)
        return text