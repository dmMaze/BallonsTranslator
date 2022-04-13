import os.path as osp
import os
import json

from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QWidget, QSizePolicy, QComboBox, QListView, QToolBar, QMenu, QSpacerItem, QPushButton, QAction, QCheckBox, QToolButton, QSplitter, QListWidget, QShortcut, QListWidgetItem
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QSize, QThread
from PyQt5.QtGui import QGuiApplication, QIcon, QCloseEvent, QKeySequence, QImage, QPainter, QFont

from typing import List, Union, Tuple

from .misc import ProjImgTrans
from .canvas import Canvas
from .configpanel import ConfigPanel
from .dl_manager import DLManager
from .imgtranspanel import TextPanel
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager
from .mainwindowbars import TitleBar, LeftBar, RightBar, BottomBar
from .io_thread import ImgSaveThread
from .stylewidgets import FrameLessMessageBox
from .constants import STYLESHEET_PATH, CONFIG_PATH, DPI, LDPI, LANG_SUPPORT_VERTICAL
from . import constants


class PageListView(QListWidget):    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMaximumWidth(512)
        self.setIconSize(QSize(70, 70))


class StackWidget(QStackedWidget):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(self.onCurrentChanged)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

    def addWidget(self, w: QWidget) -> int:
        super().addWidget(w)
        self.adjustSize()

    def onCurrentChanged(self, index: int) -> None:
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.adjustSize()


class FrameLessWindow(QMainWindow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)


class MainWindow(FrameLessWindow):

    proj_directory = None
    imgtrans_proj: ProjImgTrans = ProjImgTrans()
    save_on_page_changed = True
    
    def __init__(self, app: QApplication, open_dir='', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        global DPI, LDPI
        DPI = QGuiApplication.primaryScreen().physicalDotsPerInch()
        constants.LDPI = QGuiApplication.primaryScreen().logicalDotsPerInch()

        self.app = app
        self.setupLogger()
        self.setupUi()
        self.setupConfig()
        self.setupShortcuts()
        self.showMaximized()

        self.imsave_thread = ImgSaveThread()

        if open_dir != '' and osp.exists(open_dir):
            self.openDir(open_dir)
        elif self.config.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.openDir(proj_dir)

    def setupUi(self):
        screen_size = QApplication.desktop().screenGeometry().size()
        self.setMinimumWidth(screen_size.width()*0.5)

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.stateChanged.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        
        self.leftBar.open_dir.connect(self.openDir)
        self.leftBar.save_proj.connect(self.save_proj)

        self.pageList = PageListView()
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)
        
        self.centralStackWidget = QStackedWidget(self)
        
        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)

        self.rightBar = RightBar(self)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        # mainHLayout.addWidget(self.pageList)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.addWidget(self.rightBar)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # set up comic canvas
        self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)

        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.configPanel = ConfigPanel(self)
        self.config = self.configPanel.config

        self.drawingPanel = DrawingPanel(self.canvas, self.configPanel.inpaint_config_panel)
        self.textPanel = TextPanel(self.app)
        self.st_manager = SceneTextManager(self.app, self.canvas, self.textPanel)

        # comic trans pannel
        self.rightComicTransStackPanel = StackWidget(self)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)

        self.comicTransSplitter = QSplitter(Qt.Horizontal)
        self.comicTransSplitter.addWidget(self.pageList)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.comicTransSplitter.setStretchFactor(2, 1)

        self.centralStackWidget.addWidget(self.comicTransSplitter)
        self.centralStackWidget.addWidget(self.configPanel)

        self.centerWidget = QWidget()
        self.setCentralWidget(self.centerWidget)
        mainVBoxLayout = QVBoxLayout(self.centerWidget)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)
        self.mainvlayout = mainVBoxLayout
        
    def setupConfig(self):
        with open(STYLESHEET_PATH, "r", encoding='utf-8') as f:
            self.setStyleSheet(f.read())
        try:
            with open(CONFIG_PATH, 'r', encoding='utf8') as f:
                config_dict = json.loads(f.read())
            self.config.load_from_dict(config_dict)
        except Exception as e:
            self.logger.exception(e)
            self.logger.warning("Failed to load config file, using default config")

        self.bottomBar.originalSlider.setValue(self.config.original_transparency * 100)
        self.drawingPanel.maskTransperancySlider.setValue(self.config.mask_transparency * 100)
        self.leftBar.updateRecentProjList(self.config.recent_proj_list)
        self.leftBar.recent_proj_list = self.config.recent_proj_list
        self.leftBar.imgTransChecker.setChecked(True)
        self.st_manager.formatpanel.global_format = self.config.global_fontformat
        self.st_manager.formatpanel.set_active_format(self.config.global_fontformat)
        if self.config.imgtrans_paintmode:
            self.bottomBar.paintChecker.setChecked(True)
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.rightComicTransStackPanel.onCurrentChanged(0)
            self.canvas.setPaintMode(True)
            self.st_manager.setTextEditMode(False)
        elif self.config.imgtrans_textedit:
            self.bottomBar.texteditChecker.setChecked(True)
            self.bottomBar.originalSlider.setHidden(True)
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.canvas.setPaintMode(False)
            self.st_manager.setTextEditMode(True)
            if self.config.imgtrans_textblock:
                self.bottomBar.textblockChecker.setChecked(True)
                self.setTextBlockMode()
        else:
            self.bottomBar.originalSlider.setHidden(True)
            self.rightComicTransStackPanel.setHidden(True)
            self.st_manager.setTextEditMode(False)

        self.bottomBar.ocrChecker.setCheckState(self.config.dl.enable_ocr)
        self.bottomBar.transChecker.setChecked(self.config.dl.enable_translate)

        self.dl_manager = dl_manager = DLManager(self.config.dl, self.imgtrans_proj, self.configPanel, self.logger)
        self.dl_manager.update_translator_status.connect(self.updateTranslatorStatus)
        self.dl_manager.update_inpainter_status.connect(self.updateInpainterStatus)
        self.dl_manager.finish_translate_page.connect(self.finishTranslatePage)
        self.dl_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        self.dl_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        self.dl_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)

        self.leftBar.run_imgtrans.connect(dl_manager.runImgtransPipeline)
        self.bottomBar.ocrcheck_statechanged.connect(dl_manager.setOCRMode)
        self.bottomBar.transcheck_statechanged.connect(dl_manager.setTransMode)
        self.bottomBar.translatorStatusbtn.clicked.connect(self.translatorStatusBtnPressed)
        self.bottomBar.transTranspageBtn.run_target.connect(self.on_transpagebtn_pressed)

        self.drawingPanel.set_config(self.config.drawpanel)
        self.drawingPanel.initDLModule(dl_manager)

        if self.config.open_recent_on_startup:
            self.configPanel.open_on_startup_checker.setChecked(True)

    def setupLogger(self):
        from utils.logger import logger
        self.logger = logger

    def setupImgTransUI(self):
        self.centralStackWidget.setCurrentIndex(0)
        if self.leftBar.showPageListLabel.checkState() == 2:
            self.pageList.show()

    def setupConfigUI(self):
        self.centralStackWidget.setCurrentIndex(1)

    def openDir(self, directory: str):
        try:
            self.imgtrans_proj.load(directory)
        except Exception as e:
            self.logger.exception(e)
            self.logger.warning("Failed to load project from " + directory)
            self.dl_manager.handleRunningException(self.tr('Failed to load project ') + directory, '')
            return
        self.proj_directory = directory
        self.titleBar.setTitleContent(osp.basename(directory))
        self.updatePageList()
        
    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.imgtrans_proj.pages) >= 50:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.proj_directory, imgname)), imgname)
        for imgname in self.imgtrans_proj.pages:
            lstitem =  item_func(imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.imgtrans_proj.current_img:
                self.pageList.setCurrentItem(lstitem)

    def pageLabelStateChanged(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.leftBar.showPageListLabel.checkState() == 2:
                self.pageList.show()
            else:
                self.pageList.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.canvas.disconnect()
        self.canvas.undoStack.disconnect()
        self.config.imgtrans_paintmode = self.bottomBar.paintChecker.isChecked()
        self.config.imgtrans_textedit = self.bottomBar.texteditChecker.isChecked()
        self.config.mask_transparency = self.canvas.mask_transparency
        self.config.original_transparency = self.canvas.original_transparency
        self.config.drawpanel = self.drawingPanel.get_config()
        config_dict = self.config.to_dict()
        with open(CONFIG_PATH, 'w', encoding='utf8') as f:
            f.write(json.dumps(config_dict, ensure_ascii=False, indent=4, separators=(',', ':')))
        return super().closeEvent(event)

    def onHideCanvas(self):
        self.pageList.hide()
        self.canvas.alt_pressed = False
        self.canvas.scale_tool_mode = False

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        if item is not None:
            if self.save_on_page_changed:
                if self.canvas.projstate_unsaved:
                    self.saveCurrentPage()
            self.st_manager.canvasUndoStack.clear()
            self.imgtrans_proj.set_current_img(item.text())
            self.canvas.updateCanvas()
            self.st_manager.updateTextList()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)

    def setupShortcuts(self):
        shortcutNext = QShortcut(QKeySequence("D"), self)
        shortcutNext.activated.connect(self.shortcutNext)
        shortcutBefore = QShortcut(QKeySequence("A"), self)
        shortcutBefore.activated.connect(self.shortcutBefore)
        shortcutTextblock = QShortcut(QKeySequence("W"), self)
        shortcutTextblock.activated.connect(self.shortcutTextblock)
        shortcutZoomIn = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        shortcutZoomIn.activated.connect(self.canvas.gv.scale_up_signal)
        shortcutZoomOut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        shortcutZoomOut.activated.connect(self.canvas.gv.scale_down_signal)
        
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

    def shortcutTextblock(self):
        if self.bottomBar.texteditChecker.isChecked():
            self.bottomBar.textblockChecker.click()

    def setPaintMode(self):
        if self.bottomBar.paintChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.canvas.setPaintMode(True)
            self.bottomBar.originalSlider.show()
            self.bottomBar.textblockChecker.hide()
        else:
            self.bottomBar.originalSlider.hide()
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
            self.bottomBar.originalSlider.setHidden(True)
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
            self.st_manager.updateTextBlkList()
            self.saveCurrentPage()

    def saveCurrentPage(self, update_scene_text=True, save_proj=True):
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        if save_proj:
            self.imgtrans_proj.save()
            mask_path = self.imgtrans_proj.get_mask_path()
            mask_array = self.imgtrans_proj.mask_array
            inpainted_path = self.imgtrans_proj.get_inpainted_path()
            inpainted_array = self.imgtrans_proj.inpainted_array
        else:
            mask_path = inpainted_path = None
            
        img = QImage(self.canvas.imgLayer.pixmap().size(), QImage.Format.Format_ARGB32)

        if self.config.imgtrans_textblock:
            self.bottomBar.textblockChecker.setChecked(False)
            self.setTextBlockMode()

        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.HighQualityAntialiasing)
        self.canvas.render(painter)
        painter.end()
        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
        self.imsave_thread.saveImg(imsave_path, img)
        if mask_path is not None:
            self.imsave_thread.saveImg(mask_path, mask_array)
        if inpainted_path is not None:
            self.imsave_thread.saveImg(inpainted_path, inpainted_array)
        if hide_tsc:
            self.st_manager.txtblkShapeControl.show()
        self.canvas.setProjSaveState(False)
        
    def translatorStatusBtnPressed(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnTranslator()

    def updateTranslatorStatus(self, translator: str, source: str, target: str):
        if translator == '':
            self.bottomBar.translatorStatusbtn.hide()
            self.bottomBar.translatorStatusbtn.hide()
        else:
            self.bottomBar.translatorStatusbtn.updateStatus(translator, source, target)
            self.bottomBar.translatorStatusbtn.show()
            self.bottomBar.transTranspageBtn.show()

    def updateInpainterStatus(self, inpainter: str):
        self.bottomBar.inpainterStatBtn.updateStatus(inpainter)

    def on_transpagebtn_pressed(self, run_target: bool):
        
        page_key = self.imgtrans_proj.current_img
        if run_target:
            self.st_manager.updateTextBlkList()
        self.dl_manager.translatePage(run_target, page_key)

    def finishTranslatePage(self, page_key):
        self.bottomBar.transTranspageBtn.setRunText()
        if page_key == self.imgtrans_proj.current_img:
            self.st_manager.updateTranslation()

    def on_imgtrans_pipeline_finished(self):
        self.pageListCurrentItemChanged()

    def on_pagtrans_finished(self, page_index: int):
        if self.config.dl.translate_target not in LANG_SUPPORT_VERTICAL:
            for blk in self.imgtrans_proj.get_blklist_byidx(page_index):
                blk.vertical = False
        self.pageList.setCurrentRow(page_index)
        self.saveCurrentPage(False, False)

    def on_savestate_changed(self, unsaved: bool):
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def on_imgtrans_progressbox_showed(self):
        msg_size = self.dl_manager.progress_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.dl_manager.progress_msgbox.move(p)

    

    def on_closebtn_clicked(self):
        if self.imsave_thread.isRunning():
            self.imsave_thread.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Saving image...'))
            self.imsave_thread.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

