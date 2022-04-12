import os.path as osp
import os
import json
from collections import OrderedDict

from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QWidget, QFileDialog, QLabel, QSizePolicy, QComboBox, QListView, QToolBar, QMenu, QSpacerItem, QPushButton, QAction, QCheckBox, QToolButton, QSplitter, QListWidget, QShortcut, QListWidgetItem
from PyQt5.QtCore import Qt, QCoreApplication, pyqtSignal, QPoint, QSize, QLocale
from PyQt5.QtGui import QGuiApplication, QIcon, QMouseEvent, QCloseEvent, QKeySequence, QImage, QPainter

from typing import List, Union, Tuple

from .misc import ProjImgTrans
from .canvas import Canvas
from .configpanel import ConfigPanel
from .stylewidgets import Widget, PaintQSlider
from .dl_manager import DLManager
from .imgtranspanel import TextPanel
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager
from .constants import STYLESHEET_PATH, CONFIG_PATH, DPI, LDPI, LANG_SUPPORT_VERTICAL
from . import constants

class StatusButton(QPushButton):
    pass

class RunStopTextBtn(StatusButton):
    run_target = pyqtSignal(bool)
    def __init__(self, run_text: str, stop_text: str, run_tool_tip: str = None, stop_tool_tip: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = False
        self.run_text = run_text
        self.stop_text = stop_text
        self.run_tool_tip = run_tool_tip
        self.stop_tool_tip = stop_tool_tip
        self.setRunText()
        self.pressed.connect(self.on_pressed)

    def on_pressed(self):
        self.running = not self.running
        self.run_target.emit(self.running)
        if self.running:
            self.setStopText()
        else:
            self.setRunText()

    def setRunText(self):
        self.setText(self.run_text)
        if self.run_tool_tip is not None:
            self.setToolTip(self.run_tool_tip)

    def setStopText(self):
        self.setText(self.stop_text)
        if self.stop_tool_tip is not None:
            self.setToolTip(self.stop_tool_tip)

class TranslatorStatusButton(StatusButton):
    def updateStatus(self, translator: str, source: str, target: str):
        self.setText(self.tr('Translator: ') + translator + '   '\
                     + self.tr('Source: ') + source + '   '\
                     + self.tr('Target: ') + target)

class InpainterStatusButton(StatusButton):
    def updateStatus(self, inpainter: str):
        self.setText(self.tr('Inpainter: ') + inpainter)

class TextChecker(QLabel):
    checkStateChanged = pyqtSignal(bool)
    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setText(text)
        self.checked = False
        self.setAlignment(Qt.AlignCenter)

    def setCheckState(self, checked: bool):
        self.checked = checked
        if checked:
            self.setStyleSheet("QLabel { background-color: rgb(30, 147, 229); color: white; }")
        else:
            self.setStyleSheet("")

    def isChecked(self):
        return self.checked

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.setCheckState(not self.checked)
            self.checkStateChanged.emit(self.checked)

class ShowPageListChecker(QCheckBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class StateChecker(QCheckBox):
    checked = pyqtSignal(str)
    def __init__(self, checker_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checker_type = checker_type
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            if not self.isChecked():
                self.setChecked(True)
    def setChecked(self, check: bool) -> None:
        super().setChecked(check)
        if check:
            self.checked.emit(self.checker_type)

class PageListView(QListWidget):    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFixedWidth(512)
        self.setIconSize(QSize(70, 70))

class OpenBtn(QToolButton):
    def __init__(self, btn_width, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class RunBtn(QPushButton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText('Run')

class LeftBar(Widget):
    recent_proj_list = []
    imgTransChecked = pyqtSignal()
    configChecked = pyqtSignal()
    open_dir = pyqtSignal(str)
    save_proj = pyqtSignal()
    run_imgtrans = pyqtSignal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        bar_width = 90
        btn_width = 56
        padding = (bar_width - btn_width) // 2
        self.setFixedWidth(bar_width)
        self.showPageListLabel = ShowPageListChecker()

        self.imgTransChecker = StateChecker('imgtrans')
        self.imgTransChecker.setObjectName('ImgTransChecker')
        self.imgTransChecker.checked.connect(self.stateCheckerChanged)
        
        self.configChecker = StateChecker('config')
        self.configChecker.setObjectName('ConfigChecker')
        self.configChecker.checked.connect(self.stateCheckerChanged)

        actionOpenFolder = QAction(self)
        actionOpenFolder.setText(self.tr("Open Folder ..."))
        actionOpenFolder.triggered.connect(self.onOpenFolder)
        actionOpenFolder.setShortcut(QKeySequence.Open)

        actionOpenProj = QAction(self)
        actionOpenProj.setText(self.tr("Open Project ... *.json"))
        actionOpenProj.triggered.connect(self.onOpenProj)

        actionSaveProj = QAction(self)
        actionSaveProj.setText(self.tr("Save Project"))
        actionSaveProj.triggered.connect(self.onSaveProj)
        actionSaveProj.setShortcut(QKeySequence.StandardKey.Save)

        actionExportAsDoc = QAction(self)
        actionExportAsDoc.setText(self.tr("Export as Doc"))
        actionExportAsDoc.triggered.connect(self.onExportAsDoc)
        actionImportFromDoc = QAction(self)
        actionImportFromDoc.setText(self.tr("Import from Doc"))
        actionImportFromDoc.triggered.connect(self.onImportFromDoc)

        self.recentMenu = QMenu(self.tr("Open Recent"), self)
        
        openMenu = QMenu(self)
        openMenu.addActions([actionOpenFolder, actionOpenProj])
        openMenu.addMenu(self.recentMenu)
        openMenu.addSeparator()
        openMenu.addActions([
            actionSaveProj,
            actionExportAsDoc,
            actionImportFromDoc
        ])
        self.openBtn = OpenBtn(btn_width)
        self.openBtn.setFixedSize(btn_width, btn_width)
        self.openBtn.setMenu(openMenu)
        self.openBtn.setPopupMode(QToolButton.InstantPopup)
    
        openBtnToolBar = QToolBar(self)
        openBtnToolBar.setFixedSize(btn_width, btn_width)
        openBtnToolBar.addWidget(self.openBtn)
        
        self.runImgtransBtn = RunBtn()
        self.runImgtransBtn.setFixedSize(btn_width, btn_width)
        self.runImgtransBtn.clicked.connect(self.run_imgtrans)

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(openBtnToolBar)
        vlayout.addWidget(self.showPageListLabel)
        vlayout.addWidget(self.imgTransChecker)
        vlayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        vlayout.addWidget(self.configChecker)
        vlayout.addWidget(self.runImgtransBtn)
        vlayout.setContentsMargins(padding, 0, padding, btn_width/2)
        vlayout.setAlignment(Qt.AlignCenter)
        vlayout.setSpacing(btn_width/2)
        self.setGeometry(0, 0, 300, 500)

    def updateRecentProjList(self, proj_list: List[str]):
        if len(proj_list) == 0:
            return
        if isinstance(proj_list, str):
            proj_list = [proj_list]
        proj_list = list(OrderedDict.fromkeys(proj_list))

        actionlist = self.recentMenu.actions()
        if len(self.recent_proj_list) == 0:
            self.recent_proj_list.append(proj_list.pop())
            topAction = QAction(self.recent_proj_list[-1], self)
            topAction.triggered.connect(self.recentActionTriggered)
            self.recentMenu.addAction(topAction)
        else:
            topAction = actionlist[0]
        for proj in proj_list[::-1]:
            try:    # remove duplicated
                idx = self.recent_proj_list.index(proj)
                if idx == 0:
                    continue
                del self.recent_proj_list[idx]
                self.recentMenu.removeAction(self.recentMenu.actions()[idx])
                if len(self.recent_proj_list) == 0:
                    topAction = QAction(proj, self)
                    self.recentMenu.addAction(topAction)
                    topAction.triggered.connect(self.recentActionTriggered)
                    continue
            except ValueError:
                pass
            newTop = QAction(proj, self)
            self.recentMenu.insertAction(topAction, newTop)
            newTop.triggered.connect(self.recentActionTriggered)
            self.recent_proj_list.insert(0, proj)
            topAction = newTop

        MAXIUM_RECENT_PROJ_NUM = 5
        actionlist = self.recentMenu.actions()
        num_to_remove = len(actionlist) - MAXIUM_RECENT_PROJ_NUM
        if num_to_remove > 0:
            actions_to_remove = actionlist[-num_to_remove:]
            for action in actions_to_remove:
                self.recentMenu.removeAction(action)
                self.recent_proj_list.pop()


    def recentActionTriggered(self):
        path = self.sender().text()
        if osp.exists(path):
            self.updateRecentProjList(path)
            self.open_dir.emit(path)
        else:
            self.recent_proj_list.remove(path)
            self.recentMenu.removeAction(self.sender())
        
    def onOpenFolder(self) -> None:
        # newdir = str(QFileDialog.getExistingDirectory(self, "Select Directory")).replace("/", "\\")
        dialog = QFileDialog()
        dialog.setDefaultSuffix('.jpg')
        folder_path = str(dialog.getExistingDirectory(self, "Select Directory"))
        if osp.exists(folder_path):
            self.open_dir.emit(folder_path)
            self.updateRecentProjList(folder_path)
        # self.open_dir.emit(folder_path)

    def onOpenProj(self):
        self.open_dir.emit()

    def onSaveProj(self):
        self.save_proj.emit()
        pass

    def onExportAsDoc(self):
        raise NotImplementedError

    def onImportFromDoc(self):
        raise NotImplementedError

    def stateCheckerChanged(self, checker_type: str):
        if checker_type == 'imgtrans':
            self.configChecker.setChecked(False)
            self.imgTransChecked.emit()
        elif checker_type == 'config':
            self.imgTransChecker.setChecked(False)
            self.configChecked.emit()


class TitleBar(Widget):

    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.mainwindow : QMainWindow = parent
        self.mPos = None
        self.normalsize = False
        self.proj_name = ''
        self.page_name = ''
        self.save_state = ''
        self.setFixedHeight(40)

        self.titleLabel = QLabel('BallonTranslator')
        self.titleLabel.setObjectName('TitleLabel')
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.minBtn = QPushButton()
        self.minBtn.setObjectName('minBtn')
        self.minBtn.clicked.connect(self.onMinBtnClicked)
        self.maxBtn = QCheckBox()
        self.maxBtn.setObjectName('maxBtn')
        self.maxBtn.clicked.connect(self.onMaxBtnClicked)
        self.closeBtn = QPushButton()
        self.closeBtn.setObjectName('closeBtn')
        self.closeBtn.clicked.connect(self.onCloseBtnClicked)
        self.maxBtn.setFixedSize(72, 40)
        hlayout = QHBoxLayout(self)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hlayout.addItem(QSpacerItem(0, 0,  QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        hlayout.addWidget(self.titleLabel)
        hlayout.addItem(QSpacerItem(0, 0,  QSizePolicy.Policy.Expanding,  QSizePolicy.Policy.Expanding))
        hlayout.addWidget(self.minBtn)
        hlayout.addWidget(self.maxBtn)
        hlayout.addWidget(self.closeBtn)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)        

    def onMaxBtnClicked(self):
        if self.mainwindow.isMaximized():
            self.mainwindow.showNormal()
            self.mainwindow.updateGeometry()
        else:
            self.mainwindow.showMaximized()
            self.mainwindow.updateGeometry()

    def onMinBtnClicked(self):
        self.mainwindow.showMinimized()

    def onCloseBtnClicked(self):
        self.mainwindow.close()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mPos = event.pos()
            self.mPosGlobal = event.globalPos()
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mPos = None
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mPos is not None:
            self.mainwindow.show()
            if self.mainwindow.isMaximized():
                oldw = self.mainwindow.width()
                newgeo = self.mainwindow.normalGeometry()
                self.mainwindow.showNormal()
                
                if self.mPos.x() > newgeo.width():
                    self.mPos = QPoint(newgeo.width()-oldw+self.mPos.x(), self.mPos.y())
                else:
                    self.mainwindow.move(event.globalPos() - self.mPos)
            else:
                self.mainwindow.move(event.globalPos()-self.mPos)

    def setTitleContent(self, proj_name: str = None, page_name: str = None, save_state: str = None):
        max_proj_len = 50
        max_page_len = 50
        if proj_name is not None:
            if len(proj_name) > max_proj_len:
                proj_name = proj_name[:max_proj_len-3] + '...'
            self.proj_name = proj_name
        if page_name is not None:
            if len(page_name) > max_page_len:
                page_name = page_name[:max_page_len-3] + '...'
            self.page_name = page_name
        if save_state is not None:
            self.save_state = save_state
        title = self.proj_name + ' - ' + self.page_name
        if self.save_state != '':
            title += ' - '  + self.save_state
        self.titleLabel.setText(title)


class BottomBar(Widget):
    textedit_checkchanged = pyqtSignal()
    paintmode_checkchanged = pyqtSignal()
    textblock_checkchanged = pyqtSignal()
    ocrcheck_statechanged = pyqtSignal(bool)
    transcheck_statechanged = pyqtSignal(bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setFixedHeight(48)
        self.ocrChecker = TextChecker('ocr')
        self.ocrChecker.setObjectName('OCRChecker')
        self.ocrChecker.setToolTip(self.tr('Enable/disable ocr'))
        self.ocrChecker.checkStateChanged.connect(self.OCRStateChanged)
        self.transChecker = QCheckBox()
        self.transChecker.setObjectName('TransChecker')
        self.transChecker.setToolTip(self.tr('Enable/disable translation'))
        self.transChecker.clicked.connect(self.transCheckerStateChanged)
        self.translatorStatusbtn = TranslatorStatusButton()
        self.translatorStatusbtn.setHidden(True)
        self.transTranspageBtn = RunStopTextBtn(self.tr('translate page'),
                                                self.tr('stop'),
                                                self.tr('translate current page'),
                                                self.tr('stop translation'))
        self.inpainterStatBtn = InpainterStatusButton()
        self.transTranspageBtn.hide()
        self.hlayout = QHBoxLayout(self)
        self.paintChecker = QCheckBox()
        self.paintChecker.setObjectName('PaintChecker')
        self.paintChecker.setToolTip(self.tr('Enable/disable paint mode'))
        self.paintChecker.clicked.connect(self.onPaintCheckerPressed)
        self.texteditChecker = QCheckBox()
        self.texteditChecker.setObjectName('TexteditChecker')
        self.texteditChecker.setToolTip(self.tr('Enable/disable text edit mode'))
        self.texteditChecker.clicked.connect(self.onTextEditCheckerPressed)
        self.textblockChecker = QCheckBox()
        self.textblockChecker.setObjectName('TextblockChecker')
        self.textblockChecker.clicked.connect(self.onTextblockCheckerClicked)
        
        self.originalSlider = PaintQSlider(self.tr("Original image transparency: ") + "value%", Qt.Horizontal, self, minimumWidth=90)
        self.originalSlider.setFixedHeight(40)
        self.originalSlider.setFixedWidth(200)
        self.originalSlider.setRange(0, 100)
        
        self.hlayout.addWidget(self.ocrChecker)
        self.hlayout.addWidget(self.transChecker)
        self.hlayout.addWidget(self.translatorStatusbtn)
        self.hlayout.addWidget(self.transTranspageBtn)
        self.hlayout.addWidget(self.inpainterStatBtn)
        self.hlayout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.hlayout.addWidget(self.originalSlider)
        self.hlayout.addWidget(self.paintChecker)
        self.hlayout.addWidget(self.texteditChecker)
        self.hlayout.addWidget(self.textblockChecker)
        self.hlayout.setContentsMargins(90, 0, 15, 0)


    def onPaintCheckerPressed(self):
        if self.paintChecker.isChecked():
            self.texteditChecker.setChecked(False)
        self.paintmode_checkchanged.emit()

    def onTextEditCheckerPressed(self):
        if self.texteditChecker.isChecked():
            self.paintChecker.setChecked(False)
        self.textedit_checkchanged.emit()

    def onTextblockCheckerClicked(self):
        self.textblock_checkchanged.emit()

    def OCRStateChanged(self):
        self.ocrcheck_statechanged.emit(self.ocrChecker.isChecked())
        
    def transCheckerStateChanged(self):
        self.transcheck_statechanged.emit(self.transChecker.isChecked())
    

class FrameLessMainWindow(QMainWindow):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

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

class MainWindow(FrameLessMainWindow):
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

        self.leftBar = LeftBar()
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
        
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.pageList)
        mainHLayout.addWidget(self.centralStackWidget)
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
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)
        self.comicTransSplitter.setStretchFactor(0, 10)
        self.comicTransSplitter.setStretchFactor(1, 1)

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
            # yaml.safe_dump(config_dict, f)
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
        self.logger.info('saving ' + self.imgtrans_proj.current_img)
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        if save_proj:
            self.imgtrans_proj.save(save_mask=True, save_inpainted=True)
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
        img.save(self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img))

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