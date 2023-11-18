import os.path as osp
from typing import List, Union

from qtpy.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel, QSizePolicy, QToolBar, QMenu, QSpacerItem, QPushButton, QCheckBox, QToolButton
from qtpy.QtCore import Qt, Signal, QPoint
from qtpy.QtGui import QMouseEvent, QKeySequence, QActionGroup

from .stylewidgets import Widget, PaintQSlider, TextChecker
from utils.shared import TITLEBAR_HEIGHT, WINDOW_BORDER_WIDTH, BOTTOMBAR_HEIGHT, LEFTBAR_WIDTH, LEFTBTN_WIDTH
from .framelesswindow import startSystemMove
from utils.config import pcfg
from utils import shared as C
if C.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction

class ShowPageListChecker(QCheckBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class OpenBtn(QToolButton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class RunBtn(QPushButton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText('Run')


class StatusButton(QPushButton):
    pass


class TitleBarToolBtn(QToolButton):
    pass


class RunStopTextBtn(StatusButton):
    run_target = Signal(bool)
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
        if self.running:
            self.setStopText()
        else:
            self.setRunText()
        self.run_target.emit(self.running)

    def setRunText(self):
        self.setText(self.run_text)
        if self.run_tool_tip is not None:
            self.setToolTip(self.run_tool_tip)
        self.running = False

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


class StateChecker(QCheckBox):
    checked = Signal(str)
    def __init__(self, checker_type: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checker_type = checker_type
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.isChecked():
                self.setChecked(True)
    def setChecked(self, check: bool) -> None:
        super().setChecked(check)
        if check:
            self.checked.emit(self.checker_type)

class LeftBar(Widget):
    recent_proj_list = []
    imgTransChecked = Signal()
    configChecked = Signal()
    open_dir = Signal(str)
    open_json_proj = Signal(str)
    save_proj = Signal()
    save_config = Signal()
    run_imgtrans = Signal()
    export_doc = Signal()
    import_doc = Signal()
    def __init__(self, mainwindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.mainwindow: QMainWindow = mainwindow

        padding = (LEFTBAR_WIDTH - LEFTBTN_WIDTH) // 2
        self.setFixedWidth(LEFTBAR_WIDTH)
        self.showPageListLabel = ShowPageListChecker()

        self.globalSearchChecker = QCheckBox()
        self.globalSearchChecker.setObjectName('GlobalSearchChecker')
        self.globalSearchChecker.setToolTip(self.tr('Global Search (Ctrl+G)'))

        self.imgTransChecker = StateChecker('imgtrans')
        self.imgTransChecker.setObjectName('ImgTransChecker')
        self.imgTransChecker.checked.connect(self.stateCheckerChanged)
        
        self.configChecker = StateChecker('config')
        self.configChecker.setObjectName('ConfigChecker')
        self.configChecker.checked.connect(self.stateCheckerChanged)

        actionOpenFolder = QAction(self.tr("Open Folder ..."), self)
        actionOpenFolder.triggered.connect(self.onOpenFolder)
        actionOpenFolder.setShortcut(QKeySequence.Open)

        actionOpenProj = QAction(self.tr("Open Project ... *.json"), self)
        actionOpenProj.triggered.connect(self.onOpenProj)

        actionSaveProj = QAction(self.tr("Save Project"), self)
        self.save_proj = actionSaveProj.triggered
        actionSaveProj.setShortcut(QKeySequence.StandardKey.Save)

        actionExportAsDoc = QAction(self.tr("Export as Doc"), self)
        actionExportAsDoc.triggered.connect(self.export_doc)
        actionImportFromDoc = QAction(self.tr("Import from Doc"), self)
        actionImportFromDoc.triggered.connect(self.import_doc)

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
        self.openBtn = OpenBtn()
        self.openBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        self.openBtn.setMenu(openMenu)
        self.openBtn.setPopupMode(QToolButton.InstantPopup)
    
        openBtnToolBar = QToolBar(self)
        openBtnToolBar.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        openBtnToolBar.addWidget(self.openBtn)
        
        self.runImgtransBtn = RunBtn()
        self.runImgtransBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        self.runImgtransBtn.clicked.connect(self.run_imgtrans)

        self.runImgtransBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)

        vlayout = QVBoxLayout(self)
        vlayout.addWidget(openBtnToolBar)
        vlayout.addWidget(self.showPageListLabel)
        vlayout.addWidget(self.globalSearchChecker)
        vlayout.addWidget(self.imgTransChecker)
        vlayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        vlayout.addWidget(self.configChecker)
        vlayout.addWidget(self.runImgtransBtn)
        vlayout.setContentsMargins(padding, 0, padding, int(LEFTBTN_WIDTH / 2))
        vlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.setSpacing(int(LEFTBTN_WIDTH / 2))
        self.setGeometry(0, 0, 300, 500)
        self.setMouseTracking(True)

    def initRecentProjMenu(self, proj_list: List[str]):
        self.recent_proj_list = proj_list
        for proj in proj_list:
            action = QAction(proj, self)
            self.recentMenu.addAction(action)
            action.triggered.connect(self.recentActionTriggered)

    def updateRecentProjList(self, proj_list: Union[str, List[str]]):
        if len(proj_list) == 0:
            return
        if isinstance(proj_list, str):
            proj_list = [proj_list]
        if self.recent_proj_list == proj_list:
            return

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

        MAXIUM_RECENT_PROJ_NUM = 14
        actionlist = self.recentMenu.actions()
        num_to_remove = len(actionlist) - MAXIUM_RECENT_PROJ_NUM
        if num_to_remove > 0:
            actions_to_remove = actionlist[-num_to_remove:]
            for action in actions_to_remove:
                self.recentMenu.removeAction(action)
                self.recent_proj_list.pop()

        self.save_config.emit()

    def recentActionTriggered(self):
        path = self.sender().text()
        if osp.exists(path):
            self.updateRecentProjList(path)
            self.open_dir.emit(path)
        else:
            self.recent_proj_list.remove(path)
            self.recentMenu.removeAction(self.sender())
        
    def onOpenFolder(self) -> None:
        
        d = None
        if len(self.recent_proj_list) > 0:
            for projp in self.recent_proj_list:
                if not osp.isdir(projp):
                    projp = osp.dirname(projp)
                if osp.exists(projp):
                    d = projp
                    break
        
        dialog = QFileDialog()
        folder_path = str(dialog.getExistingDirectory(self, self.tr("Select Directory"), d))
        if osp.exists(folder_path):
            self.updateRecentProjList(folder_path)
            self.open_dir.emit(folder_path)

    def onOpenProj(self):
        dialog = QFileDialog()
        json_path = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.docx'), filter="*.json")[0].toLocalFile())
        if osp.exists(json_path):
            self.open_json_proj.emit(json_path)

    def stateCheckerChanged(self, checker_type: str):
        if checker_type == 'imgtrans':
            self.configChecker.setChecked(False)
            self.imgTransChecked.emit()
        elif checker_type == 'config':
            self.imgTransChecker.setChecked(False)
            self.configChecked.emit()

    def needleftStackWidget(self) -> bool:
        return self.showPageListLabel.isChecked() or self.globalSearchChecker.isChecked()


class TitleBar(Widget):

    closebtn_clicked = Signal()
    display_lang_changed = Signal(str)

    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        self.mainwindow : QMainWindow = parent
        self.mPos: QPoint = None
        self.normalsize = False
        self.proj_name = ''
        self.page_name = ''
        self.save_state = ''
        self.setFixedHeight(TITLEBAR_HEIGHT)
        self.setMouseTracking(True)

        self.editToolBtn = TitleBarToolBtn(self)
        self.editToolBtn.setText(self.tr('Edit'))

        undoAction = QAction(self.tr('Undo'), self)
        self.undo_trigger = undoAction.triggered
        undoAction.setShortcut(QKeySequence.StandardKey.Undo)
        redoAction = QAction(self.tr('Redo'), self)
        self.redo_trigger = redoAction.triggered
        redoAction.setShortcut(QKeySequence.StandardKey.Redo)
        pageSearchAction = QAction(self.tr('Search'), self)
        self.page_search_trigger = pageSearchAction.triggered
        pageSearchAction.setShortcut(QKeySequence('Ctrl+F'))
        globalSearchAction = QAction(self.tr('Global Search'), self)
        self.global_search_trigger = globalSearchAction.triggered
        globalSearchAction.setShortcut(QKeySequence('Ctrl+G'))

        replaceMTkeyword = QAction(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeyword_trigger = replaceMTkeyword.triggered
        replaceOCRkeyword = QAction(self.tr("Keyword substitution for OCR results"), self)
        self.replaceOCRkeyword_trigger = replaceOCRkeyword.triggered

        editMenu = QMenu(self.editToolBtn)
        editMenu.addActions([undoAction, redoAction])
        editMenu.addSeparator()
        editMenu.addActions([pageSearchAction, globalSearchAction, replaceOCRkeyword, replaceMTkeyword])
        self.editToolBtn.setMenu(editMenu)
        self.editToolBtn.setPopupMode(QToolButton.InstantPopup)

        self.viewToolBtn = TitleBarToolBtn(self)
        self.viewToolBtn.setText(self.tr('View'))

        self.displayLanguageMenu = QMenu(self.tr("Display Language"), self)
        self.lang_ac_group = lang_ac_group = QActionGroup(self)
        lang_ac_group.setExclusive(True)
        lang_actions = []
        for lang, lang_code in C.DISPLAY_LANGUAGE_MAP.items():
            la = QAction(lang, self)
            if lang_code == pcfg.display_lang:
                la.setChecked(True)
            la.triggered.connect(self.on_displaylang_triggered)
            la.setCheckable(True)
            lang_ac_group.addAction(la)
            lang_actions.append(la)
        self.displayLanguageMenu.addActions(lang_actions)

        drawBoardAction = QAction(self.tr('Drawing Board'), self)
        drawBoardAction.setShortcut(QKeySequence('P'))
        texteditAction = QAction(self.tr('Text Editor'), self)
        texteditAction.setShortcut(QKeySequence('T'))
        fontStylePresetAction = QAction(self.tr('Text Style Presets'), self)
        self.darkModeAction = darkModeAction = QAction(self.tr('Dark Mode'), self)
        darkModeAction.setCheckable(True)

        viewMenu = QMenu(self.viewToolBtn)
        viewMenu.addMenu(self.displayLanguageMenu)
        viewMenu.addActions([drawBoardAction, texteditAction])
        viewMenu.addSeparator()
        viewMenu.addAction(fontStylePresetAction)
        viewMenu.addSeparator()
        viewMenu.addAction(darkModeAction)
        self.viewToolBtn.setMenu(viewMenu)
        self.viewToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.textedit_trigger = texteditAction.triggered
        self.drawboard_trigger = drawBoardAction.triggered
        self.fontstyle_trigger = fontStylePresetAction.triggered
        self.darkmode_trigger = darkModeAction.triggered

        self.goToolBtn = TitleBarToolBtn(self)
        self.goToolBtn.setText(self.tr('Go'))
        prevPageAction = QAction(self.tr('Previous Page'), self)
        prevPageAction.setShortcuts([QKeySequence.StandardKey.MoveToPreviousPage, QKeySequence('A')])
        nextPageAction = QAction(self.tr('Next Page'), self)
        nextPageAction.setShortcuts([QKeySequence.StandardKey.MoveToNextPage, QKeySequence('D')])
        goMenu = QMenu(self.goToolBtn)
        goMenu.addActions([prevPageAction, nextPageAction])
        self.goToolBtn.setMenu(goMenu)
        self.goToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.prevpage_trigger = prevPageAction.triggered
        self.nextpage_trigger = nextPageAction.triggered

        self.runToolBtn = TitleBarToolBtn(self)
        self.runToolBtn.setText(self.tr('Run'))

        self.stageActions = stageActions = [
            QAction(self.tr('Enable Text Dection'), self),
            QAction(self.tr('Enable OCR'), self),
            QAction(self.tr('Enable Translation'), self),
            QAction(self.tr('Enable Inpainting'), self)
        ]
        for idx, sa in enumerate(stageActions):
            sa.setCheckable(True)
            sa.setChecked(pcfg.module.stage_enabled(idx))
            sa.triggered.connect(self.stageEnableStateChanged)

        runAction = QAction(self.tr('Run'), self)
        translatePageAction = QAction(self.tr('Translate page'), self)
        runMenu = QMenu(self.runToolBtn)
        runMenu.addActions(stageActions)
        runMenu.addSeparator()
        runMenu.addActions([runAction, translatePageAction])
        self.runToolBtn.setMenu(runMenu)
        self.runToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.run_trigger = runAction.triggered
        self.translate_page_trigger = translatePageAction.triggered

        self.iconLabel = QLabel(self)
        if not C.ON_MACOS:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH - 12)
        else:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH)

        self.titleLabel = QLabel('BallonTranslator')
        self.titleLabel.setObjectName('TitleLabel')
        self.titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        hlayout = QHBoxLayout(self)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hlayout.addWidget(self.iconLabel)
        hlayout.addWidget(self.editToolBtn)
        hlayout.addWidget(self.viewToolBtn)
        hlayout.addWidget(self.goToolBtn)
        hlayout.addWidget(self.runToolBtn)
        hlayout.addStretch()
        hlayout.addWidget(self.titleLabel)
        hlayout.addStretch()

        if not C.ON_MACOS:
            self.minBtn = QPushButton()
            self.minBtn.setObjectName('minBtn')
            self.minBtn.clicked.connect(self.onMinBtnClicked)
            self.maxBtn = QCheckBox()
            self.maxBtn.setObjectName('maxBtn')
            self.maxBtn.clicked.connect(self.onMaxBtnClicked)
            self.maxBtn.setFixedSize(48, 27)
            self.closeBtn = QPushButton()
            self.closeBtn.setObjectName('closeBtn')
            self.closeBtn.clicked.connect(self.closebtn_clicked)
            hlayout.addWidget(self.minBtn)
            hlayout.addWidget(self.maxBtn)
            hlayout.addWidget(self.closeBtn)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.setSpacing(0)

    def stageEnableStateChanged(self):
        sender = self.sender()
        idx= self.stageActions.index(sender)
        checked = sender.isChecked()
        if idx == 0:
            pcfg.module.enable_detect = checked
        elif idx == 1:
            pcfg.module.enable_ocr = checked
        elif idx == 2:
            pcfg.module.enable_translate = checked
        elif idx == 3:
            pcfg.module.enable_inpaint = checked

    def onMaxBtnClicked(self):
        if self.mainwindow.isMaximized():
            self.mainwindow.showNormal()
        else:
            self.mainwindow.showMaximized()

    def onMinBtnClicked(self):
        self.mainwindow.showMinimized()

    def on_displaylang_triggered(self):
        ac = self.lang_ac_group.checkedAction()
        self.display_lang_changed.emit(C.DISPLAY_LANGUAGE_MAP[ac.text()])

    def mousePressEvent(self, event: QMouseEvent) -> None:

        if C.FLAG_QT6:
            g_pos = event.globalPosition().toPoint()
        else:
            g_pos = event.globalPos()
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.mainwindow.isMaximized() and \
                event.pos().y() < WINDOW_BORDER_WIDTH:
                pass
            else:
                self.mPos = event.pos()
                self.mPosGlobal = g_pos
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mPos = None
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.mPos is not None:
            if C.FLAG_QT6:
                g_pos = event.globalPosition().toPoint()
            else:
                g_pos = event.globalPos()
            startSystemMove(self.window(), g_pos)

    def hideEvent(self, e) -> None:
        self.mPos = None
        return super().hideEvent(e)

    def leaveEvent(self, e) -> None:
        self.mPos = None
        return super().leaveEvent(e)

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
    
    textedit_checkchanged = Signal()
    paintmode_checkchanged = Signal()
    textblock_checkchanged = Signal()
    inpaint_btn_clicked = Signal()

    def __init__(self, mainwindow: QMainWindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.setFixedHeight(BOTTOMBAR_HEIGHT)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow

        self.translatorStatusbtn = TranslatorStatusButton()
        self.translatorStatusbtn.setHidden(True)
        self.transTranspageBtn = RunStopTextBtn(self.tr('translate page'),
                                                self.tr('stop'),
                                                self.tr('translate current page'),
                                                self.tr('stop translation'))
        self.inpainterStatBtn = InpainterStatusButton()
        self.inpainterStatBtn.clicked.connect(self.inpaintBtnClicked)
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
        
        self.originalSlider = PaintQSlider(self.tr("Original image transparency"), Qt.Orientation.Horizontal, self, minimumWidth=90)
        self.originalSlider.setFixedWidth(150)
        self.originalSlider.setRange(0, 100)

        self.textlayerSlider = PaintQSlider(self.tr("Lettering layer transparency"), Qt.Orientation.Horizontal, self, minimumWidth=90)
        self.textlayerSlider.setFixedWidth(150)
        self.textlayerSlider.setValue(100)
        self.textlayerSlider.setRange(0, 100)
        
        self.hlayout.addWidget(self.translatorStatusbtn)
        self.hlayout.addWidget(self.transTranspageBtn)
        self.hlayout.addWidget(self.inpainterStatBtn)
        self.hlayout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.hlayout.addWidget(self.textlayerSlider)
        self.hlayout.addWidget(self.originalSlider)
        self.hlayout.addWidget(self.paintChecker)
        self.hlayout.addWidget(self.texteditChecker)
        self.hlayout.addWidget(self.textblockChecker)
        self.hlayout.setContentsMargins(60, 0, 10, WINDOW_BORDER_WIDTH)


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

    def inpaintBtnClicked(self):
        self.inpaint_btn_clicked.emit()