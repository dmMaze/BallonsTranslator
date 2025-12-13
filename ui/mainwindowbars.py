import os.path as osp
from typing import List, Union

from qtpy.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel, QSizePolicy, QToolBar, QMenu, QSpacerItem, QPushButton, QCheckBox, QToolButton
from qtpy.QtCore import Qt, Signal, QPoint, QEvent, QSize
from qtpy.QtGui import QMouseEvent, QKeySequence, QActionGroup, QIcon

from modules.translators import BaseTranslator
from .custom_widget import Widget, PaintQSlider, SmallComboBox, ConfigClickableLabel
from utils.shared import TITLEBAR_HEIGHT, WINDOW_BORDER_WIDTH, BOTTOMBAR_HEIGHT, LEFTBAR_WIDTH, LEFTBTN_WIDTH
from .framelesswindow import FramelessMoveResize
from utils.config import pcfg
from utils import shared as C
if C.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction

class ShowPageListChecker(QCheckBox):
    ...


class OpenBtn(QToolButton):
    ...


class StatusButton(QPushButton):
    pass


class TitleBarToolBtn(QToolButton):
    pass


class StateChecker(QCheckBox):
    checked = Signal(str)
    unchecked = Signal(str)
    def __init__(self, checker_type: str, uncheckable: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checker_type = checker_type
        self.uncheckable = uncheckable

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.isChecked():
                self.setChecked(True)
            elif self.uncheckable:
                self.setChecked(False)
                
    def setChecked(self, check: bool) -> None:
        check_state = self.isChecked()
        super().setChecked(check)
        if check_state != check:
            if check:
                self.checked.emit(self.checker_type)
            else:
                self.unchecked.emit(self.checker_type)

class LeftBar(Widget):
    recent_proj_list = []
    imgTransChecked = Signal()
    configChecked = Signal()
    open_dir = Signal(str)
    open_json_proj = Signal(str)
    save_proj = Signal()
    save_config = Signal()
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
        
        self.configChecker = StateChecker('config', uncheckable=True)
        self.configChecker.setObjectName('ConfigChecker')
        self.configChecker.checked.connect(self.stateCheckerChanged)
        self.configChecker.unchecked.connect(self.stateCheckerChanged)

        actionOpenFolder = QAction(self.tr("Open Folder ..."), self)
        actionOpenFolder.triggered.connect(self.onOpenFolder)
        actionOpenFolder.setShortcut(QKeySequence.Open)

        actionOpenProj = QAction(self.tr("Open Project ... *.json"), self)
        actionOpenProj.triggered.connect(self.onOpenProj)

        actionSaveProj = QAction(self.tr("Save Project"), self)
        self.save_proj = actionSaveProj.triggered
        actionSaveProj.setShortcut(QKeySequence.StandardKey.Save)

        actionExportAsDoc = QAction(self.tr("Export as Doc"), self)
        self.export_doc = actionExportAsDoc.triggered
        actionImportFromDoc = QAction(self.tr("Import from Doc"), self)
        self.import_doc = actionImportFromDoc.triggered

        actionExportSrcTxt = QAction(self.tr("Export source text as TXT"), self)
        self.export_src_txt = actionExportSrcTxt.triggered
        actionExportTranslationTxt = QAction(self.tr("Export translation as TXT"), self)
        self.export_trans_txt = actionExportTranslationTxt.triggered

        actionExportSrcMD = QAction(self.tr("Export source text as markdown"), self)
        self.export_src_md = actionExportSrcMD.triggered
        actionExportTranslationMD = QAction(self.tr("Export translation as markdown"), self)
        self.export_trans_md = actionExportTranslationMD.triggered

        actionImportTranslationTxt = QAction(self.tr("Import translation from TXT/markdown"), self)
        self.import_trans_txt = actionImportTranslationTxt.triggered

        self.recentMenu = QMenu(self.tr("Open Recent"), self)
        
        openMenu = QMenu(self)
        openMenu.addActions([actionOpenFolder, actionOpenProj])
        openMenu.addMenu(self.recentMenu)
        openMenu.addSeparator()
        openMenu.addActions([
            actionSaveProj,
            actionExportAsDoc,
            actionImportFromDoc,
            actionExportSrcTxt,
            actionExportTranslationTxt,
            actionExportSrcMD,
            actionExportTranslationMD,
            actionImportTranslationTxt,
        ])
        self.openBtn = OpenBtn()
        self.openBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        self.openBtn.setMenu(openMenu)
        self.openBtn.setPopupMode(QToolButton.InstantPopup)
    
        openBtnToolBar = QToolBar(self)
        openBtnToolBar.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        openBtnToolBar.addWidget(self.openBtn)
        
        self.runImgtransBtn = QPushButton()
        self.runImgtransBtn.setObjectName('RunButton')
        self.runImgtransBtn.setText(self.tr('Run'))
        font = self.runImgtransBtn.font()
        font.setPixelSize(10)
        self.runImgtransBtn.setFont(font)
        self.runImgtransBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        self.run_imgtrans_clicked = self.runImgtransBtn.clicked
        self.runImgtransBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)
        
        vlayout = QVBoxLayout(self)
        vlayout.addWidget(openBtnToolBar)
        vlayout.addWidget(self.showPageListLabel)
        vlayout.addWidget(self.globalSearchChecker)
        vlayout.addWidget(self.imgTransChecker)
        vlayout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
        vlayout.addWidget(self.configChecker)
        vlayout.addWidget(self.runImgtransBtn)
        vlayout.setContentsMargins(padding, LEFTBTN_WIDTH // 2, padding, LEFTBTN_WIDTH // 2)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.setSpacing(LEFTBTN_WIDTH * 3 // 4)
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
            if self.configChecker.isChecked():
                self.imgTransChecker.setChecked(False)
                self.configChecked.emit()
            else:
                self.imgTransChecker.setChecked(True)
                

    def needleftStackWidget(self) -> bool:
        return self.showPageListLabel.isChecked() or self.globalSearchChecker.isChecked()


class TitleBar(Widget):

    closebtn_clicked = Signal()
    display_lang_changed = Signal(str)
    enable_module = Signal(int, bool)

    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.mainwindow : QMainWindow = parent
        self.mainwindow.installEventFilter(self)
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

        replacePreMTkeyword = QAction(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeyword_trigger = replacePreMTkeyword.triggered
        replaceMTkeyword = QAction(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeyword_trigger = replaceMTkeyword.triggered
        replaceOCRkeyword = QAction(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeyword_trigger = replaceOCRkeyword.triggered

        editMenu = QMenu(self.editToolBtn)
        editMenu.addActions([undoAction, redoAction])
        editMenu.addSeparator()
        editMenu.addActions([pageSearchAction, globalSearchAction, replaceOCRkeyword, replacePreMTkeyword, replaceMTkeyword])
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
        importTextStyles = QAction(self.tr('Import Text Styles'), self)
        exportTextStyles = QAction(self.tr('Export Text Styles'), self)
        self.darkModeAction = darkModeAction = QAction(self.tr('Dark Mode'), self)
        darkModeAction.setCheckable(True)

        self.viewMenu = viewMenu = QMenu(self.viewToolBtn)
        viewMenu.addMenu(self.displayLanguageMenu)
        viewMenu.addActions([drawBoardAction, texteditAction])
        viewMenu.addSeparator()
        viewMenu.addAction(importTextStyles)
        viewMenu.addAction(exportTextStyles)
        viewMenu.addSeparator()
        viewMenu.addAction(darkModeAction)
        self.viewToolBtn.setMenu(viewMenu)
        self.viewToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.textedit_trigger = texteditAction.triggered
        self.drawboard_trigger = drawBoardAction.triggered
        self.importtstyle_trigger = importTextStyles.triggered
        self.exporttstyle_trigger = exportTextStyles.triggered
        self.darkmode_trigger = darkModeAction.triggered

        self.goToolBtn = TitleBarToolBtn(self)
        self.goToolBtn.setText(self.tr('Go'))
        prevPageAction = QAction(self.tr('Previous Page'), self)
        # prevPageAction.setShortcuts([QKeySequence.StandardKey.MoveToPreviousPage, QKeySequence('A')])
        nextPageAction = QAction(self.tr('Next Page'), self)
        # nextPageAction.setShortcuts([QKeySequence.StandardKey.MoveToNextPage, QKeySequence('D')])
        goMenu = QMenu(self.goToolBtn)
        goMenu.addActions([prevPageAction, nextPageAction])
        self.goToolBtn.setMenu(goMenu)
        self.goToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.prevpage_trigger = prevPageAction.triggered
        self.nextpage_trigger = nextPageAction.triggered

        # 工具菜单
        self.toolsToolBtn = TitleBarToolBtn(self)
        self.toolsToolBtn.setText(self.tr('Tools'))
        
        # 区域合并工具
        mergeToolAction = QAction('区域合并工具', self)
        mergeToolAction.setShortcut(QKeySequence('Ctrl+Shift+M'))
        self.merge_tool_trigger = mergeToolAction.triggered
        
        toolsMenu = QMenu(self.toolsToolBtn)
        toolsMenu.addAction(mergeToolAction)
        self.toolsToolBtn.setMenu(toolsMenu)
        self.toolsToolBtn.setPopupMode(QToolButton.InstantPopup)

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
        runWoUpdateTextStyle = QAction(self.tr('Run without update textstyle'), self)
        translatePageAction = QAction(self.tr('Translate page'), self)
        runMenu = QMenu(self.runToolBtn)
        runMenu.addActions(stageActions)
        runMenu.addSeparator()
        runMenu.addActions([runAction, runWoUpdateTextStyle, translatePageAction])
        self.runToolBtn.setMenu(runMenu)
        self.runToolBtn.setPopupMode(QToolButton.InstantPopup)
        self.run_trigger = runAction.triggered
        self.run_woupdate_textstyle_trigger = runWoUpdateTextStyle.triggered
        self.translate_page_trigger = translatePageAction.triggered

        self.iconLabel = QLabel(self)
        if not C.ON_MACOS:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH - 12)
        else:
            self.iconLabel.setFixedWidth(LEFTBAR_WIDTH + 8)

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
        hlayout.addWidget(self.toolsToolBtn)
        hlayout.addStretch()
        hlayout.addWidget(self.titleLabel)
        hlayout.addStretch()
        hlayout.setContentsMargins(0, 0, 0, 0)

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

    def eventFilter(self, obj, e):
        if obj == self.mainwindow:
            if e.type() == QEvent.Type.WindowStateChange and not C.ON_MACOS:
                self.maxBtn.setChecked(self.mainwindow.isMaximized())
                return False

        return super().eventFilter(obj, e)

    def stageEnableStateChanged(self):
        sender = self.sender()
        idx= self.stageActions.index(sender)
        checked = sender.isChecked()
        self.enable_module.emit(idx, checked)

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(e)
        FramelessMoveResize.toggleMaxState(self.mainwindow)

    def onMaxBtnClicked(self):
        FramelessMoveResize.toggleMaxState(self.mainwindow)

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
            FramelessMoveResize.startSystemMove(self.window(), g_pos)

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


class SmallConfigPutton(QPushButton):
    pass


CFG_ICON  = QIcon('icons/leftbar_config_activate.svg')


class SelectionWithConfigWidget(Widget):

    cfg_clicked = Signal()

    def __init__(self, selector_name: str, add_cfg_btn=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        label = ConfigClickableLabel(text=selector_name)
        label.clicked.connect(self.cfg_clicked)
        
        self.selector = SmallComboBox()

        self.cfg_btn = None
        if add_cfg_btn:
            self.cfg_btn = SmallConfigPutton()
            self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout2 = QHBoxLayout()
        layout2.setSpacing(0)
        layout2.addWidget(self.selector)
        layout2.addWidget(self.cfg_btn)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addLayout(layout2)

    def enterEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(CFG_ICON)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(QIcon())
        return super().leaveEvent(event)
    
    def blockSignals(self, block: bool):
        self.selector.blockSignals(block)
        super().blockSignals(block)
    
    def setSelectedValue(self, value: str, block_signals=True):
        if block_signals:
            self.blockSignals(True)
        self.selector.setCurrentText(value)
        if block_signals:
            self.blockSignals(False)
    

class TranslatorSelectionWidget(Widget):

    cfg_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        label = ConfigClickableLabel(text=self.tr('Translate'))
        label.clicked.connect(self.cfg_clicked)
        label_src = ConfigClickableLabel(text=self.tr('Source'))
        label_src.clicked.connect(self.cfg_clicked)
        label_tgt = ConfigClickableLabel(text=self.tr('Target'))
        label_tgt.clicked.connect(self.cfg_clicked)
        
        self.selector = SmallComboBox()
        self.src_selector = SmallComboBox()
        self.tgt_selector = SmallComboBox()
        self.cfg_btn = SmallConfigPutton()
        self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label)
        layout.addWidget(self.selector)
        layout.addWidget(label_src)
        layout.addWidget(self.src_selector)
        layout.addWidget(label_tgt)
        layout.addWidget(self.tgt_selector)
        layout.addWidget(self.cfg_btn)
        layout.setSpacing(1)

    def enterEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(CFG_ICON)
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(QIcon())
        return super().leaveEvent(event)
    
    def blockSignals(self, block: bool):
        self.src_selector.blockSignals(block)
        self.tgt_selector.blockSignals(block)
        self.selector.blockSignals(block)
        super().blockSignals(block)
    
    def finishSetTranslator(self, translator: BaseTranslator):
        self.blockSignals(True)
        self.src_selector.clear()
        self.tgt_selector.clear()
        self.src_selector.addItems(translator.supported_src_list)
        self.tgt_selector.addItems(translator.supported_tgt_list)
        self.selector.setCurrentText(translator.name)
        self.src_selector.setCurrentText(translator.lang_source)
        self.tgt_selector.setCurrentText(translator.lang_target)
        self.blockSignals(False)



class BottomBar(Widget):
    
    textedit_checkchanged = Signal()
    paintmode_checkchanged = Signal()
    textblock_checkchanged = Signal()

    def __init__(self, mainwindow: QMainWindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.setFixedHeight(BOTTOMBAR_HEIGHT)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow
        
        self.textdet_selector = SelectionWithConfigWidget(self.tr('Text Detector'))
        self.ocr_selector = SelectionWithConfigWidget(self.tr('OCR'))
        self.inpaint_selector = SelectionWithConfigWidget(self.tr('Inpaint'))
        self.trans_selector = TranslatorSelectionWidget()

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
        
        self.originalSlider = PaintQSlider(self.tr("Original image opacity"), Qt.Orientation.Horizontal, self)
        self.originalSlider.setFixedWidth(150)
        self.originalSlider.setRange(0, 100)

        self.textlayerSlider = PaintQSlider(self.tr("Text layer opacity"), Qt.Orientation.Horizontal, self)
        self.textlayerSlider.setFixedWidth(150)
        self.textlayerSlider.setValue(100)
        self.textlayerSlider.setRange(0, 100)
        
        self.hlayout.addWidget(self.textdet_selector)
        self.hlayout.addWidget(self.ocr_selector)
        self.hlayout.addWidget(self.inpaint_selector)
        self.hlayout.addWidget(self.trans_selector)
        # self.hlayout.addWidget(self.translatorStatusbtn)
        # self.hlayout.addWidget(self.transTranspageBtn)
        # self.hlayout.addWidget(self.inpainterStatBtn)
        self.hlayout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.hlayout.addWidget(self.textlayerSlider)
        self.hlayout.addWidget(self.originalSlider)
        self.hlayout.addWidget(self.paintChecker)
        self.hlayout.addWidget(self.texteditChecker)
        self.hlayout.addWidget(self.textblockChecker)
        self.hlayout.setContentsMargins(60, 0, 10, WINDOW_BORDER_WIDTH)


    def onPaintCheckerPressed(self):
        checked = self.paintChecker.isChecked()
        if checked:
            self.texteditChecker.setChecked(False)
        pcfg.imgtrans_paintmode = checked
        self.paintmode_checkchanged.emit()

    def onTextEditCheckerPressed(self):
        checked = self.texteditChecker.isChecked()
        if checked:
            self.paintChecker.setChecked(False)
        pcfg.imgtrans_textedit = checked
        self.textedit_checkchanged.emit()

    def onTextblockCheckerClicked(self):
        self.textblock_checkchanged.emit()