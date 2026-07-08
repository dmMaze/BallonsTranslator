import json
import os
import os.path as osp
from typing import List, Union

from qtpy.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel, QSizePolicy, QToolBar, QMenu, QSpacerItem, QPushButton, QCheckBox, QToolButton, QWidgetAction
from qtpy.QtCore import Qt, Signal, QPoint, QEvent, QSize, QRectF
from qtpy.QtGui import QMouseEvent, QKeySequence, QActionGroup, QIcon, QPainterPath, QRegion

from .custom_widget import Widget, PaintQSlider, SmallComboBox
from .misc import themed_icon_path
from ballontranslator.utils.shared import TITLEBAR_HEIGHT, WINDOW_BORDER_WIDTH, BOTTOMBAR_HEIGHT, LEFTBAR_WIDTH, LEFTBTN_WIDTH
from .framelesswindow import FramelessMoveResize
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLM_TRANSLATOR_KEY, profile_by_id
from ballontranslator.utils import shared
if shared.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction

BOTTOM_BAR_ACCENT_COLOR = '#5e98f7'

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

        self.openBtn = OpenBtn()
        self.openBtn.setFixedSize(LEFTBTN_WIDTH, LEFTBTN_WIDTH)

        openMenu = QMenu(self.openBtn)
        # Keep submenu ownership aligned with the visual popup chain for Wayland.
        self.recentMenu = QMenu(self.tr("Open Recent"), openMenu)
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
                self.configChecked.emit()
                self.configChecker.blockSignals(True)
                self.configChecker.setChecked(False)
                self.configChecker.blockSignals(False)
                

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

        self.viewMenu = viewMenu = QMenu(self.viewToolBtn)
        # Keep submenu ownership aligned with the visual popup chain for Wayland.
        self.displayLanguageMenu = QMenu(self.tr("Display Language"), viewMenu)
        self.lang_ac_group = lang_ac_group = QActionGroup(self)
        lang_ac_group.setExclusive(True)
        lang_actions = []
        for lang, lang_code in shared.DISPLAY_LANGUAGE_MAP.items():
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
        if not shared.ON_MACOS:
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

        if not shared.ON_MACOS:
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
            if e.type() == QEvent.Type.WindowStateChange and not shared.ON_MACOS:
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
        self.display_lang_changed.emit(shared.DISPLAY_LANGUAGE_MAP[ac.text()])

    def mousePressEvent(self, event: QMouseEvent) -> None:

        if shared.FLAG_QT6:
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
            if shared.FLAG_QT6:
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


def cfg_icon() -> QIcon:
    return QIcon(themed_icon_path('leftbar_config_activate.svg'))


def _theme_value(key: str, fallback: str) -> str:
    theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
    try:
        with open(shared.THEME_PATH, 'r', encoding='utf8') as f:
            theme_dict = json.loads(f.read())
        return theme_dict.get(theme, {}).get(key, fallback)
    except Exception:
        return fallback


def _theme_foreground_hex() -> str:
    fallback = '#8e99b1' if pcfg.darkmode else '#5d5d5f'
    return _theme_value('@qwidgetForegroundColor', fallback)


def _theme_menu_background_hex() -> str:
    fallback = '#21252b' if pcfg.darkmode else '#e1e4eb'
    return _theme_value('@emptyContentBackgroundColor', fallback)


def _blend_hex(color: str, target: str, amount: float) -> str:
    def rgb(value: str):
        value = value.lstrip('#')
        return [int(value[i:i + 2], 16) for i in (0, 2, 4)]

    src = rgb(color)
    dst = rgb(target)
    mixed = [round(src[i] + (dst[i] - src[i]) * amount) for i in range(3)]
    return '#{:02x}{:02x}{:02x}'.format(*mixed)


def _section_label_color() -> str:
    foreground = _theme_foreground_hex()
    return _blend_hex(foreground, '#000000' if pcfg.darkmode else '#ffffff', 0.28)


def _bottom_tool_icon_path(icon_filename: str) -> str:
    icon_path = themed_icon_path(icon_filename)
    if not pcfg.darkmode or not icon_filename.lower().endswith('.svg'):
        return icon_path

    color = _theme_foreground_hex()
    cache_dir = osp.join(shared.cache_dir, 'icons', 'bottom_bar', 'eva-dark')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = osp.join(cache_dir, icon_filename)
    with open(icon_path, 'r', encoding='utf-8') as f:
        svg = f.read()
    for fill in ('#697187', '#b3b6bf', '#96a4cd', '#697186'):
        svg = svg.replace(fill, color)
    with open(cache_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    return cache_path


def _set_bottom_tool_button_icon(tool_btn: QToolButton, icon_filename: str):
    tool_btn.setIcon(QIcon(_bottom_tool_icon_path(icon_filename)))
    tool_btn.setIconSize(QSize(18, 18))
    style_enum = getattr(Qt, 'ToolButtonStyle', Qt)
    tool_btn.setToolButtonStyle(style_enum.ToolButtonTextBesideIcon)


def _instant_popup_mode():
    popup_enum = getattr(QToolButton, 'ToolButtonPopupMode', QToolButton)
    return popup_enum.InstantPopup


def _bottom_tool_button_text(name: str) -> str:
    return '  ' + name



class BottomBarMenu(QMenu):

    def showEvent(self, event) -> None:
        super().showEvent(event)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), 8, 8)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))


def _style_bottom_menu(menu: QMenu) -> QMenu:
    menu.setObjectName('BottomBarModuleMenu')
    attr_enum = getattr(Qt, 'WidgetAttribute', Qt)
    menu.setAttribute(attr_enum.WA_TranslucentBackground, True)
    return menu


def _bottom_menu(parent: QToolButton) -> QMenu:
    return _style_bottom_menu(BottomBarMenu(parent))


def _bottom_submenu(title: str, parent: QMenu) -> QMenu:
    menu = _style_bottom_menu(BottomBarMenu(parent))
    menu.setTitle(title)
    return menu


def _add_bottom_menu_section(menu: QMenu, text: str, accent: bool = False):
    label = QLabel(text, menu)
    label.setObjectName('BottomBarMenuSectionLabel')
    color = BOTTOM_BAR_ACCENT_COLOR if accent else _section_label_color()
    label.setStyleSheet(
        'QLabel#BottomBarMenuSectionLabel {{ '
        'color: {}; background-color: {}; '
        '}}'.format(color, _theme_menu_background_hex())
    )
    action = QWidgetAction(menu)
    action.setDefaultWidget(label)
    menu.addAction(action)


def _checked_action_text(text: str, checked: bool) -> str:
    return text + ('\t\u2713' if checked else '')


def _add_bottom_menu_action(menu: QMenu, text: str, checked: bool, callback):
    action = QAction(_checked_action_text(text, checked), menu)
    action.triggered.connect(callback)
    menu.addAction(action)
    return action


def _add_bottom_submenu(parent: QMenu, submenu: QMenu, text: str, checked: bool):
    parent.addMenu(submenu)
    submenu.menuAction().setText(_checked_action_text(text, checked))
    return submenu


class ModuleSelectionToolButtonWidget(Widget):

    cfg_clicked = Signal()

    def __init__(self, fallback_name: str, icon_filename: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fallback_name = fallback_name
        self.icon_filename = icon_filename
        self.selector = SmallComboBox()
        self.selector.setVisible(False)
        self.selector.currentTextChanged.connect(self.updateButtonText)

        self.tool_btn = QToolButton(self)
        self.tool_btn.setObjectName('BottomBarModuleToolButton')
        self.tool_btn.setToolTip(fallback_name)
        self.tool_btn.setPopupMode(_instant_popup_mode())
        _set_bottom_tool_button_icon(self.tool_btn, icon_filename)
        self.tool_btn.setText(fallback_name)
        self.menu = _bottom_menu(self.tool_btn)
        self.tool_btn.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.rebuildMenu)

        self.cfg_btn = SmallConfigPutton()
        self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.tool_btn)
        layout.addWidget(self.cfg_btn)
        self.updateButtonText()

    def enterEvent(self, event: QEvent) -> None:
        self.cfg_btn.setIcon(cfg_icon())
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.cfg_btn.setIcon(QIcon())
        return super().leaveEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            _set_bottom_tool_button_icon(self.tool_btn, self.icon_filename)
        return super().changeEvent(event)

    def blockSignals(self, block: bool):
        self.selector.blockSignals(block)
        super().blockSignals(block)

    def setSelectedValue(self, value: str, block_signals=True):
        if block_signals:
            self.blockSignals(True)
        self.selector.setCurrentText(value)
        if block_signals:
            self.blockSignals(False)
        self.updateButtonText()

    def rebuildMenu(self):
        self.menu.clear()
        current_module = self.selector.currentText()
        for i in range(self.selector.count()):
            module = self.selector.itemText(i)
            _add_bottom_menu_action(
                self.menu,
                module,
                module == current_module,
                lambda checked=False, value=module: self.selector.setCurrentText(value),
            )

    def updateButtonText(self, *args):
        name = self.selector.currentText() or self.fallback_name
        self.tool_btn.setText(_bottom_tool_button_text(name))


class TranslatorSelectionWidget(Widget):

    cfg_clicked = Signal()
    edit_clicked = Signal(str)
    llm_profile_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.selector = SmallComboBox()
        self.src_selector = SmallComboBox()
        self.tgt_selector = SmallComboBox()
        self.selector.setVisible(False)
        self.src_selector.setVisible(False)
        self.tgt_selector.setVisible(False)
        self.tool_btn = QToolButton(self)
        self.tool_btn.setObjectName('BottomBarModuleToolButton')
        self.tool_btn.setToolTip(self.tr('Translator'))
        self.tool_btn.setPopupMode(_instant_popup_mode())
        self.tool_btn.setText(self.tr('Translator'))
        self.icon_filename = 'bottombar_translate_activate.svg'
        _set_bottom_tool_button_icon(self.tool_btn, self.icon_filename)
        self.menu = _bottom_menu(self.tool_btn)
        self.tool_btn.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.rebuildMenu)
        self.edit_btn = SmallConfigPutton()
        self.edit_btn.clicked.connect(self.onEditClicked)
        self.cfg_btn = SmallConfigPutton()
        self.cfg_btn.clicked.connect(self.cfg_clicked)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tool_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.cfg_btn)
        layout.setSpacing(1)
        self.updateButtonText()

    def enterEvent(self, event: QEvent) -> None:
        if self.edit_btn.isVisible():
            self.edit_btn.setIcon(QIcon(themed_icon_path('edit.svg')))
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(cfg_icon())
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.edit_btn.setIcon(QIcon())
        if self.cfg_btn is not None:
            self.cfg_btn.setIcon(QIcon())
        return super().leaveEvent(event)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.PaletteChange):
            _set_bottom_tool_button_icon(self.tool_btn, self.icon_filename)
        return super().changeEvent(event)
    
    def blockSignals(self, block: bool):
        self.src_selector.blockSignals(block)
        self.tgt_selector.blockSignals(block)
        self.selector.blockSignals(block)
        super().blockSignals(block)

    def _section(self, text: str, accent: bool = False):
        _add_bottom_menu_section(self.menu, text, accent=accent)

    def rebuildMenu(self):
        self.menu.clear()
        current_translator = self.selector.currentText()
        self._section(self.tr('Translator'))
        for i in range(self.selector.count()):
            translator = self.selector.itemText(i)
            if translator == LLM_TRANSLATOR_KEY:
                continue
            _add_bottom_menu_action(
                self.menu,
                translator,
                current_translator == translator,
                lambda checked=False, value=translator: self.selector.setCurrentText(value),
            )

        self._section(self.tr('LLM'), accent=current_translator == LLM_TRANSLATOR_KEY)
        for profile in pcfg.module.llm_profiles:
            profile_id = profile.id
            profile_menu = _bottom_submenu(profile.name or profile_id, self.menu)
            _add_bottom_submenu(
                self.menu,
                profile_menu,
                profile.name or profile_id,
                current_translator == LLM_TRANSLATOR_KEY and pcfg.module.translator_llm_id == profile_id,
            )
            self._buildProfileMenu(profile_menu, profile)

        self._section(self.tr('Language'))
        source_menu = _bottom_submenu(
            self.tr('Source - {language}').format(language=self.src_selector.currentText()),
            self.menu,
        )
        self.menu.addMenu(source_menu)
        for i in range(self.src_selector.count()):
            lang = self.src_selector.itemText(i)
            _add_bottom_menu_action(
                source_menu,
                lang,
                lang == self.src_selector.currentText(),
                lambda checked=False, value=lang: self.src_selector.setCurrentText(value),
            )

        target_menu = _bottom_submenu(
            self.tr('Target - {language}').format(language=self.tgt_selector.currentText()),
            self.menu,
        )
        self.menu.addMenu(target_menu)
        for i in range(self.tgt_selector.count()):
            lang = self.tgt_selector.itemText(i)
            _add_bottom_menu_action(
                target_menu,
                lang,
                lang == self.tgt_selector.currentText(),
                lambda checked=False, value=lang: self.tgt_selector.setCurrentText(value),
            )

    def selectLLMProfile(self, profile_id: str):
        pcfg.module.translator_llm_id = profile_id
        if self.selector.currentText() != LLM_TRANSLATOR_KEY:
            self.selector.setCurrentText(LLM_TRANSLATOR_KEY)
        self.llm_profile_changed.emit(profile_id)
        self.updateButtonText()

    def selectLLMProfileSetting(self, profile_id: str, key: str, value: str):
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is not None:
            setattr(profile, key, value)
            if key == 'model':
                options = profile.model_options
                if value and value not in options:
                    options.insert(0, value)
        self.selectLLMProfile(profile_id)

    def _buildProfileMenu(self, menu: QMenu, profile: dict):
        profile_id = profile.id
        selected_profile = (
            self.selector.currentText() == LLM_TRANSLATOR_KEY
            and pcfg.module.translator_llm_id == profile_id
        )

        _add_bottom_menu_section(menu, self.tr('Thinking Level'))
        thinking_options = [str(option) for option in profile.thinking_level_options if str(option)]
        current_thinking = str(profile.thinking_level or 'None')
        for thinking_level in thinking_options:
            _add_bottom_menu_action(
                menu,
                thinking_level,
                selected_profile and thinking_level == current_thinking,
                lambda checked=False, pid=profile_id, value=thinking_level: self.selectLLMProfileSetting(pid, 'thinking_level', value),
            )

        _add_bottom_menu_section(menu, self.tr('Model'))
        model_options = [str(option) for option in profile.model_options if str(option)]
        current_model = str(profile.model or '')
        for model in model_options:
            _add_bottom_menu_action(
                menu,
                model,
                selected_profile and model == current_model,
                lambda checked=False, pid=profile_id, value=model: self.selectLLMProfileSetting(pid, 'model', value),
            )

    def onEditClicked(self):
        if self.selector.currentText() == LLM_TRANSLATOR_KEY:
            self.edit_clicked.emit(pcfg.module.translator_llm_id)

    def updateButtonText(self):
        name = self.selector.currentText()
        is_llm = name == LLM_TRANSLATOR_KEY
        if name == LLM_TRANSLATOR_KEY:
            profile = profile_by_id(pcfg.module.llm_profiles, pcfg.module.translator_llm_id)
            if profile is not None:
                model_options = [str(option) for option in profile.model_options if str(option)]
                model = str(profile.model or '').strip()
                thinking_level = str(profile.thinking_level or 'None').strip()
                if model_options and model:
                    name = model
                    if thinking_level and thinking_level != 'None':
                        name = self.tr('{model} {thinking_level}').format(model=model, thinking_level=thinking_level)
                else:
                    name = profile.name or name
        if not name:
            name = self.tr('Translator')
        self.tool_btn.setText(_bottom_tool_button_text(name))
        if self.tool_btn.property('llmActive') != is_llm:
            self.tool_btn.setProperty('llmActive', is_llm)
            self.tool_btn.style().unpolish(self.tool_btn)
            self.tool_btn.style().polish(self.tool_btn)
        self.edit_btn.setVisible(is_llm)
    
    def setTranslatorMetadata(self, name: str, supported_src_list, supported_tgt_list, lang_source: str, lang_target: str):
        # Metadata can come from ModuleSpec before the translator is imported.
        self.blockSignals(True)
        self.src_selector.clear()
        self.tgt_selector.clear()
        self.src_selector.addItems(supported_src_list)
        self.tgt_selector.addItems(supported_tgt_list)
        self.selector.setCurrentText(name)
        self.src_selector.setCurrentText(lang_source)
        self.tgt_selector.setCurrentText(lang_target)
        self.blockSignals(False)
        self.updateButtonText()



class BottomBar(Widget):
    
    textedit_checkchanged = Signal()
    paintmode_checkchanged = Signal()
    textblock_checkchanged = Signal()

    def __init__(self, mainwindow: QMainWindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.setFixedHeight(BOTTOMBAR_HEIGHT)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow
        
        self.textdet_selector = ModuleSelectionToolButtonWidget(self.tr('Text Detector'), 'textdetect.svg')
        self.ocr_selector = ModuleSelectionToolButtonWidget(self.tr('OCR'), 'small_ocr.svg')
        self.inpaint_selector = ModuleSelectionToolButtonWidget(self.tr('Inpaint'), 'drawingtools_inpaint.svg')
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
