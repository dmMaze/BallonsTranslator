import os.path as osp
from collections import OrderedDict
from typing import List

from .stylewidgets import Widget, PaintQSlider
from .constants import WINDOW_BORDER_WIDTH, BOTTOMBAR_HEIGHT, DRAG_DIR_NONE, DRAG_DIR_VER, DRAG_DIR_BDIAG, DRAG_DIR_FDIAG
from . import constants as c

from qtpy.QtWidgets import QMainWindow, QHBoxLayout, QVBoxLayout, QFileDialog, QLabel, QSizePolicy, QToolBar, QMenu, QSpacerItem, QPushButton, QCheckBox, QToolButton
from qtpy.QtCore import Qt, Signal, QPoint
from qtpy.QtGui import QMouseEvent, QKeySequence
if c.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction

class ShowPageListChecker(QCheckBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class OpenBtn(QToolButton):
    def __init__(self, btn_width, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class RunBtn(QPushButton):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText('Run')


class StatusButton(QPushButton):
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


class TextChecker(QLabel):
    checkStateChanged = Signal(bool)
    def __init__(self, text: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setText(text)
        self.checked = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setCheckState(self, checked: bool):
        self.checked = checked
        if checked:
            self.setStyleSheet("QLabel { background-color: rgb(30, 147, 229); color: white; }")
        else:
            self.setStyleSheet("")

    def isChecked(self):
        return self.checked

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCheckState(not self.checked)
            self.checkStateChanged.emit(self.checked)


class LeftBar(Widget):
    recent_proj_list = []
    imgTransChecked = Signal()
    configChecked = Signal()
    open_dir = Signal(str)
    save_proj = Signal()
    run_imgtrans = Signal()
    def __init__(self, mainwindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.mainwindow: QMainWindow = mainwindow
        self.drag_resize_pos: QPoint = None

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
        vlayout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.setSpacing(btn_width/2)
        self.setGeometry(0, 0, 300, 500)
        self.setMouseTracking(True)

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

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if not self.mainwindow.isMaximized():
            if c.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            ex = g_pos.x()
            geow = self.mainwindow.geometry()
            if ex - geow.left() < WINDOW_BORDER_WIDTH:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

            if self.drag_resize_pos is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                delta_x = ex - self.drag_resize_pos.x()
                self.drag_resize_pos = g_pos
                geow.setLeft(geow.left() + delta_x)
                self.mainwindow.setGeometry(geow)
        return super().mouseMoveEvent(e)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if c.FLAG_QT6:
            g_pos = e.globalPosition().toPoint()
        else:
            g_pos = e.globalPos()
        ex = g_pos.x()
        geow = self.mainwindow.geometry()
        if ex - geow.left() < WINDOW_BORDER_WIDTH:
            self.drag_resize_pos = g_pos
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        return super().mouseReleaseEvent(e)

    def leaveEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        return super().leaveEvent(e)


class RightBar(Widget):
    def __init__(self, mainwindow: QMainWindow):
        super().__init__()
        self.mainwindow = mainwindow
        self.drag_resize_pos: QPoint = None
        self.setFixedWidth(WINDOW_BORDER_WIDTH)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, e:  QMouseEvent) -> None:
        if not self.mainwindow.isMaximized():
            if c.FLAG_QT6:
                g_pos = e.globalPosition().toPoint()
            else:
                g_pos = e.globalPos()
            ex = g_pos.x()
            geow = self.mainwindow.geometry()
            if ex - geow.right() < WINDOW_BORDER_WIDTH:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

            if self.drag_resize_pos is not None:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
                delta_x = ex - self.drag_resize_pos.x()
                self.drag_resize_pos = g_pos
                geow.setRight(geow.right() + delta_x)
                self.mainwindow.setGeometry(geow)
        return super().mouseMoveEvent(e)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if c.FLAG_QT6:
            g_pos = e.globalPosition().toPoint()
        else:
            g_pos = e.globalPos()
        ex = g_pos.x()
        geow = self.mainwindow.geometry()
        if ex - geow.right() < WINDOW_BORDER_WIDTH:
            self.drag_resize_pos = g_pos
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        return super().mouseReleaseEvent(e)

    def leaveEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        return super().leaveEvent(e)


class TitleBar(Widget):
    closebtn_clicked = Signal()
    def __init__(self, parent, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)
        self.mainwindow : QMainWindow = parent
        self.mPos: QPoint = None
        self.drag_resize_pos: QPoint = None
        self.drag_dir = DRAG_DIR_NONE
        self.normalsize = False
        self.proj_name = ''
        self.page_name = ''
        self.save_state = ''
        self.setFixedHeight(40)
        self.setMouseTracking(True)

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
        self.closeBtn.clicked.connect(self.closebtn_clicked)
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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if c.FLAG_QT6:
            g_pos = event.globalPosition().toPoint()
        else:
            g_pos = event.globalPos()
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.mainwindow.isMaximized() and \
                event.pos().y() < WINDOW_BORDER_WIDTH:
                self.drag_resize_pos = g_pos
                x = event.pos().x()
                if x < WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    self.drag_dir = DRAG_DIR_FDIAG
                elif x > self.width() - WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    self.drag_dir = DRAG_DIR_BDIAG
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                    self.drag_dir = DRAG_DIR_VER
            else:
                self.mPos = event.pos()
                self.mPosGlobal = g_pos
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mPos = None
        self.drag_resize_pos = None
        self.drag_dir = DRAG_DIR_NONE
        return super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if c.FLAG_QT6:
            g_pos = event.globalPosition().toPoint()
        else:
            g_pos = event.globalPos()
        if self.mPos is not None:
            self.mainwindow.show()
            if self.mainwindow.isMaximized():
                oldw = self.mainwindow.width()
                newgeo = self.mainwindow.normalGeometry()
                self.mainwindow.showNormal()
                
                if self.mPos.x() > newgeo.width():
                    self.mPos = QPoint(newgeo.width()-oldw+self.mPos.x(), self.mPos.y())
                else:
                    self.mainwindow.move(g_pos - self.mPos)
            else:
                self.mainwindow.move(g_pos-self.mPos)
        elif not self.mainwindow.isMaximized():
            y = event.pos().y()
            x = event.pos().x()
            if self.drag_dir != DRAG_DIR_NONE:
                geo = self.mainwindow.geometry()
                delta_y = g_pos.y() - self.drag_resize_pos.y()
                delta_x = g_pos.x() - self.drag_resize_pos.x()
                geo.setTop(geo.top() + delta_y)
                if self.drag_dir == DRAG_DIR_BDIAG:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    geo.setRight(geo.right() + delta_x)
                elif self.drag_dir == DRAG_DIR_FDIAG:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    geo.setLeft(geo.left() + delta_x)
                elif self.drag_dir == DRAG_DIR_VER:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                self.mainwindow.setGeometry(geo)
                self.drag_resize_pos = g_pos
            elif y < WINDOW_BORDER_WIDTH:
                if x < WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif x > self.width() - WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def hideEvent(self, e) -> None:
        self.mPos = None
        self.drag_resize_pos = None
        self.drag_dir = DRAG_DIR_NONE
        return super().hideEvent(e)

    def leaveEvent(self, e) -> None:
        self.mPos = None
        self.drag_resize_pos = None
        self.drag_dir = DRAG_DIR_NONE
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
    ocrcheck_statechanged = Signal(bool)
    transcheck_statechanged = Signal(bool)
    def __init__(self, mainwindow: QMainWindow, *args, **kwargs) -> None:
        super().__init__(mainwindow, *args, **kwargs)
        self.setFixedHeight(BOTTOMBAR_HEIGHT)
        self.setMouseTracking(True)
        self.mainwindow = mainwindow
        self.drag_resize_pos: QPoint = None
        self.drag_dir = DRAG_DIR_NONE

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
        
        self.originalSlider = PaintQSlider(self.tr("Original image transparency: ") + "value%", Qt.Orientation.Horizontal, self, minimumWidth=90)
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
        self.hlayout.setContentsMargins(90, 0, 15, WINDOW_BORDER_WIDTH)


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

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self.mainwindow.isMaximized():
            if c.FLAG_QT6:
                g_pos = event.globalPosition().toPoint()
            else:
                g_pos = event.globalPos()
            ey = g_pos.y()
            ex = g_pos.x()
            geow = self.mainwindow.geometry()

            if self.drag_dir != DRAG_DIR_NONE:
                
                delta_y = g_pos.y() - self.drag_resize_pos.y()
                delta_x = g_pos.x() - self.drag_resize_pos.x()
                geow.setBottom(geow.bottom() + delta_y)
                if self.drag_dir == DRAG_DIR_BDIAG:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                    geow.setLeft(geow.left() + delta_x)
                elif self.drag_dir == DRAG_DIR_FDIAG:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                    geow.setRight(geow.right() + delta_x)
                elif self.drag_dir == DRAG_DIR_VER:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
                self.mainwindow.setGeometry(geow)
                self.drag_resize_pos = g_pos
            
            elif geow.bottom() - ey < WINDOW_BORDER_WIDTH:
                if geow.right() - ex < WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeFDiagCursor)
                elif ex - geow.left() < WINDOW_BORDER_WIDTH:
                    self.setCursor(Qt.CursorShape.SizeBDiagCursor)
                else:
                    self.setCursor(Qt.CursorShape.SizeVerCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

        return super().mouseMoveEvent(event)

    def mousePressEvent(self, e: QMouseEvent) -> None:
        if c.FLAG_QT6:
            g_pos = e.globalPosition().toPoint()
        else:
            g_pos = e.globalPos()
        ey = g_pos.y()
        ex = g_pos.x()
        geow = self.mainwindow.geometry()
        if geow.bottom() - ey < WINDOW_BORDER_WIDTH:
            if geow.right() - ex < WINDOW_BORDER_WIDTH:
                self.drag_dir = DRAG_DIR_FDIAG
            elif ex - geow.left() < WINDOW_BORDER_WIDTH:
                self.drag_dir = DRAG_DIR_BDIAG
            else:
                self.drag_dir = DRAG_DIR_VER
            self.drag_resize_pos = g_pos
        return super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        self.drag_dir = DRAG_DIR_NONE
        return super().mouseReleaseEvent(e)

    def leaveEvent(self, e: QMouseEvent) -> None:
        self.drag_resize_pos = None
        self.drag_dir = DRAG_DIR_NONE
        return super().leaveEvent(e)