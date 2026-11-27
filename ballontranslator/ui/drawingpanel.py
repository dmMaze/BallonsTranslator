from __future__ import annotations

from qtpy.QtCore import Signal, Qt, QEvent, QPointF, QRect, QSize, QSizeF, QLineF, QRectF, QSignalBlocker, QTimer
from qtpy.QtWidgets import QAbstractSpinBox, QGridLayout, QPushButton, QComboBox, QPlainTextEdit, QToolButton, QSizePolicy, QBoxLayout, QCheckBox, QHBoxLayout, QGraphicsView, QSpinBox, QStackedWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsRectItem, QToolTip
from qtpy.QtGui import QIcon, QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFont, QFontMetrics, QImage

from typing import Union, Tuple, List
from concurrent.futures import Future, ThreadPoolExecutor
from math import log
import html

import numpy as np
import cv2

from ballontranslator.utils.imgproc_utils import (
    enlarge_window,
    magic_wand_mask,
    magic_wand_preview_overlay,
    MagicWandFillMode,
)
from ballontranslator.utils.textblock_mask import canny_flood, connected_canny_flood, region_mask
from ballontranslator.utils.blob_selection import BlobSelection, RULE_ANY, RULE_MAJORITY
from ballontranslator.utils.logger import logger
from .cursor import magic_wand_cursor
from .funcmaps import get_maskseg_method
from .module_manager import ModuleManager
from .module_tool_button import ModuleSelectionMenu, ModuleSelectionToolButton
from .llm_modality import LLM_MODALITY_IMAGE
from .image_edit import ImageEditMode, PenShape, PixmapItem, StrokeImgItem, shape_fill_image
from .custom_widget import Widget, SeparatorWidget, PaintQSlider, ColorPickerLabel
from .custom_widget.helper import borderColor
from .canvas import Canvas
from .misc import ndarray2pixmap, themed_icon_path
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, LLM_INPAINT_KEY, profile_by_id
from ballontranslator.utils import shared
from ballontranslator.utils.shared import CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from ballontranslator.utils.logger import logger as LOGGER
from .drawing_commands import InpaintUndoCommand, StrokeItemUndoCommand
from .text_engine.shape_control import CONTROL_ITEM_DATA_KEY

INPAINT_BRUSH_COLOR = QColor(127, 0, 127, 127)

# Rect tool preview overlay, as RGBA. Purple marks the pixels that will be
# inpainted; cyan marks blobs that exist but are not selected, so the user
# can see what is available to click. Pixels tinted by neither colour did
# not survive binarization and cannot be selected at all.
BLOB_SELECTED_RGBA = (127, 0, 127, 127)
BLOB_AVAILABLE_RGBA = (0, 229, 255, 140)
BLOB_BOX_COLOR = QColor(0, 255, 255, 220)
# Which blobs a marquee catches by default. RULE_ANY takes a blob if any
# pixel is inside; RULE_MAJORITY needs more than half of it, which fails
# safe when a letter is fused to the balloon outline. Press O while a
# preview is up to switch between them.
BLOB_BOX_RULE_DEFAULT = RULE_MAJORITY
MAX_PEN_SIZE = 1000
MIN_PEN_SIZE = 1
TOOLNAME_POINT_SIZE = 13
MAGICWAND_TOLERANCE_MAX = 255
MAGICWAND_RANGE_LIMIT = 50


def _magic_wand_preview(
    img: np.ndarray, seed: tuple[int, int, int, int, int],
) -> tuple[np.ndarray, int, int] | None:
    """Calculate an overlay from an image snapshot without accessing Qt objects.

    >>> _magic_wand_preview(np.zeros((1, 1, 3), np.uint8), (0, 0, 0, 0, 0))[1:]
    (0, 0)
    """
    x, y, tolerance, radius, fill_mode = seed
    return magic_wand_preview_overlay(
        magic_wand_mask(img, (x, y), tolerance, radius, fill_mode)
    )


class _BrushThicknessSlider(PaintQSlider):
    """Keep exact pixel values while spacing brush sizes logarithmically."""

    def _value_to_position_ratio(self) -> float:
        minimum = self.minimum()
        maximum = self.maximum()
        if minimum < 1 or maximum <= minimum:
            return super()._value_to_position_ratio()
        return log(self.value() / minimum) / log(maximum / minimum)

    def _position_ratio_to_value(self, ratio: float) -> int:
        minimum = self.minimum()
        maximum = self.maximum()
        if minimum < 1 or maximum <= minimum:
            return super()._position_ratio_to_value(ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        return round(minimum * (maximum / minimum) ** ratio)


def _create_thickness_control(
    parent: Widget,
) -> tuple[_BrushThicknessSlider, QSpinBox, QHBoxLayout]:
    slider = _BrushThicknessSlider(parent=parent)
    slider.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
    slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    slider.show_hover_value = False

    spinbox = QSpinBox(parent)
    spinbox.setProperty('paintSliderValueEditor', True)
    spinbox.setRange(MIN_PEN_SIZE, MAX_PEN_SIZE)
    spinbox.setValue(slider.value())
    spinbox.setSuffix(' px')
    spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
    button_symbols = getattr(
        QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox
    )
    spinbox.setButtonSymbols(button_symbols.NoButtons)
    spinbox.setKeyboardTracking(False)
    spinbox.setFixedSize(60, 22)

    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    layout.addWidget(slider, 1)
    layout.addWidget(spinbox)
    return slider, spinbox, layout


class DrawToolCheckBox(QCheckBox):
    checked = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stateChanged.connect(self.on_state_changed)

    def mousePressEvent(self, event) -> None:
        if self.isChecked():
            return
        return super().mousePressEvent(event)

    def on_state_changed(self, state: int) -> None:
        if self.isChecked():
            self.checked.emit()

class ToolNameLabel(QLabel):
    def __init__(self, fix_width=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        font = self.font()
        font.setPointSizeF(TOOLNAME_POINT_SIZE)
        fmt = QFontMetrics(font)
        
        if fix_width is not None:
            self.setFixedWidth(fix_width)
            text_width = fmt.width(self.text())
            if text_width > fix_width * 0.95:
                font_size = TOOLNAME_POINT_SIZE * fix_width * 0.95 / text_width
                font.setPointSizeF(font_size)
        self.setFont(font)


def _labeled_paint_slider(
    parent: Widget,
    label: str,
    minimum: int,
    maximum: int,
    value: int,
) -> tuple[Widget, PaintQSlider]:
    row = Widget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    layout.addWidget(ToolNameLabel(100, label))
    slider = PaintQSlider(parent=row)
    slider.setRange(minimum, maximum)
    slider.setValue(value)
    layout.addWidget(slider)
    return row, slider


class DrawingInpainterMenu(ModuleSelectionMenu):
    """Use shared menu rendering with draw-panel-owned selection values.

    >>> DrawingInpainterMenu.__name__
    'DrawingInpainterMenu'
    """

    def selectedProfileId(self) -> str:
        return pcfg.drawpanel.inpaint_llm_id

    def _profileSettingValue(self, profile: LLMProfile, key: str) -> str:
        if key == 'image_model' and profile.id == self.selectedProfileId():
            return pcfg.drawpanel.inpaint_llm_model or profile.image_model
        return super()._profileSettingValue(profile, key)

    def selectLLMProfile(self, profile_id: str, *, model: str | None = None) -> None:
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is None or not profile.support_image:
            return
        if model is not None:
            pcfg.drawpanel.inpaint_llm_model = model
        elif profile_id != self.selectedProfileId():
            pcfg.drawpanel.inpaint_llm_model = profile.image_model
        pcfg.drawpanel.inpaint_llm_id = profile_id
        pcfg.drawpanel.inpainter = LLM_INPAINT_KEY
        with QSignalBlocker(self.selector):
            self.selector.setCurrentText(LLM_INPAINT_KEY)
        self.llm_profile_changed.emit(profile_id)

    def selectLLMProfileSetting(self, profile_id: str, key: str, value: str) -> None:
        if key == 'image_model':
            self.selectLLMProfile(profile_id, model=value)


class InpainterSelectorRow(Widget):
    """Shared draw-tool selection and optional per-request prompt override.

    >>> InpainterSelectorRow.__name__
    'InpainterSelectorRow'
    """

    selection_changed = Signal()
    prompt_changed = Signal()
    config_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        row = QHBoxLayout()
        row.addWidget(ToolNameLabel(100, self.tr('Inpainter')))
        self.selector = QComboBox(self)
        self.selector.hide()
        self.selector.currentTextChanged.connect(self._onSelectionChanged)
        self.tool_button = ModuleSelectionToolButton(self)
        self.tool_button.setObjectName('DrawingInpainterSelector')
        self.tool_button.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.tool_button.setMinimumWidth(CONFIG_COMBOBOX_SHORT)
        self.tool_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tool_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = DrawingInpainterMenu(self.selector, self.tr('Inpainter'), LLM_MODALITY_IMAGE, self.tool_button)
        self.menu.llm_profile_changed.connect(self._onSelectionChanged)
        self.tool_button.setMenu(self.menu)
        row.addWidget(self.tool_button, 1)
        self.config_button = QPushButton(self)
        self.config_button.setObjectName('DrawingInpainterConfigButton')
        self.config_button.setFixedSize(24, 24)
        self.config_button.setIcon(
            QIcon(themed_icon_path('leftbar_config_activate.svg'))
        )
        self.config_button.setToolTip(self.tr('Config'))
        self.config_button.setAccessibleName(self.tr('Config'))
        self.config_button.clicked.connect(self.config_requested)
        row.addWidget(self.config_button)

        self.prompt_panel = Widget(self)
        self.prompt_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        prompt_layout = QVBoxLayout(self.prompt_panel)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_label = QLabel(self.tr('Prompt override'), self.prompt_panel)
        prompt_layout.addWidget(prompt_label)
        self.prompt_edit = QPlainTextEdit(self.prompt_panel)
        self.prompt_edit.setPlaceholderText(self.tr('Leave blank to use the selected profile’s inpainting prompt.'))
        self.prompt_edit.setFixedHeight(90)
        self.prompt_edit.textChanged.connect(self._onPromptChanged)
        prompt_layout.addWidget(self.prompt_edit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        self.syncFromConfig()

    def setOptions(self, options: list[str]) -> None:
        with QSignalBlocker(self.selector):
            self.selector.clear()
            self.selector.addItems(options)
            self.selector.setCurrentText(pcfg.drawpanel.inpainter)
        self.syncFromConfig()

    def syncFromConfig(self) -> None:
        if self.selector.currentText() != pcfg.drawpanel.inpainter:
            with QSignalBlocker(self.selector):
                self.selector.setCurrentText(pcfg.drawpanel.inpainter)
        text = self.menu.selectedText()
        self.tool_button.setText(text)
        self.tool_button.setToolTip(text)
        self.prompt_panel.setVisible(self.menu.isCurrentLLM())
        prompt = pcfg.drawpanel.inpaint_prompt_override
        if self.prompt_edit.toPlainText() != prompt:
            with QSignalBlocker(self.prompt_edit):
                self.prompt_edit.setPlainText(prompt)

    def _onSelectionChanged(self) -> None:
        pcfg.drawpanel.inpainter = self.selector.currentText()
        self.selection_changed.emit()

    def _onPromptChanged(self) -> None:
        pcfg.drawpanel.inpaint_prompt_override = self.prompt_edit.toPlainText()
        self.prompt_changed.emit()


class InpaintPanel(Widget):

    thicknessChanged = Signal(int)
    toleranceChanged = Signal(int)
    rangeChanged = Signal(int)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        (
            self.thicknessSlider,
            self.thicknessSpinBox,
            thickness_control_layout,
        ) = _create_thickness_control(self)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSpinBox.valueChanged.connect(self.on_thickness_changed)

        self.thickness_row = Widget(self)
        thickness_layout = QHBoxLayout(self.thickness_row)
        thickness_layout.setContentsMargins(0, 0, 0, 0)
        thickness_layout.setSpacing(10)
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addLayout(thickness_control_layout, 1)

        shape_label = ToolNameLabel(100, self.tr('Tool Type'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle Brush'),
            self.tr('Rectangle Brush'),
            self.tr('Magic Wand'),
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        self.fill_mode_row = Widget(self)
        fill_mode_layout = QHBoxLayout(self.fill_mode_row)
        fill_mode_layout.setContentsMargins(0, 0, 0, 0)
        fill_mode_layout.setSpacing(10)
        fill_mode_layout.addWidget(ToolNameLabel(100, self.tr('Selection Mode')))
        self.fillModeCombobox = QComboBox(self.fill_mode_row)
        self.fillModeCombobox.addItems([
            self.tr('Selection'),
            self.tr('Selection Interior'),
            self.tr('Selection + Interior'),
        ])
        self.fillModeChanged = self.fillModeCombobox.currentIndexChanged
        fill_mode_layout.addWidget(self.fillModeCombobox)

        self.tolerance_row, self.toleranceSlider = _labeled_paint_slider(
            self,
            self.tr('Tolerance'),
            0,
            MAGICWAND_TOLERANCE_MAX,
            32,
        )
        self.toleranceSlider.valueChanged.connect(self.toleranceChanged)
        self.range_row, self.rangeSlider = _labeled_paint_slider(
            self,
            self.tr('Range'),
            -MAGICWAND_RANGE_LIMIT,
            MAGICWAND_RANGE_LIMIT,
            0,
        )
        self.rangeSlider.valueChanged.connect(self.rangeChanged)

        self.inpainter_selector = InpainterSelectorRow(self)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.inpainter_selector)
        layout.addWidget(self.inpainter_selector.prompt_panel)
        layout.addWidget(self.thickness_row)
        layout.addLayout(shape_layout)
        layout.addWidget(self.fill_mode_row)
        layout.addWidget(self.tolerance_row)
        layout.addWidget(self.range_row)
        layout.setSpacing(14)
        self.sync_shape_controls()

    def on_thickness_changed(self, value: int) -> None:
        sender = self.sender()
        if sender is self.thicknessSlider:
            with QSignalBlocker(self.thicknessSpinBox):
                self.thicknessSpinBox.setValue(value)
            if self.thicknessSlider.hasFocus():
                self.thicknessChanged.emit(value)
        elif sender is self.thicknessSpinBox:
            self.thicknessSlider.setValue(value)
            self.thicknessChanged.emit(value)

    @property
    def shape(self):
        return self.shapeCombobox.currentIndex()

    @property
    def fill_mode(self) -> int:
        return self.fillModeCombobox.currentIndex()

    def set_fill_mode(self, mode: int) -> None:
        self.fillModeCombobox.setCurrentIndex(mode)

    def is_magic_wand(self) -> bool:
        return self.shape == PenShape.MagicWand

    def sync_shape_controls(self) -> None:
        wand = self.is_magic_wand()
        self.thickness_row.setVisible(not wand)
        self.tolerance_row.setVisible(wand)
        self.range_row.setVisible(wand)
        self.fill_mode_row.setVisible(wand)


class PenConfigPanel(Widget):
    thicknessChanged = Signal(int)
    colorChanged = Signal(list)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        (
            self.thicknessSlider,
            self.thicknessSpinBox,
            thickness_control_layout,
        ) = _create_thickness_control(self)
        self.thicknessSlider.valueChanged.connect(self.on_thickness_changed)
        self.thicknessSpinBox.valueChanged.connect(self.on_thickness_changed)
        self.alphaSlider = PaintQSlider()
        self.alphaSlider.setRange(0, 255)
        self.alphaSlider.setValue(255)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)

        self.colorPicker = ColorPickerLabel()
        self.colorPicker.colorChanged.connect(self.on_color_changed)
        
        color_label = ToolNameLabel(None, self.tr('Color'))
        alpha_label = ToolNameLabel(None, self.tr('Alpha'))
        color_layout = QHBoxLayout()
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.colorPicker)
        color_layout.addWidget(alpha_label)
        color_layout.addWidget(self.alphaSlider)
        
        thickness_layout = QHBoxLayout()
        thickness_label = ToolNameLabel(100, self.tr('Thickness'))
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addLayout(thickness_control_layout, 1)
        thickness_layout.setSpacing(10)

        shape_label = ToolNameLabel(100, self.tr('Shape'))
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItems([
            self.tr('Circle'),
            self.tr('Rectangle'),
            # self.tr('Triangle')
        ])
        self.shapeChanged = self.shapeCombobox.currentIndexChanged
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(shape_label)
        shape_layout.addWidget(self.shapeCombobox)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(color_layout)
        layout.addLayout(thickness_layout)
        layout.addLayout(shape_layout)
        layout.setSpacing(20)

    def on_thickness_changed(self, value: int) -> None:
        sender = self.sender()
        if sender is self.thicknessSlider:
            with QSignalBlocker(self.thicknessSpinBox):
                self.thicknessSpinBox.setValue(value)
            if self.thicknessSlider.hasFocus():
                self.thicknessChanged.emit(value)
        elif sender is self.thicknessSpinBox:
            self.thicknessSlider.setValue(value)
            self.thicknessChanged.emit(value)

    def on_alpha_changed(self):
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorPicker.setPickerColor(color)
        self.colorChanged.emit(color)

    def on_color_changed(self):
        color = self.colorPicker.rgba()
        color = [color[0], color[1], color[2], self.alphaSlider.value()]
        self.colorChanged.emit(color)

    @property
    def shape(self):
        return self.shapeCombobox.currentIndex()


class ShapeFillPanel(Widget):
    def __init__(self, parent: Widget | None = None) -> None:
        super().__init__(parent)
        self.shapeCombobox = QComboBox(self)
        self.shapeCombobox.addItem(self.tr('Rectangle'), 'rectangle')
        self.shapeCombobox.addItem(self.tr('Ellipse'), 'ellipse')
        self.shapeCombobox.currentIndexChanged.connect(self.on_shape_changed)
        self.colorPicker = ColorPickerLabel(self)
        self.colorPicker.setPickerColor(QColor(pcfg.drawpanel.shape_fill_color))
        self.colorPicker.setToolTip(self.tr('Fill Color'))
        self.colorPicker.colorChanged.connect(self.on_color_changed)

        layout = QGridLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(ToolNameLabel(100, self.tr('Shape')), 0, 0)
        layout.addWidget(self.shapeCombobox, 0, 1)
        layout.addWidget(ToolNameLabel(100, self.tr('Color')), 1, 0)
        layout.addWidget(self.colorPicker, 1, 1, Qt.AlignmentFlag.AlignLeft)
        self.alphaSlider = PaintQSlider()
        self.alphaSlider.setRange(0, 255)
        self.alphaSlider.setValue(pcfg.drawpanel.shape_fill_alpha)
        self.alphaSlider.valueChanged.connect(self.on_alpha_changed)
        layout.addWidget(ToolNameLabel(100, self.tr('Alpha')), 2, 0)
        layout.addWidget(self.alphaSlider, 2, 1)

        self.borderChecker = QCheckBox(self.tr('Border'), self)
        self.borderChecker.setToolTip(self.tr('Draw shape with border'))
        self.borderChecker.setChecked(pcfg.drawpanel.shape_border_enabled)
        self.borderChecker.toggled.connect(self.on_border_enabled_changed)
        layout.addWidget(self.borderChecker, 3, 0, 1, 2)
        (
            self.borderWidthSlider,
            self.borderWidthSpinBox,
            border_width_layout,
        ) = _create_thickness_control(self)
        self.borderWidthSlider.setValue(pcfg.drawpanel.shape_border_width)
        self.borderWidthSpinBox.setValue(pcfg.drawpanel.shape_border_width)
        self.borderWidthSlider.valueChanged.connect(self.on_border_width_changed)
        self.borderWidthSpinBox.valueChanged.connect(self.on_border_width_changed)
        self.borderWidthLabel = ToolNameLabel(100, self.tr('Border Width'))
        layout.addWidget(self.borderWidthLabel, 4, 0)
        layout.addLayout(border_width_layout, 4, 1)
        self.borderColorPicker = ColorPickerLabel(self)
        self.borderColorPicker.setPickerColor(QColor(pcfg.drawpanel.shape_border_color))
        self.borderColorPicker.setToolTip(self.tr('Border Color'))
        self.borderColorPicker.colorChanged.connect(self.on_border_color_changed)
        self.borderColorLabel = ToolNameLabel(100, self.tr('Border Color'))
        layout.addWidget(self.borderColorLabel, 5, 0)
        layout.addWidget(self.borderColorPicker, 5, 1, Qt.AlignmentFlag.AlignLeft)
        layout.setVerticalSpacing(14)
        self.sync_border_controls()

    def sync_border_controls(self) -> None:
        enabled = self.borderChecker.isChecked()
        for widget in (
            self.borderWidthLabel, self.borderWidthSlider, self.borderWidthSpinBox,
            self.borderColorLabel, self.borderColorPicker,
        ):
            widget.setEnabled(enabled)

    def on_border_enabled_changed(self, checked: bool) -> None:
        pcfg.drawpanel.shape_border_enabled = checked
        self.sync_border_controls()

    def on_border_width_changed(self, value: int) -> None:
        # Mirror the pen thickness pair: either editor updates the other.
        if self.sender() is self.borderWidthSlider:
            with QSignalBlocker(self.borderWidthSpinBox):
                self.borderWidthSpinBox.setValue(value)
        else:
            with QSignalBlocker(self.borderWidthSlider):
                self.borderWidthSlider.setValue(value)
        pcfg.drawpanel.shape_border_width = value

    def on_border_color_changed(self, valid: bool) -> None:
        if valid:
            pcfg.drawpanel.shape_border_color = self.borderColorPicker.color.name()

    def set_border_width(self, value: int) -> None:
        with QSignalBlocker(self.borderWidthSlider), QSignalBlocker(self.borderWidthSpinBox):
            self.borderWidthSlider.setValue(value)
            self.borderWidthSpinBox.setValue(value)

    def on_alpha_changed(self, value: int) -> None:
        pcfg.drawpanel.shape_fill_alpha = value

    def on_shape_changed(self) -> None:
        pcfg.drawpanel.shape_fill_shape = self.shapeCombobox.currentData()

    def on_color_changed(self, valid: bool) -> None:
        if valid:
            pcfg.drawpanel.shape_fill_color = self.colorPicker.color.name()


class HelpHintButton(QToolButton):
    """A circled question mark that reads as help at a glance.

    Painted rather than styled, so the circle follows the light/dark theme
    without a stylesheet rule of its own. The outline and the glyph borrow
    the theme colours the rest of the window uses for borders and text.

    >>> HelpHintButton.__name__
    'HelpHintButton'
    """

    DIAMETER = 30
    OUTLINE_WIDTH = 1.4
    GLYPH_POINT_SIZE = 15
    GLYPH_WEIGHT = QFont.Weight.DemiBold
    #: Hover uses the accent the other toggles use.
    HOVER_INK = QColor(30, 147, 229)
    #: Qt hides a tooltip on a timer derived from its length, which a help
    #: card outlives. Qt still hides it as soon as the pointer leaves.
    TOOLTIP_MS = 120000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.DIAMETER, self.DIAMETER)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self._hovered = False

    def outline_ink(self) -> QColor:
        """The colour the rest of the UI draws its outlines in."""
        if self._hovered:
            return QColor(self.HOVER_INK)
        # Read at paint time: parse_stylesheet rewrites these from the
        # active theme, so a light/dark switch needs nothing cached here.
        return borderColor()

    def glyph_ink(self) -> QColor:
        """The colour the rest of the UI draws its text in."""
        if self._hovered:
            return QColor(self.HOVER_INK)
        return QColor(*shared.FOREGROUND_FONTCOLOR)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            # Qt's own timeout is too short to read a table; show it for
            # as long as the pointer stays here instead.
            QToolTip.showText(
                event.globalPos(),
                self.toolTip(),
                self,
                QRect(),
                self.TOOLTIP_MS,
            )
            return True
        return super().event(event)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        QToolTip.hideText()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        # Skip QToolButton's own frame entirely; this is all custom ink.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        inset = self.OUTLINE_WIDTH
        circle = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        painter.setPen(QPen(self.outline_ink(), self.OUTLINE_WIDTH))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawEllipse(circle)
        font = self.font()
        font.setWeight(self.GLYPH_WEIGHT)
        font.setPointSize(self.GLYPH_POINT_SIZE)
        painter.setFont(font)
        painter.setPen(QPen(self.glyph_ink()))
        painter.drawText(circle, Qt.AlignmentFlag.AlignCenter, '?')
        painter.end()


class RectPanel(Widget):
    dilate_ksize_changed = Signal()
    mask_changed = Signal()
    delete_btn_clicked = Signal()
    inpaint_btn_clicked = Signal()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.dilate_label = ToolNameLabel(100, self.tr('Dilate'))
        self.dilate_slider = PaintQSlider()
        self.dilate_slider.setRange(0, 100)
        self.dilate_slider.valueChanged.connect(self.dilate_ksize_changed)
        self.methodComboBox = QComboBox()
        self.methodComboBox.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.methodComboBox.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.methodComboBox.addItems([
            self.tr('method 1'), 
            self.tr('method 2'),
            self.tr('Use Existing Mask')
        ])
        self.methodComboBox.activated.connect(self.on_inpaint_seg_method_changed)
        self.use_mask_checker = QCheckBox(self.tr('Use mask'))
        self.use_mask_checker.setToolTip(self.tr(
            'When unchecked, LLM inpainters receive no mask and the entire rectangle is replaced. '
            'Local inpainters treat the entire rectangle as masked.'
        ))
        self.use_mask_checker.setChecked(pcfg.drawpanel.rectool_use_mask)
        self.use_mask_checker.toggled.connect(self.on_use_mask_changed)
        self.on_use_mask_changed()
        self.autoChecker = QCheckBox(self.tr("Auto"))
        self.autoChecker.setToolTip(self.tr("run inpainting automatically."))
        self.autoChecker.stateChanged.connect(self.on_auto_changed)
        self.inpaint_btn = QPushButton(self.tr("Inpaint"))
        self.inpaint_btn.setToolTip(self.tr("Space"))
        self.inpaint_btn.clicked.connect(self.inpaint_btn_clicked)
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setToolTip(self.tr('Ctrl+D'))
        self.delete_btn.clicked.connect(self.delete_btn_clicked)
        # The blob controls are mouse chords and a bare key, which nothing
        # on screen hints at. A hover card next to the buttons that drive
        # the same preview is where someone would look for them.
        self.blob_help_btn = HelpHintButton(self)
        self.blob_help_btn.setObjectName('BlobHelpButton')
        self.blob_help_btn.setAccessibleName(self.tr('Blob selection controls'))
        self.blob_help_btn.setToolTip(self.blob_help_text())

        self.btnlayout = QHBoxLayout()
        self.btnlayout.addWidget(self.inpaint_btn)
        self.btnlayout.addWidget(self.delete_btn)
        self.btnlayout.addWidget(self.blob_help_btn)

        self.inpainter_selector = InpainterSelectorRow(self)

        glayout = QGridLayout()
        glayout.addWidget(self.dilate_label, 0, 0)
        glayout.addWidget(self.dilate_slider, 0, 1)
        glayout.addWidget(self.autoChecker, 1, 0)
        glayout.addWidget(self.methodComboBox, 1, 1)
        glayout.addWidget(self.use_mask_checker, 2, 0, 1, 2)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.inpainter_selector)
        layout.addLayout(glayout)
        layout.addWidget(self.inpainter_selector.prompt_panel)
        layout.addLayout(self.btnlayout)
        layout.setSpacing(14)

    def blob_help_text(self) -> str:
        """Rich text listing the blob selection controls.

        Each string is translated on its own so a translator never has to
        work around markup, and the rows are escaped before being placed
        in the table.
        """
        title = self.tr('Editing the detected blobs')
        intro = self.tr(
            'While a rectangle preview is waiting to be inpainted, each '
            'detected blob is tinted by whether it is selected.'
        )
        legend = (
            self.tr('Purple: selected, and will be inpainted'),
            self.tr('Cyan: detected but not selected'),
        )
        rows = (
            (self.tr('Left click or drag'), self.tr('Add blobs')),
            (self.tr('Right click or drag'), self.tr('Remove blobs')),
            (
                self.tr('Hold one mouse button, press the other'),
                self.tr(
                    'Stop dragging a box and paint freehand instead. The '
                    'mouse button pressed first still decides add or remove'
                ),
            ),
            (
                self.tr('Shift + click'),
                self.tr('Paint freehand straight away, using only one mouse button'),
            ),
            (
                self.tr('O key'),
                self.tr(
                    'Switch the box rule between catching any blob it '
                    'touches and only blobs mostly inside it'
                ),
            ),
        )
        cells = ''.join(
            f'<tr><td><b>{html.escape(key)}</b>&nbsp;&nbsp;</td>'
            f'<td>{html.escape(action)}</td></tr>'
            for key, action in rows
        )
        legend_html = '<br>'.join(html.escape(line) for line in legend)
        return (
            f'<p><b>{html.escape(title)}</b></p>'
            f'<p>{html.escape(intro)}<br>{legend_html}</p>'
            f'<table cellspacing="0">{cells}</table>'
        )

    def on_inpaint_seg_method_changed(self) -> None:
        pcfg.drawpanel.rectool_method = self.methodComboBox.currentIndex()
        self.mask_changed.emit()

    def on_use_mask_changed(self) -> None:
        use_mask = self.use_mask_checker.isChecked()
        pcfg.drawpanel.rectool_use_mask = use_mask
        hint = '' if use_mask else self.tr('Enable Use mask to change the mask method and dilation.')
        for control in (self.dilate_label, self.dilate_slider, self.methodComboBox):
            control.setEnabled(use_mask)
            control.setToolTip(hint)
        self.mask_changed.emit()

    def on_auto_changed(self):
        if self.autoChecker.isChecked():
            self.inpaint_btn.hide()
            self.delete_btn.hide()
            pcfg.drawpanel.rectool_auto = True
        else:
            pcfg.drawpanel.rectool_auto = False
            self.inpaint_btn.show()
            self.delete_btn.show()

    def auto(self) -> bool:
        return self.autoChecker.isChecked()

    def post_process_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask is None:
            return None
        ksize = self.dilate_slider.value()
        if ksize == 0:
            return mask
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
        return cv2.dilate(mask, element)


class DrawingPanel(Widget):

    scale_tool_pos: QPointF = None
    inpainter_config_requested = Signal()

    def __init__(self, canvas: Canvas, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('DrawingPanel')
        self.module_manager: ModuleManager = None
        self.canvas = canvas
        self.inpaint_stroke: StrokeImgItem = None
        self.rect_inpaint_dict: dict = None
        self.inpaint_mask_array: np.ndarray = None
        self.extracted_imask_array: np.ndarray = None
        self.blob_selection: BlobSelection = None
        self.blob_held: List[str] = []      # buttons down, in press order
        self.blob_mode: str = None          # None | 'box' | 'paint'
        self.blob_paint_add: bool = True
        self.blob_box_add: bool = True
        self.blob_box_anchor = None         # crop coords
        self.blob_box_cur = None
        self.blob_box_rule: str = BLOB_BOX_RULE_DEFAULT

        border_pen = QPen(INPAINT_BRUSH_COLOR, 3, Qt.PenStyle.DashLine)
        self.inpaint_mask_item: PixmapItem = PixmapItem(border_pen)
        # Marquee outline. Parented to the preview pixmap, so its own
        # coordinates are crop pixels and it follows any pan or zoom.
        # Width 0 keeps the line one screen pixel at every zoom level.
        self.blob_box_item = QGraphicsRectItem(self.inpaint_mask_item)
        self.blob_box_item.setPen(QPen(BLOB_BOX_COLOR, 0, Qt.PenStyle.DashLine))
        self.blob_box_item.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.blob_box_item.hide()
        self.scale_circle = QGraphicsEllipseItem()
        self.magic_wand_preview_item = QGraphicsPixmapItem()
        self.magic_wand_preview_item.setData(CONTROL_ITEM_DATA_KEY, True)
        self.magic_wand_preview_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.magic_wand_preview_item.setAcceptHoverEvents(False)
        self.magic_wand_preview_item.setZValue(20)
        self.magic_wand_preview_item.hide()
        self._magic_wand_hover_pos: QPointF | None = None
        self._magic_wand_hover_seed = None
        self._magic_wand_generation = 0
        self._magic_wand_future: Future[tuple[np.ndarray, int, int] | None] | None = None
        self._magic_wand_future_seed = None
        self._magic_wand_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='magic-wand-preview',
        )
        # The worker owns only NumPy data; destruction never waits on it or
        # leaves a callback holding a deleted QWidget wrapper.
        self.destroyed.connect(
            lambda _obj=None, executor=self._magic_wand_executor:
                executor.shutdown(wait=False, cancel_futures=True)
        )
        self._magic_wand_hover_timer = QTimer(self)
        self._magic_wand_hover_timer.setSingleShot(True)
        self._magic_wand_hover_timer.setInterval(40)
        self._magic_wand_hover_timer.timeout.connect(self._apply_magic_wand_hover_preview)
        
        canvas.finish_painting.connect(self.on_finish_painting)
        canvas.finish_erasing.connect(self.on_finish_erasing)
        canvas.ctrl_relesed.connect(self.on_canvasctrl_released)
        canvas.begin_scale_tool.connect(self.on_begin_scale_tool)
        canvas.scale_tool.connect(self.on_scale_tool)
        canvas.end_scale_tool.connect(self.on_end_scale_tool)
        canvas.scalefactor_changed.connect(self.on_canvas_scalefactor_changed)
        canvas.end_create_rect.connect(self.on_end_create_rect)
        canvas.end_create_shape_fill.connect(self.fill_shape)
        canvas.magic_wand_clicked.connect(self.on_magic_wand_clicked)
        canvas.magic_wand_hover.connect(self.on_magic_wand_hover)
        canvas.magic_wand_hover_left.connect(self.clear_magic_wand_preview)
        canvas.image_layers_updated.connect(self._refresh_magic_wand_preview)
        canvas.blob_selection_press.connect(self.on_blob_press)
        canvas.blob_selection_move.connect(self.on_blob_move)
        canvas.blob_selection_release.connect(self.on_blob_release)
        canvas.blob_rule_toggled.connect(self.on_blob_rule_toggled)

        self.currentTool: DrawToolCheckBox = None
        self.handTool = DrawToolCheckBox()
        self.handTool.setObjectName("DrawHandTool")
        self.handTool.checked.connect(self.on_use_handtool)
        self.handTool.stateChanged.connect(self.on_handchecker_changed)
        self.inpaintTool = DrawToolCheckBox()
        self.inpaintTool.setObjectName("DrawInpaintTool")
        self.inpaintTool.checked.connect(self.on_use_inpainttool)
        self.inpaintTool.stateChanged.connect(self.on_brushchecker_changed)
        self.inpaintConfigPanel = InpaintPanel()
        self.inpaintConfigPanel.thicknessChanged.connect(self.setInpaintToolWidth)
        self.inpaintConfigPanel.shapeChanged.connect(self.setInpaintShape)
        self.inpaintConfigPanel.toleranceChanged.connect(self.on_magicwand_tolerance_changed)
        self.inpaintConfigPanel.rangeChanged.connect(self.on_magicwand_range_changed)
        self.inpaintConfigPanel.fillModeChanged.connect(self.on_magicwand_fill_mode_changed)

        self.rectTool = DrawToolCheckBox()
        self.rectTool.setObjectName("DrawRectTool")
        self.rectTool.checked.connect(self.on_use_recttool)
        self.rectTool.stateChanged.connect(self.on_rectchecker_changed)
        self.rectPanel = RectPanel()
        self.rectPanel.inpaint_btn_clicked.connect(self.on_rect_inpaintbtn_clicked)
        self.rectPanel.delete_btn_clicked.connect(self.on_rect_deletebtn_clicked)
        self.rectPanel.dilate_ksize_changed.connect(self.on_rectool_ksize_changed)
        self.rectPanel.mask_changed.connect(self._update_rect_mask)
        for selector_row in self._inpainter_selector_rows():
            selector_row.selection_changed.connect(self.refreshInpainterSelection)
            selector_row.prompt_changed.connect(self.refreshInpainterSelection)
            selector_row.config_requested.connect(
                self.inpainter_config_requested
            )

        self.penTool = DrawToolCheckBox()
        self.penTool.setObjectName("DrawPenTool")
        self.penTool.checked.connect(self.on_use_pentool)
        self.penTool.stateChanged.connect(self.on_brushchecker_changed)
        self.penConfigPanel = PenConfigPanel()
        self.penConfigPanel.thicknessChanged.connect(self.setPenToolWidth)
        self.penConfigPanel.colorChanged.connect(self.setPenToolColor)
        self.penConfigPanel.shapeChanged.connect(self.setPenShape)
        brush_tip = self.tr('Shift-click to draw a straight line. Ctrl+Shift-click constrains it horizontally or vertically.')
        self.penConfigPanel.setToolTip(brush_tip)
        self.inpaintConfigPanel.thickness_row.setToolTip(
            brush_tip + '\n' + self.tr('Hold Ctrl to accumulate brush strokes; release Ctrl to inpaint.')
        )

        self.shapeTool = DrawToolCheckBox()
        self.shapeTool.setObjectName('DrawShapeTool')
        self.shapeTool.setToolTip(self.tr('Shape Fill: left-drag to fill, right-drag to erase'))
        self.shapeTool.setAccessibleName(self.tr('Shape Fill'))
        self.shapeTool.checked.connect(self.on_use_shapetool)
        self.shapeTool.stateChanged.connect(self.on_shapechecker_changed)
        self.shapePanel = ShapeFillPanel()

        toolboxlayout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        toolboxlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        toolboxlayout.addWidget(self.handTool)
        toolboxlayout.addWidget(self.inpaintTool)
        toolboxlayout.addWidget(self.penTool)
        toolboxlayout.addWidget(self.rectTool)
        toolboxlayout.addWidget(self.shapeTool)

        self.canvas.painting_pen = self.pentool_pen = \
            QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.canvas.erasing_pen = self.erasing_pen = QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.inpaint_pen = QPen(INPAINT_BRUSH_COLOR, 1, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        
        # self.setPenToolWidth(10)
        # self.setPenToolColor([0, 0, 0, 127])

        self.toolConfigStackwidget = QStackedWidget()
        self.toolConfigStackwidget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        self.toolConfigStackwidget.addWidget(self.inpaintConfigPanel)
        self.toolConfigStackwidget.addWidget(self.penConfigPanel)
        self.toolConfigStackwidget.addWidget(self.rectPanel)
        self.toolConfigStackwidget.addWidget(self.shapePanel)

        self.maskTransperancySlider = PaintQSlider()
        self.maskTransperancySlider.valueChanged.connect(self.canvas.setMaskTransparencyBySlider)
        masklayout = QHBoxLayout()
        masklayout.addWidget(ToolNameLabel(130, self.tr('Mask Opacity')))
        masklayout.addWidget(self.maskTransperancySlider)

        layout = QVBoxLayout(self)
        layout.addLayout(toolboxlayout)
        layout.addWidget(SeparatorWidget())
        layout.addWidget(self.toolConfigStackwidget)
        layout.addWidget(SeparatorWidget())
        layout.addLayout(masklayout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

    def _inpainter_selector_rows(self):
        return (
            self.inpaintConfigPanel.inpainter_selector,
            self.rectPanel.inpainter_selector,
        )

    def setInpainterOptions(self, options: list[str]) -> None:
        for selector_row in self._inpainter_selector_rows():
            selector_row.setOptions(options)

    def refreshInpainterSelection(self) -> None:
        for selector_row in self._inpainter_selector_rows():
            selector_row.syncFromConfig()

    def setCurrentToolByName(self, tool_name: str):
        try:
            set_method = f'on_use_{tool_name}tool'
            set_method = getattr(self, set_method)
            set_method()
            if self.currentTool is not None:
                self.currentTool.setChecked(True)
        except:
            LOGGER.error(f'{set_method} not found in drawing panel')

    def shortcutSetCurrentToolByName(self, tool_name: str, key=None):
        if key is not None and self.canvas.handle_transform_modal_shortcut(key):
            return
        if self.isVisible():
            self.setCurrentToolByName(tool_name)

    def setShortcutTip(self, tool_name: str, shortcut: str):
        try:
            tool = f'{tool_name}Tool'
            tool: QStackedWidget = getattr(self, tool)
            tool.setToolTip(f'{shortcut}')
        except:
            LOGGER.error(f'{tool} not found in drawing panel')

    def initDLModule(self, module_manager: ModuleManager):
        self.module_manager = module_manager
        module_manager.canvas_inpaint_finished.connect(self.on_inpaint_finished)
        module_manager.canvas_inpaint_failed.connect(self.on_inpaint_failed)

    def setInpaintToolWidth(self, width):
        self.inpaint_pen.setWidthF(width)
        pcfg.drawpanel.inpainter_width = width
        if self.isVisible():
            self.setInpaintCursor()

    def setInpaintShape(self, shape: int) -> None:
        if self.currentTool is self.inpaintTool:
            if self.canvas.painting_shape != shape:
                self.canvas.reset_brush_line()
            if shape == PenShape.MagicWand and self.canvas.painting_shape != shape:
                # An accumulated Ctrl-brush stroke is not a wand selection.
                self.clearInpaintItems()
            self.canvas.painting_shape = shape
        pcfg.drawpanel.inpainter_shape = shape
        self.inpaintConfigPanel.sync_shape_controls()
        if self.isVisible():
            self.setInpaintCursor()
        self._sync_magic_wand_hover_tracking()
        self._refresh_magic_wand_preview()

    def on_magicwand_tolerance_changed(self, value: int) -> None:
        pcfg.drawpanel.magicwand_tolerance = int(value)
        self._refresh_magic_wand_preview()

    def on_magicwand_range_changed(self, value: int) -> None:
        pcfg.drawpanel.magicwand_range = int(value)
        self._refresh_magic_wand_preview()

    def on_magicwand_fill_mode_changed(self, mode: int) -> None:
        pcfg.drawpanel.magicwand_fill_mode = int(mode)
        self._refresh_magic_wand_preview()

    def on_magic_wand_clicked(self, scene_pos: QPointF, erasing: bool) -> None:
        """Select similar colors at the click and inpaint or restore that region.

        Example:
            >>> DrawingPanel.on_magic_wand_clicked.__annotations__['erasing']
            'bool'
        """
        if (
            not self.isVisible()
            or self.canvas.image_edit_mode != ImageEditMode.InpaintTool
            or self.currentTool is not self.inpaintTool
            or not self.inpaintConfigPanel.is_magic_wand()
        ):
            return
        self.clear_magic_wand_preview(clear_pos=False)
        proj = self.canvas.imgtrans_proj
        if not proj.img_valid:
            return
        pos = self.canvas.inpaintLayer.mapFromScene(scene_pos)
        x, y = int(round(pos.x())), int(round(pos.y()))
        img = proj.inpainted_array
        mask = magic_wand_mask(
            img,
            (x, y),
            self.inpaintConfigPanel.toleranceSlider.value(),
            self.inpaintConfigPanel.rangeSlider.value(),
            self.inpaintConfigPanel.fill_mode,
        )
        if mask is None or mask.size == 0 or int(mask.max()) == 0:
            return
        x1, y1, width, height = cv2.boundingRect(mask)
        x2, y2 = x1 + width, y1 + height
        crop_mask = mask[y1:y2, x1:x2]
        inpaint_rect = [x1, y1, x2, y2]
        if erasing:
            origin = proj.img_array[y1:y2, x1:x2]
            inpainted = img[y1:y2, x1:x2]
            existing = proj.mask_array[y1:y2, x1:x2]
            if existing.sum() == 0:
                return
            restore_mask = cv2.bitwise_and(crop_mask, existing)
            if restore_mask.max() == 0:
                return
            blend = (restore_mask > 0)[..., None]
            erased_img = np.where(blend, origin, inpainted)
            remaining_mask = cv2.bitwise_and(existing, cv2.bitwise_not(crop_mask))
            self.canvas.push_undo_command(
                InpaintUndoCommand(self.canvas, erased_img, remaining_mask, inpaint_rect)
            )
            return

        img_h, img_w = img.shape[:2]
        rect_enlarged = enlarge_window(inpaint_rect, img_w, img_h)
        top = y1 - rect_enlarged[1]
        bottom = rect_enlarged[3] - y2
        left = x1 - rect_enlarged[0]
        right = rect_enlarged[2] - x2
        padded = cv2.copyMakeBorder(
            crop_mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        inpaint_rect = rect_enlarged
        crop_img = img[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
        self.runInpaint({
            'img': crop_img,
            'mask': padded,
            'inpaint_rect': inpaint_rect,
        })

    def _sync_magic_wand_hover_tracking(self) -> None:
        enabled = (
            self.isVisible()
            and self.canvas.image_edit_mode == ImageEditMode.InpaintTool
            and self.currentTool is self.inpaintTool
            and self.inpaintConfigPanel.is_magic_wand()
        )
        self.canvas.set_magic_wand_hover_tracking(enabled)
        if not enabled:
            self.clear_magic_wand_preview()

    def on_magic_wand_hover(self, scene_pos: QPointF) -> None:
        if not self.inpaintConfigPanel.is_magic_wand() or self.currentTool is not self.inpaintTool:
            self.clear_magic_wand_preview()
            return
        self._magic_wand_hover_pos = QPointF(scene_pos)
        # Wait for the pointer to settle instead of copying/repainting a full
        # page repeatedly during mouse movement.
        self._magic_wand_hover_timer.start()

    def _refresh_magic_wand_preview(self) -> None:
        self.clear_magic_wand_preview(clear_pos=False)
        if self.canvas._magic_wand_hover_enabled and self._magic_wand_hover_pos is not None:
            self._magic_wand_hover_timer.start()

    def _apply_magic_wand_hover_preview(self) -> None:
        scene_pos = self._magic_wand_hover_pos
        if (
            scene_pos is None
            or not self.canvas._magic_wand_hover_enabled
        ):
            self.clear_magic_wand_preview()
            return
        proj = self.canvas.imgtrans_proj
        if proj is None or not proj.img_valid:
            self.clear_magic_wand_preview()
            return
        pos = self.canvas.inpaintLayer.mapFromScene(scene_pos)
        x, y = int(round(pos.x())), int(round(pos.y()))
        tolerance = self.inpaintConfigPanel.toleranceSlider.value()
        radius = self.inpaintConfigPanel.rangeSlider.value()
        fill_mode = self.inpaintConfigPanel.fill_mode
        seed = (x, y, tolerance, radius, fill_mode, self._magic_wand_generation)
        if seed == self._magic_wand_hover_seed:
            return
        self.magic_wand_preview_item.hide()
        if self._magic_wand_future is not None:
            if not self._magic_wand_future.done():
                self._magic_wand_hover_timer.start()
                return
            future = self._magic_wand_future
            self._magic_wand_future = None
            if self._magic_wand_future_seed == seed:
                self._magic_wand_hover_seed = seed
                try:
                    overlay = future.result()
                except Exception:
                    LOGGER.exception('Failed to calculate magic wand preview.')
                    return
                if overlay is not None:
                    rgba, x1, y1 = overlay
                    self.magic_wand_preview_item.setPixmap(ndarray2pixmap(rgba))
                    self.magic_wand_preview_item.setPos(x1, y1)
                    self.magic_wand_preview_item.setParentItem(self.canvas.baseLayer)
                    self.magic_wand_preview_item.show()
                return
        if proj.inpainted_array is None:
            return
        # Keep one job in flight and sample the latest pointer position when
        # it finishes. Copy before dispatch so edits cannot race the worker.
        self._magic_wand_future_seed = seed
        self._magic_wand_future = self._magic_wand_executor.submit(
            _magic_wand_preview, proj.inpainted_array.copy(), seed[:5],
        )
        self._magic_wand_hover_timer.start()

    def clear_magic_wand_preview(self, clear_pos: bool = True) -> None:
        self._magic_wand_hover_timer.stop()
        self._magic_wand_generation += 1
        self._magic_wand_hover_seed = None
        if clear_pos:
            self._magic_wand_hover_pos = None
        if self.magic_wand_preview_item.scene() is not None:
            self.magic_wand_preview_item.hide()
            self.magic_wand_preview_item.setPixmap(QPixmap())
            if self.magic_wand_preview_item.parentItem() is not None:
                self.magic_wand_preview_item.setParentItem(None)

    def setPenToolWidth(self, width):
        self.pentool_pen.setWidthF(width)
        self.erasing_pen.setWidthF(width)
        pcfg.drawpanel.pentool_width = self.pentool_pen.widthF()
        if self.isVisible():
            self.setPenCursor()

    def setPenToolColor(self, color: Union[QColor, Tuple, List]):
        if not isinstance(color, QColor):
            color = QColor(*color)
        self.pentool_pen.setColor(color)
        pcfg.drawpanel.pentool_color = [color.red(), color.green(), color.blue(), color.alpha()]
        if self.isVisible():
            self.setPenCursor()
        self.penConfigPanel.colorPicker.setPickerColor(color)
        self.penConfigPanel.alphaSlider.setValue(color.alpha())

    def setPenShape(self, shape: int) -> None:
        if self.currentTool is self.penTool and self.canvas.painting_shape != shape:
            self.canvas.reset_brush_line()
        self.setPenCursor()
        self.canvas.painting_shape = shape
        pcfg.drawpanel.pentool_shape = shape

    def on_use_handtool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.handTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.handTool
        pcfg.drawpanel.current_tool = ImageEditMode.HandTool
        self.canvas.clear_canvas_cursor()
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.canvas.image_edit_mode = ImageEditMode.HandTool
        self._sync_magic_wand_hover_tracking()

    def on_use_inpainttool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.inpaintTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.inpaintTool
        pcfg.drawpanel.current_tool = ImageEditMode.InpaintTool
        self.canvas.image_edit_mode = ImageEditMode.InpaintTool
        self.canvas.painting_pen = self.inpaint_pen
        self.canvas.erasing_pen = self.inpaint_pen
        self.canvas.painting_shape = self.inpaintConfigPanel.shape
        self.toolConfigStackwidget.setCurrentWidget(self.inpaintConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setInpaintCursor()
        self._sync_magic_wand_hover_tracking()

    def on_use_pentool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.penTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.penTool
        pcfg.drawpanel.current_tool = ImageEditMode.PenTool
        self.canvas.painting_pen = self.pentool_pen
        self.canvas.painting_shape = self.penConfigPanel.shape
        self.canvas.erasing_pen = self.erasing_pen
        self.canvas.image_edit_mode = ImageEditMode.PenTool
        self.toolConfigStackwidget.setCurrentWidget(self.penConfigPanel)
        if self.isVisible():
            self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setPenCursor()
        self._sync_magic_wand_hover_tracking()

    def on_use_recttool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.rectTool:
            self.currentTool.setChecked(False)
        self.currentTool = self.rectTool
        pcfg.drawpanel.current_tool = ImageEditMode.RectTool
        self.toolConfigStackwidget.setCurrentWidget(self.rectPanel)
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.canvas.image_edit_mode = ImageEditMode.RectTool
        self.setCrossCursor()
        self._sync_magic_wand_hover_tracking()

    def on_use_shapetool(self) -> None:
        if self.currentTool is not None and self.currentTool != self.shapeTool:
            self.canvas.clear_states()
            self.currentTool.setChecked(False)
        self.currentTool = self.shapeTool
        pcfg.drawpanel.current_tool = ImageEditMode.ShapeFillTool
        self.toolConfigStackwidget.setCurrentWidget(self.shapePanel)
        self.canvas.gv.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.canvas.image_edit_mode = ImageEditMode.ShapeFillTool
        self.setCrossCursor()
        self.clearInpaintItems()

    def on_shapechecker_changed(self) -> None:
        if not self.shapeTool.isChecked():
            self.canvas.cancel_shape_fill()

    def on_brushchecker_changed(self) -> None:
        if not self.sender().isChecked():
            # A tool change ends its brush gesture, not the physical Ctrl hold.
            self.canvas.reset_brush_line()
            self.clearInpaintItems()

    def set_config(self, config: DrawPanelConfig) -> None:
        self.refreshInpainterSelection()
        self.setPenToolWidth(config.pentool_width)
        self.setPenToolColor(config.pentool_color)
        self.penConfigPanel.thicknessSlider.setValue(int(config.pentool_width))
        self.penConfigPanel.shapeCombobox.setCurrentIndex(config.pentool_shape)
        
        self.setInpaintToolWidth(config.inpainter_width)
        self.inpaintConfigPanel.thicknessSlider.setValue(int(config.inpainter_width))
        with QSignalBlocker(self.inpaintConfigPanel.shapeCombobox):
            self.inpaintConfigPanel.shapeCombobox.setCurrentIndex(config.inpainter_shape)
        self.setInpaintShape(config.inpainter_shape)
        self.inpaintConfigPanel.toleranceSlider.setValue(config.magicwand_tolerance)
        self.inpaintConfigPanel.rangeSlider.setValue(config.magicwand_range)
        self.inpaintConfigPanel.set_fill_mode(config.magicwand_fill_mode)
        
        self.rectPanel.dilate_slider.setValue(config.recttool_dilate_ksize)
        self.rectPanel.autoChecker.setChecked(config.rectool_auto)
        self.rectPanel.methodComboBox.setCurrentIndex(config.rectool_method)
        self.rectPanel.use_mask_checker.setChecked(config.rectool_use_mask)
        self.shapePanel.shapeCombobox.setCurrentIndex(
            self.shapePanel.shapeCombobox.findData(config.shape_fill_shape)
        )
        self.shapePanel.colorPicker.setPickerColor(QColor(config.shape_fill_color))
        self.shapePanel.alphaSlider.setValue(config.shape_fill_alpha)
        self.shapePanel.borderChecker.setChecked(config.shape_border_enabled)
        self.shapePanel.set_border_width(config.shape_border_width)
        self.shapePanel.borderColorPicker.setPickerColor(QColor(config.shape_border_color))
        if config.current_tool == ImageEditMode.HandTool:
            self.handTool.setChecked(True)
        elif config.current_tool == ImageEditMode.InpaintTool:
            self.inpaintTool.setChecked(True)
        elif config.current_tool == ImageEditMode.PenTool:
            self.penTool.setChecked(True)
        elif config.current_tool == ImageEditMode.RectTool:
            self.rectTool.setChecked(True)
        elif config.current_tool == ImageEditMode.ShapeFillTool:
            self.shapeTool.setChecked(True)

    def get_pen_cursor(self, pen_color: QColor = None, pen_size = None, draw_shape=True, shape=PenShape.Circle) -> QCursor:
        cross_size = 31
        cross_len = cross_size // 4
        thickness = 3
        if pen_color is None:
            pen_color = self.pentool_pen.color()
        if pen_size is None:
            pen_size = self.pentool_pen.width()
        pen_size *= self.canvas.scale_factor
        map_size = max(cross_size+7, pen_size) if draw_shape else cross_size+7
        cursor_center = map_size // 2
        pen_radius = pen_size // 2
        pen_color.setAlpha(127)
        pen = QPen(pen_color, thickness, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        pen.setDashPattern([3, 6])
        if pen_size < 20:
            pen.setStyle(Qt.PenStyle.SolidLine)

        cur_pixmap = QPixmap(QSizeF(map_size, map_size).toSize())
        cur_pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(cur_pixmap)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if draw_shape:
            shape_rect = QRectF(cursor_center-pen_radius + thickness, 
                                cursor_center-pen_radius + thickness, 
                                pen_size - 2*thickness, 
                                pen_size - 2*thickness)
            if shape == PenShape.Circle:
                painter.drawEllipse(shape_rect)
            elif shape == PenShape.Rectangle:
                painter.drawRect(shape_rect)
            elif shape == PenShape.MagicWand:
                pass
            else:
                raise NotImplementedError
            # elif shape == PenShape.Triangle:
                # painter.drawPolygon
        cross_left = (map_size  - 1 - cross_size) // 2 
        cross_right = map_size - cross_left

        pen = QPen(Qt.GlobalColor.white, 5, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        cross_hline0 = QLineF(cross_left, cursor_center, cross_left+cross_len, cursor_center)
        cross_hline1 = QLineF(cross_right-cross_len, cursor_center, cross_right, cursor_center)
        cross_vline0 = QLineF(cursor_center, cross_left, cursor_center, cross_left+cross_len)
        cross_vline1 = QLineF(cursor_center, cross_right-cross_len, cursor_center, cross_right)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        pen.setWidth(3)
        pen.setColor(Qt.GlobalColor.black)
        painter.setPen(pen)
        painter.drawLines([cross_hline0, cross_hline1, cross_vline0, cross_vline1])
        painter.end()
        return QCursor(cur_pixmap)

    def on_incre_pensize(self):
        self.scalePen(1.1)

    def on_decre_pensize(self):
        self.scalePen(0.9)
        pass

    def scalePen(self, scale_factor):
        if self.currentTool == self.penTool:
            val = self.pentool_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.penConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setPenToolWidth(self.penConfigPanel.thicknessSlider.value())

        elif self.currentTool == self.inpaintTool:
            if self.inpaintConfigPanel.is_magic_wand():
                return
            val = self.inpaint_pen.widthF()
            new_val = round(int(val * scale_factor))
            if scale_factor > 1:
                new_val = max(val+1, new_val)
            else:
                new_val = min(val-1, new_val)
            self.inpaintConfigPanel.thicknessSlider.setValue(int(new_val))
            self.setInpaintToolWidth(self.inpaintConfigPanel.thicknessSlider.value())

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.currentTool is not None:
            self.currentTool.setChecked(False)
            self.currentTool.setChecked(True)
        self._sync_magic_wand_hover_tracking()

    def on_finish_painting(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        if not self.canvas.imgtrans_proj.img_valid:
            self.canvas.removeItem(stroke_item)
            return
        if self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if rect is not None:
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg))
            self.canvas.removeItem(stroke_item)
        elif self.currentTool == self.inpaintTool:
            self.inpaint_stroke = stroke_item
            if self.canvas.gv.ctrl_pressed:
                return
            else:
                self.runInpaint()

    def on_finish_erasing(self, stroke_item: StrokeImgItem):
        stroke_item.finishPainting()
        # inpainted-erasing logic is essentially the same as inpainting
        if self.currentTool == self.inpaintTool:
            rect, mask, _ = stroke_item.clip(mask_only=True)
            if mask is None:
                self.canvas.removeItem(stroke_item)
                return
            mask = 255 - mask
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            origin = self.canvas.imgtrans_proj.img_array
            origin = origin[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpainted = self.canvas.imgtrans_proj.inpainted_array
            inpainted = inpainted[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_mask = self.canvas.imgtrans_proj.mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            # no inpainted need to be erased
            if inpaint_mask.sum() == 0:
                self.canvas.removeItem(stroke_item)
                return
            mask = cv2.bitwise_and(mask, inpaint_mask)
            inpaint_mask = np.zeros_like(inpainted)
            inpaint_mask[mask > 0] = 1
            erased_img = inpaint_mask * inpainted + (1 - inpaint_mask) * origin
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, erased_img, mask, inpaint_rect))
            self.canvas.removeItem(stroke_item)

        elif self.currentTool == self.penTool:
            rect, _, qimg = stroke_item.clip()
            if self.canvas.erase_img_key is not None:
                self.canvas.drawingLayer.removeQImage(self.canvas.erase_img_key)
                self.canvas.erase_img_key = None
                self.canvas.stroke_img_item = None
            if rect is not None:
                self.canvas.push_undo_command(StrokeItemUndoCommand(self.canvas.drawingLayer, rect, qimg, True))
        

    def runInpaint(self, inpaint_dict: dict | None = None) -> None:
        if self.currentTool is self.inpaintTool and (
            not self.isVisible()
            or self.canvas.image_edit_mode != ImageEditMode.InpaintTool
        ):
            return

        if inpaint_dict is None:
            if self.inpaint_stroke is None:
                return
            elif self.inpaint_stroke.parentItem() is None:
                logger.warning("inpainting goes wrong")
                self.clearInpaintItems()
                return
                
            rect, mask, _ = self.inpaint_stroke.clip(mask_only=True)
            if mask is None:
                self.clearInpaintItems()
                return
            # we need to enlarge the mask window a bit to get better results
            mask_h, mask_w = mask.shape[:2]
            mask_x, mask_y = rect[0], rect[1]
            img = self.canvas.imgtrans_proj.inpainted_array
            inpaint_rect = [mask_x, mask_y, mask_w + mask_x, mask_h + mask_y]
            rect_enlarged = enlarge_window(inpaint_rect, img.shape[1], img.shape[0])
            top = mask_y - rect_enlarged[1]
            bottom = rect_enlarged[3] - inpaint_rect[3]
            left = mask_x - rect_enlarged[0]
            right = rect_enlarged[2] - inpaint_rect[2]

            mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            inpaint_rect = rect_enlarged
            img = img[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]]
            inpaint_dict = {'img': img, 'mask': mask, 'inpaint_rect': inpaint_rect}

        self.canvas.image_edit_mode = ImageEditMode.NONE
        self._sync_magic_wand_hover_tracking()
        self.canvas.set_canvas_cursor(Qt.CursorShape.WaitCursor)
        self.module_manager.canvas_inpaint(inpaint_dict)

    def on_inpaint_finished(self, inpaint_dict: dict) -> None:
        inpainted = inpaint_dict['inpainted']
        inpaint_rect = inpaint_dict['inpaint_rect']
        mask_array = self.canvas.imgtrans_proj.mask_array
        mask = cv2.bitwise_or(inpaint_dict['mask'], mask_array[inpaint_rect[1]: inpaint_rect[3], inpaint_rect[0]: inpaint_rect[2]])
        self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, inpainted, mask, inpaint_rect))
        self._finish_inpaint()

    def on_inpaint_failed(self) -> None:
        self._finish_inpaint()

    def _finish_inpaint(self) -> None:
        self.clearInpaintItems()
        # Completion may arrive after the user hid the drawing panel or
        # switched tools. Only restore the tool that is still active.
        if self.isVisible() and self.canvas.drawMode():
            if self.currentTool is self.inpaintTool:
                self.canvas.image_edit_mode = ImageEditMode.InpaintTool
                self.setInpaintCursor()
            elif self.currentTool is self.rectTool:
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                self.setCrossCursor()
        self._sync_magic_wand_hover_tracking()

    def on_canvasctrl_released(self) -> None:
        if self.isVisible() and self.currentTool == self.inpaintTool:
            # Releasing Ctrl during a held segment must wait for mouse release.
            if self.canvas.stroke_img_item is not None and self.canvas.stroke_img_item.is_painting:
                return
            self.runInpaint()

    def on_begin_scale_tool(self, pos: QPointF):
        if self.currentTool == self.inpaintTool and self.inpaintConfigPanel.is_magic_wand():
            return
        if self.currentTool == self.penTool:
            circle_pen = QPen(self.pentool_pen)
        elif self.currentTool == self.inpaintTool:
            circle_pen = QPen(self.inpaint_pen)
        else:
            return
        pen_radius = circle_pen.widthF() / 2 * self.canvas.scale_factor
        
        r, g, b, a = circle_pen.color().getRgb()

        circle_pen.setWidth(3)
        circle_pen.setStyle(Qt.PenStyle.DashLine)
        circle_pen.setDashPattern([3, 6])
        self.scale_circle.setPen(circle_pen)
        self.scale_circle.setBrush(QBrush(QColor(r, g, b, 127)))
        self.scale_circle.setPos(pos - QPointF(pen_radius, pen_radius))
        pen_size = 2 * pen_radius
        self.scale_circle.setRect(0, 0, pen_size, pen_size)
        self.scale_tool_pos = pos - QPointF(pen_size, pen_size)
        self.scale_circle.setCursor(self.get_pen_cursor(draw_shape=False))
        self.canvas.addItem(self.scale_circle)
        
    def setCrossCursor(self) -> None:
        if not self.isVisible() or self.currentTool not in (self.rectTool, self.shapeTool):
            return
        self.canvas.set_canvas_cursor(self.get_pen_cursor(draw_shape=False))

    def on_scale_tool(self, pos: QPointF):
        if self.scale_tool_pos is None:
            return
        radius = pos.x() - self.scale_tool_pos.x()
        radius = max(min(radius, MAX_PEN_SIZE * self.canvas.scale_factor), MIN_PEN_SIZE * self.canvas.scale_factor)
        self.scale_circle.setRect(0, 0, radius, radius)

    def on_end_scale_tool(self):
        if self.scale_tool_pos is None:
            return
        circle_size = int(self.scale_circle.rect().width() / self.canvas.scale_factor)
        self.scale_tool_pos = None
        self.canvas.removeItem(self.scale_circle)

        if self.currentTool == self.penTool:
            self.setPenToolWidth(circle_size)
            self.penConfigPanel.thicknessSlider.setValue(circle_size)
            self.setPenCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintToolWidth(circle_size)
            self.inpaintConfigPanel.thicknessSlider.setValue(circle_size)
            self.setInpaintCursor()

    def on_canvas_scalefactor_changed(self):
        if not self.isVisible():
            return
        if self.currentTool == self.penTool:
            self.setPenCursor()
        elif self.currentTool == self.inpaintTool:
            self.setInpaintCursor()

    def setPenCursor(self) -> None:
        if not self.isVisible() or self.currentTool != self.penTool:
            return
        self.canvas.set_canvas_cursor(
            self.get_pen_cursor(shape=self.penConfigPanel.shape)
        )

    def setInpaintCursor(self) -> None:
        if not self.isVisible() or self.currentTool != self.inpaintTool:
            return
        if self.canvas.image_edit_mode == ImageEditMode.NONE:
            self.canvas.set_canvas_cursor(Qt.CursorShape.WaitCursor)
            return
        if self.inpaintConfigPanel.is_magic_wand():
            self.canvas.set_canvas_cursor(magic_wand_cursor())
            return
        self.canvas.set_canvas_cursor(
            self.get_pen_cursor(
                INPAINT_BRUSH_COLOR,
                self.inpaint_pen.width(),
                shape=self.inpaintConfigPanel.shape,
            )
        )

    def on_handchecker_changed(self):
        if self.handTool.isChecked():
            self.toolConfigStackwidget.hide()
        else:
            self.toolConfigStackwidget.show()

    def on_end_create_rect(self, rect: QRectF, mode: int) -> None:
        if self.currentTool == self.rectTool:
            self.canvas.image_edit_mode = ImageEditMode.NONE
            img = self.canvas.imgtrans_proj.inpainted_array
            im_h, im_w = img.shape[:2]

            xyxy = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]
            xyxy = np.array(xyxy)
            xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, im_w - 1)
            xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, im_h - 1)
            x1, y1, x2, y2 = xyxy.astype(np.int64)
            if y2 - y1 < 2 or x2 - x1 < 2:
                self.canvas.image_edit_mode = ImageEditMode.RectTool
                return
            if mode == 0:
                im = np.copy(img[y1: y2, x1: x2])
                self.rect_inpaint_dict = {'img': im, 'inpaint_rect': [x1, y1, x2, y2]}
                self._update_rect_mask()
                self.inpaint_mask_item.setParentItem(self.canvas.baseLayer)
                self.inpaint_mask_item.setPos(x1, y1)
                if self.rectPanel.auto():
                    self.inpaintRect(self.rect_inpaint_dict)
            else:   # erasing
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                erased = self.canvas.imgtrans_proj.img_array[y1: y2, x1: x2]
                self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, erased, mask, [x1, y1, x2, y2]))
                self.canvas.image_edit_mode = ImageEditMode.RectTool
            self.setCrossCursor()

    def fill_shape(self, rect: QRectF, erasing: bool = False) -> None:
        """Commit one filled or erased shape to the existing drawing history.

        >>> DrawingPanel.fill_shape.__annotations__['rect']
        'QRectF'
        """
        if not self.canvas.imgtrans_proj.img_valid:
            return
        rect = rect.normalized()
        bounds = rect.intersected(self.canvas.baseLayer.rect()).toAlignedRect()
        if rect.isEmpty() or bounds.isEmpty():
            return
        config = pcfg.drawpanel
        color = QColor(config.shape_fill_color)
        # Match the pen eraser: fill opacity does not weaken erasing.
        color.setAlpha(255 if erasing else config.shape_fill_alpha)
        # Erasing clears the whole shape, so the border only applies to fills.
        border = not erasing and config.shape_border_enabled
        if color.alpha() == 0 and not border:
            return
        image = shape_fill_image(
            rect, bounds, config.shape_fill_shape, color,
            config.shape_border_width if border else 0, QColor(config.shape_border_color),
        )
        command = StrokeItemUndoCommand(
            self.canvas.drawingLayer, bounds.getRect(), image, erasing,
        )
        command.setText(self.tr('Shape Erase') if erasing else self.tr('Shape Fill'))
        self.canvas.push_undo_command(command)

    def _update_rect_mask(self) -> None:
        if self.rect_inpaint_dict is None:
            return
        request = self.rect_inpaint_dict
        x1, y1, x2, y2 = request['inpaint_rect']
        use_mask = self.rectPanel.use_mask_checker.isChecked()
        method = get_maskseg_method() if use_mask else region_mask
        self.inpaint_mask_array, balloon_mask, info = method(
            request['img'], mask=self.canvas.imgtrans_proj.mask_array[y1:y2, x1:x2],
        )
        request.update(use_mask=use_mask, need_inpaint=info['need_inpaint'],
                       bground_rgb=info['bground_rgb'], ballon_mask=balloon_mask)

        # Only the two detectors produce a mask made of text strokes.
        # 'Use Existing Mask' returns the page mask and region_mask returns
        # a solid block, neither of which has blobs worth selecting, so no
        # editable selection is built for them.
        if use_mask and method in (canny_flood, connected_canny_flood):
            try:
                self.blob_selection = BlobSelection(request['img'], self.inpaint_mask_array)
            except Exception:
                LOGGER.exception('failed to build blob selection')
                self.blob_selection = None
        else:
            self.blob_selection = None
        self.canvas.blob_selection_active = self.blob_selection is not None
        self.reset_blob_gesture()

        self._update_rect_mask_preview()

    def inpaintRect(self, inpaint_dict: dict) -> None:
        img = inpaint_dict['img']
        mask = inpaint_dict['mask']
        need_inpaint = inpaint_dict['need_inpaint']
        bground_rgb = inpaint_dict['bground_rgb']
        ballon_mask = inpaint_dict['ballon_mask']
        # The background fill below paints the balloon interior, not mask, so
        # it cannot honour a blob the user deselected, nor an added blob that
        # reaches outside the balloon (an overflowing SFX). Fall through to
        # the inpainter in those cases; an unedited selection keeps the fill.
        blob_edit_needs_inpainter = self.blob_selection is not None and (
            self.blob_selection.removed_any
            or self.blob_selection.added_outside(ballon_mask)
        )
        if (inpaint_dict['use_mask'] and pcfg.drawpanel.inpainter != LLM_INPAINT_KEY
                and not need_inpaint and pcfg.module.check_need_inpaint
                and not blob_edit_needs_inpainter):
            x1, y1, x2, y2 = inpaint_dict['inpaint_rect']
            if not np.array_equal(img, self.canvas.imgtrans_proj.inpainted_array[y1:y2, x1:x2]):
                LOGGER.info('Draw inpaint result discarded: source image changed, rectangle=%s',
                            inpaint_dict['inpaint_rect'])
                self._finish_inpaint()
                return
            bg_pixel_value = [bground_rgb[ii] for ii in range(3)]
            balloon_areas = np.where(ballon_mask > 0)
            if len(img.shape) == 3 and img.shape[2] == 4:
                avg_alpha = np.mean(img[balloon_areas][..., 3])
                avg_alpha = 0 if avg_alpha < 127 else avg_alpha
                bg_pixel_value.append(avg_alpha)
            bg_pixel_value = np.array(np.round(bg_pixel_value), dtype=np.uint8)
            img[balloon_areas] = bg_pixel_value
            self.canvas.push_undo_command(InpaintUndoCommand(self.canvas, img, mask, inpaint_dict['inpaint_rect'], merge_existing_mask=True))
            LOGGER.info('Draw inpaint completed: background fill, rectangle=%s, use_mask=True',
                        inpaint_dict['inpaint_rect'])
            self._finish_inpaint()
        else:
            self.runInpaint(inpaint_dict=inpaint_dict)

    def on_rect_inpaintbtn_clicked(self):
        if self.rect_inpaint_dict is not None:
            self.inpaintRect(self.rect_inpaint_dict)

    def on_rect_deletebtn_clicked(self) -> None:
        self._finish_inpaint()

    def on_rectool_ksize_changed(self) -> None:
        pcfg.drawpanel.recttool_dilate_ksize = self.rectPanel.dilate_slider.value()
        self._update_rect_mask_preview()

    def rect_mask_source(self) -> np.ndarray:
        """The undilated mask everything else is derived from.

        Once the user has edited the selection that becomes the source of
        truth instead of the detector's raw output. Kept undilated so the
        dilate slider still stacks on top of it.
        """
        if self.blob_selection is not None:
            return self.blob_selection.to_mask()
        return self.inpaint_mask_array

    def build_rect_preview_pixmap(self, mask: np.ndarray) -> QPixmap:
        """The purple overlay, plus cyan for blobs available to click."""
        h, w = mask.shape[:2]
        preview = np.zeros((h, w, 4), dtype=np.uint8)

        if self.blob_selection is not None:
            available = self.blob_selection.available_mask() > 0
            preview[available] = BLOB_AVAILABLE_RGBA

        preview[mask > 0] = BLOB_SELECTED_RGBA
        return ndarray2pixmap(preview)

    def _update_rect_mask_preview(self) -> None:
        if self.rect_inpaint_dict is None:
            return
        source = self.rect_mask_source()
        if source is None:
            return
        mask = (self.rectPanel.post_process_mask(source)
                if self.rect_inpaint_dict['use_mask'] else source)
        self.rect_inpaint_dict['mask'] = mask
        self.inpaint_mask_item.setPixmap(self.build_rect_preview_pixmap(mask))

    # ------------------------------------------------------ blob selection

    def blob_pos(self, scene_pos: QPointF, clamp: bool = False):
        """Scene position -> crop pixel coordinates.

        The preview pixmap sits directly over the crop, so its local
        coordinates ARE crop coordinates. Letting Qt map through the item
        survives any pan, zoom or layer transform, which hand arithmetic on
        the rect origin does not.
        """
        if self.blob_selection is None or self.inpaint_mask_item is None:
            return None
        local = self.inpaint_mask_item.mapFromScene(scene_pos)
        cx, cy = int(local.x()), int(local.y())
        h, w = self.blob_selection.shape
        if clamp:
            return max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
        return (cx, cy) if 0 <= cx < w and 0 <= cy < h else None

    def reset_blob_gesture(self) -> None:
        # getattr: hideEvent can reach clearInpaintItems before __init__ has
        # finished building the marquee item.
        self.blob_held = []
        self.blob_mode = None
        self.blob_box_anchor = None
        self.blob_box_cur = None
        box_item = getattr(self, 'blob_box_item', None)
        if box_item is not None:
            box_item.hide()

    def on_blob_rule_toggled(self) -> None:
        """Switch which blobs a marquee catches, and show the new rule.

        Applies to the next box drawn; it does not re-run the last one.
        """
        self.blob_box_rule = (
            RULE_ANY if self.blob_box_rule == RULE_MAJORITY else RULE_MAJORITY
        )
        if self.blob_box_rule == RULE_ANY:
            message = self.tr('Marquee selects blobs with any pixel inside')
        else:
            message = self.tr(
                'Marquee selects blobs with the majority of pixels inside'
            )
        # The main window is frameless and has no status bar, so the
        # feedback appears next to the pointer instead.
        QToolTip.showText(QCursor.pos(), message, self.canvas.gv)

    def update_blob_box_item(self) -> None:
        """Redraw the marquee outline from the anchor and current corner."""
        if self.blob_box_anchor is None or self.blob_box_cur is None:
            self.blob_box_item.hide()
            return
        ax, ay = self.blob_box_anchor
        cx, cy = self.blob_box_cur
        x0, x1 = sorted((ax, cx))
        y0, y1 = sorted((ay, cy))
        self.blob_box_item.setRect(QRectF(x0, y0, x1 - x0 + 1, y1 - y0 + 1))
        self.blob_box_item.show()

    def on_blob_press(self, scene_pos: QPointF, btn: int, shift: bool) -> None:
        """Left adds, right removes.

        A plain drag draws a marquee; pressing the second button turns the
        gesture into free-hand painting, with the FIRST button still deciding
        add versus remove. Shift starts painting directly, for trackpads that
        cannot chord.
        """
        if self.blob_selection is None or self.rect_inpaint_dict is None:
            return

        first_press = not self.blob_held
        self.blob_held.append('L' if btn == 0 else 'R')

        if first_press:
            self.blob_selection.begin_gesture()
            if shift:
                self.blob_mode = 'paint'
                self.blob_paint_add = (btn == 0)
                self.blob_paint_at(scene_pos)
            else:
                self.blob_mode = 'box'
                self.blob_box_add = (btn == 0)
                self.blob_box_anchor = self.blob_pos(scene_pos, clamp=True)
                self.blob_box_cur = self.blob_box_anchor
                self.update_blob_box_item()
        elif self.blob_mode == 'box':
            # The pending box applied nothing, so there is nothing to undo.
            self.blob_mode = 'paint'
            self.blob_paint_add = (self.blob_held[0] == 'L')
            self.blob_box_anchor = self.blob_box_cur = None
            self.blob_box_item.hide()
            self.blob_paint_at(scene_pos)

    def on_blob_move(self, scene_pos: QPointF) -> None:
        if self.blob_selection is None:
            return
        if self.blob_mode == 'box':
            self.blob_box_cur = self.blob_pos(scene_pos, clamp=True)
            self.update_blob_box_item()
        elif self.blob_mode == 'paint':
            self.blob_paint_at(scene_pos)

    def on_blob_release(self, scene_pos: QPointF, btn: int) -> None:
        if self.blob_selection is None:
            self.reset_blob_gesture()
            return

        tag = 'L' if btn == 0 else 'R'
        if tag in self.blob_held:
            self.blob_held.remove(tag)

        if self.blob_mode == 'box':
            self.commit_blob_box()
        self.reset_blob_gesture()

    def blob_paint_at(self, scene_pos: QPointF) -> None:
        """Free-hand painting. Blobs already touched this gesture are
        skipped, so sweeping back over one does not flip it again."""
        pos = self.blob_pos(scene_pos)
        if pos is None:
            return
        if self.blob_selection.paint(pos[0], pos[1], self.blob_paint_add):
            self._update_rect_mask_preview()

    def commit_blob_box(self) -> None:
        anchor, cur = self.blob_box_anchor, self.blob_box_cur
        if anchor is None or cur is None:
            return

        x0, x1 = sorted((anchor[0], cur[0]))
        y0, y1 = sorted((anchor[1], cur[1]))

        # A box with no drag in it is just a click on one blob.
        if x1 - x0 <= 1 and y1 - y0 <= 1:
            if self.blob_selection.click(x0, y0, add=self.blob_box_add):
                self._update_rect_mask_preview()
            return

        if self.blob_selection.box(x0, y0, x1, y1,
                                   add=self.blob_box_add,
                                   rule=self.blob_box_rule):
            self._update_rect_mask_preview()

    def on_rectchecker_changed(self):
        if not self.rectTool.isChecked():
            self.clearInpaintItems()

    def hideEvent(self, e) -> None:
        self.canvas.reset_brush_line()
        self.canvas.cancel_shape_fill()
        self.canvas.set_magic_wand_hover_tracking(False)
        self.clear_magic_wand_preview()
        self.clearInpaintItems()
        return super().hideEvent(e)

    def clearInpaintItems(self) -> None:

        self.rect_inpaint_dict = None
        self.inpaint_mask_array = None
        self.blob_selection = None
        self.canvas.blob_selection_active = False
        self.reset_blob_gesture()
        if self.inpaint_mask_item is not None:
            if self.inpaint_mask_item.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_mask_item)
            
        if self.inpaint_stroke is not None:
            if self.inpaint_stroke.scene() == self.canvas:
                self.canvas.removeItem(self.inpaint_stroke)
            self.inpaint_stroke = None
        self._sync_magic_wand_hover_tracking()

    def handle_page_changed(self) -> None:
        self.clear_magic_wand_preview()
        self._finish_inpaint()
