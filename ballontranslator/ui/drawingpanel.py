from __future__ import annotations

from qtpy.QtCore import Signal, Qt, QPointF, QSize, QSizeF, QLineF, QRectF, QSignalBlocker, QTimer, QThread
from qtpy.QtWidgets import QAbstractSpinBox, QGridLayout, QPushButton, QComboBox, QPlainTextEdit, QToolButton, QSizePolicy, QBoxLayout, QCheckBox, QHBoxLayout, QGraphicsView, QSpinBox, QStackedWidget, QVBoxLayout, QLabel, QGraphicsPixmapItem, QGraphicsEllipseItem
from qtpy.QtGui import QIcon, QPen, QColor, QCursor, QPainter, QPixmap, QBrush, QFontMetrics, QImage

from typing import Union, Tuple, List, Optional, Callable
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil, log
import numpy as np
import cv2

from ballontranslator.utils.imgproc_utils import (
    enlarge_window,
    magic_wand_mask,
    magic_wand_preview_overlay,
    MagicWandFillMode,
    MAGIC_WAND_PREVIEW_RGBA,
)
from ballontranslator.utils.textblock_mask import canny_flood, connected_canny_flood, region_mask
from ballontranslator.utils.logger import logger
from .cursor import magic_wand_cursor
from .funcmaps import get_maskseg_method
from .module_manager import ModuleManager
from .module_tool_button import ModuleSelectionMenu, ModuleSelectionToolButton
from .llm_modality import LLM_MODALITY_IMAGE
from .image_edit import ImageEditMode, PenShape, PixmapItem, StrokeImgItem, shape_fill_image
from .custom_widget import Widget, SeparatorWidget, PaintQSlider, ColorPickerLabel
from .canvas import Canvas
from .misc import ndarray2pixmap, pixmap2ndarray, themed_icon_path
from .icon_rendering import render_svg_pixmap
from ballontranslator.utils.config import DrawPanelConfig, pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, LLM_INPAINT_KEY, profile_by_id
from ballontranslator.utils.shared import CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from ballontranslator.utils.logger import logger as LOGGER
from .drawing_commands import InpaintUndoCommand, StrokeItemUndoCommand
from .text_engine.shape_control import CONTROL_ITEM_DATA_KEY

INPAINT_BRUSH_COLOR = QColor(127, 0, 127, 127)
MAX_PEN_SIZE = 1000
MIN_PEN_SIZE = 1
TOOLNAME_POINT_SIZE = 13
MAGICWAND_TOLERANCE_MAX = 255
MAGICWAND_RANGE_LIMIT = 50
REGION_ADD_COLOR = MAGIC_WAND_PREVIEW_RGBA
REGION_CLEAR_COLOR = (255, 180, 40, 110)
REGION_REJECT_COLOR = (245, 85, 110, 110)


class MaskRegionIndex:
    """Index four-connected on/off components for one immutable mask snapshot.

    Built in the preview worker and reused until the page/mask changes. Component
    statistics bound overlay work; pointer movement only reads a label by index.

    >>> index = MaskRegionIndex(np.pad(np.zeros((2, 2), np.uint8), 1, constant_values=255))
    >>> index.at(1, 1)
    (False, 1)
    """

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        binary = (mask > 0).astype(np.uint8)
        _, self.on_labels, self.on_stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)
        _, self.off_labels, self.off_stats, _ = cv2.connectedComponentsWithStats(1 - binary, connectivity=4)
        self.edge_labels = frozenset(int(label) for label in np.unique(np.concatenate((
            self.off_labels[0], self.off_labels[-1],
            self.off_labels[:, 0], self.off_labels[:, -1],
        ))) if label)

    def at(self, x: int, y: int) -> tuple[bool, int] | None:
        height, width = self.mask.shape
        if not (0 <= x < width and 0 <= y < height):
            return None
        erasing = bool(self.mask[y, x])
        return erasing, int((self.on_labels if erasing else self.off_labels)[y, x])

    def touched(self, rect: List[int], brush_mask: np.ndarray, erasing: bool) -> frozenset[int]:
        x, y = rect[:2]
        height, width = self.mask.shape
        brush_height, brush_width = brush_mask.shape
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + brush_width, width), min(y + brush_height, height)
        if x1 >= x2 or y1 >= y2:
            return frozenset()
        brush = brush_mask[y1 - y:y2 - y, x1 - x:x2 - x] > 0
        labels = (self.on_labels if erasing else self.off_labels)[y1:y2, x1:x2]
        return frozenset(int(label) for label in np.unique(labels[brush]) if label)

    def crop(self, selected: frozenset[int], erasing: bool) -> tuple[np.ndarray, int, int] | None:
        if not selected:
            return None
        stats = (self.on_stats if erasing else self.off_stats)[list(selected)]
        x1, y1 = stats[:, :2].min(axis=0)
        x2, y2 = (stats[:, :2] + stats[:, 2:4]).max(axis=0)
        labels = (self.on_labels if erasing else self.off_labels)[y1:y2, x1:x2]
        return np.isin(labels, tuple(selected)), int(x1), int(y1)

    def preview(
        self, selected: frozenset[int], erasing: bool,
    ) -> tuple[np.ndarray, int, int, str] | None:
        rejected = frozenset() if erasing else selected & self.edge_labels
        crop = self.crop(rejected or selected, erasing)
        if crop is None:
            return None
        region, x, y = crop
        status = 'clear' if erasing else ('reject' if rejected else 'add')
        color = REGION_CLEAR_COLOR if erasing else (REGION_REJECT_COLOR if rejected else REGION_ADD_COLOR)
        rgba = np.zeros((*region.shape, 4), np.uint8)
        rgba[region] = color
        if rejected:
            # Show the actual page edges that caused rejection, even when the
            # rejected region spans the whole image and surrounds valid holes.
            height, width = self.mask.shape
            for edge, touches in (
                (rgba[:3], y == 0), (rgba[-3:], y + region.shape[0] == height),
                (rgba[:, :3], x == 0), (rgba[:, -3:], x + region.shape[1] == width),
            ):
                if touches:
                    edge[edge[..., 3] > 0] = (*REGION_REJECT_COLOR[:3], 255)
        return rgba, x, y, status


def _mask_region_preview(
    index: MaskRegionIndex, selected: frozenset[int], erasing: bool,
    samples: list[tuple[List[int], np.ndarray]],
) -> tuple[frozenset[int], tuple[np.ndarray, int, int, str] | None]:
    for rect, brush in samples:
        selected |= index.touched(rect, brush, erasing)
    return selected, index.preview(selected, erasing)


def _expanded_inpaint_payload(
    image: np.ndarray,
    region_mask: np.ndarray,
) -> Optional[dict]:
    ys, xs = np.where(region_mask > 0)
    if len(ys) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    image_height, image_width = image.shape[:2]
    inpaint_rect = [x1, y1, x2, y2]
    expanded = enlarge_window(inpaint_rect, image_width, image_height)
    local_mask = region_mask[y1:y2, x1:x2]
    local_mask = cv2.copyMakeBorder(
        local_mask,
        y1 - expanded[1],
        expanded[3] - y2,
        x1 - expanded[0],
        expanded[2] - x2,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    image_crop = image[expanded[1]:expanded[3], expanded[0]:expanded[2]]
    return {
        'img': np.copy(image_crop),
        'mask': local_mask,
        'inpaint_rect': expanded,
    }


def compute_forward_flood_fill(
    rect: List[int],
    brush_mask: np.ndarray,
    full_mask: np.ndarray,
    image: np.ndarray,
    region_index: MaskRegionIndex | None = None,
) -> Optional[dict]:
    """Expand the project mask from brush-touched background components.

    The brush supplies seeds. Existing mask pixels provide the boundaries,
    so an unbounded background component is rejected instead of filling the
    rest of the page.

    >>> mask = np.full((5, 5), 255, np.uint8); mask[1:4, 1:4] = 0
    >>> result = compute_forward_flood_fill([2, 2, 3, 3], np.ones((1, 1), np.uint8) * 255, mask, np.zeros((5, 5, 3), np.uint8))
    >>> result is not None and bool(result['mask'].any())
    True
    """
    mask_x, mask_y = rect[0], rect[1]
    mask_height, mask_width = brush_mask.shape[:2]
    image_height, image_width = full_mask.shape[:2]
    x1, y1 = max(mask_x, 0), max(mask_y, 0)
    x2 = min(mask_x + mask_width, image_width)
    y2 = min(mask_y + mask_height, image_height)
    if x1 >= x2 or y1 >= y2:
        return None
    brush_roi = brush_mask[
        y1 - mask_y:y2 - mask_y,
        x1 - mask_x:x2 - mask_x,
    ]
    current_roi = full_mask[y1:y2, x1:x2]
    seed = ((brush_roi > 0) & (current_roi == 0)).astype(np.uint8) * 255
    if not np.any(seed):
        return None

    index = region_index or MaskRegionIndex(full_mask)
    seed_labels = index.touched(rect, brush_mask, False)
    if not seed_labels or seed_labels & index.edge_labels:
        return None
    crop, cx, cy = index.crop(seed_labels, False)
    new_region = np.zeros_like(full_mask)
    new_region[cy:cy + crop.shape[0], cx:cx + crop.shape[1]] = crop.astype(np.uint8) * 255

    updated_mask = cv2.bitwise_or(full_mask, new_region)
    _, updated_labels = cv2.connectedComponents(
        (updated_mask > 0).astype(np.uint8), connectivity=4
    )
    updated_roi = updated_labels[y1:y2, x1:x2]
    inpaint_labels = set(np.unique(updated_roi[seed > 0])) - {0}
    brush_on_mask = cv2.bitwise_and(brush_roi, current_roi)
    if np.any(brush_on_mask):
        inpaint_labels |= set(np.unique(updated_roi[brush_on_mask > 0])) - {0}
    if not inpaint_labels:
        return None
    region = np.isin(updated_labels, tuple(inpaint_labels)).astype(np.uint8) * 255
    return _expanded_inpaint_payload(image, region)


def compute_reverse_flood_fill(
    rect: List[int],
    brush_mask: np.ndarray,
    full_mask: np.ndarray,
    original_image: np.ndarray,
    inpainted_image: np.ndarray,
    region_index: MaskRegionIndex | None = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """Restore every connected mask component touched by the brush.

    >>> mask = np.zeros((3, 3), np.uint8); mask[1, 1] = 255
    >>> result = compute_reverse_flood_fill([1, 1, 2, 2], np.array([[255]], np.uint8), mask, np.zeros((3, 3, 3), np.uint8), np.ones((3, 3, 3), np.uint8))
    >>> result[2]
    [1, 1, 2, 2]
    """
    index = region_index or MaskRegionIndex(full_mask)
    seed_labels = index.touched(rect, brush_mask, True)
    crop = index.crop(seed_labels, True)
    if crop is None:
        return None
    region, ex1, ey1 = crop
    ex2, ey2 = ex1 + region.shape[1], ey1 + region.shape[0]
    erase_rect = [ex1, ey1, ex2, ey2]
    local_region = region.astype(np.uint8) * 255
    original_crop = original_image[ey1:ey2, ex1:ex2]
    inpainted_crop = inpainted_image[ey1:ey2, ex1:ex2]
    selector = local_region > 0
    erased_image = np.copy(inpainted_crop)
    erased_image[selector] = original_crop[selector]
    new_mask = cv2.bitwise_and(
        full_mask[ey1:ey2, ex1:ex2], cv2.bitwise_not(local_region)
    )
    return erased_image, new_mask, erase_rect


class FloodFillThread(QThread):
    """Run mask connected-component calculations outside the Qt main thread."""

    result_ready = Signal(str, object)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._mode = ''
        self._operation: Optional[Callable] = None
        self._arguments: tuple = ()

    def configure(
        self,
        mode: str,
        operation: Callable,
        arguments: tuple,
    ) -> None:
        if self.isRunning():
            raise RuntimeError('flood-fill calculation is already running')
        self._mode = mode
        self._operation = operation
        self._arguments = arguments

    def run(self) -> None:
        mode = self._mode
        operation = self._operation
        arguments = self._arguments
        try:
            result = operation(*arguments)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            self._operation = None
            self._arguments = ()
        self.result_ready.emit(mode, result)


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

        self.region_mode_row = Widget(self)
        region_layout = QHBoxLayout(self.region_mode_row)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_label = ToolNameLabel(None, self.tr('Mask Region Mode'))
        region_layout.addWidget(region_label)
        self.regionModeCheckBox = QCheckBox(self.region_mode_row)
        region_layout.addWidget(self.regionModeCheckBox)
        region_layout.addStretch(1)
        region_tip = self.tr(
            'Fill or clear connected mask regions.\n'
            'Toggle: Shift\n'
            'Fill: left-drag; clear: right-drag.\n'
            'Regions touching an image edge cannot be filled.'
        )
        self.region_mode_row.setToolTip(region_tip)
        region_label.setToolTip(region_tip)
        self.regionModeCheckBox.setToolTip(region_tip)
        self.inpainter_selector = InpainterSelectorRow(self)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.inpainter_selector)
        layout.addLayout(shape_layout)
        layout.addWidget(self.inpainter_selector.prompt_panel)
        layout.addWidget(self.thickness_row)
        layout.addWidget(self.region_mode_row)
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
        self.region_mode_row.setVisible(not wand)
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
        self.btnlayout = QHBoxLayout()
        self.btnlayout.addWidget(self.inpaint_btn)
        self.btnlayout.addWidget(self.delete_btn)

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
        self._flood_fill_page = None
        self._flood_fill_active = False
        self._flood_fill_mask_snapshot: np.ndarray | None = None
        self._flood_fill_image_snapshot: np.ndarray | None = None
        self.flood_fill_thread = FloodFillThread(self)
        self.flood_fill_thread.result_ready.connect(self.on_flood_fill_result)
        self.flood_fill_thread.failed.connect(self.on_flood_fill_failed)

        border_pen = QPen(INPAINT_BRUSH_COLOR, 3, Qt.PenStyle.DashLine)
        self.inpaint_mask_item: PixmapItem = PixmapItem(border_pen)
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
        
        self.region_preview_item = QGraphicsPixmapItem()
        self.region_preview_item.setData(CONTROL_ITEM_DATA_KEY, True)
        self.region_preview_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.region_preview_item.setAcceptHoverEvents(False)
        self.region_preview_item.setZValue(20)
        self.region_preview_item.hide()
        self._region_index: MaskRegionIndex | None = None
        self._region_index_future: Future[MaskRegionIndex] | None = None
        self._region_preview_future: Future[tuple[frozenset[int], tuple[np.ndarray, int, int, str] | None]] | None = None
        self._region_generation = 0
        self._region_request: tuple[int, bool | None, bool, frozenset[int]] | None = None
        self._region_hover_pos: QPointF | None = None
        self._region_direction: bool | None = None
        self._region_selected: frozenset[int] = frozenset()
        self._region_samples: list[tuple[List[int], np.ndarray]] = []
        self._region_display_key: tuple[bool, frozenset[int]] | None = None
        self._region_cursor_status = 'base'
        self._region_preview_timer = QTimer(self)
        self._region_preview_timer.setSingleShot(True)
        self._region_preview_timer.setInterval(20)
        self._region_preview_timer.timeout.connect(self._apply_mask_region_preview)
        canvas.mask_region_hover.connect(self.on_mask_region_hover)
        canvas.mask_region_hover_left.connect(self.hide_mask_region_preview)
        canvas.mask_region_stroke_started.connect(self.on_mask_region_stroke_started)
        canvas.mask_region_stroke_updated.connect(self.on_mask_region_stroke_updated)
        canvas.brush_gesture_cancelled.connect(self.on_brush_gesture_cancelled)
        canvas.image_layers_updated.connect(self.invalidate_mask_regions)

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
        self.inpaintConfigPanel.regionModeCheckBox.toggled.connect(self.set_mask_region_mode)
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
            self.tr('Press Shift to toggle Mask Region Mode. Hold Ctrl to accumulate brush strokes; release Ctrl to inpaint.')
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
            if (
                self.canvas.painting_shape != shape
                and (
                    shape == PenShape.MagicWand
                    or self.canvas.painting_shape == PenShape.MagicWand
                    or pcfg.drawpanel.mask_region_mode
                )
            ):
                # An accumulated stroke belongs to the brush type that made it.
                self.clearInpaintItems()
            self.canvas.painting_shape = shape
        pcfg.drawpanel.inpainter_shape = shape
        self.inpaintConfigPanel.sync_shape_controls()
        if self.isVisible():
            self.setInpaintCursor()
        self._sync_magic_wand_hover_tracking()
        self._refresh_magic_wand_preview()

    def set_mask_region_mode(self, enabled: bool) -> None:
        if pcfg.drawpanel.mask_region_mode == enabled:
            return
        if self.currentTool is self.inpaintTool and self.canvas.image_edit_mode == ImageEditMode.NONE:
            with QSignalBlocker(self.inpaintConfigPanel.regionModeCheckBox):
                self.inpaintConfigPanel.regionModeCheckBox.setChecked(pcfg.drawpanel.mask_region_mode)
            return
        pcfg.drawpanel.mask_region_mode = enabled
        with QSignalBlocker(self.inpaintConfigPanel.regionModeCheckBox):
            self.inpaintConfigPanel.regionModeCheckBox.setChecked(enabled)
        self.canvas.reset_brush_line()
        self.clearInpaintItems()
        # This slider represents displayed mask opacity, not saved mask alpha.
        self.maskTransperancySlider.setValue(20)
        self._sync_mask_region_tracking()
        self.setInpaintCursor()
        if self.canvas._mask_region_hover_enabled:
            self.on_mask_region_hover(self.canvas.scene_cursor_pos())

    def toggle_mask_region_mode(self) -> None:
        if self.isVisible() and self.canvas.mask_region_available:
            self.set_mask_region_mode(not pcfg.drawpanel.mask_region_mode)

    def _sync_mask_region_tracking(self) -> None:
        self.inpaintConfigPanel.regionModeCheckBox.setEnabled(
            self.currentTool is not self.inpaintTool or self.canvas.image_edit_mode != ImageEditMode.NONE)
        enabled = (self.isVisible() and self.currentTool is self.inpaintTool
                   and self.canvas.mask_region_mode)
        if self.canvas._mask_region_hover_enabled == enabled:
            return
        self.canvas.set_mask_region_hover_tracking(enabled)
        if enabled:
            self._region_hover_pos = self.canvas.scene_cursor_pos()
            self._region_preview_timer.start()
        else:
            self.canvas._restore_viewport_cursor()
            self.invalidate_mask_regions()

    def _set_mask_region_cursor(self, status: str) -> None:
        if status != self._region_cursor_status:
            self._region_cursor_status = status
            self.setInpaintCursor()

    def _update_mask_region_cursor(self) -> None:
        hover = self._mask_region_hover_selection()
        if self._region_index is None:
            # The current mask pixel is cheap to inspect before the async index
            # is ready. Component boundaries still wait for the worker.
            status = 'base'
            if self._region_hover_pos is not None:
                pos = self.canvas.inpaintLayer.mapFromScene(self._region_hover_pos)
                x, y = int(round(pos.x())), int(round(pos.y()))
                mask = self.canvas.imgtrans_proj.mask_array
                if mask is not None and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    on_mask = bool(mask[y, x])
                    if self._region_direction is not None and self._region_direction != on_mask:
                        status = 'none'
                    elif on_mask:
                        status = 'clear'
        elif hover is None:
            status = 'none'
        else:
            on_mask, selected = hover
            erasing = self._region_direction
            if erasing is not None and erasing != on_mask:
                status = 'none'
            elif on_mask:
                status = 'clear'
            else:
                rejected = (selected | self._region_selected) & self._region_index.edge_labels
                status = 'reject' if rejected else 'add'
        self._set_mask_region_cursor(status)

    def on_mask_region_hover(self, scene_pos: QPointF) -> None:
        if not self.canvas._mask_region_hover_enabled:
            return
        self._region_hover_pos = QPointF(scene_pos)
        self._update_mask_region_cursor()
        if not self._region_preview_timer.isActive():
            self._region_preview_timer.start()

    def hide_mask_region_preview(self) -> None:
        self._region_hover_pos = None
        self.region_preview_item.hide()
        self._region_display_key = None
        if self._region_direction is None:
            self._region_preview_timer.stop()

    def on_brush_gesture_cancelled(self) -> None:
        if self.inpaint_stroke is not None:
            if self.inpaint_stroke.scene() is self.canvas:
                self.canvas.removeItem(self.inpaint_stroke)
            self.inpaint_stroke = None
        self.clear_mask_region_gesture()

    def clear_mask_region_gesture(self) -> None:
        self._region_hover_pos = None
        self._region_direction = None
        self._region_selected = frozenset()
        self._region_samples.clear()
        self._region_display_key = None
        self.region_preview_item.hide()
        self._region_preview_timer.stop()
        if self._region_preview_future is not None:
            self._region_preview_future.cancel()
            self._region_preview_future = None

    def invalidate_mask_regions(self) -> None:
        hover_pos = self._region_hover_pos
        self._region_generation += 1
        self._region_index = None
        if self._region_index_future is not None:
            self._region_index_future.cancel()
            self._region_index_future = None
        self.clear_mask_region_gesture()
        self._region_hover_pos = hover_pos
        if self.canvas._mask_region_hover_enabled and hover_pos is not None:
            self._region_preview_timer.start()

    def on_mask_region_stroke_started(self, erasing: bool) -> None:
        hover_pos = self._region_hover_pos
        if self._region_direction != erasing:
            # An accumulated add stroke cannot become an erase stroke.
            self.clearInpaintItems()
        self._region_direction = erasing
        self._region_hover_pos = hover_pos
        self.region_preview_item.hide()
        self._update_mask_region_cursor()

    def on_mask_region_stroke_updated(self, rect: QRectF) -> None:
        stroke = self.canvas.stroke_img_item
        if stroke is None or not self.canvas.mask_region_mode:
            return
        bounds = rect.toAlignedRect().intersected(stroke._img.rect())
        if bounds.isEmpty():
            return
        # Copy only the dirty brush rectangle; workers never see a live QImage
        # or scan/copy the full page on every mouse movement.
        rgba = pixmap2ndarray(stroke._img.copy(bounds), True)
        brush = np.ascontiguousarray(rgba[..., -1])
        self._region_samples.append(([bounds.x(), bounds.y()], brush))
        if not self._region_preview_timer.isActive():
            self._region_preview_timer.start()

    def _mask_region_hover_selection(self) -> tuple[bool, frozenset[int]] | None:
        if self._region_hover_pos is None or self._region_index is None:
            return None
        pos = self.canvas.inpaintLayer.mapFromScene(self._region_hover_pos)
        region = self._region_index.at(int(round(pos.x())), int(round(pos.y())))
        if region is None:
            return None
        erasing, label = region
        return erasing, frozenset((label,))

    def _apply_mask_region_preview(self) -> None:
        if not self.canvas._mask_region_hover_enabled or not self.canvas.mask_region_mode:
            return
        project = self.canvas.imgtrans_proj
        if project is None or not project.img_valid or project.mask_array is None:
            self.hide_mask_region_preview()
            return
        if self._region_index is None:
            if self._region_index_future is None:
                self._region_index_future = self._magic_wand_executor.submit(
                    MaskRegionIndex, np.copy(project.mask_array),
                )
            elif self._region_index_future.done():
                future = self._region_index_future
                self._region_index_future = None
                try:
                    self._region_index = future.result()
                except Exception:
                    LOGGER.exception('Failed to index mask regions.')
                    return
            if self._region_index is None:
                self._region_preview_timer.start()
                return
        if self._region_preview_future is not None:
            if not self._region_preview_future.done():
                self._region_preview_timer.start()
                return
            future = self._region_preview_future
            self._region_preview_future = None
            generation, direction, erasing, requested = self._region_request
            try:
                selected, overlay = future.result()
            except Exception:
                LOGGER.exception('Failed to preview mask regions.')
                return
            if generation == self._region_generation and direction == self._region_direction:
                if direction is not None:
                    self._region_selected = selected
                if direction is not None or self._mask_region_hover_selection() == (erasing, requested):
                    if overlay is not None:
                        rgba, x, y, status = overlay
                        self.region_preview_item.setPixmap(ndarray2pixmap(rgba))
                        self.region_preview_item.setPos(x, y)
                        self.region_preview_item.setParentItem(self.canvas.baseLayer)
                        self.region_preview_item.show()
                    else:
                        self.region_preview_item.hide()
                    self._region_display_key = (erasing, selected)
        self._update_mask_region_cursor()
        if self._region_direction is None:
            hover = self._mask_region_hover_selection()
            if hover is None:
                self.region_preview_item.hide()
                self._region_display_key = None
                return
            erasing, selected = hover
            if self._region_display_key != (erasing, selected):
                self.region_preview_item.hide()
        else:
            erasing, selected = self._region_direction, self._region_selected
        if not self._region_samples and self._region_display_key == (erasing, selected):
            return
        samples, self._region_samples = self._region_samples, []
        self._region_request = (self._region_generation, self._region_direction, erasing, selected)
        self._region_preview_future = self._magic_wand_executor.submit(
            _mask_region_preview, self._region_index, selected, erasing, samples,
        )
        self._region_preview_timer.start()

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
        self._sync_mask_region_tracking()

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
        with QSignalBlocker(self.inpaintConfigPanel.regionModeCheckBox):
            self.inpaintConfigPanel.regionModeCheckBox.setChecked(config.mask_region_mode)
        
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

    def get_pen_cursor(self, pen_color: QColor = None, pen_size: float | None = None, draw_shape: bool = True, shape: int = PenShape.Circle, region_action: str | None = None) -> QCursor:
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
        if region_action is not None:
            # Extend only the bottom/right: the brush center stays the hotspot.
            icon_left = cursor_center + pen_radius + 4
            icon_top = icon_left
            map_size = max(map_size, ceil(icon_left + 37))
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
        if region_action is not None:
            if region_action == 'reject':
                # A dedicated prohibition sign replaces the add marker.
                center = QPointF(icon_left + 14, icon_top + 14)
                for color, width in ((QColor(Qt.GlobalColor.white), 5),
                                     (QColor(*REGION_REJECT_COLOR[:3]), 3)):
                    painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine))
                    painter.drawEllipse(center, 10, 10)
                    painter.drawLine(QLineF(center + QPointF(-7, 7), center + QPointF(7, -7)))
            elif region_action != 'none':
                bucket = render_svg_pixmap(themed_icon_path('cursor_mask_region.svg'), 28, 28, 1.0)
                painter.drawPixmap(int(icon_left), int(icon_top), bucket)
                if region_action in ('add', 'clear'):
                    color = REGION_CLEAR_COLOR if region_action == 'clear' else REGION_ADD_COLOR
                    badge_x, badge_y = icon_left + 25, icon_top + 25
                    lines = [QLineF(badge_x - 4, badge_y, badge_x + 4, badge_y)]
                    if region_action == 'add':
                        lines.append(QLineF(badge_x, badge_y - 4, badge_x, badge_y + 4))
                    for stroke_color, width in ((QColor(Qt.GlobalColor.white), 5), (QColor(*color[:3]), 3)):
                        painter.setPen(QPen(stroke_color, width, Qt.PenStyle.SolidLine))
                        painter.drawLines(lines)
        else:
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
        if region_action is not None:
            return QCursor(cur_pixmap, int(cursor_center), int(cursor_center))
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

    def on_finish_painting(self, stroke_item: StrokeImgItem) -> None:
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
            if self.canvas.mask_region_mode:
                self._flood_fill_forward(stroke_item)
            else:
                self.runInpaint()

    def on_finish_erasing(self, stroke_item: StrokeImgItem) -> None:
        stroke_item.finishPainting()
        # inpainted-erasing logic is essentially the same as inpainting
        if self.currentTool == self.inpaintTool:
            if self.canvas.mask_region_mode:
                self.inpaint_stroke = stroke_item
                self._flood_fill_reverse(stroke_item)
                return
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
            if self.canvas.mask_region_mode:
                if self.inpaint_stroke is not None:
                    self._flood_fill_forward(self.inpaint_stroke)
            else:
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
        cursor = self.get_pen_cursor(
            INPAINT_BRUSH_COLOR,
            self.inpaint_pen.width(),
            shape=self.inpaintConfigPanel.shape,
            region_action=self._region_cursor_status if self.canvas.mask_region_mode else None,
        )
        self.canvas.set_canvas_cursor(cursor)
        if self.canvas.mask_region_mode:
            # Qt may defer applying a scene item's new cursor until another
            # mouse move. Button presses must change the operation immediately.
            self.canvas.gv.viewport().setCursor(cursor)

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
        self._update_rect_mask_preview()

    def inpaintRect(self, inpaint_dict: dict) -> None:
        img = inpaint_dict['img']
        mask = inpaint_dict['mask']
        need_inpaint = inpaint_dict['need_inpaint']
        bground_rgb = inpaint_dict['bground_rgb']
        ballon_mask = inpaint_dict['ballon_mask']
        if (inpaint_dict['use_mask'] and pcfg.drawpanel.inpainter != LLM_INPAINT_KEY
                and not need_inpaint and pcfg.module.check_need_inpaint):
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

    def _update_rect_mask_preview(self) -> None:
        if self.rect_inpaint_dict is None or self.inpaint_mask_array is None:
            return
        mask = (self.rectPanel.post_process_mask(self.inpaint_mask_array)
                if self.rect_inpaint_dict['use_mask'] else self.inpaint_mask_array)
        self.rect_inpaint_dict['mask'] = mask
        user_preview_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        user_preview_mask[:, :, [0, 2, 3]] = (mask[:, :, np.newaxis] / 2).astype(np.uint8)
        self.inpaint_mask_item.setPixmap(ndarray2pixmap(user_preview_mask))

    def _start_flood_fill(
        self,
        mode: str,
        operation: Callable,
        arguments: tuple,
    ) -> None:
        if self.flood_fill_thread.isRunning():
            LOGGER.info('Ignoring flood fill while another fill is running.')
            self.clearInpaintItems()
            return
        self._flood_fill_page = self.canvas.imgtrans_proj.current_img
        self._flood_fill_active = True
        self._flood_fill_mask_snapshot = arguments[2]
        self._flood_fill_image_snapshot = arguments[4] if mode == 'reverse' else arguments[3]
        self.flood_fill_thread.configure(mode, operation, arguments)
        # Disable editing until the snapshot result is accepted or discarded.
        self.canvas.image_edit_mode = ImageEditMode.NONE
        self.canvas.set_canvas_cursor(Qt.CursorShape.WaitCursor)
        self._sync_mask_region_tracking()
        self.flood_fill_thread.start()

    def on_flood_fill_result(self, mode: str, result: object) -> None:
        project = self.canvas.imgtrans_proj
        same_page = (
            self._flood_fill_active
            and self._flood_fill_page == project.current_img
        )
        source_matches = (
            same_page
            and np.array_equal(self._flood_fill_mask_snapshot, project.mask_array)
            and np.array_equal(self._flood_fill_image_snapshot, project.inpainted_array)
        )
        if not source_matches:
            LOGGER.info('Discarding flood fill because its page or source changed.')
            if same_page:
                self._finish_inpaint()
            return
        if result is None:
            LOGGER.info('%s flood fill produced no bounded mask region.', mode)
            self._finish_inpaint()
            return
        if mode == 'reverse':
            erased_image, new_mask, erase_rect = result
            self.canvas.push_undo_command(InpaintUndoCommand(
                self.canvas,
                erased_image,
                new_mask,
                erase_rect,
            ))
            self._finish_inpaint()
            return
        self.clearInpaintItems()
        self.module_manager.canvas_inpaint(result)

    def on_flood_fill_failed(self, message: str) -> None:
        LOGGER.warning('Flood fill failed: %s', message)
        if (
            self._flood_fill_active
            and self._flood_fill_page == self.canvas.imgtrans_proj.current_img
        ):
            self._finish_inpaint()

    def _flood_fill_forward(self, stroke_item: StrokeImgItem) -> None:
        """Queue expansion of background regions touched by the brush."""
        rect, brush_mask, _ = stroke_item.clip(mask_only=True)
        if brush_mask is None:
            self.clearInpaintItems()
            return
        project = self.canvas.imgtrans_proj
        self._start_flood_fill(
            'forward',
            compute_forward_flood_fill,
            (
                list(rect),
                np.copy(brush_mask),
                np.copy(project.mask_array),
                np.copy(project.inpainted_array),
                self._region_index,
            ),
        )

    def _flood_fill_reverse(self, stroke_item: StrokeImgItem) -> None:
        """Queue restoration of mask components touched by the brush."""
        rect, brush_mask, _ = stroke_item.clip(mask_only=True)
        if brush_mask is None:
            self.clearInpaintItems()
            return
        project = self.canvas.imgtrans_proj
        self._start_flood_fill(
            'reverse',
            compute_reverse_flood_fill,
            (
                list(rect),
                np.copy(brush_mask),
                np.copy(project.mask_array),
                np.copy(project.img_array),
                np.copy(project.inpainted_array),
                self._region_index,
            ),
        )

    def on_rectchecker_changed(self):
        if not self.rectTool.isChecked():
            self.clearInpaintItems()

    def hideEvent(self, e) -> None:
        self.canvas.reset_brush_line()
        self.canvas.cancel_shape_fill()
        self.canvas.set_magic_wand_hover_tracking(False)
        self.canvas.set_mask_region_hover_tracking(False)
        self.invalidate_mask_regions()
        self.clear_magic_wand_preview()
        self.clearInpaintItems()
        return super().hideEvent(e)

    def clearInpaintItems(self) -> None:
        if self.module_manager is not None:
            self.module_manager.cancel_canvas_inpaint()

        self.clear_mask_region_gesture()
        self._flood_fill_active = False
        self._flood_fill_mask_snapshot = None
        self._flood_fill_image_snapshot = None
        self.rect_inpaint_dict = None
        self.inpaint_mask_array = None
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
        self.invalidate_mask_regions()
        self._finish_inpaint()
