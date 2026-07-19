import copy
import sys
from typing import List

from qtpy.QtWidgets import QLineEdit, QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QApplication, QPushButton, QLabel, QGroupBox, QCheckBox, QSlider
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QFocusEvent, QMouseEvent, QTextCursor, QKeyEvent, QFont

from ballontranslator.utils import shared
from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import FontFormat, px2pt, LineSpacingType, normalize_text_transform
from .custom_widget import Widget, ColorPickerLabel, ClickableLabel, CheckableLabel, TextCheckerLabel, AlignmentChecker, QFontChecker, SizeComboBox, SizeControlLabel
from .textitem import TextBlkItem
from .text_advanced_format import TextAdvancedFormatPanel
from .text_style_presets import TextStylePresetPanel
from .textedit_commands import SetTextTransformCommand
from . import shared_widget as SW
from . import funcmaps as FM

TEXT_TRANSFORM_FIELDS = (
    'horizontal_scale',
    'vertical_scale',
    'slant_angle',
    'glyph_slant_angle',
)


class LineEdit(QLineEdit):

    return_pressed_wochange = Signal()
    return_pressed = Signal()

    def __init__(self, content: str = None, parent = None):
        super().__init__(content, parent)
        self.textChanged.connect(self.on_text_changed)
        self._text_changed = False
        self.editingFinished.connect(self.on_editing_finished)
        # self.returnPressed.connect(self.on_return_pressed)

    def on_text_changed(self):
        self._text_changed = True

    def on_editing_finished(self):
        self._text_changed = False

    def focusOutEvent(self, e: QFocusEvent) -> None:
        self._text_changed = False
        return super().focusOutEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        super().keyPressEvent(e)
        if e.key() == Qt.Key.Key_Return:
            self.return_pressed.emit()
            if not self._text_changed:
                self.return_pressed_wochange.emit()


class IncrementalBtn(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(13, 13)


class AlignmentBtnGroup(QFrame):
    param_changed = Signal(str, int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignLeftChecker = AlignmentChecker(self)
        self.alignLeftChecker.clicked.connect(self.alignBtnPressed)
        self.alignCenterChecker = AlignmentChecker(self)
        self.alignCenterChecker.clicked.connect(self.alignBtnPressed)
        self.alignRightChecker = AlignmentChecker(self)
        self.alignRightChecker.clicked.connect(self.alignBtnPressed)
        self.alignLeftChecker.setObjectName("AlignLeftChecker")
        self.alignRightChecker.setObjectName("AlignRightChecker")
        self.alignCenterChecker.setObjectName("AlignCenterChecker")

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.alignLeftChecker)
        hlayout.addWidget(self.alignCenterChecker)
        hlayout.addWidget(self.alignRightChecker)
        hlayout.setSpacing(0)

    def alignBtnPressed(self):
        btn = self.sender()
        if btn == self.alignLeftChecker:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 0)
        elif btn == self.alignRightChecker:
            self.alignRightChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignLeftChecker.setChecked(False)
            self.param_changed.emit('alignment', 2)
        else:
            self.alignCenterChecker.setChecked(True)
            self.alignLeftChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 1)
    
    def setAlignment(self, alignment: int):
        if alignment == 0:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
        elif alignment == 1:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(True)
            self.alignRightChecker.setChecked(False)
        else:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(True)


class FormatGroupBtn(QFrame):
    param_changed = Signal(str, bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boldBtn = QFontChecker(self)
        self.boldBtn.setObjectName("FontBoldChecker")
        self.boldBtn.clicked.connect(self.setBold)
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.italicBtn.clicked.connect(self.setItalic)
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.underlineBtn.clicked.connect(self.setUnderline)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.boldBtn)
        hlayout.addWidget(self.italicBtn)
        hlayout.addWidget(self.underlineBtn)
        hlayout.setSpacing(0)

    def setBold(self):
        self.param_changed.emit('bold', self.boldBtn.isChecked())

    def setItalic(self):
        self.param_changed.emit('italic', self.italicBtn.isChecked())

    def setUnderline(self):
        self.param_changed.emit('underline', self.underlineBtn.isChecked())
    

class FontSizeBox(QFrame):
    param_changed = Signal(str, float)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.fcombobox = SizeComboBox([1, 1000], 'font_size', self)
        self.fcombobox.addItems([
            "5", "5.5", "6.5", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", '22', "26", "28",
            "36", "48", "56", "72", "93", "123", "163"
        ])
        self.fcombobox.param_changed.connect(self.param_changed)

        hlayout = QHBoxLayout(self)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.upBtn)
        vlayout.addWidget(self.downBtn)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        hlayout.addLayout(vlayout)
        hlayout.addWidget(self.fcombobox)
        hlayout.setSpacing(3)
        hlayout.setContentsMargins(0, 0, 0, 0)

    def getFontSize(self) -> str:
        return self.fcombobox.currentText()

    def onUpBtnClicked(self):
        raito = 1.25
        size = self.getFontSize()
        multi_size=False
        if "+" in size:
            size = size.strip("+")
            multi_size=True
        size = float(size)
        newsize = int(round(size * raito))
        if newsize == size:
            newsize += 1
        newsize = min(1000, newsize)
        if newsize != size:
            if not multi_size:
                self.param_changed.emit('font_size', newsize)
                self.fcombobox.setCurrentText(str(newsize))
            else:
                self.param_changed.emit('rel_font_size', raito)
                self.fcombobox.setCurrentText(str(newsize)+"+")

    def onDownBtnClicked(self):
        raito = 0.75
        size = self.getFontSize()
        multi_size=False
        if "+" in size:
            size = size.strip("+")
            multi_size=True
        size = float(size)
        newsize = int(round(size * raito))
        if newsize == size:
            newsize -= 1
        newsize = max(1, newsize)
        if newsize != size:
            if not multi_size:
                self.param_changed.emit('font_size', newsize)
                self.fcombobox.setCurrentText(str(newsize))
            else:
                self.param_changed.emit('rel_font_size', raito)
                self.fcombobox.setCurrentText(str(newsize)+"+")
    

class FontFamilyComboBox(QFontComboBox):
    param_changed = Signal(str, object)
    def __init__(self, emit_if_focused=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentFontChanged.connect(self.on_fontfamily_changed)
        self.lineedit = lineedit = LineEdit(parent=self)
        lineedit.return_pressed.connect(self.on_return_pressed)
        self.setLineEdit(lineedit)
        self.emit_if_focused = emit_if_focused
        self.return_pressed = False
        
    def apply_fontfamily(self):
        ffamily = self.currentFont().family()
        if ffamily in shared.FONT_FAMILIES:
            self.param_changed.emit('font_family', ffamily)

    def update_font_list(self, font_list):
        self.currentFontChanged.disconnect(self.on_fontfamily_changed)
        current_font = self.currentFont().family()
        self.clear()
        self.addItems(font_list)

        # If the current font is not in the list, use the first available font
        if current_font not in font_list:
            if font_list:  # If the list is not empty, use the first one.
                current_font = list(font_list)[0]
            else:  # Don't add anything if the list is empty.
                self.currentFontChanged.connect(self.on_fontfamily_changed)
                return
    
        self.setCurrentText(current_font)
        self.currentFontChanged.connect(self.on_fontfamily_changed)

    def on_return_pressed(self):
        self.return_pressed = True
        self.apply_fontfamily()

    def on_fontfamily_changed(self):
        if self.return_pressed:
            self.return_pressed = False
        else:
            self.apply_fontfamily()


class FontFormatPanel(Widget):
    
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    global_format: FontFormat = None
    restoring_textblk: bool = False

    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app
        self._transform_items = []
        self._transform_drag_before = None
        self._transform_drag_param = None

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.familybox = FontFamilyComboBox(emit_if_focused=True, parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_param_changed)
        self.familybox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.fontsizebox = FontSizeBox(self)
        self.fontsizebox.setToolTip(self.tr("Font Size"))
        self.fontsizebox.setObjectName("FontSizeBox")
        self.fontsizebox.fcombobox.setToolTip(self.tr("Change font size"))
        self.fontsizebox.param_changed.connect(self.on_param_changed)
        
        self.lineSpacingLabel = SizeControlLabel(self, direction=1, transparent_bg=False)
        self.lineSpacingLabel.setObjectName("lineSpacingLabel")
        self.lineSpacingLabel.size_ctrl_changed.connect(self.onLineSpacingCtrlChanged)
        self.lineSpacingLabel.btn_released.connect(lambda : self.on_param_changed('line_spacing', self.lineSpacingBox.value()))

        self.lineSpacingBox = SizeComboBox([0, 100], 'line_spacing', self)
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.param_changed.connect(self.on_param_changed)

        linesp_hlayout = QHBoxLayout()
        linesp_hlayout.addWidget(self.lineSpacingLabel)
        linesp_hlayout.addWidget(self.lineSpacingBox)
        linesp_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)
        
        self.colorPicker = ColorPickerLabel(self, param_name='frgb')
        self.colorPicker.setToolTip(self.tr("Change font color"))
        self.colorPicker.changingColor.connect(self.changingColor)
        self.colorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.colorPicker.apply_color.connect(self.on_apply_color)

        self.alignBtnGroup = AlignmentBtnGroup(self)
        self.alignBtnGroup.param_changed.connect(self.on_param_changed)

        self.formatBtnGroup = FormatGroupBtn(self)
        self.formatBtnGroup.param_changed.connect(self.on_param_changed)

        self.verticalChecker = QFontChecker(self)
        self.verticalChecker.setObjectName("FontVerticalChecker")
        self.verticalChecker.clicked.connect(lambda : self.on_param_changed('vertical', self.verticalChecker.isChecked()))

        self.strokeWidthBox = SizeComboBox([0, 10], 'stroke_width', self)
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.param_changed.connect(self.on_param_changed)

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        font = self.fontStrokeLabel.font()
        font.setPointSizeF(shared.CONFIG_FONTSIZE_CONTENT * 0.95)
        self.fontStrokeLabel.setFont(font)
        self.fontStrokeLabel.size_ctrl_changed.connect(self.strokeWidthBox.changeByDelta)
        self.fontStrokeLabel.btn_released.connect(lambda : self.on_param_changed('stroke_width', self.strokeWidthBox.value()))
        
        self.strokeColorPicker = ColorPickerLabel(self, param_name='srgb')
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.strokeColorPicker.apply_color.connect(self.on_apply_color)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.letterSpacingBox = SizeComboBox([0, 10], "letter_spacing", self)
        self.letterSpacingBox.addItems(["0.0"])
        self.letterSpacingBox.setToolTip(self.tr("Change letter spacing"))
        self.letterSpacingBox.setMinimumWidth(int(self.letterSpacingBox.height() * 2.5))
        self.letterSpacingBox.param_changed.connect(self.on_param_changed)

        self.letterSpacingLabel = SizeControlLabel(self, direction=0, transparent_bg=False)
        self.letterSpacingLabel.setObjectName("letterSpacingLabel")
        self.letterSpacingLabel.size_ctrl_changed.connect(self.letterSpacingBox.changeByDelta)
        self.letterSpacingLabel.btn_released.connect(lambda : self.on_param_changed('letter_spacing', self.letterSpacingBox.value()))

        lettersp_hlayout = QHBoxLayout()
        lettersp_hlayout.addWidget(self.letterSpacingLabel)
        lettersp_hlayout.addWidget(self.letterSpacingBox)
        lettersp_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.angleBox = SizeComboBox([-180, 180], "angle", self)
        self.angleBox.addItems(["0", "90", "180", "-90"])
        self.angleBox.setToolTip(self.tr("Angle"))
        self.angleBox.setMinimumWidth(int(self.angleBox.height() * 2.5))
        self.angleBox.param_changed.connect(self.on_param_changed)

        self.angleLabel = SizeControlLabel(self, direction=0, transparent_bg=False)
        self.angleLabel.setObjectName("fontAngleLabel")
        self.angleLabel.setToolTip(self.tr("Angle"))
        self.angleLabel.size_ctrl_changed.connect(self.onAngleCtrlChanged)
        self.angleLabel.btn_released.connect(lambda : self.on_param_changed('angle', self.angleBox.value()))

        angle_hlayout = QHBoxLayout()
        angle_hlayout.addWidget(self.angleLabel)
        angle_hlayout.addWidget(self.angleBox)
        angle_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.textstyle_panel = TextStylePresetPanel(
            self.global_fontfmt_str,
            config_name='show_text_style_preset',
            config_expand_name='expand_tstyle_panel'
        )
        self.textstyle_panel.active_text_style_label_changed.connect(self.on_active_textstyle_label_changed)
        self.textstyle_panel.active_stylename_edited.connect(self.on_active_stylename_edited)

        self.textadvancedfmt_panel = TextAdvancedFormatPanel(
            self.tr('Advanced Text Format'),
            config_name='text_advanced_format_panel',
            config_expand_name='expand_tadvanced_panel',
            on_format_changed=self.on_param_changed
        )
        self.textadvancedfmt_panel.transform_commit_requested.connect(
            self.on_text_transform_commit
        )
        self.textadvancedfmt_panel.transform_preview_requested.connect(
            self.on_text_transform_preview
        )
        self.textadvancedfmt_panel.transform_drag_commit_requested.connect(
            self.on_text_transform_drag_commit
        )
        self.textadvancedfmt_panel.transform_preview_canceled.connect(
            self.on_text_transform_cancel
        )
        color_label = self.textadvancedfmt_panel.shadow_group.color_label
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)

        color_label = self.textadvancedfmt_panel.gradient_group.start_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        color_label = self.textadvancedfmt_panel.gradient_group.end_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        self.foldTextBtn = CheckableLabel(self.tr("Unfold"), self.tr("Fold"), False)
        self.sourceBtn = TextCheckerLabel(self.tr("Source"))
        self.transBtn = TextCheckerLabel(self.tr("Translation"))

        FONTFORMAT_SPACING = 6

        vl0 = QVBoxLayout()
        vl0.addWidget(self.textstyle_panel.view_widget)
        vl0.addWidget(self.textadvancedfmt_panel.view_widget)
        vl0.setSpacing(0)
        vl0.setContentsMargins(0, 0, 0, 0)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.fontsizebox)
        hl1.addLayout(angle_hlayout)
        hl1.setSpacing(4)
        hl1.setContentsMargins(0, 12, 0, 0)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.setSpacing(FONTFORMAT_SPACING)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.addLayout(linesp_hlayout)
        hl3.setContentsMargins(3, 0, 3, 0)
        hl3.setSpacing(13)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setContentsMargins(0, 12, 0, 0)
        hl4.setSpacing(0)

        self.vlayout.addLayout(vl0)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.setContentsMargins(0, 0, 7, 0)
        self.vlayout.setSpacing(0)

        self.focusOnColorDialog = False
        C.active_format = self.global_format

    def global_mode(self):
        return id(C.active_format) == id(self.global_format)
    
    def active_text_style_label(self):
        return self.textstyle_panel.active_text_style_label

    def active_text_style_format(self):
        af = self.active_text_style_label()
        if af is not None:
            return af.fontfmt
        else:
            return None

    def on_param_changed(self, param_name: str, value):
        func = FM.handle_ffmt_change.get(param_name)
        func_kwargs = {}
        if param_name in {'font_size', 'rel_font_size'}:
            func_kwargs['clip_size'] = True
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True, **func_kwargs)
            self.update_text_style_label()
        else:
            func(param_name, value, C.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True, **func_kwargs)

    @staticmethod
    def _transform_with_value(transform, param_name, value):
        values = dict(zip(TEXT_TRANSFORM_FIELDS, transform))
        values[param_name] = value
        return normalize_text_transform(
            values['horizontal_scale'],
            values['vertical_scale'],
            values['slant_angle'],
            values['glyph_slant_angle'],
        )

    def _sync_text_transform_overlays(self):
        SW.canvas.sync_text_overlays()

    def _refresh_text_transform_controls(self, refresh_shape=True):
        if self._transform_items:
            self.textadvancedfmt_panel.set_transform_items(self._transform_items)
            if len(self._transform_items) == 1 and C.active_format is not None:
                transform = self._transform_items[0].blk.fontformat.text_transform
                for name, value in zip(TEXT_TRANSFORM_FIELDS, transform):
                    setattr(C.active_format, name, value)
        else:
            active_format = (
                self.global_format if self.global_mode() else C.active_format
            )
            if active_format is None:
                return
            for name, control in self.textadvancedfmt_panel.transform_controls.items():
                control.set_model_value(getattr(active_format, name))
        if refresh_shape:
            self._sync_text_transform_overlays()

    def on_text_transform_commit(self, param_name: str, value: float):
        if not self._transform_items:
            before = self.global_format.text_transform
            after = self._transform_with_value(before, param_name, value)
            if before != after:
                for name, component in zip(TEXT_TRANSFORM_FIELDS, after):
                    setattr(self.global_format, name, component)
                self.update_text_style_label()
            self._refresh_text_transform_controls(refresh_shape=False)
            return

        before = [item.blk.fontformat.text_transform for item in self._transform_items]
        after = [
            self._transform_with_value(transform, param_name, value)
            for transform in before
        ]
        command = SetTextTransformCommand.create(
            self._transform_items,
            before,
            after,
            self._refresh_text_transform_controls,
        )
        if command is not None:
            SW.canvas.push_undo_command(command)
        else:
            self._refresh_text_transform_controls(refresh_shape=False)

    def on_text_transform_preview(self, param_name: str, canonical_delta: float):
        if not self._transform_items:
            if (
                self._transform_drag_param != param_name
                or self._transform_drag_before is None
            ):
                self._transform_drag_param = param_name
                self._transform_drag_before = [self.global_format.text_transform]
            return
        if (
            self._transform_drag_param != param_name
            or self._transform_drag_before is None
        ):
            # Starting a drag must not refresh the controls: the emitting
            # control owns its cumulative display delta until release/Escape.
            # Only clear a genuinely older item preview if a different control
            # somehow begins a new session.
            if self._transform_drag_before is not None:
                for item in self._transform_items:
                    item.clear_text_transform_preview()
            self._transform_drag_param = param_name
            self._transform_drag_before = [
                item.blk.fontformat.text_transform for item in self._transform_items
            ]
        field_index = TEXT_TRANSFORM_FIELDS.index(param_name)
        preview_after = [
            self._transform_with_value(
                transform,
                param_name,
                transform[field_index] + canonical_delta,
            )
            for transform in self._transform_drag_before
        ]
        changed_items = []
        for item, transform in zip(self._transform_items, preview_after):
            # A drag can point farther out while the canonical value is already
            # clamped at its limit. Do not even enter the item transform path
            # unless the effective preview would actually change.
            if item._effective_text_transform() == transform:
                continue
            if item.set_text_transform(*transform, preview=True):
                changed_items.append(item)
        if changed_items:
            self._sync_text_transform_overlays()

    def on_text_transform_drag_commit(self, param_name: str, canonical_delta: float):
        if (
            self._transform_drag_param != param_name
            or self._transform_drag_before is None
        ):
            return
        field_index = TEXT_TRANSFORM_FIELDS.index(param_name)
        before = self._transform_drag_before
        after = [
            self._transform_with_value(
                transform,
                param_name,
                transform[field_index] + canonical_delta,
            )
            for transform in before
        ]
        items = list(self._transform_items)
        self._transform_drag_before = None
        self._transform_drag_param = None
        if not items:
            global_before = before[0]
            global_after = after[0]
            if global_before != global_after:
                for name, component in zip(TEXT_TRANSFORM_FIELDS, global_after):
                    setattr(self.global_format, name, component)
                self.update_text_style_label()
            self._refresh_text_transform_controls(refresh_shape=False)
            return
        command = SetTextTransformCommand.create(
            items, before, after, self._refresh_text_transform_controls
        )
        if command is None:
            geometry_changed = False
            for item in items:
                geometry_changed = (
                    item.clear_text_transform_preview() or geometry_changed
                )
            if geometry_changed:
                self._sync_text_transform_overlays()
        else:
            SW.canvas.push_undo_command(command)

    def on_text_transform_cancel(self, param_name=None):
        geometry_changed = False
        if self._transform_drag_before is not None:
            for item in self._transform_items:
                geometry_changed = (
                    item.clear_text_transform_preview() or geometry_changed
                )
        self._transform_drag_before = None
        self._transform_drag_param = None
        if not self._transform_items:
            self._refresh_text_transform_controls(refresh_shape=False)
            return
        if geometry_changed:
            self._sync_text_transform_overlays()

    def _cancel_text_transform_previews(self):
        for control in self.textadvancedfmt_panel.transform_controls.values():
            control.cancel_preview()
        # Keep programmatic preview callers safe even when no control owns the
        # session and therefore emitted no preview_canceled signal above.
        if self._transform_drag_before is not None:
            self.on_text_transform_cancel(self._transform_drag_param)

    def resolve_text_transform_edits_for_save(self):
        """Resolve transient transform editors before snapshot and render."""
        # Typed input is complete form state, while a drag whose mouse button is
        # still down remains a preview under the existing selection-boundary
        # policy. Commit the former and cancel the latter before either save
        # consumer runs.
        self.textadvancedfmt_panel.finish_pending_transform_edits()
        self._cancel_text_transform_previews()

    def resolve_text_transform_edits_for_history_change(self):
        """Cancel an active preview before moving the application undo stack."""
        # Pending text remains owned by its line editor. A held drag, however,
        # must not survive the model refresh performed by undo/redo.
        self._cancel_text_transform_previews()

    def resolve_text_transform_edits_for_page_change(self):
        """End old-page transform ownership before its scene is discarded."""
        self.resolve_text_transform_edits_for_save()
        self._detach_text_transform_scene_owner()

    def cancel_text_transform_edits_for_scene_change(self):
        """Discard transient transform state before scene items are removed."""
        for control in self.textadvancedfmt_panel.transform_controls.values():
            control.cancel_pending()
            control.cancel_preview()
        self.on_text_transform_cancel(self._transform_drag_param)
        self._detach_text_transform_scene_owner()

    def _detach_text_transform_scene_owner(self):
        # A format control can keep the old local owner while it has focus.
        # Scene replacement is a real ownership boundary, so bypass that policy.
        if self.textblk_item is not None:
            self.textblk_item.fontformat = copy.deepcopy(C.active_format)
        self.textblk_item = None
        self._transform_items = []
        self.set_active_format(self.global_format)
        self.set_globalfmt_title()

    def update_text_style_label(self):
        if self.global_mode():
            active_text_style_label = self.active_text_style_label()
            if active_text_style_label is not None:
                active_text_style_label.update_style(self.global_format)

    def changingColor(self):
        self.focusOnColorDialog = True

    def onColorLabelChanged(self, is_valid=True):
        self.focusOnColorDialog = False
        if is_valid:
            sender: ColorPickerLabel = self.sender()
            rgb = sender.rgb()
            self.on_param_changed(sender.param_name, rgb)

    def on_apply_color(self, param_name, rgb):
        self.on_param_changed(param_name, rgb)

    def onLineSpacingCtrlChanged(self, delta: int):
        if C.active_format.line_spacing_type == LineSpacingType.Distance:
            mul = 0.1
        else:
            mul = 0.01
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * mul)

    def onAngleCtrlChanged(self, delta: int):
        self.angleBox.setValue(round(self.angleBox.value()) + delta)

    def set_active_format(self, font_format: FontFormat, multi_size=False):
        C.active_format = font_format
        self.familybox.blockSignals(True)
        font_size = round(font_format.font_size, 1)
        if int(font_size) == font_size:
            font_size = str(int(font_size))
        else:
            font_size = f'{font_size:.1f}'
        if multi_size:
            font_size += "+"
        self.fontsizebox.fcombobox.setCurrentText(font_size)
        self.familybox.setCurrentText(font_format.font_family)
        self.familybox.setCurrentFont(QFont(font_format.font_family))
        self.colorPicker.setPickerColor(font_format.foreground_color())
        self.strokeColorPicker.setPickerColor(font_format.stroke_color())
        self.strokeWidthBox.setValue(font_format.stroke_width)
        self.lineSpacingBox.setValue(font_format.line_spacing)
        self.letterSpacingBox.setValue(font_format.letter_spacing)
        self.angleBox.setValue(0 if self.textblk_item is None else self.textblk_item.angle)
        self.verticalChecker.setChecked(font_format.vertical)
        self.formatBtnGroup.boldBtn.setChecked(font_format.bold)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.alignBtnGroup.setAlignment(font_format.alignment)
        
        self.familybox.blockSignals(False)
        self.textadvancedfmt_panel.set_active_format(font_format)

    def set_globalfmt_title(self):
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is None:
            self.textstyle_panel.setTitle(self.global_fontfmt_str)
        else:
            title = self.global_fontfmt_str + ' - ' + active_text_style_label.fontfmt._style_name
            valid_title = self.textstyle_panel.elidedText(title)
            self.textstyle_panel.setTitle(valid_title)


    def deactivate_style_label(self):
        if self.active_text_style_label() is not None:
            self.textstyle_panel.on_stylelabel_activated(False)


    def on_active_textstyle_label_changed(self):
        '''
        merge activate textstyle into global format
        '''
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is not None:
            updated_keys = self.global_format.merge(active_text_style_label.fontfmt, compare=True)
            if self.global_mode() and len(updated_keys) > 0:
                self.set_active_format(self.global_format)
            self.set_globalfmt_title()
        else:
            if self.global_mode():
                self.set_globalfmt_title()

    def on_active_stylename_edited(self):
        if self.global_mode():
            self.set_globalfmt_title()

    def set_textblk_item(self, textblk_item: TextBlkItem = None, multi_select:bool=False):
        # A selection transition is a transaction boundary for transform text.
        # Commit against the old target list before replacing it.
        self.textadvancedfmt_panel.finish_pending_transform_edits()
        if textblk_item is not None:
            transform_items = [textblk_item]
        elif multi_select:
            transform_items = SW.canvas.selected_text_items()
        else:
            transform_items = []

        preserve_local_owner = False
        if textblk_item is None:
            focus_w = self.app.focusWidget()
            focus_on_fmtoptions = self.focusOnColorDialog or (
                focus_w is not None
                and (focus_w is self or self.isAncestorOf(focus_w))
            )
            preserve_local_owner = (
                not transform_items
                and self.textblk_item is not None
                and focus_on_fmtoptions
            )
            if preserve_local_owner:
                # Formatting focus can briefly clear the canvas selection; use
                # the retained local item when comparing effective owners.
                transform_items = [self.textblk_item]

        targets_changed = len(transform_items) != len(self._transform_items) or any(
            current is not replacement
            for current, replacement in zip(self._transform_items, transform_items)
        )
        if targets_changed:
            self._cancel_text_transform_previews()
        else:
            # Preserve the physical press for a focus-only refresh, but clear
            # the item-local preview and panel snapshot as before.
            self.on_text_transform_cancel(self._transform_drag_param)
        self._transform_items = transform_items

        if textblk_item is None:
            if not preserve_local_owner:
                # Store the current text block's format before switching to global.
                # This is BASE behavior and must also preserve the transform quartet.
                if self.textblk_item is not None:
                    self.textblk_item.fontformat = copy.deepcopy(C.active_format)
                self.textblk_item = None
                self.set_active_format(self.global_format, multi_select)
                self.set_globalfmt_title()
            if transform_items:
                self.textadvancedfmt_panel.set_transform_items(transform_items)
            
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                # Preserve gradient properties from the text block's format
                if hasattr(textblk_item.fontformat, 'gradient_enabled'):
                    blk_fmt.gradient_enabled = textblk_item.fontformat.gradient_enabled
                    blk_fmt.gradient_start_color = textblk_item.fontformat.gradient_start_color
                    blk_fmt.gradient_end_color = textblk_item.fontformat.gradient_end_color
                    blk_fmt.gradient_angle = textblk_item.fontformat.gradient_angle
                    blk_fmt.gradient_size = textblk_item.fontformat.gradient_size
                self.textblk_item = textblk_item
                multi_size = not textblk_item.isEditing() and textblk_item.isMultiFontSize()
                self.set_active_format(blk_fmt, multi_size)
                self.textadvancedfmt_panel.set_transform_items(transform_items)
                self.textstyle_panel.setTitle(f'TextBlock #{textblk_item.idx}')
