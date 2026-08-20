from typing import Iterable, Union

from qtpy import QT6
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFontComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
)
from qtpy.QtCore import QSignalBlocker, Signal, Qt
from qtpy.QtGui import (
    QActionGroup,
    QColor,
    QFocusEvent,
    QFontDatabase,
    QIcon,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTextCursor,
)

from ballontranslator.utils import shared
from ballontranslator.utils import config as C
from ballontranslator.utils.fontformat import (
    FontFormat,
    FontWeight,
    LineSpacingType,
    font_weight_from_qt,
)
from ...custom_widget import (
    AlignmentChecker,
    CheckableLabel,
    ColorPickerLabel,
    QFontChecker,
    SizeComboBox,
    SizeControlLabel,
    TextCheckerLabel,
    Widget,
)
from ..item import TextBlkItem
from ..annotations import (
    DEFAULT_EMPHASIS_POSITION,
    EMPHASIS_GLYPHS,
    EMPHASIS_POSITIONS,
    EMPHASIS_STYLES,
    OLDSTYLE_NUMS,
    RubyValidationError,
)
from .advanced import TextAdvancedFormatPanel
from ..transforms.edit_session import TextTransformEditSession
from ..transforms.panel import TextTransformPanel
from .presets import TextStylePresetPanel
from .commands import handle_ffmt_change, restore_canvas_view_focus
from ... import shared_widget as SW

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
        self.setFixedSize(12, 12)


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
        hlayout.setContentsMargins(8, 8, 8, 8)

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


class EmphasisToolButton(QToolButton):
    """Toggle emphasis and select its CSS-compatible mark and position."""

    emphasis_changed = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selected_style = 'filled dot'
        self._position = DEFAULT_EMPHASIS_POSITION
        self.setObjectName('FontEmphasisToolButton')
        self.setCheckable(True)
        self.setToolTip(self.tr('Emphasis Marks'))
        popup_modes = getattr(QToolButton, 'ToolButtonPopupMode', QToolButton)
        self.setPopupMode(popup_modes.MenuButtonPopup)

        menu = QMenu(self)
        menu.setObjectName('FontEmphasisMenu')
        section_font = menu.font()
        section_font.setBold(True)
        marks_header = menu.addAction(self.tr('Marks'))
        marks_header.setEnabled(False)
        marks_header.setFont(section_font)
        self._style_group = QActionGroup(self)
        self._style_group.setExclusive(True)
        self._style_actions = {}
        style_labels = (
            self.tr('Filled Dot'),
            self.tr('Open Dot'),
            self.tr('Filled Circle'),
            self.tr('Open Circle'),
            self.tr('Filled Double Circle'),
            self.tr('Open Double Circle'),
            self.tr('Filled Triangle'),
            self.tr('Open Triangle'),
            self.tr('Filled Sesame'),
            self.tr('Open Sesame'),
        )
        for label, style in zip(style_labels, EMPHASIS_STYLES[1:]):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setData(style)
            self._style_group.addAction(action)
            self._style_actions[style] = action
        self._style_group.triggered.connect(self._on_style_selected)

        position_header = menu.addAction(self.tr('Position'))
        position_header.setEnabled(False)
        position_header.setFont(section_font)
        self._position_group = QActionGroup(self)
        self._position_group.setExclusive(True)
        self._position_actions = {}
        position_labels = (
            self.tr('Over / Right'),
            self.tr('Under / Right'),
            self.tr('Over / Left'),
            self.tr('Under / Left'),
        )
        for label, position in zip(position_labels, EMPHASIS_POSITIONS):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setData(position)
            self._position_group.addAction(action)
            self._position_actions[position] = action

        self._position_group.triggered.connect(self._on_position_selected)
        menu.aboutToShow.connect(self._update_menu_icons)
        self.setMenu(menu)
        self._update_menu_icons()
        self._style_actions[self._selected_style].setChecked(True)
        self._position_actions[self._position].setChecked(True)
        self.clicked.connect(self._on_toggled)

    def values(self) -> tuple[str, str]:
        style = self._selected_style if self.isChecked() else 'none'
        return style, self._position

    def _update_menu_icons(self) -> None:
        icon_size = 24
        ratio = max(1.0, self.devicePixelRatioF())
        font = self.font()
        font.setPixelSize(20)
        color = self.palette().text().color()
        icon_key = (ratio, font.toString(), color.rgba())
        if icon_key == getattr(self, '_menu_icon_key', None):
            return
        self._menu_icon_key = icon_key
        for style, action in self._style_actions.items():
            pixmap = QPixmap(
                round(icon_size * ratio), round(icon_size * ratio)
            )
            pixmap.setDevicePixelRatio(ratio)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.setFont(font)
            painter.setPen(color)
            painter.drawText(
                0, 0, icon_size, icon_size,
                Qt.AlignmentFlag.AlignCenter,
                EMPHASIS_GLYPHS[style],
            )
            painter.end()
            action.setIcon(QIcon(pixmap))

    def set_values(self, style: str, position: str) -> None:
        enabled = style in self._style_actions
        if enabled:
            self._selected_style = style
            self._style_actions[style].setChecked(True)
        if position in self._position_actions:
            self._position = position
            self._position_actions[position].setChecked(True)
        with QSignalBlocker(self):
            self.setChecked(enabled)
        self.update()

    def _on_toggled(self, _checked: bool) -> None:
        self.emphasis_changed.emit(*self.values())

    def _on_style_selected(self, action) -> None:
        self._selected_style = str(action.data())
        self.setChecked(True)
        self.update()
        self.emphasis_changed.emit(*self.values())

    def _on_position_selected(self, action) -> None:
        self._position = str(action.data())
        if self.isChecked():
            self.emphasis_changed.emit(*self.values())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        arrow_width = 11
        icon_width = max(1, self.width() - arrow_width)
        icon_rect = self.rect()
        icon_rect.setWidth(icon_width)
        if self.isChecked() and self.isEnabled():
            painter.fillRect(
                icon_rect.adjusted(2, 2, -2, -2), QColor(30, 147, 229)
            )
        if self.isEnabled() and (self.isChecked() or self.underMouse()):
            painter.setPen(QPen(QColor(30, 147, 229), 2))
            painter.drawRect(icon_rect.adjusted(1, 1, -1, -1))
        color = (
            QColor('white')
            if self.isChecked() and self.isEnabled()
            else self.palette().text().color()
        )
        if not self.isEnabled():
            color.setAlpha(110)
        painter.setPen(color)

        glyph_font = self.font()
        glyph_font.setPixelSize(16)
        mark_font = self.font()
        mark_font.setPixelSize(12)
        painter.setFont(mark_font)
        mark = EMPHASIS_GLYPHS[self._selected_style]
        mark_bounds = painter.fontMetrics().tightBoundingRect(mark)
        mark_x = round((icon_width - mark_bounds.width()) / 2 - mark_bounds.left())
        mark_y = self.height() - 3 - mark_bounds.bottom()
        painter.drawText(mark_x, mark_y, mark)

        glyph = 'あ'
        glyph_bottom = mark_y + mark_bounds.top() - 1
        painter.setFont(glyph_font)
        glyph_bounds = painter.fontMetrics().tightBoundingRect(glyph)
        available_height = max(1, glyph_bottom - 1)
        if glyph_bounds.height() > 0:
            fitted_size = round(
                glyph_font.pixelSize()
                * available_height
                / glyph_bounds.height()
            )
            glyph_font.setPixelSize(min(19, max(16, fitted_size)))
            painter.setFont(glyph_font)
            glyph_bounds = painter.fontMetrics().tightBoundingRect(glyph)
        glyph_x = round(
            (icon_width - glyph_bounds.width()) / 2 - glyph_bounds.left()
        )
        glyph_y = glyph_bottom - glyph_bounds.bottom()
        painter.drawText(glyph_x, glyph_y, glyph)

        separator = QColor(color)
        separator.setAlpha(90)
        painter.setPen(QPen(separator, 1))
        painter.drawLine(icon_width, 3, icon_width, self.height() - 4)
        painter.setPen(QPen(color, 1.2))
        arrow_x = self.width() - arrow_width // 2
        arrow_y = self.height() // 2
        painter.drawLine(arrow_x - 3, arrow_y - 1, arrow_x, arrow_y + 2)
        painter.drawLine(arrow_x, arrow_y + 2, arrow_x + 3, arrow_y - 1)


class FormatGroupBtn(QFrame):
    param_changed = Signal(str, bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.italicBtn.clicked.connect(self.setItalic)
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.underlineBtn.clicked.connect(self.setUnderline)
        self.emphasisBtn = EmphasisToolButton(self)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.italicBtn)
        hlayout.addWidget(self.underlineBtn)
        hlayout.addWidget(self.emphasisBtn)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(8, 8, 8, 8)

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
        self.fcombobox.setObjectName("FontFormatSizeBox")
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

    def _change_font_size(self, ratio: float) -> None:
        size_text = self.getFontSize()
        multi_size = size_text.endswith('+')
        size = float(size_text[:-1] if multi_size else size_text)
        new_size = int(round(size * ratio))
        if new_size == size:
            new_size += 1 if ratio > 1 else -1
        new_size = min(1000, max(1, new_size))
        if new_size == size:
            return

        display_text = f'{new_size}+' if multi_size else str(new_size)
        with QSignalBlocker(self.fcombobox):
            self.fcombobox.setCurrentText(display_text)
        # The arrows are part of the editable size control. Keep focus on that
        # control so effects and the live text use the same unfocused layout.
        self.fcombobox.setFocus()
        self.param_changed.emit(
            'rel_font_size' if multi_size else 'font_size',
            ratio if multi_size else new_size,
        )

    def onUpBtnClicked(self) -> None:
        self._change_font_size(1.25)

    def onDownBtnClicked(self) -> None:
        self._change_font_size(0.75)
    

class FontWeightComboBox(QComboBox):
    param_changed = Signal(str, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        labels = {
            FontWeight.Thin: self.tr('Thin'),
            FontWeight.ExtraLight: self.tr('Extra Light'),
            FontWeight.Light: self.tr('Light'),
            FontWeight.Normal: self.tr('Normal'),
            FontWeight.Medium: self.tr('Medium'),
            FontWeight.DemiBold: self.tr('Demi Bold'),
            FontWeight.Bold: self.tr('Bold'),
            FontWeight.ExtraBold: self.tr('Extra Bold'),
            FontWeight.Black: self.tr('Black'),
        }
        for weight in FontWeight:
            self.addItem(labels[weight], int(weight))
        self.activated.connect(self._on_activated)
        self.set_weight(FontWeight.Normal)

    def _on_activated(self, index: int) -> None:
        self.param_changed.emit('font_weight', self.weight())

    def weight(self) -> FontWeight:
        return FontWeight(int(self.currentData()))

    def set_weight(self, weight: FontWeight) -> None:
        index = self.findData(int(FontWeight(weight)))
        if index >= 0:
            self.setCurrentIndex(index)


_FONT_WEIGHT_SUFFIXES = (
    ('extra light', FontWeight.ExtraLight),
    ('extra bold', FontWeight.ExtraBold),
    ('semi bold', FontWeight.DemiBold),
    ('demi bold', FontWeight.DemiBold),
    ('extralight', FontWeight.ExtraLight),
    ('extrabold', FontWeight.ExtraBold),
    ('semibold', FontWeight.DemiBold),
    ('demibold', FontWeight.DemiBold),
    ('regular', FontWeight.Normal),
    ('normal', FontWeight.Normal),
    ('medium', FontWeight.Medium),
    ('light', FontWeight.Light),
    ('black', FontWeight.Black),
    ('bold', FontWeight.Bold),
    ('thin', FontWeight.Thin),
)


def _split_weight_family_name(
    family: str,
) -> tuple[str, FontWeight] | tuple[None, None]:
    """Split a family only when it has a recognized final weight token.

    >>> _split_weight_family_name('Inter Display SemiBold')
    ('Inter Display', <FontWeight.DemiBold: 600>)
    >>> _split_weight_family_name('Blackadder ITC')
    (None, None)
    """
    folded = family.casefold()
    for suffix, weight in _FONT_WEIGHT_SUFFIXES:
        marker = f' {suffix}'
        if folded.endswith(marker):
            return family[:-len(marker)], weight
    return None, None


def _font_database() -> Union[type[QFontDatabase], QFontDatabase]:
    return QFontDatabase if QT6 else QFontDatabase()


def _family_weights(
    database: Union[type[QFontDatabase], QFontDatabase],
    family: str,
) -> set[FontWeight]:
    weights = set()
    for style in database.styles(family):
        weights.add(font_weight_from_qt(int(database.weight(family, style))))
    return weights


def _weight_family_aliases(
    font_families: Iterable[str],
) -> dict[str, tuple[str, FontWeight]]:
    """Return suffix aliases that resolve to a base family's same face."""
    families = list(font_families)
    by_folded_name = {family.casefold(): family for family in families}
    database = _font_database()
    weights_by_family = {}
    aliases = {}
    for alias in families:
        base_name, weight = _split_weight_family_name(alias)
        if base_name is None:
            continue
        base = by_folded_name.get(base_name.casefold())
        if base is None:
            continue
        if base not in weights_by_family:
            weights_by_family[base] = _family_weights(database, base)
        if alias not in weights_by_family:
            weights_by_family[alias] = _family_weights(database, alias)
        base_weights = weights_by_family[base]
        alias_weights = weights_by_family[alias]
        if (
            weight in base_weights
            and alias_weights == {weight}
        ):
            aliases[alias] = (base, weight)
    return aliases


class FontFamilyComboBox(QFontComboBox):
    param_changed = Signal(str, object)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentFontChanged.connect(self.on_fontfamily_changed)
        self.lineedit = lineedit = LineEdit(parent=self)
        lineedit.return_pressed.connect(self.on_return_pressed)
        self.setLineEdit(lineedit)
        self.return_pressed = False
        self.weight_aliases = {}
        self.canonical_weight_aliases = {}
        self._weight_alias_source = None
        
    def apply_fontfamily(self) -> None:
        ffamily = self.currentText()
        if ffamily in shared.FONT_FAMILIES:
            self.param_changed.emit('font_family', ffamily)

    def set_displayed_font(self, font_family: str) -> None:
        """Show a family without changing the filtered popup model."""
        index = self.findText(font_family)
        self.setCurrentIndex(index)
        if index < 0:
            # setCurrentFont() rebuilds QFontComboBox's database-backed model.
            self.setEditText(font_family)

    def update_font_list(self, font_list: Iterable[str]) -> None:
        font_list = list(font_list)
        alias_source = frozenset(shared.FONT_FAMILIES or font_list)
        if alias_source != self._weight_alias_source:
            self.weight_aliases = _weight_family_aliases(alias_source)
            self._weight_alias_source = alias_source
        visible_families = set(font_list)
        self.canonical_weight_aliases = {
            alias: target
            for alias, target in self.weight_aliases.items()
            if target[0] in visible_families
        }
        font_list = [
            family
            for family in font_list
            if family not in self.canonical_weight_aliases
        ]
        if font_list == [self.itemText(i) for i in range(self.count())]:
            return

        current_font = self.currentText()
        self.currentFontChanged.disconnect(self.on_fontfamily_changed)
        try:
            self.clear()
            self.addItems(font_list)
            # Keep an applied hidden font visible in the editable field without
            # putting it back in the popup or changing the underlying format.
            self.set_displayed_font(current_font)
        finally:
            self.currentFontChanged.connect(self.on_fontfamily_changed)

    def canonical_family(
        self, font_family: str
    ) -> tuple[str, FontWeight | None]:
        return self.canonical_weight_aliases.get(
            font_family, (font_family, None)
        )

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

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.familybox = FontFamilyComboBox(parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_font_family_changed)
        self.familybox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if shared.FONT_FAMILIES:
            self.familybox.update_font_list(
                shared.get_filtered_font_list(shared.FONT_FAMILIES)
            )

        self.fontWeightBox = FontWeightComboBox(self)
        self.fontWeightBox.setObjectName('FontWeightBox')
        self.fontWeightBox.setToolTip(self.tr('Font Weight'))
        self.fontWeightBox.param_changed.connect(
            self.on_font_weight_changed
        )

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
        self.lineSpacingBox.setObjectName("FontFormatSizeBox")
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.param_changed.connect(self.on_param_changed)

        linesp_hlayout = QHBoxLayout()
        linesp_hlayout.addWidget(self.lineSpacingLabel)
        linesp_hlayout.addWidget(self.lineSpacingBox)
        linesp_hlayout.setSpacing(7)
        
        self.colorPicker = ColorPickerLabel(self, param_name='frgb')
        self.colorPicker.setObjectName('FontFormatColorPicker')
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

        self._tate_chu_yoko_tooltip = self.tr(
            'Combine the selected text into one upright vertical cell'
        )
        self.tateChuYokoChecker = QFontChecker(self)
        self.tateChuYokoChecker.setObjectName("FontTateChuYokoChecker")
        self.tateChuYokoChecker.setToolTip(self._tate_chu_yoko_tooltip)
        self.tateChuYokoChecker.clicked.connect(
            self.on_tate_chu_yoko_changed
        )
        self.romanAlignmentChecker = QFontChecker(self)
        self.romanAlignmentChecker.setObjectName(
            'FontRomanAlignmentChecker'
        )
        self.romanAlignmentChecker.setToolTip(
            self.tr('Standard Vertical Roman Alignment')
        )
        self.romanAlignmentChecker.clicked.connect(
            lambda checked: self.on_param_changed(
                'standard_vertical_roman_alignment', checked
            )
        )

        self.strokeWidthBox = SizeComboBox([0, 10], 'stroke_width', self)
        self.strokeWidthBox.setObjectName("FontFormatSizeBox")
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.param_changed.connect(self.on_param_changed)

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        self.fontStrokeLabel.size_ctrl_changed.connect(self.strokeWidthBox.changeByDelta)
        self.fontStrokeLabel.btn_released.connect(lambda : self.on_param_changed('stroke_width', self.strokeWidthBox.value()))
        
        self.strokeColorPicker = ColorPickerLabel(self, param_name='srgb')
        self.strokeColorPicker.setObjectName('FontFormatColorPicker')
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.strokeColorPicker.apply_color.connect(self.on_apply_color)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(7)

        self.letterSpacingBox = SizeComboBox([0, 10], "letter_spacing", self)
        self.letterSpacingBox.setObjectName("FontFormatSizeBox")
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
        lettersp_hlayout.setSpacing(7)

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
        self.formatBtnGroup.emphasisBtn.emphasis_changed.connect(
            self.on_emphasis_changed
        )
        self.textadvancedfmt_panel.ruby_apply_requested.connect(
            self.on_ruby_apply_requested
        )
        self.textadvancedfmt_panel.ligature_axis_changed.connect(
            self.on_ligature_axis_changed
        )
        self.textadvancedfmt_panel.ruby_remove_requested.connect(
            self.on_ruby_remove_requested
        )
        self.texttransform_panel = TextTransformPanel(
            self.tr('Text Transform'),
            config_name='text_transform_panel',
            config_expand_name='expand_ttransform_panel',
        )
        self.text_transform_session = TextTransformEditSession(
            self,
            self.texttransform_panel,
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
        for label in (self.foldTextBtn, self.sourceBtn, self.transBtn):
            label.setObjectName("FontFormatActionLabel")

        FONTFORMAT_SPACING = 5

        vl0 = QVBoxLayout()
        vl0.addWidget(self.textstyle_panel.view_widget)
        vl0.addWidget(self.textadvancedfmt_panel.view_widget)
        vl0.addWidget(self.texttransform_panel.view_widget)
        vl0.setSpacing(0)
        vl0.setContentsMargins(0, 0, 0, 0)
        hl1 = QHBoxLayout()
        font_selector_layout = QHBoxLayout()
        font_selector_layout.addWidget(self.colorPicker)
        font_selector_layout.addWidget(self.familybox, 1)
        font_selector_layout.addWidget(self.fontWeightBox)
        font_selector_layout.setSpacing(7)
        font_selector_layout.setContentsMargins(0, 0, 0, 0)
        hl1.addLayout(font_selector_layout, 1)
        hl1.addWidget(self.fontsizebox)
        hl1.setSpacing(4)
        hl1.setContentsMargins(0, 11, 0, 0)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        vertical_layout = QHBoxLayout()
        vertical_layout.addWidget(self.verticalChecker)
        vertical_layout.addWidget(self.tateChuYokoChecker)
        vertical_layout.addWidget(self.romanAlignmentChecker)
        vertical_layout.setSpacing(0)
        vertical_layout.setContentsMargins(0, 0, 0, 0)
        hl2.addLayout(vertical_layout)
        hl2.setSpacing(FONTFORMAT_SPACING)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.addLayout(linesp_hlayout)
        hl3.setContentsMargins(3, 0, 3, 0)
        hl3.setSpacing(12)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setContentsMargins(0, 11, 0, 0)
        hl4.setSpacing(0)

        self.vlayout.addLayout(vl0)
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.setContentsMargins(0, 0, 6, 0)
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
        func = handle_ffmt_change.get(param_name)
        func_kwargs = {}
        if param_name in {'font_size', 'rel_font_size'}:
            func_kwargs['clip_size'] = True
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True, **func_kwargs)
            self.update_text_style_label()
        else:
            func(param_name, value, C.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True, **func_kwargs)

    def on_font_family_changed(
        self, param_name: str, font_family: str
    ) -> None:
        canonical_family, inferred_weight = (
            self.familybox.canonical_family(font_family)
        )
        if canonical_family != font_family:
            with QSignalBlocker(self.familybox):
                self.familybox.set_displayed_font(canonical_family)
        self.on_param_changed(param_name, canonical_family)
        if inferred_weight is not None:
            self.fontWeightBox.set_weight(inferred_weight)
            self.on_param_changed('font_weight', inferred_weight)

    def on_font_weight_changed(
        self, param_name: str, weight: FontWeight
    ) -> None:
        self._apply_font_weight(FontWeight(weight))

    def toggle_bold(self) -> None:
        current_weight = FontWeight(C.active_format.font_weight)
        weight = (
            FontWeight.Normal
            if current_weight >= FontWeight.DemiBold
            else FontWeight.Bold
        )
        self._apply_font_weight(weight)

    def _apply_font_weight(self, weight: FontWeight) -> None:
        font_family = self.familybox.currentText()
        canonical_family, _ = self.familybox.canonical_family(font_family)
        if canonical_family != font_family:
            with QSignalBlocker(self.familybox):
                self.familybox.set_displayed_font(canonical_family)
            self.on_param_changed('font_family', canonical_family)
        self.fontWeightBox.set_weight(weight)
        self.on_param_changed('font_weight', weight)

    def on_emphasis_changed(self, style: str, position: str) -> None:
        if self.textblk_item is not None:
            items = [self.textblk_item]
        else:
            items = SW.canvas.selected_text_items()
        for item in items:
            item.setEmphasis(style, position)
        if items:
            restore_canvas_view_focus()

    def on_ligature_axis_changed(self, axis: str, state: str) -> None:
        if self.global_mode():
            attribute = (
                'oldstyle_nums'
                if axis == OLDSTYLE_NUMS
                else f'ligature_{axis}'
            )
            setattr(self.global_format, attribute, state)
            self.update_text_style_label()
        if self.textblk_item is not None:
            items = [self.textblk_item]
        else:
            items = SW.canvas.selected_text_items()
        for item in items:
            if axis == OLDSTYLE_NUMS:
                item.setOldstyleNums(state)
            else:
                item.setLigatureAxis(axis, state)
        if items:
            restore_canvas_view_focus()

    def on_tate_chu_yoko_changed(self, enabled: bool) -> None:
        if self.textblk_item is not None:
            items = [self.textblk_item]
        else:
            items = SW.canvas.selected_text_items()
        try:
            for item in items:
                item.setTateChuYoko(enabled)
        except RubyValidationError as error:
            if str(error) == 'Tate-chu-yoko cannot overlap Ruby':
                message = self.tr('Tate-chu-yoko cannot overlap Ruby.')
            else:
                message = self.tr(
                    'Unable to apply Tate-chu-yoko to this selection.'
                )
            current = (
                self.textblk_item.tate_chu_yoko_enabled()
                if self.textblk_item is not None else False
            )
            self.set_tate_chu_yoko_enabled(current)
            self.tateChuYokoChecker.setToolTip(message)
            QToolTip.showText(
                self.tateChuYokoChecker.mapToGlobal(
                    self.tateChuYokoChecker.rect().bottomLeft()
                ),
                message,
                self.tateChuYokoChecker,
            )
            return
        self.tateChuYokoChecker.setToolTip(self._tate_chu_yoko_tooltip)
        if items:
            restore_canvas_view_focus()

    def _restore_ruby_edit_focus(self, item: TextBlkItem) -> None:
        if item.isEditing():
            SW.canvas.gv.setFocus()
            item.setFocus(Qt.FocusReason.OtherFocusReason)
        else:
            restore_canvas_view_focus()

    def on_ruby_apply_requested(
        self, ruby_type: str, text: str, position: str
    ) -> None:
        item = self.textblk_item
        if item is None:
            self.textadvancedfmt_panel.ruby_group.set_error(
                self.tr('Select base text to apply Ruby.')
            )
            return
        try:
            item.setRuby(ruby_type, text, position)
        except RubyValidationError as error:
            messages = {
                'Select non-empty base text before applying Ruby': self.tr(
                    'Select base text to apply Ruby.'
                ),
                'Ruby text cannot be empty': self.tr(
                    'Ruby text cannot be empty.'
                ),
                'Mono Ruby needs one whitespace-separated reading per base grapheme': self.tr(
                    'Mono Ruby needs one whitespace-separated reading per base grapheme.'
                ),
                'Ruby cannot partially overlap an existing container': self.tr(
                    'Ruby cannot partially overlap an existing container.'
                ),
                'Ruby cannot overlap Tate-chu-yoko': self.tr(
                    'Ruby cannot overlap Tate-chu-yoko.'
                ),
                'Ruby base text cannot contain paragraph or forced line breaks': self.tr(
                    'Ruby base text cannot contain paragraph or forced line breaks.'
                ),
            }
            self.textadvancedfmt_panel.ruby_group.set_error(
                messages.get(
                    str(error),
                    self.tr('Unable to apply Ruby to this selection.'),
                )
            )
            self._restore_ruby_edit_focus(item)
            return
        self.textadvancedfmt_panel.set_ruby_state(*item.ruby_editor_values())
        self._restore_ruby_edit_focus(item)

    def on_ruby_remove_requested(self) -> None:
        item = self.textblk_item
        if item is None:
            return
        item.removeRuby()
        self.textadvancedfmt_panel.set_ruby_state(*item.ruby_editor_values())
        self._restore_ruby_edit_focus(item)

    def resolve_text_transform_edits_for_save(self):
        self.text_transform_session.resolve_for_save()

    def resolve_text_transform_edits_for_history_change(self):
        self.text_transform_session.resolve_for_history_change()

    def resolve_text_transform_edits_for_page_change(self):
        self.text_transform_session.resolve_for_page_change()

    def cancel_text_transform_edits_for_scene_change(self):
        self.text_transform_session.cancel_for_scene_change()

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

    def sync_inline_format(
        self,
        font_format: FontFormat,
        multi_size: bool = False,
        *,
        preserve_focused_editors: bool = True,
    ) -> None:
        C.active_format = font_format
        font_size = round(font_format.font_size, 1)
        if int(font_size) == font_size:
            font_size = str(int(font_size))
        else:
            font_size = f'{font_size:.1f}'
        if multi_size:
            font_size += "+"

        if not preserve_focused_editors or not self.familybox.hasFocus():
            with QSignalBlocker(self.familybox):
                self.familybox.set_displayed_font(font_format.font_family)
        if (
            not preserve_focused_editors
            or not self.fontsizebox.fcombobox.hasFocus()
        ):
            with QSignalBlocker(self.fontsizebox.fcombobox):
                self.fontsizebox.fcombobox.setCurrentText(font_size)
        if not preserve_focused_editors or not self.fontWeightBox.hasFocus():
            with QSignalBlocker(self.fontWeightBox):
                self.fontWeightBox.set_weight(font_format.font_weight)
        foreground = tuple(font_format.foreground_color())
        if (
            (not preserve_focused_editors or not self.focusOnColorDialog)
            and (
                self.colorPicker.color is None
                or self.colorPicker.rgb() != foreground
            )
        ):
            self.colorPicker.setPickerColor(foreground)
        if not preserve_focused_editors or not self.lineSpacingBox.hasFocus():
            with QSignalBlocker(self.lineSpacingBox):
                self.lineSpacingBox.setValue(font_format.line_spacing)
        if not preserve_focused_editors or not self.letterSpacingBox.hasFocus():
            with QSignalBlocker(self.letterSpacingBox):
                self.letterSpacingBox.setValue(font_format.letter_spacing)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.textadvancedfmt_panel.set_line_spacing_type(
            font_format.line_spacing_type
        )
        if self.textblk_item is None:
            self.formatBtnGroup.emphasisBtn.set_values(
                'none', 'over right'
            )
            self.set_tate_chu_yoko_enabled(False)
            self.textadvancedfmt_panel.set_ruby_state(
                'group', '', 'over', False,
            )
        else:
            self.formatBtnGroup.emphasisBtn.set_values(
                *self.textblk_item.emphasis_values()
            )
            self.set_tate_chu_yoko_enabled(
                self.textblk_item.tate_chu_yoko_enabled()
            )
            self.textadvancedfmt_panel.set_ruby_state(
                *self.textblk_item.ruby_editor_values()
            )
        for axis in self.textadvancedfmt_panel.ligature_comboboxes:
            if axis == OLDSTYLE_NUMS:
                value = (
                    font_format.oldstyle_nums
                    if self.textblk_item is None
                    else self.textblk_item.oldstyle_nums_value()
                )
            else:
                value = (
                    getattr(font_format, f'ligature_{axis}')
                    if self.textblk_item is None
                    else self.textblk_item.ligature_axis_value(axis)
                )
            self.textadvancedfmt_panel.set_ligature_axis(
                axis,
                value,
            )

    def set_active_format(
        self,
        font_format: FontFormat,
        multi_size: bool = False,
        *,
        update_transform_panel: bool = True,
    ) -> None:
        self.sync_inline_format(
            font_format,
            multi_size,
            preserve_focused_editors=False,
        )
        self.strokeColorPicker.setPickerColor(font_format.stroke_color())
        self.strokeWidthBox.setValue(font_format.stroke_width)
        self.verticalChecker.setChecked(font_format.vertical)
        self.romanAlignmentChecker.setChecked(
            font_format.standard_vertical_roman_alignment
        )
        self.alignBtnGroup.setAlignment(font_format.alignment)
        self.textadvancedfmt_panel.set_active_format(font_format)
        if update_transform_panel:
            self.texttransform_panel.set_active_format(font_format)

    def set_tate_chu_yoko_enabled(self, enabled: bool) -> None:
        self.tateChuYokoChecker.setChecked(enabled)
        self.tateChuYokoChecker.setToolTip(self._tate_chu_yoko_tooltip)

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
        self.text_transform_session.finish_pending_edits()
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

        self.text_transform_session.replace_targets(transform_items)

        if textblk_item is None:
            if not preserve_local_owner:
                # Store the current text block's format before switching to global.
                # This existing owner switch must preserve the complete transform.
                if self.textblk_item is not None:
                    # Keep the TextBlock-owned object shared with its live
                    # layout; replacing it desynchronizes fill and effects.
                    letter_spacing = (
                        self.textblk_item.fontformat.letter_spacing
                    )
                    line_spacing = self.textblk_item.fontformat.line_spacing
                    line_spacing_type = (
                        self.textblk_item.fontformat.line_spacing_type
                    )
                    self.textblk_item.fontformat.merge(C.active_format)
                    self.textblk_item.fontformat.letter_spacing = (
                        letter_spacing
                    )
                    self.textblk_item.fontformat.line_spacing = line_spacing
                    self.textblk_item.fontformat.line_spacing_type = (
                        line_spacing_type
                    )
                self.textblk_item = None
                self.set_active_format(
                    self.global_format,
                    multi_select,
                    update_transform_panel=not transform_items,
                )
                self.set_globalfmt_title()
            if transform_items:
                self.texttransform_panel.set_transform_items(transform_items)
            
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                self.textblk_item = textblk_item
                multi_size = not textblk_item.isEditing() and textblk_item.isMultiFontSize()
                self.set_active_format(blk_fmt, multi_size)
                self.textstyle_panel.setTitle(f'TextBlock #{textblk_item.idx}')
