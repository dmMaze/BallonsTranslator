"""Expandable controls for item-wide text effects."""

from dataclasses import replace
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import (
    QCoreApplication,
    QEvent,
    QPoint,
    QRectF,
    QSignalBlocker,
    QTimer,
    Signal,
    QSize,
    Qt,
)
from qtpy.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QResizeEvent,
)
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import profile_by_id
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_alpha_mask import TextAlphaMask
from ballontranslator.utils.text_effects import (
    EffectPaint,
    FilterEffect,
    GeneratedEffectPaint,
    GlowEffect,
    LinearGradientPaint,
    HollowEffect,
    ImageEffect,
    ImageGenerationRecipe,
    SHADOW_BLUR_LIMIT,
    SHADOW_DISTANCE_LIMIT,
    SHADOW_SPREAD_LIMIT,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextFillEffect,
    TextEffectStack,
    TexturePaint,
    effect_structure_key,
    without_project_raster_effects,
)

from ...custom_widget import PanelArea
from ... import shared_widget as SW
from ...custom_widget.combobox import BottomBorderComboBox
from ...icon_rendering import render_svg_pixmap
from ...misc import themed_icon_path
from ...llm_modality import LLM_MODALITY_IMAGE_COLOR
from ...module_tool_button import (
    _add_bottom_menu_action,
    _add_bottom_menu_section,
    _add_bottom_submenu,
    _bottom_submenu,
    _simplify_llm_model_name,
)
from ..transforms.controls import CommittedTransformControl, TransformDragLabel
from .gradient_editor import GradientAngleDial, InlineLinearGradientEditor
from .paint import paint_effect_paint_preview
from .filters import (
    FilterParamSpec,
    FilterSpec,
    FilterUnavailableError,
    get_filter_registry,
)
from .edit_session import (
    effect_reorder_is_aligned,
    matched_effect_occurrences,
)

if TYPE_CHECKING:
    from .alpha_mask_edit_session import TextAlphaMaskEditSession
    from ..item import TextBlkItem


def _filter_ui_text(spec: FilterSpec, text: str) -> str:
    """Translate static built-in metadata in one extractable UI context."""
    if not spec.builtin:
        return text
    translations = {
        'Noise': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Noise'
        ),
        'Grain': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Grain'
        ),
        'Rough Edge': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Rough Edge'
        ),
        'Gaussian Blur': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Gaussian Blur'
        ),
        'Bloom': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Bloom'
        ),
        'Glitch': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Glitch'
        ),
        'Amount': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Amount'
        ),
        'Color': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Color'
        ),
        'Monochrome': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Monochrome'
        ),
        'Seed': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Seed'
        ),
        'Size': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Size'
        ),
        'Hardness': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Hardness'
        ),
        'Radius': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Radius'
        ),
        'Threshold': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Threshold'
        ),
        'Intensity': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Intensity'
        ),
        'Shift': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Shift'
        ),
        'Block Size': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Block Size'
        ),
        'Activity': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'Activity'
        ),
        'RGB Split': lambda: QCoreApplication.translate(
            'TextEffectPanel', 'RGB Split'
        ),
    }
    translator = translations.get(text)
    return text if translator is None else translator()


class EffectVisibilityButton(QToolButton):
    """Compact enabled, disabled, or mixed visibility control.

    >>> EffectVisibilityButton.__name__
    'EffectVisibilityButton'
    """

    visibility_requested = Signal(bool)

    def __init__(
        self,
        show_tooltip: str,
        hide_tooltip: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._show_tooltip = show_tooltip
        self._hide_tooltip = hide_tooltip
        self._visibility: Optional[bool] = None
        self.setObjectName('TextEffectVisibilityButton')
        self.setFixedSize(18, 18)
        self.setIconSize(QSize(16, 16))
        self.clicked.connect(self._on_clicked)
        self.set_visibility(None)

    def set_visibility(self, visible: Optional[bool]) -> None:
        self._visibility = visible
        if visible is True:
            icon_name = 'text-effect-visibility-open.svg'
            description = self._hide_tooltip
        elif visible is False:
            icon_name = 'text-effect-visibility-closed.svg'
            description = self._show_tooltip
        else:
            icon_name = 'text-effect-visibility-mixed.svg'
            description = self._show_tooltip
        self.setIcon(QIcon(themed_icon_path(icon_name)))
        self.setToolTip(description)
        self.setAccessibleName(description)

    def _on_clicked(self) -> None:
        self.visibility_requested.emit(self._visibility is not True)


class _EffectCard(QFrame):
    """Keep pointer-only action icons keyboard reachable."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._hovered = False
        self._matched = False
        self._keyboard_focused_action: Optional[QToolButton] = None
        self._hover_actions: Tuple[Tuple[QToolButton, QIcon], ...] = ()
        self.setProperty('matched', False)

    def set_matched(self, matched: bool) -> None:
        matched = bool(matched)
        if self._matched == matched:
            return
        self._matched = matched
        self.setProperty('matched', matched)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_hover_actions(
        self, buttons: Sequence[QToolButton]
    ) -> None:
        self._hover_actions = tuple(
            (button, button.icon()) for button in buttons
        )
        for button, _icon in self._hover_actions:
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.installEventFilter(self)
        self._sync_action_icons()

    def _sync_action_icons(self) -> None:
        visible = self._hovered or self._keyboard_focused_action is not None
        for button, icon in self._hover_actions:
            button.setIcon(icon if visible else QIcon())

    def eventFilter(self, watched: QWidget, event: QEvent) -> bool:
        if any(watched is button for button, _icon in self._hover_actions):
            if event.type() == QEvent.Type.FocusIn:
                keyboard_reasons = {
                    Qt.FocusReason.TabFocusReason,
                    Qt.FocusReason.BacktabFocusReason,
                    Qt.FocusReason.ShortcutFocusReason,
                }
                self._keyboard_focused_action = (
                    watched if event.reason() in keyboard_reasons else None
                )
                self._sync_action_icons()
            elif (
                event.type() == QEvent.Type.FocusOut
                and watched is self._keyboard_focused_action
            ):
                self._keyboard_focused_action = None
                self._sync_action_icons()
        return super().eventFilter(watched, event)

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self._sync_action_icons()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self._hovered = False
        self._sync_action_icons()
        super().leaveEvent(event)


def _effect_icon_label(
    icon_name: str,
    parent: QWidget,
) -> QLabel:
    label = QLabel(parent)
    label.setObjectName('TextEffectParameterIcon')
    label.setFixedSize(16, 16)
    label.setPixmap(render_svg_pixmap(
        themed_icon_path(icon_name),
        16,
        16,
        parent.devicePixelRatioF(),
    ))
    return label


def _effect_action_widget(
    parent: _EffectCard,
    buttons: Sequence[QToolButton],
) -> QWidget:
    widget = QWidget(parent)
    widget.setObjectName('TextEffectPanelActions')
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    for button in buttons:
        button.setFixedSize(18, 18)
        icon_size = (
            12 if button.objectName() == 'TextEffectCloseButton' else 16
        )
        button.setIconSize(QSize(icon_size, icon_size))
        layout.addWidget(button)
    widget.setFixedWidth(18 * len(buttons) + 4 * max(0, len(buttons) - 1))
    parent.set_hover_actions(buttons)
    return widget


class BlendModeSelector(QToolButton):
    """Compact selector with native blend-family submenus.

    >>> issubclass(BlendModeSelector, QToolButton)
    True
    """

    mode_changed = Signal(str)
    ARROW_SIZE = 12

    def __init__(
        self,
        accessible_context: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._accessible_context = accessible_context
        self._current_mode: Optional[str] = None
        self._actions_by_mode: Dict[str, QAction] = {}
        self.setObjectName('TextEffectBlendSelector')
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )

        menu = QMenu(self)
        menu.setObjectName('TextEffectBlendMenu')
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)
        self._add_action(
            menu,
            QCoreApplication.translate('TextEffectPanel', 'Normal'),
            'normal',
        )
        darken_menu = menu.addMenu(
            QCoreApplication.translate('TextEffectPanel', 'Darken')
        )
        darken_menu.setObjectName('TextEffectBlendMenu')
        for label, mode in (
            (QCoreApplication.translate('TextEffectPanel', 'Darken'), 'darken'),
            (
                QCoreApplication.translate('TextEffectPanel', 'Multiply'),
                'multiply',
            ),
            (
                QCoreApplication.translate('TextEffectPanel', 'Color Burn'),
                'color_burn',
            ),
            (
                QCoreApplication.translate('TextEffectPanel', 'Linear Burn'),
                'linear_burn',
            ),
            (
                QCoreApplication.translate('TextEffectPanel', 'Darker Color'),
                'darker_color',
            ),
        ):
            self._add_action(darken_menu, label, mode)

        lighten_menu = menu.addMenu(
            QCoreApplication.translate('TextEffectPanel', 'Lighten')
        )
        lighten_menu.setObjectName('TextEffectBlendMenu')
        for label, mode in (
            (
                QCoreApplication.translate('TextEffectPanel', 'Lighten'),
                'lighten',
            ),
            (QCoreApplication.translate('TextEffectPanel', 'Screen'), 'screen'),
            (
                QCoreApplication.translate('TextEffectPanel', 'Color Dodge'),
                'color_dodge',
            ),
            (
                QCoreApplication.translate(
                    'TextEffectPanel', 'Linear Dodge (Add)'
                ),
                'linear_dodge',
            ),
            (
                QCoreApplication.translate('TextEffectPanel', 'Lighter Color'),
                'lighter_color',
            ),
        ):
            self._add_action(lighten_menu, label, mode)
        self._action_group.triggered.connect(self._on_action_triggered)
        self.setMenu(menu)
        self.set_mode('normal')

    def _add_action(self, menu: QMenu, label: str, mode: str) -> None:
        action = menu.addAction(label)
        action.setCheckable(True)
        action.setData(mode)
        self._action_group.addAction(action)
        self._actions_by_mode[mode] = action

    def current_mode(self) -> Optional[str]:
        return self._current_mode

    def set_mode(self, mode: Optional[str]) -> None:
        action = self._actions_by_mode.get(mode)
        self._current_mode = mode if action is not None else None
        for candidate in self._action_group.actions():
            candidate.setChecked(candidate is action)
        label = (
            action.text()
            if action is not None
            else QCoreApplication.translate('TextEffectPanel', 'Mixed')
        )
        self.setText(label)
        self.setAccessibleName(f'{self._accessible_context}: {label}')

    def _on_action_triggered(self, action: QAction) -> None:
        mode = str(action.data())
        if mode == self._current_mode or mode not in self._actions_by_mode:
            return
        self.set_mode(mode)
        self.mode_changed.emit(mode)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        pixmap = render_svg_pixmap(
            themed_icon_path('chevron-down.svg'),
            self.ARROW_SIZE,
            self.ARROW_SIZE,
            self.devicePixelRatioF(),
        )
        x = self.width() - self.ARROW_SIZE - 4
        y = (self.height() - self.ARROW_SIZE) // 2
        painter.drawPixmap(x, y, pixmap)
        painter.end()


def _blend_control(
    parent: QWidget,
    accessible_name: str,
) -> Tuple[QWidget, BlendModeSelector]:
    """Build the shared blend-mode row."""
    selector = BlendModeSelector(accessible_name, parent)
    tooltip = QCoreApplication.translate(
        'TextEffectPanel',
        'Blends with earlier output in the text-effect stack, not the page '
        'image or backdrop.',
    )
    selector.setToolTip(tooltip)
    selector.setAccessibleDescription(tooltip)
    return _labeled_effect_editor(
        parent,
        QCoreApplication.translate('TextEffectPanel', 'Blend'),
        selector,
    ), selector


def _labeled_effect_editor(
    parent: QWidget, label_text: str, editor: QWidget
) -> QWidget:
    """Build the shared compact label/editor row used by effect cards."""
    label = QLabel(label_text, parent)
    label.setObjectName('TextEffectParamLabel')
    label.setAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    widget = QWidget(parent)
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    layout.addWidget(label)
    layout.addWidget(editor, 1)
    return widget


def _set_blend_values(
    selector: BlendModeSelector,
    effects: Sequence[object],
) -> None:
    values = [getattr(effect, 'blend_mode') for effect in effects]
    common = (
        values[0]
        if values and all(value == values[0] for value in values)
        else None
    )
    selector.set_mode(common)


def _choose_project_raster(parent: QWidget, title: str) -> str:
    """Run the shared native chooser for project-managed raster assets."""
    path, _selected_filter = QFileDialog.getOpenFileName(
        parent,
        title,
        '',
        parent.tr('Images (*.png *.jpg *.jpeg *.webp *.bmp *.jxl)'),
    )
    return path


class EffectNumericControl(CommittedTransformControl):
    """Reuse the committed numeric editor with typed-text preview signals.

    >>> issubclass(EffectNumericControl, CommittedTransformControl)
    True
    """

    value_preview_requested = Signal(str, object)
    value_preview_canceled = Signal(str)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('TextEffectControl')
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.label.setObjectName('TextEffectParamLabel')
        self.label.setWordWrap(False)
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.editor.setObjectName('TextEffectParamEditor')
        self.editor.setProperty('cardEditor', True)
        self.editor.setMinimumWidth(0)
        self.editor.setMaximumWidth(16777215)
        self.editor.setFixedHeight(22)
        self.editor.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.layout().setSpacing(8)
        self.layout().setStretch(0, 0)
        self.layout().setStretch(1, 1)

    def _on_text_edited(self) -> None:
        super()._on_text_edited()
        try:
            value = self._parse(self.editor.text())
        except (TypeError, ValueError):
            return
        self.value_preview_requested.emit(self.param_name, value)

    @property
    def model_value(self) -> Optional[float]:
        return self._model_value

    def show_preview_value(self, value: float) -> None:
        self.editor.setText(self._format(value))

    def restore_model_display(self) -> None:
        self._restore_display()

    def commit_pending(self) -> bool:
        was_pending = self.state == self.PENDING_TEXT
        committed = super().commit_pending()
        if was_pending and not committed:
            self.value_preview_canceled.emit(self.param_name)
        return committed

    def cancel_pending(self) -> None:
        was_pending = self.state == self.PENDING_TEXT
        super().cancel_pending()
        if was_pending:
            self.value_preview_canceled.emit(self.param_name)


class EffectPaintButton(QToolButton):
    """Compact solid swatch or rendered linear-gradient strip.

    >>> issubclass(EffectPaintButton, QToolButton)
    True
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paint: Optional[GeneratedEffectPaint] = None
        self._mixed = False
        self.setObjectName('TextEffectPaintButton')
        self.setMinimumHeight(24)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_paint(
        self,
        paint: Optional[GeneratedEffectPaint],
        mixed: bool = False,
        editable: bool = True,
        description: Optional[str] = None,
    ) -> None:
        self._paint = paint
        self._mixed = bool(mixed)
        self.setIcon(QIcon())
        if mixed:
            self.setText(self.tr('Mixed'))
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            if description is None:
                if editable:
                    description = self.tr('Choose Shared Stroke Color')
                elif isinstance(paint, LinearGradientPaint):
                    description = self.tr('Mixed Gradient Paint')
                else:
                    description = self.tr('Mixed Stroke Paint')
            self.setToolTip(description)
            self.setAccessibleName(description)
            self.setEnabled(editable)
            self.update()
            return
        if paint is None:
            raise ValueError('non-mixed effect paint button requires paint')
        self.setText('')
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        if description is None:
            description = (
                self.tr('Edit Gradient')
                if isinstance(paint, LinearGradientPaint)
                else self.tr('Choose Stroke Color')
            )
        self.setToolTip(description)
        self.setAccessibleName(description)
        self.setEnabled(True)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._paint is None or self._mixed:
            return
        rect = QRectF(self.contentsRect()).adjusted(4.0, 3.0, -4.0, -3.0)
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        painter = QPainter(self)
        paint_effect_paint_preview(
            painter,
            rect,
            self._paint,
            self.palette(),
            self.devicePixelRatioF(),
        )


class StrokeEffectCard(_EffectCard):
    """One Stroke at its complete-stack semantic index.

    >>> StrokeEffectCard.__name__
    'StrokeEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-stroke.svg', self
        )
        self.title_label = QLabel(self.tr('Stroke'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Stroke'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')

        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Stroke'), self.tr('Hide Stroke'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        self.position_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.position_selector.setObjectName('TextEffectParamEditor')
        self.position_selector.setPlaceholderText(self.tr('Mixed'))
        self.position_selector.setAccessibleName(self.tr('Stroke Position'))
        for label, value in (
            (self.tr('Inside'), 'inside'),
            (self.tr('Center'), 'center'),
            (self.tr('Outside'), 'outside'),
        ):
            self.position_selector.addItem(label, value)
        self.position_selector.currentIndexChanged.connect(
            self._on_position_changed
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.position_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.width_control = EffectNumericControl(
            self.tr('Width'), 'width', 1.0, 0.0, 10.0, '', 0.01,
            self, decimals=2,
        )
        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        blend_widget, self.blend_selector = _blend_control(
            self, self.tr('Stroke Blend')
        )
        self.blend_selector.mode_changed.connect(
            self._on_blend_changed
        )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Stroke Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(self.tr('Gradient'), 'linear_gradient')
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        for control in (self.width_control, self.opacity_control):
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(
                self._on_value_preview
            )
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(
                self._on_parameter_commit
            )
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[GeneratedEffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.width_control, 0, 0)
        controls.addWidget(self.opacity_control, 0, 1)
        controls.addLayout(paint_row, 1, 0, 1, 2)
        controls.addWidget(blend_widget, 2, 0)
        controls.addWidget(self.gradient_editor, 3, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, strokes: Sequence[StrokeEffect]) -> None:
        enabled_values = [stroke.enabled for stroke in strokes]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        _set_blend_values(self.blend_selector, strokes)

        positions = [stroke.position for stroke in strokes]
        common_position = (
            positions[0]
            if positions
            and all(position == positions[0] for position in positions)
            else None
        )
        with QSignalBlocker(self.position_selector):
            self.position_selector.setCurrentIndex(
                -1
                if common_position is None
                else self.position_selector.findData(common_position)
            )

        paints = [stroke.paint for stroke in strokes]
        common_paint_type = (
            paints[0].paint_type
            if paints
            and all(
                value.paint_type == paints[0].paint_type
                for value in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1
                if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )

        for name, control in (
            ('width', self.width_control),
            ('opacity', self.opacity_control),
        ):
            values = [getattr(stroke, name) for stroke in strokes]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)

        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=(common_paint_type == 'solid') if mixed_paint else True,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (self.width_control, self.opacity_control)

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_position_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'position',
                self.position_selector.itemData(combo_index),
            )

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
            )

    def _on_blend_changed(self, blend_mode: str) -> None:
        self.value_commit_requested.emit(
            self.index, 'blend_mode', blend_mode
        )

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Stroke Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class ShadowEffectCard(_EffectCard):
    """Edit one typed Shadow at its complete-stack index.

    >>> ShadowEffectCard.__name__
    'ShadowEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-shadow.svg', self
        )
        self.title_label = QLabel(self.tr('Shadow'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Shadow'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')

        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Shadow'), self.tr('Hide Shadow'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        self.type_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.type_selector.setObjectName('TextEffectParamEditor')
        self.type_selector.setPlaceholderText(self.tr('Mixed'))
        self.type_selector.setAccessibleName(self.tr('Shadow Type'))
        for label, value in (
            (self.tr('Drop'), 'drop'),
            (self.tr('Inner'), 'inner'),
            (self.tr('Long / Extrude'), 'long'),
        ):
            self.type_selector.addItem(label, value)
        self.type_selector.currentIndexChanged.connect(
            self._on_type_changed
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.type_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        self.angle_control = EffectNumericControl(
            self.tr('Angle'), 'angle', 1.0, 0.0, 359.9, '°', 1.0,
            self, decimals=1,
        )
        self.angle_dial = GradientAngleDial(self.angle_control)
        self.angle_dial.setToolTip(self.tr('Drag to set shadow angle'))
        self.angle_dial.setAccessibleName(self.tr('Shadow Angle'))
        self.angle_control.label.hide()
        angle_layout = self.angle_control.layout()
        angle_layout.insertWidget(0, self.angle_dial)
        angle_layout.setStretch(0, 0)
        angle_layout.setStretch(1, 0)
        angle_layout.setStretch(2, 1)
        self.angle_dial.angle_previewed.connect(
            self._on_angle_dial_preview
        )
        self.angle_dial.angle_commit_requested.connect(
            self._on_angle_dial_commit
        )
        self.angle_dial.angle_preview_canceled.connect(
            self._on_angle_dial_cancel
        )
        self.distance_control = EffectNumericControl(
            self.tr('Distance'), 'distance', 1.0,
            0.0, SHADOW_DISTANCE_LIMIT, '', 0.01,
            self, decimals=2,
        )
        self.blur_control = EffectNumericControl(
            self.tr('Blur'), 'blur', 1.0, 0.0,
            SHADOW_BLUR_LIMIT, '', 0.01,
            self, decimals=2,
        )
        self.spread_control = EffectNumericControl(
            self.tr('Spread'), 'spread', 1.0, 0.0,
            SHADOW_SPREAD_LIMIT, '', 0.01,
            self, decimals=2,
        )
        blend_widget, self.blend_selector = _blend_control(
            self, self.tr('Shadow Blend')
        )
        self.blend_selector.mode_changed.connect(
            self._on_blend_changed
        )
        for control in self.iter_controls():
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(self._on_parameter_commit)
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Shadow Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(
            self.tr('Gradient'), 'linear_gradient'
        )
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[GeneratedEffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.opacity_control, 0, 0)
        controls.addWidget(self.blur_control, 0, 1)
        controls.addWidget(self.angle_control, 1, 0)
        controls.addWidget(self.distance_control, 1, 1)
        controls.addWidget(self.spread_control, 2, 0)
        controls.addWidget(blend_widget, 2, 1)
        controls.addLayout(paint_row, 3, 0, 1, 2)
        controls.addWidget(self.gradient_editor, 4, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, shadows: Sequence[ShadowEffect]) -> None:
        enabled_values = [shadow.enabled for shadow in shadows]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        _set_blend_values(self.blend_selector, shadows)

        types = [shadow.shadow_type for shadow in shadows]
        common_type = (
            types[0]
            if types and all(value == types[0] for value in types)
            else None
        )
        with QSignalBlocker(self.type_selector):
            self.type_selector.setCurrentIndex(
                -1 if common_type is None
                else self.type_selector.findData(common_type)
            )
        show_soft_controls = common_type != 'long'
        self.blur_control.setVisible(show_soft_controls)
        self.spread_control.setVisible(show_soft_controls)
        if common_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        elif common_type is None:
            self.spread_control.label.setText(self.tr('Spread / Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('angle', self.angle_control),
            ('distance', self.distance_control),
            ('blur', self.blur_control),
            ('spread', self.spread_control),
        ):
            values = [getattr(shadow, name) for shadow in shadows]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)
        self.angle_dial.end_interaction()
        if shadows:
            self.angle_dial.set_angle(shadows[0].angle)

        paints = [shadow.paint for shadow in shadows]
        common_paint_type = (
            paints[0].paint_type
            if paints and all(
                paint.paint_type == paints[0].paint_type for paint in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1 if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        editable = common_paint_type == 'solid' if mixed_paint else True
        if mixed_paint:
            if editable:
                description = self.tr('Choose Shared Shadow Color')
            elif common_paint_type == 'linear_gradient':
                description = self.tr('Mixed Shadow Gradient Paint')
            else:
                description = self.tr('Mixed Shadow Paint')
        else:
            description = (
                self.tr('Edit Shadow Gradient')
                if isinstance(common_paint, LinearGradientPaint)
                else self.tr('Choose Shadow Color')
            )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=editable,
            description=description,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = self.gradient_editor.isHidden() == show_gradient
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            self.opacity_control,
            self.angle_control,
            self.distance_control,
            self.blur_control,
            self.spread_control,
        )

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'shadow_type',
                self.type_selector.itemData(combo_index),
            )

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
            )

    def _on_blend_changed(self, blend_mode: str) -> None:
        self.value_commit_requested.emit(
            self.index, 'blend_mode', blend_mode
        )

    def _on_control_commit(self, name: str, value) -> None:
        if name == 'angle':
            self.angle_dial.set_angle(value)
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        if name == 'angle':
            self.angle_dial.set_angle(value)
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        if name == 'angle' and self.angle_control.model_value is not None:
            self.angle_dial.set_angle(
                self.angle_control.model_value + delta
            )
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        if name == 'angle' and self.angle_control.model_value is not None:
            self.angle_dial.set_angle(
                self.angle_control.model_value + delta
            )
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        if name == 'angle' and self.angle_control.model_value is not None:
            self.angle_dial.set_angle(self.angle_control.model_value)
        self.preview_canceled.emit(self.index, name)

    def _on_angle_dial_preview(self, angle: float) -> None:
        self.angle_control.show_preview_value(angle)
        self.value_preview_requested.emit(self.index, 'angle', angle)

    def _on_angle_dial_commit(self) -> None:
        angle = self.angle_dial.angle
        self.angle_control.set_model_value(angle, (angle,))
        self.value_commit_requested.emit(self.index, 'angle', angle)

    def _on_angle_dial_cancel(self) -> None:
        self.angle_control.restore_model_display()
        self.preview_canceled.emit(self.index, 'angle')

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Shadow Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class GlowEffectCard(_EffectCard):
    """Edit one typed Glow at its complete-stack index.

    >>> GlowEffectCard.__name__
    'GlowEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-glow.svg', self
        )
        self.title_label = QLabel(self.tr('Glow'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Glow'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Glow'), self.tr('Hide Glow'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )

        action_widget = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )
        self.type_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.type_selector.setObjectName('TextEffectParamEditor')
        self.type_selector.setPlaceholderText(self.tr('Mixed'))
        self.type_selector.setAccessibleName(self.tr('Glow Type'))
        self.type_selector.addItem(self.tr('Outer'), 'outer')
        self.type_selector.addItem(self.tr('Inner'), 'inner')
        self.type_selector.currentIndexChanged.connect(
            self._on_type_changed
        )
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addWidget(self.type_selector)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        self.size_control = EffectNumericControl(
            self.tr('Size'), 'size', 1.0, 0.0,
            SHADOW_BLUR_LIMIT, '', 0.01, self, decimals=2,
        )
        self.spread_control = EffectNumericControl(
            self.tr('Spread'), 'spread', 1.0, 0.0,
            SHADOW_SPREAD_LIMIT, '', 0.01, self, decimals=2,
        )
        blend_widget, self.blend_selector = _blend_control(
            self, self.tr('Glow Blend')
        )
        self.blend_selector.mode_changed.connect(
            self._on_blend_changed
        )
        for control in self.iter_controls():
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(
                self._on_parameter_commit
            )
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )

        fill_label = QLabel(self.tr('Fill'), self)
        fill_label.setObjectName('TextEffectParamLabel')
        fill_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.fill_type_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.fill_type_selector.setObjectName('TextEffectParamEditor')
        self.fill_type_selector.setPlaceholderText(self.tr('Mixed'))
        self.fill_type_selector.setAccessibleName(self.tr('Glow Fill'))
        self.fill_type_selector.addItem(self.tr('Solid'), 'solid')
        self.fill_type_selector.addItem(
            self.tr('Gradient'), 'linear_gradient'
        )
        self.fill_type_selector.currentIndexChanged.connect(
            self._on_fill_type_changed
        )
        fill_widget = QWidget(self)
        fill_row = QHBoxLayout(fill_widget)
        fill_row.setContentsMargins(0, 0, 0, 0)
        fill_row.setSpacing(4)
        fill_row.addWidget(fill_label)
        fill_row.addWidget(self.fill_type_selector, 1)

        self.paint_button = EffectPaintButton(self)
        self.paint_button.clicked.connect(self._on_paint_clicked)
        self._paint_seed: Optional[GeneratedEffectPaint] = None
        self.gradient_editor = InlineLinearGradientEditor(
            LinearGradientPaint(), self
        )
        self.gradient_editor.paint_previewed.connect(
            self._on_gradient_preview
        )
        self.gradient_editor.paint_commit_requested.connect(
            self._on_gradient_commit
        )
        self.gradient_editor.paint_preview_canceled.connect(
            self._on_gradient_cancel
        )
        self.gradient_editor.color_dialog_active_changed.connect(
            self.color_dialog_active_changed.emit
        )
        self.gradient_editor.hide()

        paint_row = QGridLayout()
        paint_row.setContentsMargins(0, 0, 0, 0)
        paint_row.setHorizontalSpacing(8)
        paint_row.addWidget(fill_widget, 0, 0)
        paint_row.addWidget(self.paint_button, 0, 1)
        paint_row.setColumnStretch(0, 1)
        paint_row.setColumnStretch(1, 1)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(self.opacity_control, 0, 0)
        controls.addWidget(self.size_control, 0, 1)
        controls.addWidget(self.spread_control, 1, 0)
        controls.addWidget(blend_widget, 1, 1)
        controls.addLayout(paint_row, 2, 0, 1, 2)
        controls.addWidget(self.gradient_editor, 3, 0, 1, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        self._controls_layout = controls

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, glows: Sequence[GlowEffect]) -> None:
        enabled_values = [glow.enabled for glow in glows]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        _set_blend_values(self.blend_selector, glows)

        types = [glow.glow_type for glow in glows]
        common_type = (
            types[0]
            if types and all(value == types[0] for value in types)
            else None
        )
        with QSignalBlocker(self.type_selector):
            self.type_selector.setCurrentIndex(
                -1 if common_type is None
                else self.type_selector.findData(common_type)
            )
        if common_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        elif common_type is None:
            self.spread_control.label.setText(self.tr('Spread / Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('size', self.size_control),
            ('spread', self.spread_control),
        ):
            values = [getattr(glow, name) for glow in glows]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control.set_model_value(common, values)

        paints = [glow.paint for glow in glows]
        common_paint_type = (
            paints[0].paint_type
            if paints and all(
                paint.paint_type == paints[0].paint_type for paint in paints
            )
            else None
        )
        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                -1 if common_paint_type is None
                else self.fill_type_selector.findData(common_paint_type)
            )
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        mixed_paint = common_paint is None
        self._paint_seed = common_paint or (
            paints[0] if paints and common_paint_type is not None else None
        )
        editable = (
            common_paint_type == 'solid' if mixed_paint else True
        )
        if mixed_paint:
            if editable:
                description = self.tr('Choose Shared Glow Color')
            elif common_paint_type == 'linear_gradient':
                description = self.tr('Mixed Glow Gradient Paint')
            else:
                description = self.tr('Mixed Glow Paint')
        else:
            description = (
                self.tr('Edit Glow Gradient')
                if isinstance(common_paint, LinearGradientPaint)
                else self.tr('Choose Glow Color')
            )
        self.paint_button.set_paint(
            self._paint_seed,
            mixed=mixed_paint,
            editable=editable,
            description=description,
        )
        show_gradient = common_paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        if visibility_changed:
            self._controls_layout.invalidate()
            self.layout().invalidate()
            self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            self.opacity_control,
            self.size_control,
            self.spread_control,
        )

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'glow_type',
                self.type_selector.itemData(combo_index),
            )

    def _on_fill_type_changed(self, combo_index: int) -> None:
        if combo_index >= 0:
            self.value_commit_requested.emit(
                self.index,
                'paint_type',
                self.fill_type_selector.itemData(combo_index),
            )

    def _on_blend_changed(self, blend_mode: str) -> None:
        self.value_commit_requested.emit(
            self.index, 'blend_mode', blend_mode
        )

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _on_paint_clicked(self) -> None:
        paint = self._paint_seed
        if not isinstance(paint, SolidPaint):
            return
        self.color_dialog_active_changed.emit(True)
        try:
            color = QColorDialog.getColor(
                QColor(*paint.color), self.window(), self.tr('Glow Color')
            )
            if color.isValid():
                self.value_commit_requested.emit(
                    self.index,
                    'paint',
                    SolidPaint((color.red(), color.green(), color.blue())),
                )
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class TextFillEffectCard(_EffectCard):
    """Edit one fixed Gradient or Texture foreground layer.

    >>> TextFillEffectCard.__name__
    'TextFillEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)
    texture_file_requested = Signal(int, str)

    def __init__(
        self,
        index: int,
        paint_type: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.index = int(index)
        if paint_type not in {'linear_gradient', 'texture'}:
            raise ValueError('unsupported foreground paint card type')
        self.paint_type = paint_type
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        if paint_type == 'linear_gradient':
            title = self.tr('Gradient')
            icon_name = 'text-effect-gradient.svg'
        else:
            title = self.tr('Texture')
            icon_name = 'text-effect-texture.svg'
        self.title_icon_label = _effect_icon_label(icon_name, self)
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show {effect}').format(effect=title),
            self.tr('Hide {effect}').format(effect=title),
            self,
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg',
            self.tr('Delete {effect}').format(effect=title),
            0,
        )
        self.delete_button.setObjectName('TextEffectCloseButton')

        action_widget = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        self._paint_seed: Optional[EffectPaint] = None
        self.gradient_editor: Optional[InlineLinearGradientEditor] = None
        self.texture_field: Optional[RasterAssetField] = None
        self.texture_button: Optional[QPushButton] = None
        self.texture_mapping_selector: Optional[BottomBorderComboBox] = None
        self.texture_scale_control: Optional[EffectNumericControl] = None

        texture_image_widget = None
        mapping_widget = None
        if paint_type == 'linear_gradient':
            self.gradient_editor = InlineLinearGradientEditor(
                LinearGradientPaint(), self
            )
            self.gradient_editor.paint_previewed.connect(
                self._on_gradient_preview
            )
            self.gradient_editor.paint_commit_requested.connect(
                self._on_gradient_commit
            )
            self.gradient_editor.paint_preview_canceled.connect(
                self._on_gradient_cancel
            )
            self.gradient_editor.color_dialog_active_changed.connect(
                self.color_dialog_active_changed.emit
            )
        else:
            self.texture_field = RasterAssetField(self)
            self.texture_button = self.texture_field.select_button
            self.texture_button.setAccessibleName(
                self.tr('Choose Texture Image')
            )
            self.texture_field.activated.connect(self._on_texture_clicked)
            texture_image_widget = _labeled_effect_editor(
                self, self.tr('Image'), self.texture_field
            )

            self.texture_mapping_selector = BottomBorderComboBox(
                self, text_alignment=Qt.AlignmentFlag.AlignCenter
            )
            self.texture_mapping_selector.setObjectName(
                'TextEffectParamEditor'
            )
            self.texture_mapping_selector.setAccessibleName(
                self.tr('Texture Mapping')
            )
            for label, value in (
                (self.tr('Fill'), 'fill'),
                (self.tr('Fit'), 'fit'),
                (self.tr('Crop'), 'crop'),
                (self.tr('Tile'), 'tile'),
            ):
                self.texture_mapping_selector.addItem(label, value)
            self.texture_mapping_selector.currentIndexChanged.connect(
                self._on_texture_mapping_changed
            )
            mapping_widget = _labeled_effect_editor(
                self, self.tr('Mapping'), self.texture_mapping_selector
            )
            self.texture_scale_control = EffectNumericControl(
                self.tr('Scale'), 'texture_scale', 100.0, 0.1, 4.0,
                '%', 1.0, self, decimals=1,
            )

        self.opacity_control = EffectNumericControl(
            self.tr('Opacity'), 'opacity', 100.0, 0.0, 1.0, '%', 1.0,
            self, decimals=1,
        )
        for control in self.iter_controls():
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(
                self._on_parameter_commit
            )
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(
                self._on_preview_canceled
            )
        blend_widget, self.blend_selector = _blend_control(
            self, self.tr('{effect} Blend').format(effect=title)
        )
        self.blend_selector.mode_changed.connect(
            self._on_blend_changed
        )

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        if paint_type == 'linear_gradient':
            controls.addWidget(self.opacity_control, 0, 0)
            controls.addWidget(blend_widget, 0, 1)
        else:
            assert texture_image_widget is not None
            assert mapping_widget is not None
            assert self.texture_scale_control is not None
            controls.addWidget(texture_image_widget, 0, 0)
            controls.addWidget(self.opacity_control, 0, 1)
            controls.addWidget(mapping_widget, 1, 0)
            controls.addWidget(blend_widget, 1, 1)
            # Scale remains available only for Tile mapping.
            controls.addWidget(self.texture_scale_control, 2, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        if self.gradient_editor is not None:
            layout.addWidget(self.gradient_editor)
        layout.addLayout(controls)

    def set_values(
        self,
        fills: Sequence[TextFillEffect],
        texture_available: Optional[bool] = None,
    ) -> None:
        """Project matching foreground-layer values into this fixed card."""
        if not fills or any(
            fill.paint.paint_type != self.paint_type for fill in fills
        ):
            raise ValueError('foreground card values must match its paint type')
        enabled_values = [fill.enabled for fill in fills]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        _set_blend_values(self.blend_selector, fills)
        opacity_values = [fill.opacity for fill in fills]
        common_opacity = (
            opacity_values[0]
            if opacity_values
            and all(value == opacity_values[0] for value in opacity_values)
            else None
        )
        self.opacity_control.set_model_value(common_opacity, opacity_values)
        paints = [fill.paint for fill in fills]
        common_paint = (
            paints[0]
            if paints and all(paint == paints[0] for paint in paints)
            else None
        )
        self._paint_seed = common_paint or paints[0]
        mixed_paint = common_paint is None
        if self.paint_type == 'linear_gradient':
            assert self.gradient_editor is not None
            assert isinstance(self._paint_seed, LinearGradientPaint)
            self.gradient_editor.set_paint(
                self._paint_seed, editable=not mixed_paint
            )
        else:
            assert self.texture_field is not None
            assert self.texture_button is not None
            assert self.texture_mapping_selector is not None
            assert self.texture_scale_control is not None
            textures = [paint for paint in paints if isinstance(
                paint, TexturePaint
            )]
            has_common_asset = bool(
                textures
                and all(
                    value.asset == textures[0].asset for value in textures
                )
            )
            common_asset = (
                textures[0].asset
                if has_common_asset
                else None
            )
            if not has_common_asset:
                display_name = self.tr('Mixed')
                hint = self.tr(
                    'Choose one image for the selected text items'
                )
                accessible_name = self.tr('Mixed Texture Images')
            elif common_asset is None:
                display_name = ''
                hint = self.tr('Choose an image for this Texture')
                accessible_name = self.tr('No Texture Image Selected')
            else:
                name = (
                    common_asset.display_name
                    or common_asset.path.rsplit('/', 1)[-1]
                )
                display_name = (
                    self.tr('Missing: {name}').format(name=name)
                    if texture_available is False else name
                )
                hint = name + '\n' + common_asset.path
                accessible_name = display_name
            self.texture_field.setText(display_name)
            self.texture_field.setCursorPosition(0)
            self.texture_field.setToolTip(hint)
            self.texture_field.setAccessibleName(accessible_name)
            self.texture_button.setToolTip(hint)
            mappings = [paint.mapping for paint in textures]
            common_mapping = (
                mappings[0]
                if mappings and all(value == mappings[0] for value in mappings)
                else None
            )
            with QSignalBlocker(self.texture_mapping_selector):
                self.texture_mapping_selector.setCurrentIndex(
                    -1
                    if common_mapping is None
                    else self.texture_mapping_selector.findData(common_mapping)
                )
            scales = [paint.scale for paint in textures]
            common_scale = (
                scales[0]
                if scales and all(value == scales[0] for value in scales)
                else None
            )
            self.texture_scale_control.set_model_value(common_scale, scales)
            self.texture_scale_control.setVisible(
                any(mapping == 'tile' for mapping in mappings)
            )
        self.layout().invalidate()
        self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            (self.opacity_control,)
            if self.texture_scale_control is None
            else (self.opacity_control, self.texture_scale_control)
        )

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)

    def _choose_texture_file(self) -> bool:
        self.color_dialog_active_changed.emit(True)
        try:
            path = _choose_project_raster(
                self, self.tr('Choose Texture Image')
            )
            if path:
                # The synchronous import/error chain stays pinned too.
                self.texture_file_requested.emit(self.index, path)
                return True
            return False
        finally:
            self.color_dialog_active_changed.emit(False)

    def _on_texture_clicked(self) -> None:
        self._choose_texture_file()

    def _on_texture_mapping_changed(self, combo_index: int) -> None:
        if combo_index >= 0 and self.texture_mapping_selector is not None:
            self.value_commit_requested.emit(
                self.index,
                'texture_mapping',
                self.texture_mapping_selector.itemData(combo_index),
            )

    def _on_blend_changed(self, blend_mode: str) -> None:
        self.value_commit_requested.emit(
            self.index, 'blend_mode', blend_mode
        )

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_gradient_preview(self, paint: LinearGradientPaint) -> None:
        self.value_preview_requested.emit(self.index, 'paint', paint)

    def _on_gradient_commit(self, paint: LinearGradientPaint) -> None:
        self.value_commit_requested.emit(self.index, 'paint', paint)

    def _on_gradient_cancel(self) -> None:
        self.preview_canceled.emit(self.index, 'paint')


class FilterEffectCard(_EffectCard):
    """One repeatable lazy filter at its complete-stack index."""

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    remove_requested = Signal(int)
    move_requested = Signal(int, int)

    def __init__(
        self,
        index: int,
        filter_id: str,
        spec: Optional[FilterSpec],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.index = int(index)
        self.filter_id = filter_id
        self.spec = spec
        self.numeric_controls = {}
        self.choice_selectors = {}
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-filter.svg', self
        )
        title = (
            _filter_ui_text(spec, spec.name)
            if spec is not None
            else self.tr('Missing Filter: {id}').format(id=filter_id)
        )
        self.title_label = QLabel(title, self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Filter'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Filter'), self.tr('Hide Filter'), self
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        actions = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(actions)
        header.addWidget(self.visibility_button)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        if spec is not None:
            for position, parameter in enumerate(spec.params):
                widget = self._parameter_widget(parameter)
                controls.addWidget(widget, position // 2, position % 2)
            controls.setColumnStretch(0, 1)
            controls.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        if spec is not None and spec.params:
            layout.addLayout(controls)

    def _parameter_widget(self, parameter: FilterParamSpec) -> QWidget:
        assert self.spec is not None
        signal_name = 'param:' + parameter.key
        label_text = _filter_ui_text(self.spec, parameter.label)
        if parameter.kind in {'float', 'int'}:
            assert parameter.minimum is not None
            assert parameter.maximum is not None
            control = EffectNumericControl(
                label_text,
                signal_name,
                parameter.display_factor,
                parameter.minimum,
                parameter.maximum,
                parameter.suffix,
                parameter.step,
                self,
                decimals=parameter.decimals,
            )
            control.layout().setSpacing(4)
            control.commit_requested.connect(self._on_control_commit)
            control.value_preview_requested.connect(self._on_value_preview)
            control.preview_requested.connect(self._on_parameter_preview)
            control.drag_commit_requested.connect(self._on_parameter_commit)
            control.preview_canceled.connect(self._on_preview_canceled)
            control.value_preview_canceled.connect(self._on_preview_canceled)
            self.numeric_controls[parameter.key] = control
            return control

        label = QLabel(label_text, self)
        label.setObjectName('TextEffectParamLabel')
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        selector.setObjectName('TextEffectParamEditor')
        selector.setProperty('filter-param', parameter.key)
        selector.setPlaceholderText(self.tr('Mixed'))
        selector.setAccessibleName(label_text)
        choices = (
            (('Off', False), ('On', True))
            if parameter.kind == 'bool'
            else parameter.choices
        )
        for choice_label, value in choices:
            selector.addItem(
                _filter_ui_text(self.spec, choice_label), value
            )
        selector.currentIndexChanged.connect(self._on_choice_changed)
        row_widget = QWidget(self)
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        row.addWidget(selector, 1)
        self.choice_selectors[parameter.key] = selector
        return row_widget

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_values(self, effects: Sequence[FilterEffect]) -> None:
        enabled_values = [effect.enabled for effect in effects]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        if self.spec is None:
            return
        failure = get_filter_registry().get_runtime_failure(self.filter_id)
        if failure is not None:
            self._set_parameter_controls_enabled(False)
            self.setToolTip(str(failure))
            return
        try:
            active_params = []
            for effect in effects:
                if effect.schema_version == self.spec.schema_version:
                    active_params.append(
                        self.spec.normalize_params(effect.params_dict())
                    )
                elif (
                    effect.enabled
                    and effect.schema_version < self.spec.schema_version
                ):
                    active_params.append(
                        dict(get_filter_registry().resolve(effect).params)
                    )
                else:
                    raise FilterUnavailableError(
                        f'{self.spec.name} schema {effect.schema_version} '
                        'is incompatible; enable/update it to migrate.'
                    )
        except (FilterUnavailableError, KeyError, ValueError) as error:
            self._set_parameter_controls_enabled(False)
            self.setToolTip(str(error))
            return
        self._set_parameter_controls_enabled(True)
        self.setToolTip('')
        for parameter in self.spec.params:
            values = [params[parameter.key] for params in active_params]
            common = (
                values[0]
                if values and all(value == values[0] for value in values)
                else None
            )
            control = self.numeric_controls.get(parameter.key)
            if control is not None:
                control.set_model_value(common, values)
                continue
            selector = self.choice_selectors[parameter.key]
            with QSignalBlocker(selector):
                selector.setCurrentIndex(
                    -1 if common is None else selector.findData(common)
                )

    def _set_parameter_controls_enabled(self, enabled: bool) -> None:
        for control in self.numeric_controls.values():
            control.setEnabled(enabled)
        for selector in self.choice_selectors.values():
            selector.setEnabled(enabled)

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return tuple(self.numeric_controls.values())

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(self.index, 'enabled', bool(enabled))

    def _on_control_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(self.index, name, value)

    def _on_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(self.index, name, value)

    def _on_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(self.index, name, delta)

    def _on_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(self.index, name, delta)

    def _on_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(self.index, name)

    def _on_choice_changed(self, combo_index: int) -> None:
        selector = self.sender()
        if combo_index < 0 or not isinstance(selector, BottomBorderComboBox):
            return
        key = selector.property('filter-param')
        if isinstance(key, str) and key:
            self.value_commit_requested.emit(
                self.index, 'param:' + key, selector.itemData(combo_index)
            )

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)


class RasterAssetField(QLineEdit):
    """Read-only raster name with a Glossary-style embedded picker.

    >>> RasterAssetField.__name__
    'RasterAssetField'
    """

    activated = Signal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        button_size: int = 20,
    ) -> None:
        super().__init__(parent)
        self._button_size = int(button_size)
        self.setObjectName('TextEffectRasterAssetField')
        self.setReadOnly(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.setFixedHeight(24)
        self.select_button = QPushButton(self)
        self.select_button.setObjectName('TextEffectRasterFileButton')
        self.select_button.setFixedSize(
            self._button_size, self._button_size
        )
        self.select_button.setIcon(QIcon(themed_icon_path('files.svg')))
        self.select_button.setIconSize(QSize(16, 16))
        self.select_button.clicked.connect(self.activated.emit)
        self.setTextMargins(4, 0, self._button_size + 2, 0)
        self._position_select_button()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_select_button()

    def _position_select_button(self) -> None:
        self.select_button.move(
            self.width() - self._button_size,
            max(0, (self.height() - self.select_button.height()) // 2),
        )


class ImageGenerationModelSelector(QToolButton):
    """Card-local image-capable LLM profile/model menu."""

    selection_changed = Signal(str, str)
    popup_active_changed = Signal(bool)
    ARROW_SIZE = 12

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.profile_id = ''
        self.model = ''
        self.backend = 'llm'
        self.setObjectName('TextEffectGenerationModelSelector')
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.setFixedHeight(24)
        self.menu = QMenu(self)
        self.menu.setObjectName('TextEffectAddMenu')
        self.menu.aboutToShow.connect(self._on_menu_about_to_show)
        self.clicked.connect(self._exec_menu)

    @staticmethod
    def _image_profiles() -> tuple:
        return tuple(
            profile
            for profile in pcfg.module.llm_profiles
            if profile.support_image
        )

    @classmethod
    def _default_selection(cls) -> Tuple[str, str]:
        profiles = cls._image_profiles()
        selected = profile_by_id(profiles, pcfg.module.inpaint_llm_id)
        profile = selected if selected is not None else (
            profiles[0] if profiles else None
        )
        if profile is None:
            return '', ''
        options = [
            str(option).strip()
            for option in profile.image_model_options
            if str(option).strip()
        ]
        model = str(profile.image_model or '').strip()
        if not model:
            model = options[0] if options else ''
        return profile.id, model

    def set_recipe(self, recipe: ImageGenerationRecipe) -> None:
        self.backend = recipe.backend
        self.profile_id = recipe.profile_id
        self.model = recipe.model
        if (
            self.backend == 'llm'
            and not self.profile_id
            and not self.model
        ):
            self.profile_id, self.model = self._default_selection()
        self._sync_text()

    def has_available_selection(self) -> bool:
        if self.backend != 'llm' or not self.model.strip():
            return False
        profile = profile_by_id(pcfg.module.llm_profiles, self.profile_id)
        return bool(profile is not None and profile.support_image)

    def _sync_text(self) -> None:
        profile = profile_by_id(pcfg.module.llm_profiles, self.profile_id)
        if self.backend != 'llm':
            text = self.tr('Unavailable: {backend}').format(
                backend=self.backend
            )
        elif profile is None and self.profile_id:
            text = self.tr('Missing: {profile}').format(
                profile=self.profile_id
            )
        elif self.model:
            text = _simplify_llm_model_name(self.model)
        elif profile is not None:
            text = profile.name or profile.id
        else:
            text = self.tr('No Models')
        self.setText(text)
        self.setToolTip(text)

    def _on_menu_about_to_show(self) -> None:
        self.popup_active_changed.emit(True)
        self._rebuild_menu()

    def _exec_menu(self) -> None:
        self.setDown(True)
        try:
            self.menu.exec_(
                self.mapToGlobal(QPoint(0, self.height()))
            )
        finally:
            # exec_ returns only after Qt has completed the popup close path.
            self.setDown(False)
            self.popup_active_changed.emit(False)

    def _rebuild_menu(self) -> None:
        self.menu.clear()
        _add_bottom_menu_section(
            self.menu, self.tr('LLM'), color=LLM_MODALITY_IMAGE_COLOR
        )
        profiles = self._image_profiles()
        if not profiles:
            action = QAction(self.tr('No image profiles'), self.menu)
            action.setEnabled(False)
            self.menu.addAction(action)
            return
        for profile in profiles:
            profile_menu = _bottom_submenu(
                profile.name or profile.id, self.menu
            )
            _add_bottom_submenu(
                self.menu,
                profile_menu,
                profile.name or profile.id,
                self.backend == 'llm' and self.profile_id == profile.id,
            )
            _add_bottom_menu_section(
                profile_menu,
                self.tr('Image Model'),
                color=LLM_MODALITY_IMAGE_COLOR,
            )
            options = [
                str(option).strip()
                for option in profile.image_model_options
                if str(option).strip()
            ]
            configured = str(profile.image_model or '').strip()
            if configured and configured not in options:
                options.insert(0, configured)
            for model in options:
                _add_bottom_menu_action(
                    profile_menu,
                    model,
                    self.backend == 'llm'
                    and self.profile_id == profile.id
                    and self.model == model,
                    (profile.id, model),
                    self._select_action,
                )
            if not options:
                action = QAction(self.tr('No image models'), profile_menu)
                action.setEnabled(False)
                profile_menu.addAction(action)

    def _select_action(self, _checked: bool = False) -> None:
        action = self.sender()
        if not isinstance(action, QAction):
            return
        profile_id, model = action.data()
        self.backend = 'llm'
        self.profile_id = str(profile_id)
        self.model = str(model)
        self._sync_text()
        self.selection_changed.emit(self.profile_id, self.model)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        pixmap = render_svg_pixmap(
            themed_icon_path('chevron-down.svg'),
            self.ARROW_SIZE,
            self.ARROW_SIZE,
            self.devicePixelRatioF(),
        )
        x = self.width() - self.ARROW_SIZE - 4
        y = (self.height() - self.ARROW_SIZE) // 2
        painter.drawPixmap(x, y, pixmap)
        painter.end()


class ImageGenerationPromptEditor(QPlainTextEdit):
    """Compact wrapping prompt editor with bounded natural height."""

    natural_height_changed = Signal()

    MIN_HEIGHT = 42
    MAX_HEIGHT = 88

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName('TextEffectGenerationPromptEditor')
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setMinimumHeight(self.MIN_HEIGHT)
        self.document().documentLayout().documentSizeChanged.connect(
            self._adjust_to_document
        )
        self._adjust_to_document()

    def showEvent(self, event: QEvent) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._adjust_to_document)

    def _adjust_to_document(self, *_size: object) -> None:
        layout = self.document().documentLayout()
        block = self.document().begin()
        content_height = 0.0
        while block.isValid() and content_height < self.MAX_HEIGHT:
            content_height += layout.blockBoundingRect(block).height()
            block = block.next()
        height = max(
            self.MIN_HEIGHT,
            min(
                self.MAX_HEIGHT,
                round(content_height + self.frameWidth() * 2 + 6),
            ),
        )
        if (
            self.minimumHeight() != height
            or self.maximumHeight() != height
        ):
            self.setFixedHeight(height)
            self.natural_height_changed.emit()


class ImageEffectCard(_EffectCard):
    """Edit one project-owned Image effect at its stack index.

    >>> ImageEffectCard.__name__
    'ImageEffectCard'
    """

    value_commit_requested = Signal(int, str, object)
    image_requested = Signal(int)
    generate_requested = Signal(int, object)
    stop_requested = Signal()
    remove_requested = Signal(int)
    move_requested = Signal(int, int)
    natural_height_changed = Signal()

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.index = int(index)
        self._generation_draft = ImageGenerationRecipe()
        self._generation_draft_dirty = False
        self._generation_eligible = False
        self._generation_eligibility_hint = ''
        self._generation_state = 'idle'
        self._generation_panel_busy = False
        self.setObjectName('TextEffectParameterPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-image.svg', self
        )
        self.title_label = QLabel(self.tr('Image'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self._editing_hint = self.tr(
            'Hidden while editing so the caret and selection match the text.'
        )
        self._choose_hint = self.tr('Choose an image...')
        self._choose_mixed_hint = self.tr(
            'Choose one image for all selected text...'
        )
        self.setToolTip(self._editing_hint)
        self.title_label.setToolTip(self._editing_hint)
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Image'),
            self.tr('Hide Image'),
            self,
        )
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        self.move_up_button = self._action_button(
            'chevron-up.svg', self.tr('Move Up'), -1
        )
        self.move_down_button = self._action_button(
            'chevron-down.svg', self.tr('Move Down'), 1
        )
        self.delete_button = self._action_button(
            'titlebar_close.svg', self.tr('Delete Image'), 0
        )
        self.delete_button.setObjectName('TextEffectCloseButton')
        actions = _effect_action_widget(
            self,
            (
                self.move_up_button,
                self.move_down_button,
                self.delete_button,
            ),
        )

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(actions)
        header.addWidget(self.visibility_button)

        self.image_field = RasterAssetField(self)
        self.image_button = self.image_field.select_button
        self.image_button.setToolTip(self._choose_hint)
        self.image_button.setAccessibleName(self.tr('Choose Image'))
        self.image_field.activated.connect(self._on_image_requested)
        image_widget = _labeled_effect_editor(
            self, self.tr('Image'), self.image_field
        )

        self.mode_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.mode_selector.setObjectName('TextEffectParamEditor')
        self.mode_selector.setAccessibleName(self.tr('Image Placement'))
        for label, mode, hint in (
            (
                self.tr('In Front'),
                'foreground',
                self.tr(
                    'Draws the Image over everything rendered before it.'
                ),
            ),
            (
                self.tr('Behind'),
                'background',
                self.tr(
                    'Draws the Image behind everything rendered before it.'
                ),
            ),
        ):
            self.mode_selector.addItem(label, mode)
            self.mode_selector.setItemData(
                self.mode_selector.count() - 1,
                hint,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_changed
        )
        self._sync_placement_hint(self.mode_selector.currentIndex())
        mode_widget = _labeled_effect_editor(
            self, self.tr('Placement'), self.mode_selector
        )

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(image_widget, 0, 0)
        controls.addWidget(mode_widget, 0, 1)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

        generate_label = QLabel(self.tr('Generate'), self)
        generate_label.setObjectName('TextEffectGenerateSectionTitle')

        self.model_selector = ImageGenerationModelSelector(self)
        self.model_selector.setAccessibleName(
            self.tr('Image Generation Model')
        )
        self.model_selector.selection_changed.connect(
            self._on_model_changed
        )
        model_widget = _labeled_effect_editor(
            self, self.tr('Model'), self.model_selector
        )

        self.context_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.context_selector.setObjectName('TextEffectParamEditor')
        self.context_selector.setAccessibleName(
            self.tr('Image Generation Context')
        )
        self.context_selector.addItem(self.tr('Source'), 'source')
        self.context_selector.addItem(self.tr('Inpainted'), 'inpainted')
        self.context_selector.addItem(self.tr('Lettered'), 'lettered')
        self.context_selector.addItem(self.tr('None'), 'none')
        self.context_selector.currentIndexChanged.connect(
            self._on_context_changed
        )
        context_widget = _labeled_effect_editor(
            self, self.tr('Context'), self.context_selector
        )

        self.prompt_editor = ImageGenerationPromptEditor(self)
        self.prompt_editor.setPlaceholderText(
            self.tr('Describe the image to generate or edit')
        )
        self.prompt_editor.setAccessibleName(
            self.tr('Image Generation Prompt')
        )
        self.prompt_editor.textChanged.connect(self._on_prompt_changed)
        self.prompt_editor.natural_height_changed.connect(
            self._on_prompt_height_changed
        )
        prompt_label = QLabel(self.tr('Prompt'), self)
        prompt_label.setObjectName('TextEffectParamLabel')
        prompt_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        prompt_layout = QVBoxLayout()
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        prompt_layout.setSpacing(4)
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_editor)

        self.generate_button = QToolButton(self)
        self.generate_button.setObjectName('TextEffectGenerateButton')
        self.generate_button.setProperty('running', False)
        self.generate_button.setText(self.tr('Generate'))
        self.generate_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.generate_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.generate_button.setFixedHeight(26)
        self.generate_button.clicked.connect(self._on_generate_clicked)

        generation_actions = QHBoxLayout()
        generation_actions.setContentsMargins(0, 0, 0, 0)
        generation_actions.setSpacing(8)
        generation_actions.addWidget(context_widget, 1)
        generation_actions.addWidget(self.generate_button, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)
        layout.addWidget(generate_label)
        layout.addWidget(model_widget)
        layout.addLayout(prompt_layout)
        layout.addLayout(generation_actions)
        self._sync_generation_controls()

    def _action_button(
        self, icon_name: str, tooltip: str, direction: int
    ) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName('TextEffectMoveButton')
        button.setIcon(QIcon(themed_icon_path(icon_name)))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setProperty('move-direction', direction)
        button.setFixedSize(18, 18)
        button.clicked.connect(self._on_action_clicked)
        return button

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def iter_controls(self) -> tuple:
        return ()

    def _on_prompt_height_changed(self) -> None:
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
        self.updateGeometry()
        self.natural_height_changed.emit()

    def set_values(
        self,
        effects: Sequence[ImageEffect],
        available: Optional[bool],
        *,
        generation_eligible: bool = False,
        generation_eligibility_hint: str = '',
    ) -> None:
        enabled_values = [effect.enabled for effect in effects]
        enabled = (
            enabled_values[0]
            if enabled_values
            and all(value == enabled_values[0] for value in enabled_values)
            else None
        )
        self.visibility_button.set_visibility(enabled)
        assets = [effect.asset for effect in effects]
        common_asset = (
            assets[0]
            if assets and all(asset == assets[0] for asset in assets)
            else None
        )
        assets_mixed = bool(assets) and any(
            asset != assets[0] for asset in assets[1:]
        )
        if assets_mixed:
            display_name = self.tr('Mixed')
            hint = self._choose_mixed_hint + '\n' + self._editing_hint
            accessible_name = self.tr('Mixed Images')
        elif common_asset is None:
            display_name = ''
            hint = self._choose_hint + '\n' + self._editing_hint
            accessible_name = self.tr('No Image Selected')
        else:
            name = (
                common_asset.display_name
                or common_asset.path.rsplit('/', 1)[-1]
            )
            display_name = (
                name if available is not False
                else self.tr('Missing: {name}').format(name=name)
            )
            hint = (
                name + '\n' + common_asset.path + '\n' + self._editing_hint
            )
            accessible_name = display_name
        self.image_field.setText(display_name)
        self.image_field.setCursorPosition(0)
        self.image_field.setToolTip(hint)
        self.image_field.setAccessibleName(accessible_name)
        self.image_button.setToolTip(hint)
        modes = [effect.mode for effect in effects]
        common_mode = (
            modes[0]
            if modes and all(mode == modes[0] for mode in modes)
            else None
        )
        with QSignalBlocker(self.mode_selector):
            self.mode_selector.setCurrentIndex(
                -1
                if common_mode is None
                else self.mode_selector.findData(common_mode)
            )
        self._sync_placement_hint(self.mode_selector.currentIndex())
        self._generation_eligible = bool(generation_eligible)
        self._generation_eligibility_hint = generation_eligibility_hint
        if (
            len(effects) == 1
            and self._generation_draft_dirty
            and effects[0].generation == self._generation_draft
        ):
            # A successful generation committed this exact draft. Future
            # history changes should once again project the persisted recipe.
            self._generation_draft_dirty = False
        if len(effects) == 1 and not self._generation_draft_dirty:
            recipe = effects[0].generation
            self.model_selector.set_recipe(recipe)
            self._generation_draft = replace(
                recipe,
                backend=self.model_selector.backend,
                profile_id=self.model_selector.profile_id,
                model=self.model_selector.model,
            )
            with QSignalBlocker(self.context_selector):
                self.context_selector.setCurrentIndex(
                    self.context_selector.findData(recipe.context)
                )
            if self.prompt_editor.toPlainText() != recipe.prompt:
                with QSignalBlocker(self.prompt_editor):
                    self.prompt_editor.setPlainText(recipe.prompt)
        self._sync_generation_controls()

    def reset_generation_draft(self) -> None:
        """Project the next selected item's persisted recipe."""
        self._generation_draft_dirty = False

    def set_generation_state(
        self, state: str, *, panel_busy: bool = False
    ) -> None:
        if state not in {'idle', 'running', 'stopping'}:
            raise ValueError('unsupported Image generation state')
        self._generation_state = state
        self._generation_panel_busy = bool(panel_busy)
        self._sync_generation_controls()

    def _sync_generation_controls(self) -> None:
        active = self._generation_state != 'idle'
        ready = (
            self._generation_eligible
            and self.model_selector.has_available_selection()
            and not self._generation_panel_busy
        )
        for control in (
            self.model_selector,
            self.context_selector,
            self.prompt_editor,
        ):
            control.setEnabled(
                self._generation_eligible
                and not active
                and not self._generation_panel_busy
            )
        self.generate_button.setText(
            self.tr('Stop') if active else self.tr('Generate')
        )
        running_changed = (
            bool(self.generate_button.property('running')) != active
        )
        if running_changed:
            self.generate_button.setProperty('running', active)
        self.generate_button.setEnabled(
            (active and self._generation_state == 'running')
            or (not active and ready)
        )
        if not self._generation_eligible:
            hint = self._generation_eligibility_hint or self.tr(
                'Select exactly one text item to generate an Image.'
            )
        elif not self.model_selector.has_available_selection():
            hint = self.tr('Select an available image generation model.')
        elif self._generation_panel_busy:
            hint = self.tr(
                'Another Image generation request is in progress.'
            )
        elif self._generation_state == 'stopping':
            hint = self.tr(
                'Waiting for the current image request to stop.'
            )
        elif active:
            hint = self.tr('Stop image generation')
        else:
            hint = self.tr('Generate an image for this effect')
        self.generate_button.setToolTip(hint)
        self.generate_button.setAccessibleName(
            self.tr('Stop Image Generation')
            if active
            else self.tr('Generate Image')
        )
        if running_changed:
            # Dynamic property styling changes only from a normal signal slot.
            self.generate_button.style().unpolish(self.generate_button)
            self.generate_button.style().polish(self.generate_button)
            self.generate_button.update()

    def _on_model_changed(self, profile_id: str, model: str) -> None:
        self._generation_draft = replace(
            self._generation_draft,
            backend='llm',
            profile_id=profile_id,
            model=model,
        )
        self._generation_draft_dirty = True
        self._sync_generation_controls()

    def _on_context_changed(self, index: int) -> None:
        context = self.context_selector.itemData(index)
        if context not in {'source', 'inpainted', 'lettered', 'none'}:
            return
        self._generation_draft = replace(
            self._generation_draft, context=context
        )
        self._generation_draft_dirty = True

    def _on_prompt_changed(self) -> None:
        self._generation_draft = replace(
            self._generation_draft,
            prompt=self.prompt_editor.toPlainText(),
        )
        self._generation_draft_dirty = True

    def _on_generate_clicked(self) -> None:
        if self._generation_state != 'idle':
            self.stop_requested.emit()
            return
        if not self._generation_eligible:
            return
        recipe = replace(
            self._generation_draft,
            backend=self.model_selector.backend,
            profile_id=self.model_selector.profile_id,
            model=self.model_selector.model,
            context=str(self.context_selector.currentData()),
            prompt=self.prompt_editor.toPlainText(),
        )
        self._generation_draft = recipe
        self._generation_draft_dirty = True
        self.generate_requested.emit(self.index, recipe)

    def _on_mode_changed(self, index: int) -> None:
        self._sync_placement_hint(index)
        mode = self.mode_selector.itemData(index)
        if mode in {'foreground', 'background'}:
            self.value_commit_requested.emit(self.index, 'mode', mode)

    def _sync_placement_hint(self, index: int) -> None:
        hint = self.mode_selector.itemData(
            index, Qt.ItemDataRole.ToolTipRole
        )
        text = hint if isinstance(hint, str) else ''
        self.mode_selector.setToolTip(text)
        self.mode_selector.setAccessibleDescription(text)

    def _on_enabled_clicked(self, enabled: bool) -> None:
        self.value_commit_requested.emit(
            self.index, 'enabled', bool(enabled)
        )

    def _on_image_requested(self) -> None:
        self.image_requested.emit(self.index)

    def _on_action_clicked(self) -> None:
        button = self.sender()
        direction = int(button.property('move-direction'))
        if direction == 0:
            self.remove_requested.emit(self.index)
        else:
            self.move_requested.emit(self.index, direction)


class AlphaMaskCard(_EffectCard):
    """Pinned controls for the selected TextBlock-owned mask.

    >>> AlphaMaskCard.__name__
    'AlphaMaskCard'
    """

    enabled_requested = Signal(bool)
    mode_changed = Signal(str)
    diameter_changed = Signal(float)
    clear_requested = Signal()
    remove_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName('TextAlphaMaskCard')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.title_icon_label = _effect_icon_label(
            'text-effect-alpha-mask.svg', self
        )
        self.title_label = QLabel(self.tr('Eraser'), self)
        self.title_label.setObjectName('TextEffectParameterTitle')
        self.title_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.visibility_button = EffectVisibilityButton(
            self.tr('Show Eraser'), self.tr('Hide Eraser'), self
        )
        self.visibility_button.visibility_requested.connect(
            self.enabled_requested.emit
        )
        self.remove_button = QToolButton(self)
        self.remove_button.setObjectName('TextEffectCloseButton')
        self.remove_button.setIcon(
            QIcon(themed_icon_path('titlebar_close.svg'))
        )
        self.remove_button.setToolTip(self.tr('Remove Eraser'))
        self.remove_button.setAccessibleName(self.tr('Remove Eraser'))
        self.remove_button.setFixedSize(18, 18)
        self.remove_button.clicked.connect(self.remove_requested.emit)

        action_widget = _effect_action_widget(self, (self.remove_button,))

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_icon_label)
        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(action_widget)
        header.addWidget(self.visibility_button)

        mode_label = QLabel(self.tr('Mode'), self)
        mode_label.setObjectName('TextEffectParamLabel')
        mode_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.mode_selector = BottomBorderComboBox(
            self, text_alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.mode_selector.setObjectName('TextEffectParamEditor')
        self.mode_selector.addItem(self.tr('Erase'), 'erase')
        self.mode_selector.addItem(self.tr('Restore'), 'restore')
        self.mode_selector.currentIndexChanged.connect(
            self._on_mode_changed
        )
        mode_widget = QWidget(self)
        mode_layout = QHBoxLayout(mode_widget)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(8)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_selector, 1)

        self.size_label = TransformDragLabel(
            self,
            direction=0,
            text=self.tr('Size'),
            alignment=(
                Qt.AlignmentFlag.AlignLeft
                | Qt.AlignmentFlag.AlignVCenter
            ),
        )
        self.size_label.setObjectName('TextEffectParamLabel')
        self.diameter_editor = QDoubleSpinBox(self)
        self.diameter_editor.setObjectName('TextEffectParamEditor')
        self.diameter_editor.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.diameter_editor.setRange(1.0, 500.0)
        self.diameter_editor.setDecimals(1)
        self.diameter_editor.setSingleStep(1.0)
        self.diameter_editor.setSuffix(self.tr(' px'))
        button_symbols = getattr(
            QAbstractSpinBox, 'ButtonSymbols', QAbstractSpinBox
        )
        self.diameter_editor.setButtonSymbols(button_symbols.NoButtons)
        self.diameter_editor.valueChanged.connect(
            self.diameter_changed.emit
        )
        self._diameter_drag_start: Optional[float] = None
        self.size_label.drag_started.connect(self._begin_diameter_drag)
        self.size_label.size_ctrl_changed.connect(
            self._change_diameter_by_drag
        )
        self.size_label.btn_released.connect(self._finish_diameter_drag)
        self.size_label.drag_canceled.connect(self._cancel_diameter_drag)
        size_widget = QWidget(self)
        size_layout = QHBoxLayout(size_widget)
        size_layout.setContentsMargins(0, 0, 0, 0)
        size_layout.setSpacing(8)
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.diameter_editor, 1)

        self.clear_button = QToolButton(self)
        self.clear_button.setObjectName('TextAlphaMaskClearButton')
        self.clear_button.setText(self.tr('Clear'))
        self.clear_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.clear_button.clicked.connect(self.clear_requested.emit)

        controls = QGridLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(8)
        controls.addWidget(mode_widget, 0, 0)
        controls.addWidget(size_widget, 0, 1)
        controls.addWidget(self.clear_button, 0, 2)
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(8)
        layout.addLayout(header)
        layout.addLayout(controls)

    def set_values(
        self,
        mask: TextAlphaMask,
        mode: str,
        diameter: float,
    ) -> None:
        blockers = (
            QSignalBlocker(self.mode_selector),
            QSignalBlocker(self.diameter_editor),
        )
        self.visibility_button.set_visibility(mask.enabled)
        index = self.mode_selector.findData(mode)
        self.mode_selector.setCurrentIndex(max(0, index))
        self.diameter_editor.setValue(diameter)
        del blockers

    def _on_mode_changed(self, index: int) -> None:
        mode = self.mode_selector.itemData(index)
        if mode in {'erase', 'restore'}:
            self.mode_changed.emit(mode)

    def _begin_diameter_drag(self) -> None:
        self._diameter_drag_start = self.diameter_editor.value()

    def _change_diameter_by_drag(self, delta: int) -> None:
        self.diameter_editor.setValue(
            self.diameter_editor.value()
            + delta * self.diameter_editor.singleStep()
        )

    def _finish_diameter_drag(self) -> None:
        self._diameter_drag_start = None

    def _cancel_diameter_drag(self) -> None:
        if self._diameter_drag_start is None:
            return
        value = self._diameter_drag_start
        self._diameter_drag_start = None
        self.diameter_editor.setValue(value)


class TextEffectPanel(PanelArea):
    """Own Overall Opacity and typed effect cards.

    >>> TextEffectPanel.__name__
    'TextEffectPanel'
    """

    value_commit_requested = Signal(int, str, object)
    value_preview_requested = Signal(int, str, object)
    parameter_preview_requested = Signal(int, str, object)
    parameter_commit_requested = Signal(int, str, object)
    preview_canceled = Signal(int, str)
    add_effect_requested = Signal(str)
    add_filter_requested = Signal(str)
    hollow_enabled_requested = Signal(bool)
    remove_effect_requested = Signal(int)
    move_effect_requested = Signal(int, int)
    color_dialog_active_changed = Signal(bool)
    mask_edit_requested = Signal(bool)
    mask_enabled_requested = Signal(bool)
    mask_mode_changed = Signal(str)
    mask_diameter_changed = Signal(float)
    mask_clear_requested = Signal()
    mask_remove_requested = Signal()
    texture_file_requested = Signal(int, str)
    image_file_requested = Signal(int, str)
    image_generation_requested = Signal(int, object)
    image_generation_stop_requested = Signal()

    MAX_CONTENT_HEIGHT = 480

    def __init__(
        self,
        panel_name: str,
        config_name: str,
        config_expand_name: str,
    ) -> None:
        super().__init__(panel_name, config_name, config_expand_name)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.scrollContent.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.setMaximumHeight(self.MAX_CONTENT_HEIGHT)

        self.overall_opacity_control = EffectNumericControl(
            self.tr('Opacity'),
            'overall_opacity',
            100.0,
            0.0,
            1.0,
            '%',
            1.0,
            self.scrollContent,
            decimals=1,
        )
        self.overall_opacity_control.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred
        )
        overall_opacity_hint = self.tr(
            'Overall opacity of the text and all effects'
        )
        self.overall_opacity_control.label.setToolTip(overall_opacity_hint)
        self.overall_opacity_control.editor.setToolTip(overall_opacity_hint)
        self.overall_opacity_control.commit_requested.connect(
            self._on_overall_commit
        )
        self.overall_opacity_control.value_preview_requested.connect(
            self._on_overall_value_preview
        )
        self.overall_opacity_control.preview_requested.connect(
            self._on_overall_parameter_preview
        )
        self.overall_opacity_control.drag_commit_requested.connect(
            self._on_overall_parameter_commit
        )
        self.overall_opacity_control.preview_canceled.connect(
            self._on_overall_preview_canceled
        )
        self.overall_opacity_control.value_preview_canceled.connect(
            self._on_overall_preview_canceled
        )

        self.mask_brush_button = QToolButton(self.scrollContent)
        self.mask_brush_button.setObjectName('TextEffectBrushButton')
        self.mask_brush_button.setIcon(
            QIcon(themed_icon_path('text-effect-alpha-mask.svg'))
        )
        self.mask_brush_button.setIconSize(QSize(16, 16))
        self.mask_brush_button.setFixedSize(26, 26)
        self.mask_brush_button.setCheckable(True)
        self.mask_brush_button.setEnabled(False)
        self.mask_brush_button.setToolTip(
            self.tr('Select one text block in text edit mode.')
        )
        self.mask_brush_button.setAccessibleName(
            self.tr('Text Eraser')
        )
        self.mask_brush_button.clicked.connect(
            self._on_mask_brush_clicked
        )

        self.hollow_toggle_button = QToolButton(self.scrollContent)
        self.hollow_toggle_button.setObjectName('TextEffectHollowButton')
        self.hollow_toggle_button.setIcon(
            QIcon(themed_icon_path('text-effect-hollow.svg'))
        )
        self.hollow_toggle_button.setIconSize(QSize(16, 16))
        self.hollow_toggle_button.setFixedSize(26, 26)
        self.hollow_toggle_button.setCheckable(True)
        self.hollow_toggle_button.setProperty('mixed', False)
        self.hollow_toggle_button.clicked.connect(
            self._on_hollow_toggled
        )
        self._set_hollow_toggle_state(False)

        self.add_effect_button = QToolButton(self.scrollContent)
        self.add_effect_button.setObjectName('AddTextEffectButton')
        self.add_effect_button.setText(self.tr('Add'))
        self.add_effect_button.setToolTip(self.tr('Add Effect'))
        self.add_effect_button.setAccessibleName(self.tr('Add Effect'))
        self.add_effect_button.setFixedSize(72, 26)
        self.add_effect_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.add_effect_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        add_menu = QMenu(self.add_effect_button)
        add_menu.setObjectName('TextEffectAddMenu')
        self.add_effect_actions = {}
        for label, effect_type, icon_name in (
            (self.tr('Stroke'), 'stroke', 'text-effect-stroke.svg'),
            (self.tr('Shadow'), 'shadow', 'text-effect-shadow.svg'),
            (self.tr('Glow'), 'glow', 'text-effect-glow.svg'),
            (
                self.tr('Gradient'),
                'gradient',
                'text-effect-gradient.svg',
            ),
            (
                self.tr('Texture'),
                'texture',
                'text-effect-texture.svg',
            ),
            (
                self.tr('Image'),
                'image',
                'text-effect-image.svg',
            ),
        ):
            action = add_menu.addAction(
                QIcon(themed_icon_path(icon_name)), label
            )
            action.setData(effect_type)
            action.triggered.connect(self._on_add_effect_triggered)
            self.add_effect_actions[effect_type] = action
        self.filter_add_menu = add_menu.addMenu(
            QIcon(themed_icon_path('text-effect-filter.svg')),
            self.tr('Filter'),
        )
        for spec in get_filter_registry().specs:
            action = self.filter_add_menu.addAction(
                QIcon(themed_icon_path('text-effect-filter.svg')),
                _filter_ui_text(spec, spec.name),
            )
            action.setData(spec.filter_id)
            action.triggered.connect(self._on_add_filter_triggered)
        self.add_effect_button.setMenu(add_menu)
        self.add_effect_actions['texture'].setEnabled(False)
        self.add_effect_actions['image'].setEnabled(False)

        self.faster_preview_toggle = QCheckBox(
            self.tr('Faster Preview'), self.view_widget.title_label
        )
        self.faster_preview_toggle.setObjectName(
            'TextEffectFasterPreviewToggle'
        )
        faster_preview_hint = self.tr(
            'Render live effect changes at half resolution. Committed and '
            'exported text keep full quality.'
        )
        self.faster_preview_toggle.setToolTip(faster_preview_hint)
        self.faster_preview_toggle.setAccessibleDescription(
            faster_preview_hint
        )
        self.faster_preview_toggle.toggled.connect(
            self._on_faster_preview_toggled
        )
        title_layout = self.view_widget.title_label.layout()
        title_layout.insertWidget(
            title_layout.count() - 1, self.faster_preview_toggle
        )

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)
        top_row.addWidget(self.add_effect_button)
        top_row.addWidget(self.mask_brush_button)
        top_row.addWidget(self.hollow_toggle_button)
        top_row.addStretch()
        top_row.addWidget(self.overall_opacity_control)

        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.effect_cards = []
        self.stroke_cards = []
        self.shadow_cards = []
        self.glow_cards = []
        self.text_fill_cards = []
        self.image_cards = []
        self.filter_cards = []
        self._effect_types = None
        self._image_generation_index = -1
        self._image_generation_state = 'idle'
        self.alpha_mask_card = None
        self._block_items = ()
        self._alpha_mask_session = None
        self.base_card_layout = QVBoxLayout()
        self.base_card_layout.setContentsMargins(0, 0, 0, 0)
        self.mask_card_layout = QVBoxLayout()
        self.mask_card_layout.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top_row)
        layout.addLayout(self.base_card_layout)
        layout.addLayout(self.cards_layout)
        layout.addLayout(self.mask_card_layout)
        self.setContentLayout(layout)
        self.content_layout = layout
        self.scrollContent.after_resized.connect(self._sync_content_height)
        self._sync_content_height()
        QTimer.singleShot(0, self._sync_content_height)

    def set_alpha_mask_session(
        self, session: "TextAlphaMaskEditSession"
    ) -> None:
        self._alpha_mask_session = session
        self.refresh_alpha_mask_state()

    def _set_alpha_mask_card(self, present: bool) -> None:
        if present and self.alpha_mask_card is None:
            card = AlphaMaskCard(self.scrollContent)
            card.enabled_requested.connect(
                self.mask_enabled_requested.emit
            )
            card.mode_changed.connect(self.mask_mode_changed.emit)
            card.diameter_changed.connect(
                self.mask_diameter_changed.emit
            )
            card.clear_requested.connect(self.mask_clear_requested.emit)
            card.remove_requested.connect(self.mask_remove_requested.emit)
            self.mask_card_layout.addWidget(card)
            card.show()
            self.alpha_mask_card = card
        elif not present and self.alpha_mask_card is not None:
            card = self.alpha_mask_card
            self.alpha_mask_card = None
            self.mask_card_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()

    def refresh_alpha_mask_state(self) -> None:
        session = self._alpha_mask_session
        item = self._block_items[0] if len(self._block_items) == 1 else None
        if item is not None and session is not None:
            try:
                attached = item.scene() is session.canvas
            except RuntimeError:
                attached = False
            if not attached:
                self._block_items = ()
                item = None
        mask = None if item is None else item.blk.text_alpha_mask
        self._set_alpha_mask_card(mask is not None)
        eligible = bool(session is not None and session.can_activate(item))
        active = bool(
            session is not None and session.active and session.target is item
        )
        self.mask_brush_button.setEnabled(eligible)
        blocker = QSignalBlocker(self.mask_brush_button)
        self.mask_brush_button.setChecked(active)
        del blocker
        self.mask_brush_button.setToolTip(
            self.tr('Edit Text Eraser')
            if eligible
            else self.tr('Select one text block in text edit mode.')
        )
        if self.alpha_mask_card is not None and mask is not None:
            self.alpha_mask_card.setEnabled(eligible)
            self.alpha_mask_card.set_values(
                mask,
                session.mode if session is not None else 'erase',
                session.diameter if session is not None else 24.0,
            )
        self._sync_content_height()

    def _choose_image_file(self, index: int) -> None:
        if not self._block_items:
            return
        self.color_dialog_active_changed.emit(True)
        try:
            path = _choose_project_raster(
                self, self.tr('Choose Image')
            )
            if path:
                # The synchronous import/error chain stays pinned too.
                self.image_file_requested.emit(index, path)
        finally:
            self.color_dialog_active_changed.emit(False)

    def show_image_import_error(self, index: int, message: str) -> None:
        if not any(card.index == index for card in self.image_cards):
            return
        QMessageBox.warning(
            self.window(),
            self.tr('Unable to Import Image'),
            self.tr(
                'The selected image could not be added to this project.'
                '\n\n{message}'
            ).format(message=message),
        )

    def _clear_effect_cards(self) -> None:
        for card in self.effect_cards:
            (
                self.base_card_layout
                if isinstance(card, TextFillEffectCard)
                else self.cards_layout
            ).removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self.effect_cards = []
        self.stroke_cards = []
        self.shadow_cards = []
        self.glow_cards = []
        self.text_fill_cards = []
        self.image_cards = []
        self.filter_cards = []

    def _rebuild_effect_cards(
        self,
        effect_keys: Sequence[object],
        seed: Optional[TextEffectStack] = None,
    ) -> None:
        effect_keys = tuple(effect_keys)
        if effect_keys == self._effect_types:
            return
        self._clear_effect_cards()
        self._effect_types = effect_keys
        # The model stays topmost-first, while the panel shows the renderer's
        # bottom-to-top application order.
        for index, effect_key in reversed(tuple(enumerate(effect_keys))):
            effect_type = (
                effect_key[0]
                if isinstance(effect_key, tuple)
                else effect_key
            )
            if effect_type == 'stroke':
                card = StrokeEffectCard(index, self.scrollContent)
                self.stroke_cards.append(card)
            elif effect_type == 'shadow':
                card = ShadowEffectCard(index, self.scrollContent)
                self.shadow_cards.append(card)
            elif effect_type == 'glow':
                card = GlowEffectCard(index, self.scrollContent)
                self.glow_cards.append(card)
            elif effect_type == 'hollow':
                continue
            elif effect_type == 'text_fill':
                fill_effect = (
                    None if seed is None else seed.effects[index]
                )
                if not isinstance(fill_effect, TextFillEffect):
                    continue
                card = TextFillEffectCard(
                    index, fill_effect.paint.paint_type, self.scrollContent
                )
                card.texture_file_requested.connect(
                    self.texture_file_requested.emit
                )
                self.text_fill_cards.append(card)
            elif effect_type == 'image':
                card = ImageEffectCard(index, self.scrollContent)
                card.image_requested.connect(self._choose_image_file)
                card.model_selector.popup_active_changed.connect(
                    self.color_dialog_active_changed.emit
                )
                card.natural_height_changed.connect(
                    self._on_image_card_natural_height_changed
                )
                card.generate_requested.connect(
                    self.image_generation_requested.emit
                )
                card.stop_requested.connect(
                    self.image_generation_stop_requested.emit
                )
                self.image_cards.append(card)
            elif effect_type == 'filter':
                filter_effect = (
                    None
                    if seed is None
                    else seed.effects[index]
                )
                if not isinstance(filter_effect, FilterEffect):
                    continue
                spec = get_filter_registry().get_spec(
                    filter_effect.filter_id
                )
                card = FilterEffectCard(
                    index, filter_effect.filter_id, spec, self.scrollContent
                )
                self.filter_cards.append(card)
            else:
                continue
            card.value_commit_requested.connect(
                self.value_commit_requested.emit
            )
            if isinstance(
                card,
                (
                    StrokeEffectCard,
                    ShadowEffectCard,
                    GlowEffectCard,
                    TextFillEffectCard,
                    FilterEffectCard,
                ),
            ):
                card.value_preview_requested.connect(
                    self.value_preview_requested.emit
                )
                card.parameter_preview_requested.connect(
                    self.parameter_preview_requested.emit
                )
                card.parameter_commit_requested.connect(
                    self.parameter_commit_requested.emit
                )
                card.preview_canceled.connect(self.preview_canceled.emit)
            if isinstance(
                card,
                (
                    StrokeEffectCard,
                    ShadowEffectCard,
                    GlowEffectCard,
                    TextFillEffectCard,
                ),
            ):
                card.color_dialog_active_changed.connect(
                    self.color_dialog_active_changed.emit
                )
            if isinstance(
                card,
                (
                    StrokeEffectCard,
                    ShadowEffectCard,
                    GlowEffectCard,
                    TextFillEffectCard,
                    ImageEffectCard,
                    FilterEffectCard,
                ),
            ):
                card.move_requested.connect(self._move_visual_effect)
            card.remove_requested.connect(self.remove_effect_requested.emit)
            (
                self.base_card_layout
                if isinstance(card, TextFillEffectCard)
                else self.cards_layout
            ).addWidget(card)
            card.show()
            self.effect_cards.append(card)

    def _move_visual_effect(self, index: int, direction: int) -> None:
        self.move_effect_requested.emit(index, -direction)

    @staticmethod
    def _effect_sequence(stack: TextEffectStack) -> Tuple[object, ...]:
        return tuple(
            effect_structure_key(effect) for effect in stack.effects
        )

    def _set_effect_states(
        self, states: Sequence[TextEffectStack]
    ) -> None:
        states = tuple(states)
        if not states or any(
            not isinstance(state, TextEffectStack) for state in states
        ):
            raise TypeError('effect panel requires TextEffectStack values')

        opacity_values = [state.overall_opacity for state in states]
        common_opacity = (
            opacity_values[0]
            if all(value == opacity_values[0] for value in opacity_values)
            else None
        )
        self.overall_opacity_control.set_model_value(
            common_opacity, opacity_values
        )

        hollow_values = [
            next(
                (
                    effect.enabled
                    for effect in state.effects
                    if isinstance(effect, HollowEffect)
                ),
                False,
            )
            for state in states
        ]
        common_hollow = (
            hollow_values[0]
            if all(value == hollow_values[0] for value in hollow_values)
            else None
        )
        self._set_hollow_toggle_state(common_hollow)

        reference = states[0]
        self.add_effect_button.setEnabled(True)
        self._rebuild_effect_cards(self._effect_sequence(reference), reference)
        matched = matched_effect_occurrences(states)
        gradient_visibility_changed = False
        movable_types = (
            StrokeEffect, ShadowEffect, GlowEffect,
            ImageEffect, FilterEffect,
        )
        movable_indices = [
            index
            for index, effect in enumerate(reference.effects)
            if isinstance(effect, movable_types)
        ]
        fill_indices = [
            index
            for index, effect in enumerate(reference.effects)
            if isinstance(effect, TextFillEffect)
        ]
        movable_aligned = (
            effect_reorder_is_aligned(states, movable_indices[0])
            if movable_indices else False
        )
        fill_aligned = (
            effect_reorder_is_aligned(states, fill_indices[0])
            if fill_indices else False
        )
        for card in self.effect_cards:
            # Cards always expose the primary item's exact values. Matching is
            # derived only for fan-out and never creates a synthetic stack.
            values = [reference.effects[card.index]]
            card_matched = len(states) > 1 and card.index in matched
            card.set_matched(card_matched)
            gradient_editor = getattr(card, 'gradient_editor', None)
            gradient_was_hidden = (
                gradient_editor.isHidden()
                if isinstance(
                    card,
                    (StrokeEffectCard, ShadowEffectCard, GlowEffectCard),
                )
                else None
            )
            if isinstance(card, TextFillEffectCard):
                card.set_values(
                    values,
                    texture_available=self._texture_available(values),
                )
            elif isinstance(card, ImageEffectCard):
                card.set_values(
                    values,
                    available=self._image_available(values),
                    generation_eligible=len(self._block_items) == 1,
                    generation_eligibility_hint=self.tr(
                        'Select exactly one text item to generate an Image.'
                    ),
                )
                self._apply_image_generation_state(card)
            elif isinstance(card, FilterEffectCard):
                card.set_values(values)
            else:
                card.set_values(values)
            if (
                gradient_was_hidden is not None
                and gradient_editor.isHidden() != gradient_was_hidden
            ):
                gradient_visibility_changed = True

            if isinstance(card, (
                StrokeEffectCard,
                ShadowEffectCard,
                GlowEffectCard,
                ImageEffectCard,
                FilterEffectCard,
            )):
                position = movable_indices.index(card.index)
                reorder_enabled = not card_matched or movable_aligned
                card.set_move_enabled(
                    reorder_enabled and position + 1 < len(movable_indices),
                    reorder_enabled and position > 0,
                )
            elif isinstance(card, TextFillEffectCard):
                position = fill_indices.index(card.index)
                reorder_enabled = not card_matched or fill_aligned
                card.set_move_enabled(
                    reorder_enabled and position + 1 < len(fill_indices),
                    reorder_enabled and position > 0,
                )
        if gradient_visibility_changed:
            self.cards_layout.invalidate()
            self.content_layout.invalidate()
        self.add_effect_actions['gradient'].setEnabled(True)
        self.add_effect_actions['texture'].setEnabled(
            bool(self._block_items)
            and getattr(SW.canvas, 'imgtrans_proj', None) is not None
        )
        self.add_effect_actions['image'].setEnabled(
            len(self._block_items) == 1
            and getattr(SW.canvas, 'imgtrans_proj', None) is not None
        )
        self.filter_add_menu.setEnabled(True)
        self._sync_content_height()

    def set_active_format(self, font_format: FontFormat) -> None:
        self._block_items = ()
        portable_effects = without_project_raster_effects(
            font_format.text_effects
        )
        self._set_effect_states([portable_effects])
        self.refresh_alpha_mask_state()

    def set_effect_items(self, items: Sequence["TextBlkItem"]) -> None:
        targets_changed = len(items) != len(self._block_items) or any(
            current is not replacement
            for current, replacement in zip(self._block_items, items)
        )
        if targets_changed:
            for card in self.image_cards:
                card.reset_generation_draft()
        self._block_items = tuple(items)
        faster_preview = self.faster_preview_toggle.isChecked()
        for item in self._block_items:
            item.effect_renderer.set_faster_preview(faster_preview)
        self._set_effect_states(
            [item.blk.fontformat.text_effects for item in items]
        )
        self.refresh_alpha_mask_state()

    def set_image_generation_state(self, index: int, state: str) -> None:
        self._image_generation_index = int(index) if state != 'idle' else -1
        self._image_generation_state = state
        for card in self.image_cards:
            self._apply_image_generation_state(card)

    def detach_image_generation_card(self) -> None:
        """Keep the panel busy after the request's stack target changed."""
        if self._image_generation_state == 'idle':
            return
        self._image_generation_index = -1
        self._image_generation_state = 'stopping'
        for card in self.image_cards:
            self._apply_image_generation_state(card)

    def _apply_image_generation_state(self, card: ImageEffectCard) -> None:
        busy = self._image_generation_state != 'idle'
        owns_request = busy and card.index == self._image_generation_index
        card.set_generation_state(
            self._image_generation_state if owns_request else 'idle',
            panel_busy=busy and not owns_request,
        )

    def show_image_generation_error(
        self, index: int, error: Exception
    ) -> None:
        if not any(card.index == index for card in self.image_cards):
            return
        from ballontranslator.modules.exceptions import (
            LLMApiKeyRequiredError,
            LLMBaseURLRequiredError,
            LLMModelRequiredError,
        )
        from ballontranslator.utils import shared
        from ballontranslator.utils.message import create_error_dialog

        if isinstance(error, LLMApiKeyRequiredError):
            shared.show_llm_key_dialog_in_mainthread(
                error.profile_id, error.profile_name
            )
        elif isinstance(error, LLMModelRequiredError):
            shared.show_llm_model_dialog_in_mainthread(
                error.profile_id, error.profile_name, error.target
            )
        elif isinstance(error, LLMBaseURLRequiredError):
            shared.show_llm_base_url_dialog_in_mainthread(
                error.profile_id, error.profile_name, error.target
            )
        else:
            create_error_dialog(
                error,
                self.tr('Image Generation Failed.'),
                'ImageGenerationFailed',
            )

    def show_image_generation_context_error(
        self, index: int, message: str
    ) -> None:
        if not any(card.index == index for card in self.image_cards):
            return
        QMessageBox.warning(
            self.window(),
            self.tr('Unable to Generate Image'),
            message,
        )

    def project_assets_changed(self) -> None:
        """Refresh imported rasters without changing immutable model values."""
        for item in self._block_items:
            item.effect_renderer.project_assets_changed()
        if self._block_items:
            self._set_effect_states([
                item.blk.fontformat.text_effects
                for item in self._block_items
            ])

    def set_alpha_mask_items(self, items: Sequence["TextBlkItem"]) -> None:
        """Refresh only the TextBlock-owned mask target boundary."""
        self._block_items = tuple(items)
        self.refresh_alpha_mask_state()

    def _texture_available(
        self, fills: Sequence[TextFillEffect]
    ) -> Optional[bool]:
        paints = [fill.paint for fill in fills]
        if (
            not paints
            or not all(isinstance(paint, TexturePaint) for paint in paints)
        ):
            return None
        return self._common_project_asset_available(
            [paint.asset for paint in paints]
        )

    def _image_available(
        self, effects: Sequence[ImageEffect]
    ) -> Optional[bool]:
        return self._common_project_asset_available(
            [effect.asset for effect in effects]
        )

    def _common_project_asset_available(
        self, assets: Sequence[Optional[RasterAssetRef]]
    ) -> Optional[bool]:
        """Resolve one common card asset across current project targets."""
        if (
            not assets
            or any(asset != assets[0] for asset in assets[1:])
            or assets[0] is None
            or not self._block_items
        ):
            return None
        for item in self._block_items:
            scene = item.scene()
            project = (
                None if scene is None else getattr(scene, 'imgtrans_proj', None)
            )
            if (
                project is None
                or project.resolve_raster_asset(assets[0]) is None
            ):
                return False
        return True

    def show_texture_import_error(self, index: int, message: str) -> None:
        if not any(card.index == index for card in self.text_fill_cards):
            return
        QMessageBox.warning(
            self.window(),
            self.tr('Unable to Import Image'),
            self.tr(
                'The selected image could not be added to this project.'
                '\n\n{message}'
            ).format(message=message),
        )

    def iter_controls(self) -> Iterator[EffectNumericControl]:
        yield self.overall_opacity_control
        for card in self.effect_cards:
            yield from card.iter_controls()

    def iter_gradient_editors(self) -> Iterator[InlineLinearGradientEditor]:
        for card in self.effect_cards:
            editor = getattr(card, 'gradient_editor', None)
            if isinstance(editor, InlineLinearGradientEditor):
                yield editor

    def finish_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.commit_pending()
        for editor in tuple(self.iter_gradient_editors()):
            editor.commit_pending()

    def cancel_pending_effect_edits(self) -> None:
        for control in self.iter_controls():
            control.cancel_pending()
        for editor in tuple(self.iter_gradient_editors()):
            editor.cancel_pending()

    def cancel_effect_previews(self) -> None:
        for control in self.iter_controls():
            control.cancel_preview()
        for editor in tuple(self.iter_gradient_editors()):
            editor.cancel_pending()

    def _sync_content_height(self) -> None:
        if not hasattr(self, 'content_layout'):
            return
        self._sync_scroll_content_height(self.content_layout)

    def _on_image_card_natural_height_changed(self) -> None:
        self.cards_layout.invalidate()
        QTimer.singleShot(0, self._sync_content_height)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if not hasattr(self, 'content_layout'):
            return hint
        hint.setHeight(min(
            self.content_layout.sizeHint().height(),
            self.MAX_CONTENT_HEIGHT,
        ))
        return hint

    def _on_overall_commit(self, name: str, value) -> None:
        self.value_commit_requested.emit(-1, name, value)

    def _on_overall_value_preview(self, name: str, value) -> None:
        self.value_preview_requested.emit(-1, name, value)

    def _on_overall_parameter_preview(self, name: str, delta) -> None:
        self.parameter_preview_requested.emit(-1, name, delta)

    def _on_overall_parameter_commit(self, name: str, delta) -> None:
        self.parameter_commit_requested.emit(-1, name, delta)

    def _on_overall_preview_canceled(self, name: str) -> None:
        self.preview_canceled.emit(-1, name)

    def _on_add_effect_triggered(self, _checked: bool = False) -> None:
        action = self.sender()
        if action is not None and action.data() in {
            'stroke', 'shadow', 'glow', 'gradient',
            'texture', 'image',
        }:
            self.add_effect_requested.emit(action.data())

    def _on_add_filter_triggered(self, _checked: bool = False) -> None:
        action = self.sender()
        filter_id = None if action is None else action.data()
        if isinstance(filter_id, str):
            self.add_filter_requested.emit(filter_id)

    def _set_hollow_toggle_state(
        self, enabled: Optional[bool]
    ) -> None:
        mixed = enabled is None
        blocker = QSignalBlocker(self.hollow_toggle_button)
        self.hollow_toggle_button.setChecked(enabled is True)
        del blocker
        if self.hollow_toggle_button.property('mixed') != mixed:
            self.hollow_toggle_button.setProperty('mixed', mixed)
            style = self.hollow_toggle_button.style()
            style.unpolish(self.hollow_toggle_button)
            style.polish(self.hollow_toggle_button)
        if mixed:
            description = self.tr('Enable Hollow for All Selected Text')
        elif enabled is True:
            description = self.tr('Disable Hollow')
        else:
            description = self.tr('Enable Hollow')
        self.hollow_toggle_button.setToolTip(description)
        self.hollow_toggle_button.setAccessibleName(description)

    def _on_hollow_toggled(self, enabled: bool) -> None:
        self.hollow_enabled_requested.emit(enabled)

    def _on_faster_preview_toggled(self, enabled: bool) -> None:
        for item in self._block_items:
            item.effect_renderer.set_faster_preview(enabled)

    def _on_mask_brush_clicked(self, checked: bool) -> None:
        self.mask_edit_requested.emit(checked)
