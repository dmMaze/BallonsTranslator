"""Reusable cards and controls for item-wide text effects."""

from dataclasses import replace
from typing import Dict, Optional, Sequence, Tuple

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
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import profile_by_id
from ballontranslator.utils.text_alpha_mask import (
    ALPHA_BRUSH_MAX_DIAMETER,
    ALPHA_BRUSH_MIN_DIAMETER,
    TextAlphaMask,
)
from ballontranslator.utils.text_effects import (
    EffectPaint,
    FilterEffect,
    GeneratedEffectPaint,
    GlowEffect,
    LinearGradientPaint,
    ImageEffect,
    ImageGenerationRecipe,
    SHADOW_BLUR_LIMIT,
    SHADOW_DISTANCE_LIMIT,
    SHADOW_SPREAD_LIMIT,
    ShadowEffect,
    SolidPaint,
    StrokeEffect,
    TextFillEffect,
    TexturePaint,
)

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


class _EffectActionButton(QToolButton):
    """Shared construction for compact effect-card actions."""

    def __init__(
        self,
        icon_name: str,
        hint: str,
        object_name: str,
        direction: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setIcon(QIcon(themed_icon_path(icon_name)))
        icon_extent = 12 if direction == 0 else 16
        self.setIconSize(QSize(icon_extent, icon_extent))
        self.setToolTip(hint)
        self.setAccessibleName(hint)
        self.setProperty('move-direction', direction)
        self.setFixedSize(18, 18)


class EffectDeleteButton(_EffectActionButton):
    """Delete an effect card."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            'titlebar_close.svg',
            QCoreApplication.translate('EffectDeleteButton', 'Delete'),
            'TextEffectCloseButton',
            0,
            parent,
        )


class EffectMoveUpButton(_EffectActionButton):
    """Move an effect toward the start of its stack."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            'chevron-up.svg',
            QCoreApplication.translate('EffectMoveUpButton', 'Move Up'),
            'TextEffectMoveButton',
            -1,
            parent,
        )


class EffectMoveDownButton(_EffectActionButton):
    """Move an effect toward the end of its stack."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            'chevron-down.svg',
            QCoreApplication.translate(
                'EffectMoveDownButton', 'Move Down'
            ),
            'TextEffectMoveButton',
            1,
            parent,
        )


class EffectVisibilityButton(QToolButton):
    """Compact enabled or disabled visibility control.

    >>> EffectVisibilityButton.__name__
    'EffectVisibilityButton'
    """

    visibility_requested = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._visibility = True
        self.setObjectName('TextEffectVisibilityButton')
        self.setFixedSize(18, 18)
        self.setIconSize(QSize(16, 16))
        self.clicked.connect(self._on_clicked)
        self.set_visibility(True)

    def set_visibility(self, visible: bool) -> None:
        self._visibility = bool(visible)
        if self._visibility:
            icon_name = 'text-effect-visibility-open.svg'
            hint = self.tr('Hide')
        else:
            icon_name = 'text-effect-visibility-closed.svg'
            hint = self.tr('Show')
        self.setIcon(QIcon(themed_icon_path(icon_name)))
        self.setToolTip(hint)
        self.setAccessibleName(hint)

    def _on_clicked(self) -> None:
        self.visibility_requested.emit(not self._visibility)


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
        layout.addWidget(button)
    widget.setFixedWidth(18 * len(buttons) + 4 * max(0, len(buttons) - 1))
    parent.set_hover_actions(buttons)
    return widget


def _set_effect_header_selector_width(
    selector: BottomBorderComboBox,
) -> None:
    """Give every effect header selector Shadow's natural content width."""
    selector.setWidthSampleText(QCoreApplication.translate(
        'TextEffectPanel', 'Long / Extrude'
    ))


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
        self._current_mode = 'normal'
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

    def current_mode(self) -> str:
        return self._current_mode

    def set_mode(self, mode: str) -> None:
        action = self._actions_by_mode.get(mode)
        if action is None:
            raise ValueError('unsupported blend mode')
        self._current_mode = mode
        for candidate in self._action_group.actions():
            candidate.setChecked(candidate is action)
        label = action.text()
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


def _set_blend_value(
    selector: BlendModeSelector,
    effect: object,
) -> None:
    selector.set_mode(getattr(effect, 'blend_mode'))


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
        self.setObjectName('TextEffectPaintButton')
        self.setMinimumHeight(24)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

    def set_paint(
        self,
        paint: GeneratedEffectPaint,
        description: Optional[str] = None,
    ) -> None:
        self._paint = paint
        self.setIcon(QIcon())
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
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._paint is None:
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

        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)

        self.visibility_button = EffectVisibilityButton(self)
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
        self.position_selector.setAccessibleName(self.tr('Stroke Position'))
        for label, value in (
            (self.tr('Inside'), 'inside'),
            (self.tr('Center'), 'center'),
            (self.tr('Outside'), 'outside'),
        ):
            self.position_selector.addItem(label, value)
        _set_effect_header_selector_width(self.position_selector)
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

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_value(self, stroke: StrokeEffect) -> None:
        self.visibility_button.set_visibility(stroke.enabled)
        _set_blend_value(self.blend_selector, stroke)

        with QSignalBlocker(self.position_selector):
            self.position_selector.setCurrentIndex(
                self.position_selector.findData(stroke.position)
            )

        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                self.fill_type_selector.findData(stroke.paint.paint_type)
            )

        for name, control in (
            ('width', self.width_control),
            ('opacity', self.opacity_control),
        ):
            value = getattr(stroke, name)
            control.set_model_value(value)

        self._paint_seed = stroke.paint
        self.paint_button.set_paint(self._paint_seed)
        show_gradient = stroke.paint.paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(self._paint_seed)
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
        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)

        self.visibility_button = EffectVisibilityButton(self)
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
        self.type_selector.setAccessibleName(self.tr('Shadow Type'))
        for label, value in (
            (self.tr('Drop'), 'drop'),
            (self.tr('Inner'), 'inner'),
            (self.tr('Long / Extrude'), 'long'),
        ):
            self.type_selector.addItem(label, value)
        _set_effect_header_selector_width(self.type_selector)
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

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_value(self, shadow: ShadowEffect) -> None:
        self.visibility_button.set_visibility(shadow.enabled)
        _set_blend_value(self.blend_selector, shadow)

        with QSignalBlocker(self.type_selector):
            self.type_selector.setCurrentIndex(
                self.type_selector.findData(shadow.shadow_type)
            )
        show_soft_controls = shadow.shadow_type != 'long'
        self.blur_control.setVisible(show_soft_controls)
        self.spread_control.setVisible(show_soft_controls)
        if shadow.shadow_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('angle', self.angle_control),
            ('distance', self.distance_control),
            ('blur', self.blur_control),
            ('spread', self.spread_control),
        ):
            control.set_model_value(getattr(shadow, name))
        self.angle_dial.end_interaction()
        self.angle_dial.set_angle(shadow.angle)

        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                self.fill_type_selector.findData(shadow.paint.paint_type)
            )
        self._paint_seed = shadow.paint
        description = (
            self.tr('Edit Shadow Gradient')
            if isinstance(shadow.paint, LinearGradientPaint)
            else self.tr('Choose Shadow Color')
        )
        self.paint_button.set_paint(
            self._paint_seed,
            description=description,
        )
        show_gradient = shadow.paint.paint_type == 'linear_gradient'
        visibility_changed = self.gradient_editor.isHidden() == show_gradient
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(self._paint_seed)
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
        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)
        self.visibility_button = EffectVisibilityButton(self)
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
        self.type_selector.setAccessibleName(self.tr('Glow Type'))
        self.type_selector.addItem(self.tr('Outer'), 'outer')
        self.type_selector.addItem(self.tr('Inner'), 'inner')
        _set_effect_header_selector_width(self.type_selector)
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

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_value(self, glow: GlowEffect) -> None:
        self.visibility_button.set_visibility(glow.enabled)
        _set_blend_value(self.blend_selector, glow)

        with QSignalBlocker(self.type_selector):
            self.type_selector.setCurrentIndex(
                self.type_selector.findData(glow.glow_type)
            )
        if glow.glow_type == 'inner':
            self.spread_control.label.setText(self.tr('Choke'))
        else:
            self.spread_control.label.setText(self.tr('Spread'))

        for name, control in (
            ('opacity', self.opacity_control),
            ('size', self.size_control),
            ('spread', self.spread_control),
        ):
            control.set_model_value(getattr(glow, name))

        with QSignalBlocker(self.fill_type_selector):
            self.fill_type_selector.setCurrentIndex(
                self.fill_type_selector.findData(glow.paint.paint_type)
            )
        self._paint_seed = glow.paint
        description = (
            self.tr('Edit Glow Gradient')
            if isinstance(glow.paint, LinearGradientPaint)
            else self.tr('Choose Glow Color')
        )
        self.paint_button.set_paint(
            self._paint_seed,
            description=description,
        )
        show_gradient = glow.paint.paint_type == 'linear_gradient'
        visibility_changed = (
            self.gradient_editor.isHidden() == show_gradient
        )
        self.paint_button.setVisible(not show_gradient)
        self.gradient_editor.setVisible(show_gradient)
        if show_gradient and isinstance(self._paint_seed, LinearGradientPaint):
            self.gradient_editor.set_paint(self._paint_seed)
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
        self.visibility_button = EffectVisibilityButton(self)
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)

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

    def set_value(
        self,
        fill: TextFillEffect,
        texture_available: Optional[bool] = None,
    ) -> None:
        """Project one foreground layer into this fixed card."""
        if fill.paint.paint_type != self.paint_type:
            raise ValueError('foreground card values must match its paint type')
        self.visibility_button.set_visibility(fill.enabled)
        _set_blend_value(self.blend_selector, fill)
        self.opacity_control.set_model_value(fill.opacity)
        self._paint_seed = fill.paint
        if self.paint_type == 'linear_gradient':
            assert self.gradient_editor is not None
            assert isinstance(self._paint_seed, LinearGradientPaint)
            self.gradient_editor.set_paint(self._paint_seed)
        else:
            assert self.texture_field is not None
            assert self.texture_button is not None
            assert self.texture_mapping_selector is not None
            assert self.texture_scale_control is not None
            assert isinstance(fill.paint, TexturePaint)
            asset = fill.paint.asset
            if asset is None:
                display_name = ''
                hint = self.tr('Choose an image for this Texture')
                accessible_name = self.tr('No Texture Image Selected')
            else:
                name = (
                    asset.display_name
                    or asset.path.rsplit('/', 1)[-1]
                )
                display_name = (
                    self.tr('Missing: {name}').format(name=name)
                    if texture_available is False else name
                )
                hint = name + '\n' + asset.path
                accessible_name = display_name
            self.texture_field.setText(display_name)
            self.texture_field.setCursorPosition(0)
            self.texture_field.setToolTip(hint)
            self.texture_field.setAccessibleName(accessible_name)
            self.texture_button.setToolTip(hint)
            with QSignalBlocker(self.texture_mapping_selector):
                self.texture_mapping_selector.setCurrentIndex(
                    self.texture_mapping_selector.findData(fill.paint.mapping)
                )
            self.texture_scale_control.set_model_value(fill.paint.scale)
            self.texture_scale_control.setVisible(
                fill.paint.mapping == 'tile'
            )
        self.layout().invalidate()
        self.updateGeometry()

    def iter_controls(self) -> Tuple[EffectNumericControl, ...]:
        return (
            (self.opacity_control,)
            if self.texture_scale_control is None
            else (self.opacity_control, self.texture_scale_control)
        )

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
        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)
        self.visibility_button = EffectVisibilityButton(self)
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

    def set_move_enabled(self, up: bool, down: bool) -> None:
        self.move_up_button.setEnabled(up)
        self.move_down_button.setEnabled(down)

    def set_value(self, effect: FilterEffect) -> None:
        self.visibility_button.set_visibility(effect.enabled)
        if self.spec is None:
            return
        failure = get_filter_registry().get_runtime_failure(self.filter_id)
        if failure is not None:
            self._set_parameter_controls_enabled(False)
            self.setToolTip(str(failure))
            return
        try:
            if effect.schema_version == self.spec.schema_version:
                active_params = self.spec.normalize_params(
                    effect.params_dict()
                )
            elif (
                effect.enabled
                and effect.schema_version < self.spec.schema_version
            ):
                active_params = dict(
                    get_filter_registry().resolve(effect).params
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
            value = active_params[parameter.key]
            control = self.numeric_controls.get(parameter.key)
            if control is not None:
                control.set_model_value(value)
                continue
            selector = self.choice_selectors[parameter.key]
            with QSignalBlocker(selector):
                selector.setCurrentIndex(selector.findData(value))

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
        self._model_effect: Optional[ImageEffect] = None
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
        self.setToolTip(self._editing_hint)
        self.title_label.setToolTip(self._editing_hint)
        self.visibility_button = EffectVisibilityButton(self)
        self.visibility_button.visibility_requested.connect(
            self._on_enabled_clicked
        )
        self.move_up_button = EffectMoveUpButton(self)
        self.move_down_button = EffectMoveDownButton(self)
        self.delete_button = EffectDeleteButton(self)
        for button in (
            self.move_up_button,
            self.move_down_button,
            self.delete_button,
        ):
            button.clicked.connect(self._on_action_clicked)
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

    def set_value(
        self,
        effect: ImageEffect,
        available: Optional[bool],
        *,
        generation_eligible: bool = False,
        generation_eligibility_hint: str = '',
    ) -> None:
        self._model_effect = effect
        self.visibility_button.set_visibility(effect.enabled)
        if effect.asset is None:
            display_name = ''
            hint = self._choose_hint + '\n' + self._editing_hint
            accessible_name = self.tr('No Image Selected')
        else:
            name = (
                effect.asset.display_name
                or effect.asset.path.rsplit('/', 1)[-1]
            )
            display_name = (
                name if available is not False
                else self.tr('Missing: {name}').format(name=name)
            )
            hint = (
                name + '\n' + effect.asset.path + '\n' + self._editing_hint
            )
            accessible_name = display_name
        self.image_field.setText(display_name)
        self.image_field.setCursorPosition(0)
        self.image_field.setToolTip(hint)
        self.image_field.setAccessibleName(accessible_name)
        self.image_button.setToolTip(hint)
        with QSignalBlocker(self.mode_selector):
            self.mode_selector.setCurrentIndex(
                self.mode_selector.findData(effect.mode)
            )
        self._sync_placement_hint(self.mode_selector.currentIndex())
        self._generation_eligible = bool(generation_eligible)
        self._generation_eligibility_hint = generation_eligibility_hint
        if (
            self._generation_draft_dirty
            and effect.generation == self._generation_draft
        ):
            # A successful generation committed this exact draft. Future
            # history changes should once again project the persisted recipe.
            self._generation_draft_dirty = False
        if not self._generation_draft_dirty:
            self._project_generation_recipe(effect.generation)
        self._sync_generation_controls()

    def _dirty_generation_draft(
        self,
    ) -> Optional[Tuple[ImageEffect, ImageGenerationRecipe]]:
        if not self._generation_draft_dirty or self._model_effect is None:
            return None
        return self._model_effect, self._generation_draft

    def _restore_generation_draft(
        self,
        effect: ImageEffect,
        recipe: ImageGenerationRecipe,
    ) -> None:
        """Restore a draft only for the same surviving immutable effect."""
        self._model_effect = effect
        self._project_generation_recipe(recipe)
        self._generation_draft_dirty = True

    def _project_generation_recipe(
        self, recipe: ImageGenerationRecipe
    ) -> None:
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

    def reset_generation_draft(self) -> None:
        """Project the next selected item's persisted recipe."""
        self._generation_draft_dirty = False
        self._model_effect = None

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
        self.visibility_button = EffectVisibilityButton(self)
        self.visibility_button.visibility_requested.connect(
            self.enabled_requested.emit
        )
        self.remove_button = EffectDeleteButton(self)
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
        self.diameter_editor.setRange(
            ALPHA_BRUSH_MIN_DIAMETER,
            ALPHA_BRUSH_MAX_DIAMETER,
        )
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
