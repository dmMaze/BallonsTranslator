"""Selection and stack orchestration for item-wide text effects."""

from typing import Iterator, Optional, Sequence, Tuple, TYPE_CHECKING

from qtpy.QtCore import QSignalBlocker, QTimer, Signal, QSize, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)

from ballontranslator.utils.fontformat import FontFormat
from ballontranslator.utils.raster_assets import RasterAssetRef
from ballontranslator.utils.text_effects import (
    FilterEffect,
    GlowEffect,
    HollowEffect,
    ImageEffect,
    ShadowEffect,
    StrokeEffect,
    TextEffectStack,
    TextFillEffect,
    TexturePaint,
    effect_structure_key,
    without_project_raster_effects,
)

from ... import shared_widget as SW
from ...custom_widget import PanelArea
from ...misc import themed_icon_path
from .cards import (
    AlphaMaskCard,
    EffectNumericControl,
    FilterEffectCard,
    GlowEffectCard,
    ImageEffectCard,
    ShadowEffectCard,
    StrokeEffectCard,
    TextFillEffectCard,
    _choose_project_raster,
    _filter_ui_text,
)
from .edit_session import (
    effect_reorder_is_aligned,
    matched_effect_occurrences,
)
from .filters import get_filter_registry
from .gradient_editor import InlineLinearGradientEditor

if TYPE_CHECKING:
    from .alpha_mask_edit_session import TextAlphaMaskEditSession
    from ..item import TextBlkItem


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
        self._effect_types = None
        self._image_generation_index = -1
        self._image_generation_state = 'idle'
        self._pending_visible_effect_index: Optional[int] = None
        self._reveal_effect_timer = QTimer(self)
        self._reveal_effect_timer.setSingleShot(True)
        self._reveal_effect_timer.timeout.connect(
            self._reveal_pending_effect_card
        )
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

    def _image_cards(self) -> Iterator[ImageEffectCard]:
        return (
            card for card in self.effect_cards
            if isinstance(card, ImageEffectCard)
        )

    def _text_fill_cards(self) -> Iterator[TextFillEffectCard]:
        return (
            card for card in self.effect_cards
            if isinstance(card, TextFillEffectCard)
        )

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
        if not any(card.index == index for card in self._image_cards()):
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

    def _rebuild_effect_cards(
        self,
        effect_keys: Sequence[object],
        seed: Optional[TextEffectStack] = None,
    ) -> None:
        effect_keys = tuple(effect_keys)
        if effect_keys == self._effect_types:
            return
        dirty_image_drafts = tuple(
            draft
            for card in self._image_cards()
            if (draft := card._dirty_generation_draft()) is not None
        )
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
            elif effect_type == 'shadow':
                card = ShadowEffectCard(index, self.scrollContent)
            elif effect_type == 'glow':
                card = GlowEffectCard(index, self.scrollContent)
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
                image_effect = (
                    None if seed is None else seed.effects[index]
                )
                if isinstance(image_effect, ImageEffect):
                    for previous_effect, draft in dirty_image_drafts:
                        if previous_effect is image_effect:
                            card._restore_generation_draft(image_effect, draft)
                            break
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
            value = reference.effects[card.index]
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
                assert isinstance(value, TextFillEffect)
                card.set_value(
                    value,
                    texture_available=self._texture_available(value),
                )
            elif isinstance(card, ImageEffectCard):
                assert isinstance(value, ImageEffect)
                card.set_value(
                    value,
                    available=self._project_asset_available(value.asset),
                    generation_eligible=len(self._block_items) == 1,
                    generation_eligibility_hint=self.tr(
                        'Select exactly one text item to generate an Image.'
                    ),
                )
                self._apply_image_generation_state(card)
            elif isinstance(card, FilterEffectCard):
                assert isinstance(value, FilterEffect)
                card.set_value(value)
            else:
                card.set_value(value)
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
            for card in self._image_cards():
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
        for card in self._image_cards():
            self._apply_image_generation_state(card)

    def detach_image_generation_card(self) -> None:
        """Keep the panel busy after the request's stack target changed."""
        if self._image_generation_state == 'idle':
            return
        self._image_generation_index = -1
        self._image_generation_state = 'stopping'
        for card in self._image_cards():
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
        if not any(card.index == index for card in self._image_cards()):
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
        if not any(card.index == index for card in self._image_cards()):
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

    def _texture_available(
        self, fill: TextFillEffect
    ) -> Optional[bool]:
        if not isinstance(fill.paint, TexturePaint):
            return None
        return self._project_asset_available(fill.paint.asset)

    def _project_asset_available(
        self, asset: Optional[RasterAssetRef]
    ) -> Optional[bool]:
        """Resolve the primary card asset across current project targets."""
        if asset is None or not self._block_items:
            return None
        for item in self._block_items:
            scene = item.scene()
            project = (
                None if scene is None else getattr(scene, 'imgtrans_proj', None)
            )
            if (
                project is None
                or project.resolve_raster_asset(asset) is None
            ):
                return False
        return True

    def show_texture_import_error(self, index: int, message: str) -> None:
        if not any(card.index == index for card in self._text_fill_cards()):
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

    def reveal_effect_card(self, index: int) -> None:
        """Scroll a newly inserted effect card into the viewport."""
        self._pending_visible_effect_index = int(index)
        self._reveal_effect_timer.start(0)

    def _reveal_pending_effect_card(self) -> None:
        index = self._pending_visible_effect_index
        self._pending_visible_effect_index = None
        if index is None:
            return
        self._sync_content_height()
        card = next(
            (card for card in self.effect_cards if card.index == index),
            None,
        )
        if card is not None:
            self.ensureWidgetVisible(card, 0, self.cards_layout.spacing())

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
