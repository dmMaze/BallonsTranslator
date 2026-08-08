"""Expandable controls for composable text transforms."""

from typing import Sequence

from qtpy.QtCore import QCoreApplication, QEvent, QSize, QTimer, Signal, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ballontranslator.utils.fontformat import FontFormat, TextTransformStack

from ...custom_widget import PanelArea
from ...misc import themed_icon_path
from .controls import (
    CommittedTransformControl,
    TransformParameterPanel,
)
from .registry import (
    GLYPH_SLANT_CONTROL,
    TEXT_TRANSFORM_VARIANTS,
)


class TextTransformPanel(PanelArea):
    """Own the transform settings shown under one expandable title.

    >>> TextTransformPanel.__name__
    'TextTransformPanel'
    """

    transform_commit_requested = Signal(int, str, object)
    transform_preview_requested = Signal(int, str, object)
    transform_drag_commit_requested = Signal(int, str, object)
    transform_preview_canceled = Signal(int, str)
    transform_add_requested = Signal(str)
    transform_remove_requested = Signal(int)
    transform_move_requested = Signal(int, int)
    transform_selected = Signal(int)

    MAX_CONTENT_HEIGHT = 480

    def __init__(
        self,
        panel_name: str,
        config_name: str,
        config_expand_name: str,
    ):
        super().__init__(panel_name, config_name, config_expand_name)
        self._base_width_hint = 1
        self._syncing_geometry = False
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.scrollContent.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.scrollContent.after_resized.connect(self._sync_content_height)
        self.setMaximumHeight(self.MAX_CONTENT_HEIGHT)

        self.transform_variants = TEXT_TRANSFORM_VARIANTS
        glyph = GLYPH_SLANT_CONTROL
        self.glyph_slant_control = CommittedTransformControl(
            glyph.label(),
            glyph.attribute_name,
            glyph.factor,
            glyph.minimum,
            glyph.maximum,
            glyph.suffix,
            1.0,
            self.scrollContent,
        )
        self.glyph_slant_control.editor.setProperty(
            'glyphSlantEditor', True
        )
        self.glyph_slant_control.editor.setFixedWidth(84)
        self.glyph_slant_control.label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.glyph_slant_control.layout().setSpacing(8)
        self.glyph_slant_control.layout().setStretch(0, 1)
        self.glyph_slant_control.layout().setStretch(1, 2)
        setattr(self, glyph.name, self.glyph_slant_control)
        self.glyph_slant_control.commit_requested.connect(
            lambda name, value:
            self.transform_commit_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.preview_requested.connect(
            lambda name, value:
            self.transform_preview_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.drag_commit_requested.connect(
            lambda name, value:
            self.transform_drag_commit_requested.emit(-1, name, value)
        )
        self.glyph_slant_control.preview_canceled.connect(
            lambda name: self.transform_preview_canceled.emit(-1, name)
        )

        self.add_transform_button = QToolButton(self.scrollContent)
        self.add_transform_button.setObjectName('AddTextTransformButton')
        self.add_transform_button.setText(self.tr('Add'))
        self.add_transform_button.setToolTip(self.tr('Add Transform'))
        self.add_transform_button.setAccessibleName(self.tr('Add Transform'))
        self.add_transform_button.setFixedSize(72, 26)
        self.add_transform_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.add_transform_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup
        )
        add_menu = QMenu(self.add_transform_button)
        add_menu.setObjectName('TextTransformAddMenu')
        for variant in self.transform_variants:
            action = add_menu.addAction(
                QIcon(themed_icon_path(variant.icon_name)),
                variant.label(),
            )
            action.triggered.connect(
                lambda _checked=False, transform_type=variant.transform_type:
                self.transform_add_requested.emit(transform_type)
            )
        self.add_transform_button.setMenu(add_menu)

        self.transform_mixed_label = QLabel(
            self.tr('Mixed'), self.scrollContent
        )
        self.transform_mixed_label.setObjectName('TextTransformMixedLabel')
        self.transform_mixed_label.setVisible(False)

        self.transform_rows_layout = QVBoxLayout()
        self.transform_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.transform_rows_layout.setSpacing(10)
        self.transform_panels = []
        self._transform_panel_types = ()
        self._selected_transform_index = None

        self.transform_layout = QVBoxLayout()
        self.transform_layout.setContentsMargins(8, 8, 8, 8)
        self.transform_layout.setSpacing(6)
        self.transform_header_layout = QHBoxLayout()
        self.transform_header_layout.setContentsMargins(0, 0, 0, 0)
        self.transform_header_layout.setSpacing(6)
        self.add_transform_layout = QHBoxLayout()
        self.add_transform_layout.setContentsMargins(0, 0, 0, 0)
        self.add_transform_layout.addWidget(
            self.add_transform_button,
            alignment=Qt.AlignmentFlag.AlignVCenter,
        )
        self.add_transform_layout.addStretch()
        self.transform_header_layout.addLayout(self.add_transform_layout, 1)
        self.transform_header_layout.addWidget(self.glyph_slant_control, 1)
        self.transform_layout.addLayout(self.transform_header_layout)
        self.transform_layout.addSpacing(6)
        self.transform_layout.addWidget(self.transform_mixed_label)
        self.transform_layout.addLayout(self.transform_rows_layout)
        self.setContentLayout(self.transform_layout)
        self._base_width_hint = super().sizeHint().width()
        self._sync_content_height()
        QTimer.singleShot(0, self._sync_content_height)

    def _sync_content_height(self):
        if self._syncing_geometry:
            return
        self._syncing_geometry = True
        try:
            # The viewport can still report its pre-show size here. Overlay
            # scrollbars consume no layout width, so the frame gives the
            # responsive content width directly.
            content_width = max(
                1, self.width() - 2 * self.frameWidth()
            )
            self.scrollContent.setMinimumWidth(content_width)
            self.scrollContent.setMaximumWidth(content_width)
            self.scrollContent.resize(
                content_width,
                max(1, self.scrollContent.height()),
            )
            self.transform_layout.invalidate()
            content_height = (
                self.transform_layout.heightForWidth(content_width)
                if self.transform_layout.hasHeightForWidth()
                else self.transform_layout.sizeHint().height()
            )
            self.scrollContent.setMinimumHeight(content_height)
            self.scrollContent.resize(
                content_width,
                max(content_height, self.viewport().height()),
            )
            self.transform_layout.activate()
            self.transform_rows_layout.invalidate()
            self.transform_rows_layout.activate()
            target = min(
                content_height + 2 * self.frameWidth(),
                self.MAX_CONTENT_HEIGHT,
            )
            self.setMinimumHeight(target)
            self.scrollContent.updateGeometry()
            self.updateGeometry()
            self.view_widget.updateGeometry()
            # A hidden resizable child does not always update QScrollArea's
            # range after its minimum height changes.
            QCoreApplication.sendEvent(
                self, QEvent(QEvent.Type.LayoutRequest)
            )
        finally:
            self._syncing_geometry = False

    def sizeHint(self):
        hint = super().sizeHint()
        if not hasattr(self, 'transform_layout'):
            return hint
        return QSize(
            self._base_width_hint,
            min(
                (
                    self.transform_layout.heightForWidth(
                        max(1, self.width() - 2 * self.frameWidth())
                    )
                    if self.transform_layout.hasHeightForWidth()
                    else self.transform_layout.sizeHint().height()
                )
                + 2 * self.frameWidth(),
                self.MAX_CONTENT_HEIGHT,
            ),
        )

    def _clear_transform_panels(self):
        for panel in self.transform_panels:
            self.transform_rows_layout.removeWidget(panel)
            panel.setParent(None)
            panel.deleteLater()
        self.transform_panels = []
        self._transform_panel_types = ()

    def _rebuild_transform_panels(self, transform_types):
        transform_types = tuple(transform_types)
        if transform_types == self._transform_panel_types:
            return
        self._clear_transform_panels()
        variants = {
            variant.transform_type: variant
            for variant in self.transform_variants
        }
        for index, transform_type in enumerate(transform_types):
            panel = TransformParameterPanel(
                index, variants[transform_type], self.scrollContent
            )
            panel.commit_requested.connect(
                self.transform_commit_requested.emit
            )
            panel.preview_requested.connect(
                self.transform_preview_requested.emit
            )
            panel.drag_commit_requested.connect(
                self.transform_drag_commit_requested.emit
            )
            panel.preview_canceled.connect(
                self.transform_preview_canceled.emit
            )
            panel.remove_requested.connect(
                self.transform_remove_requested.emit
            )
            panel.move_requested.connect(self.transform_move_requested.emit)
            panel.card_clicked.connect(self.toggle_transform)
            panel.selected.connect(self.select_transform)
            self.transform_rows_layout.addWidget(panel)
            self.transform_panels.append(panel)
        self._transform_panel_types = transform_types
        count = len(self.transform_panels)
        for index, panel in enumerate(self.transform_panels):
            panel.set_index(index)
            panel.set_move_enabled(index > 0, index + 1 < count)
            panel.set_selected(index == self._selected_transform_index)
        if (
            self._selected_transform_index is not None
            and self._selected_transform_index >= count
        ):
            self.clear_transform_selection()

    def select_transform(self, index: int, *, emit: bool = True):
        index = int(index)
        if index < 0 or index >= len(self.transform_panels):
            self.clear_transform_selection(emit=emit)
            return
        if self._selected_transform_index == index:
            return
        self._selected_transform_index = index
        for panel_index, panel in enumerate(self.transform_panels):
            panel.set_selected(panel_index == index)
        if emit:
            self.transform_selected.emit(index)

    def toggle_transform(self, index: int):
        if self._selected_transform_index == int(index):
            self.clear_transform_selection()
        else:
            self.select_transform(index)

    def clear_transform_selection(self, *, emit: bool = True):
        if self._selected_transform_index is None:
            return
        self._selected_transform_index = None
        for panel in self.transform_panels:
            panel.set_selected(False)
        if emit:
            self.transform_selected.emit(-1)

    def _set_transform_states(
        self, states: Sequence[TextTransformStack]
    ) -> None:
        if not all(isinstance(state, TextTransformStack) for state in states):
            raise TypeError('transform panel requires TextTransformStack values')
        glyph_values = [state.glyph_slant_angle for state in states]
        common_glyph = (
            glyph_values[0]
            if glyph_values
            and all(value == glyph_values[0] for value in glyph_values)
            else None
        )
        self.glyph_slant_control.set_model_value(common_glyph, glyph_values)

        sequences = [
            tuple(transform.transform_type for transform in state)
            for state in states
        ]
        common_sequence = (
            sequences[0]
            if sequences
            and all(sequence == sequences[0] for sequence in sequences)
            else None
        )
        mixed = common_sequence is None
        self.transform_mixed_label.setVisible(mixed)
        if mixed:
            self.clear_transform_selection()
            self._rebuild_transform_panels(())
        else:
            self._rebuild_transform_panels(common_sequence)
            for index, panel in enumerate(self.transform_panels):
                panel.set_values([state[index] for state in states])
        self._sync_content_height()

    def set_active_format(self, font_format: FontFormat) -> None:
        self._set_transform_states([font_format.text_transform])

    def set_transform_items(self, items) -> None:
        self._set_transform_states(
            [item.blk.fontformat.text_transform for item in items]
        )

    def iter_transform_controls(self):
        yield self.glyph_slant_control
        for panel in self.transform_panels:
            yield from panel.iter_controls()

    def cancel_pending_transform_edits(self):
        for control in self.iter_transform_controls():
            control.cancel_pending()

    def cancel_transform_previews(self):
        for control in self.iter_transform_controls():
            control.cancel_preview()

    def finish_pending_transform_edits(self):
        for control in self.iter_transform_controls():
            control.commit_pending()
