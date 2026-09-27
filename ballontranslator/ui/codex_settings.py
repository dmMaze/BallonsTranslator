"""Dedicated settings for the application's shared Codex account."""

from __future__ import annotations

from qtpy.QtCore import QSignalBlocker, QTimer, Qt, Signal
from qtpy.QtGui import QIcon, QPaintEvent, QPainter
from qtpy.QtWidgets import QAbstractButton, QLabel, QMessageBox, QPushButton, QHBoxLayout, QLayout, QScrollArea, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import LLMProfile, codex_thinking_options, image_model_choices, profile_by_id, split_image_model_selection
from ballontranslator.utils.shared import LLM_PROFILE_EDITOR_WIDTH_SCALE, LLM_PROMPT_EDITOR_WIDTH

from .codex_account import CodexAccountController
from .codex_sign_in import CodexSignInButton
from .custom_widget import ParamComboBox, RefreshButton, ScrollBar
from .misc import themed_icon_path
from .module_parse_widgets import ParamWidget


class _AccountStatusLabel(QLabel):
    """Elide long account names; the tooltip retains their full text."""

    def paintEvent(self, event: QPaintEvent) -> None:
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width())
        painter = QPainter(self)
        self.style().drawItemText(painter, self.contentsRect(), self.alignment(), self.palette(),
                                  self.isEnabled(), text, self.foregroundRole())
        painter.end()


class CodexSettingsPanel(QWidget):
    """Edit the canonical Codex settings using only cached account and model data.

    >>> CodexSettingsPanel.__name__
    'CodexSettingsPanel'
    """

    profile_ui_updated = Signal()
    profile_summary_changed = Signal()

    def __init__(self, scrollWidget: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName('CodexSettingsPanel')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMaximumWidth(LLM_PROMPT_EDITOR_WIDTH)
        self.profile: LLMProfile = profile_by_id(pcfg.module.llm_profiles, 'codex')
        self.account_controller = CodexAccountController.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        account_controls = QWidget(self)
        account_controls.setObjectName('CodexAccountControls')
        account_controls.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        account_row = QHBoxLayout(account_controls)
        account_row.setContentsMargins(0, 0, 0, 0)
        account_row.setSpacing(12)
        self.account_status = _AccountStatusLabel(account_controls)
        self.account_status.setObjectName('ParamFieldLabel')
        self.account_status.setTextFormat(Qt.TextFormat.PlainText)
        self.account_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        account_row.addWidget(self.account_status, 1)
        self.account_buttons: dict[str, QAbstractButton] = {}
        for action, title, name in (
            ('login', self.tr('Sign in'), 'CodexSignInButton'),
            ('logout', self.tr('Sign out'), 'CodexSignOutButton'),
            ('cancel', self.tr('Cancel'), 'CodexCancelButton'),
        ):
            button = CodexSignInButton(account_controls) if action == 'login' else QPushButton(title, account_controls)
            button.setObjectName(name)
            account_row.addWidget(button)
            self.account_buttons[action] = button
        account_row.addStretch()
        layout.addWidget(account_controls)

        self.refresh_button = RefreshButton(self)
        self.refresh_button.setAccessibleName(self.tr('Refresh models'))
        self.refresh_button.setToolTip(self.tr('Refresh models from your ChatGPT account.'))
        self.account_buttons['refresh'] = self.refresh_button
        for action, button in self.account_buttons.items():
            button.setProperty('codexAction', action)
            button.clicked.connect(self.performAccountAction)

        model_fields = (
            ('model', self.tr('Text')),
            ('vision_model', self.tr('Vision')),
            ('image_model', self.tr('Image')),
        )
        models = ParamWidget({
            key: {'type': 'selector', 'display_name': title, 'value': getattr(self.profile, key),
                  'options': image_model_choices(self.profile) if key == 'image_model' else getattr(self.profile, key + '_options'),
                  'editable': key == 'image_model'}
            for key, title in model_fields
        }, scrollWidget=scrollWidget, spaced_fields=True, parent=self)
        image_model = models.param_widgets['image_model']
        image_model.fit_popup_contents = True
        # Commit typed IDs on Enter/focus loss, not on each partial keystroke.
        image_model.paramwidget_edited.disconnect(models.on_paramwidget_edited)
        image_model.setInsertPolicy(ParamComboBox.InsertPolicy.NoInsert)
        image_model.lineEdit().setObjectName('LLMProfileModelEditor')
        image_model.lineEdit().setPlaceholderText(self.tr('Image model name'))
        image_model.lineEdit().editingFinished.connect(self.finishImageModelEdit)
        image_model.activated.connect(self.finishImageModelEdit)
        image_model.setToolTip(self.tr(
            'Choose an image model directly, or a GPT → GPT Image combination for assisted editing. '
            'Use + to add an image model ID. Availability is checked when used.'))
        image_label = models.param_rows['image_model'][0]
        row, column, _, _ = models.param_layout.getItemPosition(models.param_layout.indexOf(image_label))
        models.param_layout.removeWidget(image_label)
        image_label_row = QHBoxLayout()
        image_label_row.setContentsMargins(0, 0, 0, 0)
        image_label_row.setSpacing(4)
        image_label_row.addWidget(image_label)
        self.add_image_model_btn = QToolButton(self)
        self.remove_image_model_btn = QToolButton(self)
        for button, name, icon, title, slot in (
            (self.add_image_model_btn, 'LLMProfileModelAddButton', 'add.svg',
             self.tr('Add image model'), self.startImageModelEdit),
            (self.remove_image_model_btn, 'LLMProfileModelRemoveButton', 'titlebar_min.svg',
             self.tr('Delete image model and its combinations'), self.deleteCurrentImageModel),
        ):
            button.setObjectName(name)
            button.setIcon(QIcon(themed_icon_path(icon)))
            button.setToolTip(title)
            button.setAccessibleName(title)
            button.setFixedSize(16, 16)
            button.clicked.connect(slot)
            image_label_row.addWidget(button)
        image_label_row.addStretch()
        models.param_layout.addLayout(image_label_row, row, column)
        image_model.lineEdit().textChanged.connect(self.updateImageModelRemoveButton)
        self.updateImageModelRemoveButton(image_model.currentText())
        request_fields = (
            ('thinking_level', 'selector', self.tr('Reasoning level'), self.tr(
                'Auto uses the provider default. Disabled requests no reasoning; explicit levels set the reasoning effort.')),
            ('vision_detail_level', 'selector', self.tr('Vision detail level'), self.tr(
                'Image detail level sent to the vision model.')),
            ('prompt', 'editor', self.tr('Translation prompt'), self.tr(
                'Additional translation instructions for style and wording.')),
            ('vision_prompt', 'editor', self.tr('OCR prompt'), self.tr(
                'Instructions sent to the vision model for OCR.')),
            ('image_prompt', 'editor', self.tr('Inpainting prompt'), self.tr(
                'Instructions sent to the image model for cleanup.')),
        )
        params = {}
        for key, kind, title, description in request_fields:
            params[key] = {'type': kind, 'display_name': title, 'description': description,
                           'value': getattr(self.profile, key), 'label_above': kind == 'editor'}
            if kind == 'selector':
                params[key]['options'] = getattr(self.profile, key + '_options')
        requests = ParamWidget(params, scrollWidget=scrollWidget, spaced_fields=True, parent=self)
        for widget, name in ((models, 'CodexModelSettings'), (requests, 'CodexRequestSettings')):
            widget.setObjectName(name)
            widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            for editor in widget.param_widgets.values():
                if isinstance(editor, ParamComboBox):
                    editor.setFixedWidth(round(editor.width() * LLM_PROFILE_EDITOR_WIDTH_SCALE))
            for row in widget.param_rows.values():
                if len(row) > 1:
                    row[0].setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.param_widgets = {**models.param_widgets, **requests.param_widgets}
        for key in ('prompt', 'vision_prompt', 'image_prompt'):
            editor = self.param_widgets[key]
            editor.setFixedWidth(LLM_PROMPT_EDITOR_WIDTH)
            editor.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, editor, fadeout=False, hover_style=True)
            editor.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, editor, fadeout=False, hover_style=True)
        models.paramwidget_edited.connect(self.onSettingEdited)
        requests.paramwidget_edited.connect(self.onSettingEdited)

        for title, section in ((self.tr('Models'), models),
                               (self.tr('Request settings'), requests)):
            heading = QLabel(title, self)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            if section is models:
                model_heading = QHBoxLayout()
                model_heading.addWidget(heading)
                model_heading.addWidget(self.refresh_button)
                model_heading.addStretch(1)
                layout.addLayout(model_heading)
            else:
                layout.addWidget(heading)
            layout.addWidget(section)

        self.account_controller.changed.connect(self.refreshAccount)
        self.account_controller.catalog_changed.connect(self.onCatalogChanged)
        self.syncFromProfile()
        self.refreshAccount()

    def performAccountAction(self) -> None:
        action = self.sender().property('codexAction')
        if action == 'cancel':
            self.account_controller.cancel()
            return
        if action == 'logout' and QMessageBox.question(
            self, self.tr('Sign out of Codex'), self.tr('Sign out of your ChatGPT account?'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.account_controller.start(action)

    def refreshAccount(self) -> None:
        controller = self.account_controller
        busy = controller.worker is not None
        account_text = self.tr('Connected: {account}').format(account=controller.account) if controller.account else ''
        self.account_status.setText(account_text)
        self.account_status.setToolTip(account_text)
        self.account_status.setVisible(bool(controller.account))
        for action, visible in (('login', not controller.account), ('logout', bool(controller.account))):
            button = self.account_buttons[action]
            button.setVisible(visible)
            button.setEnabled(visible and not busy)
        self.refresh_button.set_busy(busy and controller.worker.action != 'logout')
        self.refresh_button.setEnabled(not busy)
        cancel = self.account_buttons['cancel']
        cancel.setVisible(busy)
        cancel.setEnabled(busy and not controller.worker.stop_event.is_set())

    def syncFromProfile(self) -> None:
        self.profile = profile_by_id(pcfg.module.llm_profiles, 'codex')
        for key, editor in self.param_widgets.items():
            value = getattr(self.profile, key)
            if isinstance(editor, ParamComboBox):
                if key == 'image_model' and editor.hasFocus() and editor.lineEdit().isModified():
                    # Startup/model refresh may finish while the user types an ID.
                    continue
                if key == 'image_model':
                    self._syncCombo(editor, image_model_choices(self.profile), value)
                else:
                    self._syncCombo(editor, getattr(self.profile, key + '_options'), value)
            elif editor.toPlainText() != value:
                with QSignalBlocker(editor):
                    editor.setPlainText(value)
        for key in ('model', 'vision_model'):
            editor = self.param_widgets[key]
            editor.setPlaceholderText(self.tr('Select a model'))
            editor.setToolTip(self.tr('Using default models. Sign in and refresh for available account models.')
                              if not pcfg.module.codex_models else '')

    @staticmethod
    def _syncCombo(combo: ParamComboBox, options: list[str], value: str) -> None:
        available = set(options)
        displayed = options + ([value] if value and value not in available else [])
        with QSignalBlocker(combo):
            if [combo.itemText(index) for index in range(combo.count())] != displayed:
                combo.clear()
                combo.addItems(displayed)
            # Removed selections remain visible, without being offered as usable models.
            for index, option in enumerate(displayed):
                combo.model().item(index).setEnabled(option in available)
            combo.setCurrentIndex(combo.findText(value))

    def onCatalogChanged(self) -> None:
        self.syncFromProfile()
        self.profile_ui_updated.emit()

    def finishImageModelEdit(self) -> bool:
        self.profile = profile_by_id(pcfg.module.llm_profiles, 'codex')
        combo = self.param_widgets['image_model']
        value = combo.currentText().strip() or self.profile.image_model
        choices = image_model_choices(self.profile)
        try:
            reasoning, image = split_image_model_selection(value)
            value = f'{reasoning} → {image}' if reasoning else image
            if reasoning and value not in choices and value != self.profile.image_model:
                raise ValueError('Unavailable image model combination.')
        except ValueError:
            self._syncCombo(combo, choices, self.profile.image_model)
            QMessageBox.warning(self, self.tr('Invalid image model'), self.tr(
                'Enter an image model ID, or choose a GPT → GPT Image combination from the list.'))
            return False
        if image and image not in self.profile.image_model_options:
            self.profile.image_model_options.append(image)
            choices = image_model_choices(self.profile)
        self._syncCombo(combo, choices, value)
        combo.lineEdit().setModified(False)
        self.onSettingEdited('image_model', {'content': value})
        return True

    def startImageModelEdit(self) -> None:
        if not self.finishImageModelEdit():
            return
        combo = self.param_widgets['image_model']
        combo.setEditText('')
        # An empty add is a draft too; catalog refresh must not restore it yet.
        combo.lineEdit().setModified(True)
        combo.setFocus(Qt.FocusReason.OtherFocusReason)

    def updateImageModelRemoveButton(self, text: str) -> None:
        # Delete only a committed selection, never the previous model behind a draft.
        self.remove_image_model_btn.setEnabled(bool(text.strip()) and text.strip() == self.profile.image_model)

    def deleteCurrentImageModel(self) -> None:
        if not self.finishImageModelEdit():
            return
        options = self.profile.image_model_options
        image = split_image_model_selection(self.profile.image_model)[1]
        if image not in options:
            return
        index = options.index(image)
        options.pop(index)
        value = options[min(index, len(options) - 1)] if options else ''
        self._syncCombo(self.param_widgets['image_model'], image_model_choices(self.profile), value)
        self.onSettingEdited('image_model', {'content': value})

    def onSettingEdited(self, key: str, content: dict[str, str]) -> None:
        self.profile = profile_by_id(pcfg.module.llm_profiles, 'codex')
        value = content['content']
        if getattr(self.profile, key) == value:
            return
        setattr(self.profile, key, value)
        if key == 'image_model':
            self.updateImageModelRemoveButton(value)
        if key == 'model':
            self.profile.thinking_level_options = codex_thinking_options(self.profile, pcfg.module.codex_models)
            self._syncCombo(self.param_widgets['thinking_level'], self.profile.thinking_level_options, self.profile.thinking_level)
        if isinstance(self.param_widgets[key], ParamComboBox):
            self.profile_summary_changed.emit()

    def focusControl(self, target: str = 'model') -> None:
        account_button = self.account_buttons['logout' if self.account_controller.account else 'login']
        editor = self.param_widgets.get(target, account_button)
        editor.setFocus(Qt.FocusReason.OtherFocusReason)
        # Prompt editors finish sizing after their first show event.
        QTimer.singleShot(0, self._scrollToFocusedControl)

    def _scrollToFocusedControl(self) -> None:
        editor = self.focusWidget()
        if editor is None:
            return
        # Settle the owning layouts before using the scroll area's range.
        parent = self
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(editor)
                break
            if parent.layout() is not None:
                parent.layout().activate()
            parent = parent.parentWidget()
