import json
from typing import Callable

from qtpy.QtCore import QEvent, QSize, Qt, Signal
from qtpy.QtGui import QIcon, QPainter
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QWidgetAction,
)

from .custom_widget import SmallComboBox, Widget
from .icon_rendering import render_svg_pixmap
from .llm_modality import (
    LLM_MODALITY_IMAGE,
    LLM_MODALITY_IMAGE_COLOR,
    LLM_MODALITY_TEXT,
    LLM_MODALITY_TEXT_COLOR,
    LLM_MODALITY_VISION,
    LLM_MODALITY_VISION_COLOR,
)
from .misc import themed_icon_path
from ballontranslator.utils import shared
from ballontranslator.utils.config import pcfg
from ballontranslator.utils.llm_profiles import (
    LLM_INPAINT_KEY,
    LLM_OCR_KEY,
    LLM_TRANSLATOR_KEY,
    profile_by_id,
)

if shared.FLAG_QT6:
    from qtpy.QtGui import QAction
else:
    from qtpy.QtWidgets import QAction


class SmallConfigPutton(QPushButton):
    pass


class BottomBarModuleToolButton(QToolButton):
    """Bottom module selector with a cached themed dropdown chevron.

    >>> BottomBarModuleToolButton.__name__
    'BottomBarModuleToolButton'
    """

    CHEVRON_SIZE = 12
    CHEVRON_RIGHT_MARGIN = 8

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = render_svg_pixmap(
            themed_icon_path('chevron-down.svg'),
            self.CHEVRON_SIZE,
            self.CHEVRON_SIZE,
            self.devicePixelRatioF(),
        )
        painter = QPainter(self)
        x = self.width() - self.CHEVRON_RIGHT_MARGIN - self.CHEVRON_SIZE
        y = (self.height() - self.CHEVRON_SIZE) // 2
        painter.drawPixmap(x, y, pixmap)
        painter.end()


def _set_bottom_aux_button_visible(button: QPushButton, visible: bool):
    if visible == (not button.isHidden()):
        return
    button.setVisible(visible)
    parent = button.parentWidget()
    if parent is not None:
        layout = parent.layout()
        if layout is not None:
            layout.invalidate()
        parent.updateGeometry()


def cfg_icon() -> QIcon:
    return QIcon(themed_icon_path('leftbar_config_activate.svg'))


def _theme_value(key: str, fallback: str) -> str:
    theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
    try:
        with open(shared.THEME_PATH, 'r', encoding='utf8') as f:
            theme_dict = json.loads(f.read())
        return theme_dict.get(theme, {}).get(key, fallback)
    except Exception:
        return fallback


def _theme_foreground_hex() -> str:
    fallback = '#8e99b1' if pcfg.darkmode else '#5d5d5f'
    return _theme_value('@qwidgetForegroundColor', fallback)


def _theme_menu_background_hex() -> str:
    fallback = '#21252b' if pcfg.darkmode else '#e1e4eb'
    return _theme_value('@emptyContentBackgroundColor', fallback)


def _blend_hex(color: str, target: str, amount: float) -> str:
    def rgb(value: str):
        value = value.lstrip('#')
        return [int(value[i:i + 2], 16) for i in (0, 2, 4)]

    src = rgb(color)
    dst = rgb(target)
    mixed = [round(src[i] + (dst[i] - src[i]) * amount) for i in range(3)]
    return '#{:02x}{:02x}{:02x}'.format(*mixed)


def _section_label_color() -> str:
    foreground = _theme_foreground_hex()
    return _blend_hex(foreground, '#000000' if pcfg.darkmode else '#ffffff', 0.28)


def _hex_to_rgb(color: str):
    color = color.lstrip('#')
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def _rgba_from_hex(color: str, alpha_percent: int) -> str:
    red, green, blue = _hex_to_rgb(color)
    return 'rgba({}, {}, {}, {}%)'.format(red, green, blue, alpha_percent)


def _set_bottom_tool_button_visuals(tool_btn: QToolButton, icon_filename: str, color: str = ''):
    tool_btn.setIcon(QIcon(themed_icon_path(icon_filename)))
    tool_btn.setIconSize(QSize(18, 18))
    style_enum = getattr(Qt, 'ToolButtonStyle', Qt)
    tool_btn.setToolButtonStyle(style_enum.ToolButtonTextBesideIcon)

    if not color:
        tool_btn.setStyleSheet('')
        return
    hover_color = _rgba_from_hex(color, 20)
    tool_btn.setStyleSheet(
        'QToolButton#BottomBarModuleToolButton:hover, '
        'QToolButton#BottomBarModuleToolButton:pressed, '
        'QToolButton#BottomBarModuleToolButton:open {{ '
        'border: none; background-color: {}; '
        '}}'.format(hover_color)
    )


def _instant_popup_mode():
    popup_enum = getattr(QToolButton, 'ToolButtonPopupMode', QToolButton)
    return popup_enum.InstantPopup


def _bottom_tool_button_text(name: str) -> str:
    return '  ' + name


def _simplify_llm_model_name(model: str) -> str:
    parts = [part.strip() for part in str(model or '').split('/') if part.strip()]
    return parts[-1] if parts else ''


def _bottom_submenu(title: str, parent: QMenu) -> QMenu:
    menu = QMenu(parent)
    menu.setTitle(title)
    return menu


def _add_bottom_menu_section(menu: QMenu, text: str, color: str = ''):
    label = QLabel(text, menu)
    label.setObjectName('MenuSectionLabel')
    color = color or _section_label_color()
    label.setStyleSheet(
        'QLabel#MenuSectionLabel {{ '
        'color: {}; background-color: {}; '
        '}}'.format(color, _theme_menu_background_hex())
    )
    action = QWidgetAction(menu)
    action.setDefaultWidget(label)
    menu.addAction(action)


def _checked_action_text(text: str, checked: bool) -> str:
    return text + ('\t\u2713' if checked else '')


def _add_bottom_menu_action(
    menu: QMenu,
    text: str,
    checked: bool,
    data: object,
    slot: Callable[[bool], None],
) -> QAction:
    action = QAction(_checked_action_text(text, checked), menu)
    action.setData(data)
    action.triggered.connect(slot)
    menu.addAction(action)
    return action


def _add_bottom_submenu(parent: QMenu, submenu: QMenu, text: str, checked: bool):
    parent.addMenu(submenu)
    submenu.menuAction().setText(_checked_action_text(text, checked))
    return submenu


class ModuleSelectionWidget(Widget):

    cfg_clicked = Signal()
    edit_clicked = Signal(str)
    llm_profile_changed = Signal(str)

    def __init__(
        self,
        fallback_name: str,
        icon_filename: str,
        llm_modality: str = '',
        icon_color: str = '',
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fallback_name = fallback_name
        self.icon_filename = icon_filename
        self.icon_color = icon_color
        self.llm_modality = llm_modality
        if self._has_llm_modality():
            self._configure_modality(llm_modality)
        self.selector = SmallComboBox()
        self.selector.setVisible(False)
        self.selector.currentTextChanged.connect(self.updateButtonText)
        if self._is_text_modality():
            self.src_selector = SmallComboBox()
            self.tgt_selector = SmallComboBox()
            self.src_selector.setVisible(False)
            self.tgt_selector.setVisible(False)

        self.tool_btn = BottomBarModuleToolButton(self)
        self.tool_btn.setObjectName('BottomBarModuleToolButton')
        self.tool_btn.setToolTip(fallback_name)
        self.tool_btn.setPopupMode(_instant_popup_mode())
        _set_bottom_tool_button_visuals(
            self.tool_btn,
            self.icon_filename,
            self.icon_color,
        )
        self.tool_btn.setText(fallback_name)
        self.menu = QMenu(self.tool_btn)
        self.tool_btn.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.rebuildMenu)

        self.cfg_btn = SmallConfigPutton()
        self.cfg_btn.clicked.connect(self.cfg_clicked)
        self.cfg_btn.setVisible(False)
        self.edit_btn = SmallConfigPutton()
        self.edit_btn.clicked.connect(self.onEditClicked)
        self.edit_btn.setVisible(False)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.tool_btn)
        layout.addWidget(self.edit_btn)
        layout.addWidget(self.cfg_btn)
        self.updateButtonText()

    def _has_llm_modality(self) -> bool:
        return bool(self.llm_modality)

    def _configure_modality(self, modality: str):
        if modality == LLM_MODALITY_TEXT:
            self.llm_key = LLM_TRANSLATOR_KEY
            self.profile_id_attr = 'translator_llm_id'
            self.profile_support_attr = 'support_text'
            self.model_attr = 'model'
            self.model_options_attr = 'model_options'
            self.modality_color = LLM_MODALITY_TEXT_COLOR
            self.icon_filename = 'text.svg'
            self.icon_color = self.modality_color
            self.module_attr_to_set = ''
        elif modality == LLM_MODALITY_VISION:
            self.llm_key = LLM_OCR_KEY
            self.profile_id_attr = 'ocr_llm_id'
            self.profile_support_attr = 'support_vision'
            self.model_attr = 'vision_model'
            self.model_options_attr = 'vision_model_options'
            self.modality_color = LLM_MODALITY_VISION_COLOR
            self.icon_filename = 'eye.svg'
            self.icon_color = self.modality_color
            self.module_attr_to_set = ''
        elif modality == LLM_MODALITY_IMAGE:
            self.llm_key = LLM_INPAINT_KEY
            self.profile_id_attr = 'inpaint_llm_id'
            self.profile_support_attr = 'support_image'
            self.model_attr = 'image_model'
            self.model_options_attr = 'image_model_options'
            self.modality_color = LLM_MODALITY_IMAGE_COLOR
            self.icon_filename = 'image.svg'
            self.icon_color = self.modality_color
            self.module_attr_to_set = 'inpainter'
        else:
            raise ValueError('Unknown LLM modality: {}'.format(modality))

    def _is_text_modality(self) -> bool:
        return self.llm_modality == LLM_MODALITY_TEXT

    def _is_current_llm(self) -> bool:
        return self._has_llm_modality() and self.selector.currentText() == self.llm_key

    def _selected_profile_id(self) -> str:
        return getattr(pcfg.module, self.profile_id_attr)

    def enterEvent(self, event: QEvent) -> None:
        show_edit = self.shouldShowEditButton()
        _set_bottom_aux_button_visible(self.edit_btn, show_edit)
        if show_edit:
            self.edit_btn.setIcon(QIcon(themed_icon_path('edit.svg')))
        _set_bottom_aux_button_visible(self.cfg_btn, True)
        self.cfg_btn.setIcon(cfg_icon())
        return super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        self.edit_btn.setIcon(QIcon())
        self.cfg_btn.setIcon(QIcon())
        _set_bottom_aux_button_visible(self.edit_btn, False)
        _set_bottom_aux_button_visible(self.cfg_btn, False)
        return super().leaveEvent(event)

    def blockSignals(self, block: bool):
        self.selector.blockSignals(block)
        if self._is_text_modality():
            self.src_selector.blockSignals(block)
            self.tgt_selector.blockSignals(block)
        super().blockSignals(block)

    def setSelectedValue(self, value: str, block_signals=True):
        if block_signals:
            self.blockSignals(True)
        self.selector.setCurrentText(value)
        if block_signals:
            self.blockSignals(False)
        self.updateButtonText()

    def rebuildMenu(self):
        self.menu.clear()
        current_module = self.selector.currentText()
        if self._has_llm_modality():
            self._section(self.fallback_name)
        for i in range(self.selector.count()):
            module = self.selector.itemText(i)
            if self._has_llm_modality() and module == self.llm_key:
                continue
            _add_bottom_menu_action(
                self.menu,
                module,
                module == current_module,
                module,
                self._select_module_action,
            )
        if self._has_llm_modality():
            self._addLlmProfileMenus(current_module)

    def _addLlmProfileMenus(self, current_module: str):
        self._section(self.tr('LLM'), color=self.modality_color)
        added = False
        for profile in pcfg.module.llm_profiles:
            if not getattr(profile, self.profile_support_attr):
                continue
            added = True
            profile_id = profile.id
            profile_menu = _bottom_submenu(profile.name or profile_id, self.menu)
            _add_bottom_submenu(
                self.menu,
                profile_menu,
                profile.name or profile_id,
                current_module == self.llm_key and self._selected_profile_id() == profile_id,
            )
            self._buildProfileMenu(profile_menu, profile)
        if not added:
            action = QAction(self._no_profiles_text(), self.menu)
            action.setEnabled(False)
            self.menu.addAction(action)

        if self._is_text_modality():
            self._addLanguageMenus()

    def _section(self, text: str, color: str = ''):
        _add_bottom_menu_section(self.menu, text, color=color)

    def _no_profiles_text(self) -> str:
        if self.llm_modality == LLM_MODALITY_TEXT:
            return self.tr('No text profiles')
        if self.llm_modality == LLM_MODALITY_VISION:
            return self.tr('No vision profiles')
        return self.tr('No image profiles')

    def _addLanguageMenus(self):
        self._section(self.tr('Language'))
        source_menu = _bottom_submenu(
            self.tr('Source - {language}').format(language=self.src_selector.currentText()),
            self.menu,
        )
        self.menu.addMenu(source_menu)
        for i in range(self.src_selector.count()):
            lang = self.src_selector.itemText(i)
            _add_bottom_menu_action(
                source_menu,
                lang,
                lang == self.src_selector.currentText(),
                lang,
                self._select_source_language_action,
            )

        target_menu = _bottom_submenu(
            self.tr('Target - {language}').format(language=self.tgt_selector.currentText()),
            self.menu,
        )
        self.menu.addMenu(target_menu)
        for i in range(self.tgt_selector.count()):
            lang = self.tgt_selector.itemText(i)
            _add_bottom_menu_action(
                target_menu,
                lang,
                lang == self.tgt_selector.currentText(),
                lang,
                self._select_target_language_action,
            )

    def _select_module_action(self, _checked: bool = False) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            self.selector.setCurrentText(str(action.data()))

    def _select_source_language_action(self, _checked: bool = False) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            self.src_selector.setCurrentText(str(action.data()))

    def _select_target_language_action(self, _checked: bool = False) -> None:
        action = self.sender()
        if isinstance(action, QAction):
            self.tgt_selector.setCurrentText(str(action.data()))

    def selectLLMProfile(self, profile_id: str):
        if not self._has_llm_modality():
            return
        setattr(pcfg.module, self.profile_id_attr, profile_id)
        if self.module_attr_to_set:
            setattr(pcfg.module, self.module_attr_to_set, self.llm_key)
        if self.selector.currentText() != self.llm_key:
            self.selector.setCurrentText(self.llm_key)
        self.llm_profile_changed.emit(profile_id)
        self.updateButtonText()

    def selectLLMProfileSetting(self, profile_id: str, key: str, value: str):
        profile = profile_by_id(pcfg.module.llm_profiles, profile_id)
        if profile is not None:
            setattr(profile, key, value)
            if key == self.model_attr:
                options = getattr(profile, self.model_options_attr)
                if value and value not in options:
                    options.insert(0, value)
        self.selectLLMProfile(profile_id)

    def _profile_menu_groups(self):
        if self.llm_modality == LLM_MODALITY_TEXT:
            return [
                (self.tr('Thinking Level'), 'thinking_level', 'thinking_level_options'),
                (self.tr('Model'), self.model_attr, self.model_options_attr),
            ]
        if self.llm_modality == LLM_MODALITY_VISION:
            return [
                (self.tr('Vision Model'), self.model_attr, self.model_options_attr),
                (self.tr('Vision Detail Level'), 'vision_detail_level', 'vision_detail_level_options'),
            ]
        return [
            (self.tr('Image Model'), self.model_attr, self.model_options_attr),
        ]

    def _buildProfileMenu(self, menu: QMenu, profile):
        profile_id = profile.id
        selected_profile = self._is_current_llm() and self._selected_profile_id() == profile_id
        for section, value_attr, options_attr in self._profile_menu_groups():
            _add_bottom_menu_section(menu, section, color=self.modality_color)
            options = [str(option) for option in getattr(profile, options_attr) if str(option)]
            current_value = str(getattr(profile, value_attr) or 'None')
            for option in options:
                _add_bottom_menu_action(
                    menu,
                    option,
                    selected_profile and option == current_value,
                    (profile_id, value_attr, option),
                    self._select_profile_setting_action,
                )

    def _select_profile_setting_action(self, _checked: bool = False) -> None:
        action = self.sender()
        if not isinstance(action, QAction):
            return
        profile_id, key, value = action.data()
        self.selectLLMProfileSetting(profile_id, key, value)

    def _buttonTextForProfile(self, profile) -> str:
        if self._is_text_modality():
            model_options = [str(option) for option in profile.model_options if str(option)]
            model = str(profile.model or '').strip()
            thinking_level = str(profile.thinking_level or 'None').strip()
            if model_options and model:
                name = _simplify_llm_model_name(model)
                if thinking_level and thinking_level != 'None':
                    name = self.tr('{model} {thinking_level}').format(model=name, thinking_level=thinking_level)
                return name
            return profile.name or self.llm_key

        model = str(getattr(profile, self.model_attr) or '').strip()
        return _simplify_llm_model_name(model) or profile.name or self.llm_key

    def updateButtonText(self, *args):
        name = self.selector.currentText()
        is_llm = self._is_current_llm()
        if is_llm:
            profile = profile_by_id(pcfg.module.llm_profiles, self._selected_profile_id())
            if profile is not None:
                name = self._buttonTextForProfile(profile)
        if not name:
            name = self.fallback_name
        self.tool_btn.setText(_bottom_tool_button_text(name))
        if self._has_llm_modality():
            _set_bottom_aux_button_visible(self.edit_btn, self.shouldShowEditButton() and self.underMouse())

    def shouldShowEditButton(self) -> bool:
        return self._is_current_llm()

    def onEditClicked(self):
        if self._is_current_llm():
            self.edit_clicked.emit(self._selected_profile_id())

    def setTranslatorMetadata(self, name: str, supported_src_list, supported_tgt_list, lang_source: str, lang_target: str):
        if not self._is_text_modality():
            return
        self.blockSignals(True)
        self.src_selector.clear()
        self.tgt_selector.clear()
        self.src_selector.addItems(supported_src_list)
        self.tgt_selector.addItems(supported_tgt_list)
        self.selector.setCurrentText(name)
        self.src_selector.setCurrentText(lang_source)
        self.tgt_selector.setCurrentText(lang_target)
        self.blockSignals(False)
        self.updateButtonText()
