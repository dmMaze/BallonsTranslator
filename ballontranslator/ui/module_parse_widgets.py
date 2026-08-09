from .custom_widget import ParamComboBox, ParamNameLabel
from ballontranslator.utils.shared import (
    CONFIG_COMBOBOX_HEIGHT,
    CONFIG_COMBOBOX_LONG,
    CONFIG_CONTENT_ROW_SPACING,
    size2width,
)
from ballontranslator.utils.config import save_config
from .framelesswindow import OutsideClickFramelessMixin
from .module_param_i18n import (
    tr_module_description,
    tr_param_description,
    tr_param_display_name,
)

from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QApplication,
    QPlainTextEdit,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QPushButton,
    QSizePolicy,
    QMenu,
    QDialog,
    QFrame,
    QLabel,
    QScrollArea,
)
from qtpy.QtCore import QTimer, Qt, Signal
from qtpy.QtGui import QDoubleValidator, QKeySequence

try:
    from qtpy.QtGui import QAction
except ImportError:
    from qtpy.QtWidgets import QAction

class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=k, parent=self)
            checker.setObjectName('ConfigCheckBox')
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())


class SecretLineEditor(QLineEdit):
    """Password editor with an explicit copy path for hidden text.

    Example:
        >>> SecretLineEditor.__name__
        'SecretLineEditor'
    """

    def copySecretText(self):
        text = self.selectedText() if self.hasSelectedText() else self.text()
        if not text:
            return
        QApplication.clipboard().setText(text)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copySecretText()
            event.accept()
            return
        return super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        copy_action = QAction(self.tr('Copy'), menu)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.setEnabled(bool(self.text()))
        paste_action = QAction(self.tr('Paste'), menu)
        paste_action.setEnabled(not self.isReadOnly() and bool(QApplication.clipboard().text()))
        select_all_action = QAction(self.tr('Select All'), menu)
        select_all_action.setEnabled(bool(self.text()))
        menu.addAction(copy_action)
        menu.addAction(paste_action)
        menu.addSeparator()
        menu.addAction(select_all_action)
        action = menu.exec(event.globalPos()) if hasattr(menu, 'exec') else menu.exec_(event.globalPos())
        if action == copy_action:
            self.copySecretText()
        elif action == paste_action:
            self.paste()
        elif action == select_all_action:
            self.selectAll()


class SecretParamWidget(QWidget):
    """Password-style parameter editor.

    Example:
        >>> SecretParamWidget.__name__
        'SecretParamWidget'
    """

    paramwidget_edited = Signal(str, str)

    def __init__(self, param_key: str, size='short', fixed_size: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName('SecretParamWidget')
        self.param_key = param_key
        self.editor = SecretLineEditor(self)
        if fixed_size:
            self.editor.setFixedWidth(size2width(size))
            self.setFixedWidth(size2width(size))
        else:
            self.editor.setMinimumWidth(size2width(size))
            self.editor.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.setMinimumWidth(size2width(size))
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.editor.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.editor.setEchoMode(QLineEdit.EchoMode.Password)
        self.editor.setToolTip(self.tr(
            'Stored in portable obfuscated form. This hides the key from plain-text scans, '
            'but it is not a secure password vault.'
        ))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor)
        self.editor.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.editor.text())

    def setText(self, text: str):
        self.editor.setText(text)

    def text(self):
        return self.editor.text()

    def setFocus(self, *args, **kwargs):
        return self.editor.setFocus(*args, **kwargs)

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self._auto_max_height = 100

        self._showed = False

        self.setFixedWidth(int(CONFIG_COMBOBOX_LONG))
        self.textChanged.connect(self.on_text_changed)
        self.document().documentLayout().documentSizeChanged.connect(lambda *_: self.adjustSize())

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._showed:
            self._showed = True
            QTimer.singleShot(0, self.adjustSize)

    def adjustSize(self):

        # QPlainTextDocumentLayout.documentSize().height() reports a block
        # count here; block bounds provide the actual wrapped visual heights.
        document_layout = self.document().documentLayout()
        block = self.document().begin()
        content_height = 0.0
        while block.isValid():
            content_height += document_layout.blockBoundingRect(block).height()
            if content_height >= self._auto_max_height:
                break
            block = block.next()

        content_height += self.frameWidth() * 2
        height = min(round(content_height), self._auto_max_height)
        if self.height() != height:
            self.setFixedHeight(height)


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        self.checker.setObjectName('ConfigCheckBox')
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.checker)
        hlayout.addWidget(name_label)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)

    def isChecked(self):
        return self.checker.isChecked()


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setObjectName('ParamCheckBox')
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def ensure_current_device_option(param_dict: dict):
    not_supported = param_dict.get('__device_not_supported', [])
    current_value = str(param_dict.get('value', 'cpu'))
    options = [str(opt) for opt in param_dict.get('options', [])]
    if current_value not in options and all(device not in current_value for device in not_supported):
        options.append(current_value)
    param_dict['options'] = options
    param_dict['value'] = current_value if current_value in options else 'cpu'


def set_label_tooltip_from_widget(label: QWidget, widget: QWidget):
    """Mirror an editor tooltip onto its visible label.

    Example:
        >>> set_label_tooltip_from_widget.__name__
        'set_label_tooltip_from_widget'
    """
    tooltip = widget.toolTip()
    if tooltip:
        label.setToolTip(tooltip)


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, display_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(display_name)
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(
        self,
        params,
        scrollWidget: QWidget = None,
        module_type: str = '',
        module_key: str = '',
        spaced_fields: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.module_type = module_type
        self.module_key = module_key
        horizontal_policy = (
            QSizePolicy.Policy.Expanding
            if spaced_fields
            else QSizePolicy.Policy.Maximum
        )
        self.setSizePolicy(horizontal_policy, QSizePolicy.Policy.Maximum)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.param_layout = param_layout = QGridLayout()
        self.param_widgets = {}
        self.param_rows = {}
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setVerticalSpacing(CONFIG_CONTENT_ROW_SPACING)
        if spaced_fields:
            param_layout.setColumnStretch(1, 1)
            layout.addLayout(param_layout, 1)
        else:
            param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            layout.addLayout(param_layout)
            layout.addStretch(-1)

        module_description = tr_module_description(params, module_type, module_key)
        if module_description:
            self.setToolTip(module_description)

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = tr_param_display_name(
                params, param_key, module_type=module_type, module_key=module_key)
            param_description = tr_param_description(
                params, param_key, module_type=module_type, module_key=module_key)

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None
            label_above = False

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = tr_param_display_name(
                    params, param_key, param_dict, module_type, module_key)
                param_description = tr_param_description(
                    params, param_key, param_dict, module_type, module_key)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                label_above = param_dict.get('label_above', False)
                if param_key == 'device' and param_type == 'selector':
                    ensure_current_device_option(param_dict)
                    value = param_dict['value']
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    param_widget = ParamComboBox(
                        param_key, param_dict['options'], size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    param_widget.setEditable(param_dict.get('editable', False))
                    param_widget.setCurrentText(str(value))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, display_param_name)
                    require_label = False

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=isinstance(value, (float, int)))
                    param_widget.setText(str(value))

                elif param_type == 'secret':
                    param_widget = SecretParamWidget(param_key, size=param_size)
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if param_description:
                        param_widget.setToolTip(param_description)

            if param_widget is not None and require_label and label_above:
                self.param_widgets[param_key] = param_widget
                row_widget = QWidget(self)
                row_widget.setObjectName('ParamLabelAboveRow')
                row_widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
                row_layout = QVBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                param_label = ParamNameLabel(display_param_name)
                param_label.setObjectName('ParamLabelAboveLabel')
                set_label_tooltip_from_widget(param_label, param_widget)
                row_layout.addWidget(param_label, 0, Qt.AlignmentFlag.AlignLeft)
                row_layout.addWidget(param_widget, 0, Qt.AlignmentFlag.AlignLeft)
                column_span = 3 if spaced_fields else 2
                param_layout.addWidget(row_widget, ii, 0, 1, column_span)
                self.param_rows[param_key] = [row_widget]
                continue

            widget_idx = 2 if spaced_fields else 0
            row_widgets = []
            if require_label:
                label_alignment = None
                if spaced_fields:
                    label_alignment = (
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                param_label = ParamNameLabel(
                    display_param_name,
                    alignment=label_alignment,
                )
                param_label.setObjectName('ParamFieldLabel')
                if param_widget is not None:
                    set_label_tooltip_from_widget(param_label, param_widget)
                param_layout.addWidget(param_label, ii, 0)
                row_widgets.append(param_label)
                widget_idx = 2 if spaced_fields else 1
            if param_widget is not None:
                self.param_widgets[param_key] = param_widget
                row_widgets.append(param_widget)
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(
                        param_widget,
                        ii,
                        widget_idx,
                        Qt.AlignmentFlag.AlignLeft,
                    )
                else:
                    param_layout.addLayout(
                        pw_lo,
                        ii,
                        widget_idx,
                        Qt.AlignmentFlag.AlignLeft,
                    )
                self.param_rows[param_key] = row_widgets
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

    def setParamVisible(self, param_key: str, visible: bool):
        for widget in self.param_rows.get(param_key, []):
            widget.setVisible(visible)

    def setRuntimeActionsEnabled(self, enabled: bool) -> None:
        """Disable commands that require a loaded module instance.

        >>> hasattr(ParamWidget, 'setRuntimeActionsEnabled')
        True
        """
        for widget in self.param_widgets.values():
            if isinstance(widget, ParamPushButton):
                widget.setEnabled(enabled)
            flush_button = getattr(widget, 'flush_btn', None)
            if flush_button is not None:
                flush_button.setEnabled(enabled)


def has_configurable_params(params: dict) -> bool:
    return isinstance(params, dict) and any(
        key != 'description' and not key.startswith('__')
        for key in params
    )


class ModuleParamDialog(OutsideClickFramelessMixin, QDialog):
    """Ephemeral editor for one selected module's live parameter mapping.

    >>> issubclass(ModuleParamDialog, QDialog)
    True
    """

    paramwidget_edited = Signal(str, str, str, dict)

    def __init__(
        self,
        module_type: str,
        module_key: str,
        params: dict,
        runtime_actions_enabled: bool,
        parent: QWidget = None,
    ) -> None:
        window_type = getattr(Qt, 'WindowType', Qt)
        super().__init__(
            parent,
            window_type.Dialog | window_type.FramelessWindowHint,
        )
        self.module_type = module_type
        self.module_key = module_key
        self.setObjectName('ModuleParamDialog')
        self.setWindowTitle(module_key)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumWidth(420)
        self.setMaximumHeight(640)
        widget_attribute = getattr(Qt, 'WidgetAttribute', Qt)
        self.setAttribute(widget_attribute.WA_TranslucentBackground)
        self.setAttribute(widget_attribute.WA_DeleteOnClose)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 5, 5)
        surface = QFrame(self)
        surface.setObjectName('ModuleParamSurface')
        root_layout.addWidget(surface)

        layout = QVBoxLayout(surface)
        layout.setContentsMargins(22, 16, 22, 18)
        layout.setSpacing(14)
        self.title_bar = QWidget(surface)
        self.title_bar.setObjectName('ModuleParamTitleBar')
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(module_key, self.title_bar)
        self.title_label.setObjectName('ModuleParamTitle')
        title_layout.addWidget(self.title_label)
        title_layout.addStretch()
        self.close_button = None
        layout.addWidget(self.title_bar)

        if has_configurable_params(params):
            scroll = QScrollArea(surface)
            scroll.setObjectName('ModuleParamScrollArea')
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setSizeAdjustPolicy(
                QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
            )
            scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            self.param_widget = ParamWidget(
                params,
                scrollWidget=scroll,
                module_type=module_type,
                module_key=module_key,
                spaced_fields=True,
            )
            self.param_widget.setObjectName('ModuleParamContent')
            self.param_widget.setRuntimeActionsEnabled(runtime_actions_enabled)
            self.param_widget.paramwidget_edited.connect(self._on_paramwidget_edited)
            scroll.setWidget(self.param_widget)
            scroll.setMinimumWidth(
                self.param_widget.minimumSizeHint().width()
                + scroll.verticalScrollBar().sizeHint().width()
            )
            layout.addWidget(scroll)
        else:
            self.param_widget = None
            empty_label = QLabel(self.tr('No configurable param'), surface)
            empty_label.setObjectName('ModuleParamEmptyLabel')
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(empty_label)
        self.adjustSize()

    def _on_paramwidget_edited(self, param_key: str, content: dict) -> None:
        self.paramwidget_edited.emit(
            self.module_type,
            self.module_key,
            param_key,
            content,
        )

    def _dismiss_transient_window(self) -> None:
        parent = self.parentWidget()
        parent_window = parent.window() if parent is not None else None
        self.close()
        if (
            parent_window is not None
            and parent_window.windowModality() != Qt.WindowModality.NonModal
        ):
            # Windows can miss this handoff when close() runs in the mouse filter.
            parent_window.activateWindow()

    def _preserve_on_outside_click(self) -> bool:
        active_modal = QApplication.activeModalWidget()
        parent = self.parentWidget()
        parent_window = parent.window() if parent is not None else None
        return active_modal not in (None, self, parent_window)

    def closeEvent(self, event) -> None:
        save_config()
        super().closeEvent(event)
