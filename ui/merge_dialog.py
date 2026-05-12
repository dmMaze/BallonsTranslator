from qtpy.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QLineEdit, QPlainTextEdit, QCheckBox, QSpinBox, QLabel, QRadioButton, QButtonGroup, QHBoxLayout, QPushButton
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QSizePolicy

class MergeDialog(QDialog):
    run_current_clicked = Signal()
    run_all_clicked = Signal()

    def _hint_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        return label

    def _form_label(self, text: str, tooltip: str = None) -> QLabel:
        label = QLabel(text)
        if tooltip:
            label.setToolTip(tooltip)
        return label

    def _set_tooltip(self, widget, tooltip: str):
        widget.setToolTip(tooltip)
        return widget

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Region Merge Tool Settings"))
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.adjustSize()
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(8, 8, 8, 8)

        # --- Mappings for translation ---
        self.merge_mode_map = {
            self.tr("Vertical Merge"): "VERTICAL",
            self.tr("Horizontal Merge"): "HORIZONTAL",
            self.tr("Vertical then Horizontal"): "VERTICAL_THEN_HORIZONTAL",
            self.tr("Horizontal then Vertical"): "HORIZONTAL_THEN_VERTICAL",
            self.tr("None"): "NONE",
        }
        self.label_strategy_map = {
            self.tr("Prefer Shorter Label"): "PREFER_SHORTER",
            self.tr("Use First Box Label"): "FIRST",
            self.tr("Combine Labels (label1+label2)"): "COMBINE",
            self.tr("Prefer Non-default Label"): "PREFER_NON_DEFAULT",
        }

        # --- Main Settings --- #
        main_group = QGroupBox(self.tr("Main Settings"))
        main_layout = QFormLayout(main_group)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.addRow(self._hint_label(self.tr(
            "Choose which neighboring text regions are combined before the result is written back."
        )))

        self.merge_mode = QComboBox()
        for text, data in self.merge_mode_map.items():
            self.merge_mode.addItem(text, userData=data)
        self._set_tooltip(self.merge_mode, self.tr(
            "Controls the merge direction. Sequential modes run one direction first and then merge the result in the second direction."
        ))
        main_layout.addRow(self._form_label(self.tr("Merge Mode:"), self.merge_mode.toolTip()), self.merge_mode)
        self.layout.addWidget(main_group)

        # --- Text Reading Order Settings ---
        reading_order_group = QGroupBox(self.tr("Text Merge Order (by Label)"))
        reading_order_layout = QFormLayout(reading_order_group)
        reading_order_layout.setSpacing(4)
        reading_order_layout.setContentsMargins(8, 6, 8, 6)
        reading_order_layout.addRow(self._hint_label(self.tr(
            "Assign labels to a reading direction so merged text keeps the expected order."
        )))

        self.ltr_labels_edit = QLineEdit()
        self.ltr_labels_edit.setPlaceholderText(self.tr("label1,label2,..."))
        self.rtl_labels_edit = QLineEdit()
        self.rtl_labels_edit.setText("balloon,qipao,shuqing")
        self.ttb_labels_edit = QLineEdit()
        self.ttb_labels_edit.setText("changfangtiao,hengxie")

        ltr_tip = self.tr("Comma-separated labels whose text should be ordered from left to right after merging.")
        rtl_tip = self.tr("Comma-separated labels whose text should be ordered from right to left after merging.")
        ttb_tip = self.tr("Comma-separated labels whose text should be ordered from top to bottom after merging.")
        self._set_tooltip(self.ltr_labels_edit, ltr_tip)
        self._set_tooltip(self.rtl_labels_edit, rtl_tip)
        self._set_tooltip(self.ttb_labels_edit, ttb_tip)

        reading_order_layout.addRow(self._form_label(self.tr("Left-to-right (LTR) Labels:"), ltr_tip), self.ltr_labels_edit)
        reading_order_layout.addRow(self._form_label(self.tr("Right-to-left (RTL) Labels:"), rtl_tip), self.rtl_labels_edit)
        reading_order_layout.addRow(self._form_label(self.tr("Top-to-bottom (TTB) Labels:"), ttb_tip), self.ttb_labels_edit)

        self.layout.addWidget(reading_order_group)

        # --- Labeling Rules --- #
        label_group = QGroupBox(self.tr("Label Merge Rules"))
        label_layout = QFormLayout(label_group)
        label_layout.setSpacing(4)
        label_layout.setContentsMargins(8, 6, 8, 6)
        label_layout.addRow(self._hint_label(self.tr(
            "Limit which detected region labels may merge and decide how the merged label is named."
        )))

        self.label_merge_strategy = QComboBox()
        for text, data in self.label_strategy_map.items():
            self.label_merge_strategy.addItem(text, userData=data)
        label_strategy_tip = self.tr("Selects which label is kept when multiple labeled regions become one region.")
        self._set_tooltip(self.label_merge_strategy, label_strategy_tip)
        label_layout.addRow(self._form_label(self.tr("Label Merge Strategy:"), label_strategy_tip), self.label_merge_strategy)

        self.enable_exclude_labels = QCheckBox(self.tr("Enable labels excluded from merging (blacklist)"))
        self.enable_exclude_labels.setChecked(True)
        self.enable_exclude_labels.setToolTip(self.tr("When enabled, regions with blacklisted labels are never merged."))
        label_layout.addRow(self.enable_exclude_labels)

        self.exclude_labels = QLineEdit()
        self.exclude_labels.setText("other")
        self.exclude_labels.setPlaceholderText(self.tr("Example: label1,label2"))
        blacklist_tip = self.tr("Comma-separated labels to leave untouched, even when their geometry matches the merge rules.")
        self._set_tooltip(self.exclude_labels, blacklist_tip)
        label_layout.addRow(self._form_label(self.tr("Blacklisted Labels:"), blacklist_tip), self.exclude_labels)

        self.enable_exclude_labels.toggled.connect(self.exclude_labels.setEnabled)

        self.require_same_label = QCheckBox(self.tr("Require exactly matching labels to merge"))
        self.require_same_label.setToolTip(self.tr("Only merge regions when their labels are identical. Disabled while specific label groups are active."))
        label_layout.addRow(self.require_same_label)

        self.use_specific_groups = QCheckBox(self.tr("Merge only within specific label groups"))
        self.use_specific_groups.setToolTip(self.tr("Restricts merging to labels that appear together on the same group line below."))
        self.specific_groups_edit = QPlainTextEdit()
        self.specific_groups_edit.setPlaceholderText(self.tr("One group per line, labels separated by commas\nExample:\nballoon,balloon2\nqipao,qipao2"))
        self.specific_groups_edit.setPlainText("balloon\nqipao\nshuqing\nchangfangtiao\nhengxie")
        self.specific_groups_edit.setMinimumHeight(100)
        self.specific_groups_edit.setMaximumHeight(120)
        self.specific_groups_edit.setEnabled(False)
        self.specific_groups_edit.setToolTip(self.tr("One merge group per line. Labels on different lines will not merge with each other."))
        self.use_specific_groups.toggled.connect(self.specific_groups_edit.setEnabled)
        self.use_specific_groups.toggled.connect(lambda checked: self.require_same_label.setDisabled(checked))

        label_layout.addRow(self.use_specific_groups)
        label_layout.addRow(self.specific_groups_edit)

        self.layout.addWidget(label_group)

        # --- Geometric Rules ---
        geo_group = QGroupBox(self.tr("Geometric Merge Parameters"))
        geo_layout = QFormLayout(geo_group)
        geo_layout.setSpacing(4)
        geo_layout.setContentsMargins(8, 6, 8, 6)
        geo_layout.addRow(self._hint_label(self.tr(
            "Geometry thresholds decide how close regions must be and how much they must overlap before merging."
        )))

        # Vertical merge parameters
        self.max_vertical_gap = QSpinBox()
        self.max_vertical_gap.setRange(0, 1000)
        self.max_vertical_gap.setValue(10)
        self.min_width_overlap_ratio = QSpinBox()
        self.min_width_overlap_ratio.setRange(0, 100)
        self.min_width_overlap_ratio.setValue(90)
        self.min_width_overlap_ratio.setSuffix(" %")

        # Horizontal merge parameters
        self.max_horizontal_gap = QSpinBox()
        self.max_horizontal_gap.setRange(0, 1000)
        self.max_horizontal_gap.setValue(10)
        self.min_height_overlap_ratio = QSpinBox()
        self.min_height_overlap_ratio.setRange(0, 100)
        self.min_height_overlap_ratio.setValue(90)
        self.min_height_overlap_ratio.setSuffix(" %")

        vgap_tip = self.tr("Maximum pixel distance between vertically stacked boxes. Lower values merge only tighter rows.")
        woverlap_tip = self.tr("Minimum left/right overlap required for vertical merging. Higher values require better column alignment.")
        hgap_tip = self.tr("Maximum pixel distance between side-by-side boxes. Lower values merge only closer boxes.")
        hoverlap_tip = self.tr("Minimum top/bottom overlap required for horizontal merging. Higher values require better row alignment.")
        self._set_tooltip(self.max_vertical_gap, vgap_tip)
        self._set_tooltip(self.min_width_overlap_ratio, woverlap_tip)
        self._set_tooltip(self.max_horizontal_gap, hgap_tip)
        self._set_tooltip(self.min_height_overlap_ratio, hoverlap_tip)

        # Add separator and widgets to layout
        geo_layout.addRow(QLabel(self.tr("<b>Vertical Merge (up/down)</b>")))
        geo_layout.addRow(self._form_label(self.tr("Maximum Vertical Gap (pixels):"), vgap_tip), self.max_vertical_gap)
        geo_layout.addRow(self._form_label(self.tr("Minimum Horizontal Overlap Ratio:"), woverlap_tip), self.min_width_overlap_ratio)
        geo_layout.addRow(QLabel(self.tr("<b>Horizontal Merge (left/right)</b>")))
        geo_layout.addRow(self._form_label(self.tr("Maximum Horizontal Gap (pixels):"), hgap_tip), self.max_horizontal_gap)
        geo_layout.addRow(self._form_label(self.tr("Minimum Vertical Overlap Ratio:"), hoverlap_tip), self.min_height_overlap_ratio)

        self.layout.addWidget(geo_group)

        # --- Advanced Options --- #
        advanced_group = QGroupBox(self.tr("Advanced Options"))
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(4)
        advanced_layout.setContentsMargins(8, 6, 8, 6)
        self.allow_negative_gap = QCheckBox(self.tr("Allow negative gaps (overlapping boxes)"))
        self.allow_negative_gap.setChecked(True)
        self.allow_negative_gap.setToolTip(self.tr("Allows already-overlapping boxes to merge. Disable this to merge only boxes separated by a positive gap."))
        advanced_layout.addWidget(self.allow_negative_gap)

        self.layout.addWidget(advanced_group)

        # --- Merge Result Type --- #
        result_type_group = QGroupBox(self.tr("Merge Result Type"))
        result_type_layout = QVBoxLayout(result_type_group)
        result_type_layout.setSpacing(4)
        result_type_layout.setContentsMargins(8, 6, 8, 6)

        self.output_type_group = QButtonGroup(self)
        self.radio_output_rectangle = QRadioButton(self.tr("Merge as Axis-aligned Rectangle"))
        self.radio_output_rotation = QRadioButton(self.tr("Merge as Rotated Rectangle"))
        self.radio_output_rectangle.setToolTip(self.tr("Creates a standard rectangular text region that is aligned to the image axes."))
        self.radio_output_rotation.setToolTip(self.tr("Keeps a rotated rectangle when the merged region should preserve angled text orientation."))

        self.radio_output_rectangle.setChecked(True) # Default to rectangle

        self.output_type_group.addButton(self.radio_output_rectangle, 1)
        self.output_type_group.addButton(self.radio_output_rotation, 2)

        result_type_layout.addWidget(self.radio_output_rectangle)
        result_type_layout.addWidget(self.radio_output_rotation)

        self.layout.addWidget(result_type_group)

        # --- Buttons --- #
        button_layout = QHBoxLayout()
        self.run_current_button = QPushButton(self.tr("Run on Current File"))
        self.run_all_button = QPushButton(self.tr("Run on All Files"))
        self.cancel_button = QPushButton(self.tr("Cancel"))
        self.run_current_button.setToolTip(self.tr("Apply these merge settings only to the currently open file."))
        self.run_all_button.setToolTip(self.tr("Apply these merge settings to every file in the current project."))

        button_layout.addWidget(self.run_current_button)
        button_layout.addWidget(self.run_all_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()

        self.run_current_button.clicked.connect(self.on_run_current)
        self.run_all_button.clicked.connect(self.on_run_all)
        self.cancel_button.clicked.connect(self.reject)

        self.layout.addLayout(button_layout)

    def on_run_current(self):
        self.run_current_clicked.emit()

    def on_run_all(self):
        self.run_all_clicked.emit()

    def get_config(self):
        config = {}
        config["MERGE_MODE"] = self.merge_mode.currentData()
        # Set a default reading direction, as the UI for a global default has been removed.
        # The logic in merger.py uses this as a fallback.
        config["READING_DIRECTION"] = "LTR"

        # Parse per-label directions from the new QLineEdits
        per_label_directions = {}
        for label in [l.strip() for l in self.ltr_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'LTR'
        for label in [l.strip() for l in self.rtl_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'RTL'
        for label in [l.strip() for l in self.ttb_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'TTB'
        config["PER_LABEL_DIRECTIONS"] = per_label_directions

        if self.enable_exclude_labels.isChecked():
            excluded = self.exclude_labels.text().strip()
            config["LABELS_TO_EXCLUDE_FROM_MERGE"] = set(l.strip() for l in excluded.split(",") if l.strip())
        else:
            config["LABELS_TO_EXCLUDE_FROM_MERGE"] = set()

        config["USE_SPECIFIC_MERGE_GROUPS"] = self.use_specific_groups.isChecked()
        if config["USE_SPECIFIC_MERGE_GROUPS"]:
            groups_text = self.specific_groups_edit.toPlainText().strip()
            groups = []
            for line in groups_text.split('\n'):
                if line.strip():
                    groups.append([l.strip() for l in line.split(',')])
            config["SPECIFIC_MERGE_GROUPS"] = groups
            config["REQUIRE_SAME_LABEL"] = False # This is disabled in UI
        else:
            config["SPECIFIC_MERGE_GROUPS"] = []
            config["REQUIRE_SAME_LABEL"] = self.require_same_label.isChecked()

        config["LABEL_MERGE_STRATEGY"] = self.label_merge_strategy.currentData()

        config["VERTICAL_MERGE_PARAMS"] = {
            "max_vertical_gap": self.max_vertical_gap.value(),
            "min_width_overlap_ratio": self.min_width_overlap_ratio.value(),
            "overlap_epsilon": 1e-6
        }

        config["HORIZONTAL_MERGE_PARAMS"] = {
            "max_horizontal_gap": self.max_horizontal_gap.value(),
            "min_height_overlap_ratio": self.min_height_overlap_ratio.value(),
            "overlap_epsilon": 1e-6
        }

        config["ADVANCED_MERGE_OPTIONS"] = {
            "allow_negative_gap": self.allow_negative_gap.isChecked(),
            "debug_mode": False # Not exposed in UI
        }

        config["OUTPUT_SHAPE_TYPE"] = "rectangle" if self.output_type_group.checkedId() == 1 else "rotation"

        return config
