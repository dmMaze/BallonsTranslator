from qtpy.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QLineEdit, QPlainTextEdit, QCheckBox, QSpinBox, QLabel, QRadioButton, QButtonGroup, QHBoxLayout, QPushButton
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QSizePolicy

class MergeDialog(QDialog):
    # 定义信号：当用户点击运行按钮时发出
    run_current_clicked = Signal()  # 对当前文件运行
    run_all_clicked = Signal()  # 对所有文件运行
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("区域合并工具设置")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.adjustSize()
        # 设置窗口标志：移除帮助按钮,添加最小化按钮
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
            "垂直合并": "VERTICAL",
            "水平合并": "HORIZONTAL",
            "先垂直后水平": "VERTICAL_THEN_HORIZONTAL",
            "先水平后垂直": "HORIZONTAL_THEN_VERTICAL",
            "无": "NONE",
        }
        self.label_strategy_map = {
            "优先使用较短的标签": "PREFER_SHORTER",
            "使用第一个框的标签": "FIRST",
            "组合标签 (label1+label2)": "COMBINE",
            "优先使用非默认标签": "PREFER_NON_DEFAULT",
        }

        # --- Main Settings --- #
        main_group = QGroupBox("主要设置")
        main_layout = QFormLayout(main_group)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(8, 6, 8, 6)

        self.merge_mode = QComboBox()
        for text, data in self.merge_mode_map.items():
            self.merge_mode.addItem(text, userData=data)
        main_layout.addRow("合并模式:", self.merge_mode)
        self.layout.addWidget(main_group)

        # --- Text Reading Order Settings ---
        reading_order_group = QGroupBox("文本合并顺序 (按标签)")
        reading_order_layout = QFormLayout(reading_order_group)
        reading_order_layout.setSpacing(4)
        reading_order_layout.setContentsMargins(8, 6, 8, 6)
        
        self.ltr_labels_edit = QLineEdit()
        self.ltr_labels_edit.setPlaceholderText("标签1,标签2,...")
        self.rtl_labels_edit = QLineEdit()
        self.rtl_labels_edit.setText("balloon,qipao,shuqing")
        self.ttb_labels_edit = QLineEdit()
        self.ttb_labels_edit.setText("changfangtiao,hengxie")

        reading_order_layout.addRow("从左到右 (LTR) 标签:", self.ltr_labels_edit)
        reading_order_layout.addRow("从右到左 (RTL) 标签:", self.rtl_labels_edit)
        reading_order_layout.addRow("从上到下 (TTB) 标签:", self.ttb_labels_edit)
        
        self.layout.addWidget(reading_order_group)

        # --- Labeling Rules --- #
        label_group = QGroupBox("标签合并规则")
        label_layout = QFormLayout(label_group)
        label_layout.setSpacing(4)
        label_layout.setContentsMargins(8, 6, 8, 6)

        self.label_merge_strategy = QComboBox()
        for text, data in self.label_strategy_map.items():
            self.label_merge_strategy.addItem(text, userData=data)
        label_layout.addRow("标签合并策略:", self.label_merge_strategy)

        # 黑名单启用复选框
        self.enable_exclude_labels = QCheckBox("启用排除合并的标签 (黑名单)")
        self.enable_exclude_labels.setChecked(True)  # 默认启用
        label_layout.addRow(self.enable_exclude_labels)
        
        self.exclude_labels = QLineEdit()
        self.exclude_labels.setText("other")  # 默认填入other
        self.exclude_labels.setPlaceholderText("例如: label1,label2")
        label_layout.addRow("黑名单标签:", self.exclude_labels)
        
        # 连接复选框信号，控制输入框的启用状态
        self.enable_exclude_labels.toggled.connect(self.exclude_labels.setEnabled)
        
        self.require_same_label = QCheckBox("要求标签完全相同才合并")
        label_layout.addRow(self.require_same_label)

        self.use_specific_groups = QCheckBox("仅在特定标签组内合并")
        self.specific_groups_edit = QPlainTextEdit()
        self.specific_groups_edit.setPlaceholderText("每行一个分组, 组内标签用逗号分隔\n例如:\nballoon,balloon2\nqipao,qipao2")
        self.specific_groups_edit.setPlainText("balloon\nqipao\nshuqing\nchangfangtiao\nhengxie")
        self.specific_groups_edit.setMinimumHeight(100)
        self.specific_groups_edit.setMaximumHeight(120)
        self.specific_groups_edit.setEnabled(False)
        self.use_specific_groups.toggled.connect(self.specific_groups_edit.setEnabled)
        self.use_specific_groups.toggled.connect(lambda checked: self.require_same_label.setDisabled(checked))

        label_layout.addRow(self.use_specific_groups)
        label_layout.addRow(self.specific_groups_edit)

        self.layout.addWidget(label_group)

        # --- Geometric Rules ---
        geo_group = QGroupBox("几何合并参数")
        geo_layout = QFormLayout(geo_group)
        geo_layout.setSpacing(4)
        geo_layout.setContentsMargins(8, 6, 8, 6)

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

        # Add separator and widgets to layout
        geo_layout.addRow(QLabel("<b>垂直合并 (上下)</b>"))
        geo_layout.addRow("最大垂直间隙 (像素):", self.max_vertical_gap)
        geo_layout.addRow("最小水平重叠比例:", self.min_width_overlap_ratio)
        geo_layout.addRow(QLabel("<b>水平合并 (左右)</b>"))
        geo_layout.addRow("最大水平间隙 (像素):", self.max_horizontal_gap)
        geo_layout.addRow("最小垂直重叠比例:", self.min_height_overlap_ratio)

        self.layout.addWidget(geo_group)

        # --- Advanced Options --- #
        advanced_group = QGroupBox("高级选项")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(4)
        advanced_layout.setContentsMargins(8, 6, 8, 6)
        self.allow_negative_gap = QCheckBox("允许负间隙 (即允许框本身有重叠)")
        self.allow_negative_gap.setChecked(True)
        advanced_layout.addWidget(self.allow_negative_gap)

        self.layout.addWidget(advanced_group)

        # --- Merge Result Type --- #
        result_type_group = QGroupBox("合并结果类型")
        result_type_layout = QVBoxLayout(result_type_group)
        result_type_layout.setSpacing(4)
        result_type_layout.setContentsMargins(8, 6, 8, 6)
        
        self.output_type_group = QButtonGroup(self)
        self.radio_output_rectangle = QRadioButton("合并水平矩形")
        self.radio_output_rotation = QRadioButton("合并旋转矩形")
        
        self.radio_output_rectangle.setChecked(True) # Default to rectangle
        
        self.output_type_group.addButton(self.radio_output_rectangle, 1)
        self.output_type_group.addButton(self.radio_output_rotation, 2)
        
        result_type_layout.addWidget(self.radio_output_rectangle)
        result_type_layout.addWidget(self.radio_output_rotation)
        
        self.layout.addWidget(result_type_group)

        # --- Buttons --- #
        button_layout = QHBoxLayout()
        self.run_current_button = QPushButton("对当前文件运行")
        self.run_all_button = QPushButton("对所有文件运行")
        self.cancel_button = QPushButton("取消")
        
        button_layout.addWidget(self.run_current_button)
        button_layout.addWidget(self.run_all_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        
        # 连接按钮信号
        self.run_current_button.clicked.connect(self.on_run_current)
        self.run_all_button.clicked.connect(self.on_run_all)
        self.cancel_button.clicked.connect(self.reject)
        
        self.layout.addLayout(button_layout)

    def on_run_current(self):
        """对当前文件运行合并"""
        self.run_current_clicked.emit()
        # 不关闭对话框，让用户可以继续调整参数

    def on_run_all(self):
        """对所有文件运行合并"""
        self.run_all_clicked.emit()
        # 不关闭对话框，让用户可以继续调整参数

    def get_config(self):
        """获取用户配置的合并参数"""
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

        # 只有当黑名单启用时才使用排除标签
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
