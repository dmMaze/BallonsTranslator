"""Navigator widget for image navigation - adapted from X-AnyLabeling"""

from typing import List, Optional, Any

from qtpy.QtCore import Qt, Signal, QRect, QPoint, QSize, QTimer
from qtpy.QtGui import QPainter, QPen, QBrush, QPixmap, QColor, QMouseEvent
from qtpy.QtWidgets import (QWidget, QSizePolicy, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QDialog, QComboBox,
                             QMenu, QCheckBox, QSpinBox, QColorDialog, QGroupBox,
                             QFormLayout)
from qtpy import QtCore, QtWidgets


class NavigatorSettingsDialog(QDialog):
    """导航器设置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导航器设置")
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setFixedSize(280, 250)
        
        layout = QVBoxLayout(self)
        
        # 鼠标指示器设置组
        mouse_group = QGroupBox("鼠标位置指示器")
        mouse_layout = QFormLayout(mouse_group)
        
        # 显示开关
        self.show_indicator_cb = QCheckBox("显示鼠标位置")
        self.show_indicator_cb.setChecked(True)
        mouse_layout.addRow(self.show_indicator_cb)
        
        # 大小设置
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(1, 20)
        self.size_spinbox.setValue(4)
        self.size_spinbox.setSuffix(" px")
        mouse_layout.addRow("指示器大小:", self.size_spinbox)
        
        # 颜色设置
        color_layout = QHBoxLayout()
        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(60, 24)
        self._indicator_color = QColor(255, 0, 0, 255)
        self._update_color_button()
        # 不在这里连接，由 _open_settings_dialog 连接
        color_layout.addWidget(self.color_btn)
        color_layout.addStretch()
        mouse_layout.addRow("指示器颜色:", color_layout)
        
        layout.addWidget(mouse_group)
        
        # 视口设置组
        viewport_group = QGroupBox("视口框")
        viewport_layout = QFormLayout(viewport_group)
        
        self.show_cross_cb = QCheckBox("显示对角线")
        self.show_cross_cb.setChecked(False)
        viewport_layout.addRow(self.show_cross_cb)
        
        # 视口框颜色设置
        viewport_color_layout = QHBoxLayout()
        self.viewport_color_btn = QPushButton()
        self.viewport_color_btn.setFixedSize(60, 24)
        self._viewport_color = QColor(255, 0, 0, 255)
        self._update_viewport_color_button()
        # 不在这里连接，由 _open_settings_dialog 连接
        viewport_color_layout.addWidget(self.viewport_color_btn)
        viewport_color_layout.addStretch()
        viewport_layout.addRow("框体颜色:", viewport_color_layout)
        
        layout.addWidget(viewport_group)
        
        layout.addStretch()
        
    def _update_color_button(self):
        """更新颜色按钮的背景色"""
        self.color_btn.setStyleSheet(
            f"background-color: {self._indicator_color.name()}; border: 1px solid #888;"
        )

    def _update_viewport_color_button(self):
        """更新视口框颜色按钮的背景色"""
        self.viewport_color_btn.setStyleSheet(
            f"background-color: {self._viewport_color.name()}; border: 1px solid #888;"
        )
        
    def _choose_color(self):
        """打开颜色选择对话框"""
        color = QColorDialog.getColor(self._indicator_color, self, "选择指示器颜色")
        if color.isValid():
            self._indicator_color = color
            self._update_color_button()

    def _choose_viewport_color(self):
        """打开视口框颜色选择对话框"""
        color = QColorDialog.getColor(self._viewport_color, self, "选择视口框颜色")
        if color.isValid():
            self._viewport_color = color
            self._update_viewport_color_button()
            
    def get_indicator_color(self) -> QColor:
        return self._indicator_color
    
    def set_indicator_color(self, color: QColor):
        self._indicator_color = color
        self._update_color_button()

    def get_viewport_color(self) -> QColor:
        return self._viewport_color
    
    def set_viewport_color(self, color: QColor):
        self._viewport_color = color
        self._update_viewport_color_button()


class ClickableSlider(QSlider):
    """Custom slider that supports clicking anywhere on the track to jump to position"""
    
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for click-to-jump functionality"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.orientation() == Qt.Orientation.Horizontal:
                handle_width = self.style().pixelMetric(self.style().PM_SliderThickness)
                slider_min = self.minimum()
                slider_max = self.maximum()
                current_value = self.value()
                slider_width = self.width() - handle_width
                
                if slider_max > slider_min:
                    handle_ratio = (current_value - slider_min) / (slider_max - slider_min)
                    handle_pos = handle_width // 2 + handle_ratio * slider_width
                    
                    click_x = event.x()
                    if abs(click_x - handle_pos) <= handle_width // 2 + 5:
                        super().mousePressEvent(event)
                        return
                
                click_x = event.x()
                effective_x = max(handle_width // 2, min(slider_width + handle_width // 2, click_x))
                ratio = (effective_x - handle_width // 2) / slider_width
                new_value = slider_min + ratio * (slider_max - slider_min)
                new_value = max(slider_min, min(slider_max, int(new_value)))
                self.setValue(new_value)
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)


class NavigatorWidget(QWidget):
    """Navigator widget showing thumbnail with viewport rectangle"""
    
    navigation_requested = Signal(float, float)  # x_ratio, y_ratio
    viewport_update_needed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWindowTitle("导航器")
        
        self.original_image: QPixmap = None
        self.thumbnail: QPixmap = None
        self.viewport_rect = QRect()
        self.image_rect = QRect()
        
        self.dragging = False
        self.last_drag_pos = QPoint()
        
        self.viewport_pen = QPen(QColor(255, 0, 0, 255), 2)
        self.background_brush = QBrush(QColor(64, 64, 64))
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.show_viewport_cross = False
        
        # 鼠标位置指示器（图像坐标）
        self.canvas_mouse_pos = None  # QPointF in image coordinates
        self.mouse_indicator_color = QColor(255, 0, 0, 255)  # 红色
        self.mouse_indicator_size = 4  # 指示器大小（圆点半径）
        self.show_mouse_indicator = True  # 是否显示鼠标位置指示器
        
    def set_viewport_cross(self, show: bool):
        """设置是否显示视口框对角线"""
        self.show_viewport_cross = show
        self.update()

    def set_viewport_color(self, color: QColor):
        """设置视口框颜色"""
        self.viewport_pen = QPen(color, self.viewport_pen.width())
        self.update()

    def get_viewport_color(self) -> QColor:
        """获取视口框颜色"""
        return self.viewport_pen.color()

    def set_canvas_mouse_pos(self, pos):
        """设置画布上的鼠标位置（图像坐标）
        
        Args:
            pos: QPointF 图像坐标系中的鼠标位置，或 None 清除指示器
        """
        self.canvas_mouse_pos = pos
        self.update()

    def set_mouse_indicator_visible(self, visible: bool):
        """设置是否显示鼠标位置指示器"""
        self.show_mouse_indicator = visible
        self.update()

    def set_mouse_indicator_color(self, color: QColor):
        """设置鼠标位置指示器颜色"""
        self.mouse_indicator_color = color
        self.update()

    def set_mouse_indicator_size(self, size: int):
        """设置鼠标位置指示器大小（圆点半径）"""
        self.mouse_indicator_size = size
        self.update()

    def set_image(self, image_data: Any) -> None:
        """Set the image to display in the navigator widget."""
        if image_data is None:
            self.original_image = None
            self.thumbnail = None
            self.update()
            return
            
        if isinstance(image_data, bytes):
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            self.original_image = pixmap
        elif isinstance(image_data, QPixmap):
            self.original_image = image_data
        else:
            try:
                self.original_image = QPixmap(str(image_data))
            except:
                return
                
        self._update_thumbnail()
        self.update()
        
    def _update_thumbnail(self):
        """Update thumbnail to fit widget size"""
        if not self.original_image or self.original_image.isNull():
            return
            
        widget_size = self.size()
        available_size = QSize(widget_size.width() - 4, widget_size.height() - 4)
        
        self.thumbnail = self.original_image.scaled(
            available_size, 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        thumb_size = self.thumbnail.size()
        x = (widget_size.width() - thumb_size.width()) // 2
        y = (widget_size.height() - thumb_size.height()) // 2
        self.image_rect = QRect(x, y, thumb_size.width(), thumb_size.height())
        
    def set_viewport(self, x_ratio: float, y_ratio: float, width_ratio: float, height_ratio: float) -> None:
        """Set the viewport rectangle that shows the visible area of the main canvas."""
        if not self.thumbnail or self.image_rect.isEmpty():
            return
            
        thumb_width = self.image_rect.width()
        thumb_height = self.image_rect.height()
        
        x = int(self.image_rect.x() + x_ratio * thumb_width)
        y = int(self.image_rect.y() + y_ratio * thumb_height)
        width = max(1, int(width_ratio * thumb_width))
        height = max(1, int(height_ratio * thumb_height))
        
        self.viewport_rect = QRect(x, y, width, height)
        self.update()
        
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_thumbnail()
        self.update()
        self.viewport_update_needed.emit()
        
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        painter.fillRect(self.rect(), self.background_brush)
        
        if self.thumbnail and not self.thumbnail.isNull():
            painter.drawPixmap(self.image_rect, self.thumbnail)
            
            if not self.viewport_rect.isEmpty():
                painter.setPen(self.viewport_pen)
                painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
                painter.drawRect(self.viewport_rect)
                
                if self.show_viewport_cross:
                    painter.drawLine(
                        self.viewport_rect.topLeft(),
                        self.viewport_rect.bottomRight()
                    )
                    painter.drawLine(
                        self.viewport_rect.topRight(),
                        self.viewport_rect.bottomLeft()
                    )
            
            # 绘制鼠标位置指示器
            self._draw_mouse_indicator(painter)
                
    def _draw_mouse_indicator(self, painter):
        """绘制鼠标位置指示器（红色圆点）"""
        if not self.show_mouse_indicator or self.canvas_mouse_pos is None:
            return
        
        if not self.original_image or self.original_image.isNull():
            return
        
        if self.image_rect.isEmpty():
            return
        
        # 获取原始图像尺寸
        original_width = self.original_image.width()
        original_height = self.original_image.height()
        
        if original_width <= 0 or original_height <= 0:
            return
        
        # 将图像坐标转换为缩略图坐标
        thumb_x = self.image_rect.x() + (self.canvas_mouse_pos.x() / original_width) * self.image_rect.width()
        thumb_y = self.image_rect.y() + (self.canvas_mouse_pos.y() / original_height) * self.image_rect.height()
        
        # 检查点是否在缩略图区域内
        if not self.image_rect.contains(int(thumb_x), int(thumb_y)):
            return
        
        # 绘制红色填充圆点
        painter.setPen(QPen(self.mouse_indicator_color, 1))
        painter.setBrush(QBrush(self.mouse_indicator_color))
        painter.drawEllipse(
            int(thumb_x - self.mouse_indicator_size),
            int(thumb_y - self.mouse_indicator_size),
            self.mouse_indicator_size * 2,
            self.mouse_indicator_size * 2
        )
                
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键弹出菜单
            self._show_context_menu(event.globalPos())
            return
        if event.button() == Qt.MouseButton.LeftButton and self.image_rect.contains(event.pos()):
            self.dragging = True
            self.last_drag_pos = event.pos()
            self._emit_navigation_signal(event.pos())
            
    def mouseMoveEvent(self, event) -> None:
        if self.dragging and self.image_rect.contains(event.pos()):
            self._emit_navigation_signal(event.pos())
            self.last_drag_pos = event.pos()
            
    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            
    def wheelEvent(self, event) -> None:
        if hasattr(self.parent(), 'handle_wheel_zoom'):
            self.parent().handle_wheel_zoom(event)
        else:
            event.accept()
            
    def _emit_navigation_signal(self, pos):
        """Emit navigation signal with position ratios"""
        if self.image_rect.isEmpty():
            return
            
        relative_x = pos.x() - self.image_rect.x()
        relative_y = pos.y() - self.image_rect.y()
        
        x_ratio = max(0.0, min(1.0, relative_x / self.image_rect.width()))
        y_ratio = max(0.0, min(1.0, relative_y / self.image_rect.height()))
        
        self.navigation_requested.emit(x_ratio, y_ratio)

    def _show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        
        # 设置选项
        settings_action = menu.addAction("设置...")
        settings_action.triggered.connect(self._open_settings_dialog)
        
        menu.exec_(pos)
        
    def _open_settings_dialog(self):
        """打开设置对话框"""
        # 获取或创建设置对话框
        parent_dialog = self.parent()
        if hasattr(parent_dialog, '_settings_dialog') and parent_dialog._settings_dialog is not None:
            dialog = parent_dialog._settings_dialog
        else:
            dialog = NavigatorSettingsDialog(self)
            if hasattr(parent_dialog, '_settings_dialog'):
                parent_dialog._settings_dialog = dialog
        
        # 同步当前设置到对话框
        dialog.show_indicator_cb.setChecked(self.show_mouse_indicator)
        dialog.size_spinbox.setValue(self.mouse_indicator_size)
        dialog.set_indicator_color(self.mouse_indicator_color)
        dialog.show_cross_cb.setChecked(self.show_viewport_cross)
        dialog.set_viewport_color(self.get_viewport_color())
        
        # 断开之前的连接（避免重复连接）
        try:
            dialog.show_indicator_cb.toggled.disconnect()
            dialog.size_spinbox.valueChanged.disconnect()
            dialog.show_cross_cb.toggled.disconnect()
            dialog.color_btn.clicked.disconnect()
            dialog.viewport_color_btn.clicked.disconnect()
        except:
            pass
        
        # 连接信号（实时更新）
        dialog.show_indicator_cb.toggled.connect(self.set_mouse_indicator_visible)
        dialog.size_spinbox.valueChanged.connect(self.set_mouse_indicator_size)
        dialog.show_cross_cb.toggled.connect(self.set_viewport_cross)
        
        # 指示器颜色变化
        def on_indicator_color_changed():
            self.set_mouse_indicator_color(dialog.get_indicator_color())
        
        def choose_and_apply_indicator():
            dialog._choose_color()
            on_indicator_color_changed()
        dialog.color_btn.clicked.connect(choose_and_apply_indicator)
        
        # 视口框颜色变化
        def on_viewport_color_changed():
            self.set_viewport_color(dialog.get_viewport_color())
        
        def choose_and_apply_viewport():
            dialog._choose_viewport_color()
            on_viewport_color_changed()
        dialog.viewport_color_btn.clicked.connect(choose_and_apply_viewport)
        
        dialog.show()
        dialog.raise_()


class NavigatorDialog(QDialog):
    """Standalone navigator window with zoom controls"""
    
    zoom_changed = Signal(int)
    zoom_at_point = Signal(int, float, float)  # zoom_percentage, x_ratio, y_ratio
    viewport_update_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.base_title = "导航器"
        self.setWindowTitle(self.base_title)
        self.setWindowFlags(
            Qt.WindowType.Tool |
            Qt.WindowType.WindowCloseButtonHint
        )
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        self.navigator = NavigatorWidget(self)
        main_layout.addWidget(self.navigator, 1)
        
        self.navigator.viewport_update_needed.connect(
            self.viewport_update_requested.emit
        )
        
        zoom_container = QWidget()
        zoom_container.setFixedHeight(60)
        zoom_layout = QVBoxLayout(zoom_container)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(3)
        
        percentage_layout = QHBoxLayout()
        percentage_layout.setContentsMargins(0, 0, 0, 0)
        
        self.file_info_label = QLabel()
        self.file_info_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.file_info_label.setStyleSheet("""
            QLabel { 
                color: #000000; 
                font-size: 11px; 
                font-weight: bold;
                background: transparent;
                padding: 2px 4px;
            }
        """)
        self.file_info_label.setMinimumWidth(80)
        
        self.zoom_input = QComboBox()
        self.zoom_input.setEditable(True)
        self.zoom_input.setFixedWidth(60)
        self.zoom_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        
        presets = [str(i) for i in range(25, 501, 25)]
        self.zoom_input.addItems(presets)
        self.zoom_input.setCurrentText("100")
        
        self.zoom_input.lineEdit().returnPressed.connect(self.on_zoom_input_changed)
        self.zoom_input.activated.connect(self.on_zoom_preset_selected)
        
        percentage_label = QLabel("%")
        percentage_label.setFixedWidth(15)
        
        percentage_layout.addWidget(self.file_info_label)
        percentage_layout.addStretch()
        percentage_layout.addWidget(self.zoom_input)
        percentage_layout.addWidget(percentage_label)
        
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedSize(24, 24)
        zoom_out_btn.clicked.connect(self.zoom_out)
        
        self.zoom_slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(1, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoom_slider.setTickInterval(100)
        self.zoom_slider.valueChanged.connect(self.on_slider_changed)
        
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedSize(24, 24)
        zoom_in_btn.clicked.connect(self.zoom_in)
        
        slider_layout.addWidget(zoom_out_btn)
        slider_layout.addWidget(self.zoom_slider)
        slider_layout.addWidget(zoom_in_btn)
        
        zoom_layout.addLayout(percentage_layout)
        zoom_layout.addLayout(slider_layout)
        
        main_layout.addWidget(zoom_container, 0)
        
        self.setLayout(main_layout)
        
        self.resize(240, 300)
        self.setMinimumSize(180, 220)
        
        self.current_zoom = 100
        
        # 保存的位置和大小
        self._saved_geometry = None
        
        # 设置对话框
        self._settings_dialog = None
        
    def load_config(self):
        """从配置加载导航器设置"""
        from utils.config import pcfg
        
        # 加载窗口位置和大小
        if pcfg.navigator_x >= 0 and pcfg.navigator_y >= 0:
            self.move(pcfg.navigator_x, pcfg.navigator_y)
        self.resize(pcfg.navigator_width, pcfg.navigator_height)
        
        # 加载鼠标指示器设置
        self.navigator.show_mouse_indicator = pcfg.navigator_show_mouse_indicator
        self.navigator.mouse_indicator_size = pcfg.navigator_mouse_indicator_size
        color = pcfg.navigator_mouse_indicator_color
        self.navigator.mouse_indicator_color = QColor(color[0], color[1], color[2], color[3] if len(color) > 3 else 255)
        self.navigator.show_viewport_cross = pcfg.navigator_show_viewport_cross
        
        # 加载视口框颜色
        vp_color = pcfg.navigator_viewport_color
        self.navigator.set_viewport_color(QColor(vp_color[0], vp_color[1], vp_color[2], vp_color[3] if len(vp_color) > 3 else 255))
        
    def save_config(self):
        """保存导航器设置到配置"""
        from utils.config import pcfg, save_config
        
        # 保存窗口位置和大小
        pos = self.pos()
        size = self.size()
        pcfg.navigator_x = pos.x()
        pcfg.navigator_y = pos.y()
        pcfg.navigator_width = size.width()
        pcfg.navigator_height = size.height()
        
        # 保存鼠标指示器设置
        pcfg.navigator_show_mouse_indicator = self.navigator.show_mouse_indicator
        pcfg.navigator_mouse_indicator_size = self.navigator.mouse_indicator_size
        color = self.navigator.mouse_indicator_color
        pcfg.navigator_mouse_indicator_color = [color.red(), color.green(), color.blue(), color.alpha()]
        pcfg.navigator_show_viewport_cross = self.navigator.show_viewport_cross
        
        # 保存视口框颜色
        vp_color = self.navigator.get_viewport_color()
        pcfg.navigator_viewport_color = [vp_color.red(), vp_color.green(), vp_color.blue(), vp_color.alpha()]
        
        save_config()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.viewport_update_requested.emit()

    def moveEvent(self, event):
        """窗口移动时保存位置"""
        super().moveEvent(event)

    def hideEvent(self, event):
        """关闭时保存窗口位置和大小"""
        self._saved_geometry = self.geometry()
        self.save_config()
        super().hideEvent(event)

    def showEvent(self, event):
        """显示时恢复窗口位置和大小"""
        super().showEvent(event)
        if self._saved_geometry is not None:
            self.setGeometry(self._saved_geometry)

    def set_image(self, image_data):
        """Set image in navigator"""
        self.navigator.set_image(image_data)
        self._update_title()

    def _update_title(self):
        """Update the full window title with resolution info."""
        new_title = self.base_title
        if self.navigator.original_image and not self.navigator.original_image.isNull():
            image = self.navigator.original_image
            img_width = image.width()
            img_height = image.height()
            new_title = f"{new_title} - {img_width}x{img_height}"
        self.setWindowTitle(new_title)
        
    def set_viewport(self, x_ratio, y_ratio, width_ratio, height_ratio):
        """Set viewport in navigator"""
        self.navigator.set_viewport(x_ratio, y_ratio, width_ratio, height_ratio)

    def set_canvas_mouse_pos(self, pos):
        """设置画布上的鼠标位置（图像坐标）"""
        self.navigator.set_canvas_mouse_pos(pos)

    def set_mouse_indicator_visible(self, visible: bool):
        """设置是否显示鼠标位置指示器"""
        self.navigator.set_mouse_indicator_visible(visible)

    def set_mouse_indicator_color(self, color: QColor):
        """设置鼠标位置指示器颜色"""
        self.navigator.set_mouse_indicator_color(color)

    def set_mouse_indicator_size(self, size: int):
        """设置鼠标位置指示器大小（圆点半径）"""
        self.navigator.set_mouse_indicator_size(size)

    def set_zoom_value(self, zoom_percentage: int) -> None:
        """Set the zoom value from external sources like the main canvas."""
        self.current_zoom = zoom_percentage
        
        self.zoom_slider.blockSignals(True)
        self.zoom_input.blockSignals(True)
        
        self.zoom_slider.setValue(zoom_percentage)
        self.zoom_input.setCurrentText(str(zoom_percentage))
        
        self.zoom_slider.blockSignals(False)
        self.zoom_input.blockSignals(False)
        
    def on_slider_changed(self, value):
        """Handle slider value change"""
        self.current_zoom = value
        self.zoom_input.setCurrentText(str(value))
        self.zoom_changed.emit(value)
        
    def on_zoom_input_changed(self):
        """Handle zoom input change from QComboBox's line edit"""
        self.apply_zoom_value(self.zoom_input.currentText())

    def on_zoom_preset_selected(self, index):
        """Handle zoom preset selection from dropdown"""
        text = self.zoom_input.itemText(index)
        self.apply_zoom_value(text)

    def apply_zoom_value(self, text):
        """Apply a zoom value from a text string."""
        try:
            value = int(text)
            value = max(1, min(1000, value))
            self.current_zoom = value
            self.zoom_slider.setValue(value)
            self.zoom_input.setCurrentText(str(value))
            self.zoom_changed.emit(value)
        except ValueError:
            self.zoom_input.setCurrentText(str(self.current_zoom))
            
    def zoom_in(self):
        """Zoom in by 10%"""
        new_zoom = min(1000, self.current_zoom + 10)
        self.set_zoom_value(new_zoom)
        self.zoom_changed.emit(new_zoom)
        
    def zoom_out(self):
        """Zoom out by 10%"""
        new_zoom = max(1, self.current_zoom - 10)
        self.set_zoom_value(new_zoom)
        self.zoom_changed.emit(new_zoom)
        
    def handle_wheel_zoom(self, event) -> None:
        """Handle mouse wheel events for zoom control.
        
        普通滚轮: +/- 1%
        Ctrl+滚轮: +/- 5%
        """
        delta = event.angleDelta().y()
        
        # 检查是否按住 Ctrl 键
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            zoom_step = 5
        else:
            zoom_step = 1
        
        if delta > 0:
            zoom_increment = zoom_step
        elif delta < 0:
            zoom_increment = -zoom_step
        else:
            zoom_increment = 0
        
        new_zoom = self.current_zoom + zoom_increment
        new_zoom = max(1, min(1000, new_zoom))
        
        self.set_zoom_value(new_zoom)
        
        # 获取鼠标在导航器缩略图上的位置比例
        mouse_pos = event.position().toPoint() if hasattr(event, 'position') else event.pos()
        nav_widget = self.navigator
        if nav_widget.image_rect.contains(mouse_pos):
            # 计算鼠标相对于图像的比例位置
            rel_x = mouse_pos.x() - nav_widget.image_rect.x()
            rel_y = mouse_pos.y() - nav_widget.image_rect.y()
            x_ratio = rel_x / nav_widget.image_rect.width()
            y_ratio = rel_y / nav_widget.image_rect.height()
            # 发射带位置的缩放信号
            self.zoom_at_point.emit(new_zoom, x_ratio, y_ratio)
        else:
            # 鼠标不在图像区域，使用普通缩放
            self.zoom_changed.emit(new_zoom)
        
        event.accept()
