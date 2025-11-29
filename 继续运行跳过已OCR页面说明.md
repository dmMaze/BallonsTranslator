# 继续运行跳过已OCR页面功能说明

## 问题描述
之前的"继续运行"功能虽然可以跳过已有文本的文本块，但是在检测阶段仍然会重新检测所有页面，导致已经OCR完成的页面也会被重新处理。

例如：OCR了10张图片后停止，点击"继续运行"应该从第11张开始，但实际上会从第1张重新开始检测。

## 解决方案
修改代码，让"继续运行"模式真正跳过已经完成OCR的页面，只处理未完成的页面。

## 修改的文件

### 1. BallonsTranslator/ui/mainwindow.py

**修改内容**：
- 在 `on_run_imgtrans()` 方法中，将需要处理的页面列表传递给 `module_manager`

```python
# 传递需要处理的页面列表给module_manager
self.module_manager.runImgtransPipeline(pages_to_process if continue_mode else None)
```

### 2. BallonsTranslator/ui/module_manager.py

**修改内容**：

#### 2.1 ModuleManager.runImgtransPipeline()
- 接收 `pages_to_process` 参数
- 将参数传递给 `ImgtransThread`

```python
def runImgtransPipeline(self, pages_to_process=None):
    # ...
    self.imgtrans_thread.runImgtransPipeline(self.imgtrans_proj, pages_to_process)
```

#### 2.2 ImgtransThread.__init__()
- 添加 `pages_to_process` 属性

```python
self.pages_to_process = None  # 需要处理的页面列表（用于继续运行模式）
```

#### 2.3 ImgtransThread.runImgtransPipeline()
- 接收并保存 `pages_to_process` 参数

```python
def runImgtransPipeline(self, imgtrans_proj: ProjImgTrans, pages_to_process=None):
    self.pages_to_process = pages_to_process
    # ...
```

#### 2.4 ImgtransThread._imgtrans_pipeline()
- 在主循环中检查页面是否需要处理
- 如果不需要处理，跳过该页面但更新进度计数器

```python
for imgname in self.imgtrans_proj.pages:
    # 继续模式：跳过不需要处理的页面
    if self.pages_to_process is not None and imgname not in self.pages_to_process:
        # 跳过此页面，但更新计数器以保持进度正确
        if cfg_module.enable_detect:
            self.detect_counter += 1
            self.update_detect_progress.emit(self.detect_counter)
        if cfg_module.enable_ocr:
            self.ocr_counter += 1
            self.update_ocr_progress.emit(self.ocr_counter)
        if cfg_module.enable_translate:
            self.translate_counter += 1
            self.update_translate_progress.emit(self.translate_counter)
        if cfg_module.enable_inpaint:
            self.inpaint_counter += 1
            self.update_inpaint_progress.emit(self.inpaint_counter)
        continue
    
    # 处理页面...
```

- 在低显存翻译模式的循环中也添加相同的跳过逻辑

```python
if cfg_module.enable_translate and low_vram_trans:
    for imgname in self.imgtrans_proj.pages:
        # 继续模式：跳过不需要处理的页面
        if self.pages_to_process is not None and imgname not in self.pages_to_process:
            self.translate_counter += 1
            self.update_translate_progress.emit(self.translate_counter)
            continue
        # 翻译页面...
```

## 工作原理

### 1. 识别需要处理的页面
在 `mainwindow.py` 的 `on_run_imgtrans()` 中：
- 遍历所有页面
- 检查每个页面的文本块
- 如果页面没有文本块，或者有文本块但没有文本，则加入 `pages_to_process` 列表

### 2. 传递页面列表
- 将 `pages_to_process` 列表传递给 `ModuleManager`
- `ModuleManager` 再传递给 `ImgtransThread`

### 3. 跳过已完成页面
在 `_imgtrans_pipeline()` 中：
- 遍历所有页面时，检查当前页面是否在 `pages_to_process` 列表中
- 如果不在列表中，跳过处理，但更新进度计数器
- 这样进度条仍然正确显示，但实际上跳过了已完成的页面

## 使用效果

### 场景示例
1. 项目有20张图片
2. 运行OCR，处理了10张后点击停止
3. 点击"继续运行"
4. 系统识别出前10张已完成，后10张未完成
5. 直接从第11张开始处理，跳过前10张
6. 进度条显示正确（前10张立即标记为完成）

### 优势
- ✅ 真正的断点续传
- ✅ 节省时间，不重复处理
- ✅ 进度显示正确
- ✅ 支持所有处理阶段（检测、OCR、翻译、修复）

## 注意事项

1. **进度计数器**：跳过的页面仍然会更新计数器，确保进度条正确显示
2. **所有阶段**：检测、OCR、翻译、修复阶段都会跳过已完成的页面
3. **低显存模式**：低显存翻译模式也支持跳过功能
4. **页面判断**：根据文本块是否有文本来判断页面是否完成

## 测试建议

1. 创建包含多张图片的测试项目
2. 运行OCR，处理一半后停止
3. 点击"继续运行"
4. 观察是否从停止处继续，而不是从头开始
5. 检查进度条是否正确显示
6. 验证最终结果是否完整
