# Changelogs

### 2022-05-19
[v1.2.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.2.0)发布

1. 支持DeepL翻译器, 感谢[@Snowad14](https://github.com/Snowad14)
2. 增加来自manga-image-translator的新OCR模型, 支持韩语识别
3. 修bug


### 2022-04-17
[v1.1.0](https://github.com/dmMaze/BallonsTranslator/releases/tag/v1.1.0)发布

1. 用qthread存编辑图片, 避免翻页卡顿
2. 图像修复策略优化: 
   - 修复算法和**CPU模式**下的修复模型输入由整张图片改为文本块
   - 可选由程序自动评估当前块是否有必要调用开销大的修复方法, 在设置-图像修复启用/禁用, 启用后纯色背景对话泡将会由计算出的背景色直接填充  
  
    优化后图像修复阶段速度提升至原来的2x-5x不等

3. 添加矩形工具
4. 更多快捷键
5. 修bug

### 2022-04-09
v1.0.0发布