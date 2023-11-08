# BallonTranslator
简体中文 | [English](README_EN.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md)

深度学习辅助漫画翻译工具, 支持一键机翻和简单的图像/文本编辑  

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
界面预览
</p>

# Features
* 一键机翻  
  - 译文回填参考对原文排版的估计, 包括颜色, 轮廓, 角度, 朝向, 对齐方式等
  - 最后效果取决于文本检测, 识别, 抹字, 机翻四个模块的整体表现  
  - 支持日漫和美漫
  - 英译中, 日译英排版已优化, 文本布局以提取到的背景泡为参考, 中文基于pkuseg进行断句, 日译中竖排待改善
  
* 图像编辑  
  支持掩膜编辑和修复画笔
  
* 文本编辑  
  - 支持所见即所得地富文本编辑和一些基础排版格式调整、字体样式预设
  - 支持全文/源文/译文查找替换
  - 支持导入导出word文档

* 适用于条漫

# 使用说明

## Windows
如果用windows而且不想自己手动配置环境, 而且能正常访问互联网:  
从[MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) 下载BallonsTranslator_dev_src_with_gitpython.7z, 解压并运行launch.bat启动程序。如果无法自动下载库和模型，手动下载data和ballontrans_pylibs_win.7z并解压到程序目录下.  
运行scripts/local_gitpull.bat获取更新.  

## 运行源码

安装[Python](https://www.python.org/downloads/release/python-31011) **< 3.12** (别用微软应用商店版) 和[Git](https://git-scm.com/downloads)

```bash
# 克隆仓库
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# 启动程序
$ python3 launch.py
```

第一次运行会自动安装torch等依赖项并下载所需模型和文件, 如果模型下载失败, 需要手动从[MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)下载data文件夹(或者报错里提到缺失的文件), 并保存到源码目录下的对应位置. 

如果要使用Sugoi翻译器(仅日译英), 下载[离线模型](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), 将 "sugoi_translator" 移入BallonsTranslator/ballontranslator/data/models.  

## 构建macOS应用（本方法兼容intel和apple silicon芯片）
<i>如果构建不成功也可以直接跑源码</i>

![录屏2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

#### 通过远程脚本一键构建应用
在终端中输入下面的命令自动完成所有构建步骤，由于从github和huggingface下载模型，需要比较好的网络条件
```
curl -L https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/macos-build-script.sh | bash
```

⚠️ 如果网络条件不佳，需要从网盘下载需要的文件，请按照下面的步骤操作
#### 1、准备工作
-   从[MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)下载`libs`和`models`.

-  将下载的资源全部放入名为`data`文件夹，最后的目录树结构应该如下所示：

```
data
├── libs
│   └── patchmatch_inpaint.dll
└── models
    ├── aot_inpainter.ckpt
    ├── comictextdetector.pt
    ├── comictextdetector.pt.onnx
    ├── lama_mpe.ckpt
    ├── manga-ocr-base
    │   ├── README.md
    │   ├── config.json
    │   ├── preprocessor_config.json
    │   ├── pytorch_model.bin
    │   ├── special_tokens_map.json
    │   ├── tokenizer_config.json
    │   └── vocab.txt
    ├── mit32px_ocr.ckpt
    ├── mit48pxctc_ocr.ckpt
    └── pkuseg
        ├── postag
        │   ├── features.pkl
        │   └── weights.npz
        ├── postag.zip
        └── spacy_ontonotes
            ├── features.msgpack
            └── weights.npz

7 directories, 23 files
```

-  安装pyenv命令行工具，这是用于管理Python版本的工具
```
# 通过Homebrew途径安装
brew install pyenv

# 通过官方自动脚本途径安装
curl https://pyenv.run | bash

# 安装完后需要设置shell环境
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


#### 2、构建应用
```
# 进入`data`工作目录
cd data

# 克隆仓库`dev`分支
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git

# 进入`BallonsTranslator`工作目录
cd BallonsTranslator

# 运行构建脚本，运行到pyinstaller环节会要求输入开机密码，输入密码后按下回车即可
sh scripts/build-macos-app.sh
```
> 📌打包好的应用在`./data/BallonsTranslator/dist/BallonsTranslator.app`，将应用拖到macOS的应用程序文件夹即完成安装，开箱即用，不需要另外配置Python环境.  

## 一键翻译
**建议在命令行终端下运行程序**, 首次运行请先配置好源语言/目标语言, 打开一个带图片的文件夹, 点击Run等待翻译完成  
<img src="doc/src/run.gif">  

一键机翻嵌字格式如大小、颜色等默认是由程序决定的, 可以在设置面板->嵌字菜单中改用全局设置. 全局字体格式就是未编辑任何文本块时右侧字体面板显示的格式:  
<img src="doc/src/global_font_format.png"> 

## 画板

## 修复画笔
<img src="doc/src/imgedit_inpaint.gif">
<p align = "center">
修复画笔
</p>

### 矩形工具
<img src="doc/src/rect_tool.gif">
<p align = "center">
矩形工具
</p>

按下鼠标左键拖动矩形框抹除框内文字, 按下右键拉框清除框内修复结果.  
抹除结果取决于算法(gif中的"方法1"和"方法2")对文字区域估算的准确程度, 一般拉的框最好稍大于需要抹除的文本块. 两种方法都比较玄学, 能够应付绝大多数简单文字简单背景, 部分复杂背景简单文字/简单背景复杂文字, 少数复杂背景复杂文字, 可以多拉几次试试.  
勾选"自动"拉完框立即修复, 否则需要按下"修复"或者空格键才进行修复, 或"Ctrl+D"删除矩形选框.  

## 文本编辑
<img src="doc/src/textedit.gif">


<p align = "center">
文本编辑
</p>

<img src="doc/src/multisel_autolayout.gif" div align=center>
<p align=center>
批量文本格式调整及自动排版
</p>

<img src="doc/src/ocrselected.gif" div align=center>
<p align=center>
OCR并翻译选中文本框
</p>

## 界面说明及快捷键
* Ctrl+Z, Ctrl+Y可以撤销重做大部分操作，注意翻页后撤消重做栈会清空
* A/D或pageUp/Down翻页, 如果当前页面未保存会自动保存
* "T"切换到文本编辑模式下(底部最右"T"图标), W激活文本块创建模式后在画布右键拉文本框
* "P"切换到画板模式, 右下角滑条改原图透明度
* 底部左侧"OCR"和"A"按钮控制启用/禁用OCR翻译功能, 禁用后再Run程序就只做文本检测和抹字  
* 设置面板配置各自动化模块参数
* Ctrl++/-或滚轮缩放画布
* Ctrl+A可选中界面中所有文本块
* Ctrl+F查找当前页, Ctrl+G全局查找
* 0-9调整嵌字/原图透明度

<img src="doc/src/configpanel.png">  

# 自动化模块
本项目重度依赖[manga-image-translator](https://github.com/zyddnys/manga-image-translator), 在线服务器和模型训练需要费用, 有条件请考虑支持一下
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

Sugoi翻译器作者: [mingshiba](https://www.patreon.com/mingshiba).
  
### 文本检测
暂时仅支持日文(方块字都差不多)和英文检测, 训练代码和说明见https://github.com/dmMaze/comic-text-detector

### OCR
 * mit_32px模型来自manga-image-translator, 支持日英汉识别和颜色提取
 * mit48px_ctc模型来自manga-image-translator, 支持日英汉韩语识别和颜色提取
 * [manga_ocr](https://github.com/kha-white/manga-ocr)来自[kha-white](https://github.com/kha-white), 支持日语识别, 注意选用该模型程序不会提取颜色

### 图像修复
  * AOT修复模型来自manga-image-translator
  * patchmatch是非深度学习算法, 也是PS修复画笔背后的算法, 实现来自[PyPatchMatch](https://github.com/vacancy/PyPatchMatch), 本程序用的是我的[修改版](https://github.com/dmMaze/PyPatchMatchInpaint)
  

### 翻译器

 * <s>谷歌翻译能挂代理建议把url从cn改成com</s> 谷歌翻译器已经关闭中国服务, 大陆再用需要设置全局代理, 并在设置面板把url换成*.com
 * 彩云, 需要申请[token](https://dashboard.caiyunapp.com/)
 * papago  
 * DeepL 和 Sugoi(及它的CT2 Translation转换)翻译器, 感谢[Snowad14](https://github.com/Snowad14)  

 如需添加新的翻译器请参考[加别的翻译器](doc/加别的翻译器.md), 本程序添加新翻译器只需要继承基类实现两个接口即可不需要理会代码其他部分, 欢迎大佬提pr

## 杂
* 如果电脑带N卡, 程序默认对所有模型启用GPU加速, 默认配置下显存占用在6G左右. 4G显存调小修复器inpaint_size即可.  
* 感谢[bropines](https://github.com/bropines)提供俄语翻译
* 第三方输入法可能会造成右侧编辑框显示bug, 见[#76](https://github.com/dmMaze/BallonsTranslator/issues/76), 暂时不打算修
* 选中文本迷你菜单支持*聚合词典专业划词翻译*[沙拉查词](https://saladict.crimx.com): [安装说明](doc/saladict_chs.md)
