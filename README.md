# BallonTranslator
简体中文 | [English](README_EN.md) | [Русский](doc/README_RU.md) | [日本語](doc/README_JA.md) | [Indonesia](doc/README_ID.md)

深度学习辅助漫画翻译工具，支持一键机翻和简单的图像/文本编辑  

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
界面预览
</p>

# Features
* 一键机翻  
  - 译文回填参考对原文排版的估计，包括颜色，轮廓，角度，朝向，对齐方式等
  - 最后效果取决于文本检测，识别，抹字，机翻四个模块的整体表现  
  - 支持日漫和美漫
  - 英译中，日译英排版已优化，文本布局以提取到的背景泡为参考，中文基于 pkuseg 进行断句，日译中竖排待改善
  
* 图像编辑  
  支持掩膜编辑和修复画笔
  
* 文本编辑  
  - 支持所见即所得地富文本编辑和一些基础排版格式调整、[字体样式预设](https://github.com/dmMaze/BallonsTranslator/pull/311)
  - 支持全文/源文/译文查找替换
  - 支持导入导出 word 文档

* 适用于条漫

# 使用说明

## Windows
如果用 Windows 而且不想自己手动配置环境，而且能正常访问互联网:  
从 [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) 下载 BallonsTranslator_dev_src_with_gitpython.7z，解压并运行 launch_win.bat 启动程序。如果无法自动下载库和模型，手动下载 data 和 ballontrans_pylibs_win.7z 并解压到程序目录下。  
运行 scripts/local_gitpull.bat 获取更新。 

## 运行源码

安装 [Python](https://www.python.org/downloads/release/python-31011) **< 3.12** (别用微软应用商店版) 和 [Git](https://git-scm.com/downloads)

```bash
# 克隆仓库
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# 启动程序
$ python3 launch.py
```

第一次运行会自动安装 torch 等依赖项并下载所需模型和文件，如果模型下载失败，需要手动从 [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) 或 [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) 下载 data 文件夹(或者报错里提到缺失的文件)，并保存到源码目录下的对应位置。

如果要使用Sugoi翻译器(仅日译英)，下载[离线模型](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm)，将 ```sugoi_translator``` 移入 BallonsTranslator/ballontranslator/data/models。 

## 构建 macOS 应用（适用 apple silicon 芯片）
<i>如果构建不成功也可以直接跑源码</i>

![录屏2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

```
# 第1步：打开终端并确保当前终端窗口的Python大版本号是3.12，可以用下面的命令确认版本号
python3 -V
# 如果没有安装Python 3.12，可以通过Homebrew安装
brew install python@3.12 python-tk@3.12

# 第2步：克隆仓库并进入仓库工作目录
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git
cd BallonsTranslator

# 第3步：创建和启用 Python 3.12 虚拟环境
python3 -m venv venv
source venv/bin/activate

# 第4步：安装依赖
pip3 install -r requirements.txt

# 第5步：源码运行程序，会自动下载 data 文件，每个文件在20-400MB左右，合计大约1.67GB，需要比较稳定的网络，如果下载报错，请重复运行下面的命令直至不再下载报错并启动程序
# 下载完毕后运行下面的命令，如果正常运行且未报错，则继续进入打包应用程序的步骤
python3 launch.py

# 第6步：下载macos_arm64_patchmatch_libs.7z到项目根目录下的'.btrans_cache'隐藏文件夹
# 该步骤是为了防止打包好的应用程序首次启动时重新下载macos_arm64_patchmatch_libs.7z导致启动失败（大概率）
mkdir ./.btrans_cache2
curl -L https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z -o ./.btrans_cache/macos_arm64_patchmatch_libs.7z

# 第7步：下载微软雅黑字体并放到fonts文件夹下，该步骤为可选项，不影响打包，只影响字体报错信息

# 第8步：构建 macOS 应用程序中途 sudo 命令需要输入开机密码授予权限
# 安装打包工具pyinstaller
pip3 install pyinstaller
# 删除MacOS下特有的.DS_Store文件，这些文件可能导致打包失败（中概率）
sudo find ./ -name '.DS_Store' -delete
# 开始打包.app应用程序
sudo pyinstaller launch.spec
```
> 📌打包好的应用在`./data/BallonsTranslator/dist/BallonsTranslator.app`，将应用拖到 macOS 的应用程序文件夹即完成安装，开箱即用，不需要另外配置 Python 环境。 

## 一键翻译
**建议在命令行终端下运行程序**，首次运行请先配置好源语言/目标语言，打开一个带图片的文件夹，点击 Run 等待翻译完成  
<img src="doc/src/run.gif">  

一键机翻嵌字格式如大小、颜色等默认是由程序决定的，可以在设置面板->嵌字菜单中改用全局设置。全局字体格式就是未编辑任何文本块时右侧字体面板显示的格式:  
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

按下鼠标左键拖动矩形框抹除框内文字，按下右键拉框清除框内修复结果。  
抹除结果取决于算法(gif 中的"方法1"和"方法2")对文字区域估算的准确程度，一般拉的框最好稍大于需要抹除的文本块。两种方法都比较玄学，能够应付绝大多数简单文字简单背景，部分复杂背景简单文字/简单背景复杂文字，少数复杂背景复杂文字，可以多拉几次试试。  
勾选"自动"拉完框立即修复，否则需要按下"修复"或者空格键才进行修复，或 ```Ctrl+D``` 删除矩形选框。 

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
* Ctrl+Z，Ctrl+Y 可以撤销重做大部分操作，注意翻页后撤消重做栈会清空
* A/D 或 pageUp/Down 翻页，如果当前页面未保存会自动保存
* T 切换到文本编辑模式下(底部最右"T"图标)，W激活文本块创建模式后在画布右键拉文本框
* P 切换到画板模式，右下角滑条改原图透明度
* 标题栏->运行 可以启用/禁用任意自动化模块，全部禁用后Run会根据全局字体样式和嵌字设置重新渲染文本  
* 设置面板配置各自动化模块参数
* Ctrl++/- 或滚轮缩放画布
* Ctrl+A 可选中界面中所有文本块
* Ctrl+F 查找当前页，Ctrl+G全局查找
* 0-9调整嵌字/原图透明度
* 文本编辑下 ```Ctrl+B``` 加粗，```Ctrl+U``` 下划线，```Ctrl+I``` 斜体
* 字体样式面板-"特效"修改透明度添加阴影

<img src="doc/src/configpanel.png">  

## 命令行模式 (无GUI)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
所有设置 (如检测模型, 原语言目标语言等) 会从 config/config.json 导入。  
如果渲染字体大小不对, 通过 ```--ldpi ``` 指定 Logical DPI 大小, 通常为 96 和 72。

# 自动化模块
本项目重度依赖 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)，在线服务器和模型训练需要费用，有条件请考虑支持一下
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

Sugoi 翻译器作者: [mingshiba](https://www.patreon.com/mingshiba)
  
### 文本检测
暂时仅支持日文(方块字都差不多)和英文检测，训练代码和说明见https://github.com/dmMaze/comic-text-detector

### OCR
 * 所有 mit 模型来自 manga-image-translator，支持日英汉识别和颜色提取
 * [manga_ocr](https://github.com/kha-white/manga-ocr) 来自 [kha-white](https://github.com/kha-white)，支持日语识别，注意选用该模型程序不会提取颜色

### 图像修复
  * AOT 修复模型来自 manga-image-translator
  * patchmatch 是非深度学习算法，也是PS修复画笔背后的算法，实现来自 [PyPatchMatch](https://github.com/vacancy/PyPatchMatch)，本程序用的是我的[修改版](https://github.com/dmMaze/PyPatchMatchInpaint)
  * lama* 是微调过的[lama](https://github.com/advimman/lama)
  

### 翻译器

 * 谷歌翻译器已经关闭中国服务，大陆再用需要设置全局代理，并在设置面板把 url 换成*.com
 * 彩云，需要申请 [token](https://dashboard.caiyunapp.com/)
 * papago  
 * DeepL 和 Sugoi (及它的 CT2 Translation 转换)翻译器，感谢 [Snowad14](https://github.com/Snowad14)  
 * 支持 [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame)

 如需添加新的翻译器请参考[加别的翻译器](doc/加别的翻译器.md)，本程序添加新翻译器只需要继承基类实现两个接口即可不需要理会代码其他部分，欢迎大佬提 pr

## 杂
* 电脑带N卡或 Apple silicon 默认启用 GPU 加速
* 感谢 [bropines](https://github.com/bropines) 提供俄语翻译
* 第三方输入法可能会造成右侧编辑框显示 bug，见[#76](https://github.com/dmMaze/BallonsTranslator/issues/76)，暂时不打算修
* 选中文本迷你菜单支持*聚合词典专业划词翻译*[沙拉查词](https://saladict.crimx.com): [安装说明](doc/saladict_chs.md)
