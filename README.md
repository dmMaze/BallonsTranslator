<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">深度学习辅助漫画翻译工具，支持一键机翻和简单的图像/文本编辑</p>



<p align="center">
  简体中文 | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | <a href="/doc/README_ID.md">Indonesia</a> | <a href="/doc/README_VI.md">Tiếng Việt</a>
</p>




<p align="center">交流反馈 QQ 群：719881337 </p>

 


# Features
> [!IMPORTANT]  
> **如打算公开分享本工具的机翻结果，且没有有经验的译者进行过完整的翻译或校对，请在显眼位置注明机翻。**


* 一键机翻  
  - 译文回填参考对原文排版的估计，包括颜色，轮廓，角度，朝向，对齐方式等
  - 最后效果取决于文本检测，识别，抹字，机翻四个模块的整体表现  
  - 支持日漫和美漫
  - 英译中，日译英排版已优化，文本布局以提取到的背景泡为参考，中文基于 pkuseg 进行断句，日译中竖排待改善
  
* 图像编辑  
  支持掩膜编辑和修复画笔
  
* 文本编辑  
  - 支持所见即所得地富文本编辑和一些基础排版格式调整、[字体样式预设](https://github.com/dmMaze/BallonsTranslator/pull/311)
  - [文本变形](https://github.com/dmMaze/BallonsTranslator/pull/1238)、全文/原文/译文查找替换
  - 支持导入导出 word 文档

* <details>
  <summary><i>支持上下文和术语表的 LLM 翻译</i></summary>

  **翻译历史**

  - 将 **LLM 上下文** 设为 **+翻译历史** 后，`LLMTranslator` 会参考之前已完成的页面，有助于统一人名、术语和语气。继续运行和选定范围也可使用范围之前符合条件的页面。
  - **Token 预算** 控制加入多少较早的译文，并优先保留较新的页面。当前页面、指令、术语表和生成回复还需要额外的上下文空间。默认值为 `4096`。
  - 较大预算可提供更多剧情上下文并减少旧页淘汰，但会发送更多输入，可能需要更长时间。本地模型还可能显著增加内存/显存占用。默认值 `4096` 是特意设置的保守选择；DeepSeek 等具有较大上下文窗口的主流服务通常可以使用更高上限。模型上下文上限的约 70% 可作为合理上限（128K 模型约为 `90000`）。
  - 历史预算也会影响提示词缓存。历史在预算内增长时，连续请求会保留相同的开头，OpenAI、DeepSeek 等服务可按折扣价复用这些输入 token，并可能降低延迟。预算迫使程序淘汰旧页后，公共开头会改变，缓存复用随之重置。较大预算可减少重置次数，但也会发送更多历史，因此总费用不一定更低。

  下表以 DeepSeek 为例，对普通漫画页面进行粗略估算；其缓存输入 token 的价格为普通输入 token 的 10%。实际结果会因项目、模型和服务商而异。

  | 历史预算 (tokens) | 预计保留的翻译历史（页） | 相对不使用历史的预计总费用 |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **可复用术语表**

  - 在运行对话框中设置 **术语表**，可使用 UTF-8 编码的 `.json`、`.txt` 或 `.tsv` 文件。文件只会被读取，并可在多个项目间复用。
  - **仅匹配** 只发送原词出现在相关页面中的条目；**全表** 会发送全部条目，可能明显增加 token 用量。
  - 支持以下格式：

    ```text
    # Sakura 格式文本
    原词->译词 # 可选备注

    # 制表符分隔文本
    原词<TAB>译词<TAB>可选备注
    ```

    ```json
    [
      {"src": "原词", "dst": "译词", "info": "可选备注"}
    ]
    ```

  - 匹配不区分大小写，并按字面文本匹配。条目冲突、文件格式错误、不支持的扩展名或文件不存在时，翻译会在发送 LLM 请求前停止。
  - 历史页面上下文和术语表注入只对 `LLMTranslator` 生效，其他翻译器会忽略这些设置。

  </details>

* 适用于条漫

# 使用说明

## Windows

**方式 A（一键自动配置本地环境，需要系统支持 PowerShell）**：
该脚本会在执行目录安装 `BallonsTranslator`：
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
或者在系统的命令提示符 (`cmd.exe`) 中运行：
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**方式 B（下载免配置压缩包）**：
从 [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases) 下载 `Ballonstranslator_win_minium.zip`，解压并双击运行 `launch_win.bat` 启动程序。  
  
以上方式不支持 Windows 7，Windows 7 用户需要自行安装 [Python 3.8](https://www.python.org/downloads/release/python-3810/) 运行源码。 


如果遇到 `msvcp140.dll`、`c10.dll` 或 `[WinError 1114]` 相关错误，请安装或更新 [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe)（Visual Studio 2015-2022；[官方下载说明](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)）。  

## macOS / Linux

该脚本会在执行目录安装 `BallonsTranslator`：
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

如果系统没有 `curl`，也可以用 `wget -O ...` 下载脚本。安装完成后会自动启动程序；之后可运行 `cd BallonsTranslator && ./launch.sh` 再次启动。  

启动程序会检查核心依赖；选择需要额外库的模块时，程序会提示安装缺失的可选依赖（也可在设置中启用自动安装）。

## 一键翻译
**建议在命令行终端下运行程序**，首次运行请先配置好源语言/目标语言，打开一个带图片的文件夹，点击 Run 等待翻译完成  
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  

一键机翻嵌字格式如大小、颜色等默认是由程序决定的，可以在设置面板->嵌字菜单中改用全局设置。全局字体格式就是未编辑任何文本块时右侧字体面板显示的格式:  
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80"> 

## 画板

## 修复画笔
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
修复画笔
</p>

### 矩形工具
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
矩形工具
</p>

按下鼠标左键拖动矩形框抹除框内文字，按下右键拉框清除框内修复结果。  
抹除结果取决于算法(gif 中的"方法1"和"方法2")对文字区域估算的准确程度，一般拉的框最好稍大于需要抹除的文本块。两种方法都比较玄学，能够应付绝大多数简单文字简单背景，部分复杂背景简单文字/简单背景复杂文字，少数复杂背景复杂文字，可以多拉几次试试。  
勾选"自动"拉完框立即修复，否则需要按下"修复"或者空格键才进行修复，或 ```Ctrl+D``` 删除矩形选框。 

## 文本编辑
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">


<p align = "center">
文本编辑
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
批量文本格式调整及自动排版
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
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
* ```Alt+Arrow Keys``` 或 ```Alt+WASD``` (正在编辑文本块时 ```pageDown``` 或 ```pageUp```) 在文本块间切换

<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">  

## 命令行模式 (无GUI)
``` python
python -m ballontranslator --headless --exec_dirs "[DIR_1],[DIR_2]..."
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
 * 暂时仅支持日文(方块字都差不多)和英文检测，训练代码和说明见https://github.com/dmMaze/comic-text-detector
 * 支持使用 [星河云(团子漫画OCR)](https://cloud.stariver.org.cn/)的文本检测，需要填写用户名和密码，每次启动时会自动登录。
   * 详细说明见 [团子OCR说明](doc/团子OCR说明.md)
 * `YSGDetector` 是由 [lhj5426](https://github.com/lhj5426) 训练的模型，能更好地过滤日漫/CG里的拟声词。需要手动从 [YSGYoloDetector](https://huggingface.co/YSGforMTL/YSGYoloDetector) 下载模型放到 data/models 目录下。


### OCR
 * 所有 mit 模型来自 manga-image-translator，支持日英汉识别和颜色提取
 * [manga_ocr](https://github.com/kha-white/manga-ocr) 来自 [kha-white](https://github.com/kha-white)，支持日语识别，注意选用该模型程序不会提取颜色
 * [PaddleOCRVLManga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga) 支持日语识别，选用该模型程序不会提取颜色
 * 支持使用 [星河云(团子漫画OCR)](https://cloud.stariver.org.cn/)的OCR，需要填写用户名和密码，每次启动时会自动登录。
   * 目前的实现方案是逐个textblock进行OCR，速度较慢，准确度没有明显提升，不推荐使用。如果有需要，请使用团子Detector。
   * 推荐文本检测设置为团子Detector时，将OCR设为none_ocr，直接读取文本，节省时间和请求次数。
   * 详细说明见 [团子OCR说明](doc/团子OCR说明.md)
 * OCR设置项: 字体识别。把[字体识别模型（YuzuMarker.FontDetection）](https://github.com/JeffersonQin/YuzuMarker.FontDetection)下载下来放在data\models\YuzuMarker.FontDetection目录下。
  需要的三个文件分别是```data\models\YuzuMarker.FontDetection\font_dataset``` ，  ```data\models\YuzuMarker.FontDetection\name=4x-epoch=18-step=368676.ckpt```，  ```data\font_demo_cache.bin```  
  识别到的置信率大于60%的字体名称会保存在json文件的```_detected_font_name```字段中。目前没做可视化外显，使用[脚本](scripts/BTjson_to_LPtxt.pyw)导出LabelPlus txt时可选带上字体字号信息，导入到其他软件（如PS/ID）嵌字用。

### 图像修复
  * AOT 修复模型来自 manga-image-translator
  * patchmatch 是非深度学习算法，也是PS修复画笔背后的算法，实现来自 [PyPatchMatch](https://github.com/vacancy/PyPatchMatch)，本程序用的是我的[修改版](https://github.com/dmMaze/PyPatchMatchInpaint)
  * lama* 是微调过的[lama](https://github.com/advimman/lama)
  

### 翻译器

 * 谷歌翻译器已经关闭中国服务，大陆再用需要设置全局代理，并在设置面板把 url 换成*.com
 * 彩云，需要申请 [token](https://dashboard.caiyunapp.com/)
 * papago  
 * DeepL 和 Sugoi (及它的 CT2 Translation 转换)翻译器，感谢 [Snowad14](https://github.com/Snowad14)，如果要使用Sugoi翻译器(仅日译英)，下载[离线模型](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm)，将 ```sugoi_translator``` 移入 BallonsTranslator/ballontranslator/data/models。 
 * 支持 [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame)。如果在本地单卡上运行且显存不足，可以在设置面板里勾选 ```low vram mode``` (默认启用)。
 * DeepLX 请参考[Vercel](https://github.com/bropines/Deeplx-vercel) 或 [deeplx](https://github.com/OwO-Network/DeepLX)
 * 支持两个版本的 OpenAI 兼容翻译器，支持兼容 OpenAI API 的官方或第三方LLM提供商，需要在设置面板里配置。
   * 无后缀版本token消耗更小，但分句稳定性稍差，长文本翻译可能有问题。
   * exp后缀版本token消耗更大，但稳定性更好，且在Prompt中进行了“越狱”，适合长文本翻译。
 * [m2m100](https://huggingface.co/facebook/m2m100_1.2B): 下载并将 m2m100-1.2B-ctranslate2 移到 data/models 目录下

其它优秀的离线英文翻译模型请参考[这条讨论](https://github.com/dmMaze/BallonsTranslator/discussions/515)  
如需添加新的翻译器请参考[加别的翻译器](doc/加别的翻译器.md)，本程序添加新翻译器只需要继承基类实现两个接口即可不需要理会代码其他部分，欢迎大佬提 pr

## 杂
* 电脑带 Nvidia 显卡或 Apple silicon 默认启用 GPU 加速
* 感谢 [bropines](https://github.com/bropines) 提供俄语翻译
* 第三方输入法可能会造成右侧编辑框显示 bug，见[#76](https://github.com/dmMaze/BallonsTranslator/issues/76)，暂时不打算修
<details>
  <summary><i>启用 AMD ROCm 显卡加速方法</i></summary>

### 通用方案 ZLUDA (ROCm)

**优点:**
文本和文本框识别速度比社区预览版略快，当然比 CPU 更快

**缺点:**
需要额外安装并进行相关配置才可工作，首次启动以及更换识别模型、显卡驱动升级等都需要长时间预热缓存

**安装步骤:**

1. 更新显卡驱动至最新版 (建议 24.12.1 及以上，根据自身系统环境下载并安装 [AMD HIP SDK Page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)  )
2. 下载 [ZLUDA](https://github.com/lshqqytiger/ZLUDA/releases) 并解压到 zluda 文件夹内，复制 zluda 文件夹到系统盘下: 比如c盘 (C:\zluda)  
3. 配置系统环境变量，以 windows 10 系统为例:设置 - 系统属性 - 高级系统设置 - 环境变量 - 系统变量 - 找到 path 变量，点击编辑，在最后添加 `C:\zluda` 和 `%HIP_PATH%bin` 两项  
4. 替换 CUDA 库的动态链接文件: 将 `C:\zluda` 文件夹内的 `cublas.dll` `cusparse.dll` 和 `nvrtc.dll` 复制出一份到桌面，按如下规则重命名复制出来的文件  

**注意事项**

注意 HIP SDK 和 ZLUDA 版本对应关系，建议使用较新的AMD显卡驱动程序。

|Windows 版本 | HIP SDK 版本 | ZLUDA 版本 |
|---|---|---|
|Windows 11 | 7.1.1 | 3.9.6 |
|Windows 10 和 11 | 6.4.2 | 3.9.5 |  
|Windows 10 和 11 | 6.2.4 | 3.9.5 |  
|Windows 10 和 11 | 6.1.2 | 3.9.5 |  


```
  原文件名 → 新文件名

  cublas.dll → cublas64_11.dll

  cusparse.dll → cusparse64_11.dll

  nvrtc.dll → nvrtc64_112_0.dll
```
  将已经重命名的文件替换掉 `BallonsTranslator\ballontrans_pylibs_win\Lib\site-packages\torch\lib\` 目录中的同名文件

5. 启动程序并设置 OCR 和文本检测 为 Cuda **(图像修复请继续使用 CPU)**
6. 运行 OCR 并等待 ZLUDA 编译 PTX 文件 **(首次编译大概需要 5-10 分钟，取决于 CPU 性能)**,**下次运行无需编译**
