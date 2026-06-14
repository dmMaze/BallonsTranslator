# BallonTranslator
[简体中文](/README.md) | [English](/README_EN.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | 日本語 | [Indonesia](../doc/README_ID.md) | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md)

ディープラーニングを活用したマンガ翻訳支援ツール。

<img src="./src/ui0.jpg" div align=center>

<p align=center>
プレビュー
</p>

# 特徴
* 完全自動翻訳
  - 自動テキスト検出、認識、削除、翻訳をサポートし、全体的な性能はこれらのモジュールに依存します。
  - 文字配置は、原文の書式推定に基づいています。
  - 漫画やコミックでまともに動作します。
  - マンガ->英語、英語->中国語の組版が改善されました（バルーン領域の抽出に基づく）。

* 画像編集
  マスク編集とインペイントのサポート（PSのスポットヒーリングブラシツールのようなもの）

* テキストの編集
  リッチテキストフォーマットをサポートし、翻訳されたテキストはインタラクティブに編集することができます。

# 使用方法

Windowsユーザーは、[腾讯云](https://share.weiyun.com/xoRhz9i4)または[MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)(note: you also need to download latest Ballonstranslator-1.3.xx from GitHub release and extract it to overwrite **Ballontranslator-1.3.0-core** or older installation to get the app updated.)

## ソースコードの実行

```bash
# このリポジトリのクローン
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# macOSの場合、requirements_macOS.txtをインストール
$ pip install -r requirements.txt
```

NVIDIA GPUをお持ちの場合、GPUアクセラレーションを有効にするためにpytorch-cudaをインストールします。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

[MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)  から **data** フォルダをダウンロードし、BallonsTranslator/ballontranslatorに移動して、最後に以下を実行します
```bash
python ballontranslator
```


Sugoi Translator（日英のみ）を使用するには、[オフラインモデル](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm)をダウンロードし、"sugoi_translator"をBallonsTranslator/ballontranslator/data/modelsに移動してください。

## 完全自動翻訳
**万が一、プログラムがクラッシュして情報が残らなかった場合に備えて、以下のgifを参考に、ターミナルで実行することをお勧めします。**また、初回実行時に希望するトランスレータを選択し、ソース言語とターゲット言語を設定してください。翻訳が必要な画像が入ったフォルダを開き、
「実行」ボタンをクリックして処理が完了するのを待ちます。
<img src="./src/run.gif">

このとき、フォントサイズや色などのフォントフォーマットはプログラムによって自動的に決定されますが、panel->Letteringで、対応するオプションを"decide by program"から"use global setting"に変更すれば、これらのフォーマットを事前に決定できます（グローバル設定とは、シーン内の
テキストブロックを編集していないときに右フォントフォーマットパネルで表示されるフォーマットのことです）。

## 画像編集

### 修復ツール
<img src="./src/imgedit_inpaint.gif">
<p align = "center">
画像編集モード、修復ツール
</p>

### 長方形ツール
<img src="./src/rect_tool.gif">
<p align = "center">
長方形ツール
</p>

不要なインペイント結果を"消去"するには、**右ボタン**を押した状態でインペイントツールまたは矩形ツールを使用します。
結果はアルゴリズム(gifの"方法1"と"方法2")がどれだけ正確にテキストマスクを抽出するかに依存します。複雑なテキストと背景の場合、パフォーマンスが低下する可能性があります。

## テキスト編集
<img src="./src/textedit.gif">
<p align = "center">
テキスト編集モード
</p>

<img src="./src/multisel_autolayout.gif" div align=center>
<p align=center>
テキストの一括書式設定と自動レイアウト
</p>

## ショートカット
* A/D または pageUp/Down でページをめくります。
* Ctrl+Z, Ctrl+Y でほとんどの操作を元に戻す/やり直すことができます。
* T でテキスト編集モード、（または下部のツールバーの「T」ボタン）W を押してテキストブロック作成モードを起動し、右ボタンをクリックしたままキャンバス上でマウスをドラッグすると、新しいテキストブロックが追加されます。(テキスト編集のgifを参照）。
* Pで画像編集モードへ。
* 画像編集モードでは、右下のスライダーでオリジナル画像の透明度を調整します。
* 下のツールバーの「OCR」と「A」ボタンは、OCRと翻訳を有効にするかどうかを制御し、それらを無効にした場合、プログラムはテキストの検出と削除を行いますだけです。
* 設定パネルで自動モジュールのパラメータを設定します。
* 画像のサイズを変更するには、Ctrl + +/。

<img src="./src/configpanel.png">


# Automation modules
このプロジェクトは[manga-image-translator](https://github.com/zyddnys/manga-image-translator)に大きく依存しており、オンラインサービスやモデルトレーニングは安くないので、プロジェクトの寄付を検討してください:
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>

Sugoi translatorは、[mingshiba](https://www.patreon.com/mingshiba)によって作成されています。

## 文字検出
英語と日本語のテキスト検出をサポートし、学習コードと詳細は[comic-text-detector](https://github.com/dmMaze/comic-text-detector)に掲載されています

## OCR
 * mit_32pxのテキスト認識モデルは、manga-image-translatorのもので、英語と日本語の認識とテキスト色の抽出をサポートしています。
 * mit_48pxのテキスト認識モデルは、manga-image-translatorのもので、英語、日本語、韓国語の認識とテキストカラーの抽出をサポートしています。
 * [manga_ocr](https://github.com/kha-white/manga-ocr)は[kha-white](https://github.com/kha-white)からです、

## 修復
  * AOTは、manga-image-translatorからです
  * patchmatchは[PyPatchMatch](https://github.com/vacancy/PyPatchMatch)のnondl algrithomで、このプログラムは私による[修正版](https://github.com/dmMaze/PyPatchMatchInpaint)を使用しています。


## 翻訳者

 * GFW によってブロックされていない場合は、goolge トランスレータの URL を *.cn から *.com に変更してください。
 * Caiyunの翻訳者は[token](https://dashboard.caiyunapp.com/)を必要とします
 * papago
 * DeepL & Sugoi translator(およびCT2変換)、[Snowad14](https://github.com/Snowad14)に感謝します

 新しいトランスレータを追加するには、[how_to_add_new_translator](doc/how_to_add_new_translator.md)を参照してください。これはBaseClassをサブクラスにして、2つのインターフェースを実装するだけでアプリケーションで使用できますので、プロジェクトへのコントリビュートは歓迎します。


## その他
* あなたのコンピュータにNvidia GPUがある場合、プログラムはデフォルトですべてのモデルのcudaアクセラレーションを有効にし、およそ6G GPUメモリを必要とします。
* ロシア語のローカライズを担当した[bropines](https://github.com/bropines)に感謝します。
