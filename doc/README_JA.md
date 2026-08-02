<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">ディープラーニングを活用したマンガ翻訳支援ツール。</p>

<p align="center">
  <a href="/README.md">简体中文</a> | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | 日本語 | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | <a href="/doc/README_ID.md">Indonesia</a> | <a href="/doc/README_VI.md">Tiếng Việt</a>
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
  - リッチテキストフォーマットをサポートし、翻訳されたテキストはインタラクティブに編集することができます。
  - [テキスト変形](https://github.com/dmMaze/BallonsTranslator/pull/1238)、検索と置換

* <details>
  <summary><i>文脈対応 LLM 翻訳と用語集</i></summary>

  **翻訳履歴**

  - **LLM Context** を **+history** に設定すると、`LLMTranslator` は以前に完了したページを例として参照します。人名、用語、口調の一貫性向上に役立ちます。続行実行や選択範囲でも、対象となる以前のページを利用できます。
  - **Token budget** は、以前の訳文をどれだけ含めるかを制御し、新しいページを優先します。現在のページ、指示、用語集、生成応答には追加の空きが必要です。デフォルトは `4096` です。
  - 予算を増やすと物語のコンテキストが増え、古いページの削除回数も減りますが、入力が増えて遅くなる場合があります。ローカルモデルでは RAM/VRAM も大幅に増えることがあります。デフォルトの `4096` は意図的に控えめな値です。DeepSeek など大きなコンテキストウィンドウを持つ一般的なプロバイダーでは、より高い上限を利用できる場合が多くあります。モデルのコンテキスト上限の約 70% が妥当な上限です（128K なら `90000`）。
  - 履歴予算はプロンプトキャッシュにも影響します。履歴が予算内で増えている間は、連続するリクエストの先頭部分が同じままなので、OpenAI や DeepSeek などは入力トークンを割引価格で再利用でき、遅延が短くなる場合もあります。予算により古いページが削除されると先頭部分が変わり、キャッシュの再利用はリセットされます。予算を増やすとリセットは減りますが、送信する履歴も増えるため、総費用が必ず下がるわけではありません。

  下表は DeepSeek を使った漫画ページの概算例です。DeepSeek のキャッシュ済み入力トークンは、通常の入力トークンの 10% の価格です。実際の結果はプロジェクト、モデル、プロバイダーによって異なります。

  | Token budget | 保持される履歴の目安（ページ） | 履歴なしと比較した推定総費用 |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **再利用可能な用語集**

  - 実行ダイアログの **Glossary File** に UTF-8 の `.json`、`.txt`、`.tsv` ファイルを指定します。ファイルは読み取り専用で、複数のプロジェクトで再利用できます。
  - **Matching** は該当ページに原語が現れる項目だけを送信します。**All** は全項目を送信するため、トークン使用量が大きく増える場合があります。
  - 対応形式：

    ```text
    # Sakura 形式
    原語->訳語 # 任意の注記

    # タブ区切り
    原語<TAB>訳語<TAB>任意の注記
    ```

    ```json
    [
      {"src": "原語", "dst": "訳語", "info": "任意の注記"}
    ]
    ```

  - 大文字と小文字を区別しないリテラル一致です。競合する項目、不正な形式、未対応の拡張子、存在しないファイルがある場合、LLM リクエストを送信する前に翻訳を停止します。
  - 過去ページのコンテキストと用語集の挿入は `LLMTranslator` のみに影響し、他の翻訳器はこれらの設定を無視します。

  </details>

# インストール

### Windowsの場合

**方法 A (PowerShellを使用したワンクリック環境構築。PowerShellが必要)**:
このスクリプトは、実行したディレクトリに `BallonsTranslator` をインストールします：
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
または、通常のコマンドプロンプト (`cmd.exe`) で次のコマンドを実行します：
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**方法 B (事前構成済みパッケージのダウンロード)**:
[GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases) から `Ballonstranslator_win_minium.zip` をダウンロードし、展開して `launch_win.bat` をダブルクリックして起動します。

これらの方法は Windows 7 をサポートしていません。Windows 7 のユーザーは手動で [Python 3.8](https://www.python.org/downloads/release/python-3810/) をインストールし、ソースコードから実行する必要があります。

`msvcp140.dll`、`c10.dll`、`[WinError 1114]` に関するエラーが表示された場合は、[Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [公式ダウンロードノート](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)) をインストールまたは更新してください。

## macOS / Linux

このスクリプトは、実行したディレクトリに `BallonsTranslator` をインストールします：
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

`curl` が使用できない場合は、代わりに `wget -O ...` でスクリプトをダウンロードしてください。インストール後にアプリは自動的に起動します。次回以降は `cd BallonsTranslator && ./launch.sh` で再起動できます。

アプリは起動時にコア依存関係を確認します。追加ライブラリが必要なモジュールを選択すると、不足している任意依存関係のインストールを促します（設定で自動インストールを有効にすることもできます）。

## 完全自動翻訳
**万が一、プログラムがクラッシュして情報が残らなかった場合に備えて、以下のgifを参考に、ターミナルで実行することをお勧めします。**また、初回実行時に希望するトランスレータを選択し、ソース言語とターゲット言語を設定してください。翻訳が必要な画像が入ったフォルダを開き、
「実行」ボタンをクリックして処理が完了するのを待ちます。
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">

このとき、フォントサイズや色などのフォントフォーマットはプログラムによって自動的に決定されますが、panel->Letteringで、対応するオプションを"decide by program"から"use global setting"に変更すれば、これらのフォーマットを事前に決定できます（グローバル設定とは、シーン内の
テキストブロックを編集していないときに右フォントフォーマットパネルで表示されるフォーマットのことです）。
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## 画像編集

### 修復ツール
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
画像編集モード、修復ツール
</p>

### 長方形ツール
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
長方形ツール
</p>

不要なインペイント結果を"消去"するには、**右ボタン**を押した状態でインペイントツールまたは矩形ツールを使用します。
結果はアルゴリズム(gifの"方法1"と"方法2")がどれだけ正確にテキストマスクを抽出するかに依存します。複雑なテキストと背景の場合、パフォーマンスが低下する可能性があります。

## テキスト編集
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
テキスト編集モード
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
テキストの一括書式設定と自動レイアウト
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
選択範囲の OCR と翻訳
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

<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">


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
