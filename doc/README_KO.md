<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">딥러닝으로 구동되는 또 다른 컴퓨터 지원 만화/만화 번역 툴.</p>

<p align="center">
  <a href="/README.md">简体中文</a> | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | 한국어 | <a href="/doc/README_ID.md">Indonesia</a> | <a href="/doc/README_VI.md">Tiếng Việt</a>
</p>

# 특징
> [!IMPORTANT]
> **번역 결과물을 공개적으로 공유할 때 숙련된 번역가가 번역이나 교정에 참여하지 않았다면, 기계 번역임을 잘 보이는 곳에 표시해 주세요.**

* 완전 자동화된 번역
  - 자동 텍스트 감지, 인식, 제거 및 번역을 지원합니다. 전반적인 성능은 이러한 모듈에 따라 좌우집니다.
  - 대사는 원본 텍스트의 서식 추정치를 기반으로 합니다.
  - 망가와 코믹스 등을 작업할 수 있습니다.
  - 영어-중국어 및 일본어-영어 조판이 최적화되었습니다. 텍스트 레이아웃은 추출된 배경 풍선을 기반으로 합니다. 중국어 문장은 pkuseg를 기반으로 분할됩니다. 일본어 번역의 세로 레이아웃이 개선되었습니다.

* 이미지 편집
  - 마스크 편집 & 인페인팅 지원 (PS에 있는 스팟 힐링 브러쉬 툴 같이)
  - 웹툰과 같은 길다란 이미지도 편집 가능합니다

* 텍스트 편집
  - 풍부한 텍스트 포맷 지원 [텍스트 스타일 프리셋](https://github.com/dmMaze/BallonsTranslator/pull/311) 및, 번역된 텍스트는 대화형으로 편집할 수 있습니다.
  - [텍스트 변형](https://github.com/dmMaze/BallonsTranslator/pull/1238), 찾기 & 바꾸기 지원
  - 워드 문서를 불러오기/내보내기 지원

* <details>
  <summary><i>문맥 인식 LLM 번역 및 용어집</i></summary>

  **번역 기록**

  - **LLM Context**를 **+history**로 설정하면 `LLMTranslator`가 이전 완료 페이지를 예시로 참고합니다. 인명, 용어, 말투의 일관성을 높이는 데 도움이 됩니다. 이어서 실행하거나 선택한 범위에서도 적합한 이전 페이지를 사용할 수 있습니다.
  - **Token budget**은 이전 번역문을 얼마나 포함할지 정하며 최근 페이지를 우선합니다. 현재 페이지, 지침, 용어집, 생성 응답에는 추가 공간이 필요합니다. 기본값은 `4096`입니다.
  - 예산을 늘리면 이야기 문맥이 많아지고 오래된 페이지를 덜 자주 제거하지만, 입력이 늘어 느려질 수 있습니다. 로컬 모델은 RAM/VRAM도 크게 더 필요할 수 있습니다. 기본값 `4096`은 의도적으로 보수적인 값입니다. DeepSeek처럼 컨텍스트 창이 큰 일반적인 공급자는 더 높은 제한을 사용할 수 있는 경우가 많습니다. 모델 컨텍스트 한도의 약 70%가 합리적인 상한입니다(128K는 `90000`).
  - 기록 예산은 프롬프트 캐싱에도 영향을 줍니다. 기록이 예산 안에서 늘어나는 동안 연속 요청은 같은 앞부분을 유지하므로 OpenAI와 DeepSeek 같은 공급자가 입력 토큰을 할인된 가격으로 재사용하고 지연 시간을 줄일 수 있습니다. 예산 때문에 오래된 페이지를 제거하면 앞부분이 바뀌어 캐시 재사용이 초기화됩니다. 예산이 크면 초기화 횟수는 줄지만 더 많은 기록을 보내므로 총비용이 반드시 낮아지는 것은 아닙니다.

  아래 표는 DeepSeek를 사용한 만화 페이지의 대략적인 예시입니다. DeepSeek의 캐시된 입력 토큰 가격은 일반 입력 토큰의 10%입니다. 실제 결과는 프로젝트, 모델, 공급자에 따라 달라집니다.

  | Token budget | 예상 유지 기록(페이지) | 기록을 사용하지 않을 때와 비교한 예상 총비용 |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **재사용 가능한 용어집**

  - 실행 대화 상자의 **Glossary File**에 UTF-8 `.json`, `.txt` 또는 `.tsv` 파일을 지정하세요. 파일은 읽기 전용이며 여러 프로젝트에서 재사용할 수 있습니다.
  - **Matching**은 해당 페이지에 원문 용어가 나타나는 항목만 전송합니다. **All**은 모든 항목을 전송하므로 훨씬 더 많은 토큰을 사용할 수 있습니다.
  - 지원 형식:

    ```text
    # Sakura 형식 텍스트
    원문->번역문 # 선택 사항인 메모

    # 탭으로 구분된 텍스트
    원문<TAB>번역문<TAB>선택 사항인 메모
    ```

    ```json
    [
      {"src": "원문", "dst": "번역문", "info": "선택 사항인 메모"}
    ]
    ```

  - 일치는 대소문자를 구분하지 않는 리터럴 방식입니다. 서로 충돌하는 항목, 잘못된 파일, 지원하지 않는 형식, 누락된 파일이 있으면 LLM 요청을 보내기 전에 번역이 중단됩니다.
  - 이전 페이지 문맥과 용어집 삽입은 `LLMTranslator`에만 적용되며 다른 번역기는 이 설정을 무시합니다.

  </details>

# 설치

## Windows에서

### Windows의 경우

**방법 A (원클릭 로컬 환경 설정, PowerShell 필요)**:
이 스크립트는 실행한 디렉터리에 `BallonsTranslator`를 설치합니다:
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
또는 일반 명령 프롬프트 (`cmd.exe`)에서 다음 명령을 실행합니다:
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**방법 B (사전 구성된 패키지 다운로드)**:
[GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases)에서 `Ballonstranslator_win_minium.zip`을 다운로드하고 압축을 푼 뒤 `launch_win.bat`를 더블 클릭하여 실행합니다.

이 방법들은 Windows 7을 지원하지 않습니다. Windows 7 사용자는 [Python 3.8](https://www.python.org/downloads/release/python-3810/)을 수동으로 설치하고 소스 코드에서 실행해야 합니다.

`msvcp140.dll`, `c10.dll` 또는 `[WinError 1114]` 관련 오류가 보이면 [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe)(Visual Studio 2015-2022; [공식 다운로드 참고 사항](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist))를 설치하거나 업데이트하세요.

## macOS / Linux

이 스크립트는 실행한 디렉터리에 `BallonsTranslator`를 설치합니다:
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

`curl`을 사용할 수 없으면 대신 `wget -O ...`로 스크립트를 다운로드하세요. 설치 후 앱이 자동으로 시작됩니다. 이후에는 `cd BallonsTranslator && ./launch.sh`를 사용해 다시 실행할 수 있습니다.

앱은 시작 시 핵심 의존성을 확인합니다. 추가 라이브러리가 필요한 모듈을 선택하면 누락된 선택 의존성 설치를 안내합니다(설정에서 자동 설치도 활성화할 수 있습니다).

# 사용법

**충돌시 관련 정보를 남기기 위해 터미널 에서 프로그램을 실행하는 것이 좋습니다. 다음 GIF를 참조하십시오.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  
- 프로그램을 처음 실행하는 경우, 설정 아이콘을 클릭하여 번역기를 선택하고 소스 및 대상 언어를 설정하십시오.
- 폴더 아이콘을 클릭하여 번역이 필요한 만화(코믹,망가 등)의 이미지를 포함하는 폴더를 엽니다.
- '실행`버튼을 클릭하고 프로세스를 완료합니다.

이 과정에서 글꼴 크기 및 색상과 같은 글꼴 형식은 프로그램에 의해 자동으로 결정되며, 설정 패널->글꼴 설정에서 해당 옵션을 “프로그램이 결정”에서 “전역 설정 사용”으로 변경하여 해당 형식을 미리 결정할 수 있습니다. (전역 설정은 장면에서 텍스트 블록을 편집하지 않을 때 오른쪽 글꼴 형식 패널에 표시되는 스타일 입니다.)
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">  

## 이미지 편집

### 인페인트 도구
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
이미지 편집 모드, 인페인팅 도구
</p>

### 글상자 도구
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
글상자 도구
</p>

원하지 않는 칠한 결과를 '지우려면' **오른쪽 버튼**을 누른 상태에서 인페인팅 도구 또는 글상자 도구를 사용합니다.  
결과는 알고리즘('방법 1' 및 '방법 2'의 GIF)이 텍스트 마스크를 얼마나 정확하게 추출하는지에 따라 달라집니다. 복잡한 텍스트 및 배경에서는 성능이 저하될 수 있습니다.  

## 텍스트 편집
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
텍스트 편집 모드
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
일괄 텍스트 포맷팅 및 자동 레이아웃
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
선택 영역 OCR 및 번역
</p>

## 단축키
* ```A```/```D``` 및 ```pageUp```/```Down``` 으로 페이지를 이동합니다.
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` 로 대부분의 작업을 취소합니다. (페이지를 이동하면 작업을 취소할 수 없음에 유의하세요)
* ```T``` 를 눌러 텍스트 편집 모드로 전환합니다(또는 하단 도구 모음의 “T” 버튼).
* ```W``` 를 눌러 텍스트 블록 생성 모드를 활성화 합니다. 오른쪽 버튼을 클릭한 상태에서 마우스를 캔버스 위로 드래그하여 새 텍스트 블록을 추가합니다. (텍스트 편집 GIF 참조)
* ```P``` 를 눌러 이미지 편집 모드로 전환합니다.  
* 이미지 편집 모드에서 오른쪽 하단의 슬라이더를 사용하여 원본 이미지의 투명도를 조절합니다.
* 제목 표시줄->실행에서 자동 모듈을 활성화하거나 비활성화할 수 있으며, 모든 모듈을 비활성화한 상태로 실행하면 해당 설정에 따라 모든 텍스트가 다시 레터링되고 다시 렌더링됩니다.  
* 설정 패널에서 자동 모듈의 매개변수를 설정합니다. 
* ```Ctrl++```/```Ctrl+-``` (또는 ```Ctrl+Shift+=```) 로 이미지 크기를 조절합니다
* ```Ctrl+G```/```Ctrl+F``` 로 모든페이지 혹은 현재페이지 내에서 검색합니다.
* ```0-9``` 로 텍스트 레이어의 불투명도를 조정합니다.
* 텍스트 스타일: 두껍게 - ```Ctrl+B```, 밑줄 - ```Ctrl+U```, 이탤릭 - ```Ctrl+I``` 
* 텍스트 스타일 패널 -> 효과에서 텍스트 그림자 및 투명도를 설정합니다.  
* ```Alt+Arrow Keys``` 및 ```Alt+WASD``` (또는 텍스트 편집 모드에서 ```pageDown``` 및 ```pageUp```) 로 텍스트 블록 사이를 전환합니다.
  
<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

## 헤드리스 모드 (GUI 없이 실행)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
메모: 설정(원본 언어, 목표 언어, 인페인트 모델 등등)은 config/config.json에서 로드합니다.
렌더링 된 글꼴 크기가 맞지 않다면, ```--ldpi ```를 통해 DPI를 수동으로 지정하세요. 보편적인 값은 96 및 72입니다.


# 자동화 모듈
이 프로젝트는 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)에 크게 의존합니다. 온라인 서비스 및 모델 교육은 저렴하지 않으며 프로젝트 기부를 고려하십시오.
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/)는 [mingshiba](https://www.patreon.com/mingshiba)에 의해 개발되었습니다.

## 텍스트 검출
 * 영어 및 일본어의 텍스트 감지를 지원하며, 훈련 코드 및 자세한 내용은 [comic-text-detector](https://github.com/dmMaze/comic-text-detector)에서 확인할 수 있습니다.
* [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/)의 텍스트 감지를 지원합니다. 사용자 이름과 비밀번호는 입력해야 하며, 매 프로그램 실행 시 자동으로 로그인 됩니다.

   * 자세한 지침에 대해서는 **Tuanzi OCR 지침**: ([Chinese](./团子OCR说明.md) & [Brazilian Portuguese](./Manual_TuanziOCR_pt-BR.md) 만)
## OCR
 * 모든 mit* 모델은 manga-image-translator 에 기반하며, 영어, 일본어 및 한국어의 인식 및 텍스트 색상 추출을 지원합니다.
 * [kha-white](https://github.com/kha-white) 의 [manga_ocr](https://github.com/kha-white/manga-ocr) 는 , 일본어의 텍스트 인식을 수행하며, 일본어 만화의 초점을 맞췄습니다.
 * [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/) 를 사용한 텍스트 감지를 지원합니다. 사용자 이름과 비밀번호는 입력해야 하며, 매 프로그램 실행 시 자동으로 로그인 됩니다.
   * 현재 구현은 각 텍스트 블록에 OCR을 개별적으로 사용하며, 속도가 느리고 정확도가 크게 향상되지 않습니다. 추천되지 않습니다. 필요한 경우, 대신 Tuanzi Detector를 사용하십시오.
   * 텍스트 감지에 대한 Tuanzi 검출기를 사용하는 경우, OCR을 none_ocr로 설정하여 텍스트, 저장 시간을 직접 읽고 요청의 수를 줄이는 것이 좋습니다.
   * 자세한 지침에 대해서는 **Tuanzi OCR 지침**: ([Chinese](./团子OCR说明.md) & [Brazilian Portuguese](./Manual_TuanziOCR_pt-BR.md) 만)
* “선택적” PaddleOCR 모듈이 추가되었습니다. 디버그 모드에서는 이 모듈이 없다는 메시지가 표시됩니다. 여기에 설명된 지침에 따라 간단히 설치할 수 있습니다. 패키지를 직접 설치하지 않으려면 paddlepaddle(gpu) 및 paddleocr 줄의 주석(`#` 제거)을 해제하면 됩니다. 자신의 위험과 위험을 감수하고 모든 것을 베팅하세요. 저(브로핀)와 두 명의 테스터에게는 모든 것이 정상적으로 설치되었으므로 오류가 있을 수 있습니다. 이슈에 글을 작성하고 저를 태그하세요.

## 소개
  * AOT는 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 에 기반합니다.
  * 모든 lama*는 [LaMa](https://github.com/advimman/lama) 를 사용하여 파인튜닝 되었습니다.
  * PatchMatch는 [PyPatchMatch](https://github.com/vacancy/PyPatchMatch) 의 알고리즘입니다. 이 프로그램은 [modified version](https://github.com/dmMaze/PyPatchMatchInpaint)을 사용합니다.


## 번역기
가능한 번역기: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu. Papago 및 Yandex.
 * Google은 중국의 번역 서비스를 종료하였으니, 설정 패널에서 해당 'URL'을 *.com으로 설정하세요.
 * [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/), 및 [DeepL](https://www.deepl.com/docs-api/api-access) 번역기는 토큰 혹은 api 키를 요구합니다.
 * DeepL & Sugoi 번역기 (그리고 그것은 CT2 번역 변환입니다) [Snowad14](https://github.com/Snowad14) 에게 감사를 표합니다.
 * Sugoi는 완전히 오프라인으로 일본어를 영어로 번역합니다. [오프라인 모델](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm) 을 다운로드 하고, "sugoi_translator"를 BallonsTranslator/ballontranslator/data/models 에 이동시키세요.
 * [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame), 로컬 장치로 실행할 때 vram OOM 으로 인한 충돌이 발생하는 경우 설정 패널에서 ```low vram mode``` 를 설정하세요. (기본으로 활성화됨)
 * DeepLX: [Vercel](https://github.com/bropines/Deeplx-vercel) 또는 [deeplx](https://github.com/OwO-Network/DeepLX)를 참조하시기 바랍니다.
 * [Translators](https://github.com/UlionTse/translators) 라이브러리를 추가하여 api 키 없이 일부 번역 서비스에 액세스할 수 있습니다. 지원되는 서비스에 대해 찾을 수 있습니다 [참고](https://github.com/UlionTse/translators#supported-translation-services).
 * OpenAI API와 호환되는 공식 또는 제3자 LLM 제공 업체와 함께 작동하는 두 가지 버전의 OpenAI-compliant 번역기를 지원하며, 설정 패널에서 몇가지 설정을 필요로 합니다.
    * Non-suffix 버전은 토큰을 더 적게 사용하지만 문장 분할 안정성이 약간 약해 긴 텍스트 번역에 문제가 발생할 수 있습니다.
    * 'exp' suffix 버전은 더 많은 토큰을 사용하지만 안정성이 더 뛰어나고 프롬프트에 '탈옥'이 포함되어 있어 긴 텍스트 번역에 적합합니다.

다른 좋은 오프라인 영어 번역기를 추가하려면, 다음을 참조하시기 바랍니다 [스레드](https://github.com/dmMaze/BallonsTranslator/discussions/515).
새로운 번역기를 추가하려면 [how_to_add_new_translator](./how_to_add_new_translator.md)를 참조하시기 바랍니다. BaseClass의 하위 클래스로 두 개의 인터페이스를 구현하는 것처럼 간단합니다. 그런 다음 애플리케이션에서 이를 사용할 수 있으며, 프로젝트에 기여할 수 있습니다.


## FAQ 및 기타
* 만약 Nvidia GPU 또는 Apple silicon을 가지고 있다면, 프로그램은 하드웨어 가속을 가능하게합니다.
* 텍스트 선택의 미니 메뉴에서 [saladict](https://saladict.crimx.com)(*All-in-one 전문적인 팝업사전과 페이지 번역기*)에 대한 지원 추가. [설치 안내](./saladict.md) 
* 대부분의 모듈이 [PyTorch](https://pytorch.org/get-started/locally/) 을 사용하므로 [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) 또는 [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) 장치가 있는 경우 성능을 가속화하세요.
* 폰트는 시스템에 설치된 폰트입니다.
* 러시아 현지화를 진행한 [브로핀](https://github.com/bropines) 에 감사드립니다.
* [bropines](https://github.com/bropines) 에 의해 Photoshop JSX 스크립트로 내보내기를 추가했습니다. </br> 지침을 읽고 코드를 개선한 후 어떻게 작동하는지 확인하려면 'scripts/export to photoshop' -> 'install_manual.md'로 이동하면 됩니다.
