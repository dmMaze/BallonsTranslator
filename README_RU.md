# BallonTranslator
[简体中文](README.md) | [English](README_EN.md) | Русский | [日本語](README_JA.md) | [Indonesia](README_ID.md)

Еще один компьютерный инструмент для перевода комиксов/манги на основе глубокого обучения.

<img src="doc/src/ui0.jpg" div align=center>

<p align=center>
Пример интерфейса
</p>

# Особенности
* Полностью автоматизированный перевод  
  - Поддержка автоматического обнаружения, распознавания, удаления и перевода текста, общая производительность зависит от этих модулей.
  - Начертание букв основано на оценке форматирования оригинального текста.
  - Хорошо работает с мангой и комиксами.
  - Улучшенная верстка манга->английский, английский->китайский (основана на выделении облачков текста).
  
* Редактирование изображений  
  Поддержка редактирования масок и закраски (что-то вроде инструмента точечной лечебной кисти в PS) 
  
* Редактирование текста  
  Поддержка богатого форматирования текста, переведенные тексты можно редактировать в интерактивном режиме.

# Использование

Пользователи Windows могут загрузить Ballonstranslator-x.x.x-core.7z с сайта [腾讯云](https://share.weiyun.com/xoRhz9i4) или [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)(note: you also need to download latest Ballonstranslator-1.3.xx from GitHub release and extract it to overwrite **Ballontranslator-1.3.0-core** or older installation to get the app updated.)

## Запуск из исходного кода (работает слегка с ошибками)

```bash
# Клонируйте этот репозиторий
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# Установка зависимостей
$ pip install -r requirements.txt
```

Установите pytorch-cuda, чтобы включить ускорение GPU, если у вас есть GPU NVIDIA.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

Скачайте папку **data** с сайта [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) or [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) и переместите ее в BallonsTranslator/ballontranslator, наконец, выполните команду

```bash
python ballontranslator
```

## Полностью автоматизированный перевод
**Рекомендуется запускать программу в терминале на случай, если она аварийно завершила работу и не оставила никакой информации, см. следующий gif**.

Пожалуйста, выберите нужный переводчик и установите исходный и целевой языки при первом запуске приложения. Откройте папку с изображениями, которые необходимо перевести, нажмите кнопку "Run" и дождитесь завершения процесса.  
<img src="doc/src/run.gif">  

Форматы шрифтов, такие как размер шрифта, цвет, определяются программой автоматически в этом процессе, вы можете предопределить эти форматы, изменив соответствующие опции с "определять программой" на "использовать глобальные настройки" в панели конфигурации-> Работа с текстом. (глобальные настройки - это те форматы, которые отображаются на правой панели формата шрифта, когда вы не редактируете какой-либо текстовый блок в сцене).

## Редактирование изображений

### Инструмент закраски
<img src="doc/src/imgedit_inpaint.gif">
<p align = "center">
Режим редактирования изображений, инструмент закраски
</p>

### Инструмент "прямоугольник"
<img src="doc/src/rect_tool.gif">
<p align = "center">
Инструмент "прямоугольник"
</p>

Перетащите прямоугольник с нажатой левой кнопкой, чтобы стереть текст внутри поля, нажмите правую кнопку и перетащите, чтобы очистить закрашенный результат.  
Результат зависит от того, насколько точно алгоритм ("метод 1" и "метод 2" на рисунке) извлекает маску текста. Он может работать хуже на сложном тексте и фоне.  

## Редактирование текста
<img src="doc/src/textedit.gif">
<p align = "center">
Режим редактирования текста
</p>

<img src="doc/src/multisel_autolayout.gif" div align=center>
<p align=center>
Пакетное форматирование текста и авторазметка
</p>

## Сочетания клавиш
* A/D сменить страницу
* Ctrl+Z, Ctrl+Y для отмены/повторения большинства операций, обратите внимание, что стек отмены будет очищен после того, как вы перевернете страницу.
* T в режиме редактирования текста (или кнопка "T" на нижней панели инструментов) нажмите W, чтобы активировать режим создания текстового блока, затем перетащите мышь по холсту с нажатой правой кнопкой, чтобы добавить новый текстовый блок. (см. GIF редактирования текста)
* P в режим редактирования изображений.
* В режиме редактирования изображения используйте ползунок справа внизу для управления прозрачностью исходного изображения.
* Кнопки "OCR" и "A" в нижней панели инструментов управляют включением OCR и перевода, если вы отключите их, программа будет выполнять только обнаружение и удаление текста.  
* Установите параметры автоматических модулей в настройках
* Ctrl+ +/- или колесо прокрутки для масштабирования холста
* Ctrl+A для выделения всех текстовых блоков в интерфейсе

<img src="doc/src/configpanel.png">  


# Модули автоматизации
Этот проект в значительной степени зависит от [manga-image-translator](https://github.com/zyddnys/manga-image-translator), онлайн-сервис и обучение моделей стоит недешево, пожалуйста, рассмотрите возможность пожертвовать проект:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>
  
## Обнаружение текста
Поддержка распознавания английского и японского текста, обучающий код и более подробную информацию можно найти на сайте [comic-text-detector](https://github.com/dmMaze/comic-text-detector)

## OCR
 * mit_32px модель распознавания текста из manga-image-translator, поддержка распознавания английского и японского языков и выделения цвета текста.
 * mit_48px модель распознавания текста от manga-image-translator, поддерживает распознавание на английском, японском и корейском языках и выделение цвета текста.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) от [kha-white](https://github.com/kha-white), 

## Закрасчик
  * AOT взято с сайта manga-image-translator
  * patchmatch это не алгоритм глубокого обучения [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), эту программу использовал [modified version](https://github.com/dmMaze/PyPatchMatchInpaint) для себя.
  

## Переводчики
 * Пожалуйста, измените url переводчика goolge с *.cn на *.com, если вы находитесь за пределами китая.    
 * Переводчику Caiyun требуется [token](https://dashboard.caiyunapp.com/)
 * Papago

 Чтобы добавить новый переводчик, пожалуйста, обратитесь к [Добавление других переводчиков](doc/add_translator_ru.md), это просто как подкласс BaseClass и реализация двух интерфейсов, затем вы можете использовать его в приложении. Вы можете внести свой вклад в проект.  


## Разное

* Если ваш компьютер оснащен графическим процессором Nvidia, программа по умолчанию включает ускорение cuda для всех моделей, что требует около 6G памяти GPU, вы можете уменьшить размер inpaint_size в панели конфигурации, чтобы избежать перегрузки памяти. 

Перевел на Русский [bropines](https://github.com/bropines)

## Предварительный просмотр результатов полностью автоматизированного перевода
|            Original            |         Translated (CHS)         |         Translated (ENG)         |
| :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
|![Original](ballontranslator/data/testpacks/manga/original2.jpg 'https://twitter.com/mmd_96yuki/status/1320122899005460481')| ![Translated (CHS)](doc/src/result2.png) | ![Translated (ENG)](doc/src/original2_eng.png) |
|![Original](ballontranslator/data/testpacks/manga/original3.jpg 'https://twitter.com/_taroshin_/status/1231099378779082754')| ![Translated (CHS)](doc/src/original3.png) | ![Translated (ENG)](doc/src/original3_eng.png) |
| ![Original](ballontranslator/data//testpacks/manga/AisazuNihaIrarenai-003.jpg) | ![Translated (CHS)](doc/src/AisazuNihaIrarenai-003.png) | ![Translated (ENG)](doc/src/AisazuNihaIrarenai-003_eng.png) |
|           ![Original](ballontranslator/data//testpacks/comics/006049.jpg)           | ![Translated (CHS)](doc/src/006049.png) | |