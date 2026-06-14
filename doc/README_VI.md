# BallonTranslator
[简体中文](/README.md) | [English](/README_EN.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | [日本語](../doc/README_JA.md) | [Indonesia](../doc/README_ID.md) | Tiếng Việt | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md)

Lại thêm một công cụ, phần mềm dịch truyện siu xịn khác có áp dụng ML/AI.

<img src="./src/ui0.jpg" div align=center>

<p align=center>
preview
</p>

# Đặc trưng
* Dịch hoàn toàn tự động
  - Hỗ trợ phát hiện văn bản tự động, nhận dạng, loại bỏ và dịch thuật. Các tính năng xoay quanh hầu hết phụ thuộc vào các đặc tính này.
  - Font, kích thức chữ được ước tính dựa trên định dạng của văn bản gốc.
  - Hoạt động tốt với manga và comics.
  - Dùng siu xịn khi mà Manga -> Tiếng Anh, Tiếng Anh -> tiếng Trung (Zì app này các pháp sư Trung Hoa làm mà :> ).
  
* Chỉnh sửa hình ảnh
  - Hỗ trợ Chỉnh sửa & Inpainting (na ná brush tool trong Photoshop)
  - Thích nghi với hình ảnh có tỷ lệ khung hình cực cao như Webtoons (?? hem hỉu lém, nhưng mà nói chung sài được với cả webtoons)
  
* Chỉnh sửa văn bản
  - Hỗ trợ RTF (rich text formatting) zà [TSP (text style presets)](https://github.com/dmMaze/BallonsTranslator/pull/311), có thể chỉnh sửa lại các văn bản đã được dịch đó lun nè.
  - Hỗ trợ Tìm kiếm & Thay thế
  - Hỗ trợ cả import từ dạng word hoặc export ra dạng đó nữa

# Cài đặt

## Trên Windows

Nếu bạn lười cài Python và Git nhưng vẫn có thể truy cập vào Internet, thì có thể tải BallonsTranslator_dev_src_with_gitpython.7z từ [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) hoặc nà [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing), unzip nó rùi chạy ```launch_win.bat```.
Chạy file ```scripts/local_gitpull.bat``` để cập nhật bản mới nhất nhoa.

## Chạy mã nguồn (từ github)

*Phù hợp cho mấy bạn sài linux như tui hehe.*

Cài [Python](https://www.python.org/downloads/release/python-31011) **<= 3.12** (Đừng cóa mà sài cái bản có sẵn trên Microsoft Store) và [Git](https://git-scm.com/downloads).

```bash
# Clone this repo
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# Launch the app
$ python3 launch.py
```

**Lưu ý:** Lần đầu tiên khởi chạy, app sẽ tự động cài đặt các thư viện và tải xuống các models. Nếu tải xuống không thành công, bạn sẽ cần tải xuống thư mục **data** (hoặc các tệp bị thiếu được báo lỗi trong terminal) từ [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) hoặc [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) rùi lưu nó ở đường dẫn tương ứng trong thư mục mã nguồn.

## Chạy ứng dụng trên MacOS (tương thích với cả chip Intel và Apple Silicon)
<i>Lưu ý MacOS cũng có thể chạy cách bên trên nếu cách này không hoạt động.</i>  

![录屏2023-09-11 14 26 49](https://github.com/hyrulelinks/BallonsTranslator/assets/134026642/647c0fa0-ed37-49d6-bbf4-8a8697bc873e)

#### 1. Chuẩn bị
-   Tải libs và models từ [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw "MEGA") hoặc [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing)


<img width="1268" alt="截屏2023-09-08 13 44 55_7g32SMgxIf" src="https://github.com/dmMaze/BallonsTranslator/assets/134026642/40fbb9b8-a788-4a6e-8e69-0248abaee21a">

-  Chuyển tất cả các tài nguyên đã tải xuống vào thư mục ```data``` (chưa có thì tự tạo nhá), cấu trúc cây thư mục cuối cùng sẽ trông như nè:

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

-  Cài đặt pyenv command line tool để quản lý các phiên bản Python. Nên cài qua Homebrew.
```
# Install via Homebrew
brew install pyenv

# Install via official script
curl https://pyenv.run | bash

# Set shell environment after install
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```


#### 2. Chạy ứng dụng
```
# Enter the `data` working directory
cd data

# Clone the `dev` branch of the repo
git clone -b dev https://github.com/dmMaze/BallonsTranslator.git

# Enter the `BallonsTranslator` working directory
cd BallonsTranslator

# Run the build script, will ask for password at pyinstaller step, enter password and press enter
sh scripts/build-macos-app.sh
```
> 📌 Ứng dụng được build ra file chạy ở đường dẫn ```./data/BallonsTranslator/dist/BallonsTranslator.app```, kéo cái ```BallonsTranslator.app``` vô thư mục macOS application để cài đặt. Sẵn sàng sử dụng lun mà không cần cấu hình thêm cho Python.

</details>

Để sài Sugoi translator(Japanese-English only), tải [offline model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), chuyển "sugoi_translator" vào ```BallonsTranslator/ballontranslator/data/models```.

# Cách sử dụng

**Bạn nên chạy chương trình trong terminal trong trường hợp nó bị crashed và không để lại log, hãy xem gif sau.**
<img src="./src/run.gif">

- Lần đầu tiên chạy ứng dụng, hãy chọn Chương trình dịch, cài Ngôn ngữ gốc và Ngôn ngữ dịch bằng cách nhấp vào biểu tượng Cài đặt.
- Mở một thư mục chứa hình ảnh của truyện cần dịch (Manga/Manhua/Manhwa) bằng cách nhấp vào biểu tượng Thư mục.
- Nhấp vào nút `Run` và chờ quá trình hoàn thành.

Các định dạng phông chữ như kích thước và màu phông chữ được xác định tự động bởi chương trình, bạn có thể xác định trước các định dạng đó bằng cách thay đổi tùy chọn tương ứng từ "decide by program" sang "use global setting" trong Bảng cấu hình (Config Panel) -> Lettering. (Global setting, cấu hình toàn bộ, là những định dạng được hiển thị ở bảng định dạng phía bên phải màn hình, khi bạn đang không chỉnh sửa bất kỳ văn bản nào trong textblock).

## Chỉnh sửa hình ảnh

### Inpaint Tool
<img src="./src/imgedit_inpaint.gif">
<p align = "center">
Chế độ Chỉnh sửa hình ảnh, Inpainting Tool
</p>

### rect tool
<img src="./src/rect_tool.gif">
<p align = "center">
Chế độ Chỉnh sửa hình ảnh, Rect Tool
</p>

Để 'Xóa' những phần đã được inpainted không mong muốn, sử dụng Inpainting tool hoặc Rect tool trong khi đang bấm **chuổt phải**.  
Kết quả sẽ phụ thuộc vào độ chính xác của thuật toán trích xuất ra text mask (lớp mask chữ) (theo "Phương pháp 1" và "Phương pháp 2" trong GIF). Nếu văn bản & nền phức tạp thì kết quả tách có thể chưa tốt lắm.

## Chỉnh sửa văn bản
<img src="./src/textedit.gif">
<p align = "center">
Chế độ Chỉnh sửa văn bản
</p>

<img src="./src/multisel_autolayout.gif" div align=center>
<p align=center>
Định dạng văn bản hàng loạt & Bố cục tự động
</p>

<img src="./src/ocrselected.gif" div align=center>
<p align=center>
OCR & Chỉ dịch văn bản đã chọn
</p>

## Shortcuts
* ```A```/```D``` hoặc ```pageUp```/```pageDown``` : Chuyển trang
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` : Undo/redo hầu hết các hoạt động. (Lưu ý rằng list hoạt động có thể undo sẽ bị xóa sau khi bạn chuyển trang)
* ```T``` : Để chuyển sang chế độ chỉnh sửa văn bản (hoặc phím "T" ở thanh công cụ bên dưới).
* ```W``` : Để kích hoạt chế độ tạo khung văn bản, sau đó bấm chuột phải để thêm khung chữ mới trên canvas. (Xem GIF chỉnh sửa văn bản)
* ```P``` : Để sang chế độ chỉnh sửa hình ảnh.  
* Trong Chế độ Chỉnh sửa hình ảnh, sử dụng thanh trượt ở phía dưới bên phải để chỉnh sửa độ trong suốt của hình ảnh gốc.
* Tắt hoặc bật bất kỳ modules tự động nào qua titlebar->run, chạy chương trình khi mà tất cả modules bị vô hiệu sẽ làm lại việc soạn và render tất cả văn bản tùy theo cài đặt tương ứng.
* Đặt tham số cho các module tự động trong Bảng cấu hình.  
* ```Ctrl++```/```Ctrl+-``` (hoặc ```Ctrl+Shift+=```) Để thay đổi kích thước hình ảnh.
* ```Ctrl+G```/```Ctrl+F``` Để tìm kiếm trên tất cả hoặc trong trang hiện tại.
* ```0-9``` Để điều chỉnh độ trong suốt của lớp chữ
* Trong chỉnh sửa văn bản: **bold** - ```Ctrl+B```, <u>underline</u> - ```Ctrl+U```, *italics* - ```Ctrl+I``` 
* Cài đặt đổ bóng và độ trong suốt chữ ở text style panel -> Effect.  
  
<img src="./src/configpanel.png">

## Headless mode (Run without GUI)
``` python
python launch.py --headless --exec_dirs "[DIR_1],[DIR_2]..."
```
**Lưu ý:** Cấu hình (ngôn ngữ nguồn, ngôn ngữ đích, mô hình InPaint, v.v.) sẽ tải từ config/config.json.
Nếu kích thước phông chữ được render không đúng, hãy chỉ định DPI thủ công theo cách sau: ```--ldpi```, các giá trị thường dùng là 96 và 72.


# Các modules tự động
Dự án này phụ thuộc rất nhiều vào [manga-image-translator](https://github.com/zyddnys/manga-image-translator), Các dịch vụ trực tuyến và model training không rẻ, nếu được thì donate các dự án nè nha (Xin cám mơn :3):  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

[Sugoi translator](https://sugoitranslator.com/) is created by [mingshiba](https://www.patreon.com/mingshiba).
  
## Xác định văn bản
* Hỗ trợ phát hiện văn bản tiếng Anh và tiếng Nhật [comic-text-detector](https://github.com/dmMaze/comic-text-detector)
* Hỗ trợ Sử dụng phát hiện văn bản [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Cần điền username và password, việc đăng nhập tự động sẽ được thực hiện mỗi khi chương trình được khởi chạy.
   * Hướng dẫn chi tiết, [Tuanzi OCR Instructions (Chinese only)](doc/Tuanzi_OCR_Instructions.md)

## OCR
 * Tất cả các mô hình MIT* đều từ manga-image-translator, hỗ trợ nhận dạng tiếng Anh, Nhật Bản và Hàn Quốc và trích xuất màu văn bản.
 * [manga_ocr](https://github.com/kha-white/manga-ocr) từ [kha-white](https://github.com/kha-white), Nhận dạng văn bản cho tiêng Nhật, tập trung vào manga.
 * Support áp dụng OCR [Starriver Cloud (Tuanzi Manga OCR)](https://cloud.stariver.org.cn/). Cần điền username và password, việc đăng nhập tự động sẽ được thực hiện mỗi khi chương trình được khởi chạy.
   * Phiên bản hiện tại sử dụng OCR trên mỗi textblock riêng, dẫn đến tốc độ chậm hơn và độ chính xác không được cải thiện tốt. Điều này khum được khuyến khích (thì khum tối ưu mà :<). Nếu cần, hãy sử dụng Tuanzi Detector thay thế.
   * Khi sài Tuanzi Detector cho việc xác định văn bản, nên đặt OCR thành none_ocr để có thể đọc trực tiếp văn bản, tiết kiệm thời gian và giảm số lượng yêu cầu.
   * Cụ thể đọc thêm tại đây [Tuanzi OCR Instructions (Chinese only)](doc/Tuanzi_OCR_Instructions.md)

## Inpainting
  * AOT [manga-image-translator](https://github.com/zyddnys/manga-image-translator).
  * Tất cả lama* đều là finetuned [LaMa](https://github.com/advimman/lama)
  * PatchMatch là một thuật toán từ [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), Phần mềm này sử dụng [phiên bản đã được tu luyện (modified version)](https://github.com/dmMaze/PyPatchMatchInpaint) bởi *me*. 
  

## Dịch thụât
Trình dịch có sẵn: Google, DeepL, ChatGPT, Sugoi, Caiyun, Baidu. Papago, and Yandex.
 * Google không cung cấp dịch vụ dịch tại Trung Quốc, vui lòng đặt 'URL' tương ứng trong bảng điều khiển thành *.com.
 * [Caiyun](https://dashboard.caiyunapp.com/), [ChatGPT](https://platform.openai.com/playground), [Yandex](https://yandex.com/dev/translate/), [Baidu](http://developers.baidu.com/), èn [DeepL](https://www.deepl.com/docs-api/api-access). Các trình dịch cần có token hoặc api key.
 * DeepL & Sugoi translator (and it's CT2 Translation conversion) thanks to [Snowad14](https://github.com/Snowad14).
 * Sugoi có thể dịch từ Japanese sang English kể cả khi ngoại tuyến (hong có kết nối mạng).
 * [Sakura-13B-Galgame](https://github.com/SakuraLLM/Sakura-13B-Galgame)

 Để thêm một trình dịch mới, xem chi tiết hơn ở đây [how_to_add_new_translator](doc/how_to_add_new_translator.md), hiểu đơn giản thì nó như phân lớp của BaseClass và triển khai hai giao diện, sau đó bạn có thể sử dụng trong ứng dụng, rấc welcome đóng góp cho dự án nhe.  


## FAQ & Misc
* Nếu máy tính của bạn có GPU NVIDIA hoặc Apple Silicon, chương trình sẽ có thể kích hoạt việc tăng tốc phần cứng. 
* Thêm hỗ trợ cho [saladict](https://saladict.crimx.com) (*All-in-one professional pop-up dictionary and page translator*) trong mini menu về lựa chọn text. [Installation guide](doc/saladict.md)
* Tăng tốc hiệu suất nếu bạn có [NVIDIA's CUDA](https://pytorch.org/docs/stable/notes/cuda.html) hoặc [AMD's ROCm](https://pytorch.org/docs/stable/notes/hip.html) thiết bị, hầu hết các module sử dụng [PyTorch](https://pytorch.org/get-started/locally/).
* Fonts được lấy từ fonts có trong máy.
* Gửi lời cảm ơn tới [bropines](https://github.com/bropines) cho việc Nga hóa.
* Thêm Export to photoshop JSX bởi [bropines](https://github.com/bropines).
  Để đọc các hướng dẫn, cải thiện code hoặc nà tò mò vọc quanh quanh để xem cách hoạt động, zô `scripts/export to photoshop` -> `install_manual.md`.
