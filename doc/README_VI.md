<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">Lại thêm một công cụ, phần mềm dịch truyện siu xịn khác có áp dụng ML/AI.</p>

<p align="center">
  <a href="/README.md">简体中文</a> | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | <a href="/doc/README_ID.md">Indonesia</a> | Tiếng Việt
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
  - [Biến đổi văn bản](https://github.com/dmMaze/BallonsTranslator/pull/1238), tìm kiếm & thay thế
  - Hỗ trợ cả import từ dạng word hoặc export ra dạng đó nữa

* <details>
  <summary><i>Dịch bằng LLM có nhận biết ngữ cảnh và bảng thuật ngữ</i></summary>

  **Lịch sử bản dịch**

  - Đặt **LLM Context** thành **+history** để cho `LLMTranslator` xem ví dụ từ các trang trước đã hoàn thành. Điều này giúp tên, thuật ngữ và giọng điệu nhất quán hơn. Khi tiếp tục hoặc dịch một phạm vi, các trang đủ điều kiện trước đó cũng có thể được dùng.
  - **Token budget** kiểm soát lượng văn bản dịch trước đó được đưa vào và ưu tiên các trang mới hơn. Trang hiện tại, chỉ dẫn, bảng thuật ngữ và phản hồi được tạo cần thêm không gian. Mặc định là `4096`.
  - Ngân sách lớn hơn cung cấp nhiều ngữ cảnh câu chuyện hơn và ít loại bỏ trang cũ hơn, nhưng gửi nhiều văn bản hơn và có thể chậm hơn. Mô hình cục bộ cũng có thể cần nhiều RAM/VRAM hơn đáng kể. Mặc định `4096` được cố ý đặt ở mức thận trọng; các nhà cung cấp phổ biến có cửa sổ ngữ cảnh lớn như DeepSeek thường có thể dùng giới hạn cao hơn. Khoảng 70% giới hạn ngữ cảnh của mô hình là mức trần hợp lý (`90000` cho 128K).
  - Ngân sách lịch sử cũng ảnh hưởng đến bộ nhớ đệm lời nhắc. Khi lịch sử tăng trong giới hạn ngân sách, các yêu cầu liên tiếp giữ nguyên phần đầu để nhà cung cấp như OpenAI và DeepSeek có thể tái sử dụng với giá token đầu vào thấp hơn và đôi khi giảm độ trễ. Khi ngân sách buộc phải loại bỏ trang cũ, phần đầu đó thay đổi và việc tái sử dụng bộ nhớ đệm được đặt lại. Ngân sách lớn hơn giúp giảm số lần đặt lại nhưng gửi nhiều lịch sử hơn, nên không đảm bảo tổng chi phí thấp hơn.

  Bảng dưới đây là ước tính sơ bộ cho các trang manga khi dùng DeepSeek, trong đó token đầu vào đã lưu đệm có giá bằng 10% token đầu vào thông thường. Kết quả thực tế thay đổi theo dự án, mô hình và nhà cung cấp.

  | Token budget | Lịch sử ước tính được giữ lại (trang) | Tổng chi phí ước tính so với không dùng lịch sử |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **Bảng thuật ngữ có thể tái sử dụng**

  - Đặt **Glossary File** trong hộp thoại Run thành một tệp UTF-8 `.json`, `.txt` hoặc `.tsv`. Tệp chỉ được đọc và có thể tái sử dụng trong nhiều dự án.
  - **Matching** chỉ gửi các mục có thuật ngữ nguồn xuất hiện trên trang liên quan. **All** gửi mọi mục và có thể sử dụng nhiều token hơn đáng kể.
  - Các định dạng được hỗ trợ gồm:

    ```text
    # Văn bản kiểu Sakura
    nguồn->bản dịch # ghi chú tùy chọn

    # Văn bản phân tách bằng tab
    nguồn<TAB>bản dịch<TAB>ghi chú tùy chọn
    ```

    ```json
    [
      {"src": "nguồn", "dst": "bản dịch", "info": "ghi chú tùy chọn"}
    ]
    ```

  - Việc đối chiếu là đối chiếu theo nghĩa đen và không phân biệt chữ hoa chữ thường. Các mục xung đột, tệp sai định dạng, định dạng không được hỗ trợ và tệp bị thiếu sẽ dừng quá trình dịch trước khi gửi yêu cầu đến LLM.
  - Ngữ cảnh từ các trang trước và việc chèn bảng thuật ngữ chỉ áp dụng cho `LLMTranslator`; các trình dịch khác bỏ qua những cài đặt này.

  </details>

# Cài đặt

## Trên Windows

### Trên Windows

**Phương pháp A (Cài đặt môi trường cục bộ bằng một cú nhấp chuột, yêu cầu PowerShell)**:
Tệp lệnh sẽ cài đặt `BallonsTranslator` trong thư mục nơi bạn chạy nó:
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Hoặc chạy lệnh sau trong Command Prompt (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Phương pháp B (Tải xuống gói cấu hình sẵn)**:
Tải xuống `Ballonstranslator_win_minium.zip` từ [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), giải nén rồi nhấp đúp vào `launch_win.bat` để khởi động ứng dụng.

Các phương pháp này không hỗ trợ Windows 7; người dùng Windows 7 phải cài đặt [Python 3.8](https://www.python.org/downloads/release/python-3810/) thủ công và chạy từ mã nguồn.

Nếu bạn thấy lỗi liên quan đến `msvcp140.dll`, `c10.dll` hoặc `[WinError 1114]`, hãy cài đặt hoặc cập nhật [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [ghi chú tải xuống chính thức](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

## macOS / Linux

Tệp lệnh sẽ cài đặt `BallonsTranslator` trong thư mục nơi bạn chạy nó:
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Nếu không có `curl`, hãy tải script bằng `wget -O ...`. Ứng dụng sẽ tự khởi động sau khi cài đặt; những lần sau dùng `cd BallonsTranslator && ./launch.sh` để mở lại.

Ứng dụng kiểm tra các phụ thuộc cốt lõi khi khởi động. Khi bạn chọn một mô-đun cần thư viện bổ sung, ứng dụng sẽ nhắc cài các phụ thuộc tùy chọn còn thiếu (bạn cũng có thể bật tự động cài đặt trong phần cài đặt).

# Cách sử dụng

**Bạn nên chạy chương trình trong terminal trong trường hợp nó bị crashed và không để lại log, hãy xem gif sau.**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">

- Lần đầu tiên chạy ứng dụng, hãy chọn Chương trình dịch, cài Ngôn ngữ gốc và Ngôn ngữ dịch bằng cách nhấp vào biểu tượng Cài đặt.
- Mở một thư mục chứa hình ảnh của truyện cần dịch (Manga/Manhua/Manhwa) bằng cách nhấp vào biểu tượng Thư mục.
- Nhấp vào nút `Run` và chờ quá trình hoàn thành.

Các định dạng phông chữ như kích thước và màu phông chữ được xác định tự động bởi chương trình, bạn có thể xác định trước các định dạng đó bằng cách thay đổi tùy chọn tương ứng từ "decide by program" sang "use global setting" trong Bảng cấu hình (Config Panel) -> Lettering. (Global setting, cấu hình toàn bộ, là những định dạng được hiển thị ở bảng định dạng phía bên phải màn hình, khi bạn đang không chỉnh sửa bất kỳ văn bản nào trong textblock).
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Chỉnh sửa hình ảnh

### Inpaint Tool
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
Chế độ Chỉnh sửa hình ảnh, Inpainting Tool
</p>

### rect tool
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
Chế độ Chỉnh sửa hình ảnh, Rect Tool
</p>

Để 'Xóa' những phần đã được inpainted không mong muốn, sử dụng Inpainting tool hoặc Rect tool trong khi đang bấm **chuổt phải**.  
Kết quả sẽ phụ thuộc vào độ chính xác của thuật toán trích xuất ra text mask (lớp mask chữ) (theo "Phương pháp 1" và "Phương pháp 2" trong GIF). Nếu văn bản & nền phức tạp thì kết quả tách có thể chưa tốt lắm.

## Chỉnh sửa văn bản
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
Chế độ Chỉnh sửa văn bản
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
Định dạng văn bản hàng loạt & Bố cục tự động
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
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
  
<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">

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
