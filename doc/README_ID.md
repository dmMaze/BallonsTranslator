<p align="center">
  <img
    width="256"
    alt="Spinning fox animation"
    src="https://github.com/user-attachments/assets/fe44e9a6-c7da-4bc5-8421-87fd6c38a0ba"
  />
</p>

<h1 align="center">BallonsTranslator</h1>

<p align="center">Sebuah aplikasi penerjemahan komik/manga yang dibantu oleh deep learning.</p>

<p align="center">
  <a href="/README.md">简体中文</a> | <a href="/README_EN.md">English</a> | <a href="/doc/README_RU.md">Русский</a> | <a href="/doc/README_JA.md">日本語</a> | <a href="/doc/README_ES.md">Español</a> | <a href="/doc/README_FR.md">Français</a> | <a href="/doc/README_PT-BR.md">pt-BR</a> | <a href="/doc/README_KO.md">한국어</a> | Indonesia | <a href="/doc/README_VI.md">Tiếng Việt</a>
</p>

# Fitur
* Terjemahan otomatis  
  - Mendukung pendeteksian, pengenalan, penghapusan, dan penerjemahan teks secara otomatis, performa keseluruhan bergantung pada modul-modul ini.
  - Peletakkan kata-kata berdasarkan perkiraan letak teks aslinya.
  - Mendukung format manga dan komik.
  - Typesetting optimal untuk manga->bahasa Inggris, bahasa Inggris->Mandarin (berdasarkan ekstraksi daerah balon.).
  
* Pengeditan gambar  
  - Mendukung pengeditan mask & inpainting (seperti alat content aware fill di PS) 
  - Mendukung gambar dengan rasio aspek ekstrim seperti webtoon
  
* Pengeditan teks  
  - Mendukung format rich text dan style teks, teks yang diterjemahkan dapat diedit secara langsung.
  - [Transformasi teks](https://github.com/dmMaze/BallonsTranslator/pull/1238), pencarian & penggantian kata
  - Mendukung ekspor/impor ke/dari dokumen word

* <details>
  <summary><i>Terjemahan LLM yang peka terhadap konteks dan glosarium</i></summary>

  **Riwayat terjemahan**

  - Atur **LLM Context** ke **+history** agar `LLMTranslator` melihat contoh dari halaman sebelumnya yang telah selesai. Ini dapat menjaga konsistensi nama, istilah, dan nada. Proses lanjutan dan rentang terpilih juga dapat memakai halaman sebelumnya yang memenuhi syarat.
  - **Token budget** mengatur jumlah teks terjemahan sebelumnya yang disertakan, dengan prioritas pada halaman yang lebih baru. Halaman saat ini, instruksi, glosarium, dan respons yang dihasilkan membutuhkan ruang tambahan. Nilai defaultnya `4096`.
  - Anggaran yang lebih besar memberi lebih banyak konteks cerita dan lebih jarang membuang halaman lama, tetapi mengirim lebih banyak teks dan dapat berjalan lebih lambat. Model lokal juga dapat membutuhkan jauh lebih banyak RAM/VRAM. Nilai default `4096` sengaja dibuat konservatif; penyedia umum dengan jendela konteks besar, seperti DeepSeek, sering dapat memakai batas yang lebih tinggi. Sekitar 70% dari batas konteks model adalah batas atas yang wajar (`90000` untuk 128K).
  - Anggaran riwayat juga memengaruhi cache prompt. Selama riwayat bertambah dalam batas anggaran, permintaan berurutan mempertahankan bagian awal yang sama; penyedia seperti OpenAI dan DeepSeek dapat memakainya kembali dengan harga token input yang lebih murah dan terkadang latensi lebih rendah. Ketika anggaran memaksa halaman lama dibuang, bagian awal itu berubah dan penggunaan cache direset. Anggaran lebih besar mengurangi reset, tetapi mengirim lebih banyak riwayat sehingga tidak menjamin biaya total lebih rendah.

  Tabel berikut adalah perkiraan kasar untuk halaman manga menggunakan DeepSeek, dengan harga token input cache sebesar 10% dari token input biasa. Hasil sebenarnya berbeda menurut proyek, model, dan penyedia.

  | Token budget | Perkiraan riwayat yang dipertahankan (halaman) | Perkiraan biaya total dibanding tanpa riwayat |
  |---:|---:|---:|
  | `2048` | 3–4 | 1.65× |
  | `4096` | 6–9 | 1.79× |
  | `8192` | 12–19 | 2.10× |
  | `16384` | 23–38 | 2.66× |

  **Glosarium yang dapat digunakan kembali**

  - Atur **Glossary File** pada dialog Run ke berkas UTF-8 `.json`, `.txt`, atau `.tsv`. Berkas hanya dibaca dan dapat digunakan kembali di berbagai proyek.
  - **Matching** hanya mengirim entri yang istilah sumbernya terdapat pada halaman terkait. **All** mengirim semua entri dan dapat memakai jauh lebih banyak token.
  - Format yang didukung meliputi:

    ```text
    # Teks bergaya Sakura
    sumber->terjemahan # catatan opsional

    # Teks yang dipisahkan tab
    sumber<TAB>terjemahan<TAB>catatan opsional
    ```

    ```json
    [
      {"src": "sumber", "dst": "terjemahan", "info": "catatan opsional"}
    ]
    ```

  - Pencocokan bersifat literal dan tidak membedakan huruf besar-kecil. Entri yang bertentangan, berkas dengan format yang salah, format yang tidak didukung, dan berkas yang tidak ditemukan akan menghentikan terjemahan sebelum permintaan dikirim ke LLM.
  - Konteks halaman sebelumnya dan penyisipan glosarium hanya berlaku untuk `LLMTranslator`; penerjemah lain mengabaikan pengaturan ini.

  </details>

# Instalasi

### Di Windows

**Metode A (Penyiapan Lingkungan Lokal Satu-Klik, memerlukan PowerShell)**:
Script akan memasang `BallonsTranslator` di direktori tempat Anda menjalankannya:
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Atau jalankan perintah berikut di Command Prompt klasik (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Metode B (Unduh Paket Pra-konfigurasi)**:
Unduh `Ballonstranslator_win_minium.zip` dari [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), ekstrak, lalu klik dua kali `launch_win.bat` untuk meluncurkan aplikasi.

Metode ini tidak mendukung Windows 7; pengguna Windows 7 harus menginstal [Python 3.8](https://www.python.org/downloads/release/python-3810/) secara manual dan menjalankannya dari kode sumber.

Jika Anda melihat error yang melibatkan `msvcp140.dll`, `c10.dll`, atau `[WinError 1114]`, instal atau perbarui [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [catatan unduhan resmi](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)).

## macOS / Linux

Script akan memasang `BallonsTranslator` di direktori tempat Anda menjalankannya:
```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Jika `curl` tidak tersedia, unduh script dengan `wget -O ...` sebagai gantinya. Aplikasi akan berjalan otomatis setelah instalasi; selanjutnya gunakan `cd BallonsTranslator && ./launch.sh` untuk memulainya lagi.

Aplikasi memeriksa dependensi inti saat startup. Saat Anda memilih modul yang memerlukan pustaka tambahan, aplikasi akan meminta Anda memasang dependensi opsional yang hilang (Anda juga dapat mengaktifkan pemasangan otomatis di pengaturan).

# Penggunaan
**Disarankan untuk menjalankan program di terminal jika program ini crash dan tidak meninggalkan informasi, lihat gif berikut ini**
<img src="https://github.com/user-attachments/assets/ee92fbdc-718c-4e04-a876-0eff3ee2a989">  

- Pilih penerjemah yang diinginkan dan atur sumber dan target bahasa. 
 - Buka folder yang berisi gambar manga/manhua/webtoon yang ingin diterjemahkan.
 - Klik tombol "Run" dan tunggu hingga proses selesai.


Format font seperti ukuran font dan warna ditentukan oleh program secara otomatis dalam proses ini, Anda dapat menentukan format tersebut sebelum memulai proses dengan mengubah opsi yang sesuai dari "decide by program" menjadi "use global setting" di panel konfigurasi->Lettering. (pengaturan global adalah format yang ditampilkan oleh panel format font yang tepat ketika Anda tidak mengedit blok teks apa pun di adegan)
<img src="https://github.com/user-attachments/assets/fb8a8b2c-54e4-4579-8319-42a172296c80">

## Image editing

### inpaint tool
<img src="https://github.com/user-attachments/assets/de0bc35d-6651-4f2f-985c-cfe9bfafb124">
<p align = "center">
Mode pengeditan gambar, alat inpainting
</p>

### rect tool
<img src="https://github.com/user-attachments/assets/6c47f46f-ffd3-41fd-b667-5442be304c79">
<p align = "center">
Alat rect
</p>

Untuk 'menghapus' hasil inpainting yang tidak diinginkan, gunakan alat inpainting atau alat rect dengan menekan **tombol kanan**.  
Hasilnya tergantung pada seberapa akurat algoritme ("metode 1" dan "metode 2" dalam gif) mengekstrak mask dari teks. Ini berjalan lebih buruk pada teks & latar belakang yang kompleks.

## Pengeditan teks
<img src="https://github.com/user-attachments/assets/0f688abe-41f7-416a-85c8-e0dd6968fd00">
<p align = "center">
Mode Pengeditan teks
</p>

<img src="https://github.com/user-attachments/assets/6d31c8a5-b909-4339-8036-7fc3ba2f014c" div align=center>
<p align=center>
pemformatan kumpulan tata letak teks secara otomatis
</p>

<img src="https://github.com/user-attachments/assets/1b76c164-1454-4aa7-b60c-9fbdb0968350" div align=center>
<p align=center>
pengenalan kata & menerjemahkan area yang dipilih
</p>

## Shortcuts
* ```A```/```D``` atau ```pageUp```/```Down``` untuk pindah halaman.
* ```Ctrl+Z```, ```Ctrl+Shift+Z``` untuk undo/redo.(catatan: sejarah undo akan dihapus setelah pindah halaman)
* ```T``` untuk masuk mode text-editting (atau tombol "T" di toolbar bagian bawah).
*```W``` untuk masuk mode pembuatan text block, lalu seret mouse dengan diklik tombol kanan pada kanvas untuk menambahkan blok teks baru. (lihat gif pengeditan teks)
* ```P``` untuk mode edit gambar.  
* Di mode edit gambar, gunakan penggeser di bagian kanan bawah untuk mengontrol transparansi gambar asli.
* Tombol "OCR" dan "A" di toolbar bagian bawah dapat mengaktifkan OCR dan penerjemahan, jika Anda menonaktifkannya, program hanya akan melakukan deteksi dan penghapusan teks.
* Mengatur parameter modul otomatis di panel konfigurasi.  
* ```Ctrl++```/```Ctrl+-``` untuk mengubah ukuran gambar
* ```Ctrl+G```/```Ctrl+F``` untuk mencari secara global/dalam halaman saat ini.

<img src="https://github.com/user-attachments/assets/084a250d-6a31-4344-94c0-2a5f4ba64b96">  


# Modul otomasi
Proyek ini sangat bergantung pada [manga-image-translator](https://github.com/zyddnys/manga-image-translator), layanan online dan pelatihan model tidaklah murah, mohon pertimbangkan untuk menyumbangkan proyek ini:  
- Ko-fi: <https://ko-fi.com/voilelabs>
- Patreon: <https://www.patreon.com/voilelabs>
- 爱发电: <https://afdian.net/@voilelabs>  

Sugoi translator dibuat oleh [mingshiba](https://www.patreon.com/mingshiba).
  
## Deteksi teks
Deteksi teks bahasa Inggris dan Jepang, kode pelatihan, dan rincian lebih lanjut dapat ditemukan di [comic-text-detector](https://github.com/dmMaze/comic-text-detector)

## OCR
* Model pengenalan teks mit_32px berasal dari manga-image-translator, mendukung pengenalan teks bahasa Inggris dan Jepang dan warna teks.
 * Model pengenalan teks mit_48px berasal dari manga-image-translator, mendukung pengenalan teks bahasa Inggris, Jepang, dan Korea serta warna teks.
 * [manga_ocr] (https://github.com/kha-white/manga-ocr) berasal dari [kha-white] (https://github.com/kha-white),  pengenalan untuk teks bahasa Jepang, dengan fokus utama manga Jepang.

## Inpainting
  * AOT berasal dari manga-image-translator.
  * patchmatch adalah sebuah algoritma dari [PyPatchMatch](https://github.com/vacancy/PyPatchMatch), program ini menggunakan [versi dimodifikasi](https://github.com/dmMaze/PyPatchMatchInpaint) dari saya.
  

## Penerjemah

 * <s> Harap ubah url penerjemah goolge dari *.cn ke *.com jika Anda tidak diblokir oleh GFW. </s> Google mematikan layanan terjemahan di Cina, harap setel 'url' yang sesuai di panel konfigurasi ke *.com.
 * Penerjemah Caiyun perlu memerlukan [token] (https://dashboard.caiyunapp.com/).
 * Papago.
 * DeepL & Sugoi translator (dan konversi CT2 Translation-nya), terima kasih kepada [Snowad14](https://github.com/Snowad14).

Untuk menambahkan penerjemah baru, silakan lihat [how_to_add_new_translator](doc/how_to_add_new_translator.md), caranya mudah, cukup dengan membuat subclass dari BaseClass dan mengimplementasikan dua interface, kemudian Anda bisa menggunakannya di dalam aplikasi, Anda dipersilakan untuk berkontribusi pada proyek ini.  


## Hal lain
* Jika komputer Anda memiliki GPU Nvidia, program ini akan mengaktifkan akselerasi cuda untuk semua model secara default dan membutuhkan sekitar 6G memori GPU, Anda dapat menurunkan inpaint_size pada panel konfigurasi untuk menghindari OOM. 
* Terima kasih kepada [bropines] (https://github.com/bropines) untuk lokalisasi bahasa Rusia.  
* Menambahkan [saladict](https://saladict.crimx.com) (*Kamus pop-up dan penerjemah halaman profesional lengkap*) di menu mini ketika pilih teks. [Panduan instalasi](doc/saladict.md)
