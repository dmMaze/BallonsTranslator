# BallonTranslator
[简体中文](/README.md) | [English](/README_EN.md) | [Русский](/doc/README_RU.md) | [日本語](/doc/README_JA.md) | [Español](/doc/README_ES.md) | [Français](/doc/README_FR.md) | [pt-BR](/doc/README_PT-BR.md) | [한국어](/doc/README_KO.md) | [Indonesia](/doc/README_ID.md) | [Tiếng Việt](/doc/README_VI.md)

Sebuah aplikasi penerjemahan komik/manga yang dibantu oleh deep learning.

<img src="https://github.com/user-attachments/assets/2140c402-dda2-47bc-9e7f-83ed41ce78af" div align=center>

<p align=center>
pratinjau
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
  - Mendukung pencarian & penggantian kata
  - Mendukung ekspor/impor ke/dari dokumen word

# Instalasi

### Di Windows
Jika Anda tidak ingin menginstal Python dan mengonfigurasi lingkungan secara manual:

**Metode A (Penyiapan Lingkungan Lokal Satu-Klik, memerlukan PowerShell)**:
Script akan secara otomatis membuat folder `BallonsTranslator` di direktori Anda saat ini, mengunduh kode sumber terbaru, mengonfigurasi lingkungan Python 3.12 yang terisolasi, menginstal semua dependensi inti, dan meluncurkan aplikasi (memerlukan PowerShell diaktifkan di sistem):
```powershell
irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex
```
Atau jalankan perintah berikut di Command Prompt klasik (`cmd.exe`):
```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.ps1 | iex"
```

**Metode B (Unduh Paket Pra-konfigurasi)**:
Unduh file `Ballonstranslator_win_minium.zip` terbaru dari [GitHub Releases](https://github.com/dmMaze/BallonsTranslator/releases), ekstrak ke folder mana pun, dan klik dua kali `launch_win.bat` untuk meluncurkan aplikasi.
Paket ini tidak mendukung Windows 7; pengguna Windows 7 harus menginstal [Python 3.8](https://www.python.org/downloads/release/python-3810/) secara manual dan menjalankannya dari kode sumber.

Modul PyTorch/deep learning mungkin memerlukan [Microsoft Visual C++ Redistributable x64](https://aka.ms/vc14/vc_redist.x64.exe) (Visual Studio 2015-2022; [catatan unduhan resmi](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)). Instal atau perbarui jika Anda melihat error yang melibatkan `msvcp140.dll`, `c10.dll`, atau `[WinError 1114]`.

## macOS / Linux


```bash
curl -fLO https://raw.githubusercontent.com/dmMaze/BallonsTranslator/dev/scripts/install.sh && chmod +x install.sh && ./install.sh
```

Jika `curl` tidak tersedia, unduh script dengan `wget -O ...` sebagai gantinya.

Jalankan installer di direktori tempat Anda ingin folder `BallonsTranslator` dibuat. Aplikasi akan berjalan otomatis setelah instalasi; selanjutnya gunakan `cd BallonsTranslator && ./launch.sh` untuk memulainya lagi.

Aplikasi memeriksa dependensi inti saat startup. Saat Anda memilih modul yang memerlukan pustaka tambahan, aplikasi akan meminta Anda memasang dependensi opsional yang hilang (Anda juga dapat mengaktifkan pemasangan otomatis di pengaturan). Jika pengunduhan model gagal, periksa jaringan/proxy Anda, atau unduh model yang diperlukan dari [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) atau [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) lalu letakkan secara manual di direktori `data`.

Perangkat lunak memiliki pemeriksaan pembaruan bawaan; lihat Panel konfigurasi -> Startup & Update untuk detailnya.

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