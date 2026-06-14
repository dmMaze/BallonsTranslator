# BallonTranslator
[简体中文](/README.md) | [English](/README_EN.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | [日本語](../doc/README_JA.md) | Indonesia | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md)

Sebuah aplikasi penerjemahan komik/manga yang dibantu oleh deep learning.

<img src="./src/ui0.jpg" div align=center>

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

**Pengguna Windows** dapat unduh Ballonstranslator-x.x.x-core.7z di [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) atau [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) (catatan: Anda juga perlu mengunduh Ballonstranslator-1.3.xx terbaru di rilis GitHub mengekstraknya untuk menimpa **Ballontranslator-1.3.0-core** atau instalasi yang lebih lama agar aplikasi dapat diperbarui.)

## Jalankan kode sumber

```bash
# Clone repo ini
$ git clone https://github.com/dmMaze/BallonsTranslator.git ; cd BallonsTranslator

# instal requirements_macOS.txt di macOS
$ pip install -r requirements.txt
```

Instal pytorch-cuda untuk dapat akselerasi GPU jika Anda memiliki GPU NVIDIA.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116
```

Unduhlah folder **data** dari [MEGA](https://mega.nz/folder/gmhmACoD#dkVlZ2nphOkU5-2ACb5dKw) atau [Google Drive](https://drive.google.com/drive/folders/1uElIYRLNakJj-YS0Kd3r3HE-wzeEvrWd?usp=sharing) dan pindahkan ke dalam BallonsTranslator/ballontranslator, akhirnya jalankan
```bash
python ballontranslator
```

Untuk pengguna Linux atau MacOS, lihat [script ini](ballontranslator/scripts/download_models.sh) dan jalankan untuk mengunduh semua model

Untuk menggunakan Sugoi translator (hanya bahasa Jepang-Inggris), unduh [offline model](https://drive.google.com/drive/folders/1KnDlfUM9zbnYFTo6iCbnBaBKabXfnVJm), pindahkan "sugoi_translator" ke dalam BallonsTranslator/ballontranslator/data/models.

# Penggunaan
**Disarankan untuk menjalankan program di terminal jika program ini crash dan tidak meninggalkan informasi, lihat gif berikut ini**
<img src="./src/run.gif">  

- Pilih penerjemah yang diinginkan dan atur sumber dan target bahasa. 
 - Buka folder yang berisi gambar manga/manhua/webtoon yang ingin diterjemahkan.
 - Klik tombol "Run" dan tunggu hingga proses selesai.


Format font seperti ukuran font dan warna ditentukan oleh program secara otomatis dalam proses ini, Anda dapat menentukan format tersebut sebelum memulai proses dengan mengubah opsi yang sesuai dari "decide by program" menjadi "use global setting" di panel konfigurasi->Lettering. (pengaturan global adalah format yang ditampilkan oleh panel format font yang tepat ketika Anda tidak mengedit blok teks apa pun di adegan)

## Image editing

### inpaint tool
<img src="./src/imgedit_inpaint.gif">
<p align = "center">
Mode pengeditan gambar, alat inpainting
</p>

### rect tool
<img src="./src/rect_tool.gif">
<p align = "center">
Alat rect
</p>

Untuk 'menghapus' hasil inpainting yang tidak diinginkan, gunakan alat inpainting atau alat rect dengan menekan **tombol kanan**.  
Hasilnya tergantung pada seberapa akurat algoritme ("metode 1" dan "metode 2" dalam gif) mengekstrak mask dari teks. Ini berjalan lebih buruk pada teks & latar belakang yang kompleks.

## Pengeditan teks
<img src="./src/textedit.gif">
<p align = "center">
Mode Pengeditan teks
</p>

<img src="./src/multisel_autolayout.gif" div align=center>
<p align=center>
pemformatan kumpulan tata letak teks secara otomatis
</p>

<img src="./src/ocrselected.gif" div align=center>
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

<img src="./src/configpanel.png">  


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