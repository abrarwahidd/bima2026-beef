---
license: cc-by-nc-4.0
task_categories:
- image-classification
language:
- id
tags:
- computer-vision
- agriculture
- food-quality
- meat-quality
- unsupervised-learning
- local-feature-extraction
size_categories:
- 1K<n<10K
---

# Dataset Deteksi Mutu Kesegaran Daging Sapi Multidomain

## Ringkasan Dataset
Dataset ini berisi kumpulan citra mentah (*raw images*) daging sapi dari berbagai potongan (*cuts*) yang dikumpulkan untuk mendukung penelitian di bidang *Computer Vision*, khususnya ekstraksi fitur lokal (*Local Feature Extraction*) untuk analisis mutu daging sapi secara multidomain. 

Penelitian ini didanai melalui **Program Kementerian Pendidikan Tinggi, Sains, dan Teknologi - Skema Penelitian Fundamental Reguler Tahun Anggaran 2026**.

## Perangkat dan Lingkungan Pengambilan Data
Pengambilan citra dirancang dengan parameter perangkat keras dan lingkungan yang terukur untuk menjaga konsistensi ekstraksi fitur:
* **Kamera:** Canon EOS 1200D dan Canon EOS 3000D
* **Pencahayaan (*Lighting*):** Sumber cahaya konstan sebesar **5000 lumen**.
* **Latar Belakang (*Background*):** 
  * **Day-1:** Backdrop *anti-glare* berwarna hitam.
  * **Day-2:** Karton berwarna biru muda. 
  * *Catatan Penting:* Perubahan warna latar belakang antara Day-1 dan Day-2 dilakukan dengan sengaja untuk mengakomodasi perubahan visual karakteristik daging; warna daging menjadi lebih gelap ketika memasuki tahap kurang segar/tidak segar, sehingga latar belakang disesuaikan agar kontras tetap optimal.

### Spesifikasi Citra
* **Resolusi:** 4608 x 3456 pixels
* **Resolusi Spasial:** 72 DPI
* **Kedalaman Warna:** 24-bit depth
* **Total Ukuran File:** ~9 GB

## Detail Pengambilan Data dan Kategori
Pengambilan gambar dilakukan secara simultan bersamaan dengan proses pemotongan daging setelah diterima dari tempat pengambilan dataset pada pukul 07:00 WIB. Daging bersumber dari dua domain berbeda: Rumah Potong Hewan (RPH) dan Usaha Mikro, Kecil, dan Menengah (UMKM).

### Distribusi Data (Total: 2.456 Citra)
| Kode | Jenis Potongan | Sumber | Day-1 (29 Jun 26) | Day-2 (30 Jun 26) | Total Citra |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **SMC** | Samcam | RPH | 125 | 137 | 262 |
| **HDM** | Has Dalam | RPH | 132 | 140 | 272 |
| **HLR** | Has Luar | RPH | 130 | 133 | 263 |
| **LMS** | Lemusir | RPH | 132 | 140 | 272 |
| **KLP** | Kelapa | RPH | 133 | 139 | 272 |
| **PNT** | Penutup | UMKM | 139 | 140 | 279 |
| **PDS** | Pendasar | UMKM | 140 | 140 | 280 |
| **SKL** | Sengkel | UMKM | 140 | 140 | 280 |
| **SDL** | Sandung Lamur | UMKM | 136 | 140 | 276 |
| **Total** | | | **1.207** | **1.249** | **2.456** |

## Struktur Berkas & Penamaan File

### Format Penamaan File (*Naming Convention*)
Setiap file citra dinamai menggunakan pola standar berikut untuk memudahkan *parsing* metadata secara otomatis:
`[HARI]_[DOMAIN]_[KODE_DAGING]_[NOMOR_URUT].JPG`

*Contoh:* `DAY-1_RPH_HLR_001.JPG` (Citra Hari ke-1, bersumber dari RPH, jenis potongan Has Luar, nomor urut 001).

### Struktur Direktori Folder
Dataset ini disediakan dalam bentuk data mentah (*raw*) tanpa prapembagian (*pre-defined splits* seperti train/test). Berkas disusun berdasarkan folder Hari (`DAY`) untuk mengisolasi domain waktu serta karakteristik latar belakang:

```text
dataset_root/
├── DAY-1/
│   ├── DAY-1_RPH_HLR_001.JPG
│   ├── DAY-1_RPH_SMC_002.JPG
│   └── ...
└── DAY-2/
    ├── DAY-2_UMKM_PNT_001.JPG
    ├── DAY-2_UMKM_PDS_002.JPG
    └── ...
```
## Rekomendasi Penggunaan Dataset

Mengingat durasi pengambilan data yang berkesinambungan dan cukup panjang, dataset ini **sangat direkomendasikan untuk pendekatan *Unsupervised Learning*** (seperti klasterisasi atau ekstraksi fitur berbasis *self-supervised*).

Pengguna disarankan untuk melatih model guna mengenali pola ekstraksi fitur lokal secara mandiri dan mengelompokkannya ke dalam **2 label akhir (Segar dan Tidak Segar)** dengan menyerahkan pembagian klaster sepenuhnya kepada performa algoritma. Pembagian proporsi data (*data splits*) diserahkan penuh kepada pengguna sesuai kebutuhan eksperimen masing-masing.

## Lisensi dan Atribusi

Dataset ini didistribusikan di bawah lisensi **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Anda diizinkan untuk menggunakan, membagikan, dan memodifikasi dataset ini untuk keperluan riset akademik non-komersial, dengan memberikan atribusi yang sesuai kepada pembuat dataset.

---

## Ucapan Terima Kasih

[Tuliskan ucapan terima kasih Anda di sini. Contoh: "Penelitian ini didanai oleh Direktorat Riset, Teknologi, dan Pengabdian kepada Masyarakat (DRTPM) Kementerian Pendidikan Tinggi, Sains, dan Teknologi melalui Skema Penelitian Fundamental Reguler Tahun Anggaran 2026. Kami juga berterima kasih kepada [Nama RPH/UMKM atau Pihak Lain] atas kerja samanya dalam penyediaan sampel daging sapi."]

## Sitasi / Citation

Jika Anda menggunakan dataset ini dalam penelitian Anda, silakan sitasi karya kami menggunakan format berikut:

**BibTeX:**

```bibtex
@misc{mardiana_dataset_2026,
  author    = {Mardiana, Ardi and [Nama Anggota Tim Lainnya]},
  title     = {Dataset Deteksi Mutu Kesegaran Daging Sapi Multidomain Berbasis Local Feature Extraction},
  year      = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{[Masukkan URL Repository Hugging Face Anda, misal: [https://huggingface.co/datasets/username/nama-dataset](https://huggingface.co/datasets/username/nama-dataset)]}}
}

```

**APA Style:**
Mardiana, A., & [Nama Tim Lainnya, Inisial.] (2026). *Dataset Deteksi Mutu Kesegaran Daging Sapi Multidomain Berbasis Local Feature Extraction*. Hugging Face. [Masukkan URL Repository Hugging Face Anda]

```