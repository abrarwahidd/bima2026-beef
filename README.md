---
license: mit
library_name: scikit-learn
tags:
- computer-vision
- image-processing
- unsupervised-learning
- bag-of-visual-words
- sift
- surf
- kmeans
- meat-quality
- beef-grading
---

# Pipeline Beef Research BIMA: Deteksi Mutu Kesegaran Daging Sapi Multidomain

## Daftar Isi

1. [Latar Belakang](#1-latar-belakang)
2. [Tujuan Penelitian](#2-tujuan-penelitian)
3. [Dataset](#3-dataset)
4. [Metodologi](#4-metodologi)
5. [Arsitektur Pipeline](#5-arsitektur-pipeline)
6. [Tahapan Pipeline (Detail Teknis)](#6-tahapan-pipeline-detail-teknis)
7. [Hasil dan Evaluasi](#7-hasil-dan-evaluasi)
8. [Struktur Output](#8-struktur-output)
9. [Setup dan Instalasi](#9-setup-dan-instalasi)
10. [Cara Menjalankan Pipeline](#10-cara-menjalankan-pipeline)
11. [Struktur Proyek](#11-struktur-proyek)

---

## 1. Latar Belakang

Penilaian mutu dan kesegaran daging sapi secara konvensional masih mengandalkan inspeksi visual oleh tenaga ahli—sebuah proses yang tidak skalabel, subjektif, dan rentan terhadap inkonsistensi antar penilai. Di sisi lain, pendekatan *machine learning* supervised (SVM, KNN, CNN) membutuhkan dataset berlabel dalam jumlah besar yang mahal dan memakan waktu untuk dikumpulkan, terutama di konteks penelitian lapangan (RPH dan UMKM).

Proyek ini menawarkan pendekatan alternatif: pipeline **sepenuhnya unsupervised** berbasis computer vision yang mampu memisahkan citra daging sapi ke dalam dua klaster mutu (Segar / Tidak Segar) **tanpa memerlukan label training satu pun**. Sistem ini dirancang untuk berjalan lintas domain (multi-domain), yaitu dapat menangani dataset dari kondisi pengambilan gambar yang berbeda (latar belakang hitam dari RPH Day-1 dan karton biru dari UMKM Day-2) tanpa memerlukan penyesuaian aturan warna yang berbeda per domain.

---

## 2. Tujuan Penelitian

- Membangun pipeline deteksi mutu kesegaran daging sapi berbasis unsupervised learning yang tidak memerlukan proses labeling supervised.
- Membandingkan kinerja dua algoritma ekstraksi fitur lokal: **SIFT** (*Scale-Invariant Feature Transform*) dan **SURF** (*Speeded-Up Robust Features*).
- Mengembangkan representasi citra berbasis **Bags of Visual Words (BoVW)** yang diperkaya dengan momen warna HSV (fusi fitur tekstur + warna).
- Mengevaluasi kualitas pemisahan klaster menggunakan metrik validasi internal (tanpa label) dan eksternal (opsional, bila label tersedia).
- Menghasilkan model produksi yang dapat digunakan untuk klasifikasi otomatis gambar daging baru.

---

## 3. Dataset

Dataset terdiri dari citra daging sapi yang dikumpulkan dari dua domain berbeda:

| Domain | Kondisi Pengambilan | Latar Belakang | Label Temporal |
|--------|---------------------|----------------|----------------|
| **DAY-1** | RPH (Rumah Potong Hewan), hari pertama pasca penyembelihan | Kain hitam | Segar |
| **DAY-2** | UMKM/pasar, hari berikutnya | Karton biru muda | Mendekati tidak segar |

### Konvensi Penamaan File

```
[HARI]_[DOMAIN]_[KODE]_[NOMOR].JPG
Contoh: DAY-1_RPH_SIR_001.JPG
```

- `HARI`: `DAY-1` atau `DAY-2`
- `DOMAIN`: kode lokasi pengambilan gambar
- `KODE`: jenis potongan daging (misal `SIR` untuk sirloin)
- `NOMOR`: nomor urut citra

### Label Ground Truth (Opsional)

File `data/labels.csv` dengan kolom `filename,label` (nilai: `segar` / `tidak_segar`) bersifat **opsional** dan **tidak digunakan untuk training**. Label ini hanya dipakai untuk validasi eksternal pasca-hoc (menghitung ARI, NMI, dan Purity) guna mengukur seberapa baik klaster otomatis bersesuaian dengan penilaian pakar.

---

## 4. Metodologi

### Prinsip Dasar: Unsupervised Sepenuhnya

Pipeline ini **tidak menggunakan label apapun dalam proses pelatihan**. Seluruh pemisahan mutu dilakukan secara otonom berdasarkan struktur visual inheren dari citra daging. Pendekatan ini dipilih karena:

1. **Data berlabel sulit dikumpulkan** di lapangan (membutuhkan ahli peternakan).
2. **Validitas lintas domain** — fitur tekstur dan warna permukaan daging yang terurai seharusnya universal, tidak bergantung pada latar belakang foto.
3. **Skalabilitas** — sistem dapat langsung diterapkan pada dataset baru tanpa proses re-labeling.

### Alur Metodologi

```
Citra Mentah
    │
    ▼
Preprocessing & Segmentasi ROI
(Resize → Denoising → HSV Thresholding → Morfologi → Bounding Box Crop)
    │
    ▼
Ekstraksi Fitur Lokal
(SIFT / SURF, dibatasi pada area mask daging)
    │
    ├── Descriptor lokal (128-dim SIFT / 64-dim SURF)
    └── Momen HSV (Mean, Std, Skewness per channel H/S/V → 9-dim)
    │
    ▼
Bags of Visual Words (BoVW)
(MiniBatchKMeans codebook → Histogram Hellinger → Fusi HSV)
    │
    ▼
Reduksi Dimensi (PCA)
    │
    ▼
Clustering Unsupervised
(K-Means / GMM, k=2)
    │
    ▼
Evaluasi & Komparasi
(Metrik internal + eksternal opsional + visualisasi PCA/t-SNE)
```

### Keputusan Desain Kunci

#### Segmentasi ROI Agnostik Domain

Segmentasi menggunakan **Strict HSV Thresholding** pada rentang pigmen mioglobin (H: 0–30 dan 160–180, S > 30, V > 25), diikuti **Margin Annihilation** (menghapus 12% tepi citra untuk mengeliminasi sisa label/stiker), lalu morfologi ellips (close + open) untuk menutup porositas tanpa mendistorsi batas. Pendekatan ini bekerja untuk kedua domain (hitam dan biru) tanpa aturan warna yang berbeda per domain.

#### Root-BoVW (Transformasi Hellinger)

Histogram BoVW dinormalisasi menggunakan **transformasi Hellinger** (√(histogram/total)) bukan L2 biasa. Ini menghasilkan vektor dengan norma L2 = 1 dan mengurangi sensitivitas terhadap frekuensi kata visual yang sangat dominan—ekuivalen dengan menggunakan kernel Hellinger pada SVM, namun bekerja langsung di ruang Euclidean sehingga K-Means dan GMM dapat diterapkan secara efektif.

#### Fusi Warna HSV yang Terkalibrasi

Momen warna HSV (mean, std, skewness untuk channel H, S, dan V = 9 dimensi) difusikan ke histogram BoVW setelah dua tahap normalisasi:
1. **MinMaxScaler** (fit di seluruh dataset, bukan per citra)
2. **L2-normalize per baris** agar norma-L2 = 1, setara dengan blok tekstur Hellinger

Dengan demikian, parameter `HSV_FUSION_WEIGHT` benar-benar merepresentasikan "bobot proporsi" yang terkendali: nilai 3.0 berarti kontribusi warna diberi bobot 3× lipat kontribusi tekstur.

#### Perbandingan Algoritma yang Adil (Ruang Fitur Seragam)

Semua algoritma clustering (K-Means dan GMM) dijalankan **di ruang fitur yang sama setelah PCA** (bukan K-Means di data mentah sementara GMM di PCA). Ini memastikan perbandingan murni pada level algoritma, bukan perbandingan ruang fitur yang tercampur.

#### Uji Stabilitas Klaster

Untuk menghindari *metric-chasing* (melaporkan Silhouette dari satu kali fit yang berpotensi terlalu optimis), pipeline menjalankan **10 kali subsample 80% data** dan melaporkan Silhouette mean ± std serta ARI vs klaster data penuh. ARI mendekati 1 berarti klaster stabil terhadap perturbasi subsampling.

---

## 5. Arsitektur Pipeline

Pipeline terdiri dari 8 tahap yang dapat dijalankan secara terpisah (dengan cache pickle antar tahap) atau sekaligus:

```
┌─────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                          │
│                                                             │
│  step_preprocess  →  step_extract  →  step_bovw             │
│       │                  │               │                  │
│  manifest.pkl      descriptors_*.pkl  histograms_*.pkl      │
│  masks/*.npz       timing_*.pkl       codebooks_*.pkl       │
│  processed/*.npz   hsv_*.pkl                                │
│                                                             │
│  step_cluster  →  step_evaluate  →  step_export             │
│       │                │               │                    │
│  cluster_labels.pkl  figures/       Laporan_Prediksi.csv    │
│  cluster_metrics.csv tables/                                │
│                                                             │
│  step_tune  →  step_build_production_model                  │
│       │                │                                    │
│  tuning_results.csv  meat_grading_model.joblib              │
│                                                             │
│  step_batch_experiment                                      │
│       │                                                     │
│  archives/  (14 skenario + Master_Log.csv)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Tahapan Pipeline (Detail Teknis)

### Tahap 1 — Preprocessing & Segmentasi ROI (`src/preprocessing.py`, `src/segmentation.py`)

**Alur per citra:**
1. **Resize** mempertahankan aspek rasio, sisi terpanjang menjadi 800px (`RESIZE_MAX_SIDE`).
2. **Denoising** menggunakan `fastNlMeansDenoisingColored` (opsional, diaktifkan jika `DENOISE_H` bukan `None`).
3. **Segmentasi ROI** — Strict HSV Thresholding pada rentang mioglobin + Margin Annihilation 12% + morfologi ellips 11×11 + pemilihan kontur terbaik berdasarkan skor `area / (dist_dari_pusat + 1)`.
4. **QA Visual** — menyimpan gambar berdampingan (asli + kontur merah | daging tersegmentasi di atas latar putih) ke `outputs/visualized_segments/`.
5. **Bounding Box Crop** — memotong citra ke bounding box foreground dengan padding 1% agar detektor tepi SURF tidak terpotong.
6. **Konversi** ke grayscale (untuk SIFT/SURF) dan HSV (untuk momen warna).

**Output:** `outputs/interim/processed/*.npz` (gray, hsv, mask per citra), `manifest.pkl`.

### Tahap 2 — Ekstraksi Fitur Lokal (`src/feature_extraction.py`)

- **SIFT**: `cv2.xfeatures2d.SIFT_create`, descriptor 128-dimensi.
- **SURF**: `cv2.xfeatures2d.SURF_create`, Hessian threshold = 500 (hasil tuning), descriptor 64-dimensi.
- Keypoint dibatasi **hanya pada area mask** menggunakan parameter `mask` di `detectAndCompute`.
- Descriptor di-cap maksimum 800 per citra (subsample acak) untuk menjaga efisiensi codebook.
- **Momen HSV**: Mean, Std, dan Skewness untuk channel H, S, dan V → vektor 9-dimensi per citra.
- Waktu ekstraksi per citra direkam untuk perbandingan efisiensi komputasi.

**Output:** `descriptors_sift.pkl`, `descriptors_surf.pkl`, `hsv_sift.pkl`, `hsv_surf.pkl`, `timing_*.pkl`.

### Tahap 3 — Bags of Visual Words (`src/bovw.py`)

- **Sampling descriptor**: Maksimum 200.000 descriptor dari seluruh dataset digabung untuk melatih codebook (menghindari memory error untuk dataset besar).
- **Pembangunan codebook**: `MiniBatchKMeans` dengan batch size 2000, `n_init=10`, `max_iter=200`, untuk ukuran k ∈ {50, 100, 200}.
- **Komputasi histogram**: Setiap citra dikuantisasi ke dalam histogram k-bin menggunakan `codebook.predict(descriptors)`, lalu dinormalisasi dengan transformasi Hellinger.
- **Fusi warna HSV**: Jika `USE_COLOR_FUSION=True`, vektor HSV dinormalisasi (MinMax → L2-norm) lalu dikali `HSV_FUSION_WEIGHT` dan di-concatenate ke histogram BoVW.

**Output:** `codebooks_*.pkl` (dict k → MiniBatchKMeans), `histograms_*.pkl` (dict k → matrix N×(k+9)).

### Tahap 4 — Clustering Unsupervised (`src/clustering.py`)

**4a. Sensitivitas Ukuran Codebook** (ruang fitur mentah, K-Means saja):
- Menguji pengaruh granularitas codebook (k=50, 100, 200) terhadap kualitas klaster.
- Ditandai `feature_space="raw"` agar tidak tercampur dengan analisis perbandingan algoritma.

**4b. Perbandingan Algoritma Clustering** (ruang fitur PCA, codebook k=200):
- PCA diaplikasikan ke data default codebook, mempertahankan hingga 30 komponen utama.
- K-Means (`n_init=20`) dan GMM (`covariance_type="full"`, `n_init=10`) dibandingkan di ruang PCA yang sama.
- Label hasil clustering disimpan untuk visualisasi dan ekspor.

**4c. Uji Stabilitas Klaster**:
- 10 iterasi subsample 80%, masing-masing dijalankan K-Means independen.
- Melaporkan Silhouette mean ± std dan ARI vs klaster data penuh.

**Output:** `cluster_labels.pkl`, `cluster_metrics_all.csv`, `cluster_stability.csv`.

### Tahap 5 — Evaluasi & Visualisasi (`src/evaluation.py`, `src/visualization.py`)

**Metrik Validasi Internal** (tanpa label):
| Metrik | Interpretasi | Arah Optimal |
|--------|-------------|--------------|
| **Silhouette Score** | Kerapatan intra-klaster vs. separasi antar-klaster, rentang [-1, 1] | ↑ Lebih tinggi lebih baik |
| **Davies-Bouldin Index** | Rata-rata similaritas tiap klaster dengan klaster paling mirip | ↓ Lebih rendah lebih baik |
| **Calinski-Harabasz Index** | Rasio dispersi antar-klaster / intra-klaster | ↑ Lebih tinggi lebih baik |

**Metrik Validasi Eksternal** (bila `labels.csv` tersedia):
| Metrik | Interpretasi |
|--------|-------------|
| **ARI** (Adjusted Rand Index) | Kesesuaian klaster dengan ground truth, dikoreksi untuk chance, rentang [-1, 1] |
| **NMI** (Normalized Mutual Info) | Informasi mutual antara klaster dan ground truth, rentang [0, 1] |
| **Purity** | Proporsi anggota klaster yang berasal dari kelas mayoritas |

**Visualisasi yang dihasilkan:**
- Proyeksi PCA 2D dan t-SNE 2D dari klaster terbaik.
- Bar chart Silhouette, Davies-Bouldin, dan Calinski-Harabasz per kombinasi (fitur × algoritma).
- Ablation Study: dampak PCA terhadap Silhouette Score (raw vs. PCA space).
- Boxplot efisiensi komputasi (waktu ekstraksi dan jumlah keypoint).
- Radar Chart multidimensional SIFT vs. SURF (performa + efisiensi).
- Visual proof keypoint SIFT vs. SURF pada citra sampel.
- Stacked bar chart distribusi klaster per hari (temporal degradation).

### Tahap 6 — Hyperparameter Tuning (`step_tune`)

Grid search terhadap:
- **HSV Fusion Weight**: [2.5, 2.8, 3.0, 3.2, 3.5]
- **PCA Components**: [15, 20, 25, 30, 35]
- **GMM Covariance Type**: ['full', 'tied', 'diag']

Total 75 kombinasi diuji, diurutkan berdasarkan Silhouette Score tertinggi lalu Davies-Bouldin terendah. Konfigurasi terbaik kemudian diuji stabilitasnya (10× subsample) dan hasilnya **dilaporkan sebagai mean ± std**, bukan nilai mentah dari pencarian, untuk menghindari bias optimistis.

> **Peringatan metodologis**: Silhouette Score hasil grid search tidak boleh dilaporkan langsung sebagai bukti separabilitas karena merupakan target optimisasi itu sendiri. Selalu gunakan angka stabilitas (mean ± std dari subsample berulang).

### Tahap 7 — Ekspor Prediksi (`step_export`)

Menghasilkan tabel prediksi per citra dari model terbaik (SIFT + K-Means) dalam format CSV yang siap dibuka di Excel, mencakup: nama file, domain asal, hari ke-, jenis potongan, ID klaster, dan asumsi kondisi (Klaster A / Klaster B).

### Tahap 8 — Pembuatan Model Produksi (`step_build_production_model`)

Merakit dan menyimpan seluruh komponen pipeline ke dalam satu file `.joblib`:

```python
model_package = {
    "codebook":   MiniBatchKMeans,   # Kamus visual BoVW
    "scaler":     MinMaxScaler,      # Normalisasi HSV
    "pca":        PCA,               # Reduksi dimensi
    "gmm":        GaussianMixture,   # Classifier akhir
    "hsv_weight": float,             # Bobot fusi warna
    "metadata":   dict               # Konfigurasi eksperimen
}
```

Model ini dapat dimuat dan digunakan untuk klasifikasi citra daging baru tanpa menjalankan ulang seluruh pipeline.

---

## 7. Hasil dan Evaluasi

### Komparasi SIFT vs. SURF

| Aspek | SIFT | SURF |
|-------|------|------|
| **Dimensi Descriptor** | 128-dim | 64-dim |
| **Kecepatan Ekstraksi** | Lebih lambat | ~2–3× lebih cepat |
| **Jumlah Keypoint** | Lebih banyak (threshold berbasis DoG) | Bergantung Hessian threshold (500 optimal) |
| **Separabilitas Klaster** | Silhouette umumnya lebih tinggi di ruang PCA | Silhouette kompetitif setelah fusi HSV |
| **Keunggulan** | Akurasi separasi tekstur | Efisiensi komputasi |

### Dampak Komponen Pipeline (Ablation Study)

| Komponen | Dampak pada Silhouette |
|----------|----------------------|
| Raw BoVW tanpa PCA | Baseline (rendah akibat *curse of dimensionality*) |
| + PCA (30 komponen) | Peningkatan signifikan |
| + Fusi HSV (weight=3.0) | Peningkatan tambahan — warna permukaan merupakan indikator kesegaran yang kuat |
| + Normalisasi L2 HSV | Memastikan bobot HSV terkendali dan dapat diinterpretasi |

### Uji Stabilitas (Laporan yang Jujur)

Kualitas klaster dilaporkan sebagai:
- **Silhouette mean ± std** dari 10× subsample 80%
- **ARI vs klaster data penuh** — nilai mendekati 1 menunjukkan struktur klaster yang stabil, bukan artefak dari satu kali fitting

### Sensitivitas Ukuran Codebook

Ukuran codebook k ∈ {50, 100, 200} diuji. Secara umum, k=200 memberikan representasi yang lebih granular dan Silhouette Score yang lebih baik. Perbedaan antara k=100 dan k=200 biasanya tidak dramatis, menunjukkan saturasi representasi.

---

## 8. Struktur Output

```
outputs/
├── interim/                       # Cache pickle antar tahap (dapat dihapus & diulang)
│   ├── manifest.pkl               # Metadata seluruh citra (hari/domain/kode/label)
│   ├── processed/                 # *.npz per citra (gray, hsv, mask)
│   ├── masks/                     # Mask ROI terpisah (*.npz)
│   ├── descriptors_sift.pkl
│   ├── descriptors_surf.pkl
│   ├── hsv_sift.pkl
│   ├── hsv_surf.pkl
│   ├── codebooks_sift.pkl
│   ├── codebooks_surf.pkl
│   ├── histograms_sift.pkl
│   ├── histograms_surf.pkl
│   ├── timing_sift.pkl
│   ├── timing_surf.pkl
│   └── cluster_labels.pkl
├── figures/                       # Seluruh grafik (PCA, t-SNE, radar, boxplot, dsb.)
├── tables/                        # Seluruh tabel hasil (CSV)
│   ├── manifest_summary.csv
│   ├── efficiency_raw.csv
│   ├── efficiency_summary.csv
│   ├── cluster_metrics_all.csv
│   ├── cluster_stability.csv
│   ├── tuning_results_deep_search.csv
│   ├── Laporan_Prediksi_SIFT_KMEANS.csv
│   └── final_summary.json
├── visualized_segments/           # Hasil QA segmentasi (asli | tersegmentasi)
├── meat_grading_model.joblib      # Model produksi siap pakai
└── logs/
    └── pipeline.log
```

```
archives/
└── DATA-BIMA_EXP-YYYYMMDD-XX/    # Satu folder per skenario eksperimen batch
    ├── code_snapshot/             # Snapshot kode saat eksperimen berjalan
    └── results/
        └── evaluation_metrics.json
Master_Log_DATA-BIMA_YYYYMMDD_HHMMSS.csv  # Rekapitulasi 14 skenario
```

---

## 9. Setup dan Instalasi

### Prasyarat

- Python 3.7 (wajib — OpenCV 3.4.2.16 dengan SIFT/SURF tersedia hanya di Python 3.7)
- [uv](https://github.com/astral-sh/uv) (package manager)

### Instalasi

```bash
# Install Python 3.7 lewat uv (jika belum ada di mesin)
uv python install 3.7

# Buat & sync environment sesuai pyproject.toml
uv sync

# Verifikasi SIFT & SURF tersedia (harus mencetak True True)
uv run python -c "import cv2; s=cv2.xfeatures2d.SIFT_create(); f=cv2.xfeatures2d.SURF_create(); print(True, True)"
```

> **Catatan:** Jika `uv python install 3.7` gagal karena python-build-standalone tidak lagi menyediakan build 3.7 di platform Anda, gunakan pyenv/conda untuk menyediakan interpreter 3.7, lalu arahkan uv ke sana:
> ```bash
> uv venv --python /path/to/python3.7
> uv sync
> ```

### Dependensi Utama

| Paket | Versi | Fungsi |
|-------|-------|--------|
| `opencv-contrib-python` | 3.4.2.16 | SIFT, SURF (`xfeatures2d`) |
| `scikit-learn` | ≥1.0.2 | KMeans, GMM, PCA, metrik |
| `numpy` | ≥1.21.6 | Operasi matriks |
| `pandas` | ≥1.1.5 | Tabel hasil |
| `matplotlib` | ≥3.5.3 | Visualisasi |
| `seaborn` | ≥0.12.2 | Visualisasi statistik |

---

## 10. Cara Menjalankan Pipeline

### Persiapan Data

1. Susun dataset sesuai konvensi penamaan (`DAY-1_[DOMAIN]_[KODE]_[NOMOR].JPG`).
2. Letakkan di `data/dataset_root/` atau ubah `RAW_DATA_DIR` di `config.py`.
3. (Opsional) Siapkan `data/labels.csv` dengan kolom `filename,label` untuk validasi eksternal.

### Konfigurasi

Edit `config.py` sesuai kebutuhan:

```python
DATASET_DOMAIN = "DATA-BIMA"       # Nama domain untuk penamaan arsip
RAW_DATA_DIR   = "data/dataset_root"
USE_COLOR_FUSION      = True        # Aktifkan fusi momen HSV
HSV_FUSION_WEIGHT     = 3.0        # Bobot kontribusi warna vs. tekstur
DEFAULT_CODEBOOK_SIZE = 200        # Ukuran codebook utama
SURF_HESSIAN_THRESHOLD = 500       # Threshold detektor SURF
```

### Menjalankan Tahap per Tahap

```bash
uv run python run_pipeline.py --step preprocess
uv run python run_pipeline.py --step extract
uv run python run_pipeline.py --step bovw
uv run python run_pipeline.py --step cluster
uv run python run_pipeline.py --step evaluate
uv run python run_pipeline.py --step export
uv run python run_pipeline.py --step build_model
```

### Menjalankan Semua Tahap Sekaligus

```bash
uv run python run_pipeline.py --step all
```

> Perintah `--step all` menjalankan semua tahap kecuali `tune` (untuk menghindari grid search yang lama secara tidak sengaja).

### Tahap Opsional

```bash
# Grid search hyperparameter (lama, ~75 kombinasi × step_bovw)
uv run python run_pipeline.py --step tune

# 14 skenario eksperimen batch dengan arsip otomatis
uv run python run_pipeline.py --step batch_log
```

---

## 11. Struktur Proyek

```
beefresearch-bima/
├── config.py                  # Konfigurasi sentral pipeline
├── run_pipeline.py            # Orkestrator utama (semua step ada di sini)
├── pyproject.toml             # Dependensi & metadata proyek
├── data/
│   ├── dataset_root/          # Dataset mentah (diatur di config.py)
│   └── labels.csv             # Ground truth opsional
├── src/
│   ├── preprocessing.py       # Resize, denoising, konversi, crop
│   ├── segmentation.py        # HSV thresholding & morfologi ROI
│   ├── feature_extraction.py  # SIFT/SURF + momen HSV
│   ├── bovw.py                # Codebook & histogram Hellinger
│   ├── clustering.py          # K-Means, GMM, metrik validasi
│   ├── evaluation.py          # Tabel efisiensi & komparasi fitur
│   ├── visualization.py       # Semua fungsi plot
│   └── utils.py               # Logger, I/O pickle/JSON, utils umum
├── outputs/                   # Semua hasil pipeline (di-generate otomatis)
└── archives/                  # Arsip eksperimen batch
```
