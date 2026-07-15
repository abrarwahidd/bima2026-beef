# Pipeline Computer Vision: Deteksi Mutu Kesegaran Daging Sapi Multidomain

Pipeline **unsupervised** (SIFT/SURF -> Bags of Visual Words -> Clustering) untuk
dataset citra daging sapi multidomain (RPH & UMKM), seluruh proses klasifikasi mutu dilakukan tanpa pelatihan supervised
(tanpa SVM/KNN), murni berbasis unsupervised learning.

## 1. Setup Environment (uv, Python 3.7, OpenCV 3.4.2.16)

```bash
# Install python 3.7 lewat uv (jika belum ada di mesin)
uv python install 3.7

# Buat & sync environment sesuai pyproject.toml
uv sync

# (opsional) verifikasi SIFT & SURF tersedia (harus tercetak True True)
uv run python -c "import cv2; s=cv2.xfeatures2d.SIFT_create(); f=cv2.xfeatures2d.SURF_create(); print(True, True)"
```

> Catatan: jika `uv python install 3.7` gagal karena python-build-standalone tidak
> lagi menyediakan build 3.7 di platform Anda, gunakan pyenv/conda untuk
> menyediakan interpreter 3.7, lalu arahkan uv ke sana dengan
> `uv venv --python /path/to/python3.7` sebelum `uv sync`.

## 2. Siapkan Data

Susun dataset mentah sesuai struktur pada README dataset (folder `DAY-1/`, `DAY-2/`,
penamaan `[HARI]_[DOMAIN]_[KODE]_[NOMOR].JPG`). Atur path pada `config.py`
(`RAW_DATA_DIR`).

Jika Anda sudah punya sebagian label ground truth (dari pelabelan awal tim
peneliti / Peternakan), siapkan file CSV opsional `labels.csv` dengan kolom
`filename,label` (label: `segar` / `tidak_segar`) — dipakai HANYA untuk
validasi eksternal klaster (ARI/NMI/Purity), bukan untuk training.

## 3. Menjalankan Pipeline

Pipeline terbagi menjadi 5 tahap, masing-masing bisa dijalankan terpisah
(hasil antara di-cache di `outputs/interim/`) atau sekaligus dengan `all`.

```bash
uv run python run_pipeline.py --step preprocess
uv run python run_pipeline.py --step extract
uv run python run_pipeline.py --step bovw
uv run python run_pipeline.py --step cluster
uv run python run_pipeline.py --step evaluate

# atau jalankan semua tahap sekaligus
uv run python run_pipeline.py --step all
```

## 4. Struktur Output

```
outputs/
├── interim/                     # cache pickle antar tahap (boleh dihapus & rerun)
│   ├── manifest.pkl             # metadata semua citra (hari/domain/kode/label)
│   ├── masks/                   # mask ROI hasil segmentasi (npz per citra)
│   ├── descriptors_sift.pkl
│   ├── descriptors_surf.pkl
│   ├── codebook_sift.pkl
│   ├── codebook_surf.pkl
│   ├── histograms_sift.npy
│   └── histograms_surf.npy
├── figures/                     # semua grafik (PCA/TSNE, perbandingan metrik, dsb.)
├── tables/                      # semua tabel hasil (CSV) siap ditempel ke laporan
└── logs/
    └── pipeline.log
```

## 5. Tahapan Pipeline (ringkas)

1. **Preprocessing & Segmentasi ROI** (`src/preprocessing.py`, `src/segmentation.py`)
   Resize, denoising, konversi HSV, dan **segmentasi latar belakang berbasis
   flood-fill dari tepi citra** (bekerja untuk backdrop hitam Day-1 maupun
   karton biru muda Day-2 tanpa perlu aturan warna berbeda per hari) sehingga
   fitur yang diekstraksi terfokus pada permukaan daging, bukan latar
   belakang yang kebetulan berkorelasi dengan hari/tahap kesegaran.

2. **Ekstraksi Fitur Lokal** (`src/feature_extraction.py`)
   SIFT dan SURF dijalankan terpisah, keypoint dibatasi hanya pada area mask
   ROI daging (parameter `mask` di `detectAndCompute`).

3. **Bags of Visual Words** (`src/bovw.py`)
   Codebook dibangun dengan `MiniBatchKMeans` (beberapa ukuran cluster diuji:
   50/100/150/200 sesuai indikator capaian proposal), setiap citra direpresentasikan
   sebagai histogram fitur ternormalisasi.

4. **Clustering Unsupervised** (`src/clustering.py`)
   K-Means (k=2), Gaussian Mixture, dan Agglomerative Clustering dibandingkan
   untuk membagi citra ke 2 klaster akhir (Segar/Tidak Segar).

5. **Evaluasi & Komparasi SIFT vs SURF** (`src/evaluation.py`, `src/visualization.py`)
   Validasi internal (Silhouette, Davies-Bouldin, Calinski-Harabasz), validasi
   eksternal opsional (ARI/NMI/Purity bila ada `labels.csv`), efisiensi
   komputasi (waktu ekstraksi & jumlah keypoint rata-rata), serta visualisasi
   PCA/t-SNE dari histogram BoVW.
