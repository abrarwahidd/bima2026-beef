"""
Konfigurasi terpusat pipeline computer vision deteksi mutu kesegaran daging sapi.
Ubah nilai di sini sesuai kebutuhan; jangan ubah logika di dalam modul src/.
"""
import os

# ---------------------------------------------------------------------------
# PATH DIREKTORI
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "dataset_root")
LABELS_CSV = os.path.join(PROJECT_ROOT, "data", "labels.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
INTERIM_DIR = os.path.join(OUTPUT_DIR, "interim")
MASKS_DIR = os.path.join(INTERIM_DIR, "masks")
PROCESSED_DIR = os.path.join(INTERIM_DIR, "processed")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# Direktori untuk inspeksi visual hasil segmentasi
SEGMENTED_VIEW_DIR = os.path.join(OUTPUT_DIR, "visualized_segments")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".JPG", ".JPEG")
FILENAME_REGEX = r"^(DAY-\d+)_([A-Za-z]+)_([A-Za-z]+)_(\d+)\.\w+$"

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
RESIZE_MAX_SIDE = 800

DENOISE_H = None      
DENOISE_H_COLOR = 7    
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

FLOODFILL_TOLERANCE = 12
FLOODFILL_MORPH_KERNEL = 7   
MIN_FOREGROUND_AREA_RATIO = 0.05  

# ---------------------------------------------------------------------------
# EKSTRAKSI FITUR & FUSI
# ---------------------------------------------------------------------------
SIFT_N_FEATURES = 0          
SURF_HESSIAN_THRESHOLD = 500  # Nilai default optimal dari hasil tuning
MAX_DESCRIPTORS_PER_IMAGE = 800

# Kontrol Fusi Momen Warna HSV
USE_COLOR_FUSION = True
HSV_FUSION_WEIGHT = 3.0       # Nilai default optimal dari hasil tuning

# ---------------------------------------------------------------------------
# BAGS OF VISUAL WORDS
# ---------------------------------------------------------------------------
CODEBOOK_SIZES = [200]
DEFAULT_CODEBOOK_SIZE = 200   # Nilai default optimal dari hasil tuning
MAX_DESCRIPTORS_FOR_CODEBOOK = 200_000
MINIBATCH_KMEANS_BATCH_SIZE = 2000

# ---------------------------------------------------------------------------
# CLUSTERING (K-Means / PCA)
# ---------------------------------------------------------------------------
N_CLUSTERS_FINAL = 2  
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# HYPERPARAMETER TUNING OTOMATIS (NATIVE)
# ---------------------------------------------------------------------------
# Rentang angka yang akan diuji secara otomatis saat menjalankan --step tune
TUNING_HESSIAN_THRESHOLDS = [400, 500, 600]
TUNING_HSV_WEIGHTS = [2.0, 2.5, 3.0]
TUNING_CODEBOOK_SIZES = [50, 100, 200]