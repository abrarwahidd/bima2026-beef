"""Utilitas umum: logging, I/O pickle/json, parsing metadata dari nama file."""
import glob
import json
import logging
import os
import pickle
import re
import sys

import config


def get_logger(name="pipeline"):
    """Logger tunggal yang menulis ke konsol dan ke outputs/logs/pipeline.log."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # sudah dikonfigurasi sebelumnya, hindari duplikasi handler

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        os.path.join(config.LOGS_DIR, "pipeline.log"), encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def ensure_dirs():
    """Membuat seluruh direktori output yang dibutuhkan pipeline jika belum ada."""
    for d in (
        config.OUTPUT_DIR,
        config.INTERIM_DIR,
        config.MASKS_DIR,
        config.PROCESSED_DIR,
        config.FIGURES_DIR,
        config.TABLES_DIR,
        config.LOGS_DIR,
    ):
        os.makedirs(d, exist_ok=True)


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def list_images(raw_data_dir, extensions):
    """Mencari seluruh file citra secara rekursif di bawah raw_data_dir."""
    paths = []
    for ext in extensions:
        pattern = os.path.join(raw_data_dir, "**", "*" + ext)
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def parse_filename_metadata(filepath, pattern):
    """
    Mem-parsing metadata dari nama file sesuai konvensi:
    [HARI]_[DOMAIN]_[KODE_DAGING]_[NOMOR_URUT].JPG
    Contoh: DAY-1_RPH_HLR_001.JPG

    Mengembalikan dict berisi day, domain, cut_code, seq_number, filename.
    Jika nama file tidak cocok pola, field yang tidak terparsing diisi None
    dan pipeline tetap berjalan (tidak crash), namun dicatat sebagai warning
    oleh caller.
    """
    filename = os.path.basename(filepath)
    match = re.match(pattern, filename)
    if match is None:
        return {
            "filename": filename,
            "filepath": filepath,
            "day": None,
            "domain": None,
            "cut_code": None,
            "seq_number": None,
            "parsed_ok": False,
        }
    day, domain, cut_code, seq_number = match.groups()
    return {
        "filename": filename,
        "filepath": filepath,
        "day": day,
        "domain": domain.upper(),
        "cut_code": cut_code.upper(),
        "seq_number": int(seq_number),
        "parsed_ok": True,
    }


def load_labels_csv(path):
    """
    Memuat label ground truth opsional dari CSV (kolom: filename,label).
    Mengembalikan dict {filename: label}, atau {} jika file tidak ditemukan.
    """
    if path is None or not os.path.isfile(path):
        return {}
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "filename" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "labels.csv harus memiliki kolom 'filename' dan 'label'. "
            "Kolom yang ditemukan: {}".format(list(df.columns))
        )
    return dict(zip(df["filename"], df["label"]))
