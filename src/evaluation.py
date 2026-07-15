"""Komparasi SIFT vs SURF: efisiensi komputasi & kualitas separabilitas klaster."""
import os

import pandas as pd

import config


def build_efficiency_table(manifest, timing_records):
    """
    timing_records: dict {method: {filename: {"elapsed_sec":.., "n_keypoints":..}}}
    Mengembalikan DataFrame panjang (long-format) siap untuk boxplot/agregasi.
    """
    rows = []
    for method, per_file in timing_records.items():
        for filename, rec in per_file.items():
            rows.append({
                "method": method.upper(),
                "filename": filename,
                "elapsed_sec": rec["elapsed_sec"],
                "n_keypoints": rec["n_keypoints"],
            })
    return pd.DataFrame(rows)


def summarize_efficiency(df_efficiency):
    """Ringkasan statistik (mean/median/std) waktu & jumlah keypoint per metode."""
    summary = df_efficiency.groupby("method").agg(
        mean_elapsed_sec=("elapsed_sec", "mean"),
        median_elapsed_sec=("elapsed_sec", "median"),
        std_elapsed_sec=("elapsed_sec", "std"),
        mean_n_keypoints=("n_keypoints", "mean"),
        median_n_keypoints=("n_keypoints", "median"),
        std_n_keypoints=("n_keypoints", "std"),
        n_images=("filename", "count"),
    ).reset_index()
    return summary


def build_cluster_metrics_table(results):
    """
    results: list of dict, masing-masing hasil satu kombinasi
        {feature: "SIFT"/"SURF", method: "kmeans"/"gmm"/"agglomerative",
         codebook_size: int, **internal_metrics, **external_metrics(optional)}
    """
    return pd.DataFrame(results)


def save_table(df, filename, tables_dir=None):
    tables_dir = config.TABLES_DIR if tables_dir is None else tables_dir
    os.makedirs(tables_dir, exist_ok=True)
    out_path = os.path.join(tables_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path


def pick_best_feature(cluster_metrics_df, primary_metric="silhouette_score",
                       higher_is_better=True):
    """
    Menentukan metode fitur (SIFT/SURF) mana yang secara rata-rata memberi
    separabilitas klaster terbaik, sebagai kesimpulan kuantitatif untuk
    laporan komparasi (luaran wajib proposal).
    """
    agg = cluster_metrics_df.groupby("feature")[primary_metric].mean()
    if agg.empty:
        return None, agg
    best = agg.idxmax() if higher_is_better else agg.idxmin()
    return best, agg
