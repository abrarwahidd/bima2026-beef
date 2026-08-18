#!/usr/bin/env python
"""
Orkestrator pipeline unsupervised computer vision untuk deteksi mutu
kesegaran daging sapi multidomain (SIFT/SURF -> BoVW -> Clustering).

Code pemakaian:
    uv run python run_pipeline.py --step preprocess
    uv run python run_pipeline.py --step extract
    uv run python run_pipeline.py --step bovw
    uv run python run_pipeline.py --step cluster
    uv run python run_pipeline.py --step evaluate
    uv run python run_pipeline.py --step tune
    uv run python run_pipeline.py --step all
"""
import argparse
import os
import shutil
import time
import json
import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from typing import Dict, Any
import joblib

import config
from src import bovw, clustering, evaluation, feature_extraction, preprocessing, utils, visualization

logger = utils.get_logger("run_pipeline")


# TAHAP 1: PREPROCESSING & SEGMENTASI ROI
def step_preprocess():
    logger.info("TAHAP 1 - PREPROCESSING & SEGMENTASI ROI")
    utils.ensure_dirs()

    image_paths = utils.list_images(config.RAW_DATA_DIR, config.IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"Tidak ada citra ditemukan di {config.RAW_DATA_DIR}.")
    logger.info("Ditemukan %d citra mentah.", len(image_paths))

    labels_map = utils.load_labels_csv(config.LABELS_CSV)
    
    manifest = []
    n_parse_fail = n_read_fail = n_fallback_seg = 0

    for filepath in tqdm(image_paths, desc="Preprocessing"):
        meta = utils.parse_filename_metadata(filepath, config.FILENAME_REGEX)
        if not meta["parsed_ok"]:
            n_parse_fail += 1

        try:
            result = preprocessing.preprocess_single_image(filepath)
        except Exception as exc:
            logger.error("Gagal memproses %s: %s", meta["filename"], exc)
            n_read_fail += 1
            continue

        if result is None:
            n_read_fail += 1
            continue

        if result["seg_meta"]["fallback_used"]:
            n_fallback_seg += 1

        preprocessing.save_processed(result["gray"], result["hsv"], result["mask"], meta["filename"])

        meta["label"] = labels_map.get(meta["filename"])
        meta["foreground_ratio"] = result["seg_meta"]["foreground_ratio"]
        meta["segmentation_fallback"] = result["seg_meta"]["fallback_used"]
        meta["image_shape"] = result["shape"]
        manifest.append(meta)

    utils.save_pickle(manifest, os.path.join(config.INTERIM_DIR, "manifest.pkl"))
    df = pd.DataFrame(manifest)
    evaluation.save_table(df, "manifest_summary.csv")
    logger.info("Selesai preprocessing: %d sukses.", len(manifest))


# TAHAP 2: EKSTRAKSI FITUR LOKAL (SIFT & SURF)
def step_extract():
    logger.info("TAHAP 2 - EKSTRAKSI FITUR LOKAL (SIFT & SURF)")
    manifest = utils.load_pickle(os.path.join(config.INTERIM_DIR, "manifest.pkl"))

    for method in ("sift", "surf"):
        logger.info("-- Ekstraksi %s --", method.upper())
        descriptors_dict = {}
        hsv_dict = {} 
        timing_records = {}

        for entry in tqdm(manifest, desc=f"Extract {method.upper()}"):
            filename = entry["filename"]
            gray, hsv, mask = preprocessing.load_processed(filename)

            descriptors, hsv_moments, n_kp, elapsed = feature_extraction.extract_features(
                gray, hsv, mask, method
            )

            descriptors_dict[filename] = descriptors
            if hsv_moments is not None:
                hsv_dict[filename] = hsv_moments
            timing_records[filename] = {"elapsed_sec": elapsed, "n_keypoints": n_kp}

        utils.save_pickle(descriptors_dict, os.path.join(config.INTERIM_DIR, f"descriptors_{method}.pkl"))
        if config.USE_COLOR_FUSION:
            utils.save_pickle(hsv_dict, os.path.join(config.INTERIM_DIR, f"hsv_{method}.pkl"))
        utils.save_pickle(timing_records, os.path.join(config.INTERIM_DIR, f"timing_{method}.pkl"))

    timing_sift = utils.load_pickle(os.path.join(config.INTERIM_DIR, "timing_sift.pkl"))
    timing_surf = utils.load_pickle(os.path.join(config.INTERIM_DIR, "timing_surf.pkl"))
    df_eff = evaluation.build_efficiency_table(manifest, {"sift": timing_sift, "surf": timing_surf})
    evaluation.save_table(df_eff, "efficiency_raw.csv")
    evaluation.save_table(evaluation.summarize_efficiency(df_eff), "efficiency_summary.csv")


# TAHAP 3: BAGS OF VISUAL WORDS & EARLY FUSION
def step_bovw():
    logger.info("TAHAP 3 - BAGS OF VISUAL WORDS & EARLY FUSION")
    manifest = utils.load_pickle(os.path.join(config.INTERIM_DIR, "manifest.pkl"))
    filenames = [e["filename"] for e in manifest]

    for method in ("sift", "surf"):
        logger.info("-- BoVW untuk %s --", method.upper())
        descriptors_dict = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"descriptors_{method}.pkl"))
        
        hsv_dict = None
        if config.USE_COLOR_FUSION:
            hsv_path = os.path.join(config.INTERIM_DIR, f"hsv_{method}.pkl")
            if os.path.exists(hsv_path):
                hsv_dict = utils.load_pickle(hsv_path)
                
                # Menormalkan fitur HSV dengan MinMaxScaler (0-1) - fit di SELURUH
                # dataset (benar, bukan per-citra).
                all_hsv_values = list(hsv_dict.values())
                scaler = MinMaxScaler()
                scaler.fit(all_hsv_values)
                
                for f_name in hsv_dict.keys():
                    hsv_dict[f_name] = scaler.transform(hsv_dict[f_name].reshape(1, -1))[0]

                # ----------------------------------------------------------
                # MinMaxScaler saja TIDAK cukup menyamakan skala
                # dengan blok tekstur. bovw_hist (Hellinger) punya norma-L2 = 1
                # per baris (properti transformasi sqrt(L1-normalized)),
                # sedangkan vektor HSV hasil MinMax bisa punya norma-L2
                # berbeda-beda (tergantung berapa banyak dimensi yang dekat
                # nilai maksimum). Tanpa disamakan, HSV_FUSION_WEIGHT bukan
                # lagi "bobot proporsi" yang terkendali - efeknya bisa jauh
                # lebih besar/kecil dari yang dimaksud, dan mendominasi PCA
                # yang dijalankan setelahnya (fitur bervarians besar otomatis
                # mendominasi komponen utama).
                #
                # normalisasi setiap vektor HSV ke norma-L2 = 1 SETELAH
                # MinMax, baru dikalikan HSV_FUSION_WEIGHT. Sekarang
                # HSV_FUSION_WEIGHT=1.0 berarti "kontribusi warna setara
                # dengan tekstur", weight=3.0 berarti "warna diberi bobot 3x
                # lipat kontribusi tekstur" - terkendali dan bisa diinterpretasi.
                # ----------------------------------------------------------
                for f_name in hsv_dict.keys():
                    vec = hsv_dict[f_name]
                    norm = np.linalg.norm(vec)
                    hsv_dict[f_name] = (vec / norm) if norm > 0 else vec

                logger.info(
                    "Fitur HSV dinormalisasi (MinMaxScaler lalu L2-normalize per "
                    "baris) supaya skalanya setara dengan blok tekstur (Hellinger, "
                    "norma L2=1) sebelum dikalikan HSV_FUSION_WEIGHT=%.2f.",
                    config.HSV_FUSION_WEIGHT,
                )
                
        all_descriptors = [descriptors_dict.get(f) for f in filenames]
        sample = bovw.sample_descriptors_for_codebook(
            all_descriptors, config.MAX_DESCRIPTORS_FOR_CODEBOOK, config.RANDOM_STATE
        )

        codebooks = {}
        histograms = {}
        for k in config.CODEBOOK_SIZES:
            codebook = bovw.build_codebook(sample, k)
            codebooks[k] = codebook

            hist_matrix = []
            for f in filenames:
                desc = descriptors_dict.get(f)
                bovw_hist = bovw.compute_histogram(desc, codebook, normalize="hellinger")
                
                if config.USE_COLOR_FUSION and hsv_dict is not None:
                    hsv_feats = hsv_dict.get(f, np.zeros(9, dtype=np.float32))
                    hsv_weighted = hsv_feats * config.HSV_FUSION_WEIGHT
                    final_vector = np.concatenate((bovw_hist, hsv_weighted))
                else:
                    final_vector = bovw_hist
                
                hist_matrix.append(final_vector)
                
            histograms[k] = np.vstack(hist_matrix)

        utils.save_pickle(codebooks, os.path.join(config.INTERIM_DIR, f"codebooks_{method}.pkl"))
        utils.save_pickle(histograms, os.path.join(config.INTERIM_DIR, f"histograms_{method}.pkl"))


# TAHAP 4: CLUSTERING UNSUPERVISED DENGAN PCA
def step_cluster():
    logger.info("TAHAP 4 - CLUSTERING UNSUPERVISED")
    manifest = utils.load_pickle(os.path.join(config.INTERIM_DIR, "manifest.pkl"))
    true_labels = [e.get("label") for e in manifest]
    has_ground_truth = all(l is not None for l in true_labels) and len(true_labels) > 0

    all_results = []
    all_cluster_labels = {}

    for method in ("sift", "surf"):
        histograms = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"histograms_{method}.pkl"))

        # --- 4a. Sensitivitas ukuran codebook (KMeans, ruang fitur MENTAH) ---
        # Analisis terpisah: menguji pengaruh granularitas codebook, BUKAN
        # membandingkan algoritma clustering. Ditandai feature_space="raw"
        # agar tidak tercampur dengan 4b saat dianalisis.
        for k_codebook, X in histograms.items():
            labels, _ = clustering.run_kmeans(X)
            metrics = clustering.internal_validation(X, labels)
            metrics.update({
                "feature": method.upper(), "method": "kmeans",
                "codebook_size": k_codebook, "feature_space": "raw",
            })
            all_results.append(metrics)

        # --- 4b. Perbandingan ALGORITMA clustering pada codebook default -
        # KETIGA metode (KMeans, GMM, Agglomerative) dijalankan di RUANG
        # FITUR YANG SAMA (setelah PCA), supaya perbandingan adil. Sebelumnya
        # KMeans jalan di data mentah sementara GMM/Agglomerative di PCA -
        # bukan perbandingan algoritma yang valid, melainkan perbandingan
        # algoritma SEKALIGUS ruang fitur tercampur.
        X_default = histograms[config.DEFAULT_CODEBOOK_SIZE]

        logger.info("Menerapkan PCA sebelum perbandingan algoritma clustering...")
        pca = PCA(n_components=min(30, X_default.shape[0] - 1, X_default.shape[1] - 1), random_state=config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_default)
        logger.info("Dimensi matriks dikompresi dari %d menjadi %d komponen utama "
                    "(variansi terjelaskan=%.1f%%).",
                    X_default.shape[1], X_pca.shape[1],
                    pca.explained_variance_ratio_.sum() * 100)

        for cname, cfunc in clustering.CLUSTERING_METHODS.items():
            labels, _ = cfunc(X_pca)
            metrics = clustering.internal_validation(X_pca, labels)
            metrics.update({
                "feature": method.upper(), "method": cname,
                "codebook_size": config.DEFAULT_CODEBOOK_SIZE, "feature_space": "pca",
            })
            all_results.append(metrics)
            all_cluster_labels[(method, cname)] = labels

        if has_ground_truth:
            for (m, cname), labels in list(all_cluster_labels.items()):
                if m != method:
                    continue
                ext_metrics = clustering.external_validation(true_labels, labels)
                for row in all_results:
                    if (row["feature"] == method.upper() and row["method"] == cname
                            and row["codebook_size"] == config.DEFAULT_CODEBOOK_SIZE
                            and row["feature_space"] == "pca"):
                        row.update(ext_metrics)

    df_results = evaluation.build_cluster_metrics_table(all_results)
    evaluation.save_table(df_results, "cluster_metrics_all.csv")
    utils.save_pickle(all_cluster_labels, os.path.join(config.INTERIM_DIR, "cluster_labels.pkl"))
    logger.info("Selesai clustering.")

    # --- 4c. Uji stabilitas klaster (mitigasi metric-chasing) ---
    # Silhouette tunggal dari satu kali fit rentan terlalu optimis. Di sini
    # klaster diulang pada 10 subsample acak (80% data) untuk melihat apakah
    # struktur klaster stabil (ARI tinggi & konsisten), bukan kebetulan satu
    # kali fit. Dilaporkan sebagai mean +/- std, lebih jujur untuk laporan
    # dibanding satu angka hasil pencarian hyperparameter.
    stability_rows = []
    rng = np.random.RandomState(config.RANDOM_STATE)
    for method in ("sift", "surf"):
        histograms = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"histograms_{method}.pkl"))
        X_default = histograms[config.DEFAULT_CODEBOOK_SIZE]
        pca = PCA(n_components=min(30, X_default.shape[0] - 1, X_default.shape[1] - 1), random_state=config.RANDOM_STATE)
        X_pca_full = pca.fit_transform(X_default)
        full_labels, _ = clustering.run_kmeans(X_pca_full)

        n = len(X_pca_full)
        sil_scores, ari_scores = [], []
        for trial in range(10):
            idx = rng.choice(n, size=int(n * 0.8), replace=False)
            sub_labels, _ = clustering.run_kmeans(X_pca_full[idx], random_state=trial)
            sub_metrics = clustering.internal_validation(X_pca_full[idx], sub_labels)
            if not np.isnan(sub_metrics["silhouette_score"]):
                sil_scores.append(sub_metrics["silhouette_score"])
            ari_scores.append(
                clustering.external_validation(full_labels[idx], sub_labels)["adjusted_rand_index"]
            )

        stability_rows.append({
            "feature": method.upper(),
            "silhouette_mean": float(np.mean(sil_scores)) if sil_scores else float("nan"),
            "silhouette_std": float(np.std(sil_scores)) if sil_scores else float("nan"),
            "ari_vs_full_mean": float(np.mean(ari_scores)),
            "ari_vs_full_std": float(np.std(ari_scores)),
            "n_trials": 10,
        })
        logger.info(
            "[%s] Stabilitas klaster (10x subsample 80%%): silhouette=%.3f±%.3f, "
            "ARI vs full-data=%.3f±%.3f (ARI mendekati 1 = klaster stabil)",
            method.upper(), stability_rows[-1]["silhouette_mean"],
            stability_rows[-1]["silhouette_std"], stability_rows[-1]["ari_vs_full_mean"],
            stability_rows[-1]["ari_vs_full_std"],
        )

    evaluation.save_table(pd.DataFrame(stability_rows), "cluster_stability.csv")


# TAHAP 5: EVALUASI & KOMPARASI SIFT vs SURF
def step_evaluate():
    logger.info("TAHAP 5 - EVALUASI & KOMPARASI SIFT vs SURF")
    manifest_path = os.path.join(config.INTERIM_DIR, "manifest.pkl")
    if os.path.exists(manifest_path):
        manifest = utils.load_pickle(manifest_path)
        if len(manifest) > 0:
            # Mengambil citra indeks pertama (atau indeks acak representatif)
            sample_entry = manifest[0] 
            sample_filename = sample_entry["filename"]
            gray_img, hsv_img, mask_img = preprocessing.load_processed(sample_filename)
            
            # Memanggil fungsi plot visual
            out_proof = visualization.plot_keypoint_proof(
                gray_img, mask_img, save_name=f"proof_keypoints_{sample_filename}.png"
            )
            logger.info("Bukti visual keypoint berhasil disimpan di: %s", out_proof)

    df_results = pd.read_csv(os.path.join(config.TABLES_DIR, "cluster_metrics_all.csv"))
    df_eff = pd.read_csv(os.path.join(config.TABLES_DIR, "efficiency_raw.csv"))

    logger.info("Menghasilkan visualisasi Ablation Study (PCA Impact)...")
    visualization.plot_pca_ablation_impact(
        df_results, 
        save_name="pca_impact_ablation.png"
    )

    logger.info("Menghasilkan Radar Chart untuk Komparasi Keseluruhan")
    visualization.plot_overall_radar_comparison(df_results, df_eff)

    df_default = df_results[
        (df_results["codebook_size"] == config.DEFAULT_CODEBOOK_SIZE)
        & (df_results["feature_space"] == "pca")
    ]

    visualization.plot_metric_comparison(
        df_default, "silhouette_score", "compare_silhouette.png",
        title=f"Perbandingan Silhouette Score (codebook k={config.DEFAULT_CODEBOOK_SIZE})",
        higher_is_better=True,
    )
    visualization.plot_metric_comparison(
        df_default, "davies_bouldin_score", "compare_davies_bouldin.png",
        title=f"Perbandingan Davies-Bouldin Index (codebook k={config.DEFAULT_CODEBOOK_SIZE})",
        higher_is_better=False,
    )
    visualization.plot_metric_comparison(
        df_default, "calinski_harabasz_score", "compare_calinski_harabasz.png",
        title=f"Perbandingan Calinski-Harabasz Index (codebook k={config.DEFAULT_CODEBOOK_SIZE})",
        higher_is_better=True,
    )
    visualization.plot_efficiency_comparison(df_eff)

    best_row = df_default.loc[df_default["silhouette_score"].idxmax()]
    best_feature = best_row["feature"].lower()
    best_method = best_row["method"]

    histograms = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"histograms_{best_feature}.pkl"))
    cluster_labels = utils.load_pickle(os.path.join(config.INTERIM_DIR, "cluster_labels.pkl"))
    X_best = histograms[config.DEFAULT_CODEBOOK_SIZE]
    labels_best = cluster_labels[(best_feature, best_method)]

    visualization.plot_embedding_2d(
        X_best, labels_best, f"{best_feature.upper()} + {best_method}",
        "pca_best_combination.png",
    )
    visualization.plot_tsne_2d(
        X_best, labels_best, f"{best_feature.upper()} + {best_method}",
        "tsne_best_combination.png",
    )

    best_feature_overall, agg = evaluation.pick_best_feature(
        df_default, primary_metric="silhouette_score", higher_is_better=True
    )
    
    summary = {
        "best_combination_by_silhouette": {
            "feature": best_feature.upper(),
            "clustering_method": best_method,
            "silhouette_score": float(best_row["silhouette_score"]),
        },
        "best_feature_overall_avg_silhouette": best_feature_overall,
        "avg_silhouette_per_feature": agg.to_dict(),
        "codebook_size_used_for_main_comparison": config.DEFAULT_CODEBOOK_SIZE,
    }
    utils.save_json(summary, os.path.join(config.TABLES_DIR, "final_summary.json"))
    logger.info("Ringkasan akhir disimpan ke outputs/tables/final_summary.json")


# TAHAP 6: HYPERPARAMETER TUNING NATIVE (DEEP SEARCH)
def step_tune():
    logger.info("TAHAP 6 - HYPERPARAMETER TUNING (DEEP SEARCH)")
    
    # Tuning parameter baru untuk memaksimalkan Silhouette > 0.5
    tuning_pca_components = [15, 20, 25, 30, 35]
    tuning_gmm_covariances = ['full', 'tied', 'diag']
    
    config.SURF_HESSIAN_THRESHOLD = 500
    config.CODEBOOK_SIZES = [200]
    config.DEFAULT_CODEBOOK_SIZE = 200
    
    # Bobot HSV dengan rentang desimal yang lebih rapat
    tuning_hsv_weights = [2.5, 2.8, 3.0, 3.2, 3.5]
    
    results = []
    total_runs = len(tuning_hsv_weights) * len(tuning_pca_components) * len(tuning_gmm_covariances)
    current_run = 1
    
    # Ekstraksi dilakukan satu kali saja di awal karena parameter SIFT/SURF tidak berubah
    logger.info("Mengeksekusi ekstraksi fitur dasar...")
    step_extract()
    
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    import json
    from src import clustering
    
    for w in tuning_hsv_weights:
        config.HSV_FUSION_WEIGHT = w
        logger.info("\n--- Menguji Bobot HSV: %.1f ---", w)
        step_bovw() # Hitung ulang BoVW karena bobot warna berubah
        
        histograms = utils.load_pickle(os.path.join(config.INTERIM_DIR, "histograms_surf.pkl"))
        X_default = histograms[config.DEFAULT_CODEBOOK_SIZE]
        
        for n_comp in tuning_pca_components:
            # Terapkan PCA dengan dimensi yang bervariasi
            pca = PCA(n_components=n_comp, random_state=config.RANDOM_STATE)
            X_pca = pca.fit_transform(X_default)
            
            for cov_type in tuning_gmm_covariances:
                logger.info("[%d/%d] Komponen PCA: %d | GMM Covariance: %s", 
                            current_run, total_runs, n_comp, cov_type)
                
                # Modifikasi GMM secara dinamis
                gmm = GaussianMixture(n_components=2, covariance_type=cov_type, 
                                      random_state=config.RANDOM_STATE, n_init=10)
                labels = gmm.fit_predict(X_pca)
                
                # Hitung skor metrik
                metrics = clustering.internal_validation(X_pca, labels)
                
                results.append({
                    "HSV_WEIGHT": w,
                    "PCA_COMPONENTS": n_comp,
                    "GMM_COVARIANCE": cov_type,
                    "SILHOUETTE_SCORE": metrics.get("silhouette_score", 0),
                    "DAVIES_BOULDIN": metrics.get("davies_bouldin_score", 0)
                })
                
                current_run += 1
                
    # Urutkan berdasarkan Silhouette Score tertinggi, lalu Davies-Bouldin terendah
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by=["SILHOUETTE_SCORE", "DAVIES_BOULDIN"], 
        ascending=[False, True]
    )
    
    out_csv = os.path.join(config.TABLES_DIR, "tuning_results_deep_search.csv")
    df_results.to_csv(out_csv, index=False)
    
    logger.info("Pencarian Selesai! Hasil disimpan di: %s", out_csv)
    print("\n" + "="*60)
    print("KOMBINASI TERBAIK (DEEP SEARCH):")
    print(df_results.head(1).to_string(index=False))
    print("="*60)


    logger.warning(
        "PERINGATAN METODOLOGIS: SILHOUETTE_SCORE pada tabel deep search "
        "adalah hasil PENCARIAN yang memaksimalkan metrik itu sendiri -> bias "
        "optimis. JANGAN dilaporkan langsung sebagai bukti separabilitas. "
        "Menjalankan uji stabilitas (subsample berulang) untuk konfigurasi "
        "terbaik sebagai angka yang lebih jujur untuk laporan..."
    )

    best = df_results.iloc[0]
    config.HSV_FUSION_WEIGHT = float(best["HSV_WEIGHT"])
    step_bovw()
    histograms = utils.load_pickle(os.path.join(config.INTERIM_DIR, "histograms_surf.pkl"))
    X_default = histograms[config.DEFAULT_CODEBOOK_SIZE]
    pca_best = PCA(n_components=int(best["PCA_COMPONENTS"]), random_state=config.RANDOM_STATE)
    X_pca_best = pca_best.fit_transform(X_default)
    gmm_best = GaussianMixture(
        n_components=2, covariance_type=best["GMM_COVARIANCE"],
        random_state=config.RANDOM_STATE, n_init=10,
    )
    full_labels = gmm_best.fit_predict(X_pca_best)

    rng = np.random.RandomState(config.RANDOM_STATE)
    n = len(X_pca_best)
    sil_scores, ari_scores = [], []
    for trial in range(10):
        idx = rng.choice(n, size=int(n * 0.8), replace=False)
        gmm_trial = GaussianMixture(
            n_components=2, covariance_type=best["GMM_COVARIANCE"],
            random_state=trial, n_init=10,
        )
        sub_labels = gmm_trial.fit_predict(X_pca_best[idx])
        sub_metrics = clustering.internal_validation(X_pca_best[idx], sub_labels)
        if not np.isnan(sub_metrics["silhouette_score"]):
            sil_scores.append(sub_metrics["silhouette_score"])
        ari_scores.append(
            clustering.external_validation(full_labels[idx], sub_labels)["adjusted_rand_index"]
        )

    logger.info(
        "Uji stabilitas konfigurasi terbaik (10x subsample 80%%): "
        "silhouette=%.3f±%.3f, ARI vs full-data=%.3f±%.3f",
        np.mean(sil_scores), np.std(sil_scores), np.mean(ari_scores), np.std(ari_scores),
    )
    print("\n" + "="*60)
    print("ANGKA YANG SEBAIKNYA DIPAKAI DI LAPORAN (bukan hasil pencarian mentah):")
    print(f"Silhouette (mean ± std, 10x subsample) = {np.mean(sil_scores):.4f} ± {np.std(sil_scores):.4f}")
    print(f"ARI vs klaster data penuh (stabilitas)  = {np.mean(ari_scores):.4f} ± {np.std(ari_scores):.4f}")
    print("ARI mendekati 1 = klaster stabil terhadap subsampling; jauh di bawah 1 = tidak stabil.")
    print("="*60)


def step_export():
    logger.info("TAHAP 7 - EKSPOR TABEL PREDIKSI")
    
    # 1. Memuat metadata gambar dan hasil klasterisasi
    manifest_path = os.path.join(config.INTERIM_DIR, "manifest.pkl")
    labels_path = os.path.join(config.INTERIM_DIR, "cluster_labels.pkl")
    
    if not os.path.exists(manifest_path) or not os.path.exists(labels_path):
        logger.error("Data klaster belum tersedia. Jalankan --step cluster terlebih dahulu.")
        return
        
    manifest = utils.load_pickle(manifest_path)
    cluster_labels = utils.load_pickle(labels_path)
    
    # 2. Mengambil hasil dari model terbaik kita (SIFT + KMeans)
    # KMeans terbukti paling optimal menangani distribusi elips dari data ini
    best_labels = cluster_labels.get(("sift", "kmeans"))
    
    if best_labels is None:
        logger.error("Label SIFT + KMeans tidak ditemukan.")
        return
    
    # 3. Menyusun data menjadi format tabular
    report_data = []
    for idx, entry in enumerate(manifest):
        report_data.append({
            "Nama File": entry["filename"],
            "Domain Asal": entry["domain"],
            "Hari Ke-": entry["day"],
            "Jenis Potongan": entry["cut_code"],
            "ID Klaster Kmeans": best_labels[idx],
            # Kamu bisa mengubah penamaan ini nanti setelah memvalidasi visualnya
            "Asumsi Kondisi": "Klaster A" if best_labels[idx] == 0 else "Klaster B" 
        })
        
    df_report = pd.DataFrame(report_data)
    
    # 4. Menyimpan sebagai CSV agar mudah dibuka di Excel
    out_path = os.path.join(config.TABLES_DIR, "Laporan_Prediksi_SIFT_KMEANS.csv")
    df_report.to_csv(out_path, index=False)
    
    logger.info("Berhasil mengekspor %d baris data ke %s", len(df_report), out_path)
    print("\n" + "="*70)
    print("CUPLIKAN HASIL PREDIKSI (SIFT + KMEANS):")
    print(df_report.head(10).to_string(index=False))
    print("="*70)

    if df_report['Hari Ke-'].isnull().all():
        logger.warning(
            "Dataset eksternal (publik) terdeteksi. Metadata temporal ('Hari Ke-') tidak "
            "tersedia. Pembuatan grafik Distribusi Temporal secara otomatis dilewati."
        )
    else:
        logger.info("Menghasilkan visualisasi Distribusi Temporal (DAY-1 vs DAY-2)...")
        visualization.plot_temporal_distribution(
            df_report, 
            save_name=f"temporal_cluster_{config.DATASET_DOMAIN}.png"
        )
        logger.info("Grafik distribusi klaster otonom tersimpan di direktori figures.")


def _execute_single_scenario(sc: Dict[str, Any], dict_sift: dict, dict_surf: dict, 
                             code_dir: str, results_dir: str) -> dict:
    """
    Helper function (Private Method): 
    Mengeksekusi satu iterasi eksperimen otonom (ekstraksi matriks laten, 
    reduksi dimensi, dan klastering) serta mengembalikan matriks evaluasinya.
    """
    # 1. Penentuan target dictionary berdasarkan skenario
    target_dict = dict_sift if sc["feat"] == "SIFT" else dict_surf
    X_current = target_dict[sc["k"]].copy()
    
    # 2. Operasi Ablasi Fusi Warna (HSV)
    if not sc["use_hsv"]:
        X_current = X_current[:, :-9]
        
    # 3. Operasi Reduksi Dimensi (PCA)
    if sc["space"] == "pca":
        # Mencegah komponen PCA melampaui dimensi sampel
        n_comp = min(sc["pca_comp"], X_current.shape[1] - 1)
        pca = PCA(n_components=n_comp, random_state=config.RANDOM_STATE)
        X_current = pca.fit_transform(X_current)
        
    # 4. Pemanggilan Algoritma Klastering Otonom
    if sc["algo"] == "kmeans":
        labels, _ = clustering.run_kmeans(X_current)
    else:
        labels, _ = clustering.run_gmm(X_current)
        
    # 5. Kalkulasi Metrik Separabilitas Spasial
    metrics = clustering.internal_validation(X_current, labels)
    
    return metrics


#EKSPERIMEN SENSITIVITAS ANALISIS
def step_batch_experiment() -> None:
    logger.info("=== TAHAP EKSPERIMEN BATCH & SENSITIVITY ANALYSIS ===")
    
    sift_path = os.path.join(config.INTERIM_DIR, "histograms_sift.pkl")
    surf_path = os.path.join(config.INTERIM_DIR, "histograms_surf.pkl")
    efficiency_path = os.path.join(config.TABLES_DIR, "efficiency_summary.csv")
    
    if not os.path.exists(sift_path) or not os.path.exists(efficiency_path):
        logger.warning("Fitur dasar atau data efisiensi belum ada. Harap jalankan ekstraksi terlebih dahulu.")
        return
        
    dict_sift = utils.load_pickle(sift_path)
    dict_surf = utils.load_pickle(surf_path)
    
    # 1. Pemuatan dan Pemetaan Data Efisiensi
    df_eff = pd.read_csv(efficiency_path)
    df_eff['method'] = df_eff['method'].str.upper()
    
    eff_lookup = {}
    for _, row in df_eff.iterrows():
        eff_lookup[row['method']] = {
            'elapsed_sec': round(row['mean_elapsed_sec'], 4),
            'std_sec': round(row['std_elapsed_sec'], 4),
            'n_keypoints': int(round(row['mean_n_keypoints'], 0))
        }
    
    # 2. Definisi 14 Skenario Eksperimen (SINTA 1/2 Standard)
    scenarios = [
        {"id": "EXP-001", "feat": "SIFT", "k": 200, "use_hsv": True, "space": "raw", "algo": "kmeans", "pca_comp": None},
        {"id": "EXP-002", "feat": "SURF", "k": 200, "use_hsv": True, "space": "raw", "algo": "kmeans", "pca_comp": None},
        {"id": "EXP-003", "feat": "SIFT", "k": 200, "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-004", "feat": "SURF", "k": 200, "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-005", "feat": "SIFT", "k": 200, "use_hsv": True, "space": "pca", "algo": "gmm", "pca_comp": 30},
        {"id": "EXP-006", "feat": "SURF", "k": 200, "use_hsv": True, "space": "pca", "algo": "gmm", "pca_comp": 30},
        {"id": "EXP-007", "feat": "SIFT", "k": 50,  "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-008", "feat": "SIFT", "k": 100, "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-009", "feat": "SURF", "k": 50,  "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-010", "feat": "SURF", "k": 100, "use_hsv": True, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-011", "feat": "SIFT", "k": 200, "use_hsv": False, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-012", "feat": "SURF", "k": 200, "use_hsv": False, "space": "pca", "algo": "kmeans", "pca_comp": 30},
        {"id": "EXP-013", "feat": "SIFT", "k": 200, "use_hsv": False, "space": "raw", "algo": "kmeans", "pca_comp": None},
        {"id": "EXP-014", "feat": "SURF", "k": 200, "use_hsv": False, "space": "raw", "algo": "kmeans", "pca_comp": None},
    ]
    
    experiment_logs = []
    base_archive_path = os.path.join(os.getcwd(), "archives")
    os.makedirs(base_archive_path, exist_ok=True)
    
    # Menghasilkan format tanggal (YYYYMMDD) untuk penomoran ID
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    # Mempertahankan timestamp detik untuk nama file Master Log
    timestamp_full = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Menggunakan enumerate untuk menghasilkan sekuens otomatis (01, 02, dst.)
    for idx, sc in enumerate(scenarios, start=1):
        # Membentuk ID Eksperimen: EXP-YYYYMMDD-XX
        seq_str = f"{idx:02d}"
        exp_id = f"EXP-{date_str}-{seq_str}"
        
        # Menimpa 'id' lama agar ID baru terekam di Master Log
        sc["id"] = exp_id 
        
        extractor = sc["feat"]
        logger.info(f"Mengeksekusi {exp_id}: {extractor} k={sc['k']} HSV={sc['use_hsv']} {sc['space'].upper()} {sc['algo'].upper()}")
        
        # A. Pembuatan Direktori Arsip otomatis
        folder_name = f"{config.DATASET_DOMAIN}_{exp_id}"
        exp_dir = os.path.join(base_archive_path, folder_name)
        code_dir, results_dir = os.path.join(exp_dir, "code_snapshot"), os.path.join(exp_dir, "results")
        os.makedirs(code_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # B. Snapshot Source Code
        for file in os.listdir("."):
            if file.endswith(".py") or file.endswith(".json"):
                shutil.copy(file, code_dir)
        if os.path.exists("src"):
            shutil.copytree("src", os.path.join(code_dir, "src"))
            
        # C. Eksekusi Skenario Komputasi
        metrics = _execute_single_scenario(sc, dict_sift, dict_surf, code_dir, results_dir)
        
        # D. Integrasi Metrik Efisiensi ke dalam Log
        eff_data = eff_lookup.get(extractor, {'elapsed_sec': 0, 'std_sec': 0, 'n_keypoints': 0})
        metrics["efficiency"] = eff_data
        
        # Memperbarui JSON di dalam folder arsip
        with open(os.path.join(results_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        # E. Pencatatan ke Master Log Tabular
        experiment_logs.append({
            "ID Eksperimen": exp_id,
            "Ekstraktor": extractor,
            "Konfigurasi": f"{extractor} k={sc['k']} HSV={sc['use_hsv']}",
            "Ruang & Algo": f"{sc['space'].upper()} + {sc['algo'].upper()}",
            "Silhouette (↑)": round(metrics.get("silhouette_score", 0), 4),
            "Davies-Bouldin (↓)": round(metrics.get("davies_bouldin_score", 0), 4),
            "Calinski-Harabasz (↑)": round(metrics.get("calinski_harabasz_score", 0), 2),
            "Waktu Ekstraksi (det/citra) (↓)": eff_data["elapsed_sec"],
            "Std Dev Waktu (±)": eff_data["std_sec"],
            "Rata-rata Keypoint": eff_data["n_keypoints"],
            "Folder Arsip": folder_name
        })
        
    # 3. Ekspor Tabel Rekapitulasi
    df_log = pd.DataFrame(experiment_logs)
    main_log_path = os.path.join(base_archive_path, f"Master_Log_{config.DATASET_DOMAIN}_{timestamp_full}.csv")
    df_log.to_csv(main_log_path, index=False)
    logger.info(f"Eksperimen komprehensif terintegrasi selesai. Log tersimpan di: {main_log_path}")

# TAHAP 8: PEMBUATAN MODEL PRODUKSI 
def step_build_production_model():
    logger.info("TAHAP 8 - MERAKIT MODEL PRODUKSI")
    
    # 1. Tentukan parameter terbaik dari hasil eksperimen
    logger.info("Mengambil konfigurasi terbaik dari file CSV...")
    tuning_csv_path = os.path.join(config.TABLES_DIR, "tuning_results_deep_search.csv")
    
    if not os.path.exists(tuning_csv_path):
        logger.error("File hasil tuning tidak ditemukan! Jalankan --step tune terlebih dahulu.")
        return
        
    # Membaca data dan mengambil baris paling atas (indeks ke-0)
    df_tuning = pd.read_csv(tuning_csv_path)
    best_params = df_tuning.iloc[0]
    
    # Menetapkan variabel secara dinamis
    best_method = "surf" 
    best_codebook_size = config.DEFAULT_CODEBOOK_SIZE
    best_pca_components = int(best_params["PCA_COMPONENTS"])
    best_gmm_covariance = str(best_params["GMM_COVARIANCE"])
    
    # Memperbarui bobot HSV di konfigurasi global agar fusi selaras dengan model
    config.HSV_FUSION_WEIGHT = float(best_params["HSV_WEIGHT"])
    
    logger.info("Parameter Otomatis Diterapkan -> PCA: %d | GMM: %s | Bobot HSV: %.1f", 
                best_pca_components, best_gmm_covariance, config.HSV_FUSION_WEIGHT)
    descriptors_dict = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"descriptors_{best_method}.pkl"))
    hsv_dict = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"hsv_{best_method}.pkl"))
    codebooks = utils.load_pickle(os.path.join(config.INTERIM_DIR, f"codebooks_{best_method}.pkl"))
    
    codebook = codebooks[best_codebook_size]
    filenames = list(descriptors_dict.keys())
    
    """ 2. Buat ulang scaler warna HSV (MinMax lalu L2-normalize per baris -
    HARUS sama persis dengan step_bovw, atau model produksi tidak konsisten
    dengan hasil eksperimen yang dilaporkan """
    
    logger.info("Melatih ulang MinMaxScaler...")
    raw_hsv_values = list(hsv_dict.values())
    scaler = MinMaxScaler()
    scaler.fit(raw_hsv_values)
    
    # 3. Bentuk matriks fitur gabungan (Histogram + Warna)
    logger.info("Menyusun matriks fusi fitur...")
    hist_matrix = []
    for f in filenames:
        desc = descriptors_dict.get(f)
        bovw_hist = bovw.compute_histogram(desc, codebook, normalize="hellinger")
        
        hsv_feats = hsv_dict.get(f, np.zeros(9, dtype=np.float32))
        hsv_scaled = scaler.transform(hsv_feats.reshape(1, -1))[0]
        hsv_norm = np.linalg.norm(hsv_scaled)
        hsv_scaled = (hsv_scaled / hsv_norm) if hsv_norm > 0 else hsv_scaled
        hsv_weighted = hsv_scaled * config.HSV_FUSION_WEIGHT
        
        hist_matrix.append(np.concatenate((bovw_hist, hsv_weighted)))
        
    X_fused = np.vstack(hist_matrix)
    
    # 4. Latih PCA
    logger.info("Melatih ulang PCA (%d komponen)...", best_pca_components)
    pca = PCA(n_components=best_pca_components, random_state=config.RANDOM_STATE)
    X_pca = pca.fit_transform(X_fused)
    
    # 5. Latih GMM
    logger.info("Melatih GMM Classifier...")
    gmm = GaussianMixture(n_components=2, covariance_type=best_gmm_covariance, 
                          random_state=config.RANDOM_STATE, n_init=10)
    gmm.fit(X_pca) 
    
    # 6. BUNGKUS MENJADI SATU FILE MODEL
    model_package = {
        "codebook": codebook,
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
        "hsv_weight": config.HSV_FUSION_WEIGHT,
        "metadata": {
            "feature": best_method.upper(),
            "codebook_size": best_codebook_size,
            "pca_components": best_pca_components
        }
    }
    
    model_path = os.path.join(config.OUTPUT_DIR, "meat_grading_model.joblib")
    joblib.dump(model_package, model_path)
    
    logger.info("Model Produksi berhasil disimpan di: %s", model_path)
    logger.info("Model ini siap dimuat (di-load) untuk aplikasi klasifikasi otomatis tahun depan!")

# ENTRY POINT
STEPS = {
    "preprocess": step_preprocess,
    "extract": step_extract,
    "bovw": step_bovw,
    "cluster": step_cluster,
    "evaluate": step_evaluate,
    "tune": step_tune,
    "export": step_export,
    "build_model": step_build_production_model,
    "batch_log": step_batch_experiment,
}

def main():
    parser = argparse.ArgumentParser(description="Pipeline unsupervised CV deteksi mutu kesegaran daging sapi.")
    parser.add_argument(
        "--step", required=True, choices=list(STEPS.keys()) + ["all"],
        help="Tahap pipeline yang ingin dijalankan.",
    )
    args = parser.parse_args()

    utils.ensure_dirs()
    start = time.time()

    if args.step == "all":
        for name, func in STEPS.items():
            if name != "tune":  # Menghindari tuning tereksekusi saat perintah --step all
                func()
    else:
        STEPS[args.step]()

    elapsed_min = (time.time() - start) / 60.0
    logger.info("Tahap '%s' selesai dalam %.2f menit.", args.step, elapsed_min)


if __name__ == "__main__":
    main()