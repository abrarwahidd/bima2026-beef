"""
Visualisasi hasil: reduksi dimensi, evaluasi metrik, dan analisis distribusi.
Standar output disesuaikan untuk publikasi akademik (SINTA/Scopus) dengan 
resolusi 300 DPI dan gaya minimalis-kontras tinggi.
"""
import os
import matplotlib
matplotlib.use("Agg")  # Headless mode
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from math import pi
from sklearn.preprocessing import MinMaxScaler

import config

def set_academic_style():
    """Mengatur parameter global Matplotlib untuk standar jurnal akademik."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False
    })

# Panggil styling di awal
set_academic_style()


def plot_embedding_2d(X, labels, method_name, save_name, title=None):
    """Plot reduksi dimensi PCA dengan standar visual akademik."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    embedding = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Menggunakan colormap yang ramah buta warna dan elegan
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Set1",
        s=25, alpha=0.8, edgecolors="white", linewidth=0.5
    )
    
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"Principal Component 1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({var_explained[1]:.1%} variance)")
    ax.set_title(title or f"PCA Projection - {method_name}", pad=15)
    
    legend = ax.legend(*scatter.legend_elements(), title="Cluster ID", loc="best", frameon=True)
    ax.add_artist(legend)

    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_metric_comparison(df_metrics, metric_col, save_name, title=None, higher_is_better=True):
    """Bar chart komparasi metrik internal dengan anotasi nilai."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sns.barplot(
        data=df_metrics, x="method", y=metric_col, hue="feature",
        palette="viridis", ax=ax, edgecolor="black", linewidth=0.8
    )
    
    # Menambahkan nilai eksak di atas setiap bar
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.4f}", 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 4), 
                        textcoords='offset points')

    arrow = "(↑ Higher is Better)" if higher_is_better else "(↓ Lower is Better)"
    ax.set_title(title or f"Comparison of {metric_col.replace('_', ' ').title()} {arrow}", pad=15)
    ax.set_xlabel("Clustering Algorithm")
    ax.set_ylabel("Score")
    ax.legend(title="Feature Extractor", frameon=True)

    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_pca_ablation_impact(df_metrics, save_name="pca_impact_ablation.png"):
    """
    Visualisasi krusial untuk paper: Membuktikan dampak PCA terhadap Silhouette Score 
    (Mengatasi Curse of Dimensionality).
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Filter hanya untuk KMeans sebagai representasi
    df_kmeans = df_metrics[df_metrics['method'] == 'kmeans'].copy()
    if df_kmeans.empty or 'feature_space' not in df_kmeans.columns:
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.barplot(
        data=df_kmeans, x="feature", y="silhouette_score", hue="feature_space",
        palette="mako", ax=ax, edgecolor="black", linewidth=0.8
    )
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.4f}", 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 4), 
                        textcoords='offset points')

    ax.set_title("Ablation Study: Impact of PCA on Cluster Separability", pad=15)
    ax.set_xlabel("Feature Extractor")
    ax.set_ylabel("Silhouette Score (↑)")
    ax.legend(title="Feature Space", loc='upper left')

    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_temporal_distribution(df_predictions, save_name="temporal_cluster_distribution.png"):
    """
    Stacked bar chart untuk menunjukkan anomali/penemuan utama: 
    Bagaimana model secara otonom memisahkan DAY-1 dan DAY-2.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # Hitung silang distribusi
    cross_tab = pd.crosstab(df_predictions['Hari Ke-'], df_predictions['Asumsi Kondisi'])
    
    # Normalisasi menjadi persentase untuk visualisasi yang lebih adil
    cross_tab_pct = cross_tab.div(cross_tab.sum(1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cross_tab_pct.plot(kind='bar', stacked=True, color=['#4daf4a', '#e41a1c'], ax=ax, edgecolor='black')
    
    # Anotasi persentase di tengah bar
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='center', color='white', weight='bold')

    ax.set_title("Autonomous Cluster Mapping across Temporal Degradation", pad=15)
    ax.set_xlabel("Temporal Condition (Days Post-Slaughter)")
    ax.set_ylabel("Proportion of Images (%)")
    ax.legend(title="Assigned Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)

    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_tsne_2d(X, labels, method_name, save_name, title=None, perplexity=30):
    """
    Plot proyeksi t-SNE 2D dengan standar visual akademik.
    Digunakan untuk melihat separabilitas topologi data lokal berdasarkan struktur klasternya.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    n = len(X)
    
    # Guard: Menghindari error jika jumlah subsampel dataset lebih kecil dari perplexity
    perplexity = min(perplexity, max(5, n // 4))  
    
    tsne = TSNE(
        n_components=2, random_state=config.RANDOM_STATE, perplexity=perplexity,
        init="pca", learning_rate="auto"
    )
    embedding = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Menggunakan colormap Set1 yang konsisten dengan plot PCA
    scatter = ax.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="Set1",
        s=25, alpha=0.8, edgecolors="white", linewidth=0.5
    )
    
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title or f"t-SNE Projection - {method_name}", pad=15)
    
    legend = ax.legend(*scatter.legend_elements(), title="Cluster ID", loc="best", frameon=True)
    ax.add_artist(legend)

    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_efficiency_comparison(df_efficiency, save_name="efficiency_comparison.png"):
    """Boxplot efisiensi komputasi dengan penyesuaian log-scale untuk visibilitas outlier."""
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df_efficiency, x="method", y="elapsed_sec", ax=axes[0], palette="pastel")
    axes[0].set_title("Feature Extraction Time per Image", pad=10)
    axes[0].set_xlabel("Feature Extractor")
    axes[0].set_ylabel("Elapsed Time (Seconds)")
    axes[0].set_yscale('log') # Menggunakan log scale karena variance tinggi

    sns.boxplot(data=df_efficiency, x="method", y="n_keypoints", ax=axes[1], palette="pastel")
    axes[1].set_title("Detected Keypoints Density", pad=10)
    axes[1].set_xlabel("Feature Extractor")
    axes[1].set_ylabel("Number of Keypoints")

    fig.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


import cv2

def plot_keypoint_proof(gray_image, mask, save_name="keypoint_comparison_proof.png"):
    """
    Menghasilkan visualisasi komparatif deteksi SIFT vs SURF pada satu citra sampel.
    Diformat dengan skala abu-abu (grayscale) agar warna keypoint (hijau/merah) 
    terlihat kontras untuk figur paper bereputasi.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # 1. Inisialisasi Detektor sesuai dengan parameter riset (config.py)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=config.SIFT_N_FEATURES)
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=config.SURF_HESSIAN_THRESHOLD)
    
    # 2. Deteksi Keypoints pada area daging (mask)
    kp_sift, _ = sift.detectAndCompute(gray_image, mask)
    kp_surf, _ = surf.detectAndCompute(gray_image, mask)
    
    # 3. Konversi citra ke BGR semu agar keypoint berwarna bisa digambar di atas grayscale
    canvas_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # 4. Penggambaran Rich Keypoints (Titik, Skala, dan Orientasi)
    img_sift = cv2.drawKeypoints(
        canvas_bgr, kp_sift, None, color=(0, 255, 0), # Hijau kontras
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    img_surf = cv2.drawKeypoints(
        canvas_bgr, kp_surf, None, color=(226, 43, 138), # Merah keunguan (Colorblind-safe)
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # 5. Plotting berdampingan standar akademik
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Matplotlib menggunakan format RGB, konversi BGR -> RGB
    axes[0].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"SIFT Detection (n={len(kp_sift)})", fontsize=12, pad=10)
    axes[0].axis("off")
    
    axes[1].imshow(cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"SURF Detection (n={len(kp_surf)})\nHessian Thresh={config.SURF_HESSIAN_THRESHOLD}", fontsize=12, pad=10)
    axes[1].axis("off")
    
    fig.suptitle("Visual Proof of Local Feature Extraction (Scale & Orientation)", fontsize=14, weight='bold')
    fig.tight_layout()
    
    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return out_path


def plot_overall_radar_comparison(df_metrics, df_efficiency, save_name="overall_radar_sift_vs_surf.png"):
    """
    Menghasilkan Radar Chart multidimensional untuk komparasi keseluruhan SIFT vs SURF.
    Menormalisasi metrik performa spasial dan metrik efisiensi komputasi ke dalam satu sumbu polar.
    """
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    # 1. Ekstraksi agregat dari kombinasi terbaik (PCA + KMeans)
    df_best = df_metrics[(df_metrics['method'] == 'kmeans') & (df_metrics['feature_space'] == 'pca')]
    if df_best.empty:
        return
        
    sift_metrics = df_best[df_best['feature'] == 'SIFT'].iloc[0]
    surf_metrics = df_best[df_best['feature'] == 'SURF'].iloc[0]
    
    # 2. Ekstraksi agregat rata-rata efisiensi
    eff_summary = df_efficiency.groupby("method").mean(numeric_only=True)
    
    # 3. Penyusunan Data Mentah
    categories = ['Silhouette Score', 'Calinski-Harabasz', 'Keypoint Density', 
                  'Davies-Bouldin (Inverse)', 'Time Efficiency (Inverse)']
    N = len(categories)
    
    # SIFT Data
    sift_raw = [
        sift_metrics['silhouette_score'], 
        sift_metrics['calinski_harabasz_score'], 
        eff_summary.loc['SIFT', 'n_keypoints'],
        sift_metrics['davies_bouldin_score'], 
        eff_summary.loc['SIFT', 'elapsed_sec']
    ]
    
    # SURF Data
    surf_raw = [
        surf_metrics['silhouette_score'], 
        surf_metrics['calinski_harabasz_score'], 
        eff_summary.loc['SURF', 'n_keypoints'],
        surf_metrics['davies_bouldin_score'], 
        eff_summary.loc['SURF', 'elapsed_sec']
    ]
    
    # 4. Normalisasi Min-Max (0-1) agar bisa di-plot di sumbu yang sama
    # Untuk DB dan Time, kita balik (invers) sebelum dinormalisasi agar nilai tertinggi = terbaik
    raw_data = np.array([sift_raw, surf_raw])
    
    # Inversi kolom DB (indeks 3) dan Time (indeks 4)
    raw_data[:, 3] = 1.0 / raw_data[:, 3] 
    raw_data[:, 4] = 1.0 / raw_data[:, 4]
    
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(raw_data)
    
    sift_norm = normalized_data[0].tolist()
    surf_norm = normalized_data[1].tolist()
    
    # Tutup poligon radar (kembali ke titik awal)
    sift_norm += sift_norm[:1]
    surf_norm += surf_norm[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    # 5. Plotting Radar Chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Rotasi agar kategori pertama berada di atas (jam 12)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=10, weight='bold')
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1.1)
    
    # Plot SIFT
    ax.plot(angles, sift_norm, linewidth=2, linestyle='solid', label='SIFT', color='#2b8cbe')
    ax.fill(angles, sift_norm, '#2b8cbe', alpha=0.25)
    
    # Plot SURF
    ax.plot(angles, surf_norm, linewidth=2, linestyle='solid', label='SURF', color='#e34a33')
    ax.fill(angles, surf_norm, '#e34a33', alpha=0.25)
    
    plt.title("Comprehensive Trade-off: SIFT vs SURF Architecture", size=14, weight='bold', pad=25)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True)
    
    out_path = os.path.join(config.FIGURES_DIR, save_name)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path