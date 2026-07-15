"""Clustering unsupervised (K-Means, GMM) + validasi klaster."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

import config

def run_kmeans(X, k=None, random_state=None):
    k = config.N_CLUSTERS_FINAL if k is None else k
    random_state = config.RANDOM_STATE if random_state is None else random_state
    model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    labels = model.fit_predict(X)
    return labels, model

def run_gmm(X, k=None, random_state=None):
    k = config.N_CLUSTERS_FINAL if k is None else k
    random_state = config.RANDOM_STATE if random_state is None else random_state
    model = GaussianMixture(
        n_components=k, random_state=random_state, n_init=10, covariance_type="full"
    )
    labels = model.fit_predict(X)
    return labels, model

# Pipeline utama akan membaca dictionary ini secara dinamis
# Sekarang hanya terdaftar dua algoritma
CLUSTERING_METHODS = {
    "kmeans": run_kmeans,
    "gmm": run_gmm,
}

def internal_validation(X, labels):
    """
    Metrik validasi klaster TANPA label ground truth. Dipakai sebagai bukti
    separabilitas kelas (pengganti akurasi supervised SVM/KNN), sesuai
    Tahap 5 metode proposal ("validasi separabilitas dan evaluasi kinerja fitur").

    - silhouette_score: [-1, 1], makin tinggi makin baik (klaster rapat & terpisah)
    - davies_bouldin_score: >= 0, makin RENDAH makin baik
    - calinski_harabasz_score: >= 0, makin tinggi makin baik
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2 or n_unique >= len(X):
        return {
            "silhouette_score": float("nan"),
            "davies_bouldin_score": float("nan"),
            "calinski_harabasz_score": float("nan"),
            "note": "Klaster tidak valid untuk dihitung metriknya (n_unique={})".format(
                n_unique
            ),
        }
    return {
        "silhouette_score": float(silhouette_score(X, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
    }

def _cluster_purity(true_labels, pred_labels):
    """Purity = proporsi anggota klaster yang berasal dari kelas mayoritasnya."""
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    total = len(true_labels)
    correct = 0
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        values, counts = np.unique(true_labels[mask], return_counts=True)
        correct += counts.max()
    return correct / float(total)

def external_validation(true_labels, pred_labels):
    """
    Metrik validasi klaster MENGGUNAKAN label ground truth opsional (dari
    pelabelan awal tim peneliti). Ini BUKAN supervised training - label hanya
    dipakai sebagai pembanding pasca-hoc untuk mengukur kesesuaian klaster
    otomatis dengan penilaian pakar (dosen Peternakan).

    - adjusted_rand_score (ARI): [-1, 1], 1 = identik dengan ground truth
    - normalized_mutual_info_score (NMI): [0, 1], makin tinggi makin baik
    - purity: [0, 1], makin tinggi makin baik
    """
    return {
        "adjusted_rand_index": float(adjusted_rand_score(true_labels, pred_labels)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(true_labels, pred_labels)
        ),
        "purity": float(_cluster_purity(true_labels, pred_labels)),
    }