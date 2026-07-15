"""Bags of Visual Words: pembentukan codebook (MiniBatchKMeans) & histogram fitur."""
import numpy as np
from sklearn.cluster import MiniBatchKMeans

import config


def sample_descriptors_for_codebook(all_descriptors_list, max_total, random_state):
    """
    Menggabungkan descriptor dari seluruh citra lalu men-subsample hingga
    maksimum `max_total` baris agar pelatihan codebook tetap efisien memori
    untuk dataset beribu-ribu citra.
    """
    stacked = np.vstack(
        [d for d in all_descriptors_list if d is not None and len(d) > 0]
    )
    if len(stacked) <= max_total:
        return stacked
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(stacked), size=max_total, replace=False)
    return stacked[idx]


def build_codebook(descriptor_sample, k, random_state=None, batch_size=None):
    """Melatih MiniBatchKMeans sebagai kamus visual (codebook) berukuran k."""
    random_state = config.RANDOM_STATE if random_state is None else random_state
    batch_size = (
        config.MINIBATCH_KMEANS_BATCH_SIZE if batch_size is None else batch_size
    )
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(batch_size, len(descriptor_sample)),
        n_init=10,
        max_iter=200,
    )
    kmeans.fit(descriptor_sample)
    return kmeans


def compute_histogram(descriptors, codebook, normalize="hellinger"):
    """
    Kuantisasi descriptor ke histogram BoVW dengan transformasi Root-BoVW (Hellinger).
    """
    k = codebook.n_clusters
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(k, dtype=np.float32)

    word_indices = codebook.predict(descriptors)
    histogram, _ = np.histogram(word_indices, bins=np.arange(k + 1))
    histogram = histogram.astype(np.float32)

    if normalize == "l2":
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
    elif normalize == "l1":
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
    elif normalize == "hellinger":
        # Transformasi Root-BoVW (Sangat superior untuk deteksi tekstur)
        total = histogram.sum()
        if total > 0:
            histogram = histogram / total
        histogram = np.sqrt(histogram)

    return histogram
