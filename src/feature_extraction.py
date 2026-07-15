"""
Ekstraksi fitur lokal SIFT & SURF, serta momen HSV, dibatasi pada area ROI daging (mask).

Membutuhkan opencv-contrib-python versi yang dikompilasi dengan
OPENCV_ENABLE_NONFREE=ON (mis. 3.4.2.16) agar cv2.xfeatures2d.SIFT_create()
dan cv2.xfeatures2d.SURF_create() tersedia.
"""
import time
import cv2
import numpy as np
from scipy.stats import skew

import config


def _get_detector(method):
    method = method.lower()
    if method == "sift":
        return cv2.xfeatures2d.SIFT_create(nfeatures=config.SIFT_N_FEATURES)
    elif method == "surf":
        return cv2.xfeatures2d.SURF_create(
            hessianThreshold=config.SURF_HESSIAN_THRESHOLD
        )
    raise ValueError("method harus 'sift' atau 'surf', diterima: {}".format(method))


def calculate_hsv_moments(hsv_image, mask):
    """
    Mengekstrak momen statistik (Mean, Std, Skewness) dari citra HSV.
    Hanya menghitung piksel yang berada di dalam mask (daging).
    """
    h, s, v = cv2.split(hsv_image)
    moments = []
    
    for channel in (h, s, v):
        # Terapkan mask untuk mengisolasi area daging secara eksklusif
        channel_data = channel[mask == 255]
        
        if len(channel_data) == 0:
            moments.extend([0.0, 0.0, 0.0])
            continue
            
        moments.extend([
            float(np.mean(channel_data)), 
            float(np.std(channel_data)), 
            float(skew(channel_data))
        ])
    return np.array(moments, dtype=np.float32)


def extract_features(gray_image, hsv_image, mask, method):
    """
    Mendeteksi keypoint & menghitung descriptor pada area mask saja,
    serta mengekstrak momen warna global HSV.

    Returns
    -------
    descriptors : np.ndarray (N, D) float32
    hsv_moments : np.ndarray (9,) float32
    n_keypoints : int
    elapsed_sec : float
    """
    detector = _get_detector(method)

    start = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(gray_image, mask)
    
    # Ekstraksi HSV Moments sesuai parameter konfigurasi
    hsv_moments = calculate_hsv_moments(hsv_image, mask) if config.USE_COLOR_FUSION else None
    
    elapsed_sec = time.perf_counter() - start

    n_keypoints = len(keypoints) if keypoints is not None else 0

    if descriptors is None:
        descriptors = np.empty((0, 128 if method.lower() == "sift" else 64),
                                dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32)
        cap = config.MAX_DESCRIPTORS_PER_IMAGE
        if cap is not None and len(descriptors) > cap:
            rng = np.random.RandomState(config.RANDOM_STATE)
            idx = rng.choice(len(descriptors), size=cap, replace=False)
            descriptors = descriptors[idx]

    return descriptors, hsv_moments, n_keypoints, elapsed_sec