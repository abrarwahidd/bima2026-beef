"""Preprocessing: resize, denoising, segmentasi ROI, QA visual, dan Bounding Box Crop."""
import os
import cv2
import numpy as np
import config
from src import segmentation

def resize_keep_aspect(image_bgr, max_side):
    h, w = image_bgr.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image_bgr  
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

def denoise(image_bgr):
    if config.DENOISE_H is None:
        return image_bgr
    return cv2.fastNlMeansDenoisingColored(
        image_bgr, None,
        h=config.DENOISE_H, hColor=config.DENOISE_H_COLOR,
        templateWindowSize=config.DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=config.DENOISE_SEARCH_WINDOW,
    )

def to_hsv(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

def crop_to_bounding_box(image, mask, pad_ratio=0.01):
    """
    Pemotongan matriks secara presisi absolut (tight fit) berbasis piksel aktif.
    Mengeliminasi redundansi dimensi latar belakang untuk menjaga kemurnian
    ekstraksi fitur spasial (SURF) dan momen statistik warna (HSV).
    """
    #1.Ekstraksi koordinat absolut dari piksel foreground murni
    coords = cv2.findNonZero(mask)
    
    # Fallback jika mask secara anomali kosong
    if coords is None:
        return image, mask
        
    #2.Kalkulasi bounding box yang presisi mengelilingi piksel aktif
    x, y, w, h_rect = cv2.boundingRect(coords)
    
    H, W = image.shape[:2]
    
    #3.Margin ketat (1% dari dimensi objek) untuk menyisakan sedikit ruang agar detektor tepi SURF tidak terpotong
    pad_x, pad_y = int(w * pad_ratio), int(h_rect * pad_ratio)
    
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h_rect + pad_y)
    
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def save_visual_inspection(original_bgr, mask, filename):
    """
    Pembuatan artefak visual QA berdampingan dengan resolusi asli.
    Latar belakang diubah menjadi putih murni untuk lampiran manuskrip.
    """
    os.makedirs(config.SEGMENTED_VIEW_DIR, exist_ok=True)
    
    segmented_bgr = cv2.bitwise_and(original_bgr, original_bgr, mask=mask)
    white_bg = np.full(original_bgr.shape, 255, dtype=np.uint8)
    background = cv2.bitwise_and(white_bg, white_bg, mask=cv2.bitwise_not(mask))
    segmented_clean = cv2.add(segmented_bgr, background)
    
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    annotated_bgr = original_bgr.copy()
    cv2.drawContours(annotated_bgr, contours, -1, (0, 0, 255), 3)
    
    combined_view = np.hstack((annotated_bgr, segmented_clean))
    out_path = os.path.join(config.SEGMENTED_VIEW_DIR, f"QA_{os.path.splitext(filename)[0]}.png")
    cv2.imwrite(out_path, combined_view)

def preprocess_single_image(filepath):
    """
    Alur komputasi: Normalisasi -> Denoising -> Masking -> QA Visual -> Cropping.
    """
    image_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    image_bgr = resize_keep_aspect(image_bgr, config.RESIZE_MAX_SIDE)
    image_bgr = denoise(image_bgr)

    mask, seg_meta = segmentation.segment_meat_roi(image_bgr)
    
    # Render gambar QA sebelum citra direduksi oleh Bounding Box
    filename = os.path.basename(filepath)
    save_visual_inspection(image_bgr, mask, filename)

    # Eksekusi pemotongan latar belakang (Bounding Box)
    if not seg_meta["fallback_used"]:
        image_bgr, mask = crop_to_bounding_box(image_bgr, mask)

    hsv = to_hsv(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "gray": gray,
        "hsv": hsv,
        "mask": mask,
        "seg_meta": seg_meta,
        "shape": image_bgr.shape[:2],
    }

def save_mask(mask, filename, masks_dir=None):
    masks_dir = config.MASKS_DIR if masks_dir is None else masks_dir
    os.makedirs(masks_dir, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    out_path = os.path.join(masks_dir, stem + "_mask.npz")
    np.savez_compressed(out_path, mask=mask)
    return out_path

def save_processed(gray, hsv, mask, filename, processed_dir=None):
    processed_dir = config.PROCESSED_DIR if processed_dir is None else processed_dir
    os.makedirs(processed_dir, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    out_path = os.path.join(processed_dir, stem + ".npz")
    np.savez_compressed(out_path, gray=gray, hsv=hsv, mask=mask)
    return out_path

def load_processed(filename, processed_dir=None):
    processed_dir = config.PROCESSED_DIR if processed_dir is None else processed_dir
    stem = os.path.splitext(filename)[0]
    in_path = os.path.join(processed_dir, stem + ".npz")
    data = np.load(in_path)
    return data["gray"], data["hsv"], data["mask"]