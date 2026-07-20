"""
Segmentasi ROI (Region of Interest) permukaan daging.
Pendekatan heuristik berbasis Strict HSV Thresholding & Margin Annihilation.
"""
import cv2
import numpy as np
import config

def segment_meat_roi(image_bgr, min_area_ratio=0.015):
    """
    Menghasilkan mask biner (uint8, nilai 0/255) area permukaan daging (foreground) murni.
    """
    h, w = image_bgr.shape[:2]
    total_area = float(h * w)

    # 1.Konversi ke HSV dengan reduksi blur untuk menjaga ketegasan batas tepi
    blurred = cv2.GaussianBlur(image_bgr, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2.Strict Spectrum Isolation
    # S > 50 dan V > 40 membuang warna netral (hitam/abu-abu latar & putih stiker)
    # H: 0-20 & 160-180 murni mengunci pigmen mioglobin (merah/pink/cokelat)
    mask1 = cv2.inRange(hsv, np.array([0, 30, 25]), np.array([30, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 30, 25]), np.array([180, 255, 255]))
    color_mask = cv2.bitwise_or(mask1, mask2)

    # 3. Margin Annihilation (Menghapus 12% margin untuk mematikan sisa label)
    margin_y, margin_x = int(h * 0.12), int(w * 0.12)
    color_mask[0:margin_y, :] = 0
    color_mask[h-margin_y:h, :] = 0
    color_mask[:, 0:margin_x] = 0
    color_mask[:, w-margin_x:w] = 0

    # 4. Conservative Morphology (Menutup porositas tanpa mendistorsi geometri tepi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morph_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 5. Ekstraksi Kontur Agnostik
    cnts = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    fallback_used = False

    if not contours:
        final_mask.fill(255)
        fallback_used = True
    else:
        center_x, center_y = w // 2, h // 2
        best_contour = None
        best_score = -1

        for c in contours:
            area = cv2.contourArea(c)
            # Filter noise absolut: Minimal 1.5% dari dimensi kanvas
            if area > (total_area * min_area_ratio):  
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = center_x, center_y

                dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                score = area / (dist + 1)
                
                if score > best_score:
                    best_score = score
                    best_contour = c

        if best_contour is not None:
            # Render poligon murni tanpa Gaussian Blur yang berlebihan di tahap akhir
            cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)
        else:
            final_mask.fill(255)
            fallback_used = True

    meta = {"foreground_ratio": float(np.count_nonzero(final_mask)) / total_area, "fallback_used": fallback_used}
    return final_mask, meta