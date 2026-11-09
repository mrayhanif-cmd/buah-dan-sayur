import cv2
import numpy as np

def cek_kualitas_gambar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()

    if blur > 100:
        return "Baik"
    else:
        return "Tidak Baik"


# ==========================
# DETEKSI APAKAH BUAH/SAYUR
# ==========================
def cek_objek_buah_sayur(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range warna hijau
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Range warna merah
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1)

    total_pixels = frame.shape[0] * frame.shape[1]
    colored_pixels = cv2.countNonZero(mask_green) + cv2.countNonZero(mask_red)

    ratio = colored_pixels / total_pixels

    if ratio > 0.05:   # kalau warna dominan > 5%
        return True
    else:
        return False