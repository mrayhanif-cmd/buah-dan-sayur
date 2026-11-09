import cv2
import os
import time

from quality_check import cek_kualitas_gambar, cek_objek_buah_sayur

camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def capture_image():
    success, frame = camera.read()
    if not success:
        return None, None, None

    os.makedirs("static/captures", exist_ok=True)

    filename = f"static/captures/capture_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)

    kualitas = cek_kualitas_gambar(frame)
    objek_valid = cek_objek_buah_sayur(frame)

    return filename, kualitas, objek_valid