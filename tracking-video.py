
import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# Konfigurasi
sumber = "video2.mp4"
model_path = "deteksi-topi.pt"

LEBAR = 640
TINGGI = 320
PREDIKSI_FRAME = 10  # Prediksi 10 frame ke depan
PANJANG_JEJAK = 30   # Maksimal titik jejak yang disimpan

# Load video dan model
video = cv2.VideoCapture(sumber)
deteksi = YOLO(model_path)

# Riwayat posisi setiap ID objek
riwayat = defaultdict(
    lambda: deque(maxlen=PANJANG_JEJAK)
)

while True:
    ret, frame = video.read()

    if not ret:
        break

    frame = cv2.resize(frame, (LEBAR, TINGGI))

    # Tracking menggunakan ByteTrack
    results = deteksi.track(
        frame,
        imgsz=320,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=0.25
    )

    hasil = results[0]
    tampilan = hasil.plot()

    if hasil.boxes is not None and hasil.boxes.id is not None:
        kotak = hasil.boxes.xyxy.cpu().numpy()
        ids = hasil.boxes.id.int().cpu().tolist()

        for koordinat, object_id in zip(kotak, ids):
            x1, y1, x2, y2 = map(int, koordinat)

            # Titik tengah objek (X, Y)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Simpan posisi saat ini
            riwayat[object_id].append((cx, cy))

            titik = list(riwayat[object_id])

            # Gambar jejak pergerakan
            for i in range(1, len(titik)):
                cv2.line(
                    tampilan,
                    titik[i - 1],
                    titik[i],
                    (0, 255, 255),
                    2
                )

            # Prediksi posisi berdasarkan gerakan sebelumnya
            if len(titik) >= 2:
                # Gunakan beberapa posisi terakhir
                n = min(8, len(titik))
                data = titik[-n:]

                # Kecepatan rata-rata per frame
                vx = (data[-1][0] - data[0][0]) / (n - 1)
                vy = (data[-1][1] - data[0][1]) / (n - 1)

                pred_x = int(cx + vx * PREDIKSI_FRAME)
                pred_y = int(cy + vy * PREDIKSI_FRAME)

                # Batasi titik prediksi dalam area video
                pred_x = max(0, min(LEBAR - 1, pred_x))
                pred_y = max(0, min(TINGGI - 1, pred_y))

                # Garis menuju prediksi
                cv2.arrowedLine(
                    tampilan,
                    (cx, cy),
                    (pred_x, pred_y),
                    (255, 0, 255),
                    2,
                    tipLength=0.25
                )

                cv2.circle(
                    tampilan,
                    (pred_x, pred_y),
                    6,
                    (255, 0, 255),
                    -1
                )

                teks_prediksi = f"Prediksi: ({pred_x}, {pred_y})"
                cv2.putText(
                    tampilan,
                    teks_prediksi,
                    (x1, max(18, y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1
                )

            # Tampilkan ID dan koordinat aktual
            teks = f"ID:{object_id} X:{cx} Y:{cy}"
            cv2.putText(
                tampilan,
                teks,
                (x1, min(TINGGI - 5, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Titik tengah objek
            cv2.circle(
                tampilan,
                (cx, cy),
                4,
                (0, 255, 0),
                -1
            )

    cv2.imshow("Deteksi Topi + Tracking + Prediksi", tampilan)

    # ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
