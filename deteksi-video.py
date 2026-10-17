import cv2
from ultralytics import YOLO

sumber = "video2.mp4"  # Path video
video = cv2.VideoCapture(sumber)
deteksi = YOLO("deteksi-topi.pt")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Deteksi dengan ukuran input maksimal 640
    frame = cv2.resize(frame, (640, 320))
    results = deteksi(frame, imgsz=320, verbose=False)

    box = results[0].plot()
    cv2.imshow("Deteksi Topi", box)

    if cv2.waitKey(1) == 27:  # ESC untuk keluar
        break

video.release()
cv2.destroyAllWindows()