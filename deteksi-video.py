import cv2
from ultralytics import YOLO
 
sumber = 4
video = cv2.VideoCapture(sumber)
deteksi = YOLO("yolo11n.pt")
 
while True:
    ret, frame = video.read()
    results = deteksi(frame)
    box = results[0].plot()
    cv2.imshow("Webcam", box)
    if cv2.waitKey(1) == 27:  # tombol esc
        break
 
video.release()
cv2.destroyAllWindows()
