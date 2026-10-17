from ultralytics import YOLO
model = YOLO("deteksi-topi.pt")

results = model("gambar2.jpg")
for box in results[0].boxes:
    conf = float(box.conf)
    print(conf)
results[0].show()