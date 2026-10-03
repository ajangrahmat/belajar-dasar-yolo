from ultralytics import YOLO
model = YOLO("yolo11n.pt")

results = model("kucing.jpg")
for box in results[0].boxes:
    conf = float(box.conf)
    print(conf)
results[0].show()