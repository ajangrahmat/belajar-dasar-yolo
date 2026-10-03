import torch
from ultralytics import YOLO

# 1. Cek apakah PyTorch mendeteksi GPU Anda
print("GPU Tersedia:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nama GPU:", torch.cuda.get_device_name(0))

# 2. Cek perangkat yang digunakan model aktif
model = YOLO("yolo11n.pt")
print("Perangkat Model:", model.device)
