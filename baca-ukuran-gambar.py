import cv2
 
img = cv2.imread("kucing.jpg")
print(img.shape)
 
# menghitung jumlah angka dalam gambar
h, w, c = img.shape
print(h*w*c)
