import cv2
 
img = cv2.imread("kucing.jpg")
cv2.rectangle(
    img,
    (100,100),
    (300,300),
    (0,255,0),
    3
)
cv2.imshow("Gambar", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
