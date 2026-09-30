import cv2
import numpy as np

img = cv2.imread("views/section_view.jpg")
H, W = img.shape[:2]

for x in range(0, W, 50):
    cv2.line(img, (x, 0), (x, H), (200, 200, 200), 1)
    cv2.putText(img, str(x), (x+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

for y in range(0, H, 50):
    cv2.line(img, (0, y), (W, y), (200, 200, 200), 1)
    cv2.putText(img, str(y), (5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

cv2.imwrite("grid_view.jpg", img)
