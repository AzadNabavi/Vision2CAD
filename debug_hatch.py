import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_img, w_img = gray.shape

# Mask for Section view
mask = np.zeros_like(gray)
mask[0:int(h_img*0.6), int(w_img*0.4):w_img] = 255
gray = cv2.bitwise_and(gray, mask)

edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)

hatch_pts = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if (40 < angle < 50) or (130 < angle < 140):
            hatch_pts.append((x1, y1))
            hatch_pts.append((x2, y2))

if hatch_pts:
    hatch_pts = np.array(hatch_pts)
    x, y, w, h = cv2.boundingRect(hatch_pts)
    print(f"Hatch BBox TR: x={x}, y={y}, w={w}, h={h}")
else:
    print("No hatch lines found.")
