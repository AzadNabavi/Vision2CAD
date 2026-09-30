import cv2
import numpy as np

img = cv2.imread(r"C:\Users\sazad\Desktop\CADAICO_Test\technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

h, w = binary.shape
# Focus solely on the cross-section
roi = binary[0:int(h*0.6), int(w*0.4):w]
roi_color = img[0:int(h*0.6), int(w*0.4):w].copy()

# Technical drawing outer solid lines are thick. Hatching and dimensions are thin.
# A 3x3 or 5x5 ellipse kernel for morphological opening (erode then dilate)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 1. Erode to destroy all thin lines (hatching, dimensions, text)
eroded = cv2.erode(roi, kernel, iterations=1)

# 2. Dilate to restore the surviving thick lines to their original thickness
dilated = cv2.dilate(eroded, kernel, iterations=1)

# 3. Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    
    cv2.drawContours(roi_color, [approx], -1, (0, 0, 255), 3)
    cv2.imwrite(r"C:\Users\sazad\Desktop\CADAICO_Test\Output\morph_test.png", roi_color)
    print("Success. Contour points:", len(approx))
else:
    print("No contours found.")
