import cv2
import numpy as np

img = cv2.imread(r"C:\Users\sazad\Desktop\CADAICO_Test\technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

h, w = binary.shape
roi = binary[0:int(h*0.6), int(w*0.4):w]
roi_color = img[0:int(h*0.6), int(w*0.4):w].copy()

# 1. Distance Transform to isolate thick lines
dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)

# Threshold distance transform. Thick lines will have higher values.
# Assuming thick lines are ~3-4px wide (radius > 1.2), thin lines are 1-2px (radius < 1.0)
_, core = cv2.threshold(dist, 1.2, 255, cv2.THRESH_BINARY)
core = core.astype(np.uint8)

# 2. Find contours of the thick line cores
contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    # Filter by length to get the main body outline
    valid_contours = [c for c in contours if cv2.arcLength(c, True) > 200]
    if valid_contours:
        largest = max(valid_contours, key=lambda c: cv2.arcLength(c, True))
        
        # Don't use approxPolyDP yet, just draw the raw contour to see if it hugs perfectly
        cv2.drawContours(roi_color, [largest], -1, (0, 255, 0), 2)
        cv2.imwrite(r"C:\Users\sazad\Desktop\CADAICO_Test\Output\dist_test.png", roi_color)
        print("Success. Contour area:", cv2.contourArea(largest))
    else:
        print("No valid contours after area filter.")
else:
    print("No contours found.")
