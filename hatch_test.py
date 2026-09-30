import cv2
import numpy as np
import os

image_path = r"C:\Users\sazad\Desktop\CADAICO_Test\technical_drawing.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Crop to top right to avoid other components
h, w = binary.shape
top_right = binary[0:int(h*0.6), int(w*0.5):w]

# 1. Isolate hatching (45 degree lines)
# Create a kernel for 45 degree lines (bottom-left to top-right, or top-left to bottom-right)
kernel_size = 9
diagonal_kernel_1 = np.eye(kernel_size, dtype=np.uint8)
diagonal_kernel_2 = np.fliplr(diagonal_kernel_1)

# Apply opening to keep only diagonal lines
hatch_1 = cv2.morphologyEx(top_right, cv2.MORPH_OPEN, diagonal_kernel_1)
hatch_2 = cv2.morphologyEx(top_right, cv2.MORPH_OPEN, diagonal_kernel_2)
hatching = cv2.bitwise_or(hatch_1, hatch_2)

# 2. Merge hatching into a solid blob
merge_kernel = np.ones((15, 15), np.uint8)
blob = cv2.dilate(hatching, merge_kernel, iterations=3)
blob = cv2.erode(blob, merge_kernel, iterations=1)

# 3. Find contour of the blob
contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest = max(contours, key=cv2.contourArea)
    
    # Offset coordinates back to full image
    for pt in largest:
        pt[0][0] += int(w*0.5)
        pt[0][1] += 0
        
    viz = img.copy()
    cv2.drawContours(viz, [largest], -1, (0, 0, 255), 3)
    cv2.imwrite(r"C:\Users\sazad\Desktop\CADAICO_Test\Output\hatch_test.png", viz)
    print("Success. Contour area:", cv2.contourArea(largest))
else:
    print("No contours found.")
