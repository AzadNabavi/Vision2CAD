import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
print(f"Found {num_labels} connected components.")

# Filter components by area
areas = stats[:, cv2.CC_STAT_AREA]
sorted_indices = np.argsort(areas)[::-1]

for i in range(1, min(10, num_labels)): # Skip 0 (background)
    idx = sorted_indices[i]
    print(f"Component {i}: Area = {areas[idx]}")
