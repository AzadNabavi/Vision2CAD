import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
areas = stats[:, cv2.CC_STAT_AREA]
sorted_indices = np.argsort(areas)[::-1]

# Create a color image to draw components
output = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

colors = [
    (0, 0, 255),   # Red
    (0, 255, 0),   # Green
    (255, 0, 0),   # Blue
    (0, 255, 255), # Yellow
]

for i in range(1, 5):
    idx = sorted_indices[i]
    mask = (labels == idx).astype(np.uint8)
    output[mask == 1] = colors[i-1]
    
cv2.imwrite("debug_components.jpg", output)
