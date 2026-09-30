import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Let's count how many pixels are in the thresholded image
print(f"Total foreground pixels: {np.sum(thresh == 255)}")

# Let's try to isolate the hatching lines. Hatching is typically diagonal.
# We can use morphological operations with diagonal kernels.
kernel_size = 5
kernel_diag = np.eye(kernel_size, dtype=np.uint8)
hatch = cv2.erode(thresh, kernel_diag, iterations=1)
hatch = cv2.dilate(hatch, kernel_diag, iterations=1)

print(f"Hatch pixels: {np.sum(hatch == 255)}")

cv2.imwrite("debug_hatch.jpg", hatch)
