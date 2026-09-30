import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 1. Isolate the top-right quadrant to avoid other views
h, w = gray.shape
mask_tr = np.zeros_like(thresh)
mask_tr[0:int(h*0.6), int(w*0.4):w] = 255
thresh = cv2.bitwise_and(thresh, mask_tr)

# 2. Dilate to merge hatching into a solid blob
# Gaps between hatch lines are about 10-15 pixels.
kernel_close = np.ones((11, 11), np.uint8)
solid = cv2.dilate(thresh, kernel_close, iterations=1)

# 3. Erode to destroy dimension lines (which are now 11px thick).
# Erode by a larger kernel, say 21x21
kernel_open = np.ones((21, 21), np.uint8)
eroded = cv2.erode(solid, kernel_open, iterations=1)

# 4. Dilate back by the difference (21 - 11 = 10) to restore boundaries
kernel_restore = np.ones((10, 10), np.uint8)
restored = cv2.dilate(eroded, kernel_restore, iterations=1)

# Now find the outer contour of this restored blob
contours, _ = cv2.findContours(restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    best_contour = max(contours, key=cv2.contourArea)
    highlighted = img.copy()
    # Draw in yellow
    cv2.drawContours(highlighted, [best_contour], -1, (0, 255, 255), 3)
    cv2.imwrite("debug_morph_highlight.jpg", highlighted)
    print("Success")
else:
    print("Failed")
