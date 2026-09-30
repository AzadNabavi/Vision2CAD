import cv2
import numpy as np

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find edges
edges = cv2.Canny(gray, 50, 150)

# Detect straight lines using Hough
lines = cv2.HoughLinesP(edges, 1, np.pi/2, threshold=40, minLineLength=30, maxLineGap=5)

edges_no_lines = edges.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Only remove perfectly horizontal or vertical lines (these are dimension lines)
        if x1 == x2 or y1 == y2:
            cv2.line(edges_no_lines, (x1, y1), (x2, y2), 0, 3)

# Now we have edges without the long straight dimension lines!
# But we might have cut some object lines.
# The section view is heavily hatched. If we dilate the edges heavily, we get a solid blob of the object.
kernel = np.ones((7, 7), np.uint8)
blob = cv2.dilate(edges_no_lines, kernel, iterations=2)
blob = cv2.erode(blob, kernel, iterations=2)

contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # We want the contour in the top right
    h, w = gray.shape
    best = None
    max_area = 0
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if x > w*0.4 and y < h*0.6:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best = cnt
    
    if best is not None:
        cv2.drawContours(img, [best], -1, (0, 255, 255), 3)
        cv2.imwrite("debug_hough.jpg", img)
        print("Success")
