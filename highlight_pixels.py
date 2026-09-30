import cv2
import numpy as np

def highlight_pixels():
    img = cv2.imread('views/section_view.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Threshold to get black lines
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 2. Separate thick lines (part outline) from thin lines (dimension arrows)
    # Erode will destroy 1-pixel thick dimension lines but keep the thicker part outline
    kernel_thin = np.ones((3,3), np.uint8)
    eroded = cv2.erode(thresh, kernel_thin, iterations=1)
    
    # 3. Dilate heavily to close the hatched gaps inside the part
    kernel_close = np.ones((5,5), np.uint8)
    solid = cv2.dilate(eroded, kernel_close, iterations=4)
    
    # 4. Find the largest contour which represents the solid part
    contours, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest = max(contours, key=cv2.contourArea)
        # Draw the contour in pure red (0, 0, 255) with thickness 2
        cv2.drawContours(img, [largest], -1, (0, 0, 255), 2)
        cv2.imwrite('gpt_style_highlight.jpg', img)
        print("Success! Created gpt_style_highlight.jpg using pixel tracing.")
    else:
        print("Failed to find contour.")

if __name__ == "__main__":
    highlight_pixels()
