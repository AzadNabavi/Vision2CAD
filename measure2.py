import cv2
import numpy as np

img = cv2.imread("agentic_highlighted_result_v3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_black_pixels(x_col):
    col = gray[:, x_col]
    # find where pixel is completely black
    black = np.where(col < 50)[0]
    return black

print("X=600 (flange?):", get_black_pixels(600))
print("X=670 (middle?):", get_black_pixels(670))
print("X=740 (middle?):", get_black_pixels(740))
print("X=780 (rightmost?):", get_black_pixels(780))
