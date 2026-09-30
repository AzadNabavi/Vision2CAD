import cv2
import numpy as np

img = cv2.imread("agentic_highlighted_result_v3.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Look at the vertical column in the middle of the middle cylinder (x = 500)
# and the rightmost cylinder (x = 650)
# We will just print some edges

def find_edges(x_col):
    col = gray[:, x_col]
    edges = np.where(np.abs(np.diff(col.astype(int))) > 50)[0]
    return edges

print("Middle cylinder edges (x=500):", find_edges(500))
print("Rightmost cylinder edges (x=650):", find_edges(650))
