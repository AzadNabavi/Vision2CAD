import cv2
import numpy as np
import math

def get_true_contour_points():
    pts = []
    
    # We will generate the top half of the profile.
    # The center line of the part will be Y = 10.
    # The outer edges will be Y < 10. (e.g. Y=0 is the flange).
    # X ranges from 0 to 31.
    
    # 1. Rightmost cylinder (D=12 -> Radius 6)
    # Starts at X=31, Y = 10 - 6 = 4
    pts.append((31, 4))
    pts.append((25, 4))
    
    # 2. Step up to D=15 -> Radius 7.5
    # Y = 10 - 7.5 = 2.5
    pts.append((25, 2.5))
    
    # 3. Middle cylinder (D=15)
    pts.append((12.5, 2.5))
    
    # 4. R9 Fillet from (12.5, 2.5) to (5, 0)
    x1, y1 = 12.5, 2.5
    x2, y2 = 5.0, 0.0
    r = 9.0
    
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    
    # Center of concave fillet must be outside the part (above it, so Y < 0)
    h = math.sqrt(r**2 - (dist/2)**2)
    # normal vector to the right of (dx, dy)
    nx = -dy / dist
    ny = dx / dist
    
    # We want the center to have Y < 0.
    cx = mid_x + h * nx
    cy = mid_y + h * ny
    if cy > 0: # If it went the wrong way, flip it
        cx = mid_x - h * nx
        cy = mid_y - h * ny
        
    angle1 = math.atan2(y1 - cy, x1 - cx)
    angle2 = math.atan2(y2 - cy, x2 - cx)
    
    # We want to sweep the SHORT way.
    if angle1 < 0: angle1 += 2*math.pi
    if angle2 < 0: angle2 += 2*math.pi
    
    diff = angle2 - angle1
    if diff > math.pi:
        angle1 += 2*math.pi
    elif diff < -math.pi:
        angle2 += 2*math.pi
        
    steps = 20
    for i in range(1, steps): # skip 0 and steps to avoid exact duplicates of endpoints
        a = angle1 + (angle2 - angle1) * (i / steps)
        px = cx + r * math.cos(a)
        py = cy + r * math.sin(a)
        pts.append((px, py))
        
    # Endpoint of fillet
    pts.append((5.0, 0.0))
        
    # 5. Flange D=20 -> Radius 10
    # Y = 10 - 10 = 0
    pts.append((0, 0))
    
    # 6. Concave front face R40.5
    # Goes from (0, 0) to center line (Y=10)
    # Center of this circle must be to the left (X < 0) to make it concave.
    cx2 = -39.246
    r2 = 40.5
    for y in np.linspace(0, 10, 20)[1:]: # skip Y=0 as it's already added
        # (x - cx2)^2 + (10 - y)^2 = r2^2
        # x = cx2 + sqrt(r2^2 - (10 - y)^2)
        # Wait, if center line is Y=10, the distance from center is |10 - y|
        x = cx2 + math.sqrt(r2**2 - (10 - y)**2)
        pts.append((x, y))
        
    return np.array(pts, dtype=np.float32)

def draw_full_contour(scale):
    top_pts = get_true_contour_points()
    
    # Top points go from right to left (X=31 to X=1.254 at Y=10)
    # The last point is on the center line Y=10!
    
    # Now we mirror to create the bottom half.
    # The bottom half should go from left to right to close the polygon smoothly!
    # Mirror rule: new_y = 10 + (10 - y) = 20 - y
    bottom_pts = []
    # We don't need to reverse top_pts if we just want a continuous loop.
    # Wait, if top_pts ends at the left face center (X=1.254, Y=10),
    # then bottom_pts should START at the left face center (X=1.254, Y=10)
    # and go towards the right face!
    # So we DO reverse top_pts for the bottom half!
    for pt in reversed(top_pts):
        bottom_pts.append((pt[0], 20 - pt[1]))
        
    # Remove the duplicate center-line points where top and bottom meet
    # The last point of top_pts is Y=10. The first point of bottom_pts is Y=10.
    bottom_pts = bottom_pts[1:]
    
    full_pts = np.vstack((top_pts, bottom_pts))
    
    # Scale it
    full_pts *= scale
    draw_pts = np.round(full_pts).astype(np.int32)
    
    width = int(32 * scale)
    height = int(22 * scale)
    
    img = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(img, [draw_pts], isClosed=True, color=255, thickness=2)
    return img

scale = 10.0
img = draw_full_contour(scale)
cv2.imwrite("debug_true_contour.jpg", img)
