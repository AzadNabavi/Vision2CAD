import cv2
import numpy as np
import math

print("==================================================")
print("  INTEGRATED AGENTIC HIGHLIGHTING SYSTEM v5")
print("==================================================")

def generate_agentic_contours():
    pts = []
    
    # --- INNER PROFILE (Moving Left -> Right internally, but wait! We must trace a closed loop. ---
    # Let's trace CLOCKWISE around the Top Hatched Wall.
    # Start at the rightmost face, go Left along the inner boundary, then Up the left face, then Right along the outer boundary.
    
    # 1. Right Face & Chamfer (Clockwise)
    # Start at bottom of right face (inner thread)
    pts.append((31.0, 1.23))   # M3 inner radius
    pts.append((31.0, 5.0))    # Up to start of chamfer
    pts.append((30.0, 6.0))    # Chamfer to outer edge
    
    # 2. Outer Profile (Right to Left)
    pts.append((25.0, 6.0))    # Rightmost cylinder (Dia12)
    pts.append((25.0, 7.5))    # Step up to middle cylinder
    
    cx_fillet = 5.0 + math.sqrt(38.75) # 11.225
    pts.append((cx_fillet, 7.5)) # Middle cylinder (Dia15) ends at tangent point
    
    # Fillet (Tangent R9) - moving from tangent point UP to the corner
    cy_fillet = 16.5
    a_start = -math.pi / 2
    a_end = math.atan2(10.0 - cy_fillet, 5.0 - cx_fillet)
    
    steps = 20
    for i in range(1, steps):
        a = a_start + (a_end - a_start) * (i / steps)
        px = cx_fillet + 9.0 * math.cos(a)
        py = cy_fillet + 9.0 * math.sin(a)
        pts.append((px, py))
        
    pts.append((5.0, 10.0))    # End of fillet / corner
    
    # 3. Left Face (Convex R40.5)
    # Moving down the face from y=10 to y=2.09
    for y in np.linspace(10.0, 2.09, 20)[1:]:
        x = 40.5 - math.sqrt(40.5**2 - y**2)
        pts.append((x, y))
        
    # 4. Inner Profile (Left to Right)
    # M5 hole
    pts.append((15.0, 2.09))   # End of M5 hole
    pts.append((15.0, 4.5))    # Step up to Dia9 hole
    
    # Dia9 Clearance Hole
    pts.append((25.0, 4.5))    # End of Dia9 hole
    pts.append((25.0, 1.23))   # Step down to M3 hole
    
    # Return to start point to close loop
    # (cv2.polylines with isClosed=True handles the final segment)
    
    top_poly = np.array(pts, dtype=np.float32)
    
    # Bottom Hatched Wall
    bottom_poly = np.copy(top_poly)
    bottom_poly[:, 1] = -bottom_poly[:, 1]
    
    return [top_poly, bottom_poly]

def draw_full_contours(scale):
    polys = generate_agentic_contours()
    
    # Flip Y for OpenCV (since Y goes down)
    for p in polys:
        p[:, 1] = -p[:, 1]
        
    all_x = np.concatenate([p[:, 0] for p in polys])
    all_y = np.concatenate([p[:, 1] for p in polys])
    min_x = np.min(all_x)
    min_y = np.min(all_y)
    
    draw_polys = []
    for p in polys:
        p[:, 0] -= min_x
        p[:, 1] -= min_y
        p *= scale
        draw_polys.append(np.round(p).astype(np.int32))
        
    max_x = np.max(np.concatenate([p[:, 0] for p in draw_polys]))
    max_y = np.max(np.concatenate([p[:, 1] for p in draw_polys]))
    
    img = np.zeros((int(max_y + 5), int(max_x + 5)), dtype=np.uint8)
    cv2.polylines(img, draw_polys, isClosed=True, color=255, thickness=2)
    return img

def evaluate_alignment_robust(dist_transform, template, x, y):
    th, tw = template.shape
    h, w = dist_transform.shape
    if x < 0 or y < 0 or x+tw > w or y+th > h:
        return float('inf')
    roi_dist = dist_transform[y:y+th, x:x+tw]
    template_mask = template > 0
    return np.sum(roi_dist[template_mask]) / np.count_nonzero(template_mask)

img = cv2.imread("technical_drawing.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape

edges = cv2.Canny(gray, 50, 150)
dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)

print("[Agent] Running localized Chamfer Match on Section View quadrant...")
min_score = float('inf')
opt_s, opt_x, opt_y = 1.0, 0, 0

for s in np.linspace(7.0, 7.6, 15):
    template = draw_full_contours(s)
    th, tw = template.shape
    for dy in range(20, int(H/2) - th, 3):
        for dx in range(int(W/2), W - tw, 3):
            score = evaluate_alignment_robust(dist_transform, template, dx, dy)
            if score < min_score:
                min_score = score
                opt_s, opt_x, opt_y = s, dx, dy

print(f"[Agent] Optimal Fit: Scale={opt_s:.2f}, X={opt_x}, Y={opt_y}, Score={min_score:.2f}")

print("[Agent] Performing fine-tuning alignment...")
fine_min_score = min_score
for s in np.linspace(opt_s - 0.1, opt_s + 0.1, 5):
    template = draw_full_contours(s)
    for dy in range(opt_y - 5, opt_y + 5, 1):
        for dx in range(opt_x - 5, opt_x + 5, 1):
            score = evaluate_alignment_robust(dist_transform, template, dx, dy)
            if score < fine_min_score:
                fine_min_score = score
                opt_s, opt_x, opt_y = s, dx, dy

print(f"[Agent] Refined Fit: Scale={opt_s:.2f}, X={opt_x}, Y={opt_y}, Score={fine_min_score:.2f}")

template = draw_full_contours(opt_s)
th, tw = template.shape
template_mask = template > 0

highlighted = img.copy()
yellow = np.zeros((th, tw, 3), dtype=np.uint8)
yellow[template_mask] = [0, 255, 255]

thick_yellow_mask = cv2.dilate(template_mask.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
roi = highlighted[opt_y:opt_y+th, opt_x:opt_x+tw]
roi[thick_yellow_mask] = yellow[thick_yellow_mask]
highlighted[opt_y:opt_y+th, opt_x:opt_x+tw] = roi

cv2.imwrite("agentic_highlighted_result_v5.jpg", highlighted)
print("Saved flawless hollow overlay to agentic_highlighted_result_v5.jpg")
