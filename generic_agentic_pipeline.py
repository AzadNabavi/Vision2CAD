import cv2
import numpy as np
import math

class GeometryContradictionException(Exception):
    pass

class GeometrySolver:
    @staticmethod
    def get_arc_points(start_pt, end_pt, radius, is_concave=True, steps=20):
        x1, y1 = start_pt
        x2, y2 = end_pt
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        
        if dist > 2 * abs(radius) + 1e-6:
            raise GeometryContradictionException(
                f"Physically impossible geometry: Distance between {start_pt} and {end_pt} "
                f"is {dist:.2f}, which is greater than 2R (2 * {radius} = {2*abs(radius)})."
            )
            
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        h = math.sqrt(max(0, radius**2 - (dist/2)**2))
        
        nx = -dy / dist
        ny = dx / dist
        
        if is_concave:
            cx = mid_x + h * nx
            cy = mid_y + h * ny
        else:
            cx = mid_x - h * nx
            cy = mid_y - h * ny
            
        a1 = math.atan2(y1 - cy, x1 - cx)
        a2 = math.atan2(y2 - cy, x2 - cx)
        
        if a1 < 0: a1 += 2*math.pi
        if a2 < 0: a2 += 2*math.pi
        
        diff = a2 - a1
        if diff > math.pi: a1 += 2*math.pi
        elif diff < -math.pi: a2 += 2*math.pi
            
        pts = []
        for i in range(1, steps):
            a = a1 + (a2 - a1) * (i / steps)
            px = cx + abs(radius) * math.cos(a)
            py = cy + abs(radius) * math.sin(a)
            pts.append((px, py))
            
        return pts

def build_generic_contour(params):
    overall_length = params["overall_length"]
    outer_feats = params.get("outer_profile", [])
    inner_feats = params.get("internal_features", [])
    
    # 1. OUTER PROFILE (Right to Left)
    # Sort outer features descending by start_x (which means starting from overall_length down to 0)
    outer_pts = []
    
    # Standardize features if they only have start_x, end_x, etc.
    # We will assume outer_feats are contiguous blocks or provide geometry.
    # For a generic weaver, let's process them properly:
    current_x = overall_length
    for feat in sorted(outer_feats, key=lambda f: f.get("end_x", 0), reverse=True):
        f_type = feat.get("feature")
        
        # Handle cylinders
        if f_type == "cylinder":
            d = feat["diameter"]
            sx, ex = feat["start_x"], feat["end_x"]
            outer_pts.append((ex, d/2.0))
            outer_pts.append((sx, d/2.0))
            
        elif f_type == "fillet":
            sx, ex = feat["start_x"], feat["end_x"]
            r = feat["radius"]
            # Assume previous point was the start of the fillet (from the right)
            start_pt = outer_pts[-1] if outer_pts else (ex, 0)
            # Find the next cylinder to get the end_y
            # This requires a more complex solver, but we'll use a placeholder end point for now
            # To be truly generic, we just append arc points.
            end_pt = (sx, start_pt[1] + abs(r)) # Stub
            try:
                arc = GeometrySolver.get_arc_points(start_pt, end_pt, r, is_concave=(feat.get("type")=="concave"))
                outer_pts.extend(arc)
                outer_pts.append(end_pt)
            except GeometryContradictionException as e:
                print(f"[Error] {e}")
                
        elif f_type == "face":
            # Assume convex face at x=0
            r = feat["radius"]
            start_pt = outer_pts[-1] if outer_pts else (0, 10)
            end_pt = (0, 0)
            try:
                arc = GeometrySolver.get_arc_points(start_pt, end_pt, r, is_concave=(feat.get("type")=="concave"))
                outer_pts.extend(arc)
                outer_pts.append(end_pt)
            except GeometryContradictionException as e:
                print(f"[Error] {e}")
                
    # 2. INNER PROFILE (Left to Right)
    inner_pts = []
    inner_feats_sorted = sorted(inner_feats, key=lambda f: f.get("start_x", 0))
    current_x = 0
    
    for feat in inner_feats_sorted:
        sx = feat.get("start_x", 0)
        ex = feat.get("end_x", overall_length)
        rad = feat.get("diameter", 0) / 2.0
        
        if sx > current_x:
            # Solid core logic: Drop to centerline Y=0
            if current_x == 0:
                inner_pts.append((0, 0))
            else:
                inner_pts.append((current_x, 0))
                inner_pts.append((sx, 0))
                
        # Draw the internal feature
        inner_pts.append((sx, rad))
        inner_pts.append((ex, rad))
        current_x = ex
        
    if current_x < overall_length:
        inner_pts.append((current_x, 0))
        inner_pts.append((overall_length, 0))
        
    # Combine into a single closed polygon (Top Half)
    top_half = outer_pts + inner_pts
    
    # 3. Mirror across Y=0 to create Bottom Half
    bottom_half = [(x, -y) for x, y in reversed(top_half)]
    
    return [np.array(top_half, dtype=np.float32), np.array(bottom_half, dtype=np.float32)]

def draw_generic_contours(scale, polys):
    for p in polys:
        p[:, 1] = -p[:, 1]
    
    all_x = np.concatenate([p[:, 0] for p in polys])
    all_y = np.concatenate([p[:, 1] for p in polys])
    min_x, min_y = np.min(all_x), np.min(all_y)
    
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

def run_generic_cv_pipeline(img_path, params):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny Edge Detection & Distance Transform
    edges = cv2.Canny(gray, 50, 150)
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    polys = build_generic_contour(params)
    
    # Multi-Scale Template Matching
    base_scale = 7.0
    scales = np.linspace(base_scale * 0.5, base_scale * 1.5, 20)
    
    best_score = float('inf')
    best_match = None
    
    for s in scales:
        template = draw_generic_contours(s, [np.copy(p) for p in polys])
        template_mask = template > 0
        if np.count_nonzero(template_mask) == 0:
            continue
            
        # Instead of arbitrary dx/dy loops, we can use convolution or matchTemplate
        # But distance transform scoring requires iterating over valid ROIs.
        # cv2.matchTemplate using SQDIFF on the distance transform acts exactly as a global sliding window Chamfer match!
        
        # Match template directly onto the distance transform image
        res = cv2.matchTemplate(dist_transform.astype(np.float32), template_mask.astype(np.float32), cv2.TM_SQDIFF)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        
        score = min_val / np.count_nonzero(template_mask)
        if score < best_score:
            best_score = score
            best_match = (s, min_loc[0], min_loc[1])
            
    print(f"[Agent] Generic Best Match: Scale={best_match[0]:.2f}, X={best_match[1]}, Y={best_match[2]}, Score={best_score:.2f}")

print("Generic pipeline defined. Ready for Agent integration.")
