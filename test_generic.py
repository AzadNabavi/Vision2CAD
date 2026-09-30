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

def normalize_params(params):
    """
    Normalize LLM JSON so the generic parser can handle it regardless of direction:
    - Internal features: ensure start_x < end_x (left to right)
    - Face feature: if zero-width (start_x == end_x), compute true start_x from R40.5 geometry
    - Outer profile: ensure start_x > end_x (right to left, as parser expects)
    """
    import math

    # Normalize internal features to left-to-right
    for feat in params.get("internal_features", []):
        sx, ex = feat.get("start_x", 0), feat.get("end_x", 0)
        if sx > ex:
            feat["start_x"], feat["end_x"] = ex, sx

    # Fix outer profile features to be right-to-left
    for feat in params.get("outer_profile", []):
        sx, ex = feat.get("start_x", 0), feat.get("end_x", 0)
        # For non-step features, ensure start_x >= end_x
        if feat.get("feature") not in ("step",) and sx < ex:
            feat["start_x"], feat["end_x"] = ex, sx
            feat["start_y"], feat["end_y"] = feat.get("end_y", 0), feat.get("start_y", 0)

    # Fix zero-width face: compute true start_x from R40.5 and the face's start_y
    for feat in params.get("outer_profile", []):
        if feat.get("feature") == "face":
            if abs(feat.get("start_x", 0) - feat.get("end_x", 0)) < 0.01:
                r = feat.get("radius", 40.5)
                y = feat.get("start_y", 10.0)
                # x offset from left tip for a convex dome: x = r - sqrt(r^2 - y^2)
                try:
                    start_x = r - math.sqrt(r**2 - y**2)
                    feat["start_x"] = round(start_x, 3)
                    feat["end_x"] = 0.0
                except Exception:
                    pass
            break

    return params


def build_generic_contour(params):
    overall_length = params["overall_length"]
    outer_feats = params.get("outer_profile", [])
    inner_feats = params.get("internal_features", [])
    
    # 1. OUTER PROFILE
    # Because JSON can be arbitrary, we trust start_x, end_x, start_y, end_y to build points right to left.
    outer_pts = []
    
    # We expect features to be ordered from Right to Left explicitly in this JSON design,
    # or we sort them by start_x descending.
    for feat in sorted(outer_feats, key=lambda f: f.get("start_x", 0), reverse=True):
        f_type = feat.get("feature")
        sx, ex = feat.get("start_x"), feat.get("end_x")
        sy, ey = feat.get("start_y", 0), feat.get("end_y", 0)
        
        if f_type in ["cylinder", "chamfer", "step"]:
            # Straight line from start to end
            # Only add start point if it doesn't match the last point exactly to avoid duplicates
            if not outer_pts or outer_pts[-1] != (sx, sy):
                outer_pts.append((sx, sy))
            outer_pts.append((ex, ey))
            
        elif f_type in ["fillet", "face"]:
            r = feat["radius"]
            is_concave = (feat.get("type") == "concave")
            if not outer_pts or outer_pts[-1] != (sx, sy):
                outer_pts.append((sx, sy))
            
            try:
                arc = GeometrySolver.get_arc_points((sx, sy), (ex, ey), r, is_concave=is_concave)
                outer_pts.extend(arc)
                outer_pts.append((ex, ey))
            except GeometryContradictionException as e:
                print(f"[Error] {e}")
                outer_pts.append((ex, ey)) # Fallback to straight line
                
    # 2. INNER PROFILE (Left to Right)
    inner_pts = []
    inner_feats_sorted = sorted(inner_feats, key=lambda f: f.get("start_x", 0))
    current_x = 0
    
    for feat in inner_feats_sorted:
        sx = feat.get("start_x", 0)
        ex = feat.get("end_x", overall_length)
        rad = feat.get("diameter", 0) / 2.0
        
        if sx > current_x + 1e-5:
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
    # The outer profile goes from right to left.
    # We append the inner profile which goes from left to right.
    # This forms a perfect closed loop.
    top_half = outer_pts + inner_pts
    
    # 3. Mirror across Y=0 to create Bottom Half
    # It also forms a closed loop, starting from the rightmost outer point.
    bottom_half = [(x, -y) for x, y in top_half]
    
    return [np.array(top_half, dtype=np.float32), np.array(bottom_half, dtype=np.float32)]

def draw_generic_contours(scale, polys):
    for p in polys:
        p[:, 1] = -p[:, 1] # OpenCV Y is down
    
    all_x = np.concatenate([p[:, 0] for p in polys])
    all_y = np.concatenate([p[:, 1] for p in polys])
    min_x, min_y = np.min(all_x), np.min(all_y)
    
    draw_polys = []
    for p in polys:
        p_shifted = np.copy(p)
        p_shifted[:, 0] -= min_x
        p_shifted[:, 1] -= min_y
        p_shifted *= scale
        draw_polys.append(np.round(p_shifted).astype(np.int32))
        
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
    
    # Normalize LLM output (fix directions, zero-width face, etc.)
    params = normalize_params(params)
    polys = build_generic_contour(params)
    
    base_scale = 7.0
    scales = np.linspace(7.0, 7.6, 15)
    
    best_score = float('inf')
    best_match = None
    
    print("[Agent] Sweeping MatchTemplate convolution...")
    for s in scales:
        template = draw_generic_contours(s, [np.copy(p) for p in polys])
        template_mask = template > 0
        
        # We only want to search the top right quadrant to save time and avoid title block
        H, W = dist_transform.shape
        roi = dist_transform[0:int(H/2), int(W/2):W]
        
        res = cv2.matchTemplate(roi.astype(np.float32), template_mask.astype(np.float32), cv2.TM_CCORR)
        min_val, _, min_loc, _ = cv2.minMaxLoc(res)
        
        score = min_val / np.count_nonzero(template_mask)
        if score < best_score:
            best_score = score
            # Offset min_loc back by the ROI shift
            best_match = (s, min_loc[0] + int(W/2), min_loc[1], template_mask)
            
    print(f"[Agent] Generic Best Match: Scale={best_match[0]:.2f}, X={best_match[1]}, Y={best_match[2]}, Score={best_score:.2f}")
    
    # Draw Final Overlay
    opt_s, opt_x, opt_y, template_mask = best_match
    th, tw = template_mask.shape
    
    highlighted = img.copy()
    yellow = np.zeros((th, tw, 3), dtype=np.uint8)
    yellow[template_mask] = [0, 255, 255]
    
    thick_yellow_mask = cv2.dilate(template_mask.astype(np.uint8), np.ones((5,5), np.uint8)) > 0
    roi_img = highlighted[opt_y:opt_y+th, opt_x:opt_x+tw]
    roi_img[thick_yellow_mask] = yellow[thick_yellow_mask]
    highlighted[opt_y:opt_y+th, opt_x:opt_x+tw] = roi_img
    
    cv2.imwrite("agentic_highlighted_result_v10.jpg", highlighted)
    print("Saved perfectly generic overlay to agentic_highlighted_result_v10.jpg")

# The fully parsed CAD parameters representing the part
CAD_JSON = {
    "overall_length": 31,
    "outer_profile": [
        {"feature": "chamfer", "start_x": 31, "end_x": 30, "start_y": 5.0, "end_y": 6.0},
        {"feature": "cylinder", "start_x": 30, "end_x": 25, "start_y": 6.0, "end_y": 6.0},
        {"feature": "step", "start_x": 25, "end_x": 25, "start_y": 6.0, "end_y": 7.5},
        {"feature": "cylinder", "start_x": 25, "end_x": 11.225, "start_y": 7.5, "end_y": 7.5},
        {"feature": "fillet", "start_x": 11.225, "end_x": 5, "start_y": 7.5, "end_y": 10.0, "radius": 9, "type": "concave"},
        {"feature": "cylinder", "start_x": 5, "end_x": 1.254, "start_y": 10.0, "end_y": 10.0},
        {"feature": "face", "start_x": 1.254, "end_x": 0.054, "start_y": 10.0, "end_y": 2.09, "radius": 40.5, "type": "convex"} 
    ],
    "internal_features": [
        {"feature": "thread", "start_x": 0.054, "end_x": 15, "diameter": 4.18},
        {"feature": "hole", "start_x": 15, "end_x": 25, "diameter": 9},
        {"feature": "thread", "start_x": 25, "end_x": 31, "diameter": 2.46} 
    ]
}

if __name__ == "__main__":
    import json; run_generic_cv_pipeline("technical_drawing.jpg", json.load(open("final_council_extracted_parameters.json")))
