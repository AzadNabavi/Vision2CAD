import cv2
import numpy as np
import ezdxf

def render_dxf_to_image(dxf_path, scale=10.0):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            min_x = min(min_x, entity.dxf.start.x, entity.dxf.end.x)
            max_x = max(max_x, entity.dxf.start.x, entity.dxf.end.x)
            min_y = min(min_y, entity.dxf.start.y, entity.dxf.end.y)
            max_y = max(max_y, entity.dxf.start.y, entity.dxf.end.y)
        elif entity.dxftype() == 'ARC':
            cx, cy = entity.dxf.center.x, entity.dxf.center.y
            r = entity.dxf.radius
            min_x = min(min_x, cx - r)
            max_x = max(max_x, cx + r)
            min_y = min(min_y, cy - r)
            max_y = max(max_y, cy + r)
            
    pad = 5
    width = int((max_x - min_x + 2*pad) * scale)
    height = int((max_y - min_y + 2*pad) * scale)
    
    img = np.zeros((height, width), dtype=np.uint8)
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            # Invert Y axis: max_y - y
            pt1 = (int((entity.dxf.start.x - min_x + pad) * scale), 
                   int((max_y - entity.dxf.start.y + pad) * scale))
            pt2 = (int((entity.dxf.end.x - min_x + pad) * scale), 
                   int((max_y - entity.dxf.end.y + pad) * scale))
            cv2.line(img, pt1, pt2, 255, 2)
            
        elif entity.dxftype() == 'ARC':
            cx = int((entity.dxf.center.x - min_x + pad) * scale)
            cy = int((max_y - entity.dxf.center.y + pad) * scale)
            r = int(entity.dxf.radius * scale)
            
            # DXF angles are CCW from positive X. OpenCV ellipse angles are clockwise.
            # So start and end angles are negated/swapped.
            start_angle = 360 - entity.dxf.end_angle
            end_angle = 360 - entity.dxf.start_angle
            
            cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, 255, 2)
            
    return img

def align_and_highlight(image_path, dxf_path, output_path):
    print("Loading original image...")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Isolate section view visually to help matching
    h, w = gray.shape
    mask_tr = np.zeros_like(gray)
    mask_tr[0:int(h*0.6), int(w*0.4):w] = 255
    
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, mask_tr)
    
    best_score = -1
    best_scale = 1.0
    best_loc = (0, 0)
    best_template = None
    
    scales = np.linspace(7.0, 11.0, 100)
    
    for scale in scales:
        template = render_dxf_to_image(dxf_path, scale)
        if template.shape[0] > thresh.shape[0] or template.shape[1] > thresh.shape[1]:
            continue
            
        res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = max_val
            best_scale = scale
            best_loc = max_loc
            best_template = template
            
    print(f"Best match found! Score: {best_score:.3f} at scale {best_scale:.2f} px/mm")
    
    highlighted = img.copy()
    th, tw = best_template.shape
    
    x, y = best_loc
    roi = highlighted[y:y+th, x:x+tw]
    
    # Create thick yellow mask
    kernel = np.ones((4,4), np.uint8)
    thick_mask = cv2.dilate(best_template, kernel, iterations=1) > 0
    
    yellow = np.zeros_like(roi)
    yellow[thick_mask] = [0, 255, 255]
    
    roi[thick_mask] = yellow[thick_mask]
    highlighted[y:y+th, x:x+tw] = roi
    
    cv2.imwrite(output_path, highlighted)
    print("Done")

if __name__ == "__main__":
    align_and_highlight("technical_drawing.jpg", "claude_part_contour.dxf", "highlighted_contour.jpg")
