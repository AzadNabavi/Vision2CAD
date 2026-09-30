import cv2
import numpy as np
import ezdxf

def get_hatch_bbox(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    
    mask = np.zeros_like(gray)
    mask[0:int(h_img*0.6), int(w_img*0.4):w_img] = 255
    gray = cv2.bitwise_and(gray, mask)
    
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=10, maxLineGap=5)
    
    hatch_pts = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if (40 < angle < 50) or (130 < angle < 140):
                hatch_pts.append((x1, y1))
                hatch_pts.append((x2, y2))
                
    if hatch_pts:
        hatch_pts = np.array(hatch_pts)
        return cv2.boundingRect(hatch_pts)
    return None

def render_dxf_to_image(dxf_path, scale):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # We KNOW from CAD extraction that max length is 31 and max radius is 10.
    # The profile in CadQuery is drawn in +X, +Y quadrant starting near X=0.
    min_x, max_x = 0.0, 31.0
    min_y, max_y = 0.0, 10.0
    
    pad = 2
    width = int((max_x - min_x + 2*pad) * scale)
    height = int((max_y - min_y + 2*pad) * scale)
    
    img = np.zeros((height, width), dtype=np.uint8)
    
    for entity in msp:
        if entity.dxftype() == 'LINE':
            pt1 = (int((entity.dxf.start.x - min_x + pad) * scale), 
                   int((max_y - entity.dxf.start.y + pad) * scale))
            pt2 = (int((entity.dxf.end.x - min_x + pad) * scale), 
                   int((max_y - entity.dxf.end.y + pad) * scale))
            cv2.line(img, pt1, pt2, 255, 3)
            
        elif entity.dxftype() == 'ARC':
            cx = int((entity.dxf.center.x - min_x + pad) * scale)
            cy = int((max_y - entity.dxf.center.y + pad) * scale)
            r = int(entity.dxf.radius * scale)
            
            start_angle = 360 - entity.dxf.end_angle
            end_angle = 360 - entity.dxf.start_angle
            cv2.ellipse(img, (cx, cy), (r, r), 0, start_angle, end_angle, 255, 3)
            
    return img

def highlight_contour(image_path, dxf_path, output_path):
    bbox = get_hatch_bbox(image_path)
    if bbox is None:
        return
        
    hx, hy, hw, hh = bbox
    # CadQuery DXF is just the TOP half of the cross-section.
    # So its height is 10mm (radius). But the hatching covers the FULL diameter (20mm).
    # So the DXF needs to be mirrored to form the full contour!
    scale_x = hw / 31.0
    scale_y = hh / 20.0  # FULL diameter
    scale = (scale_x + scale_y) / 2.0
    
    template_top = render_dxf_to_image(dxf_path, scale)
    
    # Mirror the top template to create the full contour
    template_bottom = cv2.flip(template_top, 0)
    template = np.vstack((template_top, template_bottom))
    
    img = cv2.imread(image_path)
    
    th, tw = template.shape
    yellow = np.zeros((th, tw, 3), dtype=np.uint8)
    yellow[template > 0] = [0, 255, 255]
    
    cx, cy = hx + hw//2, hy + hh//2
    tx = max(0, cx - tw//2)
    ty = max(0, cy - th//2)
    
    roi = img[ty:ty+th, tx:tx+tw]
    
    # In case of out-of-bounds due to integer division
    th = roi.shape[0]
    tw = roi.shape[1]
    template = template[:th, :tw]
    yellow = yellow[:th, :tw]
    
    mask = template > 0
    roi[mask] = yellow[mask]
    img[ty:ty+th, tx:tx+tw] = roi
    
    cv2.imwrite(output_path, img)

if __name__ == "__main__":
    highlight_contour("technical_drawing.jpg", "claude_part_contour.dxf", "highlighted_contour.jpg")
