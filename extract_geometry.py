import cv2
import numpy as np
import json
import os
import argparse

def extract_contour(image_path, output_dir):
    """
    Extracts the outer contour from a technical drawing.
    Handles both 'yellow highlighted' images and clean B&W images.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 1. Check for Yellow Highlight
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define range for yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    yellow_pixels = cv2.countNonZero(yellow_mask)
    contours_to_export = []

    if yellow_pixels > 50:
        print(f"[*] Detected {yellow_pixels} yellow pixels. Using color mask extraction.")
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Keep largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contours_to_export.append(largest_contour)
            
    else:
        print("[*] No yellow highlight detected. Falling back to advanced morphological extraction.")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert so lines are white, background is black
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 1. Remove the large rectangular border and title block
        # We can find the largest contour (the page border) and erase it.
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filter out the page border (massive area) and tiny noise/text
            if 1000 < area < (img.shape[0] * img.shape[1] * 0.5):
                valid_contours.append(c)
                
        # Sort by area descending
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # The cross-section (A-A) and 3D isometric view are typically the largest solid objects.
        contours_to_export = valid_contours[:2]

    if not contours_to_export:
        print("[-] Could not extract any valid contours.")
        return

    # 3. Export & Visualize
    result_viz = img.copy()
    output_json = []

    for i, contour in enumerate(contours_to_export):
        # Simplify contour for cleaner CAD output
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw on visualization image
        cv2.drawContours(result_viz, [approx], -1, (0, 255, 0), 2)
        
        # Format for JSON
        points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
        output_json.append({"part_index": i, "points": points})

    # Save JSON coordinates
    json_path = os.path.join(output_dir, "extracted_contours.json")
    with open(json_path, 'w') as f:
        json.dump({"contours": output_json}, f, indent=4)
    
    # Save visualization
    viz_path = os.path.join(output_dir, "contour_visualization.png")
    cv2.imwrite(viz_path, result_viz)

    print(f"[+] Successfully extracted {len(contours_to_export)} contours.")
    print(f"[+] Coordinates saved to {json_path}")
    print(f"[+] Visualization saved to {viz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CAD contours from drawings.")
    parser.add_argument("image", help="Path to the technical drawing image")
    parser.add_argument("--out", default="cadaico_output", help="Output directory")
    args = parser.parse_args()
    
    extract_contour(args.image, args.out)
