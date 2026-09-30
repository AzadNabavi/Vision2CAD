import cv2
import numpy as np
import json
import os
import argparse
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

def main(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Loading Image...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[-] Could not load {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    height, width, _ = image_bgr.shape

    print("[*] Loading Meta SAM (Segment Anything) Foundation Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Neural Processing Unit: {device.upper()}")
    
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    # The Drehteil cross-section is in the top-right quadrant.
    # We provide SAM with a bounding box prompt encompassing that region.
    # [xmin, ymin, xmax, ymax]
    box = [int(width * 0.45), int(height * 0.05), int(width * 0.95), int(height * 0.55)]
    input_boxes = [[box]]
    
    print(f"[*] Prompting SAM with spatial bounding box: {box}")
    
    inputs = processor(pil_image, input_boxes=[input_boxes], return_tensors="pt").to(device)
    
    print("[*] Executing Zero-Shot Neural Segmentation...")
    with torch.no_grad():
        outputs = model(**inputs)
        
    # SAM predicts 3 masks (different levels of detail). We take the one with the highest confidence.
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    iou_scores = outputs.iou_scores.cpu().numpy()[0][0]
    best_mask_idx = np.argmax(iou_scores)
    
    best_mask = masks[0][0][best_mask_idx].numpy()
    
    # Convert mask to OpenCV format
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    
    # Post-process: Close the hatched gaps inside the component
    kernel = np.ones((7,7), np.uint8)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    print("[*] Vectorizing Segmented Mask...")
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[-] Extraction failed.")
        return
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify the contour polygon for clean CAD export
    epsilon = 0.003 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Visualization
    viz_img = image_bgr.copy()
    
    # Draw SAM Box Prompt
    cv2.rectangle(viz_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.putText(viz_img, "SAM Prompt Box", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw extracted SAM boundary
    cv2.drawContours(viz_img, [approx], -1, (0, 255, 0), 3)
    
    # Export Coordinates
    points = [{"x": float(pt[0][0]), "y": float(pt[0][1])} for pt in approx]
    json_data = {
        "model": "Meta Segment Anything (SAM-ViT-Base)",
        "method": "Zero-shot Bounding Box Prompting",
        "contours": [{"part_index": 0, "points": points}]
    }
    
    json_path = os.path.join(output_dir, "sam_contours.json")
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    viz_path = os.path.join(output_dir, "sam_visualization.png")
    cv2.imwrite(viz_path, viz_img)
    
    print(f"[+] AI Geometry successfully saved to {json_path}")
    print(f"[+] Neural visualization saved to {viz_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to technical drawing")
    parser.add_argument("--out", default="Output", help="Output directory")
    args = parser.parse_args()
    main(args.image, args.out)
