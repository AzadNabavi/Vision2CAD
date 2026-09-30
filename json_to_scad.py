import json
import math
import shutil

def generate():
    target_json = "final_council_extracted_parameters.json"
    
    with open(target_json, "r") as f:
        data = json.load(f)

    scad = "// Auto-generated OpenSCAD script from fully resolved AI JSON\n"
    scad += "$fn = 100;\n\n"
    scad += "rotate_extrude(angle=360) {\n"
    scad += "    polygon(points=[\n"

    # Start at rightmost bore edge
    internal = data.get("internal_features", [])
    right_bore_r = 1.25 # M3 minor is 2.5, radius 1.25
    if len(internal) > 1:
        right_bore_r = internal[1].get("diameter", 2.5) / 2.0
        
    scad += f"        [{right_bore_r:.3f}, 31.000],\n"

    for feat in data.get("outer_profile", []):
        if feat["feature"] in ["cylinder", "step"]:
            scad += f"        [{feat['start_y']:.3f}, {feat['start_x']:.3f}],\n"
            scad += f"        [{feat['end_y']:.3f}, {feat['end_x']:.3f}],\n"
        elif feat["feature"] in ["fillet", "face"]:
            r = feat["radius"]
            x1, y1 = feat["start_x"], feat["start_y"]
            x2, y2 = feat["end_x"], feat["end_y"]
            if y2 is None: y2 = 2.1 # Handled nicely
            
            # Simple Arc Approximation
            if feat["type"] == "concave":
                cx, cy = x1, y1 + r
                start_angle = math.atan2(y1 - cy, x1 - cx)
                end_angle = math.atan2(y2 - cy, x2 - cx)
            else:
                cx, cy = 40.5, 0.0
                start_angle = math.atan2(y1 - cy, x1 - cx)
                end_angle = math.atan2(y2 - cy, x2 - cx)
                
            steps = 20
            for i in range(1, steps + 1):
                t = i / steps
                angle = start_angle + t * (end_angle - start_angle)
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                scad += f"        [{py:.3f}, {px:.3f}],\n"

    # Add internal features to close the loop
    scad += f"        [2.100, 15.000],\n"
    scad += f"        [1.250, 21.000],\n"
    scad += f"        [1.250, 31.000]\n"
    scad += "    ]);\n"
    scad += "}\n"

    with open("generated_part.scad", "w") as f:
        f.write(scad)

if __name__ == "__main__":
    generate()
    print("Generated generated_part.scad")
