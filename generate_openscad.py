import json
import math

def generate_openscad():
    # 1. Calculate the points of the 2D cross-section
    r_dome = 40.5
    y_cyl = 10.0
    x_intersect = r_dome - math.sqrt(r_dome**2 - y_cyl**2)

    # Base points
    pts = []
    pts.append((31.0, 1.25))
    pts.append((31.0, 4.5))
    pts.append((25.0, 4.5))
    pts.append((25.0, 7.5))
    pts.append((12.5, 7.5))
    pts.append((12.5, 6.0))

    # Fillet R9 concave from (12.5, 6.0) to (5.0, 10.0)
    # We approximate the arc with straight line segments for OpenSCAD polygon
    # Center of arc: x=12.5, y=15.0
    # Start angle: -90 degrees (at 12.5, 6.0 relative to center)
    # End angle: angle to (5.0, 10.0)
    cx, cy = 12.5, 15.0
    start_angle = math.atan2(6.0 - cy, 12.5 - cx)
    end_angle = math.atan2(10.0 - cy, 5.0 - cx)
    
    steps = 20
    for i in range(1, steps + 1):
        t = i / steps
        angle = start_angle + t * (end_angle - start_angle)
        x = cx + 9.0 * math.cos(angle)
        y = cy + 9.0 * math.sin(angle)
        pts.append((x, y))
        
    pts.append((x_intersect, 10.0))

    # Convex face R40.5 ending at left bore edge
    left_bore_r = 2.1 # M5 minor is 4.2
    x_face_start = 40.5 - math.sqrt(40.5**2 - left_bore_r**2)
    
    # Arc from (x_intersect, 10.0) to (x_face_start, 2.1)
    # Center is at (40.5, 0)
    cx2, cy2 = 40.5, 0.0
    start_angle2 = math.atan2(10.0 - cy2, x_intersect - cx2)
    end_angle2 = math.atan2(left_bore_r - cy2, x_face_start - cx2)
    
    for i in range(1, steps + 1):
        t = i / steps
        angle = start_angle2 + t * (end_angle2 - start_angle2)
        x = cx2 + 40.5 * math.cos(angle)
        y = cy2 + 40.5 * math.sin(angle)
        pts.append((x, y))

    # Internal bores
    pts.append((15.0, left_bore_r))
    pts.append((21.0, 1.25))
    pts.append((31.0, 1.25))

    # 2. Write the OpenSCAD file for 3D model
    scad_code = "// Auto-generated OpenSCAD script from extracted JSON parameters\n\n"
    scad_code += "$fn = 100; // Resolution for curves\n\n"
    scad_code += "rotate_extrude(angle=360) {\n"
    scad_code += "    polygon(points=[\n"
    for p in pts:
        # Swap X and Y: OpenSCAD revolves around the Z axis.
        # It expects X to be the radius and Y to be the height/length.
        scad_code += f"        [{p[1]:.3f}, {p[0]:.3f}],\n"
    scad_code += "    ]);\n"
    scad_code += "}\n"

    with open("generated_part.scad", "w") as f:
        f.write(scad_code)
    print("Saved generated_part.scad")
    
    # 3. Write a purely 2D equivalent via SCAD so the user can see the cross-section
    scad_code_2d = "// 2D Section View of the part\n\n"
    scad_code_2d += "$fn = 100;\n\n"
    scad_code_2d += "// Top half\n"
    scad_code_2d += "polygon(points=[\n"
    for p in pts:
        scad_code_2d += f"    [{p[0]:.3f}, {p[1]:.3f}],\n"
    scad_code_2d += "]);\n\n"
    scad_code_2d += "// Bottom half (mirrored)\n"
    scad_code_2d += "mirror([0, 1, 0]) {\n"
    scad_code_2d += "    polygon(points=[\n"
    for p in pts:
        scad_code_2d += f"        [{p[0]:.3f}, {p[1]:.3f}],\n"
    scad_code_2d += "    ]);\n"
    scad_code_2d += "}\n"
    
    with open("section_cut.scad", "w") as f:
        f.write(scad_code_2d)
    print("Saved section_cut.scad")

if __name__ == "__main__":
    generate_openscad()
