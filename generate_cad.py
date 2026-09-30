import json
import math
import cadquery as cq

def generate_part():
    with open("final_council_extracted_parameters.json", "r") as f:
        params = json.load(f)

    r_dome = 40.5
    y_cyl = 10.0
    x_intersect = r_dome - math.sqrt(r_dome**2 - y_cyl**2)
    print(f"Calculated dome intersection x: {x_intersect:.3f}")

    wp = cq.Workplane("XZ")
    
    # Trace the perimeter of the cross section above the X axis (y=0)
    res = wp.moveTo(31.0, 1.25) \
            .lineTo(31.0, 4.5) \
            .lineTo(25.0, 4.5) \
            .lineTo(25.0, 7.5) \
            .lineTo(12.5, 7.5) \
            .lineTo(12.5, 6.0)
            
    # Concave fillet R9
    res = res.radiusArc((5.0, 10.0), -9.0)
    
    # Cylinder Dia20
    res = res.lineTo(x_intersect, 10.0)
    
    # Convex face R40.5 ending at left bore edge
    left_bore_r = 2.1 # M5 minor is 4.2
    x_face_start = 40.5 - math.sqrt(40.5**2 - left_bore_r**2)
    res = res.radiusArc((x_face_start, left_bore_r), 40.5)
    
    # Internal bores
    res = res.lineTo(15.0, left_bore_r)
    
    # Connect to the M3 bore
    res = res.lineTo(21.0, 1.25)
    
    # M3 bore
    res = res.lineTo(31.0, 1.25)
    
    # Close profile
    res = res.close()
    
    # Revolve 360 degrees around X axis
    part = res.revolve(360, (0,0,0), (1,0,0))
    
    # Export STEP
    cq.exporters.export(part, "generated_part.step")
    print("Saved generated_part.step")
    
    # Slice the part at the XZ plane to get the 2D cross section
    # We will export the section face to SVG
    section = part.section()
    cq.exporters.export(section, "section_cut.svg")
    print("Saved section_cut.svg")

if __name__ == "__main__":
    generate_part()
