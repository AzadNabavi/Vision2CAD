import json
import os

print("\n=== AI GEOMETRY SOLVER CONSISTENCY REPORT (5 RUNS) ===\n")

for i in range(1, 6):
    path = f"ai_geometry_test_results/run_{i}.json"
    if not os.path.exists(path):
        print(f"Run {i}: File missing")
        continue
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            
        outer = data.get("outer_profile", [])
        
        # We are looking for the 'face' feature (the dome)
        dome = next((feat for feat in outer if feat.get("feature") == "face" and feat.get("type") == "convex"), None)
        
        # We are also looking for the cylinder right before it (where y=10)
        cyl = next((feat for feat in outer if feat.get("feature") == "cylinder" and feat.get("start_y") == 10.0), None)
        
        print(f"--- RUN {i} ---")
        if dome:
            print(f"  Dome start_x (Calculated X Intersection): {dome.get('start_x')}")
            print(f"  Dome end_y (Calculated Y termination):    {dome.get('end_y')}")
        else:
            print("  Dome feature missing or misclassified.")
            
        if cyl:
            print(f"  Adjacent Cylinder end_x:                  {cyl.get('end_x')}")
        else:
            print("  Adjacent cylinder missing.")
            
        print("")
            
    except Exception as e:
        print(f"Run {i}: Error reading JSON - {e}")
