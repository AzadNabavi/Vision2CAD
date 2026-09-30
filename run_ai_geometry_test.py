import os
import subprocess
import shutil
import time

output_dir = "ai_geometry_test_results"
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 6):
    print(f"--- Running AI Geometry Test Iteration {i}/5 ---")
    start_time = time.time()
    
    # Run the multi-agent council (which now includes the AI Geometric Pass)
    process = subprocess.run(["python", "multi_agent_council.py"], capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error in iteration {i}:\n{process.stderr}")
    else:
        if os.path.exists("final_council_extracted_parameters.json"):
            dest_json = os.path.join(output_dir, f"run_{i}.json")
            shutil.copy("final_council_extracted_parameters.json", dest_json)
            print(f"Saved {dest_json}")
        else:
            print(f"Warning: JSON file was not generated in iteration {i}")
            
    print(f"Iteration {i} completed in {time.time() - start_time:.1f} seconds.\n")
    
    time.sleep(2)

print("All 5 iterations completed.")
