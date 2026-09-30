import os
import subprocess
import shutil
import time

output_dir = "reliability_test_results"
os.makedirs(output_dir, exist_ok=True)

# Restore the exact backup version of the script
shutil.copy("backup_working_v1/multi_agent_council.py", "multi_agent_council.py")

for i in range(1, 11):
    print(f"--- Running Test Iteration {i}/10 ---")
    start_time = time.time()
    
    # Run the multi-agent council
    process = subprocess.run(["python", "multi_agent_council.py"], capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error in iteration {i}:\n{process.stderr}")
    else:
        # Save the resulting JSON
        if os.path.exists("final_council_extracted_parameters.json"):
            dest_json = os.path.join(output_dir, f"run_{i}.json")
            shutil.copy("final_council_extracted_parameters.json", dest_json)
            print(f"Saved {dest_json}")
        else:
            print(f"Warning: JSON file was not generated in iteration {i}")
            
    print(f"Iteration {i} completed in {time.time() - start_time:.1f} seconds.\n")
    
    # Pause slightly to avoid OpenAI rate limits
    time.sleep(2)

print("All 10 iterations completed.")
