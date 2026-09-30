import json
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Load the ground truth (backup)
ground_truth = load_json("backup_working_v1/final_council_extracted_parameters.json")

def compare_json(test, truth):
    errors = []
    
    if test.get("overall_length") != truth.get("overall_length"):
        errors.append(f"overall_length mismatch: expected {truth.get('overall_length')}, got {test.get('overall_length')}")
        
    test_outer = test.get("outer_profile", [])
    truth_outer = truth.get("outer_profile", [])
    
    if len(test_outer) != len(truth_outer):
        errors.append(f"outer_profile length mismatch: expected {len(truth_outer)}, got {len(test_outer)}")
    else:
        for i, (t_item, truth_item) in enumerate(zip(test_outer, truth_outer)):
            for key in truth_item.keys():
                if t_item.get(key) != truth_item.get(key):
                    errors.append(f"outer_profile[{i}].{key} mismatch: expected {truth_item.get(key)}, got {t_item.get(key)}")
                    
    test_inner = test.get("internal_features", [])
    truth_inner = truth.get("internal_features", [])
    
    if len(test_inner) != len(truth_inner):
        errors.append(f"internal_features length mismatch: expected {len(truth_inner)}, got {len(test_inner)}")
    else:
        for i, (t_item, truth_item) in enumerate(zip(test_inner, truth_inner)):
            for key in truth_item.keys():
                if t_item.get(key) != truth_item.get(key):
                    errors.append(f"internal_features[{i}].{key} mismatch: expected {truth_item.get(key)}, got {t_item.get(key)}")
                    
    return errors

success_count = 0
total_runs = 10
detailed_results = []

for i in range(1, 11):
    file_path = f"reliability_test_results/run_{i}.json"
    if not os.path.exists(file_path):
        detailed_results.append(f"Run {i}: Missing File")
        continue
        
    try:
        test_json = load_json(file_path)
        errors = compare_json(test_json, ground_truth)
        if len(errors) == 0:
            success_count += 1
            detailed_results.append(f"Run {i}: PERFECT MATCH")
        else:
            detailed_results.append(f"Run {i}: FAILED ({len(errors)} errors)\n   - " + "\n   - ".join(errors[:3]) + ("\n   - ..." if len(errors)>3 else ""))
    except Exception as e:
        detailed_results.append(f"Run {i}: FAILED (Parse Error: {e})")

print(f"\n--- RELIABILITY TEST RESULTS ---")
print(f"Accuracy Rate: {success_count}/{total_runs} ({(success_count/total_runs)*100:.1f}%)")
print("\nDetailed Breakdown:")
for r in detailed_results:
    print(r)
