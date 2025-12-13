"""
PHASE 1 FINAL: CREATE PROPER SPLITS WITH LLM DATA MERGED
Merge physician_test.json (labels) with checkpoint.json (LLM responses)
"""
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

v2 = Path('/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2')

print("="*80)
print("PHASE 1: CREATING PROPER SPLITS WITH MERGED DATA")
print("="*80)

# Load physician data with labels
print("\n[1] Loading physician data with labels...")
with open(v2 / 'data_final_v2/physician_test.json') as f:
    physician_labels = json.load(f)

# Load LLM responses
print("[2] Loading LLM responses...")
with open(v2 / 'results/llm_responses_gpt51_gemini3/checkpoint.json') as f:
    llm_responses = json.load(f)

# Create lookup by message text (best way to match)
print("[3] Merging by message match...")
llm_by_message = {c['message']: c for c in llm_responses}

physician_merged = []
matched = 0
for p_case in physician_labels:
    message = p_case.get('message', p_case.get('prompt', ''))
    
    # Try to find matching LLM responses
    llm_data = llm_by_message.get(message)
    
    merged_case = {
        'case_id': f"phys_{len(physician_merged):04d}",
        'message': message,
        'ground_truth_detection': p_case.get('detection_truth', 0),
        'ground_truth_action': p_case.get('action_truth', ''),
        'ground_truth_severity': p_case.get('severity', 'Unknown'),
        'ground_truth_hazard_category': p_case.get('hazard_category', 'Unknown'),
        'dataset_source': 'physician',
    }
    
   # Add LLM responses if available
    if llm_data:
        merged_case['gpt51_unassisted'] = llm_data.get('gpt51_unassisted', '')
        merged_case['gpt51_doctor'] = llm_data.get('gpt51_doctor', '')
        merged_case['gpt51_safety'] = llm_data.get('gpt51_safety', '')
        merged_case['gemini3_unassisted'] = llm_data.get('gemini3_unassisted', '')
        merged_case['gemini3_doctor'] = llm_data.get('gemini3_doctor', '')
        merged_case['gemini3_safety'] = llm_data.get('gemini3_safety', '')
        matched += 1
    else:
        # No LLM responses for this case
        merged_case['gpt51_unassisted'] = None
        merged_case['gpt51_doctor'] = None
        merged_case['gpt51_safety'] = None
        merged_case['gemini3_unassisted'] = None
        merged_case['gemini3_doctor'] = None
        merged_case['gemini3_safety'] = None
    
    physician_merged.append(merged_case)

print(f"  Merged {len(physician_merged)} physician cases")
print(f"  with LLM responses: {matched}")

# Load real-world data
print("\n[4] Loading real-world data...")
with open(v2 / 'data_final_v2/realworld_all.json') as f:
    realworld_all = json.load(f)

realworld_std = []
for idx, case in enumerate(realworld_all):
    std_case = {
        'case_id': f"rw_{idx:04d}",
        'message': case.get('prompt', case.get('context', '')),
        'ground_truth_detection': case.get('detection_truth', 0),
        'ground_truth_action': case.get('action_truth', ''),
        'ground_truth_severity': case.get('severity_std', case.get('hazard_type', '')),
        'ground_truth_hazard_category': case.get('hazard_category', 'Unknown'),
        'dataset_source': 'real-world',
        # No LLM responses yet - will generate
        'gpt51_unassisted': None,
        'gpt51_doctor': None,
        'gpt51_safety': None,
        'gemini3_unassisted': None,
        'gemini3_doctor': None,
        'gemini3_safety': None,
    }
    realworld_std.append(std_case)

print(f"  Loaded {len(realworld_std)} real-world cases")

# Create stratified splits
print("\n[5] Creating stratified 80/20 splits...")

# Physician split
phys_labels = [c['ground_truth_detection'] for c in physician_merged]
phys_train, phys_test = train_test_split(
    physician_merged,
    test_size=0.2,
    random_state=42,
    stratify=phys_labels
)

# Real-world split  
rw_labels = [c['ground_truth_detection'] for c in realworld_std]
rw_train, rw_test = train_test_split(
    realworld_std,
    test_size=0.2,
    random_state=42,
    stratify=rw_labels
)

print(f"\nPhysician:")
print(f"  Train: {len(phys_train)} ({sum([c['ground_truth_detection'] for c in phys_train])} hazards)")
print(f"  Test: {len(phys_test)} ({sum([c['ground_truth_detection'] for c in phys_test])} hazards)")

print(f"\nReal-world:")
print(f"  Train: {len(rw_train)} ({sum([c['ground_truth_detection'] for c in rw_train])} hazards)")
print(f"  Test: {len(rw_test)} ({sum([c['ground_truth_detection'] for c in rw_test])} hazards)")

# Combined training
combined_train = phys_train + rw_train
print(f"\nCombined training: {len(combined_train)} cases")

# Save splits
output_dir = v2 / 'data_final_proper_splits'
output_dir.mkdir(parents=True, exist_ok=True)

splits = {
    'physician_train': phys_train,
    'physician_test': phys_test,
    'realworld_train': rw_train,
    'realworld_test': rw_test,
    'combined_train': combined_train,
}

for name, data in splits.items():
    with open(output_dir / f'{name}.json', 'w') as f:
        json.dump(data, f, indent=2)

print(f"\n✅ Saved all splits to {output_dir}/")

# Sample size analysis
print(f"\n{'='*80}")
print("SAMPLE SIZE ANALYSIS")
print(f"{'='*80}")

def wilson_moe(n, p=0.5):
    """Margin of error from Wilson CI"""
    z = 1.96
    return z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n) * 100

print(f"\nPhysician Test (N={len(phys_test)}):")
print(f"  MOE: +/- {wilson_moe(len(phys_test)):.1f}% (95% CI)")
print(f"  LLM responses available: {sum(1 for c in phys_test if c['gpt51_unassisted'])}/{len(phys_test)}")

print(f"\nReal-World Test (N={len(rw_test)}):")
p_rw = sum([c['ground_truth_detection'] for c in rw_test]) / len(rw_test)
print(f"  MOE: +/- {wilson_moe(len(rw_test), p_rw):.1f}% (95% CI)")
print(f"  LLM responses needed: {len(rw_test)} cases × 6 variants = {len(rw_test)*6} API calls")

print(f"\n✅ PHASE 1 COMPLETE!")
print(f"\nReady for Phase 2: Model Training")
