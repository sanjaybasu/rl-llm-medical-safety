"""
SUBGROUP ANALYSIS
Compute metrics by Age, Sex, Race, and Hazard Category using the Audit CSV
"""
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import numpy as np

print("="*80)
print("SUBGROUP ANALYSIS")
print("="*80)

# Load Audit CSV
df = pd.read_csv('results/realworld_audit.csv')
print(f"Loaded {len(df)} cases")

# Helper to compute metrics for a group
def compute_metrics(group_df, system_prefix):
    y_true_det = group_df['ground_truth_detection']
    y_pred_det = group_df[f'{system_prefix}_det']
    
    # MCC
    try:
        mcc = matthews_corrcoef(y_true_det, y_pred_det)
    except:
        mcc = 0.0
        
    # Sensitivity
    tp = ((y_true_det == 1) & (y_pred_det == 1)).sum()
    fn = ((y_true_det == 1) & (y_pred_det == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return sens, mcc

# Systems to analyze
systems = ['guardrail', 'cql', 'awr'] # LLMs skipped if partial
results = []

# 1. By Hazard Category
print("\n--- By Hazard Category ---")
for cat in df['hazard_category'].unique():
    if pd.isna(cat): continue
    group = df[df['hazard_category'] == cat]
    if len(group) < 5: continue
    
    print(f"\nCategory: {cat} (N={len(group)})")
    for sys in systems:
        sens, mcc = compute_metrics(group, sys)
        print(f"  {sys}: Sens={sens:.3f}, MCC={mcc:.3f}")
        results.append({'Group_Type': 'Hazard_Category', 'Group': cat, 'N': len(group), 'System': sys, 'Sensitivity': sens, 'MCC': mcc})

# 2. By Sex
print("\n--- By Sex ---")
for sex in df['sex'].unique():
    if pd.isna(sex): continue
    group = df[df['sex'] == sex]
    print(f"\nSex: {sex} (N={len(group)})")
    for sys in systems:
        sens, mcc = compute_metrics(group, sys)
        print(f"  {sys}: Sens={sens:.3f}, MCC={mcc:.3f}")
        results.append({'Group_Type': 'Sex', 'Group': sex, 'N': len(group), 'System': sys, 'Sensitivity': sens, 'MCC': mcc})

# 3. By Race
print("\n--- By Race ---")
for race in df['race'].unique():
    if pd.isna(race): continue
    group = df[df['race'] == race]
    if len(group) < 5: continue
    print(f"\nRace: {race} (N={len(group)})")
    for sys in systems:
        sens, mcc = compute_metrics(group, sys)
        print(f"  {sys}: Sens={sens:.3f}, MCC={mcc:.3f}")
        results.append({'Group_Type': 'Race', 'Group': race, 'N': len(group), 'System': sys, 'Sensitivity': sens, 'MCC': mcc})

# Save to CSV
pd.DataFrame(results).to_csv('results/subgroup_analysis.csv', index=False)
print("\n✅ Subgroup analysis saved to results/subgroup_analysis.csv")
