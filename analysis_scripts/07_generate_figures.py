"""
GENERATE MANUSCRIPT FIGURES
1. ROC Curves (Physician vs. Real-World)
2. Action Accuracy Comparison
3. Subgroup Analysis (Forest Plot)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

print("="*80)
print("GENERATING FIGURES")
print("="*80)

# Create figures dir
Path('results/figures').mkdir(parents=True, exist_ok=True)

# Load Data
phys_metrics = pd.read_csv('results/physician_metrics.csv')
rw_metrics = pd.read_csv('results/realworld_metrics.csv')
subgroup = pd.read_csv('results/subgroup_analysis.csv')

# Parse string metrics to floats
def parse_metric(val):
    if isinstance(val, str):
        return float(val.split(' ')[0])
    return val

phys_metrics['Action_Accuracy'] = phys_metrics['Action_Accuracy'].apply(parse_metric)
phys_metrics['Critical_Under_Triage'] = phys_metrics['Critical_Under_Triage'].apply(parse_metric)
rw_metrics['Action_Accuracy'] = rw_metrics['Action_Accuracy'].apply(parse_metric)
rw_metrics['Critical_Under_Triage'] = rw_metrics['Critical_Under_Triage'].apply(parse_metric)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# --- FIGURE 1: Sensitivity vs. Specificity (Scatter) ---
print("Generating Figure 1...")
plt.figure(figsize=(10, 6))

# Prepare data
phys_metrics['Dataset'] = 'Physician (N=200)'
rw_metrics['Dataset'] = 'Real-World (N=200)'
combined = pd.concat([phys_metrics, rw_metrics])

sns.scatterplot(data=combined, x='Specificity', y='Sensitivity', hue='Dataset', style='System', s=200, palette='deep')

plt.title('Sensitivity vs. Specificity: Physician vs. Real-World', fontsize=14)
plt.xlabel('Specificity', fontsize=12)
plt.ylabel('Sensitivity', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('results/figures/figure1_sensitivity_specificity.png')
plt.close()

# --- FIGURE 2: Action Accuracy & Under-Triage ---
print("Generating Figure 2...")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Action Accuracy
sns.barplot(data=combined, x='System', y='Action_Accuracy', hue='Dataset', ax=ax[0], palette='Blues')
ax[0].set_title('Action Accuracy (Correct Recommendation)', fontsize=14)
ax[0].set_ylim(0, 1)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45)

# Critical Under-Triage
sns.barplot(data=combined, x='System', y='Critical_Under_Triage', hue='Dataset', ax=ax[1], palette='Reds')
ax[1].set_title('Critical Under-Triage Rate (Failure to Escalate)', fontsize=14)
ax[1].set_ylim(0, 1)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('results/figures/figure2_action_quality.png')
plt.close()

# --- FIGURE 3: Subgroup Analysis (Race) ---
print("Generating Figure 3...")
race_data = subgroup[subgroup['Group_Type'] == 'Race']

plt.figure(figsize=(12, 6))
sns.barplot(data=race_data, x='Group', y='Sensitivity', hue='System', palette='viridis')
plt.title('Sensitivity by Racial Group (Real-World)', fontsize=14)
plt.ylabel('Sensitivity')
plt.xlabel('Race')
plt.xticks(rotation=45)
plt.legend(title='System')
plt.tight_layout()
plt.savefig('results/figures/figure3_subgroup_race.png')
plt.close()

print("✅ Figures generated in results/figures/")
