#!/usr/bin/env python3
"""
Compute Matthews Correlation Coefficient (MCC) for all systems.
MCC provides a balanced measure for imbalanced classes.
"""
import json
import numpy as np
from pathlib import Path

def compute_mcc(tp, fp, fn, tn):
    """Compute Matthews Correlation Coefficient."""
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return numerator / denominator

# Real-world validation results
results = {
    "GPT-5": {"tp": 72, "fp": 54, "fn": 550, "tn": 324},
    "Claude Sonnet 4.5": {"tp": 87, "fp": 54, "fn": 535, "tn": 324},
    "Guardrails": {"tp": 444, "fp": 197, "fn": 178, "tn": 181},
    "Controller (CQL)": {"tp": 94, "fp": 0, "fn": 6, "tn": 100}
}

print("Matthews Correlation Coefficient (MCC) - Real-World Validation")
print("="*70)

for system, metrics in results.items():
    mcc = compute_mcc(metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'])
    sens = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    spec = metrics['tn'] / (metrics['tn'] + metrics['fp'])
    
    print(f"\n{system}:")
    print(f"  Sensitivity: {sens:.1%}")
    print(f"  Specificity: {spec:.1%}")
    print(f"  MCC: {mcc:.3f}")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")

# Save for manuscript
output = {system: compute_mcc(m['tp'], m['fp'], m['fn'], m['tn']) 
          for system, m in results.items()}

print(f"\n\nFor Appendix Table S4/S5:")
print(json.dumps(output, indent=2))
