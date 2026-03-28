#!/usr/bin/env python3
"""
finalize_deepseek_full2000.py
-----------------------------
Compute metrics from a completed predictions_deepseek_full2000.csv and print
a summary. This script is provided for reference; the primary execution is
managed by the watcher in the original clean_replication environment.

Usage (from repository root):
    python3 code/analysis/finalize_deepseek_full2000.py

Prerequisites:
    results/predictions/predictions_deepseek_full2000.csv   (2000 rows)
    results/predictions/predictions_long_new.csv
    results/predictions/predictions_actions_local_with_truth.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
PRED_FILE  = REPO_ROOT / 'results/predictions/predictions_deepseek_full2000.csv'
GT_HAZARD  = REPO_ROOT / 'results/predictions/predictions_long_new.csv'
GT_ACTION  = REPO_ROOT / 'results/predictions/predictions_actions_local_with_truth.csv'

N_BOOT   = 10_000
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

ds = pd.read_csv(PRED_FILE)
if len(ds) < 2000:
    print(f"ERROR: Only {len(ds)}/2000 predictions. Run is incomplete.")
    sys.exit(1)

hazard_gt = (pd.read_csv(GT_HAZARD)
             .query("system == 'Guardrail_new'")[['message_id', 'true_label']]
             .drop_duplicates('message_id'))
action_gt = (pd.read_csv(GT_ACTION)
             .query("system == 'Guardrail_new'")[['message_id', 'clinician_action_mapped']]
             .drop_duplicates('message_id'))

merged_h = ds.merge(hazard_gt, on='message_id', how='inner')
merged_a = ds.merge(action_gt, on='message_id', how='inner')

y_true = merged_h['true_label'].values
y_pred = merged_h['hazard_pred'].values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

boot_sens, boot_spec = [], []
n = len(merged_h)
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yt, yp = y_true[idx], y_pred[idx]
    _tn, _fp, _fn, _tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    boot_sens.append(_tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0)
    boot_spec.append(_tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0)

sens_lo, sens_hi = np.percentile(boot_sens, [2.5, 97.5])
spec_lo, spec_hi = np.percentile(boot_spec, [2.5, 97.5])

action_pred = merged_a['action_pred'].values
clin_action = merged_a['clinician_action_mapped'].values
appropriate = (action_pred == clin_action)
under_rate  = (action_pred < clin_action).mean()
over_rate   = (action_pred > clin_action).mean()
action_acc  = appropriate.mean()

boot_acc = []
na = len(merged_a)
for _ in range(N_BOOT):
    idx = rng.integers(0, na, na)
    boot_acc.append(appropriate[idx].mean())
acc_lo, acc_hi = np.percentile(boot_acc, [2.5, 97.5])

print("DeepSeek-R1 — full 2,000-message test set")
print(f"  N={len(merged_h)}, hazards={int(tp+fn)}")
print(f"  Sensitivity {sensitivity:.3f} (95% CI {sens_lo:.3f}–{sens_hi:.3f})")
print(f"  Specificity {specificity:.3f} (95% CI {spec_lo:.3f}–{spec_hi:.3f})")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"  FN/1000 messages: {fn/len(merged_h)*1000:.1f}")
print(f"  Action accuracy {action_acc:.3f} (95% CI {acc_lo:.3f}–{acc_hi:.3f})")
print(f"  Under-triage {under_rate:.3f}, Over-triage {over_rate:.3f}")
