"""
Compile action appropriateness metrics for all 9 primary configurations.

Reads per-message prediction files and computes action accuracy, under-triage
rate, over-triage rate, and bootstrap 95% CIs.

For systems with direct action predictions (local classifiers, GPT, ActionHead):
  - action_accuracy = exact match with clinician_action_mapped
  - under_rate = fraction where predicted action < clinician_action (under-triage)
  - over_rate = fraction where predicted action > clinician_action (over-triage)

For systems with only hazard-level predictions (TinyLlama, CQL-reward):
  - Binary mapping applied: hazard detected → action 4 (urgent 24-48h),
    no hazard → action 1 (self-care/no action)
  - Threshold chosen to match reported sensitivity (TinyLlama tau=0.080,
    CQL-reward tau=0.100)

Output: results/repro_round2/action_metrics_all_final.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

RESULTS = Path("../results/repro_round2")
N_BOOT = 10000
RNG = np.random.default_rng(42)


def bootstrap_ci(match_array, n_boot=N_BOOT):
    """Bootstrap 95% CI for mean of a binary array."""
    boots = [RNG.choice(match_array, size=len(match_array), replace=True).mean()
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def compute_action_metrics(action_pred, action_true, system_name, n, note=""):
    match = (action_pred == action_true)
    acc = match.mean()
    under = (action_pred < action_true).mean()
    over = (action_pred > action_true).mean()
    ci_lo, ci_hi = bootstrap_ci(match.astype(float))
    return {
        "system": system_name,
        "n": n,
        "action_accuracy": round(acc, 4),
        "under_rate": round(under, 4),
        "over_rate": round(over, 4),
        "action_acc_ci_lower": round(ci_lo, 4),
        "action_acc_ci_upper": round(ci_hi, 4),
        "note": note,
    }


def binary_hazard_to_action(hazard_pred):
    """Binary hazard-to-action mapping: hazard→4 (urgent 24-48h), no hazard→1."""
    return np.where(hazard_pred == 1, 4, 1)


rows = []

# ── Local classifiers and GPT (direct action predictions) ──────────────────
local_preds = pd.read_csv(RESULTS / "predictions_actions_local_with_truth.csv")
for sys in local_preds["system"].unique():
    df = local_preds[local_preds["system"] == sys].dropna(
        subset=["action_pred", "clinician_action_mapped"]
    )
    rows.append(compute_action_metrics(
        df["action_pred"].astype(int).values,
        df["clinician_action_mapped"].astype(int).values,
        sys, len(df), note="direct prediction"
    ))

llm_preds = pd.read_csv(RESULTS / "predictions_actions_llm_with_truth.csv")
for sys in llm_preds["system"].unique():
    df = llm_preds[llm_preds["system"] == sys].dropna(
        subset=["action_pred", "clinician_action_mapped"]
    )
    rows.append(compute_action_metrics(
        df["action_pred"].astype(int).values,
        df["clinician_action_mapped"].astype(int).values,
        sys, len(df), note="direct prediction"
    ))

# ActionHead
if (RESULTS / "predictions_actions_actionhead_sbert.csv").exists():
    ah = pd.read_csv(RESULTS / "predictions_actions_actionhead_sbert.csv").dropna(
        subset=["action_pred", "clinician_action_mapped"]
    )
    rows.append(compute_action_metrics(
        ah["action_pred"].astype(int).values,
        ah["clinician_action_mapped"].astype(int).values,
        "ActionHead_SBERT_LR", len(ah), note="direct 9-class prediction"
    ))

# ── Systems with only hazard predictions (binary mapping) ──────────────────
long_preds = pd.read_csv(RESULTS / "predictions_long_new.csv")
# Ground truth actions from local predictions file
gt = pd.read_csv(RESULTS / "predictions_actions_local_with_truth.csv")
gt = gt[gt["system"] == "Guardrail_new"][["message_id", "clinician_action_mapped"]]

for sys_name, target_sens in [("TinyLlama_MPS", 0.376), ("CQL_Controller_reward", 0.642)]:
    df = long_preds[long_preds["system"] == sys_name].copy()
    df = df.merge(gt, on="message_id")
    df = df.dropna(subset=["probability", "true_label", "clinician_action_mapped"])

    # Find threshold matching target sensitivity (±0.01 tolerance)
    best_tau, best_diff = 0.5, 1.0
    for tau in np.linspace(0.05, 0.95, 181):
        preds_t = (df["probability"] >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            df["true_label"], preds_t, labels=[0, 1]
        ).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        diff = abs(sens - target_sens)
        if diff < best_diff:
            best_diff = diff
            best_tau = tau

    hazard_pred = (df["probability"] >= best_tau).astype(int)
    action_pred = binary_hazard_to_action(hazard_pred)
    note = (
        f"binary hazard-to-action mapping (hazard→action4; no hazard→action1); "
        f"tau={best_tau:.3f}"
    )
    rows.append(compute_action_metrics(
        action_pred,
        df["clinician_action_mapped"].astype(int).values,
        sys_name, len(df), note=note
    ))

# TinyLlama via dedicated predictions file (if available)
if (RESULTS / "tinyllama_predictions.csv").exists():
    tl = pd.read_csv(RESULTS / "tinyllama_predictions.csv")
    # This file already has binary hazard predictions; check if action columns exist
    if "action_pred" in tl.columns and "clinician_action_mapped" in tl.columns:
        tl = tl.dropna(subset=["action_pred", "clinician_action_mapped"])
        rows.append(compute_action_metrics(
            tl["action_pred"].astype(int).values,
            tl["clinician_action_mapped"].astype(int).values,
            "TinyLlama_MPS_direct", len(tl), note="direct prediction from tinyllama"
        ))

# ── Save ───────────────────────────────────────────────────────────────────
out = pd.DataFrame(rows)
out_path = RESULTS / "action_metrics_all_final.csv"
out.to_csv(out_path, index=False)
print(f"Saved {len(out)} rows → {out_path}")
print(out[["system", "n", "action_accuracy", "under_rate", "over_rate"]].to_string(index=False))
