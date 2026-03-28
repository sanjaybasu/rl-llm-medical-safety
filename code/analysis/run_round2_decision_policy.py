"""
Round 2 decision policy evaluation (fast, heuristic).

Trains a guardrail detector on the round2 annotated training set, tunes a
detection threshold on the round2 validation split (spec >= 0.85, maximize
sensitivity), then applies a simple decision policy:
    if detected hazard -> action_id = 2 (contact doctor)
    else                -> action_id = 0 (self-care/none)

Outputs detection metrics to:
    results/repro_round2/architecture_eval_metrics_decisionpolicy.csv

Notes:
    - This is a lightweight placeholder (no deep RL); suitable for quick
      reproducibility. For full RL, extend this script.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from math import sqrt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - rad) / denom), min(1.0, (center + rad) / denom))


def compute_detection_metrics(y_true, probs, tau):
    preds = (probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, preds)
    mcc = matthews_corrcoef(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    return {
        "sensitivity": sens,
        "sensitivity_ci_lower": sens_ci[0],
        "sensitivity_ci_upper": sens_ci[1],
        "specificity": spec,
        "specificity_ci_lower": spec_ci[0],
        "specificity_ci_upper": spec_ci[1],
        "f1": f1,
        "mcc": mcc,
        "auroc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    base = Path("results/labeling_round2")
    train_rw = pd.read_csv(base / "realworld_labelset_train_candidates_annotated.csv")
    test_rw = pd.read_csv(base / "realworld_labelset_test_candidates_annotated.csv")
    phys_train = pd.read_json("data_final_outcome_splits/physician_train_clean.json")
    phys_new = pd.read_csv(base / "physician_new_scenarios.csv")

    phys_train = phys_train.rename(
        columns={"message": "text", "detection_truth": "hazard_detection"}
    )
    phys_new = phys_new.rename(columns={"message_text": "text"})
    phys_train["hazard_category"] = phys_train["hazard_category"].fillna("benign")
    phys_new["hazard_category"] = phys_new["hazard_category"].fillna("benign")
    phys_new["hazard_detection"] = (
        phys_new["hazard_category"] != "benign"
    ).astype(int)
    train_rw = train_rw.rename(columns={"message": "text"})

    train_df = pd.concat(
        [
            train_rw[train_rw["is_val"] != 1][["text", "hazard_detection"]],
            phys_train[["text", "hazard_detection"]],
            phys_new[["text", "hazard_detection"]],
        ],
        ignore_index=True,
    )
    val_df = train_rw[train_rw["is_val"] == 1][["text", "hazard_detection"]]
    test_df = test_rw.rename(columns={"message": "text"})

    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = enc.encode(train_df["text"].tolist(), show_progress_bar=False)
    X_val = enc.encode(val_df["text"].tolist(), show_progress_bar=False)
    X_test = enc.encode(test_df["text"].tolist(), show_progress_bar=False)

    y_train = train_df["hazard_detection"].values
    y_val = val_df["hazard_detection"].values
    y_test = test_df["hazard_detection"].values

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    best = None
    for tau in np.linspace(0.05, 0.8, 31):
        preds = (val_probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        if spec >= 0.85 and (best is None or sens > best[0]):
            best = (sens, spec, tau)
    tau = best[2] if best else 0.5

    test_probs = clf.predict_proba(X_test)[:, 1]
    det_metrics = compute_detection_metrics(y_test, test_probs, tau)

    # Simple action policy: hazard -> action 2 (contact doctor), else action 0
    pred_det = (test_probs >= tau).astype(int)
    pred_action = np.where(pred_det == 1, 2, 0)
    true_action = test_df.get("clinician_action_mapped", pd.Series(np.zeros(len(test_df)))).fillna(0).astype(int).values

    action_mask = test_df["hazard_detection"] == 1  # evaluate action on hazards
    action_acc = (pred_action[action_mask] == true_action[action_mask]).mean() if action_mask.sum() > 0 else 0.0

    out_dir = Path("results/repro_round2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "architecture_eval_metrics_decisionpolicy.csv"
    row = {
        "dataset": "RealWorld_Test",
        "system": "DecisionPolicy_Simple",
        "sensitivity": det_metrics["sensitivity"],
        "sensitivity_ci_lower": det_metrics["sensitivity_ci_lower"],
        "sensitivity_ci_upper": det_metrics["sensitivity_ci_upper"],
        "specificity": det_metrics["specificity"],
        "specificity_ci_lower": det_metrics["specificity_ci_lower"],
        "specificity_ci_upper": det_metrics["specificity_ci_upper"],
        "f1": det_metrics["f1"],
        "mcc": det_metrics["mcc"],
        "auroc": det_metrics["auroc"],
        "action_accuracy_on_hazards": action_acc,
        "n_sample": len(test_df),
        "n_hazard": int(test_df["hazard_detection"].sum()),
        "n_safe": int((1 - test_df["hazard_detection"]).sum()),
    }
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"✅ Saved decision policy metrics to {out_path}")


if __name__ == "__main__":
    main()
