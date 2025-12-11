"""
Round 2 local-only evaluation script.

This script trains/evaluates the local baselines on the latest annotated splits:
  - realworld_labelset_train_candidates_annotated.csv  (train + val flag)
  - realworld_labelset_test_candidates_annotated.csv   (test)
  - physician_train_clean.json                         (legacy physician train)
  - physician_new_scenarios.csv                        (new physician scenarios)

Models:
  1) Guardrail: SBERT embeddings + logistic regression (binary)
  2) Constellation_Multilog: SBERT + multinomial logistic (benign vs categories)
  3) LogReg_TFIDF: 1-2 gram TF-IDF + logistic regression (binary)
  4) XGBoost_SBERT: SBERT embeddings + XGBoost (binary)

Outputs:
  results/repro_round2/architecture_eval_metrics_VERIFIED_combined.csv
  (one row per system for the 2,000-message real-world test)

Notes:
  - Decision-policy / GPT are NOT run here; this is a fast local-only runner.
  - Validation split uses the is_val flag in the real-world train file.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from math import sqrt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from xgboost import XGBClassifier


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = phat + z * z / (2 * n)
    rad = z * sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return (max(0.0, (center - rad) / denom), min(1.0, (center + rad) / denom))


def compute_metrics(y_true, probs, tau):
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


def load_round2_sets():
    base = Path("results/labeling_round2")
    train_rw = pd.read_csv(base / "realworld_labelset_train_candidates_annotated.csv")
    test_rw = pd.read_csv(base / "realworld_labelset_test_candidates_annotated.csv")
    phys_train = pd.read_json(
        Path("data_final_outcome_splits/physician_train_clean.json")
    )
    phys_new = pd.read_csv(base / "physician_new_scenarios.csv")

    # Harmonize columns
    phys_train = phys_train.rename(
        columns={"message": "text", "detection_truth": "hazard_detection"}
    )
    phys_train["hazard_category"] = phys_train["hazard_category"].fillna("benign")

    phys_new = phys_new.rename(columns={"message_text": "text"})
    phys_new["hazard_category"] = phys_new["hazard_category"].fillna("benign")
    phys_new["hazard_detection"] = (
        phys_new["hazard_category"] != "benign"
    ).astype(int)

    train_rw = train_rw.rename(columns={"message": "text"})
    train_rw["hazard_category"] = train_rw["hazard_category"].fillna("benign")

    # Combine train (excluding val rows for training proper)
    train_all = pd.concat(
        [
            train_rw[train_rw["is_val"] != 1][["text", "hazard_detection", "hazard_category"]],
            phys_train[["text", "hazard_detection", "hazard_category"]],
            phys_new[["text", "hazard_detection", "hazard_category"]],
        ],
        ignore_index=True,
    )
    val_df = train_rw[train_rw["is_val"] == 1][["text", "hazard_detection", "hazard_category"]]
    test_df = test_rw.rename(columns={"message": "text"})
    return train_all, val_df, test_df


def train_guardrail(train_df, val_df, test_df):
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
    return compute_metrics(y_test, test_probs, tau)


def train_constellation(train_df, val_df, test_df):
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = enc.encode(train_df["text"].tolist(), show_progress_bar=False)
    X_val = enc.encode(val_df["text"].tolist(), show_progress_bar=False)
    X_test = enc.encode(test_df["text"].tolist(), show_progress_bar=False)

    cat_train = train_df["hazard_category"].fillna("benign")
    cat_val = val_df["hazard_category"].fillna("benign")
    cat_test = test_df["hazard_category"].fillna("benign")
    classes = sorted(cat_train.unique())
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_train = cat_train.map(class_to_id)
    y_val = cat_val.map(class_to_id)

    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", multi_class="multinomial", solver="lbfgs"
    )
    clf.fit(X_train, y_train)

    benign_id = class_to_id["benign"]
    val_probs = clf.predict_proba(X_val)
    val_det_probs = 1 - val_probs[:, benign_id]
    val_true = (cat_val != "benign").astype(int).values

    best = None
    for tau in np.linspace(0.05, 0.8, 31):
        preds = (val_det_probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_true, preds).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        score = sens + spec  # Youden
        if best is None or score > best[0]:
            best = (score, sens, spec, tau)
    tau = best[3] if best else 0.5

    test_probs = clf.predict_proba(X_test)
    test_det_probs = 1 - test_probs[:, benign_id]
    test_true = (cat_test != "benign").astype(int).values
    return compute_metrics(test_true, test_det_probs, tau)


def train_tfidf_lr(train_df, val_df, test_df):
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    X_train = vec.fit_transform(train_df["text"])
    X_val = vec.transform(val_df["text"])
    X_test = vec.transform(test_df["text"])
    y_train = train_df["hazard_detection"].values
    y_val = val_df["hazard_detection"].values
    y_test = test_df["hazard_detection"].values

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]

    best = None
    for tau in np.linspace(0.05, 0.7, 27):
        preds = (val_probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        if spec >= 0.85 and (best is None or sens > best[0]):
            best = (sens, spec, tau)
    tau = best[2] if best else 0.5
    test_probs = clf.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, test_probs, tau)


def train_xgb(train_df, val_df, test_df):
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = enc.encode(train_df["text"].tolist(), show_progress_bar=False)
    X_val = enc.encode(val_df["text"].tolist(), show_progress_bar=False)
    X_test = enc.encode(test_df["text"].tolist(), show_progress_bar=False)
    y_train = train_df["hazard_detection"].values
    y_val = val_df["hazard_detection"].values
    y_test = test_df["hazard_detection"].values

    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=42,
    )
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    best = None
    for tau in np.linspace(0.05, 0.7, 27):
        preds = (val_probs >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        if spec >= 0.85 and (best is None or sens > best[0]):
            best = (sens, spec, tau)
    tau = best[2] if best else 0.5
    test_probs = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, test_probs, tau)


def main():
    out_dir = Path("results/repro_round2")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_round2_sets()
    rows = []
    # Guardrail
    guard_metrics = train_guardrail(train_df, val_df, test_df)
    rows.append(
        {
            "dataset": "RealWorld_Test",
            "system": "Guardrail",
            **{k: v for k, v in guard_metrics.items() if k not in ["tp", "tn", "fp", "fn"]},
            "n_sample": len(test_df),
            "n_hazard": int(test_df["hazard_detection"].sum()),
            "n_safe": int((1 - test_df["hazard_detection"]).sum()),
        }
    )
    # Constellation
    const_metrics = train_constellation(train_df, val_df, test_df)
    rows.append(
        {
            "dataset": "RealWorld_Test",
            "system": "Constellation_Multilog",
            **{k: v for k, v in const_metrics.items() if k not in ["tp", "tn", "fp", "fn"]},
            "n_sample": len(test_df),
            "n_hazard": int(test_df["hazard_detection"].sum()),
            "n_safe": int((1 - test_df["hazard_detection"]).sum()),
        }
    )
    # TFIDF
    tfidf_metrics = train_tfidf_lr(train_df, val_df, test_df)
    rows.append(
        {
            "dataset": "RealWorld_Test",
            "system": "LogReg_TFIDF",
            **{k: v for k, v in tfidf_metrics.items() if k not in ["tp", "tn", "fp", "fn"]},
            "n_sample": len(test_df),
            "n_hazard": int(test_df["hazard_detection"].sum()),
            "n_safe": int((1 - test_df["hazard_detection"]).sum()),
        }
    )
    # XGB
    xgb_metrics = train_xgb(train_df, val_df, test_df)
    rows.append(
        {
            "dataset": "RealWorld_Test",
            "system": "XGBoost_SBERT",
            **{k: v for k, v in xgb_metrics.items() if k not in ["tp", "tn", "fp", "fn"]},
            "n_sample": len(test_df),
            "n_hazard": int(test_df["hazard_detection"].sum()),
            "n_safe": int((1 - test_df["hazard_detection"]).sum()),
        }
    )

    out_path = out_dir / "architecture_eval_metrics_VERIFIED_combined.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"✅ Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
