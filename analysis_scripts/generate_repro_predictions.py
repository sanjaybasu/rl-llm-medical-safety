"""
Reproducible per-message predictions for guardrail (constellation) and CQL controller.

Data:
 - Real-world: data_outcome_labeled/realworld_n1000_outcome_labeled.json
 - Physician: data_final_outcome_splits/physician_test_clean_n200.json

Models:
 - Guardrail/Constellation: archive_cleanup/old_models/models_final/guardrail_constellation.pkl
 - CQL Controller: archive_cleanup/old_models/models_final/cql_controller.pth

Outputs (new subfolder results/repro_v1):
 - realworld_predictions.csv
 - physician_predictions.csv
 - metrics_summary.csv

All randomness is seeded for reproducibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parent

# Paths to held-out test splits
DATA_REAL = ROOT / "data_final_outcome_splits" / "realworld_test_n200_complete.json"
DATA_PHYS = ROOT / "data_final_outcome_splits" / "physician_test_clean_n200.json"

MODEL_DIR = ROOT / "archive_cleanup" / "old_models" / "models_final"
CONSTELLATION_PATH = MODEL_DIR / "guardrail_constellation.pkl"
CQL_PATH = MODEL_DIR / "cql_controller.pth"

OUTPUT_DIR = ROOT / "results" / "repro_testset"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class CQLNetwork(torch.nn.Module):
    """
    Matches the saved CQL checkpoint structure (fc1/fc2/fc3 layers).
    fc1: 384 -> 128
    fc2: 128 -> 64
    fc3: 64  -> 4
    """

    def __init__(self, state_dim: int = 384, action_dim: int = 4, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden1)
        self.fc2 = torch.nn.Linear(hidden1, hidden2)
        self.fc3 = torch.nn.Linear(hidden2, action_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_json(path: Path) -> List[Dict]:
    with path.open() as f:
        return json.load(f)


def true_label_from_phys(example: Dict) -> int:
    # Prefer detection_truth if present; otherwise hazard_category != benign
    if "detection_truth" in example:
        return int(example["detection_truth"] != 0)
    return 0 if example.get("hazard_category", "benign") == "benign" else 1


def true_label_from_real(example: Dict) -> int:
    # Use ground_truth_detection field (0/1)
    if "ground_truth_detection" in example:
        return int(example["ground_truth_detection"])
    return 0


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    try:
        auroc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auroc = np.nan
    return {"sensitivity": sens, "specificity": spec, "f1": f1, "mcc": mcc, "auroc": auroc}


def run_inference(dataset_name: str, data_path: Path, label_fn, id_field: str) -> pd.DataFrame:
    data = load_json(data_path)
    texts = [d.get("message") or d.get("prompt") for d in data]
    ids = [d.get(id_field, d.get("name", f"{dataset_name}_{i}")) for i, d in enumerate(data)]
    labels = np.array([label_fn(d) for d in data])

    # Embed
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeds = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    # Constellation/Guardrail
    constellation = joblib.load(CONSTELLATION_PATH)
    clf = constellation["classifier"]
    const_probs = clf.predict_proba(embeds)[:, 1]
    const_pred = (const_probs > 0.5).astype(int)

    # CQL Controller
    cql_net = CQLNetwork()
    cql_net.load_state_dict(torch.load(CQL_PATH, map_location="cpu"))
    cql_net.eval()
    with torch.no_grad():
        x = torch.tensor(embeds, dtype=torch.float32)
        q_vals = cql_net(x).numpy()
        cql_actions = np.argmax(q_vals, axis=1)
        cql_probs = softmax(q_vals, axis=1)
    cql_pred = (cql_actions > 0).astype(int)
    cql_conf = cql_probs.max(axis=1)

    df = pd.DataFrame(
        {
            "message_id": ids,
            "dataset": dataset_name,
            "true_label": labels,
            "constellation_prediction": const_pred,
            "constellation_confidence": const_probs,
            "rl_action_idx": cql_actions,
            "rl_prediction": cql_pred,
            "rl_confidence": cql_conf,
        }
    )
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    real_df = run_inference("RealWorld", DATA_REAL, true_label_from_real, "case_id")
    phys_df = run_inference("Physician", DATA_PHYS, true_label_from_phys, "name")

    real_df.to_csv(OUTPUT_DIR / "realworld_predictions.csv", index=False)
    phys_df.to_csv(OUTPUT_DIR / "physician_predictions.csv", index=False)

    # Two-stage RL->Constellation
    def two_stage(df: pd.DataFrame) -> np.ndarray:
        # Stage1: RL; if negative => 0; if positive => use constellation
        final = []
        for _, r in df.iterrows():
            if r["rl_prediction"] == 0:
                final.append(0)
            else:
                final.append(int(r["constellation_prediction"]))
        return np.array(final)

    results = []
    for name, df in [("RealWorld", real_df), ("Physician", phys_df)]:
        y = df["true_label"].to_numpy()
        res_const = compute_metrics(y, df["constellation_prediction"], df["constellation_confidence"])
        res_rl = compute_metrics(y, df["rl_prediction"], df["rl_confidence"])
        two_stage_pred = two_stage(df)
        res_two = compute_metrics(y, two_stage_pred, df["constellation_confidence"])
        for arch, res in [
            ("Constellation", res_const),
            ("RL_CQL", res_rl),
            ("TwoStage_RL_to_Constellation", res_two),
        ]:
            results.append(
                {
                    "dataset": name,
                    "architecture": arch,
                    **res,
                    "n_sample": len(df),
                }
            )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    print("Saved predictions and metrics to", OUTPUT_DIR)
    print(metrics_df)


if __name__ == "__main__":
    main()
