"""
Generate per-message unified predictions for two-stage evaluation.

Inputs:
 - results/eval_outputs/real_world_cql.json          (RL controller actions)
 - results/eval_outputs/real_world_guardrail.json    (Guardrail hazards)
 - results/eval_outputs/real_world_constellation.json (optional, if exists)
 - results/predictions_with_demographics.csv         (ground-truth y_true for real-world)

Output:
 - results/predictions_unified.csv with columns:
   message_id, dataset, true_label,
   rl_prediction, rl_confidence, rl_action_idx,
   guardrail_prediction, guardrail_confidence,
   constellation_prediction, constellation_confidence

Physician test set is included only if matching ground truth can be located
(`data_final/physician_test.json` with `study_id` fields); otherwise only
real-world messages are emitted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "results" / "eval_outputs"
OUTPUT_CSV = ROOT / "results" / "predictions_unified.csv"

# Default mapping; update based on diagnose_cql_format.py output if needed
CQL_ACTION_TO_HAZARD: Dict[int, int] = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
}


def load_json(path: Path) -> List[dict]:
    with path.open() as f:
        return json.load(f)


def load_ground_truth_realworld() -> pd.DataFrame:
    gt_path = ROOT / "results" / "predictions_with_demographics.csv"
    df = pd.read_csv(gt_path)
    df = df.rename(columns={"study_id": "message_id", "y_true": "true_label"})
    df["dataset"] = "RealWorld_Test"
    return df[["message_id", "dataset", "true_label"]]


def load_ground_truth_physician() -> Optional[pd.DataFrame]:
    # Physician test with study_id is in data_final/physician_test.json
    path = ROOT / "data_final" / "physician_test.json"
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    rows = []
    for row in data:
        sid = row.get("study_id")
        if not sid:
            continue
        label = 0 if row.get("hazard_category", "benign") == "benign" else 1
        rows.append({"message_id": sid, "dataset": "Physician_Test", "true_label": label})
    return pd.DataFrame(rows)


def cql_to_pred(entry: dict) -> Dict[str, Optional[float]]:
    action_idx = entry.get("action_idx")
    action_probs = entry.get("action_probs") or []
    conf = max(action_probs) if action_probs else None
    pred = CQL_ACTION_TO_HAZARD.get(action_idx)
    return {"rl_action_idx": action_idx, "rl_prediction": pred, "rl_confidence": conf}


def guardrail_to_pred(entry: dict) -> Dict[str, Optional[float]]:
    return {
        "guardrail_prediction": entry.get("predicted_hazard"),
        "guardrail_confidence": entry.get("hazard_prob"),
    }


def constellation_to_pred(entry: dict) -> Dict[str, Optional[float]]:
    return {
        "constellation_prediction": entry.get("predicted_hazard"),
        "constellation_confidence": entry.get("hazard_prob") or entry.get("hazard_probabilities"),
    }


def merge_predictions(
    cql_path: Path,
    guardrail_path: Path,
    constellation_path: Optional[Path],
    ground_truth: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    truth_map = dict(zip(ground_truth["message_id"], ground_truth["true_label"]))
    cql = {p["scenario_id"]: p for p in load_json(cql_path)}
    grd = {p["scenario_id"]: p for p in load_json(guardrail_path)}
    con = {}
    if constellation_path and constellation_path.exists():
        con = {p["scenario_id"]: p for p in load_json(constellation_path)}

    rows = []
    for sid, c_entry in cql.items():
        if sid not in truth_map:
            continue
        row = {
            "message_id": sid,
            "dataset": dataset_name,
            "true_label": truth_map[sid],
        }
        row.update(cql_to_pred(c_entry))

        g_entry = grd.get(sid, {})
        row.update(guardrail_to_pred(g_entry))

        c_entry2 = con.get(sid, {})
        row.update(constellation_to_pred(c_entry2))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    real_gt = load_ground_truth_realworld()
    phys_gt = load_ground_truth_physician()

    real_df = merge_predictions(
        cql_path=EVAL_DIR / "real_world_cql.json",
        guardrail_path=EVAL_DIR / "real_world_guardrail.json",
        constellation_path=EVAL_DIR / "real_world_constellation.json",
        ground_truth=real_gt,
        dataset_name="RealWorld_Test",
    )

    dfs = [real_df]

    # Physician set if ground truth exists and files are present
    if phys_gt is not None and (EVAL_DIR / "physician_test_cql.json").exists():
        phys_df = merge_predictions(
            cql_path=EVAL_DIR / "physician_test_cql.json",
            guardrail_path=EVAL_DIR / "physician_test_guardrail.json",
            constellation_path=EVAL_DIR / "physician_test_constellation.json",
            ground_truth=phys_gt,
            dataset_name="Physician_Test",
        )
        dfs.append(phys_df)
    else:
        print("⚠️ Physician ground truth or prediction files not found; skipping physician set.")

    unified = pd.concat(dfs, ignore_index=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Saved unified predictions to {OUTPUT_CSV} ({len(unified)} rows)")
    print("Columns:", list(unified.columns))
    print(unified.groupby("dataset").size())


if __name__ == "__main__":
    main()
