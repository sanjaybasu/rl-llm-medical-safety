#!/usr/bin/env python3
"""
Evaluate AWR controller on the real-world replay validation set with hazard mapping.
Mirrors the CQL evaluation pipeline for reproducibility.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split

from evaluate_llm_safety import train_hazard_detector, get_hazard_detector
from rl_core import (
    ACTIONS,
    SAFE_ACTIONS,
    AWRAgent,
    build_dataset,
    summarize_scenario_metrics,
)

random.seed(42)
np.random.seed(42)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT_DIR / "data"

# Map real-world labels to physician taxonomy for action lookup
HAZARD_MAPPING: Dict[str, str] = {
    "emergency_symptom": "cardiac_emergency",
    "mental_health_crisis": "suicide_risk",
    "medication_safety": "drug_interaction",
    "social_needs_risk": "benign",
    "care_coordination": "med_reconciliation",
    "safety_threat": "misuse_escalation",
    "general_risk": "benign",
    "benign": "benign",
    "unknown": "unknown",
}


def load_replay_scenarios():
    with open(DATA_DIR / "replay_scenarios.json", "r") as f:
        return json.load(f)


def map_hazard(label: str) -> str:
    return HAZARD_MAPPING.get(label, "unknown")


def main():
    print("=" * 80)
    print("AWR CONTROLLER: REAL-WORLD VALIDATION")
    print("=" * 80)

    # Load and split data
    all_scenarios = load_replay_scenarios()
    train_scenarios, val_scenarios = train_test_split(
        all_scenarios,
        test_size=0.20,
        random_state=42,
        stratify=[s.get("hazard_type", "benign") for s in all_scenarios],
    )

    # Train detector on training split
    train_hazard_detector(train_scenarios)
    detector = get_hazard_detector()

    # Build offline dataset using physician taxonomy hazards
    hazard_ids = sorted(set(HAZARD_MAPPING.values()))
    dataset = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, repeats=10)
    awr = AWRAgent(hazard_ids, len(ACTIONS), alpha=0.1, beta=0.05)
    awr.train(dataset, epochs=500)

    # Evaluate with mapping
    hazard_rows = []
    benign_rows = []
    for sc in val_scenarios:
        prompt = sc.get("prompt", "")
        context = sc.get("context", {})
        true_hazard = sc.get("hazard_type", "unknown")
        det = detector.predict(prompt, context)
        pred = det.label if det and det.label else "unknown"
        mapped_pred = map_hazard(pred)
        mapped_true = map_hazard(true_hazard)
        action = awr.act(mapped_pred)
        safe = SAFE_ACTIONS.get(mapped_true, SAFE_ACTIONS.get("unknown", []))
        is_safe = action in safe
        row = {
            "hazard_true": true_hazard,
            "hazard_pred": pred,
            "mapped_true": mapped_true,
            "mapped_pred": mapped_pred,
            "chosen_action": action,
            "is_safe": is_safe,
        }
        if mapped_true == "benign":
            benign_rows.append(row)
        else:
            hazard_rows.append(row)

    metrics = summarize_scenario_metrics(hazard_rows, benign_rows)
    out = {
        "method": "AWR controller on real-world replay",
        "tp": metrics["tp"],
        "fn": metrics["fn"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "npv": metrics["npv"],
    }

    out_path = RESULTS_DIR / "awr_realworld_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
