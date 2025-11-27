#!/usr/bin/env python3
"""
Train an Advantage-Weighted controller (AWR) for the NEJM AI manuscript.
Uses the same offline dataset construction as the CQL calibrations.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np

from evaluate_llm_safety import build_core_scenarios, load_benign_scenarios, train_hazard_detector, get_hazard_detector
from rl_core import (
    ACTIONS,
    SAFE_ACTIONS,
    AWRAgent,
    build_dataset,
    evaluate_agent,
    summarize_scenario_metrics,
)

random.seed(42)
np.random.seed(42)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    scenarios = build_core_scenarios()
    benign_scenarios = load_benign_scenarios()
    train_hazard_detector(scenarios + benign_scenarios)
    detector = get_hazard_detector()

    hazard_ids = sorted({sc.hazard_type for sc in scenarios})
    if "benign" not in hazard_ids:
        hazard_ids.append("benign")
    if "unknown" not in hazard_ids:
        hazard_ids.append("unknown")

    # Balanced AWR controller
    dataset = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, repeats=10)
    awr = AWRAgent(hazard_ids, len(ACTIONS), alpha=0.1, beta=0.05)
    awr.train(dataset, epochs=500)

    rows_hazard = evaluate_agent(awr.state_to_idx, awr.act, scenarios, detector=detector)
    rows_benign = evaluate_agent(awr.state_to_idx, awr.act, benign_scenarios, detector=detector)
    metrics = summarize_scenario_metrics(rows_hazard, rows_benign)

    out_path = RESULTS_DIR / "awr_controller_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "hazard_ids": hazard_ids,
                "metrics": metrics,
                "rows_hazard": rows_hazard,
                "rows_benign": rows_benign,
            },
            f,
            indent=2,
        )

    print("\nAWR controller training complete")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"Precision:   {metrics['precision']:.3f}")
    print(f"NPV:         {metrics['npv']:.3f}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
