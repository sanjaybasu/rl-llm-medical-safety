#!/usr/bin/env python3
"""
Train a high-sensitivity CQL controller for NEJM AI manuscript.
Prioritizes hazard detection (high sensitivity) over false alarm avoidance.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(CURRENT_DIR))

from evaluate_llm_safety import build_core_scenarios, load_benign_scenarios, train_hazard_detector, get_hazard_detector
from run_rl_vs_llm_demo import (
    ACTIONS, SAFE_ACTIONS, CQLAgent, build_dataset, HazardEnv,
    make_selector_from_q, evaluate_agent, summarize_scenario_metrics
)

random.seed(42)
np.random.seed(42)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    scenarios = build_core_scenarios()
    benign_scenarios = load_benign_scenarios()
    train_hazard_detector(scenarios + benign_scenarios)
    
    hazard_ids = sorted({sc.hazard_type for sc in scenarios})
    if "benign" not in hazard_ids:
        hazard_ids.append("benign")
    if "unknown" not in hazard_ids:
        hazard_ids.append("unknown")
    SAFE_ACTIONS.setdefault("unknown", ["reassure"])
    SAFE_ACTIONS.setdefault("benign", ["reassure"])
    
    detector = get_hazard_detector()
    
    # Configuration 1: High Specificity (conservative, avoid false alarms)
    # Penalty for unnecessary escalation on non-emergencies
    emergency_hazards = {
        "suicide_risk", "metabolic_emergency", "neuro_emergency",
        "anaphylaxis", "obstetric_emergency", "cardiac_emergency",
        "pediatric_emergency", "misuse_escalation"
    }
    
    penalty_conservative: dict[str, dict[str, float]] = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards:
            penalty_conservative.setdefault(hazard, {})["activate_emergency"] = -0.5
            penalty_conservative.setdefault(hazard, {})["warn_and_escalate"] = -0.2
    
    dataset_conservative = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_conservative)
    cql_conservative = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_conservative.train(dataset_conservative, epochs=400)
    
    rows_conservative = evaluate_agent(
        cql_conservative.state_to_idx,
        make_selector_from_q(cql_conservative.state_to_idx, cql_conservative.Q),
        scenarios,
        detector=detector
    )
    rows_conservative_benign = evaluate_agent(
        cql_conservative.state_to_idx,
        make_selector_from_q(cql_conservative.state_to_idx, cql_conservative.Q),
        benign_scenarios,
        detector=detector
    )
    metrics_conservative = summarize_scenario_metrics(rows_conservative, rows_conservative_benign)
    
    # Configuration 2: Balanced
    penalty_balanced: dict[str, dict[str, float]] = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards:
            penalty_balanced.setdefault(hazard, {})["activate_emergency"] = -0.25
            penalty_balanced.setdefault(hazard, {})["warn_and_escalate"] = -0.1
    
    dataset_balanced = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_balanced)
    cql_balanced = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_balanced.train(dataset_balanced, epochs=400)
    
    rows_balanced = evaluate_agent(
        cql_balanced.state_to_idx,
        make_selector_from_q(cql_balanced.state_to_idx, cql_balanced.Q),
        scenarios,
        detector=detector
    )
    rows_balanced_benign = evaluate_agent(
        cql_balanced.state_to_idx,
        make_selector_from_q(cql_balanced.state_to_idx, cql_balanced.Q),
        benign_scenarios,
        detector=detector
    )
    metrics_balanced = summarize_scenario_metrics(rows_balanced, rows_balanced_benign)
    
    # Configuration 3: High Sensitivity (prioritize hazard detection)
    # Minimal penalty for escalation, boost rewards for correct hazard actions
    penalty_sensitive: dict[str, dict[str, float]] = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards:
            # Very small penalty for unnecessary escalation
            penalty_sensitive.setdefault(hazard, {})["activate_emergency"] = -0.05
            penalty_sensitive.setdefault(hazard, {})["warn_and_escalate"] = -0.02
    
    dataset_sensitive = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_sensitive, repeats=15)
    cql_sensitive = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_sensitive.train(dataset_sensitive, epochs=600)  # More training epochs
    
    rows_sensitive = evaluate_agent(
        cql_sensitive.state_to_idx,
        make_selector_from_q(cql_sensitive.state_to_idx, cql_sensitive.Q),
        scenarios,
        detector=detector
    )
    rows_sensitive_benign = evaluate_agent(
        cql_sensitive.state_to_idx,
        make_selector_from_q(cql_sensitive.state_to_idx, cql_sensitive.Q),
        benign_scenarios,
        detector=detector
    )
    metrics_sensitive = summarize_scenario_metrics(rows_sensitive, rows_sensitive_benign)
    
    report = {
        "configurations": [
            {
                "name": "High Specificity (Conservative)",
                "description": "Penalizes false alarms heavily",
                "penalty_weights": "activate_emergency: -0.5, warn_and_escalate: -0.2",
                "metrics": metrics_conservative,
                "hazard_rows": rows_conservative,
                "benign_rows": rows_conservative_benign
            },
            {
                "name": "Balanced",
                "description": "Moderate trade-off between sensitivity and specificity",
                "penalty_weights": "activate_emergency: -0.25, warn_and_escalate: -0.1",
                "metrics": metrics_balanced,
                "hazard_rows": rows_balanced,
                "benign_rows": rows_balanced_benign
            },
            {
                "name": "High Sensitivity",
                "description": "Prioritizes hazard detection over false alarm avoidance",
                "penalty_weights": "activate_emergency: -0.05, warn_and_escalate: -0.02",
                "metrics": metrics_sensitive,
                "hazard_rows": rows_sensitive,
                "benign_rows": rows_sensitive_benign
            }
        ]
    }
    
    out = RESULTS_DIR / "cql_sensitivity_calibrations.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("CQL CONTROLLER SENSITIVITY CALIBRATION RESULTS")
    print("="*80)
    for config in report["configurations"]:
        m = config["metrics"]
        print(f"\n{config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Penalty weights: {config['penalty_weights']}")
        print(f"  Sensitivity: {m['sensitivity']:.3f}")
        print(f"  Specificity: {m['specificity']:.3f}")
        print(f"  Precision: {m['precision']:.3f}")
        tp, fp, fn, tn = m['tp'], m['fp'], m['fn'], m['tn']
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
        print(f"  F1 Score: {f1:.3f}")
        print(f"  True Positives: {tp}, False Negatives: {fn}")
        print(f"  True Negatives: {tn}, False Positives: {fp}")
    
    print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
