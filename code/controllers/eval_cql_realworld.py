#!/usr/bin/env python3
"""
Evaluate CQL controller calibrations on REAL-WORLD validation set.
This script loads the gitignored replay data programmatically and generates
actual performance metrics for the NEJM AI manuscript.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parents[1]  # repo root
if str(CURRENT_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(CURRENT_DIR))

from evaluate_llm_safety import train_hazard_detector, get_hazard_detector
from rl_core import (
    ACTIONS,
    SAFE_ACTIONS,
    CQLAgent,
    build_dataset,
    make_selector_from_q,
    evaluate_agent,
    summarize_scenario_metrics,
)

random.seed(42)
np.random.seed(42)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT_DIR / "data"


def load_replay_scenarios():
    """Load real-world validation scenarios from replay data."""
    replay_path = DATA_DIR / "replay_scenarios.json"
    with open(replay_path, 'r') as f:
        data = json.load(f)
    return data


def load_labeled_scenarios():
    """Load physician-labeled scenarios for training."""
    train_path = DATA_DIR / "hazard_scenarios_train.json"
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    benign_path = DATA_DIR / "benign_scenarios_curated.json"
    with open(benign_path, 'r') as f:
        benign_data = json.load(f)
    
    return train_data + benign_data


def main() -> None:
    print("Loading data...")
    train_scenarios = load_labeled_scenarios()
    real_world_scenarios = load_replay_scenarios()
    
    print(f"Training set: {len(train_scenarios)} scenarios")
    print(f"Real-world validation set: {len(real_world_scenarios)} scenarios")
    
    # Train detector on labeled data
    print("\nTraining hazard detector...")
    train_hazard_detector(train_scenarios)
    detector = get_hazard_detector()
    
    # Get hazard IDs from training data
    hazard_ids = sorted({sc.get('hazard_type', 'unknown') for sc in train_scenarios})
    if "benign" not in hazard_ids:
        hazard_ids.append("benign")
    if "unknown" not in hazard_ids:
        hazard_ids.append("unknown")
    SAFE_ACTIONS.setdefault("unknown", ["reassure"])
    SAFE_ACTIONS.setdefault("benign", ["reassure"])
    
    emergency_hazards = {
        "suicide_risk", "metabolic_emergency", "neuro_emergency",
        "anaphylaxis", "obstetric_emergency", "cardiac_emergency",
        "pediatric_emergency", "misuse_escalation"
    }
    
    print("\n" + "="*80)
    print("TRAINING CQL CONTROLLERS WITH THREE CALIBRATIONS")
    print("="*80)
    
    # Configuration 1: High Specificity (Conservative)
    print("\n[1/3] Training High Specificity configuration...")
    penalty_conservative = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards and hazard not in ["benign", "unknown"]:
            penalty_conservative.setdefault(hazard, {})["activate_emergency"] = -0.5
            penalty_conservative.setdefault(hazard, {})["warn_and_escalate"] = -0.2
    
    dataset_conservative = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_conservative)
    cql_conservative = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_conservative.train(dataset_conservative, epochs=400)
    
    # Configuration 2: Balanced
    print("[2/3] Training Balanced configuration...")
    penalty_balanced = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards and hazard not in ["benign", "unknown"]:
            penalty_balanced.setdefault(hazard, {})["activate_emergency"] = -0.25
            penalty_balanced.setdefault(hazard, {})["warn_and_escalate"] = -0.1
    
    dataset_balanced = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_balanced)
    cql_balanced = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_balanced.train(dataset_balanced, epochs=400)
    
    # Configuration 3: High Sensitivity
    print("[3/3] Training High Sensitivity configuration...")
    penalty_sensitive = {}
    for hazard in hazard_ids:
        if hazard not in emergency_hazards and hazard not in ["benign", "unknown"]:
            penalty_sensitive.setdefault(hazard, {})["activate_emergency"] = -0.05
            penalty_sensitive.setdefault(hazard, {})["warn_and_escalate"] = -0.02
    
    dataset_sensitive = build_dataset(hazard_ids, ACTIONS, SAFE_ACTIONS, penalty=penalty_sensitive, repeats=15)
    cql_sensitive = CQLAgent(hazard_ids, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_sensitive.train(dataset_sensitive, epochs=600)
    
    print("\n" + "="*80)
    print("EVALUATING ON REAL-WORLD VALIDATION SET")
    print("="*80)
    
    # Separate real-world data into hazards and benign
    real_hazards = [s for s in real_world_scenarios if s.get('hazard_type', 'benign') != 'benign']
    real_benign = [s for s in real_world_scenarios if s.get('hazard_type', 'benign') == 'benign']
    
    print(f"\nReal-world hazards: {len(real_hazards)}")
    print(f"Real-world benign: {len(real_benign)}")
    
    # Evaluate Conservative
    print("\nEvaluating High Specificity...")
    rows_cons_haz = evaluate_agent(
        cql_conservative.state_to_idx,
        make_selector_from_q(cql_conservative.state_to_idx, cql_conservative.Q),
        real_hazards,
        detector=detector
    )
    rows_cons_ben = evaluate_agent(
        cql_conservative.state_to_idx,
        make_selector_from_q(cql_conservative.state_to_idx, cql_conservative.Q),
        real_benign,
        detector=detector
    )
    metrics_conservative = summarize_scenario_metrics(rows_cons_haz, rows_cons_ben)
    
    # Evaluate Balanced
    print("Evaluating Balanced...")
    rows_bal_haz = evaluate_agent(
        cql_balanced.state_to_idx,
        make_selector_from_q(cql_balanced.state_to_idx, cql_balanced.Q),
        real_hazards,
        detector=detector
    )
    rows_bal_ben = evaluate_agent(
        cql_balanced.state_to_idx,
        make_selector_from_q(cql_balanced.state_to_idx, cql_balanced.Q),
        real_benign,
        detector=detector
    )
    metrics_balanced = summarize_scenario_metrics(rows_bal_haz, rows_bal_ben)
    
    # Evaluate High Sensitivity
    print("Evaluating High Sensitivity...")
    rows_sens_haz = evaluate_agent(
        cql_sensitive.state_to_idx,
        make_selector_from_q(cql_sensitive.state_to_idx, cql_sensitive.Q),
        real_hazards,
        detector=detector
    )
    rows_sens_ben = evaluate_agent(
        cql_sensitive.state_to_idx,
        make_selector_from_q(cql_sensitive.state_to_idx, cql_sensitive.Q),
        real_benign,
        detector=detector
    )
    metrics_sensitive = summarize_scenario_metrics(rows_sens_haz, rows_sens_ben)
    
    # Compute F1 scores
    def calc_f1(m):
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        return 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    
    report = {
        "dataset": "Real-World Validation Set",
        "n_hazards": len(real_hazards),
        "n_benign": len(real_benign),
        "n_total": len(real_world_scenarios),
        "configurations": [
            {
                "name": "High Specificity (Conservative)",
                "penalty_weights": {"violation": -1.0, "false_alarm": -0.5},
                "sensitivity": metrics_conservative['sensitivity'],
                "specificity": metrics_conservative['specificity'],
                "precision": metrics_conservative['precision'],
                "f1_score": calc_f1(metrics_conservative),
                "tp": metrics_conservative['tp'],
                "fp": metrics_conservative['fp'],
                "fn": metrics_conservative['fn'],
                "tn": metrics_conservative['tn']
            },
            {
                "name": "Balanced",
                "penalty_weights": {"violation": -1.0, "false_alarm": -0.25},
                "sensitivity": metrics_balanced['sensitivity'],
                "specificity": metrics_balanced['specificity'],
                "precision": metrics_balanced['precision'],
                "f1_score": calc_f1(metrics_balanced),
                "tp": metrics_balanced['tp'],
                "fp": metrics_balanced['fp'],
                "fn": metrics_balanced['fn'],
                "tn": metrics_balanced['tn']
            },
            {
                "name": "High Sensitivity",
                "penalty_weights": {"violation": -1.0, "false_alarm": -0.05},
                "sensitivity": metrics_sensitive['sensitivity'],
                "specificity": metrics_sensitive['specificity'],
                "precision": metrics_sensitive['precision'],
                "f1_score": calc_f1(metrics_sensitive),
                "tp": metrics_sensitive['tp'],
                "fp": metrics_sensitive['fp'],
                "fn": metrics_sensitive['fn'],
                "tn": metrics_sensitive['tn']
            }
        ]
    }
    
    out = RESULTS_DIR / "cql_realworld_calibrations.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("REAL-WORLD VALIDATION RESULTS")
    print("="*80)
    for config in report["configurations"]:
        print(f"\n{config['name']}")
        print(f"  Penalty weights: Violation={config['penalty_weights']['violation']}, False Alarm={config['penalty_weights']['false_alarm']}")
        print(f"  Sensitivity: {config['sensitivity']:.1%} ({config['tp']}/{config['tp']+config['fn']})")
        print(f"  Specificity: {config['specificity']:.1%} ({config['tn']}/{config['tn']+config['fp']})")
        print(f"  Precision: {config['precision']:.1%}")
        print(f"  F1 Score: {config['f1_score']:.3f}")
        print(f"  Confusion: TP={config['tp']}, FP={config['fp']}, FN={config['fn']}, TN={config['tn']}")
    
    print(f"\n[ok] wrote {out}")
    return report


if __name__ == "__main__":
    main()
