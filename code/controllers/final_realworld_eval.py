#!/usr/bin/env python3
"""
FINAL: Real-world hazard evaluation with clinical taxonomy mapping.
Maps real-world classes to physician taxonomy for SAFE_ACTIONS compatibility.
"""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(CURRENT_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(CURRENT_DIR))

from evaluate_llm_safety import train_hazard_detector, get_hazard_detector
from run_rl_vs_llm_demo import (
    ACTIONS, SAFE_ACTIONS, CQLAgent, build_dataset,
    make_selector_from_q, summarize_scenario_metrics
)

random.seed(42)
np.random.seed(42)

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT_DIR / "data"

# Clinical mapping from real-world to physician taxonomy
HAZARD_MAPPING = {
    'emergency_symptom': 'cardiac_emergency',  # Conservative: treat as highest priority
    'mental_health_crisis': 'suicide_risk',
    'medication_safety': 'drug_interaction',  # Representative medication hazard
    'social_needs_risk': 'benign',  # Not a medical emergency
    'care_coordination': 'med_reconciliation',
    'safety_threat': 'misuse_escalation',
    'general_risk': 'benign',
    'benign': 'benign',
    'unknown': 'unknown'
}

# Extend SAFE_ACTIONS for real-world classes
SAFE_ACTIONS_REALWORLD = SAFE_ACTIONS.copy()
for rw_class, phys_class in HAZARD_MAPPING.items():
    if phys_class in SAFE_ACTIONS and rw_class not in SAFE_ACTIONS_REALWORLD:
        SAFE_ACTIONS_REALWORLD[rw_class] = SAFE_ACTIONS[phys_class]


def evaluate_with_mapping(agent, detector, scenarios):
    """Evaluate agent with hazard class mapping."""
    rows = []
    for sc in scenarios:
        prompt = sc.get('prompt', sc.get('message', ''))
        context = sc.get('context', {})
        true_hazard = sc.get('hazard_type', 'unknown')
        
        # Detect hazard
        det = detector.predict(prompt, context)
        pred_hazard = det.label if det.label else 'unknown'
        
        # Map to physician taxonomy for action selection
        mapped_true = HAZARD_MAPPING.get(true_hazard, 'unknown')
        mapped_pred = HAZARD_MAPPING.get(pred_hazard, 'unknown')
        
        # Get actions
        if mapped_pred in agent.state_to_idx:
            idx = agent.state_to_idx[mapped_pred]
            action_idx = int(np.argmax(agent.Q[idx]))
            chosen_action_pred = ACTIONS[action_idx]
        else:
            chosen_action_pred = None
        
        if mapped_true in agent.state_to_idx:
            idx = agent.state_to_idx[mapped_true]
            action_idx = int(np.argmax(agent.Q[idx]))
            chosen_action_true = ACTIONS[action_idx]
        else:
            chosen_action_true = None
        
        # Check if safe
        safe_actions = SAFE_ACTIONS_REALWORLD.get(true_hazard, [])
        is_safe_pred = chosen_action_pred in safe_actions if chosen_action_pred else False
        is_safe_true = chosen_action_true in safe_actions if chosen_action_true else False
        
        rows.append({
           'hazard_true': true_hazard,
            'hazard_pred': pred_hazard,
            'mapped_true': mapped_true,
            'mapped_pred': mapped_pred,
            'chosen_action_pred': chosen_action_pred,
            'is_safe_pred': is_safe_pred,
            'is_safe_true': is_safe_true
        })
    
    return rows


def bootstrap_ci(tp: int, fp: int, fn: int, tn: int, n_boot: int = 10000):
    """Compute bootstrap 95% CIs."""
    n_total = tp + fp + fn + tn
    if n_total == 0:
        return {'sens': (0,0,0), 'spec': (0,0,0), 'f1': (0,0,0)}
    
    probs = np.array([tp, fp, fn, tn]) / n_total
    samples = np.random.multinomial(n_total, probs, size=n_boot)
    
    sens_list, spec_list, f1_list = [], [], []
    for b_tp, b_fp, b_fn, b_tn in samples:
        sens = b_tp/(b_tp+b_fn) if (b_tp+b_fn)>0 else 0
        spec = b_tn/(b_tn+b_fp) if (b_tn+b_fp)>0 else 0
        prec = b_tp/(b_tp+b_fp) if (b_tp+b_fp)>0 else 0
        rec = b_tp/(b_tp+b_fn) if (b_tp+b_fn)>0 else 0
        f1 = 2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0
        sens_list.append(sens)
        spec_list.append(spec)
        f1_list.append(f1)
    
    return {
        'sens': (tp/(tp+fn) if (tp+fn)>0 else 0, np.percentile(sens_list, 2.5), np.percentile(sens_list, 97.5)),
        'spec': (tn/(tn+fp) if (tn+fp)>0 else 0, np.percentile(spec_list, 2.5), np.percentile(spec_list, 97.5)),
        'f1': (2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else 0, np.percentile(f1_list, 2.5), np.percentile(f1_list, 97.5))
    }


def main():
    print("="*80)
    print("NEJM AI: REAL-WORLD CQL CONTROLLER EVALUATION WITH HAZARD MAPPING")
    print("="*80)
    
    # Load data
    print("\n[1/5] Loading real-world data...")
    replay_path = DATA_DIR / "replay_scenarios.json"
    with open(replay_path, 'r') as f:
        all_scenarios = json.load(f)
    
    train_scenarios, val_scenarios = train_test_split(
        all_scenarios, test_size=0.20, random_state=42,
        stratify=[s.get('hazard_type', 'benign') for s in all_scenarios]
    )
    
    print(f"Training: {len(train_scenarios)}, Validation: {len(val_scenarios)}")
    
    # Train detector
    print("\n[2/5] Training detector on real-world data...")
    train_hazard_detector(train_scenarios)
    detector = get_hazard_detector()
    
    # Get ID from physician taxonomy (for controller training)
    physician_hazards = sorted(set(HAZARD_MAPPING.values()))
    print(f"Controller using {len(physician_hazards)} physician hazard classes")
    
    # Emergency hazards
    emergency = {'suicide_risk', 'cardiac_emergency', 'neuro_emergency', 
                 'metabolic_emergency', 'anaphylaxis', 'obstetric_emergency',
                 'pediatric_emergency', 'misuse_escalation'}
    
    print("\n[3/5] Training three CQL calibrations...")
    
    # High Specificity
    print("  [1/3] High Specificity...")
    penalty_1 = {}
    for h in physician_hazards:
        if h not in emergency and h not in ['benign', 'unknown']:
            penalty_1.setdefault(h, {})['activate_emergency'] = -0.5
            penalty_1.setdefault(h, {})['warn_and_escalate'] = -0.2
    
    dataset_1 = build_dataset(physician_hazards, ACTIONS, SAFE_ACTIONS, penalty=penalty_1)
    cql_1 = CQLAgent(physician_hazards, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_1.train(dataset_1, epochs=400)
    
    # Balanced
    print("  [2/3] Balanced...")
    penalty_2 = {}
    for h in physician_hazards:
        if h not in emergency and h not in ['benign', 'unknown']:
            penalty_2.setdefault(h, {})['activate_emergency'] = -0.25
            penalty_2.setdefault(h, {})['warn_and_escalate'] = -0.1
    
    dataset_2 = build_dataset(physician_hazards, ACTIONS, SAFE_ACTIONS, penalty=penalty_2)
    cql_2 = CQLAgent(physician_hazards, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_2.train(dataset_2, epochs=400)
    
    # High Sensitivity
    print("  [3/3] High Sensitivity...")
    penalty_3 = {}
    for h in physician_hazards:
        if h not in emergency and h not in ['benign', 'unknown']:
            penalty_3.setdefault(h, {})['activate_emergency'] = -0.05
            penalty_3.setdefault(h, {})['warn_and_escalate'] = -0.02
    
    dataset_3 = build_dataset(physician_hazards, ACTIONS, SAFE_ACTIONS, penalty=penalty_3, repeats=15)
    cql_3 = CQLAgent(physician_hazards, len(ACTIONS), alpha=0.05, beta=0.5)
    cql_3.train(dataset_3, epochs=600)
    
    print("\n[4/5] Evaluating on validation set with hazard mapping...")
    
    val_hazards = [s for s in val_scenarios if s.get('hazard_type') != 'benign']
    val_benign = [s for s in val_scenarios if s.get('hazard_type') == 'benign']
    
    print(f"  Hazards: {len(val_hazards)}, Benign: {len(val_benign)}")
    
    configs = [
        ("High Specificity", cql_1, {"violation": -1.0, "false_alarm": -0.5}),
        ("Balanced", cql_2, {"violation": -1.0, "false_alarm": -0.25}),
        ("High Sensitivity", cql_3, {"violation": -1.0, "false_alarm": -0.05})
    ]
    
    results = []
    for name, agent, weights in configs:
        print(f"\n  Evaluating {name}...")
        
        rows_haz = evaluate_with_mapping(agent, detector, val_hazards)
        rows_ben = evaluate_with_mapping(agent, detector, val_benign)
        
        metrics = summarize_scenario_metrics(rows_haz, rows_ben)
        cis = bootstrap_ci(metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn'])
        
        results.append({
            "name": name,
            "penalty_weights": weights,
            "tp": metrics['tp'], "fp": metrics['fp'],
            "fn": metrics['fn'], "tn": metrics['tn'],
            "sens": cis['sens'][0], "sens_ci": (cis['sens'][1], cis['sens'][2]),
            "spec": cis['spec'][0], "spec_ci": (cis['spec'][1], cis['spec'][2]),
            "f1": cis['f1'][0], "f1_ci": (cis['f1'][1], cis['f1'][2])
        })
    
    print("\n[5/5] Saving results...")
    
    report = {
        "method": "Real-world detector + Hazard mapping + CQL controllers",
        "hazard_mapping": HAZARD_MAPPING,
        "n_val": len(val_scenarios),
        "configurations": results
    }
    
    out = RESULTS_DIR / "cql_realworld_final.json"
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL VALIDATION RESULTS (REAL DATA + BOOTSTRAP CIs)")
    print("="*80)
    
    for cfg in results:
        print(f"\n{cfg['name']}")
        print(f"  Penalties: {cfg['penalty_weights']}")
        print(f"  Sensitivity: {cfg['sens']:.1%} (95% CI: {cfg['sens_ci'][0]:.1%}–{cfg['sens_ci'][1]:.1%})")
        print(f"  Specificity: {cfg['spec']:.1%} (95% CI: {cfg['spec_ci'][0]:.1%}–{cfg['spec_ci'][1]:.1%})")
        print(f"  F1 Score: {cfg['f1']:.3f} (95% CI: {cfg['f1_ci'][0]:.3f}–{cfg['f1_ci'][1]:.3f})")
        print(f"  Matrix: TP={cfg['tp']}, FP={cfg['fp']}, FN={cfg['fn']}, TN={cfg['tn']}")
    
    print(f"\n✓ Results saved to {out}")


if __name__ == "__main__":
    main()
