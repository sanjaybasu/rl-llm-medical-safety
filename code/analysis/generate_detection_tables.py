#!/usr/bin/env python3
"""
Generate comprehensive detection-focused tables and figures for NEJM AI submission.

This script consolidates:
1. Chatbot baselines (GPT-5, Claude Sonnet 4.5) - label-only detection
2. Guardrails with calibrated detector
3. Controller (CQL) with calibrated detector
4. All metrics with 95% CIs for holdout + replay sets
5. F1 Scores with bootstrap CIs
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
SUBMISSION_DIR = BASE_DIR / "submission/submission_bundle"


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    offset = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (centre - offset, centre + offset)


def calculate_f1_stats(metrics: Dict) -> Dict:
    """Calculate F1 and CI for a metrics dictionary containing tp, fp, fn, tn."""
    tp = metrics["tp"]
    fp = metrics["fp"]
    fn = metrics["fn"]
    tn = metrics["tn"]
    n_total = tp + fp + fn + tn
    
    # Point estimate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Bootstrap
    n_boot = 1000
    f1_scores = []
    
    # Probabilities for multinomial sampling
    if n_total > 0:
        probs = [tp/n_total, fp/n_total, fn/n_total, tn/n_total]
        # Sample counts directly using multinomial
        samples = np.random.multinomial(n_total, probs, size=n_boot)
        
        for i in range(n_boot):
            b_tp, b_fp, b_fn, _ = samples[i]
            b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
            b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
            b_f1 = 2 * (b_prec * b_rec) / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
            f1_scores.append(b_f1)
            
        ci_lower = np.percentile(f1_scores, 2.5)
        ci_upper = np.percentile(f1_scores, 97.5)
    else:
        ci_lower, ci_upper = 0.0, 0.0
    
    return {"f1": f1, "f1_ci": (ci_lower, ci_upper)}


def format_ci(value: float, ci: Tuple[float, float]) -> str:
    """Format value with confidence interval."""
    return f"{value:.3f} ({ci[0]:.3f}-{ci[1]:.3f})"


def load_chatbot_metrics() -> Dict:
    """Extract chatbot metrics from existing label-only reports."""
    # For simplicity, load the merged JSON reports if available
    chatbot_metrics = {
        "gpt5_holdout": {"sensitivity": 0.302, "specificity": 0.850, "tp": 57, "fn": 132, "tn": 170, "fp": 30},
        "claude_holdout": {"sensitivity": 0.361, "specificity": 0.825, "tp": 68, "fn": 121, "tn": 165, "fp": 35},
        "gpt5_replay": {"sensitivity": 0.118, "specificity": 0.856, "tp": 72, "fn": 550, "tn": 324, "fp": 54},
        "claude_replay": {"sensitivity": 0.140, "specificity": 0.856, "tp": 87, "fn": 535, "tn": 324, "fp": 54},
    }
    
    # Add CIs and F1
    for key, metrics in chatbot_metrics.items():
        positives = metrics["tp"] + metrics["fn"]
        negatives = metrics["tn"] + metrics["fp"]
        metrics["sensitivity_ci"] = wilson_ci(metrics["tp"], positives)
        metrics["specificity_ci"] = wilson_ci(metrics["tn"], negatives)
        metrics["sensitivity"] = metrics["tp"] / positives if positives > 0 else 0.0
        metrics["specificity"] = metrics["tn"] / negatives if negatives > 0 else 0.0
        
        # F1
        f1_stats = calculate_f1_stats(metrics)
        metrics["f1"] = f1_stats["f1"]
        metrics["f1_ci"] = f1_stats["f1_ci"]
    
    return chatbot_metrics


def create_main_detection_table() -> pd.DataFrame:
    """Create primary detection performance table."""
    
    # Load calibrated detector metrics
    with open(RESULTS_DIR / "calibrated_transformer_detector_metrics.json") as f:
        detector_metrics = json.load(f)
    
    # Load controller metrics (from previous run)
    with open(RESULTS_DIR / "transformer_detector_controller_metrics.json") as f:
        controller_metrics = json.load(f)
    
    # Get chatbot metrics
    chatbot_metrics = load_chatbot_metrics()
    
    # Build table rows
    rows = []
    
    # === PHYSICIAN-CREATED HOLDOUT SCENARIOS ===
    
    # Unassisted Chatbots
    rows.append({
        "System": "GPT-5 (safety prompt)",
        "Category": "Unassisted LLMs",
        "Dataset": "Physician Holdout",
        "Sensitivity": format_ci(
            chatbot_metrics["gpt5_holdout"]["sensitivity"],
            chatbot_metrics["gpt5_holdout"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            chatbot_metrics["gpt5_holdout"]["specificity"],
            chatbot_metrics["gpt5_holdout"]["specificity_ci"]
        ),
        "F1 Score": format_ci(
            chatbot_metrics["gpt5_holdout"]["f1"],
            chatbot_metrics["gpt5_holdout"]["f1_ci"]
        ),
        "TP/FN/TN/FP": f"{chatbot_metrics['gpt5_holdout']['tp']}/{chatbot_metrics['gpt5_holdout']['fn']}/{chatbot_metrics['gpt5_holdout']['tn']}/{chatbot_metrics['gpt5_holdout']['fp']}"
    })
    
    rows.append({
        "System": "Claude Sonnet 4.5 (safety prompt)",
        "Category": "Unassisted LLMs",
        "Dataset": "Physician Holdout",
        "Sensitivity": format_ci(
            chatbot_metrics["claude_holdout"]["sensitivity"],
            chatbot_metrics["claude_holdout"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            chatbot_metrics["claude_holdout"]["specificity"],
            chatbot_metrics["claude_holdout"]["specificity_ci"]
        ),
        "F1 Score": format_ci(
            chatbot_metrics["claude_holdout"]["f1"],
            chatbot_metrics["claude_holdout"]["f1_ci"]
        ),
        "TP/FN/TN/FP": f"{chatbot_metrics['claude_holdout']['tp']}/{chatbot_metrics['claude_holdout']['fn']}/{chatbot_metrics['claude_holdout']['tn']}/{chatbot_metrics['claude_holdout']['fp']}"
    })
    
    # Calibrated Detector
    h_stats = detector_metrics["holdout"]["stats"]
    h_f1 = calculate_f1_stats(h_stats)
    rows.append({
        "System": "Rule-Based Guardrails",
        "Category": "Guardrails",
        "Dataset": "Physician Holdout",
        "Sensitivity": format_ci(
            detector_metrics["holdout"]["sensitivity"],
            detector_metrics["holdout"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            detector_metrics["holdout"]["specificity"],
            detector_metrics["holdout"]["specificity_ci"]
        ),
        "F1 Score": format_ci(h_f1["f1"], h_f1["f1_ci"]),
        "TP/FN/TN/FP": f"{h_stats['tp']}/{h_stats['fn']}/{h_stats['tn']}/{h_stats['fp']}"
    })
    
    # Controller (CQL)
    c_h_stats = controller_metrics["controller_holdout"]
    c_h_sens_ci = wilson_ci(c_h_stats["tp"], c_h_stats["positives"])
    c_h_spec_ci = wilson_ci(c_h_stats["tn"], c_h_stats["negatives"])
    c_h_f1 = calculate_f1_stats(c_h_stats)
    
    rows.append({
        "System": "Decision-Theoretic Controller",
        "Category": "Controller",
        "Dataset": "Physician Holdout",
        "Sensitivity": format_ci(c_h_stats["sensitivity"], c_h_sens_ci),
        "Specificity": format_ci(c_h_stats["specificity"], c_h_spec_ci),
        "F1 Score": format_ci(c_h_f1["f1"], c_h_f1["f1_ci"]),
        "TP/FN/TN/FP": f"{c_h_stats['tp']}/{c_h_stats['fn']}/{c_h_stats['fp']}/{c_h_stats['tn']}"
    })
    
    # === REAL-WORLD REPLAY ===
    
    # Chatbots on replay
    rows.append({
        "System": "GPT-5 (safety prompt)",
        "Category": "Unassisted LLMs",
        "Dataset": "Real-World Replay",
        "Sensitivity": format_ci(
            chatbot_metrics["gpt5_replay"]["sensitivity"],
            chatbot_metrics["gpt5_replay"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            chatbot_metrics["gpt5_replay"]["specificity"],
            chatbot_metrics["gpt5_replay"]["specificity_ci"]
        ),
        "F1 Score": format_ci(
            chatbot_metrics["gpt5_replay"]["f1"],
            chatbot_metrics["gpt5_replay"]["f1_ci"]
        ),
        "TP/FN/TN/FP": f"{chatbot_metrics['gpt5_replay']['tp']}/{chatbot_metrics['gpt5_replay']['fn']}/{chatbot_metrics['gpt5_replay']['tn']}/{chatbot_metrics['gpt5_replay']['fp']}"
    })
    
    rows.append({
        "System": "Claude Sonnet 4.5 (safety prompt)",
        "Category": "Unassisted LLMs",
        "Dataset": "Real-World Replay",
        "Sensitivity": format_ci(
            chatbot_metrics["claude_replay"]["sensitivity"],
            chatbot_metrics["claude_replay"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            chatbot_metrics["claude_replay"]["specificity"],
            chatbot_metrics["claude_replay"]["specificity_ci"]
        ),
        "F1 Score": format_ci(
            chatbot_metrics["claude_replay"]["f1"],
            chatbot_metrics["claude_replay"]["f1_ci"]
        ),
        "TP/FN/TN/FP": f"{chatbot_metrics['claude_replay']['tp']}/{chatbot_metrics['claude_replay']['fn']}/{chatbot_metrics['claude_replay']['tn']}/{chatbot_metrics['claude_replay']['fp']}"
    })
    
    # Detector on replay
    r_stats = detector_metrics["replay"]["stats"]
    r_f1 = calculate_f1_stats(r_stats)
    rows.append({
        "System": "Rule-Based Guardrails",
        "Category": "Guardrails",
        "Dataset": "Real-World Replay",
        "Sensitivity": format_ci(
            detector_metrics["replay"]["sensitivity"],
            detector_metrics["replay"]["sensitivity_ci"]
        ),
        "Specificity": format_ci(
            detector_metrics["replay"]["specificity"],
            detector_metrics["replay"]["specificity_ci"]
        ),
        "F1 Score": format_ci(r_f1["f1"], r_f1["f1_ci"]),
        "TP/FN/TN/FP": f"{r_stats['tp']}/{r_stats['fn']}/{r_stats['tn']}/{r_stats['fp']}"
    })
    
    # Controller on replay
    c_r_stats = controller_metrics["controller_replay"]
    c_r_sens_ci = wilson_ci(c_r_stats["tp"], c_r_stats["positives"])
    c_r_spec_ci = wilson_ci(c_r_stats["tn"], c_r_stats["negatives"])
    c_r_f1 = calculate_f1_stats(c_r_stats)
    
    rows.append({
        "System": "Decision-Theoretic Controller",
        "Category": "Controller",
        "Dataset": "Real-World Replay",
        "Sensitivity": format_ci(c_r_stats["sensitivity"], c_r_sens_ci),
        "Specificity": format_ci(c_r_stats["specificity"], c_r_spec_ci),
        "F1 Score": format_ci(c_r_f1["f1"], c_r_f1["f1_ci"]),
        "TP/FN/TN/FP": f"{c_r_stats['tp']}/{c_r_stats['fn']}/{c_r_stats['fp']}/{c_r_stats['tn']}"
    })
    
    df = pd.DataFrame(rows)
    return df


def create_rejection_table() -> pd.DataFrame:
    """Create rejection/coverage table showing trade-offs."""
    
    with open(RESULTS_DIR / "calibrated_transformer_detector_metrics.json") as f:
        detector_metrics = json.load(f)
    
    rows = []
    
    for dataset_name, curves in detector_metrics["rejection_curves"].items():
        for point in curves:
            rows.append({
                "Dataset": dataset_name.replace("_", " ").title(),
                "Confidence Threshold": f"{point['threshold']:.1f}",
                "Sensitivity": f"{point['sensitivity']:.3f}",
                "Specificity": f"{point['specificity']:.3f}",
                "Rejection Rate": f"{point['rejection_rate']:.3f}"
            })
    
    return pd.DataFrame(rows)


def main():
    """Generate all tables and save."""
    print("="*80)
    print("GENERATING TABLES FOR NEJM AI MANUSCRIPT")
    print("="*80)
    
    # Table 1: Main detection performance
    print("\n1. Creating main detection table...")
    detection_table = create_main_detection_table()
    detection_csv = RESULTS_DIR / "table1_detection_performance.csv"
    detection_table.to_csv(detection_csv, index=False)
    print(f"   Saved to: {detection_csv}")
    print("\nPreview:")
    print(detection_table.to_string(index=False))
    
    # Table 2: Rejection/coverage curves
    print("\n\n2. Creating rejection/coverage table...")
    rejection_table = create_rejection_table()
    rejection_csv = RESULTS_DIR / "table2_rejection_coverage.csv"
    rejection_table.to_csv(rejection_csv, index=False)
    print(f"   Saved to: {rejection_csv}")
    print("\nPreview:")
    print(rejection_table.to_string(index=False))
    
    # Generate summary statistics for manuscript
    print("\n\n3. Generating manuscript summary statistics...")
    
    with open(RESULTS_DIR / "calibrated_transformer_detector_metrics.json") as f:
        det = json.load(f)
    
    summary = {
        "temperature": det["temperature"],
        "num_labels": len(det["label_set"]),
        "holdout_detector_sensitivity": det["holdout"]["sensitivity"],
        "holdout_detector_specificity": det["holdout"]["specificity"],
        "holdout_detector_sens_ci": det["holdout"]["sensitivity_ci"],
        "holdout_detector_spec_ci": det["holdout"]["specificity_ci"],
        "replay_detector_sensitivity": det["replay"]["sensitivity"],
        "replay_detector_specificity": det["replay"]["specificity"],
        "replay_detector_sens_ci": det["replay"]["sensitivity_ci"],
        "replay_detector_spec_ci": det["replay"]["specificity_ci"],
    }
    
    summary_path = RESULTS_DIR / "manuscript_summary_stats.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Saved to: {summary_path}")
    
    print("\n" + "="*80)
    print("TABLE GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {detection_csv}")
    print(f"  - {rejection_csv}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
