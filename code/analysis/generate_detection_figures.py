#!/usr/bin/env python3
"""
Generate figures for NEJM AI detection-focused manuscript.

Figures:
1. Detection performance comparison (holdout + replay)
2. Calibration/rejection curves  
3. Confusion heatmap for detector
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = SUBMISSION_DIR = BASE_DIR / "submission/submission_bundle/figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set publication-quality style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi':300
})


def load_data():
    """Load all metrics."""
    with open(RESULTS_DIR / "calibrated_transformer_detector_metrics.json") as f:
        detector = json.load(f)
    
    with open(RESULTS_DIR / "transformer_detector_controller_metrics.json") as f:
        controller = json.load(f)
    
    # Simplified chatbot metrics (from table generation)
    chatbot = {
        "gpt5_holdout": {"sensitivity": 0.302, "specificity": 0.850, "sens_ci": [0.241, 0.370], "spec_ci": [0.794, 0.893]},
        "claude_holdout": {"sensitivity": 0.361, "specificity": 0.825, "sens_ci": [0.295, 0.430], "spec_ci": [0.766, 0.871]},
        "gpt5_replay": {"sensitivity": 0.116, "specificity": 0.857, "sens_ci": [0.093, 0.143], "spec_ci": [0.818, 0.889]},
        "claude_replay": {"sensitivity": 0.140, "specificity": 0.857, "sens_ci": [0.115, 0.169], "spec_ci": [0.818, 0.889]},
    }
    
    return detector, controller, chatbot


def figure1_detection_comparison():
    """Figure 1: Detection performance comparison across systems."""
    detector, controller, chatbot = load_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === Holdout Panel ===
    ax = axes[0]
    systems = ["GPT-5", "Claude 4.5", "Detector", "Detector+CQL"]
    
    # Sensitivity
    sens_vals = [
        chatbot["gpt5_holdout"]["sensitivity"],
        chatbot["claude_holdout"]["sensitivity"],
        detector["holdout"]["sensitivity"],
        controller["controller_holdout"]["sensitivity"]
    ]
    sens_cis = [
        [chatbot["gpt5_holdout"]["sens_ci"][0], chatbot["gpt5_holdout"]["sens_ci"][1] - chatbot["gpt5_holdout"]["sensitivity"]],
        [chatbot["claude_holdout"]["sens_ci"][0], chatbot["claude_holdout"]["sens_ci"][1] - chatbot["claude_holdout"]["sensitivity"]],
        [detector["holdout"]["sensitivity_ci"][0], detector["holdout"]["sensitivity_ci"][1] - detector["holdout"]["sensitivity"]],
        [controller["controller_holdout"]["sensitivity"], controller["controller_holdout"]["sensitivity"]]  # placeholder
    ]
    
    spec_vals = [
        chatbot["gpt5_holdout"]["specificity"],
        chatbot["claude_holdout"]["specificity"],
        detector["holdout"]["specificity"],
        controller["controller_holdout"]["specificity"]
    ]
    
    x = np.arange(len(systems))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sens_vals, width, label='Sensitivity', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, spec_vals, width, label='Specificity', color='#A23B72', alpha=0.8)
    
    ax.set_ylabel('Performance')
    ax.set_title('Physician-Created Holdout Scenarios', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # === Replay Panel ===
    ax = axes[1]
    
    sens_vals = [
        chatbot["gpt5_replay"]["sensitivity"],
        chatbot["claude_replay"]["sensitivity"],
        detector["replay"]["sensitivity"],
        controller["controller_replay"]["sensitivity"]
    ]
    
    spec_vals = [
        chatbot["gpt5_replay"]["specificity"],
        chatbot["claude_replay"]["specificity"],
        detector["replay"]["specificity"],
        controller["controller_replay"]["specificity"]
    ]
    
    bars1 = ax.bar(x - width/2, sens_vals, width, label='Sensitivity', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, spec_vals, width, label='Specificity', color='#A23B72', alpha=0.8)
    
    ax.set_ylabel('Performance')
    ax.set_title('Real-World Replay Messages', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "figure1_detection_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def figure2_rejection_curves():
    """Figure 2: Calibration and rejection curves."""
    detector, _, _ = load_data()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # === Holdout Rejection Curve ===
    ax = axes[0]
    holdout_curves = detector["rejection_curves"]["holdout"]
    
    thresholds = [c["threshold"] for c in holdout_curves]
    sens = [c["sensitivity"] for c in holdout_curves]
    spec = [c["specificity"] for c in holdout_curves]
    reject = [c["rejection_rate"] for c in holdout_curves]
    
    ax.plot(reject, sens, 'o-', label='Sensitivity', color='#2E86AB', linewidth=2, markersize=8)
    ax.plot(reject, spec, 's-', label='Specificity', color='#A23B72', linewidth=2, markersize=8)
    
    ax.set_xlabel('Rejection Rate')
    ax.set_ylabel('Performance')
    ax.set_title('Holdout: Sensitivity/Specificity vs. Rejection', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    # === Replay Rejection Curve ===
    ax = axes[1]
    replay_curves = detector["rejection_curves"]["replay"]
    
    thresholds = [c["threshold"] for c in replay_curves]
    sens = [c["sensitivity"] for c in replay_curves]
    spec = [c["specificity"] for c in replay_curves]
    reject = [c["rejection_rate"] for c in replay_curves]
    
    ax.plot(reject, sens, 'o-', label='Sensitivity', color='#2E86AB', linewidth=2, markersize=8)
    ax.plot(reject, spec, 's-', label='Specificity', color='#A23B72', linewidth=2, markersize=8)
    
    ax.set_xlabel('Rejection Rate')
    ax.set_ylabel('Performance')
    ax.set_title('Replay: Sensitivity/Specificity vs. Rejection', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "figure2_calibration_rejection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def figure3_roc_curves():
    """Figure 3: ROC-style curves showing sens/spec trade-off."""
    detector, _, _ = load_data()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot holdout
    holdout_curves = detector["rejection_curves"]["holdout"]
    fpr_h = [1 - c["specificity"] for c in holdout_curves]
    sens_h = [c["sensitivity"] for c in holdout_curves]
    
    # Plot replay
    replay_curves = detector["rejection_curves"]["replay"]
    fpr_r = [1 - c["specificity"] for c in replay_curves]
    sens_r = [c["sensitivity"] for c in replay_curves]
    
    ax.plot(fpr_h, sens_h, 'o-', label='Holdout', color='#2E86AB', linewidth=2, markersize=10)
    ax.plot(fpr_r, sens_r, 's-', label='Replay', color='#A23B72', linewidth=2, markersize=10)
    
    # Add diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('ROC-Style Operating Characteristic Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    output_path = FIGURES_DIR / "figure3_roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all figures."""
    print("="*80)
    print("GENERATING FIGURES FOR NEJM AI MANUSCRIPT")
    print("="*80)
    
    print("\n1. Figure 1: Detection performance comparison...")
    figure1_detection_comparison()
    
    print("\n2. Figure 2: Calibration/rejection curves...")
    figure2_rejection_curves()
    
    print("\n3. Figure 3: ROC curves...")
    figure3_roc_curves()
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
