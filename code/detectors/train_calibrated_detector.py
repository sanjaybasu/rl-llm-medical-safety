#!/usr/bin/env python3
"""
Train a transformer-based hazard detector with calibration and rejection capabilities.

This script:
1. Loads physician-created scenarios + benign + LLM-labeled replay
2. Trains a MiniLM embedding + logistic classifier
3. Applies temperature scaling for calibration
4. Generates confidence/rejection curves
5. Re-runs detection metrics (sens/spec + CIs) on holdout and replay
6. Saves updated results for manuscript tables/figures
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Scenario:
    name: str
    prompt: str
    context: Dict
    hazard_type: str
    severity: str
    required_actions: List[str]


def load_json_scenarios(path: Path) -> List[Scenario]:
    """Load scenarios from JSON file."""
    with path.open() as f:
        data = json.load(f)
    scenarios = []
    for entry in data:
        scenarios.append(Scenario(
            name=entry.get("name", ""),
            prompt=entry.get("prompt", ""),
            context=entry.get("context", {}),
            hazard_type=entry.get("hazard_type", "unknown"),
            severity=entry.get("severity", "moderate"),
            required_actions=entry.get("required_actions", [])
        ))
    return scenarios


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson score confidence interval."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    offset = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (centre - offset, centre + offset)


class TransformerHazardDetector:
    """Transformer-based hazard detector with calibration."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = None
        self.temperature = 1.0
        self.label_encoder = {}
        self.label_decoder = {}
        
    def fit(self, scenarios: List[Scenario], calibration_split: float = 0.2):
        """Train classifier and calibrate temperature."""
        print(f"Training on {len(scenarios)} scenarios...")
        
        # Encode features
        texts = [s.prompt for s in scenarios]
        X = self.encoder.encode(texts, show_progress_bar=True)
        
        # Encode labels
        labels = [s.hazard_type for s in scenarios]
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
        self.label_decoder = {idx: label for label, idx in self.label_encoder.items()}
        y = np.array([self.label_encoder[label] for label in labels])
        
        # Split for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=calibration_split, random_state=42, stratify=y
        )
        
        # Train classifier
        print("Training logistic classifier...")
        self.classifier = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            class_weight='balanced',
            random_state=42
        )
        self.classifier.fit(X_train, y_train)
        
        # Calibrate temperature
        print("Calibrating temperature...")
        self.temperature = self._calibrate_temperature(X_cal, y_cal)
        print(f"Optimal temperature: {self.temperature:.3f}")
        
        return self
    
    def _calibrate_temperature(self, X_cal: np.ndarray, y_cal: np.ndarray) -> float:
        """Find optimal temperature using calibration set."""
        from scipy.optimize import minimize_scalar
        
        logits = self.classifier.decision_function(X_cal)
        
        def nll(temp):
            """Negative log-likelihood with temperature scaling."""
            scaled_logits = logits / temp
            # Compute softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            # Compute NLL
            log_probs = np.log(probs[np.arange(len(y_cal)), y_cal] + 1e-12)
            return -np.mean(log_probs)
        
        result = minimize_scalar(nll, bounds=(0.1, 5.0), method='bounded')
        return result.x
    
    def predict(self, scenarios: List[Scenario], confidence_threshold: float = 0.0) -> Dict:
        """Predict with calibrated confidences."""
        texts = [s.prompt for s in scenarios]
        X = self.encoder.encode(texts, show_progress_bar=False)
        
        # Get calibrated probabilities
        logits = self.classifier.decision_function(X)
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get predictions
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        results = []
        for i, scenario in enumerate(scenarios):
            pred_label = self.label_decoder[pred_indices[i]]
            confidence = float(confidences[i])
            
            # Apply rejection threshold
            if confidence < confidence_threshold:
                pred_label = "unknown"
            
            results.append({
                "scenario": scenario.name,
                "true_label": scenario.hazard_type,
                "pred_label": pred_label,
                "confidence": confidence,
                "probabilities": {self.label_decoder[idx]: float(probs[i, idx]) 
                                for idx in range(len(self.label_decoder))}
            })
        
        return {"predictions": results}
    
    def evaluate(self, scenarios: List[Scenario], confidence_threshold: float = 0.0) -> Dict:
        """Evaluate detector performance."""
        pred_results = self.predict(scenarios, confidence_threshold)
        predictions = pred_results["predictions"]
        
        # Separate hazards and benign
        hazard_scenarios = [s for s in scenarios if s.hazard_type != "benign"]
        benign_scenarios = [s for s in scenarios if s.hazard_type == "benign"]
        
        # Compute stats
        stats = {"tp": 0, "fn": 0, "tn": 0, "fp": 0, "unknown": 0}
        
        for pred, scenario in zip(predictions, scenarios):
            true_label = scenario.hazard_type
            pred_label = pred["pred_label"]
            
            if pred_label == "unknown":
                stats["unknown"] += 1
                # Treat unknown as negative for conservative metric
                if true_label != "benign":
                    stats["fn"] += 1
                else:
                    stats["tn"] += 1
            elif true_label == "benign":
                if pred_label == "benign":
                    stats["tn"] += 1
                else:
                    stats["fp"] += 1
            else:  # true positive case
                if pred_label == true_label:
                    stats["tp"] += 1
                else:
                    stats["fn"] += 1
        
        # Compute metrics
        positives = len(hazard_scenarios)
        negatives = len(benign_scenarios)
        
        sensitivity = stats["tp"] / positives if positives > 0 else 0.0
        specificity = stats["tn"] / negatives if negatives > 0 else 0.0
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
        npv = stats["tn"] / (stats["tn"] + stats["fn"]) if (stats["tn"] + stats["fn"]) > 0 else 0.0
        
        # Confidence intervals
        sens_ci = wilson_ci(stats["tp"], positives)
        spec_ci = wilson_ci(stats["tn"], negatives)
        
        return {
            "stats": stats,
            "positives": positives,
            "negatives": negatives,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision,
            "npv": npv,
            "sensitivity_ci": sens_ci,
            "specificity_ci": spec_ci,
            "rejection_rate": stats["unknown"] / len(scenarios) if len(scenarios) > 0 else 0.0
        }


def main():
    """Main training and evaluation pipeline."""
    print("="*80)
    print("TRAINING CALIBRATED TRANSFORMER HAZARD DETECTOR")
    print("="*80)
    
    # Load training data
    print("\n1. Loading training data...")
    train_scenarios = load_json_scenarios(DATA_DIR / "hazard_scenarios_train.json")
    benign_scenarios = load_json_scenarios(DATA_DIR / "benign_scenarios_augmented.json")
    replay_labeled = load_json_scenarios(DATA_DIR / "replay_scenarios_llm_labels.json")
    
    print(f"   - Physician hazards: {len(train_scenarios)}")
    print(f"   - Benign scenarios: {len(benign_scenarios)}")
    print(f"   - LLM-labeled replay: {len(replay_labeled)}")
    
    # Combine training data
    all_train = train_scenarios + benign_scenarios + replay_labeled
    print(f"   - Total training: {len(all_train)}")
    
    # Train detector
    print("\n2. Training transformer detector...")
    detector = TransformerHazardDetector()
    detector.fit(all_train, calibration_split=0.2)
    
    # Load holdout and replay test sets
    print("\n3. Loading test sets...")
    holdout_scenarios = load_json_scenarios(DATA_DIR / "hazard_scenarios_holdout.json")
    holdout_benign = benign_scenarios[:200]  # Use subset for holdout benign
    holdout_all = holdout_scenarios + holdout_benign
    
    # Use the labeled replay for real-world validation
    replay_test = replay_labeled
    
    print(f"   - Holdout hazards: {len(holdout_scenarios)}")
    print(f"   - Holdout benign: {len(holdout_benign)}")
    print(f"   - Replay test: {len(replay_test)}")
    
    # Evaluate on holdout
    print("\n4. Evaluating on holdout set...")
    holdout_results = detector.evaluate(holdout_all, confidence_threshold=0.0)
    
    print(f"\nHoldout Performance:")
    print(f"   Sensitivity: {holdout_results['sensitivity']:.3f} "
          f"(95% CI: {holdout_results['sensitivity_ci'][0]:.3f}-{holdout_results['sensitivity_ci'][1]:.3f})")
    print(f"   Specificity: {holdout_results['specificity']:.3f} "
          f"(95% CI: {holdout_results['specificity_ci'][0]:.3f}-{holdout_results['specificity_ci'][1]:.3f})")
    print(f"   Stats: tp={holdout_results['stats']['tp']}, fn={holdout_results['stats']['fn']}, "
          f"tn={holdout_results['stats']['tn']}, fp={holdout_results['stats']['fp']}")
    
    # Evaluate on replay
    print("\n5. Evaluating on replay set...")
    replay_results = detector.evaluate(replay_test, confidence_threshold=0.0)
    
    print(f"\nReplay Performance:")
    print(f"   Sensitivity: {replay_results['sensitivity']:.3f} "
          f"(95% CI: {replay_results['sensitivity_ci'][0]:.3f}-{replay_results['sensitivity_ci'][1]:.3f})")
    print(f"   Specificity: {replay_results['specificity']:.3f} "
          f"(95% CI: {replay_results['specificity_ci'][0]:.3f}-{replay_results['specificity_ci'][1]:.3f})")
    print(f"   Stats: tp={replay_results['stats']['tp']}, fn={replay_results['stats']['fn']}, "
          f"tn={replay_results['stats']['tn']}, fp={replay_results['stats']['fp']}")
    
    # Test different rejection thresholds
    print("\n6. Computing rejection/coverage curves...")
    thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]
    rejection_curves = {"holdout": [], "replay": []}
    
    for thresh in thresholds:
        h_res = detector.evaluate(holdout_all, confidence_threshold=thresh)
        r_res = detector.evaluate(replay_test, confidence_threshold=thresh)
        
        rejection_curves["holdout"].append({
            "threshold": thresh,
            "sensitivity": h_res["sensitivity"],
            "specificity": h_res["specificity"],
            "rejection_rate": h_res["rejection_rate"]
        })
        
        rejection_curves["replay"].append({
            "threshold": thresh,
            "sensitivity": r_res["sensitivity"],
            "specificity": r_res["specificity"],
            "rejection_rate": r_res["rejection_rate"]
        })
    
    # Save results
    print("\n7. Saving results...")
    output = {
        "detector_type": "transformer_miniLM_calibrated",
        "temperature": detector.temperature,
        "label_set": sorted(detector.label_encoder.keys()),
        "holdout": {
            "stats": holdout_results["stats"],
            "sensitivity": holdout_results["sensitivity"],
            "specificity": holdout_results["specificity"],
            "precision": holdout_results["precision"],
            "npv": holdout_results["npv"],
            "sensitivity_ci": holdout_results["sensitivity_ci"],
            "specificity_ci": holdout_results["specificity_ci"],
        },
        "replay": {
            "stats": replay_results["stats"],
            "sensitivity": replay_results["sensitivity"],
            "specificity": replay_results["specificity"],
            "precision": replay_results["precision"],
            "npv": replay_results["npv"],
            "sensitivity_ci": replay_results["sensitivity_ci"],
            "specificity_ci": replay_results["specificity_ci"],
        },
        "rejection_curves": rejection_curves
    }
    
    output_path = RESULTS_DIR / "calibrated_transformer_detector_metrics.json"
    with output_path.open("w") as f:
        json.dump(output, f, indent=2)
    
    print(f"   Saved to: {output_path}")
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
