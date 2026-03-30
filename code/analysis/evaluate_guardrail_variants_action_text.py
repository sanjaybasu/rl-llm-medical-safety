#!/usr/bin/env python3
"""Action-text evaluation of guardrail variants on holdout and replay sets.

Uses the calibrated transformer detector for hazard labels (no confidence
threshold applied). Each guardrail responder generates text; scoring follows the
physician rubric (explicit safety actions). This avoids zeroing low-confidence
cases by always using the top detector label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

from statsmodels.stats.proportion import proportion_confint

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.extend(
    [
        str(BASE_DIR / "code" / "detectors"),
        str(BASE_DIR / "code" / "baselines"),
    ]
)

from train_calibrated_detector import TransformerHazardDetector, load_json_scenarios  # type: ignore  # noqa: E402
import evaluate_llm_safety as els  # type: ignore  # noqa: E402
from hazard_detection import DetectionResult  # type: ignore  # noqa: E402


VARIANTS = {
    "rules_engine",
    "medical_exam_prompt",
    "specialist_guardrail",
    "constitutional_prompt",
    "hippocratic_filter",
    "selective_refusal",
    "naive",
}


def wilson_ci(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    low, high = proportion_confint(successes, total, method="wilson")
    return float(low), float(high)


def evaluate_variant(scenarios, preds, responder) -> Dict:
    stats = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
    for scenario, pred_item in zip(scenarios, preds):
        label = pred_item["pred_label"]  # always use top label; no thresholding
        det_obj = DetectionResult(label=label, confidence=pred_item["confidence"], probabilities=pred_item["probabilities"])
        answer = responder(scenario, label, det_obj)
        res = els.evaluate_response(scenario, answer)
        is_pos = scenario.hazard_type != "benign"
        if is_pos:
            if res["pass"]:
                stats["tp"] += 1
            else:
                stats["fn"] += 1
        else:
            if res["pass"]:
                stats["tn"] += 1
            else:
                stats["fp"] += 1
    pos = stats["tp"] + stats["fn"]
    neg = stats["tn"] + stats["fp"]
    sens = stats["tp"] / pos if pos else 0.0
    spec = stats["tn"] / neg if neg else 0.0
    sens_ci = wilson_ci(stats["tp"], pos) if pos else (0.0, 1.0)
    spec_ci = wilson_ci(stats["tn"], neg) if neg else (0.0, 1.0)
    prec = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return {**stats, "sensitivity": sens, "sensitivity_ci": sens_ci, "specificity": spec, "specificity_ci": spec_ci, "f1": f1}


def main() -> None:
    # Load data
    train = load_json_scenarios(BASE_DIR / "data" / "physician_created" / "hazard_scenarios_train.json")
    benign = load_json_scenarios(BASE_DIR / "data" / "physician_created" / "benign_scenarios.json")
    holdout = load_json_scenarios(BASE_DIR / "data" / "physician_created" / "hazard_scenarios_holdout.json")
    replay = load_json_scenarios(BASE_DIR / "data" / "real_world" / "replay_scenarios_llm_labels.json")

    detector = TransformerHazardDetector()
    detector.fit(train + benign)

    # Predictions (no confidence threshold)
    hold_preds = detector.predict(holdout + benign, confidence_threshold=0.0)["predictions"]
    replay_preds = detector.predict(replay, confidence_threshold=0.0)["predictions"]

    results = {"holdout": {}, "replay": {}}
    for name in VARIANTS:
        responder = els.SYSTEMS[name]
        results["holdout"][name] = evaluate_variant(holdout + benign, hold_preds, responder)
        results["replay"][name] = evaluate_variant(replay, replay_preds, responder)

    out_path = BASE_DIR / "results" / "guardrail_variants_action_text_no_threshold.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[ok] wrote {out_path}")
    print("Replay rules_engine:", results["replay"]["rules_engine"])


if __name__ == "__main__":
    main()
