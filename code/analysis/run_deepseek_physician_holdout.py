#!/usr/bin/env python3
"""
Run DeepSeek-R1 on the 43-item physician holdout subset (v2 split).
This resolves Editor Comment 8: re-evaluate DeepSeek on same holdout used for 9 primary architectures.
"""
import json
import subprocess
import csv
import sys
import math

SAFETY_PROMPT = (
    "You are a medical triage assistant. Your primary responsibility is patient safety. "
    "Review the following patient message and determine: "
    "1. Does this message contain a clinical safety hazard requiring urgent escalation? "
    "Please prioritize patient safety - when uncertain, err toward identifying a potential hazard. "
    "Respond with HAZARD if this message requires urgent medical attention or escalation, "
    "or SAFE if this is a routine non-urgent communication. "
    "Begin your response with HAZARD or SAFE."
)

def call_ollama(message: str, model: str = "deepseek-r1:8b") -> str:
    prompt = f"{SAFETY_PROMPT}\n\nPatient message: {message}\n\nYour assessment:"
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip()

def parse_response(response: str) -> int:
    """Return 1 if HAZARD detected, 0 if SAFE."""
    resp_upper = response.upper()
    # Check first 50 chars for the classification
    first_part = resp_upper[:50]
    if "HAZARD" in first_part:
        return 1
    elif "SAFE" in first_part:
        return 0
    # Fall back to full text
    if "HAZARD" in resp_upper and "SAFE" not in resp_upper[:resp_upper.index("HAZARD")]:
        return 1
    return 0  # default to SAFE if ambiguous

def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = (z / denom) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return max(0, center - margin), min(1, center + margin)

def main():
    with open('/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/archive_cleanup/old_data/data_final_v2/splits/physician_test.json') as f:
        holdout = json.load(f)

    print(f"Loaded {len(holdout)} holdout items")

    results = []
    for i, item in enumerate(holdout):
        message = item.get('message', item.get('prompt', ''))
        true_label = int(item.get('detection_truth', 0))
        name = item.get('name', f'item_{i}')

        print(f"[{i+1}/{len(holdout)}] {name[:50]} (true: {'HAZARD' if true_label else 'SAFE'})", end='... ', flush=True)

        response = call_ollama(message)
        pred = parse_response(response)

        results.append({
            'name': name,
            'true_label': true_label,
            'pred_label': pred,
            'response_snippet': response[:100]
        })
        print(f"pred={'HAZARD' if pred else 'SAFE'}")

    # Compute metrics
    n = len(results)
    n_hazard = sum(r['true_label'] for r in results)
    n_safe = n - n_hazard
    tp = sum(1 for r in results if r['true_label'] == 1 and r['pred_label'] == 1)
    tn = sum(1 for r in results if r['true_label'] == 0 and r['pred_label'] == 0)
    fp = sum(1 for r in results if r['true_label'] == 0 and r['pred_label'] == 1)
    fn = sum(1 for r in results if r['true_label'] == 1 and r['pred_label'] == 0)

    sensitivity = tp / n_hazard if n_hazard > 0 else 0
    specificity = tn / n_safe if n_safe > 0 else 0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0
    mcc_num = tp*tn - fp*fn
    mcc_denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = mcc_num / mcc_denom

    sens_lo, sens_hi = wilson_ci(tp, n_hazard)
    spec_lo, spec_hi = wilson_ci(tn, n_safe)

    print("\n" + "="*60)
    print(f"RESULTS: DeepSeek-R1 on Physician Holdout (n={n})")
    print(f"  n_hazard={n_hazard}, n_safe={n_safe}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Sensitivity: {sensitivity:.3f} (95% CI {sens_lo:.3f}-{sens_hi:.3f})")
    print(f"  Specificity: {specificity:.3f} (95% CI {spec_lo:.3f}-{spec_hi:.3f})")
    print(f"  F1: {f1:.3f}")
    print(f"  MCC: {mcc:.3f}")
    print("="*60)

    # Save results
    out_dir = '/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v2/final_submission/jmir_mi_revision'
    with open(f'{out_dir}/deepseek_physician_holdout43_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','true_label','pred_label','response_snippet'])
        writer.writeheader()
        writer.writerows(results)

    metrics = {
        'dataset': 'Physician_Holdout_43',
        'system': 'DeepSeek-R1_safety',
        'n_sample': n, 'n_hazard': n_hazard, 'n_safe': n_safe,
        'sensitivity': round(sensitivity, 6),
        'sensitivity_ci_lower': round(sens_lo, 6),
        'sensitivity_ci_upper': round(sens_hi, 6),
        'specificity': round(specificity, 6),
        'specificity_ci_lower': round(spec_lo, 6),
        'specificity_ci_upper': round(spec_hi, 6),
        'f1': round(f1, 6), 'mcc': round(mcc, 6),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }
    with open(f'{out_dir}/deepseek_physician_holdout43_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    print(f"\nResults saved to {out_dir}/")
    return metrics

if __name__ == '__main__':
    main()
