#!/usr/bin/env python3
"""
Final verification script for GitHub repository completeness.
Ensures all datasets, code, and results match manuscript claims.
"""

import json
import csv
import os
from pathlib import Path

def verify_github_repo():
    """Verify complete GitHub repository structure."""
    base_path = Path(__file__).parent
    
    print("="*80)
    print("GITHUB REPOSITORY VERIFICATION")
    print("="*80)
    print()
    
    errors = []
    warnings = []
    
    # 1. Check directory structure
    print("1. Directory Structure")
    print("-" * 40)
    required_dirs = [
        "data/physician_created",
        "data/real_world/prospective_eval",
        "code/detectors",
        "code/controllers",
        "code/baselines",
        "code/analysis",
        "results",
        "docs"
    ]
    
    for d in required_dirs:
        dpath = base_path / d
        if not dpath.exists():
            errors.append(f"Missing directory: {d}")
        else:
            print(f"  ✓ {d}/")
    print()
    
    # 2. Check datasets
    print("2. Datasets")
    print("-" * 40)
    
    datasets = {
        "data/physician_created/hazard_scenarios_train.json": ("Physician hazard scenarios", 811),
        "data/physician_created/benign_scenarios.json": ("Physician benign scenarios", 500),
        "data/real_world/replay_scenarios_llm_labels.json": ("Real-world validation set", 1000),
        "data/real_world/prospective_eval/harm_cases_500.csv": ("Prospective harm cases", 500),
        "data/real_world/prospective_eval/benign_cases_500.csv": ("Prospective benign cases", 500),
    }
    
    for fpath, (desc, expected_count) in datasets.items():
        full_path = base_path / fpath
        if not full_path.exists():
            errors.append(f"Missing dataset: {fpath}")
            continue
            
        # Count records
        if fpath.endswith('.json'):
            with open(full_path) as f:
                data = json.load(f)
                count = len(data)
        elif fpath.endswith('.csv'):
            with open(full_path) as f:
                count = sum(1 for _ in csv.DictReader(f))
        
        status = "✓" if count == expected_count else "⚠"
        print(f"  {status} {desc}: {count} records (expected: {expected_count})")
        
        if count != expected_count:
            warnings.append(f"{desc}: {count} != {expected_count}")
    print()
    
    #3. Check code files
    print("3. Analysis Code")
    print("-" * 40)
    
    code_files = {
        "code/detectors/train_calibrated_detector.py": "Train hazard detector",
        "code/detectors/hazard_detection.py": "Detection utilities",
        "code/controllers/train_cql_calibrations.py": "CQL controller training",
        "code/controllers/final_realworld_eval.py": "Real-world evaluation",
        "code/baselines/evaluate_llm_safety.py": "LLM baseline evaluation",
        "code/baselines/llm_openai.py": "OpenAI wrapper",
        "code/baselines/llm_anthropic.py": "Anthropic wrapper",
        "code/analysis/generate_detection_figures.py": "Generate figures",
        "code/analysis/generate_detection_tables.py": "Generate tables",
        "code/analysis/compute_mcc.py": "Compute MCC metrics",
        "code/verify_datasets_for_github.py": "Dataset verification",
    }
    
    for fpath, desc in code_files.items():
        full_path = base_path / fpath
        if not full_path.exists():
            errors.append(f"Missing code: {fpath}")
        else:
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {desc}: {fpath.split('/')[-1]} ({size_kb:.1f} KB)")
    print()
    
    # 4. Check key results
    print("4. Results Files")
    print("-" * 40)
    
    results = {
        "results/cql_realworld_final.json": "Controller real-world performance",
        "results/calibrated_transformer_detector_metrics.json": "Guardrail performance",
        "results/anthropic_replay_labelonly_safety_llm_report.json": "Claude results",
        "results/openai_replay_labelonly_safety_llm_report.json": "GPT-5 results",
        "results/manuscript_summary_stats.json": "Summary statistics",
    }
    
    for fpath, desc in results.items():
        full_path = base_path / fpath
        if not full_path.exists():
            errors.append(f"Missing result: {fpath}")
        else:
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {desc} ({size_kb:.1f} KB)")
    print()
    
    # 5. Check documentation
    print("5. Documentation")
    print("-" * 40)
    
    docs = ["README.md", "LICENSE", "requirements.txt"]
    for doc in docs:
        dpath = base_path / doc
        if not dpath.exists():
            errors.append(f"Missing doc: {doc}")
        else:
            print(f"  ✓ {doc}")
    print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for err in errors:
            print(f"  {err}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warn in warnings:
            print(f"  {warn}")
    
    if not errors and not warnings:
        print("\n✓ ALL CHECKS PASSED!")
        print("  Repository is complete and ready for GitHub upload.")
        print("\n  Recommended repository name: rl-llm-medical-safety")
        print("  GitHub URL: https://github.com/sanjaybasu/rl-llm-medical-safety")
    
    print()
    return len(errors) == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if verify_github_repo() else 1)
