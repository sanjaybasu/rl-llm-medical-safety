# Comparative Evaluation of AI Architectures for Medical Triage Safety

This repository contains code and data to reproduce all results from:

**"Comparative evaluation of AI architectures for medical triage safety: a real-world validation study"**

Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, John Morgan, Rajaie Batniji


## Repository Contents

```
rl_llm_safety_github/
├── data/                      # Public datasets (physician-created + placeholders)
│   ├── physician_created/     # Physician-authored test scenarios
│   ├── real_world/            # Placeholder only (no real-world text included)
├── code/                      # All analysis scripts
│   ├── detectors/             # Hazard detection models
│   ├── controllers/           # RL controller training (CQL + AWR)
│   ├── baselines/             # LLM baseline evaluations
│   └── analysis/              # Statistical analysis and figures
├── results/                   # Generated outputs (tables, figures, metrics)
└── README.md                  # This file
```

## Datasets

### Physician-Created Test Set (N=389)
- **Location**: `data/physician_created/`
- **Files**:
  - `hazard_scenarios_holdout.json` - 189 held-out hazard scenarios (used for manuscript evaluation)
  - `benign_scenarios.json` - 500 benign scenarios (first 200 used for manuscript evaluation)
  - `hazard_scenarios_train.json` - 811 training hazard scenarios (for detector/controller training; not used in manuscript tables)
  - `hazard_scenarios_extended.json` - 108 augmented scenarios (supplementary training data)
- **Description**: Systematically sampled from emergency triage frameworks (ESI, MTS, CTAS) and safety databases. The N=389 evaluation set comprises 189 holdout hazards + 200 benign scenarios.

### Real-World Validation Sets (not included)
- **Location**: `data/real_world/` (placeholder only)
- **Files**: none (removed to avoid any residual privacy risk)
- **Access**: Request the de-identified real-world sets (1,000-message replay and 500/500 prospective splits) from the corresponding author under a DUA. Scripts should point to your private copies via env vars/config.

## Reproducing Results

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- Python ≥3.9
- numpy, pandas, scikit-learn
- sentence-transformers (all-MiniLM-L6-v2)
- openai, anthropic (for LLM baselines)
- matplotlib, seaborn (for figures)

### Reproduction notes
- Physician-created scenarios are fully included; real-world text is not. The latest cross-architecture metrics (held-out tests) are stored in `results/architecture_eval_metrics.csv`.
- To rerun analyses without PHI, operate on `data/physician_created/` only. For full reproduction including real-world, point scripts to your private copies of the datasets via environment variables (not distributed here).

### Precomputed outputs
- `results/architecture_eval_metrics.csv` – current cross-architecture metrics (physician + real-world splits; real-world computed offline).
No other result artifacts are retained to avoid confusion with older analyses.


## Code Structure

### Detectors (`code/detectors/`)
- `train_calibrated_detector.py` - Train and calibrate hazard detector
- `hazard_detection.py` - Core detection utilities

### Controllers (`code/controllers/`)
- `train_cql_calibrations.py` - Conservative Q-Learning training
- `train_awr_calibrations.py` - Advantage-Weighted controller training
- `final_realworld_eval.py` - Real-world validation evaluation (CQL)
- `eval_awr_realworld.py` - Real-world validation evaluation (AWR)

### Baselines (`code/baselines/`)
- `evaluate_llm_safety.py` - LLM baseline evaluations (physician set)
- `llm_openai.py` - OpenAI API wrapper
- `llm_anthropic.py` - Anthropic API wrapper

### Analysis (`code/analysis/`)
- `generate_detection_figures.py` - Create all manuscript figures
- `generate_detection_tables.py` - Create all manuscript tables
- `compute_mcc.py` - Calculate Matthews Correlation Coefficient


## Citation

If you use this code or data, please cite:

```bibtex
@article{basu2026triage,
  title={Comparative evaluation of AI architectures for medical triage safety: a real-world validation study},
  author={Basu, Sanjay and Patel, Sadiq Y. and Sheth, Parth and Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and Morgan, John and Batniji, Rajaie},
  journal={JMIR Formative Research},
  year={2026},
  note={Under review}
}
```

### Key verified results (real-world test set, n=2,000, 8.25% hazard prevalence)

| Architecture | Sensitivity | Specificity | F1 | AUROC |
|---|---|---|---|---|
| Decision-theoretic controller (CQL) | 0.727 | 0.728 | 0.306 | 0.731 |
| Constellation architecture | 0.685 | 0.777 | 0.329 | 0.801 |
| Rule-based guardrails | 0.600 | 0.854 | 0.372 | 0.801 |
| XGBoost + sentence embeddings | 0.430 | 0.908 | 0.351 | 0.760 |
| Logistic regression + TF-IDF | 0.394 | 0.882 | 0.291 | 0.716 |
| DeepSeek-R1 (base) | 0.297 | 0.945 | 0.311 | 0.689 |
| DeepSeek-R1 (RAG) | 0.327 | 0.943 | 0.334 | 0.704 |
| GPT-5.1 (base; safety-prompted) | 0.400 | 0.901 | 0.320 | 0.651 |
| GPT-5.1 (RAG) | 0.400 | 0.901 | 0.320 | 0.651 |

Authoritative verified results: `results/repro_round2/architecture_eval_metrics_VERIFIED_final.csv`

## Data Availability
- Physician-created scenarios: included.
- Real-world datasets: not included; available upon request with DUA.

## Ethics and Privacy

- All real-world data de-identified per HIPAA Safe Harbor method (45 CFR §164.514(b)(2))
- Study protocol deemed exempt by WCG IRB (Princeton, NJ; tracking ID: 20253751)

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Sanjay Basu: sanjay.basu@waymarkcare.com
- GitHub Issues: https://github.com/sanjaybasu/rl-llm-medical-safety/issues

## Changelog

### 2026-03-24 — Revision for JMIR Formative Research
- Updated title (removed superlative framing per JMIR editorial guidelines)
- Added full author list
- Added verified results table; all F1 values corrected (were previously underreported by ~50%)
- Added IRB exemption details
- Citation bibtex updated to reflect journal under review
