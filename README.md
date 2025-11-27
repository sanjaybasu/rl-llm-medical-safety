# Safety Hazard Detection in Medical Triage: A Testing and Validation Study of Large Language Models, Guardrails, and Decision-Theoretic Controllers


Sanjay Basu, Sadiq Patel, John Morgan, Rajaie Batniji


## Repository Contents

```
rl_llm_safety_github/
├── data/                      # All datasets used in the study
├── code/                      # All analysis scripts
│   ├── detectors/             # Hazard detection models
│   ├── controllers/           # RL controller training (CQL + AWR)
│   ├── baselines/             # LLM baseline evaluations
│   └── analysis/              # Statistical analysis and figures
└── README.md                  # This file
```

## Datasets

- **Location**: `data/physician_created/`
- **Files**:
  - `hazard_scenarios_holdout.json` - 189 held-out hazard scenarios (used for manuscript evaluation)
  - `benign_scenarios.json` - 500 benign scenarios (first 200 used for manuscript evaluation)
  - `hazard_scenarios_train.json` - 811 training hazard scenarios (for detector/controller training; not used in manuscript tables)
  - `hazard_scenarios_extended.json` - 108 augmented scenarios (supplementary training data)
- **Description**: Systematically sampled from emergency triage frameworks (ESI, MTS, CTAS) and safety databases. The N=389 evaluation set comprises 189 holdout hazards + 200 benign scenarios.


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

### Step 1: Train Hazard Detector
```bash
python code/detectors/train_calibrated_detector.py
```
This trains the Sentence-BERT + logistic regression detector with temperature scaling on 80% of real-world data.

**Output**: Calibrated detector model with T=0.548

### Step 2: Evaluate LLM Baselines
```bash
# GPT-5
python code/baselines/evaluate_llm_safety.py --model openai_gpt_5 --dataset replay

# Claude Sonnet 4.5
python code/baselines/evaluate_llm_safety.py --model anthropic_claude_sonnet_4_5 --dataset replay
```


### Step 3: Train Controllers
```bash
# Conservative Q-Learning (CQL)
python code/controllers/train_cql_calibrations.py

# Advantage-Weighted Actor (AWR)
python code/controllers/train_awr_calibrations.py

# Evaluate on real-world hold-out (CQL + AWR)
python code/controllers/final_realworld_eval.py
python code/controllers/eval_awr_realworld.py
```


### Step 4: Generate Figures and Tables
```bash
python code/analysis/generate_detection_figures.py
python code/analysis/generate_detection_tables.py
```


### Step 5: Compute Summary Statistics
```bash
python code/analysis/compute_mcc.py
```

### Precomputed outputs and models
- Intermediate outputs are in `results/` (detector metrics, CQL/AWR evaluations, LLM reports, manuscript summary stats).
- Figures regenerate to `submission/submission_bundle/figures/`.
- Tables regenerate to `results/table1_detection_performance.csv` and `results/table2_rejection_coverage.csv`.
- Model binaries are not checked in (keep repo lightweight); training scripts are deterministic with `random.seed(42)` / `np.random.seed(42)` on Python 3.10+.


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
- `evaluate_llm_safety.py` - LLM baseline evaluations
- `llm_openai.py` - OpenAI API wrapper
- `llm_anthropic.py` - Anthropic API wrapper

### Analysis (`code/analysis/`)
- `generate_detection_figures.py` - Create all manuscript figures
- `generate_detection_tables.py` - Create all manuscript tables
- `compute_mcc.py` - Calculate Matthews Correlation Coefficient


## Citation

If you use this code or data, please cite:

```bibtex
@article{basu2025controllers,
  title={Decision-Theoretic Controllers Outperform Large Language Models and Rule-Based Guardrails for Medical Triage Safety},
  author={Basu, Sanjay and Patel, Sadiq and Morgan, John and Batniji, Rajaie},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Sanjay Basu: sanjay.basu@waymarkcare.com
- GitHub Issues: https://github.com/sanjaybasu/rl-llm-medical-safety/issues
