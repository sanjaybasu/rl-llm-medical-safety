# Safety Hazard Detection in Medical Triage
## A Testing and Validation Study of Large Language Models, Guardrails, and Decision-Theoretic Controllers

Sanjay Basu, Sadiq Patel, John Morgan, Rajaie Batniji


## Repository Contents

```
rl_llm_safety_github/
├── data/                      # All datasets used in the study
│   ├── physician_created/     # Physician-authored test scenarios
│   ├── real_world/           # De-identified real-world patient data
├── code/                      # All analysis scripts
│   ├── detectors/            # Hazard detection models
│   ├── controllers/          # RL controller training
│   ├── baselines/            # LLM baseline evaluations
│   └── analysis/             # Statistical analysis and figures
├── results/                   # Generated outputs (tables, figures, metrics)
└── README.md                  # This file
```

## Datasets

### Physician-Created Test Set (N=389)
- **Location**: `data/physician_created/`
- **Files**:
  - `hazard_scenarios_train.json` -  physician-authored hazard scenarios
  - `benign_scenarios.json` -  physician-authored benign scenarios
- **Description**: Systematically sampled from emergency triage frameworks (ESI, MTS, CTAS) and safety databases

### Real-World Validation Set (N=1,000)
- **Location**: `data/real_world/`
- **Files**:
  - `replay_scenarios_llm_labels.json` - 1,000 de-identified patient messages
  - `prospective_eval/harm_cases_500.csv` - 500 documented hazard cases
  - `prospective_eval/benign_cases_500.csv` - 500 benign cases
- **Description**: De-identified patient messages from Medicaid population health programs
- **De-identification**: All data de-identified per HIPAA Safe Harbor method

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

# Evaluate on real-world hold-out
python code/controllers/final_realworld_eval.py
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


## Code Structure

### Detectors (`code/detectors/`)
- `train_calibrated_detector.py` - Train and calibrate hazard detector
- `hazard_detection.py` - Core detection utilities

### Controllers (`code/controllers/`)
- `train_cql_calibrations.py` - Conservative Q-Learning training
- `final_realworld_eval.py` - Real-world validation evaluation

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

## Data Availability

All de-identified datasets and analysis code are publicly available at:
**https://github.com/sanjaybasu/rl-llm-medical-safety**

## Ethics and Privacy

- All real-world data de-identified per HIPAA Safe Harbor method (45 CFR §164.514(b)(2))
- Study deemed exempt from IRB review (analysis of de-identified data)

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues:
- Sanjay Basu: sanjay.basu@waymarkcare.com
- GitHub Issues: https://github.com/sanjaybasu/rl-llm-medical-safety/issues
