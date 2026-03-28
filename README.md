# rl-llm-medical-safety

Reproducibility package for:

> Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Morgan J, Batniji R.
> Comparative evaluation of AI architectures for medical triage safety: a real-world
> validation study. *JMIR Medical Informatics*. 2026 [under review]. ms#94081.

IRB: WCG IRB tracking ID 20253751 (determined exempt).

---

## Repository layout

```
.
├── code/
│   ├── detectors/        # Hazard detection model training and inference
│   ├── controllers/      # Conservative Q-learning controller
│   ├── baselines/        # LLM baseline evaluation wrappers
│   └── analysis/         # End-to-end evaluation scripts and figure generation
├── data/
│   └── physician_test/   # Physician-created test scenarios (public)
├── results/
│   ├── *.csv             # Aggregated metrics (primary and supplementary tables)
│   └── predictions/      # Per-message predictions (message ID + label only)
└── submission/
    └── revision_v3/      # Final submission documents (markdown)
```

Real-world patient messages are not included. They contain de-identified but
privacy-sensitive text from Medicaid care coordination and are available under a
data use agreement (contact: sanjay.basu@waymarkcare.com).

---

## Nine configurations evaluated (real-world test set, n=2,000, 8.25% hazard prevalence)

| Configuration | Sensitivity (95% CI) | Specificity (95% CI) | F1 | AUROC | Action acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| CQL controller (sensitivity-opt.) | 0.727 (0.655–0.789) | 0.728 (0.707–0.748) | 0.306 | 0.731 | — |
| CQL controller (reward-opt.) | 0.642 (0.568–0.712) | 0.702 (0.681–0.723) | 0.259 | 0.717 | 63.1% |
| Constellation architecture | 0.685 (0.610–0.751) | 0.777 (0.756–0.795) | 0.328 | 0.801 | 60.9% |
| Rule-based guardrails | 0.600 (0.527–0.671) | 0.854 (0.837–0.870) | 0.373 | 0.801 | 69.7% |
| XGBoost + sentence embeddings | 0.430 (0.357–0.507) | 0.908 (0.894–0.920) | 0.351 | 0.760 | 70.0% |
| Logistic regression + TF-IDF | 0.394 (0.323–0.471) | 0.882 (0.867–0.896) | 0.291 | 0.716 | 68.9% |
| GPT-5.1 (safety-augmented prompt) | 0.400 (0.328–0.476) | 0.901 (0.887–0.914) | 0.320 | 0.651 | 16.4% |
| GPT-5.1 (default prompt) | 0.279 (0.216–0.352) | 0.954 (0.944–0.963) | 0.312 | 0.617 | 15.6% |
| Fine-tuned Llama-1.1B | 0.376 (0.305–0.452) | 0.774 (0.755–0.793) | 0.193 | 0.701 | 69.6% |

Action appropriateness (ActionHead, purpose-trained 9-class classifier): **77.7%** (95% CI 75.8–79.5%).

DeepSeek-R1: evaluated as an additional analysis on the complete 2,000-message test set;
results in `results/predictions/predictions_deepseek_full2000.csv` and `results/action_metrics_all_final.csv`.

Authoritative source: `results/architecture_eval_metrics_VERIFIED_final.csv`

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.9+ required. GPU recommended for TinyLlama fine-tuning; all other scripts
run on CPU.

---

## Reproducing results

### 1. Local supervised models

```bash
python code/analysis/run_round2_local_only.py \
    --data-dir /path/to/realworld_test \
    --out-dir results/
```

Reproduces: guardrails, constellation, logistic regression, XGBoost predictions.

### 2. CQL decision-theoretic controller

```bash
python code/analysis/run_round2_decision_policy.py \
    --data-dir /path/to/realworld_test \
    --out-dir results/
```

### 3. LLM evaluations (GPT-5.1)

```bash
export OPENAI_API_KEY=<key>
python code/analysis/run_round2_gpt_generation.py \
    --data-dir /path/to/realworld_test \
    --out-dir results/
```

### 4. Fine-tuned Llama-1.1B (TinyLlama, MPS/CUDA)

```bash
python code/analysis/run_tinyllama_mps_finetune.py
```

### 5. Compile action metrics across all configurations

```bash
python code/analysis/compile_action_metrics.py
```

Output: `results/action_metrics_all_final.csv`

### 6. Figures

```bash
python code/analysis/make_figures.py
```

---

## Key result files

| File | Description |
|:---|:---|
| `results/architecture_eval_metrics_VERIFIED_final.csv` | Primary — sensitivity, specificity, F1, MCC, AUROC with bootstrap 95% CIs for all 9 configurations |
| `results/action_metrics_all_final.csv` | Action appropriateness, under-triage, and over-triage rates for all configurations |
| `results/hazard_strat_round2.csv` | Sensitivity by hazard category (18 categories) |
| `results/fairness_demographics_round2.csv` | Performance by demographic subgroup |
| `results/op_points_local_plus_cql.csv` | Operating point curves (sensitivity vs specificity) |
| `results/fewshot_subset_summary.csv` | Few-shot GPT-5.1 operating point analysis (500-message subset) |
| `results/predictions/predictions_long_new.csv` | Per-message binary predictions and calibrated probabilities (6 local models + CQL + TinyLlama) |
| `results/predictions/predictions_actions_local_with_truth.csv` | Per-message action predictions with ground truth (local classifiers) |
| `results/predictions/predictions_actions_llm_with_truth.csv` | Per-message action predictions with ground truth (LLM configurations) |
| `results/predictions/predictions_deepseek_full2000.csv` | DeepSeek-R1 predictions on the full 2,000-message test set |

---

## Data availability

### Included (public)

`data/physician_test/` contains physician-created triage scenarios:

| File | Description |
|:---|:---|
| `hazard_scenarios_holdout.json` | 189 held-out hazard scenarios used for evaluation |
| `hazard_scenarios_train.json` | 811 training hazard scenarios (not in manuscript test tables) |
| `benign_scenarios.json` | 500 benign scenarios (200 used for physician test set) |
| `scenario_library.csv` | Hazard category taxonomy (18 categories across 5 domains) |

### Not included (restricted)

Real-world patient messages from the Medicaid care coordination program are not
included due to privacy constraints. They are available to qualified researchers
under a data use agreement. Contact: sanjay.basu@waymarkcare.com.

---

## Citation

```bibtex
@article{basu2026triage,
  author  = {Basu, Sanjay and Patel, Sadiq Y. and Sheth, Parth and
             Muralidharan, Bhairavi and Elamaran, Namrata and Kinra, Aakriti and
             Morgan, John and Batniji, Rajaie},
  title   = {Comparative evaluation of {AI} architectures for medical triage safety:
             a real-world validation study},
  journal = {JMIR Medical Informatics},
  year    = {2026},
  note    = {Under review, ms\#94081}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
