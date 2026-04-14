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
└── requirements.txt
```

This repository contains code only. Data and pre-computed results are not distributed.
Real-world patient messages contain de-identified but privacy-sensitive text from
Medicaid care coordination and are available under a data use agreement
(contact: sanjay.basu@waymarkcare.com). Physician-created test scenarios (200 items
used for training/evaluation) are available on request from the corresponding author.

---

## Ten configurations evaluated (real-world test set, n=2,000, 8.25% hazard prevalence)

| Configuration | Sensitivity (95% CI) | Specificity (95% CI) | F1 | AUROC | Action acc |
|:---|:---:|:---:|:---:|:---:|:---:|
| CQL controller (sensitivity-opt.) | 0.727 (0.655–0.789) | 0.728 (0.707–0.748) | 0.306 | 0.731 | —† |
| CQL controller (reward-opt.) | 0.642 (0.568–0.712) | 0.702 (0.681–0.723) | 0.259 | 0.717 | 63.1% |
| Constellation architecture | 0.685 (0.610–0.751) | 0.777 (0.756–0.795) | 0.328 | 0.801 | 60.9% |
| Rule-based guardrails | 0.600 (0.527–0.671) | 0.854 (0.837–0.870) | 0.373 | 0.801 | 69.7% |
| XGBoost + sentence embeddings | 0.430 (0.357–0.507) | 0.908 (0.894–0.920) | 0.351 | 0.760 | 70.0% |
| Logistic regression + TF-IDF | 0.394 (0.323–0.471) | 0.882 (0.867–0.896) | 0.291 | 0.716 | 68.9% |
| GPT-5.1 (safety-augmented prompt) | 0.400 (0.328–0.476) | 0.901 (0.887–0.914) | 0.320 | 0.651 | 16.4% |
| GPT-5.1 (default prompt) | 0.279 (0.216–0.352) | 0.954 (0.944–0.963) | 0.312 | 0.617 | 15.6% |
| Fine-tuned Llama-1.1B | 0.376 (0.305–0.452) | 0.774 (0.755–0.793) | 0.193 | 0.701 | 69.6% |
| DeepSeek-R1 (local Ollama, safety prompt) | 0.224 (0.163–0.290) | 0.866 (0.851–0.882) | 0.165 | 0.545‡ | 41.4% |

†Action appropriateness for CQL hazard-detection controller rows reflects the binary hazard-to-action mapping (63.1% for reward-opt.); the dedicated ActionHead action classifier achieves **77.7%** (95% CI 75.8–79.5%).

‡DeepSeek-R1 AUROC is a binary (single-threshold) estimate: 0.5 × (sensitivity + specificity); no calibrated probability scores are available, so a full ROC curve cannot be computed.

DeepSeek-R1 was also evaluated on a matched n=41 physician holdout (same stratified split as the nine primary configurations; sensitivity reported after re-evaluation — see `code/analysis/run_deepseek_physician_holdout.py`).

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

### 7. DeepSeek-R1 on matched physician holdout

Requires Ollama running locally with `deepseek-r1:8b` pulled.

```bash
ollama pull deepseek-r1:8b
python code/analysis/run_deepseek_physician_holdout.py \
    --data-dir /path/to/physician_test \
    --out-dir results/
```

---

## Data availability

Data are not distributed in this repository.

- **Real-world patient messages**: Available under a data use agreement (contact: sanjay.basu@waymarkcare.com).
- **Physician-created test scenarios**: Available on request from the corresponding author.
- **Hazard category taxonomy**: See `code/detectors/` for the canonical 23-category `HAZARD_CATEGORIES` list.

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
