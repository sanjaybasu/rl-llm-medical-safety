# rl-llm-medical-safety

Reproducibility package for:

> Basu S, Patel SY, Sheth P, Muralidharan B, Elamaran N, Kinra A, Morgan J, Batniji R.
> Comparative evaluation of AI architectures for medical triage safety: a real-world
> validation study. 2026.

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

Requires Ollama running locally with `deepseek-r1:7b` pulled.

```bash
ollama pull deepseek-r1:7b
python code/analysis/run_deepseek_physician_holdout.py \
    --data-dir /path/to/physician_test \
    --out-dir results/
```

---

## Data availability

Data are not distributed in this repository.

- **Real-world patient messages**: Not distributed. These are de-identified but privacy-sensitive Medicaid care coordination messages; distribution is not possible under applicable data governance requirements.
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
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
