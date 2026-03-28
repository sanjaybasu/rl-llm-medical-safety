# Code

## Prerequisites

```bash
pip install -r ../requirements.txt
```

For DeepSeek-R1 local inference: install [Ollama](https://ollama.ai) and pull
`deepseek-r1:latest`.

Set the environment variable `REALWORLD_DATA_DIR` to the path of your local
copy of the real-world messages before running scripts that require it.

---

## detectors/

| File | Description |
|:---|:---|
| `train_calibrated_detector.py` | Train logistic regression hazard detector on sentence-BERT embeddings; apply temperature scaling calibration |
| `hazard_detection.py` | Inference utilities for the trained detector |

**Training data:** `data/physician_test/hazard_scenarios_train.json` (811 hazard)
+ `data/physician_test/benign_scenarios.json` (500 benign). Temperature scaling
parameter T=0.548 is fit on a held-out calibration split.

---

## controllers/

| File | Description |
|:---|:---|
| `rl_core.py` | CQL (Conservative Q-Learning) implementation: Q-network, replay buffer, training loop |
| `train_cql_calibrations.py` | Train the CQL controller on hazard probability vectors; reward function R(+10, -50, -2) |
| `eval_cql_realworld.py` | Evaluate trained CQL controller on the real-world test set |

**CQL reward function:**
- +10: correct escalation (hazard detected, appropriate action)
- -50: missed hazard (hazard present, action insufficient)
- -2: unnecessary escalation (benign, action above self-care)

The 25:1 penalty ratio reflects estimated clinical harm asymmetry between
missed emergencies and alert fatigue in Medicaid care coordination.

---

## baselines/

| File | Description |
|:---|:---|
| `llm_openai.py` | OpenAI API wrapper (GPT-5.1); temperature not user-configurable; returns structured action and hazard prediction |
| `llm_anthropic.py` | Anthropic API wrapper (Claude) for baseline comparisons |
| `evaluate_llm_safety.py` | Run LLM baselines on physician or real-world test sets; log structured outputs |

---

## analysis/

| File | Description |
|:---|:---|
| `run_round2_local_only.py` | Evaluate guardrails, constellation, logistic regression, XGBoost on real-world test set |
| `run_round2_decision_policy.py` | Evaluate CQL controller (both threshold variants) on real-world test set |
| `run_round2_gpt_generation.py` | Evaluate GPT-5.1 (default and safety-augmented) on real-world test set |
| `run_tinyllama_mps_finetune.py` | Fine-tune and evaluate TinyLlama (Llama-1.1B) on real-world test set |
| `compile_action_metrics.py` | Aggregate action appropriateness metrics across all configurations; write `results/action_metrics_all_final.csv` |
| `make_figures.py` | Generate all manuscript figures (Figures 1–4, Supplementary Figures S1–S6) |
| `finalize_deepseek_full2000.py` | Compute DeepSeek-R1 metrics from `predictions_deepseek_full2000.csv` |

---

## Reproducing the full pipeline

```bash
# 1. Train detector
python detectors/train_calibrated_detector.py

# 2. Train CQL controller
python controllers/train_cql_calibrations.py

# 3. Evaluate all configurations (requires REALWORLD_DATA_DIR)
python analysis/run_round2_local_only.py
python analysis/run_round2_decision_policy.py
python analysis/run_round2_gpt_generation.py

# 4. Fine-tune TinyLlama (requires GPU)
python analysis/run_tinyllama_mps_finetune.py

# 5. Compile action metrics
python analysis/compile_action_metrics.py

# 6. Generate figures
python analysis/make_figures.py
```

To reproduce results without the real-world dataset, use the pre-computed
prediction files in `results/predictions/` directly with step 5 and 6.
