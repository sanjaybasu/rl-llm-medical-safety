
# LLM architecture safety evaluation

Contents
--------
- code/
  - run_round2_local_only.py — trains/evaluates local baselines (guardrail, constellation, XGBoost/LogReg).
  - run_round2_decision_policy.py — trains the decision-theoretic controller (CQL variants).
  - run_round2_gpt_generation.py — runs GPT-5.1 (null/safety) and few-shot prompts; expects OPENAI_API_KEY.
  - run_tinyllama_mps_finetune.py — fine-tunes TinyLlama 1.1B on MPS (Apple Silicon) for the fairness experiment.
  - generate_repro_predictions.py — merges per-message predictions into long-format tables.
  - make_figures.py — generates manuscript figures from verified CSVs.
- requirements.txt — Python deps for the scripts above.

How to reproduce (high level)
-----------------------------
1) Install deps: `pip install -r requirements.txt`
2) Set `OPENAI_API_KEY` (for GPT runs). DeepSeek/TinyLlama runs are local.
3) Run local models: `python code/run_round2_local_only.py`
4) Train decision policy: `python code/run_round2_decision_policy.py`
5) Generate GPT outputs (batching supported): `python code/run_round2_gpt_generation.py`
6) Merge predictions: `python code/generate_repro_predictions.py`
7) Create figures/tables: `python code/make_figures.py`

Notes
-----
- No PHI is included. Message texts were excluded; only model outputs/metrics are provided.
- For full per-message audit, use `predictions_long_new.csv` (local/tinyllama) and the GPT JSONL files.

Please set your own API keys; none are hard-coded in this bundle.
