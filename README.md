Clean Replication Bundle (Round 2)
===================================

This bundle contains the PHI-free artifacts and scripts needed to audit or reproduce the round‑2 safety evaluation.

Contents
--------
- code/
  - run_round2_local_only.py — trains/evaluates local baselines (guardrail, constellation, XGBoost/LogReg).
  - run_round2_decision_policy.py — trains the decision-theoretic controller (CQL variants).
  - run_round2_gpt_generation.py — runs GPT-5.1 (null/safety) and few-shot prompts; expects OPENAI_API_KEY.
  - run_tinyllama_mps_finetune.py — fine-tunes TinyLlama 1.1B on MPS (Apple Silicon) for the fairness experiment.
  - generate_repro_predictions.py — merges per-message predictions into long-format tables.
  - make_figures.py — generates manuscript figures from verified CSVs.
- results/repro_round2/
  - architecture_eval_metrics_VERIFIED_final.csv — primary metrics (sens/spec/F1/MCC/AUROC with 95% CIs).
  - predictions_long_new.csv — per-message predictions/probabilities for all local models and tinyllama.
  - op_points_local_plus_cql.csv, op_points_all.csv — operating-point sensitivity at fixed specificity.
  - fairness_demographics_round2.csv — subgroup fairness metrics by sex/race/ethnicity.
  - hazard_strat_round2.csv — performance stratified by hazard category.
  - fewshot_subset_summary.csv, fewshot_subset_README.txt — summary of few-shot experiments (subset n=500).
  - tinyllama_metrics.csv, tinyllama_predictions.csv — TinyLlama fine-tune results.
- results/llm_responses_round2/
  - fewshot_subset500.csv — evaluation subset used for few-shot runs (stratified).
  - gpt5.1_fewshot{0,5,10}_subset500.jsonl + *_metrics.csv — GPT-5.1 outputs/metrics for 0/5/10-shot.
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
- TinyLlama weights are NOT included to keep size manageable; rerun step 4 to regenerate.
- For full per-message audit, use `predictions_long_new.csv` (local/tinyllama) and the GPT JSONL files.

Contact: Please set your own API keys; none are hard-coded in this bundle.
