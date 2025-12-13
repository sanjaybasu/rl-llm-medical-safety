Numbered analysis scripts (copied from `notebooks/rl_vs_llm_safety_v2`) for full replication of the main text and appendix analyses. No API keys are embedded; supply your own environment variables for any external LLM calls (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`).

1. `01_data_preparation.py` – Load/clean datasets, build train/test splits.
2. `02_train_models.py` – Train local models (CQL controller, constellation, guardrails, classical ML).
3. `03_evaluate_physician.py` – Evaluate on physician scenarios.
4. `04_evaluate_realworld.py` – Evaluate on real-world messages.
5. `05_generate_audit_physician.py` – Create physician audit CSVs.
6. `06_generate_audit_realworld.py` – Create real-world audit CSVs.
7. `07_generate_figures.py` – Produce figures for manuscript/appendix.
8. `08_subgroup_analysis.py` – Stratified analyses (severity, demographics).
9. `pipeline_architecture_eval.py` – End-to-end evaluation pipeline (orchestrates model runs, external LLM calls via env vars).
10. `generate_repro_predictions.py` – Generates reproducible prediction files across architectures.
11. `generate_unified_predictions.py` – Consolidates predictions for downstream metrics/tables.
12. `run_round2_decision_policy.py` – Offline RL/decision-policy training and evaluation.
13. `run_round2_gpt_generation.py` – LLM generation workflow (uses API keys via env vars).
14. `run_tinyllama_mps_finetune.py` – Fine-tune Llama/TinyLlama for task-specific adaptation.
15. `run_llama3_8b_finetune.py` – Alternative fine-tuning script for larger LLaMA variant.
16. `complete_all_evaluations.py` – Convenience orchestrator that chains the above steps.
