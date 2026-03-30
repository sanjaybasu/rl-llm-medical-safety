Few-shot GPT-5.1 evaluation (subset of 500 messages, 41 hazards, seed=42).

Files:
- fewshot_subset_summary.csv : sensitivity/specificity/F1/MCC/AUROC for 0/5/10-shot.
- source metrics in results/llm_responses_round2/gpt5.1_fewshot*_subset500_metrics.csv

Notes:
- LLMs remain at default thresholds (no calibration).
- Locals evaluated at matched spec in op_points_local_plus_cql.csv (spec targets 0.70/0.73/0.80/0.90).
- Subset used to control cost/time; full 2000-run not executed.
