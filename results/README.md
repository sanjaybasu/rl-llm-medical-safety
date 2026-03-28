# Results

All CSV files in this directory are produced by scripts in `code/analysis/`
and correspond to tables and figures in the manuscript and appendix.

## Primary metrics

### `architecture_eval_metrics_VERIFIED_final.csv`

**Primary result file.** Sensitivity, specificity, F1, MCC, and AUROC with
bootstrap 95% CIs for all nine configurations on the real-world 2,000-message
test set (8.25% hazard prevalence, 165 hazardous messages) and physician-created
test set (N=389, 48.6% hazard prevalence). Corresponds to manuscript Table 2.

Columns: `dataset, system, sensitivity, sensitivity_ci_lower, sensitivity_ci_upper,
specificity, specificity_ci_lower, specificity_ci_upper, f1, mcc, auroc,
n_sample, n_hazard, n_safe`

### `action_metrics_all_final.csv`

Action appropriateness, under-triage rate, and over-triage rate for all
configurations plus DeepSeek-R1. Corresponds to manuscript Table 4.

Columns: `system, n, action_accuracy, under_rate, over_rate,
action_acc_ci_lower, action_acc_ci_upper, note`

## Supplementary metrics

### `hazard_strat_round2.csv`

Sensitivity by hazard category (18 categories) for all six configurations with
calibrated probability scores. Corresponds to Appendix Table S3.

### `fairness_demographics_round2.csv`

Performance by demographic subgroup (sex, age group) for the CQL
sensitivity-optimized configuration. Corresponds to Appendix Table S4.

### `op_points_local_plus_cql.csv`

Sensitivity-specificity operating point curves across probability thresholds
for local supervised models and the CQL controller. Used for Figure 1.

### `fewshot_subset_summary.csv`

GPT-5.1 few-shot operating point analysis (0-shot, 5-shot, 10-shot) on the
500-message stratified subset. Corresponds to manuscript §Few-Shot Prompting.

## Per-message predictions (`predictions/`)

Per-message prediction files reference real-world patient message IDs and are
not distributed in this repository. Aggregated metrics above are sufficient
to reproduce all manuscript tables and figures.

To replicate per-message predictions from source, point the analysis scripts
to your local copy of the real-world dataset (available under a data use
agreement; contact sanjay.basu@waymarkcare.com).
