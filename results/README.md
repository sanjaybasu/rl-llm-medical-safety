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

These files contain message IDs and model predictions only; no message text
is included.

| File | Systems | N rows | Description |
|:---|:---|:---:|:---|
| `predictions_long_new.csv` | 6 local + CQL + TinyLlama | 12,000 | Binary predictions and calibrated probabilities |
| `predictions_actions_local_with_truth.csv` | 4 local classifiers | 8,000 | Action predictions with clinician ground truth |
| `predictions_actions_llm_with_truth.csv` | GPT-5.1 variants | varies | LLM action predictions with ground truth |
| `tinyllama_predictions.csv` | TinyLlama (fine-tuned Llama-1.1B) | 2,000 | Predictions on real-world test set |
| `predictions_deepseek_full2000.csv` | DeepSeek-R1 (7B, Ollama) | 2,000 | Predictions on full 2,000-message test set |

### Schema (`predictions_long_new.csv`)

```
dataset, system, message_id, true_label, probability
```

`message_id` values are CUIDs (lexicographic order approximates chronological
order within the dataset). `true_label`: 1 = hazardous, 0 = benign.

### Schema (`predictions_actions_local_with_truth.csv`)

```
system, message_id, hazard_pred, predicted_category, action_pred,
clinician_action_mapped, action_appropriate, action_match, true_action, under, over
```

Action codes: 1=self-care, 2=monitor, 3=routine appointment, 4=urgent 24-48h,
5=urgent same-day, 6=urgent 2-4h, 7=ED, 8=call 911, 9=call 911 immediately.
