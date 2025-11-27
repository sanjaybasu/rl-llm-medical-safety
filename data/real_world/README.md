This directory contains the de-identified real-world datasets used in the manuscript:

- `replay_scenarios_llm_labels.json`: Full 1,000-message validation set (fields: id, prompt, hazard_type, label, annotations).
- `prospective_eval/harm_cases_500.csv`: Prospective harm cases (500 rows).
- `prospective_eval/benign_cases_500.csv`: Prospective benign cases (500 rows).

All PHI has been removed; remaining details (e.g., care pathways, workflows) are retained for reproducibility. Downstream scripts will use these files directly. If you require an additional review copy or a minimized sample, please contact the corresponding author.
