# Data

## Physician-created test scenarios (`physician_test/`)

Triage scenarios authored by three board-certified physicians, systematically
sampled from emergency triage frameworks (ESI, MTS, CTAS) and safety databases.

| File | N | Description |
|:---|:---:|:---|
| `hazard_scenarios_holdout.json` | 189 | Held-out hazard scenarios used in manuscript evaluation |
| `hazard_scenarios_train.json` | 811 | Training hazard scenarios (not in manuscript test tables) |
| `benign_scenarios.json` | 500 | Benign scenarios (200 used in physician test set) |
| `scenario_library.csv` | — | Category taxonomy: 18 hazard categories, 5 domains |

The physician test set for manuscript Table S1 comprises 189 holdout hazards +
200 benign scenarios (N=389 total; 48.6% hazard prevalence).

### Schema (`hazard_scenarios_holdout.json` and `hazard_scenarios_train.json`)

```json
{
  "id": "string",
  "text": "string",
  "category": "string",
  "subcategory": "string",
  "hazard_level": "high|medium",
  "source_framework": "ESI|MTS|CTAS|other"
}
```

### Schema (`benign_scenarios.json`)

```json
{
  "id": "string",
  "text": "string",
  "category": "string"
}
```

## Real-world patient messages (not included)

Real-world messages from a multistate Medicaid care coordination program
(January 2023–November 2025; n=3,000 total; 8.25% hazard prevalence in
the 2,000-message test set) are not distributed due to privacy constraints.

The messages are de-identified under HIPAA Safe Harbor (45 CFR §164.514(b)(2))
but contain patient-authored text and are not suitable for public distribution
without a data use agreement.

**To request access:** sanjay.basu@waymarkcare.com
