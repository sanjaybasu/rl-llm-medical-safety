# Data availability and privacy

This repository includes:
- **Physician-created scenarios** (hazard and benign) — fully de-identified and suitable for end-to-end reproduction.
- **Real-world validation (replay) set** — 1,000 HIPAA Safe Harbor–de-identified messages (622 hazard, 378 benign).
- **Prospective evaluation samples** — 500/500 de-identified hazard/benign cases for action-text auditing.
- **Hazard taxonomy reference** — `data/physician_created/scenario_library.csv` lists every hazard class with an example prompt (mirrors Table S1 in the manuscript).

Privacy constraints:
- We cannot release raw PHI-containing transcripts; only Safe Harbor–de-identified messages are provided.
- Synthetic examples are available for testing without PHI.

If additional access is needed under a data use agreement, please contact the corresponding author.
