
# Comparative Evaluation of AI Architectures for Medical Triage Safety: A Real-World Validation Study
Authors: Sanjay Basu, MD, PhD¹,²,*; Sadiq Y. Patel, MSW, PhD²,³; Parth Sheth, MSE²,³; Bhairavi Muralidharan, MSE²; Namrata Elamaran, MSE²; Aakriti Kinra, MS²; John Morgan, MD²,⁴; Rajaie Batniji, MD, PhD²,⁵

Affiliations:
¹University of California San Francisco, San Francisco, CA, USA
²Waymark, San Francisco, CA, USA
³University of Pennsylvania, Philadelphia, PA, USA
⁴Virginia Commonwealth University, Richmond, VA, USA
⁵Stanford University, Stanford, CA, USA

Correspondence:
Sanjay Basu, MD, PhD
1001 Potrero Avenue
San Francisco, CA 94110
Email: sanjay.basu@ucsf.edu

Layout
- `data/deidentified/`: PHI-free datasets
- `results/`:  for all main-text and appendix figures and tables.
- `scripts/`: utilities to regenerate tables and audit model outputs 

What’s included
- All main-text tables (linguistic characteristics; real-world performance; operating-point analysis).
- All appendix tables (physician-set performance; degradation; hazard-category breakdown; demographic stratification).
- De-identified real-world dataset mirroring hazard labels and action classes without copying any raw text; source message IDs are hashed for linkage without PHI.

What’s intentionally excluded
- Raw PHI-bearing messages and LLM responses that may risk PHI identification.
- Model weights and API keys. Use your own credentials to re-run external LLMs if needed.
- Data-generation scripts for the de-identified set (kept offline to reduce PHI exposure risk).

