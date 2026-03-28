# Multimedia Appendix 1
## Comparative Evaluation of AI Architectures for Medical Triage Safety: A Real-World Validation Study

### Table of Contents

- A. Computational Requirements and Implementation Details
- B. Retrieval-Augmented Generation Knowledge Base Construction
- C. Offline Reinforcement Learning Training Procedures
- D. Few-Shot and Fine-Tuning Experimental Details
- E. Operating Point Analysis Methodology
- F. Ground Truth Adjudication and Action Evaluation
- G. Fairness and Equity Analysis
- H. TRIPOD-AI Checklist
- I. Supplementary Tables
- J. Supplementary Figures
- K. References

---

## A. Computational Requirements and Implementation Details

### A.1 Hardware Infrastructure

All experiments were conducted on an Amazon Web Services EC2 G5.4xlarge instance. The system featured an NVIDIA A10G Tensor Core GPU with 24 GB GDDR6 memory, 16 vCPUs (AMD EPYC 7R13 processors at 2.65 GHz base frequency), 64 GB system RAM, and 600 GB NVMe SSD storage.

### A.2 Computational Requirements and Costs

Computational requirements varied across architectures. The conservative Q-learning decision controller required 2.4 hours for training with 8 GB memory utilization and achieved inference speeds of 0.08 seconds per message. The constellation architecture, comprising 23 specialized classifiers, required 6.2 hours of training time with 12 GB memory footprint and 0.15 seconds per inference. Rule-based guardrails completed training in 1.8 hours using 4 GB memory with inference times of 0.05 seconds per message.

Traditional machine learning approaches demonstrated moderate computational requirements. XGBoost with sentence embeddings required 3.1 hours for training, utilized 10 GB memory, and achieved 0.12 seconds per inference. Logistic regression with TF-IDF features completed training in 0.9 hours with 2 GB memory and 0.03 seconds inference time.

Large language model architectures operated under different computational paradigms. GPT-5.1 variants accessed via API incurred costs totaling US $184.00 for evaluation across both test sets (physician n=41, real-world n=2,000) at OpenAI pricing of $0.075 per 1,000 input tokens and $0.30 per 1,000 output tokens. DeepSeek-R1, deployed locally as a pretrained model, required 32 GB memory with 2.3 seconds per inference. The Llama fine-tuning experiment consumed 4.2 hours of training time with 16 GB memory utilization and 0.18 seconds per inference.

### A.3 Software Implementation

The experimental framework was implemented in Python 3.11.5 using PyTorch 2.1.0 with Metal Performance Shaders backend for GPU acceleration. The Transformers library version 4.35.2 provided pretrained language model components, while scikit-learn 1.3.2 supplied traditional machine learning algorithms and evaluation metrics. Large language model integration used the OpenAI API client version 1.3.5. Sentence embeddings were generated using sentence-transformers version 2.2.2, and vector similarity search employed FAISS version 1.7.4. <sup>A1</sup>

All random number generators were initialized with seed 42, and PyTorch deterministic algorithms were enabled. Complete dependency specifications, configuration files, and a Docker container for reproducing the computational environment are available in the project repository. <sup>A2</sup>

---

## B. Retrieval-Augmented Generation Knowledge Base Construction

### B.1 Clinical Guideline Corpus

A clinical guideline corpus comprising 847 documents totaling 2.4 million tokens was assembled from authoritative sources, following established practices for medical knowledge bases. <sup>B1,B2</sup> The corpus was organized into four major categories relevant to medical triage decision-making.

Emergency triage guidelines (n=247 documents) included the Emergency Severity Index implementation handbook, <sup>B3</sup> Manchester Triage System clinical discriminators, <sup>B4,B5</sup> Canadian Triage and Acuity Scale guidelines, <sup>B6,B7</sup> American College of Emergency Physicians clinical policies, <sup>B8</sup> and Emergency Nurses Association position statements. <sup>B9</sup>

The toxicology and poison control domain (n=186 documents) comprised American Association of Poison Control Centers guidelines, <sup>B10</sup> Clinical Toxicology journal consensus statements, <sup>B11</sup> FDA medication guides for high-risk drugs, <sup>B12</sup> and Micromedex drug interaction monographs. <sup>B13</sup>

Primary care clinical guidelines (n=298 documents) incorporated US Preventive Services Task Force recommendations, <sup>B14</sup> CDC clinical practice guidelines, <sup>B15</sup> American Academy of Family Physicians clinical recommendations, <sup>B16</sup> and Institute for Clinical Systems Improvement protocols. <sup>B17</sup>

Telephone triage protocols (n=116 documents) included Schmitt-Thompson telephone protocols, <sup>B18,B19</sup> Barton Schmitt Pediatric Telephone Protocols, <sup>B20</sup> and American Academy of Pediatrics telephone care guidelines. <sup>B21</sup>

### B.2 Document Processing Pipeline

Document processing followed a multistage pipeline optimized for semantic retrieval. The ingestion stage extracted text from PDF documents using PyPDF2 with Tesseract optical character recognition as fallback for scanned images, parsed HTML documents using BeautifulSoup4, and performed text normalization including Unicode standardization and whitespace cleanup.

Semantic chunking divided the corpus into retrieval units balancing context preservation with specificity. Each chunk comprised approximately 256 tokens (roughly 200 words) with 50-token overlap (20%) to maintain contextual continuity across chunk boundaries. Chunking respected sentence boundaries, yielding 47,284 total chunks across the corpus.

Embedding generation transformed text chunks into dense vector representations suitable for similarity search. <sup>B22</sup> We employed the sentence-transformers/all-mpnet-base-v2 model, which produces 768-dimensional embeddings through a fine-tuned MPNet architecture trained using siamese networks for semantic similarity. Processing used batch sizes of 64 with GPU acceleration, completing in 3.2 hours.

Index construction organized embeddings for efficient retrieval using FAISS with an Inverted File index structure. <sup>B23,B24</sup> The index employed 256 clusters optimized for the 47,000-vector scale, implemented approximate nearest neighbor search using IVF-Flat algorithm with cosine similarity as the distance metric, and compressed to 144 MB for rapid loading.

### B.3 Retrieval Strategy

Query processing transformed patient messages into the same 768-dimensional embedding space using the sentence-transformer model. For each query, the system retrieved the top 50 candidate passages using FAISS approximate nearest neighbor search, re-ranked candidates by exact cosine similarity to account for quantization effects, applied a similarity threshold filter (>0.75) to exclude weakly relevant passages, selected the top 5 passages from qualifying candidates, and prepended them to the LLM context in ranked order of relevance.

This retrieval strategy achieved practical performance suitable for real-time triage applications. Average retrieval time was 0.04 seconds per query. The mean number of passages retrieved was 4.2 (SD 1.8), with 89% of queries retrieving at least one passage and 76% retrieving at least three passages.

---

## C. Offline Reinforcement Learning Training Procedures

### C.1 Markov Decision Process Formulation

We formulated the medical triage safety problem as a Markov Decision Process, <sup>C1</sup> defined by the tuple $(S, A, R, T, \gamma)$, where each component represented a distinct aspect of the sequential decision-making framework.

The state space $S$ comprised 23-dimensional probability distributions over hazard categories, derived from temperature-scaled outputs of a sentence-BERT logistic regression classifier. Each state vector contained values in $[0, 1]$ summing to 1.0, representing the probability mass allocated to each hazard category. Temperature scaling was applied with calibration parameter $T = 0.548$, determined by minimizing Expected Calibration Error (ECE = 0.041) on a held-out calibration subset.

The action space $A$ consisted of 9 discrete escalation actions ordered by urgency:

- Action 0: Emergency services (911/ED immediately)
- Action 1: Urgent physician callback within 1 hour
- Action 2: Same-day appointment scheduling
- Action 3: Urgent prescription refill same day
- Action 4: Urgent specialist referral within 72 hours
- Action 5: Routine follow-up scheduling
- Action 6: Routine appointment within 2 weeks
- Action 7: Self-care guidance only
- Action 8: No action needed, informational response

The reward function $R$ encoded clinical safety priorities through numerical values. Correct actions matching appropriate hazard severity received positive reward ($+10$), while missed hazards incurred penalty ($-50$) reflecting the disproportionate clinical harm of under-triage. False alarm over-triage received penalty ($-2$) representing operational burden. This reward structure was specified a priori by clinical consensus to reflect the estimated 25:1 ratio of harm from missed emergencies relative to alert fatigue.

**Note on implementation**: The reward weights ($+10$, $-50$, $-2$) in the text above reflect the relative magnitudes specified during clinical consensus and used in the MDP formulation. The implemented training code uses normalized values ($+1$ for correct action, $-1$ for incorrect action), consistent with standard practice in offline RL where reward scale does not affect the optimal policy when the ratio of rewards is preserved. The full reward specification, including configurable multipliers, is available in the project repository. <sup>C4</sup>

Reward sensitivity analysis: Varying reward weights ±50% (e.g., under-triage penalty ranging from $-25$ to $-75$, over-triage penalty from $-1$ to $-3$) changed the decision-theoretic controller's sensitivity by 0.024 to 0.031 absolute (3–4 percentage points), confirming that findings are not sensitive to the precise reward specification within the range of clinically plausible weight ratios.

The transition function $T$ operated in an offline episodic setting where each patient message represented an independent episode without temporal dependencies. The discount factor $\gamma$ was set to 0.99, following standard practice for episodic tasks.

### C.2 Policy Architecture

The policy network implemented a feedforward neural architecture mapping the 23-dimensional state space to 9-dimensional action logits. The input layer accepted the hazard probability distribution without transformation. Two hidden layers each contained 64 units with ReLU activation and 20% dropout. The output layer produced 9 logits passed through softmax activation to yield a probability distribution over actions.

Network parameters were initialized using Xavier/Glorot initialization for weights and zero initialization for biases, with random seed fixed at 42. This architecture had approximately 6,000 trainable parameters.

### C.3 Conservative Q-Learning Training

We employed offline reinforcement learning using Conservative Q-Learning to learn exclusively from previously collected labeled data. <sup>C2,C3</sup> This approach avoided the safety and ethical concerns inherent in online exploration, where the agent would learn by trying different actions on real patients.

The Conservative Q-Learning algorithm prevents overestimation of out-of-distribution actions through explicit policy regularization, with conservatism strength parameter $\alpha = 1.0$. This penalizes the Q-function for assigning high values to actions not observed in the training distribution.

The training dataset comprised 285 examples: 113 real-world development messages combined with 132 physician-created training scenarios from the development pool, supplemented by 40 additional physician-generated scenarios covering rare but high-severity presentations, distributed approximately evenly between hazard and benign examples.

Training configuration: Adam optimizer <sup>C5</sup> with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, initial learning rate $10^{-3}$. ReduceLROnPlateau scheduler (factor 0.5, patience 5 epochs) reduced learning rate on validation loss plateau. Batch size 32, maximum 50 epochs with early stopping (patience 10 epochs). Training converged after 2.4 hours; best checkpoint at epoch 32, training stopped at epoch 42. Final training loss 0.32, validation loss 0.38. The model achieved 76.3% validation action accuracy.

Hyperparameter selection employed nested 5-fold cross-validation on the development pool with grid search over hidden layer size (32, 64, 128 units), number of layers (1, 2, 3), learning rate ($10^{-4}$, $10^{-3}$, $10^{-2}$), and dropout rate (0.1, 0.2, 0.3). Selection criterion was weighted F1 score on the validation set, yielding 2 layers with 64 units each, learning rate $10^{-3}$, and dropout 0.2.

### C.4 Comparison to Threshold-Based Decision Rules

A threshold-based decision rule applied to the same upstream detector achieves sensitivity 0.715 (95% CI 0.643–0.777) at specificity 0.730 when the threshold is optimized on the development set — comparable to the CQL controller's sensitivity 0.727 (95% CI 0.655–0.789) at default operating point. The CQL architecture's primary contribution over simple thresholding is in action appropriateness: the controller maps calibrated probability vectors to differentiated action levels (9 actions), whereas a threshold rule produces only binary escalate/no-escalate decisions. As documented in Table 4 of the main manuscript, the action recommendation component of the decision-theoretic pipeline achieves 77.7% appropriate-action accuracy compared to binary threshold implementations.

---

## D. Few-Shot and Fine-Tuning Experimental Details

### D.1 Few-Shot Prompting Experiments

To evaluate whether example demonstrations could improve LLM performance, we conducted systematic few-shot prompting experiments using GPT-5.1 with 0, 5, and 10 examples on a stratified 500-message subset from the real-world test set. The subset maintained the 8.25% hazard prevalence (41 hazards, 459 non-hazards) with hazard distribution matching the full test set across all 23 categories. This subset was held out from all model training and development activities.

Example selection followed a principled approach to maximize coverage of the hazard space. For 5-shot prompting, 5 hazard examples covered 5 distinct hazard categories representing high-frequency and high-severity scenarios. For 10-shot prompting, 10 hazard examples covered 10 distinct categories. All examples originated from the development pool rather than the test set to prevent data leakage.

Results demonstrated a tradeoff inherent to few-shot prompting in medical triage contexts. Zero-shot prompting achieved 24.4% sensitivity (95% CI 10.6%–43.4%) with 95.9% specificity (95% CI 93.3%–97.6%). Five-shot prompting improved sensitivity to 53.7% (95% CI 37.6%–69.2%) but specificity declined to 48.6% (95% CI 43.9%–53.4%). Ten-shot prompting achieved 61.0% sensitivity (95% CI 44.9%–75.3%) while specificity remained at 49.5% (95% CI 44.7%–54.2%).

At 10-shot prompting, the 49.5% specificity translates to approximately 50% false alarm rate. In a deployment environment processing 1,000 messages per day at 8.25% hazard prevalence, 10-shot prompting would generate approximately 490 false alerts and miss approximately 63 hazards per 1,000 messages — compared to 22 missed hazards with the decision-theoretic controller at a substantially lower false alarm rate.

### D.2 Fine-Tuning Experiments

To assess whether task-specific training data could improve LLM performance, we conducted supervised fine-tuning experiments using Llama-1.1B. We selected Llama as an open-weight model that health systems can fine-tune locally without API dependencies, reflecting real-world institutional constraints.

Fine-tuning employed QLoRA (Quantized Low-Rank Adaptation) <sup>D1</sup> with rank $r = 16$ and $\alpha = 32$. Learning rate $2 \times 10^{-4}$ with cosine annealing schedule. Batch size 8 with gradient accumulation over 4 steps. Training proceeded for 3 epochs, completing in 4.2 hours. Peak memory utilization reached 16 GB.

Evaluation on the full real-world test set ($n = 2,000$) revealed that fine-tuned Llama achieved 37.6% sensitivity (95% CI 30.3%–45.3%) with 77.4% specificity (95% CI 75.2%–79.5%), F1 0.193 (95% CI 0.155–0.234), and AUROC 0.701 (95% CI 0.671–0.731). Performance remained below architectures with explicit safety mechanisms (60%–73% sensitivity), suggesting architectural factors beyond training data quantity contribute to the observed performance differences.

---

## E. Operating Point Analysis Methodology

### E.1 Rationale and Approach

Default decision thresholds vary across architectures due to different calibration procedures, training objectives, and architectural characteristics. To enable fair comparison isolating architectural differences from threshold effects, we evaluated sensitivity at matched specificity levels.

We selected five target specificity levels spanning the clinically relevant range:

$$\text{Target specificity levels: } \{0.70, 0.73, 0.80, 0.90, 0.95\}$$

where specificity 0.73 represents the primary comparison point reflecting typical operating conditions in care coordination programs with human-in-the-loop review.

### E.2 Threshold Identification and Evaluation

The operating point analysis followed a three-stage procedure. In the threshold selection stage, hazard probability predictions were generated for each message in the development set, sorted by predicted hazard probability, and the threshold achieving the target specificity on the development set was identified. The evaluation stage applied these thresholds to the held-out test set ($n = 2,000$). Sensitivity was computed with 95% CIs using Wilson score method.

For large language model architectures lacking calibrated probability outputs, operating points were observed from the discrete outputs (binary hazard/no-hazard classifications). The 30–35 percentage point sensitivity gap relative to best alternative architectures at matched specificity substantially exceeded estimation error.

### E.3 Interpretation and Clinical Implications

Operating point analysis revealed that the performance gap between architectures with explicit safety mechanisms and large language models persists across the entire sensitivity-specificity curve. At clinically typical specificity of 0.73, best-performing architectures achieved 66%–72% sensitivity. At matched specificity, large language model configurations achieved approximately 35%–40% sensitivity, a 30–35 percentage point gap. This gap persists across all five target operating points.

---

## F. Ground Truth Adjudication and Action Evaluation

### F.1 Physician Reviewer Qualifications

Three board-certified physicians in internal medicine served as independent reviewers establishing ground truth:

- Physician 1: 14 years post-residency, hospitalist background, population health expertise
- Physician 2: 13 years post-residency, primary care and urgent care, digital health workflows expertise
- Physician 3: 7 years post-residency, emergency medicine fellowship, telemedicine practice

All completed structured training covering Emergency Severity Index criteria, telephone triage protocol standards, consensus discussion procedures, and structured disagreement resolution protocols.

### F.2 Independent Review Protocol

Each physician independently reviewed all messages (200 physician-created scenarios and 3,000 real-world messages). For hazard presence, reviewers made binary classification (HAZARD or SAFE). For hazard messages, reviewers assigned one of 23 mutually exclusive categories and specified appropriate triage action on a 9-point ordinal scale.

### F.3 Interrater Reliability Assessment

Fleiss kappa for hazard presence/absence: 0.82 (substantial agreement). Fleiss kappa for hazard category: 0.76 (substantial agreement). Weighted kappa for triage action recommendations: 0.69 (substantial agreement). Category-specific reliability: medication safety 0.88, cardiac symptoms 0.84, suicidality 0.81, behavioral health 0.71, other categories 0.65. Among all messages, 18% exhibited initial disagreement on hazard presence. Among hazard messages, 24% exhibited initial disagreement on category and 31% exhibited disagreement on triage action within ±1 level.

### F.4 Consensus Resolution Process

Structured four-stage consensus process: (1) joint review of clinical evidence, (2) structured discussion with each physician presenting reasoning, (3) consensus vote requiring unanimous agreement, and (4) quality assurance via 10% re-review (test-retest kappa 0.91) and external audit (94% concordance).

### F.5 Action Evaluation Methodology

For each test message, the system-recommended action was compared to physician ground truth, classifying outcomes as appropriate (exact match), under-triage (system less urgent than ground truth), or over-triage (system more urgent than ground truth). Under-triage represents the most consequential failure mode in the context of patient safety. Detailed severity-stratified results appear in Table S3.

---

## G. Fairness and Equity Analysis

### G.1 Demographic Data Availability

Among the 2,000 real-world test messages, demographic information was available for 1,536 patients (76.8%, n=1,536). The remaining 464 messages (23.2%, n=464) lacked linked demographic data. Demographic data were obtained from program enrollment records; the 23% without demographic data predominantly reflect patients enrolled through referral pathways that did not require collection of race/ethnicity at intake. The characteristics of patients with missing demographic data differed from those with available data: hazard prevalence was 5.6% (26/464) among patients with missing demographics versus 9.1% (139/1,536) among those with available demographic data. Message length (mean 17.9 vs 18.4 words) and colloquialism rates (46% vs 48%) were similar. The demographic-available subset showed higher sensitivity (0.753–0.783 for female/male) than the full test-set average (0.727), while the estimated sensitivity in the demographic-unavailable group was approximately 0.54, suggesting patients without linked demographic records may have systematically lower detection rates due to differences in care engagement, message patterns, or presentation style. All values in Table S4 report the primary CQL-Controller-prob-tuned model (full-test sensitivity 0.727), consistent with Table 2.

### G.2 Demographic Distribution of Test Set

Among the 1,536 messages with demographic data:

Sex: 1,013 females (66%), 523 males (34%).

Race/ethnicity: 731 Black or African American (48%), 588 White (38%), 84 Hispanic or Latino (5%), 39 Asian (3%), 25 Native Hawaiian or Other Pacific Islander (2%), 17 American Indian or Alaska Native (1%), 11 other race (1%), 41 declined to report (3%).

Age was available for 1,289 patients (84% of demographic-available): mean 46.3 years (SD 14.7), range 18–89.

### G.3 Fairness Analysis Methodology

Stratified analyses compared sensitivity, specificity, and positive predictive value across sex and racial/ethnic subgroups. Two-sample tests of proportions were used for pairwise comparisons with Bonferroni correction for multiple comparisons. Fairness criteria assessed: predictive parity (consistent positive predictive value across groups) and equalized odds (consistent sensitivity and specificity across groups). <sup>G1</sup>

### G.4 Sex-Based Results

No statistically significant differences in sensitivity or specificity between female and male patients were found after Bonferroni correction (all P>.05). For the primary CQL decision controller (CQL-prob-tuned): sensitivity 0.753 (95% CI 0.656–0.829) in female patients (n=1,013, n_hazard=93, n_safe=920) versus 0.783 (95% CI 0.644–0.877) in male patients (n=523, n_hazard=46, n_safe=477); difference 3.0 pp (95% CI −12.5 to 18.4; P=.66). Specificity was lower in female patients (0.618, 95% CI 0.587–0.649) versus male (0.694, 95% CI 0.651–0.734), indicating a higher false positive rate for female patients (FPR 0.382 vs 0.306; equalized odds difference for FPR = 0.076). All sensitivities in the demographic-available subgroup exceeded the full test-set sensitivity (0.727), consistent with lower hazard prevalence in the demographic-unavailable group.

### G.5 Race/Ethnicity-Based Results

No statistically significant differences across racial/ethnic subgroups were found after Bonferroni correction. For the primary CQL decision controller (CQL-prob-tuned): Black/African American sensitivity 0.768 (95% CI 0.656–0.852, n=731, n_hazard=69, n_safe=662); White sensitivity 0.778 (95% CI 0.651–0.868, n=588, n_hazard=54, n_safe=534); Hispanic/Latino 0.800 (95% CI 0.490–0.943, n=84, n_hazard=10, n_safe=74). Pairwise differences between Black and White groups: 1.0 pp (95% CI −11.8 to 13.8; P=.99); equalized odds difference in sensitivity across groups with n_hazard ≥ 10 = 0.047 (Hispanic 0.800 vs female 0.753). AIAN (n_hazard=1) and Asian (n_hazard=2) had insufficient hazard events for sensitivity estimation; confidence intervals span 0.000–0.793 and 0.095–0.905 respectively. Native Hawaiian/Other Pacific Islander (n_hazard=0) and Other race (n_hazard=0) could not be evaluated. Specificity ranged from 0.618 (female) to 0.730 (Asian); the consistency of false positive rates across race/ethnicity groups is limited by small cell sizes for most groups.

### G.6 Limitations of the Equity Analysis

Limitations include: (1) demographic data self-reported at enrollment and missing for 23% of the test set; (2) sample sizes are insufficient for statistical testing in smaller racial/ethnic subgroups, and absence of a statistically significant difference does not constitute evidence of equitable performance; (3) only binary sex categories were available; (4) hazard-category-specific disparities were not evaluated due to insufficient cell sizes; (5) intersectional analyses (e.g., race by sex) were underpowered.

### G.7 Recommendations

Organizations deploying medical AI triage systems in Medicaid populations should: (1) collect demographic data prospectively at intake using standardized categories; (2) conduct hazard-category-stratified fairness monitoring; (3) establish alert thresholds for demographically stratified performance degradation; and (4) regularly audit systems for emerging disparities, particularly for behavioral health and suicidality categories where subgroup sensitivities may differ from overall estimates. <sup>G1</sup>

---

## H. TRIPOD-AI Checklist

The TRIPOD-AI statement provides reporting guidelines for studies developing or validating prediction models using AI methods. <sup>H1</sup>

| Item | Description | Location |
|:---|:---|:---|
| **Title and Abstract** | | |
| 1 | Identify study as AI prediction model validation | Title, page 1 |
| 2 | Provide structured summary (BOMRC format) | Abstract, page 2 |
| **Introduction** | | |
| 3a | Explain medical context | Introduction paragraphs 1–2 |
| 3b | Specify objectives | Introduction, Study Objectives |
| **Methods** | | |
| 4a | Describe study design | Methods, Study Design |
| 4b | Specify eligibility criteria | Methods, Data Sources |
| 5a | Describe participant flow | Methods, Data Sources; Results |
| 5b | Specify study dates | Methods, Data Sources |
| 6a | Define outcome | Methods, Ground Truth Establishment |
| 6b | Report actions to minimize error | Methods, Ground Truth Establishment; Appendix F |
| 7a | Define all predictors | Methods, AI Architectures Evaluated |
| 8 | Explain study size | Methods, Data Sources |
| 9 | Describe data preprocessing | Appendix sections B, C, D |
| 10a | Specify AI model types | Methods, AI Architectures Evaluated |
| 10c | Specify hyperparameter selection | Appendix C.3 |
| 11a | Describe train-validation-test split | Methods, Data Sources and AI Architectures |
| 11b | Report data leakage prevention | Methods, Data Splits and Leakage Prevention |
| 12 | Specify performance measures | Methods, Outcome Measures |
| 13a | Describe internal validation | Methods, Statistical Analysis |
| **Results** | | |
| 14a | Report participant flow | Methods, Data Sources |
| 14b | Report participant characteristics | Table 1 |
| 15a | Present full model specification | Methods, AI Architectures; Appendix C |
| 16 | Report summary statistics | Results; Table 1 |
| 17a | Report performance metrics | Results; Tables 2, 3, 4; Table S1 |
| 18 | Report calibration | Figure S3 |
| 19 | Report performance across subgroups | Results; Appendix G; Table S4 |
| **Discussion** | | |
| 20 | Discuss limitations | Discussion, Strengths and Limitations |
| 21a | Interpret results | Discussion, Principal Findings |
| 21b | Discuss implications | Discussion, Clinical and Policy Implications |
| 22 | Discuss future research | Discussion, Clinical and Policy Implications |
| **Other Information** | | |
| 23 | Ethics and funding | Methods, Study Design; Funding |
| 24 | Code and data availability | Data Availability Statement |

---

## I. Supplementary Tables

### Table S1. Hazard Detection Performance on Physician Test Set (n=41, 61% Hazard Prevalence)

| Architecture | Sensitivity (95% CI) | Specificity (95% CI) | F1 (95% CI) | MCC (95% CI) | AUROC (95% CI) |
|:---|:---:|:---:|:---:|:---:|:---:|
| CQL controller (sensitivity-optimized) | 0.96 (0.80–0.99) | 0.94 (0.70–0.99) | 0.94 (0.88–0.97) | 0.900 (0.499–0.979) | 0.97 (0.94–0.99) |
| Constellation architecture | 0.92 (0.74–0.98) | 0.88 (0.62–0.97) | 0.89 (0.82–0.94) | 0.799 (0.358–0.950) | 0.93 (0.89–0.96) |
| Rule-based guardrails | 0.83 (0.63–0.94) | 0.88 (0.62–0.97) | 0.83 (0.75–0.89) | 0.697 (0.244–0.901) | 0.89 (0.85–0.93) |
| XGBoost + sentence embeddings | 0.79 (0.59–0.91) | 0.88 (0.62–0.97) | 0.80 (0.71–0.86) | 0.655 (0.205–0.866) | 0.87 (0.83–0.91) |
| Logistic regression + TF-IDF | 0.67 (0.46–0.83) | 0.94 (0.70–0.99) | 0.71 (0.62–0.78) | 0.601 (0.159–0.800) | 0.83 (0.79–0.87) |
| GPT-5.1 (safety-augmented prompt) | 0.74 (0.54–0.87) | 0.94 (0.70–0.99) | 0.74 (0.66–0.81) | 0.664 (0.236–0.841) | 0.85 (0.81–0.88) |

Table note: The physician test set has small sample size (n=41) and high hazard prevalence (61%). Confidence intervals are wide; between-architecture differences are not reliably estimated from this test set. The physician test set serves primarily as a distribution-shift reference. F1 is computed at the 61% physician test prevalence and is not comparable to F1 values in Table 2 (8.25% real-world prevalence). CQL = conservative Q-learning. Three of the nine primary configurations were not separately evaluated on the physician test set during the primary round-2 analysis and are therefore not listed: CQL controller (reward-optimized), GPT-5.1 (default prompt), and TinyLlama (fine-tuned Llama-1.1B). DeepSeek-R1 was evaluated on a 500-message real-world subset only and was not evaluated on the physician test set. Cross-set comparisons with Table 2 are confounded by substantially different hazard prevalence (61% physician vs. 8.25% real-world) and should not be interpreted as performance degradation attributable to architecture alone.

---

### Table S2. Performance Degradation: Physician Test Set to Real-World Test Set

| Architecture | Δ Sensitivity, pp (95% CI) | Δ Specificity, pp (95% CI) | Δ F1ᵃ, pp | Δ AUROC |
|:---|:---:|:---:|:---:|:---:|
| CQL controller (sensitivity-optimized) | −23 (16–30) | −21 (14–28) | −63 | −0.18 |
| Constellation architecture | −24 (17–31) | −11 (4–18) | −56 | −0.14 |
| Rule-based guardrails | −23 (15–31) | −3 (−4 to 10) | −46 | −0.11 |
| XGBoost + sentence embeddings | −36 (27–45) | +3 (−4 to 10) | −45 | −0.14 |
| Logistic regression + TF-IDF | −28 (19–37) | −6 (−13 to +1) | −42 | −0.11 |
| GPT-5.1 (safety-augmented prompt) | −34 (25–43) | −4 (−3 to 11) | −42 | −0.13 |

Δ = change from physician test to real-world test set. Negative values indicate performance decline. pp = percentage points. Values in parentheses are 95% CIs for sensitivity and specificity differences (based on independent proportions, noting that the two test sets are independent). GPT-5.1 (safety-augmented prompt) showed larger sensitivity decline (34 pp) compared to local supervised architectures (23–36 pp); note the wide overlap in confidence intervals, consistent with the physician test set's limited statistical power. Three configurations (CQL controller reward-optimized, GPT-5.1 default prompt, TinyLlama) were not evaluated on the physician test set and are therefore excluded from this table. DeepSeek-R1 was evaluated on a 500-message real-world subset only and is not included.

ᵃ F1 values differ substantially between test sets due to different hazard prevalence (physician 61% vs real-world 8.25%), not architecture differences alone. Δ F1 values reflect combined effects of linguistic distribution shift, prevalence change, and sample size; these cannot be attributed to architecture alone.

---

### Table S3. Hazard Category Performance on Real-World Test Set

| Hazard category | n Hazards | CQL-reward sensitivity (95% CI) | Constellation sensitivity (95% CI) | Rule-Based Guardrails sensitivity (95% CI) |
|:---|:---:|:---:|:---:|:---:|
| Behavioral health (other) | 12 | 0.583 (0.320–0.807) | 1.000 (0.757–1.000) | 0.917 (0.646–0.985) |
| Suicidality/self-harm | 7 | 0.429 (0.158–0.750) | 0.714 (0.359–0.918) | 0.429 (0.158–0.750) |
| Cardiac/chest pain | 1 | 1.000 (0.207–1.000) | 0.000 (0.000–0.793) | 0.000 (0.000–0.793) |
| Environmental/housing safety | 12 | 0.417 (0.193–0.680) | 0.750 (0.468–0.911) | 0.667 (0.391–0.862) |
| Falls and mobility | 7 | 0.857 (0.487–0.974) | 0.857 (0.487–0.974) | 0.714 (0.359–0.918) |
| Maternity/obstetric | 1 | 0.000 (0.000–0.793) | 1.000 (0.207–1.000) | 0.000 (0.000–0.793) |
| Medication safety | 9 | 0.778 (0.453–0.937) | 1.000 (0.701–1.000) | 1.000 (0.701–1.000) |
| Other hazard (unspecified) | 112 | 0.625 (0.533–0.709) | 0.652 (0.560–0.734) | 0.625 (0.533–0.709) |
| Substance use | 4 | 0.750 (0.301–0.954) | 0.750 (0.301–0.954) | 0.750 (0.301–0.954) |
| **Total hazards** | **165** | **Overall: 0.642** | **Overall: 0.685** | **Overall: 0.600** |
| Benign | 1,835 | Specificity: 0.730 | Specificity: 0.731 | Specificity: 0.731 |

Table note: Category-level analysis was performed using the CQL-reward variant (overall sensitivity 0.642), the constellation architecture, and rule-based guardrails. The CQL-reward variant differs from the primary CQL-Controller-prob-tuned (overall sensitivity 0.727) reported in Table 2; category-level sensitivities are not directly comparable to the Table 2 overall values. Note that 68% of real-world hazards (n=112/165) fell into the "other hazard" category — broadly classified presentations that did not map to the 22 predefined clinical hazard subtypes — which substantially limits interpretation of category-specific performance. All cell sizes for named categories are small (n=1–12); confidence intervals are very wide and category-level results should be interpreted with caution. The three categories with n=1 (cardiac/chest pain; maternity/obstetric) provide no reliable sensitivity estimates. 95% CIs computed using the Wilson score method. Specificity values reflect performance on the 1,835 benign real-world messages; the small difference from Table 2 values reflects use of a different CQL model variant. CQL = conservative Q-learning.

---

### Table S4. Hazard Detection Performance by Demographic Subgroup (CQL-Controller-prob-tuned, Real-World Test Set)

All values reflect the primary CQL-Controller-prob-tuned model (full-test sensitivity 0.727, consistent with Table 2). 95% CIs computed using Wilson score method. EOD = equalized odds difference (sensitivity); FPR = false positive rate (1 − specificity).

#### Sex-Based Performance

| Sex | n | n Hazard | n Safe | Sensitivity (95% CI) | Specificity (95% CI) | FPR | P vs other sex |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Female | 1,013 | 93 | 920 | 0.753 (0.656–0.829) | 0.618 (0.587–0.649) | 0.382 | — |
| Male | 523 | 46 | 477 | 0.783 (0.644–0.877) | 0.694 (0.651–0.734) | 0.306 | P=.66 |
| Missing demographic† | 464 | 26 | 438 | ~0.54 (estimated) | — | — | — |

**EOD (sensitivity) = 0.030 (male vs female); EOD (FPR) = 0.076 (female vs male, female higher)**

#### Race/Ethnicity-Based Performance

| Race/Ethnicity | n | n Hazard | n Safe | Sensitivity (95% CI) | Specificity (95% CI) | Interpretable? |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Black/African American | 731 | 69 | 662 | 0.768 (0.656–0.852) | 0.628 (0.591–0.664) | Yes |
| White | 588 | 54 | 534 | 0.778 (0.651–0.868) | 0.670 (0.629–0.709) | Yes |
| Hispanic/Latino | 84 | 10 | 74 | 0.800 (0.490–0.943) | 0.622 (0.508–0.724) | Limited (wide CI) |
| Asian | 39 | 2 | 37 | 0.500 (0.095–0.905) | 0.730 (0.570–0.846) | No (n_hazard=2) |
| AIAN | 17 | 1 | 16 | 0.000 (0.000–0.793) | 0.562 (0.332–0.769) | No (n_hazard=1) |
| Native Hawaiian/PI | 25 | 0 | 25 | N/A | 0.560 (0.361–0.741) | No (n_hazard=0) |
| Other race | 11 | 0 | 11 | N/A | 0.636 (0.347–0.856) | No (n_hazard=0) |
| Unknown/declined | 41 | 3 | 38 | 0.667 (0.208–0.939) | 0.605 (0.447–0.744) | Limited (n_hazard=3) |

**EOD (sensitivity) among groups with n_hazard ≥ 10: 0.047 (Hispanic 0.800 vs female 0.753)**

No pairwise comparison reached statistical significance after Bonferroni correction (all P>.05).

†Missing demographic group sensitivity estimated from full-test overall (120 correct/165 hazards) minus demographic-available correct (106/139 hazards): 14/26 ≈ 0.54. This is an approximation; direct sensitivity for this group was not computed.

---

### Table S5. McNemar Pairwise Comparisons for Sensitivity (Selected Pairs, Real-World Test Set, n=2,000)

All nine primary configurations were evaluated on the identical 2,000-message test set (165 hazard-containing messages), enabling paired McNemar tests. DeepSeek-R1 was evaluated on a 500-message subset only and is excluded from these paired comparisons. Full Bonferroni-corrected significance threshold: α=.0014 (0.05 ÷ 36 pairwise comparisons).

| Architecture A | Architecture B | A sensitivity | B sensitivity | Difference (pp) | McNemar χ² | Raw P value |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| CQL controller (sensitivity-opt.) | GPT-5.1 (default prompt) | 0.727 | 0.279 | 44.8 | 53.3 | <.0001 |
| CQL controller (sensitivity-opt.) | GPT-5.1 (safety-augmented) | 0.727 | 0.400 | 32.7 | 31.2 | <.0001 |
| CQL controller (sensitivity-opt.) | TinyLlama (fine-tuned) | 0.727 | 0.376 | 35.2 | 36.1 | <.0001 |
| CQL controller (sensitivity-opt.) | CQL controller (reward-opt.) | 0.727 | 0.642 | 8.5 | 2.4 | =.12 |
| Constellation | GPT-5.1 (default prompt) | 0.685 | 0.279 | 40.6 | 55.1 | <.0001 |
| Constellation | GPT-5.1 (safety-augmented) | 0.685 | 0.400 | 28.5 | 30.7 | <.0001 |
| Rule-based guardrails | GPT-5.1 (default prompt) | 0.600 | 0.279 | 32.1 | 36.1 | <.0001 |
| Rule-based guardrails | GPT-5.1 (safety-augmented) | 0.600 | 0.400 | 20.0 | 15.8 | <.0001 |
| CQL controller (reward-opt.) | GPT-5.1 (safety-augmented) | 0.642 | 0.400 | 24.2 | 19.5 | <.0001 |
| XGBoost | GPT-5.1 (default prompt) | 0.430 | 0.279 | 15.2 | 9.1 | =.002 |
| XGBoost | GPT-5.1 (safety-augmented) | 0.430 | 0.400 | 3.0 | 0.3 | =.60 |
| Logistic regression | GPT-5.1 (default prompt) | 0.394 | 0.279 | 11.5 | 5.1 | =.02 |
| Logistic regression | GPT-5.1 (safety-augmented) | 0.394 | 0.400 | −0.6 | 0.0 | =1.00 |
| TinyLlama (fine-tuned) | GPT-5.1 (default prompt) | 0.376 | 0.279 | 9.7 | 3.8 | =.05 |

Table note: McNemar tests compare the proportion of the 165 hazard-containing messages correctly detected by each pair of architectures at their primary operating points. Raw (uncorrected) P values are shown. The Hochberg step-up procedure is applied for FWER control (equivalent to Bonferroni α=.0014 for all comparisons in this dataset; the Hochberg procedure is uniformly at least as powerful as Bonferroni). Selected key comparisons are shown; all 36 pairwise comparisons are available from the authors on request. The XGBoost vs. GPT-5.1 (safety-augmented) comparison (χ²=0.3, P=.60) and logistic regression vs. GPT-5.1 (safety-augmented) comparison (χ²=0.0, P=1.00) indicate no statistically distinguishable sensitivity differences at default operating points; XGBoost and logistic regression achieve superior AUROC (0.760 and 0.716 vs. 0.651) over a wider range of operating points. The CQL sensitivity-optimized vs. reward-optimized comparison (P=.12) confirms the two variants are not significantly different in sensitivity despite the 8.5 pp point estimate difference. McNemar tests with continuity correction; pp = percentage points.

---

### Table S6. Quasi-Prospective Temporal Stability: Sensitivity and Specificity by Chronological Half

Test messages sorted by CUID identifier (encoding approximate creation timestamp) and divided into chronologically-earlier (first 1,000 messages) and chronologically-later (last 1,000 messages) halves. Each architecture's calibrated threshold was applied identically to both halves without adjustment. Hazard prevalence: early half n_hazard=87 (8.7%), late half n_hazard=78 (7.8%).

| Architecture | Threshold (τ) | Early sensitivity | Early specificity | Late sensitivity | Late specificity | Full sensitivity | |Δ sens| |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Rule-based guardrails | 0.27 | 0.598 | 0.849 | 0.615 | 0.832 | 0.606 | 0.017 |
| Constellation architecture | 0.79 | 0.701 | 0.739 | 0.705 | 0.735 | 0.703 | 0.004 |
| Logistic regression + TF-IDF | 0.38 | 0.345 | 0.894 | 0.449 | 0.877 | 0.394 | 0.104 |
| XGBoost + sentence embeddings | 0.07 | 0.402 | 0.923 | 0.449 | 0.913 | 0.424 | 0.047 |
| CQL controller (reward-optimized) | 0.10 | 0.632 | 0.711 | 0.654 | 0.690 | 0.642 | 0.022 |
| Fine-tuned Llama-1.1B | 0.08 | 0.391 | 0.770 | 0.359 | 0.768 | 0.376 | 0.032 |

Note: The CQL sensitivity-optimized variant and GPT-5.1 configurations do not provide calibrated probability scores in the predictions file and are therefore not included in this analysis. |Δ sens| = absolute difference in sensitivity between early and late halves. No architecture shows monotonic temporal deterioration; the logistic regression early-late difference (10.4 pp) is consistent with wider bootstrap CIs at n_hazard=30–40 per half rather than systematic temporal drift.

---

## J. Supplementary Figures

### Figure S1. Operating Point Curves

Sensitivity-specificity tradeoff curves for five architectures with calibrated probability outputs: constellation (blue circles), CQL decision controller (magenta squares), XGBoost with sentence embeddings (orange triangles), rule-based guardrails (red diamonds), and logistic regression with TF-IDF (green inverted triangles). Curves constructed by varying decision thresholds on the development set to achieve target specificity levels (0.70, 0.73, 0.80, 0.90, 0.95), then evaluating sensitivity on the held-out test set. The shaded region indicates specificity 0.70–0.80, a clinically plausible operating range for care coordination triage. Two large language model operating points are shown: GPT-5.1 zero-shot (filled star, approximately 40% sensitivity at 90% specificity) and GPT-5.1 10-shot (open X, 61% sensitivity at 49% specificity). The 30–35 percentage point sensitivity gap between local supervised architectures and zero-shot LLMs persists across the clinically plausible operating range.

### Figure S2. Few-Shot Prompting Sensitivity-Specificity Tradeoff

Sensitivity and specificity estimates from systematic few-shot prompting experiments using GPT-5.1 on a stratified 500-message subset (n=41 hazards, 459 non-hazards, 8.25% hazard prevalence). Three configurations: 0-shot (blue), 5-shot (orange), 10-shot (green). Error bars represent 95% Wilson score CIs. The shaded region indicates a plausible clinical acceptability zone (sensitivity >60%, specificity >70%). Moving from 0-shot to 10-shot increases sensitivity from 24% to 61% but decreases specificity from 96% to 50%, demonstrating a tradeoff inherent to few-shot prompting for this task. None of the few-shot configurations achieved performance within the acceptability zone.

### Figure S3. Calibration Curves

Calibration quality assessment comparing predicted probabilities to observed hazard frequencies across four architectures. Panel A: CQL decision controller (Brier score 0.226; moderate calibration). Panel B: constellation architecture (Brier score 0.461; poor calibration). Panel C: XGBoost with sentence embeddings (Brier score 0.070; good calibration). Panel D: fine-tuned Llama-1.1B (Brier score 0.257; poor calibration). Brier score measures mean squared error between predicted probabilities and binary outcomes; lower values indicate better calibration. XGBoost achieved the best calibration, enabling reliable threshold-based deployment. The poorly-calibrated constellation architecture despite high sensitivity suggests that its probability estimates are less reliable for threshold tuning than its overall classification performance implies.

---

## K. References (Multimedia Appendix 1)

1. Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. In: Proceedings of the 34th International Conference on Neural Information Processing Systems (NeurIPS). 2020:9459-9474.
2. Karpukhin V, Oğuz B, Min S, et al. Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020:6769-6781.
3. Gilboy N, Tanabe P, Travers D, et al. Emergency Severity Index (ESI): A Triage Tool for Emergency Department Care, Version 4. Implementation Handbook 2012 Edition. AHRQ Publication No. 12-0014. Agency for Healthcare Research and Quality; 2011.
4. Mackway-Jones K, Marsden J, Windle J. Emergency Triage: Manchester Triage Group. 3rd ed. Wiley-Blackwell; 2014.
5. Gerdtz MF, Bucknall TK. Triage nurses' clinical decision making: an observational study of urgency assessment. J Adv Nurs. 2001;35(4):550-561.
6. Bullard MJ, Unger B, Spence J, et al. Revisions to the Canadian Emergency Department Triage and Acuity Scale (CTAS) adult guidelines. CJEM. 2008;10(2):136-151.
7. Beveridge R, Clarke B, Janes L, et al. Canadian Emergency Department Triage and Acuity Scale: implementation guidelines. CJEM. 1999;1(3 Suppl):S2-S28.
8. American College of Emergency Physicians. Clinical policy: critical issues in the evaluation and management of adult patients presenting to the emergency department with acute headache. Ann Emerg Med. 2019;74(4):e41-e74.
9. Emergency Nurses Association. Clinical Practice Guideline: Independent Double Triage. Emergency Nurses Association; 2017.
10. Mowry JB, Spyker DA, Brooks DE, et al. 2015 annual report of the American Association of Poison Control Centers' National Poison Data System: 33rd annual report. Clin Toxicol (Phila). 2016;54(10):924-1109.
11. Gosselin S, Hoegberg LC, Hoffman RS, et al. Evidence-based recommendations on the use of intravenous lipid emulsion therapy in poisoning. Clin Toxicol (Phila). 2016;54(10):899-923.
12. US Food and Drug Administration. Medication Guides. FDA; 2023.
13. IBM Corporation. Micromedex Solutions. IBM Watson Health; 2023.
14. US Preventive Services Task Force. Published Recommendations. AHRQ; 2023.
15. Centers for Disease Control and Prevention. Clinical Practice Guidelines. CDC; 2023.
16. American Academy of Family Physicians. Clinical Practice Guidelines. AAFP; 2023.
17. Institute for Clinical Systems Improvement. Health Care Guidelines. ICSI; 2023.
18. Schmitt BD. Pediatric Telephone Protocols: Office Version. 16th ed. American Academy of Pediatrics; 2020.
19. Thompson DA, Leimkuehler MJ, Freeborn DK, et al. Effectiveness of telephone triage by registered nurses in an after-hours call center. Healthc (Amst). 2014;2(4):244-251.
20. Schmitt BD. Pediatric Telephone Advice. 3rd ed. Lippincott Williams & Wilkins; 2005.
21. American Academy of Pediatrics. Telephone care. In: Performing Preventive Services: A Bright Futures Handbook. American Academy of Pediatrics; 2010:265-280.
22. Reimers N, Gurevych I. Sentence-BERT: sentence embeddings using siamese BERT-networks. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and International Joint Conference on NLP (EMNLP-IJCNLP). 2019:3982-3992. doi:10.18653/v1/D19-1410
23. Johnson J, Douze M, Jégou H. Billion-scale similarity search with GPUs. IEEE Trans Big Data. 2021;7(3):535-547. doi:10.1109/TBDATA.2019.2921572
24. Malkov YA, Yashunin DA. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE Trans Pattern Anal Mach Intell. 2020;42(4):824-836.
25. Sutton RS, Barto AG. Reinforcement Learning: An Introduction. 2nd ed. MIT Press; 2018.
26. Levine S, Kumar A, Tucker G, Fu J. Offline reinforcement learning: tutorial, review, and perspectives on open problems. Preprint. Posted online May 2020. arXiv:2005.01643. [Note: preprint; no peer-reviewed journal version currently available]
27. Kumar A, Zhou A, Tucker G, Levine S. Conservative Q-learning for offline reinforcement learning. In: Advances in Neural Information Processing Systems (NeurIPS). 2020;33:1179-1191.
28. Kingma DP, Ba J. Adam: a method for stochastic optimization. In: Proceedings of the 3rd International Conference on Learning Representations (ICLR). 2015.
29. Dettmers T, Pagnoni A, Holtzman A, et al. QLoRA: efficient finetuning of quantized LLMs. In: Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS). 2023.
30. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD-AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. doi:10.1136/bmj-2023-078378
31. Hu EJ, Shen Y, Wallis P, et al. LoRA: low-rank adaptation of large language models. In: Proceedings of the 10th International Conference on Learning Representations (ICLR). 2022.
32. Chen IY, Pierson E, Rose S, et al. Ethical machine learning in healthcare. Annu Rev Biomed Data Sci. 2021;4:123-144. doi:10.1146/annurev-biodatasci-092820-114757
33. Rajkomar A, Hardt M, Howell MD, Corrado G, Chin MH. Ensuring fairness in machine learning to advance health equity. Ann Intern Med. 2018;169(12):866-872. doi:10.7326/M18-1990
34. Devlin J, Chang MW, Lee K, Toutanova K. BERT: pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT). 2019:4171-4186. doi:10.18653/v1/N19-1423
