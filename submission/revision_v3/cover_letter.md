# Cover Letter — Second Revision — JMIR Medical Informatics

**JMIR Medical Informatics Editorial Office**
ed-support@jmir.org

Re: Second Revision of manuscript ms#94081, "Comparative evaluation of AI architectures for medical triage safety: a real-world validation study"

---

Dear Editor Coristine and Reviewers,

We are writing to submit a second major revision of manuscript ms#94081 in response to the editorial decision from JMIR Medical Informatics. We thank Editor Coristine and the three reviewers (Y, BP, and BV) for exceptionally thorough second-round reviews that have further strengthened the work. We have comprehensively addressed all 19 editorial/scientific comments and all second-round reviewer requests.

**Fit with JMIR Medical Informatics**

This study is a systematic comparative evaluation of nine AI architectural configurations for clinical decision support in medical triage, evaluated on 3,000 authentic patient messages from a Medicaid population health program. It falls squarely within the core scope of JMIR Medical Informatics: rigorous real-world evaluation of clinical decision support systems, natural language processing applied to clinical communications, and evidence-based assessment of AI system readiness for deployment in health care workflows. The manuscript addresses three clinical informatics questions with direct practical relevance — which architectural paradigms best support safe triage under real deployment conditions, whether performance leadership is metric-specific or general across sensitivity and discrimination measures, and how supervised optimization compares to architectural category in explaining performance gaps. These questions are grounded in NLP of asynchronous patient messages, model calibration methodology, and systematic head-to-head comparison across four paradigms on identical test cases. The study's use of authentic patient communications (rather than synthetic benchmarks), a medically underserved Medicaid population, and natural hazard prevalence (8.25%) addresses a representativeness gap documented in the clinical informatics evaluation literature. We therefore believe JMIR Medical Informatics is the appropriate venue to disseminate these findings to the clinical informatics community responsible for selecting, deploying, and evaluating AI triage systems in practice.

**Summary of revisions from the first JMIR AI submission (addressed in revision 1)**

1. All nine F1 values in Table 2 corrected (were systematically half the correct values due to a denominator error; corrected using verified results). PPV, NPV, and false negatives per 1,000 messages added.

2. System count standardized throughout to nine configurations across four architectural paradigms; statistical section updated to 36 pairwise McNemar comparisons (Bonferroni α=0.0014).

3. All primary pairwise comparisons changed from exact binomial to McNemar's test for paired binary outcomes, appropriate for architectures evaluated on identical test cases.

4. Comprehensive citation audit performed. The most serious citation error (Ref [37] Rajkomar 2018 fairness paper cited for BERT, FAISS, Sentence-BERT, DeepSeek-R1, MedGemma, Med-PaLM, MMLU-Pro, and fine-tuning) has been corrected with individual peer-reviewed citations for each claim. All other reviewer-identified citation mismatches corrected; duplicate references consolidated; 41 unique references remain.

5. Training composition clarified (285 hazard + 285 benign examples). Calibration procedure added to main Methods (T=0.548, ECE=0.041). Reward function clinical justification and ±50% sensitivity analysis added.

6. Data splitting described with quantified patient overlap: 41 of 1,679 test patients (2.4%) also contributed to the development set, affecting 68 of 2,000 test messages (3.4%). Models used aggregate text features without patient identifiers, limiting the practical impact. Temporal validation absence stated.

7. Architectural comparison framing revised to explicitly acknowledge that performance differences reflect combined advantages of supervised feature extraction, calibration, and decision-theoretic optimization — not architecture alone.

8. Action appropriateness endpoint elevated to formal Table 4 with 95% bootstrap confidence intervals and McNemar paired comparisons for all nine configurations.

9. Equity analysis corrected and expanded. The prior submission reported wrong fairness values (from a non-primary model variant). The revised analysis uses the primary CQL-prob-tuned configuration: sensitivity 0.753 (95% CI 0.656-0.829) for female patients and 0.783 (95% CI 0.644-0.877) for male patients — both exceeding the overall test set sensitivity of 0.727, not below it as previously reported.

10. Language tempered throughout; conclusions scoped to "the tested LLM configurations in this single-program retrospective internal validation."

**Summary of principal new revisions addressing JMIR Medical Informatics editorial decision**

11. IMRD formatting corrected: Conclusions is now the final subsection of Discussion (Heading 3 level), not a standalone section. Discussion subsection order: Principal Findings → Comparison to Prior Work → Architectural Considerations → Strengths and Limitations → Clinical and Policy Implications → Future Work → Conclusions.

12. Linguistic analysis methodology added to Methods: tools (pyspellchecker v0.7.2, spaCy v3.5, textstat v0.7.0), lexicons (847-term colloquialism list, 312-term abbreviation list), and inter-rater reliability (Cohen's kappa 0.64–0.86) are all now specified.

13. Action mapping methodology added to Methods: probability-to-9-action lookup table with threshold ranges for non-CQL architectures; LLM action classification procedure described.

14. RAG retrieval quality validation added to Methods: mean cosine similarity 0.51 (SD 0.14) between patient queries and retrieved passages vs 0.73 for same-source queries, explaining absence of RAG benefit.

15. Baseline naive comparators added to Results: prevalence-only classifier (sensitivity 0, AUROC 0.5) and random classifier benchmarks provide context for absolute performance levels.

16. LLM operating point tuning addressed: few-shot prompting results are repositioned as prompt-based operating point tuning; 10-shot sensitivity 0.610 at specificity 0.495 vs zero-shot 0.400/0.901 demonstrates tradeoff.

17. Temporal stability analysis added to Results: CUID-based approximate temporal ordering shows sensitivity varying 4–7 percentage points across three equal message tertiles (within bootstrap CI width), providing approximate evidence of stability.

18. Manski-style missing demographic bounds analysis added to Results: optimistic bound 0.800, conservative bound 0.642, observed 0.727, implying sensitivity 0.538 in missing-demographic group.

19. FDR/Bonferroni equity analysis: Statistical Analysis section now discusses both Bonferroni and Benjamini-Hochberg corrections for fairness analyses with explicit power limitation acknowledgment.

20. Six new limitations added: unanimity masking clinical uncertainty (8th), single-turn LLM evaluation (9th), health literacy and dialectal variation generalizability expansion (in Discussion), Manski bounds for missing demographics (in Results), hidden local model costs (in Clinical Implications), and approximate temporal analysis (in Results).

21. Future Work subsection added to Discussion: DPO alignment (with specific Rafailov et al. NeurIPS 2023 protocol [ref 42]), temporal split validation protocol, log-probability threshold tuning, and multi-turn dialogue evaluation.

22. Generative AI Disclosure section added (JMIR policy requirement).

A detailed three-column response to each editorial and reviewer comment accompanies this submission.

**Attestations**

This work has not been published and is not under consideration elsewhere. All authors have approved the final version. The study was determined exempt by WCG IRB (tracking ID: 20253751). No patients were involved in study design or interpretation. Funding: none. Conflicts of interest: all authors are employed by Waymark, a company that provides free social and health care services to Medicaid beneficiaries. Waymark does not sell AI triage software. Code is available at https://github.com/sanjaybasu/rl-llm-medical-safety.

We believe this revised manuscript meets JMIR Medical Informatics' scientific and reporting standards and provides a useful contribution to the clinical informatics evidence base for AI-assisted medical triage safety evaluation.

Sincerely,

Sanjay Basu, MD, PhD
University of California San Francisco and Waymark
sanjay.basu@waymarkcare.com
ORCID: 0000-0002-0599-6332

On behalf of co-authors: Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, John Morgan, Rajaie Batniji
