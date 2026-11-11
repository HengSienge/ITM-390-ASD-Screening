# ITM-390 ASD Screening Project: Presentation Slides Deck

**Project:** Comparative Analysis of Machine Learning Models for Autism Spectrum Disorder Screening in Children  
**Course:** ITM-390 Machine Learning · **American University of Phnom Penh (AUPP)**  
**Group 10:** Lyching KONG, Lishang KEAV, Sienge HENG, Lydana TAN · **Advisor:** Kuntha PIN  
**Target Duration:** 15 Minutes Defense + 5 Minutes Q&A (approx. 1.25 minutes per slide)

---

## Slide 1: Title & Team Overview

**Speaker:** Lyching KONG (Team Leader)  
**Slide Header:** Comparative Analysis of Machine Learning Models for Autism Spectrum Disorder Screening in Children  
**Sub-header:** A Standardized, Leak-Free Experimental Benchmark on the UCI Pediatric ASD Cohort

### Visual Elements:
- AUPP School of Digital Technologies Logo
- Group 10 Members & Specific Responsibilities:
  - **Lyching KONG** (Leader) — Problem Formulation, Clinical Context & Error Analysis
  - **Lishang KEAV** — Data Analysis, Leakage Elimination & Preprocessing Architecture
  - **Sienge HENG** — Model Engineering, Repeated Cross-Validation & Statistical Testing
  - **Lydana TAN** — Quality Assurance, Test Suite, Bootstrap Uncertainty & Documentation
- Project Advisor: **Kuntha PIN**

### Speaker Notes (Lyching KONG):
> "Good morning, Professor Kuntha PIN and esteemed colleagues. Today, Group 10 is proud to present our final machine learning research project: a comparative analysis of supervised learning paradigms for early Autism Spectrum Disorder screening in children. In this work, we address a critical healthcare bottleneck using accessible machine learning, resolve a pervasive data leakage flaw in existing literature, and evaluate our models under rigorous repeated cross-validation and statistical significance testing."

---

## Slide 2: Clinical Motivation — The Screening Bottleneck

**Speaker:** Lyching KONG  
**Slide Title:** Clinical Context: Screening vs. Diagnosis

### Visual Elements:
- Flowchart contrasting the **Current Diagnostic Bottleneck** vs. **Proposed ML-Assisted Triage Pathway**:
  - *Current Pathway:* Parent notices behavior $\to$ 12–18 month specialist waitlist $\to$ Expensive clinical evaluation (\$1,000+) $\to$ Delayed therapy.
  - *Proposed Pathway:* 5-minute parent-completed Q-CHAT-10 app $\to$ ML Screening Model $\to$ Instant triage recommendation $\to$ Fast-tracked clinical referral.
- Key Callout Box: **Asymmetric Error Cost in Pediatric Screening**
  - False Positive = Brief anxiety, resolved during formal testing.
  - False Negative = Missed developmental intervention window during peak neuroplasticity.

### Bullet Points:
- Formal diagnostic instruments (ADOS, ADI-R) are resource-intensive, clinical, and scarce in developing regions like Cambodia.
- Q-CHAT-10 compresses social communication behaviors into 10 rapid binary questions.
- **Fundamental Principle:** Our tool is an accessible **screening filter**, not a diagnostic replacement. Therefore, **Recall (Sensitivity) is our primary clinical optimization metric**.

### Speaker Notes (Lyching KONG):
> "Autism Spectrum Disorder affects at least 1 in 100 children globally. Clinical studies confirm that intensive intervention before age four drastically improves life outcomes. Unfortunately, specialized developmental pediatricians are virtually non-existent outside major urban centers. Families often wait over a year for a formal diagnosis. Our objective is to design a digital screening tool that acts as an intelligent sieve. Because missing a child who has autism is vastly worse than temporarily false-flagging a healthy child, our models must maximize sensitivity above all else."

---

## Slide 3: Research Question, Hypothesis & Creative Idea

**Speaker:** Lyching KONG  
**Slide Title:** Scientific Framing & Creative Innovation

### Visual Elements:
- Two central callout banners: **Research Question** and **Hypothesis**
- Creative Idea Badge: **Creative Idea 3.1 — Multi-Model Soft Voting Ensemble**

### Content:
- **Research Question:** How do Logistic Regression (linear), K-Nearest Neighbors (distance-based), and Random Forest (tree ensemble) compare in pediatric ASD screening?
- **Hypothesis:** *The three models will produce different screening performance and distinct error distributions because their mathematical decision surfaces differ fundamentally.*
- **Creative Idea (Course Proposal Box 3.1):** 
  - Combining multiple distinct algorithmic representations via a **Soft Voting Classifier**.
  - Blending linear log-odds probabilities with non-linear tree boundaries to eliminate false negatives.

### Speaker Notes (Lyching KONG):
> "Our research question asks how three classical learning paradigms—linear hyperplanes, non-parametric distance neighborhoods, and axis-aligned tree ensembles—compare when predicting ASD from behavioral screening questions. We hypothesized that these models would yield significantly different decision boundaries and error profiles. Furthermore, to satisfy Creative Idea 3.1 from our proposal, we implemented a Soft Voting Ensemble to examine whether combining continuous probabilities from all three architectures could outperform any single standalone model."

---

## Slide 4: Dataset & The Target Leakage Problem

**Speaker:** Lishang KEAV  
**Slide Title:** Dataset Provenance & Resolving Critical Pre-Work #1

### Visual Elements:
- Screenshot/chart from `01_eda.ipynb` showing the `calc_result` histogram and the exact cutoff at score $\ge 7$.
- Red Warning Banner: **Target Leakage Identified & Solved**.

### Content:
- **Dataset:** UCI *Autism Spectrum Disorder Screening Data for Children* ($N = 292$, 20 attributes).
- **The Critical Flaw in Published Benchmarks:**
  - The dataset includes screening items `A1`–`A10` and a `result` column.
  - `result` $\equiv \sum_{i=1}^{10} A_i$ (exact mathematical sum).
  - In the data generation app, $\text{result} \ge 7 \implies \text{Class/ASD} = \text{YES}$ with 100% precision.
  - Many prior papers left `result` in the feature matrix, producing artificial, scientifically meaningless 100% accuracy.
- **Our Resolution:** `result` is permanently excised before pipeline fitting.
- Constant feature `age_desc` ('4-11 years') dropped; duplicate rows purged ($N = 290$).

### Speaker Notes (Lishang KEAV):
> "I will now walk through our dataset analysis and preprocessing. In Week 1, we audited the raw UCI data and uncovered a massive methodological pitfall: the `result` column is literally the arithmetic sum of the ten question scores. When `result` is 7 or higher, the target is 100% ASD. Several published papers left this column in their models, creating trivial target leakage. We proved this mathematically in our EDA notebook and permanently excised `result` from all feature matrices, ensuring our benchmark measures genuine algorithmic learning."

---

## Slide 5: Preprocessing Pipeline & Multi-Feature Set Design

**Speaker:** Lishang KEAV  
**Slide Title:** Leak-Free Preprocessing & Experimental Feature Sets

### Visual Elements:
- Architecture diagram of the `ColumnTransformer`:
  - Numeric (`age`) $\to$ `SimpleImputer(median)` $\to$ `StandardScaler`
  - Categorical (`gender`, `ethnicity`, `jaundice`, etc.) $\to$ `SimpleImputer(mode)` $\to$ `OneHotEncoder`
  - Binary (`A1`..`A10`) $\to$ Passed as binary 0/1
- 3 Feature Set Comparison Cards: **Set A** (Primary), **Set B** (Questions Only), **Set C** (Demographics Only).

### Content:
- **Strict Leak-Free Protocol:** Preprocessing is encapsulated inside a scikit-learn `Pipeline` and fit solely on training folds.
- **Stratified Splitting:** 80% Training ($N = 232$) / 20% Held-Out Testing ($N = 58$) with fixed seed 42.
- **Three Experimental Conditions:**
  - **Set A (Primary):** Questions + Demographics (18 features).
  - **Set B:** Questions only (10 features).
  - **Set C (Control):** Demographics only (8 features).

### Speaker Notes (Lishang KEAV):
> "To guarantee experimental reproducibility, we built a unified `ColumnTransformer` that handles missing values, one-hot encodes categorical text, and scales numeric age. Everything runs strictly inside our training folds to prevent validation contamination. We also designed three feature configurations: Set A combines questions and demographics; Set B uses questions alone; and Set C uses demographics alone as a negative control to test whether demographics have any predictive validity on their own."

---

## Slide 6: Model Architectures & Hyperparameter Tuning

**Speaker:** Sienge HENG  
**Slide Title:** Algorithmic Formulations & Hyperparameter Optimization

### Visual Elements:
- Side-by-side model summary cards:
  - **Logistic Regression:** L2 penalty, $C = 0.01$, `liblinear` solver.
  - **KNN:** $K = 23$, Manhattan distance, uniform weights.
  - **Random Forest:** 100 estimators, max depth 3, min leaf 2, balanced weights.
  - **Soft Voting Ensemble:** $\frac{1}{3} (P_{\text{LR}} + P_{\text{KNN}} + P_{\text{RF}})$.
- Embedded Figure: `knn_k_sweep.png` showing the bias-variance curve from $K=1$ to $K=25$.

### Key Content:
- All tuning executed via `GridSearchCV` on the training split scored primarily on **Recall**.
- **KNN K-Sweep Diagnostic:** Lower $K$ values showed high variance and lower sensitivity; $K=23$ stabilized recall near 99.8%.
- **Random Forest:** Balanced class weights and shallow depth (3) prevented overfitting on the 232 training examples.

### Speaker Notes (Sienge HENG):
> "Now turning to model development. We evaluated three classical paradigms and one ensemble. Each model was tuned using Repeated Stratified 5-Fold Cross-Validation, scoring directly for Recall. As shown in our KNN K-sweep plot, smaller K values suffered from local noise in categorical distance space. Sweeping to K=23 produced the most robust sensitivity. For Random Forest, restricting tree depth to 3 and enforcing balanced class weights prevented the ensemble from overfitting to dominant demographic subsets."

---

## Slide 7: Primary Results — Cross-Validation & Test Benchmarks

**Speaker:** Sienge HENG  
**Slide Title:** Benchmark Performance Across Validation Folds and Held-Out Test Set

### Visual Elements:
- Table comparing Repeated CV vs. Held-Out Test Set performance.
- Highlighted boxes on **Recall = 1.0000** for Logistic Regression and Soft Voting Ensemble.

### Summary Table:
| Model | CV Recall (25 Folds) | Test Recall ($N=58$) | Test Specificity | Test Accuracy | Test ROC-AUC | Test FN Count |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **1.0000 ± 0.0000** | **1.0000** | 0.6000 | 0.7931 | 0.9798 | **0** |
| **K-Nearest Neighbors** | 0.9982 ± 0.0089 | 0.9643 | 0.7000 | 0.8276 | 0.9786 | 1 |
| **Random Forest** | 0.9676 ± 0.0301 | 0.9643 | **0.9333** | **0.9483** | **0.9976** | 1 |
| **Soft Voting Ensemble** | **0.9946 ± 0.0146** | **1.0000** | 0.7333 | 0.8621 | 0.9881 | **0** |

### Speaker Notes (Sienge HENG):
> "Here are our core quantitative findings. Across both 25 repeated cross-validation folds and the held-out test set, Logistic Regression and the Soft Voting Ensemble achieved a perfect 100% Recall—meaning zero missed cases of ASD. On the other hand, Random Forest delivered the highest overall accuracy at 94.83% and specificity at 93.33%, but it missed exactly one positive child. Our Voting Ensemble successfully caught all positive cases while achieving 73.3% specificity, cutting false positives by a third compared to standalone Logistic Regression."

---

## Slide 8: Statistical Significance & ROC Analysis

**Speaker:** Sienge HENG  
**Slide Title:** Confirming the Hypothesis: McNemar's Tests & ROC Curves

### Visual Elements:
- Embedded Figure: `roc_curves.png` (Overlaid curves showing all models exceeding 0.97 AUC).
- Embedded Figure: `confusion_matrices.png` (Side-by-side confusion heatmaps).
- McNemar Significance Matrix:
  - LR vs. RF: $\chi^2 = 5.8182, \mathbf{p = 0.0159}$ (**Statistically Significant**).
  - KNN vs. RF: $\chi^2 = 5.1429, \mathbf{p = 0.0233}$ (**Statistically Significant**).

### Key Takeaway:
- **Hypothesis Confirmed:** Differences between linear hyperplanes and tree ensembles are statistically significant ($p < 0.05$).
- High ROC-AUC values ($0.978$ to $0.998$) prove that all models possess outstanding rank-ordering discriminative capacity.

### Speaker Notes (Sienge HENG):
> "To test whether these performance differences were real or merely random noise on a small test set, we conducted continuity-corrected McNemar tests. The difference between Logistic Regression and Random Forest is statistically significant with a p-value of 0.0159, and KNN vs. Random Forest is significant at p=0.0233. This formally supports our hypothesis: these models learn fundamentally different decision boundaries, with linear models excelling at inclusive boundary recall and tree ensembles excelling at conservative specificity."

---

## Slide 9: Feature Set Comparison — Behavioral vs. Demographics

**Speaker:** Sienge HENG  
**Slide Title:** Multi-Feature Set Proof: Why Behavioral Questions Are Essential

### Visual Elements:
- Bar chart / Table comparing Feature Set A (All), Set B (Questions), and Set C (Demographics):
  - Set A: Recall $\approx 94.6\%$ to $98.0\%$
  - Set B: Recall $\approx 93.7\%$ to $99.8\%$
  - Set C: Recall $\approx \mathbf{53.2\% \text{ to } 55.0\%}$ (Collapse to Chance!)

### Key Bullet Points:
- Models trained strictly on demographic indicators (age, sex, ethnicity, jaundice, family history) drop to **53.5% recall and 51–58% accuracy**.
- Behavioral Q-CHAT-10 items provide virtually 100% of the diagnostic signal.
- Confirms that algorithmic screening is rooted in observed communicative behaviors, avoiding demographic bias.

### Speaker Notes (Sienge HENG):
> "One of our most impactful experiments was the multi-feature set benchmark. When we trained our models exclusively on demographic features in Set C, performance collapsed across the board to around 53% recall—barely better than tossing a coin. This conclusively proves that demographic traits cannot screen for autism on their own, and that the ten behavioral questions drive the model's intelligence. It also provides reassuring evidence that the models are not relying on demographic profiling."

---

## Slide 10: Clinical Threshold Optimization & Error Case Study

**Speaker:** Lydana TAN  
**Slide Title:** Clinical Decision Thresholding & The Borderline Case Autopsy

### Visual Elements:
- Embedded Figure: `threshold_analysis.png` (Sensitivity vs. Specificity across thresholds).
- Case Profile Card: **Child #Test-42 ($N=1$, Score = 7)**.

### Content:
- **The Single Missed Case:**
  - 7-year-old boy from Jordan, born with jaundice; sum score was exactly 7 (the clinical cutoff).
  - RF predicted $P = 0.498$ (missed by $0.002$!). KNN predicted $P = 0.478$.
  - Logistic Regression ($P = 0.527$) and Soft Voting Ensemble ($P = 0.501$) correctly classified the child!
- **Clinical Policy Recommendation:**
  - Lowering operational decision threshold to **$\tau = 0.40$** guarantees **100% Sensitivity across ALL models**.
  - Increases false alarms by only 1 case while guaranteeing zero missed children.

### Speaker Notes (Lydana TAN):
> "I will now present our error analysis and quality checks. In pediatric screening, an arbitrary 0.50 decision threshold is inappropriate. We conducted a deep autopsy on the single child missed on our test set: a 7-year-old boy whose score was exactly 7. Random Forest assigned a probability of 0.498, missing the diagnosis by two-thousandths of a point! Our Soft Voting Ensemble pulled the probability to 0.501, catching the child. Furthermore, our threshold analysis proves that setting the decision threshold to 0.40 brings every single model to 100% recall, eliminating false negatives completely."

---

## Slide 11: Limitations, Quality Assurance & Reproducibility

**Speaker:** Lydana TAN  
**Slide Title:** Rigor, Unit Testing & Methodological Limitations

### Visual Elements:
- Terminal output showing **9/9 Pytest Tests Passing** in 3.19 seconds.
- GitHub Repository Badge & VS Code workspace configuration diagram.

### Key Bullet Points:
- **Quality Assurance & Testing:**
  - `test_preprocessing.py`: Automated tests proving zero target leakage and strict split isolation.
  - `test_evaluate.py`: Unit tests validating sensitivity, specificity, and McNemar test symmetry.
  - Single-command orchestration via `main.py`.
- **Honest Limitations:**
  - Cohort size ($N = 292$) limits fine-grained subgroup fairness audits.
  - Mobile app parent self-report data may carry mild subjective observation bias.
  - Tool provides behavioral risk categorization, not formal medical diagnosis.

### Speaker Notes (Lydana TAN):
> "Throughout the project, quality assurance and reproducibility were our highest priorities. We built an automated test suite with 9 pytest tests covering leakage prevention, shape integrity, and statistical metric accuracy—all of which pass cleanly. We also configured the entire repository for one-command execution via `main.py` in VS Code. We recognize limitations honestly: our cohort is small at 292 children, and mobile parent reports can carry subjective variance. Therefore, this tool must always be framed as an assistive screening triage system."

---

## Slide 12: Conclusion & Future Work

**Speaker:** Lydana TAN  
**Slide Title:** Final Takeaways & Future Directions

### Visual Elements:
- Summary Checklist:
  - [x] Target leakage resolved and eliminated
  - [x] Multi-feature sets scientifically benchmarked
  - [x] Hypothesis supported with statistical significance ($p < 0.05$)
  - [x] 100% Recall achieved on held-out test split
  - [x] Production-grade, tested, documented GitHub repository
- Future Vision Graphic: Lightweight offline-first mobile app for Cambodian rural health workers.

### Concluding Points:
- **Answers to Research Question:** Logistic Regression and the Soft Voting Ensemble provide the safest clinical screening filters (100% Sensitivity), while Random Forest delivers peak diagnostic specificity.
- **Future Work:** Multi-center clinical validation in Cambodia, probability calibration via Platt scaling, and mobile PWA deployment.
- **Open Invitation for Q&A.**

### Speaker Notes (Lydana TAN):
> "In conclusion, Group 10 has delivered an end-to-end, scientifically sound machine learning screening framework. We proved that behavioral questions are indispensable, demonstrated that our Soft Voting Ensemble achieves 100% recall, and confirmed our hypothesis through rigorous statistical testing. In future work, we plan to calibrate these probabilities and package the model into a lightweight, offline mobile application for rural Cambodian community health centers. Thank you very much for your time and guidance, and we now welcome any questions."
