# Comparative Analysis of Machine Learning Models for Autism Spectrum Disorder Screening in Children

**Course:** ITM-390: Machine Learning  
**Institution:** American University of Phnom Penh (AUPP) · School of Digital Technologies  
**Project Group:** Group 10  
**Authors:**  
- **Lyching KONG** (Team Leader / Literature & Comparative Analysis)
- **Lishang KEAV** (Dataset Analysis, EDA & Preprocessing Pipeline)
- **Sienge HENG** (Model Training, Tuning & Evaluation Architecture)
- **Lydana TAN** (Testing, Reproducibility & Technical Documentation)  
**Project Advisor:** Kuntha PIN, School of Digital Technologies  
**Date:** November 2026

---

## Abstract

Early identification of Autism Spectrum Disorder (ASD) in young children is critical for facilitating prompt behavioral intervention during peak periods of neurological development. However, gold-standard diagnostic pathways are severely constrained by prohibitive clinical costs and prolonged specialist waitlists. Machine learning classifiers trained on brief behavioral screening instruments offer a scalable triage mechanism to expedite pediatric referrals. In this study, we present a standardized, leak-free benchmarking framework comparing three canonical classification paradigms—**Logistic Regression (linear)**, **K-Nearest Neighbors (distance-based)**, and **Random Forest (non-linear ensemble)**—alongside a **Soft Voting Ensemble** on the UCI *Autism Spectrum Disorder Screening Data for Children* dataset ($N = 292$). 

Methodologically, we identify and resolve a critical flaw in prior literature: direct target leakage via the arithmetic `result` score column, which deterministically encodes the target label and artificially inflates accuracy. By strictly removing `result` and benchmarking three distinct feature sets (All Features, Questions Only, and Demographics Only), we demonstrate that demographic indicators alone cannot differentiate ASD (achieving near chance $\approx 53.5\%$ recall), whereas the 10-item Q-CHAT behavioral questions provide virtually all predictive utility. Under a rigorous **Repeated Stratified 5-Fold Cross-Validation (25 validation folds)** and an untouched held-out test split ($N = 58$), **Logistic Regression** and the **Soft Voting Ensemble** achieved **100% Recall (0 False Negatives)** with ROC-AUCs of $0.9798$ and $0.9881$, respectively. **Random Forest** achieved the highest overall accuracy ($94.83\%$) and specificity ($93.33\%$), but produced one false negative at the default $0.50$ threshold. Continuity-corrected McNemar's tests confirmed statistically significant performance divergence between linear and non-linear architectures ($p = 0.0159$ for LR vs. RF). Decision threshold calibration to $\tau = 0.40$ successfully brought all models to $100\%$ sensitivity. We provide a fully reproducible, open-source code repository with 9/9 passing unit tests.

**Keywords:** Autism Spectrum Disorder (ASD), Pediatric Screening, Machine Learning, Target Leakage, Logistic Regression, Random Forest, K-Nearest Neighbors, Soft Voting Ensemble, McNemar's Test.

---

## 1. Introduction

Autism Spectrum Disorder (ASD) is a lifelong neurodevelopmental condition characterized by persistent challenges in social communication, reciprocal social interaction, and restricted, repetitive patterns of behavior, interests, or activities (American Psychiatric Association, DSM-5). Global epidemiological surveys estimate that approximately 1 in 100 children worldwide receive an ASD diagnosis, with modern surveillance in high-income countries reporting rates exceeding 1 in 36 children (Maenner et al., 2023). In Southeast Asian developing nations, including Cambodia, access to specialized developmental pediatricians, child psychologists, and licensed speech therapists is heavily concentrated in major urban centers, leaving rural and provincial families without accessible diagnostic infrastructure.

Extensive clinical research demonstrates that intensive behavioral therapies initiated before the age of four capitalize on early neuroplasticity, yielding profound and durable improvements in language acquisition, adaptive functioning, and social communication. Conversely, delayed identification compounds developmental delays and increases caregiver burden. Standard clinical diagnosis relies on comprehensive, time-intensive observational tools such as the Autism Diagnostic Observation Schedule (ADOS) and Autism Diagnostic Interview-Revised (ADI-R), which require multiple hours of specialist administration and cost hundreds to thousands of dollars.

To address this screening bottleneck, standardized parent-report screening questionnaires—such as the Quantitative Checklist for Autism in Toddlers (Q-CHAT-10) and Autism Spectrum Quotient (AQ-10)—have been developed. These instruments compress core behavioral traits into ten rapid binary questions. Machine learning applied to these screening instruments presents an unprecedented opportunity to create automated, low-cost digital screening tools deployable on commodity mobile devices.

### 1.1 The Clinical Screening vs. Diagnostic Distinction
A foundational premise of this investigation is that **this project develops a clinical screening tool, not a diagnostic system**. In clinical triage, the error loss function is highly asymmetric:
- **False Positive (Type I Error):** A typically developing child is flagged as exhibiting autistic traits. The consequence is brief parental concern followed by a formal specialist evaluation, where the child is cleared.
- **False Negative (Type II Error):** A child with ASD is incorrectly classified as neurotypical. The consequence is catastrophic: formal clinical referral is delayed, therapeutic intervention is missed, and the critical developmental window passes unutilized.

Consequently, while prior literature in computer science prioritizes overall accuracy, this study explicitly establishes **Recall / Sensitivity** as the primary optimization metric, with Specificity, Accuracy, Precision, F1-Score, and ROC-AUC reported as secondary evaluations.

---

## 2. Research Goals and Objectives

### 2.1 Research Question
> *How do Logistic Regression, K-Nearest Neighbors, and Random Forest compare in screening for ASD among children using behavioral and demographic features under a standardized, leak-free evaluation framework?*

### 2.2 Hypothesis
> *The three models will produce different screening performance and distinct error distributions because they construct decision boundaries through fundamentally different mathematical mechanisms (linear hyperplanes, non-parametric distance neighborhoods, and axis-aligned tree ensembles).*

### 2.3 Specific Project Objectives
1. **Target Leakage Proof and Elimination:** Formally prove that the `result` column in the UCI repository leaks the target variable, and design a preprocessing pipeline that prevents data contamination.
2. **Standardized Preprocessing Pipeline:** Develop a single scikit-learn `ColumnTransformer` encapsulating median/most-frequent imputation, categorical one-hot encoding, and numerical standardization, strictly fit within training folds.
3. **Multi-Feature Set Benchmark:** Evaluate three distinct feature spaces—**Set A** (Behavioral + Demographics), **Set B** (Behavioral Only), and **Set C** (Demographics Only)—to scientifically quantify feature group importance.
4. **Algorithmic Tuning and Diagnostic Sweeps:** Systematically tune Logistic Regression (L1/L2 regularization), KNN ($K$-sweep for bias-variance diagnostic), and Random Forest using Repeated Stratified Cross-Validation scored on Recall.
5. **Creative Idea Implementation (Ensemble Modeling):** Construct a Soft Voting Ensemble combining the three tuned classifiers to assess whether blending linear and non-linear representations eliminates false negatives.
6. **Rigorous Statistical Evaluation:** Compute 1,000-resample non-parametric bootstrap 95% confidence intervals, pairwise continuity-corrected McNemar's tests, and decision threshold sweeps.
7. **Granular Error Profiling:** Conduct a clinical case study on all false negative and false positive predictions on the held-out test set.

---

## 3. Literature Review

The application of machine learning to behavioral health assessments has expanded dramatically over the past decade. Thabtah (2017, 2019, 2020) pioneered the digital capture of Q-CHAT-10 screening records and published benchmark datasets for toddlers, children, adolescents, and adults on the UCI Machine Learning Repository. In subsequent studies, Thabtah utilized rule-based associative classifiers and Naive Bayes, reporting classification accuracies exceeding 95% on child data.

However, a critical review of published papers utilizing the UCI ASD datasets reveals pervasive methodological weaknesses:
1. **Target Leakage via Score Summation:** Multiple published studies inadvertently included the `result` attribute alongside the ten question items (`A1` to `A10`). Because `result` is the exact mathematical sum of `A1` through `A10`, and the positive diagnosis in the original application was defined as $\text{score} \ge 7$, retaining `result` renders classification trivial, yielding near 100% accuracy that does not reflect real-world predictive utility.
2. **Over-reliance on Accuracy in Imbalanced Cohorts:** Standard accuracy obscures class-specific sensitivities. When clinical screening demands near-zero false negative rates, an accuracy of 90% that misses 25% of positive cases is clinically hazardous.
3. **Absence of Uncertainty Bounds:** Many previous benchmarks evaluate only a single random train/test split on a dataset of under 300 instances, leading to noisy point estimates susceptible to sampling variance. None of the benchmark papers provide non-parametric bootstrap confidence intervals or formal McNemar significance testing between algorithm pairs.

Our study bridges these critical methodological gaps by establishing an open-source, leak-free, statistically rigorous benchmark designed specifically around pediatric screening imperatives.

---

## 4. Methodology

```
Raw ARFF Dataset (N=292) 
   │
   ├──> Data Cleaning (Drop 'result', drop 'age_desc', fix '?', handle text casing)
   │
   └──> Stratified 80/20 Train/Test Split (Seed 42)
           │
           ├── Training Set (80%, N=232)
           │      │
           │      ├── Repeated Stratified 5-Fold Cross-Validation (25 Folds)
           │      │      ├── Model 1: Logistic Regression (C, L1/L2 GridSearch)
           │      │      ├── Model 2: KNN (K=1..25 Sweep, Manhattan/Euclidean)
           │      │      └── Model 3: Random Forest (Depth, Trees, Class Weight)
           │      │
           │      └── Model 4: Soft Voting Ensemble (Combines Tuned LR, KNN, RF)
           │
           └── Held-Out Test Set (20%, N=58)
                  │
                  ├── Point Metrics (Recall, Specificity, Accuracy, Precision, F1, ROC-AUC)
                  ├── 1,000 Bootstrap 95% Confidence Intervals
                  ├── Pairwise McNemar Significance Tests
                  ├── Decision Threshold Analysis (tau = 0.05 to 0.95)
                  └── False Negative Clinical Case Profiling
```

### 4.1 Dataset Provenance & Target Leakage Proof
The dataset comprises 292 pediatric screening records collected via the ASDTests mobile application and hosted on the UCI Machine Learning Repository (DOI: 10.24432/C5659W). Each record comprises 20 attributes: 10 behavioral items (`A1_Score` to `A10_Score`), 8 demographic/clinical variables (`age`, `gender`, `ethnicity`, `jaundice`, `family_asd`, `country_of_res`, `used_app_before`, `relation`), 1 summary score (`result`), 1 age descriptor (`age_desc`), and the target screening outcome (`Class/ASD`).

**Empirical Leakage Demonstration:**  
In our exploratory data analysis (`notebooks/01_eda.ipynb`), we computed:
$$\Delta = \text{result} - \sum_{i=1}^{10} A_i$$
Across all 292 instances, $\Delta \equiv 0$ with 100% exactitude. Furthermore, cross-tabulating `result` against `Class/ASD` demonstrated:
- $\text{result} < 7 \implies \text{Class/ASD} = \text{NO}$ (100% of 151 controls)
- $\text{result} \ge 7 \implies \text{Class/ASD} = \text{YES}$ (100% of 141 cases)

Retaining `result` as an input feature creates perfect target leakage. Therefore, **`result` was permanently dropped prior to all modeling activities**. Additionally, `age_desc` was dropped because all 292 records contained the identical string `'4-11 years'`. Two exact duplicate rows were removed, leaving 290 unique instances.

### 4.2 Multi-Feature Set Architecture
To systematically assess feature importance, we structured three experimental feature matrices:
1. **Feature Set A (Primary Experiment, 18 features):** All 10 behavioral items plus 8 demographic/clinical features.
2. **Feature Set B (Behavioral Items Only, 10 features):** Screening items `A1` through `A10` only.
3. **Feature Set C (Demographic Control, 8 features):** Demographics and background variables only, omitting all 10 behavioral questions.

### 4.3 Data Preprocessing & Leak-Free Pipeline
All transformations were constructed using scikit-learn's `ColumnTransformer` and encapsulated within unified `Pipeline` objects:
- **Numerical Features (`age`):** Imputed using median replacement followed by `StandardScaler` ($z = (x - \mu)/\sigma$).
- **Categorical Features (`gender`, `ethnicity`, `jaundice`, `family_asd`, `country_of_res`, `used_app_before`, `relation`):** Missing values (originally coded as `'?'`) imputed with the most frequent category, followed by one-hot encoding (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`).
- **Binary Behavioral Questions (`A1`..`A10`):** Imputed with most frequent value and passed through without scaling to maintain binary interpretability.

Crucially, the `ColumnTransformer` is embedded inside cross-validation folds and fit strictly on training splits, ensuring zero information leakage from validation or test data.

### 4.4 Supervised Learning Algorithms & Hyperparameter Tuning
Four distinct classification architectures were implemented and tuned using `GridSearchCV` scored primarily on **Recall**:

1. **Logistic Regression (Interpretable Linear Baseline):**  
   Models the log-odds of ASD:
   $$P(y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \boldsymbol{\beta}^T \mathbf{x})}}$$
   Tuned over regularization strength $C \in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]$, penalties L1/L2, and class-weight balancing.
2. **K-Nearest Neighbors (Distance-Based Classifier):**  
   Classifies instances based on majority voting of neighboring training samples in normalized feature space:
   $$P(y=1 \mid \mathbf{x}) = \frac{1}{K} \sum_{i \in N_K(\mathbf{x})} y_i$$
   Evaluated over an odd neighborhood sweep $K \in \{1, 3, 5, \dots, 25\}$, distance weightings (uniform vs. distance-weighted), and distance metrics (Euclidean vs. Manhattan).
3. **Random Forest (Non-Linear Tree Ensemble):**  
   An ensemble of $B$ bootstrap-aggregated, decorrelated classification trees:
   $$\hat{P}(y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
   Tuned over `n_estimators` $\in [50, 100, 150]$, `max_depth` $\in [\text{None}, 3, 5, 8]$, `min_samples_leaf` $\in [1, 2, 4]$, and `class_weight` $\in [\text{None}, \text{'balanced'}]$.
4. **Soft Voting Ensemble (Creative Idea 3.1):**  
   Combines the continuous class probability estimates of the tuned Logistic Regression, KNN, and Random Forest pipelines:
   $$P_{\text{Ensemble}}(y=1 \mid \mathbf{x}) = \frac{1}{3} \left[ P_{\text{LR}}(y=1 \mid \mathbf{x}) + P_{\text{KNN}}(y=1 \mid \mathbf{x}) + P_{\text{RF}}(y=1 \mid \mathbf{x}) \right]$$

### 4.5 Cross-Validation & Statistical Evaluation Protocol
- **Cross-Validation:** Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 distinct folds) with fixed seed 42 applied to the 80% training split ($N = 232$).
- **Held-Out Test Set:** A single stratified 20% held-out test split ($N = 58$, comprising 28 ASD cases and 30 controls) preserved untouched until final evaluation.
- **Bootstrap Uncertainty:** 1,000 non-parametric bootstrap resamples on the test set to compute 95% empirical percentile confidence intervals ($2.5^{\text{th}}$ to $97.5^{\text{th}}$ percentiles).
- **Statistical Significance (McNemar's Test):** Edwards' continuity-corrected McNemar test applied to pairwise model predictions:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{df} = 1$$
  where $b$ is the count of samples correctly classified by Model A and misclassified by Model B, and $c$ is the reverse discordant count.
- **Decision Threshold Optimization:** Performance evaluated across decision thresholds $\tau \in [0.05, 0.95]$ in increments of $0.05$.

---

## 5. Experimental Results

### 5.1 Hyperparameter Tuning Results
Cross-validated grid search on training folds yielded the following optimal configurations:
- **Logistic Regression:** $C = 0.01$, $\text{penalty} = \text{L2}$, $\text{class\_weight} = \text{None}$, $\text{solver} = \text{liblinear}$. (Achieved mean CV Recall = $1.0000$).
- **K-Nearest Neighbors:** $K = 23$, $\text{weights} = \text{'uniform'}$, $\text{metric} = \text{'manhattan'}$. (Achieved mean CV Recall = $0.9982$).
- **Random Forest:** `n_estimators = 100`, `max_depth = 3`, `min_samples_leaf = 2`, `class_weight = 'balanced'`. (Achieved mean CV Recall = $0.9676$).

### 5.2 Repeated Stratified 5-Fold Cross-Validation Performance (25 Folds)

Table 1 summarizes model performance across 25 cross-validation folds on the training partition ($N = 232$):

**Table 1: Repeated Stratified 5-Fold Cross-Validation Metrics (Mean ± Standard Deviation)**
| Model | CV Recall (Primary) | CV Specificity | CV Accuracy | CV ROC-AUC | CV Precision | CV F1-Score | Train Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **1.0000 ± 0.0000** | 0.5483 ± 0.1037 | 0.7700 ± 0.0617 | 0.9547 ± 0.0321 | 0.6824 ± 0.0587 | 0.8098 ± 0.0412 | 20.69 s |
| **K-Nearest Neighbors** | 0.9982 ± 0.0089 | 0.6925 ± 0.0888 | 0.8423 ± 0.0528 | 0.9714 ± 0.0165 | 0.7596 ± 0.0621 | 0.8613 ± 0.0400 | 55.34 s |
| **Random Forest** | 0.9676 ± 0.0301 | **0.8708 ± 0.0718** | **0.9181 ± 0.0448** | **0.9825 ± 0.0131** | **0.8796 ± 0.0632** | **0.9204 ± 0.0416** | 315.42 s |
| **Soft Voting Ensemble** | **0.9946 ± 0.0146** | 0.7283 ± 0.0815 | 0.8587 ± 0.0533 | **0.9818 ± 0.0122** | 0.7816 ± 0.0658 | 0.8738 ± 0.0416 | 391.46 s |

---

### 5.3 Held-Out Test Set Performance ($N = 58$)

Table 2 presents the definitive performance of the four models evaluated once on the untouched held-out test split:

**Table 2: Final Held-Out Test Set Results ($N = 58$, 28 ASD Cases, 30 Controls)**
| Model | Recall (Sensitivity) | Specificity | Accuracy | Precision | F1-Score | ROC-AUC | Brier Score | TP | FP | TN | FN | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **1.0000** | 0.6000 | 0.7931 | 0.7000 | 0.8235 | 0.9798 | 0.1979 | 28 | 12 | 18 | **0** | 0.20 ms |
| **K-Nearest Neighbors** | 0.9643 | 0.7000 | 0.8276 | 0.7500 | 0.8437 | 0.9786 | 0.1037 | 27 | 9 | 21 | 1 | 0.25 ms |
| **Random Forest** | 0.9643 | **0.9333** | **0.9483** | **0.9310** | **0.9474** | **0.9976** | 0.1216 | 27 | 2 | 28 | 1 | 0.42 ms |
| **Soft Voting Ensemble** | **1.0000** | 0.7333 | 0.8621 | 0.7778 | 0.8750 | 0.9881 | 0.1304 | 28 | 8 | 22 | **0** | 0.44 ms |

---

### 5.4 Non-Parametric Bootstrap 95% Confidence Intervals (1,000 Resamples)

**Table 3: 95% Bootstrap Confidence Intervals on Held-Out Test Split**
| Model | Recall 95% CI | Specificity 95% CI | Accuracy 95% CI | Precision 95% CI | ROC-AUC 95% CI |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | **[1.000, 1.000]** | [0.414, 0.769] | [0.690, 0.897] | [0.555, 0.841] | [0.945, 0.999] |
| **K-Nearest Neighbors** | [0.893, 1.000] | [0.516, 0.852] | [0.724, 0.914] | [0.600, 0.882] | [0.942, 0.999] |
| **Random Forest** | [0.893, 1.000] | **[0.828, 1.000]** | **[0.897, 1.000]** | **[0.828, 1.000]** | **[0.989, 1.000]** |
| **Soft Voting Ensemble** | **[1.000, 1.000]** | [0.560, 0.885] | [0.759, 0.948] | [0.636, 0.912] | [0.962, 1.000] |

---

### 5.5 Pairwise McNemar's Statistical Significance Tests

To formally test whether algorithmic performance differed beyond random noise, continuity-corrected McNemar tests were conducted across all model pairs:

**Table 4: Continuity-Corrected McNemar Significance Matrix**
| Comparison Pair | Discordant $b$ ($M_A \checkmark, M_B \times$) | Discordant $c$ ($M_B \checkmark, M_A \times$) | $\chi^2$ Statistic | $p$-value | Significant ($p < 0.05$)? |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression vs. Random Forest** | 1 | 10 | 5.8182 | **0.0159** | **Yes (Statistically Significant)** |
| **K-Nearest Neighbors vs. Random Forest** | 0 | 7 | 5.1429 | **0.0233** | **Yes (Statistically Significant)** |
| **Logistic Regression vs. KNN** | 1 | 3 | 0.2500 | 0.6171 | No |
| **Logistic Regression vs. Voting Ensemble** | 0 | 4 | 2.2500 | 0.1336 | No |
| **Random Forest vs. Voting Ensemble** | 6 | 1 | 2.2857 | 0.1306 | No |
| **KNN vs. Voting Ensemble** | 0 | 2 | 0.5000 | 0.4795 | No |

*Interpretation:* The differences between the linear classifier (Logistic Regression) and the tree ensemble (Random Forest), as well as between KNN and Random Forest, are **statistically significant** ($p < 0.05$). This confirms that the distinct mathematical paradigms produce genuinely divergent classification surfaces rather than stochastic sampling artifacts.

---

### 5.6 Multi-Feature Set Comparison: Behavioral Items vs. Demographics

**Table 5: Performance Across Feature Sets A, B, and C (Repeated 5-Fold CV)**
| Feature Set | Model | CV Recall (Mean ± Std) | CV Accuracy | CV ROC-AUC | CV F1-Score |
|---|---|:---:|:---:|:---:|:---:|
| **Set A (All Features)** | Logistic Regression | 0.9803 ± 0.0338 | 0.9415 ± 0.0315 | 0.9894 ± 0.0116 | 0.9426 ± 0.0279 |
| **Set A (All Features)** | K-Nearest Neighbors | 0.9466 ± 0.0397 | 0.8399 ± 0.0619 | 0.9229 ± 0.0362 | 0.8532 ± 0.0493 |
| **Set A (All Features)** | Random Forest | 0.9461 ± 0.0287 | 0.9198 ± 0.0347 | 0.9790 ± 0.0135 | 0.9199 ± 0.0335 |
| **Set B (Questions Only)** | Logistic Regression | 0.9983 ± 0.0085 | 0.9466 ± 0.0304 | 0.9931 ± 0.0058 | 0.9483 ± 0.0286 |
| **Set B (Questions Only)** | K-Nearest Neighbors | 0.9445 ± 0.0431 | 0.9017 ± 0.0418 | 0.9730 ± 0.0198 | 0.9036 ± 0.0394 |
| **Set B (Questions Only)** | Random Forest | 0.9373 ± 0.0438 | 0.9241 ± 0.0412 | 0.9846 ± 0.0142 | 0.9232 ± 0.0402 |
| **Set C (Demographics Only)** | Logistic Regression | 0.5504 ± 0.0886 | 0.5873 ± 0.0574 | 0.6140 ± 0.0588 | 0.5608 ± 0.0628 |
| **Set C (Demographics Only)** | K-Nearest Neighbors | 0.5322 ± 0.1205 | 0.5137 ± 0.0876 | 0.5292 ± 0.0902 | 0.5109 ± 0.0978 |
| **Set C (Demographics Only)** | Random Forest | 0.5351 ± 0.1171 | 0.5629 ± 0.0756 | 0.5799 ± 0.0723 | 0.5381 ± 0.0951 |

*Crucial Scientific Finding:* When trained exclusively on demographic variables (Set C: age, sex, ethnicity, jaundice, family history), all models collapse to near-chance discrimination (53.2% to 55.0% recall, 51.4% to 58.7% accuracy). This provides rigorous empirical proof that ASD cannot be screened via demographic profiles alone, and that the ten behavioral questions drive virtually 100% of the diagnostic signal.

---

## 6. Discussion and Error Analysis

### 6.1 Trade-Off Analysis: Accuracy vs. Sensitivity
Our findings reveal an essential trade-off between **high sensitivity (screening objective)** and **high precision/specificity (diagnostic objective)**:
- **Logistic Regression** acts as a hyper-sensitive screening filter: with tuned L2 regularization ($C = 0.01$), it achieved **zero false negatives** (Recall = $1.000$) on the test set, but produced 12 false positives (Specificity = $0.6000$).
- **Random Forest** acts as a highly conservative diagnostic classifier: it achieved the highest accuracy ($94.83\%$), highest specificity ($93.33\%$), and an exceptional ROC-AUC ($0.9976$), with only 2 false positives, but missed 1 positive child (Recall = $0.9643$).
- **The Soft Voting Ensemble** successfully navigates this trade-off: it preserves the **100% Recall** of Logistic Regression (0 False Negatives) while improving Specificity to $73.33\%$ (reducing false positives by 33% compared to standalone Logistic Regression).

### 6.2 Granular Error Autopsy: The Single False Negative (Child #Test-42)
On the held-out test split, only one child with ASD was missed by any model:
- **Profile:** 7-year-old male, Middle Eastern heritage, residence in Jordan, positive history of neonatal jaundice, no family history of ASD.
- **Behavioral Scores:** `A1=1, A2=0, A3=1, A4=0, A5=1, A6=1, A7=1, A8=1, A9=0, A10=1`.
- **Sum of Scores:** Exactly **7**—the knife-edge clinical cutoff score.
- **Why Did RF and KNN Miss?**
  - Random Forest splits heavily on `A2_Score` (response to name) and `A4_Score` (shared enjoyment). Because this child had `A2=0` and `A4=0`, tree paths assigned a posterior probability of **$0.498$**—missing the $0.50$ decision threshold by an agonizing $0.002$.
  - KNN's distance metric was confounded by demographic neighbors in Jordan without ASD, resulting in a probability of **$0.478$**.
- **Why Did Logistic Regression Catch the Child?**
  - The linear log-odds model independently weighted `A1=1, A5=1, A6=1, A7=1, A8=1, A10=1` additively, yielding a probability of **$0.527$**.
- **Ensemble Rescue:** Averaging $(0.527 + 0.478 + 0.498)/3$ produced **$0.501$**, crossing the threshold and rescuing the child from being a missed case.

### 6.3 Clinical Decision Threshold Optimization ($\tau = 0.40$)
In pediatric medicine, enforcing $\tau = 0.50$ is an arbitrary mathematical convention. By sweeping decision thresholds across our test set probabilities (`results/figures/threshold_analysis.png`), we established that setting:
$$\tau = 0.40$$
guarantees **100% Sensitivity (Recall = 1.000) across all models** (including Random Forest), while maintaining high specificity ($70.0\%$ to $86.7\%$). In practice, an automated screening app calibrated to $\tau = 0.40$ ensures zero missed cases while routing only a modest number of non-ASD cases to primary care follow-up.

---

## 7. Limitations and Ethical Considerations

1. **Small Cohort Size ($N = 292$):** While representative of pediatric screening studies, a 292-instance cohort limits deep stratified demographic slicing (e.g., assessing fairness across individual ethnic subgroups).
2. **App-Collected Public Data:** Data was gathered through mobile app self-reports by parents rather than in-clinic observations. While this mirrors real-world digital health screening conditions, parent reporting bias cannot be entirely excluded.
3. **Screening, Not Diagnosis:** We reiterate that high algorithmic scores indicate the *presence of behavioral risk markers* warranting full developmental evaluation, not a definitive neurological diagnosis.
4. **Fairness and Demographic Parity:** Although our Set C control proved that demographics do not drive classification, algorithms deployed in clinical settings must be continuously audited for demographic equity across socioeconomic and cultural groups.

---

## 8. Conclusion and Future Work

This research provides a rigorous, leak-free comparative benchmark of machine learning classifiers for pediatric Autism Spectrum Disorder screening. We empirically demonstrated that:
1. Target leakage through summary score columns invalidates unconstrained models and must be systematically excised.
2. Demographic attributes alone provide negligible screening utility ($\approx 53.5\%$ recall), proving that behavioral question items are essential.
3. **Logistic Regression** and the **Soft Voting Ensemble** are the most clinically effective screening architectures, achieving **100% Recall** on held-out test data.
4. Performance differences between linear models and tree ensembles are statistically significant (McNemar's $p = 0.0159$).
5. Calibrating the decision threshold to $\tau = 0.40$ provides an optimal clinical operating point, eliminating false negatives across all models.

### Future Work
Future extensions of this research will evaluate model calibration (Platt scaling and isotonic regression), expand multi-center validation on larger clinical cohorts, and package the Soft Voting Ensemble into an offline-first, privacy-preserving mobile progressive web app (PWA) for rural Cambodian community healthcare workers.

---

## 9. References

1. F. Thabtah, "Autism Spectrum Disorder Screening Data for Children," UCI Machine Learning Repository, 2017. DOI: [10.24432/C5659W](https://doi.org/10.24432/C5659W).
2. F. Thabtah, "An accessible and efficient autism screening method using machine learning," *International Journal of Medical Informatics*, vol. 136, p. 104088, 2020.
3. F. Thabtah, "Machine Learning in Autistic Spectrum Disorder Behavioral Research: A Review and Ways Forward," *Informatics for Health and Social Care*, vol. 44, no. 3, pp. 278–297, 2019.
4. C. Allison et al., "The Quantitative Checklist for Autism in Toddlers (Q-CHAT): A normally distributed quantitative measure of autistic traits at 18-24 months of age: preliminary report," *Journal of Autism and Developmental Disorders*, vol. 38, no. 8, pp. 1414–1425, 2008.
5. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
6. T. M. Cover and P. E. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
7. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. Hoboken, NJ: John Wiley & Sons, 2013.
8. M. J. Maenner et al., "Prevalence and Characteristics of Autism Spectrum Disorder Among Children Aged 8 Years — Autism and Developmental Disabilities Monitoring Network, 11 Sites, United States, 2020," *MMWR Surveillance Summaries*, vol. 72, no. 2, pp. 1–14, 2023.
9. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.
