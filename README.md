# ITM-390 ASD Screening Project: Comparative Analysis of Machine Learning Models

*Comparative Analysis of Machine Learning Models for Autism Spectrum Disorder Screening in Children*

**Group 10** · **American University of Phnom Penh (AUPP)** · **School of Digital Technologies**  
**Course:** ITM-390 Machine Learning · **Advisor:** Kuntha PIN

---

## 1. Project Overview

Early detection of Autism Spectrum Disorder (ASD) in children enables timely behavioral interventions that significantly improve long-term developmental trajectories. However, formal clinical diagnosis is often delayed due to lengthy specialist waitlists and expensive assessment batteries. 

This project establishes a standardized, reproducible machine learning screening framework evaluating three supervised classification algorithms—**Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**—alongside a **Soft Voting Ensemble** using the UCI *Autism Spectrum Disorder Screening Data for Children* dataset (292 children, 20 features, binary screening label).

> **Screening vs. Diagnosis Notice:**  
> This study develops an accessible **screening tool**, not a diagnostic instrument. Because a missed case (false negative) delays clinical intervention, our evaluation prioritizes **Recall / Sensitivity** as the primary metric (alongside Specificity, Accuracy, Precision, F1-Score, and ROC-AUC) and conducts detailed error profiling.

### Research Question
> *How do Logistic Regression, K-Nearest Neighbors, and Random Forest compare in screening for ASD among children using behavioral and demographic features?*

### Hypothesis
> *The three models will produce different screening performance because they learn decision boundaries in fundamentally different ways (linear hyperplane, distance-based neighborhood, and non-linear tree ensemble).*

---

## 2. Team & Responsibilities (Group 10)

| Member | Role | Deliverables |
|---|---|---|
| **Lyching KONG** (Leader) | Research & Model Comparison | Literature review, comparative framework, error analysis, final conclusions |
| **Lishang KEAV** | Dataset Analysis & Preprocessing | Exploratory data analysis (EDA), data cleaning, leakage prevention, shared pipeline |
| **Sienge HENG** | Model Training & Evaluation | Pipeline architectures, hyperparameter tuning, cross-validation, evaluation suite |
| **Lydana TAN** | Documentation & Quality Assurance | Technical report, slides, unit test suite (`pytest`), reproducibility checks |

---

## 3. Methodological Rigor & Leakage Prevention

Per Section 3 of the project plan, three critical methodological risks were proactively solved:

1. **Target Leakage Elimination (`result` column):**  
   The raw dataset contains 10 screening question scores (`A1_Score` to `A10_Score`) and a `result` column that represents their direct sum. Because the target label `Class/ASD` was assigned deterministically when `result >= 7`, retaining `result` introduces 100% target leakage. **`result` is strictly dropped from all feature matrices.**
2. **Multi-Feature Set Benchmark (Sets A, B, and C):**  
   To rigorously measure feature importance and confirm that behavioral items drive predictions, all algorithms are evaluated across three feature configurations:
   - **Set A (Primary Experiment):** `A1` to `A10` + Demographics (`age`, `gender`, `ethnicity`, `jaundice`, `family_asd`, `country_of_res`, `used_app_before`, `relation`).
   - **Set B (Screening Items Only):** `A1` to `A10` behavioral scores only.
   - **Set C (Demographic Control):** Demographic features only (proves behavioral signal necessity).
3. **Repeated Cross-Validation & Bootstrap Uncertainty:**  
   Given the sample size ($N = 292$), a single test split is subject to sampling variance. We apply **Repeated Stratified 5-Fold Cross-Validation (5 repeats = 25 validation folds)** on training data and report **1,000-sample non-parametric bootstrap 95% confidence intervals** on the held-out test split ($N = 58$).
4. **Data Preprocessing & Constant Column Removal:**  
   - Converted `'?'` characters to `np.nan`.
   - Dropped constant attribute `age_desc` (`'4-11 years'` for all children).
   - Normalized category casing (e.g. `'self'` $\to$ `'Self'`, stripped trailing whitespace).
   - Encapsulated preprocessing inside a leak-free `ColumnTransformer` with `SimpleImputer(median)` and `StandardScaler` for numeric features, and `SimpleImputer(most_frequent)` and `OneHotEncoder(handle_unknown='ignore')` for categorical features. Preprocessors are strictly fit on training splits only.

---

## 4. Repository Structure

```
ITM-390-ASD-Screening/
├── .gitignore
├── .gitattributes
├── README.md                          # Full academic documentation & reproduction guide
├── pyproject.toml                     # Modern Python packaging configuration
├── requirements.txt                   # Pinned project dependencies
├── main.py                            # End-to-end orchestration CLI script
├── .vscode/
│   ├── settings.json                  # VS Code python interpreter & test settings
│   └── launch.json                    # VS Code debugging configurations
├── data/
│   ├── raw/
│   │   ├── Autism-Child-Data.arff     # Original untouched UCI dataset
│   │   ├── Autism-Child-Data.csv      # Converted CSV format
│   │   ├── data_dictionary.md         # Detailed attribute provenance & descriptions
│   │   └── Autism-Screening-Child-Data-Description.docx
│   └── processed/
│       ├── train.csv / test.csv       # Stratified 80/20 raw splits
│       ├── X_train.csv / y_train.csv
│       └── X_test.csv / y_test.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Data cleaning, feature sets, ColumnTransformer
│   ├── models.py                      # Model pipelines, GridSearchCV grids, Voting Ensemble
│   ├── evaluate.py                    # Screening metrics, bootstrap CI, McNemar test, plots
│   └── utils.py                       # Random seed controls, table formatters, JSON helpers
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory analysis & target leakage demonstration
│   ├── 02_models.ipynb                # Tuning, KNN K-sweep curve, Voting Ensemble
│   └── 03_evaluation.ipynb            # Held-out testing, threshold analysis & error profiling
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py          # Preprocessor & leakage verification unit tests
│   └── test_evaluate.py               # Metrics & statistical tests unit tests
└── results/
    ├── figures/                       # Publication-grade figures (ROC, PR, Thresholds, Heatmaps)
    ├── metrics/                       # CV & test metric tables (CSV & JSON)
    └── models/                        # Serialized tuned pipelines (joblib)
```

---

## 5. Experimental Results Summary

### 5.1 Repeated Stratified 5-Fold Cross-Validation (Training Folds, 25 Folds)

*Scored primarily on Recall / Sensitivity.*

| Model | CV Recall (Mean ± Std) | CV Accuracy (Mean ± Std) | CV ROC-AUC (Mean ± Std) | CV Precision (Mean ± Std) | CV F1 (Mean ± Std) |
|---|---|---|---|---|---|
| **Logistic Regression** | **1.0000 ± 0.0000** | 0.7700 ± 0.0617 | 0.9547 ± 0.0321 | 0.6824 ± 0.0587 | 0.8098 ± 0.0412 |
| **K-Nearest Neighbors** | 0.9982 ± 0.0089 | 0.8423 ± 0.0528 | 0.9714 ± 0.0165 | 0.7596 ± 0.0621 | 0.8613 ± 0.0400 |
| **Random Forest** | 0.9676 ± 0.0301 | **0.9181 ± 0.0448** | **0.9825 ± 0.0131** | **0.8796 ± 0.0632** | **0.9204 ± 0.0416** |
| **Soft Voting Ensemble** | **0.9946 ± 0.0146** | 0.8587 ± 0.0533 | **0.9818 ± 0.0122** | 0.7816 ± 0.0658 | 0.8738 ± 0.0416 |

---

### 5.2 Held-Out Test Set Performance (20% Split, N=58)

Evaluated once on untouched held-out data:

| Model | Recall (Sensitivity) | Specificity | Accuracy | Precision | F1-Score | ROC-AUC | TP | FP | TN | FN | Latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **1.0000** | 0.6000 | 0.7931 | 0.7000 | 0.8235 | 0.9798 | 28 | 12 | 18 | **0** | 0.20 ms |
| **K-Nearest Neighbors** | 0.9643 | 0.7000 | 0.8276 | 0.7500 | 0.8437 | 0.9786 | 27 | 9 | 21 | 1 | 0.25 ms |
| **Random Forest** | 0.9643 | **0.9333** | **0.9483** | **0.9310** | **0.9474** | **0.9976** | 27 | 2 | 28 | 1 | 0.42 ms |
| **Soft Voting Ensemble** | **1.0000** | 0.7333 | 0.8621 | 0.7778 | 0.8750 | 0.9881 | 28 | 8 | 22 | **0** | 0.44 ms |

---

### 5.3 Non-Parametric Bootstrap 95% Confidence Intervals (1,000 Resamples)

| Model | Recall 95% CI | Specificity 95% CI | Accuracy 95% CI | Precision 95% CI | ROC-AUC 95% CI |
|---|---|---|---|---|---|
| **Logistic Regression** | **[1.000, 1.000]** | [0.414, 0.769] | [0.690, 0.897] | [0.555, 0.841] | [0.945, 0.999] |
| **K-Nearest Neighbors** | [0.893, 1.000] | [0.516, 0.852] | [0.724, 0.914] | [0.600, 0.882] | [0.942, 0.999] |
| **Random Forest** | [0.893, 1.000] | **[0.828, 1.000]** | **[0.897, 1.000]** | **[0.828, 1.000]** | **[0.989, 1.000]** |
| **Soft Voting Ensemble** | **[1.000, 1.000]** | [0.560, 0.885] | [0.759, 0.948] | [0.636, 0.912] | [0.962, 1.000] |

---

### 5.4 Pairwise McNemar's Significance Tests (Continuity Corrected)

Tests whether the observed prediction differences between algorithm pairs are statistically significant:

| Model Pair | Discordant (A correct, B error) | Discordant (B correct, A error) | $\chi^2$ Statistic | $p$-value | Significant ($p < 0.05$) |
|---|---|---|---|---|---|
| **Logistic Regression vs. Random Forest** | 1 | 10 | 5.8182 | **0.0159** | **Yes (Significant)** |
| **KNN vs. Random Forest** | 0 | 7 | 5.1429 | **0.0233** | **Yes (Significant)** |
| **Logistic Regression vs. KNN** | 1 | 3 | 0.2500 | 0.6171 | No |
| **Random Forest vs. Voting Ensemble** | 6 | 1 | 2.2857 | 0.1306 | No |

---

### 5.5 Multi-Feature Set Comparison: Behavioral Items vs. Demographics

| Feature Set | Model | CV Recall (Mean ± Std) | CV Accuracy | CV ROC-AUC |
|---|---|---|---|---|
| **Set A (All Features)** | Logistic Regression | 0.9803 ± 0.0338 | 0.9415 ± 0.0315 | 0.9894 ± 0.0116 |
| **Set A (All Features)** | K-Nearest Neighbors | 0.9466 ± 0.0397 | 0.8399 ± 0.0619 | 0.9229 ± 0.0362 |
| **Set A (All Features)** | Random Forest | 0.9461 ± 0.0287 | 0.9198 ± 0.0347 | 0.9790 ± 0.0135 |
| **Set B (Questions Only)** | Logistic Regression | 0.9983 ± 0.0085 | 0.9466 ± 0.0304 | 0.9931 ± 0.0058 |
| **Set B (Questions Only)** | K-Nearest Neighbors | 0.9445 ± 0.0431 | 0.9017 ± 0.0418 | 0.9730 ± 0.0198 |
| **Set B (Questions Only)** | Random Forest | 0.9373 ± 0.0438 | 0.9241 ± 0.0412 | 0.9846 ± 0.0142 |
| **Set C (Demographics Only)** | Logistic Regression | 0.5504 ± 0.0886 | 0.5873 ± 0.0574 | 0.6140 ± 0.0588 |
| **Set C (Demographics Only)** | K-Nearest Neighbors | 0.5322 ± 0.1205 | 0.5137 ± 0.0876 | 0.5292 ± 0.0902 |
| **Set C (Demographics Only)** | Random Forest | 0.5351 ± 0.1171 | 0.5629 ± 0.0756 | 0.5799 ± 0.0723 |

> **Methodological Takeaway:** Models trained on Set C (demographics alone) drop to near chance (~53-55% recall), formally proving that the Q-CHAT-10 screening questions provide virtually all discriminative power.

---

## 6. Answers to Research Question and Hypothesis

1. **Answer to Research Question:**
   - In pediatric screening where **avoiding false negatives is the priority**, **Logistic Regression** and the **Soft Voting Ensemble** demonstrated superior screening efficacy, achieving **100% Recall (0 False Negatives)** on the held-out test split.
   - For balanced diagnostic precision, **Random Forest** achieved the highest overall accuracy (**94.83%**), highest specificity (**93.33%**), and highest ROC-AUC (**0.9976**), though it produced 1 false negative under default 0.50 thresholding.
   - **Ensemble Recommendation:** The Soft Voting Ensemble successfully blends the linear sensitivity of Logistic Regression with the non-linear boundaries of Random Forest, achieving 100% sensitivity with higher specificity (73.33%) than standalone Logistic Regression (60.00%).

2. **Hypothesis Evaluation:**
   - **Hypothesis Supported:** The algorithms produced distinct decision boundaries that led to statistically significant differences in prediction error patterns, as demonstrated by McNemar's test ($p = 0.0159$ between Logistic Regression and Random Forest; $p = 0.0233$ between KNN and Random Forest).

---

## 7. How to Run & Reproduce

### Prerequisites
- Python 3.11+
- Git

### Installation
Clone the repository and set up the virtual environment:

```powershell
# Clone the repository
git clone https://github.com/hengsienge/ITM-390-ASD-Screening.git
cd ITM-390-ASD-Screening

# Using uv (fastest):
uv venv .venv --python 3.11
uv pip install -r requirements.txt

# Or using standard python venv:
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Execute Full End-to-End Pipeline
Run the entire workflow (data loading, stratified splitting, CV tuning, test set evaluation, bootstrap intervals, McNemar tests, figures):

```powershell
.\.venv\Scripts\python.exe main.py
```

### 2. Run Automated Unit Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```

### 3. Launch Interactive Notebooks
```powershell
.\.venv\Scripts\jupyter.exe lab
```
Navigate to `notebooks/`:
- `01_eda.ipynb` (Exploratory Data Analysis & Leakage Proof)
- `02_models.ipynb` (Model Tuning & K-Sweep Curve)
- `03_evaluation.ipynb` (Final Test Set Evaluation, ROC curves, Error Profiling)

---

## 8. References

1. F. Thabtah, "Autism Spectrum Disorder Screening Data for Children," UCI Machine Learning Repository, 2017. DOI: [10.24432/C5659W](https://doi.org/10.24432/C5659W).
2. F. Thabtah, "An accessible and efficient autism screening method using machine learning," *International Journal of Medical Informatics*, vol. 136, 2020.
3. F. Thabtah, "Machine Learning in Autistic Spectrum Disorder Behavioral Research: A Review and Ways Forward," *Informatics for Health and Social Care*, vol. 44, no. 3, pp. 278–297, 2019.
4. L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.
5. T. M. Cover and P. E. Hart, "Nearest neighbor pattern classification," *IEEE Transactions on Information Theory*, vol. 13, no. 1, pp. 21–27, 1967.
6. D. W. Hosmer, S. Lemeshow, and R. X. Sturdivant, *Applied Logistic Regression*, 3rd ed. Hoboken, NJ: Wiley, 2013.
