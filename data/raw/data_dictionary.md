# Dataset Dictionary: Autism Spectrum Disorder Screening in Children

## Provenance
- **Source:** UCI Machine Learning Repository
- **Citation:** Fadi Thabtah (2017). *Autism Spectrum Disorder Screening Data for Children*. UCI Machine Learning Repository. DOI: [10.24432/C5659W](https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children).
- **Instruments:** Based on the AQ-10 (Autism Spectrum Quotient - 10 items) Child version / Q-CHAT-10.
- **Instances:** 292 children
- **Target:** `Class/ASD` (binary: `'YES'` / `'NO'`), encoded as `1` for ASD, `0` for non-ASD.

---

## Feature Descriptions

| Feature Name | Type | Range / Values | Description |
|---|---|---|---|
| `A1_Score` | Binary | 0, 1 | Q-CHAT-10 Item 1: Saccadic / eye contact |
| `A2_Score` | Binary | 0, 1 | Q-CHAT-10 Item 2: Response to name |
| `A3_Score` | Binary | 0, 1 | Q-CHAT-10 Item 3: Pointing to indicate interest |
| `A4_Score` | Binary | 0, 1 | Q-CHAT-10 Item 4: Shared enjoyment / interest |
| `A5_Score` | Binary | 0, 1 | Q-CHAT-10 Item 5: Pretend play |
| `A6_Score` | Binary | 0, 1 | Q-CHAT-10 Item 6: Following a gaze / point |
| `A7_Score` | Binary | 0, 1 | Q-CHAT-10 Item 7: Comforting / empathy behaviors |
| `A8_Score` | Binary | 0, 1 | Q-CHAT-10 Item 8: Spontaneous communicative gestures |
| `A9_Score` | Binary | 0, 1 | Q-CHAT-10 Item 9: Simple communicative sounds / words |
| `A10_Score` | Binary | 0, 1 | Q-CHAT-10 Item 10: Repetitive motor movements / staring |
| `age` | Numeric | 4 - 11 | Age of the child in years |
| `gender` | Categorical | `'m'`, `'f'` | Child biological sex |
| `ethnicity` | Categorical | `'White-European'`, `'Asian'`, `'Middle Eastern'`, `'Black'`, `'South Asian'`, `'Hispanic'`, `'Pasifika'`, `'Turkish'`, `'Latino'`, `'Others'`, `?` | Ethnic heritage |
| `jaundice` | Binary | `'yes'`, `'no'` | Whether the child was born with jaundice (neonatal jaundice) |
| `family_asd` | Binary | `'yes'`, `'no'` | Whether an immediate family member has diagnosed ASD |
| `country_of_res` | Categorical | Various countries | Country of primary residence |
| `used_app_before` | Binary | `'yes'`, `'no'` | Whether the respondent previously used the screening app |
| `result` | Numeric | 0 - 10 | **Target Leakage Column**: Sum of `A1_Score` through `A10_Score`. (Dropped from all feature sets) |
| `age_desc` | Categorical | `'4-11 years'` | Constant column across all 292 rows. (Dropped from all feature sets) |
| `relation` | Categorical | `'Parent'`, `'Self'`, `'Relative'`, `'Health care professional'`, `?` | Relationship of the respondent filling out the assessment |
| `Class/ASD` | Target | `'YES'`, `'NO'` | Final binary screening classification label |

---

## Experimental Feature Sets (Section 3 & 6 of Plan)

1. **Feature Set A (Primary Experiment):**
   - 10 Behavioral questions (`A1_Score` .. `A10_Score`) + 8 Demographic & Background features (`age`, `gender`, `ethnicity`, `jaundice`, `family_asd`, `country_of_res`, `used_app_before`, `relation`).
2. **Feature Set B (Behavioral Screening Questions Only):**
   - 10 Behavioral questions (`A1_Score` .. `A10_Score`).
3. **Feature Set C (Demographic Control Experiment):**
   - 8 Demographic & Background features (`age`, `gender`, `ethnicity`, `jaundice`, `family_asd`, `country_of_res`, `used_app_before`, `relation`).
