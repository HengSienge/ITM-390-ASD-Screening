"""
Shared Data Preprocessing Pipeline for ITM-390 ASD Screening Project.

Enforces:
- Strict target leakage prevention (removes 'result' column).
- Shared ColumnTransformer with SimpleImputer, OneHotEncoder, and StandardScaler.
- Stratified 80/20 train/test splitting.
- Support for Feature Sets A, B, and C.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Canonical column definitions
QUESTION_COLS = [f"A{i}_Score" for i in range(1, 11)]

DEMOGRAPHIC_COLS = [
    "age",
    "gender",
    "ethnicity",
    "jaundice",
    "family_asd",
    "country_of_res",
    "used_app_before",
    "relation",
]

TARGET_COL = "asd_target"
LEAKAGE_COL = "result"
CONSTANT_COLS = ["age_desc"]

COLUMN_RENAME_MAP = {
    "jundice": "jaundice",
    "austim": "family_asd",
    "contry_of_res": "country_of_res",
    "Class/ASD": TARGET_COL,
    "class": TARGET_COL,
}


def load_arff_to_df(arff_path: Union[str, Path]) -> pd.DataFrame:
    """Parse UCI ARFF format file into a pandas DataFrame."""
    arff_path = Path(arff_path)
    data_lines = []
    attributes = []
    reading_data = False

    with open(arff_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue

            lower = stripped.lower()
            if lower.startswith("@attribute"):
                parts = stripped.split()
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
            elif lower.startswith("@data"):
                reading_data = True
            elif reading_data:
                data_lines.append(stripped)

    records = []
    for line in data_lines:
        tokens = []
        in_quotes = False
        current_token = []
        for char in line:
            if char in ("'", '"'):
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                tokens.append("".join(current_token).strip())
                current_token = []
            else:
                current_token.append(char)
        tokens.append("".join(current_token).strip())
        records.append(tokens)

    return pd.DataFrame(records, columns=attributes)


def load_raw_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load raw dataset from either ARFF or CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")

    if path.suffix.lower() == ".arff":
        return load_arff_to_df(path)
    return pd.read_csv(path)


def clean_raw_data(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """
    Clean raw dataset:
    - Standardizes column names
    - Replaces '?' with np.nan
    - Strips whitespace and standardizes inconsistent categories
    - Drops constant column 'age_desc'
    - Encodes target as 0/1
    - Enforces target leakage prevention by dropping 'result' column
    """
    cleaned = df.copy()

    # Standardize column names
    cleaned = cleaned.rename(columns=COLUMN_RENAME_MAP)

    # Clean whitespace and strip quotes
    for col in cleaned.columns:
        if cleaned[col].dtype == object or isinstance(cleaned[col].iloc[0], str):
            cleaned[col] = cleaned[col].astype(str).str.strip("'\" \t\r\n")

    # Replace '?' with NaN
    cleaned = cleaned.replace({"?": np.nan, "nan": np.nan, "None": np.nan})

    # Standardize specific inconsistencies
    if "relation" in cleaned.columns:
        cleaned["relation"] = cleaned["relation"].replace({"self": "Self"})

    if "gender" in cleaned.columns:
        cleaned["gender"] = cleaned["gender"].str.lower()

    if "jaundice" in cleaned.columns:
        cleaned["jaundice"] = cleaned["jaundice"].str.lower()

    if "family_asd" in cleaned.columns:
        cleaned["family_asd"] = cleaned["family_asd"].str.lower()

    if "used_app_before" in cleaned.columns:
        cleaned["used_app_before"] = cleaned["used_app_before"].str.lower()

    # Drop constant columns
    cols_to_drop = [c for c in CONSTANT_COLS if c in cleaned.columns]
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)

    # Drop duplicates
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Convert numeric fields
    numeric_cols = [c for c in QUESTION_COLS if c in cleaned.columns] + ["age"]
    if "result" in cleaned.columns:
        numeric_cols.append("result")

    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Encode binary target: 'YES' -> 1, 'NO' -> 0
    if TARGET_COL in cleaned.columns:
        cleaned[TARGET_COL] = (
            cleaned[TARGET_COL]
            .astype(str)
            .str.upper()
            .map({"YES": 1, "NO": 0, "1": 1, "0": 0})
        )

    # Enforce Target Leakage Policy: drop 'result'
    if drop_leakage and LEAKAGE_COL in cleaned.columns:
        cleaned = cleaned.drop(columns=[LEAKAGE_COL])

    return cleaned


def get_feature_sets() -> Dict[str, List[str]]:
    """Return dictionary of feature sets A, B, and C as defined in the plan."""
    return {
        "Set A": QUESTION_COLS + DEMOGRAPHIC_COLS,  # Primary experiment
        "Set B": QUESTION_COLS,                     # Behavioral items only
        "Set C": DEMOGRAPHIC_COLS,                  # Demographics control
    }


def build_preprocessor(feature_cols: List[str]) -> ColumnTransformer:
    """
    Build a leak-free scikit-learn ColumnTransformer.
    - Numerical features: SimpleImputer(median) + StandardScaler
    - Categorical features: SimpleImputer(most_frequent) + OneHotEncoder
    - Binary questions: SimpleImputer(most_frequent) (passed without scaling)
    """
    num_features = [c for c in feature_cols if c == "age"]
    bin_features = [c for c in feature_cols if c in QUESTION_COLS]
    cat_features = [c for c in feature_cols if c in DEMOGRAPHIC_COLS and c != "age"]

    transformers = []

    if num_features:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_features))

    if bin_features:
        bin_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])
        transformers.append(("bin", bin_pipeline, bin_features))

    if cat_features:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_test_split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split.
    Fits nothing; returns raw split dataframes ready for pipeline consumption.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
