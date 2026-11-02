"""
Unit tests for data preprocessing and leakage prevention in ITM-390 ASD Screening.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    CONSTANT_COLS,
    DEMOGRAPHIC_COLS,
    LEAKAGE_COL,
    QUESTION_COLS,
    TARGET_COL,
    build_preprocessor,
    clean_raw_data,
    get_feature_sets,
    load_raw_data,
    train_test_split_dataset,
)


@pytest.fixture
def raw_data_path():
    path = Path(__file__).resolve().parent.parent / "data" / "raw" / "Autism-Child-Data.arff"
    if not path.exists():
        pytest.skip(f"Raw data file not found at {path}")
    return path


@pytest.fixture
def sample_raw_df():
    """Mock raw DataFrame with known quirks."""
    return pd.DataFrame({
        "A1_Score": ["1", "0", "1"],
        "A2_Score": ["1", "0", "0"],
        "A3_Score": ["0", "1", "1"],
        "A4_Score": ["0", "0", "1"],
        "A5_Score": ["1", "1", "1"],
        "A6_Score": ["1", "0", "1"],
        "A7_Score": ["0", "1", "0"],
        "A8_Score": ["1", "0", "1"],
        "A9_Score": ["0", "0", "0"],
        "A10_Score": ["0", "1", "1"],
        "age": ["6", "?", "11"],
        "gender": ["m", "f", "m"],
        "ethnicity": ["Others", "Middle Eastern ", "?"],
        "jundice": ["no", "yes", "no"],
        "austim": ["no", "no", "yes"],
        "contry_of_res": ["Jordan", "United States", "Egypt"],
        "used_app_before": ["no", "no", "yes"],
        "result": ["5", "4", "7"],
        "age_desc": ["4-11 years", "4-11 years", "4-11 years"],
        "relation": ["Parent", "?", "self"],
        "Class/ASD": ["NO", "NO", "YES"],
    })


def test_clean_raw_data_removes_leakage_and_constants(sample_raw_df):
    cleaned = clean_raw_data(sample_raw_df, drop_leakage=True)

    # Leakage column 'result' must be removed
    assert LEAKAGE_COL not in cleaned.columns

    # Constant column 'age_desc' must be removed
    for col in CONSTANT_COLS:
        assert col not in cleaned.columns

    # Check renamed columns
    assert "jaundice" in cleaned.columns
    assert "family_asd" in cleaned.columns
    assert "country_of_res" in cleaned.columns
    assert TARGET_COL in cleaned.columns

    # Check missing values replacement
    assert np.isnan(cleaned.loc[1, "age"])
    assert pd.isna(cleaned.loc[2, "ethnicity"])
    assert pd.isna(cleaned.loc[1, "relation"])

    # Check relation casing fix
    assert cleaned.loc[2, "relation"] == "Self"

    # Check target encoding
    assert list(cleaned[TARGET_COL]) == [0, 0, 1]


def test_feature_sets_definitions():
    sets = get_feature_sets()
    assert "Set A" in sets
    assert "Set B" in sets
    assert "Set C" in sets

    # Set A must have all questions + demographics
    assert set(sets["Set A"]) == set(QUESTION_COLS + DEMOGRAPHIC_COLS)
    # Set B must only have 10 screening questions
    assert set(sets["Set B"]) == set(QUESTION_COLS)
    assert len(sets["Set B"]) == 10
    # Set C must only have demographics
    assert set(sets["Set C"]) == set(DEMOGRAPHIC_COLS)
    # Leakage column must not be in any set
    for s_name, s_cols in sets.items():
        assert LEAKAGE_COL not in s_cols
        assert TARGET_COL not in s_cols


def test_build_preprocessor_transformation(sample_raw_df):
    cleaned = clean_raw_data(sample_raw_df, drop_leakage=True)
    feature_cols = get_feature_sets()["Set A"]

    preprocessor = build_preprocessor(feature_cols)
    transformed = preprocessor.fit_transform(cleaned[feature_cols])

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any(), "Transformed output must have no NaNs after imputation."
    assert transformed.shape[0] == len(cleaned)


def test_no_data_leakage_between_train_and_test(sample_raw_df):
    # Create distinct records so deduplication doesn't drop them
    records = []
    for i in range(20):
        row = sample_raw_df.iloc[i % len(sample_raw_df)].copy()
        row["age"] = str(4 + (i % 7))
        records.append(row)
    expanded = pd.DataFrame(records)
    cleaned = clean_raw_data(expanded, drop_leakage=True)

    train_df, test_df = train_test_split_dataset(cleaned, test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == len(cleaned)

    # Fit preprocessor only on train
    feature_cols = get_feature_sets()["Set A"]
    preprocessor = build_preprocessor(feature_cols)
    preprocessor.fit(train_df[feature_cols])

    # Transform both
    train_transformed = preprocessor.transform(train_df[feature_cols])
    test_transformed = preprocessor.transform(test_df[feature_cols])

    assert not np.isnan(train_transformed).any()
    assert not np.isnan(test_transformed).any()
