"""
Unit tests for evaluation metrics, confidence intervals, and statistical tests.
"""

import numpy as np
import pytest

from src.evaluate import (
    bootstrap_confidence_intervals,
    compute_classification_metrics,
    mcnemar_test,
    threshold_analysis,
)


def test_compute_classification_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.85])

    res = compute_classification_metrics(y_true, y_pred, y_proba)
    assert res["recall"] == 1.0
    assert res["specificity"] == 1.0
    assert res["accuracy"] == 1.0
    assert res["precision"] == 1.0
    assert res["f1"] == 1.0
    assert res["roc_auc"] == 1.0


def test_compute_classification_metrics_edge_case():
    y_true = np.array([0, 0, 1, 1])
    # Model predicts all 0s (missed all positive cases)
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4])

    res = compute_classification_metrics(y_true, y_pred, y_proba)
    assert res["recall"] == 0.0
    assert res["specificity"] == 1.0
    assert res["accuracy"] == 0.5
    assert res["precision"] == 0.0
    assert res["f1"] == 0.0


def test_bootstrap_confidence_intervals():
    y_true = np.array([0] * 30 + [1] * 20)
    # 90% accuracy mock predictions
    y_pred = np.array([0] * 28 + [1] * 2 + [0] * 2 + [1] * 18)
    y_proba = np.random.RandomState(42).uniform(0, 1, size=len(y_true))

    ci = bootstrap_confidence_intervals(y_true, y_pred, y_proba, n_bootstraps=100)

    for metric in ["recall", "specificity", "accuracy", "precision", "f1"]:
        assert metric in ci
        assert 0.0 <= ci[metric]["ci_lower"] <= ci[metric]["ci_upper"] <= 1.0


def test_mcnemar_test_symmetry_and_significance():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred_a = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Perfect model
    y_pred_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # Misses all positives

    res = mcnemar_test(y_true, y_pred_a, y_pred_b)

    assert res["b_discordant_a_not_b"] == 4
    assert res["c_discordant_b_not_a"] == 0
    assert res["statistic"] > 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_threshold_analysis():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    df = threshold_analysis(y_true, y_proba, thresholds=np.array([0.2, 0.5, 0.8]))

    assert len(df) == 3
    assert "recall" in df.columns
    assert "specificity" in df.columns
    # Lower threshold should give higher or equal recall
    assert df.loc[df["threshold"] == 0.2, "recall"].values[0] >= df.loc[df["threshold"] == 0.8, "recall"].values[0]
