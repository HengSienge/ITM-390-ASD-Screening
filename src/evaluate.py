"""
Evaluation framework for ITM-390 ASD Screening Project.

Implements:
- Comprehensive screening metrics (Recall, Specificity, Accuracy, Precision, F1, ROC-AUC)
- Repeated Cross-Validation multi-metric evaluation
- Non-parametric bootstrap confidence intervals (1,000 resamples)
- Statistical significance tests (McNemar's test & paired CV tests)
- Decision threshold sweep for screening sensitivity optimization
- Publication-quality plotting suite (saved to results/figures/)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics with Recall/Sensitivity as primary metric
    and Specificity reported side-by-side.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = np.nan
            metrics["brier_score"] = np.nan

    return metrics


def evaluate_cv_pipeline(
    pipeline,
    X,
    y,
    feature_cols: List[str],
    cv: RepeatedStratifiedKFold,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate pipeline across repeated CV folds.
    Returns mean and standard deviation for each metric across folds.
    """
    scoring = {
        "accuracy": "accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        pipeline,
        X[feature_cols],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    summary = {}
    for metric_name in scoring.keys():
        key = f"test_{metric_name}"
        summary[metric_name] = {
            "mean": float(np.mean(scores[key])),
            "std": float(np.std(scores[key])),
            "raw_scores": scores[key].tolist(),
        }

    return summary


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    n_bootstraps: int = 1000,
    alpha: float = 0.95,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute non-parametric bootstrap percentile confidence intervals on test set.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    bootstrapped_metrics = {
        "recall": [],
        "specificity": [],
        "accuracy": [],
        "precision": [],
        "f1": [],
        "roc_auc": [],
    }

    lower_pct = ((1.0 - alpha) / 2.0) * 100
    upper_pct = (alpha + ((1.0 - alpha) / 2.0)) * 100

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        # Ensure bootstrap sample contains both classes
        if len(np.unique(y_true[indices])) < 2:
            continue

        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        boot_y_proba = y_proba[indices] if y_proba is not None else None

        res = compute_classification_metrics(boot_y_true, boot_y_pred, boot_y_proba)
        for k in bootstrapped_metrics.keys():
            if k in res and not np.isnan(res[k]):
                bootstrapped_metrics[k].append(res[k])

    ci_results = {}
    for k, values in bootstrapped_metrics.items():
        if values:
            ci_results[k] = {
                "ci_lower": float(np.percentile(values, lower_pct)),
                "ci_upper": float(np.percentile(values, upper_pct)),
                "mean": float(np.mean(values)),
            }

    return ci_results


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Perform McNemar's test with Edwards' continuity correction.
    Compares two models on the exact same test predictions.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table:
    # b: A correct, B wrong
    # c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    a = int(np.sum(correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    contingency = [[a, b], [c, d]]

    # Continuity corrected statistic
    total_discordant = b + c
    if total_discordant == 0:
        stat = 0.0
        p_val = 1.0
    else:
        stat = float(((abs(b - c) - 1.0) ** 2) / total_discordant)
        p_val = float(1.0 - stats.chi2.cdf(stat, df=1))

    return {
        "statistic": stat,
        "p_value": p_val,
        "b_discordant_a_not_b": b,
        "c_discordant_b_not_a": c,
        "contingency_matrix": contingency,
        "significant_at_05": bool(p_val < 0.05),
    }


def threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """
    Evaluate screening performance across decision thresholds (0.05 to 0.95).
    Crucial for ASD screening where higher recall is prioritized over 0.50 cutoff.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    records = []
    for thresh in thresholds:
        pred = (y_proba >= thresh).astype(int)
        metrics = compute_classification_metrics(y_true, pred, y_proba)
        records.append({
            "threshold": float(round(thresh, 3)),
            "recall": metrics["recall"],
            "specificity": metrics["specificity"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "fn": metrics["fn"],
            "fp": metrics["fp"],
        })

    return pd.DataFrame(records)


# -------------------------------------------------------------
# Publication-Quality Plotting Functions
# -------------------------------------------------------------

def set_plot_style():
    """Apply consistent styling across all figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
    })


def plot_combined_roc_curves(
    model_probas: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot overlaid ROC curves for all models on one figure."""
    set_plot_style()
    plt.figure(figsize=(8, 6), dpi=300)

    for name, proba in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("Comparative ROC Curves: ASD Screening Models", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_combined_pr_curves(
    model_probas: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot overlaid Precision-Recall curves for all models."""
    set_plot_style()
    plt.figure(figsize=(8, 6), dpi=300)

    for name, proba in model_probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        plt.plot(recall, precision, lw=2, label=name)

    base_rate = np.mean(y_true)
    plt.axhline(base_rate, color="gray", linestyle="--", lw=1.5, label=f"Base Rate ({base_rate:.2f})")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curves: ASD Screening Models", pad=12)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(
    model_preds: Dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Plot side-by-side heatmaps of confusion matrices."""
    set_plot_style()
    n_models = len(model_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, model_preds.items()):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["No ASD (0)", "ASD (1)"],
            yticklabels=["No ASD (0)", "ASD (1)"],
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.suptitle("Confusion Matrix Comparison (Held-Out Test Set)", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_threshold_curves(
    threshold_dfs: Dict[str, pd.DataFrame],
    save_path: Union[str, Path],
) -> None:
    """Plot Sensitivity vs Specificity across decision thresholds."""
    set_plot_style()
    n_models = len(threshold_dfs)
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4), dpi=300)
    if n_models == 1:
        axes = [axes]

    for ax, (name, df) in zip(axes, threshold_dfs.items()):
        ax.plot(df["threshold"], df["recall"], marker="o", lw=2, label="Recall (Sensitivity)")
        ax.plot(df["threshold"], df["specificity"], marker="s", lw=2, label="Specificity")
        ax.axvline(0.5, color="gray", linestyle=":", label="Default (0.50)")
        ax.set_title(name)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="lower center", frameon=True, fontsize=8)

    plt.suptitle("Screening Decision Threshold Analysis (Recall vs Specificity)", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_knn_k_sweep(
    k_sweep_data: Dict[str, List[float]],
    save_path: Union[str, Path],
) -> None:
    """Plot KNN K-sweep bias-variance diagnostic curve."""
    set_plot_style()
    plt.figure(figsize=(7, 4.5), dpi=300)

    ks = k_sweep_data["k_values"]
    recalls = k_sweep_data["mean_recall"]
    recall_stds = k_sweep_data["std_recall"]
    accuracies = k_sweep_data["mean_accuracy"]
    acc_stds = k_sweep_data["std_accuracy"]

    plt.errorbar(
        ks, recalls, yerr=recall_stds, fmt="-o", capsize=4, lw=2, label="Recall / Sensitivity"
    )
    plt.errorbar(
        ks, accuracies, yerr=acc_stds, fmt="-s", capsize=4, lw=2, label="Accuracy"
    )

    plt.xlabel("Number of Neighbors (K)")
    plt.ylabel("Cross-Validated Score (Mean ± Std)")
    plt.title("KNN Hyperparameter Sweep: K vs Performance (Repeated 5-Fold CV)")
    plt.xticks(ks)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_feature_importance_comparison(
    rf_pipeline,
    feature_cols: List[str],
    save_path: Union[str, Path],
) -> None:
    """Plot Random Forest Gini feature importances."""
    set_plot_style()
    clf = rf_pipeline.named_steps["classifier"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(len(clf.feature_importances_))]

    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:15]  # Top 15 features

    top_names = [feature_names[i].replace("num__", "").replace("bin__", "").replace("cat__", "") for i in sorted_idx]
    top_importances = importances[sorted_idx]

    plt.figure(figsize=(8, 5.5), dpi=300)
    sns.barplot(x=top_importances, y=top_names, palette="viridis")
    plt.xlabel("Gini Feature Importance")
    plt.title("Top 15 Most Informative Features (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
