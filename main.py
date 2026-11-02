"""
ITM-390 ASD Screening Project: End-to-End Orchestration Script.

Runs the entire scientific workflow:
1. Data loading and validation.
2. Data cleaning, leakage prevention, and stratified splitting.
3. Cross-validation across Feature Sets A, B, and C.
4. Model tuning and KNN K-sweep on training data.
5. Soft Voting Ensemble construction.
6. Held-out test set evaluation with 1,000 bootstrap confidence intervals.
7. Pairwise McNemar's statistical significance tests.
8. Decision threshold analysis for high-sensitivity clinical screening.
9. Figure and metric table generation.
"""

import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.evaluate import (
    bootstrap_confidence_intervals,
    compute_classification_metrics,
    evaluate_cv_pipeline,
    mcnemar_test,
    plot_combined_pr_curves,
    plot_combined_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance_comparison,
    plot_knn_k_sweep,
    plot_threshold_curves,
    threshold_analysis,
)
from src.models import (
    build_voting_ensemble,
    create_pipeline,
    get_repeated_stratified_kfold,
    sweep_knn_k,
    tune_classifier,
)
from src.preprocessing import (
    TARGET_COL,
    clean_raw_data,
    get_feature_sets,
    load_raw_data,
    train_test_split_dataset,
)
from src.utils import ensure_dir, format_markdown_table, save_json, set_seed, setup_logger


def run_pipeline():
    logger = setup_logger("ITM-390-Pipeline")
    logger.info("Initializing ITM-390 ASD Screening End-to-End Pipeline...")
    set_seed(42)

    root_dir = Path(__file__).resolve().parent
    raw_data_dir = root_dir / "data" / "raw"
    proc_data_dir = root_dir / "data" / "processed"
    fig_dir = root_dir / "results" / "figures"
    metric_dir = root_dir / "results" / "metrics"
    models_dir = root_dir / "results" / "models"

    for d in [proc_data_dir, fig_dir, metric_dir, models_dir]:
        ensure_dir(d)

    # 1. Load Data
    raw_arff = raw_data_dir / "Autism-Child-Data.arff"
    logger.info(f"Loading raw dataset from {raw_arff.name}...")
    df_raw = load_raw_data(raw_arff)

    # Save a clean raw CSV for user inspection
    raw_csv = raw_data_dir / "Autism-Child-Data.csv"
    df_raw.to_csv(raw_csv, index=False)

    # 2. Clean Data & Prevent Leakage
    logger.info("Cleaning data, resolving inconsistencies, and dropping 'result' column...")
    df_cleaned = clean_raw_data(df_raw, drop_leakage=True)
    logger.info(f"Dataset shape after cleaning: {df_cleaned.shape} (292 instances, binary target)")

    # 3. Stratified Train / Test Split
    logger.info("Executing 80/20 Stratified Train/Test Split (fixed seed 42)...")
    train_df, test_df = train_test_split_dataset(df_cleaned, test_size=0.2, random_state=42)

    train_df.to_csv(proc_data_dir / "train.csv", index=False)
    test_df.to_csv(proc_data_dir / "test.csv", index=False)
    logger.info(f"Train instances: {len(train_df)} | Held-out test instances: {len(test_df)}")

    # Save X and y splits
    feature_sets = get_feature_sets()
    all_features = feature_sets["Set A"]

    train_df[all_features].to_csv(proc_data_dir / "X_train.csv", index=False)
    train_df[[TARGET_COL]].to_csv(proc_data_dir / "y_train.csv", index=False)
    test_df[all_features].to_csv(proc_data_dir / "X_test.csv", index=False)
    test_df[[TARGET_COL]].to_csv(proc_data_dir / "y_test.csv", index=False)

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    cv_scheme = get_repeated_stratified_kfold(n_splits=5, n_repeats=5, random_state=42)

    # 4. Multi-Feature Set Comparison (Sets A, B, C)
    logger.info("--- Comparing Feature Sets (A: All, B: Q-Chat Only, C: Demographics Only) ---")
    feat_comparison_records = []

    for set_name, cols in feature_sets.items():
        logger.info(f"Evaluating Repeated CV across models on {set_name} ({len(cols)} features)...")
        for model_name in ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]:
            pipe = create_pipeline(model_name, cols)
            cv_res = evaluate_cv_pipeline(pipe, train_df, y_train, cols, cv_scheme)
            feat_comparison_records.append({
                "Feature Set": set_name,
                "Model": model_name,
                "Recall (Sensitivity)": f"{cv_res['recall']['mean']:.4f} ± {cv_res['recall']['std']:.4f}",
                "Accuracy": f"{cv_res['accuracy']['mean']:.4f} ± {cv_res['accuracy']['std']:.4f}",
                "ROC-AUC": f"{cv_res['roc_auc']['mean']:.4f} ± {cv_res['roc_auc']['std']:.4f}",
                "F1-Score": f"{cv_res['f1']['mean']:.4f} ± {cv_res['f1']['std']:.4f}",
            })

    feat_comp_df = pd.DataFrame(feat_comparison_records)
    feat_comp_df.to_csv(metric_dir / "feature_sets_comparison.csv", index=False)

    # 5. Primary Experiment: Model Tuning on Feature Set A
    primary_cols = feature_sets["Set A"]
    logger.info("--- Phase 3: Hyperparameter Tuning on Training Set (Primary Feature Set A) ---")

    # KNN K-Sweep Bias-Variance Analysis
    logger.info("Sweeping KNN K values from 1 to 25...")
    k_sweep_data = sweep_knn_k(train_df, y_train, primary_cols, cv=cv_scheme)
    plot_knn_k_sweep(k_sweep_data, fig_dir / "knn_k_sweep.png")
    save_json(k_sweep_data, metric_dir / "knn_k_sweep_data.json")

    # Hyperparameter Tuning
    tuned_pipelines = {}
    best_params_all = {}
    train_times = {}

    for model_name in ["Logistic Regression", "K-Nearest Neighbors", "Random Forest"]:
        logger.info(f"Tuning {model_name} with GridSearchCV (scoring='recall')...")
        t0 = time.time()
        best_pipe, best_params, best_score = tune_classifier(
            model_name, train_df, y_train, primary_cols, cv=cv_scheme, scoring="recall"
        )
        t_elapsed = time.time() - t0
        tuned_pipelines[model_name] = best_pipe
        best_params_all[model_name] = best_params
        train_times[model_name] = t_elapsed
        logger.info(f"  -> Best params: {best_params}")
        logger.info(f"  -> Best CV Recall: {best_score:.4f} (elapsed: {t_elapsed:.2f}s)")

    # 6. Ensemble Construction (Creative Idea 3.1)
    logger.info("Building Soft Voting Ensemble combining tuned LR, KNN, and RF...")
    voting_pipe = build_voting_ensemble(tuned_pipelines, voting="soft")
    voting_pipe.fit(train_df[primary_cols], y_train)
    tuned_pipelines["Voting Ensemble"] = voting_pipe
    train_times["Voting Ensemble"] = sum(train_times.values())

    # Save best parameters
    save_json(best_params_all, metric_dir / "best_hyperparameters.json")

    # 7. Cross-Validation Performance Summary
    logger.info("Computing detailed Repeated 5-Fold CV metrics for tuned models...")
    cv_summary_records = []
    cv_raw_recalls = {}

    for name, pipe in tuned_pipelines.items():
        res = evaluate_cv_pipeline(pipe, train_df, y_train, primary_cols, cv_scheme)
        cv_raw_recalls[name] = res["recall"]["raw_scores"]
        cv_summary_records.append({
            "Model": name,
            "CV Recall (Mean ± Std)": f"{res['recall']['mean']:.4f} ± {res['recall']['std']:.4f}",
            "CV Accuracy (Mean ± Std)": f"{res['accuracy']['mean']:.4f} ± {res['accuracy']['std']:.4f}",
            "CV ROC-AUC (Mean ± Std)": f"{res['roc_auc']['mean']:.4f} ± {res['roc_auc']['std']:.4f}",
            "CV Precision (Mean ± Std)": f"{res['precision']['mean']:.4f} ± {res['precision']['std']:.4f}",
            "CV F1 (Mean ± Std)": f"{res['f1']['mean']:.4f} ± {res['f1']['std']:.4f}",
            "Train Time (s)": f"{train_times.get(name, 0.0):.2f}",
        })

    cv_summary_df = pd.DataFrame(cv_summary_records)
    cv_summary_df.to_csv(metric_dir / "cv_results_comparison.csv", index=False)

    # 8. Phase 4 & 5: Held-Out Test Set Evaluation
    logger.info("--- Phase 4 & 5: Final Evaluation on Held-Out Test Set ---")
    test_predictions = {}
    test_probabilities = {}
    test_metric_records = []
    ci_records = []
    threshold_data_all = {}

    for name, pipe in tuned_pipelines.items():
        # Fit on full training set
        pipe.fit(train_df[primary_cols], y_train)

        # Inference timing
        t0 = time.time()
        preds = pipe.predict(test_df[primary_cols])
        inf_time_ms = (time.time() - t0) * 1000.0 / len(test_df)

        probas = pipe.predict_proba(test_df[primary_cols])[:, 1]
        test_predictions[name] = preds
        test_probabilities[name] = probas

        # Point metrics
        metrics = compute_classification_metrics(y_test, preds, probas)
        test_metric_records.append({
            "Model": name,
            "Recall (Sensitivity)": metrics["recall"],
            "Specificity": metrics["specificity"],
            "Accuracy": metrics["accuracy"],
            "Precision (PPV)": metrics["precision"],
            "F1-Score": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
            "Brier Score": metrics["brier_score"],
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "TN": metrics["tn"],
            "FN": metrics["fn"],
            "Latency (ms/sample)": inf_time_ms,
        })

        # Bootstrap Confidence Intervals (1,000 resamples)
        ci = bootstrap_confidence_intervals(y_test, preds, probas, n_bootstraps=1000)
        ci_records.append({
            "Model": name,
            "Recall 95% CI": f"[{ci['recall']['ci_lower']:.3f}, {ci['recall']['ci_upper']:.3f}]",
            "Specificity 95% CI": f"[{ci['specificity']['ci_lower']:.3f}, {ci['specificity']['ci_upper']:.3f}]",
            "Accuracy 95% CI": f"[{ci['accuracy']['ci_lower']:.3f}, {ci['accuracy']['ci_upper']:.3f}]",
            "Precision 95% CI": f"[{ci['precision']['ci_lower']:.3f}, {ci['precision']['ci_upper']:.3f}]",
            "F1 95% CI": f"[{ci['f1']['ci_lower']:.3f}, {ci['f1']['ci_upper']:.3f}]",
            "ROC-AUC 95% CI": f"[{ci['roc_auc']['ci_lower']:.3f}, {ci['roc_auc']['ci_upper']:.3f}]",
        })

        # Threshold analysis
        thresh_df = threshold_analysis(y_test, probas)
        threshold_data_all[name] = thresh_df

        # Save model artifact
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(pipe, models_dir / f"{clean_name}.joblib")

    test_metric_df = pd.DataFrame(test_metric_records)
    test_metric_df.to_csv(metric_dir / "test_set_results.csv", index=False)

    ci_df = pd.DataFrame(ci_records)
    ci_df.to_csv(metric_dir / "bootstrap_confidence_intervals.csv", index=False)

    # 9. Statistical Significance Testing (McNemar's Test)
    logger.info("Computing pairwise McNemar's tests on held-out test predictions...")
    mcnemar_records = []
    model_names_list = list(test_predictions.keys())

    for i in range(len(model_names_list)):
        for j in range(i + 1, len(model_names_list)):
            m1 = model_names_list[i]
            m2 = model_names_list[j]
            mc_res = mcnemar_test(y_test, test_predictions[m1], test_predictions[m2])
            mcnemar_records.append({
                "Model A": m1,
                "Model B": m2,
                "Discordant (A right, B wrong)": mc_res["b_discordant_a_not_b"],
                "Discordant (B right, A wrong)": mc_res["c_discordant_b_not_a"],
                "Chi-Square Stat": mc_res["statistic"],
                "p-value": mc_res["p_value"],
                "Significant (p < 0.05)": mc_res["significant_at_05"],
            })

    mcnemar_df = pd.DataFrame(mcnemar_records)
    mcnemar_df.to_csv(metric_dir / "mcnemar_tests.csv", index=False)

    # 10. Publication Visualizations
    logger.info("Generating publication figures...")
    plot_combined_roc_curves(test_probabilities, y_test, fig_dir / "roc_curves.png")
    plot_combined_pr_curves(test_probabilities, y_test, fig_dir / "pr_curves.png")
    plot_confusion_matrices(test_predictions, y_test, fig_dir / "confusion_matrices.png")
    plot_threshold_curves(threshold_data_all, fig_dir / "threshold_analysis.png")
    plot_feature_importance_comparison(
        tuned_pipelines["Random Forest"], primary_cols, fig_dir / "rf_feature_importance.png"
    )

    # 11. Error Analysis (False Negatives & False Positives)
    logger.info("Conducting error profiling on held-out test set...")
    error_analysis_rows = test_df.copy()
    for name in model_names_list:
        error_analysis_rows[f"pred_{name}"] = test_predictions[name]
        error_analysis_rows[f"proba_{name}"] = np.round(test_probabilities[name], 3)

    # Identify children missed by any model (False Negatives)
    fn_mask = (error_analysis_rows[TARGET_COL] == 1) & (
        (error_analysis_rows["pred_Logistic Regression"] == 0)
        | (error_analysis_rows["pred_K-Nearest Neighbors"] == 0)
        | (error_analysis_rows["pred_Random Forest"] == 0)
    )
    missed_cases = error_analysis_rows[fn_mask]
    missed_cases.to_csv(metric_dir / "false_negatives_error_analysis.csv", index=False)
    logger.info(f"False negative count across any model: {len(missed_cases)}")

    # 12. Display Summary Tables
    print("\n" + "=" * 80)
    print(" ITM-390 ASD SCREENING: EXPERIMENTAL RESULTS SUMMARY")
    print(" Group 10 · American University of Phnom Penh")
    print("=" * 80)

    print("\n### 1. Repeated Stratified 5-Fold Cross-Validation (Training Folds)")
    print(format_markdown_table(cv_summary_df))

    print("\n### 2. Held-Out Test Set Performance (20% Split, N=59)")
    print(format_markdown_table(test_metric_df))

    print("\n### 3. Bootstrap 95% Confidence Intervals (1,000 Resamples)")
    print(format_markdown_table(ci_df))

    print("\n### 4. Pairwise McNemar Significance Tests")
    print(format_markdown_table(mcnemar_df))

    print("\n### 5. Multi-Feature Set Comparison (A vs B vs C)")
    print(format_markdown_table(feat_comp_df))

    logger.info("Pipeline execution successfully completed! All artifacts generated.")


if __name__ == "__main__":
    run_pipeline()
