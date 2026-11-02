"""
Model definitions, hyperparameter tuning, and ensemble construction
for ITM-390 ASD Screening Project.

Includes:
- Logistic Regression (interpretable linear baseline)
- K-Nearest Neighbors (distance-based classifier + K-sweep)
- Random Forest (non-linear ensemble classifier)
- Soft Voting Ensemble (Creative Idea 3.1)
"""

import warnings
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

# Suppress convergence and penalty deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_repeated_stratified_kfold(
    n_splits: int = 5, n_repeats: int = 5, random_state: int = 42
) -> RepeatedStratifiedKFold:
    """Standardized cross-validation scheme: Repeated Stratified 5-Fold (25 folds total)."""
    return RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )


def get_model_specs() -> Dict[str, Dict[str, Any]]:
    """Define base classifiers and hyperparameter search grids for tuning."""
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=1000, random_state=42, solver="liblinear"
            ),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "classifier__penalty": ["l1", "l2"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "K-Nearest Neighbors": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "classifier__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [None, 3, 5, 8],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__class_weight": [None, "balanced"],
            },
        },
    }


def create_pipeline(
    classifier_name: str,
    feature_cols: List[str],
    classifier_params: Dict[str, Any] = None,
) -> Pipeline:
    """Create end-to-end Pipeline combining ColumnTransformer and classifier."""
    specs = get_model_specs()
    if classifier_name not in specs and classifier_name != "Voting Ensemble":
        raise ValueError(f"Unknown classifier: {classifier_name}")

    preprocessor = build_preprocessor(feature_cols)

    if classifier_name == "Voting Ensemble":
        raise ValueError(
            "Use build_voting_ensemble() for constructing the Voting Classifier."
        )

    base_estimator = specs[classifier_name]["estimator"]
    if classifier_params:
        base_estimator.set_params(**classifier_params)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", base_estimator),
    ])


def build_voting_ensemble(
    best_pipelines: Dict[str, Pipeline],
    voting: str = "soft",
) -> Pipeline:
    """
    Construct a soft-voting ensemble from pre-tuned pipelines.
    Extracts the individual classifier estimators while sharing the fitted preprocessor.
    """
    estimators = []
    for name, pipe in best_pipelines.items():
        clean_key = name.lower().replace(" ", "_").replace("-", "_")
        # Extract the pre-configured classifier instance
        clf = pipe.named_steps["classifier"]
        estimators.append((clean_key, clf))

    voting_clf = VotingClassifier(estimators=estimators, voting=voting)

    # Use the preprocessor from the first tuned pipeline
    first_pipeline = next(iter(best_pipelines.values()))
    preprocessor = first_pipeline.named_steps["preprocessor"]

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", voting_clf),
    ])


def tune_classifier(
    classifier_name: str,
    X_train,
    y_train,
    feature_cols: List[str],
    cv: RepeatedStratifiedKFold = None,
    scoring: str = "recall",
    n_jobs: int = 1,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Tune a specific classifier using GridSearchCV on training data.
    Returns best_pipeline, best_params, best_cv_score.
    """
    if cv is None:
        cv = get_repeated_stratified_kfold()

    specs = get_model_specs()
    if classifier_name not in specs:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    pipe = Pipeline([
        ("preprocessor", build_preprocessor(feature_cols)),
        ("classifier", specs[classifier_name]["estimator"]),
    ])

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=specs[classifier_name]["param_grid"],
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
    )

    grid_search.fit(X_train[feature_cols], y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def sweep_knn_k(
    X_train,
    y_train,
    feature_cols: List[str],
    k_range: List[int] = None,
    cv: RepeatedStratifiedKFold = None,
) -> Dict[str, List[float]]:
    """
    Perform dedicated K-sweep for KNN to produce bias/variance diagnostic curves.
    """
    if k_range is None:
        k_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    if cv is None:
        cv = get_repeated_stratified_kfold()

    mean_recalls = []
    std_recalls = []
    mean_accuracies = []
    std_accuracies = []

    for k in k_range:
        pipe = Pipeline([
            ("preprocessor", build_preprocessor(feature_cols)),
            ("classifier", KNeighborsClassifier(n_neighbors=k)),
        ])

        from sklearn.model_selection import cross_validate
        scores = cross_validate(
            pipe,
            X_train[feature_cols],
            y_train,
            cv=cv,
            scoring=["recall", "accuracy"],
            n_jobs=1,
        )

        mean_recalls.append(float(np.mean(scores["test_recall"])))
        std_recalls.append(float(np.std(scores["test_recall"])))
        mean_accuracies.append(float(np.mean(scores["test_accuracy"])))
        std_accuracies.append(float(np.std(scores["test_accuracy"])))

    return {
        "k_values": k_range,
        "mean_recall": mean_recalls,
        "std_recall": std_recalls,
        "mean_accuracy": mean_accuracies,
        "std_accuracy": std_accuracies,
    }
