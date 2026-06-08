"""Train and save the senior_project_webtool machine learning models.

The project uses three separate cleaned, model-ready CSV files:

1. Pre-launch expected users model
   - Dataset: data/expected_users_4class_augmented_model_dataset.csv
   - Target: outcome_level
   - Classes: Not Viable, Low, Mid, High
   - Model: regularized multinomial logistic regression

2. Launch traction expected users model
   - Dataset: data/expected_users_4class_augmented_model_dataset.csv
   - Target: outcome_level
   - Classes: Not Viable, Low, Mid, High
   - Model: regularized multinomial logistic regression

3. Downside risk model
   - Dataset: data/downside_risk_binary_model_dataset.csv
   - Target: downside_target
   - Classes: Low, MidHigh
   - Model: K-Nearest Neighbors

4. Upside potential model
   - Dataset: data/upside_potential_binary_model_dataset.csv
   - Target: upside_target
   - Classes: Mid, High
   - Model: Support Vector Classifier with probability estimates enabled

Each CSV is expected to already contain cleaned predictor columns. This script
does not create new predictors. It only removes ID columns, target columns, and
any owner/revenue/review/post-launch columns that should not be used for
prelaunch prediction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


DATA_DIR = Path("data")
MODELS_DIR = Path("models")
OUTPUTS_DIR = Path("outputs")


# Columns that should never be used as model predictors. These include IDs,
# display names, and all target variants used across the three datasets.
EXCLUDED_COLUMNS = {
    "appid",
    "game_name",
    "outcome_level",
    "downside_target",
    "upside_target",
}


# Defensive leakage filter. The cleaned datasets should already be prelaunch
# only, but this prevents accidental use of post-launch outcome proxies if a
# future CSV includes them.
BLOCKED_FEATURE_SUBSTRINGS = (
    "owner",
    "owner_midpoint",
    "owners_per_review",
    "revenue",
    "review",
    "review_sentiment",
    "positive_review_pct",
    "log_total_reviews",
    "total_reviews",
    "positive_reviews",
    "negative_reviews",
    "sales_confidence",
    "confidence_score",
    "confidence",
    "post_launch",
    "postlaunch",
    "post-launch",
)


OUTCOME_TIER_USERS = {
    "Not Viable": 10000,
    "Low": 35000,
    "Mid": 75000,
    "High": 150000,
}


EXPECTED_USERS_4CLASS_LABELS = ("Not Viable", "Low", "Mid", "High")


EXPECTED_USERS_STABLE_FEATURES = [
    "prelaunch_publisher_type",
    "prelaunch_price_bucket",
    "prelaunch_subgenre_primary",
    "prelaunch_price_usd",
    "prelaunch_language_count",
    "prelaunch_screenshots_count",
    "prelaunch_movies_count",
    "prelaunch_achievements_count",
    "prelaunch_tag_count",
]


LAUNCH_TRACTION_FEATURES = [
    "prelaunch_price_usd",
    "prelaunch_achievements_count",
    "log_total_reviews",
    "confidence_score",
]


REVIEW_RECEPTION_COLUMN = "review_sentiment"
REVIEW_RECEPTION_TARGET_VALUE = "Positive"
REVIEW_RECEPTION_PRIOR_STRENGTH = 75
REVIEW_RECEPTION_BLEND_WEIGHT = 0.15
REVIEW_RECEPTION_SCENARIO_UPLIFT = 0.15


PUBLISHER_SUPPORT_COLUMN = "prelaunch_publisher_type"
PUBLISHER_SUPPORT_PRIOR_STRENGTH = 75
PUBLISHER_SUPPORT_BLEND_WEIGHT = 0.15
PUBLISHER_SUPPORT_SCENARIO_UPLIFT = 0.1762


DISCLAIMER = (
    "Predictions are directional estimates for planning and comparison only. "
    "They are not exact sales forecasts."
)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one model training job."""

    name: str
    dataset_path: Path
    target_column: str
    model_path: Path
    feature_path: Path
    estimator: Any
    expected_labels: tuple[str, ...]
    stable_features: tuple[str, ...] | None = None
    defaults_path: Path | None = None
    base_probs_path: Path | None = None
    review_prior_path: Path | None = None
    review_column: str | None = None
    publisher_prior_path: Path | None = None
    publisher_column: str | None = None
    evaluate_metrics: bool = True
    evaluate_probability_metrics: bool = True


def is_blocked_feature(column_name: str) -> bool:
    """Return True when a column name suggests leakage or post-launch data."""
    normalized_name = column_name.lower().replace(" ", "_")
    return any(blocked in normalized_name for blocked in BLOCKED_FEATURE_SUBSTRINGS)


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select predictor columns that are allowed for model training.

    The CSV files are already model-ready, so the main job here is exclusion:
    remove IDs, targets, and any fields that look like owner, revenue, review,
    or post-launch outcome columns.
    """
    feature_columns: list[str] = []

    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if is_blocked_feature(column):
            continue
        feature_columns.append(column)

    if not feature_columns:
        raise ValueError("No valid feature columns were found after exclusions.")

    return feature_columns


def select_stable_feature_columns(
    df: pd.DataFrame,
    stable_features: tuple[str, ...],
) -> list[str]:
    """Select the requested stable expected-users feature subset.

    The expected-users model intentionally avoids sparse tag flags and the full
    wide prelaunch feature set so that its probabilities remain directional and
    less overfit.
    """
    feature_columns = [
        feature
        for feature in stable_features
        if feature in df.columns and feature not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No stable expected-users features were found.")

    return feature_columns


def is_binary_column(series: pd.Series) -> bool:
    """Detect numeric/bool columns that are true binary 0/1 indicators.

    Binary columns are passed through with imputation only. Object columns with
    two text values are treated as categorical because models require numeric
    inputs after preprocessing.
    """
    non_null = series.dropna()

    if non_null.empty:
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    if not pd.api.types.is_numeric_dtype(series):
        return False

    unique_values = set(np.asarray(non_null.unique()).tolist())
    return unique_values.issubset({0, 1, 0.0, 1.0, True, False})


def split_feature_types(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split selected predictors into categorical, numeric, and binary groups."""
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    binary_columns: list[str] = []

    for column in feature_columns:
        series = df[column]

        if is_binary_column(series):
            binary_columns.append(column)
        elif pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return categorical_columns, numeric_columns, binary_columns


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    binary_columns: list[str],
) -> ColumnTransformer:
    """Create the shared preprocessing ColumnTransformer.

    - Categorical predictors get most-frequent imputation and one-hot encoding.
    - Numeric predictors get median imputation and standard scaling.
    - Binary predictors get most-frequent imputation only.
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
            ("binary", binary_pipeline, binary_columns),
        ],
        sparse_threshold=0.0,
    )


def validate_target_labels(
    y: pd.Series,
    expected_labels: tuple[str, ...],
    model_name: str,
) -> None:
    """Make sure a dataset contains the classes the model is designed for."""
    actual_labels = set(y.dropna().astype(str).unique())
    expected_label_set = set(expected_labels)

    if actual_labels != expected_label_set:
        raise ValueError(
            f"{model_name} target labels do not match expectations. "
            f"Expected {sorted(expected_label_set)}, found {sorted(actual_labels)}."
        )


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Write a dictionary as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def calculate_training_defaults(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, Any]:
    """Calculate median/mode defaults for expected-users prediction inputs."""
    numeric_defaults = {
        column: _json_safe_value(df[column].median())
        for column in numeric_columns
    }
    categorical_defaults = {
        column: _json_safe_value(_mode_or_unknown(df[column]))
        for column in categorical_columns
    }

    defaults: dict[str, Any] = {}
    defaults.update(numeric_defaults)
    defaults.update(categorical_defaults)

    return {
        "defaults": defaults,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
    }


def calculate_base_probabilities(
    y: pd.Series,
    expected_labels: tuple[str, ...],
) -> dict[str, float]:
    """Calculate base outcome probabilities from training class counts."""
    counts = y.value_counts(normalize=True)
    base_probs: dict[str, float] = {}

    for label in expected_labels:
        probability = float(counts.get(label, 0.0))
        base_probs[label] = probability
        base_probs[f"P_{label.replace(' ', '_')}"] = probability

    return base_probs


def calculate_publisher_support_priors(
    df: pd.DataFrame,
    y: pd.Series,
    expected_labels: tuple[str, ...],
    publisher_column: str,
) -> dict[str, Any]:
    """Calculate smoothed publisher outcome priors for scenario adjustment.

    Publisher support is intentionally kept out of the launch-traction
    classifier. The saved priors let the app apply a small post-model
    adjustment that reflects historical publisher-backed uplift while shrinking
    noisy category effects toward the overall market distribution.
    """
    if publisher_column not in df.columns:
        raise ValueError(f"Publisher support column is missing: {publisher_column}")

    global_counts = y.value_counts()
    global_distribution = {
        label: float(global_counts.get(label, 0) / len(y))
        for label in expected_labels
    }

    priors: dict[str, Any] = {}
    publisher_values = (
        df[publisher_column]
        .fillna("Unknown")
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    for publisher_value in publisher_values:
        mask = df[publisher_column].fillna("Unknown").astype(str) == publisher_value
        publisher_y = y[mask]
        sample_size = int(len(publisher_y))

        raw_counts = publisher_y.value_counts()
        raw_distribution = {
            label: float(raw_counts.get(label, 0) / sample_size)
            for label in expected_labels
        }
        shrink_weight = sample_size / (
            sample_size + PUBLISHER_SUPPORT_PRIOR_STRENGTH
        )
        smoothed_distribution = {
            label: (
                shrink_weight * raw_distribution[label]
                + (1 - shrink_weight) * global_distribution[label]
            )
            for label in expected_labels
        }
        smoothed_distribution = _normalize_probability_dict(smoothed_distribution)
        smoothed_expected_users = sum(
            smoothed_distribution[label] * OUTCOME_TIER_USERS[label]
            for label in expected_labels
        )

        priors[publisher_value] = {
            "sample_size": sample_size,
            "shrink_weight": float(shrink_weight),
            "raw_distribution": raw_distribution,
            "smoothed_distribution": smoothed_distribution,
            "smoothed_expected_users": float(smoothed_expected_users),
        }

    return {
        "publisher_column": publisher_column,
        "prior_strength": PUBLISHER_SUPPORT_PRIOR_STRENGTH,
        "adjustment_blend_weight": PUBLISHER_SUPPORT_BLEND_WEIGHT,
        "scenario_uplift": PUBLISHER_SUPPORT_SCENARIO_UPLIFT,
        "global_distribution": global_distribution,
        "publisher_priors": priors,
        "note": (
            "Publisher support is kept out of the core launch-traction model. "
            "Publishing Studio is applied afterward as a fixed expected-user "
            "scenario uplift."
        ),
    }


def calculate_review_reception_priors(
    df: pd.DataFrame,
    y: pd.Series,
    expected_labels: tuple[str, ...],
    review_column: str,
) -> dict[str, Any]:
    """Calculate smoothed review-sentiment priors for scenario adjustment.

    Review sentiment is intentionally kept out of the launch-traction
    classifier because it is derived from positive review percentage. The saved
    priors let the app preserve the directional research finding that Positive
    reception is associated with higher users without letting the overlapping
    model features create a negative sign.
    """
    if review_column not in df.columns:
        raise ValueError(f"Review reception column is missing: {review_column}")

    global_counts = y.value_counts()
    global_distribution = {
        label: float(global_counts.get(label, 0) / len(y))
        for label in expected_labels
    }

    priors: dict[str, Any] = {}
    review_values = (
        df[review_column]
        .fillna("Mixed")
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    for review_value in review_values:
        mask = df[review_column].fillna("Mixed").astype(str) == review_value
        review_y = y[mask]
        sample_size = int(len(review_y))

        raw_counts = review_y.value_counts()
        raw_distribution = {
            label: float(raw_counts.get(label, 0) / sample_size)
            for label in expected_labels
        }
        shrink_weight = sample_size / (
            sample_size + REVIEW_RECEPTION_PRIOR_STRENGTH
        )
        smoothed_distribution = {
            label: (
                shrink_weight * raw_distribution[label]
                + (1 - shrink_weight) * global_distribution[label]
            )
            for label in expected_labels
        }
        smoothed_distribution = _normalize_probability_dict(smoothed_distribution)
        smoothed_expected_users = sum(
            smoothed_distribution[label] * OUTCOME_TIER_USERS[label]
            for label in expected_labels
        )

        priors[review_value] = {
            "sample_size": sample_size,
            "shrink_weight": float(shrink_weight),
            "raw_distribution": raw_distribution,
            "smoothed_distribution": smoothed_distribution,
            "smoothed_expected_users": float(smoothed_expected_users),
        }

    return {
        "review_column": review_column,
        "target_value": REVIEW_RECEPTION_TARGET_VALUE,
        "prior_strength": REVIEW_RECEPTION_PRIOR_STRENGTH,
        "adjustment_blend_weight": REVIEW_RECEPTION_BLEND_WEIGHT,
        "scenario_uplift": REVIEW_RECEPTION_SCENARIO_UPLIFT,
        "global_distribution": global_distribution,
        "review_priors": priors,
        "note": (
            "Review reception is kept out of the core launch-traction model. "
            "Positive reception is applied afterward as a fixed expected-user "
            "scenario uplift."
        ),
    }


def _normalize_probability_dict(probabilities: dict[str, float]) -> dict[str, float]:
    """Normalize a probability dictionary with a safe equal-weight fallback."""
    cleaned = {
        label: max(float(probability), 0.0)
        for label, probability in probabilities.items()
    }
    total = sum(cleaned.values())

    if total <= 0:
        equal_probability = 1 / len(cleaned)
        return {label: equal_probability for label in cleaned}

    return {
        label: probability / total
        for label, probability in cleaned.items()
    }


def _mode_or_unknown(series: pd.Series) -> Any:
    """Return a series mode, or Unknown when no non-null mode exists."""
    modes = series.dropna().mode()
    if modes.empty:
        return "Unknown"
    return modes.iloc[0]


def _json_safe_value(value: Any) -> Any:
    """Convert numpy scalar values into JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def evaluate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    labels: tuple[str, ...],
    model_name: str,
    evaluate_probability_metrics: bool,
) -> dict[str, Any]:
    """Calculate training and stratified cross-validation metrics."""
    training_predictions = pipeline.predict(X)
    class_counts = y.value_counts()
    n_splits = min(5, int(class_counts.min()))

    evaluation: dict[str, Any] = {
        "training_accuracy": float(accuracy_score(y, training_predictions)),
        "cv_folds": int(n_splits),
    }

    if n_splits < 2:
        evaluation["cv_warning"] = "Not enough rows per class for cross-validation."
        return evaluation

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    cv_predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict",
    )
    evaluation.update(
        {
            "cv_accuracy": float(accuracy_score(y, cv_predictions)),
            "cv_balanced_accuracy": float(balanced_accuracy_score(y, cv_predictions)),
            "cv_macro_f1": float(f1_score(y, cv_predictions, average="macro")),
            "cv_confusion_matrix_labels": list(labels),
            "cv_confusion_matrix": confusion_matrix(
                y,
                cv_predictions,
                labels=list(labels),
            ).tolist(),
        }
    )

    if evaluate_probability_metrics:
        cv_probabilities = cross_val_predict(
            pipeline,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )
        probability_labels = [
            str(label)
            for label in pipeline.named_steps["model"].classes_
        ]
        evaluation.update(
            {
                "cv_log_loss": float(log_loss(y, cv_probabilities, labels=probability_labels)),
                "cv_probability_labels": probability_labels,
            }
        )

    metrics_slug = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        evaluation,
        OUTPUTS_DIR / f"{metrics_slug}_evaluation.json",
    )

    return evaluation


def train_one_model(config: ModelConfig) -> dict[str, Any]:
    """Train one configured model and save its model/features artifacts."""
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {config.dataset_path}")

    df = pd.read_csv(config.dataset_path)

    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' is missing from "
            f"{config.dataset_path}."
        )

    # Drop rows with a missing target. Feature missingness is handled by the
    # preprocessing pipelines.
    df = df.dropna(subset=[config.target_column]).copy()

    if config.stable_features is None:
        feature_columns = select_feature_columns(df)
    else:
        feature_columns = select_stable_feature_columns(df, config.stable_features)

    categorical_columns, numeric_columns, binary_columns = split_feature_types(
        df,
        feature_columns,
    )

    X = df[feature_columns]
    y = df[config.target_column].astype(str)

    validate_target_labels(y, config.expected_labels, config.name)

    preprocessor = build_preprocessor(
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        binary_columns=binary_columns,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", config.estimator),
        ]
    )

    pipeline.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.model_path)

    class_counts = y.value_counts().sort_index().to_dict()
    class_labels = list(pipeline.named_steps["model"].classes_)

    feature_payload = {
        "model_name": config.name,
        "dataset": str(config.dataset_path),
        "target_column": config.target_column,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "categorical_features": categorical_columns,
        "numeric_features": numeric_columns,
        "binary_features": binary_columns,
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "blocked_feature_substrings": list(BLOCKED_FEATURE_SUBSTRINGS),
    }
    save_json(feature_payload, config.feature_path)

    defaults_path: str | None = None
    base_probs_path: str | None = None
    review_prior_path: str | None = None
    publisher_prior_path: str | None = None

    if config.defaults_path is not None:
        defaults_payload = calculate_training_defaults(
            df=df,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        save_json(defaults_payload, config.defaults_path)
        defaults_path = str(config.defaults_path)

    if config.base_probs_path is not None:
        base_probs_payload = calculate_base_probabilities(
            y,
            config.expected_labels,
        )
        save_json(base_probs_payload, config.base_probs_path)
        base_probs_path = str(config.base_probs_path)

    if config.review_prior_path is not None and config.review_column is not None:
        review_prior_payload = calculate_review_reception_priors(
            df=df,
            y=y,
            expected_labels=config.expected_labels,
            review_column=config.review_column,
        )
        save_json(review_prior_payload, config.review_prior_path)
        review_prior_path = str(config.review_prior_path)

    if config.publisher_prior_path is not None and config.publisher_column is not None:
        publisher_prior_payload = calculate_publisher_support_priors(
            df=df,
            y=y,
            expected_labels=config.expected_labels,
            publisher_column=config.publisher_column,
        )
        save_json(publisher_prior_payload, config.publisher_prior_path)
        publisher_prior_path = str(config.publisher_prior_path)

    evaluation: dict[str, Any] = {}
    if config.evaluate_metrics:
        evaluation = evaluate_model(
            pipeline=pipeline,
            X=X,
            y=y,
            labels=config.expected_labels,
            model_name=config.name,
            evaluate_probability_metrics=config.evaluate_probability_metrics,
        )

    result = {
        "model_name": config.name,
        "dataset": str(config.dataset_path),
        "row_count": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "class_counts": {label: int(count) for label, count in class_counts.items()},
        "class_labels": class_labels,
        "model_path": str(config.model_path),
        "feature_path": str(config.feature_path),
        "categorical_feature_count": int(len(categorical_columns)),
        "numeric_feature_count": int(len(numeric_columns)),
        "binary_feature_count": int(len(binary_columns)),
    }

    if evaluation:
        result["evaluation"] = evaluation

    if defaults_path is not None:
        result["defaults_path"] = defaults_path
    if base_probs_path is not None:
        result["base_probs_path"] = base_probs_path
    if review_prior_path is not None:
        result["review_prior_path"] = review_prior_path
    if publisher_prior_path is not None:
        result["publisher_prior_path"] = publisher_prior_path

    print_training_summary(result)
    return result


def print_training_summary(result: dict[str, Any]) -> None:
    """Print the required high-level training summary for one model."""
    print(f"\n{result['model_name']}")
    print("-" * len(result["model_name"]))
    print(f"Dataset used: {result['dataset']}")
    print(f"Row count: {result['row_count']}")
    print(f"Feature count: {result['feature_count']}")
    print(f"Class counts: {result['class_counts']}")
    print(f"Saved model: {result['model_path']}")
    print(f"Saved features: {result['feature_path']}")
    evaluation = result.get("evaluation", {})
    if "cv_accuracy" in evaluation:
        print(f"Training accuracy: {evaluation['training_accuracy']:.3f}")
        print(f"CV accuracy: {evaluation['cv_accuracy']:.3f}")
        print(f"CV balanced accuracy: {evaluation['cv_balanced_accuracy']:.3f}")
        print(f"CV macro F1: {evaluation['cv_macro_f1']:.3f}")
        if "cv_log_loss" in evaluation:
            print(f"CV log loss: {evaluation['cv_log_loss']:.3f}")
    if "defaults_path" in result:
        print(f"Saved defaults: {result['defaults_path']}")
    if "base_probs_path" in result:
        print(f"Saved base probabilities: {result['base_probs_path']}")
    if "review_prior_path" in result:
        print(f"Saved review reception priors: {result['review_prior_path']}")
    if "publisher_prior_path" in result:
        print(f"Saved publisher support priors: {result['publisher_prior_path']}")


def build_model_configs() -> list[ModelConfig]:
    """Define the model training jobs for the project."""
    return [
        ModelConfig(
            name="Expected Users 4-Class Pre-Launch Model",
            dataset_path=DATA_DIR / "expected_users_4class_augmented_model_dataset.csv",
            target_column="outcome_level",
            model_path=MODELS_DIR / "expected_users_model.pkl",
            feature_path=MODELS_DIR / "expected_users_features.json",
            estimator=build_expected_users_estimator(),
            expected_labels=EXPECTED_USERS_4CLASS_LABELS,
            stable_features=tuple(EXPECTED_USERS_STABLE_FEATURES),
            defaults_path=MODELS_DIR / "expected_users_defaults.json",
            base_probs_path=MODELS_DIR / "expected_users_base_probs.json",
            evaluate_metrics=False,
            evaluate_probability_metrics=False,
        ),
        ModelConfig(
            name="Launch Traction 4-Class Expected Users Model",
            dataset_path=DATA_DIR / "expected_users_4class_augmented_model_dataset.csv",
            target_column="outcome_level",
            model_path=MODELS_DIR / "launch_traction_model.pkl",
            feature_path=MODELS_DIR / "launch_traction_features.json",
            estimator=build_expected_users_estimator(),
            expected_labels=EXPECTED_USERS_4CLASS_LABELS,
            stable_features=tuple(LAUNCH_TRACTION_FEATURES),
            defaults_path=MODELS_DIR / "launch_traction_defaults.json",
            base_probs_path=MODELS_DIR / "launch_traction_base_probs.json",
            review_prior_path=MODELS_DIR / "launch_traction_review_priors.json",
            review_column=REVIEW_RECEPTION_COLUMN,
            publisher_prior_path=MODELS_DIR / "launch_traction_publisher_priors.json",
            publisher_column=PUBLISHER_SUPPORT_COLUMN,
            evaluate_metrics=False,
            evaluate_probability_metrics=False,
        ),
        ModelConfig(
            name="Downside Risk Binary Model",
            dataset_path=DATA_DIR / "downside_risk_binary_model_dataset.csv",
            target_column="downside_target",
            model_path=MODELS_DIR / "downside_model.pkl",
            feature_path=MODELS_DIR / "downside_features.json",
            estimator=KNeighborsClassifier(n_neighbors=7),
            expected_labels=("Low", "MidHigh"),
            evaluate_metrics=False,
            evaluate_probability_metrics=False,
        ),
        ModelConfig(
            name="Upside Potential Binary Model",
            dataset_path=DATA_DIR / "upside_potential_binary_model_dataset.csv",
            target_column="upside_target",
            model_path=MODELS_DIR / "upside_model.pkl",
            feature_path=MODELS_DIR / "upside_features.json",
            estimator=SVC(
                probability=True,
                kernel="rbf",
                class_weight="balanced",
            ),
            expected_labels=("Mid", "High"),
            evaluate_metrics=False,
            evaluate_probability_metrics=False,
        ),
    ]


def build_expected_users_estimator() -> LogisticRegression:
    """Build the regularized multinomial logistic regression estimator.

    The project specification asks for multi_class="multinomial". Recent
    scikit-learn versions removed that keyword because multinomial behavior is
    now the default for multiclass classification with solvers such as lbfgs.
    This keeps the requested argument when the installed sklearn version still
    supports it, while remaining runnable on newer versions.
    """
    estimator_params: dict[str, Any] = {
        "solver": "lbfgs",
        "max_iter": 2000,
        "C": 0.25,
        "class_weight": "balanced",
    }

    if "multi_class" in signature(LogisticRegression).parameters:
        estimator_params["multi_class"] = "multinomial"

    return LogisticRegression(**estimator_params)


def main() -> None:
    """Train all models and write shared metadata."""
    results = [train_one_model(config) for config in build_model_configs()]

    metadata = {
        "project": "senior_project_webtool",
        "purpose": (
            "Estimate expected users and financial viability for indie "
            "turn-based strategy games."
        ),
        "outcome_tier_users": OUTCOME_TIER_USERS,
        "disclaimer": DISCLAIMER,
        "models": {
            "expected_users": results[0],
            "launch_traction": results[1],
            "downside_risk": results[2],
            "upside_potential": results[3],
        },
        "review_reception_adjustment": {
            "review_column": REVIEW_RECEPTION_COLUMN,
            "target_value": REVIEW_RECEPTION_TARGET_VALUE,
            "prior_strength": REVIEW_RECEPTION_PRIOR_STRENGTH,
            "scenario_uplift": REVIEW_RECEPTION_SCENARIO_UPLIFT,
            "method": (
                "Review sentiment is removed from the core launch-traction "
                "classifier. Positive reception is applied afterward as a 15% "
                "expected-user scenario uplift."
            ),
        },
        "publisher_support_adjustment": {
            "publisher_column": PUBLISHER_SUPPORT_COLUMN,
            "prior_strength": PUBLISHER_SUPPORT_PRIOR_STRENGTH,
            "scenario_uplift": PUBLISHER_SUPPORT_SCENARIO_UPLIFT,
            "method": (
                "Publisher support is removed from the core launch-traction "
                "classifier. Publishing Studio is applied afterward as a "
                "17.62% expected-user scenario uplift before the separate "
                "publisher partnership slider."
            ),
        },
    }

    metadata_path = MODELS_DIR / "model_metadata.json"
    save_json(metadata, metadata_path)

    print("\nShared metadata")
    print("---------------")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
