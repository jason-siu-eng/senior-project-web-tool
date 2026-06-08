"""Model loading, prediction, and financial helpers for the Streamlit app.

The trained model artifacts are sklearn Pipelines saved with joblib. Each
pipeline contains both preprocessing and the final classifier, so the app only
needs to provide a one-row DataFrame with the raw feature columns used during
training.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

# Avoid noisy joblib/loky CPU detection warnings in local Streamlit runs. If
# the user wants a different setting, their existing environment value wins.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd


MODELS_DIR = Path("models")

EXPECTED_USERS_MODEL_PATH = MODELS_DIR / "expected_users_model.pkl"
LAUNCH_TRACTION_MODEL_PATH = MODELS_DIR / "launch_traction_model.pkl"
DOWNSIDE_MODEL_PATH = MODELS_DIR / "downside_model.pkl"
UPSIDE_MODEL_PATH = MODELS_DIR / "upside_model.pkl"

EXPECTED_USERS_FEATURES_PATH = MODELS_DIR / "expected_users_features.json"
LAUNCH_TRACTION_FEATURES_PATH = MODELS_DIR / "launch_traction_features.json"
DOWNSIDE_FEATURES_PATH = MODELS_DIR / "downside_features.json"
UPSIDE_FEATURES_PATH = MODELS_DIR / "upside_features.json"
EXPECTED_USERS_DEFAULTS_PATH = MODELS_DIR / "expected_users_defaults.json"
EXPECTED_USERS_BASE_PROBS_PATH = MODELS_DIR / "expected_users_base_probs.json"
LAUNCH_TRACTION_DEFAULTS_PATH = MODELS_DIR / "launch_traction_defaults.json"
LAUNCH_TRACTION_BASE_PROBS_PATH = MODELS_DIR / "launch_traction_base_probs.json"
LAUNCH_TRACTION_REVIEW_PRIORS_PATH = (
    MODELS_DIR / "launch_traction_review_priors.json"
)
LAUNCH_TRACTION_PUBLISHER_PRIORS_PATH = (
    MODELS_DIR / "launch_traction_publisher_priors.json"
)
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"


DEFAULT_OUTCOME_TIER_USERS = {
    "Not Viable": 10000,
    "Low": 35000,
    "Mid": 75000,
    "High": 150000,
}
OUTCOME_LABEL_ORDER = ("Not Viable", "Low", "Mid", "High")


EXPECTED_USERS_SMOOTHING_WEIGHT = 0.60
REVIEW_RECEPTION_TARGET_VALUE = "Positive"
REVIEW_RECEPTION_SCENARIO_UPLIFT = 0.15
PUBLISHER_SUPPORT_TARGET_VALUE = "Publishing Studio"
PUBLISHER_SUPPORT_SCENARIO_UPLIFT = 0.1762
PUBLISHER_PARTNERSHIP_DEFAULT_UPLIFT = 0.0
OVERCONFIDENCE_THRESHOLD = 0.90
OVERCONFIDENCE_NOTE = (
    "Raw model probability was extreme, so smoothed probabilities are shown."
)
EXPECTED_USERS_FEATURE_CAPS = {
    "prelaunch_price_usd": (0.0, 29.99),
    "prelaunch_language_count": (1.0, 20.0),
    "prelaunch_screenshots_count": (4.0, 25.0),
    "prelaunch_movies_count": (0.0, 9.0),
    "prelaunch_achievements_count": (0.0, 185.0),
    "prelaunch_tag_count": (3.0, 20.0),
}
LAUNCH_TRACTION_FEATURE_CAPS = {
    "prelaunch_price_usd": (0.0, 29.99),
    "prelaunch_achievements_count": (0.0, 185.0),
    "total_reviews": (52.0, 10317.0),
    "log_total_reviews": (3.970291913552122, 9.241645221804594),
}
REVIEW_SCENARIO_NOTE = (
    "Review sentiment scenario adjustment, not a pre-launch observable predictor."
)
SALES_CONFIDENCE_METHODOLOGY_NOTE = (
    "Sales confidence was used in the research analysis to evaluate the "
    "reliability of SteamSpy ownership estimates. It is not used as a "
    "pre-launch user input because it depends on post-launch review/ownership "
    "support."
)


# Keep a known local joblib/loky CPU-count warning out of Streamlit logs.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"joblib\.externals\.loky\.backend\.context",
)


def load_model(model_name: str) -> Any:
    """Load one model from the models/ folder.

    This helper remains available for small one-off loads. The Streamlit app
    should usually call load_models() so all pipelines, feature definitions,
    and metadata are loaded together.
    """
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def save_model(model: Any, model_name: str) -> None:
    """Save one trained model to the models/ folder with joblib."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / model_name
    joblib.dump(model, model_path)


def load_models(
    include_expected_model: bool = True,
    include_support_models: bool = True,
) -> dict[str, Any]:
    """Load all model pipelines, feature lists, feature metadata, and metadata.

    Returns a dictionary with predictable keys:
    - launch_traction_model
    - launch_traction_features
    - launch_traction_feature_info
    - metadata

    When include_expected_model is True, the pre-launch expected-users model is
    loaded for baseline comparison.

    When include_support_models is True, the downside and upside support
    models are also loaded.

    A FileNotFoundError is raised before loading anything if one or more
    required artifacts are missing.
    """
    required_paths = [
        LAUNCH_TRACTION_MODEL_PATH,
        LAUNCH_TRACTION_FEATURES_PATH,
        LAUNCH_TRACTION_DEFAULTS_PATH,
        LAUNCH_TRACTION_BASE_PROBS_PATH,
        LAUNCH_TRACTION_REVIEW_PRIORS_PATH,
        LAUNCH_TRACTION_PUBLISHER_PRIORS_PATH,
        MODEL_METADATA_PATH,
    ]
    if include_expected_model:
        required_paths.extend(
            [
                EXPECTED_USERS_MODEL_PATH,
                EXPECTED_USERS_FEATURES_PATH,
                EXPECTED_USERS_DEFAULTS_PATH,
                EXPECTED_USERS_BASE_PROBS_PATH,
            ]
        )
    if include_support_models:
        required_paths.extend(
            [
                DOWNSIDE_MODEL_PATH,
                UPSIDE_MODEL_PATH,
                DOWNSIDE_FEATURES_PATH,
                UPSIDE_FEATURES_PATH,
            ]
        )
    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        missing_list = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing required model artifact(s). Run train_models.py first:\n"
            f"{missing_list}"
        )

    launch_traction_feature_info = _load_json(LAUNCH_TRACTION_FEATURES_PATH)
    launch_traction_defaults = _load_json(LAUNCH_TRACTION_DEFAULTS_PATH)
    launch_traction_base_probs = _load_json(LAUNCH_TRACTION_BASE_PROBS_PATH)
    launch_traction_review_priors = _load_json(LAUNCH_TRACTION_REVIEW_PRIORS_PATH)
    launch_traction_publisher_priors = _load_json(
        LAUNCH_TRACTION_PUBLISHER_PRIORS_PATH
    )
    metadata = _load_json(MODEL_METADATA_PATH)

    _validate_feature_info(launch_traction_feature_info, LAUNCH_TRACTION_FEATURES_PATH)
    _validate_expected_defaults(launch_traction_defaults)
    launch_traction_feature_info = {
        **launch_traction_feature_info,
        "defaults": launch_traction_defaults["defaults"],
    }

    bundle = {
        "launch_traction_model": joblib.load(LAUNCH_TRACTION_MODEL_PATH),
        "launch_traction_features": launch_traction_feature_info["features"],
        "launch_traction_feature_info": launch_traction_feature_info,
        "launch_traction_defaults": launch_traction_defaults,
        "launch_traction_base_probs": launch_traction_base_probs,
        "launch_traction_review_priors": launch_traction_review_priors,
        "launch_traction_publisher_priors": launch_traction_publisher_priors,
        "metadata": metadata,
    }

    if include_expected_model:
        expected_users_feature_info = _load_json(EXPECTED_USERS_FEATURES_PATH)
        expected_users_defaults = _load_json(EXPECTED_USERS_DEFAULTS_PATH)
        expected_users_base_probs = _load_json(EXPECTED_USERS_BASE_PROBS_PATH)
        _validate_feature_info(expected_users_feature_info, EXPECTED_USERS_FEATURES_PATH)
        _validate_expected_defaults(expected_users_defaults)
        expected_users_feature_info = {
            **expected_users_feature_info,
            "defaults": expected_users_defaults["defaults"],
        }
        bundle.update(
            {
                "expected_users_model": joblib.load(EXPECTED_USERS_MODEL_PATH),
                "expected_users_features": expected_users_feature_info["features"],
                "expected_users_feature_info": expected_users_feature_info,
                "expected_users_defaults": expected_users_defaults,
                "expected_users_base_probs": expected_users_base_probs,
            }
        )

    if include_support_models:
        downside_feature_info = _load_json(DOWNSIDE_FEATURES_PATH)
        upside_feature_info = _load_json(UPSIDE_FEATURES_PATH)
        _validate_feature_info(downside_feature_info, DOWNSIDE_FEATURES_PATH)
        _validate_feature_info(upside_feature_info, UPSIDE_FEATURES_PATH)
        bundle.update(
            {
                "downside_model": joblib.load(DOWNSIDE_MODEL_PATH),
                "upside_model": joblib.load(UPSIDE_MODEL_PATH),
                "downside_features": downside_feature_info["features"],
                "upside_features": upside_feature_info["features"],
                "downside_feature_info": downside_feature_info,
                "upside_feature_info": upside_feature_info,
            }
        )

    return bundle


def make_feature_row(
    user_inputs: Mapping[str, Any],
    feature_list: Sequence[str] | Mapping[str, Any],
) -> pd.DataFrame:
    """Create a one-row DataFrame with exactly the expected raw features.

    feature_list can be either:
    - a plain list of feature names, or
    - a feature metadata dictionary loaded from the feature JSON file.

    When the feature metadata dictionary is available, this function uses its
    categorical/numeric/binary lists for safer defaults. With a plain list, it
    falls back to conservative name-based type inference.
    """
    (
        feature_names,
        categorical_features,
        numeric_features,
        binary_features,
        defaults,
    ) = (
        _extract_feature_spec(feature_list)
    )

    row: dict[str, Any] = {}

    for feature in feature_names:
        if feature in user_inputs:
            row[feature] = user_inputs[feature]
        elif feature in defaults:
            row[feature] = defaults[feature]
        elif feature in categorical_features:
            row[feature] = "Unknown"
        elif feature in binary_features:
            row[feature] = 0
        elif feature in numeric_features:
            row[feature] = 0
        else:
            row[feature] = _default_value_from_name(feature)

    return pd.DataFrame([row], columns=feature_names)


def predict_expected_users(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict outcome-tier probabilities and probability-weighted users."""
    model = bundle["expected_users_model"]
    capped_user_inputs, cap_messages = cap_expected_users_inputs(user_inputs)
    feature_row = make_feature_row(
        capped_user_inputs,
        bundle.get("expected_users_feature_info", bundle["expected_users_features"]),
    )
    probability_by_class = _predict_probability_by_class(model, feature_row)
    tier_users = _outcome_tier_users(bundle)
    outcome_labels = _outcome_labels(tier_users, probability_by_class)
    raw_probability_by_class = {
        label: probability_by_class.get(label, 0.0)
        for label in outcome_labels
    }
    base_probability_by_class = _expected_users_base_probs(bundle, outcome_labels)
    smoothed_probability_by_class = _smooth_probabilities(
        model_probs=raw_probability_by_class,
        base_probs=base_probability_by_class,
        smoothing_weight=EXPECTED_USERS_SMOOTHING_WEIGHT,
        labels=outcome_labels,
    )

    expected_users = sum(
        smoothed_probability_by_class[label] * tier_users[label]
        for label in outcome_labels
    )
    overconfidence_warning = any(
        probability > OVERCONFIDENCE_THRESHOLD
        for probability in raw_probability_by_class.values()
    )

    result = {
        "probabilities": dict(smoothed_probability_by_class),
        "raw_probabilities": dict(raw_probability_by_class),
        "expected_users": expected_users,
        "predicted_class": max(smoothed_probability_by_class, key=smoothed_probability_by_class.get),
        "raw_predicted_class": str(model.predict(feature_row)[0]),
        "smoothing_weight": EXPECTED_USERS_SMOOTHING_WEIGHT,
        "overconfidence_warning": overconfidence_warning,
        "input_caps_applied": bool(cap_messages),
        "input_cap_messages": cap_messages,
    }

    for label in outcome_labels:
        result[_probability_key(label)] = smoothed_probability_by_class[label]
        result[_raw_probability_key(label)] = raw_probability_by_class[label]

    if overconfidence_warning:
        result["overconfidence_note"] = OVERCONFIDENCE_NOTE

    return result


def cap_expected_users_inputs(
    user_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Clamp expected-users numeric inputs to the model's training-safe ranges."""
    capped_inputs = dict(user_inputs)
    cap_messages: list[str] = []

    for feature, (minimum, maximum) in EXPECTED_USERS_FEATURE_CAPS.items():
        if feature not in capped_inputs:
            continue

        numeric_value = _try_float(capped_inputs[feature])
        if numeric_value is None:
            continue

        capped_value = min(max(numeric_value, minimum), maximum)
        if capped_value != numeric_value:
            capped_inputs[feature] = capped_value
            cap_messages.append(
                f"{feature} was capped from {numeric_value:g} to {capped_value:g} "
                f"to stay within the model training range ({minimum:g}-{maximum:g})."
            )

    return capped_inputs, cap_messages


def predict_launch_traction_users(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict expected users from assumed launch review/traction signals.

    This is not a pure pre-launch model. It answers a scenario question:
    if the game reaches the entered review count and sentiment, what outcome
    tier does that resemble historically?
    """
    model = bundle["launch_traction_model"]
    capped_user_inputs, cap_messages = prepare_launch_traction_inputs(user_inputs)
    feature_row = make_feature_row(
        capped_user_inputs,
        bundle.get("launch_traction_feature_info", bundle["launch_traction_features"]),
    )
    probability_by_class = _predict_probability_by_class(model, feature_row)
    tier_users = _outcome_tier_users(bundle)
    outcome_labels = _outcome_labels(tier_users, probability_by_class)
    core_probability_by_class = _normalize_probabilities(
        {
            label: probability_by_class.get(label, 0.0)
            for label in outcome_labels
        }
    )
    core_expected_users = _probability_weighted_users(
        probabilities=core_probability_by_class,
        tier_users=tier_users,
        labels=outcome_labels,
    )
    review_adjustment = apply_review_reception_smoothing(
        bundle=bundle,
        user_inputs=user_inputs,
        model_probabilities=core_probability_by_class,
        labels=outcome_labels,
        tier_users=tier_users,
    )
    review_probability_by_class = review_adjustment["adjusted_probabilities"]
    review_adjusted_expected_users = review_adjustment["adjusted_expected_users"]
    publisher_adjustment = apply_publisher_support_smoothing(
        bundle=bundle,
        user_inputs=user_inputs,
        model_probabilities=review_probability_by_class,
        labels=outcome_labels,
        tier_users=tier_users,
        base_expected_users=review_adjusted_expected_users,
    )
    probability_by_class = publisher_adjustment["adjusted_probabilities"]
    expected_users = publisher_adjustment["adjusted_expected_users"]
    publisher_partnership_scenario = apply_publisher_partnership_scenario(
        user_inputs=user_inputs,
        base_expected_users=expected_users,
    )
    overconfidence_warning = any(
        probability > OVERCONFIDENCE_THRESHOLD
        for probability in probability_by_class.values()
    )

    result = {
        "traction_core_probabilities": dict(core_probability_by_class),
        "traction_review_adjusted_probabilities": dict(review_probability_by_class),
        "traction_probabilities": dict(probability_by_class),
        "core_launch_traction_expected_users": core_expected_users,
        "review_adjusted_expected_users": review_adjusted_expected_users,
        "launch_traction_expected_users": expected_users,
        "financial_expected_users": publisher_partnership_scenario[
            "publisher_partnership_expected_users"
        ],
        "launch_traction_model_predicted_class": str(model.predict(feature_row)[0]),
        "launch_traction_predicted_class": max(
            probability_by_class,
            key=probability_by_class.get,
        ),
        "launch_traction_input_caps_applied": bool(cap_messages),
        "launch_traction_input_cap_messages": cap_messages,
        "launch_traction_inputs_used": {
            feature: capped_user_inputs.get(feature)
            for feature in bundle.get("launch_traction_features", [])
        },
        "launch_traction_note": (
            "Launch traction scenario uses assumed post-launch review signals. "
            "It should be used for goal-setting, not as a pure pre-launch input. "
            "Review reception and publisher support are handled as explicit "
            "expected-user scenario adjustments."
        ),
        "launch_traction_overconfidence_warning": overconfidence_warning,
        "review_reception_adjustment": review_adjustment,
        "review_reception_adjustment_applied": review_adjustment["applied"],
        "review_reception_expected_user_delta": review_adjustment[
            "expected_user_delta"
        ],
        "publisher_support_adjustment": publisher_adjustment,
        "publisher_support_adjustment_applied": publisher_adjustment["applied"],
        "publisher_support_expected_user_delta": publisher_adjustment[
            "expected_user_delta"
        ],
        "publisher_partnership_scenario": publisher_partnership_scenario,
        "publisher_partnership_expected_users": publisher_partnership_scenario[
            "publisher_partnership_expected_users"
        ],
    }

    for label in outcome_labels:
        result[f"traction_{_probability_key(label)}"] = probability_by_class[label]
        result[f"core_traction_{_probability_key(label)}"] = (
            core_probability_by_class[label]
        )
        result[f"review_traction_{_probability_key(label)}"] = (
            review_probability_by_class[label]
        )

    if overconfidence_warning:
        result["launch_traction_overconfidence_note"] = OVERCONFIDENCE_NOTE

    return result


def apply_review_reception_smoothing(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
    model_probabilities: Mapping[str, float],
    labels: Sequence[str],
    tier_users: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the Positive-review expected-user scenario adjustment.

    The core launch-traction classifier intentionally excludes review_sentiment
    and positive review rate because the fitted positive-rate effect was not
    directionally stable. For a Positive reception scenario, this applies a
    fixed expected-user uplift from the prior smaller-dataset analysis. The
    class probabilities are left unchanged and shown as core model context.
    """
    priors_info = bundle.get("launch_traction_review_priors", {})
    review_column = priors_info.get("review_column", "review_sentiment")
    target_value = priors_info.get("target_value", REVIEW_RECEPTION_TARGET_VALUE)
    review_value = str(user_inputs.get(review_column, "Mixed") or "Mixed")
    core_probabilities = _normalize_probabilities(
        {
            label: _to_float(model_probabilities.get(label), 0.0)
            for label in labels
        }
    )
    core_expected_users = _probability_weighted_users(
        probabilities=core_probabilities,
        tier_users=tier_users,
        labels=labels,
    )
    review_priors = priors_info.get("review_priors", {})
    review_prior = (
        review_priors.get(review_value)
        if isinstance(review_priors, Mapping)
        else None
    )
    prior_probabilities = {}
    sample_size = 0
    shrink_weight = 0.0
    if isinstance(review_prior, Mapping):
        prior_probabilities = _normalize_probabilities(
            {
                label: _to_float(
                    review_prior.get("smoothed_distribution", {}).get(label),
                    0.0,
                )
                for label in labels
            }
        )
        sample_size = int(review_prior.get("sample_size", 0))
        shrink_weight = _to_float(review_prior.get("shrink_weight"), 0.0)

    empty_result = {
        "applied": False,
        "review_column": review_column,
        "review_value": review_value,
        "target_value": target_value,
        "blend_weight": 0.0,
        "uplift_rate": 0.0,
        "adjustment_multiplier": 1.0,
        "adjustment_type": "expected_user_multiplier",
        "core_expected_users": core_expected_users,
        "adjusted_expected_users": core_expected_users,
        "expected_user_delta": 0.0,
        "prior_probabilities": dict(prior_probabilities),
        "adjusted_probabilities": dict(core_probabilities),
        "sample_size": sample_size,
        "shrink_weight": shrink_weight,
        "note": (
            "No Positive review reception scenario uplift was applied. The core "
            "launch traction estimate is shown."
        ),
    }

    if review_value != target_value:
        return empty_result

    uplift_rate = max(
        _to_float(
            priors_info.get("scenario_uplift"),
            REVIEW_RECEPTION_SCENARIO_UPLIFT,
        ),
        0.0,
    )
    adjustment_multiplier = 1 + uplift_rate
    adjusted_expected_users = core_expected_users * adjustment_multiplier

    return {
        "applied": True,
        "review_column": review_column,
        "review_value": review_value,
        "target_value": target_value,
        "blend_weight": 0.0,
        "uplift_rate": uplift_rate,
        "adjustment_multiplier": adjustment_multiplier,
        "adjustment_type": "expected_user_multiplier",
        "core_expected_users": core_expected_users,
        "adjusted_expected_users": adjusted_expected_users,
        "expected_user_delta": adjusted_expected_users - core_expected_users,
        "prior_probabilities": dict(prior_probabilities),
        "adjusted_probabilities": dict(core_probabilities),
        "sample_size": sample_size,
        "shrink_weight": shrink_weight,
        "note": (
            "Positive reception applies a 15% expected-user scenario uplift "
            "based on the prior smaller-dataset review sentiment finding. "
            "The core model probabilities are unchanged."
        ),
    }


def apply_publisher_partnership_scenario(
    user_inputs: Mapping[str, Any],
    base_expected_users: float,
) -> dict[str, Any]:
    """Apply an explicit publisher partnership uplift scenario.

    This is separate from the empirical publisher-support prior. It represents
    a user-controlled business-case assumption for marketing, visibility, and
    distribution help that a publisher might provide.
    """
    publisher_type = str(
        user_inputs.get("prelaunch_publisher_type", "Unknown") or "Unknown"
    )
    publisher_fee = _as_rate(user_inputs.get("publisher_fee"), 0.0)
    selected_uplift = _as_rate(
        user_inputs.get("publisher_user_uplift_scenario"),
        PUBLISHER_PARTNERSHIP_DEFAULT_UPLIFT,
    )
    base_expected_users = max(_to_float(base_expected_users, 0.0), 0.0)
    publisher_funded_amount = max(
        _to_float(user_inputs.get("publisher_funded_amount"), 0.0),
        0.0,
    )
    list_price = _to_float(user_inputs.get("list_price"), 0.0)
    discount_rate = _as_rate(user_inputs.get("discount_rate"), 0.0)
    platform_fee = _as_rate(user_inputs.get("platform_fee"), 0.0)

    if publisher_fee >= 1:
        raw_required_uplift = math.inf
    else:
        raw_required_uplift = 1 / (1 - publisher_fee) - 1

    effective_price = list_price * (1 - discount_rate)
    independent_net_revenue_per_user = effective_price * (1 - platform_fee)
    publisher_net_revenue_per_user = (
        independent_net_revenue_per_user
        * (1 - publisher_fee)
    )
    baseline_independent_revenue = (
        base_expected_users
        * independent_net_revenue_per_user
    )
    revenue_gap_after_funding = max(
        baseline_independent_revenue - publisher_funded_amount,
        0.0,
    )

    if base_expected_users <= 0:
        funding_adjusted_required_users = 0.0 if revenue_gap_after_funding <= 0 else math.inf
    elif publisher_net_revenue_per_user <= 0:
        funding_adjusted_required_users = 0.0 if revenue_gap_after_funding <= 0 else math.inf
    else:
        funding_adjusted_required_users = (
            revenue_gap_after_funding / publisher_net_revenue_per_user
        )

    if math.isinf(funding_adjusted_required_users):
        funding_adjusted_required_uplift = math.inf
    elif base_expected_users <= 0:
        funding_adjusted_required_uplift = 0.0
    else:
        funding_adjusted_required_uplift = max(
            funding_adjusted_required_users / base_expected_users - 1,
            0.0,
        )
        if funding_adjusted_required_uplift < 1e-9:
            funding_adjusted_required_uplift = 0.0

    if math.isinf(raw_required_uplift) or math.isinf(funding_adjusted_required_uplift):
        funding_reduction_to_required_uplift = 0.0
    else:
        funding_reduction_to_required_uplift = max(
            raw_required_uplift - funding_adjusted_required_uplift,
            0.0,
        )

    if publisher_type != PUBLISHER_SUPPORT_TARGET_VALUE:
        selected_uplift = 0.0
        adjusted_expected_users = base_expected_users
        note = (
            "No publisher partnership uplift was applied because Publishing "
            "Studio support was not selected."
        )
    else:
        adjusted_expected_users = base_expected_users * (1 + selected_uplift)
        note = (
            "Publisher partnership uplift is a user-selected business-case "
            "scenario for marketing and user acquisition, not a direct model "
            "probability."
        )

    if math.isinf(funding_adjusted_required_uplift):
        break_even_met = False
    else:
        break_even_met = selected_uplift + 1e-9 >= funding_adjusted_required_uplift

    return {
        "publisher_type": publisher_type,
        "publisher_fee": publisher_fee,
        "publisher_funded_amount": publisher_funded_amount,
        "selected_uplift": selected_uplift,
        "raw_required_uplift_to_offset_fee": raw_required_uplift,
        "required_uplift_to_offset_fee": funding_adjusted_required_uplift,
        "funding_reduction_to_required_uplift": funding_reduction_to_required_uplift,
        "break_even_uplift_met": break_even_met,
        "base_expected_users": base_expected_users,
        "publisher_partnership_expected_users": adjusted_expected_users,
        "expected_user_delta": adjusted_expected_users - base_expected_users,
        "baseline_independent_revenue": baseline_independent_revenue,
        "independent_net_revenue_per_user": independent_net_revenue_per_user,
        "publisher_net_revenue_per_user": publisher_net_revenue_per_user,
        "funding_adjusted_required_users": funding_adjusted_required_users,
        "note": note,
    }


def apply_publisher_support_smoothing(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
    model_probabilities: Mapping[str, float],
    labels: Sequence[str],
    tier_users: Mapping[str, float],
    base_expected_users: float | None = None,
) -> dict[str, Any]:
    """Apply the Publishing Studio expected-user scenario adjustment.

    The core launch-traction classifier intentionally excludes publisher type.
    For a Publishing Studio scenario, this applies the 17.62% expected-user
    uplift from the prior smaller-dataset analysis before the separate
    user-controlled publisher partnership slider. The class probabilities are
    left unchanged and shown as core model context.
    """
    priors_info = bundle.get("launch_traction_publisher_priors", {})
    publisher_column = priors_info.get(
        "publisher_column",
        "prelaunch_publisher_type",
    )
    publisher_value = str(
        user_inputs.get(publisher_column, "Unknown") or "Unknown"
    )
    core_probabilities = _normalize_probabilities(
        {
            label: _to_float(model_probabilities.get(label), 0.0)
            for label in labels
        }
    )
    probability_weighted_users = _probability_weighted_users(
        probabilities=core_probabilities,
        tier_users=tier_users,
        labels=labels,
    )
    core_expected_users = (
        probability_weighted_users
        if base_expected_users is None
        else max(_to_float(base_expected_users, probability_weighted_users), 0.0)
    )
    publisher_priors = priors_info.get("publisher_priors", {})
    publisher_prior = (
        publisher_priors.get(publisher_value)
        if isinstance(publisher_priors, Mapping)
        else None
    )
    prior_probabilities = {}
    sample_size = 0
    shrink_weight = 0.0
    if isinstance(publisher_prior, Mapping):
        prior_probabilities = _normalize_probabilities(
            {
                label: _to_float(
                    publisher_prior.get("smoothed_distribution", {}).get(label),
                    0.0,
                )
                for label in labels
            }
        )
        sample_size = int(publisher_prior.get("sample_size", 0))
        shrink_weight = _to_float(publisher_prior.get("shrink_weight"), 0.0)

    empty_result = {
        "applied": False,
        "publisher_column": publisher_column,
        "publisher_value": publisher_value,
        "blend_weight": 0.0,
        "uplift_rate": 0.0,
        "adjustment_multiplier": 1.0,
        "adjustment_type": "expected_user_multiplier",
        "probability_weighted_users": probability_weighted_users,
        "core_expected_users": core_expected_users,
        "adjusted_expected_users": core_expected_users,
        "expected_user_delta": 0.0,
        "prior_probabilities": dict(prior_probabilities),
        "adjusted_probabilities": dict(core_probabilities),
        "sample_size": sample_size,
        "shrink_weight": shrink_weight,
        "note": (
            "No Publishing Studio scenario uplift was applied. The current "
            "expected-user estimate is shown."
        ),
    }

    if publisher_value != PUBLISHER_SUPPORT_TARGET_VALUE:
        return empty_result

    uplift_rate = max(
        _to_float(
            priors_info.get("scenario_uplift"),
            PUBLISHER_SUPPORT_SCENARIO_UPLIFT,
        ),
        0.0,
    )
    adjustment_multiplier = 1 + uplift_rate
    adjusted_expected_users = core_expected_users * adjustment_multiplier

    return {
        "applied": True,
        "publisher_column": publisher_column,
        "publisher_value": publisher_value,
        "blend_weight": 0.0,
        "uplift_rate": uplift_rate,
        "adjustment_multiplier": adjustment_multiplier,
        "adjustment_type": "expected_user_multiplier",
        "probability_weighted_users": probability_weighted_users,
        "core_expected_users": core_expected_users,
        "adjusted_expected_users": adjusted_expected_users,
        "expected_user_delta": adjusted_expected_users - core_expected_users,
        "prior_probabilities": dict(prior_probabilities),
        "adjusted_probabilities": dict(core_probabilities),
        "sample_size": sample_size,
        "shrink_weight": shrink_weight,
        "note": (
            "Publishing Studio support applies a 17.62% expected-user scenario "
            "uplift based on the prior smaller-dataset publisher finding. This "
            "is applied before the separate publisher partnership uplift slider."
        ),
    }


def prepare_launch_traction_inputs(
    user_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize and cap launch traction scenario inputs."""
    prepared_inputs = dict(user_inputs)
    cap_messages: list[str] = []

    if "positive_review_pct" in prepared_inputs:
        prepared_inputs["positive_review_pct"] = _as_rate(
            prepared_inputs["positive_review_pct"],
            0.80,
        )

    if "total_reviews" in prepared_inputs:
        total_reviews = _try_float(prepared_inputs["total_reviews"])
        if total_reviews is not None:
            prepared_inputs["log_total_reviews"] = math.log1p(max(total_reviews, 0.0))

    for feature, (minimum, maximum) in LAUNCH_TRACTION_FEATURE_CAPS.items():
        if feature not in prepared_inputs:
            continue

        numeric_value = _try_float(prepared_inputs[feature])
        if numeric_value is None:
            continue

        capped_value = min(max(numeric_value, minimum), maximum)
        if capped_value != numeric_value:
            prepared_inputs[feature] = capped_value
            cap_messages.append(
                f"{feature} was capped from {numeric_value:g} to {capped_value:g} "
                f"to stay within the launch traction training range "
                f"({minimum:g}-{maximum:g})."
            )

            if feature == "total_reviews":
                prepared_inputs["log_total_reviews"] = math.log1p(capped_value)

    return prepared_inputs, cap_messages


def apply_review_scenario_adjustment(
    expected_users_result: Mapping[str, Any],
    review_scenario: str,
) -> dict[str, Any]:
    """Apply an optional post-launch review sentiment scenario adjustment.

    Review sentiment is intentionally not part of the pre-launch model because
    it is not known before launch. This helper keeps the base model estimate
    separate from scenario analysis.
    """
    base_expected_users = _to_float(
        expected_users_result.get("expected_users"),
        0.0,
    )
    normalized_scenario = str(review_scenario or "Not included").strip()

    if normalized_scenario == "Positive reception":
        multiplier = 1.15
        adjusted_expected_users = base_expected_users * multiplier
        scenario_note = (
            "Positive reception scenario applies a 15% expected-user uplift "
            "based on prior review sentiment analysis."
        )
    elif normalized_scenario == "Mixed reception":
        multiplier = 1.0
        adjusted_expected_users = base_expected_users
        scenario_note = "Mixed reception scenario uses the base expected user estimate."
    else:
        normalized_scenario = "Not included"
        multiplier = 1.0
        adjusted_expected_users = base_expected_users
        scenario_note = "No review sentiment scenario adjustment is applied."

    return {
        "review_scenario": normalized_scenario,
        "base_expected_users": base_expected_users,
        "review_adjusted_expected_users": adjusted_expected_users,
        "financial_expected_users": adjusted_expected_users,
        "review_adjustment_multiplier": multiplier,
        "review_scenario_note": scenario_note,
        "review_methodology_note": REVIEW_SCENARIO_NOTE,
    }


def predict_downside_risk(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict downside risk probabilities and convert P(Low) to a label."""
    model = bundle["downside_model"]
    feature_row = make_feature_row(
        user_inputs,
        bundle.get("downside_feature_info", bundle["downside_features"]),
    )
    probability_by_class = _predict_probability_by_class(model, feature_row)

    p_low = probability_by_class.get("Low", 0.0)
    p_mid_high = probability_by_class.get("MidHigh", 0.0)

    return {
        "P_Low_downside": p_low,
        "P_MidHigh": p_mid_high,
        "downside_risk_label": _downside_risk_label(p_low),
    }


def predict_upside_potential(
    bundle: Mapping[str, Any],
    user_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Predict Mid/High probabilities for the conditional upside model."""
    model = bundle["upside_model"]
    feature_row = make_feature_row(
        user_inputs,
        bundle.get("upside_feature_info", bundle["upside_features"]),
    )
    probability_by_class = _predict_probability_by_class(model, feature_row)

    p_mid = probability_by_class.get("Mid", 0.0)
    p_high = probability_by_class.get("High", 0.0)

    return {
        "P_Mid_given_MidHigh": p_mid,
        "P_High_given_MidHigh": p_high,
        "upside_label": _upside_label(p_high),
    }


def calculate_financials(
    user_inputs: Mapping[str, Any],
    expected_users: float,
    probs: Mapping[str, float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate project financials for tier and Expected scenarios.

    Percentage-style inputs can be provided as decimals such as 0.30 or whole
    percentages such as 30. Values are normalized into the 0.0 to 1.0 range.
    Divide-by-zero cases return safe values instead of raising errors.
    """
    development_budget = _to_float(user_inputs.get("development_budget"), 0.0)
    marketing_budget = _to_float(user_inputs.get("marketing_budget"), 0.0)
    post_launch_support_budget = _to_float(
        user_inputs.get("post_launch_support_budget"),
        0.0,
    )
    publisher_funded_amount = _to_float(
        user_inputs.get("publisher_funded_amount"),
        0.0,
    )

    contingency_pct = _as_rate(user_inputs.get("contingency_pct"), 0.0)
    discount_rate = _as_rate(user_inputs.get("discount_rate"), 0.0)
    platform_fee = _as_rate(user_inputs.get("platform_fee"), 0.0)
    publisher_fee = _as_rate(user_inputs.get("publisher_fee"), 0.0)

    list_price = _to_float(user_inputs.get("list_price"), 0.0)

    contingency_amount = development_budget * contingency_pct
    total_project_cost = (
        development_budget
        + marketing_budget
        + post_launch_support_budget
        + contingency_amount
        - publisher_funded_amount
    )
    effective_price = list_price * (1 - discount_rate)
    net_revenue_per_user = (
        effective_price
        * (1 - platform_fee)
        * (1 - publisher_fee)
    )

    break_even_units = _break_even_units(
        total_project_cost,
        net_revenue_per_user,
    )

    scenario_users = dict(DEFAULT_OUTCOME_TIER_USERS)
    scenario_users["Expected"] = expected_users

    rows = [
        _financial_scenario_row(
            scenario=scenario,
            users=users,
            total_project_cost=total_project_cost,
            net_revenue_per_user=net_revenue_per_user,
        )
        for scenario, users in scenario_users.items()
    ]

    scenario_table = pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "users",
            "net_revenue",
            "margin_of_safety",
            "margin_multiple",
            "risk_label",
        ],
    )

    summary = {
        "development_budget": development_budget,
        "marketing_budget": marketing_budget,
        "post_launch_support_budget": post_launch_support_budget,
        "contingency_pct": contingency_pct,
        "contingency_amount": contingency_amount,
        "publisher_funded_amount": publisher_funded_amount,
        "total_project_cost": total_project_cost,
        "list_price": list_price,
        "discount_rate": discount_rate,
        "effective_price": effective_price,
        "platform_fee": platform_fee,
        "publisher_fee": publisher_fee,
        "net_revenue_per_user": net_revenue_per_user,
        "break_even_units": break_even_units,
        "expected_users": expected_users,
        "probabilities": dict(probs),
    }

    return summary, scenario_table


def calculate_publisher_tradeoff(
    user_inputs: Mapping[str, Any],
    base_expected_users: float | None = None,
) -> dict[str, Any]:
    """Calculate how much unit uplift is needed to offset a publisher deal."""
    publisher_fee = _as_rate(user_inputs.get("publisher_fee"), 0.0)
    observed_publisher_uplift = _as_rate(
        user_inputs.get("observed_publisher_uplift"),
        0.1762,
    )
    scenario_publisher_uplift = _as_rate(
        user_inputs.get("publisher_user_uplift_scenario"),
        observed_publisher_uplift,
    )
    publisher_funded_amount = max(
        _to_float(user_inputs.get("publisher_funded_amount"), 0.0),
        0.0,
    )

    if publisher_fee >= 1:
        raw_required_publisher_uplift = math.inf
    else:
        raw_required_publisher_uplift = 1 / (1 - publisher_fee) - 1

    if base_expected_users is None:
        required_publisher_uplift = raw_required_publisher_uplift
        funding_reduction_to_required_uplift = 0.0
    else:
        partnership_scenario = apply_publisher_partnership_scenario(
            user_inputs=user_inputs,
            base_expected_users=base_expected_users,
        )
        required_publisher_uplift = partnership_scenario[
            "required_uplift_to_offset_fee"
        ]
        funding_reduction_to_required_uplift = partnership_scenario[
            "funding_reduction_to_required_uplift"
        ]

    if publisher_fee <= 0:
        note = "No publisher fee is entered, so no sales uplift is required to offset it."
    elif math.isinf(required_publisher_uplift):
        note = (
            "A publisher fee of 100% leaves no net publisher-side revenue share "
            "for the developer, so the required uplift is not finite."
        )
    elif scenario_publisher_uplift >= required_publisher_uplift:
        note = (
            "The selected publisher uplift is high enough to offset the entered "
            "publisher fee after accounting for publisher funding."
        )
    else:
        note = (
            "The selected publisher uplift is below the uplift required to offset "
            "the entered publisher fee after accounting for publisher funding."
        )

    return {
        "publisher_fee": publisher_fee,
        "publisher_funded_amount": publisher_funded_amount,
        "observed_publisher_uplift": observed_publisher_uplift,
        "scenario_publisher_uplift": scenario_publisher_uplift,
        "raw_required_publisher_uplift": raw_required_publisher_uplift,
        "required_publisher_uplift": required_publisher_uplift,
        "funding_reduction_to_required_uplift": funding_reduction_to_required_uplift,
        "interpretation_note": note,
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file and wrap parse failures with the file path."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON file: {path}") from exc


def _validate_feature_info(feature_info: Mapping[str, Any], path: Path) -> None:
    """Validate the minimal feature JSON schema used by this module."""
    if "features" not in feature_info or not isinstance(feature_info["features"], list):
        raise ValueError(f"Feature JSON must contain a list named 'features': {path}")


def _validate_expected_defaults(defaults_info: Mapping[str, Any]) -> None:
    """Validate the expected-users defaults JSON schema."""
    if "defaults" not in defaults_info or not isinstance(defaults_info["defaults"], dict):
        raise ValueError(
            "Expected users defaults JSON must contain a dictionary named 'defaults'."
        )


def _extract_feature_spec(
    feature_list: Sequence[str] | Mapping[str, Any],
) -> tuple[list[str], set[str], set[str], set[str], dict[str, Any]]:
    """Normalize a feature list or feature metadata dictionary."""
    if isinstance(feature_list, Mapping):
        feature_names = list(feature_list.get("features", []))
        categorical_features = set(feature_list.get("categorical_features", []))
        numeric_features = set(feature_list.get("numeric_features", []))
        binary_features = set(feature_list.get("binary_features", []))
        defaults = dict(feature_list.get("defaults", {}))
    else:
        feature_names = list(feature_list)
        categorical_features = {
            feature for feature in feature_names if _looks_categorical(feature)
        }
        binary_features = {
            feature for feature in feature_names if _looks_binary(feature)
        }
        numeric_features = (
            set(feature_names) - categorical_features - binary_features
        )
        defaults = {}

    if not feature_names:
        raise ValueError("Feature list is empty.")

    return feature_names, categorical_features, numeric_features, binary_features, defaults


def _default_value_from_name(feature: str) -> Any:
    """Fallback default when only a plain feature list is available."""
    if _looks_categorical(feature):
        return "Unknown"
    if _looks_binary(feature):
        return 0
    return 0


def _looks_categorical(feature: str) -> bool:
    """Infer common categorical feature names used by the training datasets."""
    return feature.endswith(("_bucket", "_type", "_primary", "_season"))


def _looks_binary(feature: str) -> bool:
    """Infer common binary feature names used by the training datasets."""
    return (
        feature.startswith("tag_")
        or "_has_" in feature
        or feature.endswith(("_mac", "_linux"))
        or feature == "prelaunch_holiday_release_window"
    )


def _predict_probability_by_class(model: Any, feature_row: pd.DataFrame) -> dict[str, float]:
    """Return predict_proba output mapped by the fitted model's class labels."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not support predict_proba().")

    classes = _model_classes(model)

    # Some local joblib/loky setups warn when physical CPU count cannot be
    # detected. That warning is not relevant to prediction results.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
        )
        probabilities = model.predict_proba(feature_row)[0]

    return {
        str(class_label): float(probability)
        for class_label, probability in zip(classes, probabilities)
    }


def _model_classes(model: Any) -> list[Any]:
    """Read classes_ from a pipeline or final estimator."""
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_model = model.named_steps["model"]
        if hasattr(final_model, "classes_"):
            return list(final_model.classes_)

    raise AttributeError("Loaded model does not expose fitted class labels.")


def _outcome_tier_users(bundle: Mapping[str, Any]) -> dict[str, float]:
    """Read outcome tier users from metadata, with safe defaults."""
    metadata = bundle.get("metadata", {})
    tier_users = metadata.get("outcome_tier_users", DEFAULT_OUTCOME_TIER_USERS)

    normalized: dict[str, float] = {}
    for label, default_users in DEFAULT_OUTCOME_TIER_USERS.items():
        normalized[label] = _to_float(tier_users.get(label), default_users)

    for label, users in tier_users.items():
        if label not in normalized:
            normalized[label] = _to_float(users, 0.0)

    return normalized


def _outcome_labels(
    tier_users: Mapping[str, float],
    probability_by_class: Mapping[str, float],
) -> tuple[str, ...]:
    """Return outcome labels in a stable business order."""
    labels: list[str] = [
        label
        for label in OUTCOME_LABEL_ORDER
        if label in tier_users or label in probability_by_class
    ]

    for label in probability_by_class:
        if label not in labels:
            labels.append(label)

    return tuple(labels)


def _probability_key(label: str) -> str:
    """Convert an outcome label into the public probability key name."""
    return f"P_{label.replace(' ', '_')}"


def _raw_probability_key(label: str) -> str:
    """Convert an outcome label into the public raw-probability key name."""
    return f"raw_{_probability_key(label)}"


def _expected_users_base_probs(
    bundle: Mapping[str, Any],
    labels: Sequence[str],
) -> dict[str, float]:
    """Read and normalize expected-users base probabilities."""
    base_probs = bundle.get("expected_users_base_probs", {})
    return _normalize_probabilities(
        {
            label: _to_float(
                base_probs.get(label, base_probs.get(_probability_key(label))),
                0.0,
            )
            for label in labels
        }
    )


def _smooth_probabilities(
    model_probs: Mapping[str, float],
    base_probs: Mapping[str, float],
    smoothing_weight: float,
    labels: Sequence[str],
) -> dict[str, float]:
    """Blend model probabilities with training base rates and normalize."""
    smoothed = {
        label: (
            smoothing_weight * _to_float(model_probs.get(label), 0.0)
            + (1 - smoothing_weight) * _to_float(base_probs.get(label), 0.0)
        )
        for label in labels
    }
    return _normalize_probabilities(smoothed)


def _blend_probabilities(
    primary_probabilities: Mapping[str, float],
    secondary_probabilities: Mapping[str, float],
    secondary_weight: float,
    labels: Sequence[str],
) -> dict[str, float]:
    """Blend two probability dictionaries and normalize the result."""
    primary_weight = 1 - secondary_weight
    blended = {
        label: (
            primary_weight * _to_float(primary_probabilities.get(label), 0.0)
            + secondary_weight * _to_float(secondary_probabilities.get(label), 0.0)
        )
        for label in labels
    }
    return _normalize_probabilities(blended)


def _probability_weighted_users(
    probabilities: Mapping[str, float],
    tier_users: Mapping[str, float],
    labels: Sequence[str],
) -> float:
    """Calculate expected users from tier probabilities."""
    return sum(
        _to_float(probabilities.get(label), 0.0)
        * _to_float(tier_users.get(label), 0.0)
        for label in labels
    )


def _normalize_probabilities(probs: Mapping[str, float]) -> dict[str, float]:
    """Normalize probabilities, falling back to equal weights when needed."""
    cleaned = {
        label: max(_to_float(probability, 0.0), 0.0)
        for label, probability in probs.items()
    }
    total = sum(cleaned.values())

    if total <= 0:
        equal_probability = 1 / len(cleaned)
        return {label: equal_probability for label in cleaned}

    return {
        label: probability / total
        for label, probability in cleaned.items()
    }


def _downside_risk_label(p_low: float) -> str:
    """Convert downside P(Low) into the requested qualitative label."""
    if p_low >= 0.60:
        return "High downside risk"
    if p_low >= 0.40:
        return "Moderate downside risk"
    if p_low >= 0.25:
        return "Manageable downside risk"
    return "Lower downside risk"


def _upside_label(p_high: float) -> str:
    """Convert upside P(High) into the requested qualitative label."""
    if p_high >= 0.60:
        return "Strong upside potential"
    if p_high >= 0.40:
        return "Moderate upside potential"
    return "More likely mid than high"


def _financial_scenario_row(
    scenario: str,
    users: float,
    total_project_cost: float,
    net_revenue_per_user: float,
) -> dict[str, Any]:
    """Calculate one financial scenario row."""
    users = max(_to_float(users, 0.0), 0.0)
    net_revenue = users * net_revenue_per_user
    margin_of_safety = net_revenue - total_project_cost
    margin_multiple = _margin_multiple(net_revenue, total_project_cost)

    return {
        "scenario": scenario,
        "users": users,
        "net_revenue": net_revenue,
        "margin_of_safety": margin_of_safety,
        "margin_multiple": margin_multiple,
        "risk_label": _financial_risk_label(margin_multiple),
    }


def _break_even_units(total_project_cost: float, net_revenue_per_user: float) -> float:
    """Safely calculate units needed to break even."""
    if total_project_cost <= 0:
        return 0.0
    if net_revenue_per_user <= 0:
        return math.inf
    return total_project_cost / net_revenue_per_user


def _margin_multiple(net_revenue: float, total_project_cost: float) -> float:
    """Safely calculate net revenue as a multiple of project cost."""
    if total_project_cost <= 0:
        return math.inf if net_revenue > 0 else 0.0
    return net_revenue / total_project_cost


def _financial_risk_label(margin_multiple: float) -> str:
    """Convert margin multiple into the requested viability label."""
    if math.isnan(margin_multiple) or margin_multiple < 1.0:
        return "Not viable"
    if margin_multiple < 1.3:
        return "Fragile"
    if margin_multiple < 1.5:
        return "Caution"
    if margin_multiple < 2.0:
        return "Viable"
    return "Strong"


def _to_float(value: Any, default: float = 0.0) -> float:
    """Convert common UI input values into floats without raising."""
    if value is None:
        return default

    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
        if cleaned == "":
            return default
        value = cleaned

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if np.isnan(number):
        return default

    return number


def _try_float(value: Any) -> float | None:
    """Convert to float, returning None when conversion is not possible."""
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "").replace("%", "")
        if cleaned == "":
            return None
        value = cleaned

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if np.isnan(number):
        return None

    return number


def _as_rate(value: Any, default: float = 0.0) -> float:
    """Normalize decimal or whole-percentage input into a 0.0 to 1.0 rate."""
    rate = _to_float(value, default)

    if rate > 1:
        rate = rate / 100

    return min(max(rate, 0.0), 1.0)
