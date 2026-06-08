"""Streamlit app for the senior_project_webtool."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from model_utils import (
    LAUNCH_TRACTION_FEATURE_CAPS,
    calculate_financials,
    load_models,
    predict_launch_traction_users,
)


@st.cache_resource(show_spinner="Loading trained model pipelines...")
def get_model_bundle() -> dict:
    """Load model artifacts once per Streamlit session."""
    return load_models(include_expected_model=False, include_support_models=False)


def format_users(value: float) -> str:
    """Format a user count for display."""
    return f"{value:,.0f}"


def format_percent(value: float) -> str:
    """Format a probability or rate for display."""
    return f"{value * 100:.1f}%"


def format_percent_or_not_finite(value: float) -> str:
    """Format a rate, handling infinite requirements cleanly."""
    if value == float("inf"):
        return "Not finite"
    return format_percent(value)


def build_user_inputs() -> dict:
    """Collect app inputs and return model/financial values."""
    st.sidebar.header("Game Positioning")

    prelaunch_publisher_type = st.sidebar.selectbox(
        "Publisher type",
        ["Independent", "Publishing Studio", "Unknown"],
        index=1,
        help=(
            "Used for a 17.62% Publishing Studio scenario uplift. The core "
            "launch traction classifier does not use publisher type directly."
        ),
    )
    publisher_fee_pct = st.sidebar.slider(
        "Publisher fee",
        min_value=0,
        max_value=100,
        value=30 if prelaunch_publisher_type == "Publishing Studio" else 0,
        step=5,
        format="%d%%",
    )
    publisher_fee = publisher_fee_pct / 100
    required_publisher_uplift = (
        float("inf") if publisher_fee >= 1 else 1 / (1 - publisher_fee) - 1
    )
    default_publisher_uplift = (
        min(max(required_publisher_uplift, 0.0), 1.0)
        if prelaunch_publisher_type == "Publishing Studio"
        else 0.0
    )
    publisher_user_uplift_scenario_pct = st.sidebar.slider(
        "Publisher user uplift scenario",
        min_value=0,
        max_value=100,
        value=int(round(default_publisher_uplift * 100)),
        step=1,
        format="%d%%",
        disabled=prelaunch_publisher_type != "Publishing Studio",
        help=(
            "Scenario assumption for extra users from publisher marketing, "
            "visibility, distribution, and launch support. The output compares "
            "this against the uplift needed after publisher funding is counted."
        ),
    )
    publisher_user_uplift_scenario = publisher_user_uplift_scenario_pct / 100
    if prelaunch_publisher_type != "Publishing Studio":
        publisher_user_uplift_scenario = 0.0

    prelaunch_price_usd = st.sidebar.number_input(
        "Pre-launch price",
        min_value=LAUNCH_TRACTION_FEATURE_CAPS["prelaunch_price_usd"][0],
        max_value=LAUNCH_TRACTION_FEATURE_CAPS["prelaunch_price_usd"][1],
        value=14.99,
        step=1.0,
    )
    prelaunch_achievements_count = st.sidebar.number_input(
        "Achievement count",
        min_value=int(LAUNCH_TRACTION_FEATURE_CAPS["prelaunch_achievements_count"][0]),
        max_value=int(LAUNCH_TRACTION_FEATURE_CAPS["prelaunch_achievements_count"][1]),
        value=30,
        step=1,
    )

    st.sidebar.header("Launch Traction Goals")
    review_sentiment = st.sidebar.selectbox(
        "Target review sentiment",
        ["Mixed", "Positive"],
        index=1,
    )
    total_reviews = st.sidebar.number_input(
        "Target total review count",
        min_value=int(LAUNCH_TRACTION_FEATURE_CAPS["total_reviews"][0]),
        max_value=int(LAUNCH_TRACTION_FEATURE_CAPS["total_reviews"][1]),
        value=363,
        step=25,
    )
    confidence_score = st.sidebar.selectbox(
        "Ownership estimate confidence",
        ["Low", "Mid", "High"],
        index=1,
        help=(
            "Confidence in the SteamSpy ownership estimate used by the "
            "launch-traction goal model. This is a post-launch diagnostic "
            "for how well the ownership estimate is supported by review "
            "and ownership data. In the 4-class launch-traction training set, "
            "the confidence groups contain 28 Low-confidence estimates, "
            "82 Mid-confidence estimates, and 58 High-confidence estimates."
        ),
    )

    st.sidebar.header("Financial Inputs")
    development_budget = st.sidebar.number_input(
        "Development budget",
        min_value=0,
        value=150000,
        step=10000,
        format="%d",
    )
    marketing_budget = st.sidebar.number_input(
        "Marketing budget",
        min_value=0,
        value=25000,
        step=5000,
        format="%d",
    )
    post_launch_support_budget = st.sidebar.number_input(
        "Post-launch support budget",
        min_value=0,
        value=25000,
        step=5000,
        format="%d",
    )
    contingency_pct_input = st.sidebar.slider(
        "Contingency rate",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        format="%d%%",
    )
    contingency_pct = contingency_pct_input / 100
    platform_fee_pct = st.sidebar.slider(
        "Platform fee",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        format="%d%%",
    )
    platform_fee = platform_fee_pct / 100
    publisher_funded_amount = st.sidebar.number_input(
        "Publisher funded amount",
        min_value=0,
        value=0,
        step=10000,
        format="%d",
    )

    user_inputs = {
        "prelaunch_publisher_type": prelaunch_publisher_type,
        "prelaunch_price_usd": prelaunch_price_usd,
        "prelaunch_achievements_count": prelaunch_achievements_count,
        "development_budget": development_budget,
        "marketing_budget": marketing_budget,
        "post_launch_support_budget": post_launch_support_budget,
        "contingency_pct": contingency_pct,
        "list_price": prelaunch_price_usd,
        "discount_rate": 0.0,
        "platform_fee": platform_fee,
        "publisher_fee": publisher_fee,
        "publisher_user_uplift_scenario": publisher_user_uplift_scenario,
        "publisher_funded_amount": publisher_funded_amount,
        "review_sentiment": review_sentiment,
        "total_reviews": total_reviews,
        "confidence_score": confidence_score,
    }

    return user_inputs


def render_launch_traction(launch_traction_result: dict) -> None:
    """Render the launch traction goal-setting model output."""
    st.subheader("Goal-Based User Prediction")
    st.write(
        "Set launch traction goals to estimate which historical user tier the "
        "game would most closely resemble if it reaches those review targets."
    )

    columns = st.columns(2)
    columns[0].metric(
        "Scenario-adjusted expected users",
        format_users(launch_traction_result["launch_traction_expected_users"]),
    )
    columns[1].metric(
        "Most likely model tier",
        launch_traction_result["launch_traction_predicted_class"],
    )

    partnership_scenario = launch_traction_result["publisher_partnership_scenario"]
    partnership_columns = st.columns(4)
    partnership_columns[0].metric(
        "Publisher scenario users",
        format_users(partnership_scenario["publisher_partnership_expected_users"]),
        delta=format_users(partnership_scenario["expected_user_delta"])
        if partnership_scenario["expected_user_delta"] > 0
        else None,
    )
    partnership_columns[1].metric(
        "Selected publisher uplift",
        format_percent(partnership_scenario["selected_uplift"]),
    )
    partnership_columns[2].metric(
        "Fee-offset uplift after funding",
        format_percent_or_not_finite(
            partnership_scenario["required_uplift_to_offset_fee"]
        ),
    )
    partnership_columns[3].metric(
        "No-funding fee hurdle",
        format_percent_or_not_finite(
            partnership_scenario["raw_required_uplift_to_offset_fee"]
        ),
        delta=(
            f"-{format_percent(partnership_scenario['funding_reduction_to_required_uplift'])}"
            if partnership_scenario["funding_reduction_to_required_uplift"] > 0
            else None
        ),
    )

    probability_columns = st.columns(4)
    probability_columns[0].metric(
        "Model P(Not Viable)",
        format_percent(launch_traction_result["traction_P_Not_Viable"]),
    )
    probability_columns[1].metric(
        "Model P(Low)",
        format_percent(launch_traction_result["traction_P_Low"]),
    )
    probability_columns[2].metric(
        "Model P(Mid)",
        format_percent(launch_traction_result["traction_P_Mid"]),
    )
    probability_columns[3].metric(
        "Model P(High)",
        format_percent(launch_traction_result["traction_P_High"]),
    )

    if launch_traction_result["launch_traction_input_caps_applied"]:
        st.warning("Some launch traction inputs were capped to the training range.")

    if launch_traction_result["launch_traction_overconfidence_warning"]:
        st.warning(launch_traction_result["launch_traction_overconfidence_note"])

    review_adjustment = launch_traction_result["review_reception_adjustment"]
    if review_adjustment["applied"]:
        st.info(
            "Positive reception scenario added "
            f"{format_users(review_adjustment['expected_user_delta'])} "
            "expected users to this scenario."
        )
    elif review_adjustment["review_value"] == "Positive":
        st.info(review_adjustment["note"])

    publisher_adjustment = launch_traction_result["publisher_support_adjustment"]
    if publisher_adjustment["applied"]:
        st.info(
            "Publishing Studio support scenario added "
            f"{format_users(publisher_adjustment['expected_user_delta'])} "
            "expected users before the separate publisher uplift slider."
        )
    elif publisher_adjustment["publisher_value"] == "Publishing Studio":
        st.info(publisher_adjustment["note"])

    if (
        partnership_scenario["publisher_type"] == "Publishing Studio"
        and not partnership_scenario["break_even_uplift_met"]
    ):
        st.warning(
            "Selected publisher uplift does not offset the entered publisher fee "
            "after accounting for publisher funding."
        )


def render_financials(financial_summary: dict, scenario_table: pd.DataFrame) -> None:
    """Render financial viability calculations."""
    st.subheader("Financial Viability")
    st.write(
        "The Expected scenario uses the launch-traction goal estimate, so the "
        "financial view is tied to the review-count target set in the sidebar."
    )

    columns = st.columns(4)
    columns[0].metric(
        "Total project cost",
        f"${financial_summary['total_project_cost']:,.0f}",
    )
    columns[1].metric(
        "Net revenue per user",
        f"${financial_summary['net_revenue_per_user']:,.2f}",
    )
    columns[2].metric(
        "Break-even units",
        format_users(financial_summary["break_even_units"]),
    )
    columns[3].metric(
        "Expected users used",
        format_users(financial_summary["expected_users"]),
    )

    display_table = scenario_table.copy()
    display_table["users"] = display_table["users"].map(lambda value: f"{value:,.0f}")
    display_table["net_revenue"] = display_table["net_revenue"].map(
        lambda value: f"${value:,.0f}"
    )
    display_table["margin_of_safety"] = display_table["margin_of_safety"].map(
        lambda value: f"${value:,.0f}"
    )
    display_table["margin_multiple"] = display_table["margin_multiple"].map(
        lambda value: f"{value:.2f}x"
    )
    st.dataframe(display_table, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(
        page_title="Senior Project Web Tool",
        layout="wide",
    )

    st.title("Indie Turn-Based Strategy Launch Goal Tool")

    try:
        bundle = get_model_bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    user_inputs = build_user_inputs()

    launch_traction_result = predict_launch_traction_users(bundle, user_inputs)

    combined_probs = dict(launch_traction_result)
    expected_users_for_financials = launch_traction_result["financial_expected_users"]

    financial_summary, scenario_table = calculate_financials(
        user_inputs=user_inputs,
        expected_users=expected_users_for_financials,
        probs=combined_probs,
    )

    render_launch_traction(launch_traction_result)
    render_financials(financial_summary, scenario_table)


if __name__ == "__main__":
    main()
