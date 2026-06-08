"""Focused backend smoke test for the launch traction web tool.

Run this after training:

    python3 test_backend.py

The current Streamlit tool is goal-based. This test loads only the launch
traction model, runs the same traction goals with and without Publishing Studio
support, and prints the scenario adjustments.
"""

from __future__ import annotations

from pprint import pprint

from model_utils import (
    calculate_financials,
    calculate_publisher_tradeoff,
    load_models,
    predict_launch_traction_users,
)


def build_sample_inputs(
    publisher_type: str,
    review_sentiment: str = "Positive",
) -> dict[str, float | int | str]:
    """Return one realistic sample game and financial input dictionary."""
    publisher_fee = 0.30 if publisher_type == "Publishing Studio" else 0.00
    publisher_funded_amount = 100000 if publisher_type == "Publishing Studio" else 0
    publisher_uplift = 0.20 if publisher_type == "Publishing Studio" else 0.00

    return {
        # Core launch traction model inputs
        "prelaunch_publisher_type": publisher_type,
        "prelaunch_price_usd": 14.99,
        "prelaunch_achievements_count": 30,
        "review_sentiment": review_sentiment,
        "total_reviews": 363,
        "confidence_score": "Mid",
        # Financial inputs
        "development_budget": 150000,
        "marketing_budget": 25000,
        "post_launch_support_budget": 25000,
        "contingency_pct": 0.20,
        "list_price": 19.99,
        "discount_rate": 0.25,
        "platform_fee": 0.30,
        "publisher_fee": publisher_fee,
        "publisher_user_uplift_scenario": publisher_uplift,
        "publisher_funded_amount": publisher_funded_amount,
    }


def print_section(title: str) -> None:
    """Print a readable section header."""
    print(f"\n{title}")
    print("-" * len(title))


def print_launch_traction_result(result: dict) -> None:
    """Print the key launch traction output fields."""
    review_adjustment = result["review_reception_adjustment"]
    publisher_adjustment = result["publisher_support_adjustment"]
    partnership_scenario = result["publisher_partnership_scenario"]

    print(f"Core expected users: {result['core_launch_traction_expected_users']:.0f}")
    print(
        "Scenario-adjusted expected users: "
        f"{result['launch_traction_expected_users']:.0f}"
    )
    print(
        "Publisher scenario users: "
        f"{partnership_scenario['publisher_partnership_expected_users']:.0f}"
    )
    print(f"Financial expected users: {result['financial_expected_users']:.0f}")
    print(f"Most likely model tier: {result['launch_traction_predicted_class']}")
    print(f"Review adjustment applied: {review_adjustment['applied']}")
    print(
        "Review expected-user delta: "
        f"{review_adjustment['expected_user_delta']:.0f}"
    )
    print(f"Review adjustment note: {review_adjustment['note']}")
    print(f"Publisher adjustment applied: {publisher_adjustment['applied']}")
    print(
        "Publisher expected-user delta: "
        f"{publisher_adjustment['expected_user_delta']:.0f}"
    )
    print(f"Publisher adjustment note: {publisher_adjustment['note']}")
    print(
        "Selected publisher uplift: "
        f"{partnership_scenario['selected_uplift']:.1%}"
    )
    print(
        "Required uplift to offset fee after funding: "
        f"{partnership_scenario['required_uplift_to_offset_fee']:.1%}"
    )
    print(
        "Raw no-funding fee hurdle: "
        f"{partnership_scenario['raw_required_uplift_to_offset_fee']:.1%}"
    )
    print(
        "Funding reduction to required uplift: "
        f"{partnership_scenario['funding_reduction_to_required_uplift']:.1%}"
    )
    print(
        "Selected uplift offsets fee: "
        f"{partnership_scenario['break_even_uplift_met']}"
    )

    print("\nCore model probabilities:")
    pprint(result["traction_core_probabilities"], sort_dicts=True)

    print("\nModel probabilities after review scenario:")
    pprint(result["traction_review_adjusted_probabilities"], sort_dicts=True)

    print("\nModel probabilities after publisher support scenario:")
    pprint(result["traction_probabilities"], sort_dicts=True)

    print("\nReview reception scenario details:")
    pprint(review_adjustment, sort_dicts=True)

    print("\nPublisher support scenario details:")
    pprint(publisher_adjustment, sort_dicts=True)

    print("\nPublisher partnership scenario details:")
    pprint(partnership_scenario, sort_dicts=True)


def main() -> None:
    """Run the focused launch traction smoke test."""
    print_section("Loading Launch Traction Model")
    bundle = load_models(include_expected_model=False, include_support_models=False)
    print("Loaded launch traction model, review priors, and publisher priors.")

    print_section("Review Reception Comparison")
    for review_sentiment in ("Mixed", "Positive"):
        user_inputs = build_sample_inputs("Independent", review_sentiment)
        result = predict_launch_traction_users(bundle, user_inputs)
        print(
            f"{review_sentiment}: "
            f"{result['core_launch_traction_expected_users']:.0f} core users, "
            f"{result['review_adjusted_expected_users']:.0f} after review scenario"
        )

    results: dict[str, dict] = {}
    for publisher_type in ("Independent", "Publishing Studio"):
        print_section(f"Scenario: {publisher_type}")
        user_inputs = build_sample_inputs(publisher_type)
        result = predict_launch_traction_users(bundle, user_inputs)
        results[publisher_type] = result
        print_launch_traction_result(result)

    print_section("Publisher Support Comparison")
    independent_users = results["Independent"]["launch_traction_expected_users"]
    publisher_users = results["Publishing Studio"]["launch_traction_expected_users"]
    publisher_scenario_users = results["Publishing Studio"]["financial_expected_users"]
    print(f"Independent expected users: {independent_users:.0f}")
    print(f"Publishing Studio adjusted expected users: {publisher_users:.0f}")
    print(f"Publishing Studio scenario users: {publisher_scenario_users:.0f}")
    print(f"Model difference: {publisher_users - independent_users:.0f}")
    print(f"Scenario difference: {publisher_scenario_users - independent_users:.0f}")

    print_section("Financial Viability: Publishing Studio Scenario")
    publishing_inputs = build_sample_inputs("Publishing Studio")
    publishing_result = results["Publishing Studio"]
    financial_summary, scenario_table = calculate_financials(
        user_inputs=publishing_inputs,
        expected_users=publishing_result["financial_expected_users"],
        probs=publishing_result,
    )
    print("Summary:")
    pprint(financial_summary, sort_dicts=True)
    print("\nScenario table:")
    print(scenario_table.to_string(index=False))

    print_section("Publisher Tradeoff")
    publisher_tradeoff = calculate_publisher_tradeoff(
        publishing_inputs,
        base_expected_users=publishing_result["launch_traction_expected_users"],
    )
    pprint(publisher_tradeoff, sort_dicts=True)


if __name__ == "__main__":
    main()
