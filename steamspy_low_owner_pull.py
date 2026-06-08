"""
Focused SteamSpy / Steam Store data pull for senior_project_webtool.

This script collects only the columns needed by the current prediction models:

1. Pre-launch expected users model inputs.
2. Optional launch traction scenario inputs.
3. Owner estimate fields used only to create training labels.

It intentionally avoids wide exploratory fields, sparse tag flags, revenue fields,
post-launch playtime fields, and sales_confidence.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


STEAMSPY_API_URL = "https://steamspy.com/api.php"
STEAM_STORE_APPDETAILS_URL = "https://store.steampowered.com/api/appdetails"

DEFAULT_TAGS = ["Turn-Based Strategy"]

PRELAUNCH_COLUMNS = [
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

LAUNCH_TRACTION_COLUMNS = [
    "review_sentiment",
    "positive_review_pct",
    "total_reviews",
    "log_total_reviews",
    "confidence_score",
]

LABEL_COLUMNS = [
    "owner_range",
    "owner_low",
    "owner_high",
    "owner_midpoint",
    "outcome_level",
]

FINAL_COLUMNS = [
    "appid",
    "game_name",
    *PRELAUNCH_COLUMNS,
    *LAUNCH_TRACTION_COLUMNS,
    *LABEL_COLUMNS,
]


class ApiFetchError(RuntimeError):
    """Raised when an API endpoint does not return usable JSON after retries."""


def slugify(value: str) -> str:
    """Make a safe filename fragment for cached API responses."""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()


def load_or_fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    cache_path: Path,
    sleep_seconds: float,
    refresh_cache: bool,
    max_retries: int = 3,
) -> Any:
    """
    Load a cached JSON response when available; otherwise fetch and cache it.

    The cache is important because the script may inspect hundreds of apps.
    Re-running from cache avoids unnecessary API traffic and makes debugging
    much faster.
    """
    if cache_path.exists() and not refresh_cache:
        try:
            with cache_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            print(f"Cached response was not valid JSON, refreshing: {cache_path}")

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            response.raise_for_status()

            try:
                data = response.json()
            except ValueError as exc:
                preview = response.text[:200].replace("\n", " ").strip()
                if not preview:
                    preview = "<empty response>"
                raise ApiFetchError(
                    f"Non-JSON response from {url} with params {params}: {preview!r}"
                ) from exc

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, sort_keys=True)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return data

        except (requests.RequestException, ApiFetchError) as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = max(sleep_seconds, 0.25) * attempt
                print(
                    f"API fetch failed on attempt {attempt}/{max_retries}; "
                    f"retrying in {wait_seconds:.2f}s. Error: {exc}"
                )
                time.sleep(wait_seconds)

    raise ApiFetchError(f"API fetch failed after {max_retries} attempts: {last_error}")


def fetch_steamspy_tag(
    session: requests.Session,
    tag: str,
    cache_dir: Path,
    sleep_seconds: float,
    refresh_cache: bool,
) -> dict[str, Any]:
    """Fetch SteamSpy's app list for one tag."""
    cache_path = cache_dir / "steamspy_tags" / f"{slugify(tag)}.json"
    data = load_or_fetch_json(
        session=session,
        url=STEAMSPY_API_URL,
        params={"request": "tag", "tag": tag},
        cache_path=cache_path,
        sleep_seconds=sleep_seconds,
        refresh_cache=refresh_cache,
    )
    return data if isinstance(data, dict) else {}


def fetch_steamspy_appdetails(
    session: requests.Session,
    appid: int,
    cache_dir: Path,
    sleep_seconds: float,
    refresh_cache: bool,
) -> dict[str, Any]:
    """Fetch detailed SteamSpy fields for one app."""
    cache_path = cache_dir / "steamspy_appdetails" / f"{appid}.json"
    data = load_or_fetch_json(
        session=session,
        url=STEAMSPY_API_URL,
        params={"request": "appdetails", "appid": appid},
        cache_path=cache_path,
        sleep_seconds=sleep_seconds,
        refresh_cache=refresh_cache,
    )
    return data if isinstance(data, dict) else {}


def fetch_store_appdetails(
    session: requests.Session,
    appid: int,
    cache_dir: Path,
    sleep_seconds: float,
    refresh_cache: bool,
) -> dict[str, Any]:
    """Fetch Steam Store fields that SteamSpy does not reliably expose."""
    cache_path = cache_dir / "steam_store_appdetails" / f"{appid}.json"
    data = load_or_fetch_json(
        session=session,
        url=STEAM_STORE_APPDETAILS_URL,
        params={"appids": appid, "cc": "us", "l": "en"},
        cache_path=cache_path,
        sleep_seconds=sleep_seconds,
        refresh_cache=refresh_cache,
    )

    app_payload = data.get(str(appid), {}) if isinstance(data, dict) else {}
    if not app_payload.get("success"):
        return {}

    store_data = app_payload.get("data", {})
    return store_data if isinstance(store_data, dict) else {}


def parse_int(value: Any) -> int | None:
    """Parse Steam API numbers that may arrive as strings with commas."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    cleaned = re.sub(r"[^0-9-]+", "", str(value))
    if cleaned in {"", "-"}:
        return None
    return int(cleaned)


def parse_owner_range(owner_range: Any) -> tuple[int | None, int | None, int | None]:
    """Parse SteamSpy owner ranges like '20,000 .. 50,000'."""
    if not owner_range:
        return None, None, None

    matches = re.findall(r"\d[\d,]*", str(owner_range))
    if len(matches) < 2:
        return None, None, None

    owner_low = parse_int(matches[0])
    owner_high = parse_int(matches[1])
    if owner_low is None or owner_high is None:
        return None, None, None

    owner_midpoint = int((owner_low + owner_high) / 2)
    return owner_low, owner_high, owner_midpoint


def first_not_none(*values: Any) -> Any:
    """Return the first value that is not None, preserving valid zero values."""
    for value in values:
        if value is not None:
            return value
    return None


def should_keep_owner_range(
    owner_low: int | None,
    owner_high: int | None,
    owner_midpoint: int | None,
    owner_midpoint_max: int,
    owner_high_max: int,
    include_straddlers: bool,
) -> bool:
    """
    Keep games below or near the current viable threshold.

    The straddler rule keeps ranges like 20,000 .. 50,000 because they sit right
    on the 35,000 viability boundary even when the exact value is uncertain.
    """
    if owner_low is None or owner_high is None or owner_midpoint is None:
        return False

    if owner_midpoint <= owner_midpoint_max:
        return True

    return include_straddlers and owner_low < owner_midpoint_max and owner_high <= owner_high_max


def outcome_level_from_owner_midpoint(owner_midpoint: int | None) -> str | None:
    """Create the model training label from the owner estimate midpoint."""
    if owner_midpoint is None:
        return None
    if owner_midpoint < 35_000:
        return "Low"
    if owner_midpoint < 100_000:
        return "Mid"
    return "High"


def parse_price_usd(steamspy_detail: dict[str, Any], store_data: dict[str, Any]) -> float | None:
    """
    Prefer Steam Store list price because SteamSpy price fields can be stale.

    Steam Store prices are in cents. SteamSpy prices are usually also cents, but
    this parser handles either cents-like or dollars-like values.
    """
    if store_data.get("is_free") is True:
        return 0.0

    price_overview = store_data.get("price_overview")
    if isinstance(price_overview, dict):
        price_cents = first_not_none(parse_int(price_overview.get("initial")), parse_int(price_overview.get("final")))
        if price_cents is not None:
            return round(price_cents / 100, 2)

    steamspy_price = first_not_none(parse_int(steamspy_detail.get("initialprice")), parse_int(steamspy_detail.get("price")))
    if steamspy_price is None:
        return None
    if steamspy_price > 100:
        return round(steamspy_price / 100, 2)
    return float(steamspy_price)


def price_bucket(price_usd: float | None) -> str | None:
    """Match the Low / Mid / High price bucket style used by the model."""
    if price_usd is None:
        return None
    if price_usd < 10:
        return "Low"
    if price_usd < 20:
        return "Mid"
    return "High"


def count_supported_languages(store_data: dict[str, Any]) -> int | None:
    """Count supported languages from Steam Store's HTML-ish language field."""
    raw_languages = store_data.get("supported_languages")
    if not raw_languages:
        return None

    text = html.unescape(str(raw_languages))
    text = re.sub(r"<br\s*/?>", ",", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*+\s*languages with full audio support.*", "", text, flags=re.IGNORECASE)

    languages = {
        item.strip()
        for item in re.split(r"[,;\n]+", text)
        if item.strip() and "audio support" not in item.lower()
    }
    return len(languages) if languages else None


def list_count_or_none(value: Any) -> int | None:
    """Count list-like Store API fields while preserving missing data as None."""
    if value is None:
        return None
    if isinstance(value, list):
        return len(value)
    return None


def list_count_or_zero(value: Any) -> int:
    """Count list-like fields where missing means the app has none."""
    if isinstance(value, list):
        return len(value)
    return 0


def achievement_count(store_data: dict[str, Any]) -> int | None:
    """Extract Steam achievement count; missing achievement data means zero."""
    if not store_data:
        return None

    achievements = store_data.get("achievements")
    if not isinstance(achievements, dict):
        return 0

    total = parse_int(achievements.get("total"))
    return total if total is not None else 0


def normalize_people_field(value: Any) -> list[str]:
    """Normalize developer/publisher fields from either API."""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;/]+", str(value))

    return [item.strip() for item in items if item and str(item).strip()]


def publisher_type(steamspy_detail: dict[str, Any], store_data: dict[str, Any]) -> str:
    """
    Derive the model's broad publisher category.

    This is intentionally simple:
    - Independent: developer and publisher appear to be the same entity.
    - Publishing Studio: there is a distinct listed publisher.
    - Unknown: neither API gives usable publisher/developer names.
    """
    developers = normalize_people_field(store_data.get("developers")) or normalize_people_field(
        steamspy_detail.get("developer")
    )
    publishers = normalize_people_field(store_data.get("publishers")) or normalize_people_field(
        steamspy_detail.get("publisher")
    )

    if not developers and not publishers:
        return "Unknown"
    if not publishers:
        return "Independent"

    developer_names = {name.lower() for name in developers}
    publisher_names = {name.lower() for name in publishers}

    if developer_names and developer_names.intersection(publisher_names):
        return "Independent"

    publisher_text = " ".join(publisher_names)
    if "self" in publisher_text or "independent" in publisher_text:
        return "Independent"

    return "Publishing Studio"


def extract_tags(steamspy_detail: dict[str, Any], fallback_tags: set[str]) -> set[str]:
    """Extract Steam tags without creating sparse one-hot columns."""
    raw_tags = steamspy_detail.get("tags")

    if isinstance(raw_tags, dict):
        tags = {str(tag).strip() for tag in raw_tags.keys() if str(tag).strip()}
        if tags:
            return tags

    if isinstance(raw_tags, list):
        tags = {str(tag).strip() for tag in raw_tags if str(tag).strip()}
        if tags:
            return tags

    return set(fallback_tags)


def primary_subgenre(tags: set[str], store_data: dict[str, Any]) -> str:
    """Map many possible Steam tags into the model's stable subgenre buckets."""
    tag_lookup = {tag.lower() for tag in tags}

    genres = store_data.get("genres") or []
    if isinstance(genres, list):
        tag_lookup.update(str(genre.get("description", "")).lower() for genre in genres if isinstance(genre, dict))

    categories = store_data.get("categories") or []
    if isinstance(categories, list):
        tag_lookup.update(
            str(category.get("description", "")).lower() for category in categories if isinstance(category, dict)
        )

    if "4x" in tag_lookup:
        return "4X"

    deckbuilder_terms = {
        "deckbuilding",
        "deckbuilder",
        "card battler",
        "card game",
        "roguelike deckbuilder",
    }
    if tag_lookup.intersection(deckbuilder_terms):
        return "Deckbuilder"

    if "grand strategy" in tag_lookup:
        return "Grand Strategy"

    management_terms = {
        "management",
        "simulation",
        "colony sim",
        "city builder",
        "base-building",
        "automation",
        "economy",
    }
    if tag_lookup.intersection(management_terms):
        return "Management/Sim"

    turn_based_terms = {
        "turn-based strategy",
        "turn-based tactics",
        "turn-based combat",
        "strategy rpg",
        "tactical rpg",
        "tactical",
    }
    if tag_lookup.intersection(turn_based_terms):
        return "Turn-Based Strategy"

    return "Other Strategy"


def review_metrics(steamspy_detail: dict[str, Any]) -> tuple[int | None, float | None, str | None]:
    """Create total review count, positive review percent, and sentiment bucket."""
    positive = parse_int(steamspy_detail.get("positive")) or 0
    negative = parse_int(steamspy_detail.get("negative")) or 0
    total_reviews = positive + negative

    if total_reviews <= 0:
        return 0, None, None

    positive_review_pct = positive / total_reviews
    if positive_review_pct >= 0.75:
        sentiment = "Positive"
    elif positive_review_pct >= 0.50:
        sentiment = "Mixed"
    else:
        sentiment = "Negative"

    return total_reviews, positive_review_pct, sentiment


def confidence_score(
    owner_low: int | None,
    owner_high: int | None,
    owner_midpoint: int | None,
    total_reviews: int | None,
) -> str | None:
    """
    Estimate ownership confidence from review support and owner range width.

    This is a diagnostic feature for the launch traction scenario model, not a
    user-facing sales confidence input. Thin review counts and very wide owner
    ranges are treated as lower confidence.
    """
    if owner_low is None or owner_high is None or owner_midpoint is None or total_reviews is None:
        return None

    range_width = owner_high - owner_low
    width_ratio = range_width / max(owner_midpoint, 1)

    if total_reviews >= 500 and width_ratio <= 1.5:
        return "High"
    if total_reviews >= 100 and width_ratio <= 2.5:
        return "Mid"
    return "Low"


def build_candidate_apps(
    session: requests.Session,
    tags: list[str],
    cache_dir: Path,
    sleep_seconds: float,
    refresh_cache: bool,
) -> dict[int, dict[str, Any]]:
    """Collect unique app candidates from the chosen SteamSpy tags."""
    candidates: dict[int, dict[str, Any]] = {}

    for tag in tags:
        print(f"Fetching SteamSpy tag: {tag}")
        tag_payload = fetch_steamspy_tag(
            session=session,
            tag=tag,
            cache_dir=cache_dir,
            sleep_seconds=sleep_seconds,
            refresh_cache=refresh_cache,
        )

        for raw_appid, row in tag_payload.items():
            appid = parse_int(raw_appid)
            if appid is None or not isinstance(row, dict):
                continue

            candidate = candidates.setdefault(appid, {"tag_hits": set(), "tag_rows": []})
            candidate["tag_hits"].add(tag)
            candidate["tag_rows"].append(row)

    return candidates


def get_first_tag_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Use tag-list fields as a lightweight fallback before appdetails are fetched."""
    tag_rows = candidate.get("tag_rows", [])
    if tag_rows and isinstance(tag_rows[0], dict):
        return tag_rows[0]
    return {}


def candidate_sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
    """Sort candidates by owner midpoint first so low-owner games are pulled first."""
    appid, candidate = item
    row = get_first_tag_row(candidate)
    owner_low, owner_high, owner_midpoint = parse_owner_range(row.get("owners"))
    if owner_midpoint is None:
        return (10**12, appid)
    return (owner_midpoint, appid)


def build_row(
    appid: int,
    candidate: dict[str, Any],
    steamspy_detail: dict[str, Any],
    store_data: dict[str, Any],
) -> dict[str, Any]:
    """Build one narrow model-ready row."""
    fallback_row = get_first_tag_row(candidate)
    merged_steamspy = {**fallback_row, **steamspy_detail}

    owner_range = merged_steamspy.get("owners")
    owner_low, owner_high, owner_midpoint = parse_owner_range(owner_range)
    total_reviews, positive_review_pct, sentiment = review_metrics(merged_steamspy)
    price_usd = parse_price_usd(merged_steamspy, store_data)
    tags = extract_tags(merged_steamspy, set(candidate.get("tag_hits", set())))

    return {
        "appid": appid,
        "game_name": store_data.get("name") or merged_steamspy.get("name"),
        "prelaunch_publisher_type": publisher_type(merged_steamspy, store_data),
        "prelaunch_price_bucket": price_bucket(price_usd),
        # This pull is intentionally scoped to the SteamSpy Turn-Based Strategy tag.
        # Keep the subgenre field constant so the added dataset does not mix in
        # adjacent buckets like 4X, Deckbuilder, or Grand Strategy.
        "prelaunch_subgenre_primary": "Turn-Based Strategy",
        "prelaunch_price_usd": price_usd,
        "prelaunch_language_count": count_supported_languages(store_data),
        "prelaunch_screenshots_count": list_count_or_none(store_data.get("screenshots")),
        "prelaunch_movies_count": list_count_or_zero(store_data.get("movies")),
        "prelaunch_achievements_count": achievement_count(store_data),
        "prelaunch_tag_count": len(tags) if tags else None,
        "review_sentiment": sentiment,
        "positive_review_pct": positive_review_pct,
        "total_reviews": total_reviews,
        "log_total_reviews": math.log1p(total_reviews) if total_reviews is not None else None,
        "confidence_score": confidence_score(owner_low, owner_high, owner_midpoint, total_reviews),
        "owner_range": owner_range,
        "owner_low": owner_low,
        "owner_high": owner_high,
        "owner_midpoint": owner_midpoint,
        "outcome_level": outcome_level_from_owner_midpoint(owner_midpoint),
    }


def store_app_type(store_data: dict[str, Any]) -> str | None:
    """Return the Steam Store product type when available."""
    app_type = store_data.get("type")
    return str(app_type).lower().strip() if app_type else None


def store_descriptions(store_data: dict[str, Any], key: str) -> list[str]:
    """Extract Store API genre/category descriptions for filtering."""
    values = store_data.get(key) or []
    if not isinstance(values, list):
        return []
    return [
        str(value.get("description", "")).lower()
        for value in values
        if isinstance(value, dict) and value.get("description")
    ]


def excluded_product_reason(row: dict[str, Any], store_data: dict[str, Any], require_store_game: bool) -> str | None:
    """
    Reject products that are not normal paid games.

    Steam Store type catches most DLC/demos/software. The text checks catch
    playtests, soundtracks, and edge cases that sometimes still look game-like
    in API responses.
    """
    app_type = store_app_type(store_data)
    if require_store_game and app_type != "game":
        return "not_store_game"

    game_name = str(row.get("game_name") or "").lower()
    categories = store_descriptions(store_data, "categories")
    genres = store_descriptions(store_data, "genres")
    text_blob = " ".join([game_name, *categories, *genres])

    text_reason = excluded_text_reason(text_blob)
    if text_reason:
        return text_reason

    return None


def excluded_text_reason(text_blob: str) -> str | None:
    """Catch obvious non-game product terms before Store API details are fetched."""
    excluded_terms = {
        "demo",
        "playtest",
        "soundtrack",
        "dlc",
        "downloadable content",
        "software",
        "tool",
        "editor",
    }
    for term in excluded_terms:
        if re.search(rf"\b{re.escape(term)}\b", text_blob.lower()):
            return f"excluded_product_{term.replace(' ', '_')}"
    return None


def value_in_range(value: Any, minimum: float | int | None, maximum: float | int | None) -> bool:
    """Validate numeric filter ranges while treating missing values as invalid."""
    if value is None or pd.isna(value):
        return False
    numeric_value = float(value)
    if minimum is not None and numeric_value < minimum:
        return False
    if maximum is not None and numeric_value > maximum:
        return False
    return True


def validate_row_before_store_details(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str | None]:
    """
    Reject rows that already fail using SteamSpy-only fields.

    This is the main speed optimization: Steam Store appdetails is only fetched
    for candidates that pass owner, review, price, and tag-count checks.
    """
    game_name = str(row.get("game_name") or "")
    text_reason = excluded_text_reason(game_name)
    if text_reason:
        return False, text_reason

    required_columns = [
        "game_name",
        "prelaunch_price_bucket",
        "prelaunch_subgenre_primary",
        "prelaunch_price_usd",
        "prelaunch_tag_count",
        "owner_range",
        "owner_midpoint",
        "outcome_level",
        "total_reviews",
        "positive_review_pct",
        "review_sentiment",
        "log_total_reviews",
        "confidence_score",
    ]
    for column in required_columns:
        value = row.get(column)
        if value is None or pd.isna(value) or value == "":
            return False, f"missing_{column}_before_store"

    if not args.include_free_games and row["prelaunch_price_usd"] <= 0:
        return False, "free_game_before_store"

    if row["total_reviews"] < args.min_reviews:
        return False, "too_few_reviews_before_store"

    if not value_in_range(row["prelaunch_price_usd"], args.min_price_usd, args.max_price_usd):
        return False, "outside_prelaunch_price_usd_range_before_store"

    if not value_in_range(row["prelaunch_tag_count"], args.min_tag_count, args.max_tag_count):
        return False, "outside_prelaunch_tag_count_range_before_store"

    return True, None


def prefilter_candidate_from_tag_row(
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str | None]:
    """
    Cheaply reject obvious misses from the SteamSpy tag-list response.

    Missing tag-list values are not rejected here because appdetails may fill
    them in. This function only rejects when the tag row already proves the
    candidate cannot pass the final filters.
    """
    fallback_row = get_first_tag_row(candidate)

    text_reason = excluded_text_reason(str(fallback_row.get("name") or ""))
    if text_reason:
        return False, f"{text_reason}_tag_prefilter"

    total_reviews, _, _ = review_metrics(fallback_row)
    if total_reviews is not None and total_reviews > 0 and total_reviews < args.min_reviews:
        return False, "too_few_reviews_tag_prefilter"

    price_usd = parse_price_usd(fallback_row, {})
    if price_usd is not None:
        if not args.include_free_games and price_usd <= 0:
            return False, "free_game_tag_prefilter"
        if not value_in_range(price_usd, args.min_price_usd, args.max_price_usd):
            return False, "outside_price_tag_prefilter"

    return True, None


def validate_model_row(
    row: dict[str, Any],
    store_data: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[bool, str | None]:
    """Apply the default quality constraints for the narrow training pull."""
    product_reason = excluded_product_reason(
        row=row,
        store_data=store_data,
        require_store_game=args.require_store_game,
    )
    if product_reason:
        return False, product_reason

    required_columns = [
        "game_name",
        "prelaunch_price_bucket",
        "prelaunch_subgenre_primary",
        "prelaunch_price_usd",
        "prelaunch_language_count",
        "prelaunch_screenshots_count",
        "prelaunch_movies_count",
        "prelaunch_achievements_count",
        "prelaunch_tag_count",
        "owner_range",
        "owner_midpoint",
        "outcome_level",
        "total_reviews",
        "positive_review_pct",
        "review_sentiment",
        "log_total_reviews",
        "confidence_score",
    ]
    for column in required_columns:
        value = row.get(column)
        if value is None or pd.isna(value) or value == "":
            return False, f"missing_{column}"

    if not args.include_free_games and row["prelaunch_price_usd"] <= 0:
        return False, "free_game"

    if row["total_reviews"] < args.min_reviews:
        return False, "too_few_reviews"

    range_checks = {
        "prelaunch_price_usd": (args.min_price_usd, args.max_price_usd),
        "prelaunch_language_count": (args.min_language_count, args.max_language_count),
        "prelaunch_screenshots_count": (args.min_screenshots, args.max_screenshots),
        "prelaunch_movies_count": (args.min_movies, args.max_movies),
        "prelaunch_achievements_count": (args.min_achievements, args.max_achievements),
        "prelaunch_tag_count": (args.min_tag_count, args.max_tag_count),
    }
    for column, (minimum, maximum) in range_checks.items():
        if not value_in_range(row.get(column), minimum, maximum):
            return False, f"outside_{column}_range"

    return True, None


def record_skip(skip_counts: dict[str, int], reason: str) -> None:
    """Track why candidates are filtered out."""
    skip_counts[reason] = skip_counts.get(reason, 0) + 1


def pull_dataset(args: argparse.Namespace) -> tuple[pd.DataFrame, int, dict[str, int]]:
    """Run the full pull, filter, derive features, and return the final DataFrame."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "senior_project_webtool/1.0 "
                "(focused academic data collection; contact: local project script)"
            )
        }
    )

    cache_dir = Path(args.cache_dir)
    candidates = build_candidate_apps(
        session=session,
        tags=args.tags,
        cache_dir=cache_dir,
        sleep_seconds=args.sleep,
        refresh_cache=args.refresh_cache,
    )

    print(f"\nCandidate games pulled from tags: {len(candidates)}")

    rows: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    sorted_candidates = sorted(candidates.items(), key=candidate_sort_key)

    for index, (appid, candidate) in enumerate(sorted_candidates, start=1):
        if args.max_apps is not None and len(rows) >= args.max_apps:
            break

        fallback_row = get_first_tag_row(candidate)
        owner_low, owner_high, owner_midpoint = parse_owner_range(fallback_row.get("owners"))
        if not should_keep_owner_range(
            owner_low=owner_low,
            owner_high=owner_high,
            owner_midpoint=owner_midpoint,
            owner_midpoint_max=args.owner_midpoint_max,
            owner_high_max=args.owner_high_max,
            include_straddlers=args.include_straddlers,
        ):
            record_skip(skip_counts, "outside_owner_focus")
            continue

        should_prefilter_keep, prefilter_reason = prefilter_candidate_from_tag_row(candidate, args)
        if not should_prefilter_keep:
            record_skip(skip_counts, prefilter_reason or "failed_tag_prefilter")
            continue

        print(f"[{index}/{len(sorted_candidates)}] Pulling SteamSpy appdetails for app {appid}")

        try:
            steamspy_detail = fetch_steamspy_appdetails(
                session=session,
                appid=appid,
                cache_dir=cache_dir,
                sleep_seconds=args.sleep,
                refresh_cache=args.refresh_cache,
            )
        except ApiFetchError as exc:
            print(f"Skipping app {appid}: SteamSpy appdetails failed. {exc}")
            record_skip(skip_counts, "steamspy_appdetails_api_error")
            continue

        row_before_store = build_row(
            appid=appid,
            candidate=candidate,
            steamspy_detail=steamspy_detail,
            store_data={},
        )

        # Re-check with detailed owner data in case appdetails changed the range.
        if not should_keep_owner_range(
            owner_low=row_before_store["owner_low"],
            owner_high=row_before_store["owner_high"],
            owner_midpoint=row_before_store["owner_midpoint"],
            owner_midpoint_max=args.owner_midpoint_max,
            owner_high_max=args.owner_high_max,
            include_straddlers=args.include_straddlers,
        ):
            record_skip(skip_counts, "outside_owner_focus_after_details")
            continue

        is_ready_for_store, pre_store_reason = validate_row_before_store_details(row_before_store, args)
        if not is_ready_for_store:
            record_skip(skip_counts, pre_store_reason or "failed_before_store_details")
            continue

        print(f"[{index}/{len(sorted_candidates)}] Pulling Store appdetails for app {appid}")

        try:
            store_data = fetch_store_appdetails(
                session=session,
                appid=appid,
                cache_dir=cache_dir,
                sleep_seconds=args.sleep,
                refresh_cache=args.refresh_cache,
            )
        except ApiFetchError as exc:
            print(f"Skipping app {appid}: Steam Store appdetails failed. {exc}")
            record_skip(skip_counts, "steam_store_appdetails_api_error")
            continue

        row = build_row(
            appid=appid,
            candidate=candidate,
            steamspy_detail=steamspy_detail,
            store_data=store_data,
        )

        is_valid, reason = validate_model_row(row=row, store_data=store_data, args=args)
        if not is_valid:
            record_skip(skip_counts, reason or "failed_quality_filter")
            continue

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=FINAL_COLUMNS), len(candidates), skip_counts

    df = df.reindex(columns=FINAL_COLUMNS)
    df = df.drop_duplicates(subset=["appid"]).sort_values(["owner_midpoint", "appid"], na_position="last")
    return df.reset_index(drop=True), len(candidates), skip_counts


def print_summary(
    df: pd.DataFrame,
    games_pulled: int,
    skip_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Print the quality checks requested for the focused dataset."""
    print("\nPull complete")
    print(f"Number of games pulled: {games_pulled}")
    print(f"Number of rows kept: {len(df)}")

    print("\nOutcome level counts:")
    if "outcome_level" in df and not df.empty:
        print(df["outcome_level"].value_counts(dropna=False).to_string())
    else:
        print("No rows available")

    print("\nMissing value counts for model variables:")
    model_columns = PRELAUNCH_COLUMNS + LAUNCH_TRACTION_COLUMNS
    if not df.empty:
        print(df[model_columns].isna().sum().to_string())
    else:
        print(pd.Series(0, index=model_columns).to_string())

    print("\nFiltered candidate counts:")
    if skip_counts:
        for reason, count in sorted(skip_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"{reason}: {count}")
    else:
        print("No candidates filtered")

    print("\nFinal column list:")
    for column in FINAL_COLUMNS:
        print(f"- {column}")

    print(f"\nSaved CSV: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull a narrow low-owner Steam dataset for senior_project_webtool."
    )
    parser.add_argument(
        "--output",
        default="data/new_low_owner_prediction_dataset.csv",
        help="CSV path for the final narrow dataset.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/api_cache_steamspy_low_owner",
        help="Directory for cached SteamSpy and Steam Store JSON responses.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to wait after each uncached API request.",
    )
    parser.add_argument(
        "--max-apps",
        type=int,
        default=300,
        help="Maximum number of kept apps to write. Use 0 for no limit.",
    )
    parser.add_argument(
        "--owner-midpoint-max",
        type=int,
        default=35_000,
        help="Keep games with owner_midpoint at or below this value.",
    )
    parser.add_argument(
        "--owner-high-max",
        type=int,
        default=50_000,
        help="When straddlers are included, keep ranges with owner_high at or below this value.",
    )
    parser.add_argument(
        "--include-straddlers",
        dest="include_straddlers",
        action="store_true",
        default=True,
        help="Include ranges that straddle the 35,000-user viability boundary.",
    )
    parser.add_argument(
        "--exclude-straddlers",
        dest="include_straddlers",
        action="store_false",
        help="Only keep games with owner_midpoint under the midpoint threshold.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached JSON files and fetch fresh API responses.",
    )
    parser.add_argument(
        "--min-reviews",
        type=int,
        default=10,
        help="Minimum total SteamSpy reviews required for review sentiment fields.",
    )
    parser.add_argument(
        "--include-free-games",
        action="store_true",
        help="Keep free-to-play games. By default, free games are excluded.",
    )
    parser.add_argument(
        "--require-store-game",
        dest="require_store_game",
        action="store_true",
        default=True,
        help="Require Steam Store appdetails type == game.",
    )
    parser.add_argument(
        "--allow-non-game-store-types",
        dest="require_store_game",
        action="store_false",
        help="Allow Store API products whose type is not game.",
    )
    parser.add_argument(
        "--min-price-usd",
        type=float,
        default=0.99,
        help="Minimum paid game price to keep.",
    )
    parser.add_argument(
        "--max-price-usd",
        type=float,
        default=29.99,
        help="Maximum game price to keep.",
    )
    parser.add_argument(
        "--min-language-count",
        type=int,
        default=1,
        help="Minimum supported language count to keep.",
    )
    parser.add_argument(
        "--max-language-count",
        type=int,
        default=25,
        help="Maximum supported language count to keep.",
    )
    parser.add_argument(
        "--min-screenshots",
        type=int,
        default=1,
        help="Minimum screenshot count to keep.",
    )
    parser.add_argument(
        "--max-screenshots",
        type=int,
        default=30,
        help="Maximum screenshot count to keep.",
    )
    parser.add_argument(
        "--min-movies",
        type=int,
        default=0,
        help="Minimum trailer/movie count to keep.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=10,
        help="Maximum trailer/movie count to keep.",
    )
    parser.add_argument(
        "--min-achievements",
        type=int,
        default=0,
        help="Minimum achievement count to keep.",
    )
    parser.add_argument(
        "--max-achievements",
        type=int,
        default=200,
        help="Maximum achievement count to keep.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=3,
        help="Minimum Steam tag count to keep.",
    )
    parser.add_argument(
        "--max-tag-count",
        type=int,
        default=25,
        help="Maximum Steam tag count to keep.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=DEFAULT_TAGS,
        help="SteamSpy tags to use as the candidate pool.",
    )

    args = parser.parse_args()
    if args.max_apps == 0:
        args.max_apps = None
    return args


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, candidate_count, skip_counts = pull_dataset(args)
    df.to_csv(output_path, index=False)

    print_summary(df=df, games_pulled=candidate_count, skip_counts=skip_counts, output_path=output_path)


if __name__ == "__main__":
    main()
