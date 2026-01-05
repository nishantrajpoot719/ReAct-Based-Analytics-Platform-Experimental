import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tools.resolve_filter as resolve_filter


@pytest.fixture(autouse=True)
def _reset_resolver():
    resolve_filter.reset_state()
    yield
    resolve_filter.reset_state()


def _seed_catalogs():
    resolve_filter.update_catalogs(
        products=[
            "Gala Apple - 4 pcs",
            "Washington Apple - 1 kg",
            "Honeycrisp Apple",
            "Banana (Robusta) - 500 gm",
        ],
        regions=["NCR", "Mumbai"],
        level1_classes=[
            "Product / Quality",
            "Delivery & Fulfillment",
        ],
        level2_classes=[
            "Product Not Fresh / Expired / Rotten / Infested",
            "Partial / Missing Items at Delivery",
        ],
        concern_types=[
            "Complaint",
            "Query",
        ],
        cities=["Delhi", "Mumbai"],
        product_categories=[
            "Fruits",
            "Dairy",
        ],
    )


def test_product_matches_handles_aliases():
    _seed_catalogs()

    matches, candidates = resolve_filter.product_matches("apples")

    assert matches
    assert any("Apple" in product for product in matches)
    assert candidates


def test_product_matches_handles_comma_separated_hints():
    _seed_catalogs()

    matches, _ = resolve_filter.product_matches("apple, banana")

    assert matches
    assert any("Apple" in product for product in matches)
    assert any("Banana" in product for product in matches)


def test_resolve_filters_returns_canonical_filters():
    _seed_catalogs()

    result = resolve_filter.resolve_filters(
        product_hint="apple",
        region_hint="ncr",
        issue_hint="not fresh",
        city_hint="delhi",
        level1_hint="product quality",
        concern_hint="complaint",
        product_category_hint="fruits",
        time_range={"from": "2025-09-30", "to": "2025-10-04"},
        focus_metric="ticket_count",
        now=datetime(2025, 10, 5),
    )

    filters = result["filters"]

    assert result["filter_id"]
    assert filters["products"]
    assert filters["region"] == ["NCR"]
    assert filters["l2_class"] == ["Product Not Fresh / Expired / Rotten / Infested"]
    assert filters["l1_class"] == ["Product / Quality"]
    assert filters["city"] == ["Delhi"]
    assert filters["concern_type"] == ["Complaint"]
    assert filters["product_category"] == ["Fruits"]
    assert filters["date_from"] == "2025-09-30"
    assert filters["date_to"] == "2025-10-04"
    assert filters["metric"] == "ticket_count"
    assert resolve_filter.FILTER_STORE[result["filter_id"]] == filters


def test_days_time_range_defaults_to_last_n_days():
    _seed_catalogs()

    now = datetime(2025, 10, 10)
    result = resolve_filter.resolve_filters(
        product_hint=None,
        time_range={"days": 7},
        now=now,
    )

    filters = result["filters"]
    expected_from = (now - timedelta(days=7)).date().isoformat()
    expected_to = now.date().isoformat()

    assert filters["date_from"] == expected_from
    assert filters["date_to"] == expected_to
    assert filters["products"] == []
    assert filters["region"] == []
    assert filters["l2_class"] == []
    assert filters["l1_class"] == []
    assert filters["city"] == []
    assert filters["concern_type"] == []
    assert filters["product_category"] == []


def test_metric_defaults_to_ticket_count():
    _seed_catalogs()

    outcome = resolve_filter.resolve_filters(
        product_hint="apple",
        focus_metric="unknown_metric",
    )

    assert outcome["filters"]["metric"] == "ticket_count"


def test_resolve_time_range_accepts_date_keys():
    now = datetime(2025, 10, 10)
    resolved = resolve_filter.resolve_time_range(
        {"date_from": "2025-09-30", "date_to": "2025-10-05"},
        now=now,
    )

    assert resolved["date_from"] == "2025-09-30"
    assert resolved["date_to"] == "2025-10-05"
