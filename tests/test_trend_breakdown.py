import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from tools import resolve_filter, trend_breakdown
from tests.fixtures import tickets_df_small, catalogs


def _setup_environment():
    df = tickets_df_small()
    trend_breakdown.register_ticket_dataframe(df)
    resolve_filter.reset_state()
    resolve_filter.update_catalogs(**catalogs())
    return df


def test_trend_breakdown_returns_top_categories():
    df = _setup_environment()
    resolved = resolve_filter.resolve_filters(
        product_hint="apple",
        time_range={"from": "2025-09-29", "to": "2025-10-10"},
    )

    breakdown = trend_breakdown.get_trend_breakdown(
        filter_id=resolved["filter_id"],
        group_time_bucket="day",
        category_dim="level_2_classification",
        metric="ticket_count",
        top_n_categories=2,
    )

    assert "timeseries" in breakdown
    assert breakdown["meta"]["metric"] == "ticket_count"
    assert breakdown["meta"]["top_categories"]

    filtered = trend_breakdown.apply_filter_to_df(df, resolved["filters"])
    expected_top = (
        filtered.groupby("level_2_classification")
        .size()
        .sort_values(ascending=False)
        .index[0]
    )
    assert breakdown["meta"]["top_categories"][0]["category"] == expected_top


def test_trend_breakdown_refund_metric_totals_match_dataframe_sum():
    df = _setup_environment()
    resolved = resolve_filter.resolve_filters(
        product_hint="apple",
        time_range={"from": "2025-09-29", "to": "2025-10-10"},
        focus_metric="refund_count_15d",
    )

    output = trend_breakdown.get_trend_breakdown(
        filter_id=resolved["filter_id"],
        group_time_bucket="day",
        category_dim="level_2_classification",
        metric="refund_count_15d",
        top_n_categories=3,
    )

    filtered = trend_breakdown.apply_filter_to_df(df, resolved["filters"])
    expected_total = float(filtered["refund_count_in_15_days"].fillna(0).sum())

    assert output["meta"]["total_all_categories"] == expected_total
    assert all("value" in row for row in output["timeseries"])


def test_apply_filter_respects_extended_dimensions():
    df = _setup_environment()
    filters = {
        "products": [],
        "region": [],
        "l2_class": [],
        "l1_class": [],
        "concern_type": ["Complaint"],
        "city": ["Delhi"],
        "product_category": ["Fruits"],
        "date_from": "2025-10-01",
        "date_to": "2025-10-10",
        "metric": "ticket_count",
    }

    filtered_df = trend_breakdown.apply_filter_to_df(df, filters)

    assert not filtered_df.empty
    assert set(filtered_df["city"]) == {"Delhi"}
    assert set(filtered_df["product_category"]) == {"Fruits"}


def test_create_or_update_filter_canonicalizes_hints():
    resolve_filter.reset_state()
    resolve_filter.update_catalogs(**catalogs())

    resolved = resolve_filter.resolve_filters(
        product_hint="apple",
        issue_hint="not fresh",
        level1_hint="product quality",
        concern_hint="complaint",
        city_hint="delhi",
        product_category_hint="fruits",
        time_range={"from": "2025-09-29", "to": "2025-10-04"},
        focus_metric="refund_count_15d",
    )

    payload = resolve_filter.FilterPayload(**resolved["filters"])
    record = resolve_filter.create_filter_record(
        payload,
        description="Apple quality complaints",
    )
    filters = resolve_filter.get_filter_payload(record["filter_id"])

    assert set(filters["products"]) >= {
        "Gala Apple - 4 pcs",
        "Washington Apple - 1 kg",
    }
    assert filters["l2_class"] == ["Product Not Fresh / Expired / Rotten / Infested"]
    assert filters["l1_class"] == ["Product / Quality"]
    assert filters["concern_type"] == ["Complaint"]
    assert filters["city"] == ["Delhi"]
    assert filters["product_category"] == ["Fruits"]
    assert filters["date_from"] == "2025-09-29"
    assert filters["date_to"] == "2025-10-04"
    assert filters["metric"] == "refund_count_15d"
