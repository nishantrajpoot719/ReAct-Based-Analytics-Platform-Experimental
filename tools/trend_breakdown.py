from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from . import resolve_filter

logger = logging.getLogger(__name__)

ACTIVE_TICKETS_DF: Optional[pd.DataFrame] = None


def register_ticket_dataframe(df) -> None:
    global ACTIVE_TICKETS_DF
    ACTIVE_TICKETS_DF = df
    logger.debug("Registered ticket dataframe", extra={"rows": len(df)})


def get_registered_dataframe():
    return ACTIVE_TICKETS_DF


def _ensure_dataframe(df):
    target = df if df is not None else ACTIVE_TICKETS_DF
    if target is None:
        raise ValueError("No ticket dataframe registered. Call register_ticket_dataframe first.")
    if target.empty:
        return target.copy()
    return target.copy()


def _time_bucket(values: pd.Series, bucket: str) -> pd.Series:
    ts = pd.to_datetime(values, errors="coerce")
    if bucket == "day":
        return ts.dt.strftime("%Y-%m-%d")
    if bucket == "hour":
        return ts.dt.strftime("%Y-%m-%d %H:00")
    if bucket == "week":
        periods = ts.dt.to_period("W-MON")
        return periods.apply(lambda p: p.start_time.strftime("%Y-%m-%d"))
    raise ValueError(f"Unsupported time bucket: {bucket}")


def apply_filter_to_df(df, filters: Dict[str, Any]):
    required_columns = {
        "product",
        "region",
        "level_2_classification",
        "level_1_classification",
        "concern_type",
        "product_category",
        "city",
        "created_date",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataframe missing required columns: {sorted(missing)}")

    mask = pd.Series(True, index=df.index)
    products = filters.get("products") or []
    if products:
        mask &= df["product"].isin(products)
    region = filters.get("region") or []
    if region:
        mask &= df["region"].isin(region)
    l2_class = filters.get("l2_class") or []
    if l2_class:
        mask &= df["level_2_classification"].isin(l2_class)
    l1_class = filters.get("l1_class") or []
    if l1_class:
        mask &= df["level_1_classification"].isin(l1_class)
    concern = filters.get("concern_type") or []
    if concern:
        mask &= df["concern_type"].isin(concern)
    city = filters.get("city") or []
    if city:
        mask &= df["city"].isin(city)
    product_category = filters.get("product_category") or []
    if product_category:
        mask &= df["product_category"].isin(product_category)
    date_from = filters.get("date_from")
    if date_from:
        mask &= pd.to_datetime(df["created_date"]) >= pd.to_datetime(date_from)
    date_to = filters.get("date_to")
    if date_to:
        mask &= pd.to_datetime(df["created_date"]) <= pd.to_datetime(date_to)
    return df.loc[mask].copy()


def _prepare_metric(df, metric: str):
    df = df.copy()
    if metric == "ticket_count":
        df["__value__"] = 1
    elif metric == "refund_count_15d":
        if "refund_count_in_15_days" not in df.columns:
            raise ValueError("Column 'refund_count_in_15_days' required for refund_count_15d metric.")
        df["__value__"] = df["refund_count_in_15_days"].fillna(0)
    else:
        df["__value__"] = 1
    return df


def _top_categories(df, category_dim: str, top_n: int):
    totals = (
        df.groupby(category_dim)["__value__"]
        .sum()
        .reset_index()
        .rename(columns={category_dim: "category", "__value__": "total_value"})
        .sort_values("total_value", ascending=False)
    )
    return totals.head(top_n)


def get_trend_breakdown(
    *,
    filter_id: str,
    group_time_bucket: str,
    category_dim: str,
    metric: str,
    top_n_categories: int = 5,
    df_override = None,
) -> Dict[str, Any]:
    """
    Generate a ticket trend breakdown grouped by category and time bucket.

    Args:
        filter_id: Identifier of a saved filter in ``resolve_filter.FILTER_STORE``.
        group_time_bucket: Time granularity for aggregation; accepts ``"day"``, ``"hour"``, or ``"week"``.
        category_dim: Column name in the dataframe used to split metrics by category; Literal: level_1_classification, level_2_classification, product, region, concern_type, product_category, city.
        metric: Metric identifier to aggregate (e.g., ``"ticket_count"`` or ``"refund_count_15d"``).
        top_n_categories: Number of categories to surface in the trend (ordered by total metric value).
        df_override: Dataframe to use instead of the registered active tickets dataframe.

    Returns:
        Dictionary with keys:
            - ``timeseries``: List of records containing ``time_bucket``, ``category``, and ``value``.
            - ``meta``: Summary metadata including requested filters, top categories, and total values.
        On error, returns a dictionary containing:
            - ``error``: Stable error code.
            - ``message``: Human-readable description of the failure.
            - ``hint``: Optional guidance on how to recover from the error.
    """
    if not filter_id or not str(filter_id).strip():
        return {
            "error": "INVALID_FILTER_ID",
            "message": "filter_id must be a non-empty string.",
            "hint": "Pass the identifier returned by resolve_filter.create_filter_record or resolve_filter.list_filter_ids.",
        }
    filter_id = str(filter_id).strip()

    valid_buckets = {"day", "hour", "week"}
    if group_time_bucket not in valid_buckets:
        return {
            "error": "INVALID_TIME_BUCKET",
            "message": f"Unsupported time bucket '{group_time_bucket}'.",
            "hint": "Use one of: day, hour, week.",
        }

    if not category_dim or not str(category_dim).strip():
        return {
            "error": "INVALID_CATEGORY_DIM",
            "message": "category_dim must be a non-empty column name.",
            "hint": "Provide a column from the registered dataframe, for example 'product' or 'concern_type'.",
        }
    category_dim = str(category_dim).strip()

    if not metric or not str(metric).strip():
        return {
            "error": "INVALID_METRIC",
            "message": "metric must be a non-empty string.",
            "hint": "Use a supported metric such as 'ticket_count' or 'refund_count_15d'.",
        }
    metric = str(metric).strip()

    if top_n_categories <= 0:
        return {
            "error": "INVALID_TOP_N",
            "message": "top_n_categories must be greater than zero.",
            "hint": "Choose a positive integer to indicate how many categories to include.",
        }

    if filter_id not in resolve_filter.FILTER_STORE:
        return {
            "error": "FILTER_NOT_FOUND",
            "message": f"Filter '{filter_id}' was not found in the filter store.",
            "hint": "Call resolve_filter.create_filter_record or ensure the id is still active.",
        }

    try:
        df_base = _ensure_dataframe(df_override)
    except ValueError as exc:
        return {
            "error": "NO_DATAFRAME",
            "message": str(exc),
            "hint": "Call register_ticket_dataframe with the active tickets dataframe before invoking this tool.",
        }

    if category_dim not in df_base.columns:
        available_columns = ", ".join(sorted(df_base.columns)[:10])
        return {
            "error": "INVALID_DIM",
            "message": f"Column '{category_dim}' is not present in the dataframe.",
            "hint": f"Available columns include: {available_columns}",
        }
    if "created_date" not in df_base.columns:
        return {
            "error": "MISSING_COLUMN",
            "message": "Column 'created_date' is required for time bucketing.",
            "hint": "Ensure the dataframe includes a 'created_date' column with timestamp values.",
        }

    filters = resolve_filter.get_filter_payload(filter_id)
    try:
        df_filtered = apply_filter_to_df(df_base, filters)
    except ValueError as exc:
        return {
            "error": "FILTER_APPLY_FAILED",
            "message": str(exc),
            "hint": "Verify the filter payload fields align with the dataframe schema.",
        }

    try:
        df_metric = _prepare_metric(df_filtered, metric)
    except ValueError as exc:
        return {
            "error": "INVALID_METRIC_CONFIGURATION",
            "message": str(exc),
            "hint": "Add the required metric columns to the dataframe or choose a different metric.",
        }
    total_value = df_metric["__value__"].sum()
    totals_df = _top_categories(df_metric, category_dim, top_n_categories) if not df_metric.empty else pd.DataFrame(columns=["category", "total_value"])

    if df_metric.empty or totals_df.empty:
        meta = {
            "date_from": filters.get("date_from"),
            "date_to": filters.get("date_to"),
            "category_dim": category_dim,
            "metric": metric,
            "top_categories": totals_df.to_dict(orient="records"),
            "total_all_categories": int(total_value) if metric == "ticket_count" else float(total_value),
        }
        return {"timeseries": [], "meta": meta}

    top_labels = totals_df["category"].tolist()
    df_top = df_metric[df_metric[category_dim].isin(top_labels)].copy()
    df_top["time_bucket"] = _time_bucket(df_top["created_date"], group_time_bucket)
    grouped = (
        df_top.groupby(["time_bucket", category_dim])["__value__"]
        .sum()
        .reset_index()
        .rename(columns={category_dim: "category", "__value__": "value"})
    )

    meta = {
        "date_from": filters.get("date_from"),
        "date_to": filters.get("date_to"),
        "category_dim": category_dim,
        "metric": metric,
        "top_categories": totals_df.to_dict(orient="records"),
        "total_all_categories": int(total_value) if metric == "ticket_count" else float(total_value),
    }

    return {
        "timeseries": grouped.to_dict(orient="records"),
        "meta": meta,
    }


__all__ = [
    "register_ticket_dataframe",
    "get_registered_dataframe",
    "apply_filter_to_df",
    "get_trend_breakdown",
]
