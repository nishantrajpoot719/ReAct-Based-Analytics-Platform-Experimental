import math
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple


def _safe_pct(numerator: float, denominator: float) -> Optional[float]:
    if denominator is None or denominator == 0:
        return None
    return (numerator / denominator) * 100.0


def _pct_change(curr: float, prev: Optional[float]) -> Optional[float]:
    """
    % change = (curr - prev)/prev * 100
    Returns None if prev is 0 or missing.
    """
    if prev is None or prev == 0:
        return None
    return ((curr - prev) / prev) * 100.0


def _index_by_category(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Turns:
       [{'category': 'Damaged Packaging', 'tickets': 11, 'refund_count_15d': 5}, ...]
    into:
       { 'Damaged Packaging': {'tickets': 11, 'refund_count_15d': 5}, ... }
    """
    out = {}
    for r in rows:
        cat = r["category"]
        out[cat] = {
            "tickets": r.get("tickets", 0),
            "refund_count_15d": r.get("refund_count_15d", 0),
        }
    return out


def _is_number(value: Any) -> bool:
    """Return True for int/float values (excluding bool) that are not NaN."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return value == value  # filter out NaNs


def _window_total_and_index(
    window: Optional[Dict[str, Any]]
) -> Tuple[Optional[float], Dict[str, Dict[str, Any]]]:
    if not window:
        return None, {}
    total_value = window.get("total_tickets")
    total_float = float(total_value) if total_value is not None else None
    categories = window.get("by_category") or []
    return total_float, _index_by_category(categories)


def _compute_rolling_baseline(
    history_windows: List[Dict[str, Any]],
    lookback: int,
    mode: str,
) -> Tuple[Optional[float], Dict[str, Dict[str, Any]]]:
    if not history_windows:
        return None, {}
    sample = history_windows[-lookback:] if lookback > 0 else history_windows[:]
    if not sample:
        return None, {}

    totals: List[float] = []
    indexes: List[Dict[str, Dict[str, Any]]] = []
    for window in sample:
        total, idx = _window_total_and_index(window)
        totals.append(float(total) if total is not None else 0.0)
        indexes.append(idx)

    aggregator = mean if mode == "avg" else median
    total_value = aggregator(totals) if totals else None

    all_categories = set()
    for idx in indexes:
        all_categories.update(idx.keys())

    category_baseline: Dict[str, Dict[str, Any]] = {}
    for cat in all_categories:
        values: List[float] = []
        for idx in indexes:
            stats = idx.get(cat)
            values.append(float(stats.get("tickets", 0)) if stats else 0.0)
        category_baseline[cat] = {"tickets": aggregator(values)}

    return total_value, category_baseline


def _select_baseline(
    previous_window: Optional[Dict[str, Any]],
    history_windows: List[Dict[str, Any]],
    baseline: str,
    lookback: int,
) -> Tuple[str, Optional[float], Dict[str, Dict[str, Any]]]:
    normalized = (baseline or "auto").lower()

    def _from_window(window: Optional[Dict[str, Any]], label: str):
        if window is None:
            return None
        total, idx = _window_total_and_index(window)
        return label, total, idx

    def _from_history(label: str):
        if not history_windows:
            return None
        return _from_window(history_windows[-1], label)

    def _from_rolling(label: str, mode: str):
        total, idx = _compute_rolling_baseline(history_windows, lookback, mode)
        if total is None and not idx:
            return None
        return label, total, idx

    if normalized == "previous":
        result = _from_window(previous_window, "previous")
        return result if result else ("previous", None, {})

    if normalized in {"dod", "wow", "mom"}:
        result = _from_history(normalized)
        return result if result else (normalized, None, {})

    if normalized in {"avgn", "mediann"}:
        mode = "avg" if normalized == "avgn" else "median"
        result = _from_rolling(normalized, mode)
        return result if result else (normalized, None, {})

    if normalized == "auto":
        for resolver in (
            lambda: _from_window(previous_window, "previous"),
            lambda: _from_history("history_last"),
            lambda: _from_rolling("rolling_avg", "avg"),
            lambda: _from_rolling("rolling_median", "median"),
        ):
            result = resolver()
            if result:
                return result
        return ("auto", None, {})

    return (normalized, None, {})


def _compute_mix_concentration(
    shares: List[Optional[float]],
    top_k: int,
) -> Dict[str, Optional[float]]:
    valid = [s for s in shares if s is not None and s > 0]
    if not valid:
        return {
            "hhi": None,
            "entropy": None,
            "effective_categories": None,
            "top_k_coverage_pct": None,
        }

    fractions = [s / 100.0 for s in valid]
    hhi = sum(p * p for p in fractions)
    entropy = -sum(p * math.log(p) for p in fractions if p > 0)
    effective = (1 / hhi) if hhi > 0 else None
    top_k_coverage = sum(sorted(valid, reverse=True)[:top_k]) if top_k > 0 else 0.0

    return {
        "hhi": hhi,
        "entropy": entropy,
        "effective_categories": effective,
        "top_k_coverage_pct": top_k_coverage,
    }


def calculate_metrics(
    current_window: Dict[str, Any],
    previous_window: Optional[Dict[str, Any]] = None,
    top_n: int = 5,
    history: Optional[List[Dict[str, Any]]] = None,
    baseline: str = "auto",
    lookback: int = 4,
    granularity: str = "week",
    min_count: int = 10,
    top_k_movers: int = 5,
    concentration_top_k: int = 3,
) -> Dict[str, Any]:
    """
    Compute summary metrics for the active ticket window, optional comparison window,
    and optional history for richer baselines.

    Args:
        current_window: Active aggregation window with ``total_tickets`` and optional ``by_category`` rows.
        previous_window: Optional previous aggregation window (same structure as ``current_window``).
        top_n: Number of categories to expose in ``top_categories_by_share``.
        history: Optional list of windows ordered oldest -> newest, used for fallback baselines.
        baseline: Baseline strategy. One of ``auto``, ``previous``, ``dod``, ``wow``, ``mom``,
            ``avgN``, or ``medianN`` (case-insensitive).
        lookback: Number of history periods to include when using rolling averages/medians.
        granularity: Data cadence label (``day``, ``week``, ``month``). Reserved for downstream use.
        min_count: Minimum ticket count required for a category to appear in mover lists.
        top_k_movers: Max entries to emit in the ``top_risers`` and ``top_fallers`` lists.
        concentration_top_k: Number of top categories to include when computing coverage share.

    Returns:
        Dictionary with summary metrics or a structured error payload.
    """
    if current_window is None or not isinstance(current_window, dict):
        return {
            "error": "INVALID_CURRENT_WINDOW",
            "message": "current_window must be a dictionary describing the active metric window.",
            "hint": "Pass the output of trend aggregation with keys such as 'total_tickets' and 'by_category'.",
        }

    if not isinstance(top_n, int) or top_n <= 0:
        return {
            "error": "INVALID_TOP_N",
            "message": "top_n must be a positive integer.",
            "hint": "Set top_n to a whole number greater than zero (default is 5).",
        }

    if previous_window is not None and not isinstance(previous_window, dict):
        return {
            "error": "INVALID_PREVIOUS_WINDOW",
            "message": "previous_window must be a dictionary when provided.",
            "hint": "Provide the same structure as current_window or omit the parameter.",
        }

    if history is not None and not isinstance(history, list):
        return {
            "error": "INVALID_HISTORY",
            "message": "history must be a list of metric windows if provided.",
            "hint": "Supply history as a list like [window_t_minus_2, window_t_minus_1].",
        }

    if not isinstance(baseline, str):
        return {
            "error": "INVALID_BASELINE",
            "message": "baseline must be a string strategy identifier.",
            "hint": "Pick from auto, previous, dod, wow, mom, avgN, or medianN.",
        }

    baseline_key = baseline.lower()
    allowed_baselines = {"auto", "previous", "dod", "wow", "mom", "avgn", "mediann"}
    if baseline_key not in allowed_baselines:
        return {
            "error": "INVALID_BASELINE",
            "message": f"baseline '{baseline}' is not supported.",
            "hint": "Pick from auto, previous, dod, wow, mom, avgN, or medianN.",
        }

    if not isinstance(lookback, int) or lookback <= 0:
        return {
            "error": "INVALID_LOOKBACK",
            "message": "lookback must be a positive integer.",
            "hint": "Set lookback to the number of history windows to average over (e.g., 4).",
        }

    if not isinstance(min_count, int) or min_count < 0:
        return {
            "error": "INVALID_MIN_COUNT",
            "message": "min_count must be a non-negative integer.",
            "hint": "Use min_count to hide noisy movers below a volume threshold.",
        }

    if not isinstance(top_k_movers, int) or top_k_movers <= 0:
        return {
            "error": "INVALID_TOP_K_MOVERS",
            "message": "top_k_movers must be a positive integer.",
            "hint": "Set top_k_movers to control how many risers/fallers get returned.",
        }

    if not isinstance(concentration_top_k, int) or concentration_top_k <= 0:
        return {
            "error": "INVALID_CONCENTRATION_K",
            "message": "concentration_top_k must be a positive integer.",
            "hint": "Pick how many categories to include when reporting coverage share.",
        }

    if not isinstance(granularity, str):
        return {
            "error": "INVALID_GRANULARITY",
            "message": "granularity must be a string (day, week, or month).",
            "hint": "Set granularity to describe the cadence of your trend data.",
        }

    granularity_key = granularity.lower()
    if granularity_key not in {"day", "week", "month"}:
        return {
            "error": "INVALID_GRANULARITY",
            "message": f"granularity '{granularity}' is not supported.",
            "hint": "Use 'day', 'week', or 'month'.",
        }

    def _validate_window_payload(window: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
        total = window.get("total_tickets")
        if total is not None and not _is_number(total):
            return {
                "error": "INVALID_TOTAL_TICKETS",
                "message": f"{label}.total_tickets must be numeric.",
                "hint": f"Ensure {label}.total_tickets is an integer or float value.",
            }

        categories = window.get("by_category")
        if categories is None:
            return None
        if not isinstance(categories, list):
            return {
                "error": "INVALID_CATEGORY_LIST",
                "message": f"{label}.by_category must be a list of category dictionaries.",
                "hint": "Provide by_category as a list like [{'category': 'Damaged', 'tickets': 10}].",
            }
        for idx, row in enumerate(categories):
            if not isinstance(row, dict):
                return {
                    "error": "INVALID_CATEGORY_ENTRY",
                    "message": f"{label}.by_category[{idx}] must be a dictionary.",
                    "hint": "Ensure each category entry is a dict with 'category' and 'tickets' keys.",
                }
            category = row.get("category")
            if not isinstance(category, str) or not category.strip():
                return {
                    "error": "INVALID_CATEGORY_NAME",
                    "message": f"{label}.by_category[{idx}].category must be a non-empty string.",
                    "hint": "Populate the category field with a descriptive label.",
                }
            tickets = row.get("tickets")
            if tickets is not None and not _is_number(tickets):
                return {
                    "error": "INVALID_TICKET_COUNT",
                    "message": f"{label}.by_category[{idx}].tickets must be numeric.",
                    "hint": "Provide ticket counts as integers or floats.",
                }
            refunds = row.get("refund_count_15d")
            if refunds is not None and not _is_number(refunds):
                return {
                    "error": "INVALID_REFUND_COUNT",
                    "message": f"{label}.by_category[{idx}].refund_count_15d must be numeric.",
                    "hint": "Provide refund counts as integers or floats.",
                }
        return None

    error = _validate_window_payload(current_window, "current_window")
    if error:
        return error

    if previous_window:
        error = _validate_window_payload(previous_window, "previous_window")
        if error:
            return error

    history_windows = list(history) if history else []
    for idx, window in enumerate(history_windows):
        if not isinstance(window, dict):
            return {
                "error": "INVALID_HISTORY_ENTRY",
                "message": f"history[{idx}] must be a dictionary.",
                "hint": "Ensure every history entry matches the current_window structure.",
            }
        error = _validate_window_payload(window, f"history[{idx}]")
        if error:
            return error

    curr_categories = current_window.get("by_category") or []
    prev_categories = previous_window.get("by_category") if previous_window else []

    curr_total_raw = current_window.get("total_tickets")
    if curr_total_raw is None:
        curr_total_raw = 0
    curr_total = float(curr_total_raw)

    prev_total_raw = None
    prev_total = None
    if previous_window:
        prev_total_raw = previous_window.get("total_tickets")
        if prev_total_raw is not None:
            prev_total = float(prev_total_raw)
        prev_categories = prev_categories or []

    rolling_avg_total, rolling_avg_index = _compute_rolling_baseline(history_windows, lookback, "avg")
    baseline_used, baseline_total, baseline_index = _select_baseline(
        previous_window=previous_window,
        history_windows=history_windows,
        baseline=baseline_key,
        lookback=lookback,
    )
    baseline_index = baseline_index or {}

    overall_wow = _pct_change(curr_total, prev_total) if previous_window else None
    prev_total_output = prev_total_raw if previous_window else None

    overall_change = {
        "current_total_tickets": curr_total_raw,
        "previous_total_tickets": prev_total_output,
        "baseline_total_tickets": baseline_total,
        "wow_change_pct": overall_wow,
        "baseline_label": baseline_used,
        "current_window": {
            "date_from": current_window.get("date_from"),
            "date_to": current_window.get("date_to"),
        },
        "previous_window": {
            "date_from": previous_window.get("date_from") if previous_window else None,
            "date_to": previous_window.get("date_to") if previous_window else None,
        },
    }

    curr_index = _index_by_category(curr_categories)
    prev_index = _index_by_category(prev_categories) if previous_window else {}

    category_insights = []
    for cat, curr_stats in curr_index.items():
        curr_tickets_value = curr_stats["tickets"]
        curr_tickets = float(curr_tickets_value)
        curr_refunds_value = curr_stats.get("refund_count_15d", 0)
        curr_refunds = float(curr_refunds_value)

        prev_tickets_value = prev_index.get(cat, {}).get("tickets", 0) if previous_window else None
        prev_tickets = float(prev_tickets_value) if previous_window else None

        baseline_stats = baseline_index.get(cat)
        baseline_tickets_value = None
        if baseline_stats is not None:
            baseline_tickets_value = baseline_stats.get("tickets")
        elif baseline_total is not None:
            baseline_tickets_value = 0.0
        baseline_tickets = float(baseline_tickets_value) if baseline_tickets_value is not None else None

        avg_stats = rolling_avg_index.get(cat)
        avg_tickets_value = avg_stats.get("tickets") if avg_stats else None
        avg_tickets = float(avg_tickets_value) if avg_tickets_value is not None else None

        share_pct = _safe_pct(curr_tickets, curr_total)
        wow_change_cat = _pct_change(curr_tickets, prev_tickets) if previous_window else None
        refund_rate_pct = _safe_pct(curr_refunds, curr_tickets)
        change_vs_baseline_pct = _pct_change(curr_tickets, baseline_tickets) if baseline_tickets is not None else None
        change_vs_avg_pct = _pct_change(curr_tickets, avg_tickets) if avg_tickets is not None else None
        baseline_share_pct = _safe_pct(baseline_tickets, baseline_total) if (
            baseline_tickets is not None and baseline_total not in (None, 0)
        ) else None
        delta_abs = (curr_tickets - baseline_tickets) if baseline_tickets is not None else None

        category_insights.append({
            "category": cat,
            "current_tickets": curr_tickets_value,
            "previous_tickets": prev_tickets_value if previous_window else None,
            "share_of_total_pct": share_pct,
            "wow_change_pct": wow_change_cat,
            "refund_rate_pct": refund_rate_pct,
            "baseline_tickets": baseline_tickets_value,
            "baseline_share_of_total_pct": baseline_share_pct,
            "delta_abs": delta_abs,
            "change_vs_baseline_pct": change_vs_baseline_pct,
            "change_vs_avg_pct": change_vs_avg_pct,
            "rank": None,
            "rank_change": None,
        })

    category_insights_sorted = sorted(
        category_insights,
        key=lambda row: (-(row["share_of_total_pct"] or 0), row["category"].lower()),
    )

    baseline_rank_lookup: Dict[str, int] = {}
    if baseline_index and baseline_total not in (None, 0):
        baseline_rows = []
        for cat, stats in baseline_index.items():
            base_tickets_value = stats.get("tickets")
            if base_tickets_value is None:
                continue
            base_tickets = float(base_tickets_value)
            base_share = _safe_pct(base_tickets, baseline_total)
            baseline_rows.append((cat, base_share or 0.0))
        baseline_rows.sort(key=lambda item: (-item[1], item[0].lower()))
        for rank_idx, (cat, _) in enumerate(baseline_rows, start=1):
            baseline_rank_lookup[cat] = rank_idx

    for idx, row in enumerate(category_insights_sorted, start=1):
        row["rank"] = idx
        baseline_rank = baseline_rank_lookup.get(row["category"])
        row["rank_change"] = baseline_rank - idx if baseline_rank is not None else None

    top_categories_by_share = [
        {
            "category": row["category"],
            "share_of_total_pct": row["share_of_total_pct"],
        }
        for row in category_insights_sorted[:top_n]
    ]

    shares_for_concentration = [row["share_of_total_pct"] for row in category_insights_sorted]
    mix_concentration = _compute_mix_concentration(shares_for_concentration, concentration_top_k)

    movers_payload = {
        "top_risers": [],
        "top_fallers": [],
        "share_of_growth": [],
    }
    net_total_change = None
    if baseline_total is not None:
        net_total_change = curr_total - baseline_total

    mover_candidates = []
    if net_total_change is not None:
        for row in category_insights_sorted:
            current_value = row["current_tickets"]
            if current_value is None:
                continue
            current_value_float = float(current_value)
            if current_value_float < min_count:
                continue
            delta_abs = row.get("delta_abs")
            if delta_abs is None or delta_abs == 0:
                continue
            share_of_growth_pct = None
            if net_total_change != 0:
                share_of_growth_pct = (delta_abs / net_total_change) * 100.0
            mover_candidates.append({
                "category": row["category"],
                "delta_abs": delta_abs,
                "current_tickets": current_value,
                "baseline_tickets": row.get("baseline_tickets"),
                "share_of_growth_pct": share_of_growth_pct,
            })

        if mover_candidates:
            movers_payload["top_risers"] = [
                entry for entry in sorted(mover_candidates, key=lambda item: item["delta_abs"], reverse=True)
                if entry["delta_abs"] > 0
            ][:top_k_movers]
            movers_payload["top_fallers"] = [
                entry for entry in sorted(mover_candidates, key=lambda item: item["delta_abs"])
                if entry["delta_abs"] < 0
            ][:top_k_movers]
            movers_payload["share_of_growth"] = [
                entry for entry in sorted(
                    (candidate for candidate in mover_candidates if candidate["share_of_growth_pct"] is not None),
                    key=lambda item: abs(item["share_of_growth_pct"]),
                    reverse=True,
                )
            ]

    return {
        "overall_change": overall_change,
        "category_insights": category_insights_sorted,
        "top_categories_by_share": top_categories_by_share,
        "baseline_used": baseline_used,
        "movers": movers_payload,
        "mix_concentration": mix_concentration,
    }
