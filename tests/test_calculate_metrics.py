import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import calculate_metrics


def test_metrics_wow_change():
    metrics = calculate_metrics.calculate_metrics_backend(
        current_window={
            "date_from": "2025-10-01",
            "date_to": "2025-10-07",
            "total_tickets": 120,
            "by_category": [],
        },
        previous_window={
            "date_from": "2025-09-24",
            "date_to": "2025-09-30",
            "total_tickets": 100,
            "by_category": [],
        },
    )

    assert metrics["overall_change"]["wow_change_pct"] == 20.0


def test_metrics_share_and_refund_rate():
    metrics = calculate_metrics.calculate_metrics_backend(
        current_window={
            "date_from": "2025-10-01",
            "date_to": "2025-10-07",
            "total_tickets": 120,
            "by_category": [
                {"category": "Product Not Fresh", "tickets": 62, "refund_count_15d": 41},
                {"category": "Damaged Packaging", "tickets": 20, "refund_count_15d": 5},
            ],
        },
        previous_window={
            "date_from": "2025-09-24",
            "date_to": "2025-09-30",
            "total_tickets": 100,
            "by_category": [
                {"category": "Product Not Fresh", "tickets": 60, "refund_count_15d": 30},
            ],
        },
        top_n=5,
    )

    insights = {row["category"]: row for row in metrics["category_insights"]}
    not_fresh = insights["Product Not Fresh"]

    assert round(not_fresh["share_of_total_pct"], 2) == round(62 / 120 * 100, 2)
    assert round(not_fresh["refund_rate_pct"], 2) == round(41 / 62 * 100, 2)
    assert metrics["top_categories_by_share"][0]["category"] == "Product Not Fresh"
