from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd


def tickets_df_small() -> pd.DataFrame:
    base = datetime(2025, 10, 10)
    rows: List[Dict[str, object]] = []
    products = ["Gala Apple - 4 pcs", "Washington Apple - 1 kg", "Buffalo Milk - 1L"]
    regions = ["NCR", "Mumbai"]
    l2_classes = [
        "Product Not Fresh / Expired / Rotten / Infested",
        "Partial / Missing Items at Delivery",
        "Damaged Packaging",
    ]
    for offset in range(12):
        rows.append(
            {
                "complaint_number": f"C-{1000 + offset}",
                "city": "Delhi" if offset % 2 == 0 else "Mumbai",
                "region": regions[offset % len(regions)],
                "created_date": base - timedelta(days=offset),
                "refund_count_in_15_days": float(offset % 4),
                "product": products[offset % len(products)],
                "concern_type": "Complaint",
                "level_1_classification": "Product / Quality",
                "level_2_classification": l2_classes[offset % len(l2_classes)],
                "expanded_description": f"Expanded description {offset}",
                "customer_issue": f"Customer issue {offset}",
                "root_cause": f"Root cause {offset}",
                "resolution_provided_summary": f"Resolution summary {offset}",
                "product_category": "Fruits" if offset % 2 == 0 else "Dairy",
            }
        )
    return pd.DataFrame(rows)


def catalogs() -> Dict[str, List[str]]:
    return {
        "products": [
            "Gala Apple - 4 pcs",
            "Washington Apple - 1 kg",
            "Buffalo Milk - 1L",
        ],
        "regions": ["NCR", "Mumbai"],
        "level1_classes": ["Product / Quality", "Delivery & Fulfillment"],
        "level2_classes": [
            "Product Not Fresh / Expired / Rotten / Infested",
            "Partial / Missing Items at Delivery",
            "Damaged Packaging",
        ],
        "concern_types": ["Complaint", "Query"],
        "cities": ["Delhi", "Mumbai"],
        "product_categories": ["Fruits", "Dairy"],
    }
