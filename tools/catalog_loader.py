from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from . import resolve_filter

# Mapping of resolve_filter.update_catalogs keyword args to dataframe columns.
_CATALOG_COLUMN_MAP: Dict[str, str] = {
    "products": "product",
    "regions": "region",
    "level1_classes": "level_1_classification",
    "level2_classes": "level_2_classification",
    "concern_types": "concern_type",
    "cities": "city",
    "product_categories": "product_category",
}


def _collect_unique(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Extract unique, non-null values from a dataframe column while
    preserving the original order.
    """
    if column not in df.columns:
        return []

    series = df[column].dropna()
    if series.empty:
        return []

    return series.drop_duplicates().tolist()


def sync_filter_catalogs(df: pd.DataFrame) -> None:
    """
    Push the latest catalog values from the ticket dataframe into the
    resolve_filter module so fuzzy matching works against the full universe.
    """
    if df is None or df.empty:
        resolve_filter.update_catalogs(
            products=[],
            regions=[],
            level1_classes=[],
            level2_classes=[],
            concern_types=[],
            cities=[],
            product_categories=[],
        )
        return

    payload = {
        key: _collect_unique(df, column)
        for key, column in _CATALOG_COLUMN_MAP.items()
    }
    resolve_filter.update_catalogs(**payload)
