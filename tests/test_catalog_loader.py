import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from tools import catalog_loader, resolve_filter


def test_sync_filter_catalogs_populates_resolver():
    resolve_filter.reset_state()

    df = pd.DataFrame(
        {
            "product": ["Gala Apple - 4 pcs", "Washington Apple - 1 kg"],
            "region": ["NCR", "Mumbai"],
            "level_1_classification": ["Quality", "Delivery"],
            "level_2_classification": [
                "Product Not Fresh / Expired / Rotten / Infested",
                "Partial / Missing Items at Delivery",
            ],
            "concern_type": ["Complaint", "Query"],
            "city": ["Delhi", "Mumbai"],
            "product_category": ["Fruits", "Fruits"],
        }
    )

    catalog_loader.sync_filter_catalogs(df)

    assert "Gala Apple - 4 pcs" in resolve_filter.ALL_PRODUCTS
    assert "Washington Apple - 1 kg" in resolve_filter.ALL_PRODUCTS
    assert "NCR" in resolve_filter.ALL_REGIONS
    assert "Product Not Fresh / Expired / Rotten / Infested" in resolve_filter.ALL_L2_CLASSES
    assert "Complaint" in resolve_filter.ALL_CONCERN_TYPES
    assert "Delhi" in resolve_filter.ALL_CITIES
    assert "Fruits" in resolve_filter.ALL_PRODUCT_CATEGORIES


def test_sync_filter_catalogs_feeds_product_matches():
    resolve_filter.reset_state()

    df = pd.DataFrame(
        {
            "product": ["Buffalo Milk 1L", "Cow Milk 1L"],
            "concern_type": ["Complaint", "Complaint"],
        }
    )

    catalog_loader.sync_filter_catalogs(df)

    matches, _ = resolve_filter.product_matches("milk")
    assert matches, "Expected product matcher to use catalogued products"
    assert any("Milk" in prod for prod in matches)
