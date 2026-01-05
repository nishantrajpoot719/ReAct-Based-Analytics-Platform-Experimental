from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:  # pragma: no cover - streamlit optional in tests
    import streamlit as st
except ImportError:  # pragma: no cover
    class _SessionState(dict):
        def __getattr__(self, name: str) -> Any:
            if name not in self:
                value: Any = [] if name == "filters_created" else None
                self[name] = value
            return self[name]

        def __setattr__(self, name: str, value: Any) -> None:
            self[name] = value

    class _StreamlitStub:
        def __init__(self) -> None:
            self.session_state = _SessionState()
            if "filters_created" not in self.session_state:
                self.session_state["filters_created"] = []

        def popover(self, *args: Any, **kwargs: Any) -> None:
            return None

    st = _StreamlitStub()

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

ALL_PRODUCTS: List[str] = []
ALL_REGIONS: List[str] = []
ALL_L1_CLASSES: List[str] = []
ALL_L2_CLASSES: List[str] = []
ALL_CONCERN_TYPES: List[str] = []
ALL_CITIES: List[str] = []
ALL_PRODUCT_CATEGORIES: List[str] = []

FILTER_STORE: Dict[str, Dict[str, Any]] = {}
FILTER_META: Dict[str, Dict[str, Any]] = {}
FILTER_CATALOG: List[Dict[str, Any]] = []


@dataclass(frozen=True)
class CatalogEntry:
    canonical: str
    norm: str
    tokens: Set[str]


INDEX_PRODUCTS: List[CatalogEntry] = []
INDEX_REGIONS: List[CatalogEntry] = []
INDEX_L1: List[CatalogEntry] = []
INDEX_L2: List[CatalogEntry] = []
INDEX_CONCERNS: List[CatalogEntry] = []
INDEX_CITIES: List[CatalogEntry] = []
INDEX_PRODUCT_CATEGORIES: List[CatalogEntry] = []


ALIASES: Dict[str, Sequence[str]] = {
    "apple": ("apple", "apples", "gala apple", "washington apple", "kinnaur apple"),
    "milk": ("milk", "cow milk", "buffalo milk", "toned milk"),
    "atta": ("atta", "whole wheat atta", "chakki atta", "lokwan atta"),
    "ncr": ("ncr", "delhi ncr", "gurgaon", "gurugram", "noida"),
    "bangalore": ("bangalore", "bengaluru", "blr"),
    "not fresh": ("not fresh", "expired", "rotten", "stale", "sour"),
    "damaged": ("damaged", "broken", "torn", "leaking", "leaked"),
    "complaint": ("complaint", "complaints"),
    "query": ("query", "queries", "question"),
    "request": ("request", "requests"),
    "feedback": ("feedback", "suggestion", "review"),
}


class _FilterBaseModel(BaseModel):
    """Common sanitisation for filter payload fields."""

    region: Optional[List[str]] = None
    l2_class: Optional[List[str]] = None
    l1_class: Optional[List[str]] = None
    concern_type: Optional[List[str]] = None
    city: Optional[List[str]] = None
    product_category: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    metric: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @staticmethod
    def _coerce_string(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _coerce_string_list(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            coerced = cls._coerce_string(value)
            return [coerced] if coerced else []
        if isinstance(value, dict):
            iterable = value.values()
        else:
            iterable = value
        try:
            iterator = iter(iterable)
        except TypeError:
            coerced = cls._coerce_string(value)
            return [coerced] if coerced else []
        values: List[str] = []
        for item in iterator:
            coerced = cls._coerce_string(item)
            if coerced:
                values.append(coerced)
        seen: Set[str] = set()
        deduped: List[str] = []
        for text in values:
            if text not in seen:
                seen.add(text)
                deduped.append(text)
        return deduped

    @field_validator("region", "l2_class", "l1_class", "concern_type", "city", "product_category", mode="before")
    @classmethod
    def _validate_optional_string_list(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        return cls._coerce_string_list(value)

    @field_validator("date_from", "date_to", "metric", mode="before")
    @classmethod
    def _validate_optional_string(cls, value: Any) -> Optional[str]:
        return cls._coerce_string(value)


class FilterPayload(_FilterBaseModel):
    region: List[str] = Field(default_factory=list)
    l2_class: List[str] = Field(default_factory=list)
    l1_class: List[str] = Field(default_factory=list)
    concern_type: List[str] = Field(default_factory=list)
    city: List[str] = Field(default_factory=list)
    product_category: List[str] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)

    @field_validator("products", mode="before")
    @classmethod
    def _validate_products(cls, value: Any) -> List[str]:
        if value is None:
            return []
        return cls._coerce_string_list(value)


class FilterUpdatePayload(_FilterBaseModel):
    products: Optional[List[str]] = None

    @field_validator("products", mode="before")
    @classmethod
    def _validate_products(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        return cls._coerce_string_list(value)


def _normalize(text: str) -> str:
    cleaned = (
        text.lower()
        .replace("%", " percent ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
    )
    return " ".join(cleaned.split())


def _dedupe(values: Optional[Iterable[Any]]) -> List[str]:
    if not values:
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, float) and raw != raw:  # NaN check
            continue
        text = str(raw).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _build_index(values: Sequence[str]) -> List[CatalogEntry]:
    return [
        CatalogEntry(
            canonical=value,
            norm=_normalize(value),
            tokens=set(_normalize(value).split()),
        )
        for value in values
    ]


def _rebuild_indexes() -> None:
    global INDEX_PRODUCTS, INDEX_REGIONS, INDEX_L1, INDEX_L2, INDEX_CONCERNS, INDEX_CITIES, INDEX_PRODUCT_CATEGORIES
    INDEX_PRODUCTS = _build_index(ALL_PRODUCTS)
    INDEX_REGIONS = _build_index(ALL_REGIONS)
    INDEX_L1 = _build_index(ALL_L1_CLASSES)
    INDEX_L2 = _build_index(ALL_L2_CLASSES)
    INDEX_CONCERNS = _build_index(ALL_CONCERN_TYPES)
    INDEX_CITIES = _build_index(ALL_CITIES)
    INDEX_PRODUCT_CATEGORIES = _build_index(ALL_PRODUCT_CATEGORIES)
    logger.debug("Catalog indexes rebuilt", extra={"products": len(INDEX_PRODUCTS)})


def reset_state() -> None:
    ALL_PRODUCTS.clear()
    ALL_REGIONS.clear()
    ALL_L1_CLASSES.clear()
    ALL_L2_CLASSES.clear()
    ALL_CONCERN_TYPES.clear()
    ALL_CITIES.clear()
    ALL_PRODUCT_CATEGORIES.clear()
    FILTER_STORE.clear()
    FILTER_META.clear()
    FILTER_CATALOG.clear()
    _rebuild_indexes()
    logger.debug("resolve_filter state reset")


def update_catalogs(
    *,
    products: Optional[Iterable[Any]] = None,
    regions: Optional[Iterable[Any]] = None,
    level1_classes: Optional[Iterable[Any]] = None,
    level2_classes: Optional[Iterable[Any]] = None,
    concern_types: Optional[Iterable[Any]] = None,
    cities: Optional[Iterable[Any]] = None,
    product_categories: Optional[Iterable[Any]] = None,
) -> None:
    if products is not None:
        ALL_PRODUCTS[:] = _dedupe(products)
    if regions is not None:
        ALL_REGIONS[:] = _dedupe(regions)
    if level1_classes is not None:
        ALL_L1_CLASSES[:] = _dedupe(level1_classes)
    if level2_classes is not None:
        ALL_L2_CLASSES[:] = _dedupe(level2_classes)
    if concern_types is not None:
        ALL_CONCERN_TYPES[:] = _dedupe(concern_types)
    if cities is not None:
        ALL_CITIES[:] = _dedupe(cities)
    if product_categories is not None:
        ALL_PRODUCT_CATEGORIES[:] = _dedupe(product_categories)

    _rebuild_indexes()


def _alias_variants(hint: str) -> Set[str]:
    variants: Set[str] = set()
    if not hint:
        return variants
    variants.add(hint)
    if hint.endswith("ies") and len(hint) > 3:
        variants.add(hint[:-3] + "y")
    if hint.endswith("es") and len(hint) > 2:
        variants.add(hint[:-2])
    if hint.endswith("s") and len(hint) > 1:
        variants.add(hint[:-1])
    for part in hint.split():
        variants.add(part)
    return variants


def _alias_list(hint: str) -> List[str]:
    if not hint:
        return []
    norm_hint = _normalize(hint)
    aliases = list(ALIASES.get(norm_hint, ())) or list(ALIASES.get(hint.lower(), ()))
    variants = _alias_variants(norm_hint)
    return list(dict.fromkeys([*aliases, *variants, norm_hint]))


def _score_candidate(hint_norm: str, candidate: CatalogEntry, alias_list: Sequence[str]) -> float:
    if not hint_norm:
        return 0.0
    user_tokens = set(hint_norm.split())
    overlap = len(user_tokens & candidate.tokens)
    substring_hit = any(alias in candidate.norm for alias in alias_list)
    direct_hit = 1 if hint_norm in candidate.norm else 0
    return overlap * 2 + (3 if substring_hit else 0) + (4 if direct_hit else 0)


def _rank_candidates(index: Sequence[CatalogEntry], hint: Optional[str]) -> List[Tuple[float, CatalogEntry]]:
    if not hint:
        return []
    hint_norm = _normalize(hint)
    alias_list = _alias_list(hint)
    scored = [
        (score, entry)
        for entry in index
        if (score := _score_candidate(hint_norm, entry, alias_list)) > 0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def _expand_hints(hint: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if hint is None:
        return []
    if isinstance(hint, str):
        if "," in hint or ";" in hint:
            segments = re.split(r"[;,]", hint)
        else:
            segments = [hint]
        values = [segment.strip() for segment in segments if segment and segment.strip()]
        if values:
            return values
        return [hint.strip()] if hint.strip() else []
    try:
        iterator = iter(hint)
    except TypeError:
        text = _FilterBaseModel._coerce_string(hint)
        return [text] if text else []
    expanded: List[str] = []
    for item in iterator:
        if isinstance(item, str):
            expanded.extend(_expand_hints(item))
        else:
            text = _FilterBaseModel._coerce_string(item)
            if text:
                expanded.append(text)
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in expanded:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _list_matches(
    index: Sequence[CatalogEntry],
    hint: Optional[Union[str, Sequence[str]]],
    *,
    candidate_limit: int = 5,
    keep_limit: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    hints = _expand_hints(hint)
    if not hints:
        return [], []
    keep: List[str] = []
    candidate_scores: Dict[str, float] = {}
    for single_hint in hints:
        scored = _rank_candidates(index, single_hint)
        if not scored:
            continue
        top_score = scored[0][0]
        threshold = max(1.0, top_score - 1.0)
        for score, entry in scored:
            if score < threshold:
                break
            if entry.canonical not in keep:
                keep.append(entry.canonical)
        for score, entry in scored[:candidate_limit]:
            existing = candidate_scores.get(entry.canonical)
            if existing is None or score > existing:
                candidate_scores[entry.canonical] = score
    if keep_limit is not None:
        keep = keep[:keep_limit]
    candidates = [
        {"value": value, "score": candidate_scores[value]}
        for value in sorted(candidate_scores, key=candidate_scores.get, reverse=True)[:candidate_limit]
    ]
    return keep, candidates


def product_matches(product_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    hints = _expand_hints(product_hint)
    if (
        isinstance(product_hint, str)
        and len(hints) <= 1
        and re.search(r"\band\b", product_hint, flags=re.IGNORECASE)
    ):
        segments = [
            segment.strip()
            for segment in re.split(r"\band\b", product_hint, flags=re.IGNORECASE)
            if segment and segment.strip()
        ]
        if segments:
            hints = segments
    target = hints if hints else product_hint
    return _list_matches(INDEX_PRODUCTS, target, candidate_limit=10, keep_limit=10)


def best_region_match(region_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_REGIONS, region_hint, candidate_limit=5, keep_limit=5)


def best_issue_match_l2(issue_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_L2, issue_hint, candidate_limit=5, keep_limit=5)


def best_level1_match(issue_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_L1, issue_hint, candidate_limit=5, keep_limit=5)


def best_concern_type_match(concern_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_CONCERNS, concern_hint, candidate_limit=5, keep_limit=5)


def best_city_match(city_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_CITIES, city_hint, candidate_limit=5, keep_limit=5)


def best_product_category_match(category_hint: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    return _list_matches(INDEX_PRODUCT_CATEGORIES, category_hint, candidate_limit=5, keep_limit=5)


def resolve_time_range(time_range: Optional[Dict[str, Any]], now: datetime) -> Dict[str, str]:
    if not time_range:
        time_range = {}
    start = time_range.get("from") or time_range.get("date_from")
    end = time_range.get("to") or time_range.get("date_to")
    if start and end:
        return {"date_from": str(start), "date_to": str(end)}
    if "days" in time_range and time_range["days"] is not None:
        days = int(time_range["days"])
        end_date = now.date()
        start_date = (now - timedelta(days=days)).date()
        return {"date_from": start_date.isoformat(), "date_to": end_date.isoformat()}
    if start or end:
        return {
            "date_from": str(start) if start else (now - timedelta(days=7)).date().isoformat(),
            "date_to": str(end) if end else now.date().isoformat(),
        }
    end_date = now.date()
    start_date = (now - timedelta(days=7)).date()
    return {"date_from": start_date.isoformat(), "date_to": end_date.isoformat()}


def _normalise_metric(metric: Optional[str]) -> str:
    if metric in {"ticket_count", "refund_count_15d"}:
        return metric
    return "ticket_count"


def resolve_filters(
    *,
    product_hint: Optional[List[str]] = [],
    region_hint: Optional[List[str]] = [],
    level_2_classification_hint: Optional[List[str]] = [],
    city_hint: Optional[List[str]] = [],
    level_1_classification_hint: Optional[List[str]] = [],
    concern_hint: Optional[List[str]] = [],
    product_category_hint: Optional[List[str]] = [],
    time_range: Optional[Dict[str, Any]] = None,
    focus_metric: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Build a canonical filter payload for downstream analytics tools.

    Args:
        product_hint: Free-form text describing one or more products. Comma-separated
            (\"apple, banana\") or \"and\" lists are accepted.
        region_hint: Region or cluster hint (e.g., \"NCR\", \"South\").
        level_2_classification_hint: Detailed issue description mapped against the
            level-2 classification catalog (e.g., \"not fresh\", \"missing items\").
        city_hint: City name hint.
        level_1_classification_hint: High-level classification hint (e.g., \"Product / Quality\").
        concern_hint: Ticket type hint (\"complaint\", \"request\", etc.).
        product_category_hint: Product category hint (\"Fruits\", \"Dairy\").
        time_range: Dict describing the window. Supports
            {\"from\": \"YYYY-MM-DD\", \"to\": \"YYYY-MM-DD\"},
            {\"date_from\": ..., \"date_to\": ...}, or {\"days\": 7}.
        focus_metric: Preferred metric identifier (\"ticket_count\" or \"refund_count_15d\").
        now: Override timestamp for deterministic testing; defaults to current UTC time.

    Returns:
        Dict with keys:
            - ``filter_id``: Persisted ID for the resolved filter.
            - ``filters``: Canonical payload containing list-based selections, date range,
              and metric.
            - ``meta``: Metadata with the original hints, candidate matches, and timestamps.
    """
        
    now = now or datetime.now(UTC)
    products, product_candidates = [], []
    regions, region_candidates = [], []
    l2_classes, l2_candidates = [], []
    l1_classes, l1_candidates = [], []
    concern_types, concern_candidates = [], []
    cities, city_candidates = [], []
    product_categories, category_candidates = [], []
    for i in product_hint:
        logger.info(f"Product hint: {i}")
        products_temp, product_candidates_temp = product_matches(i)
        products.extend(products_temp)
        product_candidates.extend(product_candidates_temp)
        logger.info(f"Resolved products: {products}, candidates: {product_candidates}")
    for i in region_hint:
        regions_temp, region_candidates_temp = best_region_match(i)
        regions.extend(regions_temp)
        region_candidates.extend(region_candidates_temp)
    for i in level_2_classification_hint:
        l2_classes_temp, l2_candidates_temp = best_issue_match_l2(i)
        l2_classes.extend(l2_classes_temp)
        l2_candidates.extend(l2_candidates_temp)
    for i in level_1_classification_hint:
        l1_classes_temp, l1_candidates_temp = best_level1_match(i)
        l1_classes.extend(l1_classes_temp)
        l1_candidates.extend(l1_candidates_temp)
    for i in concern_hint:
        concern_types_temp, concern_candidates_temp = best_concern_type_match(i)
        concern_types.extend(concern_types_temp)
        concern_candidates.extend(concern_candidates_temp)
    for i in city_hint:
        cities_temp, city_candidates_temp = best_city_match(i)
        cities.extend(cities_temp)
        city_candidates.extend(city_candidates_temp)
    for i in product_category_hint:
        product_categories_temp, category_candidates_temp = best_product_category_match(i)
        product_categories.extend(product_categories_temp)
        category_candidates.extend(category_candidates_temp)
    """
    products, product_candidates = product_matches(product_hint)
    regions, region_candidates = best_region_match(region_hint)
    l2_classes, l2_candidates = best_issue_match_l2(level_2_classification_hint)
    l1_classes, l1_candidates = best_level1_match(level_1_classification_hint)
    concern_types, concern_candidates = best_concern_type_match(concern_hint)
    cities, city_candidates = best_city_match(city_hint)
    product_categories, category_candidates = best_product_category_match(product_category_hint)
    """
    resolved_time = resolve_time_range(time_range, now)
    metric = _normalise_metric(focus_metric)

    filters = {
        "products": list(products),
        "region": list(regions),
        "l2_class": list(l2_classes),
        "l1_class": list(l1_classes),
        "concern_type": list(concern_types),
        "city": list(cities),
        "product_category": list(product_categories),
        "date_from": resolved_time["date_from"],
        "date_to": resolved_time["date_to"],
        "metric": metric,
    }
    filter_id = str(uuid.uuid4())
    FILTER_STORE[filter_id] = dict(filters)
    now_iso = now.isoformat()
    meta = {
        "requested": {
            "product_hint": product_hint,
            "region_hint": region_hint,
            "level_2_classification_hint": level_2_classification_hint,
            "city_hint": city_hint,
            "level_1_classification_hint": level_1_classification_hint,
            "concern_hint": concern_hint,
            "product_category_hint": product_category_hint,
            "time_range": time_range or {},
            "focus_metric": focus_metric,
        },
        "product_candidates": product_candidates,
        "region_candidates": region_candidates,
        "issue_candidates": l2_candidates,
        "level1_candidates": l1_candidates,
        "concern_type_candidates": concern_candidates,
        "city_candidates": city_candidates,
        "product_category_candidates": category_candidates,
        "source": "resolver",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    FILTER_META[filter_id] = {
        "description": "",
        "tags": [],
        "source": "resolver",
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    logger.debug(
        "resolve_filters created filter",
        extra={"filter_id": filter_id, "filters": filters},
    )
    return {"filter_id": filter_id, "filters": dict(filters), "meta": meta}


def resolve_filters_backend(**kwargs: Any) -> Dict[str, Any]:
    return resolve_filters(**kwargs)


def list_filter_ids() -> List[str]:
    return list(FILTER_STORE.keys())


def get_filter_payload(filter_id: str) -> Dict[str, Any]:
    payload = FILTER_STORE.get(filter_id)
    if payload is None:
        raise KeyError(filter_id)
    return dict(payload)


def get_filter_meta(filter_id: str) -> Dict[str, Any]:
    return dict(FILTER_META.get(filter_id, {}))


def _upsert_catalog_entry(filter_id: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
    record = {
        "filter_id": filter_id,
        "filters": dict(payload),
        "description": meta.get("description", ""),
        "tags": list(meta.get("tags", [])),
        "source": meta.get("source", ""),
        "created_at": meta.get("created_at"),
        "updated_at": meta.get("updated_at"),
    }
    for entry in FILTER_CATALOG:
        if entry.get("filter_id") == filter_id:
            entry.update(record)
            return
    FILTER_CATALOG.append(record)


def _ensure_filter_payload(filter_payload: Union[FilterPayload, Dict[str, Any]]) -> FilterPayload:
    if isinstance(filter_payload, FilterPayload):
        return filter_payload
    if isinstance(filter_payload, dict):
        return FilterPayload(**filter_payload)
    raise TypeError("filter_payload must be a FilterPayload instance or dictionary.")


def _ensure_filter_update_payload(updates: Union[FilterUpdatePayload, Dict[str, Any]]) -> FilterUpdatePayload:
    if isinstance(updates, FilterUpdatePayload):
        return updates
    if isinstance(updates, dict):
        return FilterUpdatePayload(**updates)
    raise TypeError("updates must be a FilterUpdatePayload instance or dictionary.")


def create_filter_record(
    filter_payload: Union[FilterPayload, Dict[str, Any]],
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    source: str = "agent",
) -> Dict[str, Any]:
    """
    Create and register a new filter definition in the in-memory store.

    Args:
        filter_payload: Base filter payload describing products, regions, dates, etc.
        description: Optional human-readable description associated with the filter.
        tags: Optional collection of tags used to categorise the filter for later lookup.
        source: Optional string describing the origin of the filter (defaults to ``"agent"``).

    Returns:
        On success, a dictionary containing the new ``filter_id``, ``filters`` payload, and accompanying ``meta``.
        On error, returns a dictionary with ``error``, ``message``, and an optional ``hint`` to aid recovery.
    """
    if filter_payload is None:
        return {
            "error": "MISSING_FILTER_PAYLOAD",
            "message": "filter_payload is required to create a filter.",
            "hint": "Provide a dict or FilterPayload with keys such as 'products', 'date_from', and 'metric'.",
        }

    try:
        payload_model = _ensure_filter_payload(filter_payload)
    except TypeError as exc:
        return {
            "error": "INVALID_FILTER_PAYLOAD",
            "message": str(exc),
            "hint": "Pass a dict or FilterPayload populated with recognised filter fields.",
        }
    except ValidationError as exc:
        details = "; ".join(err.get("msg", "") for err in exc.errors()) or str(exc)
        return {
            "error": "INVALID_FILTER_PAYLOAD",
            "message": f"Invalid filter payload: {details}",
            "hint": "Ensure values are strings, lists of strings, or ISO date strings as required.",
        }

    payload = payload_model.model_dump()
    payload["metric"] = _normalise_metric(payload.get("metric"))

    processed_tags: List[str] = []
    if tags is not None:
        if isinstance(tags, str):
            cleaned = tags.strip()
            if cleaned:
                processed_tags = [cleaned]
        else:
            try:
                for tag in tags:
                    text = str(tag).strip()
                    if text:
                        processed_tags.append(text)
            except TypeError:
                return {
                    "error": "INVALID_TAGS",
                    "message": "tags must be an iterable of strings.",
                    "hint": "Provide tags as a list like ['quality', 'seasonal'] or omit the parameter.",
                }

    source_value = (source or "agent").strip() or "agent"
    description_value = str(description).strip() if description is not None else ""

    filter_id = str(uuid.uuid4())
    FILTER_STORE[filter_id] = payload
    filters_created = getattr(st.session_state, "filters_created", None)
    if filters_created is None:
        filters_created = []
        setattr(st.session_state, "filters_created", filters_created)
    filters_created.append(filter_id)
    st.popover(f"Filter {filter_id} created and added to session state.", type="success")

    now_iso = datetime.now(UTC).isoformat()
    meta = {
        "description": description_value,
        "tags": processed_tags,
        "source": source_value,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    FILTER_META[filter_id] = meta
    _upsert_catalog_entry(filter_id, payload, meta)

    return {"filter_id": filter_id, "filters": dict(payload), "meta": dict(meta)}


def update_filter_record(
    filter_id: str,
    updates: Optional[Union[FilterUpdatePayload, Dict[str, Any]]] = None,
    description: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing filter record and optional metadata in the store.

    Args:
        filter_id: Identifier of the filter returned by ``create_filter_record``.
        updates: Field values to merge into the filter payload (products, region, dates, etc.).
        description: Optional human readable description to persist with the filter.
        tags: Optional collection of tags to associate with the filter.
        source: Optional string indicating where the update originated (e.g., ``"agent"``).

    Returns:
        On success, a dictionary containing the updated ``filter_id``, ``filters`` payload, and ``meta``.
        On error, returns a dictionary with ``error``, ``message``, and an optional ``hint`` field describing how to recover.
    """
    if not filter_id or not str(filter_id).strip():
        return {
            "error": "INVALID_FILTER_ID",
            "message": "filter_id must be a non-empty string.",
            "hint": "Pass the identifier previously returned by create_filter_record or list_filter_ids.",
        }
    filter_id = str(filter_id).strip()

    existing_payload = FILTER_STORE.get(filter_id)
    if existing_payload is None:
        return {
            "error": "FILTER_NOT_FOUND",
            "message": f"Filter '{filter_id}' was not found in the store.",
            "hint": "Call list_filter_ids to inspect available filter identifiers before updating.",
        }

    metadata_supplied = any(value is not None for value in (description, tags, source))

    update_values: Dict[str, Any] = {}
    if updates is None:
        if not metadata_supplied:
            return {
                "error": "NO_UPDATES_PROVIDED",
                "message": "No updates supplied. Provide fields in 'updates' or metadata parameters to modify.",
                "hint": "Include keys like 'products', 'date_from', or set description/tags/source to update metadata.",
            }
    else:
        try:
            update_model = _ensure_filter_update_payload(updates)
        except TypeError as exc:
            return {
                "error": "INVALID_UPDATE_PAYLOAD",
                "message": str(exc),
                "hint": "Pass a dict or FilterUpdatePayload with fields such as 'products', 'city', or 'metric'.",
            }
        except ValidationError as exc:
            details = "; ".join(err.get("msg", "") for err in exc.errors()) or str(exc)
            return {
                "error": "INVALID_UPDATE_PAYLOAD",
                "message": f"Invalid updates payload: {details}",
                "hint": "Verify that update values are strings, lists of strings, or ISO dates as required.",
            }
        update_values = update_model.model_dump(exclude_unset=True)

    if not update_values and not metadata_supplied:
        return {
            "error": "NO_UPDATES_PROVIDED",
            "message": "No updates supplied. Provide fields in 'updates' or metadata parameters to modify.",
            "hint": "Include keys like 'products', 'date_from', or set description/tags/source to update metadata.",
        }

    payload = dict(existing_payload)
    for key, value in update_values.items():
        if value in (None, "", []):
            payload.pop(key, None)
        else:
            payload[key] = value
    payload["metric"] = _normalise_metric(payload.get("metric"))
    FILTER_STORE[filter_id] = payload

    now_iso = datetime.now(UTC).isoformat()
    meta = dict(FILTER_META.get(filter_id, {}))
    if description is not None:
        meta["description"] = description
    if tags is not None:
        meta["tags"] = list(tags)
    if source is not None:
        meta["source"] = source
    meta.setdefault("description", "")
    meta.setdefault("tags", [])
    meta.setdefault("source", "agent")
    meta.setdefault("created_at", now_iso)
    meta["updated_at"] = now_iso
    FILTER_META[filter_id] = meta
    _upsert_catalog_entry(filter_id, payload, meta)

    return {"filter_id": filter_id, "filters": dict(payload), "meta": dict(meta)}




__all__ = [
    "FilterPayload",
    "FilterUpdatePayload",
    "resolve_filters",
    "resolve_filters_backend",
    "product_matches",
    "best_region_match",
    "best_issue_match_l2",
    "best_level1_match",
    "best_concern_type_match",
    "best_city_match",
    "best_product_category_match",
    "resolve_time_range",
    "update_catalogs",
    "reset_state",
    "create_filter_record",
    "update_filter_record",
    "list_filter_ids",
    "get_filter_payload",
    "get_filter_meta",
]
