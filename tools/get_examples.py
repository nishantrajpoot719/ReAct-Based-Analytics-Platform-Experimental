from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from . import resolve_filter
from . import ticket_summariser
from . import trend_breakdown
from . import vector_search

_ID_COLUMN = "complaint_number"
_DATE_COLUMN = "created_date"
_DEFAULT_TEXT_COLUMN = "expanded_description"
_SUMMARY_LIMIT = 100

_DEFAULT_FIELDS = [
    "expanded_description",
    "customer_issue",
    "root_cause",
    "resolution_provided_summary",
]

def _resolve_logger() -> logging.Logger:
    agent_logger = logging.getLogger("agent_steps")
    if agent_logger.handlers:
        return agent_logger
    return logging.getLogger(__name__)


_LOGGER = _resolve_logger()
_TRACE_ENABLED = os.getenv("SEMANTIC_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def _trace(level: int, message: str, *args) -> None:
    if _TRACE_ENABLED:
        _LOGGER.log(level, message, *args)



def _truncate(value: Any, limit: int = 300) -> Any:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _normalize_queries(values: Union[Sequence[str], str, None]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    cleaned: List[str] = []
    for raw in values:
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _format_created_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text[:10] if text else None
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return str(value)[:10]
    if value == value:  # NaN guard
        text = str(value).strip()
        return text[:10] if text else None
    return None


def _prepare_fields(
    fields: Optional[Sequence[str]],
    text_column: Optional[str],
) -> List[str]:
    ordered: List[str] = []
    for candidate in (fields or _DEFAULT_FIELDS):
        if candidate and candidate not in ordered:
            ordered.append(candidate)
    if text_column and text_column not in ordered:
        ordered.append(text_column)
    return ordered


def _safe_row_value(row: Dict[str, Any], field: str) -> Any:
    if field not in row:
        return None
    value = row.get(field)
    if value is None or value != value:
        return None
    return value


def _build_example(
    row: Dict[str, Any],
    fields: Sequence[str],
    match_score: Optional[float],
) -> Dict[str, Any]:
    example: Dict[str, Any] = {
        "ticket_id": row.get(_ID_COLUMN),
        "created_date": _format_created_date(row.get(_DATE_COLUMN)),
        "city": row.get("city"),
    }
    if match_score is not None:
        example["match_score"] = round(float(match_score), 4)

    for field in fields:
        value = _safe_row_value(row, field)
        if value is not None:
            example[field] = _truncate(value)
    return example


def _summarise_texts(texts: Sequence[str]) -> Optional[str]:
    if not texts:
        return None
    try:
        _trace(logging.INFO, "get_examples summary_start count=%s", len(texts))
        return ticket_summariser.summarise_texts(texts)
    except ValueError:
        _trace(logging.DEBUG, "get_examples summary_skipped_empty")
        return None
    except Exception as exc:
        _trace(logging.WARNING, "get_examples summary_error error=%s", exc)
        return f"Summary unavailable: {exc}"
    finally:
        if texts:
            _trace(logging.DEBUG, "get_examples summary_texts_sample=%s", _truncate(texts[0], 120))


def get_examples(
    *,
    filter_id: str,
    category_hint: Optional[str] = None,
    max_examples: int = 20,
    fields: Optional[Sequence[str]] = None,
    text_column: Optional[str] = None,
    text_query: Optional[Union[Sequence[str], str]] = None,
    theme_hint: Optional[Union[Sequence[str], str]] = None,
    top_k: Optional[int] = None,
    include_summary: bool = True,
) -> Dict[str, Any]:
    """
    Retrieve representative complaint examples for a saved filter.

    Args:
        filter_id: Identifier produced by ``resolve_filter.create_filter_record``.
        category_hint: Optional level-2 classification used to narrow examples.
        max_examples: Maximum number of examples to return.
        fields: Optional iterable of column names to include in each example payload.
        text_column: Column containing free-text content for semantic search and summaries.
        text_query: Optional text or list of texts used for semantic recall.
        theme_hint: Optional thematic keywords combined with ``text_query`` for semantic search.
        top_k: Maximum number of semantic matches to fetch (defaults to ``max_examples`` when omitted).
        include_summary: When ``True``, attempt to summarise the selected examples.

    Returns:
        On success, a dictionary containing ``examples``, ``matched_ticket_ids``, ``summary``, ``query_info``, and
        ``errors``. On failure, returns a dictionary with ``error``, ``message``, and optional ``hint``.
    """
    if not filter_id or not str(filter_id).strip():
        return {
            "error": "INVALID_FILTER_ID",
            "message": "filter_id must be a non-empty string.",
            "hint": "Provide the identifier returned by create_filter_record or list_filter_ids.",
        }
    filter_id = str(filter_id).strip()

    if not isinstance(max_examples, int) or max_examples <= 0:
        return {
            "error": "INVALID_MAX_EXAMPLES",
            "message": "max_examples must be a positive integer.",
            "hint": "Choose how many examples to return (e.g., 3).",
        }

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        return {
            "error": "INVALID_TOP_K",
            "message": "top_k must be a positive integer when provided.",
            "hint": "Set top_k to a whole number greater than zero or omit the parameter.",
        }
    effective_top_k = top_k if top_k is not None else max_examples

    cleaned_fields: Optional[List[str]] = None
    if fields is not None:
        try:
            cleaned_fields = []
            for candidate in fields:
                text = str(candidate).strip()
                if text:
                    cleaned_fields.append(text)
        except TypeError:
            return {
                "error": "INVALID_FIELDS",
                "message": "fields must be an iterable of column names.",
                "hint": "Provide fields like ['expanded_description', 'root_cause'] or omit the parameter.",
            }

    if text_column is not None:
        if not isinstance(text_column, str):
            return {
                "error": "INVALID_TEXT_COLUMN",
                "message": "text_column must be a string when provided.",
                "hint": "Pass the name of the dataframe column containing ticket text.",
            }
        text_column = text_column.strip()
        if not text_column:
            return {
                "error": "INVALID_TEXT_COLUMN",
                "message": "text_column must be a non-empty string.",
                "hint": "Pass the name of the dataframe column containing ticket text.",
            }
    text_column = text_column or _DEFAULT_TEXT_COLUMN

    if category_hint is not None:
        category_hint = str(category_hint).strip() or None

    _trace(
        logging.INFO,
        "get_examples called filter_id=%s max_examples=%s text_column=%s",
        filter_id,
        max_examples,
        text_column,
    )
    try:
        filters = resolve_filter.get_filter_payload(filter_id)
    except KeyError:
        _trace(logging.WARNING, "get_examples filter_missing filter_id=%s", filter_id)
        return {
            "error": "FILTER_NOT_FOUND",
            "message": f"Filter '{filter_id}' was not found in the filter store.",
            "hint": "Create the filter first or ensure the identifier is still active.",
        }

    df = trend_breakdown.get_registered_dataframe()
    if df is None or df.empty:
        _trace(logging.INFO, "get_examples dataframe_empty filter_id=%s", filter_id)
        return {
            "error": "NO_DATAFRAME",
            "message": "No ticket dataframe registered or dataframe is empty.",
            "hint": "Call trend_breakdown.register_ticket_dataframe before fetching examples.",
        }

    try:
        filtered = trend_breakdown.apply_filter_to_df(df, filters)
    except ValueError as exc:
        _trace(logging.WARNING, "get_examples filter_apply_failed filter_id=%s error=%s", filter_id, exc)
        return {
            "error": "FILTER_APPLY_FAILED",
            "message": str(exc),
            "hint": "Ensure the dataframe includes the columns referenced by the filter.",
        }
    if filtered.empty:
        _trace(logging.INFO, "get_examples no_rows_after_filter filter_id=%s", filter_id)
        return {
            "examples": [],
            "matched_ticket_ids": [],
            "summary": None,
            "query_info": {
                "mode": "structured",
                "text_column": text_column,
                "top_k": effective_top_k,
            },
            "errors": [],
        }

    if category_hint and "level_2_classification" in filtered.columns:
        filtered = filtered[filtered["level_2_classification"] == category_hint]
        if filtered.empty:
            _trace(
                logging.INFO,
                "get_examples no_rows_after_category filter_id=%s category_hint=%s",
                filter_id,
                category_hint,
            )
            return {
                "examples": [],
                "matched_ticket_ids": [],
                "summary": None,
                "query_info": {
                    "mode": "structured",
                    "text_column": text_column,
                    "top_k": effective_top_k,
                },
                "errors": [],
            }

    if text_column not in filtered.columns:
        available_cols = list(filtered.columns)[:10]
        _trace(
            logging.WARNING,
            "get_examples missing_column column=%s available=%s",
            text_column,
            available_cols,
        )
    fields_to_include = _prepare_fields(cleaned_fields, text_column)

    filtered = filtered.copy()
    if _ID_COLUMN in filtered.columns:
        filtered[_ID_COLUMN] = filtered[_ID_COLUMN].astype(str)

    if _ID_COLUMN in filtered.columns:
        rows_by_id = (
            filtered.drop_duplicates(subset=[_ID_COLUMN])
            .set_index(_ID_COLUMN, drop=False)
            .to_dict(orient="index")
        )
    else:
        rows_by_id = {}

    text_queries = _normalize_queries(text_query)
    theme_queries = _normalize_queries(theme_hint)
    queries = text_queries + theme_queries
    use_semantic = bool(queries) and text_column in filtered.columns
    _trace(
        logging.INFO,
        "get_examples mode_select filter_id=%s mode=%s text_query_count=%s theme_hint_count=%s top_k=%s",
        filter_id,
        "semantic" if use_semantic else "structured",
        len(text_queries),
        len(theme_queries),
        effective_top_k,
    )

    examples: List[Dict[str, Any]] = []
    matched_ids: List[str] = []
    seen_ids: set[str] = set()
    summary_texts: List[str] = []
    errors: List[str] = []

    if use_semantic:
        allowed_ids: Iterable[str] = rows_by_id.keys()
        try:
            matches = vector_search.semantic_search(
                column=text_column,
                query_texts=queries,
                top_k=effective_top_k,
                allowed_ticket_ids=allowed_ids,
            )
            semantic_error = None
        except vector_search.VectorSearchError as exc:
            matches = []
            semantic_error = f"Semantic search unavailable: {exc}"
            _trace(logging.WARNING, "get_examples semantic_error filter_id=%s error=%s", filter_id, exc)

        for match in matches:
            row = rows_by_id.get(match.ticket_id)
            if not row:
                continue
            example = _build_example(row, fields_to_include, match.score)
            examples.append(example)
            if match.ticket_id not in seen_ids:
                matched_ids.append(match.ticket_id)
                seen_ids.add(match.ticket_id)
            text_value = row.get(text_column)
            if text_value and len(summary_texts) < _SUMMARY_LIMIT:
                summary_texts.append(str(text_value))

        if semantic_error:
            errors.append(semantic_error)
            _trace(logging.WARNING, "get_examples semantic_error_recorded filter_id=%s", filter_id)

    if not examples:
        _trace(
            logging.INFO,
            "get_examples structured_fallback filter_id=%s reason=%s",
            filter_id,
            "no_semantic_matches" if use_semantic else "semantic_disabled",
        )
        ordered = filtered.sort_values(_DATE_COLUMN, ascending=False, na_position="last")
        sample = ordered.head(max_examples)
        for _, row in sample.iterrows():
            row_dict = row.to_dict()
            ticket_id = row_dict.get(_ID_COLUMN)
            if ticket_id:
                ticket_id_str = str(ticket_id)
                if ticket_id_str not in seen_ids:
                    matched_ids.append(ticket_id_str)
                    seen_ids.add(ticket_id_str)
            example = _build_example(row_dict, fields_to_include, None)
            examples.append(example)
            text_value = row_dict.get(text_column)
            if text_value and len(summary_texts) < _SUMMARY_LIMIT:
                summary_texts.append(str(text_value))

    summary = _summarise_texts(summary_texts) if include_summary else None
    mode = "semantic" if use_semantic and queries else "structured"
    _trace(
        logging.INFO,
        "get_examples completed filter_id=%s examples=%s matched_ids=%s summary_generated=%s errors=%s",
        filter_id,
        len(examples),
        len(matched_ids),
        bool(summary),
        bool(errors),
    )

    return {
        "examples": examples,
        "matched_ticket_ids": matched_ids,
        "summary": summary,
        "query_info": {
            "mode": mode,
            "text_column": text_column,
            "top_k": effective_top_k,
        },
        "errors": errors,
    }


__all__ = ["get_examples"]
