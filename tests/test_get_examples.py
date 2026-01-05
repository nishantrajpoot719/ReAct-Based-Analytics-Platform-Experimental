from __future__ import annotations

import logging

import pytest

from tests import fixtures
from tools import get_examples, resolve_filter, ticket_summariser, trend_breakdown, vector_search


@pytest.fixture
def filter_setup(monkeypatch):
    resolve_filter.reset_state()
    df = fixtures.tickets_df_small()
    trend_breakdown.register_ticket_dataframe(df)
    payload = resolve_filter.FilterPayload()
    record = resolve_filter.create_filter_record(payload, description="test", tags=[], source="pytest")
    monkeypatch.setattr(ticket_summariser, "summarise_texts", lambda texts, max_items=100: "stub-summary")
    return record["filter_id"], df


def test_get_examples_semantic_intersection(monkeypatch, filter_setup):
    filter_id, _ = filter_setup

    calls = {}

    def fake_semantic_search(column, query_texts, top_k, allowed_ticket_ids, **kwargs):
        calls["column"] = column
        calls["query_texts"] = list(query_texts)
        calls["top_k"] = top_k
        calls["allowed_ids"] = set(str(v) for v in allowed_ticket_ids)
        return [
            vector_search.VectorMatch(ticket_id="C-1001", score=0.91, document="doc", metadata={}),
            vector_search.VectorMatch(ticket_id="C-9999", score=0.88, document="doc", metadata={}),
        ]

    monkeypatch.setattr(vector_search, "semantic_search", fake_semantic_search)

    result = get_examples.get_examples(
        filter_id=filter_id,
        text_query="fresh apples",
        text_column="expanded_description",
        top_k=2,
        include_summary=True,
    )

    assert result["query_info"]["mode"] == "semantic"
    assert calls["column"] == "expanded_description"
    assert "fresh apples" in calls["query_texts"]
    assert "C-1001" in calls["allowed_ids"]
    assert "C-1001" in result["matched_ticket_ids"]
    assert "C-9999" not in result["matched_ticket_ids"]
    assert result["examples"][0]["ticket_id"] == "C-1001"
    assert result["summary"] == "stub-summary"
    assert result["errors"] == []


def test_get_examples_structured_fallback(monkeypatch, filter_setup):
    filter_id, _ = filter_setup

    def fail_semantic_search(*args, **kwargs):
        raise AssertionError("semantic search should not run when no query")

    monkeypatch.setattr(vector_search, "semantic_search", fail_semantic_search)

    result = get_examples.get_examples(
        filter_id=filter_id,
        max_examples=2,
        text_column="expanded_description",
        text_query=None,
        include_summary=True,
    )

    assert result["query_info"]["mode"] == "structured"
    assert len(result["examples"]) == 2
    assert result["summary"] == "stub-summary"
    assert result["errors"] == []
    ids = result["matched_ticket_ids"]
    assert len(ids) == 2
    assert ids[0] == "C-1000"


def test_get_examples_logging_trace(monkeypatch, filter_setup, caplog):
    filter_id, _ = filter_setup
    monkeypatch.setenv("SEMANTIC_TRACE", "1")
    monkeypatch.setattr(get_examples, "_TRACE_ENABLED", True, raising=False)
    monkeypatch.setattr(vector_search, "_TRACE_ENABLED", True, raising=False)

    def fake_semantic_search(column, query_texts, top_k, allowed_ticket_ids, **kwargs):
        allowed_list = list(allowed_ticket_ids or [])
        vector_search._trace(  # type: ignore[attr-defined]
            logging.INFO,
            "semantic_search start column=%s top_k=%s allowed=%s",
            column,
            top_k,
            len(allowed_list),
        )
        return [
            vector_search.VectorMatch(ticket_id="C-1001", score=0.91, document="doc", metadata={}),
        ]

    monkeypatch.setattr(vector_search, "semantic_search", fake_semantic_search)

    caplog.set_level(logging.INFO, logger="tools.get_examples")
    caplog.set_level(logging.INFO, logger="tools.vector_search")

    result = get_examples.get_examples(
        filter_id=filter_id,
        text_query="fresh apples",
        text_column="expanded_description",
        top_k=1,
        include_summary=False,
    )

    messages = [record.getMessage() for record in caplog.records]
    joined = " | ".join(messages)
    assert "get_examples mode_select" in joined
    assert any("semantic_search start" in message for message in messages)
    assert result["query_info"]["mode"] == "semantic"
