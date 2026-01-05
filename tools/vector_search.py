from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

VectorMatchMeta = dict[str, object]

_COLUMN_TO_COLLECTION = {
    "expanded_description": "Expanded_Description_Collection",
    "customer_issue": "Customer_Issue_Collection",
    "resolution_provided_summary": "Resolution_Provided_Collection",
    "root_cause": "Root_Cause_Collection",
}

_TICKET_ID_METADATA_KEY = "COMPLAINT_NUMBER"
_DEFAULT_VECTOR_PATH = Path(__file__).resolve().parent.parent / "VectorDB"
_DEFAULT_MODEL_NAME = os.getenv("SEMANTIC_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
_MAX_VECTOR_FETCH = 50
_MAX_WHERE_IN = 100


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


class VectorSearchError(RuntimeError):
    """Raised when the vector backend is misconfigured or unavailable."""


@dataclass
class VectorMatch:
    ticket_id: str
    score: float
    document: Optional[str]
    metadata: VectorMatchMeta


@lru_cache(maxsize=1)
def _embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - import guard
        raise VectorSearchError(
            "sentence-transformers is required for semantic search. Install sentence-transformers>=2.7.0."
        ) from exc
    model_kwargs: dict[str, str] = {}
    tokenizer_kwargs: dict[str, str] = {}

    attn_impl = os.getenv("SEMANTIC_EMBED_ATTN_IMPL")
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    padding_side = os.getenv("SEMANTIC_EMBED_PADDING_SIDE")
    if padding_side:
        tokenizer_kwargs["padding_side"] = padding_side
    else:
        tokenizer_kwargs["padding_side"] = "left"

    model = SentenceTransformer(
        _DEFAULT_MODEL_NAME,
        model_kwargs=model_kwargs or None,
        tokenizer_kwargs=tokenizer_kwargs or None,
    )
    return model


@lru_cache(maxsize=None)
def _get_collection(column: str, vector_path: str = str(_DEFAULT_VECTOR_PATH)):
    try:
        collection_name = _COLUMN_TO_COLLECTION[column]
    except KeyError:
        raise ValueError(f"Unsupported text column '{column}'. Supported: {sorted(_COLUMN_TO_COLLECTION)}")

    path = Path(vector_path)
    if not path.exists():
        raise VectorSearchError(f"Vector store path '{path}' not found.")

    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - import guard
        raise VectorSearchError("chromadb is required for semantic search. Install chromadb.") from exc

    client = chromadb.PersistentClient(path=str(path))
    try:
        return client.get_collection(collection_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise VectorSearchError(f"Failed to load collection '{collection_name}': {exc}") from exc


def _ensure_queries(queries: Sequence[str] | str | None) -> List[str]:
    if queries is None:
        return []
    if isinstance(queries, str):
        queries = [queries]
    cleaned = []
    for text in queries:
        if text is None:
            continue
        stripped = str(text).strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _average_embeddings(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        return vectors
    return np.mean(vectors, axis=0)


def _to_similarity(distance: Optional[float]) -> float:
    if distance is None:
        return 0.0
    return float(1.0 / (1.0 + float(distance)))


def embed_query(text_queries: Sequence[str] | str) -> Optional[np.ndarray]:
    queries = _ensure_queries(text_queries)
    if not queries:
        _trace(logging.INFO, "embed_query skipped_empty")
        return None
    _trace(logging.DEBUG, "embed_query queries=%s", queries)
    model = _embedding_model()
    vectors = model.encode(
        queries,
        prompt_name="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    _trace(logging.DEBUG, "embed_query encoded shape=%s", getattr(vectors, "shape", None))
    return _average_embeddings(np.atleast_2d(vectors))


def semantic_search(
    *,
    column: str,
    query_texts: Sequence[str] | str,
    top_k: int,
    vector_path: Optional[str] = None,
    allowed_ticket_ids: Optional[Iterable[str]] = None,
) -> List[VectorMatch]:
    if top_k <= 0:
        return []

    allowed_iterable: Optional[List[Any]] = None
    if allowed_ticket_ids is not None:
        allowed_iterable = [ticket_id for ticket_id in allowed_ticket_ids if ticket_id is not None]
    allowed_display = "all" if allowed_iterable is None else len(allowed_iterable)
    _trace(logging.INFO, "semantic_search start column=%s top_k=%s allowed=%s", column, top_k, allowed_display)

    query_embedding = embed_query(query_texts)
    if query_embedding is None:
        _trace(logging.INFO, "semantic_search skipped_no_query column=%s", column)
        return []

    collection = _get_collection(column, str(vector_path or _DEFAULT_VECTOR_PATH))

    allowed: Optional[set[str]] = None
    if allowed_iterable:
        allowed = {str(ticket_id) for ticket_id in allowed_iterable}
        _trace(logging.DEBUG, "semantic_search allowed_unique=%s", len(allowed))

    where = None
    if allowed and len(allowed) <= _MAX_WHERE_IN:
        bucket = sorted(allowed)
        where = {_TICKET_ID_METADATA_KEY: {"$in": bucket}}
        _trace(logging.DEBUG, "semantic_search using_where_in count=%s", len(bucket))

    fetch_k = min(max(top_k * 3, top_k), _MAX_VECTOR_FETCH)

    try:
        raw = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_k,
            where=where,
            include=["ids", "metadatas", "documents", "distances"],
        )
    except Exception as exc:  # pragma: no cover - backend error surface
        _trace(logging.ERROR, "semantic_search query_failed column=%s error=%s", column, exc)
        raise VectorSearchError(f"Vector search failed: {exc}") from exc

    ids = raw.get("ids", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    matches: List[VectorMatch] = []
    for idx, meta in enumerate(metadatas):
        meta = meta or {}
        ticket_id = meta.get(_TICKET_ID_METADATA_KEY) or (ids[idx] if idx < len(ids) else None)
        if ticket_id is None:
            continue
        ticket_id = str(ticket_id)
        if allowed and ticket_id not in allowed:
            continue
        doc = documents[idx] if idx < len(documents) else None
        dist = distances[idx] if idx < len(distances) else None
        matches.append(
            VectorMatch(
                ticket_id=ticket_id,
                score=_to_similarity(dist),
                document=doc,
                metadata=meta,
            )
        )
        if len(matches) >= top_k:
            break

    _trace(
        logging.INFO,
        "semantic_search raw_results=%s kept=%s column=%s",
        len(metadatas),
        len(matches),
        column,
    )
    if matches:
        preview = ", ".join(f"{m.ticket_id}:{m.score:.3f}" for m in matches[:5])
        _trace(logging.DEBUG, "semantic_search top_matches=%s", preview)

    if len(matches) < top_k and allowed and (where is not None):
        # We might have truncated the allowed set due to $in limits. Re-run without where and post-filter.
        remaining = top_k - len(matches)
        initial_count = len(matches)
        found_ids = {m.ticket_id for m in matches}
        try:
            raw_unfiltered = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(fetch_k, len(allowed)),
                include=["ids", "metadatas", "documents", "distances"],
            )
        except Exception:
            raw_unfiltered = raw

        ids_u = raw_unfiltered.get("ids", [[]])[0]
        metas_u = raw_unfiltered.get("metadatas", [[]])[0]
        docs_u = raw_unfiltered.get("documents", [[]])[0]
        dists_u = raw_unfiltered.get("distances", [[]])[0]
        for idx, meta in enumerate(metas_u):
            meta = meta or {}
            ticket_id = meta.get(_TICKET_ID_METADATA_KEY) or (ids_u[idx] if idx < len(ids_u) else None)
            if ticket_id is None:
                continue
            ticket_id = str(ticket_id)
            if ticket_id in found_ids or ticket_id not in allowed:
                continue
            doc = docs_u[idx] if idx < len(docs_u) else None
            dist = dists_u[idx] if idx < len(dists_u) else None
            matches.append(
                VectorMatch(
                    ticket_id=ticket_id,
                    score=_to_similarity(dist),
                    document=doc,
                    metadata=meta,
                )
            )
            found_ids.add(ticket_id)
            remaining -= 1
            if remaining <= 0 or len(matches) >= top_k:
                break
        _trace(logging.DEBUG, "semantic_search fallback_added=%s", len(matches) - initial_count)

    return matches


__all__ = [
    "VectorMatch",
    "VectorSearchError",
    "semantic_search",
    "embed_query",
]
