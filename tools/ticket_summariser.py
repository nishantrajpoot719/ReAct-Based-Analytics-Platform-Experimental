from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence

try:  # pragma: no cover - import guard
    import dspy  # type: ignore
except ImportError:  # pragma: no cover - import guard
    dspy = None  # type: ignore


if dspy is not None:

    class SummariseTicket(dspy.Signature):
        action: List[str] = dspy.InputField(desc="The list of tickets to be summarised")
        ticket_summary: str = dspy.OutputField(
            desc="A short concise analytical summary of the tickets provided. Answer in Points, structured format. Answer in Markdown format.",
        )
        key_insights: str = dspy.OutputField(
            desc="Key insights from the tickets provided. Answer in Points, structured format. Answer in Markdown format.",
        )

    summarise = dspy.ChainOfThought(SummariseTicket)

else:  # pragma: no cover - import guard fallback
    SummariseTicket = None  # type: ignore
    summarise = None  # type: ignore


if dspy is not None:

    class FinalSummary(dspy.Signature):
        segregated_summary: str = dspy.InputField(desc="The combined summaries from multiple models")
        integrated_summary: str = dspy.OutputField(
            desc="A concise summary of the key insights from the combined summaries provided. Answer in Points, structured format. Answer in Markdown format.",
        )

    final_summarise = dspy.ChainOfThought(FinalSummary)

else:  # pragma: no cover - import guard fallback
    FinalSummary = None  # type: ignore
    final_summarise = None  # type: ignore

MODELS = [
    "openai/gpt-oss-120b",
    "openai/qwen-3-235b-a22b-instruct-2507",
    "openai/llama-3.3-70b",
    "openai/qwen-3-32b",
    "openai/llama3.1-8b",
]

_DEFAULT_MAX_ITEMS = 100


def _chunk_equally(xs: Sequence[str], n: int) -> List[List[str]]:
    k = math.ceil(len(xs) / n) if xs else 0
    return [list(xs[i : i + k]) for i in range(0, len(xs), k)] if k else [[] for _ in range(n)]


def _run_model_summary(model_name: str, tickets_chunk: Sequence[str]) -> tuple[str, str]:
    if summarise is None:
        raise RuntimeError("Summarisation backend unavailable: dspy is not installed.")
    if not tickets_chunk:
        return model_name, ""
    with dspy.context(
        lm=dspy.LM(
            model=model_name,
            api_key=os.getenv("CEREBRAS_API_KEY"),
            api_base="https://api.cerebras.ai/v1",
        ),
        cache=True,
    ):
        result = summarise(action=list(tickets_chunk))
        text = (
            f"— Summary\n{result.ticket_summary}\n\n"
            f"— Key Insights\n{result.key_insights}\n"
        )
        return model_name, text


def _final_summary(text: str) -> str:
    if final_summarise is None:
        raise RuntimeError("Summarisation backend unavailable: dspy is not installed.")
    if not text.strip():
        return ""
    with dspy.context(
        lm=dspy.LM(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            api_base="https://api.cerebras.ai/v1",
        ),
    ):
        result = final_summarise(segregated_summary=text)
        return result.integrated_summary


def summarise_texts(texts: Sequence[str], max_items: int = _DEFAULT_MAX_ITEMS) -> str:
    if dspy is None:
        raise RuntimeError("Summarisation backend unavailable: install dspy to enable this feature.")
    cleaned = [str(text).strip() for text in texts if text and str(text).strip()]
    if not cleaned:
        raise ValueError("No tickets provided for summary.")

    trimmed = cleaned[:max_items]
    chunks = _chunk_equally(trimmed, len(MODELS))
    if len(chunks) < len(MODELS):
        chunks.extend([[] for _ in range(len(MODELS) - len(chunks))])
    elif len(chunks) > len(MODELS):
        chunks = chunks[: len(MODELS)]

    max_workers = min(len(MODELS), max(1, len([chunk for chunk in chunks if chunk])))
    results: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_model_summary, model, chunk): model
            for model, chunk in zip(MODELS, chunks)
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                name, summary_text = future.result()
                results[name] = summary_text
            except Exception as exc:  # pragma: no cover - propagates to final summary
                errors[model] = f"### {model} — Error\n{exc}\n"

    for model, message in errors.items():
        results.setdefault(model, message)

    combined = "\n\n".join(results.get(model, "") for model in MODELS)
    final = _final_summary(combined)
    return final or combined


__all__ = ["summarise_texts"]
