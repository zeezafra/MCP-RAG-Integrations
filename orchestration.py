"""Modular source catalog, routing, retrieval, and tracing."""

from __future__ import annotations

import string
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from adapters import KnowledgeAdapter


STOP_WORDS = {
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "the",
    "this",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    uri: str
    label: str
    description: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RoutingDecision:
    selected: tuple[SourceSpec, ...]
    scores: dict[str, int]
    matched_keywords: dict[str, list[str]]
    used_fallback: bool


CATALOG = (
    SourceSpec(
        key="products",
        uri="rag://products",
        label="Products",
        description="Product features, warranties, and repair eligibility.",
        keywords=(
            "product",
            "phone",
            "laptop",
            "headphones",
            "warranty",
            "repair",
            "replacement",
            "damaged",
        ),
    ),
    SourceSpec(
        key="support",
        uri="rag://support",
        label="Support FAQ",
        description="SQLite-backed customer steps, evidence, and refund timing.",
        keywords=(
            "support",
            "damaged",
            "return",
            "refund",
            "packaging",
            "photos",
            "contact",
        ),
    ),
    SourceSpec(
        key="policies",
        uri="rag://policies",
        label="Policies",
        description="Authoritative return, shipping, inspection, and refund rules.",
        keywords=(
            "policy",
            "policies",
            "return",
            "refund",
            "shipping",
            "inspection",
            "damaged",
            "defective",
        ),
    ),
    SourceSpec(
        key="status",
        uri="rag://status",
        label="Live status",
        description="Runtime-generated support, warehouse, and shipping conditions.",
        keywords=(
            "current",
            "currently",
            "status",
            "delay",
            "disruption",
            "warehouse",
            "response",
            "shipping",
        ),
    ),
)


def tokenize(text: str) -> set[str]:
    """Normalize text into useful routing and retrieval terms."""
    table = str.maketrans({character: " " for character in string.punctuation})
    words = " ".join(text.translate(table).lower().split()).split()
    return {word for word in words if len(word) >= 3 and word not in STOP_WORDS}


def route_sources(
    question: str,
    catalog: tuple[SourceSpec, ...] = CATALOG,
) -> RoutingDecision:
    """Select indexes by keyword overlap, falling back to the full catalog."""
    terms = tokenize(question)
    matched_keywords = {
        source.key: sorted(terms & set(source.keywords)) for source in catalog
    }
    scores = {
        source.key: len(matched_keywords[source.key]) for source in catalog
    }
    indexed = list(enumerate(catalog))
    selected_pairs = [
        (position, source)
        for position, source in indexed
        if scores[source.key] > 0
    ]
    selected_pairs.sort(key=lambda pair: (-scores[pair[1].key], pair[0]))
    used_fallback = not selected_pairs
    selected = (
        tuple(catalog)
        if used_fallback
        else tuple(source for _, source in selected_pairs)
    )
    return RoutingDecision(
        selected=selected,
        scores=scores,
        matched_keywords=matched_keywords,
        used_fallback=used_fallback,
    )


def _search_lines(
    question: str,
    lines: list[str],
    *,
    limit: int,
    strategy: str,
) -> list[tuple[int, str]]:
    terms = tokenize(question)
    scored: list[tuple[int, int, str]] = []
    for position, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        line_terms = tokenize(line)
        if strategy == "keyword":
            score = len(terms & line_terms)
        elif strategy == "broad":
            score = sum(
                any(
                    query_term.startswith(line_term[:5])
                    or line_term.startswith(query_term[:5])
                    for line_term in line_terms
                )
                for query_term in terms
            )
        else:
            raise ValueError("strategy must be 'keyword' or 'broad'")
        if score:
            scored.append((score, position, line))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(score, line) for score, _, line in scored[:limit]]


def build_retrieval_payload(
    question: str,
    adapters: Mapping[str, KnowledgeAdapter],
    *,
    max_per_source: int = 3,
    requested_sources: list[str] | None = None,
    strategy: str = "keyword",
    catalog: tuple[SourceSpec, ...] = CATALOG,
) -> dict[str, Any]:
    """Route and retrieve while recording an inspectable execution trace."""
    if max_per_source < 1:
        raise ValueError("max_per_source must be at least 1")
    if strategy not in {"keyword", "broad"}:
        raise ValueError("strategy must be 'keyword' or 'broad'")

    decision = route_sources(question, catalog)
    by_key = {source.key: source for source in catalog}
    if requested_sources is not None:
        if not requested_sources:
            raise ValueError("requested_sources cannot be empty when supplied")
        unknown = sorted(set(requested_sources) - set(by_key))
        if unknown:
            raise ValueError(f"Unknown source key(s): {', '.join(unknown)}")
        selected = tuple(dict.fromkeys(requested_sources))
        selected_specs = tuple(by_key[key] for key in selected)
        route_event: dict[str, Any] = {
            "step": "route",
            "mode": "model_selected",
            "selected": list(selected),
        }
    else:
        selected_specs = decision.selected
        route_event = {
            "step": "route",
            "mode": "deterministic_fallback",
            "selected": [source.key for source in decision.selected],
            "scores": decision.scores,
            "matched_keywords": decision.matched_keywords,
            "fallback_to_all_sources": decision.used_fallback,
        }
    trace: list[dict[str, Any]] = [route_event]
    matches: list[dict[str, Any]] = []

    for source in selected_specs:
        adapter = adapters.get(source.key)
        if adapter is None:
            trace.append(
                {
                    "step": "read",
                    "source": source.key,
                    "status": "missing",
                    "adapter": None,
                }
            )
            continue

        text = adapter.read()
        lines = text.splitlines()
        source_matches = _search_lines(
            question,
            lines,
            limit=max_per_source,
            strategy=strategy,
        )
        trace.append(
            {
                "step": "retrieve",
                "source": source.key,
                "uri": source.uri,
                "adapter": type(adapter).__name__,
                "strategy": strategy,
                "lines_scanned": len(lines),
                "matches_returned": len(source_matches),
            }
        )
        for score, text in source_matches:
            matches.append(
                {
                    "source": source.key,
                    "label": source.label,
                    "uri": source.uri,
                    "score": score,
                    "text": text,
                }
            )

    return {
        "question": question,
        "selected_sources": [source.key for source in selected_specs],
        "strategy": strategy,
        "matches": matches,
        "trace": trace,
    }


def catalog_as_dicts(
    catalog: tuple[SourceSpec, ...] = CATALOG,
) -> list[dict[str, Any]]:
    """Return a serializable discovery catalog."""
    return [asdict(source) for source in catalog]
