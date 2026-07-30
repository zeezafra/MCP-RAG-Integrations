"""Use Gemini to plan retrieval and answer from the returned evidence."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


DEFAULT_MODEL = "gemini-3.6-flash"


def _get_client() -> tuple[genai.Client, str]:
    load_dotenv(Path(__file__).with_name(".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add "
            "your key, or run the client with --no-llm."
        )
    return genai.Client(api_key=api_key), os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    return str(text).strip() if text else "(Gemini returned no text.)"


def _normalize_retrieval_plan(
    plan: Mapping[str, Any],
    source_keys: Sequence[str],
) -> dict[str, Any]:
    raw_sources = plan.get("sources")
    selected = (
        [key for key in raw_sources if isinstance(key, str) and key in source_keys]
        if isinstance(raw_sources, list)
        else []
    )
    if not selected:
        raise RuntimeError("Gemini's retrieval plan selected no valid sources.")

    raw_limit = plan.get("max_per_source", 3)
    max_per_source = (
        raw_limit
        if isinstance(raw_limit, int) and not isinstance(raw_limit, bool)
        else 3
    )
    strategy = plan.get("strategy")
    return {
        "sources": list(dict.fromkeys(selected)),
        "max_per_source": max(1, min(5, max_per_source)),
        "strategy": strategy if strategy in {"keyword", "broad"} else "keyword",
    }


def plan_retrieval(
    question: str,
    catalog: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Let Gemini select source keys, retrieval depth, and ranking strategy."""
    client, model = _get_client()
    source_keys = [str(source["key"]) for source in catalog]
    catalog_text = "\n".join(
        f"- {source['key']}: {source['description']}" for source in catalog
    )
    retrieval_tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="retrieve_knowledge",
                description=(
                    "Retrieve grounded evidence from the MCP sources.\n"
                    f"Available sources:\n{catalog_text}"
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string", "enum": source_keys},
                            "minItems": 1,
                        },
                        "max_per_source": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["keyword", "broad"],
                        },
                    },
                    "required": ["sources", "max_per_source", "strategy"],
                },
            )
        ]
    )
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(
            max_output_tokens=300,
            system_instruction=(
                "Select the smallest useful set of retrieval sources for the "
                "question. Use broad ranking when wording may differ from stored "
                "terms; otherwise use keyword ranking."
            ),
            tools=[retrieval_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["retrieve_knowledge"],
                ),
            ),
        ),
    )
    function_call = next(
        (
            getattr(part, "function_call", None)
            for candidate in getattr(response, "candidates", []) or []
            for part in (
                getattr(getattr(candidate, "content", None), "parts", []) or []
            )
            if getattr(getattr(part, "function_call", None), "name", None)
            == "retrieve_knowledge"
        ),
        None,
    )
    if function_call is None:
        raise RuntimeError("Gemini did not return a retrieval plan.")

    arguments = getattr(function_call, "args", None)
    if not isinstance(arguments, Mapping):
        raise RuntimeError("Gemini returned invalid retrieval plan arguments.")
    return _normalize_retrieval_plan(arguments, source_keys)


def ask_gemini(
    question: str,
    matches: Sequence[Mapping[str, Any]],
) -> str:
    client, model = _get_client()
    context = "\n".join(
        f"- [{match['label']}] {match['text']}" for match in matches
    )
    response = client.models.generate_content(
        model=model,
        contents=f"Retrieved evidence:\n{context}\n\nQuestion: {question}",
        config=types.GenerateContentConfig(
            max_output_tokens=500,
            system_instruction=(
                "Answer only from the retrieved evidence. Cite supporting source "
                "labels in square brackets. If the evidence is insufficient, say so."
            ),
        ),
    )
    return _response_text(response)
