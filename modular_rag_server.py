"""MCP server that exposes modular resources and an orchestrated retriever."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from adapters import build_adapters
from orchestration import CATALOG, build_retrieval_payload, catalog_as_dicts


SERVER = FastMCP("ModularRAGPlatform")
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = PROJECT_DIR / "knowledge"
ADAPTERS = build_adapters(DATA_DIR)


def _read_source(key: str) -> str:
    try:
        return ADAPTERS[key].read()
    except (KeyError, OSError) as error:
        return f"[Error] Could not read {key}: {error}"


@SERVER.resource("rag://catalog")
def source_catalog() -> str:
    """Discover the available retrieval sources and their specialties."""
    return json.dumps(catalog_as_dicts(), indent=2)


@SERVER.resource("rag://products")
def products_resource() -> str:
    """Read product and warranty knowledge."""
    return _read_source("products")


@SERVER.resource("rag://support")
def support_resource() -> str:
    """Read customer support procedures."""
    return _read_source("support")


@SERVER.resource("rag://policies")
def policies_resource() -> str:
    """Read authoritative return and refund policies."""
    return _read_source("policies")


@SERVER.resource("rag://status")
def status_resource() -> str:
    """Read the simulated real-time operations feed."""
    return _read_source("status")


@SERVER.tool()
def retrieve_knowledge(
    question: str,
    sources: list[str] | None = None,
    max_per_source: int = 3,
    strategy: str = "keyword",
) -> str:
    """Retrieve from model-selected sources, or route automatically."""
    payload = build_retrieval_payload(
        question,
        ADAPTERS,
        max_per_source=max_per_source,
        requested_sources=sources,
        strategy=strategy,
    )
    return json.dumps(payload, indent=2)


def main() -> None:
    SERVER.run()


if __name__ == "__main__":
    main()
