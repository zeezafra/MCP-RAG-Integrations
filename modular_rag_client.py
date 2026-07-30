"""Discover and call a modular, context-aware MCP retrieval capability."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gemini_client import ask_gemini, plan_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Question to answer (otherwise prompted)")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run discovery, routing, and retrieval without calling Gemini",
    )
    return parser.parse_args()


def extract_resource_text(result: Any) -> str:
    chunks: list[str] = []
    for content in getattr(result, "contents", []) or []:
        text = getattr(content, "text", None)
        if text is not None:
            chunks.append(str(text))
            continue
        blob = getattr(content, "blob", None)
        if blob is not None:
            chunks.append(base64.b64decode(blob).decode("utf-8"))
    return "\n".join(chunks)


def extract_tool_text(result: Any) -> str:
    return "\n".join(
        str(content.text)
        for content in getattr(result, "content", []) or []
        if getattr(content, "text", None) is not None
    )


async def run_demo(question: str | None = None, *, no_llm: bool = False) -> None:
    print("=== Scalable and Modular MCP RAG Platform ===")
    server_path = Path(__file__).with_name("modular_rag_server.py")
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources = await session.list_resources()
            tools = await session.list_tools()

            print("\nDiscoverable resources:")
            for resource in resources.resources:
                print(f"- {resource.uri}")
            print("\nDiscoverable tools:")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description or ''}")

            catalog_resource = next(
                (
                    resource
                    for resource in resources.resources
                    if str(resource.uri) == "rag://catalog"
                ),
                None,
            )
            if catalog_resource is None:
                raise RuntimeError("Server did not expose rag://catalog")
            catalog_result = await session.read_resource(catalog_resource.uri)
            catalog = json.loads(extract_resource_text(catalog_result))
            print("\nSource catalog:")
            for source in catalog:
                print(f"- {source['label']}: {source['description']}")

            user_question = question or input("\nAsk a question: ").strip()
            tool_arguments: dict[str, Any] = {
                "question": user_question,
                "max_per_source": 3,
                "strategy": "keyword",
            }
            if no_llm:
                print(
                    "\nUsing deterministic routing because --no-llm was supplied."
                )
            else:
                print("\nAsking Gemini to select sources and retrieval strategy...")
                plan = plan_retrieval(user_question, catalog)
                tool_arguments.update(plan)
                print(f"Model retrieval plan: {json.dumps(plan, sort_keys=True)}")

            tool_result = await session.call_tool(
                "retrieve_knowledge",
                tool_arguments,
            )
            if getattr(tool_result, "isError", False):
                raise RuntimeError(extract_tool_text(tool_result))
            payload = json.loads(extract_tool_text(tool_result))

            print("\nSelected sources:")
            for source in payload["selected_sources"]:
                print(f"- {source}")

            print("\nRetrieval trace:")
            for event in payload["trace"]:
                print(f"- {json.dumps(event, sort_keys=True)}")

            print("\nRetrieved evidence:")
            for match in payload["matches"]:
                print(
                    f"- [{match['label']}] (score={match['score']}) "
                    f"{match['text']}"
                )
            if not payload["matches"]:
                print("(no matching evidence)")

            if no_llm:
                print("\nSkipping Gemini because --no-llm was supplied.")
                return

            print("\nQuerying Gemini with traced, source-labeled evidence...")
            answer = ask_gemini(user_question, payload["matches"])
            print("\n--- Gemini's grounded answer ---")
            print(answer)


def main() -> None:
    args = parse_args()
    asyncio.run(run_demo(args.question, no_llm=args.no_llm))


if __name__ == "__main__":
    main()
