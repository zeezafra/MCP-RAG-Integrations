from __future__ import annotations

import unittest
from pathlib import Path

from adapters import build_adapters
from orchestration import (
    CATALOG,
    build_retrieval_payload,
    catalog_as_dicts,
    route_sources,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "knowledge"
ADAPTERS = build_adapters(DATA_DIR)


class OrchestrationTests(unittest.TestCase):
    def test_context_aware_router_selects_relevant_sources(self) -> None:
        decision = route_sources("How do I return a damaged product?")
        selected = {source.key for source in decision.selected}
        self.assertEqual(selected, {"products", "support", "policies"})
        self.assertNotIn("status", selected)
        self.assertFalse(decision.used_fallback)

    def test_router_falls_back_to_all_sources_for_unknown_topic(self) -> None:
        decision = route_sources("Explain quantum entanglement.")
        self.assertEqual(decision.selected, CATALOG)
        self.assertTrue(decision.used_fallback)

    def test_retrieval_payload_contains_evidence_and_trace(self) -> None:
        payload = build_retrieval_payload(
            "How do I return a damaged product?",
            ADAPTERS,
            max_per_source=2,
        )
        self.assertTrue(payload["matches"])
        self.assertEqual(payload["trace"][0]["step"], "route")
        retrieve_events = [
            event for event in payload["trace"] if event["step"] == "retrieve"
        ]
        self.assertEqual(len(retrieve_events), 3)
        self.assertTrue(
            all("source" in match and "score" in match for match in payload["matches"])
        )

    def test_max_per_source_is_enforced(self) -> None:
        payload = build_retrieval_payload(
            "return damaged refund shipping",
            ADAPTERS,
            max_per_source=1,
        )
        source_counts: dict[str, int] = {}
        for match in payload["matches"]:
            source_counts[match["source"]] = source_counts.get(match["source"], 0) + 1
        self.assertTrue(all(count <= 1 for count in source_counts.values()))

    def test_invalid_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_retrieval_payload("return", ADAPTERS, max_per_source=0)

    def test_public_catalog_hides_internal_file_paths(self) -> None:
        catalog = catalog_as_dicts()
        self.assertTrue(catalog)
        self.assertTrue(all("filename" not in source for source in catalog))

    def test_model_selected_sources_bypass_keyword_router(self) -> None:
        payload = build_retrieval_payload(
            "Tell me about refunds.",
            ADAPTERS,
            requested_sources=["support"],
            strategy="broad",
        )
        self.assertEqual(payload["selected_sources"], ["support"])
        self.assertEqual(payload["trace"][0]["mode"], "model_selected")
        self.assertEqual(payload["strategy"], "broad")

    def test_empty_model_source_selection_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_retrieval_payload(
                "Tell me about refunds.",
                ADAPTERS,
                requested_sources=[],
            )


if __name__ == "__main__":
    unittest.main()
