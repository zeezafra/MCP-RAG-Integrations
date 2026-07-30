from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import gemini_client


class FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return next(self.responses)


class GeminiClientTests(unittest.TestCase):
    def test_model_plan_uses_forced_function_call_and_normalizes_values(self) -> None:
        function_call = SimpleNamespace(
            name="retrieve_knowledge",
            args={
                "sources": ["support", "policies", "support"],
                "max_per_source": 4,
                "strategy": "broad",
            },
        )
        response = SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(function_call=function_call)]
                    )
                )
            ]
        )
        models = FakeModels([response])
        client = SimpleNamespace(models=models)
        catalog = [
            {"key": "support", "description": "Support answers"},
            {"key": "policies", "description": "Policies"},
        ]

        with patch.object(
            gemini_client,
            "_get_client",
            return_value=(client, "test-gemini"),
        ):
            plan = gemini_client.plan_retrieval("How do I return it?", catalog)

        self.assertEqual(
            plan,
            {
                "sources": ["support", "policies"],
                "max_per_source": 4,
                "strategy": "broad",
            },
        )
        config = models.calls[0]["config"]
        function_config = config.tool_config.function_calling_config
        self.assertEqual(function_config.mode.value, "ANY")
        self.assertEqual(function_config.allowed_function_names, ["retrieve_knowledge"])

    def test_malformed_plan_options_fall_back_safely(self) -> None:
        plan = gemini_client._normalize_retrieval_plan(
            {
                "sources": ["support"],
                "max_per_source": None,
                "strategy": "unexpected",
            },
            ["support"],
        )
        self.assertEqual(plan["max_per_source"], 3)
        self.assertEqual(plan["strategy"], "keyword")

    def test_grounded_answer_includes_source_labeled_evidence(self) -> None:
        models = FakeModels([SimpleNamespace(text="Use the policy [Policies].")])
        client = SimpleNamespace(models=models)
        with patch.object(
            gemini_client,
            "_get_client",
            return_value=(client, "test-gemini"),
        ):
            answer = gemini_client.ask_gemini(
                "How do I return it?",
                [{"label": "Policies", "text": "Return within 30 days."}],
            )

        self.assertEqual(answer, "Use the policy [Policies].")
        self.assertIn("[Policies] Return within 30 days.", models.calls[0]["contents"])


if __name__ == "__main__":
    unittest.main()
