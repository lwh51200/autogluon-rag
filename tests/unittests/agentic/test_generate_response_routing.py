import os
import unittest
from unittest.mock import MagicMock

from agrag.agrag import PRESETS_CONFIG_DIRECTORY, AutoGluonRAG

PRESET_CONFIG = os.path.join(PRESETS_CONFIG_DIRECTORY, "medium_quality_config.yaml")


class TestGenerateResponseRouting(unittest.TestCase):
    def _build_agrag(self):
        # Pass the preset config file explicitly so the constructor does not fall
        # back to argparse (which would try to parse pytest's argv). data_dir
        # satisfies the constructor's requirement without initializing anything.
        agrag = AutoGluonRAG(config_file=PRESET_CONFIG, data_dir="/tmp/does_not_matter")

        # Inject lightweight fakes for the modules generate_response depends on.
        agrag.retriever_module = MagicMock()
        agrag.retriever_module.top_k = 5
        agrag.retriever_module.retrieve.return_value = ["chunk 1", "chunk 2"]

        agrag.generator_module = MagicMock()
        agrag.generator_module.model_name = "mistral-7b"
        agrag.generator_module.generate_response.return_value = "standard answer"

        # Data processing module must NEVER be touched at query time.
        agrag.data_processing_module = MagicMock()
        return agrag

    def test_default_uses_standard_path(self):
        agrag = self._build_agrag()
        response = agrag.generate_response("what is autogluon")
        self.assertEqual(response, "standard answer")
        # Standard path retrieves exactly once, via the plain (text) retrieve.
        agrag.retriever_module.retrieve.assert_called_once()
        # Agentic module was never constructed.
        self.assertIsNone(agrag.agentic_module)
        # No data processing at query time (design invariant).
        agrag.data_processing_module.process_data.assert_not_called()

    def test_standard_path_skips_retrieval_when_top_k_zero(self):
        agrag = self._build_agrag()
        agrag.retriever_module.top_k = 0
        agrag.generate_response("q")
        agrag.retriever_module.retrieve.assert_not_called()

    def test_agentic_mode_routes_to_agentic_module(self):
        agrag = self._build_agrag()
        # Structured retrieval for the agentic path.
        agrag.retriever_module.retrieve.return_value = [
            {"text": "a", "doc_id": 0, "chunk_id": 0, "rank": 0},
            {"text": "b", "doc_id": 0, "chunk_id": 1, "rank": 1},
        ]
        agrag.generator_module.generate_response.return_value = "supported"

        response = agrag.generate_response("q", mode="agentic")
        self.assertIsInstance(response, str)
        # Agentic module was lazily built.
        self.assertIsNotNone(agrag.agentic_module)
        agrag.data_processing_module.process_data.assert_not_called()

    def test_agentic_return_trace(self):
        agrag = self._build_agrag()
        agrag.retriever_module.retrieve.return_value = [
            {"text": "a", "doc_id": 0, "chunk_id": 0, "rank": 0},
            {"text": "b", "doc_id": 0, "chunk_id": 1, "rank": 1},
        ]
        agrag.generator_module.generate_response.return_value = "supported"

        result = agrag.generate_response("q", mode="agentic", return_trace=True)
        self.assertIsInstance(result, tuple)
        answer, trace = result
        self.assertIsInstance(trace, dict)
        self.assertIn("metrics", trace)

    def test_resolve_mode_precedence(self):
        agrag = self._build_agrag()
        # Explicit arg wins.
        self.assertEqual(agrag._resolve_mode("agentic"), "agentic")
        # Default when nothing set.
        self.assertEqual(agrag._resolve_mode(None), "standard")
        # agent.enabled flips default to agentic.
        agrag.args.config.setdefault("agent", {})["enabled"] = True
        self.assertEqual(agrag._resolve_mode(None), "agentic")


if __name__ == "__main__":
    unittest.main()
