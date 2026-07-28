import unittest

from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.tools import (
    ContextCompressionTool,
    MultiQueryRetrieveTool,
    QueryRewriteTool,
    RetrieveTool,
    Tool,
    ToolRegistry,
)


class FakeRetriever:
    """Returns canned structured records per query; records calls."""

    def __init__(self, responses, reranker=None):
        self.responses = responses
        self.calls = []
        self.reranker = reranker

    def retrieve(self, query, return_metadata=False, top_k=None):
        self.calls.append((query, return_metadata, top_k))
        return self.responses.get(query)


class FakeGenerator:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        return self.response


class TestRetrieveTool(unittest.TestCase):
    def test_produces_evidence_from_records(self):
        retriever = FakeRetriever(
            {
                "q": [
                    {"text": "chunk a", "doc_id": 0, "chunk_id": 0, "rank": 0, "retrieval_score": 0.9},
                    {"text": "chunk b", "doc_id": 0, "chunk_id": 1, "rank": 1, "retrieval_score": 0.8},
                ]
            }
        )
        tool = RetrieveTool(retriever)
        result = tool.run(query="q")
        self.assertTrue(result.contains_evidence)
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(result.evidence[0].doc_id, 0)
        self.assertEqual(result.evidence[0].tool_name, "RetrieveTool")
        # Called the retriever with metadata enabled and default top_k.
        self.assertEqual(retriever.calls, [("q", True, None)])

    def test_passes_top_k_through(self):
        retriever = FakeRetriever({"q": [{"text": "a", "doc_id": 0, "chunk_id": 0}]})
        RetrieveTool(retriever, top_k=8).run(query="q")
        self.assertEqual(retriever.calls, [("q", True, 8)])

    def test_handles_none_result(self):
        tool = RetrieveTool(FakeRetriever({"q": None}))
        result = tool.run(query="q")
        self.assertFalse(result.contains_evidence)
        self.assertEqual(len(result.evidence), 0)

    def test_handles_text_only_records(self):
        tool = RetrieveTool(FakeRetriever({"q": ["plain chunk"]}))
        result = tool.run(query="q")
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0].text, "plain chunk")


class TestMultiQueryRetrieveTool(unittest.TestCase):
    def test_merges_across_subqueries(self):
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "b", "doc_id": 0, "chunk_id": 1}],
            }
        )
        tool = MultiQueryRetrieveTool(retriever)
        result = tool.run(queries=["q1", "q2"])
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual({e.retrieval_query for e in result.evidence}, {"q1", "q2"})

    def test_dedup_left_to_store(self):
        # Same chunk returned by both subqueries; the store dedups it.
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
            }
        )
        result = MultiQueryRetrieveTool(retriever).run(queries=["q1", "q2"])
        self.assertEqual(len(result.evidence), 2)  # tool returns both
        store = EvidenceStore()
        stored = store.add_many(result.evidence)
        self.assertEqual(stored, 1)  # store keeps one


class TestMultiQueryRetrieveToolFused(unittest.TestCase):
    def test_fused_mode_dedups_and_preserves_provenance(self):
        # Chunk (0,0) surfaced by both subqueries; (0,1) and (1,0) each by one.
        retriever = FakeRetriever(
            {
                "q1": [
                    {"text": "a", "doc_id": 0, "chunk_id": 0},
                    {"text": "b", "doc_id": 0, "chunk_id": 1},
                ],
                "q2": [
                    {"text": "a", "doc_id": 0, "chunk_id": 0},
                    {"text": "c", "doc_id": 1, "chunk_id": 0},
                ],
            }
        )
        tool = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True)
        result = tool.run(queries=["q1", "q2"])

        # Fused: one record per distinct chunk, not concatenated (would be 4).
        self.assertEqual(len(result.evidence), 3)
        by_text = {e.text: e for e in result.evidence}
        # The shared chunk keeps BOTH subqueries as provenance.
        self.assertEqual(by_text["a"].retrieval_queries, ["q1", "q2"])
        # Single-subquery chunks keep just their own.
        self.assertEqual(by_text["b"].retrieval_queries, ["q1"])
        self.assertEqual(by_text["c"].retrieval_queries, ["q2"])

    def test_fused_mode_assigns_fusion_rank_and_rrf_score(self):
        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
            }
        )
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        top = result.evidence[0]
        self.assertEqual(top.fusion_rank, 0)
        self.assertIsNotNone(top.rrf_score)
        self.assertGreater(top.rrf_score, 0.0)

    def test_fused_mode_empty_results(self):
        retriever = FakeRetriever({"q1": None, "q2": None})
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        self.assertFalse(result.contains_evidence)
        self.assertEqual(len(result.evidence), 0)

    def test_fused_mode_applies_global_reranker(self):
        class FakeReranker:
            def rerank(self, query, texts, return_scores=False):
                # Reverse order, attach descending scores.
                reversed_texts = list(reversed(texts))
                return [(t, float(len(reversed_texts) - i)) for i, t in enumerate(reversed_texts)]

        retriever = FakeRetriever(
            {
                "q1": [{"text": "a", "doc_id": 0, "chunk_id": 0}],
                "q2": [{"text": "c", "doc_id": 1, "chunk_id": 0}],
            },
            reranker=FakeReranker(),
        )
        result = MultiQueryRetrieveTool(retriever, use_fused_retrieval=True).run(queries=["q1", "q2"])
        # The reranker reversed fused order; evidence follows the rerank.
        self.assertEqual([e.text for e in result.evidence][0], "c")
        self.assertIsNotNone(result.evidence[0].rerank_score)


class TestLLMTools(unittest.TestCase):
    def test_query_rewrite(self):
        gen = FakeGenerator("  better query  ")
        result = QueryRewriteTool(gen).run(query="orig query")
        self.assertEqual(result.output, "better query")
        self.assertFalse(result.contains_evidence)
        self.assertIn("orig query", gen.prompts[0])

    def test_context_compression(self):
        gen = FakeGenerator("short summary")
        result = ContextCompressionTool(gen).run(query="q", texts=["c1", "c2"])
        self.assertEqual(result.output, "short summary")
        self.assertIn("c1", gen.prompts[0])
        self.assertIn("c2", gen.prompts[0])


class TestToolRegistry(unittest.TestCase):
    def test_register_run_and_names(self):
        retriever = FakeRetriever({"q": [{"text": "a", "doc_id": 0, "chunk_id": 0}]})
        registry = ToolRegistry([RetrieveTool(retriever)])
        self.assertTrue(registry.has("RetrieveTool"))
        self.assertEqual(registry.names(), ["RetrieveTool"])
        result = registry.run("RetrieveTool", query="q")
        self.assertEqual(len(result.evidence), 1)

    def test_unknown_tool_raises(self):
        registry = ToolRegistry([])
        with self.assertRaises(KeyError):
            registry.run("NopeTool", query="q")

    def test_duplicate_registration_raises(self):
        retriever = FakeRetriever({})
        registry = ToolRegistry([RetrieveTool(retriever)])
        with self.assertRaises(ValueError):
            registry.register(RetrieveTool(retriever))

    def test_nameless_tool_rejected(self):
        class Nameless(Tool):
            name = ""

        with self.assertRaises(ValueError):
            ToolRegistry([Nameless()])


if __name__ == "__main__":
    unittest.main()
