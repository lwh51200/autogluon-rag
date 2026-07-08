"""End-to-end agentic RAG test with REAL modules (no mocks).

Unlike the rest of the agentic suite (which uses fakes), this test drives the
whole stack — real HuggingFace embedding + generator models, a real FAISS index,
the real RetrieverModule — through the agentic path. It exists to catch wiring
regressions that unit tests with fakes cannot, e.g. a retriever/evidence
contract drift or a loop-budget default that abstains before it can recover.

It reuses the committed ``local_example`` corpus and its prebuilt FAISS index so
nothing is re-embedded or written. The models used are tiny CPU models
(all-MiniLM-L6-v2, tiny-gpt2). The tiny generator produces gibberish, so this
test asserts the *plumbing* (retrieval finds real evidence, the loop terminates
with a valid status, a coherent trace is produced) rather than answer quality.

If the models are not cached and cannot be downloaded (offline CI), or the
prebuilt index is missing, the test skips rather than fails.
"""

import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOCAL_EXAMPLE = os.path.join(REPO_ROOT, "local_example")
LOCAL_CONFIG = os.path.join(LOCAL_EXAMPLE, "local_config.yaml")
INDEX_PATH = os.path.join(LOCAL_EXAMPLE, "vector_db_index", "index.idx")
METADATA_PATH = os.path.join(LOCAL_EXAMPLE, "vector_db_metadata", "metadata.json")


@unittest.skipUnless(
    os.path.exists(LOCAL_CONFIG) and os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH),
    "local_example config/index not available",
)
class TestAgenticEndToEndRealModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the pipeline once (model loading is the expensive part). Load the
        # committed index instead of re-embedding, and never write anything back.
        from agrag.agrag import AutoGluonRAG

        cls._prev_cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            agrag = AutoGluonRAG(config_file=LOCAL_CONFIG, data_dir=os.path.join("local_example", "docs"))
            agrag.initialize_data_module()
            agrag.initialize_embeddings_module()
            agrag.initialize_vectordb_module()
            agrag.initialize_reranker_module()
            agrag.initialize_retriever_module()
            agrag.initialize_generator_module()
            agrag.load_existing_vector_db(INDEX_PATH, METADATA_PATH)
            if not agrag.vector_db_module.index or agrag.vector_db_module.metadata is None:
                raise unittest.SkipTest("prebuilt vector DB index could not be loaded")
        except unittest.SkipTest:
            os.chdir(cls._prev_cwd)
            raise
        except Exception as exc:  # pragma: no cover - environment dependent
            os.chdir(cls._prev_cwd)
            raise unittest.SkipTest(f"real models/index unavailable: {exc}")
        cls.agrag = agrag

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._prev_cwd)

    def _build_module(self, **overrides):
        # Build the agentic module directly over the real retriever + generator.
        # Keep the context small: the demo generator (tiny-gpt2) has a hard 1024
        # token window, so we bound per-query retrieval and context. A real model
        # (e.g. Mistral 7B) would use the larger defaults from the config.
        from agrag.modules.agentic.agentic_module import AgenticRAGModule

        config = {
            "retrieve_top_k_per_query": 2,
            "max_context_tokens": 300,
            "min_evidence_count": 2,
            "use_query_rewrite": False,
        }
        config.update(overrides)
        return AgenticRAGModule(
            retriever_module=self.agrag.retriever_module,
            generator_module=self.agrag.generator_module,
            config=config,
        )

    def test_agentic_path_runs_end_to_end_with_real_modules(self):
        # Single-part query -> single retrieve; verification on; DEFAULT loop
        # budget (max_iterations not overridden) so a too-small default would
        # surface here as a premature abstention.
        module = self._build_module()
        answer, trace = module.answer("What is AutoGluon", return_trace=True)

        # Plumbing assertions (answer text is gibberish with the tiny model).
        self.assertIsInstance(answer, str)
        self.assertTrue(answer)
        self.assertIn(trace["status"], ("answered", "abstained", "max_iterations"))

        # Real retrieval actually found evidence in the committed index.
        self.assertGreaterEqual(trace["metrics"]["retrieval_calls"], 1)
        self.assertGreaterEqual(trace["metrics"]["evidence_count"], 1)

        # Evidence carries provenance from the real ingest metadata.
        self.assertTrue(trace["evidence"])
        first = trace["evidence"][0]
        self.assertIsNotNone(first["doc_id"])
        self.assertIsNotNone(first["chunk_id"])
        self.assertTrue(first["text"])

        # The plan always includes the original query first.
        self.assertEqual(trace["plan"][0], "What is AutoGluon")

        # The loop respected its budget (default is 5).
        self.assertLessEqual(trace["metrics"]["iterations"], module.max_iterations)
        self.assertEqual(module.max_iterations, 5)

    def test_agentic_path_via_generate_response_routing(self):
        # Exercise the real AutoGluonRAG.generate_response(mode="agentic") entry
        # point (not just the module) so the routing + lazy init are covered.
        # Pre-build a bounded module and attach it so generate_response reuses it
        # instead of constructing one with the large defaults (tiny-gpt2 ctx cap).
        self.agrag.agentic_module = self._build_module()
        answer = self.agrag.generate_response("What is AutoGluon", mode="agentic")
        self.assertIsInstance(answer, str)
        self.assertTrue(answer)

    def test_standard_and_agentic_share_modules_without_reingest(self):
        # The agentic module must reuse the already-initialized retriever and
        # generator (design invariant: no re-ingest / re-embed at query time).
        module = self._build_module()
        self.assertIs(module.retriever_module, self.agrag.retriever_module)
        self.assertIs(module.generator_module, self.agrag.generator_module)


if __name__ == "__main__":
    unittest.main()
