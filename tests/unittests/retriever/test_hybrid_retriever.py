import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, PARENT_ID_KEY
from agrag.modules.retriever.retrievers.retriever_base import RetrieverModule
from agrag.modules.vector_db.vector_database import VectorDatabaseModule


class FakeSparseRetriever:
    """Minimal stand-in for BM25Retriever returning canned (row_index, score)."""

    def __init__(self, hits):
        self._hits = hits
        self._built = True

    def build(self, documents):  # pragma: no cover - already "built"
        return self

    def search(self, query, top_k):
        return self._hits[:top_k]


def _metadata():
    return pd.DataFrame(
        [
            {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "alpha chunk"},
            {DOC_ID_KEY: 0, CHUNK_ID_KEY: 1, DOC_TEXT_KEY: "beta chunk"},
            {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "gamma chunk"},
        ]
    )


def _build_retriever(**kwargs):
    vector_db = MagicMock(VectorDatabaseModule)
    vector_db.metadata = _metadata()
    retriever = RetrieverModule(
        vector_database_module=vector_db,
        embedding_module=MagicMock(),
        top_k=5,
        use_reranker=False,
        **kwargs,
    )
    # Deterministic query embedding; no real model.
    retriever.encode_query = lambda query: np.zeros(4)
    return retriever, vector_db


class TestLegacyPathUnchanged(unittest.TestCase):
    def test_all_flags_off_uses_dense_only_path(self):
        retriever, vector_db = _build_retriever()
        self.assertFalse(retriever._uses_advanced_path)
        vector_db.search_vector_database.return_value = [0, 1]
        result = retriever.retrieve("q")
        # Legacy text-only path returns chunk texts in DB order.
        self.assertEqual(result, ["alpha chunk", "beta chunk"])


class TestHybridFusion(unittest.TestCase):
    def test_hybrid_fuses_dense_and_sparse(self):
        sparse = FakeSparseRetriever([(2, 3.0), (0, 1.0)])
        retriever, vector_db = _build_retriever(sparse_retriever=sparse, use_hybrid=True, use_rrf=True)
        self.assertTrue(retriever._uses_advanced_path)
        # Dense surfaces rows 0,1; sparse surfaces rows 2,0. Row 0 is shared.
        vector_db.search_vector_database.return_value = ([0, 1], [0.9, 0.8])

        records = retriever.retrieve("q", return_metadata=True)
        texts = [r["text"] for r in records]
        # Row 0 appears in both lists, so it fuses to the top.
        self.assertEqual(texts[0], "alpha chunk")
        # Every fused row is present exactly once (deduped).
        self.assertEqual(set(texts), {"alpha chunk", "beta chunk", "gamma chunk"})
        # Shared chunk keeps both signals as provenance.
        self.assertEqual(records[0]["source_signals"], ["dense", "sparse"])
        # rrf_score threaded through, ranks reassigned in fused order.
        self.assertIn("rrf_score", records[0])
        self.assertEqual([r["rank"] for r in records], [0, 1, 2])

    def test_rerank_keys_on_identity_not_text(self):
        # Two different chunks share identical text; the global rerank must not
        # collapse them into one record.
        vector_db = MagicMock(VectorDatabaseModule)
        vector_db.metadata = pd.DataFrame(
            [
                {DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
                {DOC_ID_KEY: 1, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "same text"},
            ]
        )
        reranker = MagicMock()
        reranker.rerank.return_value = [("same text", 5.0), ("same text", 4.0)]
        retriever = RetrieverModule(
            vector_database_module=vector_db,
            embedding_module=MagicMock(),
            top_k=5,
            use_reranker=False,
            use_rrf=True,
        )
        # Attach the mock reranker directly (the isinstance guard only runs when
        # use_reranker=True); the advanced path uses whatever ``reranker`` is set.
        retriever.reranker = reranker
        retriever.encode_query = lambda query: np.zeros(4)
        vector_db.search_vector_database.return_value = ([0, 1], [0.9, 0.8])

        records = retriever.retrieve("q", return_metadata=True)
        self.assertEqual(len(records), 2)
        # Both distinct (doc_id, chunk_id) identities survive.
        self.assertEqual({(r[DOC_ID_KEY], r[CHUNK_ID_KEY]) for r in records}, {(0, 0), (1, 0)})


class TestChunkReadExpansion(unittest.TestCase):
    def test_neighbor_window_expands_context(self):
        retriever, vector_db = _build_retriever(chunk_read=1)
        vector_db.search_vector_database.return_value = ([1], [0.9])

        records = retriever.retrieve("q", return_metadata=True)
        record = records[0]
        # Precise child identity/text preserved.
        self.assertEqual(record["child_text"], "beta chunk")
        # Expanded text pulls the +/-1 neighbor window within doc 0.
        self.assertIn("alpha chunk", record["text"])
        self.assertIn("beta chunk", record["text"])
        self.assertEqual(record["neighbor_chunk_ids"], [0, 1])

    def test_parent_store_expands_to_parent(self):
        parent_store = pd.DataFrame([{PARENT_ID_KEY: "p0", DOC_TEXT_KEY: "full parent context"}])
        retriever, vector_db = _build_retriever(chunk_read=0, parent_store=parent_store)
        vector_db.metadata = pd.DataFrame(
            [{DOC_ID_KEY: 0, CHUNK_ID_KEY: 0, DOC_TEXT_KEY: "child bit", PARENT_ID_KEY: "p0"}]
        )
        vector_db.search_vector_database.return_value = ([0], [0.9])

        records = retriever.retrieve("q", return_metadata=True)
        self.assertEqual(records[0]["child_text"], "child bit")
        self.assertIn("full parent context", records[0]["text"])


if __name__ == "__main__":
    unittest.main()
