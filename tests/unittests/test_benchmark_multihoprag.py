"""Offline tests for the MultiHop-RAG benchmark runner.

These use in-memory fakes only (no dataset download, no models, no network) to
prove the correctness/reproducibility guarantees the benchmark must uphold:

* standard mode performs exactly ONE retrieval/generation run per query,
* both modes evaluate the IDENTICAL, reproducible set of dataset rows,
* the JSONL trace/evidence is consistent with what the run returned,
* the emitted JSONL is one valid JSON object per query with the required fields.
"""

import importlib.util
import json
import os
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_PATH = os.path.join(REPO_ROOT, "local_example", "benchmark_multihoprag.py")


def _load_bench():
    """Import the benchmark module by file path (local_example isn't a package)."""
    spec = importlib.util.spec_from_file_location("benchmark_multihoprag", BENCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_bench()


class FakeRetriever:
    """Counts retrieve() calls; returns canned best-first structured records."""

    def __init__(self, top_k=3):
        self.top_k = top_k
        self.calls = 0

    def retrieve(self, query, return_metadata=False):
        self.calls += 1
        return [
            {"text": f"{query}::chunk0", "doc_id": 0, "chunk_id": 0, "rank": 0, "retrieval_score": 0.9},
            {"text": f"{query}::chunk1", "doc_id": 1, "chunk_id": 0, "rank": 1, "retrieval_score": 0.8},
        ]


class FakeAgrag:
    """Minimal stand-in for AutoGluonRAG.generate_response used by the benchmark.

    Standard mode goes through the retriever exactly once and returns a standard
    trace mirroring the real ``_standard_trace`` shape. Agentic mode returns a
    trace with decomposition metrics but does NOT touch this retriever (so the
    "one retrieval per standard query" assertion is unambiguous).
    """

    def __init__(self):
        self.retriever_module = FakeRetriever()

    def generate_response(self, query, mode="standard", return_trace=None):
        if mode == "agentic":
            trace = {
                "mode": "agentic",
                "original_query": query,
                "final_answer": f"agentic:{query}",
                "evidence": [{"text": f"{query}::ev", "doc_id": 0, "chunk_id": 0, "rank": 0}],
                "subqueries": [f"{query} a", f"{query} b"],
                "metrics": {"retrieval_calls": 2, "iterations": 1},
            }
            return f"agentic:{query}", trace
        records = self.retriever_module.retrieve(query, return_metadata=True)
        trace = {
            "mode": "standard",
            "original_query": query,
            "final_answer": f"standard:{query}",
            "evidence": [dict(r) for r in records],
            "metrics": {"retrieval_calls": 1, "evidence_count": len(records)},
        }
        return f"standard:{query}", trace


class FakeEvaluator:
    """evaluate_responses returns a trivial deterministic metric dict."""

    def evaluate_responses(self, predictions, references, queries):
        return {"inclusive_exact_match": float(len(predictions))}


def _fake_queries():
    # Mixed question types; one null_query (no answer facts) and one with no
    # answer at all (ineligible -> must be excluded from selection).
    return [
        {"query": "q0", "answer": "a0", "question_type": "inference_query",
         "evidence_list": [{"fact": "q0::chunk0"}]},
        {"query": "q1", "answer": "a1", "question_type": "comparison_query",
         "evidence_list": [{"fact": "zzz"}]},
        {"query": "q2", "answer": "", "question_type": "inference_query", "evidence_list": []},
        {"query": "q3", "answer": "a3", "question_type": "temporal_query",
         "evidence_list": [{"fact": "q3::chunk1"}]},
        {"query": "q4", "answer": "a4", "question_type": "null_query", "evidence_list": []},
    ]


class TestSelectQueryIndices(unittest.TestCase):
    def test_excludes_ineligible_and_returns_first_n_in_order(self):
        ds = _fake_queries()
        # q2 has no answer -> ineligible. First 2 eligible in dataset order: 0,1.
        idx = bench.select_query_indices(ds, max_eval_size=2, seed=1234, stratify=False)
        self.assertEqual(idx, [0, 1])

    def test_selection_is_reproducible_across_modes(self):
        ds = _fake_queries()
        a = bench.select_query_indices(ds, max_eval_size=3, seed=7, stratify=True)
        b = bench.select_query_indices(ds, max_eval_size=3, seed=7, stratify=True)
        # Same seed -> identical selection; this is the SAME list passed to both
        # modes, so the comparison is paired by construction.
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        # Never includes the ineligible row (index 2, empty answer).
        self.assertNotIn(2, a)

    def test_stratified_covers_multiple_question_types(self):
        ds = _fake_queries()
        idx = bench.select_query_indices(ds, max_eval_size=3, seed=1234, stratify=True)
        qtypes = {ds[i]["question_type"] for i in idx}
        self.assertGreaterEqual(len(qtypes), 2)

    def test_returns_all_eligible_when_quota_exceeds(self):
        ds = _fake_queries()
        idx = bench.select_query_indices(ds, max_eval_size=100, seed=1234, stratify=False)
        # All eligible rows (0,1,3,4); index 2 excluded.
        self.assertEqual(idx, [0, 1, 3, 4])


class TestRunQuerySingleRetrieval(unittest.TestCase):
    def test_standard_run_performs_exactly_one_retrieval(self):
        agrag = FakeAgrag()
        run = bench.run_query(agrag, "q0", mode=None)
        self.assertEqual(agrag.retriever_module.calls, 1)
        self.assertEqual(run["answer"], "standard:q0")
        self.assertEqual(run["evidence_texts"], ["q0::chunk0", "q0::chunk1"])
        self.assertIsNone(run["agent_metrics"])
        self.assertGreaterEqual(run["latency"], 0.0)

    def test_agentic_run_exposes_decomposition_metrics(self):
        agrag = FakeAgrag()
        run = bench.run_query(agrag, "q0", mode="agentic")
        self.assertEqual(run["agent_metrics"]["retrieval_calls"], 2)
        self.assertEqual(run["agent_metrics"]["num_subqueries"], 2)


class TestRunModeJsonlAndConsistency(unittest.TestCase):
    def test_jsonl_rows_valid_and_consistent_with_run(self):
        agrag = FakeAgrag()
        evaluator = FakeEvaluator()
        ds = _fake_queries()
        selected = bench.select_query_indices(ds, max_eval_size=3, seed=1234, stratify=False)

        rows = []
        result = bench.run_mode(
            agrag, evaluator, ds, mode=None, selected_indices=selected, jsonl_writer=rows.append
        )

        # One JSONL row per selected, eligible query.
        self.assertEqual(len(rows), len(selected))
        # One retrieval per standard query, no more.
        self.assertEqual(agrag.retriever_module.calls, len(selected))

        for row in rows:
            # Row is JSON-serializable and round-trips.
            reloaded = json.loads(json.dumps(row, default=str))
            for field in (
                "mode", "dataset_index", "question_type", "query", "references",
                "prediction", "evidence_texts", "evidence_provenance",
                "retrieval_metrics", "latency_s", "trace",
            ):
                self.assertIn(field, reloaded)
            # Evidence texts in the JSONL match the trace's evidence (consistency).
            trace_texts = [e["text"] for e in reloaded["trace"]["evidence"]]
            self.assertEqual(reloaded["evidence_texts"], trace_texts)
            # Prediction matches the trace's final answer.
            self.assertEqual(reloaded["prediction"], reloaded["trace"]["final_answer"])
            self.assertEqual(reloaded["mode"], "standard")

        # Dataset indices recorded in the rows are exactly the selected rows.
        self.assertEqual([r["dataset_index"] for r in rows], selected)
        self.assertIn("cost", result)
        self.assertEqual(result["cost"]["num_queries_answered"], len(selected))

    def test_both_modes_evaluate_identical_rows(self):
        agrag = FakeAgrag()
        evaluator = FakeEvaluator()
        ds = _fake_queries()
        selected = bench.select_query_indices(ds, max_eval_size=3, seed=1234, stratify=True)

        std_rows, agt_rows = [], []
        bench.run_mode(agrag, evaluator, ds, mode=None, selected_indices=selected, jsonl_writer=std_rows.append)
        bench.run_mode(agrag, evaluator, ds, mode="agentic", selected_indices=selected, jsonl_writer=agt_rows.append)

        self.assertEqual(
            [r["dataset_index"] for r in std_rows],
            [r["dataset_index"] for r in agt_rows],
        )
        self.assertEqual({r["mode"] for r in std_rows}, {"standard"})
        self.assertEqual({r["mode"] for r in agt_rows}, {"agentic"})

    def test_jsonl_file_is_line_delimited_json(self):
        agrag = FakeAgrag()
        evaluator = FakeEvaluator()
        ds = _fake_queries()
        selected = bench.select_query_indices(ds, max_eval_size=2, seed=1234, stratify=False)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "preds.jsonl")
            with open(path, "w") as fh:
                bench.run_mode(
                    agrag, evaluator, ds, mode=None, selected_indices=selected,
                    jsonl_writer=lambda row: fh.write(json.dumps(row, default=str) + "\n"),
                )
            with open(path) as fh:
                lines = [ln for ln in fh.read().splitlines() if ln]
            self.assertEqual(len(lines), len(selected))
            for ln in lines:
                json.loads(ln)  # each line parses independently


if __name__ == "__main__":
    unittest.main()
