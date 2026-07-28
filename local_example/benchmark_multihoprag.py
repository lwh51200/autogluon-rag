"""Benchmark Standard RAG vs. Agentic RAG on the MultiHop-RAG benchmark.

MultiHop-RAG (Tang & Yang, 2024) is a RAG-specific benchmark whose evidence is
distributed across 2-4 documents per query -- the multi-hop retrieval + reasoning
that separates agentic RAG from a single-shot static pipeline. Unlike Google NQ
(single-hop, self-contained rows), MultiHop-RAG ships two subsets:

* ``corpus``      -- the 609-article knowledge base (ingested + indexed once).
* ``MultiHopRAG`` -- 2,556 queries, each tagged with a ``question_type``.

Because corpus and queries are separate, this cannot go through
``EvaluationModule.run_evaluation`` unchanged (that path derives queries from the
same rows it ingests). So this runner ingests the corpus itself, then drives the
query subset through both modes, reusing ``EvaluationModule.evaluate_responses``
for the metric math. Results are broken down by ``question_type`` because the
static-vs-agentic gap concentrates in ``comparison_query`` and ``null_query``
(answer absent from the corpus) -- an aggregate number washes that out.

Environment note
----------------
Reuses ``local_example/local_config.yaml`` (MiniLM embeddings + Bedrock Claude
Haiku generator). For a no-credentials run, swap the generator in that config
back to a local HuggingFace model; absolute quality numbers only matter with a
real generator, but the pipeline wiring and per-type deltas are what this shows.
"""

import argparse
import json
import os
import time

from datasets import load_dataset

from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.multihop_rag.multihop_rag import (
    get_multihop_rag_evidence_facts,
    get_multihop_rag_query,
    get_multihop_rag_question_type,
    get_multihop_rag_responses,
    preprocess_multihop_rag_corpus,
)
from agrag.evaluation.evaluator import EvaluationModule
from agrag.evaluation.retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics_for_query

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

CONFIG = "local_example/local_config.yaml"
DATASET = "yixuantt/MultiHopRAG"
CORPUS_CONFIG = "corpus"
QUERY_CONFIG = "MultiHopRAG"
# Pure-Python exact-match metric (no model download); matches the NQ benchmark.
METRICS = ["inclusive_exact_match"]


class TimedRAG(AutoGluonRAG):
    """AutoGluonRAG that records per-call latency, bucketed by the current mode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bench_mode = "standard"
        self._bench_timings = {"standard": [], "agentic": []}

    def generate_response(self, query, mode=None, return_trace=None):
        start = time.perf_counter()
        result = super().generate_response(query, mode=mode, return_trace=return_trace)
        self._bench_timings[self._bench_mode].append(time.perf_counter() - start)
        return result


def _cost_summary(timings):
    n = len(timings)
    total = sum(timings)
    return {
        "num_queries_answered": n,
        "total_latency_s": round(total, 3),
        "avg_latency_s": round(total / n, 3) if n else 0.0,
    }


def _agentic_behavior_summary(agent_runs):
    """Summarize whether the agentic path actually decomposed queries.

    The evidence-coverage advantage over static RAG only materializes when the
    planner issues MULTIPLE retrieval calls / subqueries. If ``retrieval_calls``
    is ~1 across the board, agentic degenerated to a single-shot run and any
    score parity with standard is expected, not a benchmark failure. This makes
    that visible at a glance instead of hidden in the per-query traces.
    """
    n = len(agent_runs)
    if not n:
        return None
    avg_retrieval = sum(r["retrieval_calls"] for r in agent_runs) / n
    avg_subqueries = sum(r["num_subqueries"] for r in agent_runs) / n
    avg_iterations = sum(r["iterations"] for r in agent_runs) / n
    multi_step = sum(1 for r in agent_runs if r["retrieval_calls"] > 1)
    return {
        "avg_retrieval_calls": round(avg_retrieval, 2),
        "avg_num_subqueries": round(avg_subqueries, 2),
        "avg_iterations": round(avg_iterations, 2),
        "pct_multi_step": round(100.0 * multi_step / n, 1),
        "note": (
            "pct_multi_step near 0 means the planner did NOT decompose; "
            "agentic ~= single-shot and score parity with standard is expected."
        ),
    }


def ingest_corpus(evaluation_dir, max_docs=None):
    """Write the MultiHop-RAG corpus subset to .txt files for ingestion.

    The whole corpus is written regardless of how many queries are evaluated --
    evidence for any query may live in any document. ``max_docs`` is a smoke-test
    escape hatch only.
    """
    corpus_dir = os.path.join(evaluation_dir, "corpus")
    os.makedirs(corpus_dir, exist_ok=True)
    corpus = load_dataset(DATASET, name=CORPUS_CONFIG, split="train")
    n = 0
    for idx, row in enumerate(corpus):
        if max_docs and idx >= max_docs:
            break
        text = preprocess_multihop_rag_corpus(row)
        with open(os.path.join(corpus_dir, f"doc_{idx}.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        n += 1
    print(f"Wrote {n} corpus documents to {corpus_dir}")
    return corpus_dir


def _texts_from_records(records):
    """Pull ordered chunk texts from retriever records (already best-first)."""
    if not records:
        return []
    out = []
    for rec in records:
        if isinstance(rec, dict):
            out.append(rec.get("text", ""))
        else:
            out.append(str(rec))
    return out


def retrieved_texts_for_query(agrag, query, mode):
    """Return the ranked chunk texts a given mode surfaced for ``query``.

    Standard mode: a single retrieval call (what the generator actually saw).
    Agentic mode: the union of evidence accumulated across all sub-query
    retrieval rounds, ordered by the trace's evidence list -- this is the
    multi-round retrieval whose coverage should beat the single-shot path.

    Returns ``(texts, generated_answer, agent_metrics)`` so the caller reuses the
    same run for answer-quality, retrieval scoring, and decomposition signals (no
    double generation). ``agent_metrics`` is ``None`` for the standard path and,
    for the agentic path, the trace's per-run metrics (retrieval_calls,
    subqueries count, iterations) that reveal whether the planner actually
    decomposed the query -- if not, agentic degenerates to a single-shot run.
    """
    if mode == "agentic":
        answer, trace = agrag.generate_response(query, mode="agentic", return_trace=True)
        trace = trace if isinstance(trace, dict) else {}
        evidence = trace.get("evidence", [])
        # Evidence is already stored best-first per retrieval; keep that order.
        texts = [ev.get("text", "") for ev in evidence]
        tmetrics = trace.get("metrics", {}) or {}
        agent_metrics = {
            "retrieval_calls": tmetrics.get("retrieval_calls", 0),
            "num_subqueries": len(trace.get("subqueries", []) or []),
            "iterations": tmetrics.get("iterations", 0),
        }
        return texts, answer, agent_metrics

    # Standard path: retrieve exactly what the generator will condition on, then
    # generate. retrieve() returns None when nothing matched.
    records = agrag.retriever_module.retrieve(query, return_metadata=True)
    texts = _texts_from_records(records)
    answer = agrag.generate_response(query, mode=None)
    return texts, answer, None


def build_evaluator(agrag):
    """An EvaluationModule with metrics initialized, ready for evaluate_responses."""
    evaluator = EvaluationModule(rag_instance=agrag)
    evaluator.metrics = METRICS
    evaluator.metric_init_params = {}
    evaluator.metric_score_params = {}
    evaluator.metric_instances = evaluator.initialize_metrics(METRICS)
    return evaluator


def run_mode(agrag, evaluator, queries_ds, mode, max_eval_size):
    """Run one evaluation pass; return overall + per-question-type metrics."""
    label = mode or "standard"
    agrag._bench_mode = label
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG on MultiHop-RAG  (max_eval_size={max_eval_size})")
    print("=" * 72)

    # Bucket by question_type so we can score each type separately and aggregate.
    # Each bucket holds answer-quality inputs plus per-query retrieval metrics.
    buckets = {}
    all_preds, all_refs, all_queries = [], [], []
    all_retrieval = []  # per-query retrieval metrics, non-null queries only
    agent_runs = []  # per-query agentic decomposition signals (agentic mode only)
    for idx, row in enumerate(queries_ds):
        if max_eval_size and idx >= max_eval_size:
            break
        expected = get_multihop_rag_responses(row)
        if not expected:
            continue
        query = get_multihop_rag_query(row)
        qtype = get_multihop_rag_question_type(row)
        gold_facts = get_multihop_rag_evidence_facts(row)

        retrieved_texts, generated, agent_metrics = retrieved_texts_for_query(agrag, query, mode)
        if agent_metrics is not None:
            agent_runs.append(agent_metrics)

        b = buckets.setdefault(qtype, {"preds": [], "refs": [], "queries": [], "retrieval": []})
        b["preds"].append(generated)
        b["refs"].append(expected)
        b["queries"].append(query)
        all_preds.append(generated)
        all_refs.append(expected)
        all_queries.append(query)

        # Retrieval scoring needs gold facts; null_query rows have none, so they
        # are excluded from retrieval metrics (undefined) but still answer-scored.
        if gold_facts:
            rmetrics = retrieval_metrics_for_query(retrieved_texts, gold_facts)
            b["retrieval"].append(rmetrics)
            all_retrieval.append(rmetrics)

    overall = evaluator.evaluate_responses(predictions=all_preds, references=all_refs, queries=all_queries)
    per_type = {}
    for qtype, b in sorted(buckets.items()):
        per_type[qtype] = {
            "count": len(b["preds"]),
            "quality": evaluator.evaluate_responses(
                predictions=b["preds"], references=b["refs"], queries=b["queries"]
            ),
            "retrieval": aggregate_retrieval_metrics(b["retrieval"]),
        }

    result = {
        "quality_overall": overall,
        "retrieval_overall": aggregate_retrieval_metrics(all_retrieval),
        "quality_by_question_type": per_type,
        "cost": _cost_summary(agrag._bench_timings[label]),
    }
    behavior = _agentic_behavior_summary(agent_runs)
    if behavior is not None:
        result["agentic_behavior"] = behavior
        print(f"\nAgentic decomposition: {json.dumps(behavior)}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-eval-size", type=int, default=20, help="Number of MultiHop-RAG queries to evaluate (default: 20)."
    )
    parser.add_argument(
        "--max-corpus-docs", type=int, default=None, help="Cap ingested corpus docs (default: all 609). Smoke-test only."
    )
    parser.add_argument(
        "--evaluation-dir", default="local_example/evaluation_data_multihoprag", help="Where corpus docs are written."
    )
    args = parser.parse_args()

    corpus_dir = ingest_corpus(args.evaluation_dir, max_docs=args.max_corpus_docs)

    # One instance -> same ingest, index, embedder, generator for both systems.
    agrag = TimedRAG(config_file=CONFIG, data_dir=corpus_dir)
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)
    queries_ds = load_dataset(DATASET, name=QUERY_CONFIG, split="train")

    results = {}
    results["standard"] = run_mode(agrag, evaluator, queries_ds, mode=None, max_eval_size=args.max_eval_size)
    results["agentic"] = run_mode(agrag, evaluator, queries_ds, mode="agentic", max_eval_size=args.max_eval_size)

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, identical corpus + settings)")
    print("=" * 72)
    print(json.dumps(results, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    os.makedirs(args.evaluation_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
