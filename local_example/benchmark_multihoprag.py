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
import random
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
# Fixed seed for reproducible query-row selection (shared across both modes).
DEFAULT_SEED = 1234


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


def run_query(agrag, query, mode):
    """Execute ONE comparable run for ``query`` in ``mode`` and time all of it.

    Both modes now go through a single ``generate_response(..., return_trace=True)``
    call, so exactly one retrieval + generation happens per standard query (the
    previous code retrieved once for scoring and again inside generation). The
    whole comparable operation -- retrieval and generation -- is timed for both.

    Returns a dict with the answer, the ranked evidence texts the answer was
    conditioned on (best-first), the full serializable trace, per-query latency,
    and agentic decomposition metrics (``None`` for standard).
    """
    resolved = mode or "standard"
    start = time.perf_counter()
    answer, trace = agrag.generate_response(query, mode=resolved, return_trace=True)
    latency = time.perf_counter() - start

    trace = trace if isinstance(trace, dict) else {}
    evidence = trace.get("evidence", []) or []
    # Evidence is stored best-first per retrieval in both modes; keep that order.
    texts = [ev.get("text", "") for ev in evidence]

    agent_metrics = None
    if resolved == "agentic":
        tmetrics = trace.get("metrics", {}) or {}
        agent_metrics = {
            "retrieval_calls": tmetrics.get("retrieval_calls", 0),
            "num_subqueries": len(trace.get("subqueries", []) or []),
            "iterations": tmetrics.get("iterations", 0),
        }

    return {
        "answer": answer,
        "evidence_texts": texts,
        "evidence": evidence,
        "trace": trace,
        "latency": latency,
        "agent_metrics": agent_metrics,
    }


def select_query_indices(queries_ds, max_eval_size, seed, stratify=False):
    """Select the query-row indices to evaluate, reproducibly.

    The SAME indices are used for both modes so the comparison is paired. Only
    rows with a non-empty expected answer are eligible (the metrics need a
    reference). Selection is deterministic given ``seed``.

    stratify=False : first ``max_eval_size`` eligible rows in dataset order
        (stable, and identical to the legacy "first N" behavior).
    stratify=True  : reproducible stratified sample by ``question_type`` --
        eligible rows are bucketed by type, each bucket shuffled with the fixed
        seed, then round-robined so every type is represented proportionally.

    Returns a sorted list of dataset indices.
    """
    eligible = [idx for idx, row in enumerate(queries_ds) if get_multihop_rag_responses(row)]
    if not max_eval_size or max_eval_size >= len(eligible):
        return eligible

    if not stratify:
        return eligible[:max_eval_size]

    buckets = {}
    for idx in eligible:
        qtype = get_multihop_rag_question_type(queries_ds[idx])
        buckets.setdefault(qtype, []).append(idx)

    rng = random.Random(seed)
    for qtype in sorted(buckets):
        rng.shuffle(buckets[qtype])

    # Round-robin across types (sorted for determinism) until the quota is met.
    ordered_types = sorted(buckets)
    selected = []
    position = 0
    while len(selected) < max_eval_size:
        progressed = False
        for qtype in ordered_types:
            bucket = buckets[qtype]
            if position < len(bucket):
                selected.append(bucket[position])
                progressed = True
                if len(selected) >= max_eval_size:
                    break
        if not progressed:
            break
        position += 1
    return sorted(selected)


def build_evaluator(agrag):
    """An EvaluationModule with metrics initialized, ready for evaluate_responses."""
    evaluator = EvaluationModule(rag_instance=agrag)
    evaluator.metrics = METRICS
    evaluator.metric_init_params = {}
    evaluator.metric_score_params = {}
    evaluator.metric_instances = evaluator.initialize_metrics(METRICS)
    return evaluator


def _evidence_provenance(evidence):
    """Compact per-chunk provenance for the JSONL row (drops bulky embeddings)."""
    provenance = []
    for rank, ev in enumerate(evidence):
        provenance.append(
            {
                "rank": ev.get("rank", rank),
                "doc_id": ev.get("doc_id"),
                "chunk_id": ev.get("chunk_id"),
                "source": ev.get("source"),
                "retrieval_score": ev.get("retrieval_score"),
                "rerank_score": ev.get("rerank_score"),
            }
        )
    return provenance


def run_mode(agrag, evaluator, queries_ds, mode, selected_indices, jsonl_writer=None):
    """Run one evaluation pass over the pre-selected rows.

    ``selected_indices`` is the shared, reproducible set of dataset indices used
    for BOTH modes (paired comparison). ``jsonl_writer`` is an optional callable
    receiving one dict per query, written as a JSONL row. Returns overall +
    per-question-type metrics plus the per-query latencies gathered here.
    """
    label = mode or "standard"
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG on MultiHop-RAG  (n={len(selected_indices)})")
    print("=" * 72)

    # Bucket by question_type so we can score each type separately and aggregate.
    buckets = {}
    all_preds, all_refs, all_queries = [], [], []
    all_retrieval = []  # per-query retrieval metrics, non-null queries only
    agent_runs = []  # per-query agentic decomposition signals (agentic mode only)
    latencies = []
    for idx in selected_indices:
        row = queries_ds[idx]
        expected = get_multihop_rag_responses(row)
        if not expected:
            continue
        query = get_multihop_rag_query(row)
        qtype = get_multihop_rag_question_type(row)
        gold_facts = get_multihop_rag_evidence_facts(row)

        run = run_query(agrag, query, mode)
        latencies.append(run["latency"])
        if run["agent_metrics"] is not None:
            agent_runs.append(run["agent_metrics"])

        b = buckets.setdefault(qtype, {"preds": [], "refs": [], "queries": [], "retrieval": []})
        b["preds"].append(run["answer"])
        b["refs"].append(expected)
        b["queries"].append(query)
        all_preds.append(run["answer"])
        all_refs.append(expected)
        all_queries.append(query)

        # Retrieval scoring needs gold facts; null_query rows have none, so they
        # are excluded from retrieval metrics (undefined) but still answer-scored.
        rmetrics = {}
        if gold_facts:
            rmetrics = retrieval_metrics_for_query(run["evidence_texts"], gold_facts)
            b["retrieval"].append(rmetrics)
            all_retrieval.append(rmetrics)

        if jsonl_writer is not None:
            jsonl_writer(
                {
                    "mode": label,
                    "dataset_index": idx,
                    "question_type": qtype,
                    "query": query,
                    "references": expected,
                    "prediction": run["answer"],
                    "evidence_texts": run["evidence_texts"],
                    "evidence_provenance": _evidence_provenance(run["evidence"]),
                    "retrieval_metrics": rmetrics,
                    "latency_s": round(run["latency"], 4),
                    "agent_metrics": run["agent_metrics"],
                    "trace": run["trace"],
                }
            )

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
        "cost": _cost_summary(latencies),
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
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Seed for reproducible query selection (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratified sampling by question_type (default: first-N eligible rows in dataset order).",
    )
    args = parser.parse_args()

    corpus_dir = ingest_corpus(args.evaluation_dir, max_docs=args.max_corpus_docs)

    # One instance -> same ingest, index, embedder, generator for both systems.
    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)
    queries_ds = load_dataset(DATASET, name=QUERY_CONFIG, split="train")

    # Pick the query rows ONCE and reuse the identical set for both modes so the
    # comparison is paired and reproducible under the fixed seed.
    selected_indices = select_query_indices(
        queries_ds, args.max_eval_size, seed=args.seed, stratify=args.stratify
    )
    print(
        f"Selected {len(selected_indices)} query rows "
        f"(seed={args.seed}, stratify={args.stratify}); same rows used for both modes."
    )

    os.makedirs(args.evaluation_dir, exist_ok=True)
    jsonl_path = os.path.join(args.evaluation_dir, "benchmark_predictions.jsonl")

    results = {}
    with open(jsonl_path, "w") as jf:
        def jsonl_writer(row):
            jf.write(json.dumps(row, default=str) + "\n")

        results["standard"] = run_mode(
            agrag, evaluator, queries_ds, mode=None, selected_indices=selected_indices, jsonl_writer=jsonl_writer
        )
        results["agentic"] = run_mode(
            agrag, evaluator, queries_ds, mode="agentic", selected_indices=selected_indices, jsonl_writer=jsonl_writer
        )
    print(f"\nSaved per-query predictions to {jsonl_path}")

    results["selection"] = {
        "seed": args.seed,
        "stratify": args.stratify,
        "num_selected": len(selected_indices),
        "dataset_indices": selected_indices,
    }

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, identical corpus + settings)")
    print("=" * 72)
    print(json.dumps({k: v for k, v in results.items() if k != "selection"}, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
