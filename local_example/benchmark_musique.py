"""Benchmark Standard RAG vs. Agentic RAG on the MuSiQue multi-hop QA benchmark.

MuSiQue (Trivedi et al., 2021 -- arXiv:2108.00573; HuggingFace mirror
``dgslibisey/MuSiQue``) builds multi-hop questions by *composing* single-hop
questions, so answering requires 2-4 connected reasoning hops. Its defining
feature for RAG is that each question ships its OWN ~20-paragraph pool -- a few
supporting paragraphs plus distractors -- rather than one shared corpus. This is
the paper's "distractor" setting: the retriever must find the supporting
paragraphs among plausible distractors, and a single-shot retrieve-then-read
pipeline is expected to lose ground to a multi-round agentic one as the hop count
grows.

How this differs from ``benchmark_multihoprag.py``
--------------------------------------------------
MultiHop-RAG has one 609-article corpus indexed ONCE. MuSiQue's corpus is
per-question, so this runner **re-indexes each question's own paragraphs** before
querying it. To avoid reloading the embedding / reranker / generator models 30+
times, it builds those modules once (via ``initialize_rag_pipeline`` on the first
question) and then, for every subsequent question, resets and rebuilds only the
vector index + BM25 index + parent store over that question's paragraphs
(``reindex_question``). Both modes (standard and agentic) then run over the
identical per-question index through the same ``generate_response(..., mode=...)``
entry point -- the agentic workflow itself is measured as-is, unmodified.

Environment note
----------------
Reuses ``local_example/local_config.yaml`` (MiniLM embeddings + Bedrock Claude
Haiku generator). Source ``credential.sh`` for Bedrock access before running. The
config's saved-index paths are NOT written to: this runner overrides the vector-DB
save/load flags in memory so per-question indexes never touch disk.
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import time

from datasets import load_dataset

from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.musique.musique import (
    get_musique_answerable,
    get_musique_evidence_facts,
    get_musique_paragraph_docs,
    get_musique_query,
    get_musique_question_type,
    get_musique_responses,
)
from agrag.evaluation.evaluator import EvaluationModule
from agrag.evaluation.retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics_for_query
from agrag.evaluation.utils import calculate_f1_score, f1_metric

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

CONFIG = "local_example/local_config.yaml"
DATASET = "dgslibisey/MuSiQue"
DEFAULT_SPLIT = "validation"
# Answer-quality metric computed via the evaluator (pure-Python, no model download).
# Matches the MultiHop-RAG / NQ benchmarks; token-F1 is computed separately below.
EM_METRIC = "inclusive_exact_match"
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
    is ~1 across the board, agentic degenerated to a single-shot run and any score
    parity with standard is expected, not a benchmark failure. This makes that
    visible at a glance instead of hidden in the per-query traces.
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


def _quality_scores(evaluator, predictions, references, queries):
    """Answer-quality metrics for one set of (prediction, references) pairs.

    Exact-match goes through the shared ``EvaluationModule`` (inclusive EM), and
    token-level F1 -- MuSiQue's official answer metric -- is computed directly via
    ``f1_metric``/``calculate_f1_score`` so it is aggregated to a mean (the
    evaluator's callable-metric path would return the raw per-example list).
    """
    if not predictions:
        return {EM_METRIC: 0.0, "f1": 0.0, "count": 0}
    em = evaluator.evaluate_responses(predictions=predictions, references=references, queries=queries)
    f1 = calculate_f1_score(f1_metric(predictions, references))
    return {**em, "f1": round(f1, 4), "count": len(predictions)}


def reindex_question(agrag, paragraph_docs, work_dir):
    """Rebuild the pipeline's index over ONE question's paragraph pool.

    Reuses the already-loaded embedding / reranker / retriever / generator modules
    and rebuilds only the corpus-dependent state, mirroring the non-batched branch
    of ``initialize_rag_pipeline`` (agrag.py). Three pieces of per-corpus state
    must be reset first or evidence would leak across questions:

    * ``vector_db_module`` appends on ``construct_vector_database`` (pd.concat +
      index.add), so its FAISS index and metadata are cleared.
    * the BM25 ``sparse_retriever`` caches a ``_built`` flag and is only rebuilt
      when that flag is False.
    * the retriever's parent-text cache is reset by ``_attach_parent_store_to_retriever``.

    Then paragraphs are written to fresh ``.txt`` files, the data module is pointed
    at them, and ``process_data -> generate_embeddings -> construct_vector_db ->
    attach parent store`` runs -- exactly the steps the initializer uses.
    """
    # Fresh corpus dir for this question.
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    for i, doc in enumerate(paragraph_docs):
        with open(os.path.join(work_dir, f"para_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(doc + "\n")

    # Reset per-corpus state so nothing leaks from the previous question.
    agrag.vector_db_module.index = None
    agrag.vector_db_module.metadata = agrag.vector_db_module.metadata.iloc[0:0]
    if getattr(agrag.retriever_module, "sparse_retriever", None) is not None:
        agrag.retriever_module.sparse_retriever._built = False
    agrag.data_processing_module.data_dir = work_dir

    processed_data = agrag.process_data()
    embeddings = agrag.generate_embeddings(processed_data=processed_data)
    agrag.construct_vector_db(embeddings=embeddings)
    agrag._attach_parent_store_to_retriever()


def run_query(agrag, query, mode):
    """Execute ONE comparable run for ``query`` in ``mode`` and time all of it.

    Both modes go through a single ``generate_response(..., return_trace=True)``
    call, so exactly one retrieval + generation happens per standard query. The
    whole comparable operation -- retrieval and generation -- is timed for both.
    (The index is already built for the current question by ``reindex_question``;
    that build is timed separately and not attributed to either mode.)

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


def select_query_indices(queries_ds, max_eval_size, seed, answerable_only=False, stratify=False):
    """Select the query-row indices to evaluate, reproducibly.

    The SAME indices are used for both modes so the comparison is paired. Only rows
    with a non-empty expected answer are eligible (the metrics need a reference);
    with ``answerable_only`` the unanswerable rows are also dropped. Selection is
    deterministic given ``seed``.

    stratify=False : first ``max_eval_size`` eligible rows in dataset order.
    stratify=True  : reproducible stratified sample by hop-count question type --
        eligible rows are bucketed by type, each bucket shuffled with the fixed
        seed, then round-robined so every hop count is represented proportionally.

    Returns a sorted list of dataset indices.
    """
    eligible = []
    for idx, row in enumerate(queries_ds):
        if not get_musique_responses(row):
            continue
        if answerable_only and not get_musique_answerable(row):
            continue
        eligible.append(idx)

    if not max_eval_size or max_eval_size >= len(eligible):
        return eligible

    if not stratify:
        return eligible[:max_eval_size]

    buckets = {}
    for idx in eligible:
        qtype = get_musique_question_type(queries_ds[idx])
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
    """An EvaluationModule with the EM metric initialized, ready for evaluate_responses."""
    evaluator = EvaluationModule(rag_instance=agrag)
    evaluator.metrics = [EM_METRIC]
    evaluator.metric_init_params = {}
    evaluator.metric_score_params = {}
    evaluator.metric_instances = evaluator.initialize_metrics([EM_METRIC])
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


def run_mode(agrag, evaluator, queries_ds, mode, selected_indices, corpus_dir, jsonl_writer=None):
    """Run one evaluation pass over the pre-selected rows.

    ``selected_indices`` is the shared, reproducible set of dataset indices used
    for BOTH modes (paired comparison). For each question the per-question index is
    rebuilt (over that question's paragraphs) before querying. ``jsonl_writer`` is
    an optional callable receiving one dict per query, written as a JSONL row.
    Returns overall + per-hop-type metrics plus the per-query latencies.
    """
    label = mode or "standard"
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG on MuSiQue  (n={len(selected_indices)})")
    print("=" * 72)

    # Bucket by hop-count question type so we can score each type separately.
    buckets = {}
    all_preds, all_refs, all_queries = [], [], []
    all_retrieval = []  # per-query retrieval metrics, rows with gold facts only
    agent_runs = []  # per-query agentic decomposition signals (agentic mode only)
    latencies = []
    for idx in selected_indices:
        row = queries_ds[idx]
        expected = get_musique_responses(row)
        if not expected:
            continue
        query = get_musique_query(row)
        qtype = get_musique_question_type(row)
        gold_facts = get_musique_evidence_facts(row)

        # Rebuild the index over THIS question's paragraph pool before querying.
        reindex_question(agrag, get_musique_paragraph_docs(row), corpus_dir)

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

        # Retrieval scoring needs gold facts; unanswerable rows may have none, so
        # they are excluded from retrieval metrics (undefined) but still answer-scored.
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
                    "answerable": get_musique_answerable(row),
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

    overall = _quality_scores(evaluator, all_preds, all_refs, all_queries)
    per_type = {}
    for qtype, b in sorted(buckets.items()):
        per_type[qtype] = {
            "count": len(b["preds"]),
            "quality": _quality_scores(evaluator, b["preds"], b["refs"], b["queries"]),
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
        "--max-eval-size", type=int, default=30, help="Number of MuSiQue questions to evaluate (default: 30)."
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help=f"MuSiQue split to evaluate (default: {DEFAULT_SPLIT})."
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Evaluate only answerable questions (drops rows where the answer is absent from the paragraphs).",
    )
    parser.add_argument(
        "--evaluation-dir", default="local_example/evaluation_data_musique", help="Where results are written."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Seed for reproducible query selection (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--stratify",
        action="store_true",
        help="Stratified sampling by hop-count question type (default: first-N eligible rows in dataset order).",
    )
    args = parser.parse_args()

    queries_ds = load_dataset(DATASET, split=args.split)

    # Pick the query rows ONCE and reuse the identical set for both modes so the
    # comparison is paired and reproducible under the fixed seed.
    selected_indices = select_query_indices(
        queries_ds, args.max_eval_size, seed=args.seed, answerable_only=args.answerable_only, stratify=args.stratify
    )
    print(
        f"Selected {len(selected_indices)} MuSiQue questions from split '{args.split}' "
        f"(seed={args.seed}, answerable_only={args.answerable_only}, stratify={args.stratify}); "
        f"same rows used for both modes."
    )
    if not selected_indices:
        raise SystemExit("No eligible questions selected; nothing to evaluate.")

    # Per-question corpus dir (rewritten before each question by reindex_question).
    corpus_dir = os.path.join(tempfile.gettempdir(), "musique_question_corpus")

    # One instance -> same embedder, reranker, retriever, generator for both
    # systems. Initialize on the FIRST selected question's paragraphs so the
    # models load exactly once; every later question reuses them via reindex.
    first_docs = get_musique_paragraph_docs(queries_ds[selected_indices[0]])
    if os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    for i, doc in enumerate(first_docs):
        with open(os.path.join(corpus_dir, f"para_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(doc + "\n")

    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    # Never load or persist a shared index: each question is indexed fresh in
    # memory. Overriding here (not in the yaml) keeps local_config.yaml untouched.
    agrag.args.use_existing_vector_db = False
    agrag.args.save_vector_db_index = False
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)

    os.makedirs(args.evaluation_dir, exist_ok=True)
    jsonl_path = os.path.join(args.evaluation_dir, "benchmark_predictions.jsonl")

    results = {}
    with open(jsonl_path, "w") as jf:
        def jsonl_writer(row):
            jf.write(json.dumps(row, default=str) + "\n")

        results["standard"] = run_mode(
            agrag, evaluator, queries_ds, mode=None, selected_indices=selected_indices,
            corpus_dir=corpus_dir, jsonl_writer=jsonl_writer,
        )
        results["agentic"] = run_mode(
            agrag, evaluator, queries_ds, mode="agentic", selected_indices=selected_indices,
            corpus_dir=corpus_dir, jsonl_writer=jsonl_writer,
        )
    print(f"\nSaved per-query predictions to {jsonl_path}")

    results["selection"] = {
        "dataset": DATASET,
        "split": args.split,
        "seed": args.seed,
        "answerable_only": args.answerable_only,
        "stratify": args.stratify,
        "num_selected": len(selected_indices),
        "dataset_indices": selected_indices,
    }

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, identical per-question corpus + settings)")
    print("=" * 72)
    print(json.dumps({k: v for k, v in results.items() if k != "selection"}, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
