"""Benchmark Standard RAG vs. Agentic RAG on the MuSiQue multi-hop QA benchmark.

MuSiQue (Trivedi et al., 2021 -- arXiv:2108.00573; HuggingFace mirror
``dgslibisey/MuSiQue``) builds multi-hop questions by *composing* single-hop
questions, so answering requires 2-4 connected reasoning hops. Its defining
feature for RAG is that each question ships its OWN ~20-paragraph pool -- a few
supporting paragraphs plus distractors. In the paper's "distractor" setting each
question is answered against only its own pool.

Global merged corpus (this runner)
----------------------------------
To make retrieval closer to real RAG, this runner does NOT use the per-question
pools in isolation: it **merges every selected question's paragraphs into ONE
deduplicated global corpus, indexed once**, and answers every question against
that whole corpus (``build_global_corpus``). A question's supporting paragraphs
must therefore be found among ALL questions' paragraphs, not just its own 20 --
a harder, more realistic global-retrieval task, and deliberately no longer the
paper's official distractor benchmark. Identical Wikipedia paragraphs shared
across questions are written only once.

How this differs from ``benchmark_multihoprag.py``
--------------------------------------------------
MultiHop-RAG has one 609-article corpus indexed ONCE; this runner likewise builds
a single corpus and index up front (via ``initialize_rag_pipeline`` over the fully
populated global corpus dir), so the embedding / reranker / generator models load
once and no per-question re-indexing happens. Both modes (standard and agentic)
run over the identical global index through the same
``generate_response(..., mode=...)`` entry point -- the agentic workflow itself is
measured as-is, unmodified. Note: total corpus size grows with the number of
questions, so very large samples will be slow (see ``build_global_corpus``).

The agentic run here uses the **LLM planner + policy** (the shared Bedrock
generator decomposes the query into subqueries and chooses the next action among
the legal set), not the deterministic regex planner / rule-based action cascade.
These flags are set in ``main`` on the ``AutoGluonRAG`` instance rather than in
the yaml, so ``configs/agent/default.yaml`` (LLM off by default) is untouched.
The three-way rule/llm/strands sweep lives in ``evaluate_agentic_musique.py``.

Data
----
By default this reads the frozen, self-contained eval set produced by
``build_musique_eval_set.py`` (``--eval-set``, default
``local_example/evaluation_data_musique/musique_eval_set.jsonl``) so runs are
offline and reproducible. Pass ``--from-hf`` (or delete the frozen file) to load
``dgslibisey/MuSiQue`` from HuggingFace instead, applying the same reproducible
row selection. Either way the loaded rows are merged into the global corpus.

Environment note
----------------
Reuses ``local_example/local_config.yaml`` (Bedrock Cohere Embed English v3
embeddings + Bedrock Claude Sonnet 4.6 generator). Source ``credential.sh`` for
Bedrock access before running. The
config's saved-index paths are NOT written to: this runner overrides the vector-DB
save/load flags in memory so the global index never touches disk.
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
from agrag.evaluation.utils import calculate_f1_score, f1_metric, rouge_geometric_mean

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = "local_example/local_config.yaml"
DATASET = "dgslibisey/MuSiQue"
DEFAULT_SPLIT = "validation"
# Answer-quality metric computed via the evaluator (pure-Python, no model download).
# Matches the MultiHop-RAG / NQ benchmarks; token-F1 is computed separately below.
EM_METRIC = "inclusive_exact_match"
# Fixed seed for reproducible query-row selection (shared across both modes).
DEFAULT_SEED = 1234
# Frozen, self-contained MuSiQue slice produced by build_musique_eval_set.py. Used
# by default so runs are offline and reproducible (no HuggingFace access needed).
DEFAULT_EVAL_SET = "local_example/evaluation_data_musique/musique_eval_set.jsonl"


def load_rows(eval_set_path, from_hf, split, size, seed, stratify, answerable_only):
    """Return the MuSiQue rows to evaluate, from the frozen file or HuggingFace.

    Frozen path (default): read the self-contained JSONL built by
    ``build_musique_eval_set.py`` -- offline and reproducible, no network. Taken
    whenever ``from_hf`` is False and ``eval_set_path`` exists. HF path
    (``--from-hf``, or when the frozen file is absent): load ``dgslibisey/MuSiQue``
    and apply the same reproducible ``select_query_indices`` the frozen builder
    uses, so both routes evaluate comparable rows.
    """
    if not from_hf and os.path.exists(eval_set_path):
        # The frozen file is evaluated as-is: its rows were already selected (size,
        # seed, split, stratify) when it was built by build_musique_eval_set.py. So
        # those flags cannot re-select here -- warn if the user passed non-defaults
        # so the ignored flags are visible rather than silently dropped. Rebuild the
        # frozen file or pass --from-hf to change the selection. (--answerable-only
        # is still applied downstream in main, so it is not listed here.)
        ignored = []
        if size and size != 30:
            ignored.append(f"--max-eval-size={size}")
        if seed != DEFAULT_SEED:
            ignored.append(f"--seed={seed}")
        if split != DEFAULT_SPLIT:
            ignored.append(f"--split={split}")
        if stratify:
            ignored.append("--stratify")
        if ignored:
            print(
                f"WARNING: {', '.join(ignored)} ignored on the frozen path "
                f"({eval_set_path}); these only take effect with --from-hf or by "
                f"rebuilding the frozen file. Evaluating all frozen rows as-is."
            )
        rows = []
        with open(eval_set_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} frozen MuSiQue rows from {eval_set_path}")
        return rows

    print(f"Loading {DATASET} split '{split}' from HuggingFace ...")
    ds = load_dataset(DATASET, split=split)
    indices = select_query_indices(ds, size, seed=seed, answerable_only=answerable_only, stratify=stratify)
    rows = [ds[i] for i in indices]
    print(f"Selected {len(rows)} rows from HuggingFace (seed={seed}, stratify={stratify}).")
    return rows


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
    pct_multi_step = round(100.0 * multi_step / n, 1)
    # Describe what actually happened rather than emitting a fixed caveat: a low
    # pct_multi_step means the planner degenerated to single-shot (score parity
    # with standard is then expected); a high one means it genuinely decomposed.
    if pct_multi_step < 20.0:
        note = (
            f"pct_multi_step={pct_multi_step} is low: the planner mostly did NOT "
            "decompose, so agentic ~= single-shot and score parity with standard "
            "is expected."
        )
    else:
        note = (
            f"pct_multi_step={pct_multi_step}: the planner decomposed most queries "
            "into multiple retrievals, so agentic is doing real multi-step work."
        )
    return {
        "avg_retrieval_calls": round(avg_retrieval, 2),
        "avg_num_subqueries": round(avg_subqueries, 2),
        "avg_iterations": round(avg_iterations, 2),
        "pct_multi_step": pct_multi_step,
        "note": note,
    }


def _quality_scores(evaluator, predictions, references, queries):
    """Answer-quality metrics for one set of (prediction, references) pairs.

    Exact-match goes through the shared ``EvaluationModule`` (inclusive EM), and
    token-level F1 -- MuSiQue's official answer metric -- is computed directly via
    ``f1_metric``/``calculate_f1_score`` so it is aggregated to a mean (the
    evaluator's callable-metric path would return the raw per-example list). The
    ROUGE geometric mean (ROUGE-1 x ROUGE-2 x ROUGE-L)^(1/3) is added the same way,
    via ``rouge_geometric_mean`` so it is aggregated (mean-then-GM) per bucket.
    """
    if not predictions:
        return {EM_METRIC: 0.0, "f1": 0.0, "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rouge_gm": 0.0, "count": 0}
    em = evaluator.evaluate_responses(predictions=predictions, references=references, queries=queries)
    f1 = calculate_f1_score(f1_metric(predictions, references))
    rouge = rouge_geometric_mean(predictions, references)
    return {**em, "f1": round(f1, 4), **rouge, "count": len(predictions)}


def build_global_corpus(rows, work_dir):
    """Merge every selected question's paragraphs into ONE deduplicated corpus dir.

    Instead of the paper's per-question distractor pools, this pools *all* selected
    questions' paragraphs into a single corpus that is indexed once and queried by
    every question -- so each question's supporting paragraphs must be found among
    ALL questions' paragraphs (a harder, more realistic global-retrieval setting).

    Paragraphs are deduplicated by exact string identity: the same Wikipedia
    paragraph appearing in multiple questions is written only once. The identity
    key is the exact ``get_musique_paragraph_docs`` string (title-prepended), which
    is also what the index ingests, so a paragraph maps to exactly one global file.

    Each unique paragraph is written once as ``para_{global_i}.txt`` (``global_i``
    is first-seen order). Returns ``doc_to_global`` mapping each paragraph string to
    its global index, so downstream Support-F1 can translate a question's gold
    supporting paragraphs into global corpus indices (see
    ``evaluate_agentic_musique.gold_global_support_indices``).

    The corpus dir is populated fully *before* the pipeline is initialized, so the
    normal one-time ``initialize_rag_pipeline`` build indexes the whole global
    corpus -- no per-question re-indexing is needed.

    Note on scale: total corpus size grows with the number of questions. The
    vector DB's duplicate removal builds an O(n^2) similarity matrix and BM25 is a
    pure-Python O(corpus) scan per query, so very large samples will be slow; the
    default ~30-question samples (a few hundred paragraphs after dedup) are fine.
    """
    if os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    doc_to_global = {}
    total = 0
    for row in rows:
        for doc in get_musique_paragraph_docs(row):
            total += 1
            if doc in doc_to_global:
                continue
            global_i = len(doc_to_global)
            doc_to_global[doc] = global_i
            with open(os.path.join(work_dir, f"para_{global_i}.txt"), "w", encoding="utf-8") as f:
                f.write(doc + "\n")

    print(
        f"Global corpus: {total} paragraphs across {len(rows)} questions "
        f"-> {len(doc_to_global)} unique written to {work_dir}"
    )
    return doc_to_global


def run_query(agrag, query, mode):
    """Execute ONE comparable run for ``query`` in ``mode`` and time all of it.

    Both modes go through a single ``generate_response(..., return_trace=True)``
    call, so exactly one retrieval + generation happens per standard query. The
    whole comparable operation -- retrieval and generation -- is timed for both.
    (The global corpus is indexed once up front; that build is not attributed to
    either mode.)

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


def apply_rerank_setting(agrag, use_reranker, reranker_top_k=None):
    """Reconfigure the already-built pipeline's reranker gate WITHOUT re-indexing.

    ``use_reranker`` only gates the retriever's rerank stage; the vector index,
    embeddings, and parent store are untouched. Re-initializing the retriever
    module rebuilds it against the same in-memory ``vector_db_module`` (cheap --
    no embedding or index work), and clearing the cached agentic module forces it
    to rebuild against the new retriever on the next agentic query. When
    ``reranker_top_k`` is given, the Reranker instance is rebuilt too (its truncation
    cutoff is baked in at construction) so the sweep can vary how many chunks
    survive reranking -- otherwise enabling reranking silently reintroduces the
    truncate-to-reranker_top_k step and confounds the comparison with "fewer chunks
    reached the generator".
    """
    agrag.args.use_reranker = use_reranker
    if reranker_top_k is not None:
        agrag.args.reranker_top_k = reranker_top_k
        agrag.initialize_reranker_module()
    agrag.initialize_retriever_module()
    # The retriever was just rebuilt; re-attach the parent store (small-to-big
    # expansion) and drop the cached agentic module so it rebinds to the new retriever.
    agrag._attach_parent_store_to_retriever()
    agrag.agentic_module = None
    print(f"  reranking {'ON' if use_reranker else 'OFF'}" + (f" (top_k={reranker_top_k})" if reranker_top_k else ""))


def rerank_variants(rerank_arg):
    """Map the ``--rerank`` choice to an ordered list of ``(use_reranker, suffix)``.

    ``config`` (default) leaves the pipeline exactly as the yaml configured it --
    a single unsuffixed run, so behavior is unchanged when the flag is omitted.
    ``on``/``off`` force a single setting; ``both`` runs the on and off settings
    back to back so the two can be compared side by side in one results file.
    """
    if rerank_arg == "both":
        return [(True, "_rerank_on"), (False, "_rerank_off")]
    if rerank_arg == "on":
        return [(True, "")]
    if rerank_arg == "off":
        return [(False, "")]
    return [(None, "")]  # "config": no override


def run_mode(agrag, evaluator, rows, mode, jsonl_writer=None, label=None):
    """Run one evaluation pass over the pre-selected rows.

    ``rows`` is the shared, reproducible list of MuSiQue rows used for BOTH modes
    (paired comparison). Every question is queried against the same global corpus,
    which was merged and indexed once before this call. ``jsonl_writer`` is an
    optional callable receiving one dict per query, written as a JSONL row.
    ``label`` overrides the ``mode`` field written to the JSONL / printed header
    (used to tag rerank-sweep variants, e.g. ``agentic_rerank_on``); ``mode`` itself
    still drives ``generate_response``. Returns overall + per-hop-type metrics plus
    the per-query latencies.
    """
    label = label or (mode or "standard")
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG on MuSiQue  (n={len(rows)})")
    print("=" * 72)

    # Bucket by hop-count question type so we can score each type separately.
    buckets = {}
    all_preds, all_refs, all_queries = [], [], []
    all_retrieval = []  # per-query retrieval metrics, rows with gold facts only
    agent_runs = []  # per-query agentic decomposition signals (agentic mode only)
    latencies = []
    for idx, row in enumerate(rows):
        expected = get_musique_responses(row)
        if not expected:
            continue
        query = get_musique_query(row)
        qtype = get_musique_question_type(row)
        gold_facts = get_musique_evidence_facts(row)

        # No per-question re-indexing: every question is queried against the same
        # global corpus that was merged and indexed once before this pass.
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
                    "row_index": idx,
                    "source_index": row.get("_source_index"),
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
    # Run relative to the repo root so config/data paths resolve regardless of the
    # caller's cwd. Done here (not at import time) so importing this module -- e.g.
    # build_musique_eval_set.py reuses select_query_indices -- has no side effects.
    os.chdir(REPO_ROOT)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        default=DEFAULT_EVAL_SET,
        help=f"Frozen MuSiQue JSONL to evaluate offline (default: {DEFAULT_EVAL_SET}).",
    )
    parser.add_argument(
        "--from-hf",
        action="store_true",
        help="Ignore the frozen file and load from HuggingFace (needs network).",
    )
    parser.add_argument(
        "--max-eval-size", type=int, default=30, help="Sample size when loading from HuggingFace (default: 30)."
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help=f"MuSiQue split when --from-hf (default: {DEFAULT_SPLIT})."
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
    parser.add_argument(
        "--rerank",
        choices=["config", "on", "off", "both"],
        default="config",
        help=(
            "Cross-encoder reranking sweep. 'config' (default): use whatever the yaml "
            "sets (unchanged behavior). 'on'/'off': force one setting. 'both': run "
            "reranking on AND off back to back and tag results (_rerank_on/_rerank_off) "
            "for side-by-side comparison. No pipeline-code change; only the retriever's "
            "use_reranker gate is toggled (no re-indexing)."
        ),
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=None,
        help=(
            "When reranking is enabled, override reranker_top_k (the truncation cutoff). "
            "Vary this so the on/off comparison is not confounded by fewer chunks reaching "
            "the generator. Defaults to the yaml value."
        ),
    )
    args = parser.parse_args()

    # Load the SAME rows for both modes (paired comparison). Frozen JSONL by
    # default (offline); HuggingFace only with --from-hf or if the file is absent.
    # This condition mirrors load_rows' own branch so the recorded selection
    # metadata below matches how the rows were actually chosen.
    used_frozen = not args.from_hf and os.path.exists(args.eval_set)
    rows = load_rows(
        args.eval_set, args.from_hf, args.split, args.max_eval_size, args.seed, args.stratify, args.answerable_only
    )
    # Keep only rows with a usable gold answer (metrics need a reference); with
    # --answerable-only also drop unanswerable rows (the frozen path may include them).
    rows = [r for r in rows if get_musique_responses(r)]
    if args.answerable_only:
        rows = [r for r in rows if get_musique_answerable(r)]
    if not rows:
        raise SystemExit("No eligible questions with a gold answer; nothing to evaluate.")
    print(f"Evaluating {len(rows)} MuSiQue questions; same rows used for both modes.")

    # Merge every selected question's paragraphs into ONE deduplicated global
    # corpus dir, populated fully before the pipeline is built so the one-time
    # initialize_rag_pipeline indexes the whole corpus (no per-question reindex).
    corpus_dir = os.path.join(tempfile.gettempdir(), "musique_global_corpus")
    build_global_corpus(rows, corpus_dir)

    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    # Never load or persist a shared index: the global corpus is indexed fresh in
    # memory. Overriding here (not in the yaml) keeps local_config.yaml untouched.
    agrag.args.use_existing_vector_db = False
    agrag.args.save_vector_db_index = False
    # Drive the agentic path with the LLM planner + policy (via the shared Bedrock
    # generator) instead of the deterministic regex planner / rule-based action
    # cascade. The agentic module is built lazily on the first mode="agentic"
    # query and reads these flags then, so setting them here (not in the yaml)
    # takes effect while leaving configs/agent/default.yaml untouched.
    agrag.args.agent_use_llm_planner = True
    agrag.args.agent_use_llm_policy = True
    # Sequential-hop execution: resolve subqueries in order, threading each hop's
    # resolved answer into the next hop's retrieval query. Targets the dominant
    # decomposition-failure bucket (late hops retrieved blind). Set here (not in the
    # yaml) so configs/agent/default.yaml stays at its opt-out default.
    agrag.args.agent_use_iterative_planner = True
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)

    os.makedirs(args.evaluation_dir, exist_ok=True)
    jsonl_path = os.path.join(args.evaluation_dir, "benchmark_predictions.jsonl")

    # Reranking sweep: one pass per (use_reranker) variant. "config" (default) is a
    # single pass that leaves the yaml setting alone, so omitting --rerank keeps the
    # original two-bucket (standard/agentic) behavior byte-for-byte.
    variants = rerank_variants(args.rerank)
    results = {}
    with open(jsonl_path, "w") as jf:
        def jsonl_writer(row):
            jf.write(json.dumps(row, default=str) + "\n")

        for use_reranker, suffix in variants:
            if use_reranker is not None:
                print(f"\n### Rerank variant: use_reranker={use_reranker} ###")
                apply_rerank_setting(agrag, use_reranker, args.rerank_top_k)
            for mode, base in ((None, "standard"), ("agentic", "agentic")):
                label = base + suffix
                results[label] = run_mode(
                    agrag, evaluator, rows, mode=mode, jsonl_writer=jsonl_writer, label=label,
                )
    print(f"\nSaved per-query predictions to {jsonl_path}")

    # On the frozen path, size/seed/split/stratify did NOT drive selection (the
    # rows were selected when the frozen file was built), so record them as null to
    # avoid implying they applied. --answerable-only IS applied on both paths above,
    # so it is reported as-is.
    results["selection"] = {
        "dataset": DATASET,
        "selection_source": "frozen" if used_frozen else "huggingface",
        "eval_set": args.eval_set if used_frozen else None,
        "from_hf": args.from_hf,
        "split": None if used_frozen else args.split,
        "seed": None if used_frozen else args.seed,
        "answerable_only": args.answerable_only,
        "stratify": None if used_frozen else args.stratify,
        "num_selected": len(rows),
        "source_indices": [r.get("_source_index") for r in rows],
        "rerank_sweep": args.rerank,
        "rerank_top_k_override": args.rerank_top_k,
    }

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, shared global corpus + settings)")
    print("=" * 72)
    print(json.dumps({k: v for k, v in results.items() if k != "selection"}, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
