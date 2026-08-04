"""Evaluate the agentic RAG workflow on MuSiQue across reasoning backends.

Goal
----
The existing ``benchmark_musique.py`` compares *standard* vs. *agentic* RAG. This
script answers a different question aimed squarely at the agentic workflow: **does
the planner/policy reasoning backend matter, and does it help more as questions
get harder (more hops)?** It sweeps the three backends the agentic path now
supports and reports, per hop count:

* ``rule``    -- deterministic planner (regex split) + rule-based policy cascade.
* ``llm``     -- raw-Bedrock LLM planner + policy (``use_llm_planner/policy``).
* ``strands`` -- Strands/Bedrock-Haiku planner + policy (``use_strands_*``), where
                 the LLM emits only the plan/action and Python derives the args.

Metrics (MuSiQue's official pair, plus supporting signals)
----------------------------------------------------------
* **Answer EM** -- inclusive exact match (shared ``EvaluationModule``).
* **Answer F1** -- token-level F1 vs. gold answer + aliases (``token_f1``); this is
  MuSiQue's headline answer metric.
* **Support F1** -- the paper's second official metric: did the system surface the
  gold *supporting* paragraphs? A RAG pipeline does not emit an explicit support
  set, so its retrieved/used evidence set is taken as its support prediction: each
  retrieved chunk is mapped back (via its ``para_{i}.txt`` source) to a paragraph
  index, and precision/recall/F1 are computed over paragraph-index sets against
  the ``is_supporting`` gold. This is an honest RAG analog of the paper's metric,
  labeled as such -- not the token-level support metric a span-predicting model
  would report.

Fair comparison
---------------
One ``AutoGluonRAG`` instance loads the models once. For each question its own
paragraph pool is indexed **once** (the paper's distractor setting), and all three
backends run against that identical index, so any score difference is attributable
to the reasoning backend and not to retrieval variance. The three agentic modules
share the same retriever + generator and are built once, then reused across
questions (re-indexing mutates the shared retriever in place).

Data
----
Reads the frozen, self-contained eval set produced by
``build_musique_eval_set.py`` (default:
``local_example/evaluation_data_musique/musique_eval_set.jsonl``) so runs are
offline and reproducible. Falls back to loading ``dgslibisey/MuSiQue`` from
HuggingFace with ``--from-hf`` if no frozen file is present.

Environment
-----------
Reuses ``local_example/local_config.yaml`` (MiniLM embeddings + Bedrock Claude
Haiku). Source ``credential.sh`` for Bedrock access before running -- the ``llm``
and ``strands`` backends make live Bedrock calls; ``rule`` does not. Per-question
indexes are built in memory and never persisted.

Usage
-----
    # one-time (needs network): freeze a real MuSiQue sample
    python local_example/build_musique_eval_set.py --size 30 --stratify

    # then (offline for retrieval; Bedrock for llm/strands backends):
    python local_example/evaluate_agentic_musique.py
    python local_example/evaluate_agentic_musique.py --backends rule strands
"""

import argparse
import json
import os
import shutil
import tempfile
import time

# Reuse the existing runner's building blocks rather than duplicating them: the
# per-question re-indexing, the evaluator/quality scoring, and the reproducible
# row selection are all already correct in benchmark_musique.
from benchmark_musique import (
    CONFIG,
    DATASET,
    DEFAULT_SEED,
    DEFAULT_SPLIT,
    build_evaluator,
    reindex_question,
    select_query_indices,
    _quality_scores,
)

from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.musique.musique import (
    get_musique_answerable,
    get_musique_evidence_facts,
    get_musique_paragraph_docs,
    get_musique_query,
    get_musique_question_type,
    get_musique_responses,
    get_musique_supporting_flags,
)
from agrag.evaluation.retrieval_metrics import aggregate_retrieval_metrics, retrieval_metrics_for_query
from agrag.modules.agentic.agentic_module import AgenticRAGModule

# Backend name -> the agent-config flag overrides that select that reasoning path.
# rule = no model calls in planner/policy; llm = raw-Bedrock; strands = Strands SDK.
BACKENDS = {
    "rule": {
        "use_llm_planner": False,
        "use_llm_policy": False,
        "use_strands_planner": False,
        "use_strands_policy": False,
    },
    "llm": {
        "use_llm_planner": True,
        "use_llm_policy": True,
        "use_strands_planner": False,
        "use_strands_policy": False,
    },
    "strands": {
        "use_llm_planner": False,
        "use_llm_policy": False,
        "use_strands_planner": True,
        "use_strands_policy": True,
    },
}

DEFAULT_EVAL_SET = "local_example/evaluation_data_musique/musique_eval_set.jsonl"


def load_rows(eval_set_path, from_hf, split, size, seed, stratify, answerable_only):
    """Yield the MuSiQue rows to evaluate, from the frozen file or HuggingFace.

    Frozen path (default): read the self-contained JSONL built by
    ``build_musique_eval_set.py`` -- offline and reproducible. HF path
    (``--from-hf``): load ``dgslibisey/MuSiQue`` and apply the same reproducible
    selection the frozen builder uses, so both routes evaluate comparable rows.
    """
    if not from_hf and os.path.exists(eval_set_path):
        rows = []
        with open(eval_set_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"Loaded {len(rows)} frozen MuSiQue rows from {eval_set_path}")
        return rows

    from datasets import load_dataset

    print(f"Loading {DATASET} split '{split}' from HuggingFace ...")
    ds = load_dataset(DATASET, split=split)
    indices = select_query_indices(ds, size, seed=seed, answerable_only=answerable_only, stratify=stratify)
    rows = [ds[i] for i in indices]
    print(f"Selected {len(rows)} rows from HuggingFace (seed={seed}, stratify={stratify}).")
    return rows


def _paragraph_index_from_source(source):
    """Map an evidence chunk's ``source`` path (``.../para_{i}.txt``) to index ``i``.

    ``reindex_question`` writes one ``para_{i}.txt`` per ingested paragraph, in the
    order ``get_musique_paragraph_docs`` returns them; ``get_musique_supporting_flags``
    uses the identical filtering, so ``i`` indexes the same paragraph in both.
    Returns ``None`` when the source does not follow that pattern.
    """
    if not source:
        return None
    base = os.path.basename(str(source))
    if not (base.startswith("para_") and base.endswith(".txt")):
        return None
    stem = base[len("para_") : -len(".txt")]
    return int(stem) if stem.isdigit() else None


def support_f1(evidence, supporting_flags):
    """Support-F1 analog: overlap of *surfaced* vs. *gold-supporting* paragraphs.

    ``evidence`` is the list of evidence dicts the run conditioned on (each with a
    ``source``); ``supporting_flags[i]`` says whether ingested paragraph ``i`` is a
    gold supporting paragraph. The predicted support set is the distinct paragraph
    indices that appear in the retrieved evidence; the gold set is the indices
    flagged supporting. Returns precision / recall / F1 over those sets.

    Returns ``None`` when there are no gold supporting paragraphs (e.g.
    unanswerable rows), so such rows are excluded from the Support-F1 mean rather
    than counted as trivially perfect or zero.
    """
    gold = {i for i, flag in enumerate(supporting_flags) if flag}
    if not gold:
        return None
    predicted = set()
    for ev in evidence:
        idx = _paragraph_index_from_source(ev.get("source"))
        if idx is not None:
            predicted.add(idx)
    if not predicted:
        return {"support_precision": 0.0, "support_recall": 0.0, "support_f1": 0.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"support_precision": precision, "support_recall": recall, "support_f1": f1}


def _mean_support(support_rows):
    """Mean of the per-query support dicts (skipping ``None`` rows). Empty -> {}."""
    rows = [s for s in support_rows if s is not None]
    if not rows:
        return {}
    keys = rows[0].keys()
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}


def build_backend_modules(agrag, base_cfg, backend_names):
    """Build one agentic module per requested backend, sharing retriever+generator.

    Each module differs only in the planner/policy backend flags overlaid on the
    shared ``base_cfg``. They wrap the same retriever/generator by reference, so
    re-indexing a question (which mutates the retriever in place) is visible to all
    of them -- letting every backend answer against the identical per-question
    index. Building once (not per question) avoids reloading models and rebuilding
    the Strands agent on every row.
    """
    modules = {}
    for name in backend_names:
        cfg = dict(base_cfg)
        cfg.update(BACKENDS[name])
        modules[name] = AgenticRAGModule(agrag.retriever_module, agrag.generator_module, config=cfg)
    return modules


def run_backend_on_question(module, query):
    """Answer one query with one backend; return answer, evidence, timing, signals."""
    start = time.perf_counter()
    answer, trace = module.answer(query, return_trace=True)
    latency = time.perf_counter() - start
    trace = trace if isinstance(trace, dict) else {}
    evidence = trace.get("evidence", []) or []
    metrics = trace.get("metrics", {}) or {}
    return {
        "answer": answer,
        "evidence": evidence,
        "evidence_texts": [ev.get("text", "") for ev in evidence],
        "latency": latency,
        "retrieval_calls": metrics.get("retrieval_calls", 0),
        "num_subqueries": len(trace.get("subqueries", []) or []),
        "iterations": metrics.get("iterations", 0),
    }


def summarize_backend(per_query, evaluator):
    """Aggregate one backend's per-query results into overall + per-hop metrics.

    ``per_query`` rows carry the prediction, references, hop type, retrieval
    metrics, support dict, and agentic signals. Answer EM/F1 go through the shared
    quality scorer; Support-F1 and retrieval are averaged over the eligible rows;
    the agentic behavior summary flags whether the planner actually decomposed
    (if it did not, backend parity is expected rather than a failure).
    """

    def quality(rows):
        preds = [r["prediction"] for r in rows]
        refs = [r["references"] for r in rows]
        queries = [r["query"] for r in rows]
        scores = _quality_scores(evaluator, preds, refs, queries)
        scores["support"] = _mean_support([r["support"] for r in rows])
        scores["retrieval"] = aggregate_retrieval_metrics([r["retrieval"] for r in rows if r["retrieval"]])
        return scores

    by_type = {}
    for r in per_query:
        by_type.setdefault(r["question_type"], []).append(r)

    n = len(per_query)
    latencies = [r["latency"] for r in per_query]
    multi_step = sum(1 for r in per_query if r["retrieval_calls"] > 1)
    return {
        "overall": quality(per_query),
        "by_question_type": {t: quality(rows) for t, rows in sorted(by_type.items())},
        "agentic_behavior": {
            "avg_retrieval_calls": round(sum(r["retrieval_calls"] for r in per_query) / n, 2) if n else 0.0,
            "avg_num_subqueries": round(sum(r["num_subqueries"] for r in per_query) / n, 2) if n else 0.0,
            "avg_iterations": round(sum(r["iterations"] for r in per_query) / n, 2) if n else 0.0,
            "pct_multi_step": round(100.0 * multi_step / n, 1) if n else 0.0,
        },
        "cost": {
            "num_queries": n,
            "total_latency_s": round(sum(latencies), 3),
            "avg_latency_s": round(sum(latencies) / n, 3) if n else 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKENDS),
        default=list(BACKENDS),
        help="Which reasoning backends to sweep (default: all three).",
    )
    parser.add_argument(
        "--eval-set",
        default=DEFAULT_EVAL_SET,
        help=f"Frozen MuSiQue JSONL to evaluate (default: {DEFAULT_EVAL_SET}).",
    )
    parser.add_argument("--from-hf", action="store_true", help="Ignore the frozen file; load from HuggingFace.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="HF split when --from-hf (default: validation).")
    parser.add_argument("--size", type=int, default=30, help="Sample size when --from-hf (default: 30).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Selection seed when --from-hf.")
    parser.add_argument("--stratify", action="store_true", help="Stratify by hop count when --from-hf.")
    parser.add_argument("--answerable-only", action="store_true", help="Keep only answerable rows when --from-hf.")
    parser.add_argument(
        "--evaluation-dir",
        default="local_example/evaluation_data_musique",
        help="Where per-query predictions and the summary are written.",
    )
    args = parser.parse_args()

    rows = load_rows(
        args.eval_set, args.from_hf, args.split, args.size, args.seed, args.stratify, args.answerable_only
    )
    # Keep only rows with a usable gold answer (metrics need a reference).
    rows = [r for r in rows if get_musique_responses(r)]
    if not rows:
        raise SystemExit("No eligible rows with a gold answer; nothing to evaluate.")
    print(f"Evaluating {len(rows)} questions across backends: {args.backends}")

    corpus_dir = os.path.join(tempfile.gettempdir(), "musique_sweep_corpus")

    # Initialize the pipeline once on the first question's paragraphs so the
    # embedding/reranker/generator models load exactly once.
    first_docs = get_musique_paragraph_docs(rows[0])
    if os.path.isdir(corpus_dir):
        shutil.rmtree(corpus_dir)
    os.makedirs(corpus_dir, exist_ok=True)
    for i, doc in enumerate(first_docs):
        with open(os.path.join(corpus_dir, f"para_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(doc + "\n")

    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=corpus_dir)
    agrag.args.use_existing_vector_db = False
    agrag.args.save_vector_db_index = False
    if not agrag.pipeline_initialized:
        agrag.initialize_rag_pipeline()

    evaluator = build_evaluator(agrag)
    base_cfg = agrag._agent_config()
    modules = build_backend_modules(agrag, base_cfg, args.backends)

    # results[backend] = list of per-query rows.
    results = {name: [] for name in args.backends}

    os.makedirs(args.evaluation_dir, exist_ok=True)
    jsonl_path = os.path.join(args.evaluation_dir, "backend_sweep_predictions.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for qi, row in enumerate(rows):
            query = get_musique_query(row)
            expected = get_musique_responses(row)
            qtype = get_musique_question_type(row)
            gold_facts = get_musique_evidence_facts(row)
            support_flags = get_musique_supporting_flags(row)

            # Index THIS question's paragraph pool once; all backends share it.
            reindex_question(agrag, get_musique_paragraph_docs(row), corpus_dir)
            print(f"[{qi + 1}/{len(rows)}] {qtype}: {query[:70]}...")

            for name in args.backends:
                run = run_backend_on_question(modules[name], query)
                retrieval = retrieval_metrics_for_query(run["evidence_texts"], gold_facts) if gold_facts else {}
                support = support_f1(run["evidence"], support_flags)
                results[name].append(
                    {
                        "prediction": run["answer"],
                        "references": expected,
                        "query": query,
                        "question_type": qtype,
                        "retrieval": retrieval,
                        "support": support,
                        "latency": run["latency"],
                        "retrieval_calls": run["retrieval_calls"],
                        "num_subqueries": run["num_subqueries"],
                        "iterations": run["iterations"],
                    }
                )
                jf.write(
                    json.dumps(
                        {
                            "backend": name,
                            "question_type": qtype,
                            "answerable": get_musique_answerable(row),
                            "query": query,
                            "references": expected,
                            "prediction": run["answer"],
                            "evidence_texts": run["evidence_texts"],
                            "retrieval_metrics": retrieval,
                            "support_metrics": support,
                            "latency_s": round(run["latency"], 4),
                            "retrieval_calls": run["retrieval_calls"],
                            "num_subqueries": run["num_subqueries"],
                            "iterations": run["iterations"],
                        },
                        default=str,
                    )
                    + "\n"
                )

    summary = {name: summarize_backend(rows_, evaluator) for name, rows_ in results.items()}

    print("\n" + "=" * 72)
    print("BACKEND SWEEP SUMMARY  (agentic MuSiQue; identical per-question index)")
    print("=" * 72)
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved per-query predictions to {jsonl_path}")

    out = os.path.join(args.evaluation_dir, "backend_sweep_results.json")
    payload = {
        "eval_set": None if args.from_hf else args.eval_set,
        "num_questions": len(rows),
        "backends": args.backends,
        "results": summary,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
