"""Benchmark Standard RAG vs. Agentic RAG on the SAME dataset and settings.

Reuses the original AutoGluon-RAG evaluation pipeline (``EvaluationModule``) and
the original Google Natural Questions dataset adapters. A single ``AutoGluonRAG``
instance (one ingest, one index, one embedder/generator) is evaluated twice --
once in the standard path and once in the agentic path -- so the only variable
between the two systems is ``mode``.

The only library change this relies on is ``EvaluationModule`` now forwarding a
``mode`` argument to ``generate_response`` (and returning its computed metrics).

Notes / environment constraints
--------------------------------
* The original ``medium_quality`` preset uses Amazon Bedrock (Claude Sonnet +
  Cohere embeddings). Those credentials are unavailable in this environment, so
  this harness runs the already-cached *local* HuggingFace models from
  ``local_config.yaml`` (MiniLM embeddings, tiny-gpt2 generator). Both systems
  share those identical models, so the comparison stays apples-to-apples; the
  absolute answer-quality numbers are not meaningful with the tiny demo
  generator -- the pipeline wiring and the cost/latency deltas are what this
  validates. Swap in a Bedrock/OpenAI/Mistral generator for real quality numbers.
"""

import argparse
import json
import os
import time

from agrag.agrag import AutoGluonRAG
from agrag.evaluation.datasets.google_natural_questions.google_nq import (
    get_google_nq_query,
    get_google_nq_responses,
    preprocess_google_nq,
)
from agrag.evaluation.evaluator import EvaluationModule

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

CONFIG = "local_example/local_config.yaml"
DATASET = "google-research-datasets/natural_questions"
# Answer-quality metric from the original pipeline. "inclusive_exact_match" is
# the pipeline's supported exact-match metric (pure Python, no model download).
# The original example's other metrics ("transformer_matcher"/"pedant") pull
# large models and are skipped here to keep the benchmark self-contained; add
# them back once running against a real generator + network-available models.
METRICS = ["inclusive_exact_match"]


class TimedRAG(AutoGluonRAG):
    """AutoGluonRAG that records per-call latency for the current mode.

    Lets the benchmark report cost metrics (latency, call count) without
    modifying the evaluation library. Timings are bucketed by the ``mode`` set
    on ``self._bench_mode`` before each evaluation run.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bench_mode = "standard"
        self._bench_timings = {"standard": [], "agentic": []}

    def generate_response(self, query, mode=None, return_trace=None):
        start = time.perf_counter()
        result = super().generate_response(query, mode=mode, return_trace=return_trace)
        elapsed = time.perf_counter() - start
        self._bench_timings[self._bench_mode].append(elapsed)
        return result


def _cost_summary(timings):
    n = len(timings)
    total = sum(timings)
    return {
        "num_queries_answered": n,
        "total_latency_s": round(total, 3),
        "avg_latency_s": round(total / n, 3) if n else 0.0,
    }


def run_mode(agrag, mode, max_eval_size, save_eval_data, evaluation_dir):
    """Run one evaluation pass in the given mode and return its metrics."""
    label = mode or "standard"
    agrag._bench_mode = label
    print("\n" + "=" * 72)
    print(f"EVALUATING: {label.upper()} RAG  (max_eval_size={max_eval_size})")
    print("=" * 72)

    evaluator = EvaluationModule(rag_instance=agrag)
    metrics = evaluator.run_evaluation(
        dataset_name=DATASET,
        metrics=METRICS,
        preprocessing_fn=preprocess_google_nq,
        query_fn=get_google_nq_query,
        response_fn=get_google_nq_responses,
        hf_dataset_params={"name": "dev"},
        # Only the first pass needs to write docs + build the index; the second
        # pass reuses the same already-initialized pipeline.
        save_evaluation_data=save_eval_data,
        evaluation_dir=evaluation_dir,
        max_eval_size=max_eval_size,
        mode=mode,
    )
    return {"quality": metrics, "cost": _cost_summary(agrag._bench_timings[label])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-eval-size", type=int, default=3, help="Number of NQ datapoints to evaluate (smoke test default: 3)."
    )
    parser.add_argument(
        "--evaluation-dir", default="local_example/evaluation_data_nq", help="Where NQ docs are written + ingested."
    )
    args = parser.parse_args()

    # Same instance -> same ingest, index, embedder, generator for both systems.
    agrag = TimedRAG(config_file=CONFIG, data_dir=args.evaluation_dir)

    results = {}
    # First pass (standard): writes NQ docs and builds the index.
    results["standard"] = run_mode(
        agrag, mode=None, max_eval_size=args.max_eval_size, save_eval_data=True, evaluation_dir=args.evaluation_dir
    )
    # Second pass (agentic): reuse the already-initialized pipeline + index.
    # The agentic module is lazily initialized from the agent config defaults
    # (configs/agent/default.yaml) on first use -- with a real generator there is
    # no need to shrink the context budget the way the tiny-gpt2 demo required.
    results["agentic"] = run_mode(
        agrag, mode="agentic", max_eval_size=args.max_eval_size, save_eval_data=False, evaluation_dir=args.evaluation_dir
    )

    print("\n" + "=" * 72)
    print("SUMMARY  (standard vs. agentic, identical dataset + settings)")
    print("=" * 72)
    print(json.dumps(results, indent=2, default=str))

    out = os.path.join(args.evaluation_dir, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
