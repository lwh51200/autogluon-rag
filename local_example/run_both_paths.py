"""Run ONE query through each path locally: standard RAG and agentic RAG.

Fully local: HuggingFace tiny models on CPU, no cloud keys. The tiny demo
generator (sshleifer/tiny-gpt2) produces gibberish text — this validates the
plumbing of each path, not answer quality.
"""
import os

from agrag.agrag import AutoGluonRAG
from agrag.modules.agentic.agentic_module import AgenticRAGModule

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

CONFIG = "local_example/local_config.yaml"
DATA_DIR = "local_example/docs"
QUERY = "What is AutoGluon and how does it build models?"


def main():
    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=DATA_DIR)
    agrag.initialize_rag_pipeline()

    # ---- 1) STANDARD RAG path (default: mode=None -> "standard") ----
    print("\n" + "=" * 70 + "\nSTANDARD RAG\n" + "=" * 70)
    print("ANSWER:", agrag.generate_response(QUERY))

    # ---- 2) AGENTIC RAG path (mode="agentic") ----
    # Bind a small context budget so tiny-gpt2's 1024-token window is respected;
    # a real model (e.g. Mistral) would just use the config defaults.
    print("\n" + "=" * 70 + "\nAGENTIC RAG\n" + "=" * 70)
    agrag.agentic_module = AgenticRAGModule(
        agrag.retriever_module,
        agrag.generator_module,
        config={"min_evidence_count": 2, "retrieve_top_k_per_query": 3, "max_context_tokens": 150},
    )
    answer, trace = agrag.generate_response(QUERY, mode="agentic", return_trace=True)
    print("ANSWER:", answer)
    print("STATUS:", trace["status"], "| METRICS:", trace["metrics"])


if __name__ == "__main__":
    main()
