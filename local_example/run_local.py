"""Minimal AutoGluon-RAG example over local documents.

Builds a RAG pipeline over the local text files in local_example/docs/ using
HuggingFace embeddings + reranker on CPU and an AWS Bedrock Claude Haiku 4.5
generator (per local_config.yaml), then answers a query. Bedrock uses the
standard AWS credential chain; swap the generator in local_config.yaml for a
HuggingFace model to run without cloud credentials.
"""
import os

from agrag.agrag import AutoGluonRAG

# Run from the repo root so the relative paths in the config resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

CONFIG = "local_example/local_config.yaml"
DATA_DIR = "local_example/docs"


def main():
    agrag = AutoGluonRAG(
        config_file=CONFIG,
        data_dir=DATA_DIR,
    )

    # Build the pipeline: process docs -> embed -> store in FAISS.
    agrag.initialize_rag_pipeline()

    # ---- Inspect retrieval directly (the interesting part of RAG) ----
    query = "What is AutoGluon and how does it build models?"
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    context = agrag.retrieve_context_for_query(query)
    print("\nRETRIEVED CONTEXT CHUNKS:")
    for i, chunk in enumerate(context, 1):
        text = chunk.get("text", chunk) if isinstance(chunk, dict) else chunk
        print(f"\n[{i}] {text}")

    # ---- Full generate (retrieve context, then generate the answer) ----
    print("\n" + "=" * 70)
    print("GENERATED RESPONSE:")
    print("=" * 70)
    response = agrag.generate_response(query)
    print(response)


if __name__ == "__main__":
    main()
