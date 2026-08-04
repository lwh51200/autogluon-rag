"""Streamlit live demo: Static RAG vs. Agentic RAG.

The project's real entry points:

  * Init (shared):  AutoGluonRAG(config_file=..., data_dir=...)  ->  .initialize_rag_pipeline()
                        (src/agrag/agrag.py)
  * Static query:   agrag.generate_response(query)                    # mode="standard" (default)
  * Agentic query:  agrag.generate_response(query, mode="agentic", return_trace=True)
                        -> AgenticRAGModule.answer(...)   (src/agrag/modules/agentic/agentic_module.py)

Launch from the project root:

    source venv/bin/activate
    streamlit run local_example/streamlit_demo.py
"""
import os
import time

import streamlit as st

# Run from the repo root so the relative paths in local_config.yaml resolve.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

from agrag.agrag import AutoGluonRAG  # noqa: E402  (import after chdir)
from agrag.modules.agentic.agentic_module import AgenticRAGModule  # noqa: E402

CONFIG = "local_example/local_config.yaml"
DATA_DIR = "local_example/docs"

DEFAULT_QUERY = "What is AutoGluon and how does it build models?"


def resolve_generator():
    """Pick the generator from env vars, falling back to the config's default.

    Both the static and agentic paths share a single ``GeneratorModule`` (the
    agentic module is constructed with ``agrag.generator_module``), so overriding
    it once here guarantees a fair, same-model comparison.

    No API keys are hard-coded. Credentials come from the environment:
      * bedrock     -> standard AWS credential chain (no key in code)
      * openai      -> OPENAI_API_KEY (read by the project's read_openai_key)
      * huggingface -> keep local_config.yaml's generator unchanged

    Env vars (all optional):
      AGRAG_GENERATOR_PLATFORM   bedrock | openai | huggingface   (default: bedrock)
      AGRAG_GENERATOR_MODEL      model id / name for that platform
      AWS_REGION / AWS_DEFAULT_REGION   region for bedrock         (default: us-east-1)

    Returns ``(platform, model_name, platform_args)`` or ``None`` to keep the
    config file's generator unchanged (Bedrock Claude Haiku 4.5 by default).
    """
    platform = os.getenv("AGRAG_GENERATOR_PLATFORM", "bedrock").lower()

    if platform == "huggingface":
        return None  # keep local_config.yaml's generator as-is

    if platform == "bedrock":
        model = os.getenv("AGRAG_GENERATOR_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        params = {"max_tokens": 512, "temperature": 0.2}
        if "claude" in model or "anthropic" in model:
            params["anthropic_version"] = "bedrock-2023-05-31"
        return platform, model, {"bedrock_aws_region": region, "bedrock_generate_params": params}

    if platform == "openai":
        model = os.getenv("AGRAG_GENERATOR_MODEL", "gpt-4o-mini")
        # openai_api_key is read from OPENAI_API_KEY by GeneratorModule/read_openai_key.
        return platform, model, {"gpt_generate_params": {"max_tokens": 512, "temperature": 0.2}}

    raise ValueError(f"Unsupported AGRAG_GENERATOR_PLATFORM: {platform!r}")


def agent_config():
    """Config for the agentic path (shared retriever/generator; small top-k)."""
    return {"min_evidence_count": 2, "retrieve_top_k_per_query": 3}


@st.cache_resource(show_spinner="Building the RAG pipeline (process → embed → index)…")
def load_pipeline():
    """Initialize the shared pipeline once, then attach the agentic module.

    Cached across reruns so the (relatively) expensive ingest/embed/index step
    runs only on the first query.
    """
    agrag = AutoGluonRAG(config_file=CONFIG, data_dir=DATA_DIR)

    # Override the single shared generator via the project's existing config
    # object (Arguments) BEFORE the pipeline is built. Both paths read this.
    override = resolve_generator()
    if override is not None:
        platform, model, platform_args = override
        agrag.args.generator_model_name = model
        agrag.args.generator_model_platform = platform
        agrag.args.generator_model_platform_args = platform_args

    agrag.initialize_rag_pipeline()
    # Reuse the same retriever + generator for the agentic path (no re-ingest).
    agrag.agentic_module = AgenticRAGModule(
        agrag.retriever_module,
        agrag.generator_module,
        config=agent_config(),
    )
    return agrag


def run_static(agrag, query):
    """Static path: retrieve context, then single-pass generate."""
    t0 = time.perf_counter()
    context = agrag.retrieve_context_for_query(query) or []
    answer = agrag.generate_response(query)  # mode defaults to "standard"
    elapsed = time.perf_counter() - t0
    return {"answer": answer, "context": context, "elapsed": elapsed}


def run_agentic(agrag, query):
    """Agentic path: bounded reasoning loop with plan/verify/abstain + trace."""
    t0 = time.perf_counter()
    answer, trace = agrag.generate_response(query, mode="agentic", return_trace=True)
    elapsed = time.perf_counter() - t0
    return {"answer": answer, "trace": trace, "elapsed": elapsed}


# ----------------------------------------------------------------------------- UI
st.set_page_config(page_title="Static vs. Agentic RAG", layout="wide")
st.title("AutoGluon-RAG — Static vs. Agentic (live demo)")

st.caption(
    "Same retriever **and the same generator model**, two answering strategies. "
    "Compare the *answers* and the *process* (retrieved chunks vs. plan → "
    "per-subquery retrieval → evidence → verification → abstention)."
)

_override = resolve_generator()
if _override is None:
    # huggingface override -> keep whatever local_config.yaml sets (Bedrock Claude
    # Haiku 4.5 by default).
    _gen_platform, _gen_model = "config default", "local_config.yaml (Bedrock Claude Haiku 4.5)"
else:
    _gen_platform, _gen_model, _ = _override

with st.sidebar:
    st.header("Config")
    st.write(f"**Config file:** `{CONFIG}`")
    st.write(f"**Data dir:** `{DATA_DIR}`")
    st.write(f"**Generator:** `{_gen_platform}` · `{_gen_model}`")
    st.caption(
        "Shared by both paths. Override with `AGRAG_GENERATOR_PLATFORM` "
        "(bedrock|openai|huggingface) and `AGRAG_GENERATOR_MODEL`. No API keys in code — "
        "bedrock uses the AWS credential chain, openai reads `OPENAI_API_KEY`."
    )
    st.write("**Static:** `generate_response(query)`")
    st.write("**Agentic:** `generate_response(query, mode='agentic', return_trace=True)`")
    st.json(agent_config())
    if st.button("Reset pipeline cache"):
        st.cache_resource.clear()
        st.rerun()

query = st.text_input("Query", value=DEFAULT_QUERY)
go = st.button("Run both pipelines", type="primary")

if go and query.strip():
    agrag = load_pipeline()

    with st.spinner("Running static and agentic paths…"):
        static = run_static(agrag, query)
        agentic = run_agentic(agrag, query)

    col_static, col_agentic = st.columns(2)

    # ---- Static ----
    with col_static:
        st.subheader("🟦 Static RAG")
        st.metric("Latency (s)", f"{static['elapsed']:.2f}")
        st.markdown("**Answer**")
        st.code(static["answer"], language="text")
        st.markdown(f"**Retrieved context** ({len(static['context'])} chunks)")
        for i, chunk in enumerate(static["context"], 1):
            text = chunk.get("text", chunk) if isinstance(chunk, dict) else chunk
            with st.expander(f"Chunk {i}"):
                st.write(text)

    # ---- Agentic ----
    with col_agentic:
        st.subheader("🟩 Agentic RAG")
        trace = agentic["trace"]
        metrics = trace.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Latency (s)", f"{agentic['elapsed']:.2f}")
        c2.metric("Status", trace.get("status", "?"))
        c3.metric("Iterations", metrics.get("iterations", "?"))

        st.markdown("**Answer**")
        st.code(agentic["answer"], language="text")

        st.markdown("**Metrics**")
        st.json(metrics)

        if trace.get("plan"):
            st.markdown("**Plan (retrieval queries)**")
            for p in trace["plan"]:
                st.write(f"- {p}")

        if trace.get("subqueries"):
            st.markdown("**Subqueries**")
            for sq in trace["subqueries"]:
                st.write(f"- {sq}")

        verification = trace.get("verification")
        if verification is not None:
            st.markdown("**Verification**")
            st.json(verification)

        steps = trace.get("steps", [])
        if steps:
            st.markdown(f"**Reasoning steps** ({len(steps)})")
            for step in steps:
                label = f"iter {step.get('iteration')} · {step.get('action_type')}"
                if step.get("tool_name"):
                    label += f" · tool={step['tool_name']}"
                with st.expander(label):
                    st.json(step)

        evidence = trace.get("evidence", [])
        if evidence:
            st.markdown(f"**Evidence** ({len(evidence)} items)")
            for ev in evidence:
                cited = "✅ cited" if ev.get("used_in_answer") else "—"
                header = f"{ev.get('evidence_id')} · {cited}"
                with st.expander(header):
                    st.write(ev.get("text", ""))
                    st.caption(
                        f"retrieval_query={ev.get('retrieval_query')!r} · "
                        f"rank={ev.get('rank')} · score={ev.get('retrieval_score')}"
                    )

    st.divider()
    st.markdown(
        "**What to notice:** the static path does one retrieve → one generate. The "
        "agentic path plans queries, retrieves per subquery, accumulates deduplicated "
        "evidence, verifies the draft against that evidence, and **abstains** when the "
        "evidence is insufficient (here `min_evidence_count`) instead of answering "
        "regardless — visible in `status`, `verification`, and the evidence `cited` flags."
    )
elif go:
    st.warning("Enter a non-empty query.")
