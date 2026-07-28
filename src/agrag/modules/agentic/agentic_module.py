"""Top-level controller for the agentic RAG path.

``AgenticRAGModule`` wires together the planner, tools, policy, synthesizer, and
verifier, runs the bounded loop via ``AgentExecutor``, and returns either a
supported answer or an abstention message (optionally with a structured trace).

It reuses the already-initialized ``RetrieverModule`` and ``GeneratorModule`` from
the RAG pipeline; it never re-ingests, re-chunks, or re-embeds data.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.executor import AgentExecutor
from agrag.modules.agentic.planner import QueryPlanner
from agrag.modules.agentic.policy import DecisionPolicy
from agrag.modules.agentic.synthesizer import AnswerSynthesizer
from agrag.modules.agentic.tools.llm_tools import ContextCompressionTool, QueryRewriteTool
from agrag.modules.agentic.tools.registry import ToolRegistry
from agrag.modules.agentic.tools.retrieve_tools import MultiQueryRetrieveTool, RetrieveTool
from agrag.modules.agentic.trace import AgentTrace
from agrag.modules.agentic.verifier import AnswerVerifier

logger = logging.getLogger(LOGGER_NAME)

DEFAULT_ABSTENTION = (
    "I don't have enough supporting evidence in the indexed documents to answer " "this question confidently."
)


class AgenticRAGModule:
    """Controller for agentic answering.

    Attributes:
    ----------
    retriever_module : RetrieverModule
        Existing retriever, wrapped by the retrieval tools.
    generator_module : GeneratorModule
        Existing generator, used for synthesis, verification, and (optionally)
        query rewrite / context compression.
    config : dict
        Agent configuration (see configs/agent/default.yaml). Recognized keys:
        max_iterations, max_subqueries, retrieve_top_k_per_query,
        use_query_rewrite, use_context_compression, use_verification,
        min_evidence_count, max_context_tokens, query_prefix.
    """

    def __init__(self, retriever_module, generator_module, config: Optional[Dict[str, Any]] = None):
        self.retriever_module = retriever_module
        self.generator_module = generator_module
        cfg = config or {}

        self.max_iterations = cfg.get("max_iterations", 5)
        self.max_subqueries = cfg.get("max_subqueries", 4)
        self.retrieve_top_k_per_query = cfg.get("retrieve_top_k_per_query", None)
        self.use_query_rewrite = cfg.get("use_query_rewrite", True)
        self.use_context_compression = cfg.get("use_context_compression", False)
        self.use_verification = cfg.get("use_verification", True)
        self.min_evidence_count = cfg.get("min_evidence_count", 2)
        self.max_context_tokens = cfg.get("max_context_tokens", 6000)
        self.query_prefix = cfg.get("query_prefix", "")

        self._build_components()

    def _build_components(self) -> None:
        tools = [
            RetrieveTool(self.retriever_module, top_k=self.retrieve_top_k_per_query),
            MultiQueryRetrieveTool(self.retriever_module, top_k=self.retrieve_top_k_per_query),
        ]
        if self.use_query_rewrite:
            tools.append(QueryRewriteTool(self.generator_module))
        if self.use_context_compression:
            tools.append(ContextCompressionTool(self.generator_module))
        self.tool_registry = ToolRegistry(tools)

        self.planner = QueryPlanner(max_subqueries=self.max_subqueries)
        self.synthesizer = AnswerSynthesizer(
            self.generator_module,
            max_context_tokens=self.max_context_tokens,
            query_prefix=self.query_prefix,
        )
        self.verifier = (
            AnswerVerifier(
                self.generator_module,
                min_evidence_count=self.min_evidence_count,
                max_context_tokens=self.max_context_tokens,
            )
            if self.use_verification
            else None
        )
        self.policy = DecisionPolicy(
            min_evidence_count=self.min_evidence_count,
            use_query_rewrite=self.use_query_rewrite,
            use_context_compression=self.use_context_compression,
            max_context_tokens=self.max_context_tokens,
            max_iterations=self.max_iterations,
        )
        self.executor = AgentExecutor(
            tool_registry=self.tool_registry,
            policy=self.policy,
            planner=self.planner,
            synthesizer=self.synthesizer,
            verifier=self.verifier,
            max_iterations=self.max_iterations,
        )

    def answer(self, query: str, return_trace: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Answer a query via the agentic loop.

        Returns the answer string (or an abstention message). When
        ``return_trace`` is True, returns ``(answer, trace_dict)``.
        """
        state, evidence_store, final_answer = self.executor.run(query)

        answer = final_answer if final_answer is not None else DEFAULT_ABSTENTION
        logger.info("Agentic run finished with status=%s", state.status.value)

        if return_trace:
            trace = AgentTrace.from_run(state, evidence_store, final_answer)
            return answer, trace.to_dict()
        return answer
