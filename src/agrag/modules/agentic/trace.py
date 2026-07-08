"""Trace schema for the agentic RAG path.

When ``generate_response(..., return_trace=True)`` is used, the agent returns a
structured trace alongside the answer. The trace is a serializable snapshot that
supports debugging, citation inspection, and the performance metrics described in
the design (latency, retrieval-call count, tool-call count, evidence count). It
assembles data already held by ``AgentState`` and ``EvidenceStore`` — it does not
own any new state.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.state import AgentState


@dataclass
class AgentTrace:
    """A serializable record of one agentic run.

    Attributes:
    ----------
    original_query : str
        The user's query.
    final_answer : Optional[str]
        The returned answer, or ``None`` when the agent abstained.
    status : str
        Terminal status of the run (see ``AgentStatus``).
    verification : Optional[Dict[str, Any]]
        The final verification result (label + details), if any. Sourced from
        ``AgentState.last_verification`` so it survives a rewrite that cleared the
        per-query ``verification`` right before the run ended.
    plan : List[str]
        Retrieval queries produced by the planner.
    subqueries : List[str]
        Subqueries used for multi-query retrieval.
    steps : List[Dict[str, Any]]
        Ordered (action, observation) records from ``AgentState.history``.
    evidence : List[Dict[str, Any]]
        Serialized evidence collected during the run.
    metrics : Dict[str, Any]
        Aggregate counters (iterations, tool calls, retrieval calls, evidence,
        cited-evidence count) plus any externally supplied timing/cost values.
    """

    original_query: str
    final_answer: Optional[str] = None
    status: str = ""
    verification: Optional[Dict[str, Any]] = None
    plan: List[str] = field(default_factory=list)
    subqueries: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the trace to plain dicts (JSON-friendly)."""
        return asdict(self)

    @classmethod
    def from_run(
        cls,
        state: AgentState,
        evidence_store: EvidenceStore,
        final_answer: Optional[str],
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> "AgentTrace":
        """Assemble a trace from the state and evidence store after a run.

        Parameters:
        ----------
        state : AgentState
            The (terminal) state of the run.
        evidence_store : EvidenceStore
            The evidence collected during the run.
        final_answer : Optional[str]
            The answer returned to the caller (``None`` if abstained).
        extra_metrics : Optional[Dict[str, Any]]
            Externally measured values to merge into ``metrics`` (e.g. latency
            seconds, token usage) that the state does not track itself.
        """
        # retrieval-call count = tool steps whose tool name mentions "retrieve"
        retrieval_calls = sum(
            1 for record in state.history if record.tool_name and "retrieve" in record.tool_name.lower()
        )
        cited = sum(1 for ev in evidence_store if ev.used_in_answer)

        metrics: Dict[str, Any] = {
            "iterations": state.iteration,
            "tool_calls": state.tool_call_count,
            "retrieval_calls": retrieval_calls,
            "evidence_count": len(evidence_store),
            "cited_evidence_count": cited,
        }
        if extra_metrics:
            metrics.update(extra_metrics)

        return cls(
            original_query=state.original_query,
            final_answer=final_answer,
            status=state.status.value,
            # Prefer the run-level verification (survives a rewrite) so the trace
            # reflects the last draft's verdict even after a final-iteration
            # rewrite cleared the per-query verification.
            verification=state.last_verification or state.verification,
            plan=list(state.plan),
            subqueries=list(state.subqueries),
            steps=[record.to_dict() for record in state.history],
            evidence=evidence_store.to_list(),
            metrics=metrics,
        )
