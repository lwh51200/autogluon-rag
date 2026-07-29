"""Bounded agent loop for the agentic RAG path.

The ``AgentExecutor`` runs the loop described in the design (section 4): plan,
then repeatedly choose an action via the ``DecisionPolicy``, run it through the
``ToolRegistry`` or synthesize/verify an answer, update state and evidence, and
stop when the answer is accepted, the agent abstains, or ``max_iterations`` is
reached. It owns no persistent state — one execution operates on one
``AgentState`` and one ``EvidenceStore``.
"""

import logging
from typing import Optional, Tuple

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.policy import ActionType, DecisionPolicy
from agrag.modules.agentic.state import AgentState, AgentStatus
from agrag.modules.agentic.tools.registry import ToolRegistry

logger = logging.getLogger(LOGGER_NAME)


class AgentExecutor:
    """Runs the bounded agent loop.

    Attributes:
    ----------
    tool_registry : ToolRegistry
        The tools the agent may call.
    policy : DecisionPolicy
        Chooses the next action each iteration.
    planner : QueryPlanner
        Produces the initial retrieval plan.
    synthesizer : AnswerSynthesizer
        Builds the grounded answer from evidence.
    verifier : Optional[AnswerVerifier]
        Verifies the draft answer; if None, verification is skipped and any draft
        is accepted.
    max_iterations : int
        Hard cap on loop iterations.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy: DecisionPolicy,
        planner,
        synthesizer,
        verifier=None,
        max_iterations: int = 5,
    ):
        self.tool_registry = tool_registry
        self.policy = policy
        self.planner = planner
        self.synthesizer = synthesizer
        self.verifier = verifier
        self.max_iterations = max_iterations

    def _ingest_evidence(self, state: AgentState, evidence_store: EvidenceStore, result) -> int:
        """Store any evidence from a tool result; track ids on the state."""
        if not getattr(result, "contains_evidence", False):
            return 0
        added = 0
        new_ids = []
        for ev in result.evidence:
            if evidence_store.add(ev):
                added += 1
                new_ids.append(ev.evidence_id)
        state.add_evidence_ids(new_ids)
        return added

    def _draft_and_verify(self, state: AgentState, evidence_store: EvidenceStore) -> Tuple[Optional[str], bool]:
        """Synthesize an answer and (optionally) verify it.

        Synthesis and verification always target ``original_query`` — the
        immutable user question. Query rewrites only steer *retrieval* (via
        ``current_query``); the answer and its verification must address what the
        user actually asked, never a rewritten working query.

        Returns (answer, accepted).
        """
        answer, used_ids = self.synthesizer.generate(
            state.original_query, evidence_store, compressed_context=state.compressed_context
        )
        evidence_store.mark_used(used_ids)
        state.draft_answer = answer

        if self.verifier is None:
            state.set_verification({"label": "unverified", "is_supported": True})
            state.record_action("draft_answer", observation_summary="drafted (unverified)")
            return answer, True

        verification = self.verifier.verify(state.original_query, answer, evidence_store)
        state.set_verification(verification)
        accepted = self.policy.accept_verification(verification)
        state.record_action(
            "draft_answer",
            observation_summary=f"drafted; verification={verification.get('label')}",
        )
        return answer, accepted

    def run(self, query: str) -> Tuple[AgentState, EvidenceStore, Optional[str]]:
        """Execute the loop for a query.

        Returns the terminal ``AgentState``, the ``EvidenceStore``, and the final
        answer (``None`` if the agent abstained).
        """
        state = AgentState(original_query=query)
        evidence_store = EvidenceStore()
        state.plan = self.planner.create_plan(query)
        # Subqueries are the plan entries beyond the original query.
        state.subqueries = state.plan[1:]

        final_answer: Optional[str] = None

        while state.iteration < self.max_iterations:
            action = self.policy.next_action(state, evidence_store)
            logger.debug("Iteration %d: action=%s", state.iteration, action.type.value)

            if action.type == ActionType.ABSTAIN:
                state.finish(AgentStatus.ABSTAINED)
                return state, evidence_store, None

            if action.type == ActionType.DRAFT_ANSWER:
                answer, accepted = self._draft_and_verify(state, evidence_store)
                if accepted:
                    final_answer = answer
                    state.finish(AgentStatus.ANSWERED)
                    return state, evidence_store, final_answer
                # Not accepted: fall through to another iteration (may rewrite,
                # retrieve more, or eventually abstain / hit the cap).
                state.iteration += 1
                continue

            if action.type == ActionType.REWRITE_QUERY:
                result = self.tool_registry.run("QueryRewriteTool", query=action.args["query"])
                if result.output:
                    state.set_current_query(result.output)
                state.record_action(
                    "rewrite_query",
                    tool_name="QueryRewriteTool",
                    args=action.args,
                    observation_summary=result.summary,
                )
                state.iteration += 1
                continue

            if action.type == ActionType.COMPRESS_CONTEXT:
                result = self.tool_registry.run(
                    "ContextCompressionTool", query=action.args["query"], texts=action.args["texts"]
                )
                if result.output:
                    state.compressed_context = result.output
                state.record_action(
                    "compress_context",
                    tool_name="ContextCompressionTool",
                    observation_summary=result.summary,
                )
                state.iteration += 1
                continue

            # Retrieval actions.
            if action.type == ActionType.RETRIEVE:
                result = self.tool_registry.run("RetrieveTool", query=action.args["query"])
                tool_name = "RetrieveTool"
            elif action.type == ActionType.MULTI_RETRIEVE:
                # Pass the immutable user question so the fused path's single
                # global cross-encoder rerank scores candidates against what the
                # user actually asked, not against any one subquery.
                result = self.tool_registry.run(
                    "MultiQueryRetrieveTool",
                    queries=action.args["queries"],
                    original_query=state.original_query,
                )
                tool_name = "MultiQueryRetrieveTool"
            else:
                raise ValueError(f"Unhandled action type: {action.type}")

            added = self._ingest_evidence(state, evidence_store, result)
            state.record_action(
                action.type.value,
                tool_name=tool_name,
                args=action.args,
                observation_summary=result.summary,
                evidence_added=added,
            )
            state.iteration += 1

        # Loop budget exhausted. Return the best draft if we have one, else abstain.
        if state.draft_answer is not None and self.policy.accept_verification(state.verification):
            state.finish(AgentStatus.MAX_ITERATIONS)
            return state, evidence_store, state.draft_answer

        state.finish(AgentStatus.MAX_ITERATIONS)
        return state, evidence_store, None
