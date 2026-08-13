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
    allow_abstention : bool
        Opt-in switch for refusing to answer. When False (the default) the loop
        never abstains: any ABSTAIN decision and the loop-budget-exhausted fallback
        return the best available draft instead of ``None``, synthesizing one on
        the spot if none was produced yet. Verification still runs (its label is
        recorded in the trace and still steers rewrites toward better evidence); it
        just no longer gates the final return. When True the framework may refuse:
        an ABSTAIN decision returns ``None`` (-> the canned abstention) and the
        budget-exhausted fallback returns a draft only if it passed verification.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        policy: DecisionPolicy,
        planner,
        synthesizer,
        verifier=None,
        max_iterations: int = 5,
        allow_abstention: bool = False,
        use_iterative_planner: bool = False,
    ):
        self.tool_registry = tool_registry
        self.policy = policy
        self.planner = planner
        self.synthesizer = synthesizer
        self.verifier = verifier
        self.max_iterations = max_iterations
        self.allow_abstention = allow_abstention
        # Opt-in sequential-hop execution (default off -> unchanged parallel loop).
        self.use_iterative_planner = use_iterative_planner

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

    # A hop subquery that still contains an unresolved reference to an earlier hop
    # (a pronoun or a definite description like "the city", "that person") cannot be
    # retrieved blind. These cues trigger an LLM fill-in using the prior hops'
    # resolved answers. Deliberately small and lowercase-matched.
    _UNRESOLVED_CUES = (
        "it", "its", "it's", "he", "she", "they", "them", "his", "her", "their",
        "this", "that", "these", "those", "the city", "the person", "the country",
        "the company", "the team", "the author", "the film", "the band", "the place",
    )

    _HOP_FILL_INSTRUCTION = (
        "You are rewriting a follow-up search query in a multi-hop question so it is "
        "self-contained. Using the resolved facts below, replace any pronoun or vague "
        "reference (e.g. 'it', 'that city', 'the person') in the follow-up query with "
        "the concrete entity it refers to. Keep it a concise search query. Do NOT "
        "answer it, and add no explanation.\n\n"
        "Resolved facts:\n{facts}\n\n"
        "Follow-up query: {subquery}\n\n"
        "Rewritten self-contained query:"
    )

    def _needs_resolution(self, subquery: str) -> bool:
        """Heuristic: does this subquery reference an as-yet-unresolved entity?"""
        tokens = set(subquery.lower().replace("?", " ").replace(",", " ").split())
        if tokens & {c for c in self._UNRESOLVED_CUES if " " not in c}:
            return True
        low = subquery.lower()
        return any(cue in low for cue in self._UNRESOLVED_CUES if " " in cue)

    def _resolve_hop_query(self, subquery: str, state: AgentState) -> str:
        """Substitute prior hops' resolved answers into a hop subquery.

        When the subquery contains no unresolved reference, or no prior answers
        exist yet, it is returned unchanged. Otherwise the generator rewrites it
        into a self-contained query using the resolved-fact chain. Any generation
        failure falls back to the original subquery so a hop never dead-ends.
        """
        if not state.hop_answers or not self._needs_resolution(subquery):
            return subquery
        facts = "\n".join(f"- {h['subquery']} -> {h['answer']}" for h in state.hop_answers)
        prompt = self._HOP_FILL_INSTRUCTION.format(facts=facts, subquery=subquery)
        try:
            rewritten = self.generator_module.generate_response(prompt)
        except Exception as exc:  # generation must never break the hop chain
            logger.debug("Hop-query fill-in failed (%s); using original subquery", exc)
            return subquery
        rewritten = (rewritten or "").strip()
        return rewritten or subquery

    @property
    def generator_module(self):
        """The shared generator (reached via the synthesizer) for hop fill-in."""
        return self.synthesizer.generator_module

    def _extract_hop_answer(self, hop_query: str, evidence_store: EvidenceStore) -> str:
        """Extract this hop's intermediate answer by reusing the synthesizer.

        Scoped to the hop question (not ``original_query``) so the resolved answer
        threaded into the next hop is the entity this hop asked for. Evidence is
        NOT marked used here -- the final synthesis over ``original_query`` still
        sees the whole store.
        """
        answer, _ = self.synthesizer.generate(hop_query, evidence_store)
        return (answer or "").strip()

    def _run_sequential(
        self, state: AgentState, evidence_store: EvidenceStore
    ) -> Tuple[AgentState, EvidenceStore, Optional[str]]:
        """Resolve the plan's subqueries in order, threading each hop's answer forward.

        Per hop: build a self-contained hop query (substituting prior answers),
        retrieve via the existing ``RetrieveTool``, ingest evidence, extract the
        hop's intermediate answer, and record it on ``state.hop_answers``. After the
        last hop, synthesize + verify the final answer against ``original_query``
        exactly as the parallel path does. Reuses ``_ingest_evidence``,
        ``_draft_and_verify`` and ``_best_effort_answer`` unchanged.
        """
        for subquery in state.subqueries:
            hop_query = self._resolve_hop_query(subquery, state)
            state.set_current_query(hop_query)
            try:
                result = self.tool_registry.run("RetrieveTool", query=hop_query)
            except Exception as exc:  # a hop failure must not dead-end the run
                logger.debug("Sequential hop retrieval failed (%s); continuing", exc)
                state.iteration += 1
                continue
            added = self._ingest_evidence(state, evidence_store, result)
            hop_answer = self._extract_hop_answer(hop_query, evidence_store)
            state.hop_answers.append(
                {"subquery": subquery, "hop_query": hop_query, "answer": hop_answer}
            )
            state.record_action(
                "retrieve",
                tool_name="RetrieveTool",
                args={"query": hop_query},
                observation_summary=f"hop {len(state.hop_answers)}: {result.summary}; answer={hop_answer!r}",
                evidence_added=added,
            )
            state.iteration += 1

        # Final answer + verification always target the immutable user question.
        state.set_current_query(state.original_query)
        answer, accepted = self._draft_and_verify(state, evidence_store)
        if accepted:
            state.finish(AgentStatus.ANSWERED)
            return state, evidence_store, answer
        # Not accepted: in never-refuse mode return the best-effort draft; if refusal
        # is allowed, abstain rather than return an unverified answer.
        if not self.allow_abstention:
            best = self._best_effort_answer(state, evidence_store)
            state.finish(AgentStatus.ANSWERED)
            return state, evidence_store, best
        state.finish(AgentStatus.ABSTAINED)
        return state, evidence_store, None

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

        # Sequential-hop path (opt-in). Only worth it when the plan actually
        # decomposed into >1 subquery; a single-subquery plan has no chain to
        # thread, so it falls through to the standard loop unchanged.
        if self.use_iterative_planner and len(state.subqueries) >= 2:
            return self._run_sequential(state, evidence_store)

        final_answer: Optional[str] = None

        while state.iteration < self.max_iterations:
            action = self.policy.next_action(state, evidence_store)
            logger.debug("Iteration %d: action=%s", state.iteration, action.type.value)

            if action.type == ActionType.ABSTAIN:
                if not self.allow_abstention:
                    # Never-refuse (default): a forced ABSTAIN terminal means the
                    # policy has exhausted iteration (no rewrite budget / no new
                    # evidence), not that we should return the canned abstention.
                    # Fall back to the best available draft, synthesizing one now if
                    # the loop never produced one. Status reflects the answer.
                    answer = self._best_effort_answer(state, evidence_store)
                    state.finish(AgentStatus.ANSWERED)
                    return state, evidence_store, answer
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

        # Loop budget exhausted.
        state.finish(AgentStatus.MAX_ITERATIONS)
        if not self.allow_abstention:
            # Never-refuse (default): return the best draft, synthesizing one if the
            # budget ran out before any draft was produced.
            return state, evidence_store, self._best_effort_answer(state, evidence_store)
        # Refusal allowed: return an accepted draft if we have one, else abstain
        # (None -> DEFAULT_ABSTENTION) rather than returning an answer the verifier
        # rejected.
        if state.draft_answer is not None and self.policy.accept_verification(state.verification):
            return state, evidence_store, state.draft_answer
        return state, evidence_store, None

    def _best_effort_answer(self, state: AgentState, evidence_store: EvidenceStore) -> str:
        """Return the best available answer for never-refuse mode.

        Uses the existing draft when one was produced; otherwise synthesizes one
        now from whatever evidence was collected (which may be empty — the
        synthesizer still returns a grounded best effort). Never returns ``None``,
        so the caller always yields an answer rather than the canned abstention.
        """
        if state.draft_answer is not None:
            return state.draft_answer
        answer, used_ids = self.synthesizer.generate(
            state.original_query, evidence_store, compressed_context=state.compressed_context
        )
        evidence_store.mark_used(used_ids)
        state.draft_answer = answer
        state.record_action("draft_answer", observation_summary="drafted (never-refuse fallback)")
        return answer
