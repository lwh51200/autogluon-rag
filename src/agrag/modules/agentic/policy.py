"""Decision policy for the agentic RAG loop.

The ``DecisionPolicy`` decides the next action: retrieve more evidence, rewrite
the query, compress context, draft an answer, or abstain. It is a rule-based
policy driven by the current ``AgentState`` and the collected evidence.

Unlike a fixed pipeline, the policy *reacts to observations*: it retrieves again
after a query rewrite, and it changes course when a draft fails verification
(rewriting to gather better evidence rather than re-drafting the same answer).
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import EvidenceStore
from agrag.modules.agentic.state import AgentState
from agrag.modules.agentic.verifier import VerificationLabel

logger = logging.getLogger(LOGGER_NAME)


class ActionType(str, Enum):
    """Actions the policy can choose."""

    RETRIEVE = "retrieve"
    MULTI_RETRIEVE = "multi_retrieve"
    REWRITE_QUERY = "rewrite_query"
    COMPRESS_CONTEXT = "compress_context"
    DRAFT_ANSWER = "draft_answer"
    ABSTAIN = "abstain"


@dataclass
class Action:
    """A chosen action plus its arguments."""

    type: ActionType
    args: Dict[str, Any] = field(default_factory=dict)


class DecisionPolicy:
    """Rule-based next-action policy.

    Attributes:
    ----------
    min_evidence_count : int
        Minimum evidence required before drafting an answer.
    use_query_rewrite : bool
        Whether the policy may choose to rewrite the query (after weak retrieval
        or a failed verification).
    use_context_compression : bool
        Whether the policy may choose to compress oversized context before
        drafting.
    max_context_tokens : int
        Approximate context-token budget; when the collected evidence exceeds it
        and compression is enabled, the policy compresses before drafting.
    max_rewrites : int
        Upper bound on query rewrites within a single run.
    max_iterations : Optional[int]
        The executor's loop budget. When set, the policy will not choose to
        rewrite the query on the final iteration: a rewrite is only useful if
        there is budget left to re-retrieve *and* re-draft afterwards. Rewriting
        with no remaining budget guarantees an abstention and wastes an LLM call.
        When None, the policy assumes budget is always available (legacy
        behavior).
    """

    # A rewrite must be followed by at least a re-retrieve and a re-draft to be
    # worth doing, so it needs this many iterations left after the current one.
    _REWRITE_FOLLOWUP_ITERATIONS = 2

    def __init__(
        self,
        min_evidence_count: int = 2,
        use_query_rewrite: bool = True,
        use_context_compression: bool = False,
        max_context_tokens: int = 6000,
        max_rewrites: int = 1,
        max_iterations: int = None,
    ):
        self.min_evidence_count = min_evidence_count
        self.use_query_rewrite = use_query_rewrite
        self.use_context_compression = use_context_compression
        self.max_context_tokens = max_context_tokens
        self.max_rewrites = max_rewrites
        self.max_iterations = max_iterations

    def _rewrite_count(self, state: AgentState) -> int:
        return sum(1 for r in state.history if r.action_type == ActionType.REWRITE_QUERY.value)

    def _has_retrieved(self, state: AgentState) -> bool:
        """Whether *any* retrieval has run this run (regardless of query)."""
        return any(
            r.action_type in (ActionType.RETRIEVE.value, ActionType.MULTI_RETRIEVE.value) for r in state.history
        )

    def _rewrite_has_budget(self, state: AgentState) -> bool:
        """Whether enough loop iterations remain to act on a rewrite.

        A rewrite consumes the current iteration and only pays off if the
        rewritten query can be retrieved and re-drafted before the loop ends.
        When ``max_iterations`` is unknown, assume budget is available.
        """
        if self.max_iterations is None:
            return True
        remaining_after_rewrite = self.max_iterations - (state.iteration + 1)
        return remaining_after_rewrite >= self._REWRITE_FOLLOWUP_ITERATIONS

    def _can_rewrite(self, state: AgentState) -> bool:
        return (
            self.use_query_rewrite
            and self._rewrite_count(state) < self.max_rewrites
            and self._rewrite_has_budget(state)
        )

    def _last_draft_failed(self, state: AgentState) -> bool:
        """True when the most recent draft was verified and rejected.

        ``set_current_query`` clears ``verification`` on a rewrite, so a stale
        rejection from a previous query does not keep forcing rewrites.
        """
        if state.draft_answer is None or state.verification is None:
            return False
        return not self.accept_verification(state.verification)

    def _compression_attempted(self, state: AgentState) -> bool:
        """Whether compression has already been tried for the current query.

        Prevents re-compressing every iteration when the tool returns nothing
        usable (which leaves ``compressed_context`` None).
        """
        return any(
            r.action_type == ActionType.COMPRESS_CONTEXT.value and r.query == state.current_query
            for r in state.history
        )

    def _context_too_large(self, evidence_store: EvidenceStore) -> bool:
        """Approximate whether collected evidence exceeds the token budget."""
        approx_tokens = sum(len(ev.text.split()) for ev in evidence_store)
        return approx_tokens > self.max_context_tokens

    def next_action(self, state: AgentState, evidence_store: EvidenceStore) -> Action:
        """Choose the next action given the current state and evidence.

        Rules:
        1. If nothing has been retrieved for the *current* query, retrieve (multi
           on the first retrieval when the plan has subqueries, else single). A
           rewrite changes the current query, so this fires again and the
           rewritten query is actually used.
        2. If evidence is below the minimum: rewrite the query (if allowed and
           under the rewrite budget) so retrieval reruns, otherwise abstain.
        3. If the latest draft failed verification: rewrite to gather better
           evidence (if allowed), otherwise abstain. This replaces the previous
           behavior of re-drafting the identical answer until the loop expired.
        4. If context compression is enabled and the evidence exceeds the token
           budget, compress before drafting.
        5. Otherwise: draft an answer.
        """
        if not state.retrieved_for_current_query():
            # Multi-query retrieval applies to the initial plan only; a rewritten
            # query is a single reformulation.
            if not self._has_retrieved(state) and len(state.plan) > 1:
                return Action(ActionType.MULTI_RETRIEVE, {"queries": state.plan})
            return Action(ActionType.RETRIEVE, {"query": state.current_query})

        if len(evidence_store) < self.min_evidence_count:
            if self._can_rewrite(state):
                return Action(ActionType.REWRITE_QUERY, {"query": state.current_query})
            return Action(ActionType.ABSTAIN, {})

        if self._last_draft_failed(state):
            if self._can_rewrite(state):
                return Action(ActionType.REWRITE_QUERY, {"query": state.current_query})
            return Action(ActionType.ABSTAIN, {})

        if (
            self.use_context_compression
            and state.compressed_context is None
            and not self._compression_attempted(state)
            and self._context_too_large(evidence_store)
        ):
            return Action(
                ActionType.COMPRESS_CONTEXT,
                {"query": state.current_query, "texts": evidence_store.texts()},
            )

        return Action(ActionType.DRAFT_ANSWER, {})

    def accept_verification(self, verification: Dict[str, Any]) -> bool:
        """Whether a verification result is good enough to return the answer."""
        label = (verification or {}).get("label")
        return label in (
            VerificationLabel.SUPPORTED.value,
            VerificationLabel.PARTIALLY_SUPPORTED.value,
        )
