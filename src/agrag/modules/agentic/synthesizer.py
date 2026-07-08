"""Answer synthesis for the agentic RAG path.

The ``AnswerSynthesizer`` builds a grounded prompt from the collected evidence and
calls the existing ``GeneratorModule``. It reuses ``format_query`` so prompt
formatting stays consistent with the standard RAG path (per-model templates).
"""

import logging
from typing import List, Tuple

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import Evidence, EvidenceStore
from agrag.modules.generator.utils import format_query

logger = logging.getLogger(LOGGER_NAME)


class AnswerSynthesizer:
    """Generates a grounded answer from evidence using the configured generator.

    Attributes:
    ----------
    generator_module : GeneratorModule
        The generator used to produce the answer.
    max_context_tokens : int
        Approximate cap on context size. Evidence is included in order until the
        budget is reached (approximated by whitespace token count).
    """

    def __init__(self, generator_module, max_context_tokens: int = 6000):
        self.generator_module = generator_module
        self.max_context_tokens = max_context_tokens

    def _select_evidence(self, evidence_store: EvidenceStore) -> List[Evidence]:
        """Select evidence up to the (approximate) context-token budget."""
        selected: List[Evidence] = []
        used = 0
        for ev in evidence_store:
            approx_tokens = len(ev.text.split())
            if selected and used + approx_tokens > self.max_context_tokens:
                break
            selected.append(ev)
            used += approx_tokens
        return selected

    def build_context(self, evidence_store: EvidenceStore) -> Tuple[List[str], List[str]]:
        """Return (context_texts, evidence_ids) for the selected evidence."""
        selected = self._select_evidence(evidence_store)
        texts = [f"[{ev.citation()}] {ev.text}" for ev in selected]
        ids = [ev.evidence_id for ev in selected if ev.evidence_id is not None]
        return texts, ids

    def generate(
        self, query: str, evidence_store: EvidenceStore, compressed_context: str = None
    ) -> Tuple[str, List[str]]:
        """Generate an answer grounded in the evidence.

        When ``compressed_context`` is provided (produced by the context
        compression tool), it is used as the context instead of the raw evidence
        chunks. All collected evidence ids are still returned as "used" so
        citation/traceability reflect what fed the compressed summary.

        Returns the answer text and the list of evidence ids that were placed in
        the prompt (the caller can mark these as used).
        """
        if compressed_context:
            context_texts = [compressed_context]
            evidence_ids = [ev.evidence_id for ev in evidence_store if ev.evidence_id is not None]
        else:
            context_texts, evidence_ids = self.build_context(evidence_store)
        formatted = format_query(
            model_name=self.generator_module.model_name,
            query=query,
            context=context_texts,
        )
        answer = self.generator_module.generate_response(formatted)
        logger.debug("Synthesized answer using %d evidence items", len(evidence_ids))
        return answer, evidence_ids
