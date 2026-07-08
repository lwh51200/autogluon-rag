"""Answer verification for the agentic RAG path.

The ``AnswerVerifier`` checks whether a draft answer is supported by the retrieved
evidence and returns a structured label. It reuses the single configured
``GeneratorModule`` as an LLM judge (per the design's "reuse one generator"
decision), with robust parsing of the model's label output.
"""

import logging
from enum import Enum
from typing import Any, Dict

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import EvidenceStore

logger = logging.getLogger(LOGGER_NAME)


class VerificationLabel(str, Enum):
    """Structured verification labels (design section 3)."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Longest labels first so substring matching does not match a prefix by mistake
# (e.g. "partially_supported" must be checked before "supported").
_LABELS_BY_LENGTH = sorted(VerificationLabel, key=lambda label: len(label.value), reverse=True)

_VERIFY_INSTRUCTION = (
    "You are a strict verifier. Given a QUESTION, a draft ANSWER, and the "
    "EVIDENCE used to produce it, decide how well the evidence supports the "
    "answer. Reply with exactly one of these labels and nothing else: "
    "supported, partially_supported, unsupported, conflicting_evidence, "
    "insufficient_evidence.\n\n"
)


class AnswerVerifier:
    """LLM-backed verifier returning a structured label.

    Attributes:
    ----------
    generator_module : GeneratorModule
        The generator used as the verification judge.
    min_evidence_count : int
        If fewer than this many evidence items are available, the verifier short
        circuits to ``insufficient_evidence`` without calling the model.
    max_context_tokens : int
        Approximate cap on the size of the evidence block placed in the verifier
        prompt (approximated by whitespace token count, matching the
        synthesizer). Prevents concatenating unbounded evidence into a single
        prompt, which would overflow the model's context window. At least the
        first evidence item is always included.
    """

    def __init__(self, generator_module, min_evidence_count: int = 2, max_context_tokens: int = 6000):
        self.generator_module = generator_module
        self.min_evidence_count = min_evidence_count
        self.max_context_tokens = max_context_tokens

    @staticmethod
    def _parse_label(text: str) -> VerificationLabel:
        """Map raw model output to a label; default to unsupported if unclear."""
        lowered = (text or "").strip().lower()
        for label in _LABELS_BY_LENGTH:
            if label.value in lowered:
                return label
        logger.debug("Verifier could not parse label from %r; defaulting to unsupported", text)
        return VerificationLabel.UNSUPPORTED

    def _build_evidence_block(self, evidence_store: EvidenceStore) -> str:
        """Join evidence text into a bounded block for the verifier prompt.

        Includes evidence in order until the approximate token budget
        (whitespace-delimited words) is reached, always keeping at least the
        first item. Mirrors ``AnswerSynthesizer._select_evidence`` so the
        verifier judges against the same slice of evidence the answer was
        grounded in.
        """
        lines = []
        used = 0
        for ev in evidence_store:
            approx_tokens = len(ev.text.split())
            if lines and used + approx_tokens > self.max_context_tokens:
                break
            lines.append(f"- {ev.text}")
            used += approx_tokens
        return "\n".join(lines)

    def verify(self, query: str, draft_answer: str, evidence_store: EvidenceStore) -> Dict[str, Any]:
        """Return a verification result dict with a structured label.

        Returns:
        -------
        Dict[str, Any]
            ``{"label": <str>, "is_supported": <bool>, "evidence_count": <int>}``.
            ``is_supported`` is True only for the ``supported`` label.
        """
        evidence_count = len(evidence_store)
        if evidence_count < self.min_evidence_count:
            label = VerificationLabel.INSUFFICIENT_EVIDENCE
            return self._result(label, evidence_count)

        evidence_block = self._build_evidence_block(evidence_store)
        prompt = f"{_VERIFY_INSTRUCTION}QUESTION: {query}\n\nANSWER: {draft_answer}\n\n" f"EVIDENCE:\n{evidence_block}"
        raw = self.generator_module.generate_response(prompt)
        label = self._parse_label(raw)
        logger.debug("Verifier label: %s", label.value)
        return self._result(label, evidence_count)

    @staticmethod
    def _result(label: VerificationLabel, evidence_count: int) -> Dict[str, Any]:
        return {
            "label": label.value,
            "is_supported": label == VerificationLabel.SUPPORTED,
            "evidence_count": evidence_count,
        }
