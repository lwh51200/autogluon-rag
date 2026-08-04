"""Query planning for the agentic RAG path.

The ``QueryPlanner`` prepares retrieval queries: it does NOT answer the user's
question directly, it only produces one or more retrieval queries (subqueries /
subgoals) from the input query.

Three modes are supported, in precedence order Strands > LLM > rule-based:

* **Rule-based** (default): a simple regex that splits the query on conjunctions
  and punctuation. Deterministic and dependency-free.
* **LLM-backed** (opt-in via ``use_llm`` + a ``generator_module``): the query is
  decomposed by the configured generator (the same shared LLM used elsewhere in
  the agentic path). The model's output is parsed and *validated* back into the
  same ``List[str]`` contract — original query first, deduped, capped at
  ``max_subqueries`` — and any malformed output falls back to the rule-based plan.
  There is no pydantic dependency; validation is stdlib JSON + type checks, in
  the same "constrained output + tolerant parse + safe fallback" spirit as
  ``AnswerVerifier``.
* **Strands-backed** (opt-in via a ``strands_backend``): the query is decomposed
  by a Strands agent driving Bedrock Haiku 4.5, which returns *only* subquery
  strings (Pydantic-validated structured output). Those strings go through the
  same normalization as the LLM path, so the ``List[str]`` contract is identical.
  Any failure falls back to the LLM path (if configured) and then to rules.

In every mode the LLM emits only the subqueries; Python owns the normalization
(original query first, dedup, cap), so the executor's contract can't be broken.
"""

import json
import logging
import re
from typing import List, Optional

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Conjunctions / cues that often separate distinct information needs in a query.
_SPLIT_PATTERN = re.compile(
    r"\b(?:and|versus|vs\.?|compared to|as well as)\b|[;?]",
    flags=re.IGNORECASE,
)

# Prompt for the LLM planner. It must decompose (not answer) and emit strict JSON
# so the output can be validated without a schema library.
_PLAN_INSTRUCTION = (
    "You are a retrieval query planner for a RAG system. Decompose the user's "
    "question into focused search subqueries (subgoals) that, retrieved "
    "together, would surface the evidence needed to answer it. Do NOT answer the "
    "question. Prefer a single subquery for simple questions; only split when the "
    "question bundles distinct information needs (e.g. multi-hop or comparison). "
    "Return at most {max_subqueries} subqueries.\n\n"
    'Return ONLY a JSON object of the form {{"subqueries": ["...", "..."]}} with '
    "no markdown, no code fences, and no explanation.\n\n"
    "Question: "
)


class QueryPlanner:
    """Planner that derives retrieval subqueries from a query.

    Attributes:
    ----------
    max_subqueries : int
        Upper bound on the number of subqueries produced (the original query
        counts as the first entry).
    generator_module : Optional[GeneratorModule]
        The shared generator used when ``use_llm`` is enabled. When ``None`` the
        planner is always rule-based regardless of ``use_llm``.
    use_llm : bool
        Whether to use the LLM to decompose the query. Effective only when a
        ``generator_module`` is also provided.
    strands_backend : Optional[StrandsReasoner]
        A Strands/Bedrock reasoning helper. When provided it takes precedence
        over the raw-LLM path: the model emits only subquery strings, which are
        normalized the same way. ``None`` disables the Strands path.
    """

    def __init__(self, max_subqueries: int = 4, generator_module=None, use_llm: bool = False, strands_backend=None):
        self.max_subqueries = max_subqueries
        self.generator_module = generator_module
        self.use_llm = use_llm
        self.strands_backend = strands_backend

    @property
    def _llm_enabled(self) -> bool:
        return self.use_llm and self.generator_module is not None

    def create_plan(self, query: str) -> List[str]:
        """Return a list of retrieval queries for the given user query.

        The original query is always the first entry. Precedence is
        Strands > raw-LLM > rule-based: each configured mode is tried in turn and
        any failure falls through to the next, so the loop never breaks.
        """
        if self.strands_backend is not None:
            strands_plan = self._strands_plan(query)
            if strands_plan:
                logger.debug("Strands planner produced %d subqueries for %r", len(strands_plan), query)
                return strands_plan
            logger.debug("Strands planner produced no usable plan for %r; falling back", query)
        if self._llm_enabled:
            llm_plan = self._llm_plan(query)
            if llm_plan:
                logger.debug("LLM planner produced %d subqueries for %r", len(llm_plan), query)
                return llm_plan
            logger.debug("LLM planner produced no usable plan for %r; falling back to rules", query)
        return self._rule_based_plan(query)

    def _strands_plan(self, query: str) -> Optional[List[str]]:
        """Decompose the query via the Strands backend; normalize its output.

        The backend returns only subquery strings; normalization (original query
        first, dedup, cap) is applied here so the ``List[str]`` contract matches
        the other modes. Returns ``None`` on any failure so the caller falls back.
        """
        try:
            subqueries = self.strands_backend.plan_subqueries(query, self.max_subqueries)
        except Exception as exc:  # backend must never break the loop
            logger.debug("Strands planner call failed (%s); falling back", exc)
            return None
        if not subqueries:
            return None
        return self._normalize_subqueries(subqueries, query)

    def _rule_based_plan(self, query: str) -> List[str]:
        """Derive subqueries by splitting on conjunctions / punctuation.

        The original query is always included first. If the query appears to
        bundle multiple information needs (e.g. contains "and", "versus", "?"),
        the parts are added as additional subqueries, up to ``max_subqueries``.
        """
        query = query.strip()
        plan: List[str] = [query] if query else []

        parts = [p.strip() for p in _SPLIT_PATTERN.split(query) if p and p.strip()]
        # Only treat as multi-part when splitting actually produced >1 meaningful
        # part and each part is a reasonable length (avoids splitting on stray
        # punctuation into tiny fragments).
        meaningful = [p for p in parts if len(p.split()) >= 2]
        if len(meaningful) > 1:
            for part in meaningful:
                if part not in plan:
                    plan.append(part)

        plan = plan[: self.max_subqueries]
        logger.debug("Rule-based planner produced %d subqueries for %r", len(plan), query)
        return plan

    def _llm_plan(self, query: str) -> Optional[List[str]]:
        """Ask the generator to decompose the query; validate its output.

        Returns a validated ``List[str]`` (original query first) or ``None`` when
        the model output cannot be parsed/validated, so the caller falls back.
        """
        prompt = _PLAN_INSTRUCTION.format(max_subqueries=self.max_subqueries) + query
        try:
            raw = self.generator_module.generate_response(prompt)
        except Exception as exc:  # generator/backend failure must not break the loop
            logger.debug("LLM planner generation failed (%s); falling back to rules", exc)
            return None
        return self._parse_subqueries(raw, query)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Best-effort parse of a JSON object embedded in model output.

        Slices from the first ``{`` to the last ``}`` so leading/trailing prose or
        code fences do not defeat parsing. Returns the parsed dict, or ``None`` if
        no JSON object can be recovered.
        """
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    def _parse_subqueries(self, raw: str, original_query: str) -> Optional[List[str]]:
        """Validate the model output into a capped, deduped ``List[str]``.

        Delegates normalization to ``_normalize_subqueries``. Returns ``None`` on
        any parse/validation failure so the caller falls back.
        """
        obj = self._extract_json(raw)
        if obj is None:
            logger.debug("LLM planner output was not valid JSON: %r", raw)
            return None
        subqueries = obj.get("subqueries")
        if not isinstance(subqueries, list):
            logger.debug("LLM planner JSON missing a 'subqueries' list: %r", obj)
            return None
        return self._normalize_subqueries(subqueries, original_query)

    def _normalize_subqueries(self, subqueries: List, original_query: str) -> Optional[List[str]]:
        """Normalize raw model subqueries into the executor's ``List[str]`` contract.

        The plan always leads with the original query; only non-empty string
        subqueries not already present are appended, capped at ``max_subqueries``.
        Shared by the LLM and Strands paths so both yield identical shapes.
        Returns ``None`` when nothing usable remains so the caller falls back.
        """
        plan: List[str] = [original_query.strip()] if original_query.strip() else []
        for item in subqueries:
            if not isinstance(item, str):
                continue
            cleaned = item.strip()
            if cleaned and cleaned not in plan:
                plan.append(cleaned)

        plan = plan[: self.max_subqueries]
        return plan or None
