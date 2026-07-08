"""Query planning for the agentic RAG path.

The ``QueryPlanner`` prepares retrieval queries. For the MVP it is a simple
rule-based component: it does NOT answer the user's question directly, it only
produces one or more retrieval queries (subqueries) from the input query.
"""

import logging
import re
from typing import List

from agrag.constants import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

# Conjunctions / cues that often separate distinct information needs in a query.
_SPLIT_PATTERN = re.compile(
    r"\b(?:and|versus|vs\.?|compared to|as well as)\b|[;?]",
    flags=re.IGNORECASE,
)


class QueryPlanner:
    """Rule-based planner that derives retrieval subqueries from a query.

    Attributes:
    ----------
    max_subqueries : int
        Upper bound on the number of subqueries produced.
    """

    def __init__(self, max_subqueries: int = 4):
        self.max_subqueries = max_subqueries

    def create_plan(self, query: str) -> List[str]:
        """Return a list of retrieval queries for the given user query.

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
        logger.debug("Planner produced %d subqueries for %r", len(plan), query)
        return plan
