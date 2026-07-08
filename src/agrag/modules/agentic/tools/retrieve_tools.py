"""Retrieval tools that wrap the existing RetrieverModule.

These tools do not re-ingest, re-chunk, or re-embed anything. They call the
existing ``RetrieverModule.retrieve(..., return_metadata=True)`` at query time and
convert the structured records into ``Evidence`` objects.
"""

import logging
from typing import List

from agrag.constants import LOGGER_NAME
from agrag.modules.agentic.evidence import Evidence
from agrag.modules.agentic.tools.base import Tool, ToolResult

logger = logging.getLogger(LOGGER_NAME)


def _records_to_evidence(records, retrieval_query: str, tool_name: str) -> List[Evidence]:
    """Convert retriever records into Evidence objects.

    ``records`` may be ``None`` (no valid indices) or a list of dicts. When the
    retriever is configured to return text only (a list of strings), each string
    is wrapped as text-only evidence so the tool still degrades gracefully.
    """
    if not records:
        return []
    evidence = []
    for rank, record in enumerate(records):
        if isinstance(record, str):
            evidence.append(Evidence(text=record, retrieval_query=retrieval_query, rank=rank, tool_name=tool_name))
        else:
            item = Evidence.from_retrieval_record(
                record, retrieval_query=retrieval_query, rank=record.get("rank", rank), tool_name=tool_name
            )
            evidence.append(item)
    return evidence


class RetrieveTool(Tool):
    """Run a single retrieval query against the existing retriever.

    Attributes:
    ----------
    top_k : Optional[int]
        Per-query number of chunks to retrieve. If None, the retriever's own
        ``top_k`` is used.
    """

    name = "RetrieveTool"

    def __init__(self, retriever_module, top_k: int = None):
        self.retriever_module = retriever_module
        self.top_k = top_k

    def run(self, query: str, **kwargs) -> ToolResult:
        records = self.retriever_module.retrieve(query, return_metadata=True, top_k=self.top_k)
        evidence = _records_to_evidence(records, retrieval_query=query, tool_name=self.name)
        logger.debug("%s retrieved %d chunks for %r", self.name, len(evidence), query)
        return self._result(
            evidence=evidence,
            summary=f"retrieved {len(evidence)} chunks for query",
        )


class MultiQueryRetrieveTool(Tool):
    """Run retrieval over several subqueries and merge the results.

    Deduplication across subqueries is left to the ``EvidenceStore``; this tool
    simply returns all evidence it produced, tagged with the subquery that found
    each item.

    Attributes:
    ----------
    top_k : Optional[int]
        Per-subquery number of chunks to retrieve. If None, the retriever's own
        ``top_k`` is used.
    """

    name = "MultiQueryRetrieveTool"

    def __init__(self, retriever_module, top_k: int = None):
        self.retriever_module = retriever_module
        self.top_k = top_k

    def run(self, queries: List[str], **kwargs) -> ToolResult:
        all_evidence: List[Evidence] = []
        for query in queries:
            records = self.retriever_module.retrieve(query, return_metadata=True, top_k=self.top_k)
            all_evidence.extend(_records_to_evidence(records, retrieval_query=query, tool_name=self.name))
        logger.debug("%s retrieved %d chunks across %d subqueries", self.name, len(all_evidence), len(queries))
        return self._result(
            evidence=all_evidence,
            summary=f"retrieved {len(all_evidence)} chunks across {len(queries)} subqueries",
        )
