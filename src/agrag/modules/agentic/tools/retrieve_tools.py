"""Retrieval tools that wrap the existing RetrieverModule.

These tools do not re-ingest, re-chunk, or re-embed anything. They call the
existing ``RetrieverModule.retrieve(..., return_metadata=True)`` at query time and
convert the structured records into ``Evidence`` objects.
"""

import logging
from typing import Dict, List

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, LOGGER_NAME
from agrag.modules.agentic.evidence import Evidence
from agrag.modules.agentic.tools.base import Tool, ToolResult
from agrag.modules.retriever.fusion import dedup_records, reciprocal_rank_fusion

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

    def __init__(self, retriever_module, top_k: int = None, use_fused_retrieval: bool = False, rrf_k: int = 60):
        self.retriever_module = retriever_module
        self.top_k = top_k
        self.use_fused_retrieval = use_fused_retrieval
        self.rrf_k = rrf_k

    def run(self, queries: List[str], **kwargs) -> ToolResult:
        if self.use_fused_retrieval:
            return self._run_fused(queries)

        all_evidence: List[Evidence] = []
        for query in queries:
            records = self.retriever_module.retrieve(query, return_metadata=True, top_k=self.top_k)
            all_evidence.extend(_records_to_evidence(records, retrieval_query=query, tool_name=self.name))
        logger.debug("%s retrieved %d chunks across %d subqueries", self.name, len(all_evidence), len(queries))
        return self._result(
            evidence=all_evidence,
            summary=f"retrieved {len(all_evidence)} chunks across {len(queries)} subqueries",
        )

    def _dedup_key(self, record: Dict) -> object:
        doc_id = record.get(DOC_ID_KEY)
        chunk_id = record.get(CHUNK_ID_KEY)
        if doc_id is not None and chunk_id is not None:
            return (doc_id, chunk_id)
        return ("text", record.get(DOC_TEXT_KEY, ""))

    def _run_fused(self, queries: List[str]) -> ToolResult:
        """Fuse per-subquery results globally with RRF instead of concatenating.

        Each subquery's records form one ranked list; the lists are fused with
        RRF over ``(doc_id, chunk_id)`` identity, deduped while preserving every
        subquery that surfaced a chunk, then reranked once globally by the
        retriever's cross-encoder (when present). This gives one globally-ordered
        evidence set whose provenance still spans all subgoals.
        """
        per_query_records: List[List[Dict]] = []
        all_records: List[Dict] = []
        for query in queries:
            records = self.retriever_module.retrieve(query, return_metadata=True, top_k=self.top_k) or []
            tagged = []
            for record in records:
                item = dict(record)
                item["retrieval_query"] = query
                item["signal"] = query
                tagged.append(item)
            per_query_records.append(tagged)
            all_records.extend(tagged)

        if not all_records:
            return self._result(evidence=[], summary=f"retrieved 0 chunks across {len(queries)} subqueries (fused)")

        # Dedup while preserving all subquery provenance, keyed by chunk identity.
        deduped = dedup_records(all_records, key_fn=self._dedup_key)
        by_key = {self._dedup_key(r): r for r in deduped}

        # RRF fuse the per-subquery ranked lists on chunk identity.
        ranked_lists = [[self._dedup_key(r) for r in recs] for recs in per_query_records]
        fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)

        ordered: List[Dict] = []
        for fusion_rank, (key, rrf_score) in enumerate(fused):
            record = by_key.get(key)
            if record is None:
                continue
            record["rrf_score"] = rrf_score
            record["fusion_rank"] = fusion_rank
            ordered.append(record)

        # One global cross-encoder rerank over the fused pool, when available.
        reranker = getattr(self.retriever_module, "reranker", None)
        if reranker is not None and ordered:
            texts = [r.get(DOC_TEXT_KEY, "") for r in ordered]
            reranked = reranker.rerank(queries[0], texts, return_scores=True)
            text_to_records: Dict[str, List[Dict]] = {}
            for record in ordered:
                text_to_records.setdefault(record.get(DOC_TEXT_KEY, ""), []).append(record)
            reranked_order = []
            for text, rerank_score in reranked:
                bucket = text_to_records.get(text)
                if not bucket:
                    continue
                record = bucket.pop(0)
                record["rerank_score"] = rerank_score
                reranked_order.append(record)
            if reranked_order:
                ordered = reranked_order

        evidence: List[Evidence] = []
        for rank, record in enumerate(ordered):
            primary_query = record.get("retrieval_query", queries[0])
            record["retrieval_queries"] = record.get("source_signals", []) or record.get("retrieval_queries", [])
            item = Evidence.from_retrieval_record(
                record, retrieval_query=primary_query, rank=rank, tool_name=self.name
            )
            evidence.append(item)

        logger.debug("%s fused %d chunks across %d subqueries", self.name, len(evidence), len(queries))
        return self._result(
            evidence=evidence,
            summary=f"fused {len(evidence)} chunks across {len(queries)} subqueries",
        )
