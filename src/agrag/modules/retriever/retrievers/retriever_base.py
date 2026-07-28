import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agrag.constants import CHUNK_ID_KEY, DOC_ID_KEY, DOC_TEXT_KEY, EMBEDDING_KEY, LOGGER_NAME, PARENT_ID_KEY
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.fusion import dedup_records, mmr, reciprocal_rank_fusion
from agrag.modules.retriever.rerankers.reranker import Reranker
from agrag.modules.retriever.retrievers.sparse_retriever import BM25Retriever
from agrag.modules.vector_db.vector_database import VectorDatabaseModule

logger = logging.getLogger(LOGGER_NAME)


class RetrieverModule:
    """
    Initializes the RetrieverModule with the VectorDatabaseModule.

    Attributes:
    ----------
    vector_database_module : VectorDatabaseModule
        The module containing the vector database and metadata.
    embedding_model: EmbeddingModule,
        The module for generating embeddings.
    top_k: int
        The top-k documents to retrieve (default is 50).
        If this is set to 0, no documents will be retrieved and the generator will be used without providing additional context.
    reranker: Reranker
        Optional Reranker object to use for reranking
    use_reranker: bool
        Whether or not to use a reranker.
    sparse_retriever : Optional[BM25Retriever]
        Optional BM25 sparse retriever for hybrid retrieval. When provided and
        ``use_hybrid`` is True, its lexical hits are fused with the dense hits.
    use_hybrid : bool
        Whether to combine dense and sparse (BM25) retrieval. Default False keeps
        the legacy dense-only behavior.
    use_rrf : bool
        Whether to fuse ranked lists with Reciprocal Rank Fusion. Implied by
        ``use_hybrid``; kept separate so RRF can also be toggled independently.
    rrf_k : int
        The RRF constant (default 60).
    dense_weight, sparse_weight : float
        Per-signal weights applied during RRF fusion.
    use_mmr : bool
        Whether to apply Maximal Marginal Relevance for diversity after reranking.
    mmr_lambda : float
        MMR relevance/diversity trade-off in ``[0, 1]``.
    chunk_read : int
        Neighbor window: after retrieval, expand each chunk with +/- this many
        adjacent chunks (same document) and its parent chunk when a parent store
        is present. 0 disables expansion (default).
    parent_store : Optional[pd.DataFrame]
        Parent-chunk store (columns ``parent_id``, ``text``) produced by
        parent-child chunking. Used by ``chunk_read`` to expand a child into its
        parent context. None when parent-child chunking was not used.
    **kwargs : dict
        Additional parameters for `RetrieverModule`.

    Methods:
    -------
    encode_query(query: str) -> np.ndarray:
        Encodes the query into an embedding.

    retrieve(query: str, return_metadata: bool = False) -> List[Any]:
        Retrieves the top_k most similar embeddings to the query. Returns text
        chunks by default, or structured records when ``return_metadata`` is True.
    """

    def __init__(
        self,
        vector_database_module: VectorDatabaseModule,
        embedding_module: EmbeddingModule,
        top_k: int = 50,
        reranker: Reranker = None,
        use_reranker: bool = True,
        sparse_retriever: Optional[BM25Retriever] = None,
        use_hybrid: bool = False,
        use_rrf: bool = False,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        use_mmr: bool = False,
        mmr_lambda: float = 0.5,
        chunk_read: int = 0,
        parent_store: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        self.embedding_module = embedding_module
        self.vector_database_module = vector_database_module
        self.top_k = top_k
        self.reranker = None
        if use_reranker:
            assert isinstance(reranker, Reranker), "reranker must be of type <class> Reranker"
            self.reranker = reranker

        self.sparse_retriever = sparse_retriever
        self.use_hybrid = use_hybrid
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda
        self.chunk_read = chunk_read
        self.parent_store = parent_store
        # Cache: parent_id -> parent text, built lazily from the parent store.
        self._parent_text_cache: Optional[Dict[Any, str]] = None

    @property
    def _uses_advanced_path(self) -> bool:
        """True when any hybrid/fusion/diversity/expansion feature is enabled.

        When False, ``retrieve`` runs the original dense-only code path unchanged,
        preserving legacy behavior byte-for-byte. A present ``parent_store`` also
        opts in, so parent-child (small-to-big) expansion takes effect whenever an
        index built with parent-child chunking is loaded.
        """
        return bool(
            self.use_hybrid or self.use_rrf or self.use_mmr or self.chunk_read or self.parent_store is not None
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes the query into an embedding.

        Parameters:
        ----------
        query : str
            The query to be encoded.

        Returns:
        -------
        np.ndarray
            The embedding of the query.
        """
        if self.embedding_module.model_platform == "bedrock" and "cohere" in self.embedding_module.model_name:
            self.embedding_module.bedrock_embedding_params["input_type"] = "search_query"
        query_embedding = self.embedding_module.encode(data=pd.DataFrame([{DOC_TEXT_KEY: query}]))
        query_embedding = query_embedding[EMBEDDING_KEY][0]
        return query_embedding

    def retrieve(self, query: str, return_metadata: bool = False, top_k: int = None) -> List[Any]:
        """
        Retrieves the top_k most similar embeddings to the query.

        Parameters:
        ----------
        query : str
            The query to retrieve documents for.
        return_metadata : bool
            If False (default), returns a list of text chunks, preserving the
            original behavior. If True, returns a list of structured records
            (dicts) that keep provenance metadata: ``doc_id`` and ``chunk_id``
            from ingest, plus ``rank`` (position in the returned order, updated
            after reranking). It also carries ``source`` provenance (file path
            or URL) when the ingest metadata has it, ``retrieval_score`` (the raw
            vector-DB similarity/distance), and ``rerank_score`` when a reranker
            ran. The agentic RAG path uses the structured form to build
            ``Evidence`` objects.
        top_k : int, optional
            Overrides the module's ``top_k`` for this call only. If None (default),
            the module-level ``self.top_k`` is used, preserving existing behavior.
            The agentic path uses this to retrieve a per-query number of chunks.

        Returns:
        -------
        List[str] or List[Dict[str, Any]]
            Text chunks (default) or structured records when ``return_metadata``
            is True. Returns None if the vector database search yields no valid
            indices.
        """
        effective_top_k = self.top_k if top_k is None else top_k

        # Hybrid / fusion / MMR / chunk_read path. Only taken when a feature is
        # enabled; otherwise the original dense-only logic below runs unchanged.
        if self._uses_advanced_path:
            return self._retrieve_advanced(query, return_metadata, effective_top_k)

        logger.info(f"\nRetrieving top {effective_top_k} most similar embeddings")
        query_embedding = self.encode_query(query)

        if not return_metadata:
            # Original behavior: indices only, return text chunks.
            indices = self.vector_database_module.search_vector_database(
                embedding=query_embedding, top_k=effective_top_k
            )
            valid_indices = [idx for idx in indices if idx < self.vector_database_module.metadata.shape[0]]
            if not valid_indices:
                logger.warning("No valid indices returned from the vector database search.")
                return None
            records = self.vector_database_module.metadata.iloc[valid_indices].to_dict(orient="records")
            retrieved_content = [chunk["text"] for chunk in records]
            if self.reranker:
                retrieved_content = self.reranker.rerank(query, retrieved_content)
            return retrieved_content

        # Structured behavior: also pull the raw similarity scores so evidence
        # can carry provenance and scoring.
        indices, scores = self.vector_database_module.search_vector_database(
            embedding=query_embedding, top_k=effective_top_k, return_scores=True
        )
        # Keep scores aligned with indices while filtering out-of-range indices.
        valid_pairs = [
            (idx, score) for idx, score in zip(indices, scores) if idx < self.vector_database_module.metadata.shape[0]
        ]
        if not valid_pairs:
            logger.warning("No valid indices returned from the vector database search.")
            return None
        valid_indices = [idx for idx, _ in valid_pairs]
        valid_scores = [score for _, score in valid_pairs]

        records = self.vector_database_module.metadata.iloc[valid_indices].to_dict(orient="records")

        # Keep metadata aligned with each chunk, tagging pre-rerank rank + score.
        text_to_record = {}
        for rank, (record, score) in enumerate(zip(records, valid_scores)):
            record["rank"] = rank
            record["retrieval_score"] = score
            # Map text -> record so we can realign after reranking. If the same
            # text appears more than once, the first occurrence is kept.
            text_to_record.setdefault(record["text"], record)

        if self.reranker:
            reranked = self.reranker.rerank(query, [record["text"] for record in records], return_scores=True)
            ordered_records = []
            for new_rank, (text, rerank_score) in enumerate(reranked):
                record = dict(text_to_record.get(text, {"text": text}))
                record["rank"] = new_rank
                record["rerank_score"] = rerank_score
                ordered_records.append(record)
            return ordered_records

        return records

    # ------------------------------------------------------------------
    # Advanced retrieval (hybrid fusion + global rerank + MMR + chunk_read)
    # ------------------------------------------------------------------
    def _dense_hits(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run dense retrieval, returning per-hit records tagged with signal/rank."""
        query_embedding = self.encode_query(query)
        indices, scores = self.vector_database_module.search_vector_database(
            embedding=query_embedding, top_k=top_k, return_scores=True
        )
        n_rows = self.vector_database_module.metadata.shape[0]
        hits = []
        rank = 0
        for idx, score in zip(indices, scores):
            if idx >= n_rows:
                continue
            record = self.vector_database_module.metadata.iloc[idx].to_dict()
            record["row_index"] = int(idx)
            record["rank"] = rank
            record["retrieval_score"] = score
            record["signal"] = "dense"
            record["retrieval_query"] = query
            hits.append(record)
            rank += 1
        return hits

    def _sparse_hits(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Run BM25 retrieval, returning per-hit records tagged with signal/rank."""
        if self.sparse_retriever is None:
            return []
        # Lazily build the BM25 index the first time it is used. The corpus (the
        # metadata ``text`` column) is only populated after the vector DB is
        # constructed or loaded, which happens after retriever init.
        if not getattr(self.sparse_retriever, "_built", False):
            metadata = self.vector_database_module.metadata
            if metadata is None or metadata.empty:
                logger.warning("Cannot build BM25 index: vector DB metadata is empty.")
                return []
            self.sparse_retriever.build(metadata[DOC_TEXT_KEY].tolist())
        n_rows = self.vector_database_module.metadata.shape[0]
        hits = []
        rank = 0
        for idx, score in self.sparse_retriever.search(query, top_k):
            if idx >= n_rows:
                continue
            record = self.vector_database_module.metadata.iloc[idx].to_dict()
            record["row_index"] = int(idx)
            record["rank"] = rank
            record["sparse_score"] = score
            record["signal"] = "sparse"
            record["retrieval_query"] = query
            hits.append(record)
            rank += 1
        return hits

    def _retrieve_advanced(self, query: str, return_metadata: bool, top_k: int):
        """Dense + sparse hybrid retrieval with RRF fusion, global rerank, MMR,
        and chunk_read expansion. Shared by standard and agentic callers."""
        logger.info(f"\nHybrid retrieval: top {top_k} (hybrid={self.use_hybrid}, rrf={self.use_rrf})")

        dense_hits = self._dense_hits(query, top_k)
        sparse_hits = self._sparse_hits(query, top_k) if self.use_hybrid else []

        if not dense_hits and not sparse_hits:
            logger.warning("No valid indices returned from hybrid retrieval.")
            return None

        # Fuse the ranked lists on row index (RRF). A single list fuses to its own
        # order, so dense-only + RRF is a no-op ordering change.
        records = self._fuse(dense_hits, sparse_hits)

        # One global cross-encoder rerank over the fused candidate pool.
        if self.reranker:
            records = self._rerank_records(query, records)

        # Optional diversity re-ordering.
        if self.use_mmr and len(records) > 1:
            records = self._apply_mmr(query, records)

        # Optional context expansion (neighbor window + parent chunk).
        if self.chunk_read or self.parent_store is not None:
            records = self._expand_context(records)

        # Finalize ranks and strip internal bookkeeping keys.
        for new_rank, record in enumerate(records):
            record["rank"] = new_rank
            record.pop("row_index", None)
            record.pop("signal", None)

        if not return_metadata:
            return [record.get("text", "") for record in records]
        return records

    def _fuse(self, dense_hits: List[Dict[str, Any]], sparse_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fuse dense + sparse hits into one provenance-preserving record list."""
        # Merge duplicate hits (same chunk from dense and sparse) first so each
        # chunk keeps every signal/query that surfaced it.
        merged = dedup_records(dense_hits + sparse_hits)
        by_index = {record["row_index"]: record for record in merged}

        if self.use_rrf or self.use_hybrid:
            ranked_lists = [[h["row_index"] for h in dense_hits]]
            weights = [self.dense_weight]
            if sparse_hits:
                ranked_lists.append([h["row_index"] for h in sparse_hits])
                weights.append(self.sparse_weight)
            fused = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k, weights=weights)
            ordered = []
            for row_index, rrf_score in fused:
                record = by_index.get(row_index)
                if record is None:
                    continue
                record["rrf_score"] = rrf_score
                ordered.append(record)
            return ordered

        # No RRF: keep dedup order (dense first, then sparse extras).
        return merged

    def _rerank_records(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply the cross-encoder globally, keying results back to records by
        identity (not text) so identical text from different docs is not merged."""
        texts = [record.get("text", "") for record in records]
        # Queue of records per text preserves identity when texts repeat.
        text_to_records: Dict[str, deque] = defaultdict(deque)
        for record in records:
            text_to_records[record.get("text", "")].append(record)

        reranked = self.reranker.rerank(query, texts, return_scores=True)
        ordered = []
        for text, rerank_score in reranked:
            queue = text_to_records.get(text)
            if not queue:
                continue
            record = queue.popleft()
            record["rerank_score"] = rerank_score
            ordered.append(record)
        return ordered

    def _apply_mmr(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-order records for diversity via MMR over freshly-encoded embeddings."""
        texts = [record.get("text", "") for record in records]
        try:
            encoded = self.embedding_module.encode(data=pd.DataFrame({DOC_TEXT_KEY: texts}))
            candidate_embeddings = list(encoded[EMBEDDING_KEY].values)
            query_embedding = self.encode_query(query)
        except Exception as exc:  # pragma: no cover - defensive; MMR is optional
            logger.warning("MMR embedding failed (%s); skipping diversity re-ordering.", exc)
            return records
        keys = list(range(len(records)))
        selected = mmr(
            query_embedding=np.asarray(query_embedding),
            candidate_keys=keys,
            candidate_embeddings=candidate_embeddings,
            lambda_mult=self.mmr_lambda,
        )
        return [records[i] for i in selected]

    def _build_parent_cache(self) -> Dict[Any, str]:
        if self._parent_text_cache is None:
            cache: Dict[Any, str] = {}
            if self.parent_store is not None and not self.parent_store.empty:
                for _, row in self.parent_store.iterrows():
                    cache[row[PARENT_ID_KEY]] = row.get(DOC_TEXT_KEY, "")
            self._parent_text_cache = cache
        return self._parent_text_cache

    def _expand_context(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand each retrieved child chunk into larger context.

        Precise retrieval stays keyed on the child chunk (``doc_id``/``chunk_id``
        are preserved and the child text is kept as ``child_text``), but ``text``
        is replaced with an expanded context: a +/-``chunk_read`` neighbor window
        of adjacent chunks in the same document, and/or the parent chunk when a
        parent store is available.
        """
        metadata = self.vector_database_module.metadata
        has_doc = DOC_ID_KEY in metadata.columns
        has_chunk = CHUNK_ID_KEY in metadata.columns
        parent_cache = self._build_parent_cache()

        for record in records:
            child_text = record.get("text", "")
            record["child_text"] = child_text
            pieces: List[str] = []

            # Parent chunk (small-to-big): prefer the larger parent context.
            parent_id = record.get(PARENT_ID_KEY)
            if parent_id is not None and parent_id in parent_cache:
                pieces.append(parent_cache[parent_id])

            # Neighbor window within the same document.
            if self.chunk_read and has_doc and has_chunk and record.get(CHUNK_ID_KEY) is not None:
                doc_id = record.get(DOC_ID_KEY)
                chunk_id = record.get(CHUNK_ID_KEY)
                lo, hi = chunk_id - self.chunk_read, chunk_id + self.chunk_read
                window = metadata[
                    (metadata[DOC_ID_KEY] == doc_id) & (metadata[CHUNK_ID_KEY] >= lo) & (metadata[CHUNK_ID_KEY] <= hi)
                ].sort_values(CHUNK_ID_KEY)
                neighbor_text = "\n".join(str(t) for t in window[DOC_TEXT_KEY].tolist())
                record["neighbor_chunk_ids"] = [int(c) for c in window[CHUNK_ID_KEY].tolist()]
                if neighbor_text:
                    pieces.append(neighbor_text)

            if pieces:
                record["text"] = "\n\n".join(pieces)
        return records
