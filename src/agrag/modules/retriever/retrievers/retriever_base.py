import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agrag.constants import DOC_TEXT_KEY, EMBEDDING_KEY, LOGGER_NAME
from agrag.modules.embedding.embedding import EmbeddingModule
from agrag.modules.retriever.rerankers.reranker import Reranker
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
        **kwargs,
    ):
        self.embedding_module = embedding_module
        self.vector_database_module = vector_database_module
        self.top_k = top_k
        self.reranker = None
        if use_reranker:
            assert isinstance(reranker, Reranker), "reranker must be of type <class> Reranker"
            self.reranker = reranker

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
            (idx, score)
            for idx, score in zip(indices, scores)
            if idx < self.vector_database_module.metadata.shape[0]
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
