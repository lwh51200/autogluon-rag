"""Retrieval-quality metrics for RAG evaluation (Hit@k, MRR, evidence coverage).

Answer exact-match tells you whether the final text was right; it does not tell
you whether the *retriever* surfaced the passages needed to get there. On a
multi-hop benchmark that distinction is the whole point: the static-vs-agentic
gap shows up first in retrieval, because the agentic path issues multiple
sub-queries and accumulates evidence across rounds. These metrics score the
retrieved passages against the benchmark's gold supporting facts.

Matching heuristic
------------------
Gold relevance in MultiHop-RAG is a set of ``fact`` snippets (sentences taken
from corpus articles). The pipeline retrieves fixed-size *chunks* (e.g. 128
tokens), so a gold fact almost never equals a retrieved chunk verbatim. A gold
fact is therefore counted as "retrieved" by a chunk when most of the fact's
content tokens appear in that chunk -- token *containment* above a threshold
(default 0.6). This is deliberately lenient on chunk boundaries but strict on
content, which is the right trade-off for chunked retrieval.

The metrics are corpus-agnostic: they take the ranked list of retrieved chunk
texts and the list of gold fact strings, so they work for any benchmark that
provides gold evidence snippets, not just MultiHop-RAG.
"""

import re
from typing import Dict, List

# Minimal stopword set: dropping these keeps containment focused on the content
# words that actually distinguish one fact from another.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "from", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "he", "she", "they", "them",
    "his", "her", "their", "has", "have", "had", "will", "would", "can", "could",
    "s", "t",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> List[str]:
    """Lowercase, split to alphanumeric tokens, drop stopwords."""
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


def fact_is_retrieved(fact: str, chunk_text: str, threshold: float = 0.6) -> bool:
    """True if ``chunk_text`` contains enough of ``fact``'s content tokens.

    Containment = (# of fact content-tokens present in the chunk's token set) /
    (# of fact content-tokens). Uses set membership so repetition does not skew
    the ratio. A fact with no content tokens can never be matched (returns False).
    """
    fact_tokens = _content_tokens(fact)
    if not fact_tokens:
        return False
    chunk_tokens = set(_content_tokens(chunk_text))
    present = sum(1 for tok in set(fact_tokens) if tok in chunk_tokens)
    return (present / len(set(fact_tokens))) >= threshold


def _first_hit_rank(retrieved_texts: List[str], gold_facts: List[str], threshold: float) -> int:
    """1-indexed rank of the first retrieved chunk matching ANY gold fact, else 0."""
    for rank, chunk in enumerate(retrieved_texts, start=1):
        if any(fact_is_retrieved(fact, chunk, threshold) for fact in gold_facts):
            return rank
    return 0


def _evidence_coverage(retrieved_texts: List[str], gold_facts: List[str], threshold: float) -> float:
    """Fraction of DISTINCT gold facts matched by at least one retrieved chunk.

    This is the multi-hop-specific signal: a single-shot retriever may nail one
    hop (coverage ~= 1/n) while missing the others; a multi-round agentic
    retriever should cover more of the required facts.
    """
    if not gold_facts:
        return 0.0
    covered = sum(
        1 for fact in gold_facts if any(fact_is_retrieved(fact, chunk, threshold) for chunk in retrieved_texts)
    )
    return covered / len(gold_facts)


def retrieval_metrics_for_query(
    retrieved_texts: List[str],
    gold_facts: List[str],
    k_values=(1, 3, 5, 10),
    threshold: float = 0.6,
) -> Dict[str, float]:
    """Compute per-query retrieval metrics against gold evidence facts.

    Parameters
    ----------
    retrieved_texts : List[str]
        Retrieved chunk texts, best-first (rank 1 = index 0).
    gold_facts : List[str]
        Gold supporting-fact snippets for this query.
    k_values : tuple of int
        Cutoffs for Hit@k.
    threshold : float
        Token-containment threshold for counting a fact as retrieved.

    Returns
    -------
    Dict[str, float]
        ``hit@k`` for each k (1.0 if any gold fact appears in the top k, else 0),
        ``mrr`` (reciprocal of the first hit's rank, 0 if none), and
        ``evidence_coverage`` (fraction of distinct gold facts retrieved anywhere
        in the returned list).
    """
    first_rank = _first_hit_rank(retrieved_texts, gold_facts, threshold)
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = 1.0 if (first_rank and first_rank <= k) else 0.0
    metrics["mrr"] = (1.0 / first_rank) if first_rank else 0.0
    metrics["evidence_coverage"] = _evidence_coverage(retrieved_texts, gold_facts, threshold)
    return metrics


def aggregate_retrieval_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean each metric across queries. Returns zeros-free empty dict if no queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    n = len(per_query)
    return {key: round(sum(q[key] for q in per_query) / n, 4) for key in keys}
