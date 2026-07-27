"""Adapters for the MultiHop-RAG benchmark (Tang & Yang, 2024).

MultiHop-RAG (HuggingFace: ``yixuantt/MultiHopRAG``) is a RAG-specific
benchmark whose evidence for each query is distributed across 2-4 documents. It
is published as *two* subsets:

* ``corpus``      -- 609 news articles (fields: ``body``, ``title``, ``source``,
                     ``author``, ``category``, ``published_at``, ``url``). This
                     is the knowledge base that gets ingested + indexed.
* ``MultiHopRAG`` -- 2,556 queries (fields: ``query``, ``answer``,
                     ``question_type``, ``evidence_list``). This drives the
                     evaluation.

Because the corpus and the queries live in different subsets, MultiHop-RAG does
NOT fit the single-dataset assumption of ``EvaluationModule.run_evaluation``
(which derives queries from the same rows it ingests, as Google NQ does). The
companion ``local_example/benchmark_multihoprag.py`` runner ingests the corpus
separately and then feeds queries through the pipeline, using these adapters.

The ``question_type`` field is the reason this benchmark separates agentic from
static RAG: ``comparison_query`` and ``null_query`` (answer absent from the
corpus) exercise multi-step retrieval and know-when-to-stop behaviour that a
single-shot retrieve-then-read pipeline cannot do. Always report the per-type
breakdown, not just an aggregate -- the gap concentrates in those buckets.
"""


def preprocess_multihop_rag_corpus(row):
    """Build the ingested document text for one ``corpus`` subset row.

    Prepends the title so lexical/semantic retrieval can match on it, then the
    article body. Mirrors the ``preprocess_*`` contract used by the Google NQ
    adapter (dataset row -> plain text for ingestion).

    Parameters
    ----------
    row : dict
        A row from the ``corpus`` subset.

    Returns
    -------
    str
        The document text to ingest and index.
    """
    title = row.get("title") or ""
    body = row.get("body") or ""
    if title:
        return f"{title}\n\n{body}"
    return body


def get_multihop_rag_query(row):
    """Extract the question from a ``MultiHopRAG`` subset row.

    Parameters
    ----------
    row : dict
        A row from the ``MultiHopRAG`` (query) subset.

    Returns
    -------
    str
        The multi-hop query.
    """
    return row["query"]


def get_multihop_rag_responses(row):
    """Extract the expected answer(s) from a ``MultiHopRAG`` subset row.

    Returns a list (the ``EvaluationModule`` metrics expect a list of
    references per query) even though MultiHop-RAG ships a single gold answer.
    ``null_query`` rows have the literal answer ``"Insufficient information."``;
    that string is preserved so a correct "I don't know" is scored as a match.

    Parameters
    ----------
    row : dict
        A row from the ``MultiHopRAG`` (query) subset.

    Returns
    -------
    List[str]
        The expected response(s).
    """
    answer = row.get("answer")
    if answer is None or answer == "":
        return []
    return [answer]


def get_multihop_rag_evidence_facts(row):
    """Extract the gold supporting-evidence snippets for retrieval scoring.

    Each item in a query's ``evidence_list`` carries a ``fact`` -- the exact text
    snippet (drawn from a corpus article) that supports the answer. These are the
    gold relevant passages used to compute Hit@k / MRR / evidence-coverage against
    what the pipeline retrieved. ``null_query`` rows have no supporting facts in
    the corpus, so this returns an empty list for them (retrieval metrics are
    undefined and those queries are skipped in the retrieval breakdown).

    Parameters
    ----------
    row : dict
        A row from the ``MultiHopRAG`` (query) subset.

    Returns
    -------
    List[str]
        The gold supporting-fact snippets (possibly empty).
    """
    evidence_list = row.get("evidence_list") or []
    facts = []
    for item in evidence_list:
        fact = item.get("fact") if isinstance(item, dict) else None
        if fact:
            facts.append(fact)
    return facts


def get_multihop_rag_question_type(row):
    """Extract the query category used for the per-type metric breakdown.

    Parameters
    ----------
    row : dict
        A row from the ``MultiHopRAG`` (query) subset.

    Returns
    -------
    str
        One of ``inference_query``, ``comparison_query``, ``temporal_query``,
        ``null_query`` (or ``"unknown"`` if the field is absent).
    """
    return row.get("question_type", "unknown")
