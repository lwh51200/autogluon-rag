import csv
import re
import string
from typing import List, Union

import numpy as np
from qa_metrics.pedant import PEDANT
from qa_metrics.transformerMatcher import TransformerMatcher


def preprocess_text(
    text: str,
    regexes_to_ignore: List[str] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> str:
    """
    Preprocesses text by applying specified transformations.

    Parameters:
    ----------
    text : str
        The text to be preprocessed.
    regexes_to_ignore : List[str]
        List of regex expressions to ignore in the text.
    ignore_case : bool
        If True, turns everything to lowercase.
    ignore_punctuation : bool
        If True, removes punctuation.
    ignore_numbers : bool
        If True, removes all digits.

    Returns:
    -------
    str
        The preprocessed text.
    """
    if regexes_to_ignore:
        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)

    if ignore_case:
        text = text.lower()

    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))

    if ignore_numbers:
        text = re.sub(r"\d+", "", text)

    return text.strip()


def inclusive_exact_match_metric(
    predictions: List[str],
    references: List[List[str]],
    regexes_to_ignore: List[str] = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> List[bool]:
    """
    Inclusive exact match metric to check if predictions match the references.

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses.
    regexes_to_ignore : List[str]
        List of regex expressions to ignore in the text.
    ignore_case : bool
        If True, turns everything to lowercase.
    ignore_punctuation : bool
        If True, removes punctuation.
    ignore_numbers : bool
        If True, removes all digits.

    Returns:
    -------
    List[bool]
        A list of boolean values indicating if the prediction matches any of the references.
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."

    exact_matches = []

    for gen_resp, exp_resps in zip(predictions, references):
        gen_resp = preprocess_text(gen_resp, regexes_to_ignore, ignore_case, ignore_punctuation, ignore_numbers)
        match_found = False

        for exp_resp in exp_resps:
            exp_resp = preprocess_text(exp_resp, regexes_to_ignore, ignore_case, ignore_punctuation, ignore_numbers)

            if gen_resp == exp_resp or exp_resp in gen_resp:
                match_found = True
                break
        exact_matches.append(match_found)
    return exact_matches


def calculate_exact_match_score(exact_matches: List[bool]) -> float:
    """
    Calculates the exact match score.

    Parameters:
    ----------
    exact_matches : List[bool]
        The exact match results.

    Returns:
    -------
    float
        The exact match score.
    """
    total_responses = len(exact_matches)
    total_matches = sum(exact_matches)
    exact_match_score = total_matches / total_responses if total_responses > 0 else 0
    return exact_match_score


def token_f1(prediction: str, references: List[str]) -> float:
    """Token-level F1 between a prediction and its best-matching reference.

    This is the standard SQuAD / MuSiQue answer-F1: normalize both sides, split on
    whitespace, and compute F1 over the multiset of shared tokens, taking the max
    over all references (so answer aliases each get a fair shot). Normalization
    reuses ``preprocess_text`` (lowercase + strip punctuation) so scoring is
    consistent with ``inclusive_exact_match_metric``.

    Parameters:
    ----------
    prediction : str
        The generated response.
    references : List[str]
        The acceptable reference answers (e.g. gold answer + aliases).

    Returns:
    -------
    float
        The best token-F1 over ``references`` (0.0 if ``references`` is empty).
    """
    pred_tokens = preprocess_text(prediction, ignore_case=True, ignore_punctuation=True).split()
    best = 0.0
    for ref in references:
        ref_tokens = preprocess_text(ref, ignore_case=True, ignore_punctuation=True).split()
        # Edge case: if either side is empty, F1 is 1.0 only when both are empty
        # (SQuAD convention), otherwise 0.0.
        if not pred_tokens or not ref_tokens:
            best = max(best, 1.0 if pred_tokens == ref_tokens else 0.0)
            continue
        common = 0
        ref_counts = {}
        for tok in ref_tokens:
            ref_counts[tok] = ref_counts.get(tok, 0) + 1
        for tok in pred_tokens:
            if ref_counts.get(tok, 0) > 0:
                common += 1
                ref_counts[tok] -= 1
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def f1_metric(predictions: List[str], references: List[List[str]]) -> List[float]:
    """Per-example token-level answer F1 (MuSiQue's / SQuAD's official metric).

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses (one list of acceptable answers per prediction).

    Returns:
    -------
    List[float]
        Per-example best token-F1 against the references.
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."
    return [token_f1(pred, refs) for pred, refs in zip(predictions, references)]


def calculate_f1_score(f1_scores: List[float]) -> float:
    """Mean token-F1 across examples (0.0 if empty), mirroring calculate_exact_match_score."""
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    """All contiguous ``n``-grams of ``tokens`` as tuples (empty if fewer than n tokens)."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Longest-common-subsequence length of two token lists (standard DP).

    Used for ROUGE-L, whose overlap unit is the LCS (in-order, not necessarily
    contiguous) rather than fixed n-grams. Runs in O(len(a) * len(b)) with a
    rolling 1-D table to keep memory linear.
    """
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _overlap_f_measure(overlap: int, pred_units: int, ref_units: int) -> float:
    """F-measure from an overlap count and the per-side unit counts.

    ``precision = overlap / pred_units``, ``recall = overlap / ref_units``,
    ``F = 2PR/(P+R)``. Shared by ROUGE-N (units = n-grams) and ROUGE-L (units =
    tokens, overlap = LCS length). Returns 0.0 when either side has no units or
    there is no overlap (so the geometric mean stays defined).
    """
    if pred_units == 0 or ref_units == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_units
    recall = overlap / ref_units
    return 2 * precision * recall / (precision + recall)


def _rouge_n_f(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    """ROUGE-N F-measure: n-gram overlap between prediction and reference.

    Overlap is counted as a multiset intersection of n-grams (repetition is
    capped by the reference count, matching the standard ROUGE definition).
    """
    pred_ngrams = _ngrams(pred_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    ref_counts = {}
    for gram in ref_ngrams:
        ref_counts[gram] = ref_counts.get(gram, 0) + 1
    overlap = 0
    for gram in pred_ngrams:
        if ref_counts.get(gram, 0) > 0:
            overlap += 1
            ref_counts[gram] -= 1
    return _overlap_f_measure(overlap, len(pred_ngrams), len(ref_ngrams))


def _rouge_l_f(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    """ROUGE-L F-measure: longest-common-subsequence overlap, normalized by length."""
    lcs = _lcs_length(pred_tokens, ref_tokens)
    return _overlap_f_measure(lcs, len(pred_tokens), len(ref_tokens))


def rouge_scores(prediction: str, references: List[str]) -> dict:
    """Best-over-references ROUGE-1 / ROUGE-2 / ROUGE-L F-measure for one example.

    F-measure, **no stemmer**: tokenization reuses ``preprocess_text`` (lowercase +
    strip punctuation, whitespace split), identical to ``token_f1`` so ROUGE is
    consistent with the other answer metrics. Each ROUGE type is scored against
    every reference (gold answer + aliases) and the max is kept, mirroring
    ``token_f1``'s best-over-references rule.

    Edge cases follow the ``token_f1`` convention: if both prediction and a
    reference tokenize to empty, that reference scores 1.0; if exactly one side is
    empty, it scores 0.0.

    Parameters:
    ----------
    prediction : str
        The generated response.
    references : List[str]
        The acceptable reference answers (e.g. gold answer + aliases).

    Returns:
    -------
    dict
        ``{"rouge1": float, "rouge2": float, "rougeL": float}`` -- the best F-measure
        over ``references`` for each type (all 0.0 if ``references`` is empty).
    """
    pred_tokens = preprocess_text(prediction, ignore_case=True, ignore_punctuation=True).split()
    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref in references:
        ref_tokens = preprocess_text(ref, ignore_case=True, ignore_punctuation=True).split()
        # Both empty -> perfect match; exactly one empty -> no overlap (0.0).
        if not pred_tokens or not ref_tokens:
            score = 1.0 if pred_tokens == ref_tokens else 0.0
            best["rouge1"] = max(best["rouge1"], score)
            best["rouge2"] = max(best["rouge2"], score)
            best["rougeL"] = max(best["rougeL"], score)
            continue
        best["rouge1"] = max(best["rouge1"], _rouge_n_f(pred_tokens, ref_tokens, 1))
        best["rouge2"] = max(best["rouge2"], _rouge_n_f(pred_tokens, ref_tokens, 2))
        best["rougeL"] = max(best["rougeL"], _rouge_l_f(pred_tokens, ref_tokens))
    return best


def rouge_geometric_mean(predictions: List[str], references: List[List[str]], ndigits: int = 4) -> dict:
    """Corpus ROUGE geometric mean = ``(ROUGE-1 x ROUGE-2 x ROUGE-L)^(1/3)``.

    Aggregation is **mean-then-GM**: the per-example ROUGE-1/2/L F-measures are each
    averaged across the dataset first, and the geometric mean is taken over the three
    means. This is deliberate -- MuSiQue answers are often 1-2 tokens, so per-example
    ROUGE-2 is frequently 0; a per-example geometric mean would collapse those
    examples to 0 and dominate the corpus number, whereas mean-then-GM stays stable.

    F-measure, no stemmer (see ``rouge_scores``). Aliases are credited via the
    best-over-references rule inside ``rouge_scores``.

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses (one list of acceptable answers per prediction).
    ndigits : int
        Rounding for the returned scores (default 4).

    Returns:
    -------
    dict
        ``{"rouge1", "rouge2", "rougeL", "rouge_gm"}`` -- the three corpus-mean
        F-measures and their geometric mean (all 0.0 for empty input).
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rouge_gm": 0.0}

    n = len(predictions)
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, refs in zip(predictions, references):
        scores = rouge_scores(pred, refs)
        for key in sums:
            sums[key] += scores[key]
    means = {key: sums[key] / n for key in sums}
    gm = (means["rouge1"] * means["rouge2"] * means["rougeL"]) ** (1.0 / 3.0)
    return {
        "rouge1": round(means["rouge1"], ndigits),
        "rouge2": round(means["rouge2"], ndigits),
        "rougeL": round(means["rougeL"], ndigits),
        "rouge_gm": round(gm, ndigits),
    }


def qa_metric_score(
    predictions: List[str],
    references: List[List[str]],
    queries: List[str],
    qa_metric: Union[PEDANT, TransformerMatcher],
) -> float:
    """
    Computes the QA metric score for the predictions.

    Parameters:
    ----------
    predictions : List[str]
        The generated responses.
    references : List[List[str]]
        The expected responses.
    queries: List[str]
        The original queries for each response.
    qa_metric : Union[PEDANT, TransformerMatcher]
        The QA metric instance to use for evaluation.

    Returns:
    -------
    float
        The average QA metric score.
    """
    assert len(predictions) == len(
        references
    ), "The length of generated responses and expected responses must be the same."

    scores = []
    for gen_resp, exp_resps, query in zip(predictions, references, queries):
        _, highest_score = qa_metric.get_highest_score(gen_resp, exp_resps, query)
        scores.append(highest_score)

    return np.mean(scores)


def save_responses_to_csv(
    generated_responses: List[str], expected_responses: List[List[str]], queries: List[str], output_csv: str
):
    """
    Saves the evaluation predictions to a CSV file.

    One row per evaluated example, capturing the query, the model's generated
    response, and the expected (reference) responses so results are inspectable.

    Parameters:
    ----------
    generated_responses : List[str]
        The generated responses.
    expected_responses : List[List[str]]
        The expected responses.
    queries : List[str]
        The original queries for each response.
    output_csv : str
        The path to the output CSV file.
    """
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Query", "Generated Response", "Expected Responses"])

        for query, gen_resp, exp_resps in zip(queries, generated_responses, expected_responses):
            writer.writerow([query, gen_resp, "; ".join(exp_resps)])
