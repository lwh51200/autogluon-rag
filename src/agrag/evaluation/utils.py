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
