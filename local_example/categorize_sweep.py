"""Categorize a MuSiQue backend-sweep prediction file into diagnostic buckets.

For each (backend, hop-type) it cross-tabs the answer outcome with whether the
gold answer was actually retrievable, so we can tell *why* a question failed:

  outcome:
    em_hit         -- inclusive exact match against any reference
    answered_no_em -- the agent answered but it did not exact-match
    abstain        -- the agent emitted the canned abstention string

  answer_in_evidence:
    True/False -- any gold reference (normalized) appears as a substring of the
                  concatenated retrieved evidence texts. This is the *retrieval*
                  success signal, independent of what the agent chose to do.

The four cells that matter:
  retrievable + abstain        -> policy over-refused (answer was there)
  retrievable + answered_no_em -> EM-strictness or synthesis issue
  not-retrievable + anything   -> genuine retrieval gap
  retrievable + em_hit         -> win

Usage:
    python local_example/categorize_sweep.py <predictions.jsonl> [--label C0]
    python local_example/categorize_sweep.py a.jsonl b.jsonl   # compare two runs
"""
import argparse
import json
import re
import string
from collections import Counter, defaultdict

# Must match agentic_module.DEFAULT_ABSTENTION exactly.
ABSTENTION_MARKER = "enough supporting evidence"


def _normalize(text):
    """Lowercase, strip punctuation, collapse whitespace (matches EM preprocessing)."""
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _em_hit(prediction, references):
    """Inclusive exact match: any normalized reference is a substring of the pred."""
    pred = _normalize(prediction)
    for ref in references:
        r = _normalize(ref)
        if r and (pred == r or r in pred):
            return True
    return False


def _answer_in_evidence(references, evidence_texts):
    """Whether any normalized reference appears in the concatenated evidence."""
    blob = _normalize(" ".join(evidence_texts or []))
    return any(_normalize(ref) and _normalize(ref) in blob for ref in references)


def _outcome(prediction, references):
    if ABSTENTION_MARKER in (prediction or "").lower():
        return "abstain"
    return "em_hit" if _em_hit(prediction, references) else "answered_no_em"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze(rows):
    """Return per-backend summary dicts keyed by backend name."""
    by_backend = defaultdict(list)
    for r in rows:
        by_backend[r["backend"]].append(r)

    summary = {}
    for backend, brows in by_backend.items():
        outcomes = Counter()
        crosstab = Counter()  # (answer_in_evidence, outcome)
        by_hop = defaultdict(Counter)
        retrievable = 0
        f1_sum = 0.0
        f1_n = 0
        supp_recall_sum = 0.0
        supp_n = 0
        for r in brows:
            refs = r.get("references", [])
            pred = str(r.get("prediction", ""))
            ev = r.get("evidence_texts", [])
            out = _outcome(pred, refs)
            in_ev = _answer_in_evidence(refs, ev)
            outcomes[out] += 1
            crosstab[(in_ev, out)] += 1
            by_hop[r.get("question_type", "?")][out] += 1
            if in_ev:
                retrievable += 1
                by_hop[r.get("question_type", "?")]["_retrievable"] += 1
            rm = r.get("retrieval_metrics") or {}
            # token f1 isn't in the jsonl per-row; approximate answer quality via EM
            # bucket only. support recall is available per-row.
            sm = r.get("support_metrics") or {}
            if sm and sm.get("support_recall") is not None:
                supp_recall_sum += sm["support_recall"]
                supp_n += 1
        n = len(brows)
        summary[backend] = {
            "n": n,
            "outcomes": dict(outcomes),
            "em": round(outcomes["em_hit"] / n, 4) if n else 0.0,
            "answer_retrievable": f"{retrievable}/{n}",
            "crosstab": {f"{'RET' if k[0] else 'MISS'}+{k[1]}": v for k, v in sorted(crosstab.items())},
            "mean_support_recall": round(supp_recall_sum / supp_n, 4) if supp_n else None,
            "by_hop": {h: dict(c) for h, c in sorted(by_hop.items())},
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="One or more predictions.jsonl files to analyze.")
    ap.add_argument("--label", action="append", default=None, help="Label(s) for the file(s).")
    args = ap.parse_args()

    for i, path in enumerate(args.paths):
        label = (args.label[i] if args.label and i < len(args.label) else path)
        print("=" * 72)
        print(f"{label}")
        print("=" * 72)
        summary = analyze(load(path))
        print(json.dumps(summary, indent=2))
        print()


if __name__ == "__main__":
    main()
