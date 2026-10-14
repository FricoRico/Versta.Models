"""
Probe quality metrics: chrF and digit preservation.

chrF is re-implemented in stdlib on purpose: it pins the pack-acceptance
score to a fully specified, dependency-free formula. The implementation is
behavior-identical to sacreBLEU's default CHRF (char_order=6, word_order=0,
beta=2, no lowercasing, whitespace stripped before n-gram extraction,
effective-order smoothing), regression-tested against sacreBLEU reference
values in tests/test_probe_metrics.py — do not change the formula without
re-deriving those vectors.

Corpus chrF micro-averages: per-order match statistics sum over the corpus
and the f-score is computed once, matching sacreBLEU's corpus_chrf.
"""

import re

from collections import Counter
from typing import Dict, Iterable, List, Tuple

_CHAR_ORDER = 6
_BETA = 2

# Maximal ASCII digit runs, e.g. "18:45" -> ["18", "45"], "+44 7788-123456"
# -> ["44", "7788", "123456"].
_DIGIT_RUN = re.compile(r"\d+")

# --- Number-surface normalization --------------------------------------------
# Digit preservation in the NEU-30 sense is about *values*, not orthography:
# models legitimately localize "9" as "nine" (ja->en) or "6:30" as "6時30分"
# (en->ja). The check therefore compares values after normalizing the output
# surface back to ASCII digit runs. V1 covers what the first sweep observed:
# English cardinal/ordinal number words, English month names, and Japanese
# clock forms. Extend the maps when a probed pair is seen to spell numbers in
# another language; do not accept lost/corrupted values by weakening the rule.

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

_ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "twentieth": 20,
    "thirtieth": 30,
}

_MONTH_WORDS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

_WORD_TO_NUMBER: Dict[str, int] = {
    **_NUMBER_WORDS,
    **_ORDINAL_WORDS,
    **_MONTH_WORDS,
}

# Clock-like value carriers, compared as (hour % 12, minute) values rather
# than digit runs. Covers "18:45" / "6:45 pm", Japanese "18時45分" / "10時"
# (minutes omitted = :00; the negative lookahead keeps duration "9時間" out),
# and bare meridiem forms "8pm". The 12-hour normalization deliberately treats
# meridiem as locale style ("las 8 de la tarde" ≡ "8 pm" ≡ "20時"): v1 guards
# value corruption (dropped/doubled/changed numbers), not am/pm choice.
_CLOCK_COLON = re.compile(r"(\d{1,2})\s*[:：]\s*(\d{2})\s*(?:am|pm|a\.m\.|p\.m\.)?\b")
_CLOCK_JA = re.compile(r"(\d{1,2})\s*時(?!\s*間)(?:\s*(\d{1,2})\s*分)?")
_CLOCK_MERIDIEM = re.compile(r"(?<![:：\d])(\d{1,2})\s*(?:am|pm|a\.m\.|p\.m\.)\b")


def _normalize_number_surface(text: str) -> str:
    """Replaces known number words (token-bounded) by their digit values."""
    # Hyphenated compounds split first: "fifteen-minute" carries 15.
    tokens = text.replace("-", " ").split(" ")
    normalized = [
        str(_WORD_TO_NUMBER[token.lower().strip(",.;:!?")])
        if token.lower().strip(",.;:!?") in _WORD_TO_NUMBER
        else token
        for token in tokens
    ]
    return " ".join(normalized)


def _clock_values(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Extracts clock values as 12h-normalized (hour, minute) tuples plus their
    matched spans (for removal before plain digit-run comparison).
    """
    values: List[Tuple[str, Tuple[int, int]]] = []
    for match in _CLOCK_COLON.finditer(text):
        values.append((match.group(0), (int(match.group(1)) % 12, int(match.group(2)))))
    for match in _CLOCK_JA.finditer(text):
        values.append(
            (match.group(0), (int(match.group(1)) % 12, int(match.group(2) or 0)))
        )
    for match in _CLOCK_MERIDIEM.finditer(text):
        values.append((match.group(0), (int(match.group(1)) % 12, 0)))
    return values


def _without_spans(text: str, spans: List[str]) -> str:
    for span in spans:
        text = text.replace(span, " ", 1)
    return text


def digit_runs(text: str) -> List[str]:
    """
    Extracts maximal ASCII digit runs from text in reading order.

    Args:
        text (str): arbitrary input or output text.

    Returns:
        List[str]: every maximal run of digits, e.g. "18:45" -> ["18", "45"].
    """
    return _DIGIT_RUN.findall(text)


def digit_preservation(source: str, output: str) -> bool:
    """
    Value-level digit preservation (the NEU-30 harm class): every number the
    source carries must survive the translation with its value intact. Clock
    values compare as 12h-normalized H:MM pairs ("10:00" -> "10時" passes,
    "6:30" -> "6 pm" fails), and all other digit runs compare as multisets
    after number-word normalization ("3枚" -> "three" passes, "3" -> "4"
    fails).

    Args:
        source (str): probe source sentence.
        output (str): engine output for that sentence.

    Returns:
        bool: True when every source number value appears in the output.
    """
    normalized_output = _normalize_number_surface(output)
    source_clocks = _clock_values(source)
    output_clocks = _clock_values(normalized_output)

    # Every source clock must find an output clock of the same value; its span
    # is then excluded from the plain-run comparison on both sides. Output
    # clocks that match no source clock keep their digits in the run
    # comparison (the source may express the same value as plain text).
    unmatched_source = Counter(value for _, value in source_clocks)
    consumed_output_spans: List[str] = []
    for span, value in output_clocks:
        if unmatched_source.get(value, 0) > 0:
            unmatched_source[value] -= 1
            consumed_output_spans.append(span)
    if any(count > 0 for count in unmatched_source.values()):
        return False

    source_rest = _without_spans(source, [span for span, _ in source_clocks])
    output_rest = _without_spans(normalized_output, consumed_output_spans)

    output_runs = Counter(digit_runs(output_rest))
    for run in digit_runs(source_rest):
        if output_runs.get(run, 0) <= 0:
            return False
        output_runs[run] -= 1
    return True


def _char_ngram_stats(hypothesis: str, reference: str) -> List[Tuple[int, int, int]]:
    """
    Builds per-order character n-gram match statistics for a sentence pair,
    with whitespace stripped (sacreBLEU CHRF default). Orders where the
    reference has no n-gram are recorded as (0, 0, 0) so the effective-order
    smoothing below drops them, exactly like sacreBLEU.
    """
    hyp = "".join(hypothesis.split())
    ref = "".join(reference.split())

    stats: List[Tuple[int, int, int]] = []
    for order in range(1, _CHAR_ORDER + 1):
        hyp_ngrams: Counter[str] = Counter(
            hyp[i : i + order] for i in range(len(hyp) - order + 1)
        )
        ref_ngrams: Counter[str] = Counter(
            ref[i : i + order] for i in range(len(ref) - order + 1)
        )
        if not ref_ngrams:
            stats.append((0, 0, 0))
            continue
        match = 0
        for ngram, count in hyp_ngrams.items():
            if ngram in ref_ngrams:
                match += min(count, ref_ngrams[ngram])
        stats.append((sum(hyp_ngrams.values()), sum(ref_ngrams.values()), match))
    return stats


def _f_score(stats: Iterable[Tuple[int, int, int]]) -> float:
    """
    Computes the chrF f-score (beta=2, effective-order smoothing) from
    per-order (hypothesis, reference, match) counts; identical to
    sacreBLEU's CHRF._compute_f_score with eps_smoothing=False.
    """
    factor = _BETA**2
    avg_prec = 0.0
    avg_rec = 0.0
    effective_order = 0
    for n_hyp, n_ref, n_match in stats:
        if n_hyp == 0 or n_ref == 0:
            continue
        prec = n_match / n_hyp
        rec = n_match / n_ref
        avg_prec += prec
        avg_rec += rec
        effective_order += 1

    if effective_order == 0:
        return 0.0

    avg_prec /= effective_order
    avg_rec /= effective_order
    if avg_prec + avg_rec == 0.0:
        return 0.0
    score = (1 + factor) * avg_prec * avg_rec
    score /= (factor * avg_prec) + avg_rec
    return 100.0 * score


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """
    Computes sentence-level chrF (0–100), sacreBLEU-default configuration.

    Args:
        hypothesis (str): the engine's output.
        reference (str): the human reference translation.

    Returns:
        float: chrF score in [0, 100].
    """
    return _f_score(_char_ngram_stats(hypothesis, reference))


def corpus_chrf(hypotheses: List[str], references: List[str]) -> float:
    """
    Computes corpus-level chrF: per-order statistics summed across the
    corpus, one f-score at the end (sacreBLEU corpus_chrf aggregation).

    Args:
        hypotheses (List[str]): engine outputs, aligned with references.
        references (List[str]): human references.

    Returns:
        float: chrF score in [0, 100].

    Raises:
        ValueError: on length mismatch or an empty corpus.
    """
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be aligned one-to-one")
    if not hypotheses:
        raise ValueError("cannot score an empty corpus")

    totals: List[List[int]] = [[0, 0, 0] for _ in range(_CHAR_ORDER)]
    for hypothesis, reference in zip(hypotheses, references):
        for order, (n_hyp, n_ref, n_match) in enumerate(
            _char_ngram_stats(hypothesis, reference)
        ):
            totals[order][0] += n_hyp
            totals[order][1] += n_ref
            totals[order][2] += n_match

    return _f_score((n_hyp, n_ref, n_match) for n_hyp, n_ref, n_match in totals)
