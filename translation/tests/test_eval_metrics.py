from versta.eval.metrics import score_translations

import pytest

from conftest import EVAL_PAIRS

REFERENCES = [pair[1] for pair in EVAL_PAIRS]


def test_identity_scores_perfect() -> None:
    scores = score_translations(REFERENCES, REFERENCES)
    assert scores.bleu == 100.0
    assert scores.chrf == 100.0


def test_signatures_and_version_recorded() -> None:
    scores = score_translations(REFERENCES, REFERENCES)
    assert "13a" in scores.bleu_signature
    # chrF2 defaults: character order 6, no word n-grams.
    assert "nc:6" in scores.chrf_signature
    assert "nw:0" in scores.chrf_signature
    assert scores.sacrebleu_version.count(".") >= 1


def test_scoring_is_deterministic() -> None:
    hypotheses = [f"hola {index}" for index in range(len(REFERENCES))]
    first = score_translations(hypotheses, REFERENCES)
    second = score_translations(hypotheses, REFERENCES)
    assert first == second


def test_garbage_scores_low() -> None:
    hypotheses = ["aaa bbb ccc ddd eee"] * len(REFERENCES)
    scores = score_translations(hypotheses, REFERENCES)
    assert scores.bleu == 0.0
    # chrF shares stray characters with any same-script text, so garbage
    # lands a few points — still an order of magnitude below a real model.
    assert scores.chrf < 10.0


def test_misaligned_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="line-aligned"):
        score_translations(["one"], REFERENCES)
