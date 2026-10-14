"""
Pack acceptance harness tests (NEU-60). Everything here is hermetic: bundles
are built through the production versta.bundle path from fake direction
fixtures, and scorecard assembly runs on canned leanmt-probe JSON — the real
engine path is exercised by the pack-acceptance CI workflow.
"""

import json
import tarfile

from pathlib import Path

import pytest

from versta.bundle.__main__ import main as bundle_main
from versta.probe.corpus import CorpusError, load_corpus, validate_corpus
from versta.probe.install import install_bundle, sha256_of
from versta.probe.metrics import corpus_chrf, digit_preservation, sentence_chrf
from versta.probe.scorecard import (
    EXPECT_DIRECTION_FAIL,
    EXPECT_PASS,
    acceptance_of,
    build_scorecard,
    score_direction,
)

from conftest import write_direction_dir


def _bundle(tmp_path: Path, directions: list, bidirectional: bool = True) -> Path:
    input_dirs = [
        write_direction_dir(tmp_path / "downloads", direction)
        for direction in directions
    ]
    result = bundle_main(
        input_dirs=input_dirs,
        output_dir=tmp_path / "bundles",
        bidirectional=bidirectional,
        keep_intermediates=False,
        keep_input=True,
    )
    return result["bundle"]


# ---------------------------------------------------------------------------
# chrF parity with sacreBLEU (values derived from sacrebleu 2.6.0 defaults:
# char_order=6, word_order=0, beta=2, no lowercase, whitespace stripped,
# effective-order smoothing. Regenerate deliberately if the formula changes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hypothesis,reference,expected",
    [
        (
            "Das Museumsticket kostet 12 Euro.",
            "Das Museumsticket kostet 12 Euro.",
            100.0,
        ),
        ("hello world", "completely different text", 5.761305),
        ("", "", 0.0),
        ("", "Something here.", 0.0),
        (
            "私の電車は18時45分に7番線から出ます。",
            "私の電車は18時45分に7番線から出ます。",
            100.0,
        ),
        (
            "私の電車は19時15分に出ます。",
            "私の電車は18時45分に7番線から出ます。",
            34.03925,
        ),
    ],
)
def test_sentence_chrf_matches_sacrebleu(hypothesis, reference, expected):
    assert sentence_chrf(hypothesis, reference) == pytest.approx(expected, abs=1e-4)


def test_corpus_chrf_matches_sacrebleu_micro_average():
    hypotheses = [
        "Das Museumsticket kostet 12 Euro.",
        "My train leaves at 18:45 from platform 7.",
        "hello world",
        "",
    ]
    references = [
        "Das Museumsticket kostet 12 Euro.",
        "My train leaves at 18:45 from platform 7.",
        "completely different text",
        "",
    ]
    # sacrebleu 2.6.0 corpus_score for this set.
    assert corpus_chrf(hypotheses, references) == pytest.approx(77.869388, abs=1e-4)


def test_corpus_chrf_rejects_misaligned_input():
    with pytest.raises(ValueError):
        corpus_chrf(["a"], [])


# ---------------------------------------------------------------------------
# Digit preservation (value-level; NEU-30 harm class)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,output,expected",
    [
        # verbatim digit copy
        (
            "The museum ticket costs 12 euros.",
            "Das Museumsticket kostet 12 Euro.",
            True,
        ),
        ("Room 214 is on the second floor.", "214号室は2階です。", True),
        # locale clock rendering
        (
            "My train leaves at 18:45 from platform 7.",
            "私の電車は18時45分に7番線から出ます。",
            True,
        ),
        (
            "Breakfast is served from 6:30 to 10:00.",
            "朝食は6時30分から10時まで提供されます。",
            True,
        ),
        (
            "私の電車は18時45分に7番線から出発します。",
            "My train leaves at 6:45 pm from platform 7.",
            True,
        ),
        # number words / ordinals / months in English output
        (
            "子供2人を含めて、全部で5人です。",
            "There are five of us, including two children.",
            True,
        ),
        (
            "私の部屋は2階の214号室です。",
            "My room is the 214th room on the second floor.",
            True,
        ),
        (
            "10月3日に着いて、10月9日に出ます。",
            "I arrive on October 3 and leave on October 9.",
            True,
        ),
        ("博物館から徒歩15分です。", "The museum is a fifteen-minute walk away.", True),
        # genuine corruption cases stay failing
        ("朝食は6時30分から10時までですか。", "Is breakfast from 6pm to 10pm?", False),
        (
            "The flight takes 9 hours and 25 minutes.",
            "El vuelo dura 9 horas y 5 minutos.",
            False,
        ),
        ("料金は4500円です。", "The fare is 5400 yen.", False),
        (
            "Someone stole my bag with 200 euros inside.",
            "Me robaron la bolsa con 20 euros dentro.",
            False,
        ),
        ("I need 3 tickets.", "Necesito 33 billetes.", False),
        ("Room 214.", "Habitación 1214.", False),
        ("Turn left and walk 200 meters.", "Gire a la izquierda.", False),
    ],
)
def test_digit_preservation(source, output, expected):
    assert digit_preservation(source, output) is expected


# ---------------------------------------------------------------------------
# Corpus loading / validation
# ---------------------------------------------------------------------------


def test_packaged_corpora_are_valid():
    corpus_dir = Path(__file__).parent.parent / "versta" / "probe" / "corpora"
    for path in sorted(corpus_dir.glob("*.json")):
        corpus = load_corpus(path)
        assert corpus["name"] == "traveler-v1"
        assert len(corpus["entries"]) == 40
        # filename and declared direction agree — a swapped corpus file is the
        # adversarial class in miniature.
        assert path.stem == f"{corpus['source_language']}-{corpus['target_language']}"


def test_corpus_rejects_missing_entries():
    bad = {
        "name": "traveler-v1",
        "version": 1,
        "source_language": "en",
        "target_language": "es",
        "chrf_floor": 20.0,
        "entries": [],
    }
    with pytest.raises(CorpusError):
        validate_corpus(bad)


def test_corpus_requires_digit_coverage():
    plain = {
        "id": "x",
        "tags": ["numbers"],
        "source": "plain words",
        "reference": "ref",
    }
    bad = {
        "name": "traveler-v1",
        "version": 1,
        "source_language": "en",
        "target_language": "es",
        "chrf_floor": 20.0,
        "entries": [plain, dict(plain, id="y")],
    }
    with pytest.raises(CorpusError, match="carry digits"):
        validate_corpus(bad)


# ---------------------------------------------------------------------------
# Install path (bundle -> versta.schema gate -> extraction)
# ---------------------------------------------------------------------------


def test_install_bundle_runs_the_archive_gate(tmp_path):
    bundle = _bundle(tmp_path, ["en-es", "es-en"])
    installed = install_bundle(bundle, tmp_path / "installed")
    assert installed.slug == "en-es"
    assert installed.sha256 == sha256_of(bundle)
    assert len(installed.directions) == 2
    forward = next(d for d in installed.directions if d.source_language == "en")
    assert forward.model.is_file()
    # Fixture models are two-vocabulary by default (mirroring en-ja).
    assert forward.target_vocabulary.name == "vocab.trg.spm"
    assert forward.encoder_layers == 6


def test_install_bundle_rejects_corrupt_archive(tmp_path):
    bundle = _bundle(tmp_path, ["en-es", "es-en"])
    blob = bytearray(bundle.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    corrupted = tmp_path / "corrupted.tar.gz"
    corrupted.write_bytes(bytes(blob))
    with pytest.raises(tarfile.ReadError):
        install_bundle(corrupted, tmp_path / "installed2")


def test_install_bundle_verifies_checksum_sidecar(tmp_path):
    bundle = _bundle(tmp_path, ["en-es", "es-en"])
    sidecar = bundle.parent / (bundle.name[: -len(".gz")] + ".sha256")
    assert sidecar.is_file()  # versta.bundle writes .tar.sha256 alongside

    # Tampered release artifact with an intact (now stale) sidecar must be
    # rejected before any schema or probing work runs.
    tampered = bundle.parent / "tampered-bundle.tar.gz"
    tampered.write_bytes(bundle.read_bytes() + b"\x00")
    tampered_sidecar = tampered.parent / (tampered.name[: -len(".gz")] + ".sha256")
    tampered_sidecar.write_text(sidecar.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(Exception, match="sha256 mismatch"):
        install_bundle(tampered, tmp_path / "installed3")


# ---------------------------------------------------------------------------
# Adversarial pack builder (the NEU-19 negative control)
# ---------------------------------------------------------------------------


def test_adversarial_pack_claims_swapped_direction(tmp_path):
    from versta.probe.adversarial import build_mislabeled_bundle

    real = write_direction_dir(tmp_path / "downloads", "en-es")
    bundle = build_mislabeled_bundle(
        model_dir=real,
        output_dir=tmp_path / "adv",
        claimed_source="es",
        claimed_target="en",
        work_dir=tmp_path / "adv-work",
    )
    assert bundle.name == "es-en-mislabeled-bundle.tar.gz"

    # The artifact passes every shape/content gate the pipeline has — the
    # defect is purely directional, visible only to the probe's chrF check.
    installed = install_bundle(bundle, tmp_path / "adv-installed")
    assert installed.slug == "en-es"
    assert [
        f"{d.source_language}-{d.target_language}" for d in installed.directions
    ] == ["es-en"]
    # Payload is untouched: the true en-es model bytes are still inside.
    source_dir = write_direction_dir(tmp_path / "again", "en-es")
    assert sha256_of(installed.directions[0].model) == sha256_of(
        source_dir / "model.bin"
    )


def test_adversarial_builder_refuses_truthful_labels(tmp_path):
    from versta.probe.adversarial import build_mislabeled_bundle

    real = write_direction_dir(tmp_path / "downloads", "en-es")
    with pytest.raises(ValueError, match="true direction"):
        build_mislabeled_bundle(
            model_dir=real,
            output_dir=tmp_path / "adv",
            claimed_source="en",
            claimed_target="es",
            work_dir=tmp_path / "adv-work",
        )


# ---------------------------------------------------------------------------
# Scorecard assembly + acceptance rules (canned probe output, no engine)
# ---------------------------------------------------------------------------


def _probe_output(arch: str = "aarch64", threads: int = 2) -> dict:
    sentences = [
        {
            "source": f"Source sentence {i} 12.",
            "target": f"Target sentence {i} 12.",
            "ms": 10.0 + i,
            "ok": True,
        }
        for i in range(40)
    ]
    return {
        "tool": "leanmt-probe",
        "arch": arch,
        "threads": threads,
        "decode": {"beam": 4, "max_seq": 256},
        "load_ok": True,
        "model_load_ms": 68.0,
        "rss_after_load_kib": 90_000,
        "sentences": sentences,
        "latency": {
            "n": 40,
            "min_ms": 10.0,
            "p50_ms": 29.0,
            "p95_ms": 48.0,
            "max_ms": 49.0,
        },
        "rss_peak_kib": 98_000,
    }


def _corpus(tmp_path: Path) -> Path:
    entries = [
        {
            "id": f"t-{i}",
            "tags": ["numbers"],
            "source": f"Source sentence {i} 12.",
            "reference": f"Target sentence {i} 12.",
        }
        for i in range(40)
    ]
    path = tmp_path / "en-es.json"
    path.write_text(
        json.dumps(
            {
                "name": "traveler-v1",
                "source_language": "en",
                "target_language": "es",
                "version": 1,
                "chrf_floor": 50.0,
                "chrf_floor_note": "test fixture",
                "entries": entries,
            }
        )
    )
    return path


def test_score_direction_clean_output(tmp_path):
    corpus = load_corpus(_corpus(tmp_path))
    score = score_direction("en-es", corpus, _probe_output())
    assert score["load_ok"] is True
    assert score["chrf"] == pytest.approx(100.0)
    assert score["direction_check"] == "pass"
    assert score["digit_preservation"]["passed"] == 40
    assert acceptance_of([score]) == "pass"


def test_direction_floor_trips_on_garbage(tmp_path):
    corpus = load_corpus(_corpus(tmp_path))
    output = _probe_output()
    for sentence in output["sentences"]:
        sentence["target"] = "garbled nonsense 999"
    score = score_direction("en-es", corpus, output)
    assert score["chrf"] < 50.0
    assert score["direction_check"] == "fail"
    assert acceptance_of([score]) == "fail"


def _scorecard_bundle_and_environment():
    from versta.probe.install import InstalledBundle

    bundle = InstalledBundle(
        archive=Path("es-en-bundle.tar.gz"),
        sha256="0" * 64,
        slug="es-en",
        bundle_version="v2.1.0",
        leanmt_binary_version=1,
        bidirectional=False,
        directions=[],
    )
    environment = {
        "host_arch": "aarch64",
        "probe_arch": "aarch64",
        "threads": 2,
        "runner": "test",
        "leanmt_commit": "abc123",
        "corpus": "traveler-v1",
        "corpus_entries": 40,
    }
    return bundle, environment


def test_scorecard_expectation_rules(tmp_path):
    corpus = load_corpus(_corpus(tmp_path))
    bundle, environment = _scorecard_bundle_and_environment()

    clean = score_direction("es-en", corpus, _probe_output())
    broken_output = _probe_output()
    for sentence in broken_output["sentences"]:
        sentence["target"] = "garbled nonsense 999"
    broken = score_direction("es-en", corpus, broken_output)
    load_failed = score_direction(
        "es-en", corpus, {"load_ok": False, "load_error": "boom"}
    )

    passthrough = build_scorecard(bundle, [clean], environment, EXPECT_PASS)
    assert passthrough["acceptance"] == "pass"
    assert passthrough["expectation_met"] is True

    negative = build_scorecard(bundle, [broken], environment, EXPECT_DIRECTION_FAIL)
    assert negative["acceptance"] == "fail"
    assert negative["directions"][0]["direction_check"] == "fail"
    assert negative["expectation_met"] is True

    # A clean pack under adversarial expectation: the control no longer
    # proves the gate works, so the run must not pass.
    bad_control = build_scorecard(bundle, [clean], environment, EXPECT_DIRECTION_FAIL)
    assert bad_control["expectation_met"] is False

    # A load failure fails acceptance but does not satisfy the adversarial
    # expectation — the control exists to prove the chrF direction check.
    load_failing = build_scorecard(
        bundle, [load_failed], environment, EXPECT_DIRECTION_FAIL
    )
    assert load_failing["acceptance"] == "fail"
    assert load_failing["expectation_met"] is False
