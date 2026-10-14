"""
Generated release eval summaries (NEU-52).

Renders the gate report as the markdown posted with every model release —
scores delta'd against the predecessor, corpus pin, engine provenance,
artifact size notes, and the gate decision including any override reason.
This replaces handwritten summaries: the document is a pure function of the
ledger JSON, so what is posted always matches what is recorded.
"""

from typing import Optional

from .check import GateReport


def _fmt_score(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _fmt_delta(delta: Optional[float]) -> str:
    return "—" if delta is None else f"{delta:+.4f}"


def _fmt_bytes(size: Optional[int]) -> str:
    if size is None:
        return "unknown"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.2f} MiB"
    if size >= 1 << 10:
        return f"{size / (1 << 10):.1f} KiB"
    return f"{size} B"


def _fmt_size_delta(size: Optional[int], previous: Optional[int]) -> str:
    if size is None or previous is None:
        return ""
    return f" ({size - previous:+,} B)"


def render_summary(report: GateReport) -> str:
    """
    Renders the release eval-summary markdown for one gate report.

    Args:
        report (GateReport): The gate evaluation to render.

    Returns:
        str: Markdown document: decision banner, metric table (predecessor
            vs. candidate with deltas), corpus/engine provenance, artifact
            checksum and size notes, and the recorded reason for override or
            rebaseline decisions.
    """
    result = report.result
    model = result["model"]
    scores = result["scores"]
    corpus = result["corpus"]
    engine = result["engine"]

    predecessor = report.predecessor
    previous = predecessor["result"] if predecessor is not None else None

    title = f"Eval summary — {report.direction} {report.version}"
    decision = report.decision.upper()
    lines = [
        f"# {title}",
        "",
        f"**Gate: {decision}** (chrF regression threshold {report.threshold_chrf})",
    ]
    if report.failure_why is not None:
        lines.append(f"**Blocked:** {report.failure_why}")
    if report.reason is not None:
        label = "Rebaseline" if report.decision == "rebaseline" else "Override"
        lines.append(f"**{label} reason:** {report.reason}")

    previous_heading = previous["model"]["version"] if previous else "—"
    lines += [
        "",
        f"| Metric | {previous_heading} (previous) | {report.version} (candidate) | Δ |",
        "|---|---|---|---|",
        _row(
            "sacreBLEU",
            previous["scores"]["bleu"] if previous else None,
            scores["bleu"],
            report.delta_bleu,
        ),
        _row(
            "chrF",
            previous["scores"]["chrf"] if previous else None,
            scores["chrf"],
            report.delta_chrf,
        ),
        _row(
            "COMET-22",
            previous["scores"]["comet22"] if previous else None,
            scores["comet22"],
            report.delta_comet22,
        ),
    ]

    corpus_line = (
        f"- Corpus: {corpus['corpus']} {corpus['split']} @ `{corpus['revision']}` "
        f"({corpus['sentences']} sentences)"
    )
    if previous is not None and previous["corpus"]["revision"] != corpus["revision"]:
        corpus_line += (
            f" — predecessor was @ `{previous['corpus']['revision']}`; scores "
            f"across corpus pins are not comparable"
        )
    engine_line = (
        f"- Engine: {engine['name']} ({engine['variant']}) {engine['version']}"
    )
    if previous is not None and (
        previous["engine"]["name"] != engine["name"]
        or previous["engine"]["version"] != engine["version"]
    ):
        engine_line += (
            f" — previous: {previous['engine']['name']} "
            f"({previous['engine']['variant']}) {previous['engine']['version']}"
        )

    artifact = model.get("bundle_file") or model.get("base_model") or "unknown"
    size = model.get("size_bytes")
    previous_size = previous["model"].get("size_bytes") if previous else None
    checksum = model.get("bundle_sha256") or model.get("model_sha256") or "unknown"
    artifact_line = (
        f"- Artifact: `{artifact}`, sha256 `{checksum[:16]}…`, "
        f"size {_fmt_bytes(size)}{_fmt_size_delta(size, previous_size)}"
    )

    run = result["run"]
    lines += [
        "",
        corpus_line,
        engine_line,
        artifact_line,
        f"- Evaluated: {run['started_utc']} ({run['duration_seconds']}s); "
        f"recorded {report.decided_utc}",
        "",
    ]
    return "\n".join(lines)


def _row(
    metric: str,
    previous: Optional[float],
    candidate: Optional[float],
    delta: Optional[float],
) -> str:
    return (
        f"| {metric} | {_fmt_score(previous)} | {_fmt_score(candidate)} "
        f"| {_fmt_delta(delta)} |"
    )
