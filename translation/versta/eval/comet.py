"""
Optional COMET-22 scoring (release tier only).

COMET-22 needs the ``unbabel-comet`` package and a ~2 GB scoring model,
neither of which belongs on the PR path or in the module's default
dependency set. This module is a lazy, fail-loud hook: it is only reached
when ``--comet`` is passed (a release-candidate job decision), imports the
package on demand and tells the operator exactly how to enable it
otherwise. Weights resolve through the standard COMET/Hugging Face cache,
shared between runs.

The harness records the model identifier in every result that uses it, so
scores stay traceable when the release tier bumps scoring models.
"""

from typing import List

from .engines import EngineError

# The COMET-22 model Mozilla reports for these pairs on flores200-plus.
COMET22_MODEL = "Unbabel/wmt22-comet-da"

_INSTALL_HINT = (
    "COMET scoring requires the optional dependency: "
    "`uv pip install unbabel-comet` (pulls torch and the scoring model on "
    "first use; release-tier jobs only, never the PR path)"
)


def comet22_available() -> bool:
    """Whether the optional unbabel-comet dependency is importable."""
    try:
        import comet  # noqa: F401
    except ImportError:
        return False
    return True


def score_comet22(
    sources: List[str], hypotheses: List[str], references: List[str]
) -> float:
    """
    Scores translations with COMET-22 (system score, rounded to 4 decimals).

    Args:
        sources (list[str]): Corpus source sentences.
        hypotheses (list[str]): Engine translations.
        references (list[str]): Reference translations.

    Returns:
        float: The COMET-22 system score.

    Raises:
        EngineError: If unbabel-comet is not installed in the environment.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise EngineError(_INSTALL_HINT)

    model = load_from_checkpoint(download_model(COMET22_MODEL))
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, hypotheses, references)
    ]
    prediction = model.predict(data, batch_size=8, gpus=0)
    return round(float(prediction.system_score), 4)
