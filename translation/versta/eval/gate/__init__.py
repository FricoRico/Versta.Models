"""
The release gate (NEU-44/NEU-52): may this model refresh ship?

Compares one candidate ``versta.eval-result/1`` against the direction's
checked-in ``evals/`` ledger and decides ``baseline`` / ``pass`` / ``fail``
/ ``override`` / ``rebaseline``. The gate also **generates** the release
eval-summary markdown, so what is posted with a release is always derived
from what is recorded in the ledger — never handwritten prose drifting
from the numbers.

Run it as ``uv run python -m versta.eval.gate --result <result.json>``;
see ``evals/README.md`` for the ledger contract and gate semantics.
"""

from .check import GateError, GateReport, evaluate_gate, gate_info
from .summary import render_summary

__all__ = [
    "GateError",
    "GateReport",
    "evaluate_gate",
    "gate_info",
    "render_summary",
]
