"""
Generate scoring tables from OmniDocBench evaluation results.

Mirrors the logic from generate_result_tables.ipynb to compute
per-category scores and an overall score from the metric_result.json files.
"""

import json
from pathlib import Path
from typing import Dict, Any


def generate_result_tables(
    result_file: Path,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Generate score tables from a single *_metric_result.json file.

    Reads the OmniDocBench metric result JSON and computes:
    - text_block_Edit_dist (from ALL_page_avg)
    - display_formula_CDM (from page ALL, * 100)
    - table_TEDS (from page ALL, * 100)
    - table_TEDS_structure_only (from page ALL, * 100)
    - reading_order_Edit_dist (from ALL_page_avg)
    - overall = ((1 - text_block_Edit_dist) * 100 + display_formula_CDM + table_TEDS) / 3

    Args:
        result_file: Path to *_metric_result.json
        model_name: Display name for the model (used in summary output)

    Returns:
        Dict with scores dict and a summary string
    """
    result_file = Path(result_file)

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores = {}

    # Edit_dist metrics use ALL_page_avg (0-1, lower is better)
    scores["text_block_Edit_dist"] = data["text_block"]["all"]["Edit_dist"].get(
        "ALL_page_avg", float("nan")
    )
    scores["reading_order_Edit_dist"] = data["reading_order"]["all"]["Edit_dist"].get(
        "ALL_page_avg", float("nan")
    )

    # CDM metric uses page ALL (0-1, higher is better)
    cdm_value = data["display_formula"]["page"].get("CDM", {}).get("ALL")
    scores["display_formula_CDM"] = (
        cdm_value * 100 if cdm_value is not None else float("nan")
    )

    # TEDS metrics use page ALL (0-1, higher is better)
    teds_value = data["table"]["page"].get("TEDS", {}).get("ALL")
    scores["table_TEDS"] = teds_value * 100 if teds_value is not None else float("nan")

    teds_structure_value = (
        data["table"]["page"].get("TEDS_structure_only", {}).get("ALL")
    )
    scores["table_TEDS_structure_only"] = (
        teds_structure_value * 100 if teds_structure_value is not None else float("nan")
    )

    # Overall score: ((1 - text_block_Edit_dist) * 100 + display_formula_CDM + table_TEDS) / 3
    # All on 0-100 scale, higher is better
    if not any(
        v is None or (isinstance(v, float) and v != v)
        for v in [
            scores["text_block_Edit_dist"],
            scores["display_formula_CDM"],
            scores["table_TEDS"],
        ]
    ):
        overall = (
            (1 - scores["text_block_Edit_dist"]) * 100
            + scores["display_formula_CDM"]
            + scores["table_TEDS"]
        ) / 3
        scores["overall"] = round(overall, 3)
    else:
        scores["overall"] = float("nan")

    summary = _format_summary(model_name, scores)

    return {
        "model_name": model_name,
        "scores": scores,
        "summary": summary,
    }


def _format_summary(model_name: str, scores: Dict[str, float]) -> str:
    """Format scores as a printable table."""
    lines = [f"Results for {model_name}:"]
    lines.append("-" * 50)

    display_fields = [
        ("text_block_Edit_dist", "Text Block Edit Dist"),
        ("display_formula_CDM", "Display Formula CDM"),
        ("table_TEDS", "Table TEDS"),
        ("table_TEDS_structure_only", "Table TEDS (Structure Only)"),
        ("reading_order_Edit_dist", "Reading Order Edit Dist"),
    ]

    for key, label in display_fields:
        value = scores.get(key)
        if value is None or (isinstance(value, float) and value != value):
            formatted = "N/A"
        elif key.endswith("_dist"):
            formatted = f"{value:.3f}"
        else:
            formatted = f"{value:.3f}"
        lines.append(f"  {label:<30} {formatted}")

    lines.append("-" * 50)
    overall = scores.get("overall")
    if overall is None or (isinstance(overall, float) and overall != overall):
        lines.append(f"  {'Overall':<30} N/A")
    else:
        lines.append(f"  {'Overall':<30} {overall:.3f}")

    return "\n".join(lines)
