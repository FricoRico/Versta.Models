"""
OmniDocBench evaluation wrapper.

Runs the full OmniDocBench end2end evaluation on prediction markdown files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def run_evaluation(
    predictions_dir: Path,
    gt_json_path: Path,
    output_dir: Path,
    match_method: str = "quick_match",
    match_workers: int = 4,
    skip_cdm: bool = True,
) -> Dict[str, Any]:
    """
    Run OmniDocBench end2end evaluation.

    Args:
        predictions_dir: Directory containing .md prediction files
        gt_json_path: Path to OmniDocBench.json ground truth
        output_dir: Directory for evaluation results
        match_method: Matching algorithm (quick_match, simple_match, no_split)
        match_workers: Number of parallel workers for matching
        skip_cdm: Deprecated, kept for backward compatibility

    Returns:
        Dictionary with evaluation results
    """
    from src.core.pipeline import run_config

    predictions_dir = Path(predictions_dir)
    gt_json_path = Path(gt_json_path)
    output_dir = Path(output_dir)

    # Create output directory
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Change to output directory for evaluation
    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        # Build config
        config = {
            "end2end_eval": {
                "metrics": {
                    "text_block": {"metric": ["Edit_dist"]},
                    "display_formula": {"metric": ["Edit_dist", "CDM"]},
                    "table": {"metric": ["Edit_dist", "TEDS"]},
                    "reading_order": {"metric": ["Edit_dist"]},
                },
                "dataset": {
                    "dataset_name": "end2end_dataset",
                    "ground_truth": {"data_path": str(gt_json_path)},
                    "prediction": {"data_path": str(predictions_dir)},
                    "match_method": match_method,
                    "match_workers": match_workers,
                },
            }
        }

        # Run evaluation
        run_config(config)

        # Load and return results
        result_files = list(result_dir.glob("*_metric_result.json"))
        if result_files:
            with open(result_files[0], "r") as f:
                return json.load(f)

        # Try alternative result files
        result_json_files = list(result_dir.glob("*_result.json"))
        if result_json_files:
            with open(result_json_files[0], "r") as f:
                return json.load(f)

        return {"status": "completed", "result_dir": str(result_dir)}

    finally:
        os.chdir(original_cwd)


def run_evaluation_from_config(
    predictions_dir: Path,
    gt_json_path: Path,
    output_dir: Path,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run OmniDocBench evaluation using a custom config file.

    Args:
        predictions_dir: Directory containing .md prediction files
        gt_json_path: Path to OmniDocBench.json ground truth
        output_dir: Directory for evaluation results
        config_path: Optional custom config YAML path

    Returns:
        Dictionary with evaluation results
    """
    from src.core.pipeline import run_config

    predictions_dir = Path(predictions_dir)
    gt_json_path = Path(gt_json_path)
    output_dir = Path(output_dir)

    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = os.getcwd()
    os.chdir(output_dir)

    try:
        if config_path:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            # Default config with CDM and TEDS
            config = {
                "end2end_eval": {
                    "metrics": {
                        "text_block": {"metric": ["Edit_dist"]},
                        "display_formula": {"metric": ["Edit_dist", "CDM"]},
                        "table": {"metric": ["Edit_dist", "TEDS"]},
                        "reading_order": {"metric": ["Edit_dist"]},
                    },
                    "dataset": {
                        "dataset_name": "end2end_dataset",
                        "ground_truth": {"data_path": str(gt_json_path)},
                        "prediction": {"data_path": str(predictions_dir)},
                        "match_method": "quick_match",
                        "match_workers": 4,
                    },
                }
            }

        run_config(config)

        result_files = list(result_dir.glob("*_metric_result.json"))
        if result_files:
            with open(result_files[0], "r") as f:
                return json.load(f)

        return {"status": "completed", "result_dir": str(result_dir)}

    finally:
        os.chdir(original_cwd)


def extract_metrics_from_results(result_dir: Path) -> Dict[str, float]:
    """Extract key metrics from evaluation results."""
    result_dir = Path(result_dir)

    metrics = {}

    # Try to find metric result file
    metric_files = list(result_dir.glob("*_metric_result.json"))
    if metric_files:
        with open(metric_files[0], "r") as f:
            data = json.load(f)
            # Extract key metrics
            if "text_block" in data:
                metrics["text_edit_dist"] = data["text_block"].get("Edit_dist", {})
            if "display_formula" in data:
                metrics["formula_edit_dist"] = data["display_formula"].get(
                    "Edit_dist", {}
                )
            if "table" in data:
                metrics["table_edit_dist"] = data["table"].get("Edit_dist", {})
            if "reading_order" in data:
                metrics["reading_order_edit_dist"] = data["reading_order"].get(
                    "Edit_dist", {}
                )

    return metrics
