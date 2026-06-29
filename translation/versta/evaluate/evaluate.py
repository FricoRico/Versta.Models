from datetime import datetime
from typing import List

import torch

from .dataset import filter_dataset, is_flores_dataset, load_dataset
from .inference import get_engine
from .metrics import BleuMetric, ChrfMetric, CometMetric
from .types import EvaluationConfig, EvaluationResult, ToneResults


def _compute_metrics(
    references: List[str],
    hypotheses: List[str],
    use_comet: bool,
    use_bleu: bool,
    use_chrf: bool,
) -> ToneResults:
    result: ToneResults = {"comet": None, "bleu": None, "chrf": None}

    if use_comet:
        comet_metric = CometMetric()
        result["comet"] = comet_metric.compute(references, hypotheses)

    if use_bleu:
        bleu_metric = BleuMetric()
        result["bleu"] = bleu_metric.compute(references, hypotheses)

    if use_chrf:
        chrf_metric = ChrfMetric()
        result["chrf"] = chrf_metric.compute(references, hypotheses)

    return result


def _aggregate_results(
    by_tone: dict[str, ToneResults],
    use_comet: bool,
    use_bleu: bool,
    use_chrf: bool,
) -> ToneResults:
    overall: ToneResults = {"comet": None, "bleu": None, "chrf": None}

    tone_scores = list(by_tone.values())

    if use_comet:
        comet_scores = [r["comet"] for r in tone_scores if r["comet"] is not None]
        overall["comet"] = (
            sum(comet_scores) / len(comet_scores) if comet_scores else None
        )

    if use_bleu:
        bleu_scores = [r["bleu"] for r in tone_scores if r["bleu"] is not None]
        overall["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None

    if use_chrf:
        chrf_scores = [r["chrf"] for r in tone_scores if r["chrf"] is not None]
        overall["chrf"] = sum(chrf_scores) / len(chrf_scores) if chrf_scores else None

    return overall


def evaluate(config: EvaluationConfig) -> EvaluationResult:
    engine = get_engine(
        config["model"], config["max_seq_length"], config.get("device")
    )
    config["model_type"] = engine.model_type

    dataset_type = "flores_plus" if is_flores_dataset(config["dataset"]) else "custom"
    raw_data = load_dataset(config["dataset"], config["source"], config["target"])

    is_flores = dataset_type == "flores_plus"

    if is_flores:
        filtered_data = raw_data
    else:
        filtered_data = filter_dataset(
            raw_data,
            config["source"],
            config["target"],
            config["tones"],
            config["percentage"],
            is_flores=is_flores,
        )

    raw_sample_count = len(filtered_data)

    if is_flores:
        expanded = []
        for item in filtered_data:
            for tone in config["tones"]:
                copy = dict(item)
                copy["_tone"] = tone
                copy["_group_key"] = tone
                expanded.append(copy)
        filtered_data = expanded
    else:
        for item in filtered_data:
            group_key = None
            instruction = item.get("instruction", "").lower()
            for tone in config["tones"]:
                if tone == "plain":
                    if not any(t in instruction for t in ["formal", "neutral", "casual"]):
                        group_key = "plain"
                        break
                elif tone in instruction:
                    group_key = tone
                    break
            item["_tone"] = group_key or "neutral"
            item["_group_key"] = group_key or "neutral"

    by_tone: dict[str, ToneResults] = {}
    references_per_tone: dict[str, List[str]] = {}
    hypotheses_per_tone: dict[str, List[str]] = {}

    for item in filtered_data:
        tone_key = item["_group_key"]
        references_per_tone[tone_key] = references_per_tone.get(tone_key, [])
        references_per_tone[tone_key].append(item.get("output", ""))

    hypotheses = engine.generate(
        data=filtered_data,
        target=config["target"],
        batch_size=config["batch_size"],
        gen_config=config["gen_config"],
        max_seq_length=config["max_seq_length"],
    )

    for idx, item in enumerate(filtered_data):
        tone_key = item["_group_key"]
        hypotheses_per_tone[tone_key] = hypotheses_per_tone.get(tone_key, [])
        hypotheses_per_tone[tone_key].append(hypotheses[idx])

    for tone_key, refs in references_per_tone.items():
        hyps = hypotheses_per_tone.get(tone_key, [])
        if refs and hyps:
            by_tone[tone_key] = _compute_metrics(
                refs,
                hyps,
                config["use_comet"],
                config["use_bleu"],
                config["use_chrf"],
            )

    overall = _aggregate_results(
        by_tone, config["use_comet"], config["use_bleu"], config["use_chrf"]
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return EvaluationResult(
        config=config,
        overall=overall,
        by_tone=by_tone,
        num_samples=raw_sample_count,
        model_name=config["model"],
        dataset_type=dataset_type,
        dataset_name=config["dataset"],
        timestamp=datetime.utcnow().isoformat() + "Z",
    )
