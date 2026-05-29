import json
import random
from pathlib import Path


def load_custom_dataset(dataset_path: str) -> list[dict]:
    try:
        dataset_path_obj = Path(dataset_path)
        if dataset_path_obj.exists() and dataset_path_obj.suffix == ".jsonl":
            with open(dataset_path_obj, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        pass

    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(dataset_path, split="train")
    return dataset.to_list()


def filter_dataset(
    data: list[dict],
    source: str,
    target: str,
    tones: list[str],
    percentage: float,
    is_flores: bool,
    seed: int = 42,
) -> list[dict]:
    filtered = [
        item
        for item in data
        if item.get("source") == source and item.get("target") == target
    ]

    if not is_flores and tones:
        filtered = [
            item
            for item in filtered
            if any(
                tone.lower() in item.get("instruction", "").lower() for tone in tones
            )
        ]

    rng = random.Random(seed)
    sample_size = int(len(filtered) * percentage)

    sampled = rng.sample(filtered, min(sample_size, len(filtered)))
    return sampled
