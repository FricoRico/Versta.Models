import json
from pathlib import Path

from .types import CorpusConfig, CorpusGroupConfig, LanguagePairConfig


def load_corpus_config(corpus_path: Path) -> list[LanguagePairConfig]:
    """Load and validate corpus configuration from a JSON file.

    Args:
        corpus_path: Path to the corpora JSON config file.

    Returns:
        List of LanguagePairConfig objects.

    Raises:
        ValueError: If the file doesn't exist or isn't a JSON file.
        KeyError: If required fields are missing from the config.
    """
    corpus_path = Path(corpus_path)

    if not corpus_path.exists() or corpus_path.suffix != ".json":
        raise ValueError(
            f"'{corpus_path}' is not a valid JSON file. "
            "Please provide a path to a corpora JSON config file."
        )

    with open(corpus_path, "r", encoding="utf-8") as f:
        configs = json.load(f)

    def _parse_entries(entries: list[dict]) -> list[CorpusConfig]:
        return [
            CorpusConfig(
                corpus=c["corpus"],
                pairs=c.get("pairs"),
                release=c.get("release"),
                register=c.get("register"),
                preprocess=c.get("preprocess"),
            )
            for c in entries
        ]

    def _parse_group(group: dict | None) -> CorpusGroupConfig:
        group = group or {}
        return CorpusGroupConfig(
            synthetic=_parse_entries(group.get("synthetic", [])),
            natural=_parse_entries(group.get("natural", [])),
        )

    typed_configs = []
    for config in configs:
        typed_configs.append(
            LanguagePairConfig(
                source=config["source"],
                target=config["target"],
                train=_parse_group(config.get("train")),
                eval=_parse_group(config.get("eval")),
            )
        )

    return typed_configs


def filter_corpus_config(
    configs: list[LanguagePairConfig],
    source: str | None = None,
    target: str | None = None,
) -> list[LanguagePairConfig]:
    """Filter corpus configurations by source/target language pair.

    Args:
        configs: List of LanguagePairConfig objects to filter.
        source: Optional source language code to filter by.
        target: Optional target language code to filter by.

    Returns:
        Filtered list of LanguagePairConfig objects.

    Raises:
        ValueError: If source and target are provided but no matching config is found.
    """
    if source is not None and target is not None:
        filtered = [
            c for c in configs if c["source"] == source and c["target"] == target
        ]
        if not filtered:
            available = ", ".join(f"{c['source']}-{c['target']}" for c in configs)
            raise ValueError(
                f"No language pair found for {source}-{target}. "
                f"Available pairs: {available}"
            )
        return filtered

    return configs
