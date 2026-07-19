import json
import re
from gzip import decompress
from hashlib import sha256
from pathlib import Path
from typing import Dict, List
from urllib.request import urlopen, Request


def load_registry(registry_url: str) -> dict:
    """
    Loads the Firefox translations model registry from the public Google Cloud Storage bucket.

    The registry is a JSON document with a `baseUrl` and a `models` dictionary keyed by language
    pair (e.g. "en-nl"). Each pair maps to a list of entries, one per model architecture
    ("tiny", "base", "base-memory").

    Args:
        registry_url (str): URL of the registry JSON document.

    Returns:
        dict: The parsed registry document.
    """
    with urlopen(Request(registry_url, method="GET"), timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_language(code: str) -> str:
    """
    Normalizes a user-supplied language code to the form used as a key in the Firefox registry.

    The registry uses bare ISO-639 codes (e.g. "en", "nl") and, for Chinese, distinguishes
    "zh" (simplified) from "zh_hant" (traditional). Script- or region-qualified inputs are mapped
    accordingly so callers may request models using common BCP-47-style tags:

      * "zh-Hans"/"zh-CN"/"zh"            -> "zh"      (simplified)
      * "zh-Hant"/"zh-TW"/"zh-HK"/"zh-MO" -> "zh_hant" (traditional)
      * "en-US"/"en-GB"/"pt-BR"          -> "en"/"pt" (region subtag stripped)
      * "no"/"nb"/"nn"/"hbs"             -> unchanged (distinct real pairs, never collapsed)

    Args:
        code (str): A language code, possibly with script/region subtags (e.g. "zh-Hans").

    Returns:
        str: The normalized language code used in registry pair keys.
    """
    parts = re.split(r"[-_]", code.lower())
    base = parts[0]

    if base == "zh":
        return "zh_hant" if any(part in ("hant", "tw", "hk", "mo") for part in parts[1:]) else "zh"

    return base


def get_pair_entries(registry: dict, source_language: str, target_language: str) -> List[dict]:
    """
    Returns all registry entries for the given language pair, regardless of architecture.

    Both language codes are normalized (see `normalize_language`) so script- or region-qualified
    inputs such as "zh-Hans" resolve to the correct registry key.

    Args:
        registry (dict): The registry document as returned by `load_registry`.
        source_language (str): Source language code (e.g. "en" or "zh-Hans").
        target_language (str): Target language code (e.g. "nl" or "zh-Hant").

    Returns:
        list[dict]: The matching registry entries.
    """
    source = normalize_language(source_language)
    target = normalize_language(target_language)
    pair = f"{source}-{target}"
    return registry.get("models", {}).get(pair, [])


ARCHITECTURE_PREFERENCE = ("tiny", "base-memory", "base")


def get_best_entry(registry: dict, source: str, target: str) -> dict:
    """
    Returns the best available registry entry for a language pair, choosing the architecture with
    the highest preference. Preference order is tiny -> base-memory -> base (see
    `ARCHITECTURE_PREFERENCE`); if none of the preferred architectures are published for the pair,
    the first available entry is returned.

    Args:
        registry (dict): The registry document as returned by `load_registry`.
        source (str): Source language code (e.g. "en").
        target (str): Target language code (e.g. "nl").

    Returns:
        dict: The selected registry entry.

    Raises:
        ValueError: If the language pair has no published models at all.
    """
    entries = get_pair_entries(registry, source, target)

    if not entries:
        raise ValueError(f"No models available for '{source}-{target}'.")

    for architecture in ARCHITECTURE_PREFERENCE:
        for entry in entries:
            if entry.get("architecture") == architecture:
                return entry

    return entries[0]


def get_entry(
        registry: dict,
        source: str,
        target: str,
        architecture: str = None,
) -> dict:
    """
    Returns the registry entry for a specific language pair and architecture.

    Args:
        registry (dict): The registry document as returned by `load_registry`.
        source (str): Source language code (e.g. "en").
        target (str): Target language code (e.g. "nl").
        architecture (str): Model architecture ("tiny", "base" or "base-memory"). When None, the
            best available architecture is selected automatically (see `get_best_entry`).

    Returns:
        dict: The matching registry entry.
    """
    if architecture is None:
        return get_best_entry(registry, source, target)

    entries = get_pair_entries(registry, source, target)

    for entry in entries:
        if entry.get("architecture") == architecture:
            return entry

    available = ", ".join(e.get("architecture") for e in entries) or "none"
    raise ValueError(
        f"No '{architecture}' model for '{source}-{target}'. "
        f"Available architectures: {available}."
    )


def download_model(
        base_url: str,
        entry: dict,
        output_dir: Path,
) -> Path:
    """
    Downloads the three Bergamot model files (model, lexical shortlist and vocabulary) for a single
    translation direction, decompresses them and writes a metadata.json describing the model.

    The downloaded files are already in the native Bergamot format (.bin/.spm).

    Args:
        base_url (str): Base URL of the storage bucket (from the registry's `baseUrl`).
        entry (dict): A registry entry as returned by `get_entry`.
        output_dir (Path): Directory where the model files and metadata will be written.

    Returns:
        Path: The output directory containing the downloaded model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_language = entry["sourceLanguage"]
    target_language = entry["targetLanguage"]
    architecture = entry["architecture"]
    files = entry["files"]

    downloaded: Dict[str, str] = {}

    for kind, descriptor in files.items():
        remote_path = descriptor["path"]
        url = f"{base_url.rstrip('/')}/{remote_path.lstrip('/')}"

        print(f"Downloading {url}")
        with urlopen(Request(url, method="GET"), timeout=300) as response:
            compressed = response.read()

        decompressed = decompress(compressed)
        file_name = Path(remote_path).stem
        output_file = output_dir / file_name

        with open(output_file, "wb") as handle:
            handle.write(decompressed)

        if "uncompressedHash" in descriptor:
            actual = sha256(decompressed).hexdigest()
            expected = descriptor["uncompressedHash"]
            if actual != expected:
                raise ValueError(
                    f"Hash mismatch for {file_name}: expected {expected}, got {actual}."
                )

        downloaded[kind] = file_name

    model_metadata = _load_model_metadata(base_url, files.get("model", {}))
    model_config = _extract_config(model_metadata)
    model_version = _extract_version(model_metadata)

    files_metadata = {
        "model": downloaded["model"],
        "vocabulary": downloaded.get("srcVocab", downloaded.get("vocab")),
        "target_vocabulary": downloaded.get("trgVocab"),
        "shortlist": downloaded["lexicalShortlist"],

    }

    config_metadata = {
        "encoder_layers": model_config["encoder_layers"],
        "decoder_layers": model_config["decoder_layers"],
        "ffn_depth": model_config["ffn_depth"],
        "num_heads": model_config["num_heads"],
        "split_mode": "sentence",
    }

    metadata = {
        "directory": output_dir.name,
        "source_language": source_language,
        "target_language": target_language,
        "architecture": architecture,
        "base_model": f"{source_language}-{target_language}:{architecture}",
        "score": _extract_score(entry),
        "version": model_version,
        "files": files_metadata,
        "config": config_metadata,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)

    return output_dir


def _extract_score(entry: dict) -> float:
    """
    Extracts a quality score (COMET-22) from the registry entry metrics, if available.
    """
    metrics = entry.get("metrics", {})
    flores = metrics.get("flores200-plus", {})
    return float(flores.get("comet22", 0.0))


def _load_model_metadata(base_url: str, model_descriptor: dict) -> dict:
    """
    Loads the per-model metadata.json published alongside the model file in the storage bucket. This
    document carries the full `modelConfig` (layer counts, heads, feed-forward depth, version).

    Returns an empty dict if the metadata cannot be fetched or parsed.
    """
    remote_path = model_descriptor.get("path", "")
    if not remote_path:
        return {}

    metadata_path = str(Path(remote_path).parent / "metadata.json")
    url = f"{base_url.rstrip('/')}/{metadata_path.lstrip('/')}"

    try:
        with urlopen(Request(url, method="GET"), timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def _extract_config(model_metadata: dict) -> dict:
    """
    Builds the per-model configuration metadata from the model's `modelConfig` block.

    Raises:
        ValueError: If the required configuration keys are missing from `modelConfig`.
    """
    cfg = model_metadata.get("modelConfig", {})
    required = {
        "encoder_layers": "enc-depth",
        "decoder_layers": "dec-depth",
        "ffn_depth": "transformer-ffn-depth",
        "num_heads": "transformer-heads",
    }

    missing = [src for src, key in required.items() if key not in cfg]
    if missing:
        raise ValueError(
            f"modelConfig is missing required keys: {', '.join(missing)}."
        )

    return {
        "encoder_layers": int(cfg["enc-depth"]),
        "decoder_layers": int(cfg["dec-depth"]),
        "ffn_depth": int(cfg["transformer-ffn-depth"]),
        "num_heads": int(cfg["transformer-heads"]),
    }


def _extract_version(model_metadata: dict) -> str:
    """
    Extracts the released model version from the per-model metadata.json. The raw version has the form
    "v1.12.14 2d067af 2024-02-16 11:44:13 -0500"; only the leading "vX.Y.Z" tag is kept.

    Returns an empty string if the version cannot be determined.
    """
    raw_version = model_metadata.get("modelConfig", {}).get("version", "")
    if not raw_version:
        return ""

    match = re.match(r"(v\d+\.\d+\.\d+)", raw_version)
    return match.group(1) if match else raw_version
