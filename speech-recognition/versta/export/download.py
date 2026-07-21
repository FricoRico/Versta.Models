import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from .metadata import generate_metadata

DEFAULT_VAD_REPO = "ggml-org/whisper-vad"
DEFAULT_VAD_FILENAME = "ggml-silero-v6.2.0.bin"


def build_filename(model_type: str) -> str:
    """
    Builds the whisper.cpp model filename for the given model type.

    whisper.cpp publishes models as ``ggml-<model_type>.bin`` (e.g. ``ggml-base-q8_0.bin``,
    ``ggml-small.en.bin``, ``ggml-large-v3-turbo-q5_0.bin``).

    Args:
        model_type (str): Whisper model variant (e.g. "base-q8_0", "small.en").

    Returns:
        str: The model filename as published on the Hugging Face repository.
    """
    return f"ggml-{model_type}.bin"


def resolve_filename(repo_id: str, filename: str) -> str:
    """
    Verifies that ``filename`` exists in the given Hugging Face repository.

    Args:
        repo_id (str): Hugging Face repository id (e.g. "ggerganov/whisper.cpp").
        filename (str): File to look for in the repository.
        revision (str): Repository revision (branch, tag or commit).

    Returns:
        str: The validated filename.

    Raises:
        ValueError: If the file is not present in the repository.
    """
    available = list_repo_files(repo_id)
    if filename not in available:
        raise ValueError(
            f"'{filename}' not found in '{repo_id}'. "
            f"Available files (matching 'ggml-'): "
            f"{[f for f in available if f.startswith('ggml-')][:20]}"
        )
    return filename


def get_file_size(repo_id: str, filename: str) -> int:
    """
    Resolves the size (in bytes) of a repository file using the Hugging Face API.

    Args:
        repo_id (str): Hugging Face repository id.
        filename (str): File to inspect.
        revision (str): Repository revision.

    Returns:
        int: File size in bytes, or 0 if it cannot be determined.
    """
    try:
        api = HfApi()
        info = api.get_paths_info(repo_id=repo_id, paths=[filename])
        if info and getattr(info[0], "size", None):
            return int(info[0].size)
    except Exception:
        pass
    return 0


def download_file(
        repo_id: str,
        filename: str,
        output_dir: Path,
) -> Path:
    """
    Downloads a single file from a Hugging Face repository into ``output_dir``.

    Args:
        repo_id (str): Hugging Face repository id.
        filename (str): File to download.
        output_dir (Path): Directory where the file will be written.

    Returns:
        Path: The local path of the downloaded file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}/{filename}")
    return Path(hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
    ))


def download_model(
        repo_id: str,
        model_type: str,
        output_dir: Path,
        languages: list = None,
        vad_repo: str = DEFAULT_VAD_REPO,
        vad_filename: str = DEFAULT_VAD_FILENAME,
) -> Path:
    """
    Downloads a whisper.cpp ggml model (and its required Silero-VAD model) from Hugging Face
    and writes a metadata.json describing the model in the format expected by the Versta
    speech-recognition module.

    No integrity verification is performed; sizes are recorded best-effort from the
    Hugging Face API.

    Args:
        repo_id (str): Hugging Face repository id holding the whisper model.
        model_type (str): Whisper model variant (e.g. "base-q8_0", "small.en").
        output_dir (Path): Directory where the model and metadata will be written.
        languages (list): Supported language codes. Defaults to the full Whisper set,
            or ``["en"]`` for English-only (".en") model variants.
        vad_repo (str): Hugging Face repository id holding the VAD model.
        vad_filename (str): VAD model filename.

    Returns:
        Path: The output directory containing the downloaded models and metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    model_filename = build_filename(model_type)
    resolve_filename(repo_id, model_filename)
    resolve_filename(vad_repo, vad_filename)

    model_path = download_file(repo_id, model_filename, model_dir)
    vad_path = download_file(vad_repo, vad_filename, model_dir)

    model_size = get_file_size(repo_id, model_filename) or os.path.getsize(model_path)
    vad_size = get_file_size(vad_repo, vad_filename) or os.path.getsize(vad_path)

    metadata_file = generate_metadata(model_dir, model_type, repo_id, model_filename, vad_filename, languages)

    with open(metadata_file, "r", encoding="utf-8") as handle:
        language_count = len(json.load(handle).get("languages", []))

    cache_dir = model_dir / ".cache"
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)

    print(
        f"Prepared '{model_filename}' ({model_size} bytes) + "
        f"'{vad_filename}' ({vad_size} bytes) with {language_count} languages."
    )

    return model_dir
