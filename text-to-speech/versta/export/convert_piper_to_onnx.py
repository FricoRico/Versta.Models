import json
from pathlib import Path
from shutil import copyfile
from struct import pack
from huggingface_hub import hf_hub_download


def convert_piper_to_onnx(repo_name: str, export_path: Path, voice: str = None) -> Path:
    """
    Downloads the specified Piper TTS model (already in ONNX format) and prepares it for export.

    Args:
        repo_name (str): Name of the Piper repository (e.g., "rhasspy/piper-voices")
        export_path (Path): Path to the directory where the ONNX model will be saved.
        voice (str): Voice path specification (e.g., "nl/nl_NL/mls/medium", "de/de_DE/mls_6892/low")
    
    Returns:
        Path: The path to the exported ONNX model.
    """
    if not voice:
        raise ValueError("Voice path must be specified for Piper models. Use --sub_voice parameter.")
    
    print(f"Downloading Piper model from {repo_name} with voice path {voice}...")
    
    model_name = _get_model_name(voice)
    
    model_file_path = hf_hub_download(
        repo_id=repo_name,
        filename=f"{voice}/{model_name}.onnx",
    )

    config_file_path = hf_hub_download(
        repo_id=repo_name,
        filename=f"{voice}/{model_name}.onnx.json",
    )
        
    model_output = export_path / "model.onnx"
    config_output = export_path / "config.json"
    
    copyfile(model_file_path, model_output)
    copyfile(config_file_path, config_output)

    _extract_vocab(export_path, config_output)

    _set_model_type(config_output, "piper")

    return model_output

def _get_model_name(voice: str) -> str:
    """
    Extracts the model name from the voice path.

    Args:
        voice (str): Voice path specification (e.g., "nl/nl_NL/mls/medium", "de/de_DE/mls_6892/low").

    Returns:
        str: The model name derived from the voice path.
    """
    path_parts = voice.strip('/').split('/')
    if len(path_parts) < 3:
        raise ValueError(f"Invalid voice path '{voice}'. Expected format: 'language/locale/voice_type/quality'")

    locale = path_parts[1]  # e.g., "nl_NL", "de_DE"
    voice_type = path_parts[2]  # e.g., "mls", "mls_6892"
    quality = path_parts[3] if len(path_parts) > 3 else "medium"  # Default to medium

    return f"{locale}-{voice_type}-{quality}"

def _extract_vocab(export_dir: Path, config_file: Path) -> Path:
    """
    Extracts the vocabulary from the Piper model config.json and saves it in the export directory.
    
    Args:
        export_dir (Path): Path to the directory where the vocabulary will be saved.
        config_file (Path): Path to the configuration file of the Piper model.
    
    Returns:
        Path: Path to the created vocab.bin file.
    """
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}.")

    source_vocabulary = _load_vocabulary(config_file)
    optimized_vocabulary = export_dir / "vocab.bin"

    with open(optimized_vocabulary, "wb") as f:
        for word, index in source_vocabulary.items():
            # Write the word as a null-terminated byte string
            f.write(word.encode('utf-8') + b'\0')
            # Write the index as a 4-byte little-endian integer
            f.write(pack('<I', index))

    return optimized_vocabulary


def _load_vocabulary(config_file: Path) -> dict:
    """
    Loads the vocabulary from the Piper model configuration file.

    Args:
        config_file (Path): Path to the configuration file of the Piper model.

    Returns:
        dict: A dictionary containing the vocabulary with phoneme->index mapping.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        phoneme_id_map = config.get("phoneme_id_map", {})
        return {phoneme: ids[0] for phoneme, ids in phoneme_id_map.items()}


def _set_model_type(config_file: Path, model_type: str):
    """
    Sets the model type in the configuration file.

    Args:
        config_file (Path): Path to the configuration file.
        model_type (str): Type of the model to set.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["model_type"] = model_type

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_language_from_config(model: str, voice: str = None) -> dict:
    """
    Extracts language information from the Piper model configuration file.

    Args:
        model (str): Name of the Piper repository (e.g., "rhasspy/piper-voices").
        voice (str): Voice path specification (e.g., "nl/nl_NL/mls/medium", "de/de_DE/mls_6892/low").

    Returns:
        dict: Language information including code, family, region, etc.
    """
    if not voice:
        raise ValueError("Voice path must be specified for Piper models. Use --sub_voice parameter.")

    model_name = _get_model_name(voice)

    config_file = hf_hub_download(
        repo_id=model,
        filename=f"{voice}/{model_name}.onnx.json",
    )

    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        return config.get("language", {})