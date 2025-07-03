import json
from pathlib import Path
from shutil import copyfile
from struct import pack
from huggingface_hub import hf_hub_download


def convert_piper_to_onnx(model_name: str, export_path: Path, sub_voice: str = None) -> Path:
    """
    Downloads the specified Piper TTS model (already in ONNX format) and prepares it for export.

    Args:
        model_name (str): Name of the pre-trained Piper model (e.g., "nl-piet-medium")
        export_path (Path): Path to the directory where the ONNX model will be saved.
        sub_voice (str): Sub-voice specification (e.g., "mls", "amy", "arctic")
    
    Returns:
        Path: The path to the exported ONNX model.
    """
    print(f"Downloading Piper model {model_name} with sub-voice {sub_voice}...")
    
    # Piper models are organized by language and voice type
    # Example: nl-piet-medium, de-thorsten-medium, etc.
    # With sub-voice like mls: nl-mls-512-medium
    if sub_voice:
        # Construct the full model name with sub-voice
        parts = model_name.split('-')
        if len(parts) >= 2:
            language = parts[0]  # e.g., "nl", "de"
            full_model_name = f"{language}-{sub_voice}"
            if len(parts) > 2:
                # Add additional parts like "medium", "low", etc.
                full_model_name = "-".join([full_model_name] + parts[2:])
        else:
            full_model_name = f"{model_name}-{sub_voice}"
    else:
        full_model_name = model_name

    # Download from rhasspy/piper-voices repository
    repo_id = "rhasspy/piper-voices"
    
    try:
        # Download the ONNX model file
        model_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{full_model_name}.onnx",
            local_dir=export_path,
            local_dir_use_symlinks=False
        )
        
        # Download the config.json file
        config_file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{full_model_name}.onnx.json",
            local_dir=export_path,
            local_dir_use_symlinks=False
        )
        
    except Exception as e:
        raise FileNotFoundError(f"Could not download Piper model {full_model_name}: {e}")

    # Rename files to standard names expected by the export pipeline
    model_output = export_path / "model.onnx"
    config_output = export_path / "config.json"
    
    # Copy the downloaded files to standard names
    copyfile(model_file_path, model_output)
    copyfile(config_file_path, config_output)

    # Extract vocabulary from config
    _extract_vocab(export_path, config_output)

    # Set model type for compatibility
    _set_model_type(config_output, "piper")

    return model_output


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
        dict: A dictionary containing the vocabulary.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        # Piper vocabulary is typically in the "phoneme_id_map" field
        return config.get("phoneme_id_map", {})


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