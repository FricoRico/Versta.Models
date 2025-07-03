# Piper TTS Model Support

This module adds support for converting Piper TTS models to ORT format for use in the Versta Android app.

## Overview

Piper TTS models are already in ONNX format and primarily require:
1. Downloading from the rhasspy/piper-voices HuggingFace repository
2. Vocabulary extraction from config.json to vocab.bin
3. Quantization for mobile deployment
4. Conversion to ORT format

## Usage

### Basic Command

```bash
python -m versta.export \
    --model rhasspy/piper-voices \
    --model_format piper \
    --sub_voice nl/nl_NL/mls/medium \
    --output_dir ./output
```

### Voice Path Specification

The `--sub_voice` parameter specifies the directory path within the Piper repository to the desired voice model. Examples:

- `nl/nl_NL/mls/medium` - Dutch MLS voice, medium quality
- `nl/nl_NL/mls_5809/low` - Dutch MLS voice with speaker ID 5809, low quality
- `de/de_DE/mls_6892/low` - German MLS voice with speaker ID 6892, low quality

### Output Structure

The export process creates:
```
output/
├── model.ort          # Quantized ORT model for inference
├── vocab.bin          # Binary vocabulary file
├── config.json        # Model configuration
└── metadata.json      # Model metadata for Android app
```

The output directory will be automatically named based on language information from the model's config.json file (e.g., `dutch_nl-nl`, `german_de-de`).

## Implementation Details

### Files Added

- `convert_piper_to_onnx.py`: Downloads Piper models and extracts vocabulary
- `quantize_piper.py`: Quantizes Piper models for mobile deployment
- Updated `convert_onnx.py` and `quantize.py` to support Piper format
- Updated `__main__.py` to add `--sub_voice` parameter

### Key Features

1. **Model Download**: Downloads ONNX models from rhasspy/piper-voices
2. **Vocabulary Extraction**: Extracts phoneme_id_map from config.json to vocab.bin
3. **Quantization**: ARM64-optimized quantization for mobile deployment
4. **Metadata Generation**: Creates metadata.json with Piper architecture info
5. **Language-based Output Naming**: Automatically names output folders based on language info

### Error Handling

- Graceful handling of network errors when downloading models
- Clear error messages for missing models or configurations
- Validation of model file existence before processing
- Validation of voice path format

## Examples

### Dutch MLS Model
```bash
python -m versta.export --model rhasspy/piper-voices --model_format piper --sub_voice nl/nl_NL/mls/medium
```

### German MLS Model  
```bash
python -m versta.export --model rhasspy/piper-voices --model_format piper --sub_voice de/de_DE/mls_6892/low
```

### Dutch MLS Model with Specific Speaker
```bash
python -m versta.export --model rhasspy/piper-voices --model_format piper --sub_voice nl/nl_NL/mls_5809/low
```