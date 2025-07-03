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
    --model nl \
    --model_format piper \
    --sub_voice mls \
    --output_dir ./output/nl-mls
```

### Supported Languages

The implementation currently supports:
- Dutch (nl) with mls sub-voice: `nl-mls_5809-medium`
- German (de) with mls sub-voice: `de-mls_6892-medium`

### Sub-voice Options

- `mls`: Multi-language speaker models (recommended for Dutch and German)
- `amy`, `arctic`: Other voice options (model-dependent)

### Output Structure

The export process creates:
```
output/
├── model.ort          # Quantized ORT model for inference
├── vocab.bin          # Binary vocabulary file
├── config.json        # Model configuration
└── metadata.json      # Model metadata for Android app
```

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

### Error Handling

- Graceful handling of network errors when downloading models
- Clear error messages for missing models or configurations
- Validation of model file existence before processing

## Examples

### Dutch MLS Model
```bash
python -m versta.export --model nl --model_format piper --sub_voice mls
```

### German MLS Model  
```bash
python -m versta.export --model de --model_format piper --sub_voice mls
```