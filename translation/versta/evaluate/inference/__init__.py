from transformers import AutoConfig

from .base import InferenceEngine


def get_engine(
    model_path: str, max_seq_length: int, device: str | None = None
) -> InferenceEngine:
    model_type = _detect_model_type(model_path)

    if model_type == "opusmt":
        from .opusmt import OpusMTEngine

        engine: InferenceEngine = OpusMTEngine()
    elif model_type == "nllb":
        from .nllb import NllbEngine

        engine = NllbEngine()
    else:
        from .versta import VerstaEngine

        engine = VerstaEngine()

    engine.load(model_path, max_seq_length, device)
    return engine


def _detect_model_type(model_path: str) -> str:
    # Fast path: detect by model name pattern
    model_lower = model_path.lower()
    if "opus-mt" in model_lower:
        return "opusmt"
    if "nllb" in model_lower:
        return "nllb"

    # Config-based detection
    try:
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "marian":
            return "opusmt"
        if config.model_type in ("nllb", "m2m_100"):
            return "nllb"
    except Exception:
        pass

    # Fallback: load raw config.json directly (works around version incompatibilities)
    try:
        import json
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_path, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        model_type = config_dict.get("model_type", "")
        if model_type == "marian":
            return "opusmt"
        if model_type in ("nllb", "m2m_100"):
            return "nllb"
    except Exception:
        pass

    return "versta"
