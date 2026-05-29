from transformers import AutoConfig

from .base import InferenceEngine


def get_engine(
    model_path: str, max_seq_length: int, device: str | None = None
) -> InferenceEngine:
    model_type = _detect_model_type(model_path)

    if model_type == "opusmt":
        from .opusmt import OpusMTEngine

        engine: InferenceEngine = OpusMTEngine()
    else:
        from .versta import VerstaEngine

        engine = VerstaEngine()

    engine.load(model_path, max_seq_length, device)
    return engine


def _detect_model_type(model_path: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_path)
        if config.model_type == "marian":
            return "opusmt"
    except Exception:
        pass
    return "versta"
