from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Any, TypedDict
from pycountry import languages


class ArchitectureConfig(TypedDict):
    num_layers: int
    num_heads: int
    head_dim: int
    d_model: int


def get_source_language(model_name: str) -> str:
    """
    Extracts the source language from the model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_config = tokenizer.init_kwargs

    language_code = tokenizer_config.get("source_lang")
    return normalize_language_code(language_code)


def get_target_language(model_name: str) -> str:
    """
    Extracts the target language from the model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_config = tokenizer.init_kwargs

    language_code = tokenizer_config.get("target_lang")
    return normalize_language_code(language_code)


def get_architecture(model_name: str) -> List[str]:
    """
    Extracts the architecture from the model name.

    Args:
        model_name (str): Name of the model to extract the architecture from.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    config = model.config

    return config.architectures


def get_architecture_config(model_name: str) -> ArchitectureConfig:
    """
    Extracts the architecture configuration needed for decoder cache initialization.

    Args:
        model_name (str): Name of the model to extract the architecture config from.

    Returns:
        ArchitectureConfig: Dictionary containing num_layers, num_heads, head_dim, d_model.
    """
    config = AutoConfig.from_pretrained(model_name)

    num_layers = getattr(config, "num_hidden_layers", None) or getattr(
        config, "decoder_layers", 6
    )
    num_heads = getattr(config, "num_attention_heads", None) or getattr(
        config, "decoder_attention_heads", 8
    )
    d_model = getattr(config, "d_model", None) or getattr(config, "hidden_size", 512)

    if d_model and num_heads:
        head_dim = d_model // num_heads
    else:
        head_dim = 64

    return ArchitectureConfig(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        d_model=d_model,
    )


def normalize_language_code(language_code: str) -> str:
    """
    Normalizes the language code to a standard format.

    Args:
        language_code (str): Language code to normalize.
    """
    if len(language_code) == 2:
        return language_code

    try:
        locale = languages.get(alpha_3=language_code)

        if hasattr(locale, "alpha_2"):
            return locale.alpha_2
        else:
            return None
    except KeyError:
        return None
