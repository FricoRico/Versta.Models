import random

from datasets import load_dataset as loader
from transformers import AutoTokenizer


def language_pair(dataset_name: str) -> tuple:
    """
    Extracts the source and target language from the dataset.

    Args:
        dataset_name: Name of the dataset to load.

    Returns:
        Tuple of (source_language, target_language).
    """
    dataset = loader(dataset_name, split="train")
    source = dataset[0]["source"]
    target = dataset[0]["target"]
    return source, target


def balance_tonalities(dataset, seed: int = 1780699690):
    """
    Resamples the dataset to achieve a 15% casual / 55% neutral / 30% formal tonality distribution.

    Keeps all neutral rows and subsamples casual/formal proportionally.

    Args:
        dataset: HuggingFace dataset with an 'instruction' field.
        seed: Random seed for deterministic sampling and shuffling.

    Returns:
        Resampled dataset with the target tonality distribution.
    """
    instructions = dataset["instruction"]
    casual_idx = [i for i, inst in enumerate(instructions) if inst.lower().endswith("casual:")]
    neutral_idx = [i for i, inst in enumerate(instructions) if inst.lower().endswith("neutral:")]
    formal_idx = [i for i, inst in enumerate(instructions) if inst.lower().endswith("formal:")]

    if not (neutral_idx and casual_idx and formal_idx):
        return dataset

    target_casual = min(len(casual_idx), int(len(neutral_idx) * 15 / 55))
    target_formal = min(len(formal_idx), int(len(neutral_idx) * 30 / 55))

    rng = random.Random(seed)
    selected = (
        rng.sample(casual_idx, target_casual)
        + neutral_idx
        + rng.sample(formal_idx, target_formal)
    )
    rng.shuffle(selected)

    return dataset.select(selected)


def load_dataset(
    dataset_name: str,
    model_name: str,
) -> object:
    """
    Loads and formats the translation dataset for SFT training.

    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model for tokenizer.

    Returns:
        Formatted dataset ready for SFT training.
    """
    dataset = loader(dataset_name, split="train")
    dataset = balance_tonalities(dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(
        lambda examples: _format(examples, tokenizer),
        batched=True,
        remove_columns=["source", "target", "instruction", "input", "output"],
        load_from_cache_file=True,
    )

    return dataset


def _format(examples: dict, tokenizer: object) -> dict:
    """
    Formats dataset examples into chat messages for SFT.

    Args:
        examples: Raw dataset examples with source, target, instruction, input, output fields.
        tokenizer: Tokenizer for the chat template.

    Returns:
        Dictionary with 'text' field containing formatted chat messages.
    """
    conversations_list = []
    for instruction, input_text, output_text in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        messages = [
            {"role": "user", "content": f"{instruction} {input_text}"},
            {"role": "assistant", "content": output_text},
        ]
        conversation_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        conversations_list.append(conversation_text)

    return {"text": conversations_list}
