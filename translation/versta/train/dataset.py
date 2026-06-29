import random

import pyarrow.compute as pc
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


def _get_balanced_indices(dataset, seed: int = 1780699690):
    """
    Returns the full index pools and target sample sizes for each tonality.
    """
    instructions = dataset.data.column("instruction")
    casual_idx = (
        pc.match_substring(instructions, "casual").to_numpy().nonzero()[0].tolist()
    )
    neutral_idx = (
        pc.match_substring(instructions, "neutral").to_numpy().nonzero()[0].tolist()
    )
    formal_idx = (
        pc.match_substring(instructions, "formal").to_numpy().nonzero()[0].tolist()
    )

    target_casual = min(len(casual_idx), int(len(neutral_idx) * 15 / 55))
    target_formal = min(len(formal_idx), int(len(neutral_idx) * 30 / 55))

    return casual_idx, neutral_idx, formal_idx, target_casual, target_formal


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
    casual_idx, neutral_idx, formal_idx, target_casual, target_formal = (
        _get_balanced_indices(dataset, seed)
    )

    if not (neutral_idx and casual_idx and formal_idx):
        return dataset

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
    Loads and formats the translation dataset for SFT training into two disjoint subsets.

    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model for tokenizer.

    Returns:
        Formatted dataset ready for SFT training.
    """
    dataset = loader(dataset_name, split="train")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = dataset.map(
        lambda examples: _format(examples, tokenizer),
        batched=True,
        remove_columns=["source", "target", "instruction", "input", "output"],
        load_from_cache_file=True,
    )

    return dataset


def load_eval_dataset(
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    split: str = "eval",
) -> object:
    """
    Loads a separate evaluation dataset with prompt masking for metric computation.

    Each row is tokenized so that only the assistant response (the translation output)
    contributes to the loss/metrics. Prompt positions (system + user) are masked with -100
    in the labels.

    Args:
        dataset_name: Name of the dataset to load.
        model_name: Name of the model for tokenizer.
        max_seq_length: Maximum sequence length for truncation.
        split: Dataset split to load (default: "eval").

    Returns:
        Dataset with input_ids, attention_mask, and labels columns.
    """
    dataset = loader(dataset_name, split=split)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format_eval(examples):
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for instruction, input_text, output_text in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            prompt_text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer.encode(
                prompt_text,
                truncation=True,
                max_length=max_seq_length,
            )
            prompt_len = len(prompt_ids)

            output_ids = tokenizer.encode(
                output_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, max_seq_length - prompt_len),
            )

            input_ids = prompt_ids + output_ids
            labels = [-100] * prompt_len + output_ids
            attention_mask = [1] * len(input_ids)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "attention_mask": batch_attention_mask,
        }

    dataset = dataset.map(
        _format_eval,
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
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        conversation_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        conversations_list.append(conversation_text)

    return {"text": conversations_list}
