
from pathlib import Path

from datasets import load_dataset


def clean_quotes(text):
    """Remove surrounding quotes and extra whitespace from text.

    Args:
        text: Text to clean.

    Returns:
        Cleaned text with surrounding quotes removed and whitespace collapsed.
    """
    if not text:
        return text

    import re

    text = str(text).strip()

    while len(text) >= 2 and text[0] == text[-1] and text[0] in ['"', '"']:
        text = text[1:-1]

    return re.sub(r'\s+', ' ', text).strip()


def clean_batch(batch):
    """Clean input and output fields in a batch by removing surrounding quotes.

    Args:
        batch: Dictionary containing 'input' and 'output' fields with list values.

    Returns:
        Cleaned batch with quotes removed from all input/output strings.
    """
    for field in ["input", "output"]:
        batch[field] = [clean_quotes(t) if t else t for t in batch[field]]
    return batch


def upload_dataset(
    input_paths: list[Path],
    dataset_name: str = "Neurora/versta-tonality-en-nl",
) -> None:
    """Load processed JSONL dataset, clean it, and push to HuggingFace Hub.

    Args:
        input_paths (list[Path]): List of paths to JSONL files.
        dataset_name (str): HuggingFace dataset repository name.
    """
    dataset = load_dataset("json", data_files=[str(p) for p in input_paths])
    dataset = dataset.map(clean_batch, batched=True, batch_size=1000)
    dataset.push_to_hub(dataset_name)
