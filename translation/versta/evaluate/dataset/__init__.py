from .custom import filter_dataset, load_custom_dataset
from .flores import is_flores_dataset, load_flores_dataset


def load_dataset(dataset_path: str, source: str, target: str) -> list[dict]:
    if is_flores_dataset(dataset_path):
        return load_flores_dataset(source, target)
    return load_custom_dataset(dataset_path)
