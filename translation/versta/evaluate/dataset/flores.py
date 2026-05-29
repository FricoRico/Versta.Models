from collections import defaultdict

import pycountry
from datasets import load_dataset as hf_load_dataset


def is_flores_dataset(dataset_path: str) -> bool:
    return dataset_path == "openlanguagedata/flores_plus"


def load_flores_dataset(source: str, target: str) -> list[dict]:
    src_iso = pycountry.languages.get(alpha_2=source).alpha_3
    tgt_iso = pycountry.languages.get(alpha_2=target).alpha_3

    dataset = hf_load_dataset("openlanguagedata/flores_plus", split="dev")

    id_to_translations: dict[str, dict[str, str]] = defaultdict(dict)
    for item in dataset:
        sent_id = item["id"]
        iso_code = item["iso_639_3"]
        text = item["text"]
        id_to_translations[sent_id][iso_code] = text

    pairs = []
    for sent_id, translations in id_to_translations.items():
        if src_iso in translations and tgt_iso in translations:
            pairs.append(
                {
                    "source": source,
                    "target": target,
                    "instruction": "",
                    "input": translations[src_iso],
                    "output": translations[tgt_iso],
                }
            )

    return pairs
