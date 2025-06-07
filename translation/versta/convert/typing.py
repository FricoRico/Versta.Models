from typing import TypedDict

class Decoder(TypedDict):
    relative_paths: bool
    models: list[str]
    vocabs: list[str]
    beam_size: int
    normalize: int
    word_penalty: int
    mini_batch: int
    maxi_batch: int
    maxi_batch_sort: str

def decoder_serializer(data: Decoder) -> dict[str, list[str] | int | bool]:
    """
    Converts the Decoder TypedDict to a dictionary format suitable for serialization.

    Args:
        data (Decoder): The Decoder TypedDict containing the configuration data.
    """
    return {
        'relative-paths': data['relative_paths'],
        'models': data['models'],
        'vocabs': data['vocabs'],
        'beam-size': data['beam_size'],
        'normalize': data['normalize'],
        'word-penalty': data['word_penalty'],
        'mini-batch': data['mini_batch'],
        'maxi-batch': data['maxi_batch'],
        'maxi-batch-sort': data['maxi_batch_sort']
    }
