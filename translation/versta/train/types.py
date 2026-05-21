from typing import TypedDict


class TrainStats(TypedDict):
    train_runtime: float
    train_total_tokens: int
    peak_memory_reserved: float
    peak_memory_for_lora: float
    peak_memory_percent: float
    lora_memory_percent: float
