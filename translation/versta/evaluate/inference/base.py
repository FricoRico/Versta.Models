from abc import ABC, abstractmethod


class InferenceEngine(ABC):
    model_type: str = "unknown"

    @abstractmethod
    def load(
        self, model_path: str, max_seq_length: int, device: str | None = None
    ) -> None: ...

    @abstractmethod
    def generate(
        self,
        data: list[dict],
        target: str,
        batch_size: int,
        gen_config: dict,
        max_seq_length: int,
    ) -> list[str]: ...
