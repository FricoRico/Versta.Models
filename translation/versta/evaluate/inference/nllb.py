import pycountry
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import InferenceEngine


def _to_nllb_lang_code(alpha_2: str) -> str:
    alpha_3 = pycountry.languages.get(alpha_2=alpha_2).alpha_3
    return f"{alpha_3}_Latn"


class NllbEngine(InferenceEngine):
    model_type = "nllb"

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self._device = None

    def load(
        self, model_path: str, max_seq_length: int, device: str | None = None
    ) -> None:
        self._device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Work around transformers 5.4.0 bug: M2M100Config types scale_embedding
        # as int but NLLB configs ship it as bool, which fails strict dataclass
        # validation in huggingface_hub.
        from transformers.models.m2m_100.configuration_m2m_100 import M2M100Config

        fields = M2M100Config.__dataclass_fields__
        if "scale_embedding" in fields:
            fields["scale_embedding"].type = bool

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(
            self._device
        )

    def generate(
        self,
        data: list[dict],
        target: str,
        batch_size: int,
        gen_config: dict,
        max_seq_length: int,
    ) -> list[str]:
        self.model.generation_config.max_new_tokens = max_seq_length

        source_lang = data[0].get("source", "") if data else ""
        src_lang = _to_nllb_lang_code(source_lang)
        tgt_code = _to_nllb_lang_code(target)
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_code)
        self.tokenizer.src_lang = src_lang

        source_texts = [item.get("input", "") for item in data]
        hypotheses = []

        for i in range(0, len(source_texts), batch_size):
            batch_texts = source_texts[i : i + batch_size]

            try:
                batch_inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                ).to(self._device)

                gen_kwargs: dict = {
                    "forced_bos_token_id": forced_bos,
                }

                if gen_config.get("do_sample", False):
                    gen_kwargs["temperature"] = gen_config.get("temperature", 0.1)
                    gen_kwargs["top_k"] = gen_config.get("top_k", 50)
                    gen_kwargs["top_p"] = gen_config.get("top_p", 0.95)
                else:
                    gen_kwargs["num_beams"] = gen_config.get("num_beams", 4)
                    gen_kwargs["early_stopping"] = gen_config.get(
                        "early_stopping", True
                    )

                gen_kwargs["repetition_penalty"] = gen_config.get(
                    "repetition_penalty", 1.0
                )

                outputs = self.model.generate(**batch_inputs, **gen_kwargs)

                for j in range(len(batch_texts)):
                    hypothesis = self.tokenizer.decode(
                        outputs[j],
                        skip_special_tokens=True,
                    )
                    hypotheses.append(hypothesis)
            except Exception:
                hypotheses.extend([""] * len(batch_texts))

        return hypotheses
