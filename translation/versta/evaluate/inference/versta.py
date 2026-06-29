import os

os.environ["UNSLOTH_COMPILE_LOCATION"] = "cache/unsloth/"

import pycountry
import torch
from unsloth import FastLanguageModel

from .base import InferenceEngine


class VerstaEngine(InferenceEngine):
    model_type = "versta"

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def load(
        self, model_path: str, max_seq_length: int, device: str | None = None
    ) -> None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
        )

        FastLanguageModel.for_inference(model)
        model.generation_config.max_length = max_seq_length
        self.model = model
        self.tokenizer = tokenizer

    def _build_prompts(self, data: list[dict]) -> list[str]:
        prompts = []
        for item in data:
            instruction = item.get("instruction", "")
            if not instruction:
                tone = item.get("_tone", "neutral")
                target = item.get("target", "")
                target_lang_obj = pycountry.languages.get(alpha_2=target)
                target_name = target_lang_obj.name if target_lang_obj else target
                if tone == "plain":
                    instruction = f"Translate to {target_name}."
                else:
                    instruction = f"Translate to {tone.lower()} {target_name}."
            input_text = item.get("input", "")
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input_text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
        return prompts

    def generate(
        self,
        data: list[dict],
        target: str,
        batch_size: int,
        gen_config: dict,
        max_seq_length: int,
    ) -> list[str]:
        all_prompts = self._build_prompts(data)
        hypotheses = []

        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i : i + batch_size]

            try:
                batch_inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                ).to(self.model.device)

                outputs = self.model.generate(
                    **batch_inputs,
                    **gen_config,
                )

                input_len = batch_inputs["input_ids"].shape[1]
                for j in range(len(batch_prompts)):
                    hypothesis = self.tokenizer.decode(
                        outputs[j][input_len:],
                        skip_special_tokens=True,
                    )
                    hypotheses.append(hypothesis)
            except Exception:
                hypotheses.extend([""] * len(batch_prompts))

        return hypotheses
