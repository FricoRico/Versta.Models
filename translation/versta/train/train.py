import os
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

from .config import get_dtype


def _train(
    model: str | Path,
    tokenizer: object,
    eval_dataset: object,
    dataset: object,
    batch_size: int,
    max_seq_length: int,
    checkpoints_dir: Path,
    logs_dir: Path,
    num_train_epochs: float,
    learning_rate: float,
    embedding_learning_rate: float,
    enable_metrics: bool,
    save_steps: int,
) -> Path:
    """
    Trains the model using SFT with the given dataset.

    Args:
        model: Model with LoRA adapters.
        eval_dataset: Separate dataset for evaluation with prompt masking.
        dataset: Formatted dataset for training.
        tokenizer: Tokenizer for the model.
        batch_size: Per-device batch size.
        max_seq_length: Maximum sequence length.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.
        embedding_learning_rate: Embedding learning rate.
        save_steps: Steps interval for saving and evaluation.
        run_name: Unique identifier for the TensorBoard run.

    Returns:
        Trained model.
    """
    os.environ["TENSORBOARD_LOGGING_DIR"] = logs_dir.as_posix()

    trainer_kwargs = {}
    if enable_metrics:
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        def _preprocess_logits_for_metrics(logits, labels):
            pred_ids = torch.argmax(logits, dim=-1)
            return pred_ids

        def _compute_metrics(eval_preds):
            predictions = eval_preds.predictions
            labels = eval_preds.label_ids

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            predictions = np.where(labels == -100, pad_id, predictions)
            labels = np.where(labels == -100, pad_id, labels)

            decoded_preds = tokenizer.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
            chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels])

            return {
                "bleu": bleu.score,
                "chrf": chrf.score,
            }

        trainer_kwargs["compute_metrics"] = _compute_metrics
        trainer_kwargs["preprocess_logits_for_metrics"] = _preprocess_logits_for_metrics

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_length,
        dataset_num_proc=16,
        dataloader_prefetch_factor=16,
        **trainer_kwargs,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 64 // batch_size),
            num_train_epochs=num_train_epochs,
            warmup_ratio=0.02,
            learning_rate=learning_rate,
            embedding_learning_rate=embedding_learning_rate,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=1779708246,
            packing=not enable_metrics,
            output_dir=checkpoints_dir.as_posix(),
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=6,
            load_best_model_at_end=True,
            metric_for_best_model="eval_chrf" if enable_metrics else "eval_loss",
            greater_is_better=enable_metrics,
            eval_strategy="steps",
            eval_steps=save_steps,
            report_to="tensorboard",
            logging_strategy="steps",
            logging_steps=100,
        ),
    )

    trainer.train()
    return trainer.model


def _load(
    model: str | Path,
    max_seq_length: int,
) -> tuple:
    """
    Loads the model with LoRA adapters.

    Args:
        model_name: Name of the model to load.
        max_seq_length: Maximum sequence length.

    Returns:
        Tuple of (model, tokenizer).
    """
    dtype = get_dtype()
    if isinstance(model, Path):
        model = model.as_posix()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w2",
            "w3",
            "embed_tokens",
            "lm_head",
        ],
        lora_alpha=128,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1779708246,
        use_rslora=True,
    )

    return model, tokenizer


def _save_adapter(model: object, tokenizer: object, output_dir: object) -> None:
    """
    Saves the LoRA adapter weights to the output directory.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer for the model.
        output_dir: Directory to save the LoRA adapter.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _save_model(model: object, tokenizer: object, output_dir: object) -> None:
    """
    Saves the merged (LoRA-unloaded) model to the output directory.

    Args:
        model: Model with LoRA adapters.
        tokenizer: Tokenizer for the model.
        output_dir: Directory to save the merged model.
    """
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def finetune(
    model: str | Path,
    dataset: object,
    eval_dataset: object,
    output_dir: Path,
    lang_pair: str,
    logs_dir: Path,
    batch_size: int,
    max_seq_length: int,
    num_train_epochs: float = 1,
    learning_rate: float = 2e-4,
    embedding_learning_rate: float = 5e-5,
    enable_metrics: bool = False,
    save_steps: int = 5000,
) -> Path:
    """
    Finetunes the model with LoRA adapters and saves outputs.

    Args:
        model: Name of the model to load.
        dataset: Formatted dataset for training.
        eval_dataset: Separate dataset for evaluation with prompt masking.
        output_dir: Directory for the final merged model.
        lang_pair: Language pair string (e.g. "en-nl").
        intermediates_dir: Directory for intermediate files.
        cache_dir: Cache directory for Unsloth.
        batch_size: Per-device batch size.
        max_seq_length: Maximum sequence length.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.
        embedding_learning_rate: Embedding learning rate.
        save_steps: Steps interval for saving and evaluation.

    Returns:
        Tuple of (tokenizer, merged_model_path).
    """
    for key in ["UNSLOTH_RETURN_LOGITS", "UNSLOTH_IS_PRESENT"]:
        os.environ.pop(key, None)

    checkpoints_dir = output_dir / "checkpoints"
    adapter_dir = output_dir / "adapter"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model with LoRA adapters")
    model, tokenizer = _load(
        model=model,
        max_seq_length=max_seq_length,
    )

    run_name = f"finetune_{lang_pair}_{int(time.time())}"

    run_logs_dir = logs_dir / run_name
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    print("Training model")
    model = _train(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        dataset=dataset,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        checkpoints_dir=checkpoints_dir,
        logs_dir=run_logs_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        enable_metrics=enable_metrics,
        save_steps=save_steps,
    )

    print("Saving finetuned LoRA adapter")
    _save_adapter(model, tokenizer, adapter_dir)

    print("Saving merged model")
    _save_model(model, tokenizer, output_dir)

    return output_dir


def recover(
    model: str | Path,
    dataset: object,
    eval_dataset: object,
    output_dir: Path,
    intermediates_dir: Path,
    logs_dir: Path,
    lang_pair: str,
    batch_size: int,
    max_seq_length: int,
    num_train_epochs: float = 1,
    learning_rate: float = 8e-5,
    embedding_learning_rate: float = 5e-5,
    save_steps: int = 5000,
    enable_metrics: bool = False,
) -> Path:
    """
    Loads a pruned model, applies LoRA adapters, trains, and saves outputs.

    Args:
        model: Path to the pruned model directory.
        tokenizer: Tokenizer for the model.
        dataset: Formatted dataset for training.
        eval_dataset: Separate dataset for evaluation with prompt masking.
        output_dir: Directory for the final merged model.
        lang_pair: Language pair string (e.g. "en-nl").
        batch_size: Per-device batch size.
        max_seq_length: Maximum sequence length.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.
        embedding_learning_rate: Embedding learning rate.
        save_steps: Steps interval for saving and evaluation.
    """
    for key in ["UNSLOTH_RETURN_LOGITS", "UNSLOTH_IS_PRESENT"]:
        os.environ.pop(key, None)

    adapter_dir = intermediates_dir / "adapter"
    checkpoints_dir = intermediates_dir / "checkpoints"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load(
        model=model,
        max_seq_length=max_seq_length,
    )

    run_name = f"recover_{lang_pair}_{int(time.time())}"
    run_logs_dir = logs_dir / run_name
    run_logs_dir.mkdir(parents=True, exist_ok=True)

    model = _train(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        dataset=dataset,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        checkpoints_dir=checkpoints_dir,
        logs_dir=run_logs_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
        enable_metrics=enable_metrics,
        save_steps=save_steps,
    )

    print("Saving recovered LoRA adapter")
    _save_adapter(model, tokenizer, adapter_dir)

    print("Saving recovered merged model")
    _save_model(model, tokenizer, output_dir)

    return output_dir
