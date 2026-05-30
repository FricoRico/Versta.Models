import os
import shutil
import time
from pathlib import Path

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

from .config import get_dtype


def _train(
    model: str | Path,
    tokenizer: object,
    dataset: object,
    batch_size: int,
    max_seq_length: int,
    checkpoints_dir: Path,
    logs_dir: Path,
    num_train_epochs: float = 1,
    learning_rate: float = 2e-4,
    embedding_learning_rate: float = 1e-5,
    save_steps: int = 10000,
) -> Path:
    """
    Trains the model using SFT with the given dataset.

    Args:
        model: Model with LoRA adapters.
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
    eval_size = 5000
    train_dataset = dataset.select(range(len(dataset) - eval_size))
    eval_dataset = dataset.select(range(len(dataset) - eval_size, len(dataset)))

    os.environ["TENSORBOARD_LOGGING_DIR"] = logs_dir.as_posix()

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=max_seq_length,
        dataset_num_proc=16,
        dataloader_prefetch_factor=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 64 // batch_size),
            num_train_epochs=num_train_epochs,
            warmup_steps=400,
            learning_rate=learning_rate,
            embedding_learning_rate=embedding_learning_rate,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=1779708246,
            packing=True,
            output_dir=checkpoints_dir.as_posix(),
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
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
        r=32,
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
        lora_alpha=64,
        lora_dropout=0.05,
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
    output_dir: Path,
    lang_pair: str,
    logs_dir: Path,
    batch_size: int,
    max_seq_length: int,
    num_train_epochs: float = 1,
    learning_rate: float = 1e-4,
    embedding_learning_rate: float = 1e-5,
    save_steps: int = 10000,
) -> Path:
    """
    Finetunes the model with LoRA adapters and saves outputs.

    Args:
        model: Name of the model to load.
        dataset: Formatted dataset for training.
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
        dataset=dataset,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        checkpoints_dir=checkpoints_dir,
        logs_dir=run_logs_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        embedding_learning_rate=embedding_learning_rate,
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
    output_dir: Path,
    intermediates_dir: Path,
    logs_dir: Path,
    lang_pair: str,
    batch_size: int,
    max_seq_length: int,
    num_train_epochs: float = 1,
    num_passes: int = 2,
    learning_rate: float = 8e-5,
    embedding_learning_rate: float = 5e-5,
    save_steps: int = 10000,
) -> Path:
    """
    Loads a pruned model, applies LoRA adapters, trains, and saves outputs.

    Args:
        model: Path to the pruned model directory.
        tokenizer: Tokenizer for the model.
        dataset: Formatted dataset for training.
        output_dir: Directory for the final merged model.
        lang_pair: Language pair string (e.g. "en-nl").
        intermediates_dir: Directory for intermediate files.
        batch_size: Per-device batch size.
        max_seq_length: Maximum sequence length.
        cache_dir: Cache directory for Unsloth.
        num_train_epochs: Number of training epochs.
        learning_rate: Learning rate.
        embedding_learning_rate: Embedding learning rate.
        save_steps: Steps interval for saving and evaluation.
    """
    model_path = model
    tokenizer = None

    for pass_num in range(num_passes):
        for key in ["UNSLOTH_RETURN_LOGITS", "UNSLOTH_IS_PRESENT"]:
            os.environ.pop(key, None)

        pass_dir = intermediates_dir / f"pass_{pass_num}"
        adapter_dir = pass_dir / "adapter"
        checkpoints_dir = pass_dir / "checkpoints"

        pass_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        model, tokenizer = _load(
            model=model_path,
            max_seq_length=max_seq_length,
        )

        model_path = pass_dir
        pass_lr = learning_rate * (0.8**pass_num)
        pass_embed_lr = embedding_learning_rate * (0.9**pass_num)

        print(
            f"Recovery pass {pass_num + 1}/{num_passes}: learning_rate: {pass_lr:.2e}, embedding_learning_rate: {pass_embed_lr:.2e}"
        )

        run_name = f"recover_{lang_pair}_pass{pass_num}_{int(time.time())}"
        run_logs_dir = logs_dir / run_name
        run_logs_dir.mkdir(parents=True, exist_ok=True)

        if num_train_epochs < 1:
            dataset = dataset.shuffle(seed=1779708246 + pass_num)

        model = _train(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            checkpoints_dir=checkpoints_dir,
            logs_dir=run_logs_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=pass_lr,
            embedding_learning_rate=pass_embed_lr,
            save_steps=save_steps,
        )

        print(f"Saving recovered LoRA adapter for pass: {pass_num + 1}/{num_passes}")
        _save_adapter(model, tokenizer, adapter_dir)

        print(f"Saving recovered merged model for pass: {pass_num + 1}/{num_passes}")
        _save_model(model, tokenizer, pass_dir)

    print("Saving final recovered model")
    _save_model(model, tokenizer, output_dir)

    return output_dir
