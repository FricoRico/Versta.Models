import torch
import torch.nn as nn
import torch_pruning as tp
from unsloth import FastLanguageModel

from .config import get_dtype


def _prune(
    model: nn.Module,
    prune_ratio: float = 0.25,
) -> nn.Module:
    """
    Apply STRUCTURED pruning with torch_pruning to actually reduce model size.

    Hardcoded:
    - round_to=8 (keeps dims as multiples of 8)
    - ignored_layers=[lm_head, embed_tokens, operator_norm, ffn_norm]
    - GroupMagnitudeImportance(p=2)

    Args:
        model: Model to prune.
        prune_ratio: Pruning ratio (0.0-1.0).
    """
    model.eval()
    torch.set_grad_enabled(True)

    example_inputs = torch.randint(0, 32000, (1, 128)).to(model.device)

    ignored_layers = [
        model.lm_head,
        model.model.embed_tokens,
    ]

    for name, module in model.named_modules():
        if any(
            x in name
            for x in [
                "operator_norm",
                "ffn_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "out_proj",
                "in_proj",
                "conv.conv",
            ]
        ):
            ignored_layers.append(module)

    imp = tp.importance.GroupMagnitudeImportance(p=2)
    pruner = tp.pruner.BasePruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
        round_to=8,
    )

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Before: {base_params / 1e6:.2f}M params")

    pruner.step()

    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    print(
        f"After: {params / 1e6:.2f}M params ({(1 - params / base_params) * 100:.1f}% reduction)"
    )

    return model


def _update_config(model: nn.Module) -> None:
    """
    Updates the model config to match the pruned architecture.

    Args:
        model: Pruned model.
    """
    pruned_intermediate = model.model.layers[0].feed_forward.w1.weight.shape[0]
    model.config.intermediate_size = pruned_intermediate
    model.config.block_auto_adjust_ff_dim = False

def _save(model: nn.Module, tokenizer: object, output_dir: object) -> None:
    """
    Saves the pruned model and tokenizer to the output directory.

    Args:
        model: Pruned model.
        tokenizer: Tokenizer for the model.
        output_dir: Directory to save the pruned model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def prune(
    merged_model_path: object,
    tokenizer: object,
    prune_ratio: float = 0.2,
    output_dir: object = None,
) -> object:
    """
    Loads a merged model, prunes it, and saves the pruned model.

    Args:
        merged_model_path: Path to the merged model directory.
        tokenizer: Tokenizer for the model.
        prune_ratio: Pruning ratio (0.0-1.0).
        output_dir: Directory to save the pruned model.

    Returns:
        Path to the pruned model directory.
    """
    dtype = get_dtype()
    print("Loading merged model for pruning")
    model, _ = FastLanguageModel.from_pretrained(
        merged_model_path.as_posix(),
        dtype=dtype,
        load_in_4bit=False,
    )

    print("Pruning model")
    model = _prune(model, prune_ratio)

    print("Saving pruned model")
    _update_config(model)
    _save(model, tokenizer, output_dir)

    return output_dir
