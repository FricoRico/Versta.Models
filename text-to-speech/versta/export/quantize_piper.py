from pathlib import Path
from shutil import copyfile
from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig
from onnxruntime.quantization.registry import QLinearOpsRegistry
from onnx import load, save, AttributeProto

# Piper models may have specific nodes that should not be quantized
BLOCKED_NODES = []
BLOCKED_OPS = []


def quantize_piper(export_dir: Path, model_filename: str, quantization_dir: Path):
    """
    Quantizes a Piper ONNX model for mobile deployment.

    Args:
        export_dir (Path): Path to the directory where the ONNX model is stored.
        model_filename (str): Name of the ONNX model file to quantize.
        quantization_dir (Path): Path to the directory where the quantized model will be saved.
    """
    print(f"Preparing Piper model for quantization: {model_filename}")
    
    # Clear model descriptions to reduce file size
    _clear_descriptions(export_dir / model_filename, export_dir / model_filename)

    print(f"Quantizing {model_filename}...")

    nodes_to_exclude = _get_excluded_nodes(export_dir / model_filename, BLOCKED_NODES)
    operators_to_quantize = _get_operators(export_dir / model_filename, BLOCKED_OPS)

    config = AutoQuantizationConfig.arm64(
        is_static=False,
        nodes_to_exclude=nodes_to_exclude,
        operators_to_quantize=operators_to_quantize
    )

    quantizer = ORTQuantizer.from_pretrained(export_dir, model_filename)
    quantizer.quantize(config, quantization_dir)


def _clear_descriptions(model_path: Path, output_path: Path):
    """
    Clears node descriptions from the ONNX model to reduce file size.
    
    Args:
        model_path (Path): Path to the input ONNX model.
        output_path (Path): Path to save the modified model.
    """
    model = load(model_path)

    for node in model.graph.node:
        node.doc_string = ""

    save(model, output_path)


def _get_excluded_nodes(model_path: Path, blocked_nodes: list[str]) -> list[str]:
    """
    Gets a list of nodes that should be excluded from quantization.
    
    Args:
        model_path (Path): Path to the ONNX model.
        blocked_nodes (list[str]): List of node name prefixes to block.
    
    Returns:
        list[str]: List of node names to exclude from quantization.
    """
    model = load(model_path)

    excluded_nodes = list()
    for node in model.graph.node:
        if any(node.name.startswith(block) for block in blocked_nodes):
            excluded_nodes.append(node.name)

    return excluded_nodes


def _get_operators(model_path: Path, blocked_ops: list[str]) -> list[str]:
    """
    Gets a list of operators that can be quantized, excluding blocked ones.
    
    Args:
        model_path (Path): Path to the ONNX model.
        blocked_ops (list[str]): List of operator types to exclude.
    
    Returns:
        list[str]: List of operator types to quantize.
    """
    model = load(model_path)

    operators = list()
    
    def traverse_graph(graph):
        for node in graph.node:
            if node.op_type not in QLinearOpsRegistry:
                continue

            if node.op_type in blocked_ops:
                continue

            operators.append(node.op_type)
            for attr in node.attribute:
                if attr.type == AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators