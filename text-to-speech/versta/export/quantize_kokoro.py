from shutil import copyfile

from huggingface_hub import hf_hub_download

from onnxruntime import SessionOptions, GraphOptimizationLevel
from optimum.onnxruntime import ORTQuantizer, AutoQuantizationConfig
from onnxruntime.quantization.registry import QLinearOpsRegistry
from onnx import load, save, AttributeProto
from pathlib import Path

BLOCKED_NODES = []
BLOCKED_OPS = []

def quantize_kokoro(export_dir: Path, model_filename: str, quantization_dir: Path):
    """
    Quantizes a specific ONNX model file and saves the quantized model to the given directory.

    Args:
        export_dir (Path): Path to the directory where the ONNX model is stored.
        model_filename (str): Name of the ONNX model file to quantize (e.g., "encoder_model.onnx").
        quantization_dir (Path): Path to the directory where the quantized model will be saved.
    """
    print("Skipping quantization for kokoro model as it is not supported. Downloading the quantized model instead.")
    model_file = hf_hub_download(
        repo_id="onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
        filename="model_uint8.onnx",
        subfolder="onnx",
    )
    copyfile(model_file, quantization_dir / "model_quantized.onnx")
    return

    print(f"Cleaning {model_filename}...")
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
    model = load(model_path)

    for node in model.graph.node:
        node.doc_string = ""

    save(model, output_path)

def _get_excluded_nodes(model_path: Path, blocked_nodes: list[str]) -> list[str]:
    model = load(model_path)

    excluded_nodes = list()
    for node in model.graph.node:
        if any(node.name.startswith(block) for block in blocked_nodes):
            excluded_nodes.append(node.name)

    return excluded_nodes

def _get_operators(model_path: Path, blocked_ops: list[str]) -> list[str]:
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
