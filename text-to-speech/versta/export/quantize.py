from typing import Set

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from onnxruntime.quantization.registry import IntegerOpsRegistry
from os import path
import onnx

from pathlib import Path

BLOCKED_OPS = [
    "/text_encoder/Unsqueeze_2_output_0"
    "/text_encoder_1/Unsqueeze_output_0",
]

def quantize_model(export_dir: Path, model_filename: str, quantization_dir: Path):
    """
    Quantizes a specific ONNX model file and saves the quantized model to the given directory.

    Args:
        export_dir (Path): Path to the directory where the ONNX model is stored.
        model_filename (str): Name of the ONNX model file to quantize (e.g., "encoder_model.onnx").
        quantization_dir (Path): Path to the directory where the quantized model will be saved.
        config (AutoQuantizationConfig): Configuration for the quantization process.
    """

    file_name_without_extension = path.splitext(path.basename(model_filename))[0]

    input = export_dir / model_filename
    preprocessed = export_dir / f"{file_name_without_extension}_preprocessed.onnx"
    cleaned = export_dir / f"{file_name_without_extension}_clean.onnx"
    optimized = export_dir / f"{file_name_without_extension}_optimized.onnx"
    output = quantization_dir / f"{file_name_without_extension}_quantized.onnx"

    print(f"Preprocessing {model_filename} for quantization...")
    quant_pre_process(
        input_model=input,
        output_model_path=preprocessed,
        skip_symbolic_shape=True,
    )

    print(f"Cleaning {preprocessed}...")
    _clear_descriptions(preprocessed, cleaned)

    print(f"Optimizing {cleaned}...")
    _optimize_graph(cleaned, optimized)

    op_types_to_quantize = _get_operators(optimized, BLOCKED_OPS)

    print(f"Quantizing {optimized}...")
    quantize_dynamic(
        model_input=optimized,
        model_output=output,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=op_types_to_quantize,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        },
    )

def _optimize_graph(model_path, output_path: Path):
    sess_options = SessionOptions()

    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = output_path.as_posix()

    InferenceSession(model_path.as_posix(), sess_options)

def _clear_descriptions(model_path: Path, output_path: Path):
    model = onnx.load(model_path)

    for node in model.graph.node:
        node.doc_string = ""

    onnx.save(model, output_path)

def _get_operators(model_path: Path, blocked: list[str]) -> Set[str]:
    model = onnx.load(model_path)

    operators = set()
    def traverse_graph(graph):
        for node in graph.node:
            if node.name in blocked:
                print("Skipping blocked operator:", node.name)
                continue

            if node.op_type not in IntegerOpsRegistry:
                print("Skipping integer operator:", node.op_type)
                continue

            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators
