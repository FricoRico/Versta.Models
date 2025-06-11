from typing import Set

from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
from onnxruntime.quantization.registry import IntegerOpsRegistry
from os import path
import onnx

from pathlib import Path

BLOCKED_OPS = (

)


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
    output = quantization_dir / f"{file_name_without_extension}_quantized.onnx"

    op_types_to_quantize = set(IntegerOpsRegistry.keys())
    if BLOCKED_OPS is not None:
        op_types_to_quantize.difference_update(BLOCKED_OPS)

    print(f"Preprocessing {model_filename} for quantization...")
    quant_pre_process(
        input_model=input,
        output_model_path=preprocessed,
        skip_symbolic_shape=True,
    )

    print(f"Quantizing {preprocessed}...")
    quantize_dynamic(
        model_input=preprocessed,
        model_output=output,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        extra_options={
            "ForceQuantizeNoInputCheck": True,
            "MatMulConstBOnly": True,
        },
    )


def _get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators
