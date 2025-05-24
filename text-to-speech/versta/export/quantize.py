from typing import Set

from onnxruntime.quantization import QuantType, QuantizationMode
from onnxruntime.quantization.registry import IntegerOpsRegistry
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from os import path
import onnx

from pathlib import Path

BLOCKED_OPS=(

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
    model = onnx.load_model(export_dir / model_filename)
    file_name_without_extension = path.splitext(path.basename(model_filename))[0]

    op_types_to_quantize = set(IntegerOpsRegistry.keys())
    if BLOCKED_OPS is not None:
        op_types_to_quantize.difference_update(BLOCKED_OPS)

    quantizer = ONNXQuantizer(
        model,
        per_channel=True,
        reduce_range=True,
        mode=QuantizationMode.IntegerOps,
        static=False,
        weight_qType=QuantType.QUInt8,
        activation_qType=QuantType.QUInt8,
        tensors_range=None,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        op_types_to_quantize=op_types_to_quantize,
        extra_options=dict(
            EnableSubgraph=True,
            MatMulConstBOnly=True,
        ),
    )
    quantizer.quantize_model()

    onnx.save(
        quantizer.model.model,
        quantization_dir / f"{file_name_without_extension}_quantized.onnx",
        convert_attribute=True,
    )

def get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    traverse_graph(attr.g)

    traverse_graph(model.graph)
    return operators