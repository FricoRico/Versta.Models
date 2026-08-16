from pathlib import Path
from typing import List

import numpy as np
import onnx

from onnx import GraphProto, ModelProto, helper, numpy_helper


def fold_deconv_weight(w: np.ndarray) -> np.ndarray:
    """
    Folds a 2x2 stride-2 ConvTranspose kernel into an equivalent 1x1 Conv
    kernel with spatially averaged weights — exact for an unpadded deconv,
    equivalent to average-pooling the full-resolution logit map.

    Args:
        w (np.ndarray): ConvTranspose weights, [Cin, Cout, 2, 2].

    Returns:
        np.ndarray: Conv weights, [Cout, Cin, 1, 1].
    """
    return w.mean(axis=(2, 3)).T[:, :, None, None].astype(np.float32)


def _resolve_initializer(graph: GraphProto, value_name: str) -> onnx.TensorProto:
    """
    Resolves an ONNX value to its source weight tensor, following Identity
    chains. paddle2onnx 2.1 emits PIR weights as Constant nodes wrapped in
    auto-cast Identity nodes rather than graph initializers.

    Args:
        graph (GraphProto): The graph to search.
        value_name (str): The value name to resolve.

    Returns:
        onnx.TensorProto: The backing weight tensor.

    Raises:
        KeyError: If the value does not trace back to a weight tensor.
    """
    inits = {i.name: i for i in graph.initializer}
    name = value_name
    while name not in inits:
        producer = next((n for n in graph.node if name in n.output), None)
        if producer is None:
            raise KeyError(f"value {value_name} does not resolve to a weight")
        if producer.op_type == "Constant":
            return next(a.t for a in producer.attribute if a.name == "value")
        if producer.op_type != "Identity":
            raise KeyError(f"value {value_name} does not resolve to a weight")
        name = producer.input[0]
    return inits[name]


def replace_deconv(graph: GraphProto, node_name: str, new_w_name: str) -> None:
    """
    Replaces a named ConvTranspose node in an ONNX graph with a 1x1 Conv using
    folded weights.

    Args:
        graph (GraphProto): The ONNX graph to edit in place.
        node_name (str): Name of the ConvTranspose node to replace.
        new_w_name (str): Name for the appended folded-weight initializer.

    Raises:
        KeyError: If no node with the given name exists in the graph.
    """
    for idx, node in enumerate(graph.node):
        if node.name != node_name:
            continue
        w = numpy_helper.to_array(_resolve_initializer(graph, node.input[1]))
        graph.initializer.append(
            numpy_helper.from_array(fold_deconv_weight(w), new_w_name)
        )
        conv = helper.make_node(
            "Conv",
            [node.input[0], new_w_name],
            list(node.output),
            name=f"{node_name}_folded",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )
        graph.node.remove(node)
        graph.node.insert(idx, conv)
        return
    raise KeyError(f"node {node_name} not found")


def clear_output_dims(graph: GraphProto) -> None:
    """
    Marks all graph output dimensions dynamic; the folded head changes the
    static output shape recorded by the exporter.

    Args:
        graph (GraphProto): The ONNX graph to edit in place.
    """
    for output in graph.output:
        for dim in output.type.tensor_type.shape.dim:
            dim.Clear()
            dim.dim_param = "dyn"


def fold_variant_graph(
    onnx_path: Path, deconv_names: List[str], out_path: Path
) -> Path:
    """
    Loads an ONNX detector model, folds the given DBNet head deconvs into
    1x1 convolutions and saves the edited graph.

    Args:
        onnx_path (Path): Source ONNX detector model.
        deconv_names (List[str]): ConvTranspose node names to fold, in order.
        out_path (Path): Destination ONNX file path.

    Returns:
        Path: The written ONNX variant path.
    """
    model: ModelProto = onnx.load(str(onnx_path))
    for deconv in deconv_names:
        replace_deconv(model.graph, deconv, f"folded_{deconv}")
    clear_output_dims(model.graph)
    onnx.save(model, str(out_path))
    return out_path
