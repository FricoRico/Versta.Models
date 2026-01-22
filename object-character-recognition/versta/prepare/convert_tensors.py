"""Convert calibration images to ONNX tensor format for AWQ."""

import numpy as np
import onnx
from pathlib import Path
from typing import List

from .typing import CropResult, ManifestEntry

# Mapping from numpy dtypes to ONNX TensorProto data types
NP_DTYPE_TO_ONNX_DTYPE = {
    np.float32: onnx.TensorProto.FLOAT,
    np.float16: onnx.TensorProto.FLOAT16,
    np.int8: onnx.TensorProto.INT8,
    np.int16: onnx.TensorProto.INT16,
    np.int32: onnx.TensorProto.INT32,
    np.int64: onnx.TensorProto.INT64,
}


def images_to_onnx_tensors(
    crops: List[np.ndarray],
    output_path: Path,
    tensor_name: str = "x",
    batch_axis: bool = True,
) -> None:
    """
    Convert cropped images to ONNX tensor format.

    Creates a minimal ONNX model containing the calibration tensors.
    The tensor is saved as a constant initializer, which can be extracted by AWQ.

    Args:
        crops: List of preprocessed images (numpy arrays)
        output_path: Path to save ONNX tensor file
        tensor_name: Name for the input tensor (default: "x")
        batch_axis: Whether to add batch dimension (default: True)
    """
    if len(crops) == 0:
        raise ValueError("No crops provided")

    first_shape = crops[0].shape
    dtype = crops[0].dtype

    print(f"\n{'=' * 70}")
    print(f"Converting to ONNX tensors")
    print(f"{'=' * 70}")
    print(f"Number of samples: {len(crops)}")
    print(f"Tensor name: {tensor_name}")
    print(f"Data type: {dtype}")
    print(f"Sample shape: {first_shape}")

    stack_axis = 0 if batch_axis else None

    stacked = np.stack(crops, axis=stack_axis)

    print(f"Stacked shape: {stacked.shape}")
    print(f"Total size: {stacked.nbytes / (1024 * 1024):.2f} MB")

    # Get ONNX dtype with fallback to FLOAT32
    onnx_dtype = NP_DTYPE_TO_ONNX_DTYPE.get(dtype, onnx.TensorProto.FLOAT)

    tensor = onnx.helper.make_tensor(
        name=tensor_name,
        data_type=onnx_dtype,
        dims=stacked.shape,
        vals=stacked.flatten().tolist(),
    )

    # Create output value info
    output_value_info = onnx.helper.make_tensor_value_info(
        tensor_name, onnx_dtype, stacked.shape
    )

    graph = onnx.helper.make_graph(
        nodes=[],  # No computation nodes
        name="calibration_tensors",
        inputs=[],  # No inputs
        outputs=[output_value_info],  # One output
        initializer=[tensor],  # Our tensor as initializer
    )

    model = onnx.helper.make_model(graph)

    print(f"\nSaving ONNX tensors to: {output_path}")
    onnx.save_model(model, str(output_path))

    print(f"✓ ONNX tensors saved")
    print(f"{'=' * 70}\n")


def prepare_calibration_manifest(
    crop_info: List[CropResult],
    image_paths: List[str],
    language_codes: List[str],
    output_path: Path,
) -> None:
    """
    Create manifest file with crop metadata including language codes.

    Args:
        crop_info: List of CropResult containing crop metadata
        image_paths: List of source image paths (one per crop)
        language_codes: Language code for each crop (same index alignment)
        output_path: Path to save manifest.json
    """
    print(f"\n{'=' * 70}")
    print(f"Creating calibration manifest")
    print(f"{'=' * 70}")
    print(f"Number of entries: {len(crop_info)}")

    manifest: List[ManifestEntry] = []

    for idx, (crop, img_path, lang_code) in enumerate(
        zip(crop_info, image_paths, language_codes)
    ):
        manifest.append(
            {
                "image_path": img_path,
                "box": crop["box"],
                "score": crop["score"],
                "crop_index": idx,
                "language_code": lang_code,
            }
        )

    print(f"\nSaving manifest to: {output_path}")

    import json

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✓ Manifest saved")
    print(f"{'=' * 70}\n")
